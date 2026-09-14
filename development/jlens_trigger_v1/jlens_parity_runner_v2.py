"""Prospective J-lens parity harness V1 (job jlens_loader_parity_20260914_v1).

This module is the control shell for a future, separately reviewed and released
single-forward parity run. It is MODEL-FREE at import and in every test: it
imports no ``torch``/``transformers``/``tokenizers``/``safetensors``/
``transformer_lens``, reads no lens/model/checkpoint/cache tensor, runs no
forward and fits nothing. All real providers are imported only inside the watched
worker, after preflight and after the snapshot bytes are verified, exactly as the
reviewed ``native_development_runner_v2`` supervisor does.

The prospective run (NOT authorized here) is one frozen raw ``Qwen/Qwen3.5-0.8B``
revision ``2fc0636...`` model, one tokenizer and exactly one forward on one
predeclared TRAIN prompt whose class label/outcome is not read. On that same
model object and same forward it compares the raw HF block outputs for layers
6/10/18 with the TransformerBridge ``blocks.{l}.hook_out`` recordings, then
compares ``jlens_core_v2`` selected-token readouts against the published
``JacobianLens`` API math (``transport`` + the library's own ``_unembed``) applied
to the same captured hidden state (never a second forward). If a same-model hook
path is unavailable the run fails closed; the exact minimal alternative is stated
in :data:`MINIMAL_ALTERNATIVE`.

Caps (PROVISIONAL, freeze in the reviewed lock): 600 s, 1 model load, 1 tokenizer
load, 1 forward, 0 fits/derivatives, 64 MiB output, <= 6 GiB working memory, and
no generation / logit-behaviour study / steering. Independent review and the
root-prepared release lock must precede any real execution. This module never
creates that lock and never launches parity.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import re
import sys
import time
import uuid

import numpy as np

import jlens_core_v2 as core
import jlens_io_v1 as io

__all__ = [
    "JOB_ID",
    "SCHEMA",
    "RECEIPT_SCHEMA",
    "CAPS",
    "TOLERANCES",
    "BLOCKS",
    "MINIMAL_ALTERNATIVE",
    "GateError",
    "need",
    "preflight",
    "verify_snapshot",
    "load_tensors",
    "capture",
    "build_parity_adapter",
    "worker",
    "supervise",
    "main",
]

runner = io.runner
GateError = io.GateError
ROOT = Path(__file__).resolve().parents[2]
STUDY = Path("development/jlens_trigger_v1")
NATIVE_STUDY = Path("development/classifier_generalization_v2")

JOB_ID = "jlens_parity_root_v2"
SCHEMA = "jlens_parity_execution.v1"
RECEIPT_SCHEMA = "jlens_parity_receipt.v1"
BLOCKS = io.BLOCKS
ARTIFACTS = ("parity_receipt.json",)
OWNER_NAME = "native_model_owner.json"
WORKER_RECEIPT = "worker_complete.json"
SUCCESS_RECEIPT = "supervisor_success.json"
STATUS_RESERVE = runner.STATUS_RESERVE

CAPS = dict(
    seconds=600,
    model_loads=1,
    tokenizer_loads=1,
    forwards=1,
    fits=0,
    derivatives=0,
    output_bytes=67108864,
    working_memory_gib=6,
    generation=False,
    logit_behavior_study=False,
    steering=False,
)

# Provisional locked tolerances. Independent review must confirm or tighten them
# in the root-prepared lock before any real execution.
TOLERANCES = {
    "hook_max_abs": 0.0,
    "hook_require_float32": True,
    "readout_max_abs": 0.004,
    "readout_relative": False,
    "provisional": False,
    "note": "hook equality is exact for one same-object forward; readout is float32 matmul order drift",
}

MINIMAL_ALTERNATIVE = (
    "If TransformerBridge blocks.{l}.hook_out cannot be produced from the same model object "
    "and the same forward, fail closed. The exact minimal alternative is one "
    "bridge.run_with_cache call on the already-loaded model with (a) a names_filter selecting "
    "only blocks.{6,10,18}.hook_out and (b) plain torch forward hooks registered on the original "
    "HF decoder layers, then comparing those two same-forward recordings. A second model load, a "
    "second forward, or a fabricated value is not an acceptable substitute."
)

SOURCE_PATHS = (
    (STUDY / "jlens_parity_runner_v2.py").as_posix(),
    (STUDY / "jlens_io_v1.py").as_posix(),
    (STUDY / "jlens_core_v2.py").as_posix(),
    (STUDY / "PARITY_PLAN_V1.json").as_posix(),
    (NATIVE_STUDY / "native_development_runner_v2.py").as_posix(),
    (NATIVE_STUDY / "snapshot_verifier.py").as_posix(),
)
PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "transformer-lens", "numpy", "torchvision")
PROVIDERS = {
    "torch": ("torch", "torch/__init__.py"),
    "config": ("transformers.models.qwen3_5.configuration_qwen3_5", "transformers/models/qwen3_5/configuration_qwen3_5.py"),
    "model": ("transformers.models.qwen3_5.modeling_qwen3_5", "transformers/models/qwen3_5/modeling_qwen3_5.py"),
    "tokenizer": ("transformers.models.qwen2.tokenization_qwen2", "transformers/models/qwen2/tokenization_qwen2.py"),
    "bridge_builder": ("transformer_lens.model_bridge.sources._bridge_builder", "transformer_lens/model_bridge/sources/_bridge_builder.py"),
    "bridge": ("transformer_lens.model_bridge.transformer_bridge", "transformer_lens/model_bridge/transformer_bridge.py"),
    "jacobian_lens": ("transformer_lens.tools.analysis.jacobian_lens", "transformer_lens/tools/analysis/jacobian_lens.py"),
}
PROMPT_FIELDS = {"case_id", "text", "sha256"}
INPUT_ROLES = ("lens", "model_snapshot_lock", "train_manifest", "prompt")
FORBIDDEN_ROOTS = {"torch", "transformers", "tokenizers", "safetensors", "transformer_lens"}


def need(condition, code, detail=""):
    if not condition:
        raise GateError(code if not detail else "%s: %s" % (code, detail))


# --------------------------------------------------------------------------- #
# model-free preflight: no provider import, no tensor, no readout, no fit
# --------------------------------------------------------------------------- #
def _runtime(ctx_lock):
    runtime = ctx_lock["runtime"]
    prefix = runner.runtime_prefix()
    need(Path(runtime["prefix"]).resolve() == prefix, "RUNTIME_PREFIX")
    need(runtime["python"] == ".".join(map(str, sys.version_info[:3])), "PYTHON_VERSION")
    need(set(runtime["packages"]) == set(PACKAGES), "RUNTIME_PACKAGE_SET")
    for name, version in runtime["packages"].items():
        need(importlib.metadata.version(name) == version, "RUNTIME_VERSION")
    need(set(runtime["provider_sources"]) == set(PROVIDERS), "PROVIDER_SOURCE_SET")
    provider_paths = {}
    for role, (_, relative) in PROVIDERS.items():
        path = runner.relative_path(prefix, "Lib/site-packages/" + relative)
        need(
            runner.sha(runner.small_bytes(path)) == runner.digest(runtime["provider_sources"][role]),
            "PROVIDER_SOURCE_HASH",
        )
        provider_paths[role] = path
    extras = runtime.get("extra_provider_sources", {})
    need(type(extras) is dict and len(extras) <= 256, "EXTRA_PROVIDER_SET")
    for relative, digest in extras.items():
        need(relative.startswith(("transformer_lens/", "torchvision/")), "EXTRA_PROVIDER_SCOPE")
        path = runner.relative_path(prefix, "Lib/site-packages/" + relative)
        need(runner.sha(runner.small_bytes(path)) == runner.digest(digest), "EXTRA_PROVIDER_HASH")
    return provider_paths


def _inputs(root, lock):
    inputs = lock["inputs"]
    need(type(inputs) is dict and set(inputs) == set(INPUT_ROLES), "INPUT_SET")

    lens = inputs["lens"]
    need(type(lens) is dict, "LENS_PIN")
    need(lens.get("revision") == io.LENS_PIN["revision"], "LENS_REVISION")
    need(lens.get("filename") == io.LENS_PIN["filename"], "LENS_FILENAME")
    need(type(lens.get("cache_root")) is str and lens["cache_root"], "LENS_CACHE_ROOT")
    lens_path = io.contained_path(lens.get("path"), lens["cache_root"])
    need(lens_path.suffix == ".pt", "LENS_SUFFIX")
    io.inspect_lens(str(lens_path), lens, full_hash=True)

    snapshot_pin = inputs["model_snapshot_lock"]
    need(type(snapshot_pin) is dict and set(snapshot_pin) == {"path", "sha256"}, "INPUT_PIN_SCHEMA")
    need(snapshot_pin["path"].startswith(NATIVE_STUDY.as_posix() + "/"), "INPUT_SCOPE")
    need("private" not in snapshot_pin["path"].split("/"), "INPUT_SCOPE")
    snapshot_raw = runner.pinned(root, snapshot_pin)
    snapshot = runner.strict_json(snapshot_raw)
    need(
        snapshot.get("schema") == "snapshot_lock.v1"
        and snapshot.get("revision") == io.MODEL_REVISION
        and snapshot.get("scientific_execution_authorized") is False,
        "SNAPSHOT_LOCK",
    )

    train_pin = inputs["train_manifest"]
    need(type(train_pin) is dict and set(train_pin) == {"path", "sha256"}, "INPUT_PIN_SCHEMA")
    need(train_pin["path"].startswith(NATIVE_STUDY.as_posix() + "/"), "INPUT_SCOPE")
    need("private" not in train_pin["path"].split("/"), "INPUT_SCOPE")
    train = runner.strict_json(runner.pinned(root, train_pin))
    cases = train.get("cases")
    need(
        train.get("split") == "TRAIN"
        and train.get("case_count") == io.ADMITTED_TRAIN
        and type(cases) is list
        and len(cases) == io.ADMITTED_TRAIN,
        "TRAIN_MANIFEST",
    )
    train_ids = {case.get("case_id") for case in cases}
    need(len(train_ids) == io.ADMITTED_TRAIN and None not in train_ids, "TRAIN_MANIFEST")

    prompt = inputs["prompt"]
    need(type(prompt) is dict and set(prompt) == PROMPT_FIELDS, "PROMPT_SCHEMA")
    need(prompt["case_id"] in train_ids, "PROMPT_NOT_TRAIN")
    need(type(prompt["text"]) is str and prompt["text"], "PROMPT_TEXT")
    need(runner.sha(prompt["text"].encode("utf-8")) == runner.digest(prompt["sha256"]), "PROMPT_HASH")
    return dict(prompt=prompt, train=train, snapshot_lock=snapshot), {"model_snapshot_lock": snapshot_raw}


def preflight(lock_path, expected_sha256, root=ROOT):
    need(
        not any(name.split(".")[0] in FORBIDDEN_ROOTS for name in sys.modules),
        "NATIVE_IMPORT_BEFORE_PREFLIGHT",
    )
    root = Path(root).resolve()
    raw = runner.small_bytes(lock_path)
    need(runner.sha(raw) == runner.digest(expected_sha256), "LOCK_DIGEST")
    lock = runner.strict_json(raw)
    need(lock.get("schema") == SCHEMA and lock.get("scientific_execution_authorized") is True, "LOCK_NOT_AUTHORIZED")
    need(lock.get("release") == io.RELEASE_V1, "RELEASE_NOT_AUTHORIZED")
    need(type(lock.get("run_id")) is str and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", lock["run_id"]), "RUN_ID")
    need(lock.get("caps") == CAPS, "CAPS")
    sources = lock.get("source_files")
    need(type(sources) is dict and set(sources) == set(SOURCE_PATHS), "SOURCE_SET")
    runner.check_sources(root, lock.get("source_commit"), sources)
    provider_paths = _runtime(lock)
    threads = lock["threads"]
    need(
        set(threads) == {"intra", "inter"}
        and all(type(v) is int and 0 < v <= (os.cpu_count() or 1) for v in threads.values()),
        "THREAD_SETTINGS",
    )
    data, raws = _inputs(root, lock)
    need(type(lock.get("snapshot_cache_root")) is str and lock["snapshot_cache_root"], "SNAPSHOT_CACHE_ROOT")
    base = (root / STUDY / "runs").resolve()
    need(base.is_relative_to(root), "OUTPUT_SCOPE")
    marker = (root / NATIVE_STUDY / "runs" / OWNER_NAME).resolve()
    need(marker.is_relative_to(root), "OWNER_SCOPE")
    return dict(
        lock=lock,
        lock_sha256=expected_sha256,
        root=root,
        data=data,
        raws=raws,
        provider_paths=provider_paths,
        base=base,
        output=base / lock["run_id"],
        marker=marker,
    )


# --------------------------------------------------------------------------- #
# verified snapshot + release-gated tensor load
# --------------------------------------------------------------------------- #
def verify_snapshot(ctx):
    return io.snapshot_verifier.verify_snapshot(
        ctx["raws"]["model_snapshot_lock"],
        expected_lock_sha256=ctx["lock"]["inputs"]["model_snapshot_lock"]["sha256"],
        allowed_root=ctx["lock"]["snapshot_cache_root"],
        max_files=10,
        max_file_bytes=1746942600,
        max_total_bytes=1769905646,
        deadline_seconds=300,
    )


def load_tensors(ctx, proof, token_ids):
    """Release-gated lens + float32 norm + selected unembedding rows.

    Only three Jacobian matrices, the 1024-wide norm vector and <= 6 unembedding
    rows are retained; the safetensors file is seeked per row and the whole
    ``248320 x 1024`` matrix is never read.
    """
    lock = ctx["lock"]
    lens_pin = lock["inputs"]["lens"]
    jacobians = io.load_lens_jacobians(
        str(io.contained_path(lens_pin["path"], lens_pin["cache_root"])), lens_pin, release=lock["release"], loader=io.torch_weights_only_loader)
    files = {item["name"]: item for item in proof["checked_files"]}
    shards = [name for name in files if name.endswith(".safetensors")]
    need(len(shards) == 1, "MODEL_SINGLE_SHARD")
    name = shards[0]
    model_file = io.contained_path(str(Path(proof["snapshot_realpath"]) / name), lock["snapshot_cache_root"])
    model_pin = {"bytes": files[name]["bytes"], "sha256": files[name]["sha256"]}
    io.strict_file(model_file, model_pin, full_hash=True)
    norm = io.load_norm_weight(model_file, model_pin, release=lock["release"], full_hash=False)
    rows = io.load_unembed_rows(model_file, model_pin, token_ids, release=lock["release"], full_hash=False)
    return {"jacobians": jacobians, "norm": norm, "rows": rows}


# --------------------------------------------------------------------------- #
# parity capture: one forward, same object, tolerance-checked, fail closed
# --------------------------------------------------------------------------- #
def capture(ctx, started, *, verify=verify_snapshot, build=None, tensors=None):
    caps = ctx["lock"]["caps"]
    need(caps == CAPS, "CAPS")
    proof = verify(ctx)
    need(type(proof) is dict and proof.get("snapshot_realpath"), "SNAPSHOT_PROOF")
    Path(ctx["output"]).mkdir(parents=True, exist_ok=True)

    adapter = (build or build_parity_adapter)(ctx, proof["snapshot_realpath"])
    prompt = ctx["data"]["prompt"]
    ids = adapter.encode(prompt["text"])
    need(type(ids) in (list, tuple) and len(ids) > 0, "PROMPT_TOKENS")
    result = adapter.run_once(list(ids))
    need(type(result) is dict, "PARITY_RESULT")
    need(result.get("forward_count") == 1 and result.get("fits", 0) == 0, "FORWARD_BUDGET")
    need(result.get("model_loads") == 1 and result.get("tokenizer_loads") == 1, "BUILD_BUDGET")
    need(
        type(result.get("working_memory_gib")) in (int, float)
        and result["working_memory_gib"] <= caps["working_memory_gib"],
        "WORKING_MEMORY",
    )
    need(result.get("same_model_object") is True, "SECOND_MODEL")
    if result.get("same_model_hook_available") is not True:
        raise GateError("SAME_MODEL_HOOK_UNAVAILABLE: " + MINIMAL_ALTERNATIVE)

    native, bridge = result["native_layers"], result["bridge_layers"]
    need(set(native) == set(BLOCKS) and set(bridge) == set(BLOCKS), "LAYER_ALIGNMENT")
    selected = io.validate_token_ids(result["selected_token_ids"])
    contract = core.ReadoutContract(d_model=io.D_MODEL, token_ids=tuple(selected), vocab_size=io.VOCAB_SIZE)
    if tensors is None:
        tensors = load_tensors(ctx, proof, selected)
    jacobians, norm, rows = tensors["jacobians"], tensors["norm"], tensors["rows"]
    need(set(jacobians) == set(BLOCKS), "JACOBIAN_LAYERS")
    need(getattr(norm, "shape", None) == (io.D_MODEL,), "NORM_SHAPE")
    need(getattr(rows, "shape", None) == (len(selected), io.D_MODEL), "ROWS_SHAPE")

    hook_errors, readout_errors = {}, {}
    for layer in BLOCKS:
        layer_native = np.asarray(native[layer], dtype=np.float32)
        layer_bridge = np.asarray(bridge[layer], dtype=np.float32)
        need(
            layer_native.ndim == 2
            and layer_native.shape == layer_bridge.shape
            and layer_native.shape[1] == io.D_MODEL,
            "HOOK_SHAPE",
        )
        if TOLERANCES["hook_require_float32"]:
            need(
                result["native_dtypes"][layer] == "float32" and result["bridge_dtypes"][layer] == "float32",
                "HOOK_DTYPE",
            )
        hook_error = float(np.max(np.abs(layer_native - layer_bridge))) if layer_native.size else 0.0
        need(hook_error <= TOLERANCES["hook_max_abs"], "HOOK_TOLERANCE")
        hook_errors[layer] = hook_error

        hidden = layer_native[-1]
        core_logits = core.raw_direct_logit(hidden, jacobians[layer], norm, rows, contract)
        reference = np.asarray(adapter.lens_api_readout(layer, layer_native, selected), dtype=np.float32)
        need(core_logits.shape == reference.shape, "READOUT_ALIGNMENT")
        readout_error = float(np.max(np.abs(core_logits - reference))) if core_logits.size else 0.0
        need(readout_error <= TOLERANCES["readout_max_abs"], "READOUT_TOLERANCE")
        readout_errors[layer] = readout_error

    need(time.monotonic() - started <= caps["seconds"], "DEADLINE")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "job_id": JOB_ID,
        "status": "parity_complete",
        "lock_sha256": ctx["lock_sha256"],
        "run_id": ctx["lock"]["run_id"],
        "prompt": {"case_id": prompt["case_id"], "sha256": prompt["sha256"]},
        "layers": list(BLOCKS),
        "selected_token_ids": selected,
        "hook_max_abs": hook_errors,
        "readout_max_abs": readout_errors,
        "tolerances": dict(TOLERANCES),
        "counters": {"model_loads": 1, "tokenizer_loads": 1, "forwards": 1, "fits": 0, "derivatives": 0},
        "caps": dict(caps),
        "adapter_provenance": adapter.provenance,
        "notes": {
            "one_model_object": True,
            "one_forward": True,
            "outcome_used": False,
            "readout_is_not_probability": True,
            "second_model_or_forward": False,
        },
    }
    raw = runner.encoded(receipt)
    runner.write_new(ctx["output"] / ARTIFACTS[0], raw)
    return {ARTIFACTS[0]: {"bytes": len(raw), "sha256": runner.sha(raw)}}


# --------------------------------------------------------------------------- #
# real provider build (release/worker path only; never in tests)
# --------------------------------------------------------------------------- #
class ParityAdapter:
    """Wraps one loaded model+tokenizer+bridge for exactly one forward."""

    def __init__(self, torch, model, tokenizer, bridge, lens, jacobian_module, surfaces):
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.bridge = bridge
        self.lens = lens
        self.jacobian_module = jacobian_module
        self.surfaces = surfaces
        self.hook_names = {layer: "blocks.%d.hook_out" % layer for layer in BLOCKS}
        self.working_memory_gib = float(
            sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 3))
        self.provenance = {
            "job_id": JOB_ID,
            "model_repo": io.MODEL_REPO,
            "model_revision": io.MODEL_REVISION,
            "bridge_blocks": list(BLOCKS),
            "bridge_hook": "blocks.{layer}.hook_out",
            "same_model_object": True,
            "same_forward": True,
            "second_model_loaded": False,
            "generation_or_steering": False,
            "surface_count": len(surfaces),
        }

    def encode(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def selected_token_ids(self):
        ids = []
        for surface in self.surfaces:
            encoded = self.tokenizer.encode(surface, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] not in ids:
                ids.append(encoded[0])
        return ids

    def run_once(self, input_ids):
        captured = {}
        handles = []
        for layer in BLOCKS:
            component = self.bridge.blocks[layer].original_component

            def hook(module, inputs, output, layer=layer):
                captured[layer] = output[0] if isinstance(output, tuple) else output

            handles.append(component.register_forward_hook(hook))
        try:
            tokens = self.torch.tensor([list(input_ids)], dtype=self.torch.long)
            with self.torch.no_grad():
                _, cache = self.bridge.run_with_cache(
                    tokens, names_filter=lambda name: name in set(self.hook_names.values()))
        finally:
            for handle in handles:
                handle.remove()
        native_layers, bridge_layers, native_dtypes, bridge_dtypes = {}, {}, {}, {}
        for layer in BLOCKS:
            native = captured[layer].detach().float().cpu().numpy()
            bridge = cache[self.hook_names[layer]].detach().float().cpu().numpy()
            # One 1-row batch is submitted, so both recordings are (1, seq, d_model);
            # capture()'s contract is the unbatched (seq, d_model) layer output.
            native_layers[layer] = native[0] if native.ndim == 3 else native
            bridge_layers[layer] = bridge[0] if bridge.ndim == 3 else bridge
            native_dtypes[layer] = str(captured[layer].dtype).replace("torch.", "")
            bridge_dtypes[layer] = str(cache[self.hook_names[layer]].dtype).replace("torch.", "")
        return dict(
            forward_count=1,
            fits=0,
            model_loads=1,
            tokenizer_loads=1,
            same_model_object=True,
            same_model_hook_available=True,
            working_memory_gib=self.working_memory_gib,
            native_layers=native_layers,
            bridge_layers=bridge_layers,
            native_dtypes=native_dtypes,
            bridge_dtypes=bridge_dtypes,
            selected_token_ids=self.selected_token_ids(),
        )

    def lens_api_readout(self, layer, hidden, token_ids):
        """Published JacobianLens math on the SAME captured hidden state (no forward)."""
        tensor = self.torch.tensor(np.asarray(hidden, dtype=np.float32)[-1])
        transported = self.lens.transport(tensor, layer)
        with self.torch.no_grad():
            logits = self.jacobian_module._unembed(self.bridge, transported)
        return logits[list(token_ids)].detach().float().cpu().numpy()


def build_parity_adapter(ctx, snapshot_dir):
    """Load one model/tokenizer, wrap the SAME object, run nothing here."""
    modules = {role: importlib.import_module(name) for role, (name, _) in PROVIDERS.items()}
    for role, module in modules.items():
        need(Path(module.__file__).resolve() == ctx["provider_paths"][role], "PROVIDER_IMPORT_ORIGIN")
        need(
            runner.sha(runner.small_bytes(module.__file__)) == runner.digest(
                ctx["lock"]["runtime"]["provider_sources"][role]),
            "PROVIDER_CHANGED",
        )
    torch = modules["torch"]
    torch.set_num_threads(ctx["lock"]["threads"]["intra"])
    torch.set_num_interop_threads(ctx["lock"]["threads"]["inter"])
    snapshot = Path(snapshot_dir)
    model = modules["model"].Qwen3_5ForConditionalGeneration.from_pretrained(
        str(snapshot),
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
        weights_only=True,
        use_safetensors=True,
    )
    model.eval()
    tokenizer = modules["tokenizer"].Qwen2Tokenizer.from_pretrained(str(snapshot), local_files_only=True)
    bridge = modules["bridge"].TransformerBridge.boot_transformers(
        io.MODEL_REPO, hf_model=model, tokenizer=tokenizer, device="cpu", dtype=torch.float32)
    need(bridge.original_model is model, "SECOND_MODEL")
    need(bridge.cfg.n_layers == io.N_LAYERS, "BRIDGE_LAYERS")
    need(hasattr(bridge, "run_with_cache"), "SAME_MODEL_HOOK_UNAVAILABLE: " + MINIMAL_ALTERNATIVE)
    for layer in BLOCKS:
        need(
            getattr(bridge.blocks[layer], "original_component", None) is not None,
            "SAME_MODEL_HOOK_UNAVAILABLE: " + MINIMAL_ALTERNATIVE,
        )
    lens_pin = ctx["lock"]["inputs"]["lens"]
    lens = io.load_lens_jacobians(
        lens_pin["path"], lens_pin, release=ctx["lock"]["release"], loader=io.torch_weights_only_loader)
    lens = modules["jacobian_lens"].JacobianLens(
        {layer: torch.tensor(matrix) for layer, matrix in lens.items()},
        n_prompts=0,
        d_model=io.D_MODEL,
    )
    return ParityAdapter(torch, model, tokenizer, bridge, lens, modules["jacobian_lens"], core.CONCEPT_SURFACES_V1)


# --------------------------------------------------------------------------- #
# watched worker + parent supervisor (shared native owner lock preserved)
# --------------------------------------------------------------------------- #
def _verify_pins(ctx, pins):
    need(set(pins) == set(ARTIFACTS), "ARTIFACT_SET")
    total = 0
    for name, pin in pins.items():
        path = ctx["output"] / name
        need(path.is_file() and not path.is_symlink() and path.stat().st_size == pin["bytes"], "ARTIFACT_SIZE")
        total += pin["bytes"]
        need(total <= CAPS["output_bytes"] - STATUS_RESERVE, "OUTPUT_CAP")
        need(runner.sha(path.read_bytes()) == pin["sha256"], "ARTIFACT_HASH")


def worker(lock_path, expected, token):
    started = time.monotonic()
    need(re.fullmatch(r"[0-9a-f]{32}", token or ""), "WORKER_TOKEN")
    ctx = preflight(lock_path, expected)
    ctx["base"].mkdir(parents=True, exist_ok=True)
    ctx["marker"].parent.mkdir(parents=True, exist_ok=True)
    owner = dict(pid=os.getpid(), token=token, run_id=ctx["lock"]["run_id"])
    runner.write_new(ctx["marker"], runner.encoded(owner))
    try:
        ctx["output"].mkdir()  # Exclusive: old outputs, even empty, refuse a retry.
        with contextlib.redirect_stdout(sys.stderr):
            pins = capture(ctx, started)
        _verify_pins(ctx, pins)
        report = dict(status="worker_complete", lock_sha256=expected, outputs=pins, **owner)
        runner.write_new(ctx["output"] / WORKER_RECEIPT, runner.encoded(report))
        return report
    finally:
        runner.release_owner(ctx["marker"], os.getpid(), token, owner["run_id"])


def supervise(lock_path, expected):
    started = time.monotonic()
    ctx = preflight(lock_path, expected)
    need(not ctx["output"].exists() and not ctx["marker"].exists(), "OWNER_OR_OUTPUT_EXISTS")
    ctx["base"].mkdir(parents=True, exist_ok=True)
    ctx["marker"].parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--lock",
        str(Path(lock_path).resolve()),
        "--sha256",
        expected,
        "--token",
        token,
    ]
    child_pid = None
    child_closed = False
    try:
        child_pid, _ = runner.watch(command, str(ctx["root"]), CAPS["seconds"] - (time.monotonic() - started))
        child_closed = True
        complete = runner.strict_json(runner.small_bytes(ctx["output"] / WORKER_RECEIPT))
        need(
            all(
                complete.get(k) == v
                for k, v in dict(
                    status="worker_complete",
                    pid=child_pid,
                    token=token,
                    run_id=ctx["lock"]["run_id"],
                    lock_sha256=expected,
                ).items()
            ),
            "WORKER_RECEIPT_BINDING",
        )
        _verify_pins(ctx, complete["outputs"])
        need(time.monotonic() - started <= CAPS["seconds"], "HARD_TIMEOUT")
        receipt = dict(complete, status="complete", controller_pid=os.getpid(),
                       elapsed_seconds=time.monotonic() - started)
        runner.write_new(ctx["output"] / SUCCESS_RECEIPT, runner.encoded(receipt))
        return receipt
    except BaseException as exc:
        child_pid = getattr(exc, "owned_child_pid", child_pid)
        child_closed = getattr(exc, "owned_child_closed", child_closed)
        failure = dict(status="failed", run_id=ctx["lock"]["run_id"], controller_pid=os.getpid(),
                       child_pid=child_pid, code=str(exc)[:1024])
        runner.write_new(ctx["base"] / ("controller_failure_" + token + ".json"), runner.encoded(failure))
        raise
    finally:
        if child_pid is not None and child_closed:
            runner.release_owner(ctx["marker"], child_pid, token, ctx["lock"]["run_id"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--sha256", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            result = worker(args.lock, args.sha256, args.token)
        elif args.run:
            result = supervise(args.lock, args.sha256)
        else:
            ctx = preflight(args.lock, args.sha256)
            result = dict(status="preflight_pass", run_id=ctx["lock"]["run_id"],
                          native_execution_performed=False, forwards=0, fits=0)
        print(runner.encoded(result).decode().strip())
        return 0
    except Exception as exc:
        print(runner.encoded(dict(status="failed", code=type(exc).__name__, detail=str(exc)[:1024])).decode().strip())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
