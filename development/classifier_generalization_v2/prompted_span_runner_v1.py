"""Root versioned extension of the reviewed span runner: prompted condition only.
Old source/caps remain unchanged. Numeric window schema is generic; prompted
render, binding, index, receipt and query pins distinguish this condition.

Minimal span-capture runner + binary window writer V1 (model-free parent).

This is the integration layer between the reviewed
``native_development_runner_v2`` supervisor helpers and ``span_capture_adapter_v1``.
It imports no ``torch``/``transformers``, performs no capture, no fit, no
dataset/feature/holdout read at import time, and no network I/O. Every real
provider is imported only inside the watched worker *after* preflight and after
the snapshot bytes are verified, exactly as the reviewed V2 runner does.

Scope
-----
* One new finite lock schema ``prompted_span_development_execution.v1``:
  640 forwards, 1800 seconds, 1 model load, 1 tokenizer load, 0 fits,
  0 derivatives, rough raw cap 125829120 B and total output cap 201326592 B.
* The 320 logical development cases come from the two already-existing
  320-forward V2 lock *metadata* files, which are validated through the reviewed
  ``runner.preflight`` (old lock preflight only). Their captures are never run
  and their output run_ids are never reused; ``run_id`` reuse is rejected.
* One native model build, wrapped in ``SpanCaptureAdapter``, never two builds.
* AB/BA ``capture_window`` calls are streamed as little-endian float32 window
  records to one exclusive binary file, with a JSON index and SHA256 pins. No
  JSON float arrays are written and no all-window accumulation occurs.

Binary / index contract for the future fit loader
-------------------------------------------------
``windows.f32`` holds one concatenated stream of little-endian float32 values
with no header. Record order is: case order x view order (AB, BA) x block order
(6, 10, 18). Each record is ``window_length`` rows of ``WIDTH`` (1024) float32
values, row-major, with no padding; ``window_length`` is
``min(16, readout_index + 1)`` and short windows are explicit. ``index.json``
carries, for every record, ``case_id``, ``order``, ``block``, ``shape``,
``dtype``/``byte_order``/``float_bytes``, ``token_positions``,
``readout_index``, ``final_input_index``, ``prefix_sha256``,
``input_ids_sha256``, ``offset``, ``length`` and ``sha256``. The fit loader
must read the index, never assume a fixed offset, and must not pass this file
or its provider provenance as authenticated native evidence.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import re
import struct
import sys
import time
import uuid

import native_capture_contract as contract
import native_development_runner_v2 as runner
import span_capture_adapter_v1 as span
import prompted_input_adapter_v1 as tokens

__all__ = [
    "GateError",
    "need",
    "SpanRunnerError",
    "preflight",
    "capture",
    "worker",
    "supervise",
    "main",
    "verify_snapshot",
    "build_span_adapter",
    "CAPS",
    "SCHEMA",
    "JOB_ID",
    "ARTIFACTS",
]

JOB_ID = "prompted_span_runner_root_20260914"
SCHEMA = "prompted_span_development_execution.v1"
INDEX_SCHEMA = "prompted_span_capture_index.v1"
RECEIPT_SCHEMA = "prompted_span_capture_receipt.v1"
BINARY_SCHEMA = "prompted_span_capture_windows.v1"
STUDY = runner.STUDY
REVISION = runner.REVISION
ORDERS = ("AB", "BA")
LABEL_TOKEN_IDS = {"A": 32, "B": 33}
BLOCKS = span.BLOCKS
WIDTH = span.WIDTH
WINDOW = span.MAX_WINDOW
FLOAT32_MAX = 3.4028234663852886e38
RAW_BYTES = 640 * len(BLOCKS) * WINDOW * WIDTH * 4
OUTPUT_BYTES = 192 * 1024 * 1024
CAPS = dict(
    seconds=1800,
    forwards=640,
    tokens_per_view=320,
    model_loads=1,
    tokenizer_loads=1,
    fits=0,
    derivatives=0,
    raw_bytes=RAW_BYTES,
    output_bytes=OUTPUT_BYTES,
)
SOURCE_NAMES = (
    "prompted_span_runner_v1.py",
    "native_development_runner_v2.py",
    "span_capture_adapter_v1.py",
    "native_capture_adapter.py",
    "native_capture_contract.py",
    "prompted_input_adapter_v1.py",
    "snapshot_verifier.py",
)
INPUT_ROLES = ("blueprint", "model_metadata", "snapshot_lock", "prompt_sanity")
REFERENCE_FIELDS = ("path", "sha256", "run_id", "forwards")
REFERENCE_FORWARDS = runner.CAPS["forwards"]
ARTIFACTS = ("windows.f32", "index.json", "capture_receipt.json")
TRAIN_CASES = 240
VALIDATION_CASES = 80
TOTAL_CASES = TRAIN_CASES + VALIDATION_CASES
TOTAL_VIEWS = TOTAL_CASES * len(ORDERS)
MAX_SMALL_FILE = runner.MAX_SMALL_FILE
STATUS_RESERVE = runner.STATUS_RESERVE
CHUNK = 1 << 20

GateError = runner.GateError


class SpanRunnerError(runner.GateError):
    """Same error family as the reviewed runner, for callers that want it."""


def need(condition, code, detail=""):
    if not condition:
        raise GateError(code if not detail else "%s: %s" % (code, detail))


# --------------------------------------------------------------------------- #
# model-free preflight: new lock, then old-lock preflight only (never capture)
# --------------------------------------------------------------------------- #
def _runtime(ctx_lock, root):
    runtime = ctx_lock["runtime"]
    prefix = runner.runtime_prefix()
    need(Path(runtime["prefix"]).resolve() == prefix, "RUNTIME_PREFIX")
    need(runtime["python"] == ".".join(map(str, sys.version_info[:3])), "PYTHON_VERSION")
    need(set(runtime["packages"]) == runner.PACKAGES, "RUNTIME_PACKAGE_SET")
    for name, version in runtime["packages"].items():
        need(importlib.metadata.version(name) == version, "RUNTIME_VERSION")
    need(set(runtime["provider_sources"]) == set(runner.PROVIDERS), "PROVIDER_SOURCE_SET")
    provider_paths = {}
    for role, (_, relative) in runner.PROVIDERS.items():
        path = runner.relative_path(prefix, "Lib/site-packages/" + relative)
        need(
            runner.sha(runner.small_bytes(path))
            == runner.digest(runtime["provider_sources"][role]),
            "PROVIDER_SOURCE_HASH",
        )
        provider_paths[role] = path
    return provider_paths


def _load_inputs(lock, root):
    need(set(lock["inputs"]) == set(INPUT_ROLES), "INPUT_SET")
    raws, data = {}, {}
    for role, pin in lock["inputs"].items():
        need(pin["path"].startswith(STUDY.as_posix() + "/"), "INPUT_SCOPE")
        need("private" not in pin["path"].split("/"), "INPUT_SCOPE")
        path = runner.relative_path(root, pin["path"])
        need("private" not in [p.lower() for p in path.relative_to(root).parts], "INPUT_SCOPE")
        raws[role] = runner.pinned(root, pin)
        data[role] = runner.strict_json(raws[role])
    metadata = data["model_metadata"]
    need(
        metadata.get("schema") == "native_model_metadata.v1"
        and metadata.get("revision") == REVISION
        and len(metadata.get("checkpoint_keys", {})) == 488,
        "MODEL_METADATA",
    )
    snapshot = data["snapshot_lock"]
    need(
        snapshot.get("schema") == "snapshot_lock.v1"
        and snapshot.get("revision") == REVISION
        and snapshot.get("scientific_execution_authorized") is False,
        "SNAPSHOT_LOCK",
    )
    return raws, data


def _old_context(root, reference):
    need(type(reference) is dict and set(reference) == set(REFERENCE_FIELDS), "REFERENCE_LOCK_SCHEMA")
    need(reference["forwards"] == REFERENCE_FORWARDS, "REFERENCE_LOCK_FORWARDS")
    path = runner.relative_path(root, reference["path"])
    raw = runner.small_bytes(path)
    need(runner.sha(raw) == runner.digest(reference["sha256"]), "REFERENCE_LOCK_DIGEST")
    old = runner.strict_json(raw)
    need(
        old.get("schema") == runner.SCHEMA and old.get("scientific_execution_authorized") is True,
        "REFERENCE_LOCK_NOT_AUTHORIZED",
    )
    need(old.get("run_id") == reference["run_id"], "REFERENCE_LOCK_RUN_ID")
    need(old.get("caps") == runner.CAPS, "REFERENCE_LOCK_CAPS")
    need(old["caps"]["forwards"] == reference["forwards"], "REFERENCE_LOCK_FORWARDS")
    # Old lock preflight only: validated metadata + input pins, never its capture.
    return runner.preflight(path, reference["sha256"], root)


def _combine(old_contexts):
    cases, ids, held = [], set(), set()
    splits = {"TRAIN": 0, "VALIDATION": 0}
    labels = {label: 0 for label in runner.LABELS}
    run_ids = set()
    for ctx in old_contexts:
        run_ids.add(ctx["lock"]["run_id"])
        for case in ctx["cases"]:
            need(case["split"] in ("TRAIN", "VALIDATION"), "SPLIT")
            need(case["case_id"] not in ids, "DUPLICATE_CASE")
            ids.add(case["case_id"])
            splits[case["split"]] += 1
            labels[case["class_label"]] += 1
            cases.append(case)
        held.update(ctx["data"]["holdout_index"]["case_ids"])
    need(len(run_ids) == len(old_contexts), "DUPLICATE_REFERENCE_RUN_ID")
    need(not ids.intersection(held), "HOLDOUT_OVERLAP")
    need(len(cases) == TOTAL_CASES, "CASE_ACCOUNTING")
    need(splits == {"TRAIN": TRAIN_CASES, "VALIDATION": VALIDATION_CASES}, "CASE_ACCOUNTING")
    need(labels == {label: TOTAL_CASES // 4 for label in runner.LABELS}, "CLASS_BALANCE")
    return cases


def preflight(lock_path, expected_sha256, root=runner.ROOT):
    need(
        not any(name.split(".")[0] in runner.PACKAGES for name in sys.modules),
        "NATIVE_IMPORT_BEFORE_PREFLIGHT",
    )
    root = Path(root).resolve()
    raw = runner.small_bytes(lock_path)
    need(runner.sha(raw) == runner.digest(expected_sha256), "LOCK_DIGEST")
    lock = runner.strict_json(raw)
    need(lock.get("schema") == SCHEMA and lock.get("scientific_execution_authorized") is True, "LOCK_NOT_AUTHORIZED")
    need(type(lock.get("run_id")) is str and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", lock["run_id"]), "RUN_ID")
    need(lock.get("caps") == CAPS and all(type(v) is int for v in lock["caps"].values()), "CAPS")
    sources = lock.get("source_files")
    need(type(sources) is dict and set(sources) == {(STUDY / n).as_posix() for n in SOURCE_NAMES}, "SOURCE_SET")
    runner.check_sources(root, lock.get("source_commit"), sources)
    provider_paths = _runtime(lock, root)
    threads = lock["threads"]
    need(
        set(threads) == {"intra", "inter"}
        and all(type(v) is int and 0 < v <= (os.cpu_count() or 1) for v in threads.values()),
        "THREAD_SETTINGS",
    )
    raws, data = _load_inputs(lock, root)
    references = lock.get("reference_locks")
    need(type(references) is list and len(references) == 2, "REFERENCE_LOCKS")
    for reference in references:
        # Old lock metadata must never name this run's output directory.
        need(type(reference) is dict, "REFERENCE_LOCK_SCHEMA")
        need(reference.get("run_id") != lock["run_id"], "OLD_RUN_ID_REUSE")
    old_contexts = [_old_context(root, reference) for reference in references]
    cases = _combine(old_contexts)
    sanity = data["prompt_sanity"]
    need(sanity.get("status") == "PASS", "PROMPT_SANITY")
    need(sanity.get("query_sha256") == tokens.QUERY_SHA256, "QUERY_PIN")
    need(sanity.get("adapter_sha256") == sources[(STUDY / "prompted_input_adapter_v1.py").as_posix()], "SANITY_SOURCE")
    need(sanity.get("reference_locks") == references, "SANITY_REFERENCES")
    need(sanity.get("max_full_tokens", 10**9) <= CAPS["tokens_per_view"], "SANITY_TOKEN_CAP")
    need(sanity.get("last_shared_token_ids") == [contract.LAST_SHARED_ID], "SANITY_BOUNDARY")
    need(sanity.get("exact_decode_reencode") is True and sanity.get("query_in_shared_prefix") is True, "SANITY_ROUNDTRIP")
    need(sanity.get("counts") == dict(cases=320, views=640, tokenizer_loads=1, model_loads=0, forwards=0, fits=0), "SANITY_COUNTS")
    rows = sanity.get("rows", [])
    need(len(rows) == 640 and {(v["case_id"], v["order"]) for v in rows} == {(c["case_id"], o) for c in cases for o in ORDERS}, "SANITY_CASE_BINDING")
    base = (root / STUDY / "runs").resolve()
    need(base.is_relative_to(root), "OUTPUT_SCOPE")
    return dict(
        lock=lock,
        lock_sha256=expected_sha256,
        root=root,
        cases=cases,
        data=data,
        raws=raws,
        provider_paths=provider_paths,
        reference_contexts=old_contexts,
        base=base,
        output=base / lock["run_id"],
        marker=base / "native_model_owner.json",
    )


# --------------------------------------------------------------------------- #
# one native build wrapped in SpanCaptureAdapter
# --------------------------------------------------------------------------- #
def verify_snapshot(ctx):
    verifier = runner.local_module("snapshot_verifier", ctx)
    return verifier.verify_snapshot(
        ctx["raws"]["snapshot_lock"],
        expected_lock_sha256=ctx["lock"]["inputs"]["snapshot_lock"]["sha256"],
        allowed_root=ctx["lock"]["snapshot_cache_root"],
        max_files=10,
        max_file_bytes=1746942600,
        max_total_bytes=1769905646,
        deadline_seconds=300,
    )


def build_span_adapter(ctx, snapshot_dir):
    native_adapter = runner.real_factory(ctx, snapshot_dir)
    module = runner.local_module("span_capture_adapter_v1", ctx)
    return module.SpanCaptureAdapter(native_adapter)


# --------------------------------------------------------------------------- #
# streaming binary capture
# --------------------------------------------------------------------------- #
def _float32_le(matrix, code):
    need(type(matrix) is list, code)
    chunks = []
    for row in matrix:
        need(type(row) in (list, tuple) and len(row) == WIDTH, code, "row width")
        values = []
        for value in row:
            need(type(value) in (int, float) and not isinstance(value, bool), code, "value type")
            number = float(value)
            need(math.isfinite(number) and abs(number) <= FLOAT32_MAX, code, "nonfinite/overflow")
            values.append(number)
        chunks.append(struct.pack("<%df" % WIDTH, *values))
    return b"".join(chunks)


def _validate_boundary(ids, view):
    """Re-check the old contract's shared-prefix/label token boundaries."""
    contract.validate_token_ids(list(ids))
    readout = view["readout_index"]
    need(ids[readout] == contract.LAST_SHARED_ID, "LAST_SHARED_ID")
    need(ids[readout + 1] in contract.LABEL_TOKEN_IDS, "LABEL_BOUNDARY")
    need(view["final_input_index"] == len(ids) - 1, "INDEX")


def _validate_window(result, view, caps):
    need(type(result) is dict, "WINDOW_RESULT")
    need(result.get("schema") == span.WINDOW_SCHEMA, "WINDOW_SCHEMA")
    need(result.get("blocks") == list(BLOCKS), "WINDOW_BLOCKS")
    need(result.get("readout_index") == view["readout_index"], "WINDOW_READOUT")
    need(result.get("final_input_index") == view["final_input_index"], "WINDOW_FINAL")
    need(result.get("width") == WIDTH and result.get("dtype") == "float32", "WINDOW_WIDTH")
    need(result.get("all_positions_unchanged") is True, "WINDOW_FLAGS")
    need(result.get("parameters_unchanged") is True, "WINDOW_FLAGS")
    need(result.get("one_forward_per_capture_call") is True, "WINDOW_FORWARD_FLAG")
    prefix_length = view["readout_index"] + 1
    need(prefix_length <= caps["tokens_per_view"], "PREFIX_TOO_LONG")
    window_length = min(WINDOW, prefix_length)
    start = prefix_length - window_length
    positions = list(range(start, prefix_length))
    need(result.get("shared_prefix_length") == prefix_length, "WINDOW_PREFIX")
    need(result.get("window_length") == window_length, "WINDOW_LENGTH")
    need(result.get("window_start") == start, "WINDOW_START")
    need(result.get("position_indices") == positions, "WINDOW_POSITIONS")
    need(result.get("hook_calls") == {block: 1 for block in BLOCKS}, "WINDOW_HOOKS")
    need(type(result.get("layers")) is dict and set(result["layers"]) == set(BLOCKS), "WINDOW_LAYERS")
    for block in BLOCKS:
        layer = result["layers"][block]
        need(type(layer) is dict, "WINDOW_LAYER")
        need(layer.get("block") == block, "WINDOW_LAYER")
        need(layer.get("shape") == [window_length, WIDTH], "WINDOW_SHAPE")
        need(layer.get("dtype") == "float32", "WINDOW_DTYPE")
        need(layer.get("row_bytes") == WIDTH * 4, "WINDOW_ROW_BYTES")
        need(layer.get("bytes") == window_length * WIDTH * 4, "WINDOW_LAYER_BYTES")
        need(layer.get("position_indices") == positions, "WINDOW_POSITIONS")
        matrix = layer.get("matrix")
        need(type(matrix) is list and len(matrix) == window_length, "WINDOW_MATRIX")


def hash_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(CHUNK)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def capture(ctx, started, *, verify=verify_snapshot, build=build_span_adapter):
    """Stream AB/BA span windows to exclusive artifacts; return artifact pins."""
    caps = ctx["lock"]["caps"]
    cases = ctx["cases"]
    need(type(cases) in (list, tuple) and len(cases) > 0, "CASES_EMPTY")
    # All source/runtime/input pins were checked at preflight; snapshot bytes are
    # verified here before the one provider factory call.
    proof = verify(ctx)
    need(type(proof) is dict and proof.get("snapshot_realpath"), "SNAPSHOT_PROOF")
    identity = proof.get("aggregate_tokenizer_identity_sha256")
    need(type(identity) is str and re.fullmatch(r"[0-9a-f]{64}", identity), "TOKENIZER_IDENTITY")
    adapter = build(ctx, proof["snapshot_realpath"])
    output = Path(ctx["output"])
    output.mkdir(parents=True, exist_ok=True)
    windows_path = output / "windows.f32"
    index_path = output / "index.json"
    receipt_path = output / "capture_receipt.json"
    windows_hash = hashlib.sha256()
    records, raw_total, forwards, views = [], 0, 0, 0
    with windows_path.open("xb") as stream:
        for case in cases:
            need(time.monotonic() - started <= caps["seconds"], "DEADLINE")
            prepared = tokens.prepare_case_inputs(
                case,
                encode=adapter.encode,
                decode=adapter.decode,
                label_token_ids=dict(LABEL_TOKEN_IDS),
                expected_identity_sha256=identity,
                observed_identity_sha256=identity,
                max_tokens=caps["tokens_per_view"],
            )
            binding = prepared["binding"]["bindings"]
            prefix_sha = prepared["provenance"]["prefix_hash"]
            need(prepared["provenance"]["condition"] == tokens.PROMPTED_CONDITION, "PROMPT_CONDITION")
            need(prepared["provenance"]["query_sha256"] == tokens.QUERY_SHA256, "QUERY_PIN")
            for order in ORDERS:
                need(time.monotonic() - started <= caps["seconds"], "DEADLINE")
                view = binding[order]
                ids = list(prepared["input_ids"][order])
                _validate_boundary(ids, view)
                result = adapter.capture_window(
                    input_ids=ids,
                    readout_index=view["readout_index"],
                    final_input_index=view["final_input_index"],
                )
                forwards += 1
                views += 1
                _validate_window(result, view, caps)
                for block in BLOCKS:
                    layer = result["layers"][block]
                    payload = _float32_le(layer["matrix"], "WINDOW_VALUES")
                    need(len(payload) == layer["bytes"], "WINDOW_BYTES")
                    offset = raw_total
                    stream.write(payload)
                    windows_hash.update(payload)
                    raw_total += len(payload)
                    need(raw_total <= caps["raw_bytes"], "RAW_CAP")
                    records.append(
                        {
                            "case_id": prepared["provenance"]["case_id"],
                            "condition": tokens.PROMPTED_CONDITION,
                            "query_sha256": tokens.QUERY_SHA256,
                            "original_context_sha256": prepared["provenance"]["original_context_sha256"],
                            "split": prepared["provenance"]["split"],
                            "order": order,
                            "block": block,
                            "shape": [result["window_length"], WIDTH],
                            "dtype": "float32",
                            "byte_order": "little",
                            "float_bytes": 4,
                            "token_positions": list(layer["position_indices"]),
                            "prefix_length": view["shared_prefix_length"],
                            "readout_index": view["readout_index"],
                            "final_input_index": view["final_input_index"],
                            "prefix_sha256": prefix_sha,
                            "input_ids_sha256": prepared["provenance"]["input_hashes"][order],
                            "offset": offset,
                            "length": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    )
    need(forwards == len(cases) * len(ORDERS), "FORWARD_ACCOUNTING")
    need(views == len(cases) * len(ORDERS), "VIEW_ACCOUNTING")
    index_doc = {
        "schema": INDEX_SCHEMA,
        "condition": tokens.PROMPTED_CONDITION,
        "query_sha256": tokens.QUERY_SHA256,
        "query": tokens.FIXED_QUERY,
        "job_id": JOB_ID,
        "execution_schema": SCHEMA,
        "lock_sha256": ctx["lock_sha256"],
        "run_id": ctx["lock"]["run_id"],
        "binary_file": "windows.f32",
        "binary_schema": BINARY_SCHEMA,
        "blocks": list(BLOCKS),
        "orders": list(ORDERS),
        "record_order": "case_order x orders(AB,BA) x blocks(6,10,18)",
        "dtype": "float32",
        "byte_order": "little",
        "float_bytes": 4,
        "width": WIDTH,
        "window_max": WINDOW,
        "case_order": [case["case_id"] for case in cases],
        "case_count": len(cases),
        "view_count": views,
        "forward_count": forwards,
        "raw_bytes": raw_total,
        "raw_cap_bytes": caps["raw_bytes"],
        "records": records,
    }
    index_raw = runner.encoded(index_doc)
    runner.write_new(index_path, index_raw)
    pins = {
        "windows.f32": {"bytes": raw_total, "sha256": windows_hash.hexdigest()},
        "index.json": {"bytes": len(index_raw), "sha256": runner.sha(index_raw)},
    }
    receipt_doc = {
        "schema": RECEIPT_SCHEMA,
        "condition": tokens.PROMPTED_CONDITION,
        "query_sha256": tokens.QUERY_SHA256,
        "prompt_sanity_sha256": ctx["lock"]["inputs"]["prompt_sanity"]["sha256"],
        "job_id": JOB_ID,
        "status": "capture_complete",
        "lock_sha256": ctx["lock_sha256"],
        "run_id": ctx["lock"]["run_id"],
        "counters": {
            "cases": len(cases),
            "views": views,
            "forwards": forwards,
            "raw_bytes": raw_total,
            "model_loads": 1,
            "tokenizer_loads": 1,
            "fits": 0,
            "derivatives": 0,
        },
        "caps": dict(caps),
        "artifacts": {name: dict(pin) for name, pin in pins.items()},
        "adapter_provenance": adapter.provenance,
        "snapshot_proof": proof,
        "threads": ctx["lock"]["threads"],
        "notes": {
            "float_windows_in_binary_only": True,
            "json_float_arrays_written": False,
            "old_capture_executed": False,
            "old_output_run_id_reused": False,
        },
    }
    receipt_raw = runner.encoded(receipt_doc)
    runner.write_new(receipt_path, receipt_raw)
    pins["capture_receipt.json"] = {"bytes": len(receipt_raw), "sha256": runner.sha(receipt_raw)}
    total = sum(pin["bytes"] for pin in pins.values())
    need(total <= caps["output_bytes"] - STATUS_RESERVE, "OUTPUT_CAP")
    return pins


# --------------------------------------------------------------------------- #
# watched worker + parent supervisor (same ownership model as V2)
# --------------------------------------------------------------------------- #
def _verify_pins(ctx, pins):
    need(set(pins) == set(ARTIFACTS), "ARTIFACT_SET")
    total = 0
    for name, pin in pins.items():
        path = ctx["output"] / name
        need(
            path.is_file() and not path.is_symlink() and path.stat().st_size == pin["bytes"],
            "ARTIFACT_SIZE",
        )
        total += pin["bytes"]
        need(total <= CAPS["output_bytes"] - STATUS_RESERVE, "OUTPUT_CAP")
        need(hash_file(path) == pin["sha256"], "ARTIFACT_HASH")


def worker(lock_path, expected, token):
    started = time.monotonic()
    need(re.fullmatch(r"[0-9a-f]{32}", token or ""), "WORKER_TOKEN")
    ctx = preflight(lock_path, expected)
    ctx["base"].mkdir(parents=True, exist_ok=True)
    owner = dict(pid=os.getpid(), token=token, run_id=ctx["lock"]["run_id"])
    runner.write_new(ctx["marker"], runner.encoded(owner))
    try:
        ctx["output"].mkdir()  # Exclusive: old outputs, even empty, refuse a retry.
        with contextlib.redirect_stdout(sys.stderr):
            pins = capture(ctx, started)
        _verify_pins(ctx, pins)
        report = dict(status="worker_complete", lock_sha256=expected, outputs=pins, **owner)
        runner.write_new(ctx["output"] / "worker_complete.json", runner.encoded(report))
        return report
    finally:
        runner.release_owner(ctx["marker"], owner["pid"], token, owner["run_id"])


def supervise(lock_path, expected):
    started = time.monotonic()
    ctx = preflight(lock_path, expected)
    need(not ctx["output"].exists() and not ctx["marker"].exists(), "OWNER_OR_OUTPUT_EXISTS")
    ctx["base"].mkdir(parents=True, exist_ok=True)
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
        complete = runner.strict_json(runner.small_bytes(ctx["output"] / "worker_complete.json"))
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
        # Parent success receipt only after every output pin is checked.
        _verify_pins(ctx, complete["outputs"])
        need(time.monotonic() - started <= CAPS["seconds"], "HARD_TIMEOUT")
        receipt = dict(
            complete,
            status="complete",
            controller_pid=os.getpid(),
            elapsed_seconds=time.monotonic() - started,
        )
        runner.write_new(ctx["output"] / "supervisor_success.json", runner.encoded(receipt))
        return receipt
    except BaseException as exc:
        child_pid = getattr(exc, "owned_child_pid", child_pid)
        child_closed = getattr(exc, "owned_child_closed", child_closed)
        failure = dict(
            status="failed",
            run_id=ctx["lock"]["run_id"],
            controller_pid=os.getpid(),
            child_pid=child_pid,
            code=str(exc)[:1024],
        )
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
            result = dict(
                status="preflight_pass",
                run_id=ctx["lock"]["run_id"],
                cases=len(ctx["cases"]),
                views=len(ctx["cases"]) * len(ORDERS),
                native_execution_performed=False,
            )
        print(runner.encoded(result).decode().strip())
        return 0
    except Exception as exc:
        print(
            runner.encoded(
                dict(status="failed", code=type(exc).__name__, detail=str(exc)[:1024])
            ).decode().strip()
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
