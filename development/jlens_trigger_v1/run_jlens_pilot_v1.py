"""PLAN_V3 stage-3 cached J-lens pilot supervisor (job ``jlens_pilot_v1``).

MODEL-FREE CONTROL SHELL. This module imports no ``torch``/``transformers``/
``tokenizers``/``safetensors``/``transformer_lens`` at import time and in every
test: the real lens loader is reached only inside the watched worker, after
preflight, through ``jlens_io_v1.torch_weights_only_loader``. It never loads a
model, tokenizer, forward, derivative, PCA, cone or estimator pickle, and it
neither creates nor claims a scientific result.

What the real (still UNAUTHORIZED) run would do
-----------------------------------------------
After a finite root-prepared lock and the zero-fit preflight:
1. authenticate the pinned lens, the two cached activation indexes
   (unprompted + ``prompted_fixed_query_v1``), the four manifests plus group
   blueprint, and the pinned single-token surface table;
2. read only the last shared pre-option position of blocks 6/10/18 from the two
   caches (AB/BA pair-averaged exactly as the reviewed classifier reader does)
   and the three Jacobian matrices, the 1024-wide final-norm vector and at most
   six selected unembedding rows;
3. score the 72 fit-free raw cells and the 24 learned C-cells (132 sklearn
   ``lbfgs`` fits maximum) with only TRAIN-fold thresholds/standardization/C;
4. write exclusive, versioned artifacts and preserve failures.

Caps are ``jlens_pilot_v1.CAPS``: 900 s readout, 600 s fit, 1800 s hard total,
64 MiB output, 2.5 GiB working memory, 0 model/tokenizer/forward/derivative.
The plan stays unauthorized until a finite lock, this preflight and one
independent review exist.
"""

from __future__ import annotations

import argparse
import contextlib
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
import jlens_pilot_v1 as pilot

__all__ = [
    "JOB_ID",
    "SCHEMA",
    "RECEIPT_SCHEMA",
    "CAPS",
    "ARTIFACTS",
    "SOURCE_PATHS",
    "INPUT_ROLES",
    "KNOWN_INPUT_PINS",
    "GateError",
    "need",
    "verify_safe_interfaces",
    "preflight",
    "verify_snapshot",
    "load_tensors",
    "read_hidden",
    "compute_scores",
    "capture",
    "worker",
    "supervise",
    "main",
]

runner = io.runner
GateError = io.GateError
ROOT = Path(__file__).resolve().parents[2]
STUDY = Path("development/jlens_trigger_v1")
NATIVE_STUDY = Path("development/classifier_generalization_v2")

JOB_ID = pilot.JOB_ID
SCHEMA = "jlens_pilot_execution.v1"
RECEIPT_SCHEMA = pilot.RECEIPT_SCHEMA
CAPS = dict(pilot.CAPS)
BLOCKS = tuple(io.BLOCKS)
CONDITIONS = tuple(pilot.CONDITIONS)
METHODS = tuple(pilot.METHODS)
SURFACE_CONDITION = {"unprompted_cache": None, "prompted_cache": "prompted_fixed_query_v1"}
ARTIFACTS = ("cell_results.json", "pilot_receipt.json")
OWNER_NAME = "native_model_owner.json"
WORKER_RECEIPT = "worker_complete.json"
SUCCESS_RECEIPT = "supervisor_success.json"
STATUS_RESERVE = runner.STATUS_RESERVE
FORBIDDEN_ROOTS = {"torch", "transformers", "tokenizers", "safetensors", "transformer_lens"}
SNAPSHOT_LIMITS = dict(
    max_files=10,
    max_file_bytes=1746942600,
    max_total_bytes=1769905646,
    deadline_seconds=300,
)

SOURCE_PATHS = (
    (STUDY / "jlens_pilot_v1.py").as_posix(),
    (STUDY / "run_jlens_pilot_v1.py").as_posix(),
    (STUDY / "jlens_io_v1.py").as_posix(),
    (STUDY / "jlens_core_v2.py").as_posix(),
    (STUDY / "PLAN_V3.json").as_posix(),
    (STUDY / "JLENS_PILOT_PLAN_V1.json").as_posix(),
    (NATIVE_STUDY / "native_development_runner_v2.py").as_posix(),
    (NATIVE_STUDY / "snapshot_verifier.py").as_posix(),
    (NATIVE_STUDY / "harness.py").as_posix(),
    (NATIVE_STUDY / "grouped_driver.py").as_posix(),
    (NATIVE_STUDY / "span_classifier_driver_v1.py").as_posix(),
    (NATIVE_STUDY / "span_feature_transforms_v1.py").as_posix(),
)

INPUT_ROLES = ("lens", "model_snapshot_lock", "unprompted_cache", "prompted_cache", "manifests", "surfaces")
MANIFEST_ROLES = ("original_train", "original_validation", "added_train", "added_validation", "blueprint")

# Resolved input pins carried prospectively in ``JLENS_PILOT_PLAN_V1.json``. The
# finite lock must repeat them; ``verify_safe_interfaces`` checks them early
# without reading a tensor body.
KNOWN_INPUT_PINS = {
    "lens": {
        "repo": io.LENS_PIN["repo"],
        "revision": io.LENS_PIN["revision"],
        "filename": io.LENS_PIN["filename"],
        "bytes": io.LENS_PIN["bytes"],
        "sha256": io.LENS_PIN["sha256"],
        "cache_root": "C:/Users/farha/.cache/huggingface/hub",
        "path": (
            "C:/Users/farha/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/"
            "6bb49967d3c51a12ccb5beac7146f6f5781f9d06/qwen3.5-0.8b/jlens/Salesforce-wikitext/"
            "Qwen3.5-0.8B_jacobian_lens.pt"
        ),
    },
    "unprompted_cache": {
        "condition": None,
        "index": {
            "path": "development/classifier_generalization_v2/runs/span_capture_20260914_v1/index.json",
            "bytes": 1054612,
            "sha256": "10457cf2668d72b1b65968b58fd94113681d028e844ade46f3d52a958c072418",
        },
        "windows": {
            "path": "development/classifier_generalization_v2/runs/span_capture_20260914_v1/windows.f32",
            "bytes": 125829120,
            "sha256": "cc4c756d5bd23f40d0e4781886008c2bdcc4b21b228299746aececaaaf88d103",
        },
    },
    "prompted_cache": {
        "condition": "prompted_fixed_query_v1",
        "index": {
            "path": "development/classifier_generalization_v2/runs/prompted_capture_20260914_v1/index.json",
            "bytes": 1485448,
            "sha256": "8a476cf3d01c77e1d549b9757734bbb8a03edfb60f10e761ed3a20df36fd15ac",
        },
        "windows": {
            "path": "development/classifier_generalization_v2/runs/prompted_capture_20260914_v1/windows.f32",
            "bytes": 125829120,
            "sha256": "6cef755d4da7bfe49a6cad07f9a92b7f1e24c8421f69acad2eabf4a52fedad9b",
        },
    },
    "manifests": {
        "original_train": {
            "path": "development/classifier_generalization_v2/TRAIN_ACCEPTED_V10.json",
            "bytes": 188753,
            "sha256": "60a9ee7bd5c02f8656f495b6cf74a6b9e97a7c6a0df622d78a01c4cdebe52fb9",
        },
        "original_validation": {
            "path": "development/classifier_generalization_v2/VALIDATION_ACCEPTED_V3.json",
            "bytes": 65706,
            "sha256": "0bf23931085c453655a2c8c164b76393077d1254ac096de15c05454f616fd617",
        },
        "added_train": {
            "path": "development/classifier_generalization_v2/EXPANSION_TRAIN_ROOT_V2.json",
            "bytes": 142657,
            "sha256": "beae36ebad963f6af69bdfdb3e2b176a0b57634c1322e35a23f6a28485f5db83",
        },
        "added_validation": {
            "path": "development/classifier_generalization_v2/EXPANSION_VALIDATION_ROOT_V2.json",
            "bytes": 49725,
            "sha256": "1a34cf78c79ed0e74f435248161eed3518d2c85ae79fe560a50b0e8820209ccf",
        },
        "blueprint": {
            "path": "development/classifier_generalization_v2/GROUP_BLUEPRINT_V4.json",
            "bytes": 12140,
            "sha256": "46c0f5abe41e68bdd456ec956c74ad8c01b85085101b3c2ab83fc7ad72cf4593",
        },
    },
}


def need(condition, code, detail=""):
    if not condition:
        raise GateError(code if not detail else "%s: %s" % (code, detail))


def _pin(pin, code="PIN_SCHEMA"):
    need(type(pin) is dict and {"path", "sha256"} <= set(pin), code)
    return pin


def _scope_native(pin, role):
    need(type(pin.get("path")) is str, "PIN_PATH", role)
    need(pin["path"].startswith(NATIVE_STUDY.as_posix() + "/"), "INPUT_SCOPE", role)
    try:
        pilot.reject_forbidden_path(pin["path"])
    except pilot.PilotError as exc:
        raise GateError(str(exc)) from None


# --------------------------------------------------------------------------- #
# model-free preflight: no provider import, no tensor body, no fit
# --------------------------------------------------------------------------- #
def _runtime(lock):
    runtime = lock["runtime"]
    need(
        type(runtime) is dict
        and set(runtime) == {"prefix", "python", "packages", "provider_sources", "classifier_sources"},
        "RUNTIME_SCHEMA",
    )
    need(Path(runtime["prefix"]).resolve() == runner.runtime_prefix(), "RUNTIME_PREFIX")
    need(runtime["python"] == ".".join(map(str, sys.version_info[:3])), "PYTHON_VERSION")
    need(type(runtime["packages"]) is dict and runtime["packages"], "RUNTIME_PACKAGES")
    for name, version in runtime["packages"].items():
        need(type(name) is str and type(version) is str, "RUNTIME_PACKAGE_SCHEMA")
        need(importlib.metadata.version(name) == version, "RUNTIME_VERSION", name)
    need(
        type(runtime["classifier_sources"]) is dict
        and set(runtime["classifier_sources"]) == set(pilot.CLASSIFIER_SOURCES),
        "CLASSIFIER_SOURCE_SET",
    )
    for name, digest in runtime["classifier_sources"].items():
        need(digest == pilot.CLASSIFIER_SOURCES[name], "CLASSIFIER_SOURCE_PIN", name)
        raw = (NATIVE_STUDY / (name + ".py")).read_bytes()
        need(runner.sha(raw) == runner.digest(digest), "CLASSIFIER_SOURCE_HASH", name)
    need(type(runtime["provider_sources"]) is dict, "PROVIDER_SOURCE_SCHEMA")
    provider_paths = {}
    for relative, digest in runtime["provider_sources"].items():
        need(
            type(relative) is str
            and relative.startswith(
                ("numpy/", "scipy/", "sklearn/", "torch/", "transformers/", "tokenizers/",
                 "safetensors/", "transformer_lens/")
            ),
            "PROVIDER_SCOPE",
            relative,
        )
        path = runner.relative_path(runtime["prefix"], "Lib/site-packages/" + relative)
        need(runner.sha(runner.small_bytes(path)) == runner.digest(digest), "PROVIDER_SOURCE_HASH", relative)
        provider_paths[relative] = path
    return provider_paths


def _cache_pin(entry, role, condition):
    need(type(entry) is dict and set(entry) == {"condition", "index", "windows"}, "CACHE_ENTRY_SCHEMA", role)
    need(entry["condition"] == condition, "CACHE_CONDITION", role)
    for key in ("index", "windows"):
        _pin(entry[key], "CACHE_PIN_SCHEMA")
        _scope_native(entry[key], "%s.%s" % (role, key))
    return entry


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

    snapshot_pin = _pin(inputs["model_snapshot_lock"])
    need(set(snapshot_pin) == {"path", "sha256"}, "INPUT_PIN_SCHEMA")
    _scope_native(snapshot_pin, "model_snapshot_lock")
    snapshot = runner.strict_json(runner.pinned(root, snapshot_pin))
    need(
        snapshot.get("schema") == "snapshot_lock.v1"
        and snapshot.get("revision") == io.MODEL_REVISION
        and snapshot.get("scientific_execution_authorized") is False,
        "SNAPSHOT_LOCK",
    )

    manifest_pins = inputs["manifests"]
    need(type(manifest_pins) is dict and set(manifest_pins) == set(MANIFEST_ROLES), "MANIFEST_SET")
    for role in MANIFEST_ROLES:
        _pin(manifest_pins[role], "MANIFEST_PIN_SCHEMA")
        _scope_native(manifest_pins[role], role)
    manifest = pilot.load_pilot_manifests(manifest_pins, root)

    admitted_ids = set(manifest["cases"])
    cache_indexes = {}
    case_order = None
    for role in ("unprompted_cache", "prompted_cache"):
        entry = _cache_pin(inputs[role], role, SURFACE_CONDITION[role])
        index_path = (root / entry["index"]["path"]).resolve()
        windows_path = (root / entry["windows"]["path"]).resolve()
        cache_indexes[role] = pilot.inspect_pilot_cache(
            index_path,
            entry["index"],
            windows_path,
            entry["windows"],
            admitted_ids=admitted_ids,
            held_ids=set(),
            condition=entry["condition"],
        )
        order = list(cache_indexes[role].document["case_order"])
        if case_order is None:
            case_order = order
        else:
            need(order == case_order, "CACHE_CASE_ORDER_MISMATCH")
    need(set(case_order) == admitted_ids, "CACHE_MANIFEST_CASES")

    retained = pilot.validate_surfaces(inputs["surfaces"])
    need(len(retained) <= io.MAX_SURFACES, "SURFACE_BUDGET")
    labels, splits, folds, group_of = [], [], [], {}
    for case_id in case_order:
        info = manifest["cases"][case_id]
        labels.append(info["class_label"])
        splits.append(info["split"])
        folds.append(info["fold"])
        group_of[case_id] = info["group_id"]
    need(splits.count("TRAIN") == 240 and splits.count("VALIDATION") == 80, "SPLIT_COUNTS")
    group_fold = {}
    for case_id in case_order:
        info = manifest["cases"][case_id]
        if info["split"] != "TRAIN":
            continue
        need(type(info["fold"]) is int and info["fold"] in pilot.FOLDS, "TRAIN_FOLD", case_id)
        group_fold[info["group_id"]] = info["fold"]
    need(set(group_fold.values()) == set(pilot.FOLDS), "TRAIN_FOLD_SET")
    for held in pilot.FOLDS:
        fit_groups = {
            group_of[case_id]
            for case_id in case_order
            if manifest["cases"][case_id]["split"] == "TRAIN" and manifest["cases"][case_id]["fold"] != held
        }
        test_groups = {
            group_of[case_id]
            for case_id in case_order
            if manifest["cases"][case_id]["split"] == "TRAIN" and manifest["cases"][case_id]["fold"] == held
        }
        need(not (fit_groups & test_groups), "FOLD_GROUP_LEAK", str(held))

    need(type(lock.get("snapshot_cache_root")) is str and lock["snapshot_cache_root"], "SNAPSHOT_CACHE_ROOT")
    base = (root / STUDY / "runs").resolve()
    need(base.is_relative_to(root), "OUTPUT_SCOPE")
    marker = (root / NATIVE_STUDY / "runs" / OWNER_NAME).resolve()
    need(marker.is_relative_to(root), "OWNER_SCOPE")
    data = {
        "case_order": case_order,
        "labels": labels,
        "splits": splits,
        "folds": folds,
        "surfaces": retained,
        "cache_indexes": cache_indexes,
        "manifest": manifest,
    }
    return data, {"model_snapshot_lock": runner.pinned(root, snapshot_pin)}


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
        and all(type(value) is int and 0 < value <= (os.cpu_count() or 1) for value in threads.values()),
        "THREAD_SETTINGS",
    )
    data, raws = _inputs(root, lock)
    base = (root / STUDY / "runs").resolve()
    return {
        "lock": lock,
        "lock_sha256": expected_sha256,
        "root": root,
        "data": data,
        "raws": raws,
        "provider_paths": provider_paths,
        "base": base,
        "output": base / lock["run_id"],
        "marker": (root / NATIVE_STUDY / "runs" / OWNER_NAME).resolve(),
    }


def verify_snapshot(ctx):
    return io.snapshot_verifier.verify_snapshot(
        ctx["raws"]["model_snapshot_lock"],
        expected_lock_sha256=ctx["lock"]["inputs"]["model_snapshot_lock"]["sha256"],
        allowed_root=ctx["lock"]["snapshot_cache_root"],
        max_files=SNAPSHOT_LIMITS["max_files"],
        max_file_bytes=SNAPSHOT_LIMITS["max_file_bytes"],
        max_total_bytes=SNAPSHOT_LIMITS["max_total_bytes"],
        deadline_seconds=SNAPSHOT_LIMITS["deadline_seconds"],
    )


# --------------------------------------------------------------------------- #
# release-gated tensor slices + cached-position read
# --------------------------------------------------------------------------- #
def load_tensors(ctx, proof, token_ids):
    """Load only the three Jacobians, the norm vector and <= 6 unembedding rows.

    The ``torch`` import happens inside ``io.torch_weights_only_loader`` and only
    here, after preflight, in the watched worker. No model and no tokenizer is
    instantiated; the safetensors body is seeked per selected row.
    """
    lock = ctx["lock"]
    lens_pin = lock["inputs"]["lens"]
    jacobians = io.load_lens_jacobians(
        str(io.contained_path(lens_pin["path"], lens_pin["cache_root"])),
        lens_pin,
        release=lock["release"],
        loader=io.torch_weights_only_loader,
    )
    files = {item["name"]: item for item in proof["checked_files"]}
    shards = [name for name in files if name.endswith(".safetensors")]
    need(len(shards) == 1, "MODEL_SINGLE_SHARD")
    name = shards[0]
    model_file = io.contained_path(str(Path(proof["snapshot_realpath"]) / name), lock["snapshot_cache_root"])
    model_pin = {"bytes": files[name]["bytes"], "sha256": files[name]["sha256"]}
    io.strict_file(model_file, model_pin, full_hash=True)
    norm = io.load_norm_weight(model_file, model_pin, release=lock["release"], full_hash=False)
    rows = io.load_unembed_rows(model_file, model_pin, token_ids, release=lock["release"], full_hash=False)
    need(set(jacobians) == set(BLOCKS), "JACOBIAN_LAYERS")
    need(getattr(norm, "shape", None) == (io.D_MODEL,), "NORM_SHAPE")
    need(getattr(rows, "shape", None) == (len(token_ids), io.D_MODEL), "ROWS_SHAPE")
    return {"jacobians": jacobians, "norm": norm, "rows": rows}


def read_hidden(ctx):
    """Read the last shared pre-option position per case from both caches.

    Only the final position of each authenticated 16-position window is used;
    AB/BA are pair-averaged exactly as the reviewed classifier reader does. At
    most six ``(n_cases, 1024)`` activation tensors are retained.
    """
    data = ctx["data"]
    release = ctx["lock"]["release"]
    hidden = {condition: {} for condition in CONDITIONS}
    for role, condition in (("unprompted_cache", "unprompted"), ("prompted_cache", "prompted")):
        index = data["cache_indexes"][role]
        for block in BLOCKS:
            rows = np.empty((len(data["case_order"]), io.D_MODEL), dtype=np.float32)
            for row, case_id in enumerate(data["case_order"]):
                view_ab = index.view(case_id, "AB", block, release=release)
                view_ba = index.view(case_id, "BA", block, release=release)
                need(view_ab.shape == view_ba.shape, "VIEW_SHAPE", case_id)
                need(view_ab.shape[1] == io.D_MODEL, "VIEW_WIDTH", case_id)
                rows[row] = pilot.harness.pair_average(view_ab[-1], view_ba[-1]).astype(np.float32)
            need(bool(np.isfinite(rows).all()), "HIDDEN_NONFINITE", "%s|%s" % (condition, block))
            hidden[condition][block] = rows
    return hidden


def compute_scores(hidden, tensors, retained, counters):
    token_ids = tuple(token_id for _, token_id in retained)
    contract = core.ReadoutContract(d_model=io.D_MODEL, token_ids=token_ids, vocab_size=io.VOCAB_SIZE)
    scores = {}
    for method in METHODS:
        scores[method] = {}
        for condition in CONDITIONS:
            scores[method][condition] = {}
            for layer in pilot.SCORE_LAYERS:
                scores[method][condition][layer] = pilot.method_scores(
                    hidden[condition][layer], layer, method, tensors, contract, counters=counters
                )
    return scores


def _working_memory_gib(tensors, hidden, scores):
    total = 0
    for matrix in tensors["jacobians"].values():
        total += int(np.asarray(matrix).nbytes)
    total += int(np.asarray(tensors["norm"]).nbytes)
    total += int(np.asarray(tensors["rows"]).nbytes)
    for condition in hidden:
        for layer in hidden[condition]:
            total += int(np.asarray(hidden[condition][layer]).nbytes)
    for method in scores:
        for condition in scores[method]:
            for layer in scores[method][condition]:
                total += int(np.asarray(scores[method][condition][layer]).nbytes)
    return total / float(1024 ** 3)


def capture(ctx, started, *, verify=verify_snapshot, tensors=None, hidden=None, scores=None, factory=None):
    caps = ctx["lock"]["caps"]
    need(caps == CAPS, "CAPS")
    proof = verify(ctx)
    need(type(proof) is dict and proof.get("snapshot_realpath"), "SNAPSHOT_PROOF")
    Path(ctx["output"]).mkdir(parents=True, exist_ok=True)
    retained = ctx["data"]["surfaces"]
    counters = {"cv_fits": 0, "refits": 0, "fit_errors": 0, "selected_logit_reads": 0}

    if scores is None:
        if hidden is None:
            hidden = read_hidden(ctx)
        need(time.monotonic() - started <= caps["readout_seconds"], "READOUT_DEADLINE")
        if tensors is None:
            tensors = load_tensors(ctx, proof, [token_id for _, token_id in retained])
        scores = compute_scores(hidden, tensors, retained, counters)
    else:
        hidden = hidden if hidden is not None else {}
        tensors = tensors if tensors is not None else {"jacobians": {}, "norm": np.zeros(1), "rows": np.zeros((1, 1))}
    need(time.monotonic() - started <= caps["readout_seconds"], "READOUT_DEADLINE")

    fit_started = time.monotonic()

    def fit_check():
        need(time.monotonic() - fit_started <= caps["fit_seconds"], "FIT_DEADLINE")
        need(time.monotonic() - started <= caps["hard_total_seconds"], "HARD_DEADLINE")

    result = pilot.run_pilot(
        scores=scores,
        case_ids=ctx["data"]["case_order"],
        labels=ctx["data"]["labels"],
        splits=ctx["data"]["splits"],
        folds=ctx["data"]["folds"],
        surface_names=[name for name, _ in retained],
        factory=factory,
        deadline=fit_check,
        counters=counters,
    )
    working_memory = _working_memory_gib(tensors, hidden, scores)
    need(working_memory <= caps["working_memory_gib"], "WORKING_MEMORY")
    need(time.monotonic() - started <= caps["hard_total_seconds"], "HARD_DEADLINE")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "job_id": JOB_ID,
        "status": "pilot_complete",
        "lock_sha256": ctx["lock_sha256"],
        "run_id": ctx["lock"]["run_id"],
        "counts": result["counts"],
        "fits": result["fits"],
        "selected_logit_reads": result["selected_logit_reads"],
        "working_memory_gib": float(working_memory),
        "caps": dict(caps),
        "counters": {
            "model_loads": 0,
            "tokenizer_loads": 0,
            "forwards": 0,
            "derivatives": 0,
            "cone_fits": 0,
            "pca_fits": 0,
            "hyperparameter_search": False,
        },
        "surfaces": [{"surface": name, "token_id": token_id} for name, token_id in retained],
        "conditions": list(CONDITIONS),
        "layers": list(pilot.SCORE_LAYERS),
        "methods": list(METHODS),
        "manifest_case_count": len(ctx["data"]["case_order"]),
        "notes": {
            "cached_readout_only": True,
            "no_model_or_tokenizer_load": True,
            "holdout_accessed": False,
            "validation_used_for_selection": False,
            "raw_logit_is_not_a_probability": True,
            "fit_backend": "sklearn LogisticRegression lbfgs via grouped_driver._default_factory",
        },
    }
    files = {
        ARTIFACTS[0]: runner.encoded(result),
        ARTIFACTS[1]: runner.encoded(receipt),
    }
    total_bytes = sum(len(raw) for raw in files.values())
    need(total_bytes <= caps["output_bytes"] - STATUS_RESERVE, "OUTPUT_CAP", str(total_bytes))
    pins = {}
    for name, raw in files.items():
        runner.write_new(Path(ctx["output"]) / name, raw)
        pins[name] = {"bytes": len(raw), "sha256": runner.sha(raw)}
    return pins


# --------------------------------------------------------------------------- #
# watched worker + parent supervisor (shared native owner lock preserved)
# --------------------------------------------------------------------------- #
def _verify_pins(ctx, pins):
    need(set(pins) == set(ARTIFACTS), "ARTIFACT_SET")
    total = 0
    for name, pin in pins.items():
        path = Path(ctx["output"]) / name
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
        Path(ctx["output"]).mkdir()  # Exclusive: old outputs, even empty, refuse a retry.
        with contextlib.redirect_stdout(sys.stderr):
            pins = capture(ctx, started)
        _verify_pins(ctx, pins)
        report = dict(status="worker_complete", lock_sha256=expected, outputs=pins, **owner)
        runner.write_new(Path(ctx["output"]) / WORKER_RECEIPT, runner.encoded(report))
        return report
    finally:
        runner.release_owner(ctx["marker"], os.getpid(), token, owner["run_id"])


def supervise(lock_path, expected):
    started = time.monotonic()
    ctx = preflight(lock_path, expected)
    need(not Path(ctx["output"]).exists() and not ctx["marker"].exists(), "OWNER_OR_OUTPUT_EXISTS")
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
        child_pid, _ = runner.watch(
            command, str(ctx["root"]), CAPS["hard_total_seconds"] - (time.monotonic() - started)
        )
        child_closed = True
        complete = runner.strict_json(runner.small_bytes(Path(ctx["output"]) / WORKER_RECEIPT))
        need(
            all(
                complete.get(key) == value
                for key, value in dict(
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
        need(time.monotonic() - started <= CAPS["hard_total_seconds"], "HARD_TIMEOUT")
        receipt = dict(
            complete,
            status="complete",
            controller_pid=os.getpid(),
            elapsed_seconds=time.monotonic() - started,
        )
        runner.write_new(Path(ctx["output"]) / SUCCESS_RECEIPT, runner.encoded(receipt))
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
            fits=0,
            forwards=0,
            holdout_accessed=False,
        )
        runner.write_new(ctx["base"] / ("controller_failure_" + token + ".json"), runner.encoded(failure))
        raise
    finally:
        if child_pid is not None and child_closed:
            runner.release_owner(ctx["marker"], child_pid, token, ctx["lock"]["run_id"])


def verify_safe_interfaces(root=ROOT, pins=None):
    """Zero-fit, metadata-only check of the real resolved input interfaces.

    Reads only the two authenticated cache indexes, the four manifests plus
    blueprint and the pinned lens header/hash. No tensor body, model, tokenizer,
    score or fit is touched; this is the early proof that the real safe
    interfaces load under the locked helper hashes.
    """
    root = Path(root).resolve()
    if pins is None:
        document = runner.strict_json((root / STUDY / "JLENS_PILOT_PLAN_V1.json").read_bytes())
        pins = document["known_pins"]
    manifest = pilot.load_pilot_manifests(pins["manifests"], root)
    admitted_ids = set(manifest["cases"])
    summary = {"cases": len(admitted_ids), "caches": {}}
    for role in ("unprompted_cache", "prompted_cache"):
        entry = pins[role]
        index = pilot.inspect_pilot_cache(
            (root / entry["index"]["path"]).resolve(),
            entry["index"],
            (root / entry["windows"]["path"]).resolve(),
            entry["windows"],
            admitted_ids=admitted_ids,
            held_ids=set(),
            condition=entry["condition"],
        )
        summary["caches"][role] = {
            "cases": len(index.document["case_order"]),
            "records": len(index.document["records"]),
            "schema": index.document["schema"],
            "condition": index.document.get("condition"),
        }
    lens = pins["lens"]
    io.inspect_lens(lens["path"], lens, full_hash=True)
    summary["lens"] = {"bytes": lens["bytes"], "sha256": lens["sha256"], "revision": lens["revision"]}
    return summary


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
                fits=0,
                forwards=0,
                model_loads=0,
                tokenizer_loads=0,
                tensor_reads=0,
                scientific_execution_performed=False,
            )
        print(runner.encoded(result).decode().strip())
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI reports a structured failure
        print(
            runner.encoded(
                dict(status="failed", code=type(exc).__name__, detail=str(exc)[:1024])
            ).decode().strip()
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
