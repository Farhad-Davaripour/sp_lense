"""Versioned model-free loader + supervised wrapper for the prompted-compression comparison.

Job ``compression_integration_20260914_v2``. This module is the smallest
integration that lets ``PROMPTED_COMPRESSION_PLAN_V1.md`` execute against the
already-completed unprompted span capture and the already-completed prompted
capture that root owns in parallel. It owns no capture, no lock authoring, no
model/tokenizer/provider, no network/install/Git action and no holdout access.
V2 is a minimal successor of ``compression_integration_v1``: it wraps
``compression_comparison_v2`` (binary targets encoded 1/0, string targets kept
for four-class) and treats the refit cap as a maximum so partially invalid runs
stay honest instead of being padded to 14. The root additions
``native_development_runner_v2.py`` (source pin) and ``threadpoolctl`` (runtime
pin) are retained.

``build_inputs(plan, root)`` authenticates the prospective source/runtime pins,
the existing unprompted span capture, the prompted index/receipt/success/window
pins and query fingerprint against the prompted lock, re-derives prompt
last-token provenance from the lock's tokenizer-sanity rows, reads prompted
case labels/groups/folds from the *same admitted manifests* as the unprompted
capture, checks that both conditions cover identical cases/labels/groups/folds
and builds per-layer-L2 pair-averaged 3072-wide matrices with the existing
``linear_span_control_v1`` features. The old unprompted full-3072 baseline is
hash-pinned from ``runs/linear_span_20260914_v1`` and passed to the core as a
cited reference; it is never refit.

``preflight`` performs that work with zero estimator/PCA fits. ``run`` calls the
reviewed ``compression_comparison_v2.run`` (210 CV attempts + at most 14 refits
+ <=12 PCA fits when every fold is valid) into an exclusive fresh directory and
enforces the 256 MiB total-output bound. The root wrapper owns the release, the
external 600-second watch and the process boundary.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import numpy as np

import compression_comparison_v2 as core
import linear_span_control_v1 as linear
import native_capture_contract as contract
import prompted_input_adapter_v1 as tokens
import span_classifier_driver_v1 as span

__all__ = [
    "CompressionIntegrationError", "JOB_ID", "PLAN_SCHEMA", "PROMPTED_LOCK_SCHEMA",
    "PROMPTED_INDEX_SCHEMA", "PROMPTED_RECEIPT_SCHEMA", "SECONDS", "OUTPUT_BYTES",
    "CV_FIT_LIMIT", "REFIT_LIMIT", "PCA_FIT_LIMIT", "TOTAL_CASES", "TRAIN_CASES",
    "VALIDATION_CASES", "VALIDATION_ORIGINAL", "VALIDATION_ADDED", "FORWARDS",
    "BASELINE_RUN_ID", "BASELINE_NAMES", "SOURCE_FILES", "load_plan",
    "verify_sources_and_runtime", "load_reference_span", "authenticate_prompted",
    "verify_last_token_provenance", "load_prompted_case_data", "build_conditions",
    "build_reference", "build_inputs", "output_dir", "preflight", "run",
]

JOB_ID = "compression_integration_20260914_v2"
PLAN_SCHEMA = "compression_integration_plan.v2"
PROMPTED_LOCK_SCHEMA = "prompted_span_development_execution.v1"
PROMPTED_INDEX_SCHEMA = "prompted_span_capture_index.v1"
PROMPTED_RECEIPT_SCHEMA = "prompted_span_capture_receipt.v1"

SECONDS = 600
OUTPUT_BYTES = 256 * 1024 * 1024
CV_FIT_LIMIT = core.CV_FIT_LIMIT
REFIT_LIMIT = core.REFIT_LIMIT
PCA_FIT_LIMIT = core.PCA_FIT_LIMIT

TOTAL_CASES = span.TOTAL_CASES
TRAIN_CASES = span.TRAIN_CASES
VALIDATION_CASES = span.VALIDATION_CASES
VALIDATION_ORIGINAL = span.VALIDATION_ORIGINAL
VALIDATION_ADDED = span.VALIDATION_ADDED
FORWARDS = TOTAL_CASES * len(span.ORDERS)
BASELINE_RUN_ID = "linear_span_20260914_v1"
STUDY_DIR = "development/classifier_generalization_v2"
BASELINE_NAMES = (
    "cv_scores.json", "family_binary.json", "family_fourclass.json",
    "development_results.json", "fit_plan.json", "supervisor_success.json",
    "estimator_linear_binary.pkl", "estimator_linear_fourclass.pkl",
    "predictions_linear_binary.json", "predictions_linear_fourclass.json",
)
# Prospective code pins the root wrapper snapshots; hashing only, no import.
SOURCE_FILES = (
    STUDY_DIR + "/compression_integration_v2.py",
    STUDY_DIR + "/compression_comparison_v2.py",
    STUDY_DIR + "/linear_span_control_v1.py",
    STUDY_DIR + "/span_classifier_driver_v1.py",
    STUDY_DIR + "/span_feature_transforms_v1.py",
    STUDY_DIR + "/grouped_driver.py",
    STUDY_DIR + "/harness.py",
    STUDY_DIR + "/prompted_input_adapter_v1.py",
    STUDY_DIR + "/native_capture_contract.py",
    STUDY_DIR + "/native_development_runner_v2.py",
    STUDY_DIR + "/run_compression_supervised_v2.py",
)
RUNTIME_PACKAGES = ("numpy", "scipy", "scikit-learn", "threadpoolctl")
PIN_FIELDS = ("path", "sha256")


class CompressionIntegrationError(ValueError):
    """Structured rejection raised by this module."""


def _need(condition, code, detail=None):
    if not condition:
        raise CompressionIntegrationError(code if detail is None else "%s: %s" % (code, detail))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha_path(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _read_pin(root, pin, role):
    """Reuse the reviewed span pin reader; normalise its error family."""
    try:
        return span._pinned(root, pin, role)
    except span.SpanDriverError as exc:
        raise CompressionIntegrationError(str(exc)) from None


def _load_json(raw, code):
    try:
        return span._load_json(raw, code)
    except span.SpanDriverError as exc:
        raise CompressionIntegrationError(str(exc)) from None


def _span_loader(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except span.SpanDriverError as exc:
        raise CompressionIntegrationError(str(exc)) from None


# --------------------------------------------------------------------------- #
# prospective plan + source/runtime pins
# --------------------------------------------------------------------------- #
def load_plan(plan_path, expected_sha256):
    path = Path(plan_path).resolve()
    _need(path.is_file(), "PLAN_MISSING", str(path))
    raw = path.read_bytes()
    _need(_sha(raw) == expected_sha256, "PLAN_DIGEST")
    plan = _load_json(raw, "PLAN_JSON")
    _need(type(plan) is dict, "PLAN_SCHEMA")
    _need(plan.get("schema") == PLAN_SCHEMA, "PLAN_SCHEMA")
    _need(plan.get("job_id") == JOB_ID, "PLAN_JOB_ID")
    _need(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", str(plan.get("run_id", ""))), "PLAN_RUN_ID")
    _need(plan.get("seconds") == SECONDS, "PLAN_SECONDS")
    _need(plan.get("output_bytes") == OUTPUT_BYTES, "PLAN_OUTPUT_BYTES")
    _need(plan.get("cv_fit_limit") == CV_FIT_LIMIT, "PLAN_CV_LIMIT")
    _need(plan.get("refit_limit") == REFIT_LIMIT, "PLAN_REFIT_LIMIT")
    _need(plan.get("pca_fit_limit") == PCA_FIT_LIMIT, "PLAN_PCA_LIMIT")
    _need(plan.get("holdout_access") is False, "PLAN_HOLDOUT_ACCESS")
    _need(list(plan.get("C", [])) == list(core.C_VALUES), "PLAN_C")
    _need(list(plan.get("thresholds", [])) == list(core.TAUS), "PLAN_TAUS")
    _need(list(plan.get("folds", [])) == list(core.FOLDS), "PLAN_FOLDS")
    _need(list(plan.get("conditions", [])) == list(core.CONDITIONS), "PLAN_CONDITIONS")
    for role in ("reference_span", "reference_baseline", "prompted_source"):
        _need(type(plan.get(role)) is dict, "PLAN_ROLE", role)
    _need(type(plan.get("source_files")) is dict and bool(plan["source_files"]), "PLAN_SOURCE_FILES")
    _need(type(plan.get("runtime_packages")) is dict and bool(plan["runtime_packages"]), "PLAN_RUNTIME")
    return plan


def verify_sources_and_runtime(plan, root):
    """Prospective code and runtime hashes; no Git, no import of pinned code."""
    for relative, expected in plan["source_files"].items():
        _need(type(relative) is str and relative != "", "SOURCE_PATH")
        _need(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected), "SOURCE_PIN_FORMAT", relative)
        path = _span_loader(span._resolve, root, relative)
        _need(path.is_file(), "SOURCE_MISSING", relative)
        _need(_sha_path(path) == expected, "SOURCE_MISMATCH", relative)
    for name, version in plan["runtime_packages"].items():
        _need(type(name) is str and type(version) is str, "RUNTIME_PIN_FORMAT")
        _need(importlib.metadata.version(name) == version, "RUNTIME_VERSION", name)


# --------------------------------------------------------------------------- #
# unprompted reference: existing checked span loader, never captured here
# --------------------------------------------------------------------------- #
def load_reference_span(plan, root):
    block = plan["reference_span"]
    pin = block.get("plan")
    _need(type(pin) is dict and set(PIN_FIELDS) <= set(pin), "REFERENCE_SPAN_PLAN")
    path, _raw = _read_pin(root, pin, "reference_span_plan")
    span_plan = _span_loader(span.load_plan, path, pin["sha256"])
    source_sha = span_plan["source"]["lock"]["sha256"]
    source = _span_loader(span.load_source, span_plan, source_sha, root)
    expected_run = block.get("run_id")
    _need(expected_run is None or source["lock"]["run_id"] == expected_run, "REFERENCE_SPAN_RUN_ID")
    case_data = _span_loader(span.load_case_data, source, span_plan, root)
    return {"plan": span_plan, "plan_sha256": pin["sha256"], "source": source, "case_data": case_data}


# --------------------------------------------------------------------------- #
# prompted capture: pins, query fingerprint, records, last-token provenance
# --------------------------------------------------------------------------- #
def _prompted_record_map(index, caps):
    records = index.get("records")
    _need(type(records) is list and records, "PROMPTED_RECORDS")
    mapping = {}
    for record in records:
        _need(type(record) is dict, "PROMPTED_RECORD")
        case_id, order, block = record.get("case_id"), record.get("order"), record.get("block")
        _need(type(case_id) is str and case_id, "PROMPTED_RECORD_CASE")
        _need(order in span.ORDERS, "PROMPTED_RECORD_ORDER")
        _need(block in span.BLOCKS, "PROMPTED_RECORD_BLOCK")
        key = (case_id, order, block)
        _need(key not in mapping, "PROMPTED_RECORD_DUPLICATE", str(key))
        prefix, readout = record.get("prefix_length"), record.get("readout_index")
        _need(type(prefix) is int and type(readout) is int and prefix == readout + 1, "PROMPTED_RECORD_PREFIX")
        _need(1 <= prefix <= caps["tokens_per_view"], "PROMPTED_PREFIX_BOUND", str(prefix))
        _need(isinstance(record.get("input_ids_sha256"), str), "PROMPTED_RECORD_HASH")
        mapping[key] = {"prefix_length": prefix, "input_ids_sha256": record["input_ids_sha256"]}
    case_order = index.get("case_order")
    _need(type(case_order) is list and case_order and len(set(case_order)) == len(case_order), "PROMPTED_CASE_ORDER")
    expected = len(case_order) * len(span.ORDERS) * len(span.BLOCKS)
    _need(len(mapping) == expected, "PROMPTED_RECORD_ACCOUNTING", str(len(mapping)))
    _need(set(key[0] for key in mapping) == set(case_order), "PROMPTED_RECORD_CASES")
    for case_id in case_order:
        for order in span.ORDERS:
            for block in span.BLOCKS:
                _need((case_id, order, block) in mapping, "PROMPTED_RECORD_MISSING",
                      "%s|%s|%s" % (case_id, order, block))
    return mapping, case_order


def verify_last_token_provenance(index, sanity, caps):
    """Every case/order/block and the last shared pre-option token come from the lock."""
    mapping, case_order = _prompted_record_map(index, caps)
    _need(type(sanity) is dict and sanity.get("status") == "PASS", "PROMPTED_SANITY_STATUS")
    _need(sanity.get("query_sha256") == tokens.QUERY_SHA256, "PROMPTED_SANITY_QUERY")
    _need(sanity.get("exact_decode_reencode") is True and sanity.get("query_in_shared_prefix") is True,
          "PROMPTED_SANITY_ROUNDTRIP")
    _need(sanity.get("last_shared_token_ids") == [contract.LAST_SHARED_ID], "PROMPTED_SANITY_BOUNDARY")
    _need(sanity.get("max_full_tokens", 10 ** 9) <= caps["tokens_per_view"], "PROMPTED_SANITY_TOKEN_CAP")
    counts = sanity.get("counts")
    _need(type(counts) is dict, "PROMPTED_SANITY_COUNTS")
    _need(counts.get("cases") == len(case_order) and counts.get("views") == len(case_order) * len(span.ORDERS),
          "PROMPTED_SANITY_COUNT_BINDING")
    _need(counts.get("tokenizer_loads") == 1 and counts.get("model_loads") == 0, "PROMPTED_SANITY_LOADS")
    _need(counts.get("forwards") == 0 and counts.get("fits") == 0, "PROMPTED_SANITY_WORK")
    rows = sanity.get("rows")
    _need(type(rows) is list and len(rows) == len(case_order) * len(span.ORDERS), "PROMPTED_SANITY_ROWS")
    seen = set()
    expected_keys = {(case_id, order) for case_id in case_order for order in span.ORDERS}
    for row in rows:
        _need(type(row) is dict, "PROMPTED_SANITY_ROW")
        case_id, order = row.get("case_id"), row.get("order")
        key = (case_id, order)
        _need(key in expected_keys and key not in seen, "PROMPTED_SANITY_ROW_KEY", str(key))
        seen.add(key)
        _need(row.get("last_shared_token_id") == contract.LAST_SHARED_ID, "PROMPTED_SANITY_LAST_TOKEN", str(key))
        prefixes = {mapping[(case_id, order, block)]["prefix_length"] for block in span.BLOCKS}
        _need(len(prefixes) == 1, "PROMPTED_BLOCK_PREFIX_MISMATCH", str(key))
        _need(row.get("prefix_tokens") == prefixes.pop(), "PROMPTED_SANITY_PREFIX", str(key))
        hashes = {mapping[(case_id, order, block)]["input_ids_sha256"] for block in span.BLOCKS}
        _need(len(hashes) == 1 and row.get("input_sha256") == hashes.pop(), "PROMPTED_SANITY_INPUT", str(key))
    _need(seen == expected_keys, "PROMPTED_SANITY_COVERAGE")
    return {"cases": len(case_order), "rows": len(rows)}


def authenticate_prompted(plan, root):
    """Authenticate prompted lock, index, receipt, success and window pins."""
    block = plan["prompted_source"]
    for role in ("lock", "index", "windows", "receipt", "success"):
        _need(isinstance(block.get(role), dict), "PROMPTED_PIN", role)
    lock_path, lock_raw = _read_pin(root, block["lock"], "prompted_lock")
    lock = _load_json(lock_raw, "PROMPTED_LOCK_JSON")
    _need(lock.get("schema") == PROMPTED_LOCK_SCHEMA, "PROMPTED_LOCK_SCHEMA")
    _need(lock.get("scientific_execution_authorized") is True, "PROMPTED_LOCK_NOT_AUTHORIZED")
    run_id = lock.get("run_id")
    _need(type(run_id) is str and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", run_id), "PROMPTED_RUN_ID")
    _need(block.get("run_id") is None or block["run_id"] == run_id, "PROMPTED_RUN_ID")
    caps = lock.get("caps")
    _need(type(caps) is dict, "PROMPTED_CAPS")
    _need(caps.get("fits") == 0 and caps.get("derivatives") == 0, "PROMPTED_CAPS_FITS")
    _need(caps.get("model_loads") == 1 and caps.get("tokenizer_loads") == 1, "PROMPTED_CAPS_LOADS")
    _need(type(caps.get("forwards")) is int and caps["forwards"] > 0, "PROMPTED_CAPS_FORWARDS")
    _need(type(caps.get("tokens_per_view")) is int and caps["tokens_per_view"] > 0, "PROMPTED_CAPS_TOKENS")
    _need(block.get("forwards") is None or block["forwards"] == caps["forwards"], "PROMPTED_FORWARDS_PIN")
    _need(block.get("tokens_per_view") is None or block["tokens_per_view"] == caps["tokens_per_view"],
          "PROMPTED_TOKENS_PIN")
    references = lock.get("reference_locks")
    _need(type(references) is list and len(references) == 2, "PROMPTED_REFERENCE_LOCKS")
    _need(all(isinstance(reference, dict) for reference in references), "PROMPTED_REFERENCE_LOCKS")

    index_path, index_raw = _read_pin(root, block["index"], "prompted_index")
    windows_path, windows_raw = _read_pin(root, block["windows"], "prompted_windows")
    receipt_path, receipt_raw = _read_pin(root, block["receipt"], "prompted_receipt")
    success_path, success_raw = _read_pin(root, block["success"], "prompted_success")
    index = _load_json(index_raw, "PROMPTED_INDEX_JSON")
    receipt = _load_json(receipt_raw, "PROMPTED_RECEIPT_JSON")
    success = _load_json(success_raw, "PROMPTED_SUCCESS_JSON")

    _need(index.get("schema") == PROMPTED_INDEX_SCHEMA, "PROMPTED_INDEX_SCHEMA")
    _need(index.get("condition") == tokens.PROMPTED_CONDITION, "PROMPTED_CONDITION")
    _need(index.get("query") == tokens.FIXED_QUERY and index.get("query_sha256") == tokens.QUERY_SHA256,
          "PROMPTED_QUERY")
    _need(index.get("lock_sha256") == block["lock"]["sha256"], "PROMPTED_INDEX_LOCK")
    _need(index.get("run_id") == run_id, "PROMPTED_INDEX_RUN_ID")
    _need(index.get("blocks") == list(span.BLOCKS) and index.get("orders") == list(span.ORDERS), "PROMPTED_INDEX_LAYOUT")
    _need(index.get("dtype") == "float32" and index.get("byte_order") == "little", "PROMPTED_INDEX_DTYPE")
    _need(index.get("float_bytes") == span.FLOAT_BYTES and index.get("width") == span.WIDTH, "PROMPTED_INDEX_WIDTH")
    _need(index.get("window_max") == span.WINDOW_MAX, "PROMPTED_INDEX_WINDOW")
    _need(index.get("raw_bytes") == len(windows_raw), "PROMPTED_INDEX_BYTES")
    _need(index.get("case_count") == len(index.get("case_order", [])), "PROMPTED_INDEX_CASES")
    views = index.get("case_count", 0) * len(span.ORDERS)
    _need(index.get("view_count") == views and index.get("forward_count") == views, "PROMPTED_INDEX_VIEWS")
    _need(index.get("forward_count") == caps["forwards"], "PROMPTED_FORWARD_ACCOUNTING")

    expected_outputs = {"windows.f32": block["windows"], "index.json": block["index"],
                        "capture_receipt.json": block["receipt"]}
    _need(success.get("status") == "complete" and success.get("lock_sha256") == block["lock"]["sha256"],
          "PROMPTED_SUCCESS")
    _need(success.get("run_id") == run_id, "PROMPTED_SUCCESS_RUN_ID")
    outputs = success.get("outputs")
    _need(type(outputs) is dict, "PROMPTED_SUCCESS_OUTPUTS")
    for name, pin in expected_outputs.items():
        _need(name in outputs and outputs[name].get("sha256") == pin["sha256"], "PROMPTED_SUCCESS_PIN", name)
        if "bytes" in pin:
            _need(outputs[name].get("bytes") == pin["bytes"], "PROMPTED_SUCCESS_BYTES", name)

    _need(receipt.get("schema") == PROMPTED_RECEIPT_SCHEMA and receipt.get("status") == "capture_complete",
          "PROMPTED_RECEIPT")
    _need(receipt.get("lock_sha256") == block["lock"]["sha256"] and receipt.get("run_id") == run_id,
          "PROMPTED_RECEIPT_BINDING")
    _need(receipt.get("query_sha256") == tokens.QUERY_SHA256, "PROMPTED_RECEIPT_QUERY")
    artifacts = receipt.get("artifacts")
    _need(type(artifacts) is dict, "PROMPTED_RECEIPT_ARTIFACTS")
    for name in ("windows.f32", "index.json"):
        _need(name in artifacts and artifacts[name].get("sha256") == expected_outputs[name]["sha256"],
              "PROMPTED_RECEIPT_PIN", name)

    sanity_pin = (lock.get("inputs") or {}).get("prompt_sanity")
    _need(isinstance(sanity_pin, dict), "PROMPTED_SANITY_PIN")
    _, sanity_raw = _read_pin(root, sanity_pin, "prompt_sanity")
    sanity = _load_json(sanity_raw, "PROMPTED_SANITY_JSON")
    _need(sanity.get("adapter_sha256") == plan["source_files"].get(STUDY_DIR + "/prompted_input_adapter_v1.py"),
          "PROMPTED_SANITY_ADAPTER")
    provenance = verify_last_token_provenance(index, sanity, caps)
    return {
        "lock": lock, "lock_sha256": block["lock"]["sha256"], "lock_path": str(lock_path),
        "index": index, "index_path": str(index_path), "index_sha256": block["index"]["sha256"],
        "windows_raw": windows_raw, "windows_path": str(windows_path), "windows_sha256": block["windows"]["sha256"],
        "receipt": receipt, "success": success, "sanity": sanity,
        "query_sha256": tokens.QUERY_SHA256, "provenance": provenance,
    }


def load_prompted_case_data(auth, span_plan, root):
    """Reuse the reviewed loader; labels/groups/folds come only from the span plan manifests."""
    source = {"index": auth["index"], "windows_raw": auth["windows_raw"]}
    return _span_loader(span.load_case_data, source, span_plan, root)


# --------------------------------------------------------------------------- #
# identical-case check and matrix assembly
# --------------------------------------------------------------------------- #
def _verify_identity(unprompted, prompted):
    for key in ("case_order", "train_ids", "validation_ids", "original_ids", "added_ids"):
        _need(type(unprompted.get(key)) is list and type(prompted.get(key)) is list, "CASE_DATA_IDS", key)
        _need(set(unprompted[key]) == set(prompted[key]), "CONDITION_CASE_MISMATCH", key)
    _need(len(unprompted["case_order"]) == TOTAL_CASES, "CASE_TOTAL", str(len(unprompted["case_order"])))
    _need(len(unprompted["train_ids"]) == TRAIN_CASES, "TRAIN_TOTAL")
    _need(len(unprompted["validation_ids"]) == VALIDATION_CASES, "VALIDATION_TOTAL")
    _need(len(unprompted["original_ids"]) == VALIDATION_ORIGINAL, "ORIGINAL_TOTAL")
    _need(len(unprompted["added_ids"]) == VALIDATION_ADDED, "ADDED_TOTAL")
    _need(set(unprompted["validation_ids"]) == set(unprompted["original_ids"]) | set(unprompted["added_ids"]),
          "VALIDATION_PARTITION")
    _need(not (set(unprompted["original_ids"]) & set(unprompted["added_ids"])), "VALIDATION_OVERLAP")
    _need(not (set(unprompted["train_ids"]) & set(unprompted["validation_ids"])), "SPLIT_OVERLAP")
    for key in ("labels", "groups", "folds"):
        _need(type(unprompted.get(key)) is dict and type(prompted.get(key)) is dict, "CASE_DATA_MAP", key)
        _need(unprompted[key] == prompted[key], "CONDITION_META_MISMATCH", key)
    return {
        "train_ids": list(unprompted["train_ids"]),
        "original_ids": list(unprompted["original_ids"]),
        "added_ids": list(unprompted["added_ids"]),
        "validation_ids": list(unprompted["original_ids"]) + list(unprompted["added_ids"]),
        "case_order": list(unprompted["train_ids"]) + list(unprompted["original_ids"]) + list(unprompted["added_ids"]),
        "labels": dict(unprompted["labels"]), "groups": dict(unprompted["groups"]),
        "folds": dict(unprompted["folds"]),
    }


def _canonical_matrix(case_data, shared):
    order, matrix = linear.build_features(case_data)
    _need(set(order) == set(shared["case_order"]), "FEATURE_CASE_MISMATCH")
    row_of = {case_id: row for row, case_id in enumerate(order)}
    return np.ascontiguousarray(matrix[[row_of[case_id] for case_id in shared["case_order"]]])


def _canonical_order_features(case_data, key, shared):
    _need(case_data.get(key) is not None, "ORDER_WINDOWS_MISSING", key)
    rows = [linear.features_for_case(case_data[key][case_id]) for case_id in shared["validation_ids"]]
    matrix = np.ascontiguousarray(np.stack(rows))
    _need(matrix.shape == (VALIDATION_CASES, core.FULL_DIM), "ORDER_MATRIX_SHAPE", str(matrix.shape))
    return matrix


def build_conditions(unprompted, prompted, provenance):
    shared = _verify_identity(unprompted, prompted)
    _need(len(shared["case_order"]) == len(set(shared["case_order"])), "CASE_ORDER_DUPLICATE")
    metadata = {
        "case_order": shared["case_order"], "labels": shared["labels"], "groups": shared["groups"],
        "folds": shared["folds"], "train_ids": shared["train_ids"], "validation_ids": shared["validation_ids"],
        "original_ids": shared["original_ids"], "added_ids": shared["added_ids"],
    }
    return {
        "metadata": metadata,
        "unprompted": {"matrix": _canonical_matrix(unprompted, shared)},
        "prompted": {
            "matrix": _canonical_matrix(prompted, shared),
            "ab_matrix": _canonical_order_features(prompted, "ab_windows", shared),
            "ba_matrix": _canonical_order_features(prompted, "ba_windows", shared),
        },
        "provenance": provenance,
    }


# --------------------------------------------------------------------------- #
# hash-pinned old unprompted full-3072 baseline (never refit)
# --------------------------------------------------------------------------- #
def build_reference(plan, root, case_data):
    block = plan["reference_baseline"]
    _need(block.get("run_id") == BASELINE_RUN_ID, "BASELINE_RUN_ID")
    files = block.get("files")
    _need(type(files) is dict, "BASELINE_FILES")
    raws = {}
    for name in BASELINE_NAMES:
        _need(name in files, "BASELINE_PIN", name)
        _, raw = _read_pin(root, files[name], name)
        raws[name] = raw
    cv = _load_json(raws["cv_scores.json"], "BASELINE_CV_JSON")
    train_ids = list(case_data["train_ids"])
    _need(cv.get("oof_case_ids") == train_ids, "BASELINE_OOF_IDS")
    truth = [case_data["labels"][case_id] for case_id in train_ids]
    _need(cv.get("oof_truth") == truth, "BASELINE_OOF_TRUTH")
    candidates = cv.get("candidates")
    _need(type(candidates) is list and candidates, "BASELINE_CANDIDATES")
    development = _load_json(raws["development_results.json"], "BASELINE_RESULTS_JSON")
    _need(development.get("status") == "COMPLETE", "BASELINE_STATUS")
    counters = development.get("counters") or {}
    _need(counters.get("cv_fits") == 30 and counters.get("refits") == 2, "BASELINE_COUNTERS")
    _need(development.get("holdout_accessed") is False, "BASELINE_HOLDOUT")

    families, artifacts = {}, {}
    for family in core.FAMILIES:
        detail = _load_json(raws["family_%s.json" % family], "BASELINE_FAMILY_JSON")
        _need(detail.get("family") == family and detail.get("status") == "FITTED", "BASELINE_FAMILY_STATUS", family)
        _need(detail.get("reload_exact") is True, "BASELINE_RELOAD", family)
        artifact = detail.get("artifact")
        model_name = "estimator_linear_%s.pkl" % family
        _need(artifact == model_name, "BASELINE_ARTIFACT_NAME", family)
        _need(_sha(raws[model_name]) == detail.get("artifact_sha256"), "BASELINE_ARTIFACT_SHA", family)
        prediction_name = "predictions_linear_%s.json" % family
        _load_json(raws[prediction_name], "BASELINE_PREDICTIONS_JSON")
        criterion = {"family": family, "C": float(detail["C"]), "selected_tau": float(detail["selected_tau"]),
                     "valid": True}
        matches = [entry for entry in candidates
                   if entry.get("family") == criterion["family"] and entry.get("valid") is True
                   and float(entry.get("C")) == criterion["C"]
                   and float(entry.get("selected_tau", entry.get("tau"))) == criterion["selected_tau"]]
        _need(len(matches) == 1, "BASELINE_CANDIDATE_MATCH", family)
        p_self = np.asarray(matches[0].get("oof_p_self"), float)
        _need(p_self.shape == (len(train_ids),) and bool(np.isfinite(p_self).all()), "BASELINE_OOF")
        _need(float(p_self.min()) >= 0.0 and float(p_self.max()) <= 1.0, "BASELINE_OOF_RANGE")
        families[family] = {
            "C": criterion["C"], "tau": criterion["selected_tau"], "oof_p_self": p_self.tolist(),
            "oof_metrics": detail.get("oof_metrics"), "artifact": "runs/%s/%s" % (BASELINE_RUN_ID, model_name),
            "artifact_sha256": detail["artifact_sha256"], "evaluation": detail.get("evaluation"),
            "citation": {"run_id": BASELINE_RUN_ID, "family_file": "family_%s.json" % family,
                          "oof_file": "cv_scores.json", "model_file": model_name,
                          "predictions_file": prediction_name},
        }
        artifacts[family] = {"family_file": _sha(raws["family_%s.json" % family]),
                             "model_file": detail["artifact_sha256"],
                             "predictions_file": _sha(raws[prediction_name])}
    provenance = {
        "job_id": development.get("job_id"), "run_id": BASELINE_RUN_ID, "refit": False,
        "source": "REFERENCE_REUSED_NOT_REFIT", "holdout_accessed": False,
        "files": {name: files[name]["sha256"] for name in BASELINE_NAMES},
        "artifacts": artifacts,
    }
    return {"condition": "unprompted", "dimension": "full3072", "provenance": provenance, "families": families}


# --------------------------------------------------------------------------- #
# assembled inputs, zero-fit preflight and supervised run
# --------------------------------------------------------------------------- #
def output_dir(plan, root):
    base = (Path(root).resolve() / STUDY_DIR / "runs").resolve()
    _need(base.is_relative_to(Path(root).resolve()), "OUTPUT_SCOPE")
    return base / plan["run_id"]


def _provenance(plan, auth, reference_span):
    return {
        "job_id": JOB_ID,
        "reference_span": {"run_id": reference_span["source"]["lock"]["run_id"],
                           "lock_sha256": reference_span["source"]["lock_sha256"],
                           "plan_sha256": reference_span["plan_sha256"]},
        "reference_baseline": {"run_id": BASELINE_RUN_ID,
                               "files": {name: plan["reference_baseline"]["files"][name]["sha256"]
                                         for name in BASELINE_NAMES}},
        "prompted": {"run_id": auth["lock"]["run_id"], "lock_sha256": auth["lock_sha256"],
                     "index_sha256": auth["index_sha256"], "windows_sha256": auth["windows_sha256"],
                     "query_sha256": auth["query_sha256"]},
        "holdout_accessed": False,
    }


def build_inputs(plan, root):
    """Authenticate every prospective pin and assemble aligned conditions plus the reference."""
    verify_sources_and_runtime(plan, root)
    reference_span = load_reference_span(plan, root)
    auth = authenticate_prompted(plan, root)
    prompted = load_prompted_case_data(auth, reference_span["plan"], root)
    conditions = build_conditions(reference_span["case_data"], prompted, _provenance(plan, auth, reference_span))
    reference = build_reference(plan, root, reference_span["case_data"])
    return {"conditions": conditions, "reference": reference, "unprompted": reference_span["case_data"],
            "prompted": prompted, "auth": auth, "span_plan": reference_span["plan"],
            "output": output_dir(plan, root)}


def preflight(plan_path, plan_sha256, root):
    """Authenticate and assemble everything with zero estimator/PCA fits."""
    plan = load_plan(plan_path, plan_sha256)
    inputs = build_inputs(plan, root)
    _need(not inputs["output"].exists(), "PREFLIGHT_OUTPUT_EXISTS", str(inputs["output"]))
    _need(inputs["conditions"]["unprompted"]["matrix"].shape == (TOTAL_CASES, core.FULL_DIM),
          "UNPROMPTED_MATRIX_SHAPE")
    _need(inputs["conditions"]["prompted"]["matrix"].shape == (TOTAL_CASES, core.FULL_DIM),
          "PROMPTED_MATRIX_SHAPE")
    return {
        "status": "preflight_pass", "job_id": JOB_ID, "run_id": plan["run_id"],
        "cases": TOTAL_CASES, "train": TRAIN_CASES, "validation": VALIDATION_CASES,
        "original": VALIDATION_ORIGINAL, "added": VALIDATION_ADDED, "feature_width": core.FULL_DIM,
        "conditions": list(core.CONDITIONS), "query_sha256": inputs["auth"]["query_sha256"],
        "prompted_run_id": inputs["auth"]["lock"]["run_id"],
        "prompted_rows": inputs["auth"]["provenance"]["rows"],
        "reference_baseline_run_id": BASELINE_RUN_ID,
        "fits": 0, "pca_fits": 0, "holdout_accessed": False,
    }


def _verify_result(result, plan, output):
    _need(result.get("status") == "COMPLETE", "RESULT_STATUS", str(result.get("status")))
    counters = result.get("counters") or {}
    _need(counters.get("cv_fits") == CV_FIT_LIMIT, "RESULT_CV_BUDGET", str(counters.get("cv_fits")))
    # The refit cap is a maximum, not an equality: invalid candidates are never
    # padded or refit to reach 14, so a partially invalid run legitimately refits
    # fewer family winners. Failures stay explicit in candidate_failures.json.
    _need(counters.get("refits", REFIT_LIMIT + 1) <= REFIT_LIMIT, "RESULT_REFIT_BUDGET",
          str(counters.get("refits")))
    _need(counters.get("pca_fits", PCA_FIT_LIMIT + 1) <= PCA_FIT_LIMIT, "RESULT_PCA_BUDGET")
    _need(counters.get("reference_fits") == 0, "RESULT_REFERENCE_REFIT")
    _need(result.get("holdout_accessed") is False and result.get("validation_used_for_selection") is False,
          "RESULT_SCOPE")
    total = sum(path.stat().st_size for path in Path(output).rglob("*") if path.is_file())
    _need(total <= plan["output_bytes"], "RESULT_OUTPUT_BYTES", str(total))
    return total


def run(plan_path, plan_sha256, root, *, factory=None, deadline=None):
    """Fit the comparison into an exclusive fresh directory, then enforce the output cap."""
    plan = load_plan(plan_path, plan_sha256)
    inputs = build_inputs(plan, root)
    output = inputs["output"]
    _need(not output.exists(), "OUTPUT_EXISTS", str(output))
    result = core.run(inputs["conditions"], inputs["reference"], output, factory=factory, deadline=deadline)
    _verify_result(result, plan, output)
    return result
