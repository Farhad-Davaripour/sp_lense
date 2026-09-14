"""Minimal authenticated loader + supervised comparison for the three-question identity pipeline.

Job ``identity_fit_pipeline_20260914_v1``. This module is the smallest finite
integration that lets the already-derived three-question identity prompt be
compared against the already-completed single-question prompted control. It owns
no capture, no lock authoring, no model/tokenizer/provider, no network/install/Git
action and no holdout access.

It reuses, rather than re-implements, the reviewed machinery:

* ``span_classifier_driver_v1`` for plan-independent manifest loading, index
  record authentication/window reconstruction (``decode_index``), case-data
  assembly (``load_case_data``), split evaluation (``_evaluate_split``) and the
  pin reader (``_pinned``);
* ``linear_span_control_v1`` for the per-layer L2 last-token concatenation
  (3072 coordinates);
* ``compression_comparison_v2.PCA32`` for the exact full-SVD 32-component PCA;
* ``harness`` for the binary gate metrics / threshold eligibility;
* ``grouped_driver._default_factory`` for the fixed L2 logistic regression.

The identity condition is fit fresh: 5 grouped TRAIN folds + 1 full-TRAIN refit
(6 classifier fits) and 5 fold PCA fits + 1 full-TRAIN PCA fit (6 PCA fits), all
at binary C=10 with the 19-threshold TRAIN-OOF selection rule
``max F1, then max min(precision, recall), then |tau-0.5|, then smaller tau``.
The old prompted control contributes zero fits: its saved TRAIN OOF and its saved
per-split prediction JSON are re-read (never the pickled estimator) and the same
current TRAIN-only threshold rule is applied to the saved OOF.

``build_inputs`` authenticates the identity lock/index/window/receipt/success and
sanity pins, the identity source pins, the shared manifests and the old-control
artifacts, and proves that the identity cases/order/labels/groups/folds match the
control. ``preflight`` does that with zero estimator/PCA fits. ``run`` writes the
comparison into an exclusive fresh directory, preserves a failure receipt and
enforces the 64 MiB total-output bound.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

import compression_comparison_v2 as core
import grouped_driver as driver
import harness
import identity_input_adapter_v1 as tokens
import linear_span_control_v1 as linear
import span_classifier_driver_v1 as span

__all__ = [
    "IdentityFitError", "JOB_ID", "PLAN_SCHEMA", "IDENTITY_LOCK_SCHEMA", "IDENTITY_INDEX_SCHEMA",
    "IDENTITY_RECEIPT_SCHEMA", "IDENTITY_WINDOWS_SCHEMA", "SECONDS", "OUTPUT_BYTES", "CV_FITS",
    "REFITS", "CLASSIFIER_FITS", "PCA_FITS", "FAMILY", "C_VALUE", "TAUS", "FOLDS", "CLASS_ORDER",
    "PCA_COMPONENTS", "FEATURE_DIM", "SPLITS", "OLD_CONTROL_RUN_ID", "OLD_CONTROL_CONDITION",
    "IDENTITY_CONDITION", "SOURCE_FILES", "RUNTIME_PACKAGES", "load_plan",
    "verify_sources_and_runtime", "authenticate_identity", "load_identity_case_data",
    "load_old_control", "select_tau", "comparison_key", "build_inputs", "output_dir", "preflight",
    "compare", "run",
]

STUDY_DIR = "development/classifier_generalization_v2"

JOB_ID = "identity_fit_pipeline_20260914_v1"
PLAN_SCHEMA = "identity_fit_plan.v1"
IDENTITY_LOCK_SCHEMA = "identity_span_development_execution.v1"
IDENTITY_INDEX_SCHEMA = "identity_span_capture_index.v1"
IDENTITY_RECEIPT_SCHEMA = "identity_span_capture_receipt.v1"
IDENTITY_WINDOWS_SCHEMA = "identity_span_capture_windows.v1"

SECONDS = 60
OUTPUT_BYTES = 64 * 1024 * 1024
CV_FITS = 5
REFITS = 1
CLASSIFIER_FITS = CV_FITS + REFITS
PCA_FITS = 6

FAMILY = "binary"
C_VALUE = 10.0
TAUS = tuple(span.TAUS)
FOLDS = tuple(span.FOLDS)
CLASS_ORDER = tuple(span.CLASS_ORDER)
PCA_COMPONENTS = int(core.PCA_COMPONENTS)
FEATURE_DIM = int(linear.FEATURE_DIM)
SPLITS = (("train", "train_ids"), ("original40", "original_ids"), ("added40", "added_ids"),
          ("combined80", "validation_ids"))

IDENTITY_CONDITION = tokens.IDENTITY_CONDITION
OLD_CONTROL_RUN_ID = "compression_supervised_20260914_v2"
OLD_CONTROL_CONDITION = "prompted"  # saved compression_supervised_20260914_v2 artifact tag
OLD_DIMENSION = "pca32"
OLD_FAMILY = "binary"
OLD_C = 10.0

SOURCE_FILES = (
    STUDY_DIR + "/identity_fit_v1.py",
    STUDY_DIR + "/run_identity_fit_v1.py",
    STUDY_DIR + "/compression_comparison_v2.py",
    STUDY_DIR + "/linear_span_control_v1.py",
    STUDY_DIR + "/span_classifier_driver_v1.py",
    STUDY_DIR + "/grouped_driver.py",
    STUDY_DIR + "/harness.py",
    STUDY_DIR + "/native_development_runner_v2.py",
    STUDY_DIR + "/identity_input_adapter_v1.py",
    STUDY_DIR + "/span_feature_transforms_v1.py",
    STUDY_DIR + "/native_capture_contract.py",
)
RUNTIME_PACKAGES = ("numpy", "scipy", "scikit-learn", "threadpoolctl")
PIN_FIELDS = ("path", "sha256")


class IdentityFitError(ValueError):
    """Structured rejection raised by this module."""


def _need(condition, code, detail=None):
    if not condition:
        raise IdentityFitError(code if detail is None else "%s: %s" % (code, detail))


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


def _load_json(raw, code):
    try:
        return span._load_json(raw, code)
    except span.SpanDriverError as exc:
        raise IdentityFitError(str(exc)) from None


def _read_pin(root, pin, role):
    try:
        return span._pinned(root, pin, role)
    except span.SpanDriverError as exc:
        raise IdentityFitError(str(exc)) from None


def _resolve(root, relative):
    try:
        return span._resolve(root, relative)
    except span.SpanDriverError as exc:
        raise IdentityFitError(str(exc)) from None


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(driver._jsonable(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def _configured(estimator):
    return core._configured(estimator)


def _short(exc):
    return "%s: %s" % (type(exc).__name__, str(exc)[:240])


def _resolved(pin):
    return (type(pin) is dict and type(pin.get("path")) is str
            and re.fullmatch(r"[0-9a-f]{64}", str(pin.get("sha256") or "")) is not None)


def _require_pin(pin, role):
    _need(_resolved(pin), "UNRESOLVED_PIN", role)
    return pin


# --------------------------------------------------------------------------- #
# prospective plan (unresolved pins stay explicit; root pins them before preflight)
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
    _need(plan.get("classifier_fits") == CLASSIFIER_FITS, "PLAN_CLASSIFIER_FITS")
    _need(plan.get("cv_fits") == CV_FITS, "PLAN_CV_FITS")
    _need(plan.get("refit_fits") == REFITS, "PLAN_REFIT_FITS")
    _need(plan.get("pca_fits") == PCA_FITS, "PLAN_PCA_FITS")
    _need(plan.get("family") == FAMILY and float(plan.get("C", -1)) == C_VALUE, "PLAN_MODEL")
    _need(list(plan.get("thresholds", [])) == list(TAUS), "PLAN_TAUS")
    _need(list(plan.get("folds", [])) == list(FOLDS), "PLAN_FOLDS")
    _need(plan.get("holdout_access") is False, "PLAN_HOLDOUT_ACCESS")
    _need(type(plan.get("identity")) is dict, "PLAN_IDENTITY")
    _need(type(plan.get("manifests")) is dict and bool(plan["manifests"]), "PLAN_MANIFESTS")
    _need(type(plan.get("old_control")) is dict, "PLAN_OLD_CONTROL")
    _need(type(plan.get("source_files")) is dict and bool(plan["source_files"]), "PLAN_SOURCE_FILES")
    _need(type(plan.get("runtime_packages")) is dict and bool(plan["runtime_packages"]), "PLAN_RUNTIME")
    return plan


def verify_sources_and_runtime(plan, root):
    """Prospective code/runtime pins; no Git and no import of pinned code."""
    for relative, expected in plan["source_files"].items():
        _need(type(relative) is str and relative != "", "SOURCE_PATH")
        _need(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected or ""),
              "UNRESOLVED_PIN", "source %s" % relative)
        path = _resolve(root, relative)
        _need(path.is_file(), "SOURCE_MISSING", relative)
        _need(_sha_path(path) == expected, "SOURCE_MISMATCH", relative)
    for name, expected in plan["runtime_packages"].items():
        _need(type(name) is str and type(expected) is str, "UNRESOLVED_PIN", "runtime %s" % name)
        _need(importlib.metadata.version(name) == expected, "RUNTIME_VERSION", name)


# --------------------------------------------------------------------------- #
# identity capture authentication (pure verifiers, then the pinned composer)
# --------------------------------------------------------------------------- #
def verify_identity_lock(lock, run_id):
    _need(type(lock) is dict and lock.get("schema") == IDENTITY_LOCK_SCHEMA, "IDENTITY_LOCK_SCHEMA")
    _need(lock.get("scientific_execution_authorized") is True, "IDENTITY_LOCK_NOT_AUTHORIZED")
    _need(lock.get("run_id") == run_id, "IDENTITY_LOCK_RUN_ID")
    caps = lock.get("caps")
    _need(type(caps) is dict, "IDENTITY_CAPS")
    _need(caps.get("fits") == 0 and caps.get("derivatives") == 0, "IDENTITY_CAPS_FITS")
    _need(caps.get("model_loads") == 1 and caps.get("tokenizer_loads") == 1, "IDENTITY_CAPS_LOADS")
    _need(caps.get("forwards") == span.TOTAL_VIEWS and caps.get("tokens_per_view") == 320,
          "IDENTITY_CAPS_WORK")
    _need(type(caps.get("raw_bytes")) is int and caps["raw_bytes"] > 0, "IDENTITY_CAPS_RAW")
    return caps


def verify_source_pins(lock, root):
    """Every identity lock source pin must hash-match the working tree."""
    sources = lock.get("source_files")
    _need(type(sources) is dict and bool(sources), "IDENTITY_SOURCE_SET")
    for relative, expected in sources.items():
        _need(type(relative) is str and re.fullmatch(r"[0-9a-f]{64}", str(expected or "")),
              "IDENTITY_SOURCE_PIN", str(relative))
        path = _resolve(root, relative)
        _need(path.is_file(), "IDENTITY_SOURCE_MISSING", relative)
        _need(_sha_path(path) == expected, "IDENTITY_SOURCE_MISMATCH", relative)
    return sources


def verify_index_binding(index, caps, lock_sha256, run_id, windows_raw):
    """Identity index lock/query/condition/layout/accounting; rejects a cache mix."""
    _need(type(index) is dict and index.get("schema") == IDENTITY_INDEX_SCHEMA, "IDENTITY_INDEX_SCHEMA")
    _need(index.get("condition") == IDENTITY_CONDITION, "IDENTITY_INDEX_CONDITION")
    _need(index.get("query") == tokens.IDENTITY_QUERY and index.get("query_sha256") == tokens.QUERY_SHA256,
          "IDENTITY_INDEX_QUERY")
    _need(index.get("lock_sha256") == lock_sha256, "IDENTITY_INDEX_LOCK")
    _need(index.get("run_id") == run_id, "IDENTITY_INDEX_RUN_ID")
    _need(index.get("binary_file") == "windows.f32" and index.get("binary_schema") == IDENTITY_WINDOWS_SCHEMA,
          "IDENTITY_INDEX_BINARY")
    _need(index.get("blocks") == list(span.BLOCKS) and index.get("orders") == list(span.ORDERS),
          "IDENTITY_INDEX_LAYOUT")
    _need(index.get("dtype") == "float32" and index.get("byte_order") == "little", "IDENTITY_INDEX_DTYPE")
    _need(index.get("float_bytes") == span.FLOAT_BYTES and index.get("width") == span.WIDTH,
          "IDENTITY_INDEX_WIDTH")
    _need(index.get("window_max") == span.WINDOW_MAX, "IDENTITY_INDEX_WINDOW")
    _need(index.get("raw_bytes") == len(windows_raw) <= caps["raw_bytes"], "IDENTITY_INDEX_RAW")
    case_order = index.get("case_order")
    _need(type(case_order) is list and case_order and len(set(case_order)) == len(case_order),
          "IDENTITY_INDEX_CASE_ORDER")
    views = len(case_order) * len(span.ORDERS)
    _need(index.get("case_count") == len(case_order), "IDENTITY_INDEX_CASES")
    _need(index.get("view_count") == views and index.get("forward_count") == views, "IDENTITY_INDEX_VIEWS")
    _need(views == caps["forwards"], "IDENTITY_FORWARD_ACCOUNTING", str(views))
    return case_order


def _identity_record_map(index, caps):
    records = index.get("records")
    _need(type(records) is list and records, "IDENTITY_RECORDS")
    mapping = {}
    for record in records:
        _need(type(record) is dict, "IDENTITY_RECORD")
        case_id, order, block = record.get("case_id"), record.get("order"), record.get("block")
        _need(type(case_id) is str and case_id, "IDENTITY_RECORD_CASE")
        _need(order in span.ORDERS, "IDENTITY_RECORD_ORDER")
        _need(block in span.BLOCKS, "IDENTITY_RECORD_BLOCK")
        _need(record.get("condition") == IDENTITY_CONDITION, "IDENTITY_RECORD_CONDITION")
        _need(record.get("query_sha256") == tokens.QUERY_SHA256, "IDENTITY_RECORD_QUERY")
        key = (case_id, order, block)
        _need(key not in mapping, "IDENTITY_RECORD_DUPLICATE", str(key))
        prefix, readout = record.get("prefix_length"), record.get("readout_index")
        _need(type(prefix) is int and type(readout) is int and prefix == readout + 1, "IDENTITY_RECORD_PREFIX")
        _need(1 <= prefix <= caps["tokens_per_view"], "IDENTITY_PREFIX_BOUND", str(prefix))
        _need(type(record.get("last_shared_token_id")) is int, "IDENTITY_RECORD_LAST_TOKEN")
        _need(isinstance(record.get("prefix_sha256"), str), "IDENTITY_RECORD_PREFIX_SHA")
        _need(isinstance(record.get("input_ids_sha256"), str), "IDENTITY_RECORD_INPUT_SHA")
        mapping[key] = {"prefix_length": prefix, "readout_index": readout,
                        "last_shared_token_id": record["last_shared_token_id"],
                        "prefix_sha256": record["prefix_sha256"],
                        "input_ids_sha256": record["input_ids_sha256"]}
    case_order = index.get("case_order")
    expected = len(case_order) * len(span.ORDERS) * len(span.BLOCKS)
    _need(len(mapping) == expected, "IDENTITY_RECORD_ACCOUNTING", str(len(mapping)))
    _need(set(key[0] for key in mapping) == set(case_order), "IDENTITY_RECORD_CASES")
    for case_id in case_order:
        for order in span.ORDERS:
            for block in span.BLOCKS:
                _need((case_id, order, block) in mapping, "IDENTITY_RECORD_MISSING",
                      "%s|%s|%s" % (case_id, order, block))
    return mapping, case_order


def verify_last_token_provenance(index, sanity, caps):
    """Bind every record and the last shared pre-option token to the lock sanity."""
    mapping, case_order = _identity_record_map(index, caps)
    _need(type(sanity) is dict and sanity.get("status") == "PASS", "IDENTITY_SANITY_STATUS")
    _need(sanity.get("condition") == IDENTITY_CONDITION, "IDENTITY_SANITY_CONDITION")
    _need(sanity.get("query_sha256") == tokens.QUERY_SHA256, "IDENTITY_SANITY_QUERY")
    _need(sanity.get("exact_decode_reencode") is True and sanity.get("query_in_shared_prefix") is True,
          "IDENTITY_SANITY_ROUNDTRIP")
    _need(type(sanity.get("last_shared_token_ids")) is list and sanity["last_shared_token_ids"]
          and all(type(value) is int for value in sanity["last_shared_token_ids"]),
          "IDENTITY_SANITY_BOUNDARY")
    _need(sanity.get("max_full_tokens", 10 ** 9) <= caps["tokens_per_view"], "IDENTITY_SANITY_TOKEN_CAP")
    counts = sanity.get("counts")
    _need(type(counts) is dict, "IDENTITY_SANITY_COUNTS")
    _need(counts.get("cases") == len(case_order) and counts.get("views") == len(case_order) * len(span.ORDERS),
          "IDENTITY_SANITY_COUNT_BINDING")
    _need(counts.get("tokenizer_loads") == 1 and counts.get("model_loads") == 0, "IDENTITY_SANITY_LOADS")
    _need(counts.get("forwards") == 0 and counts.get("fits") == 0, "IDENTITY_SANITY_WORK")
    rows = sanity.get("rows")
    _need(type(rows) is list and len(rows) == len(case_order) * len(span.ORDERS), "IDENTITY_SANITY_ROWS")
    seen = set()
    expected = {(case_id, order) for case_id in case_order for order in span.ORDERS}
    for row in rows:
        _need(type(row) is dict, "IDENTITY_SANITY_ROW")
        key = (row.get("case_id"), row.get("order"))
        _need(key in expected and key not in seen, "IDENTITY_SANITY_ROW_KEY", str(key))
        seen.add(key)
        _need(row.get("last_shared_token_id") in sanity["last_shared_token_ids"],
              "IDENTITY_SANITY_LAST_TOKEN", str(key))
        prefixes = {mapping[(key[0], key[1], block)]["prefix_length"] for block in span.BLOCKS}
        _need(len(prefixes) == 1 and row.get("prefix_tokens") == prefixes.pop(),
              "IDENTITY_BLOCK_PREFIX_MISMATCH", str(key))
        last_ids = {mapping[(key[0], key[1], block)]["last_shared_token_id"] for block in span.BLOCKS}
        _need(len(last_ids) == 1 and row.get("last_shared_token_id") == last_ids.pop(),
              "IDENTITY_BLOCK_LAST_TOKEN", str(key))
        hashes = {mapping[(key[0], key[1], block)]["input_ids_sha256"] for block in span.BLOCKS}
        _need(len(hashes) == 1 and row.get("input_sha256") == hashes.pop(), "IDENTITY_SANITY_INPUT", str(key))
        _need(type(row.get("tokens")) is int and row["tokens"] <= caps["tokens_per_view"],
              "IDENTITY_SANITY_TOKENS", str(key))
    _need(seen == expected, "IDENTITY_SANITY_COVERAGE")
    return {"cases": len(case_order), "rows": len(rows)}


def verify_identity_receipt(receipt, lock_sha256, run_id, windows_pin, index_pin):
    _need(type(receipt) is dict and receipt.get("schema") == IDENTITY_RECEIPT_SCHEMA, "IDENTITY_RECEIPT_SCHEMA")
    _need(receipt.get("status") == "capture_complete", "IDENTITY_RECEIPT_STATUS")
    _need(receipt.get("condition") == IDENTITY_CONDITION, "IDENTITY_RECEIPT_CONDITION")
    _need(receipt.get("query_sha256") == tokens.QUERY_SHA256, "IDENTITY_RECEIPT_QUERY")
    _need(receipt.get("lock_sha256") == lock_sha256 and receipt.get("run_id") == run_id,
          "IDENTITY_RECEIPT_BINDING")
    counters = receipt.get("counters") or {}
    _need(counters.get("fits") == 0 and counters.get("derivatives") == 0, "IDENTITY_RECEIPT_FITS")
    _need(counters.get("model_loads") == 1 and counters.get("tokenizer_loads") == 1, "IDENTITY_RECEIPT_LOADS")
    artifacts = receipt.get("artifacts")
    _need(type(artifacts) is dict, "IDENTITY_RECEIPT_ARTIFACTS")
    # The runner records only the two streamed artifacts in the receipt body; the
    # receipt's own pin exists only in supervisor_success.json outputs.
    for name, pin in (("windows.f32", windows_pin), ("index.json", index_pin)):
        _need(name in artifacts and artifacts[name].get("sha256") == pin["sha256"]
              and artifacts[name].get("bytes") == pin.get("bytes"), "IDENTITY_RECEIPT_PIN", name)


def verify_identity_success(success, lock_sha256, run_id, windows_pin, index_pin, receipt_pin):
    _need(type(success) is dict and success.get("status") == "complete", "IDENTITY_SUCCESS")
    _need(success.get("lock_sha256") == lock_sha256 and success.get("run_id") == run_id,
          "IDENTITY_SUCCESS_BINDING")
    outputs = success.get("outputs")
    _need(type(outputs) is dict, "IDENTITY_SUCCESS_OUTPUTS")
    for name, pin in (("windows.f32", windows_pin), ("index.json", index_pin),
                      ("capture_receipt.json", receipt_pin)):
        _need(name in outputs and outputs[name].get("sha256") == pin["sha256"]
              and outputs[name].get("bytes") == pin.get("bytes"), "IDENTITY_SUCCESS_PIN", name)


def verify_identity_sanity(sanity, lock, caps, case_order):
    _need(sanity.get("adapter_sha256")
          == lock["source_files"].get(STUDY_DIR + "/identity_input_adapter_v1.py"), "IDENTITY_SANITY_ADAPTER")
    _need(sanity.get("reference_locks") == lock.get("reference_locks"), "IDENTITY_SANITY_REFERENCES")
    _need(all(type(value) is int for value in sanity.get("last_shared_token_ids", [])),
          "IDENTITY_SANITY_BOUNDARY")
    _need({row.get("case_id") for row in sanity.get("rows", [])}
          == {case_id for case_id in case_order}, "IDENTITY_SANITY_CASE_BINDING")
    _need(sanity.get("fits_existing_320_cap") is True, "IDENTITY_SANITY_CAP_FLAG")
    return sanity


def authenticate_identity(plan, root):
    """Authenticate the identity lock, source pins, index, windows, receipt and success."""
    block = plan["identity"]
    run_id = block.get("run_id")
    _need(type(run_id) is str and run_id, "IDENTITY_RUN_ID")
    for role in ("lock", "index", "windows", "receipt", "success"):
        _require_pin(block.get(role), "identity %s" % role)
    lock_path, lock_raw = _read_pin(root, block["lock"], "identity_lock")
    lock = _load_json(lock_raw, "IDENTITY_LOCK_JSON")
    caps = verify_identity_lock(lock, run_id)
    verify_source_pins(lock, root)

    index_path, index_raw = _read_pin(root, block["index"], "identity_index")
    windows_path, windows_raw = _read_pin(root, block["windows"], "identity_windows")
    receipt_path, receipt_raw = _read_pin(root, block["receipt"], "identity_receipt")
    success_path, success_raw = _read_pin(root, block["success"], "identity_success")
    index = _load_json(index_raw, "IDENTITY_INDEX_JSON")
    receipt = _load_json(receipt_raw, "IDENTITY_RECEIPT_JSON")
    success = _load_json(success_raw, "IDENTITY_SUCCESS_JSON")

    case_order = verify_index_binding(index, caps, block["lock"]["sha256"], run_id, windows_raw)
    verify_identity_receipt(receipt, block["lock"]["sha256"], run_id, block["windows"], block["index"])
    verify_identity_success(success, block["lock"]["sha256"], run_id, block["windows"], block["index"],
                            block["receipt"])
    sanity_pin = (lock.get("inputs") or {}).get("identity_sanity")
    _require_pin(sanity_pin, "identity sanity")
    _, sanity_raw = _read_pin(root, sanity_pin, "identity_sanity")
    sanity = verify_identity_sanity(_load_json(sanity_raw, "IDENTITY_SANITY_JSON"), lock, caps, case_order)
    provenance = verify_last_token_provenance(index, sanity, caps)
    return {
        "lock": lock, "lock_sha256": block["lock"]["sha256"], "lock_path": str(lock_path),
        "index": index, "index_path": str(index_path), "index_sha256": block["index"]["sha256"],
        "windows_raw": windows_raw, "windows_path": str(windows_path),
        "windows_sha256": block["windows"]["sha256"], "receipt": receipt, "success": success,
        "sanity": sanity, "query_sha256": tokens.QUERY_SHA256, "run_id": run_id,
        "provenance": provenance,
    }


def load_identity_case_data(auth, plan, root):
    """Reuse the reviewed span loader; labels/groups/folds come only from the pinned manifests."""
    source = {"index": auth["index"], "windows_raw": auth["windows_raw"]}
    try:
        return span.load_case_data(source, {"manifests": plan["manifests"]}, root)
    except span.SpanDriverError as exc:
        raise IdentityFitError(str(exc)) from None


# --------------------------------------------------------------------------- #
# old prompted control: saved OOF + saved predictions, never the pickled model
# --------------------------------------------------------------------------- #
def load_old_control(plan, root):
    block = plan["old_control"]
    _need(block.get("run_id") == OLD_CONTROL_RUN_ID, "OLD_RUN_ID")
    _need(block.get("condition") == OLD_CONTROL_CONDITION, "OLD_CONDITION")
    _need(block.get("dimension") == OLD_DIMENSION and block.get("family") == OLD_FAMILY
          and float(block.get("C", -1)) == OLD_C, "OLD_MODEL")
    _require_pin(block.get("cv_scores"), "old cv_scores")
    _require_pin(block.get("model_results"), "old model_results")
    _, cv_raw = _read_pin(root, block["cv_scores"], "old_cv_scores")
    _, prediction_raw = _read_pin(root, block["model_results"], "old_model_results")
    cv = _load_json(cv_raw, "OLD_CV_JSON")
    prediction = _load_json(prediction_raw, "OLD_PREDICTION_JSON")

    matches = [entry for entry in cv.get("candidates", [])
               if entry.get("condition") == OLD_CONTROL_CONDITION and entry.get("dimension") == OLD_DIMENSION
               and entry.get("family") == OLD_FAMILY and entry.get("valid") is True
               and float(entry.get("C", -1)) == OLD_C]
    _need(len(matches) == 1, "OLD_CANDIDATE_MATCH", str(len(matches)))
    candidate = matches[0]
    oof = np.asarray(candidate.get("oof_p_self"), float)
    _need(oof.shape == (span.TRAIN_CASES,), "OLD_OOF_SHAPE", str(oof.shape))
    _need(bool(np.isfinite(oof).all()) and float(oof.min()) >= 0.0 and float(oof.max()) <= 1.0,
          "OLD_OOF_RANGE")
    truth = [1 if label == "SELF" else 0 for label in cv.get("oof_truth", [])]
    _need(len(truth) == span.TRAIN_CASES, "OLD_OOF_TRUTH")
    _need(prediction.get("condition") == OLD_CONTROL_CONDITION and prediction.get("dimension") == OLD_DIMENSION
          and prediction.get("family") == OLD_FAMILY and float(prediction.get("C", -1)) == OLD_C,
          "OLD_PREDICTION_MODEL")
    _need(prediction.get("status") == "FITTED" and prediction.get("reload_exact") is True,
          "OLD_PREDICTION_STATUS")
    evaluation = prediction.get("evaluation")
    _need(type(evaluation) is dict and all(split in evaluation for split, _ in SPLITS), "OLD_PREDICTION_SPLITS")
    folds = [record for record in cv.get("folds", [])
             if record.get("condition") == OLD_CONTROL_CONDITION]
    _need(len(folds) == len(FOLDS), "OLD_FOLD_RECORDS")
    group_fold = {}
    for record in folds:
        for group in record.get("test_groups", []):
            _need(group not in group_fold, "OLD_GROUP_MULTI_FOLD", group)
            group_fold[group] = record.get("fold")
    seen = set(group_fold)
    for record in folds:
        _need(set(record.get("train_groups", [])) <= seen, "OLD_GROUP_MISSING", str(record.get("fold")))
    return {
        "run_id": OLD_CONTROL_RUN_ID, "condition": OLD_CONTROL_CONDITION, "dimension": OLD_DIMENSION,
        "family": OLD_FAMILY, "C": OLD_C, "oof_case_ids": list(cv.get("oof_case_ids", [])),
        "oof_truth": list(cv.get("oof_truth", [])), "oof_p_self": oof.tolist(),
        "saved_tau": float(candidate.get("selected_tau")), "saved_oof_metrics": candidate.get("oof_metrics"),
        "saved_thresholds": candidate.get("thresholds"), "group_fold": group_fold,
        "saved_prediction": prediction, "saved_evaluation": evaluation,
        "cv_scores_sha256": block["cv_scores"]["sha256"],
        "model_results_sha256": block["model_results"]["sha256"],
        "artifact": prediction.get("artifact"), "artifact_sha256": prediction.get("artifact_sha256"),
        "pickle_loaded": False, "classifier_fits": 0, "pca_fits": 0, "refits": 0,
    }


# --------------------------------------------------------------------------- #
# TRAIN-OOF threshold rule (F1 first) and the two-condition comparison key
# --------------------------------------------------------------------------- #
def select_tau(oof_p_self, truth):
    """19-threshold TRAIN-OOF choice: max F1, max min(P,R), |tau-0.5|, smaller tau."""
    p = np.asarray(oof_p_self, float)
    y = np.asarray(truth)
    _need(p.ndim == 1 and p.size == y.size, "SELECT_INPUT_SHAPE")
    _need(bool(np.isfinite(p).all()), "SELECT_INPUT_NONFINITE")
    thresholds, best = [], None
    for tau in TAUS:
        metrics = harness.binary_gate_metrics(y, p, tau)
        eligible = harness.is_eligible(metrics)
        thresholds.append({"tau": float(tau), "precision": metrics["precision"], "recall": metrics["recall"],
                           "f1": metrics["f1"], "eligible": eligible})
        if not eligible:
            continue
        key = (-float(metrics["f1"]), -min(float(metrics["precision"]), float(metrics["recall"])),
               abs(float(tau) - 0.5), float(tau))
        if best is None or key < best[0]:
            best = (key, float(tau), metrics)
    if best is None:
        return {"selected_tau": None, "oof_metrics": None, "thresholds": thresholds,
                "error": "NO_ELIGIBLE_THRESHOLD"}
    return {"selected_tau": best[1], "oof_metrics": best[2], "thresholds": thresholds, "error": None}


def comparison_key(entry):
    """max pooled TRAIN OOF F1, max min(P,R), lower width/C, binary, |tau-0.5|, tau, old control tie."""
    metrics = entry["oof_metrics"]
    _need(type(metrics) is dict, "COMPARISON_METRICS")
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    return (-float(metrics["f1"]), -min(precision, recall), int(entry["width"]), float(entry["C"]),
            0 if entry["family"] == FAMILY else 1, abs(float(entry["tau"]) - 0.5), float(entry["tau"]),
            0 if entry["condition"] == OLD_CONTROL_CONDITION else 1)


def _saved_split_evaluation(predictions, tau):
    truth = np.asarray([row["truth"] for row in predictions])
    p_self = np.asarray([row["p_self"] for row in predictions], float)
    predicted = p_self >= tau
    return {
        "self_gate": harness.binary_metrics(truth == "SELF", predicted),
        "negative_class_false_positives": {
            label: {"false_positives": int(predicted[truth == label].sum()),
                    "total": int((truth == label).sum())}
            for label in CLASS_ORDER[1:]
        },
        "predictions": [{"case_id": row["case_id"], "truth": str(row["truth"]),
                         "p_self": float(row["p_self"]), "predicted_self": bool(flag)}
                        for row, flag in zip(predictions, predicted)],
        "threshold": float(tau),
    }


# --------------------------------------------------------------------------- #
# case-data validation, alignment with the control, grouped CV and refit
# --------------------------------------------------------------------------- #
def _validate_case_data(case_data):
    for key in ("case_order", "train_ids", "validation_ids", "original_ids", "added_ids",
                "labels", "groups", "folds", "windows"):
        _need(key in case_data, "CASE_DATA_FIELD", key)
    case_order = list(case_data["case_order"])
    _need(bool(case_order) and len(set(case_order)) == len(case_order), "CASE_ORDER")
    train_ids = list(case_data["train_ids"])
    validation_ids = list(case_data["validation_ids"])
    original_ids = list(case_data["original_ids"])
    added_ids = list(case_data["added_ids"])
    _need(len(train_ids) == span.TRAIN_CASES, "TRAIN_TOTAL")
    _need(len(validation_ids) == span.VALIDATION_CASES, "VALIDATION_TOTAL")
    _need(len(original_ids) == span.VALIDATION_ORIGINAL and len(added_ids) == span.VALIDATION_ADDED,
          "VALIDATION_PARTITION")
    _need(set(validation_ids) == set(original_ids) | set(added_ids) and not (set(original_ids) & set(added_ids)),
          "VALIDATION_PARTITION")
    _need(not (set(train_ids) & set(validation_ids)), "SPLIT_OVERLAP")
    _need(set(case_order) == set(train_ids) | set(validation_ids), "CASE_COVERAGE")
    labels, groups, folds = case_data["labels"], case_data["groups"], case_data["folds"]
    for case_id in case_order:
        _need(labels.get(case_id) in CLASS_ORDER, "CASE_LABEL", case_id)
        _need(type(groups.get(case_id)) is str and groups[case_id], "CASE_GROUP", case_id)
        if case_id in set(train_ids):
            _need(type(folds.get(case_id)) is int and folds[case_id] in FOLDS, "TRAIN_FOLD", case_id)
        else:
            _need(folds.get(case_id) is None, "VALIDATION_FOLD", case_id)
    group_fold = {}
    for case_id in train_ids:
        known = group_fold.setdefault(groups[case_id], folds[case_id])
        _need(known == folds[case_id], "GROUP_MULTI_FOLD", groups[case_id])
    for held in FOLDS:
        fit_ids = [case_id for case_id in train_ids if folds[case_id] != held]
        test_ids = [case_id for case_id in train_ids if folds[case_id] == held]
        _need(bool(fit_ids) and bool(test_ids), "FOLD_EMPTY", str(held))
        _need({labels[case_id] for case_id in fit_ids} == set(CLASS_ORDER), "FOLD_CLASSES", str(held))
        _need(not ({groups[case_id] for case_id in fit_ids} & {groups[case_id] for case_id in test_ids}),
              "FOLD_GROUP_LEAK", str(held))
    return {"case_order": case_order, "train_ids": train_ids, "validation_ids": validation_ids,
            "original_ids": original_ids, "added_ids": added_ids, "labels": labels, "groups": groups,
            "folds": folds, "group_fold": group_fold}


def verify_control_alignment(case_data, old_control):
    """Identity/control must share case order, labels, groups, folds and the matched C=10 candidate."""
    _need(old_control["oof_case_ids"] == case_data["train_ids"], "CONTROL_OOF_IDS")
    expected_truth = [case_data["labels"][case_id] for case_id in case_data["train_ids"]]
    _need(old_control["oof_truth"] == expected_truth, "CONTROL_OOF_TRUTH")
    _need(old_control["group_fold"] == case_data["group_fold"], "CONTROL_GROUP_FOLD")
    for split, key in SPLITS:
        saved = old_control["saved_evaluation"][split]["predictions"]
        _need([row["case_id"] for row in saved] == list(case_data[key]), "CONTROL_SPLIT_IDS", split)
        _need([row["truth"] for row in saved] == [case_data["labels"][case_id] for case_id in case_data[key]],
              "CONTROL_SPLIT_TRUTH", split)
    _need(float(old_control["C"]) == C_VALUE and old_control["family"] == FAMILY, "CONTROL_C10")
    return True


def _matrix_and_rows(case_data):
    order, matrix = linear.build_features(case_data)
    _need(order == list(case_data["train_ids"]) + list(case_data["validation_ids"]), "FEATURE_ORDER")
    row_of = {case_id: row for row, case_id in enumerate(order)}
    return np.ascontiguousarray(matrix), row_of


def _order_matrix(case_data, key, row_of):
    rows = [linear.features_for_case(case_data[key][case_id]) for case_id in case_data["validation_ids"]]
    matrix = np.ascontiguousarray(np.stack(rows))
    _need(matrix.shape == (span.VALIDATION_CASES, FEATURE_DIM), "ORDER_MATRIX_SHAPE", str(matrix.shape))
    return matrix


def _identity_cv(matrix, row_of, case_data, factory, counts, check, pca_states):
    train_ids = case_data["train_ids"]
    labels, groups, folds = case_data["labels"], case_data["groups"], case_data["folds"]
    truth = np.asarray([1 if labels[case_id] == "SELF" else 0 for case_id in train_ids])
    train_pos = {case_id: index for index, case_id in enumerate(train_ids)}
    oof = np.full(len(train_ids), np.nan)
    fold_records = []
    for held in FOLDS:
        check()
        fit_ids = [case_id for case_id in train_ids if folds[case_id] != held]
        test_ids = [case_id for case_id in train_ids if folds[case_id] == held]
        fit_rows = [row_of[case_id] for case_id in fit_ids]
        test_rows = [row_of[case_id] for case_id in test_ids]
        fit_groups = {groups[case_id] for case_id in fit_ids}
        test_groups = {groups[case_id] for case_id in test_ids}
        _need(not (fit_groups & test_groups), "FOLD_GROUP_LEAK", str(held))
        counts["pca_fits"] += 1
        _need(counts["pca_fits"] <= PCA_FITS, "PCA_BUDGET", str(counts["pca_fits"]))
        pca = core.PCA32(matrix[fit_rows])
        pca_states.append({"key": "identity__fold%d" % held, "condition": IDENTITY_CONDITION, "scope": "fold",
                           "fold": int(held), "n_train": len(fit_ids), "state": pca.state()})
        transformed = pca.transform(matrix, PCA_COMPONENTS)
        counts["cv_fits"] += 1
        _need(counts["cv_fits"] <= CV_FITS, "CV_BUDGET", str(counts["cv_fits"]))
        estimator = factory(FAMILY, C_VALUE)
        target = [1 if labels[case_id] == "SELF" else 0 for case_id in fit_ids]
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(transformed[fit_rows], target)
        error = _configured(estimator)
        _need(error is None, "CV_FIT_UNCONVERGED", error or "")
        p_self, _probs = driver._fold_probabilities(harness.MODEL_BINARY, estimator,
                                                    transformed[test_rows], len(test_rows))
        oof[[train_pos[case_id] for case_id in test_ids]] = p_self
        fold_records.append({"condition": IDENTITY_CONDITION, "fold": int(held), "n_train": len(fit_ids),
                             "n_test": len(test_ids), "train_groups": sorted(fit_groups),
                             "test_groups": sorted(test_groups), "error": None})
    _need(bool(np.isfinite(oof).all()), "OOF_NONFINITE")
    return oof, truth, fold_records


def _identity_condition(matrix, row_of, case_data, factory, counts, check, pca_states):
    oof, truth, fold_records = _identity_cv(matrix, row_of, case_data, factory, counts, check, pca_states)
    selection = select_tau(oof, truth)
    _need(selection["selected_tau"] is not None, "IDENTITY_NO_ELIGIBLE_THRESHOLD")
    train_rows = [row_of[case_id] for case_id in case_data["train_ids"]]
    check()
    counts["pca_fits"] += 1
    _need(counts["pca_fits"] <= PCA_FITS, "PCA_BUDGET", str(counts["pca_fits"]))
    full_pca = core.PCA32(matrix[train_rows])
    pca_states.append({"key": "identity__full", "condition": IDENTITY_CONDITION, "scope": "full_train",
                       "fold": None, "n_train": len(train_rows), "state": full_pca.state()})
    transformed = full_pca.transform(matrix, PCA_COMPONENTS)
    counts["refits"] += 1
    _need(counts["refits"] <= REFITS, "REFIT_BUDGET", str(counts["refits"]))
    estimator = factory(FAMILY, C_VALUE)
    target = [1 if case_data["labels"][case_id] == "SELF" else 0 for case_id in case_data["train_ids"]]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(transformed[train_rows], target)
    error = _configured(estimator)
    _need(error is None, "REFIT_UNCONVERGED", error or "")
    tau = float(selection["selected_tau"])
    evaluation = {}
    for split, key in SPLITS:
        split_ids = list(case_data[key])
        rows = [row_of[case_id] for case_id in split_ids]
        ab_matrix = ba_matrix = None
        if split == "combined80":
            ab_matrix = full_pca.transform(_order_matrix(case_data, "ab_windows", row_of), PCA_COMPONENTS)
            ba_matrix = full_pca.transform(_order_matrix(case_data, "ba_windows", row_of), PCA_COMPONENTS)
        evaluation[split] = span._evaluate_split(
            FAMILY, estimator, transformed[rows], split_ids,
            np.asarray([case_data["labels"][case_id] for case_id in split_ids]), tau,
            ab_matrix=ab_matrix, ba_matrix=ba_matrix)
    return {
        "condition": IDENTITY_CONDITION, "dimension": OLD_DIMENSION, "family": FAMILY, "C": C_VALUE,
        "width": PCA_COMPONENTS, "tau": tau, "source": "NEW_FIT",
        "oof_metrics": selection["oof_metrics"], "thresholds": selection["thresholds"],
        "oof_p_self": oof.tolist(), "fold_records": fold_records,
        "evaluation": evaluation, "saved_tau": None,
        "retained_variance": {
            "n_components": int(full_pca.n_components_),
            "explained_variance_ratio": [float(value) for value in full_pca.explained_variance_ratio_],
            "cumulative_explained_variance": float(np.sum(full_pca.explained_variance_ratio_)),
        },
    }


def _old_condition(old_control):
    truth = [1 if label == "SELF" else 0 for label in old_control["oof_truth"]]
    selection = select_tau(old_control["oof_p_self"], truth)
    _need(selection["selected_tau"] is not None, "CONTROL_NO_ELIGIBLE_THRESHOLD")
    tau = float(selection["selected_tau"])
    evaluation = {split: _saved_split_evaluation(old_control["saved_evaluation"][split]["predictions"], tau)
                  for split, _ in SPLITS}
    saved = {split: old_control["saved_evaluation"][split] for split, _ in SPLITS}
    return {
        "condition": OLD_CONTROL_CONDITION, "dimension": OLD_DIMENSION, "family": FAMILY, "C": OLD_C,
        "width": PCA_COMPONENTS, "tau": tau, "source": "SAVED_REUSE", "oof_metrics": selection["oof_metrics"],
        "thresholds": selection["thresholds"], "oof_p_self": list(old_control["oof_p_self"]),
        "evaluation": evaluation, "saved_tau": old_control["saved_tau"],
        "saved_evaluation": saved, "saved_oof_metrics": old_control["saved_oof_metrics"],
        "artifact": old_control["artifact"], "artifact_sha256": old_control["artifact_sha256"],
    }


def _write_pca_index(output, pca_states):
    fits = []
    for entry in pca_states:
        state = entry["state"]
        fits.append({"key": entry["key"], "condition": entry["condition"], "scope": entry["scope"],
                     "fold": entry["fold"], "n_train": entry["n_train"], "n_components": state["n_components"],
                     "svd_solver": state["svd_solver"], "whiten": state["whiten"],
                     "explained_variance_ratio": [float(value) for value in state["explained_variance_ratio_"]],
                     "mean_sha256": _sha(np.asarray(state["mean_"], float).tobytes()),
                     "components_sha256": _sha(np.asarray(state["components_"], float).tobytes())})
    _write_json(output / "pca_index.json", {"schema": "identity_fit_pca_index.v1", "fits": fits})


def _counters():
    return {"cv_fits": 0, "refits": 0, "pca_fits": 0, "fit_errors": 0,
            "old_cv_fits": 0, "old_refits": 0, "old_pca_fits": 0}


def _compare(case_data, old_control, output, counts, *, factory, deadline, provenance, started):
    case = _validate_case_data(case_data)
    verify_control_alignment(case, old_control)

    def check():
        _need(deadline is None or not deadline(), "DEADLINE")

    check()
    output.mkdir(parents=True)
    owners = factory if factory is not None else driver._default_factory
    matrix, row_of = _matrix_and_rows(case_data)
    _need(matrix.shape == (span.TOTAL_CASES, FEATURE_DIM), "IDENTITY_MATRIX_SHAPE", str(matrix.shape))
    pca_states = []
    identity = _identity_condition(matrix, row_of, case_data, owners, counts, check, pca_states)
    control = _old_condition(old_control)
    entries = [identity, control]
    ranked = sorted(entries, key=comparison_key)
    counts["cells"] = len(entries)
    result = {
        "status": "COMPLETE",
        "job_id": JOB_ID,
        "counters": dict(counts, seconds=float(time.monotonic() - started)),
        "limits": {"classifier_fits": CLASSIFIER_FITS, "cv_fits": CV_FITS, "refits": REFITS,
                   "pca_fits": PCA_FITS},
        "data_counts": {"train": span.TRAIN_CASES, "original40": span.VALIDATION_ORIGINAL,
                        "added40": span.VALIDATION_ADDED, "combined80": span.VALIDATION_CASES},
        "conditions": [OLD_CONTROL_CONDITION, IDENTITY_CONDITION],
        "candidate_order": "TRAIN OOF max F1, max min(precision,recall), lower width, lower C, binary family, "
                           "|tau-0.5|, smaller tau, old control tie",
        "selection_policy": "per candidate: max TRAIN-OOF F1, max min(precision,recall), |tau-0.5|, smaller tau",
        "cells": {"%s|%s|%s" % (entry["condition"], entry["dimension"], entry["family"]): entry
                  for entry in entries},
        "winner": ranked[0],
        "matched_control_baseline": {
            "run_id": OLD_CONTROL_RUN_ID, "condition": OLD_CONTROL_CONDITION, "dimension": OLD_DIMENSION,
            "family": OLD_FAMILY, "C": OLD_C, "saved_tau": old_control["saved_tau"],
            "saved_oof_metrics": old_control["saved_oof_metrics"],
            "current_rule_tau": control["tau"], "current_rule_oof_metrics": control["oof_metrics"],
            "cv_scores_sha256": old_control["cv_scores_sha256"],
            "model_results_sha256": old_control["model_results_sha256"],
            "pickle_loaded": False,
        },
        "provenance": provenance,
        "validation_used_for_selection": False,
        "holdout_accessed": False,
    }
    _write_json(output / "cv_scores.json", {
        "job_id": JOB_ID, "candidate_order": result["candidate_order"],
        "identity": {"oof_case_ids": case["train_ids"],
                     "oof_truth": [case["labels"][case_id] for case_id in case["train_ids"]],
                     "selected_tau": identity["tau"], "oof_metrics": identity["oof_metrics"],
                     "thresholds": identity["thresholds"], "oof_p_self": identity["oof_p_self"],
                     "fold_records": identity["fold_records"]},
        "control": {"oof_case_ids": old_control["oof_case_ids"], "oof_truth": old_control["oof_truth"],
                    "saved_tau": control["saved_tau"], "selected_tau": control["tau"],
                    "oof_metrics": control["oof_metrics"], "thresholds": control["thresholds"],
                    "oof_p_self": control["oof_p_self"], "source": "SAVED_REUSE"},
        "train_ranking": [{"condition": entry["condition"], "tau": entry["tau"],
                           "oof_metrics": entry["oof_metrics"], "key": list(comparison_key(entry))}
                          for entry in ranked],
    })
    _write_json(output / "identity_results.json", identity)
    _write_json(output / "old_control.json", {
        "job_id": JOB_ID, "run_id": OLD_CONTROL_RUN_ID, "condition": OLD_CONTROL_CONDITION,
        "dimension": OLD_DIMENSION, "family": OLD_FAMILY, "C": OLD_C,
        "saved_tau": old_control["saved_tau"], "saved_oof_metrics": old_control["saved_oof_metrics"],
        "saved_evaluation": control["saved_evaluation"], "current_rule": control,
        "cv_scores_sha256": old_control["cv_scores_sha256"],
        "model_results_sha256": old_control["model_results_sha256"],
        "artifact": old_control["artifact"], "artifact_sha256": old_control["artifact_sha256"],
        "pickle_loaded": False, "classifier_fits": 0, "pca_fits": 0, "refits": 0,
    })
    _write_json(output / "comparison.json", {
        "job_id": JOB_ID, "conditions": result["conditions"], "candidate_order": result["candidate_order"],
        "selection_policy": result["selection_policy"], "cells": result["cells"],
        "winner_condition": result["winner"]["condition"],
        "matched_control_baseline": result["matched_control_baseline"],
        "identity_retained_variance": identity["retained_variance"],
        "validation_used_for_selection": False, "holdout_accessed": False,
    })
    _write_json(output / "feature_definition.json", {
        "schema": "identity_fit_feature_definition.v1", "job_id": JOB_ID,
        "blocks": list(span.BLOCKS), "position": "last_shared_preoption_input",
        "aggregation": "per-layer L2 of each 1024 vector then concatenation in block order 6,10,18 -> 3072",
        "pca": {"components": PCA_COMPONENTS, "svd_solver": "full", "whiten": False,
                "fit_scope": "TRAIN rows only, 5 folds + 1 full TRAIN"},
        "family": "fixed L2 logistic regression", "C": C_VALUE, "folds": list(FOLDS), "taus": list(TAUS),
    })
    _write_pca_index(output, pca_states)
    _write_json(output / "development_results.json", result)
    return result


def compare(case_data, old_control, output_dir, *, factory=None, deadline=None, provenance=None):
    """Compare the fresh identity condition against the saved prompted control."""
    output = Path(output_dir)
    _need(not output.exists(), "OUTPUT_EXISTS", str(output))
    counts = _counters()
    started = time.monotonic()
    try:
        return _compare(case_data, old_control, output, counts, factory=factory, deadline=deadline,
                        provenance=provenance, started=started)
    except BaseException as exc:
        if output.exists():
            try:
                _write_json(output / "failure.json", {"status": "failed", "job_id": JOB_ID,
                                                      "code": _short(exc), "error_type": type(exc).__name__,
                                                      "counters": dict(counts,
                                                                       seconds=time.monotonic() - started),
                                                      "holdout_accessed": False})
            except OSError:
                pass
        raise


# --------------------------------------------------------------------------- #
# assembled inputs, zero-fit preflight and supervised run
# --------------------------------------------------------------------------- #
def output_dir(plan, root):
    base = (Path(root).resolve() / STUDY_DIR / "runs").resolve()
    _need(base.is_relative_to(Path(root).resolve()), "OUTPUT_SCOPE")
    return base / plan["run_id"]


def _provenance(plan, auth):
    return {
        "job_id": JOB_ID,
        "identity": {"run_id": auth["run_id"], "lock_sha256": auth["lock_sha256"],
                     "index_sha256": auth["index_sha256"], "windows_sha256": auth["windows_sha256"],
                     "query_sha256": auth["query_sha256"], "rows": auth["provenance"]["rows"]},
        "old_control": {"run_id": OLD_CONTROL_RUN_ID,
                        "cv_scores_sha256": plan["old_control"]["cv_scores"]["sha256"],
                        "model_results_sha256": plan["old_control"]["model_results"]["sha256"],
                        "pickle_loaded": False},
        "holdout_accessed": False,
    }


def build_inputs(plan, root):
    """Authenticate every prospective pin and assemble aligned comparison inputs."""
    verify_sources_and_runtime(plan, root)
    auth = authenticate_identity(plan, root)
    case_data = load_identity_case_data(auth, plan, root)
    validated = _validate_case_data(case_data)
    case_data["group_fold"] = validated["group_fold"]
    old_control = load_old_control(plan, root)
    verify_control_alignment(validated, old_control)
    return {"case_data": case_data, "old_control": old_control, "auth": auth,
            "provenance": _provenance(plan, auth), "output": output_dir(plan, root)}


def preflight(plan_path, plan_sha256, root):
    """Authenticate and assemble everything with zero estimator/PCA fits."""
    plan = load_plan(plan_path, plan_sha256)
    inputs = build_inputs(plan, root)
    _need(not inputs["output"].exists(), "PREFLIGHT_OUTPUT_EXISTS", str(inputs["output"]))
    matrix, _rows = _matrix_and_rows(inputs["case_data"])
    _need(matrix.shape == (span.TOTAL_CASES, FEATURE_DIM), "IDENTITY_MATRIX_SHAPE")
    return {
        "status": "preflight_pass", "job_id": JOB_ID, "run_id": plan["run_id"],
        "cases": span.TOTAL_CASES, "train": span.TRAIN_CASES, "validation": span.VALIDATION_CASES,
        "original": span.VALIDATION_ORIGINAL, "added": span.VALIDATION_ADDED, "feature_width": FEATURE_DIM,
        "conditions": [OLD_CONTROL_CONDITION, IDENTITY_CONDITION], "query_sha256": inputs["auth"]["query_sha256"],
        "identity_run_id": inputs["auth"]["run_id"], "identity_rows": inputs["auth"]["provenance"]["rows"],
        "old_control_run_id": OLD_CONTROL_RUN_ID, "old_control_saved_tau": inputs["old_control"]["saved_tau"],
        "classifier_fits": 0, "pca_fits": 0, "holdout_accessed": False,
    }


def _verify_result(result, plan, output):
    _need(result.get("status") == "COMPLETE", "RESULT_STATUS", str(result.get("status")))
    counters = result.get("counters") or {}
    _need(counters.get("cv_fits") == CV_FITS, "RESULT_CV_BUDGET", str(counters.get("cv_fits")))
    _need(counters.get("refits") == REFITS, "RESULT_REFIT_BUDGET", str(counters.get("refits")))
    _need(counters.get("pca_fits") == PCA_FITS, "RESULT_PCA_BUDGET", str(counters.get("pca_fits")))
    _need(counters.get("old_cv_fits") == 0 and counters.get("old_pca_fits") == 0
          and counters.get("old_refits") == 0, "RESULT_OLD_FITS")
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
    result = compare(inputs["case_data"], inputs["old_control"], output, factory=factory, deadline=deadline,
                     provenance=inputs["provenance"])
    _verify_result(result, plan, output)
    return result
