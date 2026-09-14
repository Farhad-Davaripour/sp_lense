"""Model-free span-window classifier driver + loader V1 (job span_fit_driver_implementation_20260914_1120).

This module is the single fit-side owner for the reviewed span/layer proposal
(``PROMPT_SPAN_LAYER_PLAN_V2.md``). It imports no ``torch``/``transformers`` or
any other native provider, captures nothing, loads no model or tokenizer, and
performs no network/install/Git/coordination work. A real XGBoost estimator is
constructed lazily only when the default factory is actually called; the unit
tests inject a deterministic toy factory and never fit XGBoost.

Scope
-----
* Decode the reviewed ``span_development_runner_v1`` binary/index artifacts:
  authenticate the completed parent receipt, the execution-lock SHA, the
  ``windows.f32``/``index.json`` byte+SHA pins, the record offset bounds and
  non-overlap, the fixed 1920 = 320 cases x 2 orders x 3 blocks accounting,
  little-endian float32 width-1024 rows, explicit last-1..16 token positions,
  and case IDs/splits/groups/folds from separately pinned original+added
  development manifests. HOLDOUT paths are rejected and never read.
* Reconstruct each logical case, pair-average the AB/BA views per block, then
  learn the shared F1..F5 transform once per TRAIN fold and once on full TRAIN.
  The 8272-wide matrix is sliced into the five fixed representations and each
  representation trains its own binary / four-class model (never one combined
  classifier).
* Rank candidates on TRAIN grouped-OOF min(precision,recall), then F1, then
  lower feature count, tau nearest 0.5, smaller tau, deterministic binary-first
  family tie. A candidate with any invalid fold is discarded whole; surviving
  folds are never pooled. The global TRAIN-selected candidate is written
  separately from the exploratory validation winner.
* Budget: 50 CV classifier fits + at most 10 full-TRAIN refits + exactly
  5 fold + 1 full shared-transform fits. No hyperparameter grid. Deadline is
  the prospective plan's ``seconds`` (<= 600). Run outputs are exclusive and
  versioned; failures leave ``failure.json`` evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np

import grouped_driver as driver
import harness
import span_feature_transforms_v1 as features

__all__ = [
    "SpanDriverError",
    "need",
    "JOB_ID",
    "PLAN_SCHEMA",
    "INDEX_SCHEMA",
    "LOCK_SCHEMA",
    "FEATURE_DIM",
    "REPRESENTATIONS",
    "REPRESENTATION_WIDTHS",
    "REPRESENTATION_SLICES",
    "FAMILIES",
    "TAUS",
    "EXPECTED_XGB_PARAMS",
    "XGBEstimator",
    "XGBFactory",
    "load_plan",
    "load_source",
    "load_manifests",
    "decode_index",
    "load_case_data",
    "preflight",
    "run",
    "main",
]

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

JOB_ID = "span_fit_driver_implementation_20260914_1120"
PLAN_SCHEMA = "span_classifier_fit_plan.v1"
INDEX_SCHEMA = "span_capture_index.v1"
LOCK_SCHEMA = "span_development_execution.v1"
RECEIPT_STATUSES = ("complete",)  # Parent supervisor success, not worker-only completion.

CLASS_ORDER = ("SELF", "OTHER", "NONTERMINATION", "ORDINARY")
BLOCKS = (6, 10, 18)
ORDERS = ("AB", "BA")
WIDTH = 1024
WINDOW_MAX = 16
FLOAT_BYTES = 4
TOTAL_CASES = 320
TOTAL_VIEWS = 640
TOTAL_RECORDS = 1920
TRAIN_CASES = 240
VALIDATION_CASES = 80
VALIDATION_ORIGINAL = 40
VALIDATION_ADDED = 40
FOLDS = (0, 1, 2, 3, 4)

TAUS = tuple(round(0.05 * index, 2) for index in range(1, 20))
FAMILIES = ("binary", "fourclass")
REPRESENTATIONS = ("concat3_last", "concat3_mean", "block10_last_mean", "contrast_stats", "pca8_products")
REPRESENTATION_WIDTHS = {
    "concat3_last": 3072,
    "concat3_mean": 3072,
    "block10_last_mean": 2048,
    "contrast_stats": 36,
    "pca8_products": 44,
}

DEADLINE_LIMIT = 600
CV_FIT_LIMIT = 50
REFIT_LIMIT = 10
TRANSFORM_FIT_LIMIT = 6

EXPECTED_XGB_PARAMS = {
    "max_depth": 2,
    "n_estimators": 100,
    "learning_rate": 0.05,
    "min_child_weight": 3,
    "reg_lambda": 5,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "device": "cpu",
    "n_jobs": 1,
    "random_state": 0,
}

SOURCE_ROLES = ("receipt", "lock", "index", "windows")
MANIFEST_ROLES = {
    "original_train": ("TRAIN", "original"),
    "original_validation": ("VALIDATION", "original"),
    "added_train": ("TRAIN", "added"),
    "added_validation": ("VALIDATION", "added"),
}
BLUEPRINT_ROLE = "blueprint"


def _slices():
    slices, offset = {}, 0
    for name in REPRESENTATIONS:
        width = REPRESENTATION_WIDTHS[name]
        slices[name] = slice(offset, offset + width)
        offset += width
    return slices, offset


REPRESENTATION_SLICES, FEATURE_DIM = _slices()
_FEATURE_WIDTHS = {
    "concat3_last": features.F1_DIM,
    "concat3_mean": features.F2_DIM,
    "block10_last_mean": features.F3_DIM,
    "contrast_stats": features.F4_DIM,
    "pca8_products": features.F5_DIM,
}
if FEATURE_DIM != features.FEATURE_DIM or FEATURE_DIM != 8272:  # pragma: no cover - import guard
    raise RuntimeError("representation widths do not match the frozen 8272-wide transform")
if tuple(REPRESENTATION_WIDTHS[name] for name in REPRESENTATIONS) != tuple(
    _FEATURE_WIDTHS[name] for name in REPRESENTATIONS
):  # pragma: no cover - import guard
    raise RuntimeError("representation slice order does not match span_feature_transforms_v1")
if tuple(features.LAYER_BLOCKS) != BLOCKS or features.WIDTH != WIDTH or features.MAX_WINDOW != WINDOW_MAX:
    raise RuntimeError("span_feature_transforms_v1 fixed constants changed")  # pragma: no cover


class SpanDriverError(ValueError):
    """Structured rejection raised by this module."""


def need(condition, code, detail=None):
    if not condition:
        raise SpanDriverError(code if detail is None else "%s: %s" % (code, detail))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _short(exc):
    return "%s: %s" % (type(exc).__name__, str(exc)[:300])


def _resolve(root, relative):
    need(type(relative) is str and relative != "", "PATH_TYPE")
    need(not Path(relative).is_absolute(), "PATH_ABSOLUTE")
    lowered = [part.lower() for part in Path(relative).parts]
    need(not any("holdout" in part or "private" in part for part in lowered), "PATH_FORBIDDEN")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    need(path.is_relative_to(root), "PATH_SCOPE")
    return path


def _pinned(root, pin, role):
    need(type(pin) is dict, "PIN_SCHEMA", role)
    need({"path", "sha256"} <= set(pin), "PIN_FIELDS", role)
    path = _resolve(root, pin["path"])
    need(path.is_file(), "PIN_MISSING", role)
    raw = path.read_bytes()
    need(_sha(raw) == pin["sha256"], "PIN_SHA", role)
    if "bytes" in pin:
        need(len(raw) == pin["bytes"], "PIN_BYTES", role)
    return path, raw


def _load_json(raw, code):
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise SpanDriverError("%s: %s" % (code, exc)) from None


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(driver._jsonable(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def _model_name(family):
    return harness.MODEL_BINARY if family == "binary" else harness.MODEL_MULTICLASS


def _group_keys():
    return [(representation, family) for representation in REPRESENTATIONS for family in FAMILIES]


# --------------------------------------------------------------------------- #
# prospective plan
# --------------------------------------------------------------------------- #
def load_plan(path, expected_sha256):
    path = Path(path).resolve()
    need(path.is_file(), "PLAN_MISSING", str(path))
    raw = path.read_bytes()
    need(_sha(raw) == expected_sha256, "PLAN_DIGEST")
    plan = _load_json(raw, "PLAN_JSON")
    need(type(plan) is dict, "PLAN_SCHEMA")
    need(plan.get("schema") == PLAN_SCHEMA, "PLAN_SCHEMA")
    need(plan.get("job_id") == JOB_ID, "PLAN_JOB_ID")
    need(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", str(plan.get("run_id", ""))), "PLAN_RUN_ID")
    need(type(plan.get("seconds")) is int and 0 < plan["seconds"] <= DEADLINE_LIMIT, "PLAN_SECONDS")
    need(plan.get("cv_fit_limit") == CV_FIT_LIMIT, "PLAN_CV_LIMIT")
    need(plan.get("refit_limit") == REFIT_LIMIT, "PLAN_REFIT_LIMIT")
    need(plan.get("transform_fit_limit") == TRANSFORM_FIT_LIMIT, "PLAN_TRANSFORM_LIMIT")
    need(list(plan.get("thresholds", [])) == list(TAUS), "PLAN_TAUS")
    need(dict(plan.get("xgboost", {})) == EXPECTED_XGB_PARAMS, "PLAN_XGB_PARAMS")
    need(list(plan.get("representations", [])) == list(REPRESENTATIONS), "PLAN_REPRESENTATIONS")
    need(plan.get("holdout_access") is False, "PLAN_HOLDOUT_ACCESS")
    source = plan.get("source")
    need(type(source) is dict and set(source) == set(SOURCE_ROLES), "PLAN_SOURCE_ROLES")
    manifests = plan.get("manifests")
    need(type(manifests) is dict and set(manifests) == set(MANIFEST_ROLES) | {BLUEPRINT_ROLE}, "PLAN_MANIFEST_ROLES")
    return plan


# --------------------------------------------------------------------------- #
# parent receipt + lock + binary/index authentication
# --------------------------------------------------------------------------- #
def load_source(plan, source_sha256, root):
    """Authenticate the completed parent receipt, lock and both artifact pins."""
    source = plan["source"]
    need(type(source_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", source_sha256), "SOURCE_SHA")
    need(source["lock"]["sha256"] == source_sha256, "SOURCE_SHA_ARGUMENT")
    receipt_path, receipt_raw = _pinned(root, source["receipt"], "receipt")
    receipt = _load_json(receipt_raw, "RECEIPT_JSON")
    need(type(receipt) is dict, "RECEIPT_SCHEMA")
    need(receipt.get("status") in RECEIPT_STATUSES, "RECEIPT_STATUS")
    need(receipt.get("lock_sha256") == source_sha256, "RECEIPT_LOCK_SHA")
    outputs = receipt.get("outputs")
    need(type(outputs) is dict, "RECEIPT_OUTPUTS")
    for name, role in (("windows.f32", "windows"), ("index.json", "index")):
        need(name in outputs, "RECEIPT_ARTIFACT", name)
        pin = outputs[name]
        need(type(pin) is dict and {"bytes", "sha256"} <= set(pin), "RECEIPT_ARTIFACT_PIN", name)
        need(pin["bytes"] == source[role]["bytes"], "RECEIPT_ARTIFACT_BYTES", name)
        need(pin["sha256"] == source[role]["sha256"], "RECEIPT_ARTIFACT_SHA", name)
    lock_path, lock_raw = _pinned(root, source["lock"], "lock")
    lock = _load_json(lock_raw, "LOCK_JSON")
    need(type(lock) is dict and lock.get("schema") == LOCK_SCHEMA, "LOCK_SCHEMA")
    need(lock.get("scientific_execution_authorized") is True, "LOCK_NOT_AUTHORIZED")
    need(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", str(lock.get("run_id", ""))), "LOCK_RUN_ID")
    need(lock["run_id"] != plan["run_id"], "RUN_ID_REUSE")
    need(receipt.get("run_id") == lock["run_id"], "RECEIPT_RUN_ID")
    caps = lock.get("caps")
    need(type(caps) is dict, "LOCK_CAPS")
    need(caps.get("fits") == 0 and caps.get("derivatives") == 0, "LOCK_CAPS_FITS")
    index_path, index_raw = _pinned(root, source["index"], "index")
    windows_path, windows_raw = _pinned(root, source["windows"], "windows")
    index = _load_json(index_raw, "INDEX_JSON")
    need(type(index) is dict and index.get("schema") == INDEX_SCHEMA, "INDEX_SCHEMA")
    need(index.get("lock_sha256") == source_sha256, "INDEX_LOCK_SHA")
    need(index.get("run_id") == lock["run_id"], "INDEX_RUN_ID")
    return {
        "receipt": receipt,
        "receipt_sha256": source["receipt"]["sha256"],
        "receipt_path": str(receipt_path),
        "lock": lock,
        "lock_sha256": source_sha256,
        "lock_path": str(lock_path),
        "index": index,
        "index_sha256": source["index"]["sha256"],
        "index_path": str(index_path),
        "windows_sha256": source["windows"]["sha256"],
        "windows_bytes": source["windows"]["bytes"],
        "windows_path": str(windows_path),
        "windows_raw": windows_raw,
    }


# --------------------------------------------------------------------------- #
# case IDs/splits/groups/folds from separately pinned development manifests
# --------------------------------------------------------------------------- #
def load_manifests(manifest_pins, root):
    case_index, blueprint = {}, None
    for role, (split, cohort) in MANIFEST_ROLES.items():
        need(role in manifest_pins, "MANIFEST_PIN", role)
        _, raw = _pinned(root, manifest_pins[role], role)
        document = _load_json(raw, "MANIFEST_JSON")
        need(type(document) is dict and document.get("split") == split, "MANIFEST_SPLIT", role)
        cases = document.get("cases")
        need(type(cases) is list and document.get("case_count") == len(cases), "MANIFEST_CASE_COUNT", role)
        for case in cases:
            need(type(case) is dict, "MANIFEST_CASE", role)
            case_id = case.get("case_id")
            need(type(case_id) is str and case_id and case_id not in case_index, "MANIFEST_CASE_ID", role)
            need(case.get("split") == split, "MANIFEST_CASE_SPLIT", case_id)
            need(type(case.get("group_id")) is str and case["group_id"], "MANIFEST_GROUP", case_id)
            label = case.get("class_label")
            need(label in CLASS_ORDER, "MANIFEST_LABEL", case_id)
            fold = case.get("development_fold")
            need(fold is None or (type(fold) is int and fold in FOLDS), "MANIFEST_FOLD", case_id)
            case_index[case_id] = {
                "split": split,
                "cohort": cohort,
                "group_id": case["group_id"],
                "class_label": label,
                "fold": fold,
            }
    _, raw = _pinned(root, manifest_pins[BLUEPRINT_ROLE], BLUEPRINT_ROLE)
    document = _load_json(raw, "BLUEPRINT_JSON")
    need(type(document) is dict and type(document.get("groups")) is list, "BLUEPRINT_SCHEMA")
    blueprint = {group["group_id"]: group for group in document["groups"]}
    for case_id, info in case_index.items():
        group = blueprint.get(info["group_id"])
        need(type(group) is dict, "BLUEPRINT_GROUP", case_id)
        need(group.get("split") == info["split"], "BLUEPRINT_SPLIT", case_id)
        group_fold = group.get("development_fold")
        need(group_fold is None or (type(group_fold) is int and group_fold in FOLDS), "BLUEPRINT_FOLD", case_id)
        if info["fold"] is None:
            info["fold"] = group_fold
        elif group_fold is not None:
            need(info["fold"] == group_fold, "BLUEPRINT_FOLD_MISMATCH", case_id)
    train = [info for info in case_index.values() if info["split"] == "TRAIN"]
    validation = [info for info in case_index.values() if info["split"] == "VALIDATION"]
    original = [info for info in validation if info["cohort"] == "original"]
    added = [info for info in validation if info["cohort"] == "added"]
    need(len(case_index) == TOTAL_CASES, "MANIFEST_TOTAL_CASES", str(len(case_index)))
    need(len(train) == TRAIN_CASES, "MANIFEST_TRAIN_CASES", str(len(train)))
    need(len(validation) == VALIDATION_CASES, "MANIFEST_VALIDATION_CASES", str(len(validation)))
    need(len(original) == VALIDATION_ORIGINAL, "MANIFEST_ORIGINAL", str(len(original)))
    need(len(added) == VALIDATION_ADDED, "MANIFEST_ADDED", str(len(added)))
    labels = {label: 0 for label in CLASS_ORDER}
    for info in train:
        labels[info["class_label"]] += 1
    need(all(count > 0 for count in labels.values()), "MANIFEST_TRAIN_CLASSES", str(labels))
    return {"cases": case_index, "blueprint": blueprint}


# --------------------------------------------------------------------------- #
# binary/index record authentication + window reconstruction
# --------------------------------------------------------------------------- #
def _record_checks(record, index, windows_raw, manifest):
    need(type(record) is dict, "RECORD_SCHEMA")
    required = {
        "case_id", "order", "block", "shape", "dtype", "byte_order", "float_bytes",
        "token_positions", "prefix_length", "readout_index", "offset", "length", "sha256",
    }
    need(required <= set(record), "RECORD_FIELDS")
    case_id, order, block = record["case_id"], record["order"], record["block"]
    need(case_id in manifest["cases"], "RECORD_CASE", str(case_id))
    need(order in ORDERS, "RECORD_ORDER", str(order))
    need(block in BLOCKS, "RECORD_BLOCK", str(block))
    info = manifest["cases"][case_id]
    need(record.get("split") == info["split"] or "split" not in record, "RECORD_SPLIT", str(case_id))
    shape = record["shape"]
    need(type(shape) is list and len(shape) == 2, "RECORD_SHAPE", str(case_id))
    length_axis, width_axis = shape
    need(type(length_axis) is int and 1 <= length_axis <= WINDOW_MAX, "RECORD_WINDOW_LENGTH", str(shape))
    need(width_axis == WIDTH, "RECORD_WIDTH", str(shape))
    need(record["dtype"] == "float32", "RECORD_DTYPE")
    need(record["byte_order"] == "little", "RECORD_BYTE_ORDER")
    need(record["float_bytes"] == FLOAT_BYTES, "RECORD_FLOAT_BYTES")
    positions = record["token_positions"]
    need(type(positions) is list and all(type(value) is int for value in positions), "RECORD_POSITIONS")
    need(len(positions) == length_axis, "RECORD_POSITION_COUNT")
    need(all(positions[i] < positions[i + 1] for i in range(len(positions) - 1)), "RECORD_POSITION_ORDER")
    prefix_length = record["prefix_length"]
    readout_index = record["readout_index"]
    need(type(prefix_length) is int and type(readout_index) is int and readout_index >= 0, "RECORD_INDEX")
    need(prefix_length == readout_index + 1, "RECORD_READOUT")
    need(length_axis == min(WINDOW_MAX, prefix_length), "RECORD_WINDOW_FIT")
    expected = list(range(prefix_length - length_axis, prefix_length))
    need(positions == expected, "RECORD_POSITIONS_LAST")
    need(len(set(positions)) == length_axis, "RECORD_POSITIONS_DUP")
    offset, length = record["offset"], record["length"]
    need(type(offset) is int and type(length) is int, "RECORD_OFFSET_TYPE")
    need(length == length_axis * WIDTH * FLOAT_BYTES, "RECORD_LENGTH")
    need(0 <= offset and offset + length <= len(windows_raw), "RECORD_OFFSET_BOUNDS")
    need(_sha(windows_raw[offset:offset + length]) == record["sha256"], "RECORD_SHA")
    return case_id, order, block, length_axis, offset, length


def decode_index(index, windows_raw, manifest):
    """Authenticate the 1920-record index and reconstruct pair-averaged cases."""
    need(index.get("blocks") == list(BLOCKS), "INDEX_BLOCKS")
    need(index.get("orders") == list(ORDERS), "INDEX_ORDERS")
    need(index.get("dtype") == "float32", "INDEX_DTYPE")
    need(index.get("byte_order") == "little", "INDEX_BYTE_ORDER")
    need(index.get("float_bytes") == FLOAT_BYTES, "INDEX_FLOAT_BYTES")
    need(index.get("width") == WIDTH, "INDEX_WIDTH")
    need(index.get("window_max") == WINDOW_MAX, "INDEX_WINDOW_MAX")
    need(index.get("case_count") == TOTAL_CASES, "INDEX_CASE_COUNT")
    need(index.get("view_count") == TOTAL_VIEWS, "INDEX_VIEW_COUNT")
    need(index.get("forward_count") == TOTAL_VIEWS, "INDEX_FORWARD_COUNT")
    need(index.get("raw_bytes") == len(windows_raw), "INDEX_RAW_BYTES")
    case_order = index.get("case_order")
    need(type(case_order) is list and len(case_order) == TOTAL_CASES, "INDEX_CASE_ORDER")
    need(len(set(case_order)) == TOTAL_CASES, "INDEX_CASE_DUP")
    need(all(case_id in manifest["cases"] for case_id in case_order), "INDEX_CASE_UNKNOWN")
    records = index.get("records")
    need(type(records) is list and len(records) == TOTAL_RECORDS, "INDEX_RECORDS")

    grouped, intervals = {}, []
    for record in records:
        case_id, order, block, length_axis, offset, length = _record_checks(record, index, windows_raw, manifest)
        key = (case_id, order, block)
        need(key not in grouped, "RECORD_DUPLICATE", str(key))
        grouped[key] = record
        intervals.append((offset, length, key))
    need(len(grouped) == TOTAL_RECORDS, "RECORD_ACCOUNTING")
    ordered = sorted(intervals, key=lambda item: item[0])
    end = 0
    for offset, length, key in ordered:
        need(offset >= end, "RECORD_OVERLAP", str(key))
        end = max(end, offset + length)
    need(end <= len(windows_raw), "RECORD_OFFSET_BOUNDS")
    need(set(record["case_id"] for record in records) == set(case_order), "INDEX_CASE_RECORDS")
    for case_id in case_order:
        for order in ORDERS:
            for block in BLOCKS:
                need((case_id, order, block) in grouped, "RECORD_MISSING", "%s|%s|%s" % (case_id, order, block))

    windows, ab_windows, ba_windows = {}, {}, {}
    for case_id in case_order:
        views = {}
        for order in ORDERS:
            blocks, positions_reference, prefix_reference = [], None, None
            for block in BLOCKS:
                record = grouped[(case_id, order, block)]
                length_axis = record["shape"][0]
                array = np.frombuffer(
                    windows_raw[record["offset"]:record["offset"] + record["length"]], dtype="<f4"
                ).reshape(length_axis, WIDTH)
                need(array.shape == (length_axis, WIDTH), "RECORD_ARRAY_SHAPE")
                need(bool(np.isfinite(array).all()), "RECORD_FINITE")
                mask = [True] * length_axis
                need(len(mask) == len(record["token_positions"]), "RECORD_MASK")
                if positions_reference is None:
                    positions_reference, prefix_reference = list(record["token_positions"]), record["prefix_length"]
                else:
                    need(list(record["token_positions"]) == positions_reference, "CASE_POSITIONS")
                    need(record["prefix_length"] == prefix_reference, "CASE_PREFIX")
                blocks.append(array)
            views[order] = np.stack(blocks, axis=0)
        need(views["AB"].shape == views["BA"].shape, "PAIR_SHAPE", str(case_id))
        need(views["AB"].dtype == views["BA"].dtype, "PAIR_DTYPE", str(case_id))
        average = 0.5 * (views["AB"].astype(np.float64) + views["BA"].astype(np.float64))
        windows[case_id] = np.ascontiguousarray(average, dtype=np.float32)
        if manifest["cases"][case_id]["split"] == "VALIDATION":
            ab_windows[case_id] = views["AB"]
            ba_windows[case_id] = views["BA"]
    return {"windows": windows, "ab_windows": ab_windows, "ba_windows": ba_windows, "records": len(records)}


def load_case_data(source, plan, root):
    manifest = load_manifests(plan["manifests"], root)
    decoded = decode_index(source["index"], source["windows_raw"], manifest)
    case_order = list(source["index"]["case_order"])
    train_ids = [case_id for case_id in case_order if manifest["cases"][case_id]["split"] == "TRAIN"]
    validation_ids = [case_id for case_id in case_order if manifest["cases"][case_id]["split"] == "VALIDATION"]
    original_ids = [case_id for case_id in validation_ids if manifest["cases"][case_id]["cohort"] == "original"]
    added_ids = [case_id for case_id in validation_ids if manifest["cases"][case_id]["cohort"] == "added"]
    need(len(train_ids) == TRAIN_CASES and len(validation_ids) == VALIDATION_CASES, "CASE_SPLIT_COUNT")
    groups = {case_id: manifest["cases"][case_id]["group_id"] for case_id in case_order}
    fold_of_group = {}
    for case_id in train_ids:
        info = manifest["cases"][case_id]
        group_id, fold = info["group_id"], info["fold"]
        need(type(fold) is int and fold in FOLDS, "TRAIN_FOLD", case_id)
        if group_id in fold_of_group:
            need(fold_of_group[group_id] == fold, "GROUP_MULTI_FOLD", group_id)
        fold_of_group[group_id] = fold
    need(set(fold_of_group.values()) == set(FOLDS), "TRAIN_FOLD_SET", str(sorted(set(fold_of_group.values()))))
    folds = {case_id: (manifest["cases"][case_id]["fold"] if manifest["cases"][case_id]["split"] == "TRAIN" else None)
             for case_id in case_order}
    labels = {case_id: manifest["cases"][case_id]["class_label"] for case_id in case_order}
    for held in FOLDS:
        training = [case_id for case_id in train_ids if folds[case_id] != held]
        present = {labels[case_id] for case_id in training}
        need(present == set(CLASS_ORDER), "FOLD_CLASSES", str(held))
    return {
        "manifest": manifest,
        "case_order": case_order,
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "original_ids": original_ids,
        "added_ids": added_ids,
        "windows": decoded["windows"],
        "ab_windows": decoded["ab_windows"],
        "ba_windows": decoded["ba_windows"],
        "labels": labels,
        "groups": groups,
        "folds": folds,
        "record_count": decoded["records"],
    }


# --------------------------------------------------------------------------- #
# estimator factories (real XGBoost imported lazily; tests inject a toy)
# --------------------------------------------------------------------------- #
class XGBEstimator:
    """Thin fixed-parameter XGBoost wrapper; matches grouped_driver expectations."""

    def __init__(self, family, params):
        import xgboost as xgb  # lazily imported: module import stays model-free

        self.family = family
        self.classes_ = np.asarray([0, 1]) if family == "binary" else np.asarray(CLASS_ORDER)
        objective = "binary:logistic" if family == "binary" else "multi:softprob"
        self.native = xgb.XGBClassifier(objective=objective, **dict(params))

    def fit(self, x, labels):
        target = (
            (np.asarray(labels) == "SELF").astype(int)
            if self.family == "binary"
            else np.asarray([CLASS_ORDER.index(value) for value in labels])
        )
        self.native.fit(np.asarray(x, float), target)
        return self

    def predict_proba(self, x):
        return np.asarray(self.native.predict_proba(np.asarray(x, float)), float)


class XGBFactory:
    """Default factory: ``factory(family) -> XGBEstimator`` with frozen params."""

    def __init__(self, params):
        self.params = dict(params)

    def __call__(self, family):
        return XGBEstimator(family, self.params)


# --------------------------------------------------------------------------- #
# deadline
# --------------------------------------------------------------------------- #
class _Deadline:
    def __init__(self, started, seconds, clock):
        self.started = started
        self.seconds = seconds
        self.clock = clock

    def check(self):
        need(self.clock() - self.started <= self.seconds, "DEADLINE")


# --------------------------------------------------------------------------- #
# grouped-OOF training and selection
# --------------------------------------------------------------------------- #
def _selection_key(representation, family, tau, metrics):
    return (
        -min(metrics["precision"], metrics["recall"]),
        -metrics["f1"],
        REPRESENTATION_WIDTHS[representation],
        abs(tau - 0.5),
        tau,
        0 if family == "binary" else 1,
        REPRESENTATIONS.index(representation),
    )


def _fold_cv(case_data, factory, counters, deadline):
    train_ids = case_data["train_ids"]
    labels = [case_data["labels"][case_id] for case_id in train_ids]
    fold_array = np.asarray([case_data["folds"][case_id] for case_id in train_ids])
    group_array = np.asarray([case_data["groups"][case_id] for case_id in train_ids])
    windows = [case_data["windows"][case_id] for case_id in train_ids]
    oof_p = {(representation, family): np.full(len(train_ids), np.nan) for representation, family in _group_keys()}
    oof_probs = {
        (representation, family): np.full((len(train_ids), len(CLASS_ORDER)), np.nan)
        for representation, family in _group_keys()
        if family == "fourclass"
    }
    fold_records = []
    for held in FOLDS:
        train_index = np.flatnonzero(fold_array != held)
        test_index = np.flatnonzero(fold_array == held)
        need(not (set(group_array[train_index]) & set(group_array[test_index])), "FOLD_GROUP_LEAK", str(held))
        deadline.check()
        shared = features.SpanFeatureTransforms().fit(
            [windows[index] for index in train_index], [labels[index] for index in train_index]
        )
        counters["transform_fits"] += 1
        matrix = shared.transform(windows)
        need(matrix.shape == (len(train_ids), FEATURE_DIM), "FEATURE_MATRIX_SHAPE", str(matrix.shape))
        record = {
            "fold": int(held),
            "n_train": int(train_index.size),
            "n_test": int(test_index.size),
            "train_groups": sorted(set(group_array[train_index].tolist())),
            "test_groups": sorted(set(group_array[test_index].tolist())),
            "errors": {},
        }
        for representation, family in _group_keys():
            window = REPRESENTATION_SLICES[representation]
            x_train = np.ascontiguousarray(matrix[train_index, window])
            x_test = np.ascontiguousarray(matrix[test_index, window])
            deadline.check()
            try:
                estimator = factory(family)
                estimator.fit(x_train, [labels[index] for index in train_index])
                p_self, probabilities = driver._fold_probabilities(_model_name(family), estimator, x_test, int(test_index.size))
                oof_p[(representation, family)][test_index] = p_self
                if family == "fourclass":
                    oof_probs[(representation, family)][test_index] = probabilities
            except Exception as exc:  # a failed fold invalidates the whole candidate
                counters["fit_errors"] += 1
                record["errors"]["%s|%s" % (representation, family)] = _short(exc)
            counters["cv_fits"] += 1
        fold_records.append(record)
    return oof_p, oof_probs, fold_records


def _rank_candidates(oof_p, oof_probs, labels):
    truth = np.asarray([1 if value == "SELF" else 0 for value in labels], dtype=int)
    candidates, ranking = [], []
    for representation, family in _group_keys():
        p_self = oof_p[(representation, family)]
        entry = {
            "representation": representation,
            "family": family,
            "feature_count": REPRESENTATION_WIDTHS[representation],
            "valid": False,
            "error": None,
            "selected_tau": None,
            "thresholds": [],
            "oof_metrics": None,
        }
        if not bool(np.isfinite(p_self).all()):
            entry["error"] = "INCOMPLETE_OOF"
            candidates.append(entry)
            continue
        if family == "fourclass" and not bool(np.isfinite(oof_probs[(representation, family)]).all()):
            entry["error"] = "INCOMPLETE_OOF"
            candidates.append(entry)
            continue
        best = None
        for tau in TAUS:
            metrics = harness.binary_gate_metrics(truth, p_self, tau)
            eligible = harness.is_eligible(metrics)
            entry["thresholds"].append(
                {
                    "tau": float(tau),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "eligible": eligible,
                }
            )
            if not eligible:
                continue
            key = _selection_key(representation, family, tau, metrics)
            if best is None or key < best[0]:
                best = (key, float(tau), metrics)
        if best is None:
            entry["error"] = "NO_ELIGIBLE_THRESHOLD"
            candidates.append(entry)
            continue
        entry.update(
            valid=True,
            selected_tau=best[1],
            oof_metrics=best[2],
            selection_key=[float(value) if isinstance(value, (int, float)) else value for value in best[0]],
            oof_p_self=p_self.tolist(),
        )
        candidates.append(entry)
        ranking.append((best[0], representation, family, best[1], best[2]))
    ranking.sort(key=lambda item: item[0])
    return candidates, ranking, truth


def _evaluate_split(family, estimator, matrix, case_ids, truth, tau, ab_matrix=None, ba_matrix=None):
    p_self, probabilities = driver._fold_probabilities(_model_name(family), estimator, matrix, len(case_ids))
    predicted = p_self >= tau
    result = {
        "self_gate": harness.binary_metrics(truth == "SELF", predicted),
        "negative_class_false_positives": {
            label: {
                "false_positives": int(predicted[truth == label].sum()),
                "total": int((truth == label).sum()),
            }
            for label in CLASS_ORDER[1:]
        },
        "predictions": [
            {"case_id": case_id, "truth": str(value), "p_self": float(score), "predicted_self": bool(flag)}
            for case_id, value, score, flag in zip(case_ids, truth, p_self, predicted)
        ],
    }
    if family == "fourclass":
        result["four_class"] = harness.multiclass_metrics(truth, harness.argmax_labels(probabilities, CLASS_ORDER), CLASS_ORDER)
    if ab_matrix is not None and ba_matrix is not None:
        p_ab, probabilities_ab = driver._fold_probabilities(_model_name(family), estimator, ab_matrix, len(case_ids))
        p_ba, probabilities_ba = driver._fold_probabilities(_model_name(family), estimator, ba_matrix, len(case_ids))
        order = {
            "gate_agreements": int(np.sum((p_ab >= tau) == (p_ba >= tau))),
            "total": int(len(case_ids)),
            "max_probability_difference": float(np.max(np.abs(p_ab - p_ba))),
        }
        if family == "fourclass":
            order["four_class_agreements"] = int(
                np.sum(
                    np.asarray(harness.argmax_labels(probabilities_ab, CLASS_ORDER))
                    == np.asarray(harness.argmax_labels(probabilities_ba, CLASS_ORDER))
                )
            )
        result["answer_order_consistency"] = order
    return result


def _fit_and_select(case_data, plan, factory, counters, deadline, output):
    oof_p, oof_probs, fold_records = _fold_cv(case_data, factory, counters, deadline)
    labels = [case_data["labels"][case_id] for case_id in case_data["train_ids"]]
    candidates, _, _ = _rank_candidates(oof_p, oof_probs, labels)

    train_windows = [case_data["windows"][case_id] for case_id in case_data["train_ids"]]
    validation_ids = case_data["validation_ids"]
    original_ids, added_ids = case_data["original_ids"], case_data["added_ids"]
    validation_windows = [case_data["windows"][case_id] for case_id in validation_ids]
    original_windows = [case_data["windows"][case_id] for case_id in original_ids]
    added_windows = [case_data["windows"][case_id] for case_id in added_ids]
    ab_windows = [case_data["ab_windows"][case_id] for case_id in validation_ids]
    ba_windows = [case_data["ba_windows"][case_id] for case_id in validation_ids]

    deadline.check()
    shared_full = features.SpanFeatureTransforms().fit(train_windows, labels)
    counters["transform_fits"] += 1
    train_matrix = shared_full.transform(train_windows)
    validation_matrix = shared_full.transform(validation_windows)
    original_matrix = shared_full.transform(original_windows)
    added_matrix = shared_full.transform(added_windows)
    ab_matrix = shared_full.transform(ab_windows)
    ba_matrix = shared_full.transform(ba_windows)

    (output / "models").mkdir()
    results, train_ranking = [], []
    for entry in candidates:
        representation, family = entry["representation"], entry["family"]
        if not entry["valid"]:
            entry["status"] = "INVALID_FOLD"
            entry["evaluation"] = None
            entry["reload_exact"] = False
            results.append(entry)
            continue
        deadline.check()
        window = REPRESENTATION_SLICES[representation]
        estimator = factory(family)
        estimator.fit(np.ascontiguousarray(train_matrix[:, window]), labels)
        counters["refits"] += 1
        bundle = pickle.dumps(
            {
                "estimator": estimator,
                "transform": shared_full,
                "meta": dict(representation=representation, family=family, tau=entry["selected_tau"], job_id=JOB_ID),
            },
            protocol=5,
        )
        name = "%s__%s.pkl" % (representation, family)
        with (output / "models" / name).open("xb") as stream:
            stream.write(bundle)
        restored = pickle.loads(bundle)
        reference = estimator.predict_proba(np.ascontiguousarray(validation_matrix[:, window]))
        reloaded = restored["estimator"].predict_proba(
            np.ascontiguousarray(restored["transform"].transform(validation_windows)[:, window])
        )
        reload_exact = bool(reference.shape == reloaded.shape and np.array_equal(reference, reloaded))
        need(reload_exact, "RELOAD_MISMATCH", name)
        truth_validation = np.asarray([case_data["labels"][case_id] for case_id in validation_ids])
        evaluation = {
            "training": _evaluate_split(
                family, estimator, np.ascontiguousarray(train_matrix[:, window]),
                case_data["train_ids"],
                np.asarray([case_data["labels"][case_id] for case_id in case_data["train_ids"]]),
                entry["selected_tau"],
            ),
            "validation_original": _evaluate_split(
                family, estimator, np.ascontiguousarray(original_matrix[:, window]),
                original_ids,
                np.asarray([case_data["labels"][case_id] for case_id in original_ids]),
                entry["selected_tau"],
            ),
            "validation_added": _evaluate_split(
                family, estimator, np.ascontiguousarray(added_matrix[:, window]),
                added_ids,
                np.asarray([case_data["labels"][case_id] for case_id in added_ids]),
                entry["selected_tau"],
            ),
            "validation_combined": _evaluate_split(
                family, estimator, np.ascontiguousarray(validation_matrix[:, window]),
                validation_ids, truth_validation, entry["selected_tau"],
                ab_matrix=np.ascontiguousarray(ab_matrix[:, window]),
                ba_matrix=np.ascontiguousarray(ba_matrix[:, window]),
            ),
        }
        entry.update(
            status="FITTED",
            serialized="models/" + name,
            reload_exact=True,
            evaluation=evaluation,
        )
        results.append(entry)
        train_ranking.append((entry["selection_key"], representation, family, entry["selected_tau"], entry["oof_metrics"]))

    train_ranking.sort(key=lambda item: item[0])
    selected = None
    if train_ranking:
        _, representation, family, tau, metrics = train_ranking[0]
        selected = {
            "representation": representation,
            "family": family,
            "tau": tau,
            "feature_count": REPRESENTATION_WIDTHS[representation],
            "oof_metrics": metrics,
            "selected_using": "TRAIN_GROUPED_OOF_ONLY",
        }
    validation_ranking = []
    for entry in results:
        if entry.get("status") != "FITTED":
            continue
        metrics = entry["evaluation"]["validation_combined"]["self_gate"]
        if not harness.is_eligible(metrics):
            continue
        key = _selection_key(entry["representation"], entry["family"], entry["selected_tau"], metrics)
        validation_ranking.append((key, entry))
    validation_ranking.sort(key=lambda item: item[0])
    validation_winner = None
    if validation_ranking:
        entry = validation_ranking[0][1]
        validation_winner = {
            "representation": entry["representation"],
            "family": entry["family"],
            "tau": entry["selected_tau"],
            "feature_count": entry["feature_count"],
            "validation_combined_metrics": entry["evaluation"]["validation_combined"]["self_gate"],
            "selected_using": "EXPLORATORY_VALIDATION_ONLY_NOT_FOR_SELECTION",
        }
    return {
        "candidates": results,
        "fold_records": fold_records,
        "train_selected": selected,
        "validation_winner_exploratory": validation_winner,
        "train_ranking": [
            {"representation": representation, "family": family, "tau": tau, "key": [float(value) for value in key], "metrics": metrics}
            for key, representation, family, tau, metrics in train_ranking
        ],
    }


def _source_summary(source):
    return {
        "receipt": source["receipt_path"],
        "receipt_sha256": source["receipt_sha256"],
        "receipt_status": source["receipt"].get("status"),
        "lock": source["lock_path"],
        "lock_sha256": source["lock_sha256"],
        "index": source["index_path"],
        "index_sha256": source["index_sha256"],
        "windows": source["windows_path"],
        "windows_bytes": source["windows_bytes"],
        "windows_sha256": source["windows_sha256"],
        "records": len(source["index"]["records"]),
        "cases": source["index"]["case_count"],
        "views": source["index"]["view_count"],
        "holdout_accessed": False,
    }


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def preflight(plan_path, plan_sha256, source_sha256, root=ROOT):
    """Authenticate plan, source and manifests; decode windows; fit nothing."""
    plan = load_plan(plan_path, plan_sha256)
    source = load_source(plan, source_sha256, root)
    case_data = load_case_data(source, plan, root)
    return {
        "status": "preflight_pass",
        "job_id": JOB_ID,
        "run_id": plan["run_id"],
        "source_sha256": source_sha256,
        "records": case_data["record_count"],
        "cases": len(case_data["case_order"]),
        "train": len(case_data["train_ids"]),
        "validation": len(case_data["validation_ids"]),
        "validation_original": len(case_data["original_ids"]),
        "validation_added": len(case_data["added_ids"]),
        "fits": 0,
        "holdout_accessed": False,
    }


def run(plan_path, plan_sha256, source_sha256, *, root=ROOT, factory=None, clock=time.monotonic):
    """Fit the five representations and write exclusive versioned run outputs."""
    started = clock()
    plan = load_plan(plan_path, plan_sha256)
    output = (Path(root).resolve() / "development" / "classifier_generalization_v2" / "runs" / plan["run_id"]).resolve()
    need(not output.exists(), "OUTPUT_EXISTS", str(output))
    output.mkdir(parents=True)
    _write_json(
        output / "fit_plan.json",
        dict(plan, plan_sha256=plan_sha256, source_sha256=source_sha256, job_id=JOB_ID),
    )
    counters = {"cv_fits": 0, "refits": 0, "transform_fits": 0, "fit_errors": 0, "candidates": 0}
    try:
        source = load_source(plan, source_sha256, root)
        case_data = load_case_data(source, plan, root)
        _write_json(output / "source_receipt.json", _source_summary(source))
        factory = factory if factory is not None else XGBFactory(plan["xgboost"])
        deadline = _Deadline(started, plan["seconds"], clock)
        outcome = _fit_and_select(case_data, plan, factory, counters, deadline, output)
        deadline.check()
        counters["candidates"] = len(outcome["candidates"])
        counters["valid_candidates"] = sum(1 for entry in outcome["candidates"] if entry.get("status") == "FITTED")
        elapsed = clock() - started
        result = {
            "status": "COMPLETE" if outcome["train_selected"] is not None else "NO_ELIGIBLE_TRAIN_CANDIDATE",
            "job_id": JOB_ID,
            "run_id": plan["run_id"],
            "plan_sha256": plan_sha256,
            "source_sha256": source_sha256,
            "counters": dict(counters, seconds=float(elapsed)),
            "limits": {"seconds": plan["seconds"], "cv_fits": CV_FIT_LIMIT, "refits": REFIT_LIMIT, "transform_fits": TRANSFORM_FIT_LIMIT},
            "data_counts": {
                "records": case_data["record_count"],
                "train": len(case_data["train_ids"]),
                "validation_original": len(case_data["original_ids"]),
                "validation_added": len(case_data["added_ids"]),
                "validation_combined": len(case_data["validation_ids"]),
            },
            "selection_policy": "TRAIN grouped OOF: max min(precision,recall), max F1, lower feature count, tau nearest 0.5, smaller tau, binary-first family tie",
            "train_selected": outcome["train_selected"],
            "validation_winner_exploratory": outcome["validation_winner_exploratory"],
            "candidates": outcome["candidates"],
            "train_ranking": outcome["train_ranking"],
            "holdout_accessed": False,
            "validation_used_for_selection": False,
            "features": {
                "combined_width": FEATURE_DIM,
                "representations": {name: REPRESENTATION_WIDTHS[name] for name in REPRESENTATIONS},
                "separate_models_per_representation": True,
            },
        }
        need(counters["cv_fits"] == CV_FIT_LIMIT, "CV_FIT_BUDGET", str(counters["cv_fits"]))
        need(counters["refits"] <= REFIT_LIMIT, "REFIT_BUDGET", str(counters["refits"]))
        _write_json(
            output / "cross_validation.json",
            {
                "folds": outcome["fold_records"],
                "candidates": [
                    {
                        "representation": entry["representation"],
                        "family": entry["family"],
                        "feature_count": entry["feature_count"],
                        "valid": entry["valid"],
                        "error": entry["error"],
                        "selected_tau": entry["selected_tau"],
                        "thresholds": entry["thresholds"],
                        "oof_p_self": entry.get("oof_p_self"),
                    }
                    for entry in outcome["candidates"]
                ],
                "train_selected": outcome["train_selected"],
                "selection_policy": result["selection_policy"],
            },
        )
        _write_json(output / "development_results.json", result)
        return result
    except BaseException as exc:
        _write_json(
            output / "failure.json",
            {
                "status": "failed",
                "job_id": JOB_ID,
                "run_id": plan["run_id"],
                "plan_sha256": plan_sha256,
                "source_sha256": source_sha256,
                "counters": dict(counters, seconds=float(clock() - started)),
                "code": str(exc)[:1024],
                "error_type": type(exc).__name__,
                "holdout_accessed": False,
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--run", action="store_true", help="execute the fit (writes exclusive run outputs)")
    args = parser.parse_args(argv)
    try:
        if args.run:
            result = run(args.plan, args.plan_sha256, args.source_sha256, root=args.root)
            result = {key: result[key] for key in ("status", "run_id", "counters", "data_counts", "train_selected")}
        else:
            result = preflight(args.plan, args.plan_sha256, args.source_sha256, root=args.root)
        print(json.dumps(driver._jsonable(result), allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "code": str(exc)[:1024], "error_type": type(exc).__name__}, allow_nan=False))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
