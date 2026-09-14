"""F1-first pure cached-prediction selector (job ``f1_cached_selection_20260914_v1``).

This module is the smallest *pure* successor of the frozen min(precision,recall)-first
ranking used by the completed linear-span and compression runs. It performs **no
scientific work**: no loader, no capture, no model/tokenizer/lens, no estimator, no
PCA, no fit, no pickle deserialization, no network/install/Git and no real
dataset/result/holdout read. ``select(pool)`` accepts already-authenticated TRAIN
grouped-OOF probabilities plus the saved full-TRAIN artifact *pins* supplied by the
root-owned wrapper and returns the prospective F1-first selection and a frozen
identity.

Selection rule (prospective, distinct from the frozen historical minPR-first rule)::

    max pooled TRAIN OOF F1, then max min(precision, recall), then lower width,
    lower C, binary family, |tau - 0.5|, smaller tau, unprompted

The candidate pool is exactly the 42 saved ``compression_comparison_v2`` candidates
(7 cells x 2 families x 3 C) plus the 6 reused unprompted full-3072
``linear_span_20260914_v1`` candidates (2 families x 3 C) = 48 candidates over 16
cells. Thresholds are the original finite 19 values 0.05..0.95. Validation rows are
never read or used during selection; the only post-freeze action is recomputing
already-saved predictions at the newly selected tau, and only when that cell's saved
full-TRAIN refit is at the *same* C. Otherwise the result fails closed with
``REQUIRES_SEPARATELY_LOCKED_REFIT`` and never borrows a different-C model.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

import harness

__all__ = [
    "F1SelectionError", "JOB_ID", "PLAN_SCHEMA", "RESULT_SCHEMA", "SPLIT_SCHEMA", "POLICY_ID",
    "POLICY", "SELECTION_RULE", "VALIDATION_POLICY", "READ_ORDER", "SOURCES",
    "SOURCE_COMPRESSION", "SOURCE_REFERENCE", "CONDITIONS", "CONDITION_DIMS", "DIM_WIDTH",
    "FAMILIES", "C_VALUES", "THRESHOLDS", "TAUS", "FOLDS", "CLASS_ORDER", "SPLITS",
    "SPLIT_ROWS", "POST_FREEZE_SPLITS", "OOF_ROWS", "VALIDATION_ROWS", "THRESHOLD_LIMIT",
    "CANDIDATE_LIMIT", "NEW_CANDIDATE_LIMIT", "REFERENCE_CANDIDATE_LIMIT", "CELL_LIMIT",
    "SECONDS", "OUTPUT_BYTES", "FIT_LIMIT", "MODEL_LOAD_LIMIT", "BOUNDS", "candidate_id",
    "cell_key", "expected_candidate_ids", "expected_cell_keys", "artifact_template",
    "policy_key", "select", "recompute_split_metrics",
]

JOB_ID = "f1_cached_selection_20260914_v1"
PLAN_SCHEMA = "f1_cached_selection_plan.v1"
RESULT_SCHEMA = "f1_cached_selection_result.v1"
SPLIT_SCHEMA = "f1_cached_selection_split_metrics.v1"
POLICY_ID = "max_train_oof_f1_then_min_pr_v1"
POLICY = ("TRAIN grouped OOF only: max pooled OOF F1, then max min(precision,recall), then lower "
          "width, lower C, binary family, |tau-0.5|, smaller tau, unprompted")
SELECTION_RULE = ("max pooled TRAIN OOF F1", "max min(precision, recall)", "lower width", "lower C",
                  "binary family first", "|tau - 0.5| smallest", "smaller tau", "unprompted first")
VALIDATION_POLICY = ("validation arrays are never passed to select() and never influence ranking, "
                     "tau choice or the frozen identity; validation predictions may be read only "
                     "after the selection is frozen and only at the frozen tau")
READ_ORDER = (
    "1. root pins every source file by sha256 and rejects non-finite/malformed input",
    "2. read TRAIN OOF only: oof_case_ids/oof_truth/oof_p_self for 42 new + 6 reference candidates",
    "3. read the saved full-TRAIN artifact C index for the 16 cells",
    "4. select on TRAIN OOF only and freeze selection + frozen_identity",
    "5. only after freeze: if the winning cell's artifact C equals the selected C, read the saved "
    "predictions and recompute at the selected tau",
    "6. otherwise stop and report REQUIRES_SEPARATELY_LOCKED_REFIT; never borrow a different-C model",
)

SOURCE_COMPRESSION = "compression_comparison_v2"
SOURCE_REFERENCE = "linear_span_v1"
SOURCES = (SOURCE_COMPRESSION, SOURCE_REFERENCE)
CONDITIONS = ("unprompted", "prompted")
CONDITION_DIMS = {"unprompted": ("pca8", "pca16", "pca32"),
                  "prompted": ("pca8", "pca16", "pca32", "full3072")}
DIM_WIDTH = {"pca8": 8, "pca16": 16, "pca32": 32, "full3072": 3072}
FAMILIES = ("binary", "fourclass")
C_VALUES = (0.1, 1.0, 10.0)
THRESHOLDS = tuple(round(0.05 * index, 2) for index in range(1, 20))  # original finite 19
TAUS = THRESHOLDS
FOLDS = (0, 1, 2, 3, 4)
CLASS_ORDER = ("SELF", "OTHER", "NONTERMINATION", "ORDINARY")

OOF_ROWS = 240
VALIDATION_ROWS = 80
SPLITS = ("original40", "added40", "combined80")
SPLIT_ROWS = {"original40": 40, "added40": 40, "combined80": 80}
POST_FREEZE_SPLITS = SPLITS
THRESHOLD_LIMIT = 19
CANDIDATE_LIMIT = 48
NEW_CANDIDATE_LIMIT = 42
REFERENCE_CANDIDATE_LIMIT = 6
CELL_LIMIT = 16
SECONDS = 60
OUTPUT_BYTES = 16 * 1024 * 1024
FIT_LIMIT = 0
MODEL_LOAD_LIMIT = 0

BOUNDS = {
    "candidates_total": CANDIDATE_LIMIT,
    "new_candidates": NEW_CANDIDATE_LIMIT,
    "reference_candidates": REFERENCE_CANDIDATE_LIMIT,
    "cells": CELL_LIMIT,
    "artifacts": CELL_LIMIT,
    "thresholds": THRESHOLD_LIMIT,
    "oof_rows": OOF_ROWS,
    "validation_rows": VALIDATION_ROWS,
    "estimator_fits": FIT_LIMIT,
    "model_loads": MODEL_LOAD_LIMIT,
    "pickle_deserializations": MODEL_LOAD_LIMIT,
    "seconds": SECONDS,
    "output_bytes": OUTPUT_BYTES,
}

POOL_FIELDS = ("source_pins", "train_ids", "truth", "candidates", "artifacts")
CANDIDATE_FIELDS = ("source", "condition", "dimension", "family", "C", "oof_case_ids", "oof_truth",
                    "oof_p_self")
ARTIFACT_FIELDS = ("C", "model", "model_sha256", "predictions", "predictions_sha256")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class F1SelectionError(ValueError):
    """Structured rejection raised by this module."""


def _require(condition, code, detail=None):
    if not condition:
        raise F1SelectionError(code if detail is None else "%s: %s" % (code, detail))


def _sha256_hex(value, code, detail=None):
    _require(type(value) is str and bool(_SHA256.fullmatch(value)), code, detail)
    return value


# --------------------------------------------------------------------------- #
# canonical ids: 42 new (7 cells x 2 families x 3 C) + 6 reference (2 x 3)
# --------------------------------------------------------------------------- #
def candidate_id(source, condition, dimension, family, C):
    """Canonical candidate id ``source|condition|dimension|family|C<repr(C)>``."""
    return "|".join((str(source), str(condition), str(dimension), str(family), "C%s" % repr(float(C))))


def cell_key(source, condition, dimension, family):
    """Canonical cell id shared by all three C candidates of one saved refit cell."""
    return "|".join((str(source), str(condition), str(dimension), str(family)))


def expected_candidate_ids():
    """The exact 48-candidate pool bound (order is deterministic)."""
    ids = []
    for condition in CONDITIONS:
        for dimension in CONDITION_DIMS[condition]:
            for family in FAMILIES:
                for C in C_VALUES:
                    ids.append(candidate_id(SOURCE_COMPRESSION, condition, dimension, family, C))
    for family in FAMILIES:
        for C in C_VALUES:
            ids.append(candidate_id(SOURCE_REFERENCE, "unprompted", "full3072", family, C))
    return tuple(ids)


def expected_cell_keys():
    """The exact 16 saved-refit cells (14 new + 2 reused reference)."""
    cells = []
    for condition in CONDITIONS:
        for dimension in CONDITION_DIMS[condition]:
            for family in FAMILIES:
                cells.append(cell_key(SOURCE_COMPRESSION, condition, dimension, family))
    for family in FAMILIES:
        cells.append(cell_key(SOURCE_REFERENCE, "unprompted", "full3072", family))
    return tuple(cells)


def _compression_artifact(condition, dimension, family):
    stem = "%s__%s__%s" % (condition, dimension, family)
    return {"model": "models/%s.pkl" % stem, "index": "model_results_%s.json" % stem,
            "predictions": "model_results_%s.json" % stem, "C_field": "C",
            "predictions_path": "evaluation.<split>.predictions[].p_self"}


def _reference_artifact(family):
    return {"model": "estimator_linear_%s.pkl" % family, "index": "family_%s.json" % family,
            "predictions": "predictions_linear_%s.json" % family, "C_field": "C",
            "predictions_path": "splits.<split>.predictions[].p_self"}


def artifact_template():
    """Cell -> relative file names and field paths; shas are pinned later by root."""
    template = {}
    for condition in CONDITIONS:
        for dimension in CONDITION_DIMS[condition]:
            for family in FAMILIES:
                template[cell_key(SOURCE_COMPRESSION, condition, dimension, family)] = \
                    _compression_artifact(condition, dimension, family)
    for family in FAMILIES:
        template[cell_key(SOURCE_REFERENCE, "unprompted", "full3072", family)] = _reference_artifact(family)
    return template


def _decode_id(cid):
    parts = cid.split("|")
    _require(len(parts) == 5, "CANDIDATE_ID_FORMAT", cid)
    source, condition, dimension, family, token = parts
    _require(source in SOURCES, "CANDIDATE_SOURCE", cid)
    _require(condition in CONDITIONS, "CANDIDATE_CONDITION", cid)
    _require(family in FAMILIES, "CANDIDATE_FAMILY", cid)
    if source == SOURCE_REFERENCE:
        _require((condition, dimension) == ("unprompted", "full3072"), "CANDIDATE_REFERENCE_CELL", cid)
    else:
        _require(dimension in CONDITION_DIMS[condition], "CANDIDATE_DIMENSION", cid)
    _require(token.startswith("C"), "CANDIDATE_C_TOKEN", cid)
    try:
        C = float(token[1:])
    except ValueError:
        raise F1SelectionError("CANDIDATE_C_TOKEN: %s" % cid) from None
    _require(C in C_VALUES, "CANDIDATE_C", cid)
    _require(candidate_id(source, condition, dimension, family, C) == cid, "CANDIDATE_ID_CANONICAL", cid)
    return source, condition, dimension, family, C


# --------------------------------------------------------------------------- #
# input validation (everything already authenticated by root)
# --------------------------------------------------------------------------- #
def _validate_candidate(cid, record, train_ids, truth):
    _require(type(record) is dict, "CANDIDATE_SCHEMA", cid)
    unknown = sorted(set(record) - set(CANDIDATE_FIELDS))
    _require(not unknown, "CANDIDATE_UNKNOWN_FIELD", "%s %s" % (cid, ",".join(unknown)))
    missing = [field for field in CANDIDATE_FIELDS if field not in record]
    _require(not missing, "CANDIDATE_FIELD_MISSING", "%s %s" % (cid, ",".join(missing)))
    source, condition, dimension, family, C = _decode_id(cid)
    for field, expected in (("source", source), ("condition", condition), ("dimension", dimension),
                            ("family", family)):
        _require(record[field] == expected, "CANDIDATE_METADATA_MISMATCH", "%s %s" % (cid, field))
    _require(type(record["C"]) in (int, float) and not isinstance(record["C"], bool)
             and float(record["C"]) == C, "CANDIDATE_C_MISMATCH", cid)
    case_ids = record["oof_case_ids"]
    _require(type(case_ids) is list and case_ids == train_ids, "CANDIDATE_CASE_ORDER", cid)
    cand_truth = record["oof_truth"]
    _require(type(cand_truth) is list and cand_truth == truth, "CANDIDATE_TRUTH_ORDER", cid)
    raw = record["oof_p_self"]
    _require(type(raw) in (list, tuple) and len(raw) == OOF_ROWS, "CANDIDATE_OOF_SHAPE", cid)
    try:
        p_self = np.asarray(raw, float)
    except (TypeError, ValueError):
        raise F1SelectionError("CANDIDATE_OOF_TYPE: %s" % cid) from None
    _require(p_self.shape == (OOF_ROWS,), "CANDIDATE_OOF_SHAPE", cid)
    _require(bool(np.isfinite(p_self).all()), "CANDIDATE_OOF_FINITE", cid)
    _require(float(p_self.min()) >= 0.0 and float(p_self.max()) <= 1.0, "CANDIDATE_OOF_RANGE", cid)
    return {"source": source, "condition": condition, "dimension": dimension, "family": family,
            "C": C, "p_self": p_self}


def _validate_artifact(cell, record):
    _require(type(record) is dict, "ARTIFACT_SCHEMA", cell)
    unknown = sorted(set(record) - set(ARTIFACT_FIELDS))
    _require(not unknown, "ARTIFACT_UNKNOWN_FIELD", "%s %s" % (cell, ",".join(unknown)))
    missing = [field for field in ARTIFACT_FIELDS if field not in record]
    _require(not missing, "ARTIFACT_FIELD_MISSING", "%s %s" % (cell, ",".join(missing)))
    _require(type(record["C"]) in (int, float) and not isinstance(record["C"], bool)
             and float(record["C"]) in C_VALUES, "ARTIFACT_C", cell)
    _require(type(record["model"]) is str and record["model"] != "", "ARTIFACT_MODEL", cell)
    _require(type(record["predictions"]) is str and record["predictions"] != "",
             "ARTIFACT_PREDICTIONS", cell)
    _sha256_hex(record["model_sha256"], "ARTIFACT_MODEL_SHA", cell)
    _sha256_hex(record["predictions_sha256"], "ARTIFACT_PREDICTIONS_SHA", cell)
    return {"C": float(record["C"]), "model": record["model"], "model_sha256": record["model_sha256"],
            "predictions": record["predictions"], "predictions_sha256": record["predictions_sha256"]}


def _validate_pool(pool):
    _require(type(pool) is dict, "POOL_SCHEMA")
    unknown = sorted(set(pool) - set(POOL_FIELDS))
    _require(not unknown, "POOL_UNKNOWN_FIELD", ",".join(unknown))
    missing = [field for field in POOL_FIELDS if field not in pool]
    _require(not missing, "POOL_FIELD_MISSING", ",".join(missing))
    pins = pool["source_pins"]
    _require(type(pins) is dict and pins, "SOURCE_PINS_SCHEMA")
    source_pins = {}
    for role in sorted(pins):
        _require(type(role) is str and role != "", "SOURCE_PIN_ROLE")
        source_pins[role] = _sha256_hex(pins[role], "SOURCE_PIN_SHA", role)
    train_ids = pool["train_ids"]
    _require(type(train_ids) is list and len(train_ids) == OOF_ROWS, "TRAIN_IDS_SHAPE",
             str(len(train_ids) if type(train_ids) is list else train_ids))
    _require(all(type(case_id) is str and case_id != "" for case_id in train_ids), "TRAIN_IDS_ENTRY")
    _require(len(set(train_ids)) == len(train_ids), "TRAIN_IDS_DUPLICATE")
    truth = pool["truth"]
    _require(type(truth) is list and len(truth) == OOF_ROWS, "TRUTH_SHAPE")
    _require(all(label in CLASS_ORDER for label in truth), "TRUTH_LABEL")
    _require("SELF" in set(truth) and bool(set(truth) - {"SELF"}), "TRUTH_COVERAGE")
    candidates = pool["candidates"]
    _require(type(candidates) is dict, "CANDIDATES_SCHEMA")
    _require(set(candidates) == set(expected_candidate_ids()), "CANDIDATE_SET",
             "expected %d" % CANDIDATE_LIMIT)
    parsed = {cid: _validate_candidate(cid, candidates[cid], train_ids, truth) for cid in candidates}
    artifacts = pool["artifacts"]
    _require(type(artifacts) is dict, "ARTIFACTS_SCHEMA")
    _require(set(artifacts) == set(expected_cell_keys()), "ARTIFACT_SET",
             "expected %d" % CELL_LIMIT)
    validated = {cell: _validate_artifact(cell, artifacts[cell]) for cell in artifacts}
    return {"source_pins": source_pins, "train_ids": list(train_ids), "truth": list(truth),
            "candidates": parsed, "artifacts": validated}


# --------------------------------------------------------------------------- #
# prospective F1-first policy
# --------------------------------------------------------------------------- #
def policy_key(family, C, width, condition, tau, metrics):
    """Ordered prospective key: max F1, max minPR, lower width, lower C, binary, tau, unprompted."""
    precision, recall, f1 = metrics["precision"], metrics["recall"], metrics["f1"]
    _require(all(value is not None for value in (precision, recall, f1)), "POLICY_METRIC_NONE")
    return (-float(f1), -min(float(precision), float(recall)), int(width), float(C),
            0 if family == "binary" else 1, abs(float(tau) - 0.5), float(tau),
            0 if condition == "unprompted" else 1)


def _oof_digest(candidates):
    hasher = hashlib.sha256()
    for cid in sorted(candidates):
        hasher.update(cid.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(np.ascontiguousarray(candidates[cid]["p_self"], dtype="<f8").tobytes())
        hasher.update(b"\x1e")
    return hasher.hexdigest()


def _entry(cid, record, key, tau, metrics):
    return {"candidate_id": cid, "source": record["source"], "condition": record["condition"],
            "dimension": record["dimension"], "family": record["family"], "C": record["C"],
            "width": DIM_WIDTH[record["dimension"]], "tau": float(tau),
            "oof_metrics": {"tp": int(metrics["tp"]), "tn": int(metrics["tn"]), "fp": int(metrics["fp"]),
                            "fn": int(metrics["fn"]), "precision": metrics["precision"],
                            "recall": metrics["recall"], "f1": metrics["f1"]},
            "selection_key": [float(value) for value in key]}


def _frozen_payload(data, ranking, match):
    winner = None
    if ranking:
        winner = {field: ranking[0][field] for field in ("candidate_id", "source", "condition", "dimension",
                                                         "family", "C", "width", "tau", "oof_metrics",
                                                         "selection_key")}
    return {
        "schema": RESULT_SCHEMA, "job_id": JOB_ID, "policy_id": POLICY_ID,
        "source_pins": {role: data["source_pins"][role] for role in sorted(data["source_pins"])},
        "train_ids": data["train_ids"], "truth": data["truth"],
        "oof_sha256": _oof_digest(data["candidates"]),
        "artifacts": {cell: {"C": data["artifacts"][cell]["C"],
                             "model_sha256": data["artifacts"][cell]["model_sha256"],
                             "predictions_sha256": data["artifacts"][cell]["predictions_sha256"]}
                      for cell in sorted(data["artifacts"])},
        "ranking_ids": [entry["candidate_id"] for entry in ranking],
        "winner": winner, "artifact_match": match,
    }


def select(pool):
    """Select the F1-first winner from authenticated TRAIN OOF and freeze its identity.

    No validation array is accepted or read. The result carries the full ranking, the
    frozen identity and a fail-closed ``post_freeze`` instruction for the saved
    predictions at the selected tau (or ``REQUIRES_SEPARATELY_LOCKED_REFIT``).
    """
    data = _validate_pool(pool)
    y = np.asarray([label == "SELF" for label in data["truth"]], dtype=bool)
    ranked = []
    for cid in sorted(data["candidates"]):
        record = data["candidates"][cid]
        best = None
        for tau in THRESHOLDS:
            try:
                metrics = harness.binary_gate_metrics(y, record["p_self"], tau)
            except ValueError as exc:
                raise F1SelectionError("METRICS: %s" % exc) from None
            if not harness.is_eligible(metrics):
                continue
            key = policy_key(record["family"], record["C"], DIM_WIDTH[record["dimension"]],
                             record["condition"], tau, metrics)
            if best is None or key < best[0]:
                best = (key, float(tau), metrics)
        if best is not None:
            ranked.append((best[0], _entry(cid, record, best[0], best[1], best[2])))
    ranked.sort(key=lambda item: item[0])
    ranking = [entry for _, entry in ranked]

    match = None
    post = None
    if ranking:
        winner = ranking[0]
        cell = cell_key(winner["source"], winner["condition"], winner["dimension"], winner["family"])
        pin = data["artifacts"][cell]
        if float(pin["C"]) == float(winner["C"]):
            match = {"status": "MATCHING_C_ARTIFACT", "cell": cell, "C": float(winner["C"]),
                     "model": pin["model"], "model_sha256": pin["model_sha256"],
                     "predictions": pin["predictions"], "predictions_sha256": pin["predictions_sha256"]}
            post = {"action": "recompute_saved_predictions_at_tau", "cell": cell, "tau": float(winner["tau"]),
                    "C": float(winner["C"]), "predictions": pin["predictions"],
                    "predictions_sha256": pin["predictions_sha256"], "splits": list(POST_FREEZE_SPLITS)}
        else:
            match = {"status": "REQUIRES_SEPARATELY_LOCKED_REFIT", "cell": cell,
                     "selected_C": float(winner["C"]), "artifact_C": float(pin["C"]),
                     "reason": "no saved full-TRAIN refit at the newly selected C; a different-C "
                               "model must never be borrowed"}
            post = {"action": "requires_separately_locked_refit", "cell": cell,
                    "selected_C": float(winner["C"]), "artifact_C": float(pin["C"]),
                    "reason": match["reason"]}

    payload = _frozen_payload(data, ranking, match)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    counts = {"candidates": len(data["candidates"]),
              "new_candidates": sum(1 for cid in data["candidates"]
                                    if data["candidates"][cid]["source"] == SOURCE_COMPRESSION),
              "reference_candidates": sum(1 for cid in data["candidates"]
                                          if data["candidates"][cid]["source"] == SOURCE_REFERENCE),
              "cells": len(data["artifacts"]), "artifacts": len(data["artifacts"]),
              "thresholds": len(THRESHOLDS), "oof_rows": len(data["train_ids"]),
              "eligible_candidates": len(ranking)}
    result = {
        "schema": RESULT_SCHEMA, "job_id": JOB_ID,
        "status": "SELECTED" if ranking else "NO_ELIGIBLE_CANDIDATE",
        "policy_id": POLICY_ID, "policy": POLICY, "selection_rule": list(SELECTION_RULE),
        "validation_policy": VALIDATION_POLICY, "read_order": list(READ_ORDER),
        "validation_used_for_selection": False, "validation_rows_read_during_selection": 0,
        "estimator_fits": 0, "model_deserializations": 0,
        "bounds": dict(BOUNDS), "counts": counts, "thresholds": list(THRESHOLDS),
        "winner": payload["winner"], "ranking": ranking, "artifact_match": match, "post_freeze": post,
        "frozen_identity": hashlib.sha256(canonical).hexdigest(), "frozen_identity_payload": payload,
    }
    probe = json.dumps(result, allow_nan=False).encode("utf-8")
    _require(len(probe) <= OUTPUT_BYTES, "RESULT_BYTES", str(len(probe)))
    result["result_json_bytes"] = len(probe)
    return result


# --------------------------------------------------------------------------- #
# post-freeze: recompute already-saved predictions at the frozen tau
# --------------------------------------------------------------------------- #
def recompute_split_metrics(frozen, split, case_ids, truth, p_self, *, expected_case_ids=None,
                            expected_truth=None):
    """Post-freeze recomputation of saved split predictions at the frozen tau.

    Refuses unless the frozen selection had a matching-C saved artifact. It changes no
    ranking and is the only place validation predictions are consumed.
    """
    _require(type(frozen) is dict and frozen.get("schema") == RESULT_SCHEMA, "FROZEN_SCHEMA")
    _require(frozen.get("validation_used_for_selection") is False, "FROZEN_VALIDATION_FLAG")
    _sha256_hex(frozen.get("frozen_identity"), "FROZEN_IDENTITY")
    post = frozen.get("post_freeze") or {}
    _require(post.get("action") == "recompute_saved_predictions_at_tau", "REFIT_REQUIRED",
             str(post.get("action")))
    _require(split in POST_FREEZE_SPLITS, "SPLIT", str(split))
    _require(type(case_ids) is list and type(truth) is list and len(case_ids) == len(truth),
             "SPLIT_INPUT")
    _require(len(case_ids) == SPLIT_ROWS[split], "SPLIT_ROWS", "%s %d" % (split, len(case_ids)))
    _require(all(type(case_id) is str and case_id != "" for case_id in case_ids), "SPLIT_CASE_IDS")
    _require(len(set(case_ids)) == len(case_ids), "SPLIT_CASE_DUPLICATE")
    _require(all(label in CLASS_ORDER for label in truth), "SPLIT_TRUTH")
    if expected_case_ids is not None:
        _require(case_ids == list(expected_case_ids), "SPLIT_CASE_ORDER")
    if expected_truth is not None:
        _require(truth == list(expected_truth), "SPLIT_TRUTH_ORDER")
    try:
        p = np.asarray(p_self, float)
    except (TypeError, ValueError):
        raise F1SelectionError("SPLIT_OOF_TYPE") from None
    _require(p.shape == (len(case_ids),), "SPLIT_OOF_SHAPE")
    _require(bool(np.isfinite(p).all()), "SPLIT_OOF_FINITE")
    _require(float(p.min()) >= 0.0 and float(p.max()) <= 1.0, "SPLIT_OOF_RANGE")
    tau = float(post["tau"])
    _require(tau in THRESHOLDS, "SPLIT_TAU", str(tau))
    y = np.asarray([label == "SELF" for label in truth], dtype=bool)
    metrics = harness.binary_gate_metrics(y, p, tau)
    _require(harness.is_eligible(metrics), "SPLIT_METRICS_INELIGIBLE")
    return {"schema": SPLIT_SCHEMA, "split": split, "tau": tau, "case_count": len(case_ids),
            "metrics": {"tp": int(metrics["tp"]), "tn": int(metrics["tn"]), "fp": int(metrics["fp"]),
                        "fn": int(metrics["fn"]), "precision": metrics["precision"],
                        "recall": metrics["recall"], "f1": metrics["f1"]},
            "frozen_identity": frozen["frozen_identity"], "selection_unchanged": True,
            "validation_used_for_selection": False}
