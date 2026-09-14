"""PLAN_V3 stage-3 cached J-lens pilot logic (job ``jlens_pilot_v1``).

MODEL-FREE GLUE. This module imports only the standard library, NumPy, the two
reviewed J-lens modules (``jlens_core_v2`` pure readout math and ``jlens_io_v1``
loader/pinning) and the pinned classifier helpers ``harness`` /
``grouped_driver``. It never imports ``torch``/``transformers``/``tokenizers``/
``safetensors``/``transformer_lens``, never loads a model, tokenizer, cache
tensor, lens tensor, forward, derivative, PCA or cone, and never runs the real
pilot. The estimator backend is the existing sklearn ``lbfgs``
``LogisticRegression`` factory re-exported from ``grouped_driver``; no custom
IRLS is present.

What is frozen here (authoritative spec: ``PLAN_V3.json``)
----------------------------------------------------------
* ``cell_A_fixed_concept_raw_score``: 6 frozen single-token surfaces x 3 layers
  x 2 conditions x 2 methods = 72 fit-free cells. Score = the raw direct logit
  (``jlens_core_v2.raw_direct_logit``) at the surface token. Threshold from the
  finite 19-point TRAIN quantile grid, per held-out fold only from that fold's
  TRAIN partition.
* ``cell_B_regularized_binary_combination``: 2 C values x 3 layers x 2
  conditions x 2 methods = 24 learned C-cells over all R <= 6 retained raw
  scores jointly. Standardization and C selection use TRAIN-fold rows only;
  5 grouped folds (120 CV fits) plus one best-C full-TRAIN refit per
  (method, condition, layer) (12 refits) = 132 fits maximum.
* The V1 tie-break chain is reproduced verbatim; validation is reported and is
  never used for selection. Any nonconverged, non-finite or incomplete-fold cell
  is invalidated whole and reported; folds are never pooled across cells.

Only the real runner (``run_jlens_pilot_v1.py``) reads the two authenticated
cached activation sets and the pinned lens/norm/unembed slices. This module is
pure arithmetic over supplied arrays so it is testable with synthetic scores.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
import warnings
from pathlib import Path

import numpy as np

import jlens_core_v2 as core
import jlens_io_v1 as io

__all__ = [
    "PilotError",
    "BudgetError",
    "need",
    "JOB_ID",
    "PILOT_SCHEMA",
    "CELLS_SCHEMA",
    "RECEIPT_SCHEMA",
    "CONCEPT_SURFACES",
    "SCORE_LAYERS",
    "CONDITIONS",
    "METHODS",
    "CLASS_ORDER",
    "NEGATIVE_CLASSES",
    "C_VALUES",
    "FOLDS",
    "QUANTILE_GRID",
    "TIE_BREAK_CHAIN",
    "R_MAX",
    "CELL_A_CELLS",
    "CELL_B_CELLS",
    "TOTAL_CELLS",
    "CV_FITS_MAX",
    "REFITS_MAX",
    "TOTAL_FITS_MAX",
    "MAX_SELECTED_LOGIT_READS",
    "MAX_RECORDS_DECODED",
    "MAX_CACHED_ACTIVATION_TENSORS",
    "CAPS",
    "CLASSIFIER_SOURCES",
    "PILOT_CACHE_SCHEMAS",
    "FORBIDDEN_PATH_PARTS",
    "harness",
    "driver",
    "span_module",
    "reject_forbidden_path",
    "inspect_pilot_cache",
    "load_pilot_manifests",
    "validate_surfaces",
    "binary_targets",
    "wilson_interval",
    "cell_selection_key",
    "fold_partitions",
    "method_scores",
    "select_threshold_cell",
    "split_metrics",
    "_split_metrics_by_group",
    "cross_validate_binary",
    "refit_binary",
    "run_pilot",
]

JOB_ID = "jlens_pilot_v1"
PILOT_SCHEMA = "jlens_pilot.v1"
CELLS_SCHEMA = "jlens_pilot_cells.v1"
RECEIPT_SCHEMA = "jlens_pilot_receipt.v1"

CONCEPT_SURFACES = tuple(core.CONCEPT_SURFACES_V1)
SCORE_LAYERS = tuple(core.SCORE_LAYERS_V1)
CONDITIONS = tuple(core.CONDITIONS_V1)
METHODS = tuple(core.READOUT_METHODS_V1)
CLASS_ORDER = ("SELF", "OTHER", "NONTERMINATION", "ORDINARY")
NEGATIVE_CLASSES = ("OTHER", "NONTERMINATION", "ORDINARY")

C_VALUES = (0.1, 1.0)
FOLDS = (0, 1, 2, 3, 4)
QUANTILE_GRID = tuple(round(0.05 * index, 2) for index in range(1, 20))  # 19 values

# PLAN_V3.threshold_plan.tie_break_chain, in order, highest priority first.
TIE_BREAK_CHAIN = (
    "max min(precision, recall)",
    "max f1",
    "lower C",
    "earlier method index (j_lens before J=I)",
    "earlier condition index (unprompted before prompted)",
    "lower layer",
    "lower cell index",
    "binary-first",
    "abs(tau-0.5)",
    "smaller tau",
    "lower feature count",
    "lower concept-surface index",
)

R_MAX = 6
CELL_A_CELLS = len(CONCEPT_SURFACES) * len(SCORE_LAYERS) * len(CONDITIONS) * len(METHODS)  # 72
CELL_B_CELLS = len(C_VALUES) * len(SCORE_LAYERS) * len(CONDITIONS) * len(METHODS)  # 24
TOTAL_CELLS = CELL_A_CELLS + CELL_B_CELLS  # 96
CV_FITS_MAX = CELL_B_CELLS * len(FOLDS)  # 120
REFITS_MAX = len(SCORE_LAYERS) * len(CONDITIONS) * len(METHODS)  # 12
TOTAL_FITS_MAX = CV_FITS_MAX + REFITS_MAX  # 132
MAX_SELECTED_LOGIT_READS = CELL_A_CELLS * (240 + 80)  # 23040
MAX_RECORDS_DECODED = (240 + 80) * 2 * len(SCORE_LAYERS) * len(CONDITIONS)  # 3840
MAX_CACHED_ACTIVATION_TENSORS = len(SCORE_LAYERS) * len(CONDITIONS)  # 6

CAPS = dict(
    readout_seconds=900,
    fit_seconds=600,
    hard_total_seconds=1800,
    output_bytes=67108864,
    working_memory_gib=2.5,
    model_loads=0,
    tokenizer_loads=0,
    forwards=0,
    derivatives=0,
    cone_fits=0,
    pca_fits=0,
    hyperparameter_search=False,
    holdout_reads=0,
)

# Frozen reviewed classifier helpers (hashes recorded in SPAN_FIT_PLAN_V2.json).
CLASSIFIER_SOURCES = {
    "harness": "b14930e9ca6e5d8350569b986c4584135caa7ba2a5e6685e3629f5762b07e9e4",
    "grouped_driver": "d44064e1e3035455ab1b7bb36f0ecc7112965470bf0d2a3a9e47811e0548cc4b",
    "span_feature_transforms_v1": "a23aaa38dc36a764af8cd30d2d8ed40193a9a3496a9d8687778c96f74741c5c0",
    "span_classifier_driver_v1": "69b42fff78bae74414d291328345577667ce84d8e16be07a62b4404a15acbebf",
}

PILOT_CACHE_SCHEMAS = ("span_capture_index.v1", "prompted_span_capture_index.v1")
FORBIDDEN_PATH_PARTS = ("holdout", "private")

_HERE = Path(__file__).resolve().parent
_CLASSIFIER = (_HERE.parent / "classifier_generalization_v2").resolve()
_SPAN = None


class PilotError(ValueError):
    """Structured rejection raised by this module."""


class BudgetError(PilotError):
    """A frozen fit/read budget was exceeded; it must abort, never be pooled."""


def need(condition, code, detail=""):
    if not condition:
        raise PilotError(code if not detail else "%s: %s" % (code, detail))


def _short(exc):
    return "%s: %s" % (type(exc).__name__, str(exc)[:240])


# --------------------------------------------------------------------------- #
# reviewed classifier helpers loaded by explicit path (never by sys.path)
# --------------------------------------------------------------------------- #
def _load_classifier_module(name):
    if name in sys.modules:
        return sys.modules[name]
    need(name in CLASSIFIER_SOURCES, "CLASSIFIER_MODULE_UNKNOWN", name)
    path = _CLASSIFIER / (name + ".py")
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == CLASSIFIER_SOURCES[name], "CLASSIFIER_SOURCE_HASH", name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ``harness`` and ``grouped_driver`` are NumPy-only at import time; sklearn is
# reached lazily inside ``grouped_driver._default_factory``. The manifest loader
# (``span_classifier_driver_v1``) pulls sklearn at import and is therefore lazy.
harness = _load_classifier_module("harness")
driver = _load_classifier_module("grouped_driver")


def span_module():
    """Load the reviewed manifest/fold reader lazily (it imports sklearn)."""
    global _SPAN
    if _SPAN is None:
        _load_classifier_module("span_feature_transforms_v1")
        _SPAN = _load_classifier_module("span_classifier_driver_v1")
    return _SPAN


def load_pilot_manifests(manifest_pins, root):
    """Reuse the reviewed ``span_classifier_driver_v1.load_manifests`` reader."""
    return span_module().load_manifests(manifest_pins, root)


# --------------------------------------------------------------------------- #
# input metadata (no tensor body is ever read here)
# --------------------------------------------------------------------------- #
def reject_forbidden_path(path):
    """Reject any path whose components mention a holdout or private corpus."""
    for part in Path(path).parts:
        lowered = part.lower()
        for banned in FORBIDDEN_PATH_PARTS:
            need(banned not in lowered, "FORBIDDEN_PATH_PART", "%s in %s" % (banned, path))


def inspect_pilot_cache(index_path, index_pin, windows_path, windows_pin, *,
                        admitted_ids, held_ids, condition=None):
    """Authenticate one cached-view index and return a lazy ``io.CacheIndex``.

    This is the ``jlens_io_v1.inspect_cache`` contract relaxed for the second
    authenticated cache: the unprompted index uses ``span_capture_index.v1`` and
    the prompted index uses ``prompted_span_capture_index.v1`` (with a
    ``prompted_fixed_query_v1`` condition). Every structural, alignment,
    accounting and holdout check is retained; the per-record alignment check is
    the reviewed ``jlens_io_v1._check_record_alignment``. No tensor body is read.
    """
    reject_forbidden_path(index_path)
    reject_forbidden_path(windows_path)
    io.strict_file(index_path, index_pin, full_hash=True)
    windows_size = io.strict_file(windows_path, windows_pin, full_hash=False)
    document = io.runner.strict_json(Path(index_path).read_bytes())
    need(type(document) is dict, "CACHE_INDEX_SCHEMA")
    need(document.get("schema") in PILOT_CACHE_SCHEMAS, "CACHE_INDEX_SCHEMA")
    if condition is not None:
        need(document.get("condition") == condition, "CACHE_CONDITION")
    need(document.get("blocks") == list(io.BLOCKS), "CACHE_BLOCKS")
    need(document.get("dtype") == "float32", "CACHE_DTYPE")
    need(document.get("byte_order") == "little", "CACHE_BYTE_ORDER")
    need(document.get("float_bytes") == io.FLOAT_BYTES, "CACHE_FLOAT_BYTES")
    need(document.get("width") == io.D_MODEL, "CACHE_WIDTH")
    need(document.get("case_count") == io.CACHE_CASES, "CACHE_CASE_COUNT")
    need(document.get("view_count") == io.CACHE_VIEWS, "CACHE_VIEW_COUNT")
    need(document.get("forward_count") == io.CACHE_VIEWS, "CACHE_FORWARD_COUNT")
    need(document.get("raw_bytes") == windows_size, "CACHE_RAW_BYTES")
    case_order = document.get("case_order")
    need(
        type(case_order) is list
        and len(case_order) == io.CACHE_CASES
        and len(set(case_order)) == io.CACHE_CASES,
        "CACHE_CASE_ORDER",
    )
    need(set(case_order) <= set(admitted_ids), "CACHE_CASE_NOT_ADMITTED")
    need(not (set(case_order) & set(held_ids)), "CACHE_HOLDOUT_OVERLAP")
    records = document.get("records")
    need(type(records) is list and len(records) == io.CACHE_VIEWS * len(io.BLOCKS), "CACHE_RECORD_COUNT")

    names, seen, splits = set(), set(), {"TRAIN": set(), "VALIDATION": set()}
    total, previous, grouped = 0, 0, {}
    for record in records:
        io._check_record_alignment(record, names)
        key = (record["case_id"], record["order"], record["block"])
        need(key not in seen, "CACHE_RECORD_DUPLICATE")
        seen.add(key)
        need(record["offset"] == previous, "CACHE_OFFSET_ORDER")
        previous += record["length"]
        total += record["length"]
        splits[record["split"]].add(record["case_id"])
        grouped.setdefault((record["case_id"], record["order"]), []).append(record["block"])
    need(total == document["raw_bytes"], "CACHE_RAW_BYTES")
    need(names == set(case_order), "CACHE_CASE_ORDER")
    need(len(splits["TRAIN"]) == io.ADMITTED_TRAIN, "CACHE_TRAIN_COUNT")
    need(len(splits["VALIDATION"]) == io.ADMITTED_VALIDATION, "CACHE_VALIDATION_COUNT")
    for blocks in grouped.values():
        need(sorted(blocks) == list(io.BLOCKS), "CACHE_LAYER_ALIGNMENT")
    return io.CacheIndex(Path(index_path), Path(windows_path), document, windows_size)


def validate_surfaces(surfaces):
    """Validate the pinned single-token surface table; return ``[(surface, id)]``.

    Tokenization is a stage-1 bounded job; the pilot never loads a tokenizer, so
    it consumes the already-pinned ``(surface, token_id)`` pairs and requires the
    reviewed single-token assertion to be explicit in the lock.
    """
    need(type(surfaces) is list and 1 <= len(surfaces) <= R_MAX, "SURFACE_BUDGET")
    retained, names, ids = [], set(), set()
    for item in surfaces:
        need(type(item) is dict and set(item) == {"surface", "token_id", "single_token"}, "SURFACE_SCHEMA")
        need(item["single_token"] is True, "SURFACE_NOT_SINGLE_TOKEN")
        name = item["surface"]
        token_id = item["token_id"]
        need(name in CONCEPT_SURFACES, "SURFACE_UNKNOWN", str(name))
        need(
            type(token_id) is int and not isinstance(token_id, bool) and 0 <= token_id < io.VOCAB_SIZE,
            "SURFACE_TOKEN_ID",
            str(token_id),
        )
        need(name not in names, "SURFACE_DUPLICATE", name)
        need(token_id not in ids, "SURFACE_DUPLICATE_TOKEN", str(token_id))
        names.add(name)
        ids.add(token_id)
        retained.append((name, token_id))
    return retained


# --------------------------------------------------------------------------- #
# readout + deterministic metrics
# --------------------------------------------------------------------------- #
def _finite_2d(scores, name):
    need(isinstance(scores, np.ndarray), "ARRAY_TYPE", name)
    need(scores.ndim == 2 and scores.shape[0] > 0 and scores.shape[1] > 0, "ARRAY_SHAPE", name)
    need(scores.dtype.kind == "f", "ARRAY_DTYPE", name)
    need(bool(np.isfinite(scores).all()), "ARRAY_NONFINITE", name)
    return scores


def method_scores(hidden, layer, method, tensors, contract, counters=None):
    """Raw direct logits ``(n_cases, R)`` for one method/layer readout.

    ``j_lens_transport`` uses the pinned ``J_l``; the matched control replaces it
    with the identity so the final norm and selected unembedding path are byte-for-
    byte the same code. Only the selected unembedding rows are contracted.
    """
    need(method in METHODS, "METHOD_UNKNOWN", str(method))
    hidden = _finite_2d(hidden, "hidden")
    need(hidden.shape[1] == contract.d_model, "HIDDEN_WIDTH")
    if method == METHODS[0]:
        jacobian = tensors["jacobians"][layer]
    else:
        jacobian = np.eye(contract.d_model, dtype=np.float32)
    logits = core.raw_direct_logit(hidden, jacobian, tensors["norm"], tensors["rows"], contract)
    reads = int(hidden.shape[0]) * int(len(contract.token_ids))
    if counters is not None:
        counters["selected_logit_reads"] = int(counters.get("selected_logit_reads", 0)) + reads
        if counters["selected_logit_reads"] > MAX_SELECTED_LOGIT_READS:
            raise BudgetError("SELECTED_LOGIT_READ_BUDGET")
    return np.ascontiguousarray(logits, dtype=np.float32)


def binary_targets(labels):
    return np.asarray([1 if label == "SELF" else 0 for label in labels], dtype=np.int64)


def wilson_interval(successes, total, z=1.959963984540054):
    """Wilson score interval; ``(None, None)`` when the denominator is zero."""
    if total <= 0:
        return None, None
    p = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    low = 0.0 if successes <= 0 else max(0.0, center - half)
    high = 1.0 if successes >= total else min(1.0, center + half)
    return low, high


def cell_selection_key(*, metrics, tau, c_value, method_index, condition_index, layer,
                       cell_index, is_binary, feature_count, surface_index):
    """PLAN_V3 threshold_plan.tie_break_chain as a deterministic sort key."""
    precision, recall, f1 = metrics["precision"], metrics["recall"], metrics["f1"]
    if precision is None or recall is None or f1 is None:
        return None
    return (
        -min(float(precision), float(recall)),
        -float(f1),
        float(c_value),
        int(method_index),
        int(condition_index),
        int(layer),
        int(cell_index),
        0 if is_binary else 1,
        abs(float(tau) - 0.5),
        float(tau),
        int(feature_count),
        int(surface_index),
    )


def fold_partitions(train_folds, folds=FOLDS):
    """Grouped folds over the TRAIN subset; each part is ``(held, train, test)``."""
    train_folds = np.asarray(train_folds, dtype=int)
    need(set(train_folds.tolist()) == set(folds), "FOLD_SET")
    partitions = []
    for held in folds:
        test_index = np.flatnonzero(train_folds == held)
        train_index = np.flatnonzero(train_folds != held)
        need(test_index.size > 0 and train_index.size > 0, "EMPTY_FOLD", str(held))
        partitions.append((held, train_index, test_index))
    return partitions


def select_threshold_cell(*, fold_train_scores, fold_test_scores, full_train_scores,
                          y_train, partitions, key_context):
    """Select tau(q) on the 19-point grid using TRAIN-fold scores only."""
    y_train = np.asarray(y_train)
    n_train = y_train.size
    need(n_train > 0, "EMPTY_TRAIN")
    thresholds, best = [], None
    for q in QUANTILE_GRID:
        oof_pred = np.zeros(n_train, dtype=bool)
        for held, _, test_index in partitions:
            tau_q = core.select_quantile_threshold(fold_train_scores[held], q)
            oof_pred[test_index] = core.predict_positive(fold_test_scores[held], tau_q)
        metrics = harness.binary_metrics(y_train, oof_pred)
        eligible = bool(harness.is_eligible(metrics))
        tau_star_q = core.select_quantile_threshold(full_train_scores, q)
        thresholds.append(
            {
                "q": float(q),
                "tau": float(tau_star_q),
                "eligible": eligible,
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
        if eligible:
            key = cell_selection_key(metrics=metrics, tau=float(tau_star_q), **key_context)
            if best is None or key < best[0]:
                best = (key, float(q), float(tau_star_q), metrics)
    if best is None:
        return {
            "status": "NO_ELIGIBLE_TRAIN_CANDIDATE",
            "error": "NO_ELIGIBLE_TRAIN_CANDIDATE",
            "thresholds": thresholds,
            "selected_q": None,
            "tau": None,
            "oof_metrics": None,
            "selection_key": None,
        }
    key, selected_q, tau_star, metrics = best
    return {
        "status": "VALID",
        "error": None,
        "thresholds": thresholds,
        "selected_q": selected_q,
        "tau": tau_star,
        "oof_metrics": metrics,
        "selection_key": [float(value) for value in key],
    }


def split_metrics(scores, tau, labels_subset, p_self=None):
    """Binary SELF-gate metrics for one split with negative-class breakdown."""
    pred = core.predict_positive(scores, tau)
    labels_array = np.asarray(labels_subset)
    y = binary_targets(labels_subset)
    metrics = harness.binary_metrics(y, pred)
    negative_counts = {}
    for cls in NEGATIVE_CLASSES:
        mask = labels_array == cls
        negative_counts[cls] = {
            "n": int(mask.sum()),
            "predicted_positive": int(np.sum(pred & mask)),
        }
    wilson_precision = wilson_interval(metrics["tp"], metrics["tp"] + metrics["fp"])
    wilson_recall = wilson_interval(metrics["tp"], metrics["tp"] + metrics["fn"])
    record = dict(metrics)
    record["negative_counts"] = negative_counts
    record["wilson_precision"] = None if wilson_precision[0] is None else list(wilson_precision)
    record["wilson_recall"] = None if wilson_recall[0] is None else list(wilson_recall)
    if p_self is not None:
        record["p_self"] = [float(value) for value in np.asarray(p_self, dtype=np.float64)]
    return record


def _split_metrics_by_group(scores, tau, labels, groups, combined_name="combined80"):
    """Per-manifest validation metrics (original40 / added40 / combined80).

    ``groups`` is one label per validation row in the frozen validation order.
    Selection never sees this; the breakdown is reporting only. Returns ``None``
    when no grouping is supplied so the V1 pooled behavior is unchanged.
    """
    if groups is None:
        return None
    scores = np.asarray(scores)
    labels = list(labels)
    groups = list(groups)
    need(scores.shape[0] == len(labels) == len(groups), "SPLIT_GROUP_ALIGNMENT")
    out = {}
    for name in sorted(set(groups)):
        index = [i for i, group in enumerate(groups) if group == name]
        out[name] = split_metrics(scores[index], tau, [labels[i] for i in index])
    out[combined_name] = split_metrics(scores, tau, labels)
    return out


# --------------------------------------------------------------------------- #
# learned cell: standardization on TRAIN-fold rows, sklearn lbfgs, honest gates
# --------------------------------------------------------------------------- #
def _fit_with_convergence_gate(estimator, x_train, target):
    """Fit once; a ConvergenceWarning or fit failure invalidates the whole cell."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(x_train, target)
    for warning in caught:
        if "ConvergenceWarning" in type(warning.category).__name__:
            return "FIT_CONVERGENCE_WARNING"
    coefficient = getattr(estimator, "coef_", None)
    if coefficient is None or not bool(np.isfinite(np.asarray(coefficient, float)).all()):
        return "NONFINITE_COEFFICIENTS"
    if not driver._converged(estimator):
        return "NOT_CONVERGED"
    return None


def _standardize(train_rows, apply_rows):
    mean = np.asarray(train_rows, float).mean(axis=0)
    std = np.asarray(train_rows, float).std(axis=0, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0.0):
        raise PilotError("DEGENERATE_STANDARDIZATION")
    return (np.asarray(apply_rows, float) - mean) / std


def cross_validate_binary(raw_train, y_train, partitions, c_value, factory, counters, deadline):
    """Five grouped CV folds; a single failed fold invalidates the whole C-cell.

    Fold thresholds come from the fold model's in-sample TRAIN-fold logits, so
    the held-out fold never enters its own threshold.
    """
    n_train = raw_train.shape[0]
    oof_logit = np.full(n_train, np.nan)
    fold_train_logits, fold_test_logits, errors = {}, {}, {}
    for held, train_index, test_index in partitions:
        deadline()
        if set(np.asarray(y_train)[train_index].tolist()) != {0, 1}:
            errors[str(held)] = "DEGENERATE_FOLD"
            continue
        try:
            x_train = _standardize(raw_train[train_index], raw_train[train_index])
            x_test = _standardize(raw_train[train_index], raw_train[test_index])
        except PilotError as exc:
            errors[str(held)] = str(exc)
            continue
        try:
            estimator = factory(harness.MODEL_BINARY, c_value)
            counters["cv_fits"] = int(counters.get("cv_fits", 0)) + 1
            _check_fit_caps(counters)
            error = _fit_with_convergence_gate(estimator, x_train, np.asarray(y_train)[train_index])
        except BudgetError:
            raise
        except Exception as exc:  # noqa: BLE001 - any fit failure voids this cell
            counters["fit_errors"] = int(counters.get("fit_errors", 0)) + 1
            errors[str(held)] = _short(exc)
            continue
        if error is not None:
            counters["fit_errors"] = int(counters.get("fit_errors", 0)) + 1
            errors[str(held)] = error
            continue
        try:
            driver._fold_probabilities(harness.MODEL_BINARY, estimator, x_test, test_index.size)
        except (TypeError, ValueError) as exc:
            errors[str(held)] = _short(exc)
            continue
        logit_test = np.asarray(estimator.decision_function(x_test), float)
        logit_train = np.asarray(estimator.decision_function(x_train), float)
        if not (np.isfinite(logit_test).all() and np.isfinite(logit_train).all()):
            errors[str(held)] = "NONFINITE_LOGIT"
            continue
        fold_train_logits[held] = logit_train
        fold_test_logits[held] = logit_test
        oof_logit[test_index] = logit_test
    complete = len(fold_train_logits) == len(partitions)
    return {
        "complete": bool(complete),
        "errors": errors,
        "oof_logit": oof_logit,
        "fold_train_logits": fold_train_logits,
        "fold_test_logits": fold_test_logits,
    }


def refit_binary(raw_train, y_train, raw_validation, c_value, factory, counters, deadline):
    """One full-TRAIN refit at the selected C; standardization on TRAIN only."""
    deadline()
    x_train = _standardize(raw_train, raw_train)
    x_validation = _standardize(raw_train, raw_validation)
    estimator = factory(harness.MODEL_BINARY, c_value)
    counters["refits"] = int(counters.get("refits", 0)) + 1
    _check_fit_caps(counters)
    error = _fit_with_convergence_gate(estimator, x_train, np.asarray(y_train))
    need(error is None, "REFIT_FAILED", error or "")
    logit_train = np.asarray(estimator.decision_function(x_train), float)
    logit_validation = np.asarray(estimator.decision_function(x_validation), float)
    need(
        bool(np.isfinite(logit_train).all() and np.isfinite(logit_validation).all()),
        "REFIT_NONFINITE_LOGIT",
    )
    probabilities = np.asarray(estimator.predict_proba(x_validation), float)
    need(probabilities.shape == (raw_validation.shape[0], 2), "REFIT_PROBA_SHAPE", str(probabilities.shape))
    return {
        "logit_train": logit_train,
        "logit_validation": logit_validation,
        "p_train": core.logistic_probability(logit_train),
        "p_validation": core.logistic_probability(logit_validation),
        "proba_validation": probabilities,
        "classes": [int(value) for value in np.asarray(estimator.classes_).tolist()],
        "n_iter": int(np.max(np.asarray(getattr(estimator, "n_iter_", [0]), float))),
    }


def _check_fit_caps(counters):
    if int(counters.get("cv_fits", 0)) > CV_FITS_MAX:
        raise BudgetError("CV_FIT_BUDGET")
    if int(counters.get("refits", 0)) > REFITS_MAX:
        raise BudgetError("REFIT_BUDGET")
    if int(counters.get("cv_fits", 0)) + int(counters.get("refits", 0)) > TOTAL_FITS_MAX:
        raise BudgetError("TOTAL_FIT_BUDGET")


def _invalid_cell(base, error):
    record = dict(base)
    record.update(
        status="INVALID",
        error=error,
        selected_q=None,
        tau=None,
        oof_metrics=None,
        selection_key=None,
        thresholds=[],
        frozen_train_metrics=None,
        validation_metrics=None,
        validation_split_metrics=None,
    )
    return record


# --------------------------------------------------------------------------- #
# the frozen 96-cell pilot
# --------------------------------------------------------------------------- #
def run_pilot(*, scores, case_ids, labels, splits, folds, surface_names=None,
              factory=None, deadline=None, counters=None, validation_groups=None):
    """Evaluate the frozen PLAN_V3 96 cells over supplied raw scores.

    ``scores[method][condition][layer]`` is a ``(n_cases, R)`` float array of raw
    direct logits with columns in the retained surface order given by
    ``surface_names``. Selection uses only TRAIN-fold rows; validation is
    reported and never used for selection. A fold whose TRAIN partition lacks
    either class invalidates every cell that depends on it, whole and reported.
    """
    need(factory is None or callable(factory), "FACTORY_TYPE")
    factory = factory if factory is not None else driver._default_factory
    deadline = deadline if deadline is not None else (lambda: None)
    counters = dict(counters) if counters is not None else {}
    for key in ("cv_fits", "refits", "fit_errors", "selected_logit_reads"):
        counters.setdefault(key, 0)
    labels, case_ids = list(labels), list(case_ids)
    n_cases = len(case_ids)
    need(len(labels) == n_cases and len(splits) == n_cases and len(folds) == n_cases, "CASE_LENGTHS")
    y = binary_targets(labels)
    split_array = np.asarray(splits)
    train_index = np.flatnonzero(split_array == "TRAIN")
    validation_index = np.flatnonzero(split_array == "VALIDATION")
    need(train_index.size == 240 and validation_index.size == 80, "SPLIT_COUNTS")
    train_folds = np.asarray([folds[index] for index in train_index.tolist()], dtype=int)
    partitions = fold_partitions(train_folds)
    y_train = y[train_index]
    labels_train = [labels[index] for index in train_index.tolist()]
    labels_validation = [labels[index] for index in validation_index.tolist()]
    validation_groups = list(validation_groups) if validation_groups is not None else None
    if validation_groups is not None:
        need(len(validation_groups) == validation_index.size, "VALIDATION_GROUPS")
    fold_classes_ok = all(set(y_train[train].tolist()) == {0, 1} for _, train, _ in partitions)
    probe = _finite_2d(
        np.asarray(scores[METHODS[0]][CONDITIONS[0]][SCORE_LAYERS[0]], dtype=np.float32), "scores"
    )
    need(probe.shape[0] == n_cases, "SCORE_CASE_COUNT")
    n_features = int(probe.shape[1])
    need(1 <= n_features <= R_MAX, "FEATURE_COUNT")
    names = list(surface_names) if surface_names is not None else list(CONCEPT_SURFACES[:n_features])
    need(len(names) == n_features, "SURFACE_NAME_COUNT")

    cells, refits = [], []
    for method_index, method in enumerate(METHODS):
        for condition_index, condition in enumerate(CONDITIONS):
            for layer_index, layer in enumerate(SCORE_LAYERS):
                raw = _finite_2d(np.asarray(scores[method][condition][layer], dtype=np.float32), "scores")
                need(raw.shape[0] == n_cases, "SCORE_CASE_COUNT")
                need(int(raw.shape[1]) == n_features, "FEATURE_COUNT_MISMATCH")
                raw_train = raw[train_index]
                raw_validation = raw[validation_index]

                # cell_A: one fit-free raw cell per retained surface.
                for surface_index in range(n_features):
                    cell_index = (
                        (method_index * len(CONDITIONS) + condition_index) * len(SCORE_LAYERS) + layer_index
                    ) * R_MAX + surface_index
                    base = {
                        "scheme": "cell_A",
                        "cell_index": int(cell_index),
                        "method": method,
                        "condition": condition,
                        "layer": int(layer),
                        "surface_index": int(surface_index),
                        "surface": names[surface_index],
                        "C": None,
                        "feature_count": 1,
                    }
                    if not fold_classes_ok:
                        cells.append(_invalid_cell(base, "DEGENERATE_FOLD"))
                        continue
                    key_context = dict(
                        c_value=0.0,
                        method_index=method_index,
                        condition_index=condition_index,
                        layer=layer,
                        cell_index=cell_index,
                        is_binary=True,
                        feature_count=1,
                        surface_index=surface_index,
                    )
                    fold_train = {held: raw_train[train, surface_index] for held, train, _ in partitions}
                    fold_test = {held: raw_train[test, surface_index] for held, _, test in partitions}
                    selection = select_threshold_cell(
                        fold_train_scores=fold_train,
                        fold_test_scores=fold_test,
                        full_train_scores=raw_train[:, surface_index],
                        y_train=y_train,
                        partitions=partitions,
                        key_context=key_context,
                    )
                    if selection["status"] != "VALID":
                        cells.append(_invalid_cell(base, selection["error"]))
                        continue
                    tau = selection["tau"]
                    record = dict(base)
                    record.update(
                        status="VALID",
                        error=None,
                        selected_q=selection["selected_q"],
                        tau=float(tau),
                        thresholds=selection["thresholds"],
                        oof_metrics=selection["oof_metrics"],
                        selection_key=selection["selection_key"],
                        frozen_train_metrics=split_metrics(
                            raw_train[:, surface_index], tau, labels_train
                        ),
                        validation_metrics=split_metrics(
                            raw_validation[:, surface_index], tau, labels_validation
                        ),
                        validation_split_metrics=_split_metrics_by_group(
                            raw_validation[:, surface_index], tau, labels_validation, validation_groups
                        ),
                    )
                    cells.append(record)

                # cell_B: one learned C-cell per C value over all R features jointly.
                learned = []
                for c_index, c_value in enumerate(C_VALUES):
                    cell_index = CELL_A_CELLS + (
                        (method_index * len(CONDITIONS) + condition_index) * len(SCORE_LAYERS) + layer_index
                    ) * len(C_VALUES) + c_index
                    base = {
                        "scheme": "cell_B",
                        "cell_index": int(cell_index),
                        "method": method,
                        "condition": condition,
                        "layer": int(layer),
                        "surface_index": -1,
                        "surface": None,
                        "C": float(c_value),
                        "feature_count": n_features,
                    }
                    cv = cross_validate_binary(
                        raw_train, y_train, partitions, c_value, factory, counters, deadline
                    )
                    if not cv["complete"]:
                        record = _invalid_cell(base, "INCOMPLETE_FOLD")
                        record["fold_errors"] = cv["errors"]
                        cells.append(record)
                        continue
                    key_context = dict(
                        c_value=float(c_value),
                        method_index=method_index,
                        condition_index=condition_index,
                        layer=layer,
                        cell_index=cell_index,
                        is_binary=True,
                        feature_count=n_features,
                        surface_index=-1,
                    )
                    selection = select_threshold_cell(
                        fold_train_scores=cv["fold_train_logits"],
                        fold_test_scores=cv["fold_test_logits"],
                        full_train_scores=cv["oof_logit"],
                        y_train=y_train,
                        partitions=partitions,
                        key_context=key_context,
                    )
                    if selection["status"] != "VALID":
                        record = _invalid_cell(base, selection["error"])
                        record["thresholds"] = selection["thresholds"]
                        cells.append(record)
                        continue
                    record = dict(base)
                    oof_p_self = core.logistic_probability(cv["oof_logit"])
                    record.update(
                        status="VALID",
                        error=None,
                        selected_q=selection["selected_q"],
                        tau=float(selection["tau"]),
                        thresholds=selection["thresholds"],
                        oof_metrics=selection["oof_metrics"],
                        selection_key=selection["selection_key"],
                        oof_logit=[float(value) for value in cv["oof_logit"]],
                        oof_p_self=[float(value) for value in oof_p_self],
                        frozen_train_metrics=split_metrics(
                            cv["oof_logit"], selection["tau"], labels_train, p_self=oof_p_self
                        ),
                        validation_metrics=None,
                        validation_split_metrics=None,
                    )
                    cells.append(record)
                    learned.append(record)

                if learned:
                    best = min(learned, key=lambda record: record["selection_key"])
                    try:
                        refit = refit_binary(
                            raw_train, y_train, raw_validation, best["C"], factory, counters, deadline
                        )
                    except BudgetError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - a failed refit voids that frozen cell
                        refits.append(
                            {
                                "method": method,
                                "condition": condition,
                                "layer": int(layer),
                                "C": best["C"],
                                "tau": best["tau"],
                                "status": "INVALID",
                                "error": _short(exc),
                                "validation_split_metrics": None,
                            }
                        )
                        continue
                    need(refit["classes"] == [0, 1], "REFIT_CLASSES", str(refit["classes"]))
                    refits.append(
                        {
                            "method": method,
                            "condition": condition,
                            "layer": int(layer),
                            "C": best["C"],
                            "tau": best["tau"],
                            "status": "VALID",
                            "error": None,
                            "fits": 1,
                            "classes": refit["classes"],
                            "n_iter": refit["n_iter"],
                            "train_metrics": split_metrics(
                                refit["logit_train"], best["tau"], labels_train, p_self=refit["p_train"]
                            ),
                            "validation_metrics": split_metrics(
                                refit["logit_validation"],
                                best["tau"],
                                labels_validation,
                                p_self=refit["p_validation"],
                            ),
                            "validation_split_metrics": _split_metrics_by_group(
                                refit["logit_validation"], best["tau"], labels_validation, validation_groups
                            ),
                        }
                    )

    valid = [cell for cell in cells if cell["status"] == "VALID"]
    no_eligible = [cell for cell in cells if cell["status"] == "NO_ELIGIBLE_TRAIN_CANDIDATE"]
    invalid = [cell for cell in cells if cell["status"] == "INVALID"]
    cell_a_built = sum(1 for cell in cells if cell["scheme"] == "cell_A")
    cell_b_built = sum(1 for cell in cells if cell["scheme"] == "cell_B")
    ranked = sorted(valid, key=lambda cell: cell["selection_key"])
    result = {
        "schema": CELLS_SCHEMA,
        "job_id": JOB_ID,
        "status": "COMPLETE" if ranked else "NO_ELIGIBLE_CANDIDATE",
        "counts": {
            "retained_surfaces": n_features,
            "retained_surface_names": names,
            "cell_A_cells_built": cell_a_built,
            "cell_B_cells_built": cell_b_built,
            "total_cells_built": len(cells),
            "cell_A_cells_max": CELL_A_CELLS,
            "cell_B_cells_max": CELL_B_CELLS,
            "total_cells_max": TOTAL_CELLS,
            "valid": len(valid),
            "invalid": len(invalid),
            "no_eligible_train_candidate": len(no_eligible),
        },
        "fits": {
            "cv_fits": int(counters.get("cv_fits", 0)),
            "refits": int(counters.get("refits", 0)),
            "total": int(counters.get("cv_fits", 0)) + int(counters.get("refits", 0)),
            "fit_errors": int(counters.get("fit_errors", 0)),
            "cv_fits_max": CV_FITS_MAX,
            "refits_max": REFITS_MAX,
            "total_fits_max": TOTAL_FITS_MAX,
        },
        "selected_logit_reads": int(counters.get("selected_logit_reads", 0)),
        "selected_logit_reads_max": MAX_SELECTED_LOGIT_READS,
        "cells": cells,
        "refits": refits,
        "global_ranking": [
            {
                "rank": rank,
                "cell_index": cell["cell_index"],
                "scheme": cell["scheme"],
                "method": cell["method"],
                "condition": cell["condition"],
                "layer": cell["layer"],
                "surface": cell["surface"],
                "C": cell["C"],
                "selected_q": cell["selected_q"],
                "tau": cell["tau"],
                "selection_key": cell["selection_key"],
                "oof_metrics": cell["oof_metrics"],
            }
            for rank, cell in enumerate(ranked, start=1)
        ],
        "selection_policy": "TRAIN grouped OOF only: " + "; ".join(TIE_BREAK_CHAIN),
        "validation_used_for_selection": False,
        "holdout_accessed": False,
        "tie_break_chain": list(TIE_BREAK_CHAIN),
    }
    return driver._jsonable(result)
