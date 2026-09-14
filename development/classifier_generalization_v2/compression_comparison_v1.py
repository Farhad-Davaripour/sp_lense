"""Compact model-free prompted-compression comparison core V1 (job compression_core_implementation_20260914_1437).

Compares unprompted and prompted normalized 3072-wide readouts at PCA8/16/32 and
(prompted only) full3072. ``run(conditions, reference_full, output_dir, ...)``
takes already-authenticated, aligned case metadata plus per-condition matrices from
the root loader; this module owns no loader, pin, watch, capture, provider,
tokenizer or model snapshot, and it never reads a real dataset/cache/holdout.

The old unprompted full-3072 baseline is accepted as a root-verified reference
(``reference_full``), cited in the report and admitted to the global ranking, and
never refit. PCA32 is fit with exact full SVD and ``whiten=False`` once per
condition per TRAIN fold and once per condition on full TRAIN (at most 12 shared
PCA fits), then sliced to 8/16/32 with no whitening and no validation fitting.
Fixed L2 logistic regression at C in {0.1,1,10}, five grouped folds, 19 thresholds
and both families give exactly 210 CV attempts plus at most 14 full-TRAIN refits
when every fold is valid. A failed/non-converged/non-finite fold voids the whole
candidate; folds are never pooled and failure evidence is kept. Selection is
TRAIN-OOF only; validation is reported, never selected on.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

import grouped_driver as driver
import harness
import span_classifier_driver_v1 as span

__all__ = [
    "CompressionError", "JOB_ID", "CONDITIONS", "CONDITION_DIMS", "PCA_DIMS", "PCA_COMPONENTS",
    "FULL_DIM", "DIM_WIDTH", "FAMILIES", "C_VALUES", "TAUS", "FOLDS", "CLASS_ORDER",
    "CV_FIT_LIMIT", "REFIT_LIMIT", "PCA_FIT_LIMIT", "FEATURE_DEFINITION", "PCA32", "run",
]

JOB_ID = "compression_core_implementation_20260914_1437"
CONDITIONS = ("unprompted", "prompted")
CONDITION_DIMS = {
    "unprompted": ("pca8", "pca16", "pca32"),
    "prompted": ("pca8", "pca16", "pca32", "full3072"),
}
PCA_DIMS = (8, 16, 32)
PCA_COMPONENTS = 32
FULL_DIM = 3072
DIM_WIDTH = {"pca8": 8, "pca16": 16, "pca32": 32, "full3072": FULL_DIM}
FOLDS = tuple(span.FOLDS)
FAMILIES = ("binary", "fourclass")
C_VALUES = (0.1, 1.0, 10.0)
TAUS = tuple(span.TAUS)
CLASS_ORDER = tuple(span.CLASS_ORDER)
CV_FIT_LIMIT = 210
REFIT_LIMIT = 14
PCA_FIT_LIMIT = 12
REQUIRED_META = ("case_order", "labels", "groups", "folds", "train_ids", "validation_ids",
                 "original_ids", "added_ids")
SPLITS = (("train", "train_ids"), ("original40", "original_ids"), ("added40", "added_ids"),
          ("combined80", "validation_ids"))

FEATURE_DEFINITION = {
    "schema": "compression_comparison_feature_definition.v1",
    "job_id": JOB_ID,
    "conditions": {condition: list(CONDITION_DIMS[condition]) for condition in CONDITIONS},
    "full_dim": FULL_DIM,
    "pca": {
        "components": PCA_COMPONENTS, "svd_solver": "full", "whiten": False,
        "fit_scope": "TRAIN rows only, once per condition per grouped fold and once per condition on full TRAIN",
        "slicing": "first 8/16/32 components of the same 32-component fit",
    },
    "unprompted_full": "root-verified old full baseline reused as a cited reference; never refit",
    "family": "fixed L2 logistic regression", "C": list(C_VALUES), "folds": list(FOLDS), "taus": list(TAUS),
}


class CompressionError(ValueError):
    """Structured rejection raised by this module."""


def _need(condition, code, detail=None):
    if not condition:
        raise CompressionError(code if detail is None else "%s: %s" % (code, detail))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _short(exc):
    return "%s: %s" % (type(exc).__name__, str(exc)[:240])


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(driver._jsonable(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def _write_bytes(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)


def _model_name(family):
    return harness.MODEL_BINARY if family == "binary" else harness.MODEL_MULTICLASS


# --------------------------------------------------------------------------- #
# PCA32: exact full SVD, whiten=False, fit on TRAIN rows only
# --------------------------------------------------------------------------- #
class PCA32:
    """32-component exact full-SVD PCA without whitening; slice at transform time."""

    svd_solver, whiten = "full", False

    def __init__(self, matrix):
        x = np.asarray(matrix, float)
        _need(x.ndim == 2 and x.shape[0] > 0 and x.shape[1] >= PCA_COMPONENTS, "PCA_INPUT_SHAPE", str(x.shape))
        _need(bool(np.isfinite(x).all()), "PCA_INPUT_FINITE")
        components = min(PCA_COMPONENTS, x.shape[0], x.shape[1])
        _need(components == PCA_COMPONENTS, "PCA_COMPONENTS_INSUFFICIENT", str(components))
        from sklearn.decomposition import PCA  # imported lazily; module import stays model-free

        model = PCA(n_components=components, svd_solver="full", whiten=False)
        model.fit(x)
        self.n_components_ = int(components)
        self.mean_ = np.asarray(model.mean_, float)
        self.components_ = np.asarray(model.components_, float)
        self.explained_variance_ = np.asarray(model.explained_variance_, float)
        self.explained_variance_ratio_ = np.asarray(model.explained_variance_ratio_, float)

    def transform(self, matrix, dim):
        x = np.asarray(matrix, float)
        _need(int(dim) in PCA_DIMS, "PCA_DIM", str(dim))
        return np.ascontiguousarray((x - self.mean_) @ self.components_[:int(dim)].T)

    def state(self):
        return {"svd_solver": self.svd_solver, "whiten": self.whiten, "n_components": self.n_components_,
                "mean_": self.mean_, "components_": self.components_,
                "explained_variance_": self.explained_variance_,
                "explained_variance_ratio_": self.explained_variance_ratio_}


def _configured(estimator):
    """Estimator fit quality: finite coefficients and a converging lbfgs run."""
    coefficient = getattr(estimator, "coef_", None)
    if coefficient is not None and not bool(np.isfinite(np.asarray(coefficient, float)).all()):
        return "non-finite coefficients"
    n_iter, max_iter = getattr(estimator, "n_iter_", None), getattr(estimator, "max_iter", None)
    if n_iter is not None and max_iter is not None:
        try:
            return None if int(np.max(np.asarray(n_iter, float))) < int(max_iter) else "estimator did not converge"
        except (TypeError, ValueError):
            return "malformed n_iter_"
    return None


def _row_min_metrics(metrics):
    return -min(float(metrics["precision"]), float(metrics["recall"]))


def _cell_key(candidate, tau, metrics):
    return (_row_min_metrics(metrics), -float(metrics["f1"]), float(candidate["C"]),
            0 if candidate["family"] == "binary" else 1, abs(float(tau) - 0.5), float(tau))


def _global_key(entry):
    metrics = entry["oof_metrics"]
    return (_row_min_metrics(metrics), -float(metrics["f1"]), int(entry["width"]), float(entry["C"]),
            0 if entry["family"] == "binary" else 1, abs(float(entry["tau"]) - 0.5), float(entry["tau"]),
            0 if entry["condition"] == "unprompted" else 1)


# --------------------------------------------------------------------------- #
# input validation (metadata is supplied by root; no loader lives here)
# --------------------------------------------------------------------------- #
def _validate_conditions(conditions):
    _need(type(conditions) is dict, "CONDITIONS_SCHEMA")
    meta = conditions.get("metadata")
    _need(type(meta) is dict, "CONDITIONS_METADATA")
    for key in REQUIRED_META:
        _need(key in meta, "CONDITIONS_METADATA_FIELD", key)
    case_order = list(meta["case_order"])
    _need(bool(case_order) and all(type(case_id) is str for case_id in case_order), "CASE_ORDER")
    _need(len(set(case_order)) == len(case_order), "CASE_ORDER_DUPLICATE")
    ids = {}
    for key in ("train_ids", "validation_ids", "original_ids", "added_ids"):
        value = list(meta[key])
        _need(all(type(case_id) is str for case_id in value), "CASE_IDS", key)
        ids[key] = value
    _need(set(ids["validation_ids"]) == set(ids["original_ids"]) | set(ids["added_ids"]), "VALIDATION_PARTITION")
    _need(not (set(ids["original_ids"]) & set(ids["added_ids"])), "VALIDATION_OVERLAP")
    _need(not (set(ids["train_ids"]) & set(ids["validation_ids"])), "SPLIT_OVERLAP")
    _need(set(ids["train_ids"]) | set(ids["validation_ids"]) == set(case_order), "CASE_COVERAGE")
    labels, groups, folds = dict(meta["labels"]), dict(meta["groups"]), dict(meta["folds"])
    for case_id in case_order:
        _need(labels.get(case_id) in CLASS_ORDER, "CASE_LABEL", case_id)
        _need(type(groups.get(case_id)) is str and groups[case_id], "CASE_GROUP", case_id)
        fold = folds.get(case_id)
        if case_id in set(ids["train_ids"]):
            _need(type(fold) is int and fold in FOLDS, "TRAIN_FOLD", case_id)
        else:
            _need(fold is None, "VALIDATION_FOLD", case_id)
    group_fold = {}
    for case_id in ids["train_ids"]:
        known = group_fold.setdefault(groups[case_id], folds[case_id])
        _need(known == folds[case_id], "GROUP_MULTI_FOLD", groups[case_id])
    for held in FOLDS:
        fit_ids = [case_id for case_id in ids["train_ids"] if folds[case_id] != held]
        test_ids = [case_id for case_id in ids["train_ids"] if folds[case_id] == held]
        _need(bool(fit_ids) and bool(test_ids), "FOLD_EMPTY", str(held))
        _need({labels[case_id] for case_id in fit_ids} == set(CLASS_ORDER), "FOLD_CLASSES", str(held))
        _need(not ({groups[case_id] for case_id in fit_ids} & {groups[case_id] for case_id in test_ids}),
              "FOLD_GROUP_LEAK", str(held))
    matrices, ab, ba = {}, {}, {}
    for condition in CONDITIONS:
        value = conditions.get(condition)
        _need(type(value) is dict, "CONDITION_SCHEMA", condition)
        try:
            matrix = np.asarray(value.get("matrix"), float)
        except (TypeError, ValueError):
            raise CompressionError("CONDITION_MATRIX: %s" % condition) from None
        _need(matrix.ndim == 2 and matrix.shape == (len(case_order), FULL_DIM),
              "CONDITION_MATRIX_SHAPE", "%s %s" % (condition, matrix.shape))
        _need(bool(np.isfinite(matrix).all()), "CONDITION_MATRIX_FINITE", condition)
        for key in REQUIRED_META:
            if key in value:
                _need(value[key] == meta[key], "CONDITION_METADATA_MISMATCH", "%s %s" % (condition, key))
        matrices[condition] = np.ascontiguousarray(matrix)
        pair = [value.get("ab_matrix"), value.get("ba_matrix")]
        _need((pair[0] is None) == (pair[1] is None), "CONDITION_ORDER_PAIR", condition)
        for name, raw in zip(("ab_matrix", "ba_matrix"), pair):
            if raw is None:
                ab[condition] = ba[condition] = None
                continue
            ordered = np.asarray(raw, float)
            _need(ordered.ndim == 2 and ordered.shape == (len(ids["validation_ids"]), FULL_DIM),
                  "CONDITION_ORDER_SHAPE", "%s %s" % (condition, name))
            _need(bool(np.isfinite(ordered).all()), "CONDITION_ORDER_FINITE", "%s %s" % (condition, name))
            (ab if name == "ab_matrix" else ba)[condition] = np.ascontiguousarray(ordered)
    row_of = {case_id: row for row, case_id in enumerate(case_order)}
    return {"case_order": case_order, "labels": labels, "groups": groups, "folds": folds,
            "train_ids": ids["train_ids"], "validation_ids": ids["validation_ids"],
            "original_ids": ids["original_ids"], "added_ids": ids["added_ids"],
            "matrices": matrices, "ab": ab, "ba": ba, "row_of": row_of}


def _validate_reference(reference_full, case_data):
    """Recompute the cited OOF metrics from the supplied reference; never fit it."""
    _need(type(reference_full) is dict, "REFERENCE_SCHEMA")
    _need(reference_full.get("condition", "unprompted") == "unprompted", "REFERENCE_CONDITION")
    families = reference_full.get("families")
    _need(type(families) is dict and all(family in families for family in FAMILIES), "REFERENCE_FAMILIES")
    truth = np.asarray([1 if case_data["labels"][case_id] == "SELF" else 0 for case_id in case_data["train_ids"]])
    provenance = reference_full.get("provenance") or {}
    entries, detail = [], {}
    for family in FAMILIES:
        record = families[family]
        _need(type(record) is dict, "REFERENCE_FAMILY_SCHEMA", family)
        raw = record.get("oof_p_self")
        if raw is None:
            oof = reference_full.get("oof")
            _need(type(oof) is dict and family in oof, "REFERENCE_OOF", family)
            nested = oof[family]
            raw = nested.get("p_self") if type(nested) is dict else nested
        p_self = np.asarray(raw, float)
        _need(p_self.shape == (len(case_data["train_ids"]),), "REFERENCE_OOF_SHAPE", family)
        _need(bool(np.isfinite(p_self).all()) and float(p_self.min()) >= 0.0 and float(p_self.max()) <= 1.0,
              "REFERENCE_OOF_RANGE", family)
        _need(type(record.get("C")) in (int, float) and type(record.get("tau")) in (int, float),
              "REFERENCE_HYPERPARAM", family)
        metrics = harness.binary_gate_metrics(truth, p_self, float(record["tau"]))
        provided = record.get("oof_metrics")
        if type(provided) is dict:
            for key in ("precision", "recall", "f1"):
                if provided.get(key) is not None and metrics[key] is not None:
                    _need(abs(float(provided[key]) - float(metrics[key])) <= 1e-9,
                          "REFERENCE_METRICS_MISMATCH", "%s %s" % (family, key))
        entry = {"condition": "unprompted", "dimension": "full3072", "family": family,
                 "C": float(record["C"]), "tau": float(record["tau"]), "width": FULL_DIM,
                 "oof_metrics": metrics, "source": "REFERENCE_REUSED_NOT_REFIT", "refit": False,
                 "provenance": provenance, "citation": record.get("citation")}
        entry["selection_key"] = [float(value) for value in _global_key(entry)]
        entries.append(entry)
        detail[family] = {"C": entry["C"], "tau": entry["tau"], "oof_metrics": metrics,
                          "evaluation": record.get("evaluation"), "artifact": record.get("artifact"),
                          "artifact_sha256": record.get("artifact_sha256")}
    return entries, detail, provenance


# --------------------------------------------------------------------------- #
# grouped OOF over the two conditions and their representations
# --------------------------------------------------------------------------- #
def _finalize_and_rank(candidates, truth):
    for candidate in candidates.values():
        candidate["valid"] = (
            len(candidate["folds"]) == len(FOLDS)
            and not candidate["errors"]
            and bool(np.isfinite(candidate["oof_p_self"]).all())
            and (candidate["oof_probs"] is None or bool(np.isfinite(candidate["oof_probs"]).all()))
        )
        candidate["error"] = None if candidate["valid"] else "INVALID_CANDIDATE"
        candidate["thresholds"] = []
        candidate["selected_tau"] = candidate["oof_metrics"] = candidate["selection_key"] = None
        if not candidate["valid"]:
            continue
        best = None
        for tau in TAUS:
            metrics = harness.binary_gate_metrics(truth, candidate["oof_p_self"], tau)
            eligible = harness.is_eligible(metrics)
            candidate["thresholds"].append({"tau": float(tau), "precision": metrics["precision"],
                                            "recall": metrics["recall"], "f1": metrics["f1"],
                                            "eligible": eligible})
            if eligible:
                key = _cell_key(candidate, tau, metrics)
                if best is None or key < best[0]:
                    best = (key, float(tau), metrics)
        if best is None:
            candidate["error"] = "NO_ELIGIBLE_THRESHOLD"
            continue
        candidate["selected_tau"], candidate["oof_metrics"] = best[1], best[2]
        candidate["selection_key"] = [float(value) for value in best[0]]
    return sorted((candidate for candidate in candidates.values() if candidate["selected_tau"] is not None),
                  key=lambda candidate: candidate["selection_key"])


def _candidate_entry(candidate):
    entry = {"condition": candidate["condition"], "dimension": candidate["dimension"],
             "family": candidate["family"], "C": float(candidate["C"]), "tau": float(candidate["selected_tau"]),
             "width": int(candidate["width"]), "oof_metrics": candidate["oof_metrics"],
             "source": "NEW_FIT", "refit": False, "candidate_key": candidate["key"]}
    entry["selection_key"] = [float(value) for value in _global_key(entry)]
    return entry


def _cross_validate(case_data, counts, factory, check, pca_states):
    matrices, row_of = case_data["matrices"], case_data["row_of"]
    labels, groups, folds = case_data["labels"], case_data["groups"], case_data["folds"]
    train_ids = case_data["train_ids"]
    train_pos = {case_id: index for index, case_id in enumerate(train_ids)}
    truth = np.asarray([1 if labels[case_id] == "SELF" else 0 for case_id in train_ids])
    candidates, fold_records = {}, []
    for condition in CONDITIONS:
        matrix, dims = matrices[condition], CONDITION_DIMS[condition]
        for held in FOLDS:
            check()
            fit_ids = [case_id for case_id in train_ids if folds[case_id] != held]
            test_ids = [case_id for case_id in train_ids if folds[case_id] == held]
            fit_rows = [row_of[case_id] for case_id in fit_ids]
            test_rows = [row_of[case_id] for case_id in test_ids]
            fit_groups = {groups[case_id] for case_id in fit_ids}
            test_groups = {groups[case_id] for case_id in test_ids}
            _need(not (fit_groups & test_groups), "FOLD_GROUP_LEAK", "%s|%s" % (condition, held))
            check()
            counts["pca_fits"] += 1
            pca = PCA32(matrix[fit_rows])
            pca_states.append({"key": "%s__fold%d" % (condition, held), "condition": condition, "scope": "fold",
                               "fold": int(held), "n_train": len(fit_ids), "state": pca.state()})
            transformed = {"full3072": matrix}
            for dim in dims:
                if dim != "full3072":
                    transformed[dim] = pca.transform(matrix, DIM_WIDTH[dim])
            record = {"condition": condition, "fold": int(held), "n_train": len(fit_ids), "n_test": len(test_ids),
                      "train_groups": sorted(fit_groups), "test_groups": sorted(test_groups), "errors": {}}
            positions = [train_pos[case_id] for case_id in test_ids]
            for dim in dims:
                x_fit = transformed[dim][fit_rows]
                x_test = transformed[dim][test_rows]
                for family in FAMILIES:
                    for C in C_VALUES:
                        check()
                        key = "%s|%s|%s|%s" % (condition, dim, family, C)
                        candidate = candidates.get(key)
                        if candidate is None:
                            candidate = {
                                "key": key, "condition": condition, "dimension": dim, "family": family,
                                "C": float(C), "width": DIM_WIDTH[dim], "cv_fits": 0,
                                "oof_p_self": np.full(len(train_ids), np.nan), "folds": {}, "errors": {},
                                "oof_probs": np.full((len(train_ids), len(CLASS_ORDER)), np.nan)
                                if family == "fourclass" else None,
                            }
                            candidates[key] = candidate
                        counts["cv_fits"] += 1
                        candidate["cv_fits"] += 1
                        error = None
                        try:
                            estimator = factory(family, C)
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", ConvergenceWarning)
                                estimator.fit(x_fit, [labels[case_id] for case_id in fit_ids])
                            error = _configured(estimator)
                            if error is None:
                                p_self, probs = driver._fold_probabilities(
                                    _model_name(family), estimator, x_test, len(test_ids))
                                candidate["oof_p_self"][positions] = p_self
                                if probs is not None:
                                    candidate["oof_probs"][positions] = probs
                        except Exception as exc:  # a failed fold voids this whole candidate
                            error = _short(exc)
                        if error is None:
                            candidate["folds"][held] = {"fold": int(held), "error": None, "valid": True}
                        else:
                            counts["fit_errors"] += 1
                            candidate["folds"][held] = {"fold": int(held), "error": error, "valid": False}
                            candidate["errors"][str(held)] = error
                            record["errors"]["%s|%s|%s" % (dim, family, C)] = error
            fold_records.append(record)
    ranked = _finalize_and_rank(candidates, truth)
    return candidates, fold_records, ranked, truth


# --------------------------------------------------------------------------- #
# refits, disk reload proof and split evaluation
# --------------------------------------------------------------------------- #
def _transform_rows(dim, matrix, pca, rows):
    if dim == "full3072":
        return np.ascontiguousarray(matrix[rows])
    return np.ascontiguousarray(pca.transform(matrix, DIM_WIDTH[dim])[rows])


def _transform_all(dim, matrix, pca):
    return np.ascontiguousarray(matrix) if dim == "full3072" else pca.transform(matrix, DIM_WIDTH[dim])


def _evaluate_and_save(case_data, condition, dim, family, candidate, estimator, pca, output):
    matrix, row_of, labels = case_data["matrices"][condition], case_data["row_of"], case_data["labels"]
    tau = float(candidate["selected_tau"])
    name = "%s__%s__%s.pkl" % (condition, dim, family)
    bundle = pickle.dumps({
        "estimator": estimator, "condition": condition, "dimension": dim, "family": family,
        "C": float(candidate["C"]), "tau": tau, "pca": None if dim == "full3072" else pca.state(),
        "feature_kind": "full3072" if dim == "full3072" else "pca", "job_id": JOB_ID,
        "reference": False,
    }, protocol=5)
    _write_bytes(output / "models" / name, bundle)
    _need(_sha(bundle) == _sha((output / "models" / name).read_bytes()), "MODEL_FILE_CHANGED", name)
    restored = pickle.loads((output / "models" / name).read_bytes())
    evaluation = {}
    for split, key in SPLITS:
        split_ids = list(case_data[key])
        rows = [row_of[case_id] for case_id in split_ids]
        ordered_ab = ordered_ba = None
        if split == "combined80" and case_data["ab"][condition] is not None:
            ordered_ab = _transform_all(dim, case_data["ab"][condition], pca)
            ordered_ba = _transform_all(dim, case_data["ba"][condition], pca)
        evaluation[split] = span._evaluate_split(
            family, estimator, _transform_rows(dim, matrix, pca, rows), split_ids,
            np.asarray([labels[case_id] for case_id in split_ids]), tau,
            ab_matrix=ordered_ab, ba_matrix=ordered_ba)
    combined_rows = [row_of[case_id] for case_id in case_data["validation_ids"]]
    x_combined = _transform_rows(dim, matrix, pca, combined_rows)
    reference = np.asarray(estimator.predict_proba(x_combined), float)
    reloaded = np.asarray(restored["estimator"].predict_proba(x_combined), float)
    exact = bool(reference.shape == reloaded.shape and np.array_equal(reference, reloaded))
    if dim != "full3072":
        for field in ("mean_", "components_", "explained_variance_", "explained_variance_ratio_"):
            exact = exact and bool(np.array_equal(np.asarray(restored["pca"][field], float),
                                                  np.asarray(getattr(pca, field), float)))
    _need(exact, "RELOAD_MISMATCH", name)
    return {"condition": condition, "dimension": dim, "family": family, "status": "FITTED",
            "source": "NEW_FIT", "C": float(candidate["C"]), "tau": tau,
            "oof_metrics": candidate["oof_metrics"], "selection_key": candidate["selection_key"],
            "pca": None if dim == "full3072" else {
                "n_components": int(pca.n_components_),
                "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_]},
            "artifact": "models/" + name, "artifact_sha256": _sha(bundle), "artifact_bytes": len(bundle),
            "reload_exact": exact, "evaluation": evaluation}


def _refit(case_data, ranked, counts, factory, check, full_pca, output):
    """One full-TRAIN refit per (condition, dimension, family) family winner."""
    family_best = {}
    for candidate in ranked:
        key = (candidate["condition"], candidate["dimension"], candidate["family"])
        if key not in family_best or candidate["selection_key"] < family_best[key]["selection_key"]:
            family_best[key] = candidate
    results = {}
    for condition in CONDITIONS:
        matrix, pca = case_data["matrices"][condition], full_pca[condition]
        for dim in CONDITION_DIMS[condition]:
            for family in FAMILIES:
                candidate = family_best.get((condition, dim, family))
                if candidate is None:
                    continue
                check()
                counts["refits"] += 1
                _need(counts["refits"] <= REFIT_LIMIT, "REFIT_BUDGET", str(counts["refits"]))
                try:
                    estimator = factory(family, candidate["C"])
                    x_train = _transform_rows(dim, matrix, pca, [case_data["row_of"][case_id]
                                                                 for case_id in case_data["train_ids"]])
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        estimator.fit(x_train, [case_data["labels"][case_id] for case_id in case_data["train_ids"]])
                    error = _configured(estimator)
                    _need(error is None, "REFIT_NOT_CONVERGED", error or "")
                except Exception as exc:
                    counts["refit_errors"] += 1
                    raise CompressionError("REFIT_FAILED: " + _short(exc)) from exc
                results[(condition, dim, family)] = _evaluate_and_save(
                    case_data, condition, dim, family, candidate, estimator, pca, output)
    return family_best, results


# --------------------------------------------------------------------------- #
# exclusive artifacts
# --------------------------------------------------------------------------- #
def _write_pca_artifacts(output, pca_states):
    arrays = {}
    index = []
    for entry in pca_states:
        state, key = entry["state"], entry["key"]
        for field in ("mean_", "components_", "explained_variance_", "explained_variance_ratio_"):
            arrays["%s__%s" % (key, field.rstrip("_"))] = np.asarray(state[field], float)
        index.append({"key": key, "condition": entry["condition"], "scope": entry["scope"],
                      "fold": entry["fold"], "n_train": entry["n_train"], "n_components": state["n_components"],
                      "svd_solver": state["svd_solver"], "whiten": state["whiten"],
                      "explained_variance": [float(value) for value in state["explained_variance_"]],
                      "explained_variance_ratio": [float(value) for value in state["explained_variance_ratio_"]],
                      "mean_sha256": _sha(np.asarray(state["mean_"], float).tobytes()),
                      "components_sha256": _sha(np.asarray(state["components_"], float).tobytes())})
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _write_bytes(output / "pca_states.npz", buffer.getvalue())
    _write_json(output / "pca_index.json", {"schema": "compression_comparison_pca_index.v1",
                                            "array_file": "pca_states.npz", "fits": index})


def _write_reports(output, case_data, counts, candidates, fold_records, ranked, cell_winners,
                   family_best, family_results, reference_detail, provenance, result):
    _json = _write_json
    _json(output / "feature_definition.json", FEATURE_DEFINITION)
    _json(output / "source_provenance.json", {
        "job_id": JOB_ID, "provided": provenance,
        "reference_provenance": result["reference"]["provenance"],
        "conditions": {condition: {"rows": int(case_data["matrices"][condition].shape[0]),
                                   "columns": FULL_DIM,
                                   "matrix_sha256": _sha(case_data["matrices"][condition].tobytes())}
                       for condition in CONDITIONS},
        "note": "metadata, matrices and reference results supplied by root; core does no loader/pin/capture work",
        "holdout_accessed": False})
    ordered = sorted(candidates.values(), key=lambda item: (item["condition"], item["dimension"],
                                                            item["family"], item["C"]))
    _json(output / "cv_scores.json", {
        "job_id": JOB_ID, "folds": fold_records, "oof_case_ids": list(case_data["train_ids"]),
        "oof_truth": [case_data["labels"][case_id] for case_id in case_data["train_ids"]],
        "candidates": [{"condition": entry["condition"], "dimension": entry["dimension"],
                        "family": entry["family"], "C": entry["C"], "width": entry["width"],
                        "valid": entry["valid"], "error": entry["error"], "errors": entry["errors"],
                        "folds": [entry["folds"][held] for held in FOLDS], "cv_fits": entry["cv_fits"],
                        "selected_tau": entry["selected_tau"], "thresholds": entry["thresholds"],
                        "selection_key": entry["selection_key"], "oof_metrics": entry["oof_metrics"],
                        "oof_p_self": [None if not np.isfinite(value) else float(value)
                                       for value in entry["oof_p_self"]]} for entry in ordered],
        "family_best": {"%s|%s|%s" % key: {"C": entry["C"], "tau": entry["selected_tau"],
                                           "selection_key": entry["selection_key"], "oof_metrics": entry["oof_metrics"]}
                        for key, entry in sorted(family_best.items())},
        "cell_winners": {"%s|%s" % key: {"condition": entry["condition"], "dimension": entry["dimension"],
                                         "family": entry["family"], "C": entry["C"], "tau": entry["selected_tau"],
                                         "selection_key": entry["selection_key"], "oof_metrics": entry["oof_metrics"]}
                         for key, entry in sorted(cell_winners.items())},
        "train_ranking": [{"condition": entry["condition"], "dimension": entry["dimension"],
                           "family": entry["family"], "C": entry["C"], "tau": entry["tau"],
                           "selection_key": entry["selection_key"]} for entry in ranked],
        "selection_policy": result["selection_policy"]})
    _json(output / "candidate_failures.json", {
        "job_id": JOB_ID, "pooled_valid_folds": False,
        "invalid_candidates": [{"key": entry["key"], "condition": entry["condition"],
                                "dimension": entry["dimension"], "family": entry["family"], "C": entry["C"],
                                "error": entry["error"], "errors": entry["errors"]}
                               for entry in ordered if not entry["valid"] or entry["selected_tau"] is None],
        "fit_errors": counts["fit_errors"], "refit_errors": counts["refit_errors"]})
    _json(output / "reference_full.json", {
        "cited": True, "refit": False, "condition": "unprompted", "dimension": "full3072",
        "provenance": provenance, "families": reference_detail,
        "note": "root-verified old full baseline accepted as reference and placed in the global ranking without fitting"})
    _json(output / "global_ranking.json", {
        "job_id": JOB_ID, "reference_included": True, "reference_fitted": False,
        "selection_policy": result["selection_policy"],
        "entries": [dict(entry, rank=index) for index, entry in enumerate(ranked, start=1)],
        "winner": dict(ranked[0], rank=1) if ranked else None})
    _json(output / "family_results.json", {
        "job_id": JOB_ID, "families": ["%s|%s|%s" % key for key in sorted(family_results)],
        "detail_files": {"%s|%s|%s" % key: "model_results_%s__%s__%s.json" % key for key in sorted(family_results)},
        "reference_included": True, "validation_used_for_selection": False, "holdout_accessed": False})
    for key, entry in sorted(family_results.items()):
        _json(output / ("model_results_%s__%s__%s.json" % key), entry)
    _json(output / "development_results.json", result)


def _nonfinite_count(value):
    if isinstance(value, dict):
        return sum(_nonfinite_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nonfinite_count(item) for item in value)
    if isinstance(value, np.ndarray):
        try:
            return int((~np.isfinite(np.asarray(value, float))).sum())
        except (TypeError, ValueError):
            return 0
    if isinstance(value, (float, np.floating)):
        return 0 if math.isfinite(float(value)) else 1
    return 0


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def _run(conditions, reference_full, output, counts, *, factory, deadline, provenance, started):
    case_data = _validate_conditions(conditions)
    reference_entries, reference_detail, reference_provenance = _validate_reference(reference_full, case_data)

    def check():
        _need(deadline is None or not deadline(), "DEADLINE")

    check()
    output.mkdir(parents=True)
    (output / "models").mkdir()
    supplied = dict(conditions.get("provenance") or {})
    supplied.update(provenance or {})
    owners = factory if factory is not None else driver._default_factory
    pca_states = []
    candidates, fold_records, ranked_candidates, _truth = _cross_validate(case_data, counts, owners, check, pca_states)
    full_pca = {}
    for condition in CONDITIONS:
        check()
        counts["pca_fits"] += 1
        pca = PCA32(case_data["matrices"][condition][[case_data["row_of"][case_id]
                                                      for case_id in case_data["train_ids"]]])
        full_pca[condition] = pca
        pca_states.append({"key": "%s__full" % condition, "condition": condition, "scope": "full_train",
                           "fold": None, "n_train": len(case_data["train_ids"]), "state": pca.state()})
    _need(counts["pca_fits"] <= PCA_FIT_LIMIT, "PCA_FIT_BUDGET", str(counts["pca_fits"]))
    _need(counts["cv_fits"] <= CV_FIT_LIMIT, "CV_FIT_BUDGET", str(counts["cv_fits"]))
    cell_winners = {}
    for candidate in ranked_candidates:
        cell_key = (candidate["condition"], candidate["dimension"])
        if cell_key not in cell_winners or _global_key(_candidate_entry(candidate)) < \
                _global_key(_candidate_entry(cell_winners[cell_key])):
            cell_winners[cell_key] = candidate
    family_best, family_results = _refit(case_data, ranked_candidates, counts, owners, check, full_pca, output)
    entries = [_candidate_entry(winner) for winner in cell_winners.values()] + reference_entries
    ranked = sorted(entries, key=_global_key)
    cells_report = {}
    for condition in CONDITIONS:
        for dim in CONDITION_DIMS[condition]:
            key = "%s|%s" % (condition, dim)
            winner = cell_winners.get((condition, dim))
            cells_report[key] = {"source": "NEW_FIT", "family": winner["family"], "C": float(winner["C"]),
                                 "tau": float(winner["selected_tau"]), "width": DIM_WIDTH[dim],
                                 "oof_metrics": winner["oof_metrics"]} if winner is not None else {
                "source": "NO_VALID_CANDIDATE"}
        if condition == "unprompted":
            cells_report["unprompted|full3072"] = {"source": "REFERENCE_REUSED_NOT_REFIT",
                                                   "families": list(FAMILIES), "width": FULL_DIM}
    counts["cells"] = len(cells_report)
    counts["reference_entries"] = len(reference_entries)
    counts["reference_fits"] = 0
    result = {
        "status": "COMPLETE" if ranked else "NO_ELIGIBLE_CANDIDATE",
        "job_id": JOB_ID,
        "counters": dict(counts, seconds=float(time.monotonic() - started)),
        "limits": {"cv_fits": CV_FIT_LIMIT, "refits": REFIT_LIMIT, "pca_fits": PCA_FIT_LIMIT},
        "data_counts": {"train": len(case_data["train_ids"]), "validation": len(case_data["validation_ids"]),
                        "original40": len(case_data["original_ids"]), "added40": len(case_data["added_ids"])},
        "conditions": list(CONDITIONS),
        "cells": cells_report,
        "reference": {"cited": True, "refit": False, "fitted": False, "provenance": reference_provenance,
                      "families": reference_detail},
        "global_ranking": ranked,
        "global_winner": ranked[0] if ranked else None,
        "selection_policy": ("TRAIN grouped OOF only: max min(precision,recall), max F1, lower dimension, "
                             "lower C, binary family tie, tau nearest 0.5, smaller tau, unprompted tie"),
        "validation_used_for_selection": False, "holdout_accessed": False,
    }
    nonfinite = _nonfinite_count({"family_results": family_results, "reference": reference_detail,
                                  "global_ranking": ranked})
    _need(nonfinite == 0, "NONFINITE_RESULT", str(nonfinite))
    result["finite_checks"] = {"nonfinite_values": 0, "model_results_checked": len(family_results),
                               "reference_checked": len(reference_entries), "passed": True}
    _write_pca_artifacts(output, pca_states)
    _write_reports(output, case_data, counts, candidates, fold_records, ranked, cell_winners, family_best,
                   family_results, reference_detail, supplied, result)
    check()
    return result


def run(conditions, reference_full, output_dir, *, factory=None, deadline=None, provenance=None):
    """Compare unprompted/prompted compression on authenticated aligned inputs.

    ``conditions`` carries ``{"metadata": {...}, "unprompted": {"matrix": ...}, "prompted": {...}}``.
    ``reference_full`` carries the root-verified unprompted full-3072 baseline; it is cited, not fit.
    """
    output = Path(output_dir)
    _need(not output.exists(), "OUTPUT_EXISTS", str(output))
    counts = {"cv_fits": 0, "refits": 0, "pca_fits": 0, "fit_errors": 0, "refit_errors": 0}
    started = time.monotonic()
    try:
        return _run(conditions, reference_full, output, counts, factory=factory, deadline=deadline,
                    provenance=provenance, started=started)
    except BaseException as exc:
        if output.exists():
            _write_json(output / "failure.json", {"status": "failed", "job_id": JOB_ID,
                                                  "code": _short(exc), "error_type": type(exc).__name__,
                                                  "counters": dict(counts, seconds=time.monotonic() - started),
                                                  "holdout_accessed": False})
        raise
