"""Grouped-CV fitting driver for classifier_generalization_v2 (code only).

Validates raw paired features/labels/groups/folds, then screens 2 model
families x 4 C values x 5 grouped folds (<= 40 fits). Training-only centering
and row-L2 come from harness; any failed / nonconvergent / non-finite-coef /
malformed-probability fold invalidates its whole candidate (surviving folds are
never pooled). Complete OOF candidates become 19 fixed-tau SELF-gate candidates
ranked by harness.select_oof. No real data, cache, holdout, or network access.
"""
from __future__ import annotations

import contextlib
import math
import os
import warnings

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

import harness

try:  # optional, limits BLAS/OpenMP pools at runtime
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

CANON = ("SELF", "OTHER", "NONTERMINATION", "ORDINARY")
FOLDS = (0, 1, 2, 3, 4)


def _default_factory(model, C):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, l1_ratio=0, solver="lbfgs", max_iter=1000,
                              tol=1e-4, class_weight=None, random_state=0)


def _as_1d(values, name, n):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size != n:
        raise ValueError(f"{name} must be 1-D with length {n}")
    return raw


def _validate(x, labels, groups, fold_ids, fit_factory):
    """Reject malformed input loudly, before any estimator is constructed."""
    if fit_factory is not None and not callable(fit_factory):
        raise TypeError("fit_factory must be callable or None")
    try:
        x = np.asarray(x, float)
    except (TypeError, ValueError):
        raise ValueError("x must be a finite 2-D numeric array") from None
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("x must be a non-empty 2-D array")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")
    n = x.shape[0]
    lab = [str(v) for v in _as_1d(labels, "labels", n).tolist()]
    if set(lab) != set(CANON):
        raise ValueError(f"labels must be exactly {CANON}")
    grp = _as_1d(groups, "groups", n).tolist()
    try:
        fold = np.asarray(_as_1d(fold_ids, "fold_ids", n), float)
    except (TypeError, ValueError):
        raise ValueError("fold_ids must be numeric") from None
    if not np.all(np.isfinite(fold)) or not np.all(fold == np.floor(fold)):
        raise ValueError("fold_ids must be finite integers")
    fold = fold.astype(int)
    if set(fold.tolist()) != set(FOLDS):
        raise ValueError(f"fold_ids must contain exactly {FOLDS}")
    seen = {}
    for g, f in zip(grp, fold.tolist()):
        if g in seen and seen[g] != f:
            raise ValueError(f"group {g!r} spans folds {seen[g]} and {f}")
        seen[g] = f
    for f in FOLDS:
        if {lab[i] for i in range(n) if fold[i] != f} != set(CANON):
            raise ValueError(f"training fold {f} lacks all four classes")
    return x, lab, grp, fold


def _positive_index(classes):
    for i, cls in enumerate(classes):
        if cls is True or cls == 1:
            return i
    raise ValueError("binary estimator does not expose positive class 1")


def _converged(est):
    n_iter, max_iter = getattr(est, "n_iter_", None), getattr(est, "max_iter", None)
    if n_iter is None or max_iter is None:
        return True
    try:
        return int(np.max(np.asarray(n_iter, float))) < int(max_iter)
    except (TypeError, ValueError):
        return True


def _fold_probabilities(model, est, x_te, n_test):
    classes = list(np.asarray(est.classes_).ravel().tolist())
    if len(set(classes)) != len(classes):
        raise ValueError("duplicate estimator classes")
    raw = np.asarray(est.predict_proba(x_te), float)
    if raw.ndim != 2 or raw.shape != (n_test, len(classes)):
        raise ValueError(f"malformed predict_proba shape {raw.shape}")
    harness.validate_probabilities(raw, n_columns=len(classes))
    if model == harness.MODEL_BINARY:
        if set(classes) != {0, 1}:
            raise ValueError("binary estimator classes must be exactly 0 and 1")
        return raw[:, _positive_index(classes)], None
    missing = [c for c in CANON if c not in classes]
    if missing:
        raise ValueError(f"multiclass estimator missing classes {missing}")
    probs = np.column_stack([raw[:, classes.index(c)] for c in CANON])
    return probs[:, CANON.index("SELF")], probs


def _run_fold(model, C, x, labels, fold_ids, held_out, factory, state):
    test_idx, train_idx = np.flatnonzero(fold_ids == held_out), np.flatnonzero(fold_ids != held_out)
    rec = {"model": model, "C": float(C), "fold": int(held_out), "n_train": int(train_idx.size),
           "n_test": int(test_idx.size), "valid": False, "error": None, "metrics": None,
           "p_self": None, "probs": None}
    try:
        center = harness.fit_center(x[train_idx])
        x_tr = harness.transform_features(x[train_idx], center)
        x_te = harness.transform_features(x[test_idx], center)
    except (TypeError, ValueError) as exc:
        rec["error"] = f"feature transform failed: {exc}"
        return rec
    y_test = [labels[i] for i in test_idx.tolist()]
    if model == harness.MODEL_BINARY:
        target = (np.asarray([labels[i] for i in train_idx.tolist()]) == "SELF").astype(int)
    else:
        target = [labels[i] for i in train_idx.tolist()]
    try:
        est = factory(model, C)
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            state["fits"] += 1
            est.fit(x_tr, target)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            rec["error"] = "fit emitted ConvergenceWarning"
            return rec
    except Exception as exc:  # any fit failure invalidates this fold
        rec["error"] = f"fit failed: {type(exc).__name__}: {exc}"
        return rec
    coef = getattr(est, "coef_", None)
    if coef is not None and not np.all(np.isfinite(np.asarray(coef, float))):
        rec["error"] = "non-finite coefficients"
        return rec
    if not _converged(est):
        rec["error"] = "estimator did not converge"
        return rec
    try:
        p_self, probs = _fold_probabilities(model, est, x_te, test_idx.size)
    except (TypeError, ValueError) as exc:
        rec["error"] = str(exc)
        return rec
    rec["p_self"], rec["probs"] = p_self, probs
    if model == harness.MODEL_BINARY:
        y_bin = np.asarray([1 if s == "SELF" else 0 for s in y_test])
        metrics = harness.evaluate_fold(y_bin, (p_self >= 0.5).astype(int), converged=True, coef=coef)
        rec["valid"] = bool(metrics.get("valid", True))
        if not rec["valid"]:
            rec["error"] = metrics.get("error", "malformed predictions")
    else:
        metrics = harness.multiclass_metrics(y_test, harness.argmax_labels(probs, CANON), CANON)
        rec["valid"] = True
    rec["metrics"] = metrics
    return rec


def _assemble_oof(model, folds, fold_ids, n):
    p_self = np.full(n, np.nan)
    probs = None if model == harness.MODEL_BINARY else np.full((n, len(CANON)), np.nan)
    for rec in folds:
        idx = np.flatnonzero(fold_ids == rec["fold"])
        p_self[idx] = rec["p_self"]
        if probs is not None:
            probs[idx] = rec["probs"]
    return {"p_self": p_self, "probs": probs}


def _select(oof_candidates, labels):
    entries = []
    for cand in oof_candidates:
        p_self, probs = cand["oof"]["p_self"], cand["oof"]["probs"]
        for tau in harness.GRID_TAU:
            entry = {"model": cand["model"], "C": cand["C"], "tau": float(tau),
                     "p_self": p_self.tolist()}
            if cand["model"] == harness.MODEL_BINARY:
                entry["y_true"] = [1 if s == "SELF" else 0 for s in labels]
            else:
                entry.update({"y_true": list(labels), "class_order": list(CANON),
                              "probs": probs.tolist()})
            entries.append(entry)
    result = harness.select_oof(entries)
    ranking = [{"model": r["candidate"]["model"], "C": r["candidate"]["C"],
                "tau": r["candidate"]["tau"], "eligible": r["eligible"],
                "key": _jsonable(r["key"]), "metrics": _jsonable(r["metrics"])}
               for r in result["ranking"]]
    selected = result["selected"]
    top = {"model": selected["model"], "C": selected["C"], "tau": selected["tau"],
           "metrics": ranking[0]["metrics"]} if selected is not None else None
    return {"selected": top, "ranking": ranking,
            "n_candidates": len(entries), "n_eligible": len(ranking)}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_grouped_screen(x, labels, groups, fold_ids, fit_factory=None):
    """Screen binary/multiclass x four C values over five grouped folds."""
    x, labels, groups, fold_ids = _validate(x, labels, groups, fold_ids, fit_factory)
    factory = fit_factory if fit_factory is not None else _default_factory
    state, fold_results, candidates, complete_oof = {"fits": 0}, [], [], []
    ctx = threadpool_limits(limits=1) if threadpool_limits is not None else contextlib.nullcontext()
    with ctx:
        for model in (harness.MODEL_BINARY, harness.MODEL_MULTICLASS):
            for C in harness.GRID_C:
                folds = [_run_fold(model, C, x, labels, fold_ids, f, factory, state) for f in FOLDS]
                fold_results.extend(folds)
                valid = [r["fold"] for r in folds if r["valid"]]
                cand = {"model": model, "C": float(C), "complete": len(valid) == len(FOLDS),
                        "valid_folds": valid,
                        "missing_folds": [r["fold"] for r in folds if not r["valid"]],
                        "errors": {str(r["fold"]): r["error"] for r in folds if r["error"]},
                        "oof": None}
                if cand["complete"]:
                    cand["oof"] = _assemble_oof(model, folds, fold_ids, len(labels))
                    complete_oof.append(cand)
                candidates.append(cand)
        selection = _select(complete_oof, labels)
    return _jsonable({"fit_count": state["fits"], "fold_results": fold_results,
                      "candidate_results": candidates, "selection": selection})
