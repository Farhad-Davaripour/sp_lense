"""Narrow synthetic-only core harness for classifier_generalization_v2.

Approved core only: pair-average features, training-only centering + row L2,
honest metrics with undefined denominators, separated multiclass-argmax and
binary-SELF-gate metrics, deterministic pooled-OOF selection, pure group/schema
validation, frozen-threshold inference. Grouped-CV/fitting driver and
real-cache adapter are NOT implemented (see HARNESS_HANDOFF.md). No real data.
"""
from __future__ import annotations
import math
import numpy as np
MODEL_BINARY, MODEL_MULTICLASS = "binary", "multiclass"
GRID_C = (0.01, 0.1, 1.0, 10.0)
GRID_TAU = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 .. 0.95
PROB_TOL = 1e-6

def pair_average(x_a, x_b):
    a, b = np.asarray(x_a, float), np.asarray(x_b, float)
    if a.shape != b.shape: raise ValueError(f"pair shape mismatch: {a.shape} != {b.shape}")
    return 0.5 * (a + b)

def fit_center(x_train):
    """Mean over training rows only; validation rows never enter the center."""
    x = np.asarray(x_train, float)
    if x.ndim != 2 or x.shape[0] == 0: raise ValueError("x_train must be non-empty 2-D")
    return x.mean(axis=0)

def row_l2(x, eps=1e-12):
    x = np.asarray(x, float)
    if x.ndim != 2: raise ValueError("x must be 2-D")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)

def transform_features(x, center):
    """Training-fitted center, then per-row L2; never re-center on validation."""
    x, center = np.asarray(x, float), np.asarray(center, float)
    if x.ndim != 2 or x.shape[1] != center.shape[0]: raise ValueError("width mismatch")
    return row_l2(x - center)

def _ratio(num, den):
    """Undefined (None) whenever the denominator is zero.

    F1 = 2TP/(2TP+FP+FN) therefore stays 0.0 when TP == 0 but FP+FN > 0
    (a legitimate all-false-negative result), and is None only when all of
    TP, FP and FN are zero.
    """
    return None if den == 0 else float(num) / float(den)

def precision_score(tp, fp):
    return _ratio(tp, tp + fp)

def recall_score(tp, fn):
    return _ratio(tp, tp + fn)

def f1_score(tp, fp, fn):
    return _ratio(2 * tp, 2 * tp + fp + fn)

def _as_finite_float(value, name):
    """Reject NaN/inf (and non-numeric values) loudly."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number, got {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out

def validate_probabilities(probs, n_columns=None, name="probs", sum_tol=PROB_TOL):
    """Reject malformed SELF probabilities: shape, non-finite, out-of-range, non-simplex.

    Raises ValueError; never returns a silent sentinel count.
    """
    p = np.asarray(probs, float)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array, got shape {p.shape}")
    if n_columns is not None and p.shape[1] != int(n_columns):
        raise ValueError(f"{name} column mismatch: {p.shape[1]} != {n_columns}")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} contains non-finite probabilities")
    if float(p.min()) < -sum_tol or float(p.max()) > 1.0 + sum_tol:
        raise ValueError(f"{name} outside [0, 1]")
    totals = p.sum(axis=1)
    if not np.all(np.abs(totals - 1.0) <= sum_tol):
        raise ValueError(f"{name} rows must sum to 1 within {sum_tol}")
    return p

def binary_counts(y_true, y_pred):
    t, p = _binary_values(y_true, "y_true"), _binary_values(y_pred, "y_pred")
    if t.shape != p.shape: raise ValueError("y_true/y_pred shape mismatch")
    return (int(np.sum(t & p)), int(np.sum(~t & ~p)),
            int(np.sum(~t & p)), int(np.sum(t & ~p)))

def binary_metrics(y_true, y_pred):
    tp, tn, fp, fn = binary_counts(y_true, y_pred)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision_score(tp, fp),
            "recall": recall_score(tp, fn), "f1": f1_score(tp, fp, fn)}

def confusion_matrix(y_true, y_pred, labels):
    labels = list(labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("labels must be non-empty and unique")
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    if truth.ndim != 1 or truth.shape != pred.shape or truth.size == 0:
        raise ValueError("label arrays must be non-empty, 1-D and equal length")
    idx = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(truth, pred):
        if t not in idx or p not in idx:
            raise ValueError(f"label outside declared labels: {t!r}/{p!r}")
        m[idx[t], idx[p]] += 1
    return m

def _class_defined(per_class):
    return {lab: all(v[key] is not None for key in ("precision", "recall", "f1"))
            for lab, v in per_class.items()}

def multiclass_metrics(y_true, y_pred, labels):
    """Argmax confusion plus per-class and fixed-denominator macro P/R/F1.

    Undefined classes are never dropped. Each macro uses all declared labels
    and is None if that metric has an undefined constituent or true support is
    missing. A class F1 of zero can be defined even when its precision is not.
    """
    labels = list(labels)
    if not labels: raise ValueError("labels must be non-empty")
    n = len(labels)
    m = confusion_matrix(y_true, y_pred, labels)
    per_class = {}
    for i, lab in enumerate(labels):
        tp = int(m[i, i])
        fp, fn = int(m[:, i].sum()) - tp, int(m[i, :].sum()) - tp
        per_class[lab] = {"precision": precision_score(tp, fp),
                          "recall": recall_score(tp, fn), "f1": f1_score(tp, fp, fn)}
    defined = _class_defined(per_class)
    present = {lab for lab in labels if int(m[labels.index(lab), :].sum()) > 0}
    undefined = [lab for lab in labels if not defined[lab]]
    absent = [lab for lab in labels if lab not in present]

    def macro(key):
        if absent or any(per_class[lab][key] is None for lab in labels): return None
        return _ratio(sum(per_class[lab][key] for lab in labels), n)
    return {"confusion": m, "per_class": per_class, "labels": labels,
            "defined_count": int(sum(defined[lab] for lab in present)),
            "present_count": len(present), "undefined_classes": undefined,
            "absent_classes": absent, "fully_defined": not undefined,
            "support": {lab: int(m[i, :].sum()) for i, lab in enumerate(labels)},
            "fully_covered": not absent, "macro_precision": macro("precision"),
            "macro_recall": macro("recall"), "macro_f1": macro("f1")}

def _binary_values(values, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a non-empty 1-D binary array")
    if not np.all(np.isfinite(raw)) or not np.all((raw == 0) | (raw == 1)):
        raise ValueError(f"{name} must contain only finite 0/1 values")
    return raw.astype(bool)

def _self_probabilities(p_self, tau):
    p = np.asarray(p_self, float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p_self must be non-empty and 1-D")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p_self must contain finite probabilities in [0, 1]")
    threshold = _as_finite_float(tau, "tau")
    if not 0 <= threshold <= 1:
        raise ValueError("tau must be in [0, 1]")
    return p, threshold

def binary_gate_metrics(y_true, p_self, tau):
    """SELF gate ON iff p_self >= tau; binary model's own metrics only.

    Bug-2 fix: non-finite/out-of-range probabilities and non-finite tau raise
    instead of silently becoming false negatives.
    """
    y = _binary_values(y_true, "y_true")
    p, tau = _self_probabilities(p_self, tau)
    if p.shape != y.shape: raise ValueError("p_self/y_true shape mismatch")
    return binary_metrics(y, p >= tau)

def argmax_labels(probs, class_order):
    classes = list(class_order)
    if not classes: raise ValueError("class_order must be non-empty")
    p = validate_probabilities(probs, n_columns=len(classes), name="probs")
    return [classes[i] for i in np.argmax(p, axis=1)]

def multiclass_argmax_metrics(y_true, probs, class_order):
    return multiclass_metrics(y_true, argmax_labels(probs, class_order), class_order)

def _coverage(candidates, labels):
    """Which declared labels actually occur in y_true; absent classes stay explicit."""
    support = {lab: 0 for lab in labels}
    for val in candidates:
        if val in support: support[val] += 1
    absent = [lab for lab in labels if support[lab] == 0]
    return {"support": support, "absent_classes": absent, "fully_covered": not absent}

def candidate_metrics(candidate):
    """Pooled development metrics for one candidate; no fitting, no tuning."""
    model = candidate.get("model")
    if model == MODEL_BINARY:
        metrics = binary_gate_metrics(candidate["y_true"], candidate["p_self"], candidate["tau"])
        cov = _coverage(np.asarray(candidate["y_true"]).astype(bool).ravel(), [False, True])
        return {**metrics, **cov}
    if model == MODEL_MULTICLASS:
        y_true, labels = candidate["y_true"], list(candidate["class_order"])
        if len(set(labels)) != len(labels) or "SELF" not in labels:
            raise ValueError("unique class_order must declare SELF")
        mc = multiclass_argmax_metrics(y_true, candidate["probs"], labels)
        cov = _coverage(y_true, labels)
        p_self = np.asarray(candidate["probs"], float)[:, labels.index("SELF")]
        gate = binary_gate_metrics(np.asarray(y_true) == "SELF", p_self, candidate["tau"])
        return {**gate, **cov, "multiclass_argmax": mc}
    raise ValueError(f"unknown model: {model!r}")

def selection_key(candidate, metrics):
    """max min(P,R); max F1; lower C; simpler binary; |tau-0.5|; smaller tau.

    Bug-2 fix: NaN/inf metrics are rejected rather than sorted unpredictably.
    """
    precision = _as_finite_float(metrics["precision"], "metrics['precision']")
    f1 = _as_finite_float(metrics["f1"], "metrics['f1']")
    recall = _as_finite_float(metrics["recall"], "metrics['recall']")
    tau = candidate.get("tau")
    tau_key = 0.0 if tau is None else _as_finite_float(tau, "candidate['tau']")
    return (-min(precision, recall), -f1, _as_finite_float(candidate["C"], "candidate['C']"),
            0 if candidate["model"] == MODEL_BINARY else 1,
            abs(tau_key - 0.5), tau_key)

def is_eligible(metrics):
    """Defined pooled P/R/F1 AND complete declared-class coverage are required.

    Bug-3 fix: a multiclass candidate whose declared labels include absent
    classes is not eligible, even if its macro average would look defined.
    """
    for key in ("precision", "recall", "f1"):
        try:
            value = _as_finite_float(metrics.get(key), key)
        except ValueError:
            return False
        if not 0 <= value <= 1: return False
    if "fully_defined" in metrics and not metrics["fully_defined"]: return False
    if "fully_covered" in metrics and not metrics["fully_covered"]: return False
    return True

def select_oof(candidates):
    rows = []
    for cand in candidates:
        metrics = candidate_metrics(cand)
        eligible = is_eligible(metrics)
        rows.append({"candidate": cand, "metrics": metrics, "eligible": eligible,
                     "key": selection_key(cand, metrics) if eligible else None})
    ranked = sorted((r for r in rows if r["eligible"]), key=lambda r: r["key"])
    return {"selected": ranked[0]["candidate"] if ranked else None,
            "ranking": ranked, "all": rows}

def validate_schema(obj, required_fields, name="record"):
    if not isinstance(obj, dict): raise TypeError(f"{name} must be a mapping")
    missing = [f for f in required_fields if f not in obj]
    if missing: raise ValueError(f"{name} missing required fields: {missing}")
    return True

def find_group_overlap(group_ids_a, group_ids_b):
    return set(map(str, group_ids_a)) & set(map(str, group_ids_b))

def assert_disjoint_groups(group_ids_a, group_ids_b, name_a="fold_a", name_b="fold_b"):
    overlap = find_group_overlap(group_ids_a, group_ids_b)
    if overlap: raise ValueError(f"group leakage {name_a}/{name_b}: {sorted(overlap)}")
    return True

def assert_disjoint_variants(variant_keys_a, variant_keys_b, name="split"):
    overlap = find_group_overlap(variant_keys_a, variant_keys_b)
    if overlap: raise ValueError(f"variant leakage across {name}: {sorted(overlap)}")
    return True

def assert_pairs_together(pairs, group_of, name="pair"):
    split = []
    for a, b in pairs:
        ga, gb = group_of.get(a), group_of.get(b)
        if ga is None or gb is None: raise ValueError(f"unmapped case in {name}: {a!r}/{b!r}")
        if ga != gb: split.append((a, b))
    if split: raise ValueError(f"{name} variants split across groups: {split}")
    return True

def predict_frozen(p_self, tau):
    p, threshold = _self_probabilities(p_self, tau)
    return p >= threshold

def predict_frozen_multiclass(probs, class_order):
    return argmax_labels(probs, class_order)

def evaluate_fold(y_true, y_pred, converged=True, coef=None):
    """A fold without true positives has undefined recall, not a bad fit.

    A legitimate no-positive-prediction fold keeps its false negatives (F1 0.0)
    and stays valid. Only nonconvergence, non-finite coef, or malformed/non-
    finite predictions mark the fold invalid.
    """
    valid = bool(converged)
    if coef is not None and not np.all(np.isfinite(np.asarray(coef, float))): valid = False
    t = _binary_values(y_true, "y_true")
    try:
        pred = _binary_values(y_pred, "y_pred")
        if pred.shape != t.shape: raise ValueError("prediction shape mismatch")
    except (TypeError, ValueError) as error:
        return {"valid": False, "n_pos": int(np.sum(t)), "error": str(error),
                **{key: None for key in ("tp", "tn", "fp", "fn", "precision", "recall", "f1")}}
    result = {"valid": valid, "n_pos": int(np.sum(t))}
    result.update(binary_metrics(t, pred))
    return result
