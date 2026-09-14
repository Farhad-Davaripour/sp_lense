"""Model-free three-layer linear span control V1 (job linear_span_implementation_20260914_1233).

Implements ``LINEAR_SPAN_CONTROL_PLAN_V1.md``. No torch/transformers or other native
provider, no capture, no model/tokenizer/data cache, no network/install/Git work. The
real estimator is built lazily by ``grouped_driver._default_factory`` only if the
default factory is called; tests inject a toy factory and never fit a real classifier.
``run(case_data, output_dir)`` takes already-authenticated case_data from
``span_classifier_driver_v1.load_case_data`` plus a fresh output directory; the binary
loader, source audits, runner and supervisor are untouched. Features are the last token
vector of blocks 6,10,18 of the per-case pair-averaged windows, each 1024 vector
L2-normalized independently (zero maps to zero), concatenated to 3072; no learned
transform and no PCA.

Binary SELF-vs-rest and four-class LogisticRegression at C in {0.1,1,10} over exactly 5
grouped TRAIN folds = 30 CV attempts plus at most 2 full-TRAIN family refits; any
failed/nonconverged/non-finite fold invalidates its whole C/family candidate, folds are
never pooled and failure counters are preserved. Thresholds 0.05..0.95 step 0.05 are
chosen on TRAIN OOF by max min(precision,recall), max F1, lower C, binary-first family
tie, tau nearest 0.5, smaller tau; validation is reported, never selected on. Outputs
are exclusive. The root wrapper owns the prospective plan, source/runtime/input checks,
the external time watch and the 256MiB output cap.
"""
from __future__ import annotations

import hashlib, json, pickle, time, warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

import grouped_driver as driver
import harness
import span_classifier_driver_v1 as span

__all__ = ["LinearSpanError", "JOB_ID", "BLOCKS", "WIDTH", "FEATURE_DIM", "FOLDS", "FAMILIES", "C_VALUES",
           "TAUS", "CLASS_ORDER", "CV_FIT_LIMIT", "REFIT_LIMIT", "OUTPUT_CAP_BYTES", "FEATURE_DEFINITION",
           "features_for_case", "build_features", "run"]

JOB_ID = "linear_span_implementation_20260914_1233"
BLOCKS, WIDTH, FOLDS = (6, 10, 18), 1024, (0, 1, 2, 3, 4)
FAMILIES, C_VALUES, TAUS, CLASS_ORDER = ("binary", "fourclass"), (0.1, 1.0, 10.0), span.TAUS, span.CLASS_ORDER
FEATURE_DIM = len(BLOCKS) * WIDTH
CV_FIT_LIMIT = len(FAMILIES) * len(C_VALUES) * len(FOLDS)  # 30 grouped-OOF attempts
REFIT_LIMIT = len(FAMILIES)                               # 2 full-TRAIN family refits
OUTPUT_CAP_BYTES = 256 * 1024 * 1024
SPLITS = (("train", "train_ids"), ("original40", "original_ids"), ("added40", "added_ids"),
          ("combined80", "validation_ids"))
REQUIRED_KEYS = ("train_ids", "validation_ids", "original_ids", "added_ids", "folds", "labels", "windows", "groups")
FEATURE_DEFINITION = {
    "schema": "linear_span_feature_definition.v1", "blocks": list(BLOCKS), "token_position": "last",
    "vector_width": WIDTH, "normalization": "independent L2 per 1024 vector; zero vector maps to zero",
    "concatenation": "blocks in order 6,10,18 -> 3072 coordinates", "learned_transform": False, "pca": False,
    "windows": "pair_averaged AB/BA windows supplied per case by the existing span loader"}

class LinearSpanError(ValueError):
    """Structured rejection raised by this module."""

def _need(condition, code, detail=None):
    if not condition:
        raise LinearSpanError(code if detail is None else "%s: %s" % (code, detail))

def _sha(raw):
    return hashlib.sha256(raw).hexdigest()

def _short(exc):
    return "%s: %s" % (type(exc).__name__, str(exc)[:240])

def _json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(driver._jsonable(value), stream, indent=2, allow_nan=False)
        stream.write("\n")

# features: pair-averaged windows -> normalized last-vector concat
def features_for_case(window):
    """Pair-averaged window (3, L, 1024) -> 3072 coordinates, normalized per block."""
    array = np.asarray(window, dtype=np.float64)
    _need(array.ndim == 3 and array.shape[0] == len(BLOCKS) and 1 <= array.shape[1] <= 16 and array.shape[2] == WIDTH,
          "CASE_WINDOW_SHAPE", str(array.shape))
    _need(bool(np.isfinite(array).all()), "CASE_WINDOW_FINITE")
    vectors = []
    for index in range(len(BLOCKS)):
        vector = array[index, -1]
        norm = float(np.linalg.norm(vector))
        vectors.append(vector / norm if norm > 0.0 else np.zeros(WIDTH))
    return np.concatenate(vectors)

def build_features(case_data):
    """Stack case features in canonical case order; reject malformed input loudly."""
    _need(type(case_data) is dict and all(key in case_data for key in REQUIRED_KEYS), "CASE_DATA_FIELDS")
    order = list(case_data["train_ids"]) + list(case_data["validation_ids"])
    _need(len(set(order)) == len(order), "CASE_DATA_DUPLICATE")
    matrix = np.empty((len(order), FEATURE_DIM), dtype=np.float64)
    for row, case_id in enumerate(order):
        _need(case_id in case_data["windows"] and case_id in case_data["labels"], "CASE_DATA_ID", str(case_id))
        matrix[row] = features_for_case(case_data["windows"][case_id])
    _need(bool(np.isfinite(matrix).all()), "FEATURE_MATRIX_FINITE")
    return order, matrix

def matrix_for(matrix, row_of, case_ids):
    return np.ascontiguousarray(np.stack([matrix[row_of[case_id]] for case_id in case_ids]))

# grouped OOF
def _configured(estimator):
    """Estimator fit quality: finite coefficients and a converging lbfgs run."""
    coefficient, n_iter, max_iter = (getattr(estimator, name, None) for name in ("coef_", "n_iter_", "max_iter"))
    if coefficient is not None and not bool(np.isfinite(np.asarray(coefficient, float)).all()):
        return "non-finite coefficients"
    if n_iter is not None and max_iter is not None:
        try:
            return None if int(np.max(np.asarray(n_iter, float))) < int(max_iter) else "estimator did not converge"
        except (TypeError, ValueError):
            return "malformed n_iter_"
    return None

def _fold_oof(family, C, matrix, case_data, train_ids, rows, labels, factory, counters, deadline):
    """One family/C candidate over exactly five group-disjoint folds; never pools folds."""
    _need(set(case_data["folds"][case_id] for case_id in train_ids) == set(FOLDS), "TRAIN_FOLD_SET")
    model = harness.MODEL_BINARY if family == "binary" else harness.MODEL_MULTICLASS
    p_self = np.full(len(labels), np.nan)
    probs = None if family == "binary" else np.full((len(labels), len(CLASS_ORDER)), np.nan)
    errors, folds = {}, []
    for held in FOLDS:
        deadline()
        test_index = [i for i, case_id in enumerate(train_ids) if case_data["folds"][case_id] == held]
        train_index = [i for i, case_id in enumerate(train_ids) if case_data["folds"][case_id] != held]
        train_groups = {case_data["groups"][train_ids[i]] for i in train_index}
        test_groups = {case_data["groups"][train_ids[i]] for i in test_index}
        _need(not (train_groups & test_groups), "FOLD_GROUP_LEAK", str(held))
        record = {"fold": int(held), "n_train": len(train_index), "n_test": len(test_index),
                  "train_groups": sorted(train_groups), "test_groups": sorted(test_groups), "error": None}
        counters["cv_fits"] += 1
        try:
            estimator = factory(family, C)
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                estimator.fit(matrix[[rows[i] for i in train_index]],
                              [1 if labels[i] == "SELF" else 0 for i in train_index] if family == "binary"
                              else [labels[i] for i in train_index])
            error = _configured(estimator)
            if error is None:
                scores, probabilities = driver._fold_probabilities(
                    model, estimator, matrix[[rows[i] for i in test_index]], len(test_index))
                p_self[test_index] = scores
                if probs is not None:
                    probs[test_index] = probabilities
        except Exception as exc:  # any fit/probability failure invalidates this fold
            error = _short(exc)
        if error is not None:
            counters["fit_errors"] += 1
            record["error"] = error
            errors[str(held)] = error
        folds.append(record)
    valid = not errors and bool(np.isfinite(p_self).all()) and (probs is None or bool(np.isfinite(probs).all()))
    return {"family": family, "C": float(C), "valid": valid, "errors": errors, "folds": folds,
            "error": "INCOMPLETE_OR_FAILED_FOLD" if not valid else None, "p_self": p_self, "probs": probs}

def _selection_key(family, C, tau, metrics):
    return (-min(metrics["precision"], metrics["recall"]), -metrics["f1"], float(C),
            0 if family == "binary" else 1, abs(float(tau) - 0.5), float(tau))

def _rank_taus(candidate, truth):
    """TRAIN OOF threshold grid; a candidate missing any OOF fold is rejected whole."""
    if not candidate["valid"]:
        return
    thresholds, best = [], None
    for tau in TAUS:
        metrics = harness.binary_gate_metrics(truth, candidate["p_self"], tau)
        entry = {"tau": float(tau), "eligible": harness.is_eligible(metrics), "precision": metrics["precision"],
                 "recall": metrics["recall"], "f1": metrics["f1"]}
        thresholds.append(entry)
        if entry["eligible"]:
            key = _selection_key(candidate["family"], candidate["C"], tau, metrics)
            if best is None or key < best[0]:
                best = (key, float(tau), metrics)
    if best is not None:
        candidate["thresholds"], candidate["selected_tau"] = thresholds, best[1]
        candidate["oof_metrics"], candidate["selection_key"] = best[2], [float(v) for v in best[0]]

def _cross_validate(matrix, case_data, row_of, factory, counters, deadline):
    train_ids = list(case_data["train_ids"])
    truth = np.asarray([1 if case_data["labels"][case_id] == "SELF" else 0 for case_id in train_ids])
    rows = [row_of[case_id] for case_id in train_ids]
    labels = [case_data["labels"][case_id] for case_id in train_ids]
    candidates = []
    for family in FAMILIES:
        for C in C_VALUES:
            _need(counters["cv_fits"] < CV_FIT_LIMIT, "CV_FIT_BUDGET", str(counters["cv_fits"]))
            candidate = _fold_oof(family, C, matrix, case_data, train_ids, rows, labels, factory, counters, deadline)
            _rank_taus(candidate, truth)
            candidates.append(candidate)
    _need(counters["cv_fits"] <= CV_FIT_LIMIT, "CV_FIT_BUDGET", str(counters["cv_fits"]))
    return candidates, sorted([entry for entry in candidates if entry.get("selected_tau") is not None],
                              key=lambda entry: entry["selection_key"])

# evaluation, refits and exclusive artifacts
def _prepare(matrix, estimator, family, candidate, row_of, case_data, output):
    """Serialize, reload from disk, prove exact equality, then evaluate every split."""
    name, tau = "estimator_linear_%s.pkl" % family, float(candidate["selected_tau"])
    bundle = pickle.dumps({"estimator": estimator, "feature_definition": FEATURE_DEFINITION, "family": family,
                           "C": float(candidate["C"]), "tau": tau, "job_id": JOB_ID}, protocol=5)
    with (output / name).open("xb") as stream:
        stream.write(bundle)
    case_ids = list(case_data["validation_ids"])
    reference = np.asarray(estimator.predict_proba(matrix_for(matrix, row_of, case_ids)), float)
    saved = (output / name).read_bytes()
    _need(_sha(saved) == _sha(bundle), 'MODEL_FILE_CHANGED', name)
    reloaded = np.asarray(pickle.loads(saved)["estimator"].predict_proba(
        matrix_for(matrix, row_of, case_ids)), float)
    exact = bool(reference.shape == reloaded.shape and np.array_equal(reference, reloaded))
    _need(exact, "RELOAD_MISMATCH", name)
    evaluation = {}
    for split, key in SPLITS:
        split_ids = list(case_data[key])
        paired = {}
        if split == 'combined80' and 'ab_windows' in case_data and 'ba_windows' in case_data:
            paired = {order + '_matrix': np.stack([features_for_case(case_data[order + '_windows'][i])
                       for i in split_ids]) for order in ('ab', 'ba')}
        evaluation[split] = span._evaluate_split(
            family, estimator, matrix_for(matrix, row_of, split_ids), split_ids,
            np.asarray([case_data["labels"][case_id] for case_id in split_ids]), tau, **paired)
    return {"family": family, "status": "FITTED", "C": float(candidate["C"]), "selected_tau": tau,
            "oof_metrics": candidate["oof_metrics"], "selection_key": candidate["selection_key"],
            "artifact": name, "artifact_sha256": _sha(bundle), "artifact_bytes": len(bundle),
            "reload_exact": exact, "evaluation": evaluation}

def _refit(queue, selected, matrix, case_data, row_of, owners, counts, check, output):
    """One full-TRAIN refit per family in ranked order; failed refits stay evidence, not results."""
    family_results, refitted = {}, set()
    for candidate in queue:
        family = candidate["family"]
        if family in refitted:
            continue
        refitted.add(family)
        check()
        counts["refits"] += 1
        _need(counts["refits"] <= REFIT_LIMIT, "REFIT_BUDGET", str(counts["refits"]))
        target = np.asarray([case_data["labels"][case_id] for case_id in case_data["train_ids"]])
        try:
            estimator = owners(family, candidate["C"])
            train_matrix = matrix_for(matrix, row_of, case_data['train_ids'])
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                estimator.fit(train_matrix, (target == "SELF").astype(int) if family == "binary" else target)
            _need(_configured(estimator) is None, 'REFIT_NOT_CONVERGED')
        except Exception as exc:
            counts["refit_errors"] += 1
            raise LinearSpanError('REFIT_FAILED: ' + _short(exc)) from exc
        entry = _prepare(matrix, estimator, family, candidate, row_of, case_data, output)
        entry["train_selected"] = bool(candidate is selected)
        family_results[family] = entry
    return family_results

def _write_reports(output, candidates, ranked, case_data, result, family_results):
    _json(output / "feature_definition.json", FEATURE_DEFINITION)
    _json(output / "cv_scores.json", {
        "folds": [record for entry in candidates for record in entry["folds"]],
        "oof_case_ids": list(case_data["train_ids"]),
        "oof_truth": [case_data["labels"][case_id] for case_id in case_data["train_ids"]],
        "candidates": [{"family": e["family"], "C": e["C"], "valid": e["valid"], "error": e["error"],
                        "errors": e["errors"], "folds": e["folds"], "selected_tau": e.get("selected_tau"),
                        "thresholds": e.get("thresholds", []), "selection_key": e.get("selection_key"),
                        "oof_metrics": e.get("oof_metrics"),
                        "oof_p_self": [None if not np.isfinite(v) else float(v) for v in e["p_self"]]}
                       for e in candidates],
        "train_ranking": [{"family": e["family"], "C": e["C"], "tau": e["selected_tau"],
                           "selection_key": e["selection_key"]} for e in ranked],
        "train_selected": result["train_selected"]})
    for family, entry in family_results.items():
        _json(output / ("predictions_linear_%s.json" % family), {
            "family": family, "C": entry["C"], "tau": entry["selected_tau"],
            "splits": {name: {"self_gate": value["self_gate"],
                              "negative_class_false_positives": value["negative_class_false_positives"],
                              "four_class": value.get("four_class"),
                              "answer_order_consistency": value.get("answer_order_consistency"),
                              "predictions": value["predictions"]}
                       for name, value in entry["evaluation"].items()}})
        _json(output / ("family_%s.json" % family), entry)
    _json(output / "family_results.json", {
        "families": list(family_results),
        "family_detail_files": dict((family, "family_%s.json" % family) for family in family_results),
        "validation_used_for_selection": False, "holdout_accessed": False})
    _json(output / "development_results.json", result)

def _validate(case_data):
    _need(type(case_data) is dict, "CASE_DATA_SCHEMA")
    for key in ("folds", "labels"):
        _need(type(case_data.get(key)) is dict, "CASE_DATA_SCHEMA", key)
    for key in REQUIRED_KEYS:
        _need(key in case_data, "CASE_DATA_FIELDS", key)
    for key in ("train_ids", "validation_ids", "original_ids", "added_ids"):
        _need(type(case_data[key]) is list and case_data[key], "CASE_DATA_SCHEMA", key)

# entry point
def _run(case_data, output_dir, counts, *, factory=None, deadline=None):
    """Fit the linear span control on authenticated ``case_data`` into a fresh output directory."""
    output = Path(output_dir)
    _need(not output.exists(), "OUTPUT_EXISTS", str(output))
    _validate(case_data)
    case_data = dict(case_data)
    order, matrix = build_features(case_data)
    row_of = dict((case_id, row) for row, case_id in enumerate(order))
    output.mkdir(parents=True)
    started = time.monotonic()

    def check():
        _need(deadline is None or not deadline(), "DEADLINE")

    owners = factory if factory is not None else driver._default_factory
    candidates, ranked = _cross_validate(matrix, case_data, row_of, owners, counts, check)
    selected = ranked[0] if ranked else None
    queue = [entry for entry in ranked if entry["family"] == "binary"] + \
            [entry for entry in ranked if entry["family"] == "fourclass"]
    family_results = _refit(queue, selected, matrix, case_data, row_of, owners, counts, check, output)
    _need(bool(family_results), "NO_FITTED_FAMILY")
    result = {
        "status": "COMPLETE" if selected is not None else "NO_ELIGIBLE_TRAIN_CANDIDATE",
        "job_id": JOB_ID, "counters": dict(counts, seconds=float(time.monotonic() - started)),
        "limits": {"cv_fits": CV_FIT_LIMIT, "refits": REFIT_LIMIT, "output_bytes": OUTPUT_CAP_BYTES},
        "feature_definition": FEATURE_DEFINITION,
        "data_counts": {"train": len(case_data["train_ids"]), "validation_original": len(case_data["original_ids"]),
                        "validation_added": len(case_data["added_ids"]),
                        "validation_combined": len(case_data["validation_ids"])},
        "selection_policy": ("TRAIN grouped OOF: max min(precision,recall), max F1, lower C, binary-first "
                             "family tie, tau nearest 0.5, smaller tau"),
        "train_selected": None if selected is None else {
            "family": selected["family"], "C": selected["C"], "tau": selected["selected_tau"],
            "oof_metrics": selected["oof_metrics"], "selected_using": "TRAIN_GROUPED_OOF_ONLY"},
        "family_results": {family: {k: v for k, v in entry.items() if k != "evaluation"}
                           for family, entry in family_results.items()},
        "validation_used_for_selection": False, "holdout_accessed": False}
    _write_reports(output, candidates, ranked, case_data, result, family_results)
    check()
    return result


def run(case_data, output_dir, *, factory=None, deadline=None):
    """Preserve attempt counters on failure; never overwrite an old output."""
    output = Path(output_dir)
    _need(not output.exists(), 'OUTPUT_EXISTS', str(output))
    counts = {'cv_fits': 0, 'refits': 0, 'fit_errors': 0, 'refit_errors': 0}
    started = time.monotonic()
    try:
        return _run(case_data, output_dir, counts, factory=factory, deadline=deadline)
    except BaseException as exc:
        if output.exists():
            _json(output / 'failure.json', {'status': 'failed', 'error': _short(exc),
                'counters': dict(counts, seconds=time.monotonic()-started), 'holdout_accessed': False})
        raise
