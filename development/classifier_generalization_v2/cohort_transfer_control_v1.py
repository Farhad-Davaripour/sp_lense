"""Cohort-transfer control V1 (job cohort_transfer_implementation_20260914_1319).

Single model-free owner and minimal wrapper over the reviewed, unchanged
``linear_span_control_v1`` core. It derives two disjoint 120-case TRAIN arms
(30 per class, same 7 groups and identical group->fold map) from authenticated
``case_data['manifest']['cases'][cid]['cohort']`` -- never from case-ID spelling --
leaves the identical 40+40 validation evaluation untouched, calls the core exactly
once per arm (30 CV + <=2 refits each, <=64 classifier fits total), shares one
deadline, and enforces the 256 MiB aggregate output cap across ``<root>/original``
and ``<root>/added``.

Root disposition of review condition 5: the core ``JOB_ID`` is implementation
provenance and is NOT monkeypatched or otherwise overridden. This wrapper records
its own experiment/run identity, cohort role, parent plan hash and core artifact
hashes separately while preserving the core implementation ID. No dataset, cache,
model, provider, network, install, Git or coordination action is performed here.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import linear_span_control_v1 as core

__all__ = [
    "CohortTransferError", "EXPERIMENT_ID", "ARMS", "ARM_CASES", "PER_CLASS",
    "GROUP_COUNT", "FOLD_SET", "EVAL_ORIGINAL", "EVAL_ADDED", "OUTPUT_CAP_BYTES",
    "CORE_CV_LIMIT", "CORE_REFIT_LIMIT", "TOTAL_FIT_LIMIT", "FAMILIES", "SPLITS",
    "derive_arms", "run",
]

EXPERIMENT_ID = "cohort_transfer_implementation_20260914_1319"
ARMS = ("original", "added")
ARM_CASES, PER_CLASS, GROUP_COUNT = 120, 30, 7
FOLD_SET = (0, 1, 2, 3, 4)
EVAL_ORIGINAL, EVAL_ADDED = 40, 40
OUTPUT_CAP_BYTES = 256 * 1024 * 1024
CORE_CV_LIMIT, CORE_REFIT_LIMIT, TOTAL_FIT_LIMIT = 30, 2, 64
FAMILIES = ("binary", "fourclass")
SPLITS = (("original40", "original_ids"), ("added40", "added_ids"), ("combined80", "validation_ids"))


class CohortTransferError(ValueError):
    """Structured rejection raised by this wrapper."""


def _need(condition, code, detail=None):
    if not condition:
        raise CohortTransferError(code if detail is None else "%s: %s" % (code, detail))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _short(exc):
    return "%s: %s" % (type(exc).__name__, str(exc)[:240])


def _core_path():
    return Path(core.__file__).resolve()


def _ids_sha(case_ids):
    return _sha(json.dumps(list(case_ids), separators=(",", ":")).encode("utf-8"))


def _tree_bytes(path):
    root = Path(path)
    return sum(entry.stat().st_size for entry in root.rglob("*") if entry.is_file())


# --------------------------------------------------------------------------- #
# zero-fit subset derivation (cohort field, not case-ID spelling)
# --------------------------------------------------------------------------- #
def _cohort(manifest, case_id):
    entry = manifest["cases"].get(case_id)
    _need(type(entry) is dict, "MANIFEST_CASE", str(case_id))
    cohort = entry.get("cohort")
    _need(cohort in ARMS, "MANIFEST_COHORT", "%s:%s" % (case_id, cohort))
    return cohort


def _class_counts(case_data, case_ids):
    counts = {label: 0 for label in core.CLASS_ORDER}
    for case_id in case_ids:
        _need(case_id in case_data["labels"], "ARM_LABEL", str(case_id))
        counts[case_data["labels"][case_id]] += 1
    return counts


def _fold_map(case_data, case_ids):
    mapping = {}
    for case_id in case_ids:
        group, fold = case_data["groups"][case_id], case_data["folds"][case_id]
        _need(type(fold) is int and fold in FOLD_SET, "ARM_FOLD", "%s:%s" % (case_id, fold))
        if group in mapping:
            _need(mapping[group] == fold, "GROUP_MULTI_FOLD", str(group))
        mapping[group] = fold
    return mapping


def derive_arms(case_data):
    """Split authenticated TRAIN ids into class-balanced, group-aligned cohorts."""
    _need(type(case_data) is dict, "CASE_DATA_SCHEMA")
    for key in ("train_ids", "validation_ids", "original_ids", "added_ids", "labels", "groups", "folds", "manifest"):
        _need(key in case_data, "CASE_DATA_FIELDS", key)
    manifest = case_data["manifest"]
    _need(type(manifest) is dict and type(manifest.get("cases")) is dict, "MANIFEST_SCHEMA")
    train_ids = list(case_data["train_ids"])
    _need(len(set(train_ids)) == len(train_ids), "TRAIN_DUPLICATE")
    arms = {cohort: [case_id for case_id in train_ids if _cohort(manifest, case_id) == cohort]
            for cohort in ARMS}
    maps = {}
    for cohort in ARMS:
        case_ids = arms[cohort]
        _need(len(case_ids) == ARM_CASES and len(set(case_ids)) == ARM_CASES, "ARM_COUNT",
              "%s:%d" % (cohort, len(case_ids)))
        counts = _class_counts(case_data, case_ids)
        _need(set(counts) == set(core.CLASS_ORDER) and all(value == PER_CLASS for value in counts.values()),
              "ARM_CLASS_BALANCE", "%s:%s" % (cohort, counts))
        maps[cohort] = _fold_map(case_data, case_ids)
        _need(len(maps[cohort]) == GROUP_COUNT, "ARM_GROUPS", "%s:%d" % (cohort, len(maps[cohort])))
        _need(set(maps[cohort].values()) == set(FOLD_SET), "ARM_FOLDS", cohort)
    _need(set(maps["original"]) == set(maps["added"]), "ARM_GROUP_SET")
    _need(all(maps["original"][group] == maps["added"][group] for group in maps["original"]), "ARM_FOLD_MAP")
    original, added = set(arms["original"]), set(arms["added"])
    _need(not (original & added), "ARM_OVERLAP")
    _need(not ((original | added) & set(case_data["validation_ids"])), "EVAL_LEAK")
    _need(original | added == set(train_ids), "TRAIN_PARTITION")
    _need(len(case_data["original_ids"]) == EVAL_ORIGINAL and len(case_data["added_ids"]) == EVAL_ADDED,
          "EVAL_PRESERVED")
    _need(len(case_data["validation_ids"]) == EVAL_ORIGINAL + EVAL_ADDED, "EVAL_COMBINED")
    _need(set(case_data["validation_ids"]) == set(case_data["original_ids"]) | set(case_data["added_ids"]),
          "EVAL_PARTITION")
    return arms


# --------------------------------------------------------------------------- #
# per-arm budget and family-matched endpoint extraction
# --------------------------------------------------------------------------- #
def _check_arm(cohort, result, case_ids):
    _need(type(result) is dict, "RESULT_SCHEMA", cohort)
    _need(result.get('status') == 'COMPLETE', 'CORE_STATUS', cohort)
    _need(len(set(case_ids)) == ARM_CASES, "ARM_SIZE", cohort)
    counters = result.get("counters")
    _need(type(counters) is dict, "COUNTERS", cohort)
    _need(int(counters.get("cv_fits", -1)) == CORE_CV_LIMIT, "CORE_CV_BUDGET", str(counters.get("cv_fits")))
    _need(0 <= int(counters.get("refits", -1)) <= CORE_REFIT_LIMIT, "CORE_REFIT_BUDGET", str(counters.get("refits")))
    _need(int(counters.get("refit_errors", 0)) == 0, "REFIT_ERRORS", str(counters.get("refit_errors")))


def _check_budget(results):
    cv = sum(int(result["counters"].get("cv_fits", 0)) for result in results.values())
    refits = sum(int(result["counters"].get("refits", 0)) for result in results.values())
    _need(cv <= CORE_CV_LIMIT * len(ARMS), "TOTAL_CV_BUDGET", str(cv))
    _need(refits <= CORE_REFIT_LIMIT * len(ARMS), "TOTAL_REFIT_BUDGET", str(refits))
    _need(cv + refits <= TOTAL_FIT_LIMIT, "TOTAL_FIT_BUDGET", str(cv + refits))


def _endpoints(arm_dir, result):
    entries = result.get("family_results")
    _need(type(entries) is dict and set(entries) == set(FAMILIES), "FAMILY_RESULTS",
          str(sorted(entries or ())))
    endpoints = {}
    for family in FAMILIES:
        entry = entries[family]
        _need(type(entry) is dict, "FAMILY_ENTRY", family)
        evaluation = entry.get("evaluation")
        if type(evaluation) is not dict:
            path = Path(arm_dir) / ("family_%s.json" % family)
            _need(path.is_file(), "FAMILY_ARTIFACT", family)
            document = json.loads(path.read_text(encoding="utf-8"))
            evaluation = document.get("evaluation")
            entry = dict(entry, C=document.get("C", entry.get("C")),
                         selected_tau=document.get("selected_tau", entry.get("selected_tau")))
        _need(type(evaluation) is dict, "EVALUATION_MISSING", family)
        splits = {}
        for split, _ in SPLITS:
            value = evaluation.get(split)
            _need(type(value) is dict, "SPLIT_EVALUATION", "%s/%s" % (family, split))
            splits[split] = {"self_gate": value.get("self_gate"), "four_class": value.get("four_class")}
        endpoints[family] = {
            "C": entry.get("C"), "tau": entry.get("selected_tau"),
            "core_artifact_sha256": entry.get("artifact_sha256")
                                    or _sha((Path(arm_dir) / ("family_%s.json" % family)).read_bytes()),
            "endpoints": splits,
        }
    return endpoints


def _family_matched(arms_endpoints):
    return {
        family: {
            "scope": "family-matched endpoints only; no winner-to-winner comparison",
            "original_trained": arms_endpoints["original"][family],
            "added_trained": arms_endpoints["added"][family],
        }
        for family in FAMILIES
    }


def _record(case_data, arms, results, endpoints, root, seconds, plan_sha256):
    return {
        "schema": "cohort_transfer_record.v1",
        "experiment_id": EXPERIMENT_ID,
        "run_id": Path(root).name,
        "control": "cohort_transfer_control_v1",
        "parent_plan_sha256": plan_sha256,
        "core_implementation_id": core.JOB_ID,
        "core_module": str(_core_path()),
        "core_module_sha256": _sha(_core_path().read_bytes()),
        "job_id_policy": "core JOB_ID preserved as implementation provenance; not overridden",
        "arms": {
            cohort: {
                "cohort_role": cohort,
                "n_train": len(arms[cohort]),
                "train_ids_sha256": _ids_sha(arms[cohort]),
                "output_dir": str(Path(root) / cohort),
                "status": results[cohort].get("status"),
                "counters": results[cohort].get("counters"),
                "families": endpoints[cohort],
            }
            for cohort in ARMS
        },
        "family_matched": _family_matched(endpoints),
        "comparison_policy": "family-matched endpoints only; no winner-to-winner comparison",
        "evaluation": {
            "original_ids": len(case_data["original_ids"]),
            "added_ids": len(case_data["added_ids"]),
            "validation_ids": len(case_data["validation_ids"]),
            "validation_used_for_selection": False,
        },
        "aggregate": {"core_calls": len(results), "seconds": float(seconds),
                      "output_bytes": _tree_bytes(root), "output_cap_bytes": OUTPUT_CAP_BYTES},
        "holdout_accessed": False,
    }


def _failed(root, arms, results, calls, exc, seconds, plan_sha256):
    root = Path(root)
    if not root.exists():
        root.mkdir(parents=True)
    path = root / "failure.json"
    if path.exists():
        return
    core._json(path, {
        "schema": "cohort_transfer_failure.v1",
        "experiment_id": EXPERIMENT_ID,
        "run_id": root.name,
        "parent_plan_sha256": plan_sha256,
        "core_implementation_id": core.JOB_ID,
        "core_module_sha256": _sha(_core_path().read_bytes()),
        "arm_sizes": {cohort: len(arms[cohort]) for cohort in ARMS},
        "completed_arms": {cohort: results[cohort].get("status") for cohort in results},
        "core_calls": calls,
        "error": _short(exc),
        "error_type": type(exc).__name__,
        "seconds": float(seconds),
        "holdout_accessed": False,
    })


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def run(case_data, fresh_output_root, deadline=None, runner=core.run, *,
        plan_sha256=None, clock=time.monotonic):
    """Run one unchanged core call per cohort arm under a shared deadline and output cap."""
    root = Path(fresh_output_root)
    _need(not root.exists(), "OUTPUT_EXISTS", str(root))
    arms = derive_arms(case_data)
    _need(deadline is None or not deadline(), "DEADLINE")
    root.mkdir(parents=True)
    started = clock()
    results, endpoints, calls = {}, {}, 0
    try:
        for cohort in ARMS:
            _need(deadline is None or not deadline(), "DEADLINE")
            arm_dir = root / cohort
            arm_data = dict(case_data, train_ids=list(arms[cohort]))
            calls += 1
            result = runner(arm_data, arm_dir, deadline=deadline)
            _check_arm(cohort, result, arms[cohort])
            results[cohort] = result
            endpoints[cohort] = _endpoints(arm_dir, result)
            total = _tree_bytes(root)
            _need(total <= OUTPUT_CAP_BYTES, "OUTPUT_CAP", str(total))
        _need(len(results) == len(ARMS), "CALL_COUNT", str(len(results)))
        _check_budget(results)
        record = _record(case_data, arms, results, endpoints, root, clock() - started, plan_sha256)
        core._json(root / "cohort_transfer_record.json", record)
        _need(_tree_bytes(root) <= OUTPUT_CAP_BYTES, 'OUTPUT_CAP')
        _need(deadline is None or not deadline(), 'DEADLINE')
        return record
    except BaseException as exc:
        _failed(root, arms, results, calls, exc, clock() - started, plan_sha256)
        raise
