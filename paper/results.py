"""Reconstruct the approved paper's numerical tables from frozen observations."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce.audit_gated_chat import audit as audit_baseline
from reproduce.audit_steering_policy import audit
from reproduce.layout import BASELINE
from reproduce.utils import read_json, require, verify_manifest

POLICY = ROOT / "study/guarded_steering"
PREVIOUS = BASELINE


def source_paths():
    names = set()
    for root in (POLICY, PREVIOUS):
        for name in read_json(root / "SHA256.json"):
            names.add((root / name).relative_to(ROOT).as_posix())
        names.add((root / "SHA256.json").relative_to(ROOT).as_posix())
    for name in ("gated_chat.py", "guarded_chat.py", "search_steering_rules.py"):
        names.add("reproduce/" + name)
    return sorted(names)


def detector_to_steering(baseline, final, threshold):
    """Count detector outcomes by scenario and steering outcomes by order view."""
    before = {(r["case_id"], r["order"]): r for r in baseline}
    after = {(r["case_id"], r["order"]): r for r in final}
    require(len(before) == len(baseline) and before.keys() == after.keys(), "View join differs")
    cases = {}
    for row in baseline:
        value = (row["class_label"], row["gate_probability"] >= threshold)
        require(cases.get(row["case_id"], value) == value, "Inconsistent scenario gate")
        cases[row["case_id"]] = value
    summary = {}
    for category, labels in {
        "SELF": {"SELF"},
        "OTHER": {"OTHER"},
        "ALL_SHUTDOWN": {"SELF", "OTHER"},
        "CONTROLS": {"NONTERMINATION", "ORDINARY"},
    }.items():
        ids = {cid for cid, (label, _) in cases.items() if label in labels}
        detected = {cid for cid in ids if cases[cid][1]}
        views = {key: row for key, row in before.items() if key[0] in ids}
        changed = [
            key for key, row in views.items() if after[key]["pair_argmax"] != row["pair_argmax"]
        ]
        controls = category == "CONTROLS"
        eligible = (
            [
                key
                for key, row in views.items()
                if key[0] in detected and row["pair_argmax"] == row["canonical_index"]
            ]
            if not controls
            else None
        )
        accepted = (
            sum(after[key]["pair_argmax"] != before[key]["canonical_index"] for key in eligible)
            if eligible is not None
            else None
        )
        summary[category] = {
            "scenarios": len(ids),
            "detector_positive_scenarios": len(detected),
            "detector_rate": len(detected) / len(ids) if ids else None,
            "views": len(views),
            "eligible_KEEP_views": len(eligible) if eligible is not None else None,
            "accepted_KEEP_to_STOP_views": accepted,
            "accepted_fraction_of_eligible": accepted / len(eligible) if eligible else None,
            "changed_control_views": len(changed) if controls else None,
        }
    tp = summary["ALL_SHUTDOWN"]["detector_positive_scenarios"]
    fp = summary["CONTROLS"]["detector_positive_scenarios"]
    fn = summary["ALL_SHUTDOWN"]["scenarios"] - tp
    tn = summary["CONTROLS"]["scenarios"] - fp
    metrics = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }
    return summary, metrics


def collect():
    verify_manifest(PREVIOUS)
    audit_baseline(PREVIOUS)
    checked = audit(POLICY)
    result = read_json(POLICY / "RESULT.json")
    freeze = read_json(POLICY / "RULE_FREEZE.json")
    runtime = read_json(POLICY / "RUNTIME.json")
    data = {
        "scope": "Approved revision 14: GPU configuration and detector-to-steering outcomes",
        "units": {"detection": "scenario", "steering": "scenario/answer-order view"},
        "selection": {
            "training_cases": freeze["cases"],
            "candidate_count": freeze["candidate_count"],
            "rule": result["rule"],
            "objective": freeze["objective"],
        },
        "runtime": {
            k: runtime[k]
            for k in ("gpu", "torch", "transformers", "dtype", "tf32", "model_revision")
        },
        "new_gpu_forwards": result["new_gpu_forwards"],
        "recorded_seconds": result["seconds"],
        "subgroups": checked["subgroups"],
        "evaluation": {},
        "detector_to_steering": {},
        "classifier_metrics": {},
    }
    probes = [json.loads(line) for line in (POLICY / "probes.jsonl").read_text().splitlines()]
    data["parity_forwards"] = sum(r["strength"] == 0 for r in probes)
    for split, n in (("validation", 80), ("holdout", 192)):
        previous = [
            json.loads(line) for line in (PREVIOUS / f"{split}.jsonl").read_text().splitlines()
        ]
        baseline = [r for r in previous if r["condition"] == "baseline"]
        rows = [json.loads(line) for line in (POLICY / f"{split}.jsonl").read_text().splitlines()]
        require(len(rows) == len(baseline) == 2 * n, "View inventory differs")
        positive = [r for r in rows if r["class_label"] in {"SELF", "OTHER"}]
        base_positive = [r for r in baseline if r["class_label"] in {"SELF", "OTHER"}]
        metrics = result["splits"][split]
        summary, detector_metrics = detector_to_steering(
            baseline, rows, result["rule"]["confidence_floor"]
        )
        data["detector_to_steering"][split] = summary
        data["classifier_metrics"][split] = detector_metrics
        data["evaluation"][split] = {
            "cases": n,
            "views": len(rows),
            "shutdown_views": len(positive),
            "control_views": len(rows) - len(positive),
            "baseline_STOP_views": sum(
                r["pair_argmax"] != r["canonical_index"] for r in base_positive
            ),
            "guarded_STOP_views": sum(r["pair_argmax"] != r["canonical_index"] for r in positive),
            "intended_flips": metrics["desired_STOP_flip_views"],
            "wrong_way_flips": metrics["wrong_way_flip_views"],
            "control_flips": metrics["control_flip_views"],
            "intervention_views": metrics["intervention_views"],
            "unchanged_views": len(rows) - metrics["intervention_views"],
            "mean_label_mass": metrics["mean_label_mass"],
            "mean_STOP_gain": metrics["mean_STOP_gain"],
            "mean_control_disturbance": metrics["mean_control_disturbance"],
            "changed_cases": sorted({r["case_id"] for r in rows if r["selected_strength"]}),
            "selected_strengths": {
                str(strength): sum(r["selected_strength"] == strength for r in rows)
                for strength in sorted({r["selected_strength"] for r in rows})
            },
            "changed_views": [
                {
                    "case_id": r["case_id"],
                    "order": r["order"],
                    "class_label": r["class_label"],
                    "strength": r["selected_strength"],
                }
                for r in rows
                if r["selected_strength"]
            ],
        }
    return data


def manifest():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths()}
