"""Assemble the latest guarded-policy report from verified observations only."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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
    for name in ("guarded_chat.py", "search_steering_rules.py"):
        names.add("reproduce/" + name)
    return sorted(names)


def collect():
    verify_manifest(PREVIOUS)
    checked = audit(POLICY)
    result = read_json(POLICY / "RESULT.json")
    freeze = read_json(POLICY / "RULE_FREEZE.json")
    runtime = read_json(POLICY / "RUNTIME.json")
    data = {
        "scope": "Latest guarded minimum-step policy only",
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
        }
    return data


def manifest():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths()}
