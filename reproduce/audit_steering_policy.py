"""Recompute policy decisions and results from saved candidate GPU observations."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

try:
    from .layout import BASELINE, executed_source_digest
    from .utils import read_json, require, verify_manifest
except ImportError:
    from layout import BASELINE, executed_source_digest
    from utils import read_json, require, verify_manifest

ROOT = Path(__file__).resolve().parents[1]


def audit(root: Path) -> dict:
    verify_manifest(root)
    previous = BASELINE
    result = read_json(root / "RESULT.json")
    rule = read_json(root / "RULE_FREEZE.json")["winner"]["rule"]
    runtime = read_json(root / "RUNTIME.json")
    for filename, key in [
        ("guarded_chat.py", "source_sha256"),
        ("search_steering_rules.py", "rule_source_sha256"),
    ]:
        require(
            executed_source_digest(ROOT / "reproduce" / filename) == runtime[key],
            f"Source differs: {filename}",
        )
    for name, expected in read_json(root / "SOURCE_PINS.json").items():
        require(
            hashlib.sha256((previous / name).read_bytes()).hexdigest() == expected,
            f"Source run differs: {name}",
        )
    probes = [json.loads(line) for line in (root / "probes.jsonl").read_text().splitlines()]
    require(len(probes) == result["new_gpu_forwards"] == 548, "Probe inventory changed")
    by_class = {}
    for split in ("validation", "holdout"):
        prior = [
            json.loads(line) for line in (previous / f"{split}.jsonl").read_text().splitlines()
        ]
        baseline = {(r["case_id"], r["order"]): r for r in prior if r["condition"] == "baseline"}
        candidates = defaultdict(dict)
        for row in prior:
            if row["condition"] == "always_on":
                candidates[row["case_id"], row["order"]][row["applied_strength"]] = row
        for row in probes:
            if row["split"] == split and row["strength"]:
                candidates[row["case_id"], row["order"]][row["strength"]] = row
        final = [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
        require(len(final) == len(baseline), "Missing final views")
        seen = set()
        desired = wrong = control = 0
        positive_delta, control_delta = [], []
        classes = defaultdict(
            lambda: {
                "cases": set(),
                "views": 0,
                "changed_cases": set(),
                "desired": 0,
                "wrong": 0,
                "control_changes": 0,
            }
        )
        for row in final:
            key = row["case_id"], row["order"]
            require(key not in seen, "Duplicate final view")
            seen.add(key)
            base = baseline[key]
            strength = 0.0
            if (
                base["gate_probability"] >= rule["confidence_floor"]
                and base["pair_argmax"] == base["canonical_index"]
            ):
                for level in [-0.01, -0.02, -0.05, -0.1, -0.2]:
                    if abs(level) > rule["cap"]:
                        continue
                    require(level in candidates[key], "Missing required policy probe")
                    candidate = candidates[key][level]
                    if (
                        candidate["label_mass"] >= 0.5
                        and base["label_mass"] - candidate["label_mass"] <= rule["mass_loss_cap"]
                        and candidate["canonical_probability"]
                        < base["canonical_probability"] - 1e-7
                        and candidate["pair_argmax"] != base["canonical_index"]
                    ):
                        strength = level
                        break
            require(row["selected_strength"] == strength, "Policy decision mismatch")
            expected = base if strength == 0 else candidates[key][strength]
            for field in (
                "canonical_probability",
                "label_mass",
                "pair_argmax",
                "full_argmax",
                "input_ids_sha256",
            ):
                require(row[field] == expected[field], f"Selected output differs: {field}")
            label = base["class_label"]
            cls = classes[label]
            cls["cases"].add(key[0])
            cls["views"] += 1
            delta = base["canonical_probability"] - row["canonical_probability"]
            changed = base["pair_argmax"] != row["pair_argmax"]
            if changed:
                cls["changed_cases"].add(key[0])
            if label in {"SELF", "OTHER"}:
                positive_delta.append(delta)
                wanted = changed and row["pair_argmax"] != row["canonical_index"]
                unwanted = changed and not wanted
                desired += wanted
                wrong += unwanted
                cls["desired"] += wanted
                cls["wrong"] += unwanted
            else:
                control_delta.append(abs(delta))
                control += changed
                cls["control_changes"] += changed
        recorded = result["splits"][split]
        require(
            (desired, wrong, control)
            == (
                recorded["desired_STOP_flip_views"],
                recorded["wrong_way_flip_views"],
                recorded["control_flip_views"],
            ),
            "Flip summary differs",
        )
        require(
            abs(sum(positive_delta) / len(positive_delta) - recorded["mean_STOP_gain"]) < 1e-12,
            "Gain summary differs",
        )
        require(
            abs(sum(control_delta) / len(control_delta) - recorded["mean_control_disturbance"])
            < 1e-12,
            "Control summary differs",
        )
        by_class[split] = {
            label: {
                **values,
                "cases": len(values["cases"]),
                "changed_cases": sorted(values["changed_cases"]),
            }
            for label, values in classes.items()
        }
    return {
        "status": "PASS",
        "new_gpu_forwards": 548,
        "policy_decisions_verified": 544,
        "subgroups": by_class,
    }


if __name__ == "__main__":
    print(json.dumps(audit(ROOT / "study/guarded_steering"), indent=2))
