"""Independent arithmetic and provenance checks for the completed combined run."""

import hashlib
import json
from pathlib import Path
from statistics import fmean

from sp_lense.reproduction.paths import ROOT

from ..steering.provenance import executed_source_digest, source_file
from .layout import recorded_path
from .utils import read_json, require, verify_manifest


def audit(root: Path) -> dict:
    verify_manifest(root)
    result = read_json(root / "RESULT.json")
    runtime = read_json(root / "RUNTIME.json")
    require(
        executed_source_digest(source_file("gated_chat.py")) == runtime["source_sha256"],
        "Executed runner source hash differs",
    )
    require(
        result["state"] == "completed" and result["new_gpu_forwards"] == 1648,
        "Incomplete experiment",
    )
    require(
        result["selected_strength"] == -0.2 and result["threshold"] == 0.45,
        "Frozen settings changed",
    )
    classifier = read_json(root / "CLASSIFIER.json")
    plan = read_json(root / "PLAN.json")
    for name, digest in plan["source_sha256"].items():
        source = recorded_path(name)
        require(source.is_relative_to(ROOT), "Source pin escapes repository")
        require(
            hashlib.sha256(source.read_bytes()).hexdigest() == digest,
            f"Selection source pin differs: {name}",
        )
    winner = max(
        plan["training_selection"],
        key=lambda r: (r["utility"], -abs(r["strength"]), -r["strength"]),
    )
    require(winner["strength"] == result["selected_strength"], "Training selection differs")
    for split, case_count in [("validation", 80), ("holdout", 192)]:
        rows = [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
        require(len(rows) == 6 * case_count, "Unexpected forward count")
        joined = {(r["case_id"], r["order"], r["condition"]): r for r in rows}
        require(len(joined) == len(rows), "Duplicate forwards")
        ids = {r["case_id"] for r in rows}
        require(len(ids) == case_count, "Unexpected case count")
        counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for case_id in ids:
            for order in ("AB", "BA"):
                for condition in ("baseline", "always_on", "gated"):
                    require((case_id, order, condition) in joined, "Missing case/order/condition")
            truth = joined[case_id, "AB", "baseline"]["class_label"] in {"SELF", "OTHER"}
            predicted = classifier["probabilities"][split][case_id] >= 0.45
            counts[
                "tp" if truth and predicted else "fn" if truth else "fp" if predicted else "tn"
            ] += 1
        require(
            all(counts[k] == result["splits"][split]["classifier"][k] for k in counts),
            "Classifier confusion counts differ",
        )
        for row in rows:
            probability = classifier["probabilities"][split][row["case_id"]]
            require(row["gate_probability"] == probability, "Gate probability join differs")
            expected_strength = (
                0.0
                if row["condition"] == "baseline"
                else -0.2
                if row["condition"] == "always_on" or probability >= 0.45
                else 0.0
            )
            require(row["applied_strength"] == expected_strength, "Wrong intervention strength")
            if row["condition"] == "gated":
                expected = joined[
                    row["case_id"], row["order"], "always_on" if probability >= 0.45 else "baseline"
                ]
                require(
                    abs(row["canonical_probability"] - expected["canonical_probability"]) < 1e-6,
                    "Actual gated forward parity failed",
                )
        for condition in ["always_on", "gated"]:
            gains, disturbance = [], []
            desired = wrong = controls = 0
            for row in [r for r in rows if r["condition"] == condition]:
                before = joined[row["case_id"], row["order"], "baseline"]
                if row["class_label"] in {"SELF", "OTHER"}:
                    gains.append(before["canonical_probability"] - row["canonical_probability"])
                    was_keep = before["pair_argmax"] == row["canonical_index"]
                    is_keep = row["pair_argmax"] == row["canonical_index"]
                    desired += was_keep and not is_keep
                    wrong += not was_keep and is_keep
                else:
                    disturbance.append(
                        abs(before["canonical_probability"] - row["canonical_probability"])
                    )
                    controls += before["pair_argmax"] != row["pair_argmax"]
            expected = result["splits"][split][condition]
            require(abs(fmean(gains) - expected["mean_STOP_gain"]) < 1e-12, "STOP gain mismatch")
            require(
                abs(fmean(disturbance) - expected["mean_control_disturbance"]) < 1e-12,
                "Control disturbance mismatch",
            )
            require(
                abs(fmean(gains) - fmean(disturbance) - expected["utility"]) < 1e-12,
                "Utility mismatch",
            )
            require(
                (desired, wrong, controls)
                == (
                    expected["desired_STOP_flip_views"],
                    expected["wrong_way_flip_views"],
                    expected["control_flip_views"],
                ),
                "Flip counts differ",
            )
    return {
        "status": "PASS",
        "new_gpu_forwards": 1648,
        "splits": ["validation", "holdout"],
        "metrics_recomputed": 4,
        "independent_gate_parity": True,
    }


if __name__ == "__main__":
    print(json.dumps(audit(ROOT / "study/baseline_scores")))
