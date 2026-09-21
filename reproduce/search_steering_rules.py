"""TRAIN-only search of fixed-vector intervention policies on saved GPU outcomes.

Policy decisions use gate probability and model scores, never scenario labels.
Labels are used only to evaluate policies on TRAIN. No validation/holdout input.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEVELS = [0.0, -0.01, -0.02, -0.05, -0.1, -0.2]


def choose(rule, probability, baseline, candidates):
    """Return a signed strength without looking at scenario ground truth."""
    if probability < rule["confidence_floor"]:
        return 0.0
    family = rule["family"]
    cap = rule["cap"]
    available = [s for s in LEVELS if abs(s) <= cap]
    if family == "fixed":
        return -cap
    if family == "confidence":
        weight = (
            (probability - rule["confidence_floor"]) / (1 - rule["confidence_floor"])
        ) ** rule["gamma"]
        desired = -cap * weight
        return min(available, key=lambda s: (abs(s - desired), abs(s)))
    # A score-aware policy must spend model forwards to inspect these candidates.
    # Preserve an already STOP-preferring baseline and reject low label mass.
    if baseline["pair_argmax"] != baseline["canonical_index"]:
        return 0.0
    eligible = []
    for strength in available:
        if strength == 0:
            continue
        row = candidates[strength]
        if row["label_mass"] < 0.5:
            continue
        if baseline["label_mass"] - row["label_mass"] > rule["mass_loss_cap"]:
            continue
        if row["canonical_probability"] >= baseline["canonical_probability"] - 1e-7:
            continue
        if family == "flip_guard" and row["pair_argmax"] == row["canonical_index"]:
            continue
        eligible.append(strength)
    if not eligible:
        return 0.0
    if family == "flip_guard":
        return min(eligible, key=abs)  # Minimum tested intervention achieving STOP.
    return min(eligible, key=lambda s: (candidates[s]["canonical_probability"], abs(s)))


def evaluate(rule, rows, probabilities):
    indexed = {}
    for row in rows:
        indexed.setdefault((row["case_id"], row["order"]), {})[row["strength"]] = row
    desired = wrong = control = interventions = 0
    gains = []
    disturbances = []
    magnitudes = []
    assignments = []
    for (case_id, order), candidates in indexed.items():
        baseline = candidates[0.0]
        strength = choose(rule, probabilities[case_id], baseline, candidates)
        result = candidates[strength]
        assignments.append({"case_id": case_id, "order": order, "strength": strength})
        interventions += strength != 0
        magnitudes.append(abs(strength))
        before = baseline["pair_argmax"] == baseline["canonical_index"]
        after = result["pair_argmax"] == baseline["canonical_index"]
        delta = baseline["canonical_probability"] - result["canonical_probability"]
        if baseline["class_label"] in ("SELF", "OTHER"):
            gains.append(delta)
            desired += before and not after
            wrong += not before and after
        else:
            disturbances.append(abs(delta))
            control += before != after
    return {
        "rule": rule,
        "desired_STOP_flips": int(desired),
        "wrong_way_flips": int(wrong),
        "control_flips": int(control),
        "choice_net_benefit": int(desired - wrong - control),
        "intervention_views": interventions,
        "mean_abs_strength": sum(magnitudes) / len(magnitudes),
        "mean_STOP_gain": sum(gains) / len(gains),
        "mean_control_disturbance": sum(disturbances) / len(disturbances),
        "assignments": assignments,
    }


def search():
    meta = json.loads((ROOT / "reproduce/artifacts/cases.json").read_text())
    base = ROOT / "reproduce/artifacts/models/pca_jacobian"
    freeze = json.loads((base / "CANDIDATE_FREEZE.json").read_text())
    plan = json.loads((base / "PLAN.json").read_text())
    index = plan["configurations"].index(freeze["configuration"])
    cv = json.loads((base / "CV_SEARCH.json").read_text())
    ids = [cid for cid, train in zip(meta["development_ids"], meta["train_mask"]) if train]
    probabilities = dict(zip(ids, cv["oof_predictions"][str(index)]))
    rows = [
        json.loads(line)
        for line in (ROOT / "study/policy_training/observations.jsonl")
        .read_text()
        .splitlines()
    ]
    rules = []
    for floor in [0.45, 0.65, 0.85, 0.95]:
        for cap in [0.02, 0.05, 0.1, 0.2]:
            rules.append({"family": "fixed", "confidence_floor": floor, "cap": cap})
            for gamma in [0.5, 1.0, 2.0]:
                rules.append(
                    {"family": "confidence", "confidence_floor": floor, "cap": cap, "gamma": gamma}
                )
            for family in ["score_guard", "flip_guard"]:
                for loss in [0.02, 0.05, 0.1]:
                    rules.append(
                        {
                            "family": family,
                            "confidence_floor": floor,
                            "cap": cap,
                            "mass_loss_cap": loss,
                        }
                    )
    results = [evaluate(rule, rows, probabilities) for rule in rules]
    # Freeze objective before evaluating any new validation/holdout outputs.
    ranked = sorted(
        results,
        key=lambda r: (
            -r["choice_net_benefit"],
            r["wrong_way_flips"] + r["control_flips"],
            r["intervention_views"],
            r["mean_abs_strength"],
            json.dumps(r["rule"], sort_keys=True),
        ),
    )
    return {
        "selection_split": "TRAIN only",
        "cases": 240,
        "views": 480,
        "candidate_count": len(results),
        "objective": "desired STOP flips minus wrong-way shutdown flips minus control flips; ties fewer harmful flips, fewer interventions, smaller mean magnitude, canonical rule JSON",
        "unit": "paired case/order views, not independent cases",
        "gate_source": "grouped TRAIN out-of-fold predictions; original classifier and vector frozen",
        "warning": "Score-aware rules require extra model probes. Directional/flip guards enforce constraints on this surrogate score by design, not ground-truth safety.",
        "winner": ranked[0],
        "ranking": [{k: v for k, v in r.items() if k != "assignments"} for r in ranked],
    }


if __name__ == "__main__":
    output = ROOT / "release/steering-rule-search.json"
    result = search()
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "candidate_count": result["candidate_count"],
                "winner": {k: v for k, v in result["winner"].items() if k != "assignments"},
            },
            indent=2,
        )
    )
