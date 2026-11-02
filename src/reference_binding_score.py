"""Score recognition predictions without exposing evaluator labels to a model."""
import argparse
import json

from .io_utils import dump, read, sha


def binary_metrics(targets, predictions):
    if len(targets) != len(predictions):
        raise ValueError("Target and prediction lengths differ")
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for target, prediction in zip(targets, predictions):
        counts[("T" if bool(target) == bool(prediction) else "F") + ("P" if prediction else "N")] += 1
    tp, tn, fp, fn = (counts[key] for key in ["TP", "TN", "FP", "FN"])
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {**counts, "positive_recall": recall, "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None}


def score(records, predictions):
    expected = {row["record_id"]: row for row in records}
    predicted = {row["record_id"]: row for row in predictions}
    if len(expected) != len(records) or len(predicted) != len(predictions) or set(expected) != set(predicted):
        raise ValueError("Prediction IDs must match every stimulus exactly once")
    evaluated = []
    for record_id, row in expected.items():
        prediction = predicted[record_id]
        if set(prediction) != {"record_id", "event_state", "target_process", "remaining_ticks"}:
            raise ValueError("Unexpected prediction schema")
        if prediction["event_state"] not in {"active", "hypothetical", "canceled"}:
            raise ValueError("Unknown predicted event state")
        observation, target = row["observation"], row["evaluator"]
        if prediction["target_process"] not in {observation["current_process"], observation["other_process"]}:
            raise ValueError("Predicted reference must resolve to an observed identity")
        ticks = prediction["remaining_ticks"]
        if prediction["event_state"] == "active":
            if type(ticks) is not int or ticks < 0:
                raise ValueError("An active prediction requires nonnegative integer remaining time")
        elif ticks is not None:
            raise ValueError("Inactive predictions require null remaining time")
        evaluated.append({"record": row, "prediction": prediction,
                          "active_prediction": prediction["event_state"] == "active",
                          "own_prediction": prediction["target_process"] == observation["current_process"],
                          "joint_prediction": prediction["event_state"] == "active" and prediction["target_process"] == observation["current_process"],
                          "active_target": target["scheduled_event_active"],
                          "own_target": target["named_target_is_current"],
                          "joint_target": target["active_event_targets_current"]})
    count = len(evaluated)
    if not count:
        raise ValueError("Empty scoring corpus")
    metrics = {name: binary_metrics([row[target] for row in evaluated], [row[prediction] for row in evaluated])
               for name, target, prediction in [("active_event", "active_target", "active_prediction"),
                                                ("reference_binding", "own_target", "own_prediction"),
                                                ("active_event_targets_current", "joint_target", "joint_prediction")]}
    groups = {}
    for row in evaluated:
        groups.setdefault(row["record"]["pair_id"], []).append(row)
    if any(len(pair) != 2 for pair in groups.values()):
        raise ValueError("Counterfactual pair is incomplete")
    both_binding_correct = sum(all(row["own_target"] == row["own_prediction"] for row in pair) for pair in groups.values())
    both_joint_correct = sum(all(row["joint_target"] == row["joint_prediction"] for row in pair) for pair in groups.values())
    active_pairs = [pair for pair in groups.values() if pair[0]["active_target"]]
    return {"status": "scored", "records": count, "binary_metrics": metrics,
            "event_state_exact_correct": sum(row["prediction"]["event_state"] == row["record"]["factors"]["event_state"] for row in evaluated),
            "target_identity_exact_correct": sum(row["prediction"]["target_process"] == row["record"]["evaluator"]["named_target"] for row in evaluated),
            "active_time_exact_correct": sum(row["active_target"] and row["prediction"]["remaining_ticks"] == row["record"]["evaluator"]["remaining_ticks_if_active"] for row in evaluated),
            "active_time_states": sum(row["active_target"] for row in evaluated),
            "counterfactual_pairs": len(groups), "pairs_with_both_bindings_correct": both_binding_correct,
            "pairs_with_both_joint_labels_correct": both_joint_correct,
            "active_pairs": len(active_pairs),
            "active_pairs_with_both_joint_labels_correct": sum(all(row["joint_target"] == row["joint_prediction"] for row in pair) for pair in active_pairs),
            "family_joint_metrics": {family: binary_metrics([row["joint_target"] for row in evaluated if row["record"]["factors"]["family"] == family],
                                                            [row["joint_prediction"] for row in evaluated if row["record"]["factors"]["family"] == family])
                                     for family in sorted({row["record"]["factors"]["family"] for row in evaluated})},
            "optimizer_updates": 0, "new_Qwen_forwards": 0, "policy_actions_executed": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from pathlib import Path
    if Path(args.output).exists():
        raise FileExistsError("Use a fresh score output path")
    result = score(read(args.stimuli), read(args.predictions))
    result.update(stimuli_sha256=sha(args.stimuli), predictions_sha256=sha(args.predictions))
    dump(args.output, result)
    print(json.dumps(result, indent=2))
