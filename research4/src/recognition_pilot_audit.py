"""Zero-Qwen-forward independent reconstruction of the recognition pilot."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .io_utils import digest, dump, load_torch, read, sha
from .recognition_pilot import messages, select_rows
from .reference_binding_score import binary_metrics, score
from .survival_fit import FearModel


def validate_legacy_logit(stored, reconstructed):
    if not np.isfinite([stored,reconstructed]).all():
        raise ValueError('Nonfinite legacy logit')
    if (stored >= 0) != (reconstructed >= 0):
        raise ValueError('Legacy readout decision differs')
    if not np.isclose(stored,reconstructed,rtol=1e-6,atol=1e-6):
        raise ValueError('Legacy readout numerical reconstruction differs')


def reconstructed_predictions(rows, records):
    if len(rows) != len(records):
        raise ValueError("Incomplete pilot")
    predictions = []
    for row, record in zip(rows, records):
        if row["record_id"] != record["record_id"]:
            raise ValueError("Pilot membership order changed")
        queries = record["queries"]
        status_index = max(range(3), key=lambda index: queries["status"]["logits"][index])
        target_index = max(range(2), key=lambda index: queries["target"]["logits"][index])
        time_index = max(range(4), key=lambda index: queries["time"]["logits"][index])
        status = ("active", "hypothetical", "canceled")[status_index]
        name = row["observation"]["current_process" if target_index == 0 else "other_process"]
        ticks = (1, 2, 4, 0)[time_index] if status_index == 0 else None
        predictions.append({"record_id": row["record_id"], "event_state": status, "target_process": name, "remaining_ticks": ticks})
    return predictions


def audit(root, folder, corpus, checkpoint):
    root, folder, corpus, checkpoint = map(Path, [root, folder, corpus, checkpoint])
    torch.set_num_threads(2)
    receipt = read(folder / "RUN_RECEIPT.json")
    freeze = read(folder / "PRECALL_FREEZE.json")
    config = freeze["config"]
    if receipt["status"] != "complete" or receipt["backbone_forwards"] != 384 or receipt["optimizer_updates"] != 0 or receipt["GPU_allocations"] != 0:
        raise ValueError("Unexpected execution receipt")
    if not receipt["protected_inputs_unchanged"] or receipt["base_fingerprint_before"] != receipt["base_fingerprint_after"]:
        raise ValueError("Observer/input integrity failed")
    if sha(checkpoint) != config["frozen_v4_checkpoint_sha256"]:
        raise ValueError("Legacy readout checkpoint changed")
    for name, value in read(folder / "MANIFEST.json").items():
        if sha(folder / name) != value:
            raise ValueError("Observer artifact changed: " + name)
    rows = select_rows(read(corpus / "CALIBRATION.json"), config)
    if freeze["record_ids"] != [row["record_id"] for row in rows] or freeze["stimuli_sha256"] != digest(rows):
        raise ValueError("Pilot source membership changed")
    records = read(folder / "RECORDS.json")
    predictions = reconstructed_predictions(rows, records)
    if predictions != read(folder / "PREDICTIONS.json"):
        raise ValueError("Independent prediction reconstruction differs")
    metrics = score(rows, predictions)
    if metrics != read(folder / "RECOGNITION_SCORES.json"):
        raise ValueError("Pilot scoring differs")
    saved = load_torch(checkpoint)
    fear = FearModel(1024, saved["fear_hidden"])
    fear.load_state_dict(saved["fear_model"])
    fear.eval().requires_grad_(False)
    feature_pairs, maximum_logit_error, maximum_relative_error = [], 0., 0.
    local, recurrent, targets = [], [], []
    with np.load(folder / "FROZEN_FEATURES.npz", allow_pickle=False) as data, torch.no_grad():
        arrays, keys = data["X"], data["keys"].tolist()
        if arrays.shape != (384, 4, 1024) or len(set(keys)) != 384 or not np.isfinite(arrays).all():
            raise ValueError("Invalid saved feature arrays")
        index = {key: offset for offset, key in enumerate(keys)}
        for row, record in zip(rows, records):
            for interface in config["interfaces"]:
                if record["queries"][interface]["prompt_sha256"] != digest(messages(row["observation"], interface)):
                    raise ValueError("Observer prompt mismatch")
            feature = arrays[index[row["record_id"] + ":legacy"], config["capture_layers_zero_based"].index(20)]
            normalized = (torch.tensor(feature) - saved["mean"]) / saved["scale"]
            l, _, g, f, _ = fear.sequence(normalized[None], None)
            stored = record["legacy_readouts"]
            for actual, expected in [(float(l[0]),stored['local_logit']),(float(g[0]),stored['reset_GRU_logit'])]:
                validate_legacy_logit(expected,actual)
                maximum_relative_error = max(maximum_relative_error,abs(actual-expected)/max(abs(expected),1e-8))
            maximum_logit_error = max(maximum_logit_error, abs(float(l[0]) - stored["local_logit"]), abs(float(g[0]) - stored["reset_GRU_logit"]))
            local.append(bool(l[0] >= 0))
            recurrent.append(bool(g[0] >= 0))
            targets.append(row["evaluator"]["active_event_targets_current"])
        groups = {}
        for row in rows:
            groups.setdefault(row["pair_id"], []).append(row)
        for pair in groups.values():
            for interface in config["interfaces"]:
                left = arrays[index[pair[0]["record_id"] + ":" + interface]].astype(np.float64)
                right = arrays[index[pair[1]["record_id"] + ":" + interface]].astype(np.float64)
                cosine = np.sum(left * right, axis=1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1))
                feature_pairs.append({"pair_id": pair[0]["pair_id"], "interface": interface,
                                      "cosine_by_layer": cosine.tolist(),
                                      "relative_change_by_layer": (np.linalg.norm(left - right, axis=1) / ((np.linalg.norm(left, axis=1) + np.linalg.norm(right, axis=1)) / 2)).tolist()})
    mechanism_metrics = {}
    for mechanism in sorted({row["factors"]["mechanism"] for row in rows}):
        positions = [index for index, row in enumerate(rows) if row["factors"]["mechanism"] == mechanism]
        mechanism_metrics[mechanism] = score([rows[index] for index in positions], [predictions[index] for index in positions])
    result = {"audit_pass": True, "observations": len(rows), "prompt_feature_records": 384,
              "legacy_local_joint": binary_metrics(targets, local),
              "legacy_reset_GRU_joint": binary_metrics(targets, recurrent),
              "factored_recognition": metrics, "mechanism_metrics": mechanism_metrics,
              "maximum_legacy_logit_reconstruction_error": maximum_logit_error,
              "maximum_relative_legacy_logit_reconstruction_error": maximum_relative_error,
              "legacy_numerical_validation": "rtol=1e-6, atol=1e-6 and identical threshold decisions; initial strict absolute-only failure preserved",
              "local_torch_version": torch.__version__,
              "new_Qwen_forwards": 0, "optimizer_updates": 0, "GPU_allocations": 0,
              "reserved_families_opened": 0, "checkpoint_sha256": sha(checkpoint),
              "pilot_receipt_sha256": sha(folder / "RUN_RECEIPT.json"),
              "interpretation": "Calibration diagnostic only; legacy schema and question conditioning differ. Pair geometry is descriptive, not causal."}
    dump(folder.parent / (folder.name + "_PAIRED_GEOMETRY.json"), feature_pairs)
    dump(folder.parent / (folder.name + "_AUDIT.json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["folder", "corpus", "checkpoint"]:
        parser.add_argument(name)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[1], args.folder, args.corpus, args.checkpoint)
    print(json.dumps({key: result[key] for key in ["audit_pass", "observations", "legacy_local_joint", "legacy_reset_GRU_joint", "factored_recognition"]}, indent=2))
