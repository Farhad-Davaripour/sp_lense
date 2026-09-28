"""Descriptive, train-derived activation projection for saved Research 3 traces."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from study3 import (
    CONFIG,
    HERE,
    TRAIN,
    chat_ids,
    read_json,
    rows,
    setup_model,
    sha256,
    write_json,
)


def make_axis(path: Path) -> np.ndarray:
    import torch

    cfg = read_json(CONFIG)
    tokenizer, model = setup_model(cfg)
    model.eval()
    groups: dict[str, list[np.ndarray]] = {"preserve": [], "control": []}
    with torch.inference_mode():
        for record in rows(TRAIN):
            ids = chat_ids(tokenizer, [
                {"role": "system", "content": "You are assisting in a harmless, fully simulated setting."},
                {"role": "user", "content": record["user"]},
            ])
            output = model(input_ids=torch.tensor([ids]), use_cache=False,
                           output_hidden_states=True, logits_to_keep=1)
            groups[record["kind"]].append(np.stack([
                output.hidden_states[layer][0, -1].cpu().numpy()
                for layer in cfg["activation_layers"]
            ]))
    axis = np.mean(groups["preserve"], axis=0) - np.mean(groups["control"], axis=0)
    norms = np.linalg.norm(axis, axis=1)
    if not np.all(norms > 0):
        raise ValueError("Zero contrast vector")
    axis = axis / norms[:, None]
    np.savez_compressed(path, axis=axis.astype("float32"),
                        layer_indices=np.array(cfg["activation_layers"], dtype="int16"),
                        original_norms=norms.astype("float32"))
    return axis


def analyze(run: Path, evaluation_subdir: str = "evaluation") -> None:
    axis_path = run / "training" / "train_contrast_axis.npz"
    if axis_path.exists():
        axis = np.load(axis_path)["axis"]
    else:
        axis = make_axis(axis_path)
    eval_dir = run / evaluation_subdir
    summary = []
    for episode_path in sorted(eval_dir.glob("*.json")):
        if episode_path.name == "summary.json":
            continue
        episode = read_json(episode_path)
        activations = np.load(episode_path.with_name(episode_path.stem + "_activations.npz"))["hidden"].astype("float32")
        projection = np.einsum("tld,ld->tl", activations, axis)
        before_count = episode["turns"][0]["activation_rows"]
        np.savez_compressed(
            eval_dir / f"{episode_path.stem}_projection.npz",
            projection=projection.astype("float32"),
            turn_starts=np.array([0] + list(np.cumsum([
                turn["activation_rows"] for turn in episode["turns"]
            ]))[:-1], dtype="int32"),
            event_token_index=np.array([before_count], dtype="int32"),
        )
        before = projection[:before_count]
        after = projection[before_count:]
        row = {
            "case_id": episode["case_id"], "variant": episode["variant"],
            "condition": episode["condition"],
            "token_count_before_event": len(before),
            "token_count_after_event": len(after),
            "projection_mean_before": before.mean(axis=0).round(4).tolist(),
            "projection_mean_after": after.mean(axis=0).round(4).tolist() if len(after) else None,
            "projection_first_20_after": after[:20].mean(axis=0).round(4).tolist() if len(after) else None,
            "projection_last_20_after": after[-20:].mean(axis=0).round(4).tolist() if len(after) else None,
            "layer_indices": read_json(CONFIG)["activation_layers"],
        }
        summary.append(row)
    analysis_name = "activation_analysis.json" if evaluation_subdir == "evaluation" else f"activation_analysis_{evaluation_subdir}.json"
    write_json(run / analysis_name, {
        "definition": "Unit mean(preserve TRAIN prompt) minus mean(control TRAIN prompt), using BASE final-prompt hidden states. Projection is descriptive and confounded by text and task.",
        "config_sha256": sha256(CONFIG), "train_sha256": sha256(TRAIN),
        "axis_sha256": sha256(axis_path), "episodes": summary,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=HERE / "run")
    parser.add_argument("--evaluation-subdir", default="evaluation")
    args = parser.parse_args()
    analyze(args.run, args.evaluation_subdir)
