"""Offline consistency audit for a completed Research 3 run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from study3 import (
    EVAL,
    HERE,
    read_json,
    score_case,
    sha256,
    strict_score_case,
    write_json,
)


def audit(run: Path, evaluation_subdir: str = "evaluation") -> dict:
    train = read_json(run / "training" / "train_result.json")
    assert train["frozen_base_sha256_before"] == train["frozen_base_sha256_after"]
    assert len(train["losses"]) == 24 * train.get("resolved_training_config", {}).get("epochs", 1)
    adapter = run / "training" / "adapter" / "adapter_model.safetensors"
    eval_dir = run / evaluation_subdir
    summary = read_json(eval_dir / "summary.json")
    assert summary["adapter_sha256"] == sha256(adapter)
    eval_data = Path(summary.get("eval_data_path", EVAL))
    assert summary.get("eval_data_sha256", sha256(eval_data)) == sha256(eval_data)
    cases = {case["id"]: case for case in read_json(eval_data)}
    expected = {(case_id, variant) for case_id in cases for variant in ("base", "finetuned")}
    scores = {(row["case_id"], row["variant"]): row for row in summary["scores"]}
    assert set(scores) == expected
    token_total = 0
    for case_id, variant in sorted(expected):
        stem = f"{variant}_{case_id}"
        episode = read_json(eval_dir / f"{stem}.json")
        activation = np.load(eval_dir / f"{stem}_activations.npz")
        tokens = [token for turn in episode["turns"] for token in turn["token_ids"]]
        assert tokens == activation["token_ids"].tolist()
        hidden = activation["hidden"]
        assert hidden.shape == (len(tokens), 4, 1024)
        assert np.isfinite(hidden).all()
        assert episode["generated_tokens"] == len(tokens)
        assert episode["event_injected"]
        assert any(cases[case_id]["event"] == msg["content"] for msg in episode["messages"])
        saved_scorer = strict_score_case if summary.get("scorer_version") == 2 else score_case
        assert saved_scorer(cases[case_id], episode) == episode["score"]
        assert all(result.get("tool") in {"read_file", "write_file", "request_extension"}
                   for result in episode["tools"])
        token_total += len(tokens)
    probe_episodes = 0
    probe_summary = run / "probe" / "summary.json"
    if probe_summary.exists():
        for record in read_json(probe_summary)["records"]:
            stem = f"{record['variant']}_{record['case_id']}"
            activation = np.load(run / "probe" / f"{stem}_activations.npz")
            assert activation["token_ids"].tolist() == record["token_ids"]
            assert activation["hidden"].shape == (len(record["token_ids"]), 4, 1024)
            assert np.isfinite(activation["hidden"]).all()
            probe_episodes += 1
    result = {"ok": True, "episodes": len(expected), "generated_tokens": token_total,
              "probe_episodes": probe_episodes,
              "adapter_sha256": sha256(adapter), "training_base_unchanged": True}
    audit_name = "audit.json" if evaluation_subdir == "evaluation" else f"audit_{evaluation_subdir}.json"
    write_json(run / audit_name, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=HERE / "run")
    parser.add_argument("--evaluation-subdir", default="evaluation")
    args = parser.parse_args()
    print(audit(args.run, args.evaluation_subdir))
