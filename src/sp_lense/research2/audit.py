"""Independently recompute pilot comparisons from saved per-view GPU outputs."""

import argparse
import hashlib
import json
from pathlib import Path

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.metrics import evaluate, key, summarize, transfer_recovery
from sp_lense.research2.report import write
from sp_lense.steering.gated import classifier_metrics, require


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(output):
    output = Path(output)
    native = json.loads((output / "METRICS.json").read_text())
    require(native["state"] == "completed", "Incomplete native execution")
    manifest = json.loads((output / "ARTIFACTS.json").read_text())
    for name, digest in manifest.items():
        path = (output / name).resolve()
        require(path.is_relative_to(output.resolve()), "Invalid output manifest path")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Hash mismatch: {name}")
    result = {
        "state": "completed",
        "runtime": native["runtime"],
        "splits": {},
        "transfer": {},
        "elapsed_seconds": native["elapsed_seconds"],
        "scored_forwards": native["scored_forwards"],
        "training_forwards": native["training_forwards"],
        "interpretation": native["interpretation"],
        "native_manifest": manifest,
    }
    for split in ("validation", "holdout"):
        base = read_rows(output / f"{split}_base.jsonl")
        teacher = read_rows(output / f"{split}_teacher.jsonl")
        cases = json.loads((ROOT / f"data/{split}.json").read_text())["cases"]
        require(
            {key(r) for r in base} == {(c["case_id"], o) for c in cases for o in ("AB", "BA")},
            "Evaluation case/order set mismatch",
        )
        result["splits"][split] = evaluate(base, teacher)
        require(
            result["splits"][split]["methods"] == native["splits"][split]["methods"],
            "Native teacher metrics disagree with replay",
        )
        result["splits"][split]["detector"] = classifier_metrics(
            cases, {r["case_id"]: r["gate_probability"] for r in base}
        )
        original = read_rows(ROOT / f"study/guarded_steering/{split}.jsonl")
        result["splits"][split]["research1"] = summarize(
            base, original, {key(r) for r in original if r["applied_strength"] != 0}
        )
        if native["gate1_pass"]:
            result["transfer"][split] = {}
            for method in ("oracle", "training_mean", "random"):
                patched = read_rows(output / f"{split}_{method}.jsonl")
                calculation = evaluate(base, patched)
                require(
                    calculation["methods"] == native["transfer"][split][method]["methods"],
                    "Native transfer metrics disagree with replay",
                )
                result["transfer"][split][method] = calculation | {
                    "teacher_recovery": transfer_recovery(base, teacher, patched)
                }
    result["gate1_pass"] = result["splits"]["validation"]["gate1_pass"]
    require(result["gate1_pass"] == native["gate1_pass"], "Continuation gate changed")
    result["reporting_note"] = (
        "Research 1 accepted-intervention counts use recorded nonzero selections. Raw LoRA counts describe enabled views. Native GPU receipt is preserved."
    )
    result["provenance"] = {
        "baseline_commit": "cff7542cd0f9abd16c1ed9d590d168079ab6dc94",
        "input_manifest": "../INPUT_PINS.json",
        "native_execution_metrics": "../METRICS.json",
        "adapter_checkpoint": "../adapter/adapter_model.safetensors",
        "training_receipt": "../TRAINING.json",
        "validation_decision_before_holdout": "../VALIDATION_DECISION.json",
        "baseline_smoke": "../SMOKE.json",
        "relative_paths_from": "audited/",
    }
    target = output / "audited"
    target.mkdir(exist_ok=True)
    (target / "METRICS.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    write(target)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    result = audit(parser.parse_args().output)
    print(json.dumps({"state": result["state"], "gate1_pass": result["gate1_pass"]}))
