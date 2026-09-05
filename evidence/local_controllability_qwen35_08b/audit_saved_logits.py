"""Post-exposure arithmetic diagnosis, not a replacement preregistered verifier.

Uses saved arrays only: no model, tokenizer, forward, derivative, fit, or threshold change.
"""

import hashlib
import json
import math
from pathlib import Path
import sys
import zlib

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import verify_local_controllability as frozen

OUTPUT = Path(__file__).resolve().parent


def main():
    rows = frozen.rows_at(OUTPUT / "rows.jsonl")
    prereg = frozen.read(OUTPUT / "preregistration.json")
    for path, digest in prereg["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    frozen.verify_journal(OUTPUT / "forward_events.jsonl", prereg["plan"]["cells"])
    frozen.verify_journal(OUTPUT / "derivative_events.jsonl", prereg["plan"]["derivative_cells"])
    baseline = {}
    diagnostic = []
    for row in rows:
        logits64 = frozen.read_logits(OUTPUT, row)
        logp64, probs64 = frozen.distribution(logits64)
        raw = zlib.decompress((OUTPUT / row["logits_file"]).read_bytes())
        logits32 = torch.frombuffer(bytearray(raw), dtype=torch.float32)
        logp32 = torch.log_softmax(logits32, dim=-1)
        probs32 = logp32.exp()
        pi = row["choice_a_token_id"] if row["preserve_label"] == "A" else row["choice_b_token_id"]
        ci = row["choice_b_token_id"] if row["preserve_label"] == "A" else row["choice_a_token_id"]
        if row["condition"] == "baseline":
            baseline[row["prompt_id"]] = logp32, logp64
        b32, b64 = baseline[row["prompt_id"]]
        f32_values = {
            "answer_pair_mass": float((probs32[pi] + probs32[ci]).item()),
            "preserve_log_odds": float((logits32[pi] - logits32[ci]).item()),
            "preserve_pair_probability": float(torch.softmax(torch.stack([logits32[pi], logits32[ci]]), dim=0)[0].item()),
            "kl_from_baseline": float((probs32 * (logp32 - b32)).sum().item()),
        }
        f64_mass = probs64[pi] + probs64[ci]
        f64_kl = math.fsum(p * (lp - bp) for p, lp, bp in zip(probs64, logp64, b64, strict=True))
        diagnostic.append({
            "cell_id": row["cell_id"],
            "float32_reproduction_exact": all(row[k] == v for k, v in f32_values.items()),
            "float32_absolute_errors": {k: abs(row[k] - v) for k, v in f32_values.items()},
            "saved_pair_mass": row["answer_pair_mass"],
            "float64_pair_mass": f64_mass,
            "pair_mass_absolute_difference": abs(row["answer_pair_mass"] - f64_mass),
            "frozen_pair_mass_arithmetic_check_passes": math.isclose(row["answer_pair_mass"], f64_mass, rel_tol=frozen.ARITHMETIC_TOL, abs_tol=frozen.ARITHMETIC_TOL),
            "saved_kl": row["kl_from_baseline"],
            "float64_kl": f64_kl,
            "kl_absolute_difference": abs(row["kl_from_baseline"] - f64_kl),
            "frozen_kl_arithmetic_check_passes": math.isclose(row["kl_from_baseline"], f64_kl, rel_tol=frozen.ARITHMETIC_TOL, abs_tol=frozen.ARITHMETIC_TOL),
        })
    result = {
        "status": "supplemental_post_exposure_arithmetic_diagnosis_only",
        "final_verification_status": "INCONCLUSIVE",
        "frozen_verifier_or_threshold_modified": False,
        "additional_model_calls": 0,
        "source_hashes_unchanged": True,
        "journals_verified": {"forwards": 40, "derivatives": 4},
        "all_float32_scores_reproduced_exactly": all(d["float32_reproduction_exact"] for d in diagnostic),
        "pair_mass_frozen_arithmetic_failures": sum(not d["frozen_pair_mass_arithmetic_check_passes"] for d in diagnostic),
        "kl_frozen_arithmetic_failures": sum(not d["frozen_kl_arithmetic_check_passes"] for d in diagnostic),
        "maximum_pair_mass_float64_difference": max(d["pair_mass_absolute_difference"] for d in diagnostic),
        "maximum_kl_float64_difference": max(d["kl_absolute_difference"] for d in diagnostic),
        "rows": diagnostic,
    }
    with (OUTPUT / "ARITHMETIC_DIAGNOSIS.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    analysis = frozen.read(OUTPUT / "analysis.json")
    for kind in ("local", "shared"):
        print(kind, json.dumps({k: v for k, v in analysis[kind].items() if k != "cells"}))
        if kind == "local":
            for cell in analysis[kind]["cells"]:
                print(json.dumps(cell))


if __name__ == "__main__":
    main()
