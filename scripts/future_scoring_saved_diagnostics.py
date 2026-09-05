"""Supplemental saved-array numerics and scalar feasibility only; no model/tokenizer calls.

Never constructs/applies a successor edit. Never overwrites original scores/verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import zlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.future_choice_scoring_reference import NUMERIC_FIELDS, verify_record
from sp_lense.future_choice_scoring import score_float32_logits

ORIGINAL = ROOT / "evidence/local_controllability_qwen35_08b"
OUTPUT = ROOT / "evidence/future_scoring_numerical_amendment/canonical_float64_diagnostics.json"
REQUESTED_MARGIN = 0.05  # Proposal only; small, fixed positive pair-logit margin.
PROPOSED_RADIUS_CAP = 0.10  # Proposal only; no successor edit is constructed here.


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if OUTPUT.exists():
        raise FileExistsError("supplemental artifact already exists; no overwrite")
    original_hashes = {
        str(p.relative_to(ORIGINAL)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ORIGINAL.rglob("*"))
        if p.is_file()
    }
    prereg = read(ORIGINAL / "preregistration.json")
    for path, digest in prereg["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert read(ORIGINAL / "VERIFICATION_FAILURE.json")["classification"] == "INCONCLUSIVE"
    rows = [
        json.loads(line)
        for line in (ORIGINAL / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 40 and [r["cell_id"] for r in rows] == [
        c["cell_id"] for c in prereg["plan"]["cells"]
    ]
    baseline = {}
    canonical = []
    maximum_error = dict.fromkeys(NUMERIC_FIELDS, 0.0)
    for row in rows:
        path = (ORIGINAL / row["logits_file"]).resolve()
        assert path.is_relative_to((ORIGINAL / "logits").resolve())
        raw = zlib.decompress(path.read_bytes())
        assert (
            len(raw) == row["logit_count"] * 4
            and hashlib.sha256(raw).hexdigest() == row["logits_sha256"]
        )
        logits = torch.frombuffer(bytearray(raw), dtype=torch.float32)
        if row["condition"] == "baseline":
            baseline[row["prompt_id"]] = logits, row
        blogits, baseline_row = baseline[row["prompt_id"]]
        assert row["baseline_cell_id"] == baseline_row["cell_id"]
        assert row["baseline_argmax_id"] == int(blogits.argmax().item())
        args = {
            "choice_a_token_id": row["choice_a_token_id"],
            "choice_b_token_id": row["choice_b_token_id"],
            "preserve_label": row["preserve_label"],
        }
        measured = score_float32_logits(torch, logits, blogits, **args)
        errors = verify_record(measured, logits.tolist(), blogits.tolist(), **args)
        for key, value in errors.items():
            maximum_error[key] = max(maximum_error[key], value)
        # Exact array ordering/argmax does not depend on probability normalizers.
        assert measured["actual_next_token_id"] == row["actual_next_token_id"]
        canonical.append(
            {
                "cell_id": row["cell_id"],
                "prompt_id": row["prompt_id"],
                "condition": row["condition"],
                "original_raw_logits_sha256": row["logits_sha256"],
                "metrics": measured,
                "reference_absolute_errors": errors,
            }
        )
    by_cell = {r["cell_id"]: r["metrics"] for r in canonical}
    facts = {}
    for kind in ("local", "shared"):
        selected = [r for r in rows if r["condition"].startswith(kind + "_")]
        facts[kind] = {
            "target_cells": len(selected),
            "baseline_opposed_requests": sum(
                r["baseline_argmax_id"] != r["requested_token_id"] for r in selected
            ),
            "argmax_changes_from_baseline": sum(
                by_cell[r["cell_id"]]["actual_next_token_id"] != r["baseline_argmax_id"]
                for r in selected
            ),
            "new_requested_argmax_flips": sum(
                by_cell[r["cell_id"]]["actual_next_token_id"] == r["requested_token_id"]
                and r["baseline_argmax_id"] != r["requested_token_id"]
                for r in selected
            ),
            "requested_argmax_retentions": sum(
                by_cell[r["cell_id"]]["actual_next_token_id"]
                == r["requested_token_id"]
                == r["baseline_argmax_id"]
                for r in selected
            ),
        }
    feasibility = []
    for row in rows:
        if row["condition"] != "gradient":
            continue
        gnorm = math.sqrt(math.fsum(x * x for x in row["gradient"]))
        hnorm = math.sqrt(math.fsum(x * x for x in row["hidden_before"]))
        margin = by_cell[row["baseline_cell_id"]]["preserve_log_odds"]
        assert gnorm > 1e-12 and hnorm > 0
        for requested, sign in (("preserve", 1), ("comply", -1)):
            deficit = max(0.0, REQUESTED_MARGIN - sign * margin)
            required = deficit / (gnorm * hnorm)
            feasibility.append(
                {
                    "prompt_id": row["prompt_id"],
                    "variant_id": row["variant_id"],
                    "order": row["order"],
                    "requested": requested,
                    "baseline_signed_margin": sign * margin,
                    "gradient_norm": gnorm,
                    "hidden_norm": hnorm,
                    "requested_signed_margin": REQUESTED_MARGIN,
                    "first_order_required_relative_radius": required,
                    "proposed_cap_would_bind": required > PROPOSED_RADIUS_CAP,
                    "proposed_noop": deficit == 0,
                    "first_order_signed_margin_at_proposed_cap": sign * margin
                    + min(required, PROPOSED_RADIUS_CAP) * gnorm * hnorm,
                }
            )
    result = {
        "status": "supplemental_only_not_reclassification",
        "original_verdict_unchanged": "INCONCLUSIVE",
        "original_verifier_unchanged": True,
        "original_arithmetic_tolerance_unchanged": 2e-5,
        "new_contract_arithmetic_tolerance": 2e-5,
        "model_calls": 0,
        "tokenizer_calls": 0,
        "derivative_calls": 0,
        "all_40_canonical_float64_records_match_independent_reference": True,
        "maximum_reference_absolute_errors": maximum_error,
        "exact_saved_array_argmax_facts": facts,
        "proposal_only_feasibility": {
            "requested_margin": REQUESTED_MARGIN,
            "radius_cap": PROPOSED_RADIUS_CAP,
            "informed_by_exposed_discovery": True,
            "successor_recipe_implemented": False,
            "cases": feasibility,
        },
        "original_namespace_sha256": original_hashes,
        "canonical_rows": canonical,
    }
    assert original_hashes == {
        str(p.relative_to(ORIGINAL)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ORIGINAL.rglob("*"))
        if p.is_file()
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("canonical_rows", "original_namespace_sha256")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
