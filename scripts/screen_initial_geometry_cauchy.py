"""One exact saved-initial-row Cauchy screen; no solver, model or full re-audit."""

import hashlib
import json
import math
import struct
import subprocess
import sys
from fractions import Fraction as F
from pathlib import Path

COMMIT = "d3338172b1e4560b74f28f14652c219fe4ac279b"
NAMESPACE = "evidence/soft_drift_constrained_comply_v1_qwen35_08b"
ROOT = Path(__file__).resolve().parents[1]
SCALE = 10**12


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def rational(value):
    assert type(value) in (float, int) and math.isfinite(value)
    return F.from_float(float(value))


def norm_enclosure(square):
    assert square >= 0
    k = math.isqrt(square.numerator * SCALE * SCALE // square.denominator)
    exact = k * k * square.denominator == square.numerator * SCALE * SCALE
    upper = k if exact else k + 1
    assert F(k, SCALE) ** 2 <= square <= F(upper, SCALE) ** 2

    def decimal(integer):
        return f"{integer // SCALE}.{integer % SCALE:012d}"

    return {"lower": decimal(k), "upper": decimal(upper)}, F(upper, SCALE)


def main():
    source = {}
    for name in ("rows.jsonl", "updates.jsonl", "preregistration.json"):
        path = NAMESPACE + "/" + name
        assert git("hash-object", "--no-filters", "--", path) == git(
            "rev-parse", COMMIT + ":" + path
        )
        source[name] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    p = ROOT / NAMESPACE
    plan = json.loads((p / "preregistration.json").read_bytes())["plan"]
    ids = plan["construction_ids"]
    assert len(ids) == len(set(ids)) == 12
    base, gradients = {}, {}
    with (p / "rows.jsonl").open() as stream:
        for consumed, line in enumerate(stream, 1):
            row = json.loads(line)
            assert row["condition"] in {"baseline", "gradient_1"}
            target = base if row["condition"] == "baseline" else gradients
            assert row["prompt_id"] not in target
            target[row["prompt_id"]] = row
            if len(gradients) == 12:
                break
    assert consumed == 24 and set(base) == set(gradients) == set(ids)
    with (p / "updates.jsonl").open() as stream:
        update = json.loads(next(stream))
    assert update["stage"] == 1 and update["path_before"] == 0
    assert len(update["w_before"]) == 1024 and all(x == 0 for x in update["w_before"])
    assert update["gradient_cell_ids"] == [gradients[pid]["cell_id"] for pid in ids]
    # The slightly larger binary64 radius makes a violation sufficient for
    # BOTH literal-implementation and exact-decimal .05 cap interpretations.
    rho_decimal, rho_binary = F(1, 20), rational(0.05)
    assert rho_binary >= rho_decimal
    rows = []
    for index, pid in enumerate(ids):
        row, baseline = gradients[pid], base[pid]
        assert row["current_cell_id"] == baseline["cell_id"]
        assert row["preserve_log_odds"] == baseline["preserve_log_odds"]
        assert row["h0_norm"] == baseline["h0_norm"] > 0
        assert len(row["shared_w"]) == 1024 and all(x == 0 for x in row["shared_w"])
        assert len(row["gradient"]) == 1024
        n, S = float(row["h0_norm"]), float(row["preserve_log_odds"])
        exact_A = [-rational(n) * rational(g) for g in row["gradient"]]
        A = [-n * float(g) for g in row["gradient"]]
        rounded_A = [rational(x) for x in A]
        A_hash = hashlib.sha256(struct.pack("<1024d", *A)).hexdigest()
        b_float = 0.10 - (-S)
        assert A_hash == update["qp_inputs"]["A_row_sha256"][index]
        assert b_float == update["qp_inputs"]["b"][index]
        b_rounded, b_expression = rational(b_float), F(1, 10) + rational(S)
        square = sum((x * x for x in rounded_A), F(0))
        square_expression = sum((x * x for x in exact_A), F(0))
        norm_bounds, norm_upper = norm_enclosure(square)
        squared_gap = b_rounded**2 - rho_binary**2 * square
        violates = b_rounded > 0 and squared_gap > 0
        exact_expression_violates = (
            b_expression > 0 and b_expression**2 > rho_binary**2 * square_expression
        )
        lower_gap = b_rounded - rho_binary * norm_upper
        if violates:
            assert lower_gap > 0  # the readable norm enclosure ALSO proves it
        rows.append(
            {
                "index": index + 1,
                "prompt_id": pid,
                "family_id": row["family_id"],
                "order": row["order"],
                "display_order": row["display_order"],
                "current_comply_margin": -S,
                "b": b_float,
                "b_exact_fraction": str(b_rounded),
                "A_row_sha256": A_hash,
                "A_norm_squared_exact": str(square),
                "A_norm_enclosure": norm_bounds,
                "radius_times_norm_upper_exact": str(rho_binary * norm_upper),
                "cauchy_gap_lower_exact": str(lower_gap),
                "squared_gap_exact": str(squared_gap),
                "single_row_infeasibility_certified": violates,
                "exact_product_rational_target_crosscheck_violates": exact_expression_violates,
                "assembly_classifications_agree": violates == exact_expression_violates,
            }
        )
    violations = [r["index"] for r in rows if r["single_row_infeasibility_certified"]]
    assert all(r["assembly_classifications_agree"] for r in rows)
    assert not any(
        m.split(".")[0] in {"torch", "transformers", "transformer_lens", "numpy", "scipy"}
        for m in sys.modules
    )
    return {
        "status": "INITIAL_LOCAL_SUBPROBLEM_INFEASIBLE_CERTIFIED"
        if violations
        else "CAUCHY_SCREEN_INCONCLUSIVE",
        "artifact_commit": COMMIT,
        "namespace": NAMESPACE,
        "input_sha256": source,
        "rows_numerically_read": 24,
        "update_records_read": 1,
        "full_file_hashes_for_identity_only": True,
        "w_initial_zero": True,
        "path_initial": 0,
        "step_radius_exact_decimal": str(rho_decimal),
        "step_radius_binary64_exact": str(rho_binary),
        "initial_net_and_remaining_path_radii": ["1/5", "2/5"],
        "effective_initial_radius_is_step": True,
        "proof": "For b_i>0, b_i^2>rho^2*sum_j A_ij^2 contradicts A_i*r>=b_i and ||r||<=rho by Cauchy-Schwarz.",
        "violating_row_indices": violations,
        "rows": rows,
        "fraction_arithmetic": True,
        "integer_isqrt_readable_enclosures": True,
        "solver_or_witness_search": False,
        "full_historical_reaudit": False,
        "model_loads": 0,
        "real_forwards": 0,
        "real_derivatives": 0,
        "nonlinear_multistep_behavioral_or_overall_impossibility_claim": False,
    }


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))
