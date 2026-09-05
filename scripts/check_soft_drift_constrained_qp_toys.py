"""Exact hand-supplied 2D design toys; no QP solver, model or repository-data reads."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = "docs/SOFT_DRIFT_CONSTRAINED_QP_DESIGN.md"
REPORT = ROOT / "docs/soft_drift_constrained_qp_toy_checks.json"
ZERO = (F(0), F(0))
TAU, STEP, PAIRS = F(1, 10), F(1, 20), 6
KAPPA = STEP**2 / (PAIRS * TAU**2)


def dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), F(0))


def problem(pairs):
    """Only 2D abstract toys with supplied rows; pad to the declared six-pair weighting."""
    assert len(pairs) <= 6
    pairs = pairs + [(ZERO, ZERO, F(-1), F(-1), F(0))] * (6 - len(pairs))
    A, b, c, D = [], [], [], []
    for aa, ab, ba, bb, drift in pairs:
        A.extend((aa, ab))
        b.extend((ba, bb))
        c.append(drift)
        D.append(tuple((x - y) / 2 for x, y in zip(aa, ab, strict=True)))
    assert len(A) == len(b) == 12 and len(c) == len(D) == 6
    return A, b, c, D


def kkt(given, d, multipliers, penalty=F(1)):
    """Verify a hand-written answer exactly; do not search, optimize or invert matrices."""
    A, b, c, D = given
    assert len(d) == 2 and len(multipliers) == 12
    residuals = [dot(a, d) - rhs for a, rhs in zip(A, b, strict=True)]
    drift = [cp + dot(dp, d) for cp, dp in zip(c, D, strict=True)]
    gradient = [
        d[j] + penalty * KAPPA * sum((dp[j] * cp for dp, cp in zip(D, drift, strict=True)), F(0))
        for j in range(2)
    ]
    stationarity = [
        gradient[j] - sum((mu * a[j] for mu, a in zip(multipliers, A, strict=True)), F(0))
        for j in range(2)
    ]
    complementarity = [mu * r for mu, r in zip(multipliers, residuals, strict=True)]
    assert all(r >= 0 for r in residuals) and all(mu >= 0 for mu in multipliers)
    assert stationarity == [0, 0] and all(value == 0 for value in complementarity)
    value = dot(d, d) / 2 + penalty * KAPPA * sum((cp * cp for cp in drift), F(0)) / 2
    return {
        "status": "EXACT_KKT_MATCH",
        "hand_supplied_d": [str(x) for x in d],
        "minimum_primal_slack": str(min(residuals)),
        "stationarity": [str(x) for x in stationarity],
        "complementarity_max": "0",
        "objective_after_multiplication_by_step_squared": str(value),
        "residual_pair_drifts": [str(x) for x in drift],
    }


def run():
    started = time.perf_counter()
    assert not REPORT.exists(), "exclusive toy report; do not overwrite previous execution"
    assert KAPPA == F(1, 24)
    cases = {}
    aa = -F(2) * F(-1)
    ab_letter = -F(4) * F(1, 2)
    ab_semantic = -F(4) * F(-1, 2)
    dsmall = F(1, 100)
    assert (aa, ab_letter, ab_semantic) == (2, -2, 2)
    x, y = aa * dsmall, -ab_letter * dsmall
    assert ((x - y) / 2, (x + y) / 2) == (0, F(1, 50))
    x_sem, y_sem = aa * dsmall, -ab_semantic * dsmall
    assert ((x_sem - y_sem) / 2, (x_sem + y_sem) / 2) == (F(1, 50), 0)
    x_unequal, y_unequal = F(2) * dsmall, F(4) * dsmall
    assert ((x_unequal - y_unequal) / 2, (x_unequal + y_unequal) / 2) == (F(-1, 100), F(3, 100))
    cases["signs_own_norms_letter_semantic"] = {
        "status": "PASS",
        "own_norm_margin_jacobians": [str(aa), str(ab_letter)],
        "pure_letter_a_c": ["0", "1/50"],
        "pure_semantic_a_c": ["1/50", "0"],
        "unequal_sensitivity_a_c": ["-1/100", "3/100"],
        "mean_norm_substitution_allowed": False,
    }

    margins, baselines = (F(1, 25), F(1, 5)), (F(-3, 50), F(3, 10))
    rhs = tuple(TAU - m for m in margins)
    drift = ((margins[0] - baselines[0]) - (margins[1] - baselines[1])) / 2
    assert rhs == (F(3, 50), F(-1, 10)) and drift == F(1, 10)
    negative = problem([((F(1), F(0)), (F(-1), F(0)), *rhs, drift)])
    raw = (F(3, 50), F(0))
    cases["negative_rhs_soft_drift"] = kkt(negative, raw, [F(1, 15)] + [F(0)] * 11)
    assert margins[1] - raw[0] == F(7, 50) >= TAU
    assert drift + raw[0] == F(4, 25) > drift
    assert margins[0] + STEP == F(9, 100) < TAU
    cases["negative_rhs_soft_drift"].update(
        rhs=[str(x) for x in rhs],
        raw_local_margins=["1/10", "7/50"],
        clipped_first_margin="9/100",
        clamping_negative_rhs_would_require="3/50 <= d_x <= 0",
        soft_drift_can_increase=True,
        bounded_weakening_allowed=True,
    )

    affine = problem(
        [
            ((F(1), F(0)), (F(-1), F(0)), F(-1, 20), F(-1, 20), F(1)),
            ((F(0), F(4)), (F(0), F(4)), TAU, TAU, F(0)),
        ]
    )
    solution = (F(-1, 25), F(1, 40))
    dual = [F(0), F(0), F(1, 160)] + [F(0)] * 9
    cases["affine_current_drift"] = kkt(affine, solution, dual)
    assert dot(solution, solution) == F(89, 40000) < STEP**2
    assert F(3, 20) + solution[0] == F(11, 100)
    assert F(3, 20) - solution[0] == F(19, 100)
    split_dual = [F(0), F(0), F(1, 320), F(1, 320)] + [F(0)] * 8
    assert kkt(affine, solution, split_dual) == cases["affine_current_drift"]
    cases["affine_current_drift"].update(
        empty_set_unconstrained_d0=["-1/25", "0"],
        empty_set_violates_second_pair=True,
        dual_not_unique_primal_unique=True,
    )
    cases["zero_penalty_old_minimum_norm"] = kkt(affine, (F(0), F(1, 40)), dual, penalty=F(0))
    cases["zero_penalty_old_minimum_norm"]["algebraic_limit_only_not_a_sweep"] = True

    infeasible = problem([((F(1), F(0)), (F(-1), F(0)), TAU, TAU, F(0))])
    A, b, _, _ = infeasible
    y = [F(1), F(1)] + [F(0)] * 10
    assert all(sum((mu * a[j] for mu, a in zip(y, A, strict=True)), F(0)) == 0 for j in range(2))
    assert dot(y, b) == F(1, 5) > 0
    cases["infeasible_exact_farkas_witness"] = {
        "status": "PASS",
        "A_transpose_y": ["0", "0"],
        "b_dot_y": "1/5",
        "objective_cannot_change_feasibility": True,
        "absence_of_numerical_kkt_certificate_is_not_this_witness": True,
    }
    assert F(1, 26) < F(199, 1000) ** 2
    cases["net_projection_negative_rhs"] = {
        "status": "PASS",
        "w": ["1/5", "0"],
        "raw_d": ["0", "1/25"],
        "inequality": "d_x >= -1/1000",
        "post_projection_r_x": "1/sqrt(26)-1/5 < -1/1000",
        "exact_positive_square_comparison": "1/26 < (199/1000)^2",
        "radial_clipping_alone_preserves_feasible_negative_rhs": True,
    }

    assert not {"torch", "transformers", "transformer_lens", "tokenizers"} & set(sys.modules)
    result = {
        "status": "DESIGN_TOYS_EXACT_CHECKS_PASSED",
        "production_solver_implemented": False,
        "native_benchmark_run": False,
        "model_loads": 0,
        "tokenizer_loads": 0,
        "real_forwards": 0,
        "real_derivatives": 0,
        "penalty_weight": "1 (fixed)",
        "kappa": str(KAPPA),
        "pair_count_in_all_qp_toys": 6,
        "constraint_count": 12,
        "toy_dimension_only": 2,
        "numeric_elapsed_seconds": time.perf_counter() - started,
        "source_sha256": {
            str(Path(__file__).relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            DOC: hashlib.sha256((ROOT / DOC).read_bytes()).hexdigest(),
        },
        "cases": cases,
        "original_evidence_untouched": True,
        "original_recording_status": "INCONCLUSIVE ARTIFACT_BYTE_CAP",
        "scope": "New whole-method adaptive-development design; no candidate or isolated penalty-causality claim. STOP.",
    }
    raw_json = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    assert len(raw_json) < 16384
    with REPORT.open("xb") as stream:
        stream.write(raw_json)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": len(cases),
                "report_bytes": len(raw_json),
                "numeric_elapsed_seconds": result["numeric_elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    assert len(sys.argv) == 1, "fixed design toys only; no recipe switches"
    run()
