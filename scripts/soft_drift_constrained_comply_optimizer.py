"""One fixed soft-drift QP proposal plus original applied geometry; no model code."""

from __future__ import annotations

import hashlib
import math

from scripts import paired_common_drift_comply_optimizer as contracts
from scripts import soft_drift_qp_solver as solver
from scripts import verify_soft_drift_qp as numeric

require, vector, scalar = contracts.require, contracts.vector, contracts.scalar
norm, vector_sha = contracts.norm, contracts.vector_sha
PAIRS = ((2, 0), (3, 1), (6, 4), (7, 5), (10, 8), (11, 9))
ROUND_EPS = 1e-12


def objective(states, baselines):
    """Descriptive response trajectory, not the old squared-hinge objective."""
    rows, base = contracts._bound_rows(states, baselines)
    margins = [-scalar(row["preserve_log_odds"]) for row in rows]
    original = [-scalar(row["preserve_log_odds"]) for row in base]
    drift = [((margins[a] - original[a]) - (margins[b] - original[b])) / 2.0 for a, b in PAIRS]
    return {
        "comply_margins": margins,
        "baseline_relative_common_letter_drift": drift,
        "behavioral_acceptance_gate": False,
        "local_objective_descent_guaranteed": False,
    }


def increment(gradients, w, path, stage, baselines):
    require(type(stage) is int and 1 <= stage <= 8, "one of eight attempted updates")
    w, path = vector(w), scalar(path)
    require(
        0 <= path <= 0.40 + ROUND_EPS
        and norm(w) <= 0.20 + ROUND_EPS
        and norm(w) <= path + ROUND_EPS,
        "existing shared net/path bounds",
    )
    rows, base = contracts._bound_rows(gradients, baselines)
    require(rows[0]["shared_w"] == w, "gradient cache belongs to exact current w")
    for row in rows:
        condition = f"gradient_{stage}"
        previous = "baseline" if stage == 1 else f"step_{stage - 1}"
        require(
            row["condition"] == condition
            and row["stage"] == stage
            and row["cell_id"] == row["prompt_id"] + "__" + condition
            and row["current_cell_id"] == row["prompt_id"] + "__" + previous,
            "fresh current-gradient stage/cache binding",
        )
        require(
            0 <= scalar(row["maximum_current_logit_difference"]) <= 1e-6
            and 0 <= scalar(row["maximum_current_h_difference"]) <= 1e-6,
            "current-gradient identity arm",
        )
        vector(row["gradient"])
    inputs = solver.assemble(
        [row["h0_norm"] for row in rows],
        [row["gradient"] for row in rows],
        [-scalar(row["preserve_log_odds"]) for row in rows],
        [-scalar(row["preserve_log_odds"]) for row in base],
    )
    A, b, c = inputs["A"], inputs["b"], inputs["c"]
    solved = solver.solve(A, b, c)
    certificate = (
        numeric.verify(A, b, c, solved["solution"])
        if solved["solution"]
        else {"status": "NUMERICALLY_UNRESOLVED", "accepted": False, "reason": "NO_POINT"}
    )
    compact = {key: value for key, value in solved.items() if key not in {"solution", "mask_trace"}}
    compact["mask_trace_sha256"] = hashlib.sha256(solved["mask_trace"].encode()).hexdigest()
    compact["mask_trace"] = solved["mask_trace"]
    result = {
        "stage": stage,
        "gradient_cell_ids": [row["cell_id"] for row in rows],
        "w_before": w,
        "w_before_sha256": vector_sha(w),
        "path_before": path,
        "status": "NUMERICALLY_UNRESOLVED",
        "solver": compact,
        "numeric_certificate": certificate,
        "infeasibility_certified": False,
        "raw_feasibility_is_applied_feasibility": False,
    }
    if not certificate["accepted"]:
        return result
    point = solved["solution"]
    d = vector(point["vector"])
    dn = scalar(norm(d))
    clip = min(1.0, 0.05 / dn) if dn else 0.0
    s = vector([clip * x for x in d])
    u = vector([x + y for x, y in zip(w, s, strict=True)])
    un = scalar(norm(u))
    projection = 1.0 if un <= 0.20 else 0.20 / un
    after = list(u) if projection == 1.0 else vector([projection * x for x in u])
    actual = vector([x - y for x, y in zip(after, w, strict=True)])
    sn, rn, net = scalar(norm(s)), scalar(norm(actual)), scalar(norm(after))
    path_after = scalar(path + rn)
    require(
        rn <= 0.05 + ROUND_EPS and net <= 0.20 + ROUND_EPS and path_after <= 0.40 + ROUND_EPS,
        "actual post-projection step/net/path bounds",
    )
    D = [[(x - y) / 2.0 for x, y in zip(A[a], A[brow], strict=True)] for a, brow in PAIRS]
    dot = math.fsum
    predictions = {}
    margins = [-scalar(row["preserve_log_odds"]) for row in rows]
    for label, move in (("raw", d), ("clipped", s), ("applied", actual)):
        changes = [dot(x * y for x, y in zip(row, move, strict=True)) for row in A]
        predictions[label] = {
            "slacks": [scalar(change - rhs) for change, rhs in zip(changes, b, strict=True)],
            "predicted_comply_margins": [
                scalar(m + change) for m, change in zip(margins, changes, strict=True)
            ],
            "residual_common_letter_drift": [
                scalar(cp + dot(x * y for x, y in zip(dp, move, strict=True)))
                for cp, dp in zip(c, D, strict=True)
            ],
        }
    result.update(
        d=d,
        s=s,
        u=u,
        w_after=after,
        r=actual,
        w_after_sha256=vector_sha(after),
        d_norm=dn,
        clip_factor=clip,
        proposed_step_norm=sn,
        unprojected_net_norm=un,
        projection_factor=projection,
        projection_distance=scalar(norm([x - y for x, y in zip(u, after, strict=True)])),
        step_norm=rn,
        net_norm=net,
        path_after=path_after,
        proposed_path_after=scalar(path + sn),
        predictions=predictions,
        qp_witness={key: value for key, value in point.items() if key != "vector"},
        qp_inputs={
            "A_row_sha256": [vector_sha(row) for row in A],
            "b": b,
            "c": c,
            "D_row_sha256": [vector_sha(row) for row in D],
        },
        status="ready" if rn else "method_zero_increment" if dn == 0 else "projection_stall",
    )
    return result
