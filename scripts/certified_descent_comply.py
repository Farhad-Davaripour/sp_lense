"""Fresh-zero certified-descent runtime; no real run without a separate lock authority."""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paired_common_drift_comply_1800_binding import load_bound

bound = load_bound(
    "scripts._certified_descent_comply_runtime_shell",
    "scripts/paired_common_drift_comply.py",
    "1c6f375facf793109ede21f8f2e067aed1ffcef343750a463ffdd6278d774aa6",
    (
        (
            "from scripts import paired_common_drift_comply_optimizer as optimizer",
            "from scripts import certified_descent_comply_optimizer as optimizer",
            1,
        ),
        (
            "from scripts import paired_common_drift_comply_plan as protocol",
            "from scripts import certified_descent_comply_plan as protocol",
            1,
        ),
        (
            "from scripts import paired_common_drift_comply_recording as recording",
            "from scripts import certified_descent_comply_recording as recording",
            1,
        ),
        (
            "from scripts.verify_paired_common_drift_comply import",
            "from scripts.verify_certified_descent_comply import",
            2,
        ),
        ("scripts._paired_comply_runtime", "scripts._certified_descent_comply_runtime", 1),
        ("paired_objective_trajectory", "paired_response_trajectory", 1),
        ("1200", "1800", 6),
        ('value["standard_used_percent"] < 90', 'value["standard_used_percent"] <= 100', 1),
        (
            "finite fresh standard usage below90 required",
            "finite fresh standard usage at most100 required",
            1,
        ),
    ),
)


def drive(plan, call, skip, propose, save_update, save_endpoint):
    """Same 216/96/eight-stage conditional schedule, with exact path/history binding."""
    require = bound.require
    cells = {(c["prompt_id"], c["condition"]): c for c in plan["cells"]}
    ids = plan["construction_ids"]
    require(
        len(ids) == 12 and not plan["control_ids"] and not plan["transfer_ids"], "12 training only"
    )
    w, path, path_upper = [0.0] * plan["model"]["d_model"], 0.0, "0"
    history = [w[:]]
    current = [call(cells[pid, "baseline"], w, None) for pid in ids]
    baselines = list(current)
    stop, applied, attempted = bound.stopping(current), 0, 0
    anchor = current[-1]["row"]["cell_id"]
    for stage in range(1, 9):
        gc = [cells[pid, f"gradient_{stage}"] for pid in ids]
        sc = [cells[pid, f"step_{stage}"] for pid in ids]
        if stop:
            for cell in gc + sc:
                skip(cell, stop, anchor)
            continue
        attempted += 1
        gradients = [call(c, w, state) for c, state in zip(gc, current, strict=True)]
        anchor = gradients[-1]["row"]["cell_id"]
        if any(not bound.quality(state["row"]) for state in gradients):
            stop = "quality_failure"
        else:
            proposal = propose(
                gradients,
                w,
                path,
                stage,
                baselines,
                history=[point[:] for point in history],
                path_upper=path_upper,
            )
            save_update(proposal)
            require(
                not proposal.get("technical_failure", False),
                "technical/incomplete local solve; inconclusive without more forwards",
            )
            require(
                proposal["status"] in ("ready", "exact_zero_optimum", "no_certified_step"),
                "one declared dyadic disposition; no alternative optimizer",
            )
            if proposal["status"] != "ready":
                stop = proposal["status"]
            else:
                next_path = Fraction(proposal["path_after_upper"])
                require(
                    Fraction(path_upper) <= next_path <= Fraction(2, 5)
                    and proposal["path_after"] == float(next_path)
                    and proposal["step_norm"] <= 0.05 + bound.ROUND_EPS
                    and proposal["net_norm"] <= 0.20 + bound.ROUND_EPS,
                    "certified path chain and descriptive step/net bounds",
                )
                w, path, path_upper = (
                    proposal["w_after"],
                    proposal["path_after"],
                    proposal["path_after_upper"],
                )
                history.append(w[:])
                applied += 1
        if stop:
            for cell in sc:
                skip(cell, stop, anchor)
            continue
        current = [call(c, w, state) for c, state in zip(sc, current, strict=True)]
        anchor, stop = current[-1]["row"]["cell_id"], bound.stopping(current)
    endpoint = {
        "w": w,
        "vector_float64_le_sha256": bound.vector_sha(w),
        "path": path,
        "path_upper": path_upper,
        "history_w_sha256": [bound.vector_sha(point) for point in history],
        "net": bound.norm(w),
        "updates": applied,
        "attempted_updates": attempted,
        "stop_reason": stop or "max_updates",
        "endpoint_cell_ids": [state["row"]["cell_id"] for state in current],
    }
    save_endpoint(endpoint)
    final = [call(cells[pid, "final"], w, state) for pid, state in zip(ids, current, strict=True)]
    return {
        **endpoint,
        "final_cell_ids": [state["row"]["cell_id"] for state in final],
        "transfer_ran": False,
        "candidate_eligible": all(bound.accepts(state["row"]) for state in final),
    }


bound.drive = drive
bound.DRIVER_OVERRIDE = {
    "path": "scripts/certified_descent_comply.py",
    "change": "exact conservative path/history chaining and three certified-descent stop dispositions",
    "model_prompts_hooks_and_counters_changed": False,
}

if __name__ == "__main__":
    commands = {
        "freeze": bound.freeze,
        "preflight": bound.preflight,
        "run": bound.run,
        "_worker": bound.worker,
    }
    bound.require(
        len(sys.argv) == 2 and sys.argv[1] in commands,
        "Use freeze/preflight/run; no recipe switches",
    )
    result = commands[sys.argv[1]]()
    print(json.dumps(result, indent=2, allow_nan=False))
    if sys.argv[1] == "run" and result["status"] != "complete_valid":
        raise SystemExit(1)
else:
    sys.modules[__name__] = bound
