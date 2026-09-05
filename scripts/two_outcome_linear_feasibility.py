"""Exactly two fixed-outcome convex problems using the unchanged active-set solver."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import shared_direction_linear_feasibility as engine
from scripts import two_outcome_feasibility_io as io

solve = engine.solve


def formulation(records, outcome, margin):
    io.require(outcome in io.OUTCOMES, "fixed outcome only")
    sign = 1 if outcome == "preserve" else -1
    A = []
    for row in records:
        scale = sign * engine.norm(row["h0"])
        A.append([scale * x for x in row["g"]])
    return A, [margin - sign * row["S"] for row in records]


def table(records, vector, outcome, margin):
    sign = 1 if outcome == "preserve" else -1
    output = []
    for row in records:
        effect = (
            engine.norm(row["h0"]) * engine.dot(row["g"], vector) if vector is not None else None
        )
        predicted = sign * (row["S"] + effect) if effect is not None else None
        output.append(
            {k: row[k] for k in ("dataset", "order", "prompt_id", "initial_gradient_cell_id")}
        )
        output[-1].update(
            S=row["S"],
            outcome=outcome,
            rhs=margin - sign * row["S"],
            baseline_already_correct=sign * row["S"] >= margin,
            predicted_signed_margin=predicted,
            slack=predicted - margin if predicted is not None else None,
        )
    return output


def calculate(config, loader, freezer):
    io.require(
        config["construction"] == ["f01_v1", "f01_v2"]
        and config["descriptive_comparison"] == ["f02_v1"]
        and config["outcomes"] == list(io.OUTCOMES),
        "fixed split and two outcomes",
    )
    records = [row for key in config["construction"] for row in loader(key)]
    io.require(
        [(r["dataset"], r["order"]) for r in records]
        == [(key, order) for key in config["construction"] for order in config["order"]],
        "all four construction rows",
    )
    outcomes, frozen = {}, {}
    for outcome in io.OUTCOMES:
        A, b = formulation(records, outcome, config["margin"])
        result = solve(A, b, config)
        solution = result["solution"]
        vector = solution["vector"] if solution else None
        frozen[outcome] = {
            "outcome": outcome,
            "status": result["status"],
            "solution": solution,
            "vector_float64_le_sha256": io.sha(struct.pack(f"<{len(vector)}d", *vector))
            if vector is not None
            else None,
            "construction": [
                {k: row[k] for k in ("dataset", "prompt_id", "initial_gradient_cell_id")}
                for row in records
            ],
            "hypothetical_edit": "norm(h0_i)*this_vector; no extra sign",
            "role": "unapplied mathematical diagnostic; no normalization/clipping/controller",
            "model_calls": 0,
        }
        outcomes[outcome] = {
            "status": result["status"],
            "active_sets": result["active_sets"],
            "construction_rows": table(records, vector, outcome, config["margin"]),
        }
    hashes = freezer(frozen)  # BOTH exclusive durable records must precede ANY f02 numbers.
    io.require(set(hashes) == set(io.OUTCOMES), "both outcome records frozen")
    exposed = (
        [row for key in config["descriptive_comparison"] for row in loader(key)]
        if any(frozen[o]["solution"] is not None for o in io.OUTCOMES)
        else []
    )
    for outcome in io.OUTCOMES:
        solution = frozen[outcome]["solution"]
        outcomes[outcome]["exposed_f02_rows"] = table(
            exposed, solution["vector"] if solution else None, outcome, config["margin"]
        )
    return {
        "status": "TWO_SEPARATE_KKT_ESTIMATES_PENDING_AUDIT",
        "solution_sha256": hashes,
        "outcomes": outcomes,
        "model_calls": 0,
        "causal_test_performed": False,
    }


def analyze():
    config = io.locked()["config"]

    def freezer(values):
        hashes = {}
        for outcome in io.OUTCOMES:
            path = io.OUTPUT / (outcome + "_solution.json")
            io.write_new(path, values[outcome])
            hashes[outcome] = io.sha(path.read_bytes())
        io.write_new(
            io.OUTPUT / "construction_frozen.json",
            {"solution_sha256": hashes, "f02_numeric_loading_started": False},
        )
        return hashes

    result = calculate(config, lambda key: io.initial_rows(key, config), freezer)
    io.write_new(io.OUTPUT / "analysis.json", result)


if __name__ == "__main__":
    if sys.argv[1:] == ["freeze"]:
        print(json.dumps(io.freeze(), indent=2))
    elif sys.argv[1:] == ["run"]:
        status = io.launch(io.SCRIPT, "_analyze")
        print(json.dumps(status, indent=2))
        if status["status"] != "completed":
            raise SystemExit(1)
    elif sys.argv[1:] == ["_analyze"]:
        analyze()
    else:
        raise SystemExit("Use freeze or run; no adjustable outcomes/inputs.")
