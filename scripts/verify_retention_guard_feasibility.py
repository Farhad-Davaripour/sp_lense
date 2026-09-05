"""Independent 80-digit reconstruction/certificates; never imports a QP solver."""

from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext

from scripts import retention_guard_feasibility_io as io
from scripts import verify_shared_direction_feasibility as dec


def rebuild(records, name):
    io.require(name in io.ORDER and len(records) == 8, "four fixed eight-row systems")
    target = 1 if name[-1] == "P" else -1
    floating_A, floating_b, A, b, c0, goals = [], [], [], [], [], []
    for row in records:
        c = target * row["S0"]
        rhs = 0.10 - c
        exact_rhs = dec.Interval("0.10") - dec.Interval(row["S0"]) * target
        guarded = name[:7] == "guarded"
        if guarded:
            rhs = max(0.0, rhs)
            exact_rhs = dec.Interval(max(Decimal(0), exact_rhs.lo), max(Decimal(0), exact_rhs.hi))
        hn = math.sqrt(math.fsum(x * x for x in row["h0"]))
        floating_A.append([target * hn * g for g in row["g"]])
        floating_b.append(rhs)
        exact_hn = dec.length(row["h0"])
        A.append([exact_hn * (target * g) for g in row["g"]])
        b.append(exact_rhs)
        c0.append(c)
        goals.append(max(0.10, c) if guarded else 0.10)
    R = max(
        1.0,
        *(abs(x) for x in floating_b),
        *(math.sqrt(math.fsum(x * x for x in a)) for a in floating_A),
    )
    tolerances = {
        "rank_relative_pivot_floor": 1e-12,
        "primal_absolute_tolerance": 1e-9 * R,
        "kkt_absolute_tolerance": 1e-8 * R * R,
    }
    return (
        {
            "name": name,
            "target_sign": target,
            "c0": c0,
            "A": floating_A,
            "b": floating_b,
            "scale": R,
            "endpoint_goals": goals,
            "tolerances": tolerances,
        },
        A,
        b,
    )


def certificate_policy(cert, config):
    # Independently check the categorical claim without rounding negative slack into feasibility.
    lower = cert["dual_radius_lower_bound_interval"]
    radius, guard = Decimal(str(config["radius"])), Decimal(config["radius_comparison_guard"])
    slack = [Decimal(x[0]) for x in cert["primal_residual_intervals"]]
    wn_hi = Decimal(cert["norm_interval"][1])
    nonnegative = cert["lambda_nonnegative"]
    bound_permitted = (
        nonnegative
        and Decimal(cert["dual_numerator_interval"][0]) > 0
        and Decimal(cert["dual_denominator_interval"][0]) > 0
    )
    decision = "NUMERICALLY_UNRESOLVED"
    if bound_permitted and lower is not None and Decimal(lower[0]) > radius + guard:
        decision = "OUTSIDE_RADIUS_LINEAR_SURROGATE_CERTIFIED"
    elif cert["kkt_verified"] and min(slack) >= 0 and wn_hi <= radius - guard:
        decision = "WITHIN_RADIUS_LINEAR_SURROGATE_VERIFIED"
    io.require(cert["decision"] == decision, "corrupt conservative certificate decision")
    return {
        "conservative_status": decision,
        "dual_radius_bound_permitted": bound_permitted,
        "minimum_primal_interval_lower": str(min(slack)),
        "negative_primal_interval_slack": min(slack) < 0,
        "numerical_within_radius_is_not_certified_feasibility": True,
    }


def verify_objective(item, records, config):
    expected, A, b = rebuild(records, item["name"])
    io.require(all(item[k] == value for k, value in expected.items()), "system/sign/RHS/scale")
    solved = item["solver"]
    logs = solved["active_sets"]
    io.require([x["mask"] for x in logs] == list(range(256)), "all256 masks in fixed order")
    valid_status = {
        "rank_deficient_or_near_dependent_skipped",
        "negative_or_nonfinite_multiplier_independent_set",
        "kkt_valid",
        "kkt_rejected",
    }
    for entry in logs:
        io.require(
            entry["active"] == [i for i in range(8) if entry["mask"] & (1 << i)]
            and entry["status"] in valid_status
            and all(math.isfinite(p) and p >= 0 for p in entry["pivots"]),
            "mask/rank log",
        )
    solution = solved["solution"]
    base = {
        "name": item["name"],
        "scale": expected["scale"],
        "tolerances": expected["tolerances"],
        "masks": 256,
        "row_ids": [r["prompt_id"] for r in records],
        "c0": expected["c0"],
        "b": expected["b"],
        "endpoint_goals": expected["endpoint_goals"],
    }
    if solution is None:
        io.require(
            solved["status"] == "NUMERICALLY_UNRESOLVED"
            and not any(x["status"] == "kkt_valid" for x in logs),
            "unresolved solve",
        )
        return {
            **base,
            "numeric_status": "NUMERICALLY_UNRESOLVED",
            "norm_estimate": None,
            "certificate": None,
            "conservative_status": "NUMERICALLY_UNRESOLVED",
        }
    mask = solution["active_mask"]
    io.require(
        solved["status"] == "KKT_ESTIMATE_ONLY" and type(mask) is int and 0 <= mask < 256,
        "numeric solution status/mask",
    )
    active = [i for i in range(8) if mask & (1 << i)]
    multipliers, vector = solution["multipliers"], solution["vector"]
    io.require(
        solution["active"] == active
        and len(multipliers) == 8
        and len(vector) == len(records[0]["g"])
        and all(math.isfinite(x) for x in multipliers + vector)
        and all(x >= 0 for x in multipliers)
        and all(multipliers[i] == 0 for i in range(8) if i not in active)
        and logs[mask]["status"] == "kkt_valid"
        and logs[mask]["metrics"] == solution["metrics"],
        "selected active set",
    )
    best = min(
        (entry for entry in logs if entry["status"] == "kkt_valid"),
        key=lambda e: (e["metrics"]["primal_objective"], e["mask"]),
    )
    io.require(best["mask"] == mask, "unchanged first-minimum tie policy")
    certificate_config = {**config, **expected["tolerances"]}
    cert = dec.certificate(A, b, vector, multipliers, certificate_config)
    dec.compare(solution["metrics"], cert["metrics"], 1e-9 * max(1.0, expected["scale"] ** 2))
    policy = certificate_policy(cert, config)
    estimate = solution["metrics"]["norm"]
    numeric_radius = (
        "WITHIN_RADIUS_NUMERIC_ESTIMATE_ONLY"
        if estimate <= 0.20 + 1e-12
        else "OVER_RADIUS_NUMERIC_ESTIMATE_ONLY"
    )
    return {
        **base,
        "norm_estimate": estimate,
        "numeric_radius_status": numeric_radius,
        "numeric_status": "INDEPENDENT_KKT_POLICY_MATCH"
        if cert["kkt_verified"]
        else "INDEPENDENT_KKT_UNRESOLVED",
        "active_constraints": active,
        "active_prompt_ids": [records[i]["prompt_id"] for i in active],
        "multipliers": multipliers,
        "solver_metrics": solution["metrics"],
        "certificate": cert,
        **policy,
        "solution_vector_float64_le_sha256": io.vector_sha(vector),
    }


def verify_events(events, started):
    expected = [(e, n) for n in io.ORDER for e in ("attempt", "complete")]
    expected.append(("attempt", "independent_audit"))
    io.require(
        [(e["event"], e["name"]) for e in events] == expected,
        "four solves then single audit attempt",
    )
    times = [e["monotonic"] for e in events]
    io.require(
        times == sorted(times)
        and all(started["started_monotonic"] <= t <= started["deadline_monotonic"] for t in times)
        and started["maximum_seconds"] == 180
        and started["deadline_monotonic"] - started["started_monotonic"] == 180
        and started["qp_ceiling"] == 4
        and started["audit_ceiling"] == 1
        and started["model_calls"] == started["derivatives"] == 0,
        "independent journal clock/budget",
    )


def verify(analysis, record):
    events = [
        json.loads(line)
        for line in (io.OUTPUT / "events.jsonl").read_text().splitlines()
    ]
    verify_events(events, io.read(io.OUTPUT / "RUN_STARTED.json"))
    records, paths = io.load_inputs(record["config"])
    io.require(
        records == record["selected_rows"]
        and io.canonical_sha(records)
        == record["selected_rows_sha256"]
        == analysis["selected_rows_sha256"],
        "independent archived input reconstruction",
    )
    io.require(
        all(record["sha256"][path] == digest for path, digest in paths.items()),
        "independent source/input binding",
    )
    io.require(
        analysis["solve_order"] == io.ORDER
        and analysis["qp_solves"] == 4
        and analysis["model_calls"] == analysis["derivatives"] == 0
        and [x["name"] for x in analysis["objectives"]] == io.ORDER,
        "four solves only",
    )
    with localcontext() as ctx:
        ctx.prec = 80
        results = [
            verify_objective(item, records, record["config"]) for item in analysis["objectives"]
        ]
    costs = []
    for target, first, second in (("P", 0, 1), ("C", 2, 3)):
        original, guarded = results[first], results[second]
        a, b = original["norm_estimate"], guarded["norm_estimate"]
        costs.append(
            {
                "target": target,
                "original_norm_estimate": a,
                "guarded_norm_estimate": b,
                "guarded_minus_original_norm": b - a if a is not None and b is not None else None,
                "guarded_over_original_norm": b / a if a and b is not None else None,
                "guarded_minus_original_objective": (b * b - a * a) / 2
                if a is not None and b is not None
                else None,
                "role": "numerical initial-linearization comparison only",
            }
        )
    return {
        "status": "INDEPENDENT_INPUT_NUMERIC_CERTIFICATE_AUDIT_COMPLETE",
        "objectives": results,
        "cost_comparison": costs,
        "decimal_precision": 80,
        "relative_tolerance": 0,
        "radius": 0.20,
        "predictor_aim": 0.10,
        "qp_solves": 4,
        "masks_per_solve": 256,
        "independent_audit_passes": 1,
        "model_calls": 0,
        "derivatives": 0,
        "selected_rows_sha256": record["selected_rows_sha256"],
        "analysis_sha256": io.canonical_sha(analysis),
        "is_neural_feasibility_or_global_impossibility_proof": False,
        "old_passes_unchanged": True,
        "follow_on_authorized": False,
    }


def report(result):
    lines = [
        "# Optional retention-guard: initial linearized feasibility",
        "",
        f"Audit: {result['status']}. Four fixed QPs, 256 masks each; one 80-digit audit.",
        "ZERO model/tokenizer imports or loads, forwards, derivatives or new gradients.",
        "Only the same eight f01/f02 v1/v2 AB rows at original shared_w=0 were used.",
        "",
        "| Objective | Numeric norm estimate | Numeric KKT | Conservative radius status |",
        "|---|---:|---|---|",
    ]
    for row in result["objectives"]:
        lines.append(
            f"| {row['name']} | {row['norm_estimate']} | {row['numeric_status']} | {row['conservative_status']} |"
        )
    lines += [
        "",
        "Numerical within-cap estimates are not automatically exact certified feasibility.",
        "Negative conservative primal slack or ambiguous bounds remain unresolved.",
        "",
        "## Original versus guarded numerical cost",
        "",
        "| Target | Original norm | Guarded norm | Difference | Ratio | Objective difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in result["cost_comparison"]:
        lines.append(
            f"| {c['target']} | {c['original_norm_estimate']} | {c['guarded_norm_estimate']} | {c['guarded_minus_original_norm']} | {c['guarded_over_original_norm']} | {c['guarded_minus_original_objective']} |"
        )
    for row in result["objectives"]:
        lines += [
            "",
            "## " + row["name"],
            "",
            f"Scale R={row['scale']}; tolerances={row['tolerances']}.",
        ]
        cert = row["certificate"]
        if cert is None:
            lines += ["Numerically unresolved; no replacement solve or certificate claim."]
            continue
        lines += [
            f"Active zero-based constraints: {row['active_constraints']}.",
            f"Norm interval: {cert['norm_interval']}.",
            f"Conservative dual radius lower-bound interval: {cert['dual_radius_lower_bound_interval']}.",
            f"Minimum conservative primal slack: {row['minimum_primal_interval_lower']}.",
            f"Stationarity upper: {cert['stationarity_max_upper']}; complementarity upper: {cert['complementarity_max_upper']}.",
            f"Primal-dual gap interval: {cert['primal_dual_gap_interval']}.",
            "",
            "| Row / prompt ID | Baseline target margin | Goal | RHS | Multiplier | Numeric primal slack |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for i, pid in enumerate(row["row_ids"]):
            lines.append(
                f"| {i} / {pid} | {row['c0'][i]} | {row['endpoint_goals'][i]} | {row['b'][i]} | {row['multipliers'][i]} | {cert['metrics']['primal_residuals'][i]} |"
            )
    lines += [
        "",
        "## Limits of this diagnostic",
        "",
        "The original objective PERMITTED retention weakening; observed runs used that freedom,",
        "but this does not prove it CAUSED letter/display failures.",
        "Protecting retention margins is an OPTIONAL method-development hypothesis, stricter",
        "than needed for actual-outcome control. Old passes remain passes; this is not a",
        "retrospective evaluation gate or mandatory definition of useful control.",
        "",
        "A guarded over-cap certificate concerns ONLY this proposed initial LINEARIZED objective.",
        "It can reject/reconsider this local guard, not the user goal, the existing radius for",
        "more general nonlinear methods, or any previous pass. A numerical within-cap KKT",
        "candidate is not exact feasibility unless its conservative certificate establishes it.",
        "",
        "AB-only model-free feasibility cannot establish BA robustness, selectivity,",
        "ordinary-task preservation or A-to-B coverage. There is no causal diagnosis of the",
        "crossed display failure and no neural/global impossibility claim.",
        "",
        "Solutions are audit numbers only; none was scaled, clipped, projected, frozen as a",
        "deployable steering candidate or applied. No model run, 16-row training, f03 repair,",
        "threshold/norm increase, gate or controller follows. REPORT+STOP.",
        "",
    ]
    return "\n".join(lines)
