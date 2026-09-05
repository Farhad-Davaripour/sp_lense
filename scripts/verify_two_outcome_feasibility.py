"""Two signed formulations; immutable independent Decimal certificate arithmetic."""

from __future__ import annotations

import json
import struct
import sys
from decimal import localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import two_outcome_feasibility_io as io
from scripts import verify_shared_direction_feasibility as old

certificate, compare = old.certificate, old.compare


def rebuild(records, outcome, config):
    io.require(outcome in io.OUTCOMES, "fixed outcome only")
    A, b = [], []
    for row in records:
        hn = old.length(row["h0"])
        native = [hn * x for x in row["g"]]
        if outcome == "preserve":
            A.append(native)
            b.append(old.Interval(str(config["margin"])) - row["S"])
        else:
            A.append([-x for x in native])
            b.append(old.Interval(str(config["margin"])) + row["S"])
    return A, b


def table(records, vector, outcome, config):
    output = []
    for row in records:
        hn = old.length(row["h0"])
        native = [hn * x for x in row["g"]]
        prediction = (
            old.Interval(row["S"]) + old.inner(native, vector) if vector is not None else None
        )
        margin = old.Interval(str(config["margin"]))
        rhs = margin - row["S"] if outcome == "preserve" else margin + row["S"]
        if outcome == "comply" and prediction is not None:
            prediction = -prediction  # Score sign only; do not negate the applied wC.
        baseline = (
            old.number(row["S"]) if outcome == "preserve" else old.number(row["S"]).copy_negate()
        )
        output.append(
            {k: row[k] for k in ("dataset", "order", "prompt_id", "initial_gradient_cell_id")}
        )
        output[-1].update(
            S=row["S"],
            outcome=outcome,
            rhs=float(rhs.midpoint()),
            baseline_already_correct=baseline >= old.number(str(config["margin"])),
            predicted_signed_margin=float(prediction.midpoint())
            if prediction is not None
            else None,
            slack=float((prediction - margin).midpoint()) if prediction is not None else None,
        )
    return output


def joint_status(certificates):
    decisions = [certificates[o]["decision"] for o in io.OUTCOMES]
    both = all(d == "WITHIN_RADIUS_LINEAR_SURROGATE_VERIFIED" for d in decisions)
    return {
        "both_outcomes_within_radius_verified": both,
        "decision": "BOTH_OUTCOMES_WITHIN_LINEAR_SURROGATE"
        if both
        else (
            "AT_LEAST_ONE_OUTCOME_OUTSIDE_LINEAR_SURROGATE"
            if any(d == "OUTSIDE_RADIUS_LINEAR_SURROGATE_CERTIFIED" for d in decisions)
            else "BOTH_OUTCOME_FEASIBILITY_NUMERICALLY_UNRESOLVED"
        ),
        "joint_sum_of_norms_cap_used": False,
        "causal_pass": False,
    }


def verify():
    config = io.locked()["config"]
    io.require(
        config["decimal_precision"] == old.PRECISION and config["outcomes"] == list(io.OUTCOMES),
        "unchanged precision/two outcomes",
    )
    analysis = io.read(io.OUTPUT / "analysis.json")
    frozen = io.read(io.OUTPUT / "construction_frozen.json")
    io.require(
        frozen["f02_numeric_loading_started"] is False
        and set(frozen["solution_sha256"]) == set(io.OUTCOMES),
        "both records frozen",
    )
    saved = {}
    for outcome in io.OUTCOMES:
        raw = (io.OUTPUT / (outcome + "_solution.json")).read_bytes()
        io.require(
            io.sha(raw)
            == frozen["solution_sha256"][outcome]
            == analysis["solution_sha256"][outcome],
            "outcome freeze hash",
        )
        saved[outcome] = json.loads(raw)
    records = [r for key in config["construction"] for r in io.initial_rows(key, config)]
    io.require(len(records) == 4, "four construction constraints per outcome")
    certs, result = {}, {}
    for outcome in io.OUTCOMES:
        value, observed = saved[outcome], analysis["outcomes"][outcome]
        io.require(
            value["outcome"] == outcome
            and value["construction"]
            == [
                {k: r[k] for k in ("dataset", "prompt_id", "initial_gradient_cell_id")}
                for r in records
            ],
            "all four outcome construction identities",
        )
        solution = value["solution"]
        vector = solution["vector"] if solution else None
        io.require(
            value["vector_float64_le_sha256"]
            == (io.sha(struct.pack(f"<{len(vector)}d", *vector)) if vector is not None else None),
            "native outcome vector bytes",
        )
        io.require(
            len(observed["active_sets"]) == 16
            and [r["mask"] for r in observed["active_sets"]] == list(range(16)),
            "16 subsets per outcome",
        )
        cert = {
            "decision": "NUMERICALLY_UNRESOLVED",
            "kkt_verified": False,
            "reason": "no qualifying independent active set",
        }
        if solution:
            io.require(
                solution["active"] == [i for i in range(4) if solution["active_mask"] & (1 << i)]
                and all(
                    solution["multipliers"][i] == 0 for i in range(4) if i not in solution["active"]
                )
                and observed["active_sets"][solution["active_mask"]]["status"] == "kkt_valid",
                "active set identity",
            )
            A, b = rebuild(records, outcome, config)
            cert = certificate(A, b, vector, solution["multipliers"], config)
            compare(
                solution["metrics"],
                cert["metrics"],
                config["scalar_reconstruction_absolute_tolerance"],
            )
        construction = table(records, vector, outcome, config)
        compare(
            observed["construction_rows"],
            construction,
            config["scalar_reconstruction_absolute_tolerance"],
        )
        certs[outcome] = cert
        result[outcome] = {
            "certificate": cert,
            "construction_rows": construction,
            "solution_sha256": frozen["solution_sha256"][outcome],
            "vector_float64_le_sha256": value["vector_float64_le_sha256"],
        }
    # Both solution hashes were authenticated above, before any exposed f02 numeric load.
    exposed = (
        [r for key in config["descriptive_comparison"] for r in io.initial_rows(key, config)]
        if any(saved[o]["solution"] is not None for o in io.OUTCOMES)
        else []
    )
    for outcome in io.OUTCOMES:
        solution = saved[outcome]["solution"]
        descriptive = table(exposed, solution["vector"] if solution else None, outcome, config)
        compare(
            analysis["outcomes"][outcome]["exposed_f02_rows"],
            descriptive,
            config["scalar_reconstruction_absolute_tolerance"],
        )
        result[outcome]["exposed_f02_rows"] = descriptive
    return {
        "status": "INDEPENDENT_TWO_OUTCOME_SCALAR_RECONSTRUCTION_MATCH",
        "outcomes": result,
        "joint": joint_status(certs),
        "shared_comparison": io.comparison(config),
        "decimal_precision": old.PRECISION,
        "scalar_absolute_tolerance": config["scalar_reconstruction_absolute_tolerance"],
        "relative_tolerance": 0,
        "radius_per_outcome": config["radius"],
        "margin": config["margin"],
        "model_calls": 0,
        "causal_test_performed": False,
    }


def report(result):
    lines = [
        "# Two-outcome initial-gradient feasibility",
        "",
        f"Independent audit: **{result['status']}**.",
        f"Joint status: **{result['joint']['decision']}**.",
        "",
        "One assumption changed: two fixed outcome vectors need not be negatives of each other.",
        "Both problems retain all four f01/v1-v2 construction constraints, including negative RHS",
        "retention constraints. Each applied edit has its own 0.20 cap; no joint norm-sum cap.",
        "",
        "| Outcome | Norm estimate | Conservative radius decision | Dual bound lower | KKT |",
        "|---|---:|---|---:|---|",
    ]
    for outcome in io.OUTCOMES:
        cert = result["outcomes"][outcome]["certificate"]
        bound = cert.get("dual_radius_lower_bound_interval")
        lines.append(
            f"| {outcome} | {cert.get('metrics', {}).get('norm')} | {cert['decision']} | {bound[0] if bound else 'unresolved'} | {cert['kkt_verified']} |"
        )
    lines += [
        "",
        "The full outward-rounded certificates are in [verification.json](verification.json).",
        "A norm estimate alone is not a radius certificate. Rounded boundary slacks that fail the",
        "strict unchanged within-radius test remain numerically unresolved; no vectors are corrected.",
        "",
        "| Outcome | Max primal violation | Min lambda | Max stationarity | Max complementarity | Signed gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for outcome in io.OUTCOMES:
        m = result["outcomes"][outcome]["certificate"].get("metrics", {})
        lines.append(
            f"| {outcome} | {m.get('primal_violation')} | {m.get('minimum_multiplier')} | {m.get('stationarity_max')} | {m.get('complementarity_max')} | {m.get('gap')} |"
        )
    lines += [
        "",
        "## All construction constraints",
        "",
        "| Row | Preserve RHS | Preserve margin / slack | Comply RHS | Comply margin / slack |",
        "|---|---:|---:|---:|---:|",
    ]
    pairs = zip(
        result["outcomes"]["preserve"]["construction_rows"],
        result["outcomes"]["comply"]["construction_rows"],
        strict=True,
    )
    for p, c in pairs:
        lines.append(
            f"| {p['dataset']} / {p['order']} | {p['rhs']} | {p['predicted_signed_margin']} / {p['slack']} | {c['rhs']} | {c['predicted_signed_margin']} / {c['slack']} |"
        )
    lines += [
        "",
        "## Exposed f02 predictions after BOTH solution freezes",
        "",
        "| Order | Preserve S+A*wP / slack | Comply -(S+A*wC) / slack |",
        "|---|---:|---:|",
    ]
    pairs = zip(
        result["outcomes"]["preserve"]["exposed_f02_rows"],
        result["outcomes"]["comply"]["exposed_f02_rows"],
        strict=True,
    )
    for p, c in pairs:
        lines.append(
            f"| {p['order']} | {p['predicted_signed_margin']} / {p['slack']} | {c['predicted_signed_margin']} / {c['slack']} |"
        )
    previous = result["shared_comparison"]
    lines += [
        "",
        f"Descriptive comparison: old shared sign-reversible estimate {previous['norm_estimate']},",
        f"old certificate {previous['decision']}; historical verification hash {previous['verification_sha256']}.",
        "",
        "f02 is exposed development, never sealed confirmation or a selection/correction source.",
        "A hypothetical comply edit would use norm(h0)*wC directly, without an extra minus sign.",
        "No solution was normalized, clipped or applied. Over-cap points are mathematical diagnostics only.",
        "",
        "There were no model/tokenizer loads, forwards, derivatives, vocabulary-array decompression,",
        "new data/strengths or controller/gate training. This is S-only initial-gradient geometry:",
        "nonlinearity, A/B mass, competing tokens, ordinary-task collateral and actual choices are",
        "unproved. No causal PASS, natural self-preservation claim or global neural-impossibility claim.",
        "The real 0.05/0.20 failures remain unchanged. No follow-on has started.",
        "",
    ]
    return "\n".join(lines)


def audit():
    with localcontext() as ctx:
        ctx.prec = old.PRECISION
        result = verify()
    io.write_new(io.OUTPUT / "verification.json", result)
    with (io.OUTPUT / "PILOT_REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report(result))


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        status = io.launch(io.AUDIT, "_audit")
        print(json.dumps(status, indent=2))
        if status["status"] != "completed":
            raise SystemExit(1)
    elif sys.argv[1:] == ["_audit"]:
        audit()
    else:
        raise SystemExit("Use run; one bounded two-outcome audit.")
