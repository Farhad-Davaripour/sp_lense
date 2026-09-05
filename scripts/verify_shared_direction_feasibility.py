"""Independent directed-rounding Decimal certificates; never imports the solver."""

from __future__ import annotations

import json
import struct
import sys
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import shared_direction_feasibility_io as io

PRECISION = 80


def number(value):
    return Decimal.from_float(value) if isinstance(value, float) else Decimal(value)


class Interval:
    """Outward interval operations at the frozen Decimal precision."""

    def __init__(self, lo, hi=None):
        self.lo, self.hi = number(lo), number(lo if hi is None else hi)
        io.require(
            self.lo.is_finite() and self.hi.is_finite() and self.lo <= self.hi, "finite interval"
        )

    @staticmethod
    def cast(other):
        return other if isinstance(other, Interval) else Interval(other)

    def __add__(self, other):
        other = self.cast(other)
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = PRECISION, ROUND_FLOOR
            lo = self.lo + other.lo
            ctx.rounding = ROUND_CEILING
            hi = self.hi + other.hi
        return Interval(lo, hi)

    def __neg__(self):
        return Interval(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other):
        return self + -self.cast(other)

    def __mul__(self, other):
        other = self.cast(other)
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = PRECISION, ROUND_FLOOR
            lo = min(x * y for x in (self.lo, self.hi) for y in (other.lo, other.hi))
            ctx.rounding = ROUND_CEILING
            hi = max(x * y for x in (self.lo, self.hi) for y in (other.lo, other.hi))
        return Interval(lo, hi)

    def __truediv__(self, other):
        other = self.cast(other)
        io.require(other.lo > 0 or other.hi < 0, "denominator interval touches zero")
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = PRECISION, ROUND_FLOOR
            lo = min(x / y for x in (self.lo, self.hi) for y in (other.lo, other.hi))
            ctx.rounding = ROUND_CEILING
            hi = max(x / y for x in (self.lo, self.hi) for y in (other.lo, other.hi))
        return Interval(lo, hi)

    def square(self):
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = PRECISION, ROUND_FLOOR
            lo = (
                Decimal(0) if self.lo <= 0 <= self.hi else min(self.lo * self.lo, self.hi * self.hi)
            )
            ctx.rounding = ROUND_CEILING
            hi = max(self.lo * self.lo, self.hi * self.hi)
        return Interval(lo, hi)

    def sqrt(self):
        io.require(self.lo >= 0, "nonnegative square root")
        with localcontext() as ctx:
            ctx.prec = PRECISION
            lo = max(Decimal(0), self.lo.sqrt().next_minus())
            hi = self.hi.sqrt().next_plus() if self.hi else Decimal(0)
        return Interval(lo, hi)

    def padded(self, config):
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = PRECISION, ROUND_CEILING
            safety = number(config["interval_safety_absolute"]) + number(
                config["interval_safety_relative"]
            ) * max(abs(self.lo), abs(self.hi))
        return self + Interval(safety.copy_negate(), safety)

    def midpoint(self):
        with localcontext() as ctx:
            ctx.prec = PRECISION
            return (self.lo + self.hi) / 2

    def abs_upper(self):
        return max(self.lo.copy_abs(), self.hi.copy_abs())

    def strings(self):
        return [str(self.lo), str(self.hi)]


def total(items):
    result = Interval(0)
    for item in items:
        result = result + item
    return result


def inner(a, b):
    return total(Interval.cast(x) * y for x, y in zip(a, b, strict=True))


def length(vector):
    return total(Interval.cast(x).square() for x in vector).sqrt()


def rebuild(records, config):
    A, b = [], []
    for row in records:
        hn = length(row["h0"])
        A.append([hn * x for x in row["g"]])
        b.append(Interval(str(config["margin"])) + number(row["S"]).copy_abs())
    return A, b


def certificate(A, b, vector, multipliers, config):
    io.require(
        len(A) == len(b) == len(multipliers) and len(vector) == len(A[0]), "certificate dimensions"
    )
    lam = [Interval(x) for x in multipliers]
    w = [Interval(x) for x in vector]
    at = [total(l * row[j] for l, row in zip(lam, A, strict=True)) for j in range(len(w))]
    residuals = [(inner(row, w) - bi).padded(config) for row, bi in zip(A, b, strict=True)]
    stationarity = [(x - y).padded(config) for x, y in zip(w, at, strict=True)]
    complementarity = [(x * y).padded(config) for x, y in zip(lam, residuals, strict=True)]
    primal = (inner(w, w) / 2).padded(config)
    dual = (inner(b, lam) - inner(at, at) / 2).padded(config)
    gap = (primal - dual).padded(config)
    wn = length(w).padded(config)
    denominator = length(at).padded(config)
    numerator = inner(b, lam).padded(config)
    nonnegative = all(x.lo >= 0 for x in lam)
    lower_bound = None
    if (
        nonnegative
        and denominator.lo > 0
        and denominator.hi > number(config["dual_denominator_floor"])
    ):
        lower_bound = (numerator / denominator).padded(config)
    primal_violation = max(Decimal(0), -min(x.lo for x in residuals))
    stat = max(x.abs_upper() for x in stationarity)
    comp = max(x.abs_upper() for x in complementarity)
    kkt = (
        nonnegative
        and primal_violation <= number(config["primal_absolute_tolerance"])
        and max(stat, comp, gap.abs_upper()) <= number(config["kkt_absolute_tolerance"])
    )
    radius = number(str(config["radius"]))
    guard = number(config["radius_comparison_guard"])
    decision = "NUMERICALLY_UNRESOLVED"
    if lower_bound is not None and lower_bound.lo > radius + guard:
        decision = "OUTSIDE_RADIUS_LINEAR_SURROGATE_CERTIFIED"
    elif kkt and min(x.lo for x in residuals) >= 0 and wn.hi <= radius - guard:
        decision = "WITHIN_RADIUS_LINEAR_SURROGATE_VERIFIED"
    metrics = {
        "norm": float(wn.midpoint()),
        "primal_residuals": [float(x.midpoint()) for x in residuals],
        "primal_violation": float(primal_violation),
        "minimum_multiplier": min(multipliers),
        "stationarity_max": float(stat),
        "complementarity_max": float(comp),
        "primal_objective": float(primal.midpoint()),
        "dual_objective": float(dual.midpoint()),
        "gap": float(gap.midpoint()),
    }
    return {
        "decision": decision,
        "kkt_verified": kkt,
        "metrics": metrics,
        "primal_residual_intervals": [x.strings() for x in residuals],
        "norm_interval": wn.strings(),
        "dual_numerator_interval": numerator.strings(),
        "dual_denominator_interval": denominator.strings(),
        "dual_radius_lower_bound_interval": lower_bound.strings() if lower_bound else None,
        "stationarity_max_upper": str(stat),
        "complementarity_max_upper": str(comp),
        "primal_dual_gap_interval": gap.strings(),
        "lambda_nonnegative": nonnegative,
        "denominator_is_exact_infeasibility_claim": False,
    }


def independent_table(records, vector, config):
    output = []
    A, b = rebuild(records, config)
    for row, a, bi in zip(records, A, b, strict=True):
        hn, gn, an = length(row["h0"]), length(row["g"]), length(a)
        effect = inner(a, vector) if vector is not None else None
        output.append(
            {
                k: row[k]
                for k in (
                    "dataset",
                    "order",
                    "prompt_id",
                    "baseline_cell_id",
                    "initial_gradient_cell_id",
                )
            }
        )
        output[-1].update(
            S=row["S"],
            h0_norm=float(hn.midpoint()),
            gradient_norm=float(gn.midpoint()),
            necessary_norm_bound=float((bi / an).midpoint()) if an.lo > 0 else None,
            predicted_preserve_margin=float((effect + row["S"]).midpoint())
            if effect is not None
            else None,
            predicted_comply_margin=float((effect - row["S"]).midpoint())
            if effect is not None
            else None,
            residual=float((effect - bi).midpoint()) if effect is not None else None,
        )
    return output


def compare(actual, expected, tolerance, path="root"):
    if isinstance(expected, dict):
        io.require(actual.keys() == expected.keys(), f"fields differ: {path}")
        for key, value in expected.items():
            compare(actual[key], value, tolerance, path + "." + key)
    elif isinstance(expected, list):
        io.require(len(actual) == len(expected), f"length differs: {path}")
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            compare(a, e, tolerance, path + "." + str(index))
    elif isinstance(expected, float):
        io.require(
            abs(number(actual) - number(expected)) <= number(tolerance),
            f"absolute scalar mismatch: {path}",
        )
    else:
        io.require(actual == expected, f"identity mismatch: {path}")


def verify():
    config = io.locked()["config"]
    io.require(
        config["decimal_precision"] == PRECISION and config["relative_tolerance"] == 0,
        "frozen precision",
    )
    analysis = io.read(io.OUTPUT / "analysis.json")
    frozen = io.read(io.OUTPUT / "construction_frozen.json")
    raw = (io.OUTPUT / "solution.json").read_bytes()
    saved = json.loads(raw)
    io.require(
        io.sha(raw) == frozen["solution_sha256"] == analysis["solution_sha256"]
        and frozen["f02_numeric_loading_started"] is False,
        "construction freeze digest",
    )
    records = [r for key in config["construction"] for r in io.initial_rows(key, config)]
    solution = saved["solution"]
    vector = solution["vector"] if solution else None
    io.require(
        saved["construction"]
        == [
            {k: r[k] for k in ("dataset", "prompt_id", "initial_gradient_cell_id")} for r in records
        ],
        "construction input identities",
    )
    io.require(
        saved["vector_float64_le_sha256"]
        == (io.sha(struct.pack(f"<{len(vector)}d", *vector)) if vector is not None else None),
        "frozen native vector bytes",
    )
    io.require(
        len(analysis["active_sets"]) == 16
        and [r["mask"] for r in analysis["active_sets"]] == list(range(16)),
        "all16 active sets recorded",
    )
    cert = {
        "decision": "NUMERICALLY_UNRESOLVED",
        "kkt_verified": False,
        "reason": "no qualifying independent active-set estimate",
    }
    if solution:
        io.require(
            solution["active"] == [i for i in range(4) if solution["active_mask"] & (1 << i)]
            and all(
                solution["multipliers"][i] == 0 for i in range(4) if i not in solution["active"]
            )
            and analysis["active_sets"][solution["active_mask"]]["status"] == "kkt_valid",
            "selected independent active set identity",
        )
        A, b = rebuild(records, config)
        cert = certificate(A, b, vector, solution["multipliers"], config)
        compare(
            solution["metrics"], cert["metrics"], config["scalar_reconstruction_absolute_tolerance"]
        )
    construction = independent_table(records, vector, config)
    compare(
        analysis["construction_rows"],
        construction,
        config["scalar_reconstruction_absolute_tolerance"],
    )
    pairs = []
    for key in config["construction"]:
        first, second = [r for r in records if r["dataset"] == key]
        den = length(first["g"]) * length(second["g"])
        pairs.append(
            {
                "dataset": key,
                "semantic_gradient_cosine": float((inner(first["g"], second["g"]) / den).midpoint())
                if den.lo > 0
                else None,
            }
        )
    compare(analysis["within_pair"], pairs, config["scalar_reconstruction_absolute_tolerance"])
    # This independent audit also authenticates the already-frozen solution before f02 loads.
    exposed = (
        [r for key in config["descriptive_comparison"] for r in io.initial_rows(key, config)]
        if vector is not None
        else []
    )
    descriptive = independent_table(exposed, vector, config)
    compare(
        analysis["exposed_f02_rows"],
        descriptive,
        config["scalar_reconstruction_absolute_tolerance"],
    )
    return {
        "status": "INDEPENDENT_SCALAR_RECONSTRUCTION_MATCH",
        "certificate": cert,
        "construction_rows": construction,
        "within_pair": pairs,
        "exposed_f02_rows": descriptive,
        "solution_sha256": io.sha(raw),
        "vector_float64_le_sha256": saved["vector_float64_le_sha256"],
        "decimal_precision": PRECISION,
        "scalar_absolute_tolerance": config["scalar_reconstruction_absolute_tolerance"],
        "relative_tolerance": 0,
        "radius": config["radius"],
        "margin": config["margin"],
        "model_calls": 0,
        "causal_test_performed": False,
        "no_global_impossibility_claim": True,
    }


def report(result):
    cert = result["certificate"]
    body = [
        "# Shared-direction initial-gradient linear feasibility",
        "",
        f"Independent audit: **{result['status']}**.",
        f"Radius decision: **{cert['decision']}**.",
        "",
        "One sign-reversible dimensionless displacement; four f01/v1-v2 construction constraints.",
        "This is a local first-order surrogate, not actual model behavior. No model/tokenizer was loaded,",
        "and no forwards, derivatives, generation, vocabulary-array decompression or controller training occurred.",
        "",
        "| Construction row | Baseline S | Individual necessary norm | Shared-solution residual |",
        "|---|---:|---:|---:|",
    ]
    for row in result["construction_rows"]:
        body.append(
            f"| {row['dataset']} / {row['order']} | {row['S']:+.9f} | {row['necessary_norm_bound']} | {row['residual']} |"
        )
    body += ["", "Within-pair semantic-gradient cosines: " + json.dumps(result["within_pair"]), ""]
    if "metrics" in cert:
        body += [
            f"Shared minimum-norm estimate: {cert['metrics']['norm']}; existing radius: 0.20.",
            f"Independent KKT verified: {cert['kkt_verified']}.",
            "",
            "```json",
            json.dumps(cert, indent=2),
            "```",
            "",
        ]
    body += [
        "## Exposed f02 descriptive check after construction freeze",
        "",
        "| Order | Predicted preserve margin | Predicted comply margin | Both-sign residual |",
        "|---|---:|---:|---:|",
    ]
    for row in result["exposed_f02_rows"]:
        body.append(
            f"| {row['order']} | {row['predicted_preserve_margin']:+.9f} | {row['predicted_comply_margin']:+.9f} | {row['residual']:+.9f} |"
        )
    body += [
        "",
        "The solution bytes were frozen before f02 numeric loading. f02 is exposed development,",
        "not sealed confirmation; no corrections or selection used its numbers. An over-radius",
        "solution is only a mathematical diagnostic and was not normalized, clipped or applied.",
        "",
        "S-only geometry cannot diagnose or fix the real 0.05/0.20 quality failures. Pair mass,",
        "other-token competition, nonlinear effects, collateral preservation and actual choices",
        "remain unproved. A certificate outside this radius concerns only these saved initial",
        "gradients and this sign-reversible linear surrogate, not a natural self-preservation",
        "mechanism or globally impossible neural intervention. No strength escalation or follow-on started.",
        "",
    ]
    return "\n".join(body)


def audit():
    with localcontext() as ctx:
        ctx.prec = PRECISION
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
        raise SystemExit("Use run; one bounded independent reconstruction only.")
