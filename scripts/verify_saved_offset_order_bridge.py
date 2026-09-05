"""Independent Decimal scalar/vector reconstruction; stdlib only, no model calls."""

from __future__ import annotations

import json
import math
import struct
import sys
import time
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import saved_offset_order_bridge_io as io

SCRIPT = "scripts/verify_saved_offset_order_bridge.py"


def number(x):
    return Decimal.from_float(float(x))


def product(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), Decimal(0))


def length(a):
    return product(a, a).sqrt()


def angle(a, b):
    denominator = length(a) * length(b)
    return float(product(a, b) / denominator) if denominator else None


def parts(a, b):
    return (
        [(x - y) / 2 for x, y in zip(a, b, strict=True)],
        [(x + y) / 2 for x, y in zip(a, b, strict=True)],
    )


def coordinates(records, config):
    io.require([r["order"] for r in records] == config["order"], "independent pair order")
    output = []
    for row in records:
        h0 = list(map(number, row["h0"]))
        D = [number(x) - y for x, y in zip(row["h"], h0, strict=True)]
        dn, hn = length(D), length(h0)
        error = max(abs(x - number(y)) for x, y in zip(D, row["offset"], strict=True))
        io.require(dn > 0 and hn > 0, "independent nonzero vectors")
        io.require(
            error <= number(config["offset_component_absolute_tolerance"]), "independent offset"
        )
        io.require(
            abs(dn - number(row["saved_D_norm"]))
            <= number(config["saved_scalar_absolute_tolerance"])
            and abs(hn - number(row["saved_h0_norm"]))
            <= number(config["saved_scalar_absolute_tolerance"]),
            "independent saved norms",
        )
        output.append(
            {
                **row,
                "raw": D,
                "unit": [x / dn for x in D],
                "h0_relative": [x / hn for x in D],
                "dn": dn,
                "hn": hn,
                "error": error,
                "gradient": list(map(number, row["g"])),
            }
        )
    return output


def pair_numbers(pair):
    result = {}
    for key in ("unit", "raw", "h0_relative"):
        a, b = pair[0][key], pair[1][key]
        odd, even = parts(a, b)
        result[key] = {
            "first_norm": float(length(a)),
            "second_norm": float(length(b)),
            "endpoint_cosine": angle(a, b),
            "order_odd_norm": float(length(odd)),
            "order_even_norm": float(length(even)),
            "odd_even_cosine": angle(odd, even),
        }
    return result


def endpoint_numbers(pair, candidate, split):
    output = []
    for row in pair:
        slope = product(row["gradient"], candidate) if candidate is not None else None
        output.append(
            {
                k: row[k]
                for k in (
                    "dataset",
                    "order",
                    "prompt_id",
                    "final_cell_id",
                    "initial_gradient_cell_id",
                    "t",
                    "requested",
                    "baseline_label",
                    "final_label",
                    "steps",
                    "baseline_signed_margin",
                    "final_signed_margin",
                )
            }
        )
        output[-1].update(
            D_norm=float(row["dn"]),
            h0_norm=float(row["hn"]),
            offset_max_error=float(row["error"]),
            split=split,
            relative_D_norm=float(row["dn"] / row["hn"]),
            semantic_gradient_dot=float(slope) if slope is not None else None,
            predicted_S_slope_per_unit_relative_edit=float(row["hn"] * slope)
            if slope is not None
            else None,
        )
    return output


def reconstruct(config, loader, check_candidate):
    with localcontext() as ctx:
        ctx.prec = 60
        pairs = {key: coordinates(loader(key), config) for key in config["construction"]}
        io.require(
            len(pairs) == 2 and all([r["t"] for r in pair] == [1, -1] for pair in pairs.values()),
            "independent construction signs/pair count",
        )
        odd = [parts(pair[0]["unit"], pair[1]["unit"])[0] for pair in pairs.values()]
        mean = [(x + y) / 2 for x, y in zip(odd[0], odd[1], strict=True)]
        magnitude = length(mean)
        table = {
            "status": "DEGENERATE_INCONCLUSIVE",
            "U_norm": float(magnitude),
            "construction_u_cosine": angle(*odd),
            "pairs": {k: pair_numbers(p) for k, p in pairs.items()},
            "endpoints": [
                e for pair in pairs.values() for e in endpoint_numbers(pair, None, "construction")
            ],
            "f02_alignment": [],
            "causal_test_performed": False,
        }
        if magnitude <= number(config["degeneracy_norm_threshold"]):
            check_candidate(None, float(magnitude))
            return None, table
        candidate = [x / magnitude for x in mean]
        check_candidate([float(x) for x in candidate], float(magnitude))
        table["status"] = "CANDIDATE_CONSTRUCTED_NO_CAUSAL_TEST"
        table["endpoints"] = []
        for pair in pairs.values():
            table["endpoints"] += endpoint_numbers(pair, candidate, "construction")
        for key in config["descriptive_comparison"]:
            pair = coordinates(loader(key), config)
            table["pairs"][key] = pair_numbers(pair)
            table["endpoints"] += endpoint_numbers(
                pair, candidate, "exposed_second_family_descriptive"
            )
            for row in pair:
                table["f02_alignment"].append(
                    {
                        "dataset": key,
                        "order": row["order"],
                        "t": row["t"],
                        "cosine_with_t_q": angle(candidate, [row["t"] * x for x in row["unit"]]),
                    }
                )
        return [float(x) for x in candidate], table


def compare(actual, expected, tolerance=1e-10, path="root"):
    if type(expected) is float:
        io.require(
            type(actual) in (float, int)
            and math.isfinite(actual)
            and abs(actual - expected) <= tolerance,
            f"independent absolute mismatch: {path}",
        )
        return abs(actual - expected)
    if isinstance(expected, dict):
        io.require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"keys: {path}")
        return max(
            (compare(actual[k], v, tolerance, path + "." + k) for k, v in expected.items()),
            default=0,
        )
    if isinstance(expected, list):
        io.require(isinstance(actual, list) and len(actual) == len(expected), f"length: {path}")
        return max(
            (
                compare(a, b, tolerance, f"{path}[{i}]")
                for i, (a, b) in enumerate(zip(actual, expected, strict=True))
            ),
            default=0,
        )
    io.require(type(actual) is type(expected) and actual == expected, f"identity: {path}")
    return 0


def report(table, audit, receipt):
    lines = [
        "# Saved-offset order bridge",
        "",
        f"Status: **{table['status']}**. No causal test or causal PASS.",
        "",
        f"Independent 60-digit Decimal reconstruction matched within absolute1e-10, zero relative tolerance; maximum discrepancy {audit['maximum_absolute_difference']:.3g}.",
        "",
        f"Primary U norm: {table['U_norm']:.6f}; construction cross-variant u cosine: {table['construction_u_cosine']}.",
        "",
    ]
    if receipt is None:
        return (
            "\n".join(
                lines
                + [
                    "Degenerate fixed construction. No candidate, f02 comparison, substitution or follow-on."
                ]
            )
            + "\n"
        )
    lines += [
        f"Candidate file SHA256: `{receipt['candidate_sha256']}`.",
        f"Float64 little-endian vector SHA256: `{receipt['vector_float64_le_sha256']}`.",
        "",
        "The primary arrow equally weights unit endpoint directions from f01/v1 and f01/v2. It is the normalized mean of their (first-order minus second-order) half-differences. It was saved and hashed before the descriptive f02 values were loaded.",
        "",
        "| Pair | Representation | First / second norm | Endpoint cosine | Odd / even norm | Odd-even cosine |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, pair in table["pairs"].items():
        for representation, m in pair.items():
            fmt = lambda x: "undefined" if x is None else f"{x:.6f}"
            lines.append(
                f"| {key} | {representation} | {m['first_norm']:.6f} / {m['second_norm']:.6f} | {fmt(m['endpoint_cosine'])} | {m['order_odd_norm']:.6f} / {m['order_even_norm']:.6f} | {fmt(m['odd_even_cosine'])} |"
            )
    lines += [
        "",
        "The unit-pair odd norm is the preregistered cancellation diagnostic ||q_first-q_second||/2. Raw and h0-relative rows are sensitivity descriptions only; they did not select another candidate.",
        "",
        "| Data / order | Role | Steps | D / h0 norm | D/h0 | Baseline / final signed margin | g dot v | h0-norm-scaled g dot v |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for e in table["endpoints"]:
        lines.append(
            f"| {e['dataset']} / {e['order']} | {e['split']} | {e['steps']} | {e['D_norm']:.6f} / {e['h0_norm']:.6f} | {e['relative_D_norm']:.6f} | {e['baseline_signed_margin']:+.6f} / {e['final_signed_margin']:+.6f} | {e['semantic_gradient_dot']:+.6f} | {e['predicted_S_slope_per_unit_relative_edit']:+.6f} |"
        )
    lines += [
        "",
        "f02 alignment with the saved t*q endpoints: "
        + "; ".join(f"{a['order']}: {a['cosine_with_t_q']:+.6f}" for a in table["f02_alignment"])
        + ".",
        "",
        "g differentiates S=z_preserve-z_comply. The h0-norm-scaled dot is a local predicted S slope per unit relative edit, not observed finite-edit behavior. No alpha or sign was selected from these diagnostics.",
        "",
        "The order-odd component can contain prompt/order/optimization effects; it is not identified semantics. The order-even component is not proven label A. Different trajectory lengths and stopping rules remain confounds. f01 paraphrases are not independent families; f02 is exposed development, not sealed validation. All original opposed flips were B-to-A. Existing nonself controls were off, so this candidate has no collateral-effect evidence.",
        "",
        "Smallest next causal-transfer question for separate prospective authorization: does this frozen candidate, at one prospectively fixed relative magnitude and its fixed semantic sign, move S consistently across matched A/B orders when actually injected? A diagnostic dot product cannot answer that question. No follow-on run, portfolio search, gate/controller or other model work was performed.",
    ]
    return "\n".join(lines) + "\n"


def verify():
    config = io.read(ROOT / io.CONFIG)
    output = ROOT / config["output_namespace"]
    io.require(
        io.identity(config) == io.read(output / "provenance.json")["source_sha256"], "frozen source"
    )
    io.require(
        io.read(output / "analysis_status.json")["status"] == "completed", "analysis incomplete"
    )
    receipt = None
    if (output / "candidate.json").exists():
        receipt = io.read(output / "CANDIDATE_FROZEN.json")
        io.require(
            io.sha((output / "candidate.json").read_bytes()) == receipt["candidate_sha256"],
            "candidate hash",
        )
        start = io.read(output / "DESCRIPTIVE_STARTED.json")
        io.require(
            receipt["frozen_monotonic"] <= start["started_monotonic"]
            and receipt["descriptive_values_loaded"] is False
            and start["candidate_sha256"] == receipt["candidate_sha256"],
            "freeze-before-comparison",
        )
    maximum = 0

    def check_candidate(vector, un):
        nonlocal maximum
        if vector is None:
            io.require(
                receipt is None
                and (output / "DEGENERATE.json").exists()
                and not (output / "DESCRIPTIVE_STARTED.json").exists(),
                "degeneracy path",
            )
            return
        actual = io.read(output / "candidate.json")
        maximum = max(compare(actual["vector"], vector), compare(actual["U_norm"], un))
        io.require(abs(actual["vector_norm"] - 1) <= 1e-10, "candidate unit norm")
        io.require(
            actual["construction"] == config["construction"]
            and actual["construction_input_sha256"]
            == {k: config["inputs"][k]["sha256"] for k in config["construction"]}
            and actual["basis"]["dimension"] == 1024
            and actual["basis"]["hook"] == "blocks.10.hook_out"
            and actual["causal_test_performed"] is False,
            "candidate metadata",
        )
        io.require(
            io.sha(struct.pack(f"<{len(vector)}d", *actual["vector"]))
            == receipt["vector_float64_le_sha256"]
            == actual["vector_float64_le_sha256"],
            "vector byte identity",
        )

    _, expected = reconstruct(config, lambda key: io.load_endpoints(key, config), check_candidate)
    actual = io.read(output / "diagnostics.json")
    maximum = max(maximum, compare(actual, expected))
    result = {
        "status": "RECONSTRUCTION_MATCH_NO_CAUSAL_TEST",
        "maximum_absolute_difference": maximum,
        "absolute_tolerance": 1e-10,
        "relative_tolerance": 0,
        "arithmetic": "independent stdlib Decimal, precision60",
        "model_calls": 0,
    }
    io.write_new(output / "verification.json", result)
    with (output / "BRIDGE_REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report(actual, result, receipt))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == ["_verify"]:
        verify()
    elif not sys.argv[1:]:
        started = time.monotonic()
        config = io.read(ROOT / io.CONFIG)
        result = io.bounded(SCRIPT, "_verify", ROOT / config["output_namespace"], started=started)
        print(json.dumps(result, indent=2))
        if result["status"] != "completed":
            raise SystemExit(1)
    else:
        raise SystemExit("No alternative inputs or construction options.")
