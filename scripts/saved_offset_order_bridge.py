"""Exactly one prospectively specified model-free saved-offset candidate."""

from __future__ import annotations

import json
import math
import os
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import saved_offset_order_bridge_io as io

SCRIPT = "scripts/saved_offset_order_bridge.py"


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def norm(a):
    return math.sqrt(dot(a, a))


def combine(a, b, sign=1):
    return [(x + sign * y) / 2 for x, y in zip(a, b, strict=True)]


def cosine(a, b):
    n = norm(a) * norm(b)
    return dot(a, b) / n if n else None


def prepare(endpoints, config):
    io.require([e["order"] for e in endpoints] == config["order"], "ordered pair")
    prepared = []
    for e in endpoints:
        D = [x - y for x, y in zip(e["h"], e["h0"], strict=True)]
        dn, hn = norm(D), norm(e["h0"])
        error = max(abs(x - y) for x, y in zip(D, e["offset"], strict=True))
        io.require(
            math.isfinite(dn) and dn > 0 and math.isfinite(hn) and hn > 0,
            "nonzero finite displacement/h0",
        )
        io.require(error <= config["offset_component_absolute_tolerance"], "D/offset agreement")
        io.require(
            abs(dn - e["saved_D_norm"]) <= config["saved_scalar_absolute_tolerance"]
            and abs(hn - e["saved_h0_norm"]) <= config["saved_scalar_absolute_tolerance"],
            "saved norm agreement",
        )
        prepared.append(
            {
                **e,
                "D": D,
                "q": [x / dn for x in D],
                "relative": [x / hn for x in D],
                "D_norm": dn,
                "h0_norm": hn,
                "offset_max_error": error,
            }
        )
    return prepared


def pair_metrics(a, b):
    u, e = combine(a, b, -1), combine(a, b)
    return {
        "first_norm": norm(a),
        "second_norm": norm(b),
        "endpoint_cosine": cosine(a, b),
        "order_odd_norm": norm(u),
        "order_even_norm": norm(e),
        "odd_even_cosine": cosine(u, e),
    }


def pair_table(pair):
    return {
        name: pair_metrics(pair[0][field], pair[1][field])
        for name, field in (("unit", "q"), ("raw", "D"), ("h0_relative", "relative"))
    }


def endpoint_table(pair, v, split):
    result = []
    for e in pair:
        slope = dot(e["g"], v) if v is not None else None
        result.append(
            {
                k: e[k]
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
                    "D_norm",
                    "h0_norm",
                    "offset_max_error",
                )
            }
        )
        result[-1].update(
            split=split,
            relative_D_norm=e["D_norm"] / e["h0_norm"],
            semantic_gradient_dot=slope,
            predicted_S_slope_per_unit_relative_edit=e["h0_norm"] * slope
            if slope is not None
            else None,
        )
    return result


def calculate(config, loader, freeze_candidate):
    construction = {key: prepare(loader(key), config) for key in config["construction"]}
    io.require(len(construction) == 2, "exactly two construction variants")
    for pair in construction.values():
        io.require([e["t"] for e in pair] == [1, -1], "construction semantic signs")
    us = [combine(pair[0]["q"], pair[1]["q"], -1) for pair in construction.values()]
    U = combine(*us)
    un = norm(U)
    table = {
        "status": "DEGENERATE_INCONCLUSIVE",
        "U_norm": un,
        "construction_u_cosine": cosine(*us),
        "pairs": {key: pair_table(pair) for key, pair in construction.items()},
        "endpoints": [
            e for pair in construction.values() for e in endpoint_table(pair, None, "construction")
        ],
        "f02_alignment": [],
        "causal_test_performed": False,
    }
    if un <= config["degeneracy_norm_threshold"]:
        return None, table
    v = [x / un for x in U]
    candidate = {
        "vector": v,
        "vector_norm": norm(v),
        "U_norm": un,
        "sign_convention": "positive v = normalized mean of (q_first-q_second)/2; t_first=preserve +1, t_second=comply -1",
        "weighting": "equal unit-displacement direction per construction endpoint",
        "construction": config["construction"],
        "basis": {
            "model": "Qwen/Qwen3.5-0.8B",
            "revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "hook": "blocks.10.hook_out",
            "position": "final encoded prompt token",
            "dimension": config["dimension"],
            "coordinates": "unchanged native activation coordinates",
            "saved_activation_dtype": "float32",
            "candidate_dtype": "float64",
        },
        "vector_float64_le_sha256": io.sha(struct.pack(f"<{len(v)}d", *v)),
        "causal_test_performed": False,
    }
    freeze_candidate(candidate)  # Must complete before any descriptive endpoint load.
    table["status"] = "CANDIDATE_CONSTRUCTED_NO_CAUSAL_TEST"
    table["endpoints"] = []
    for pair in construction.values():
        table["endpoints"] += endpoint_table(pair, v, "construction")
    for key in config["descriptive_comparison"]:
        pair = prepare(loader(key), config)
        table["pairs"][key] = pair_table(pair)
        table["endpoints"] += endpoint_table(pair, v, "exposed_second_family_descriptive")
        table["f02_alignment"] += [
            {
                "dataset": key,
                "order": e["order"],
                "t": e["t"],
                "cosine_with_t_q": cosine(v, [e["t"] * x for x in e["q"]]),
            }
            for e in pair
        ]
    return candidate, table


def analyze():
    config = io.read(ROOT / io.CONFIG)
    output = ROOT / config["output_namespace"]
    source = io.identity(config)
    io.require(source == io.read(output / "provenance.json")["source_sha256"], "source changed")

    def freeze(candidate):
        candidate["construction_input_sha256"] = {
            k: config["inputs"][k]["sha256"] for k in config["construction"]
        }
        io.write_new(output / "candidate.json", candidate)
        io.write_new(
            output / "CANDIDATE_FROZEN.json",
            {
                "candidate_sha256": io.sha((output / "candidate.json").read_bytes()),
                "vector_float64_le_sha256": candidate["vector_float64_le_sha256"],
                "frozen_monotonic": time.monotonic(),
                "descriptive_values_loaded": False,
            },
        )

    def loader(key):
        if key in config["descriptive_comparison"]:
            receipt = io.read(output / "CANDIDATE_FROZEN.json")
            io.require(
                io.sha((output / "candidate.json").read_bytes()) == receipt["candidate_sha256"],
                "candidate changed before descriptive comparison",
            )
            io.write_new(
                output / "DESCRIPTIVE_STARTED.json",
                {
                    "dataset": key,
                    "started_monotonic": time.monotonic(),
                    "candidate_sha256": receipt["candidate_sha256"],
                },
            )
        return io.load_endpoints(key, config)

    _, table = calculate(config, loader, freeze)
    io.write_new(output / "diagnostics.json", table)
    if table["status"] == "DEGENERATE_INCONCLUSIVE":
        io.write_new(
            output / "DEGENERATE.json", {"U_norm": table["U_norm"], "no_substitution": True}
        )
    print(json.dumps({k: table[k] for k in ("status", "U_norm", "construction_u_cosine")}))


def run():
    started = time.monotonic()
    config = io.read(ROOT / io.CONFIG)
    usage = json.loads(os.environ.get("SP_LENSE_USAGE_PREFLIGHT", "null"))
    io.require(
        isinstance(usage, dict)
        and type(usage.get("standard_used_percent")) in (int, float)
        and 0 <= usage["standard_used_percent"] < 90
        and type(usage.get("checked_at_unix")) in (int, float)
        and 0 <= time.time() - usage["checked_at_unix"] <= 60,
        "fresh usage below90 required",
    )
    source = io.identity(config)
    output = ROOT / config["output_namespace"]
    output.mkdir(parents=True, exist_ok=False)
    io.write_new(
        output / "provenance.json",
        {
            "source_commit": io.git("rev-parse", "HEAD"),
            "recipe_commit": "847b71c",
            "source_sha256": source,
            "config": config,
            "python": sys.version,
            "no_model_or_tokenizer_loading": True,
        },
    )
    return io.bounded(SCRIPT, "_analyze", output, usage, started=started)


if __name__ == "__main__":
    if sys.argv[1:] == ["_analyze"]:
        analyze()
    elif sys.argv[1:] == ["run"]:
        result = run()
        print(json.dumps(result, indent=2))
        if result["status"] != "completed":
            raise SystemExit(1)
    else:
        raise SystemExit("Use run; no alternate construction or input arguments.")
