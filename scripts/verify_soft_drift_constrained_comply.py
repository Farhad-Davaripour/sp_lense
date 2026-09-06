"""Independent compact raw-record/QP/geometry audit; no solver or model imports.

Native arrays are inspected in memory, never copied into the scientific report.
The unchanged interval checker certifies only the recorded raw local QP point;
clipped/applied predictions and observed nonlinear acceptance remain separate.
"""

from __future__ import annotations

import argparse
import dis
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import soft_drift_constrained_comply_plan as protocol
from scripts import verify_soft_drift_qp as numeric
from scripts.paired_common_drift_comply_1800_binding import load_bound

FROZEN_CHECKER_SHA256 = "31eeceaa34a9049ab745e5ca4593513054dba882f261f7d3a9b007327c34ddb5"
inherited = load_bound(
    "scripts._soft_drift_constrained_independent_checker",
    "scripts/verify_paired_common_drift_comply.py",
    FROZEN_CHECKER_SHA256,
    (
        (
            "from scripts import paired_common_drift_comply_plan as protocol",
            "from scripts import soft_drift_constrained_comply_plan as protocol",
            1,
        ),
        (
            '"scripts._paired_common_drift_independent_base"',
            '"scripts._soft_drift_constrained_independent_base"',
            1,
        ),
    ),
)
legacy, engine = inherited.legacy, inherited.engine
OUTPUT = ROOT / protocol.OUTPUT
require, read, sha = protocol.require, protocol.read, protocol.sha
norm, vector_sha = engine.norm, engine.vector_sha
accepts, quality = engine.accepts, engine.quality
AUDIT_MATCH = inherited.AUDIT_MATCH
FAMILIES = inherited.FAMILIES
PAIR_INDICES = ((2, 0), (3, 1), (6, 4), (7, 5), (10, 8), (11, 9))
arithmetic_match = inherited.arithmetic_match
_raw_verify_data = inherited._raw_verify_data
_behavior_summary = inherited._behavior_summary
_old_compare_summary = inherited._old_compare_summary
_inherited_replay = inherited.replay

# Two isolated deadline comparisons only; raw-scoring/geometry code is unchanged.
_old_code = engine.verify.__code__
require(
    sum(type(x) is int and x == 1200 for x in _old_code.co_consts) == 1
    and sum(
        x.opname == "LOAD_CONST" and x.argval == 1200 for x in dis.get_instructions(engine.verify)
    )
    == 2,
    "exact inherited deadline sites",
)
_new_code = _old_code.replace(
    co_consts=tuple(1800 if type(x) is int and x == 1200 else x for x in _old_code.co_consts)
)
require(_new_code.co_code == _old_code.co_code, "unchanged raw verifier bytecode")
engine.verify.__code__ = _new_code

VECTOR_KEYS = ("w_before", "d", "s", "u", "w_after", "r")
GEOMETRY_KEYS = (
    "stage",
    "status",
    "path_before",
    "path_after",
    "w_before_sha256",
    "w_after_sha256",
    "d_norm",
    "clip_factor",
    "proposed_step_norm",
    "unprojected_net_norm",
    "projection_factor",
    "projection_distance",
    "step_norm",
    "net_norm",
    "proposed_path_after",
)
ROW_KEYS = (
    "cell_id",
    "condition",
    "actual_next_token_label",
    "baseline_label",
    "comply_label",
    "signed_margin",
    "signed_delta_log_odds",
    "answer_pair_mass",
    "kl_from_baseline",
    "net_relative_norm",
    "path_relative_norm",
    "maximum_delta_error",
    "maximum_offset_error",
    "maximum_step_error",
    "maximum_current_logit_difference",
    "maximum_current_h_difference",
    "requested_accepted",
    "quality_valid",
    "shared_w_sha256",
    "logits_file",
    "logits_sha256",
)


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode()


def _number(value):
    require(type(value) in (int, float) and math.isfinite(value), "finite nonboolean number")
    return float(value)


def _vector(value, n=None):
    require(
        isinstance(value, list) and 1 <= len(value) <= 1024 and (n is None or len(value) == n),
        "native vector shape",
    )
    return [_number(x) for x in value]


def _sha(value):
    require(
        type(value) is str and len(value) == 64 and all(x in "0123456789abcdef" for x in value),
        "lowercase SHA256",
    )
    return value


def geometry_match(actual, expected, label="local geometry"):
    """Fixed scalar/prediction tolerance; native geometry arrays stay exact."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label + " keys")
        for key, value in expected.items():
            geometry_match(actual[key], value, label + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + " shape")
        for a, b in zip(actual, expected, strict=True):
            geometry_match(a, b, label)
    elif isinstance(expected, float):
        require(
            abs(_number(actual) - expected) <= 1e-12 * max(1.0, abs(expected)),
            label + " arithmetic",
        )
    else:
        require(type(actual) is type(expected) and actual == expected, label + " identity")


def _bound_rows(states, baselines):
    rows = inherited._ordered_rows(states)
    if isinstance(baselines, dict):
        by_id = {k: v.get("row", v) for k, v in baselines.items()}
    else:
        by_id = {row["prompt_id"]: row for row in inherited._ordered_rows(baselines)}
    require(set(by_id) == {row["prompt_id"] for row in rows}, "exact baseline membership")
    baseline = [by_id[row["prompt_id"]] for row in rows]
    inherited._ordered_rows(baseline)
    n = len(_vector(rows[0]["shared_w"]))
    for row, base in zip(rows, baseline, strict=True):
        h0 = _vector(base["h0"], n)
        hn = norm(h0)
        require(
            base["condition"] == "baseline"
            and base["cell_id"] == row["prompt_id"] + "__baseline"
            and _vector(base["shared_w"], n) == [0.0] * n
            and row["baseline_cell_id"] == base["cell_id"]
            and row["baseline_margin"] == base["preserve_log_odds"]
            and row["h0"] == base["h0"]
            and row["h0_norm"] == base["h0_norm"] == hn
            and hn > 0,
            "own original baseline and norm",
        )
        require(
            _vector(row["shared_w"], n) == rows[0]["shared_w"]
            and row["shared_w_sha256"] == vector_sha(row["shared_w"]),
            "all paired rows at the same exact current vector",
        )
        _number(row["preserve_log_odds"])
        _number(base["preserve_log_odds"])
    return rows, baseline


def reference_objective(states, baselines):
    """Descriptive current response proxies; never an additional acceptance gate."""
    rows, base = _bound_rows(states, baselines)
    margins = [-_number(row["preserve_log_odds"]) for row in rows]
    original = [-_number(row["preserve_log_odds"]) for row in base]
    drift = [
        ((margins[a] - original[a]) - (margins[b] - original[b])) / 2.0 for a, b in PAIR_INDICES
    ]
    return {
        "comply_margins": margins,
        "baseline_relative_common_letter_drift": drift,
        "behavioral_acceptance_gate": False,
        "local_objective_descent_guaranteed": False,
    }


def assemble_current(gradients, baselines, w, stage):
    """Independently reconstruct signed raw rows and six affine drift rows."""
    rows, base = _bound_rows(gradients, baselines)
    w = _vector(w)
    require(type(stage) is int and 1 <= stage <= 8, "one of eight attempted stages")
    require(rows[0]["shared_w"] == w, "exact current-w binding")
    A = []
    for row in rows:
        previous = "baseline" if stage == 1 else f"step_{stage - 1}"
        require(
            row["condition"] == f"gradient_{stage}"
            and row["stage"] == stage
            and row["cell_id"] == row["prompt_id"] + f"__gradient_{stage}"
            and row["current_cell_id"] == row["prompt_id"] + "__" + previous,
            "exact gradient/cache lineage",
        )
        require(
            0 <= _number(row["maximum_current_logit_difference"]) <= 1e-6
            and 0 <= _number(row["maximum_current_h_difference"]) <= 1e-6,
            "current-logit/current-hidden identity",
        )
        hn = norm(row["h0"])
        A.append([-hn * x for x in _vector(row["gradient"], len(w))])
    b = [0.10 + _number(row["preserve_log_odds"]) for row in rows]
    m = [-_number(row["preserve_log_odds"]) for row in rows]
    m0 = [-_number(row["preserve_log_odds"]) for row in base]
    c = [((m[a] - m0[a]) - (m[brow] - m0[brow])) / 2.0 for a, brow in PAIR_INDICES]
    D = [[(x - y) / 2.0 for x, y in zip(A[a], A[brow], strict=True)] for a, brow in PAIR_INDICES]
    a = [((m[ia] - m0[ia]) + (m[ib] - m0[ib])) / 2.0 for ia, ib in PAIR_INDICES]
    legacy.finite_tree([A, b, c, D, a])
    return {"A": A, "b": b, "c": c, "D": D, "margins": m, "semantic_proxy": a, "rows": rows}


def reference_geometry(d, w, path, stage):
    """Original clip, project, and actual-increment path accounting."""
    w, d, path = _vector(w), _vector(d, len(w)), _number(path)
    require(
        0 <= path <= 0.40 + 1e-12 and norm(w) <= 0.20 + 1e-12 and norm(w) <= path + 1e-12,
        "existing net/path caps",
    )
    dn = norm(d)
    clip = min(1.0, 0.05 / dn) if dn else 0.0
    s = [clip * x for x in d]
    u = [x + y for x, y in zip(w, s, strict=True)]
    un = norm(u)
    projection = 1.0 if un <= 0.20 else 0.20 / un
    after = list(u) if projection == 1.0 else [projection * x for x in u]
    r = [x - y for x, y in zip(after, w, strict=True)]
    sn, rn, net = norm(s), norm(r), norm(after)
    require(
        rn <= sn + 1e-12
        and rn <= 0.05 + 1e-12
        and net <= 0.20 + 1e-12
        and path + rn <= 0.40 + 1e-12,
        "actual projected step/net/path caps",
    )
    return {
        "stage": stage,
        "w_before": w,
        "w_before_sha256": vector_sha(w),
        "path_before": path,
        "d": d,
        "s": s,
        "u": u,
        "w_after": after,
        "r": r,
        "w_after_sha256": vector_sha(after),
        "d_norm": dn,
        "clip_factor": clip,
        "proposed_step_norm": sn,
        "unprojected_net_norm": un,
        "projection_factor": projection,
        "projection_distance": norm([x - y for x, y in zip(u, after, strict=True)]),
        "step_norm": rn,
        "net_norm": net,
        "path_after": path + rn,
        "proposed_path_after": path + sn,
        "status": "ready" if rn else "method_zero_increment" if dn == 0 else "projection_stall",
    }


def reference_predictions(inputs, moves):
    predictions, semantic = {}, {}
    for label, move in moves.items():
        changes = [math.fsum(x * y for x, y in zip(row, move, strict=True)) for row in inputs["A"]]
        predictions[label] = {
            "slacks": [x - b for x, b in zip(changes, inputs["b"], strict=True)],
            "predicted_comply_margins": [
                m + x for m, x in zip(inputs["margins"], changes, strict=True)
            ],
            "residual_common_letter_drift": [
                c + math.fsum(x * y for x, y in zip(row, move, strict=True))
                for c, row in zip(inputs["c"], inputs["D"], strict=True)
            ],
        }
        semantic[label] = [
            a + (changes[ia] + changes[ib]) / 2.0
            for a, (ia, ib) in zip(inputs["semantic_proxy"], PAIR_INDICES, strict=True)
        ]
    return predictions, semantic


def _solver_metadata(record, witness, n):
    """Check bounded claimed search accounting, without rerunning its search."""
    require(
        isinstance(record, dict)
        and record.get("status") == "KKT_ESTIMATE_ONLY"
        and record.get("reason") is None
        and record.get("maximum_masks") == 4096
        and record.get("dimension") == n
        and record.get("input_sha256") == witness["input_sha256"]
        and record.get("geometry_applied") is False
        and record.get("infeasibility_certified") is False
        and record.get("independent_checker_required") is True,
        "recorded solver estimate metadata",
    )
    count = record["masks_visited"]
    statuses = record["mask_status_counts"]
    require(
        type(count) is int
        and count == witness["active_mask"] + 1
        and isinstance(statuses, dict)
        and set(statuses) <= set("RDPFCX")
        and all(type(x) is int and 0 < x <= 4096 for x in statuses.values())
        and sum(statuses.values()) == count
        and statuses.get("C") == 1
        and not statuses.get("X"),
        "bounded increasing-mask claimed accounting",
    )
    _sha(record["mask_trace_sha256"])
    trace = record["mask_trace"]
    require(
        type(trace) is str
        and len(trace) == count <= 4096
        and trace.endswith("C")
        and set(trace) <= set("RDPFC")
        and {key: trace.count(key) for key in sorted(set(trace))} == statuses
        and sha(trace.encode()) == record["mask_trace_sha256"],
        "full bounded mask trace structure and hash",
    )
    legacy.finite_tree(record)
    return {
        "status": record["status"],
        "masks_visited": count,
        "maximum_masks": 4096,
        "mask_status_counts": statuses,
        "mask_trace_sha256": record["mask_trace_sha256"],
        "mask_search_independently_replayed": False,
    }


def verify_update(update, gradients, w, path, baselines):
    """No rescue: reject an uncertified point; only return bounded scalar evidence."""
    require(isinstance(update, dict), "update record object")
    inputs = assemble_current(gradients, baselines, w, update["stage"])
    require(update["status"] != "NUMERICALLY_UNRESOLVED", "unresolved QP is technical INCONCLUSIVE")
    d = _vector(update["d"], len(w))
    witness = update["qp_witness"]
    require(
        isinstance(witness, dict)
        and set(witness) == {"multipliers", "active_mask", "input_sha256"},
        "exact compact QP witness keys",
    )
    certificate = numeric.verify(inputs["A"], inputs["b"], inputs["c"], {"vector": d, **witness})
    require(certificate["accepted"] is True, "independent raw QP interval certificate rejected")
    require(update["numeric_certificate"] == certificate, "exact independent interval certificate")
    source = {
        "A_row_sha256": [vector_sha(row) for row in inputs["A"]],
        "b": inputs["b"],
        "c": inputs["c"],
        "D_row_sha256": [vector_sha(row) for row in inputs["D"]],
    }
    require(update["qp_inputs"] == source, "exact independently reconstructed QP inputs")
    expected = reference_geometry(d, w, path, update["stage"])
    predictions, semantic = reference_predictions(
        inputs, {"raw": d, "clipped": expected["s"], "applied": expected["r"]}
    )
    required = set(expected) | {
        "gradient_cell_ids",
        "solver",
        "numeric_certificate",
        "qp_inputs",
        "qp_witness",
        "predictions",
        "infeasibility_certified",
        "raw_feasibility_is_applied_feasibility",
    }
    require(set(update) == required, "exact update schema; no unreviewed fields")
    for key, value in expected.items():
        if key in VECTOR_KEYS:
            require(
                _vector(update[key], len(w)) == value
                and vector_sha(update[key]) == vector_sha(value),
                "exact independently applied binary64 geometry",
            )
        elif key.endswith("sha256"):
            require(
                update[key] == value == vector_sha(update[key.removesuffix("_sha256")]),
                "actual and independently reconstructed vector hash",
            )
        else:
            geometry_match(update[key], value, "geometry." + key)
    require(update["w_before"] == w and update["path_before"] == path, "exact update origin")
    geometry_match(update["predictions"], predictions, "three distinct local predictions")
    ids = [row["cell_id"] for row in inputs["rows"]]
    require(
        update["gradient_cell_ids"] == ids
        and update["infeasibility_certified"] is False
        and update["raw_feasibility_is_applied_feasibility"] is False,
        "gradient references and limited claims",
    )
    solver = _solver_metadata(update["solver"], witness, len(w))
    result = {
        **{key: update[key] for key in GEOMETRY_KEYS},
        "gradient_cell_ids": ids,
        "update_canonical_json_sha256": sha(encoded(update)),
        "vector_sha256": {key: vector_sha(update[key]) for key in VECTOR_KEYS},
        "qp_inputs": source,
        "qp_witness": witness,
        "numeric_certificate": certificate,
        "solver": solver,
        "predictions": predictions,
        "predicted_semantic_proxy": semantic,
        "recorded_gradient_arithmetic": "MATCH",
        "real_derivatives_independently_rerun": False,
        "raw_feasibility_is_applied_feasibility": False,
        "nonlinear_acceptance_inferred_from_qp": False,
    }
    require(len(encoded(result)) <= 32768, "compact QP update certificate allocation")
    return result


def replay(plan, rows, updates, skips, output, events):
    # The inherited implementation resolves this module's independently replaced
    # verify_update via its isolated globals and retains exact first-accept replay.
    result, audits = _inherited_replay(plan, rows, updates, skips, output, events)
    for line, item in enumerate(audits, 1):
        item["updates_jsonl_line"] = line
        require(len(encoded(item)) <= 32768, "compact update with journal reference")
    return result, audits


def summary(rows, result):
    value = _behavior_summary(rows, result)
    baselines = {row["prompt_id"]: row for row in rows if row["condition"] == "baseline"}
    value["paired_response_trajectory"] = [
        {"condition": condition, **reference_objective(group, baselines)}
        for condition in ("baseline", *(f"step_{i}" for i in range(1, 9)), "final")
        if (group := [row for row in rows if row["condition"] == condition])
    ]
    return value


def compare_summary(actual, expected):
    key = "paired_response_trajectory"
    _old_compare_summary(
        {k: v for k, v in actual.items() if k != key},
        {k: v for k, v in expected.items() if k != key},
    )
    arithmetic_match(actual[key], expected[key], "independent response trajectory")


def compact_row(row, line):
    values = {"maximum_step_error": None, **row}
    result = {key: values[key] for key in ROW_KEYS}
    require(
        all(type(x) in (str, int, float, bool) or x is None for x in result.values()),
        "scalar row whitelist",
    )
    result.update(
        rows_jsonl_line=line,
        recorded_gradient_present=row["gradient"] is not None,
        reconstruction="MATCH",
    )
    require(len(encoded(result)) <= 2048, "compact row allocation")
    return result


def verify_plan(plan):
    require(plan == protocol.build_plan(), "exact prospective QP plan")
    legacy.verify_layout(plan)
    inherited._ordered_rows(plan["prompts"])
    for path, digest in plan["input_sha256"].items():
        require(sha((ROOT / path).read_bytes()) == digest, "authenticated prospective input")
    return {"rendering_rows": 12, "pairs": 6, "maximum_forwards": 216, "maximum_derivatives": 96}


def verify_data(plan, rows, output):
    # Capture canonical scores only AFTER the immutable raw validator checks them.
    # No second logits decode, no model call, no retained cross-audit state.
    canonical = []
    previous = engine.summary

    def capture_summary(values, result):
        canonical.extend(values)
        return summary(values, result)

    engine.summary = capture_summary
    try:
        result = _raw_verify_data(plan, rows, output)
    finally:
        engine.summary = previous
    require(len(canonical) == len(rows), "one canonical row per raw journal row")
    result["row_checks"] = [compact_row(row, i) for i, row in enumerate(canonical, 1)]
    # The inherited wrapper has already checked/augmented these scalar rows;
    # every one is represented in row_checks, so do not serialize them twice.
    result.pop("construction_stages")
    return result


def verify():
    verify_plan(read(OUTPUT / "preregistration.json")["plan"])
    for path in OUTPUT.iterdir():
        if path.is_file() and path.suffix == ".json":
            legacy.finite_tree(read(path))
    result = engine.verify()
    if result["status"] != "INCONCLUSIVE":
        arithmetic_match(
            read(OUTPUT / "analysis.json")["paired_response_trajectory"],
            result["summary"]["paired_response_trajectory"],
            "response trajectory",
        )
    return result


def prepare_candidate_records(result, verification_sha256, output=None):
    """Pure preparation: read audited endpoint, return payload; never open a writer."""
    output = Path(OUTPUT if output is None else output)
    _sha(verification_sha256)
    if result.get("status") != AUDIT_MATCH or not result.get("summary", {}).get(
        "candidate_eligible"
    ):
        return {}
    value = result["summary"]
    require(
        value["candidate_eligible"] is True
        and type(value["final_accepted"]) is int
        and value["final_accepted"] == 12
        and len(value["final_cells"]) == 12,
        "all twelve independently accepted final rows",
    )
    require(
        not any(
            (output / name).exists()
            for name in ("INVALID.json", "VERIFICATION_FAILURE.json", "RECORDING_FAILURE.json")
        ),
        "no technical fault before candidate preparation",
    )
    endpoint, run_result = read(output / "endpoint.json"), read(output / "result.json")
    w = _vector(endpoint["w"], 1024)
    require(
        norm(w) <= 0.20 + 1e-12
        and w == run_result["w"]
        and vector_sha(w)
        == endpoint["vector_float64_le_sha256"]
        == run_result["vector_float64_le_sha256"]
        and run_result["final_cell_ids"] == [row["cell_id"] for row in value["final_cells"]]
        and all(row["accepted"] is True for row in value["final_cells"])
        and sha((output / "endpoint.json").read_bytes()) == result["audited_endpoint_sha256"]
        and sha((output / "result.json").read_bytes()) == result["audited_result_sha256"],
        "audited unnormalized native endpoint identity",
    )
    return {
        "comply_vector.json": {
            "vector": w,
            "vector_float64_le_sha256": vector_sha(w),
            "final_cell_ids": run_result["final_cell_ids"],
            "requested": "comply",
            "construction_only": True,
            "transfer_ran": False,
            "training_families": list(FAMILIES),
            "verification_sha256": verification_sha256,
            "endpoint_sha256": result["audited_endpoint_sha256"],
            "valid_only_with_complete_final_recording_inventory": True,
        }
    }


def render_report(result):
    if result.get("status") != AUDIT_MATCH:
        return "# Soft-drift constrained COMPLY\n\nINCONCLUSIVE; no candidate and no retry.\n"
    value = result["summary"]
    lines = [
        "# Soft-drift constrained COMPLY",
        "",
        f"Audit: {result['status']}; construction: {value['status']}; stop: {value['stop_reason']}.",
        f"Final acceptance {value['final_accepted']}/12; forwards {value['forward_count']}/216; derivatives {value['derivative_count']}/96.",
        f"Applied updates {value['updates']}/{value['attempted_updates']}; native net {value['shared_net']}; actual path {value['shared_path']}.",
        "",
        "| Rendering | COMPLY margin | Pair mass | Raw KL | Accepted |",
        "|---|---:|---:|---:|---|",
    ]
    for row in value["final_cells"]:
        lines.append(
            f"| {row['cell_id']} | {row['signed_margin']:+.12g} | {row['answer_pair_mass']:.12g} | {row['kl_from_baseline']:.12g} | {row['accepted']} |"
        )
    lines += [
        "",
        "QP certificates apply only to raw local proposals. Clipped and projected increments have separately reported slacks and drift predictions; they do not imply observed answer changes.",
        "Recorded derivative arithmetic is reconstructed; real derivatives are not rerun. Exact feasibility, nonlinear descent, causal features, transfer, gate efficacy and general reliability are not established. Three development/training situations only; previous findings remain unchanged. Native endpoints are not renormalized. STOP.",
        "",
    ]
    return "\n".join(lines)


report = render_report


def finalize_recording(budget, capture_receipt, runtime_status, *, audit=None):
    from scripts import soft_drift_constrained_comply_recording as recording

    return recording.finalize_recording(
        budget,
        capture_receipt,
        runtime_status,
        audit=verify if audit is None else audit,
        prepare_candidate_records=lambda result, digest: prepare_candidate_records(
            result, digest, budget.root
        ),
        render_report=render_report,
    )


def read_sealed_recording(output=None):
    from scripts import soft_drift_constrained_comply_recording as recording

    return recording.read_sealed_recording(
        OUTPUT if output is None else output, audit_match=AUDIT_MATCH
    )


inherited.protocol = legacy.protocol = engine.protocol = protocol
inherited.OUTPUT = legacy.OUTPUT = engine.OUTPUT = OUTPUT
inherited.verify_update = verify_update
engine.replay, engine.summary, engine.verify_update = replay, summary, verify_update
engine.compare_summary, engine.verify_data = compare_summary, verify_data


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    disposition = read_sealed_recording()
    print(
        json.dumps(
            {
                k: v
                for k, v in disposition.items()
                if k not in {"summary", "construction_stages", "optimizer_checks", "row_checks"}
            },
            indent=2,
            allow_nan=False,
        )
    )
    if disposition["status"] == "INCONCLUSIVE":
        raise SystemExit(1)
