"""Independent fresh-row / exact serialized descent / complete schedule audit.

No optimizer, production gradient/projection, model, or tokenizer imports.
Raw scoring and complete finalization are authenticated unchanged predecessors.
"""

from __future__ import annotations

import argparse
import dis
import json
import math
import time
from fractions import Fraction as F
from pathlib import Path

from scripts import certified_descent_comply_plan as protocol
from scripts import verify_certified_descent_dyadic as numeric
from scripts.paired_common_drift_comply_1800_binding import load_bound

ROOT = Path(__file__).resolve().parents[1]
inherited = load_bound(
    "scripts._certified_descent_comply_independent",
    "scripts/verify_soft_drift_constrained_comply.py",
    "f96fa37c84c8c1d6b5cd2f432a7fb3342744069170c670ca61c15176c02e47d6",
    (
        (
            "from scripts import soft_drift_constrained_comply_plan as protocol",
            "from scripts import certified_descent_comply_plan as protocol",
            2,
        ),
        (
            '"scripts._soft_drift_constrained_independent_checker"',
            '"scripts._certified_descent_independent_checker"',
            1,
        ),
        (
            '"scripts._soft_drift_constrained_independent_base"',
            '"scripts._certified_descent_independent_base"',
            1,
        ),
        (
            "from scripts import soft_drift_constrained_comply_recording as recording",
            "from scripts import certified_descent_comply_recording as recording",
            2,
        ),
    ),
)
engine, legacy = inherited.engine, inherited.legacy
OUTPUT = ROOT / protocol.OUTPUT
require, read, sha = protocol.require, protocol.read, protocol.sha
norm, vector_sha = inherited.norm, inherited.vector_sha
accepts, quality, AUDIT_MATCH = inherited.accepts, inherited.quality, inherited.AUDIT_MATCH
PAIR_INDICES = ((2, 0), (3, 1), (6, 4), (7, 5), (10, 8), (11, 9))
_vector, _number = inherited._vector, inherited._number
reference_objective, summary = inherited.reference_objective, inherited.summary
encoded, arithmetic_match = inherited.encoded, inherited.arithmetic_match
_original_assemble = inherited.assemble_current
_original_raw_verify = engine.verify
_usage_code = _original_raw_verify.__code__
require(
    sum(type(v) is int and v == 90 for v in _usage_code.co_consts) == 1
    and sum(
        i.opname == "LOAD_CONST" and type(i.argval) is int and i.argval == 90
        for i in dis.get_instructions(_original_raw_verify)
    )
    == 1,
    "one authenticated inherited usage ceiling site",
)
_usage_updated = _usage_code.replace(
    co_consts=tuple(101 if type(v) is int and v == 90 else v for v in _usage_code.co_consts)
)
require(
    _usage_updated.co_code == _usage_code.co_code,
    "usage-only constant binding; raw verifier bytecode unchanged",
)
_original_raw_verify.__code__ = _usage_updated


def _raw_verify_with_usage_guard():
    # The old <101 guard is redundant behind this supervisor-authorized exact
    # finite <=100 check. No scoring, model, timing or numerical-method change.
    value = read(engine.OUTPUT / "RUN_STARTED.json")["usage_preflight"]["standard_used_percent"]
    require(
        type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100,
        "finite nonboolean standard usage at most100",
    )
    return _original_raw_verify()


engine.verify = _raw_verify_with_usage_guard


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _same(actual, expected, label):
    require(canonical(actual) == canonical(expected), label)


def assemble_current(gradients, baselines, w, stage, history, path_upper):
    # The independent predecessor reconstructs row identities, own h0 norms,
    # fresh current-gradient lineage and original binary64 A/b/c only.
    inputs = _original_assemble(gradients, baselines, w, stage)
    require(
        all(vector_sha(row["shared_w"]) == vector_sha(w) for row in inputs["rows"]),
        "every fresh row at exact current-w bytes",
    )
    require(type(history) is list and len(history) == stage, "exact applied history length")
    require(type(path_upper) is str, "canonical rational path string")
    problem = {k: inputs[k] for k in ("A", "b", "c")}
    problem.update(w=_vector(w), w_sha256=vector_sha(w), history=history, path_upper=path_upper)
    d_exact = [
        [str((F(x) - F(y)) / 2) for x, y in zip(problem["A"][a], problem["A"][b], strict=True)]
        for a, b in PAIR_INDICES
    ]
    references = {
        "A_row_sha256": [vector_sha(row) for row in problem["A"]],
        "b": problem["b"],
        "c": problem["c"],
        "D_exact_row_sha256": [sha(canonical(row)) for row in d_exact],
    }
    return problem, references, inputs["rows"]


def reference_proposal(context):
    """Reconstruct the prescribed one floating proposal, not call a generator.

    Independent implementation of the frozen finite sums and analytic ball cases.
    Admission still requires exact serialized checks; nominal geometry is not a gate.
    """
    A, b, c, D = (context[k] for k in ("A", "b", "c", "D"))
    n, M, w, rho = context["dimension"], context["M"], context["w"], context["rho"]
    g = []
    for j in range(n):
        numeric.auth._tick(context["deadline"])
        margin = math.fsum(row[j] * max(0.0, rhs) for row, rhs in zip(A, b, strict=True)) / 12.0
        drift = math.fsum(row[j] * cp for row, cp in zip(D, c, strict=True)) / 24.0
        g.append(_number(0.0 - margin + drift))
    y = [_number(-v / M) for v in g]
    length = math.hypot
    zero = [0.0] * n
    yn = length(*y)
    require(0 < rho <= 0.05 and length(*w) <= 0.20, "nominal projection origin")
    if yn <= rho and length(*(v + q for v, q in zip(y, w, strict=True))) <= 0.20:
        p, ns, nn, ls, ln, case = list(y), zero, zero, 0.0, 0.0, "interior"
    else:
        if yn <= rho:
            sp, sl = list(y), 0.0
        else:
            scale = rho / yn
            sp, sl = [scale * v for v in y], yn / rho - 1.0
        if length(*(v + q for v, q in zip(sp, w, strict=True))) <= 0.20:
            p, ns, nn, ls, ln, case = sp, [sl * v for v in sp], zero, sl, 0.0, "step_only"
        else:
            shifted = [v + q for v, q in zip(y, w, strict=True)]
            tn = length(*shifted)
            if tn <= 0.20:
                np, nl = list(y), 0.0
            else:
                scale = 0.20 / tn
                np, nl = [scale * v - q for v, q in zip(shifted, w, strict=True)], tn / 0.20 - 1.0
            if length(*np) <= rho:
                p, ns, nn, ls, ln, case = (
                    np,
                    zero,
                    [nl * (v + q) for v, q in zip(np, w, strict=True)],
                    0.0,
                    nl,
                    "net_only",
                )
            else:
                distance = length(*w)
                require(distance > 0, "nondegenerate two-boundary origin")
                axis = [v / distance for v in w]
                ay = math.fsum(v * q for v, q in zip(y, axis, strict=True))
                perpendicular = [v - ay * q for v, q in zip(y, axis, strict=True)]
                pn = length(*perpendicular)
                ap = (0.20 * 0.20 - rho * rho - distance * distance) / (2.0 * distance)
                hs = rho * rho - ap * ap
                require(hs > 0 and pn > 0, "nondegenerate analytic circle")
                height = math.sqrt(hs)
                scale = height / pn
                p = [ap * q + scale * v for q, v in zip(axis, perpendicular, strict=True)]
                total = pn / height
                ln, ls = (ay - total * ap) / distance, None
                ls = total - 1.0 - ln
                ns, nn, case = (
                    [ls * v for v in p],
                    [ln * (v + q) for v, q in zip(p, w, strict=True)],
                    "both_active",
                )
    require(ls >= 0 and ln >= 0, "strict projection normal signs")
    residual = [v - yi + a + b for v, yi, a, b in zip(p, y, ns, nn, strict=True)]
    rn = length(*residual)
    scale = max(1.0, length(*y), length(*p), math.hypot(length(*ns), length(*nn)))
    limit = 128.0 * n * 2.0**-52 * scale
    require(rn <= limit, "frozen nominal normal-residual screen")
    result = {
        "y": y,
        "p": p,
        "M": M,
        "projection": {
            "case": case,
            "step_multiplier": ls,
            "net_multiplier": ln,
            "residual_norm": rn,
            "residual_limit": limit,
            "step_normal": ns,
            "net_normal": nn,
        },
    }
    legacy.finite_tree(result)
    return result


def compact_certificate(certificate):
    result = dict(certificate)
    displacement = result.pop("actual_displacement")
    result["actual_displacement_sha256"] = sha(canonical(displacement))
    result["actual_displacement_dimension"] = len(displacement)
    return result


def _solver_audit(record, context):
    require(type(record) is dict, "raw solver journal object")
    required = {
        "status",
        "terminal_reason",
        "problem_sha256",
        "gradient_count",
        "proposal_count",
        "trial_count",
        "trials",
        "chosen_j",
        "first_passing_j",
        "proposal",
        "zero_certificate",
        "descriptive_gap",
        "diagnostic_count",
        "observer_calls",
        "step_admitted",
        "exact_zero_optimum",
        "maximum_trials",
        "scale_indices",
        "combined_seconds_limit",
        "point_repaired",
        "alternate_proposal_used",
        "near_optimality_gate",
        "mathematical_trial_pass_is_not_step_issuance",
        "model_calls_authorized",
        "serialization_checked",
        "dimension",
        "fixed_M",
        "elapsed_seconds",
        "elapsed_sample",
        "selected_trial_j",
    }
    require(
        set(record) == required, "complete exact immutable solver schema; no technical fallback"
    )
    require(record["problem_sha256"] == context["_sealed"].problem_sha256, "solver problem hash")
    require(
        record["status"] in {"ADMITTED_DESCENT", "EXACT_ZERO_OPTIMUM", "NO_CERTIFIED_STEP"},
        "frozen terminal status",
    )
    require(
        record["point_repaired"] is False
        and record["alternate_proposal_used"] is False
        and record["near_optimality_gate"] is False
        and record["model_calls_authorized"] is False,
        "no new method or implied model authority",
    )
    require(record.get("serialization_checked") is True, "complete raw solver serialization")
    _number(record["elapsed_seconds"])
    require(0 <= record["elapsed_seconds"] <= 10, "bounded recorded solver elapsed")
    trials = record["trials"]
    require(
        type(trials) is list and len(trials) == record["trial_count"] <= 53,
        "bounded complete fixed trial count",
    )
    require(
        record["observer_calls"] == len(trials) + 1
        and record["mathematical_trial_pass_is_not_step_issuance"] is True,
        "recording-only observer and issuance distinction",
    )
    zero = numeric.exact_zero(context)
    _same(record.get("zero_certificate"), zero, "exact-zero proof reconstructed")
    require(
        record.get("failure_receipt_only") is not True,
        "incomplete solver witness is technical INCONCLUSIVE",
    )
    require(
        record["dimension"] == context["dimension"]
        and record["fixed_M"] == context["M"]
        and record["maximum_trials"] == 53
        and record["scale_indices"] == [0, 52]
        and record["combined_seconds_limit"] == 10.0,
        "frozen solver constants",
    )
    endpoint, selected = None, None
    if zero["exact_zero_optimum"]:
        require(
            record["status"] == "EXACT_ZERO_OPTIMUM"
            and not trials
            and record["gradient_count"]
            == record["proposal_count"]
            == record["diagnostic_count"]
            == 0
            and record["exact_zero_optimum"] is True
            and record["step_admitted"] is False
            and record["proposal"] is None,
            "exact-zero-only terminal branch",
        )
        require(
            record["terminal_reason"] == zero["reason"] and record["descriptive_gap"] is None,
            "exact-zero terminal proof, no diagnostic gate",
        )
    else:
        require(
            record["gradient_count"] == record["proposal_count"] == 1,
            "one original gradient/projection",
        )
        proposal = reference_proposal(context)
        _same(record["proposal"], proposal, "independently reconstructed prescribed proposal")
        require(trials, "complete numerical ray journal required")
        for j, trial in enumerate(trials):
            require(
                set(trial) == {"j", "lambda", "certificate", "check_status"}
                and trial["j"] == j
                and trial["lambda"] == math.ldexp(1.0, -j)
                and trial["check_status"] == "COMPLETED",
                "contiguous complete dyadic prefix",
            )
            scaled = [trial["lambda"] * v for v in proposal["p"]]
            candidate = [v + q for v, q in zip(context["w"], scaled, strict=True)]
            certificate = numeric.check(context, candidate)
            _same(
                trial["certificate"],
                compact_certificate(certificate),
                "exact serialized trial certificate and displacement hash",
            )
            if certificate["status"] != "REJECTED_TRIAL":
                require(
                    j == len(trials) - 1,
                    "first passing or serialized-zero point terminates immediately",
                )
                if certificate["status"] == "ADMITTED_DESCENT":
                    endpoint, selected = candidate, certificate
        if selected is not None:
            j = len(trials) - 1
            require(
                record["status"] == "ADMITTED_DESCENT"
                and record["step_admitted"] is True
                and record["exact_zero_optimum"] is False
                and record["chosen_j"]
                == record["first_passing_j"]
                == record["selected_trial_j"]
                == j
                and record["diagnostic_count"] == 1
                and record["terminal_reason"] == "FIRST_CERTIFIED_DYADIC_TRIAL",
                "first passing admission only",
            )
            _same(
                record["descriptive_gap"],
                numeric.diagnostic(context, endpoint),
                "one descriptive diagnostic, never gate",
            )
        else:
            require(
                record["status"] == "NO_CERTIFIED_STEP"
                and record["step_admitted"] is False
                and record["exact_zero_optimum"] is False
                and record["chosen_j"] is None
                and record["first_passing_j"] is None
                and record["selected_trial_j"] is None
                and record["diagnostic_count"] == 0
                and record["descriptive_gap"] is None,
                "finite no-step, not stationarity",
            )
            last = trials[-1]["certificate"]
            require(
                (
                    len(trials) == 53
                    and last["status"] == "REJECTED_TRIAL"
                    and record["terminal_reason"] == "FIXED_DYADIC_SCHEDULE_EXHAUSTED"
                )
                or (
                    last["status"] == "NO_CERTIFIED_STEP"
                    and record["terminal_reason"] == "serialized_zero_increment"
                ),
                "fixed ray exhausted or increment erased",
            )
    if selected is None:
        require(
            record["selected_trial_j"] is None
            and record["chosen_j"] is None
            and record["first_passing_j"] is None,
            "no selected trial without descent",
        )
    require(
        "w_next" not in record and "selected_certificate" not in record,
        "no duplicate native or certificate payload",
    )
    audit = {
        k: record[k]
        for k in (
            "status",
            "terminal_reason",
            "trial_count",
            "chosen_j",
            "first_passing_j",
            "gradient_count",
            "proposal_count",
            "diagnostic_count",
        )
    }
    audit.update(
        trial_journal_sha256=sha(canonical(trials)),
        proposal_sha256=sha(canonical(record["proposal"])),
        zero_certificate=zero,
        selected_certificate=None if selected is None else compact_certificate(selected),
        descriptive_gap=record["descriptive_gap"],
        maximum_trials=53,
    )
    return audit, endpoint, selected


def verify_update(update, gradients, w, path, baselines, history=None, path_upper=None):
    require(type(update) is dict, "raw update object")
    required = {
        "stage",
        "status",
        "terminal_reason",
        "adapter_fault",
        "technical_failure",
        "solver",
        "combined_seconds_limit",
        "step_admitted",
        "exact_zero_optimum",
        "near_optimality_gate",
        "point_repaired",
        "nonlinear_acceptance_inferred",
        "float_geometry_is_descriptive",
        "serialization_checked",
        "elapsed_seconds",
        "elapsed_sample",
        "gradient_cell_ids",
        "baseline_cell_ids",
        "w_before",
        "w_before_sha256",
        "path_before",
        "path_before_upper",
        "history_w_sha256",
        "problem_sha256",
        "inputs",
    }
    if update.get("status") == "ready":
        required |= {
            "w_after",
            "w_after_sha256",
            "path_after_upper",
            "path_after",
            "step_norm",
            "net_norm",
        }
    require(set(update) == required, "exact fully recorded adapter schema")
    require(
        update["adapter_fault"] is False
        and update["technical_failure"] is False
        and update["serialization_checked"] is True,
        "technical or incomplete no-step is INCONCLUSIVE",
    )
    require(
        update["combined_seconds_limit"] == 10.0
        and 0 <= _number(update["elapsed_seconds"]) < 10.0
        and update["near_optimality_gate"] is False
        and update["point_repaired"] is False
        and update["nonlinear_acceptance_inferred"] is False
        and update["float_geometry_is_descriptive"] is True,
        "frozen adapter accounting and limited claims",
    )
    if history is None:
        require(
            update["stage"] == 1 and w == [0.0] * len(w) and path == 0,
            "later audit needs independently replayed history",
        )
        history, path_upper = [list(w)], "0"
    problem, references, rows = assemble_current(
        gradients, baselines, w, update["stage"], history, path_upper
    )
    digest = sha(canonical(problem))
    context = numeric.prepare(problem, digest, time.monotonic() + 10.0)
    require(
        update["w_before"] == w
        and update["w_before_sha256"] == vector_sha(w)
        and update["path_before_upper"] == path_upper
        and update["path_before"] == path == float(F(path_upper)),
        "exact current vector and conservative path",
    )
    _same(update["inputs"], references, "independent original own-norm signed inputs")
    require(
        update["history_w_sha256"] == [vector_sha(v) for v in history]
        and update["problem_sha256"] == digest
        and update["gradient_cell_ids"] == [r["cell_id"] for r in rows],
        "history/problem/fresh-gradient identity",
    )
    require(
        update["baseline_cell_ids"] == [r["baseline_cell_id"] for r in rows],
        "own original baseline references",
    )
    solver, after, certificate = _solver_audit(update["solver"], context)
    expected_status = {
        "ADMITTED_DESCENT": "ready",
        "EXACT_ZERO_OPTIMUM": "exact_zero_optimum",
        "NO_CERTIFIED_STEP": "no_certified_step",
    }[solver["status"]]
    require(update["status"] == expected_status, "exact driver disposition")
    require(
        update["terminal_reason"] == solver["terminal_reason"]
        and update["step_admitted"] is (after is not None)
        and update["exact_zero_optimum"] is (expected_status == "exact_zero_optimum"),
        "adapter disposition matches complete immutable result",
    )
    result = {
        "stage": update["stage"],
        "status": expected_status,
        "gradient_cell_ids": update["gradient_cell_ids"],
        "update_canonical_json_sha256": sha(encoded(update)),
        "w_before_sha256": vector_sha(w),
        "problem_sha256": digest,
        "path_before_upper": path_upper,
        "path_before": path,
        "history_w_sha256": update["history_w_sha256"],
        "inputs": references,
        "solver": solver,
        "recorded_gradient_arithmetic": "MATCH",
        "real_derivatives_independently_rerun": False,
        "first_passing_verified": True,
        "exact_serialized_geometry_verified": True,
        "nominal_proposal_reconstructed": after is not None or solver["proposal_count"] == 1,
        "nonlinear_acceptance_inferred_from_surrogate": False,
    }
    if after is not None:
        require(
            update["w_after"] == after
            and vector_sha(update["w_after"]) == vector_sha(after) == update["w_after_sha256"],
            "exact admitted serialized endpoint",
        )
        require(
            update["path_after_upper"] == certificate["geometry"]["path_after_upper"]
            and update["path_after"] == float(F(update["path_after_upper"])),
            "conservative exact path carried forward",
        )
        require(
            update["step_norm"]
            == math.sqrt(float(F(certificate["geometry"]["actual_step_squared"])))
            and update["net_norm"] == math.sqrt(float(F(certificate["geometry"]["net_squared"]))),
            "descriptive norms derived from exact serialized squares",
        )
        result.update(
            {
                k: update[k]
                for k in (
                    "w_after_sha256",
                    "path_after_upper",
                    "path_after",
                    "step_norm",
                    "net_norm",
                )
            }
        )
    else:
        require(
            not set(update).intersection(
                {
                    "w_after",
                    "w_after_sha256",
                    "path_after",
                    "path_after_upper",
                    "step_norm",
                    "net_norm",
                }
            ),
            "no endpoint issued on stop",
        )
    require(len(encoded(result)) <= 32768, "compact independent update bound")
    return result


def replay(plan, rows, updates, skips, output, events):
    del events  # Immutable raw validator owns journal order/completion/timestamps.
    output = Path(output)
    by_id = {r["cell_id"]: r for r in rows}
    cells = {(c["prompt_id"], c["condition"]): c for c in plan["cells"]}
    executed, omitted, audits = [], [], []

    def take(pid, condition, w, current=None):
        row = by_id[cells[pid, condition]["cell_id"]]
        require(
            row["shared_w"] == w
            and row["current_cell_id"] == (current["cell_id"] if current else None),
            "exact row vector/cache lineage",
        )
        executed.append(row["cell_id"])
        return row

    def skip(group, reason, anchor):
        omitted.extend({"cell": c, "reason": reason, "after_cell_id": anchor} for c in group)

    w = [0.0] * plan["model"]["d_model"]
    path, path_upper, applied, attempted, ui, history = 0.0, "0", 0, 0, 0, [list(w)]
    current = [take(pid, "baseline", w) for pid in plan["construction_ids"]]
    baselines = {r["prompt_id"]: r for r in current}
    stop, anchor = engine.stop_for(current), current[-1]["cell_id"]
    for stage in range(1, 9):
        gc = [cells[pid, f"gradient_{stage}"] for pid in plan["construction_ids"]]
        sc = [cells[pid, f"step_{stage}"] for pid in plan["construction_ids"]]
        if stop:
            skip(gc + sc, stop, anchor)
            continue
        attempted += 1
        gradients = [
            take(pid, f"gradient_{stage}", w, r)
            for pid, r in zip(plan["construction_ids"], current, strict=True)
        ]
        anchor = gradients[-1]["cell_id"]
        if any(not quality(r) for r in gradients):
            stop = "quality_failure"
        else:
            require(ui < len(updates) and updates[ui]["stage"] == stage, "exact attempted update")
            update = updates[ui]
            audit = verify_update(update, gradients, w, path, baselines, history, path_upper)
            ui += 1
            audit["updates_jsonl_line"] = ui
            require(len(encoded(audit)) <= 32768, "compact journal reference bound")
            audits.append(audit)
            if update["status"] != "ready":
                stop = update["status"]
            else:
                w, path, path_upper, applied = (
                    update["w_after"],
                    update["path_after"],
                    update["path_after_upper"],
                    applied + 1,
                )
                history.append(list(w))
        if stop:
            skip(sc, stop, anchor)
            continue
        current = [
            take(pid, f"step_{stage}", w, r)
            for pid, r in zip(plan["construction_ids"], current, strict=True)
        ]
        anchor, stop = current[-1]["cell_id"], engine.stop_for(current)
    require(ui == len(updates), "no extra optimizer records")
    endpoint = {
        "w": w,
        "vector_float64_le_sha256": vector_sha(w),
        "path": path,
        "path_upper": path_upper,
        "history_w_sha256": [vector_sha(v) for v in history],
        "net": norm(w),
        "updates": applied,
        "attempted_updates": attempted,
        "stop_reason": stop or "max_updates",
        "endpoint_cell_ids": [r["cell_id"] for r in current],
    }
    _same(read(output / "endpoint.json"), endpoint, "last endpoint without selection or repair")
    final = [
        take(pid, "final", w, r) for pid, r in zip(plan["construction_ids"], current, strict=True)
    ]
    require(
        len(final) == 12 and not plan["control_ids"] and not plan["transfer_ids"],
        "original twelve final replays",
    )
    require(
        not any(
            (output / name).exists()
            for name in (
                "comply_vector.json",
                "candidate_freeze.json",
                "transfer_vector.json",
                "transfer_freeze.json",
            )
        ),
        "candidate cannot precede independent audit",
    )
    require(
        executed == [r["cell_id"] for r in rows] and len(rows) + len(skips) == 216,
        "complete nonpadded conditional schedule",
    )
    _same(
        omitted,
        [{k: s[k] for k in ("cell", "reason", "after_cell_id")} for s in skips],
        "exact deterministic skips",
    )
    result = {
        **endpoint,
        "final_cell_ids": [r["cell_id"] for r in final],
        "transfer_ran": False,
        "candidate_eligible": all(accepts(r) for r in final),
    }
    _same(read(output / "result.json"), result, "independent complete result replay")
    return result, audits


def render_report(result):
    if result.get("status") != AUDIT_MATCH:
        return "# Certified-descent COMPLY\n\nINCONCLUSIVE; no candidate or retry.\n"
    value = result["summary"]
    return (
        "# Certified-descent COMPLY\n\n"
        f"Audit: {result['status']}; stop: {value['stop_reason']}.\n"
        f"Final acceptance {value['final_accepted']}/12; forwards {value['forward_count']}/216; derivatives {value['derivative_count']}/96.\n"
        f"Updates {value['updates']}/{value['attempted_updates']}; native net {value['shared_net']}; conservative shared path {value['shared_path']}.\n\n"
        "Recorded fresh-gradient arithmetic and any issued endpoint's fixed first-passing certificate are independently reconstructed; exact-zero and ordinary no-step stops follow distinct rules. Technical or incomplete attempts are INCONCLUSIVE. Certificates concern the saved local surrogate and exact serialized shared-vector balls/path only; descriptive optimality bounds never gate admission. Real derivatives are not rerun. Neither surrogate decrease nor a no-step status establishes nonlinear progress, stationarity without exact proof, a causal feature, transfer, or gate efficacy. Actual twelve-row behavior and matching final replays remain separate. No warmstart, vector repair, or renormalization. STOP.\n"
    )


inherited.replay = replay
inherited.verify_update = verify_update
inherited.render_report = render_report
inherited.inherited.verify_update = verify_update
engine.replay, engine.verify_update = replay, verify_update
verify_plan, verify_data = inherited.verify_plan, inherited.verify_data
prepare_candidate_records = inherited.prepare_candidate_records
report = render_report


def verify():
    # Synchronize only this fresh isolated checker tree, supporting actual
    # temporary-namespace fixtures without changing predecessor modules.
    inherited.OUTPUT = inherited.inherited.OUTPUT = engine.OUTPUT = legacy.OUTPUT = OUTPUT
    inherited.ROOT = inherited.inherited.ROOT = engine.ROOT = legacy.ROOT = ROOT
    return inherited.verify()


def finalize_recording(budget, capture_receipt, runtime_status, *, audit=None):
    from scripts import certified_descent_comply_recording as recording

    return recording.finalize_recording(
        budget,
        capture_receipt,
        runtime_status,
        audit=verify if audit is None else audit,
        prepare_candidate_records=lambda value, digest: prepare_candidate_records(
            value, digest, budget.root
        ),
        render_report=render_report,
    )


def read_sealed_recording(output=None):
    from scripts import certified_descent_comply_recording as recording

    return recording.read_sealed_recording(
        OUTPUT if output is None else output, audit_match=AUDIT_MATCH
    )


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    value = read_sealed_recording()
    print(
        json.dumps(
            {
                k: v
                for k, v in value.items()
                if k not in {"summary", "optimizer_checks", "row_checks"}
            },
            allow_nan=False,
        )
    )
    if value["status"] == "INCONCLUSIVE":
        raise SystemExit(1)
