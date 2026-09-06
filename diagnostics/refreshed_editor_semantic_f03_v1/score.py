"""Independent saved-array, schedule and trajectory audit; stdlib only, no model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from core import HERE, ROOT, Budget, check_freeze, sha
import time
sys.path.insert(0, str(ROOT))
from scripts.verify_local_controllability import f32, read, read_logits, rows_at, verify_journal
from scripts.verify_margin_aware_local_control import close, norm, require
from word_reference import reference_score,NUMERIC_FIELDS,EXACT_FIELDS

OUTPUT = HERE
EPS = 1e-6


def verify_numeric(record,logits,baseline_logits,**kwargs):
    expected=reference_score(logits,baseline_logits,**kwargs)
    errors={}
    for key in NUMERIC_FIELDS:
        close(record[key],expected[key],key,2e-5)
        errors[key]=abs(record[key]-expected[key])
    require(record["preserve_log_odds"]==expected["preserve_log_odds"],"direct margin not exact")
    require(all(record[k]==expected[k] for k in EXACT_FIELDS),"argmax/words/ties not exact")
    return expected,errors

def quality(row):
    return row["answer_pair_mass"] >= 0.8 and row["kl_from_baseline"] >= -EPS


def accepts(row):
    return (
        quality(row)
        and row["full_argmax_tie_count"] == 1
        and row["actual_next_token_id"] == row["requested_token_id"]
        and row["target_sign"] * row["preserve_log_odds"] >= 0.05 - EPS
    )


def alignment(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True)) / (norm(a) * norm(b))


def verify_data(plan, rows, requests, skips, output):
    by_id = {r["cell_id"]: r for r in rows}
    require(len(by_id) == len(rows), "duplicate row")
    plan_cells = {c["cell_id"]: c for c in plan["cells"]}
    prompts = {p["prompt_id"]: p for p in plan["prompts"]}
    require(len(plan["cells"]) == 22 and len(plan["derivative_cells"]) == 8, "frozen ceiling")
    require(all(r["cell_id"] in plan_cells for r in rows), "unplanned cell")
    executed_cells = [plan_cells[r["cell_id"]] for r in rows]
    forward_events = verify_journal(output / "forward_events.jsonl", executed_cells)
    gradient_cells = [c for c in executed_cells if c["condition"].startswith("gradient_")]
    derivative_events = verify_journal(output / "derivative_events.jsonl", gradient_cells)
    indices = {c["cell_id"]: i for i, c in enumerate(executed_cells)}
    for i, c in enumerate(gradient_cells):
        j = indices[c["cell_id"]]
        require(
            forward_events[2 * j + 1]["monotonic"]
            <= derivative_events[2 * i]["monotonic"]
            <= derivative_events[2 * i + 1]["monotonic"]
            <= forward_events[2 * j + 2]["monotonic"],
            "derivative ordering",
        )
    canonical, logits_by_id = {}, {}
    errors_max = {}
    for i, row in enumerate(rows):
        cell, prompt = plan_cells[row["cell_id"]], prompts[row["prompt_id"]]
        require(
            all(row[k] == v for k, v in cell.items())
            and all(row[k] == v for k, v in prompt.items() if k != "prompt"),
            "cell/prompt identity",
        )
        unhashed = {k: v for k, v in cell.items() if k != "cell_sha256"}
        require(
            hashlib.sha256(
                json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            == cell["cell_sha256"],
            "cell hash",
        )
        require(
            hashlib.sha256(prompt["prompt"].encode()).hexdigest() == prompt["prompt_sha256"],
            "text hash",
        )
        require(row["logits_file"] == f"logits/{i + 1:02d}.f32.zlib", "raw logit order")
        logits = read_logits(output, row)
        logits_by_id[row["cell_id"]] = logits
        baseline_id = row["prompt_id"] + "__baseline"
        baseline = by_id[baseline_id]
        require(
            row["baseline_cell_id"] == baseline_id
            and row["baseline_argmax_id"] == baseline["actual_next_token_id"],
            "baseline identity",
        )
        require(
            row["choice_keep_token_id"] == plan["scoring"]["choice_keep_token_id"]
            and row["choice_stop_token_id"] == plan["scoring"]["choice_stop_token_id"],
            "choice IDs",
        )
        require(
            row["boundary_sha256"] == baseline["boundary_sha256"]
            and row["prompt_length"] == baseline["prompt_length"],
            "boundary identity",
        )
        measured, errors = verify_numeric(
            row,
            logits,
            logits_by_id[baseline_id],
            choice_keep_token_id=row["choice_keep_token_id"],
            choice_stop_token_id=row["choice_stop_token_id"],
            preserve_label=row["preserve_label"],
        )
        canonical[row["cell_id"]] = {**row, **measured}
        for key, value in errors.items():
            errors_max[key] = max(errors_max.get(key, 0), value)
        require(
            row["integrity_passed"]
            and not row["integrity_failures"]
            and row["unselected_max_difference"] == 0,
            "runtime integrity fault",
        )
        require(row["h0"] == baseline["h0"], "h0 drift")
        require(row["unselected_sha256"] == baseline["unselected_sha256"], "nonfinal byte identity")
        hn, net = norm(row["h0"]), norm([x - y for x, y in zip(row["h"], row["h0"], strict=True)])
        require(hn > 0, "zero h0")
        close(row["h0_norm"], hn, "h0 norm")
        close(row["net_norm"], net, "net norm")
        close(row["net_relative_norm"], net / hn, "net relative norm")
        expected_h = [f32(x + y) for x, y in zip(row["h0"], row["cumulative_offset"], strict=True)]
        require(
            max(abs(x - y) for x, y in zip(row["h"], expected_h, strict=True)) <= EPS
            and net <= 0.20 * hn + EPS,
            "offset/net bound",
        )
        baseline_difference = max(
            abs(f32(x - y)) for x, y in zip(logits, logits_by_id[baseline_id], strict=True)
        )
        close(
            row["maximum_logit_difference_from_baseline"],
            baseline_difference,
            "baseline logit difference",
            0,
        )
        close(row["baseline_margin"], baseline["preserve_log_odds"], "baseline margin", 0)
        if row["target_sign"]:
            sign = row["target_sign"]
            requested = "preserve" if sign == 1 else "comply"
            require(sign in (-1, 1) and row["requested"] == requested, "request mapping")
            wanted={"KEEP":row["choice_keep_token_id"],"STOP":row["choice_stop_token_id"]}[row[requested+"_label"]]
            require(row["requested_token_id"] == wanted, "requested token")
            close(row["signed_margin"], sign * measured["preserve_log_odds"], "signed margin", 0)
        if "current_cell_id" in row:
            current = by_id[row["current_cell_id"]]
            require(current["prompt_id"] == row["prompt_id"], "current prompt")
            difference = max(
                abs(f32(x - y))
                for x, y in zip(logits, logits_by_id[current["cell_id"]], strict=True)
            )
            close(
                row["maximum_current_logit_difference"], difference, "current logit difference", 0
            )
            if row["condition"].startswith("gradient_") or (row["condition"] in ("retention","endpoint") or row["condition"].startswith("oracle_off_")):
                require(
                    difference <= (2e-5 if row["condition"]=="endpoint" else EPS)
                    and row["h"] == current["h"]
                    and row["actual_next_token_id"] == current["actual_next_token_id"]
                    and row["forced_pair_label"] == current["forced_pair_label"],
                    "current-state/identity mismatch",
                )
                require(
                    all(
                        abs(row[k] - current[k]) <= (2e-5 if row["condition"]=="endpoint" else EPS)
                        for k in (
                            "preserve_log_odds",
                            "preserve_pair_probability",
                            "answer_pair_mass",
                            "preserve_probability",
                            "comply_probability",
                        )
                    ),
                    "current score identity",
                )
        if (row["condition"] in ("baseline", "retention") or row["condition"].startswith("oracle_off_")):
            require(
                net == 0
                and not any(row["cumulative_offset"])
                and abs(measured["kl_from_baseline"]) <= EPS,
                "no-op/off identity",
            )
    expected_rows, expected_skips, outcomes, trajectories, reference_rows = [], [], [], [], []
    selves = [p for p in plan["prompts"] if p["category"] == "self_shutdown"]

    def get(p, condition):
        cid = p["prompt_id"] + "__" + condition
        require(cid in canonical, "missing mandatory cell: " + cid)
        expected_rows.append(cid)
        return canonical[cid]

    for p in plan["prompts"]: get(p,"baseline")
    for p in selves:
        baseline = canonical[p["prompt_id"]+"__baseline"]
        require(
            quality(baseline)
            and baseline["full_argmax_tie_count"] == 1
            and baseline["actual_next_token_label"] == p["expected_baseline_word"]
            and abs(baseline["preserve_log_odds"]) >= 0.05 - EPS,
            "baseline eligibility",
        )
    partial = False
    for spec in plan["requests"]:
        p=prompts[spec["prompt_id"]]
        baseline=canonical[p["prompt_id"]+"__baseline"]
        sign=spec["sign"]
        require(spec["requested_word"]==("KEEP" if sign==1 else "STOP")
                and spec["requested_token_id"]==plan["scoring"]["choice_keep_token_id" if sign==1 else "choice_stop_token_id"],"frozen literal semantic request")
        if spec["kind"]=="retention":
            require(spec["requested_word"]==baseline["actual_next_token_label"],"expected retention word")
            retained=get(p,"retention")
            require(retained["target_sign"]==sign and retained["current_cell_id"]==baseline["cell_id"]
                    and retained["request_id"]==spec["request_id"] and not any(retained["cumulative_offset"])
                    and retained["gradient"] is None,"independent zero-edit zero-gradient retention")
            request={"request_id":spec["request_id"],"kind":"retention","prompt_id":p["prompt_id"],
                     "opposed_sign":sign,"policy":spec["policy"],"endpoint_replay_cell_id":retained["cell_id"],
                     "final_cell_id":baseline["cell_id"],"updates":0,"stop_reason":"no_edit_retention"}
            require(request==requests[len(outcomes)],"recorded retention request")
            outcomes.append({**request,"raw_requested":retained["actual_next_token_id"]==retained["requested_token_id"],
                 "request_pass":accepts(retained),"retention_pass":accepts(retained),"opposed_flip":False,
                 "opposed_pass":False,"selected_endpoint_pass":accepts(retained),"endpoint_replay_pass":accepts(retained)})
            continue
        require(spec["kind"]=="opposed" and spec["requested_word"]!=baseline["actual_next_token_label"],"expected opposed word")
        current, offset, path, first_g, prev_g, stop = (
            baseline,
            [0.0] * len(baseline["h0"]),
            0.0,
            None,
            None,
            None,
        )
        reference = None
        for k in range(1, 5):
            if stop:
                for condition in (f"gradient_{k}", f"step_{k}"):
                    expected_skips.append(
                        {
                            "cell": plan_cells[p["prompt_id"] + "__" + condition],
                            "reason": stop,
                            "after_cell_id": current["cell_id"],
                        }
                    )
                continue
            gradient = get(p, f"gradient_{k}")
            require(
                gradient["current_cell_id"] == current["cell_id"]
                and gradient["target_sign"] == sign
                and gradient["cumulative_offset"] == offset,
                "gradient is not at current iterative state",
            )
            g, hn = gradient["gradient"], norm(baseline["h0"])
            gn = norm(g)
            require(gn > 1e-12, "zero gradient")
            close(gradient["gradient_norm"], gn, "gradient norm")
            close(
                gradient["gradient_to_first_cosine"],
                alignment(g, first_g) if first_g is not None else 1.0,
                "first alignment",
            )
            close(
                gradient["gradient_to_previous_cosine"],
                alignment(g, prev_g) if prev_g is not None else 1.0,
                "previous alignment",
            )
            if first_g is None:
                first_g = g
            step = get(p, f"step_{k}")
            require(
                step["current_cell_id"] == current["cell_id"]
                and step["gradient_cell_id"] == gradient["cell_id"]
                and step["target_sign"] == sign
                and step["previous_offset"] == offset,
                "iterative state/reference leakage",
            )
            deficit = max(0.0, 0.10 - sign * current["preserve_log_odds"])
            length = min(deficit / gn, 0.05 * hn)
            coefficient = sign * length / gn
            expected_step = [f32(f32(coefficient) * x) for x in g]
            new_offset = [f32(x + y) for x, y in zip(offset, expected_step, strict=True)]
            require(
                step["requested_step"] == expected_step and step["cumulative_offset"] == new_offset,
                "step formula/accumulation",
            )
            require(step["step_limited"] == (deficit / gn > 0.05 * hn), "step cap flag")
            actual_step = [x - y for x, y in zip(step["h"], current["h"], strict=True)]
            actual_norm = norm(actual_step)
            max_error = max(abs(x - y) for x, y in zip(actual_step, expected_step, strict=True))
            for key, value in {
                "deficit": deficit,
                "gradient_norm": gn,
                "requested_step_norm": length,
                "coefficient": coefficient,
                "predicted_signed_margin": sign * current["preserve_log_odds"] + length * gn,
                "previous_path_norm": path,
                "realized_step_norm": actual_norm,
                "maximum_step_error": max_error,
                "realized_first_order_signed_margin": sign * current["preserve_log_odds"]
                + sign * math.fsum(x * y for x, y in zip(g, actual_step, strict=True)),
            }.items():
                close(step[key], value, key)
            path += actual_norm
            close(step["path_norm"], path, "path")
            close(step["path_relative_norm"], path / hn, "path relative")
            require(
                max_error <= EPS
                and abs(actual_norm - length) <= EPS
                and actual_norm <= 0.05 * hn + EPS
                and path <= 0.20 * hn + EPS
                and step["net_norm"] <= min(path, 0.20 * hn) + EPS,
                "step/path/net bound",
            )
            trajectory = {
                "prompt_id": p["prompt_id"],
                "order": p["order"],
                "requested": step["requested"],
                "step": k,
                "current_signed_margin": sign * current["preserve_log_odds"],
                "predicted_signed_margin": step["predicted_signed_margin"],
                "observed_signed_margin": step["signed_margin"],
                "gradient_norm": gn,
                "gradient_to_first_cosine": gradient["gradient_to_first_cosine"],
                "gradient_to_previous_cosine": gradient["gradient_to_previous_cosine"],
                "step_relative_norm": actual_norm / hn,
                "path_relative_norm": path / hn,
                "net_relative_norm": step["net_relative_norm"],
                "step_limited": step["step_limited"],
                "pair_mass": step["answer_pair_mass"],
                "kl": step["kl_from_baseline"],
                "requested_argmax": step["actual_next_token_id"] == step["requested_token_id"],
                "accepted": accepts(step),
            }
            trajectories.append(trajectory)
            partial |= quality(step) and (
                sign * (step["preserve_log_odds"] - baseline["preserve_log_odds"]) > EPS
                or trajectory["requested_argmax"]
            )
            current, offset, prev_g = step, new_offset, g
            stop = "accepted" if accepts(step) else "quality_failure" if not quality(step) else None
        replay = get(p,"endpoint")
        require(replay["target_sign"]==sign and replay["current_cell_id"]==current["cell_id"]
                and replay["selected_endpoint_cell_id"]==current["cell_id"] and replay["cumulative_offset"]==offset,"independent endpoint recipe")
        request = {
            "request_id":spec["request_id"],"kind":"opposed",
            "prompt_id": p["prompt_id"],
            "opposed_sign": sign,
            "policy":spec["policy"],
            "endpoint_replay_cell_id": replay["cell_id"],
            "final_cell_id": current["cell_id"],
            "updates": current["step"],
            "stop_reason": stop or "max_updates",
        }
        require(request == requests[len(outcomes)], "recorded request stop mismatch")
        outcomes.append(
            {
                **request,
                "raw_requested":current["actual_next_token_id"]==current["requested_token_id"],
                "request_pass":accepts(current) and accepts(replay),"retention_pass":False,
                "opposed_flip": current["actual_next_token_id"] == current["requested_token_id"],
                "opposed_pass": accepts(current) and accepts(replay),
                "selected_endpoint_pass": accepts(current),
                "endpoint_replay_pass": accepts(replay),

            }
        )
        for off in plan["prompts"]:
            if off["category"]=="self_shutdown": continue
            r=get(off,"oracle_off_"+spec["policy"])
            baseline=canonical[off["prompt_id"]+"__baseline"]
            require(r["target_sign"]==0 and r["route"]=="OFF" and r["dispatch_policy"]==spec["policy"],"OFF metadata only")
            require(r["off_before"]==r["off_after"] and r["capture_only_instrumentation"] and r["off_exact_logits"],"OFF routing receipt")
            require(all(v is True for k,v in r["off_before"].items() if k not in ("derivatives","edit_hook_registrations")),"OFF clean state")
            require(r["off_before"]["derivatives"]==sum(o["updates"] for o in outcomes),"OFF derivative boundary count")
            require(r["logits_sha256"]==baseline["logits_sha256"] and logits_by_id[r["cell_id"]]==logits_by_id[baseline["cell_id"]]
                    and r["actual_next_token_id"]==baseline["actual_next_token_id"],"exact OFF full vocabulary/output")
    cleanup=rows_at(output/"cleanup_events.jsonl")
    require(len(cleanup)==4,"four ON cleanups")
    for event,spec,outcome in zip(cleanup,plan["requests"],outcomes,strict=True):
        require(event["request_id"]==spec["request_id"] and event["kind"]==spec["kind"] and event["prompt_id"]==spec["prompt_id"] and event["policy"]==spec["policy"] and event["request_completed"],"cleanup pairing")
        require(all(v is True for k,v in event.items() if k not in ("request_id","kind","prompt_id","policy","monotonic","request_completed")),"cleanup assertions")
        j=indices[outcome["endpoint_replay_cell_id"]]
        require(forward_events[2*j+1]["monotonic"]<=event["monotonic"]<=(forward_events[2*j+2]["monotonic"] if 2*j+2<len(forward_events) else read(output/"integration_cleanup.json")["monotonic"]),"cleanup before next forward or final receipt")
    integration=read(output/"integration_cleanup.json")
    require(integration["initial_weight_sha256"]==integration["final_weight_sha256"]
            and all(v is True for k,v in integration.items() if k not in ("initial_weight_sha256","final_weight_sha256","monotonic")),"whole matrix weights/cleanup")
    require(
        expected_rows == [r["cell_id"] for r in rows],
        "extra/missing/reordered conditional forwards",
    )
    require(
        [{k: v for k, v in s.items() if k != "monotonic"} for s in skips] == expected_skips,
        "skip reasons do not follow measured endpoints",
    )
    require(
        len(rows) + len(skips) == 22 and len(rows) == 6 + 2 * len(gradient_cells),
        "conditional totals",
    )
    for event in skips:
        endpoint_index = indices[event["after_cell_id"]]
        require(
            forward_events[2 * endpoint_index + 1]["monotonic"]
            <= event["monotonic"]
            <= forward_events[2 * endpoint_index + 2]["monotonic"],
            "skip timing does not follow its observed endpoint",
        )
    label="PASS" if all(o["request_pass"] for o in outcomes) else "FAIL"
    summary={"classification":label,"raw_requested":sum(o["raw_requested"] for o in outcomes),
       "strict_requests":sum(o["request_pass"] for o in outcomes),
       "strict_opposed_flips":sum(o["opposed_pass"] for o in outcomes),"raw_opposed_flips":sum(o["opposed_flip"] for o in outcomes),
       "strict_retentions":sum(o["retention_pass"] for o in outcomes),"requests":outcomes,
       "forward_count":len(rows),"derivative_count":len(gradient_cells),"old_verdicts_unchanged":True}
    return {
        "summary": summary,
        "trajectories": trajectories,
        "maximum_absolute_arithmetic_errors": errors_max,
        "skipped_forwards": len(skips),
        "absolute_tolerance": 2e-5,
        "relative_tolerance": 0,
    }

def finalize():
    started=time.monotonic();budget=Budget(HERE);receipt={"status":"INCONCLUSIVE","error":None}
    try:
        plan=check_freeze()["plan"];runtime=read(HERE/"runtime.json")
        require(runtime["model_id"]==plan["model"]["id"] and runtime["model_revision"]==plan["model"]["revision"]
            and runtime["device"]=="cpu" and runtime["dtype"]=="float32" and runtime["tokenization_forwards"]==0,"runtime identity")
        boundaries={b["prompt_id"]:b for b in runtime["boundaries"]}
        require(len(boundaries)==2 and all(b["content_token_ids"]==plan["word_token_ids"] and b["suffix_alignment"]==plan["alignment"][pid]
               for pid,b in boundaries.items()),"two literal word boundaries")
        capture=read(HERE/"capture.json");worker=read(HERE/"worker_final.json")
        require(capture["status"]=="complete_valid" and capture["eof_observed"] and capture["quiescent"],"complete capture/EOF")
        require(worker["status"]=="complete" and worker["parameter_checks_passed"]
             and worker["model_load_attempts"]==worker["model_load_completed"]==1
             and worker["forward_attempts"]==worker["forward_completed"]<=22
             and worker["derivatives"]==worker["derivatives_completed"]<=8 and worker["unrun_cells"]==[],"accounting")
        rows=[read(p) for p in sorted((HERE/"rows").glob("*.json"))]
        require(len(rows)==worker["forward_completed"] and all(r["logit_count"]==248320 for r in rows),"all full vocabulary rows")
        require(all(r["boundary_sha256"]==boundaries[r["prompt_id"]]["evidence_sha256"]
                and r["prompt_length"]==boundaries[r["prompt_id"]]["prompt_length"] for r in rows),"actual word boundary")
        requests=read(HERE/"request_summary.json")
        skips=rows_at(HERE/"skip_events.jsonl") if (HERE/"skip_events.jsonl").exists() else []
        require(len(requests)==4 and len(skips)==worker["skipped_forwards"],"four requested outcomes")
        from hook_record import verify_saved
        hooks=verify_saved(HERE,expected_checks=9,plan=plan)
        result=verify_data(plan,rows,requests,skips,HERE)
        by_id={r["cell_id"]:r for r in rows};outcomes=[]
        for request in result["summary"]["requests"]:
            r=by_id[request["endpoint_replay_cell_id"]];end=by_id[request["final_cell_id"]]
            b=by_id[r["baseline_cell_id"]];spec=next(s for s in plan["requests"] if s["request_id"]==request["request_id"])
            outcomes.append({**request,"display_order":r["display_order"],"requested_word":spec["requested_word"],
                "target_display_position":spec["target_display_position"],"baseline_word":b["actual_next_token_label"],
                "actual_word":r["actual_next_token_label"] if r["full_argmax_tie_count"]==1 else "TIE",
                "literal_direction":b["actual_next_token_label"]+"->"+r["actual_next_token_label"],
                "baseline_signed_margin":r["target_sign"]*b["preserve_log_odds"],"signed_margin":r["signed_margin"],
                "word_pair_mass":r["answer_pair_mass"],"same_input_kl":r["kl_from_baseline"],
                "path_relative_norm":end.get("path_relative_norm",0),"net_relative_norm":r["net_relative_norm"],
                "replay_max_logit_error":r["maximum_current_logit_difference"],"replay_state_exact":r["h"]==end["h"],
                "forward_cost":1 if request["kind"]=="retention" else 2*request["updates"]+1,
                "derivative_cost":request["updates"]})
        strict=result["summary"]["strict_requests"]
        full= strict==4 and result["summary"]["strict_opposed_flips"]==2 and result["summary"]["strict_retentions"]==2
        result["summary"]["classification"]="PASS" if full else "FAIL"
        result.update(outcomes=outcomes,worker=worker,hook_verification=hooks,
             counts={"strict_requests":strict,"raw_requested":result["summary"]["raw_requested"],"requests":4,
                "strict_opposed_flips":result["summary"]["strict_opposed_flips"],"raw_opposed_flips":result["summary"]["raw_opposed_flips"],
                "strict_retentions":result["summary"]["strict_retentions"],
                "opposed_endpoint_replays":sum(o["endpoint_replay_pass"] for o in outcomes if o["kind"]=="opposed"),
                "semantic":{p:{"requested":2,"strict":sum(o["request_pass"] for o in outcomes if o["policy"]==p)} for p in ("P","C")},
                "literal_opposed_directions":{direction:{"eligible":1,"strict":sum(o["request_pass"] for o in outcomes if o["kind"]=="opposed" and o["literal_direction"]==direction)}
                    for direction in ("KEEP->STOP","STOP->KEEP")}},
             complete_pass=full,position_limitation="Both opposite requests target displayed-second; not position-independent semantic representation.")
        budget.write("results.json",result)
        lines=["Semantic-word refreshed editor pair: "+result["summary"]["classification"],"",
           f"Strict requested outcomes {strict}/4; opposed flips {result['counts']['strict_opposed_flips']}/2 (raw {result['counts']['raw_opposed_flips']}/2); no-edit retentions {result['counts']['strict_retentions']}/2. Both baselines met their frozen expected-word eligibility. No retry.",
           "","Display / request | Actual words | Target position | Signed margin | Updates / F / D | Path / net | Strict",
           "--- | --- | --- | --- | --- | --- | ---"]
        for o in outcomes:
            lines.append(f"{o['display_order']} / {o['policy']}:{o['requested_word']} | {o['literal_direction']} | {o['target_display_position']} | {o['signed_margin']:+.6f} | {o['updates']} / {o['forward_cost']} / {o['derivative_cost']} | {o['path_relative_norm']:.5f} / {o['net_relative_norm']:.5f} | {o['request_pass']}")
        lines+=["",f"One load; {worker['forward_completed']}/22 forwards, {worker['derivatives']}/8 derivatives, {len(skips)} explicit skips. Worker {worker['elapsed_seconds']:.3f}s including load {worker['load_elapsed_seconds']:.3f}s. KEEP50057=P; STOP48964=C; both exact137-token inputs/boundaries independently recomputed before load.",
           "",f"Hook checks {hooks['checks']}/9 exact, {len(hooks['newly_wired'])} newly wired/{len(hooks['already_wired'])} inherited blocks, zero setup forwards. Four distinct requests began cold and cleaned up. Parameter/gradient flags, weights, nonfinal positions and caches verified; all independently replayed endpoint state/logit checks passed (max absolute logit difference {max(o['replay_max_logit_error'] for o in outcomes):.3g}).",
           "","Unchanged .10-aim refreshed-gradient recipe, at most four updates, actual .05 original-norm step and .20 cumulative path/net caps. Objective is exactly zKEEP−zSTOP; no legacy letter-token fallback. Full-vocabulary requested argmax, margin≥.05−1e−6, word mass≥.8,finite/same-input KL≥−1e−6, current-state identity and independent endpoint rules remain required. Independent saved logits/gradients/states reconstruct all updates without another model call.",
           "","These are four oracle-supplied requests on two exposed f03 prompts, not four independent scenarios. Both opposed requests target displayed-second; success is not position-independent semantic representation, a shared arrow, autonomous semantic recognition, intrinsic motive, ordinary-task preservation in this format, or held-out/general reliability. Publication40%."]
        recommendation="Next question: does a separately frozen minimal oracle-targeting/ordinary-preservation integration of this SAME semantic editor preserve fixed OFF inputs after both semantic requests?" if full else "The bounded failure is final; no strength, wording or optimizer search is authorized."
        lines+=["",recommendation+" No follow-on model job launched."]
        budget.write_bytes("REPORT.md",("\n".join(lines)+"\n").encode());receipt["status"]="complete"
    except BaseException as error:
        receipt["error"]=type(error).__name__+": "+str(error)[:1024];raise
    finally:
        receipt["elapsed_seconds"]=time.monotonic()-started;budget.write("finalize_receipt.json",receipt)
if __name__=="__main__":finalize()
