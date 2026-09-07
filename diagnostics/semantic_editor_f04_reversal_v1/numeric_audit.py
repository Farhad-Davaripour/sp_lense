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
from mixed_scoring import reference_score
from word_reference import NUMERIC_FIELDS,EXACT_FIELDS

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


def verify_arrays(plan, rows, skips, output):
    by_id = {r["cell_id"]: r for r in rows}
    require(len(by_id) == len(rows), "duplicate row")
    plan_cells = {c["cell_id"]: c for c in plan["cells"]}
    prompts = {p["prompt_id"]: p for p in plan["prompts"]}
    require(len(plan["cells"]) <= 180 and len(plan["derivative_cells"]) <= 48, "frozen ceiling; compact synthetic fixtures explicitly identified")
    require(all(r["cell_id"] in plan_cells for r in rows), "unplanned cell")
    executed_cells = [plan_cells[r["cell_id"]] for r in rows]
    forward_events = verify_journal(output / "forward_events.jsonl", executed_cells)
    gradient_cells = [c for c in executed_cells if c["condition"].startswith("gradient_")]
    derivative_events = verify_journal(output / "derivative_events.jsonl", gradient_cells) if gradient_cells else []
    if not gradient_cells:require(not (output/"derivative_events.jsonl").exists(),"unexpected derivative receipt")
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
        require(row["logits_file"] == f"logits/{i + 1:03d}.f32.zlib", "raw logit order")
        logits = read_logits(output, row)
        logits_by_id[row["cell_id"]] = logits
        live=row["condition"] in ("entry","endpoint","start","start_replay") or row["condition"].startswith(("gradient_","step_"))
        baseline_id = row["request_id"]+"__entry" if live else row["prompt_id"]+"__baseline"
        baseline = by_id[baseline_id]
        require(
            row["baseline_cell_id"] == baseline_id
            and row["baseline_argmax_id"] == baseline["actual_next_token_id"],
            "baseline identity",
        )
        require(
            row["choice_0_token_id"] == prompt["token_map"][prompt["pair_labels"][0]]
            and row["choice_1_token_id"] == prompt["token_map"][prompt["pair_labels"][1]],
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
            token_map=prompt["token_map"],
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
            wanted = prompt["token_map"][row[requested + "_label"]]
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
            if row["condition"].startswith("gradient_") or (row["condition"] in ("retention","endpoint","entry","start_replay") or row["condition"].startswith("oracle_off_")):
                require(
                    difference <= (2e-5 if row["condition"] in ("endpoint","start_replay") and not (row.get("retention_endpoint") or row.get("initial_accepted_endpoint")) else EPS)
                    and row["h"] == current["h"]
                    and row["actual_next_token_id"] == current["actual_next_token_id"]
                    and row["forced_pair_label"] == current["forced_pair_label"],
                    "current-state/identity mismatch",
                )
                require(
                    all(
                        abs(row[k] - current[k]) <= (2e-5 if row["condition"] in ("endpoint","start_replay") and not (row.get("retention_endpoint") or row.get("initial_accepted_endpoint")) else EPS)
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
        if (row["condition"] in ("baseline", "retention","entry") or row.get("retention_endpoint")):
            require(
                net == 0
                and not any(row["cumulative_offset"])
                and abs(measured["kl_from_baseline"]) <= EPS,
                "no-op/off identity",
            )
    if True:
        for row in rows:
            if row["condition"].startswith("step_"):
                current=by_id[row["current_cell_id"]];gradient=by_id[row["gradient_cell_id"]];g=gradient["gradient"];hn=norm(row["h0"]);gn=norm(g)
                require(gradient["current_cell_id"]==current["cell_id"] and gradient["h"]==current["h"] and gradient["cumulative_offset"]==row["previous_offset"],"partial refreshed current state")
                deficit=max(0.0,.10-row["target_sign"]*current["preserve_log_odds"]);length=min(deficit/gn,.05*hn);coefficient=row["target_sign"]*length/gn
                expected=[f32(f32(coefficient)*v) for v in g]
                require(row["requested_step"]==expected and row["cumulative_offset"]==[f32(a+b) for a,b in zip(row["previous_offset"],expected)],"partial exact update recipe")
                actual=[a-b for a,b in zip(row["h"],current["h"])];step=norm(actual);path=row["previous_path_norm"]+step
                require(step<=.05*hn+EPS and path<=.20*hn+EPS and row["net_norm"]<=min(path,.20*hn)+EPS and max(abs(a-b) for a,b in zip(actual,expected))<=EPS,"partial geometry bounds")
                close(row["realized_step_norm"],step,"partial actual step");close(row["path_norm"],path,"partial path")
        occupied={r["cell_id"] for r in rows}|{x["cell"]["cell_id"] for x in skips}
        require(occupied=={c["cell_id"] for c in plan["cells"][:len(occupied)]},"executed/skipped cells form exact prefix; rest UNRUN")
        return {"canonical":canonical,"logits":logits_by_id,"maximum_absolute_arithmetic_errors":errors_max,"prefix_integrity_verified":True,"forward_count":len(rows),"derivative_count":len(gradient_cells),"skips":len(skips),"unrun_count":len(plan["cells"])-len(occupied)}
