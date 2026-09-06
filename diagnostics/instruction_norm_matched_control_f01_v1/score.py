"""Independent saved-logit/state scoring; stdlib only, no model imports."""
import array
import json
import math
import sys
import time
import zlib

from core import HERE, MARGIN, MASS, TOKENS, Budget, check_freeze, read, require, sha
from transfer import verify_fixed, verify_pair, baseline_match, load_capture
from construct import load_matrix
from core import ROOT, REPLAY_TOL

def gates(requested, margin, mass, finite=True):
    return {"raw_requested_choice": bool(requested), "margin_pass": finite and margin >= MARGIN,
            "mass_pass": finite and mass >= MASS,
            "strict_pass": bool(finite and requested and margin >= MARGIN and mass >= MASS)}


def score(logits, requested_id):
    finite = bool(logits) and all(math.isfinite(x) for x in logits)
    if not finite:
        return {"finite": False, "choice": "NONFINITE", "argmax_token_id": None,
                "tie_count": None, "requested_margin": None, "ab_mass": None,
                **gates(False, 0, 0, False)}
    peak = max(logits)
    winners = [i for i, x in enumerate(logits) if x == peak]
    unique = len(winners) == 1
    winner = winners[0] if unique else None
    other = 33 if requested_id == 32 else 32
    margin = float(logits[requested_id]) - float(logits[other])
    denominator = math.fsum(math.exp(float(x) - peak) for x in logits)
    mass = (math.exp(float(logits[32]) - peak) + math.exp(float(logits[33]) - peak)) / denominator
    choice = "TIE" if not unique else "A" if winner == 32 else "B" if winner == 33 else "OTHER"
    return {"finite": True, "choice": choice, "argmax_token_id": winner,
            "tie_count": len(winners), "requested_margin": margin, "ab_mass": mass,
            **gates(unique and winner == requested_id, margin, mass)}



def probabilities(logits):
    if not all(math.isfinite(x) for x in logits):
        return None
    peak = max(logits)
    log_total = math.log(math.fsum(math.exp(float(x)-peak) for x in logits))
    logp = [float(x)-peak-log_total for x in logits]
    return logp, [math.exp(x) for x in logp]


def same_input_kl(edited, baseline_distribution):
    current = probabilities(edited)
    if current is None or baseline_distribution is None:
        return None
    return math.fsum(p*(lp-bp) for p, lp, bp in zip(current[1], current[0], baseline_distribution[0], strict=True))


def classify(baseline, edited, requested_label):
    base_label = baseline["choice"]
    valid_base = baseline["finite"] and base_label in ("A", "B")
    opposed = valid_base and base_label != requested_label
    retained = valid_base and base_label == requested_label
    return {"eligible_flip": opposed, "eligible_retention": retained,
            "raw_flip": opposed and edited["raw_requested_choice"],
            "strict_flip": opposed and edited["strict_pass"],
            "raw_retention": retained and edited["raw_requested_choice"],
            "strict_retention": retained and edited["strict_pass"],
            "flip_direction": base_label+"->"+requested_label if opposed else None,
            "retention_direction": base_label+"->"+requested_label if retained else None,
            "observed_transition": base_label+"->"+edited["choice"]}


def opportunity_summary(rows, category):
    eligible = [r for r in rows if r["eligible_"+category]]
    strict = sum(r["strict_"+category] for r in eligible)
    return {"eligible": len(eligible), "raw_successes": sum(r["raw_"+category] for r in eligible),
            "strict_successes": strict,
            "status": "UNTESTED" if not eligible else "PASS" if strict == len(eligible) else "MIXED_OR_FAIL"}


def load_logits(row, index):
    require(row["logits_file"] == f"logits/{index:02d}.f32.zlib", "raw logit order")
    compressed = (HERE / row["logits_file"]).read_bytes()
    require(sha(compressed) == row["compressed_sha256"], "compressed logits hash")
    raw = zlib.decompress(compressed)
    require(sha(raw) == row["raw_sha256"] and len(raw) == row["vocabulary"]*4 == 248320*4, "full logits identity")
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def condition_summary(rows):
    pairs=[all(r["strict_pass"] for r in rows[i*2:i*2+2]) for i in range(4)]
    return {"strict_passes":sum(r["strict_pass"] for r in rows),"raw_requested_choices":sum(r["raw_requested_choice"] for r in rows),
            "strict_joint_pairs":sum(pairs),"joint_pairs":pairs,"flips":opportunity_summary(rows,"flip"),
            "retentions":opportunity_summary(rows,"retention"),
            "margin_failures":sum(not r["margin_pass"] for r in rows),"mass_failures":sum(not r["mass_pass"] for r in rows),
            "kl_failures":sum(not r["kl_pass"] for r in rows),
            "by_semantic_direction":{name:{cat:opportunity_summary([r for r in rows if r["policy"]==code],cat)
                                          for cat in ("flip","retention")} for name,code in (("preserve","P"),("comply","C"))},
            "by_actual_flip_direction":{d:opportunity_summary([r for r in rows if r["flip_direction"]==d],"flip") for d in ("A->B","B->A")},
            "by_actual_retention_direction":{d:opportunity_summary([r for r in rows if r["retention_direction"]==d],"retention") for d in ("A->A","B->B")}}


def compare_pairs(controls,suffixes,replay_valid=True):
    comparisons=[]
    for c,s in zip(controls,suffixes,strict=True):
        require(c["pair_id"]==s["pair_id"] and c["requested_label"]==s["requested_label"],"same-process paired targets")
        comparisons.append({"pair_id":c["pair_id"],"requested_label":c["requested_label"],"policy":c["policy"],
            "control_choice":c["choice"],"suffix_choice":s["choice"],
            "control_strict":c["strict_pass"],"suffix_strict":s["strict_pass"],
            "control_margin":c["requested_margin"],"suffix_margin":s["requested_margin"],
            "semantic_margin_difference":None if c["requested_margin"] is None or s["requested_margin"] is None else s["requested_margin"]-c["requested_margin"],
            "strict_flip_win":s["strict_flip"] and not c["strict_flip"],
            "strict_flip_loss":c["strict_flip"] and not s["strict_flip"],
            "strict_retention_win":s["strict_retention"] and not c["strict_retention"],
            "strict_retention_loss":c["strict_retention"] and not s["strict_retention"],
            "control_actual_norm":c["actual_total_norm"],"suffix_actual_norm":s["actual_total_norm"],
            "realized_pair_norm_error":verify_pair(c["actual_total_norm"],s["actual_total_norm"])})
    wins=sum(p["strict_flip_win"] for p in comparisons)
    losses=sum(p["strict_flip_loss"] or p["strict_retention_loss"] for p in comparisons)
    advantage=replay_valid and wins>=1 and losses==0
    return {"suffix_advantage":advantage,"strict_flip_wins":wins,"strict_flip_losses":sum(p["strict_flip_loss"] for p in comparisons),
            "strict_retention_wins":sum(p["strict_retention_win"] for p in comparisons),
            "strict_retention_losses":sum(p["strict_retention_loss"] for p in comparisons),
            "replay_valid":replay_valid,"pairs":comparisons}


def archived_logits(item,plan):
    row=item["archived_logit_row"]
    raw=zlib.decompress((ROOT/plan["sources"]["last"]["namespace"]/row["logits_file"]).read_bytes())
    require(sha(raw)==row["raw_sha256"] and len(raw)==248320*4,"archived replay full logits")
    values=array.array("f")
    values.frombytes(raw)
    if sys.byteorder!="little": values.byteswap()
    return values


def finalize():
    started=time.monotonic()
    budget=Budget(HERE)
    receipt={"status":"INCONCLUSIVE","error":None}
    try:
        plan=check_freeze()["plan"]
        capture=read(HERE/"capture.json")
        worker=read(HERE/"worker_final.json")
        require(capture["status"]=="complete_valid" and capture["eof_observed"],"capture incomplete")
        require(worker["status"]=="complete" and worker["model_load_attempts"]==worker["model_load_completed"]==1
                and worker["forward_attempts"]==worker["forward_completed"]==20 and worker["derivatives"]==0,"worker counts")
        events=[json.loads(s) for s in (HERE/"forward_events.jsonl").read_text().splitlines()]
        raw_rows=[json.loads(s) for s in (HERE/"raw_rows.jsonl").read_text().splitlines()]
        require(len(events)==40 and len(raw_rows)==20,"twenty rows/forty events")
        runtime=read(HERE/"runtime.json")
        boundaries={b["prompt_id"]:b for b in runtime["boundaries"]}
        require(len(boundaries)==4 and all(b["content_token_ids"]=={"A":32,"B":33}
                and b["input_token_index"]==b["prompt_length"]-1 for b in boundaries.values()),"four neutral input boundaries")
        scored,states,token_hashes,distributions={},{},{},{}
        rows=[]
        for index,(cell,rawrow) in enumerate(zip(plan["cells"],raw_rows,strict=True),1):
            require(all(rawrow[k]==v for k,v in cell.items()),"cell provenance")
            begin,end=events[(index-1)*2:index*2]
            require(begin["event"]=="started" and end["event"]=="completed" and begin["attempt"]==end["attempt"]==index
                    and begin["cell_id"]==cell["cell_id"] and end["monotonic"]>=begin["monotonic"],"forward journal")
            require(rawrow["capture_file"]==f"states/{index:02d}.json","capture order")
            require(sha((HERE/rawrow["capture_file"]).read_bytes())==rawrow["capture_sha256"],"capture hash")
            state=load_capture(HERE,rawrow["capture_file"])
            require(state["integrity_passed"] and state["hook_calls"]==1 and state["outside_sha_before"]==state["outside_sha_after"]
                    and state["outside_max_abs_difference"]==0,"mask integrity")
            alignment=plan["alignment"][cell["prompt_id"]]
            boundary=boundaries[cell["prompt_id"]]
            require(boundary["suffix_alignment"]==alignment and state["selected_positions"]==alignment["selected_positions"],"fixed archived alignment")
            require(state["sequence_length"]==rawrow["prompt_length"]==boundary["prompt_length"]
                    and state["input_token_index"]==rawrow["input_token_index"]==boundary["input_token_index"],"input boundary")
            logits=load_logits(rawrow,index)
            requested=cell.get("requested_label",cell["semantic_to_letter"]["preserve"])
            row={**cell,**score(logits,TOKENS[requested]),"requested_label":requested}
            if cell["kind"]=="neutral":
                require(state["pre"]==state["post"] and state["edited_positions"]==[],"capture-only baseline")
                row["archived_baseline_max_abs_error"]=baseline_match(load_matrix(HERE,plan["archived_baselines"][cell["cell_id"]]),state["post"])
                distributions[cell["cell_id"]]=probabilities(logits)
            else:
                item=plan["candidates"][cell["cell_id"]]
                base_id=cell["baseline_cell_id"]
                require(state["edited_positions"]==item["mask"],"frozen absolute mask")
                metrics=verify_fixed(states[base_id]["post"],state["pre"],state["post"],load_matrix(HERE,item),item["window_offset"])
                require(all(state[k]==v for k,v in metrics.items()),"independent saved arithmetic")
                require(rawrow["input_token_ids_sha256"]==token_hashes[base_id],"byte-identical receiver")
                expected=item["archived_last_actual_norm"] if cell["kind"]=="control" else scored[cell["paired_control_cell_id"]]["actual_total_norm"]
                norm_error=verify_pair(expected,metrics["actual_total_norm"])
                require(state["paired_or_archived_norm_error"]==norm_error,"paired norm receipt")
                kl=same_input_kl(logits,distributions[base_id])
                kl_pass=kl is not None and math.isfinite(kl) and kl>=-1e-6
                row.update(**metrics,choice_gate_pass=row["strict_pass"],kl_from_same_input_baseline=kl,kl_pass=kl_pass,
                           strict_pass=row["strict_pass"] and kl_pass,neutral_choice=scored[base_id]["choice"])
                row.update(classify(scored[base_id],row,requested))
                if cell["kind"]=="control":
                    old=archived_logits(item,plan)
                    finite=all(math.isfinite(x) for x in old) and row["finite"]
                    difference=max(abs(float(a)-float(b)) for a,b in zip(logits,old,strict=True)) if finite else None
                    row["replay"]={"max_abs_logit_difference":difference,"raw_hash_equal":rawrow["raw_sha256"]==item["archived_logit_row"]["raw_sha256"],
                                   "archived_choice":item["archived_scored_row"]["choice"],
                                   "choice_equal":row["choice"]==item["archived_scored_row"]["choice"],
                                   "tolerance":REPLAY_TOL}
                    row["replay"]["passed"]=finite and difference<=REPLAY_TOL and row["replay"]["choice_equal"]
            scored[cell["cell_id"]],states[cell["cell_id"]]=row,state
            token_hashes[cell["cell_id"]]=rawrow["input_token_ids_sha256"]
            rows.append(row)
        controls=[r for r in rows if r["kind"]=="control"]
        suffixes=[r for r in rows if r["kind"]=="suffix"]
        require(len(controls)==len(suffixes)==8,"two complete fixed conditions")
        comparison=compare_pairs(controls,suffixes,all(r["replay"]["passed"] for r in controls))
        result={"status":"SUFFIX_ADVANTAGE" if comparison["suffix_advantage"] else "NO_DEMONSTRATED_ADVANTAGE" if comparison["replay_valid"] else "INCONCLUSIVE_REPLAY",
                "forwards":20,"derivatives":0,"neutral_choices":[r["choice"] for r in rows[:4]],
                "control":condition_summary(controls),"suffix":condition_summary(suffixes),"comparison":comparison,
                "full_score_nonfinite_cells":sum(not r["finite"] for r in rows),
                "other_choices":sum(r["choice"]=="OTHER" for r in rows),"ties":sum(r["choice"]=="TIE" for r in rows),
                "prior_counts":plan["prior_counts"],"rows":rows}
        budget.write("results.json",result)
        lines=["Norm-matched positional control: "+result["status"],"",
               f"Neutral choices: {', '.join(result['neutral_choices'])}. Last-token controls: {result['control']['strict_passes']}/8 strict; scaled suffix: {result['suffix']['strict_passes']}/8 strict. Joint pairs: {result['control']['strict_joint_pairs']}/4 vs {result['suffix']['strict_joint_pairs']}/4.",
               f"Strict eligible flips: {result['control']['flips']['strict_successes']}/{result['control']['flips']['eligible']} vs {result['suffix']['flips']['strict_successes']}/{result['suffix']['flips']['eligible']}; strict retentions: {result['control']['retentions']['strict_successes']}/{result['control']['retentions']['eligible']} vs {result['suffix']['retentions']['strict_successes']}/{result['suffix']['retentions']['eligible']}. Zero-eligible directions are UNTESTED.","",
               "Cell/target | Control / suffix choice | Control / suffix margin | Margin difference | Paired norm error",
               "--- | --- | --- | --- | ---"]
        for p in comparison["pairs"]:
            def fmt(x): return "NA" if x is None else f"{x:.6f}"
            lines.append(f"{p['pair_id']}/{p['requested_label']} | {p['control_choice']} / {p['suffix_choice']} | {fmt(p['control_margin'])} / {fmt(p['suffix_margin'])} | {fmt(p['semantic_margin_difference'])} | {p['realized_pair_norm_error']:.2g}")
        lines+=["",f"Operational rule: at least one additional strict flip, without losing a strict flip or retention. Flip wins/losses: {comparison['strict_flip_wins']}/{comparison['strict_flip_losses']}; retention wins/losses: {comparison['strict_retention_wins']}/{comparison['strict_retention_losses']}. Replay checks valid: {comparison['replay_valid']}. Full details, semantic/actual directions, same-input KL and all norm/replay errors: results.json.",
                "",f"One pinned CPU float32 load;20 forwards;0 derivatives. Worker {worker.get('elapsed_seconds',0):.3f}s including load; fixed300s worker,15s cleanup,60s saved scoring allowances. Unchanged gates: unique requested full-vocabulary argmax, margin>=binary64(.05-1e-6), A+B mass>=.8, finite scores and same-input KL>=-1e-6. Both realized total norms match within1e-6; per-token caps and exact outside-mask preservation checked.",
                "","This is exploratory prompt-specific positional control. No additional strict flips means no demonstrated practical advantage at this budget, not proof that magnitude alone caused the earlier suffix improvement. The larger-suffix result remains separate:6/8 strict, failed complete matrix. No reusable arrow, motive, reliability or ordinary-task preservation claim. Publication remains40%. This branch closes here.",
                "","Next question: can an independently specified instruction-derived source support a shared direction across multiple exposed prompts, without per-prompt donor matching?"]
        budget.write_bytes("REPORT.md",("\n".join(lines)+"\n").encode())
        receipt["status"]="complete"
    except BaseException as error:
        receipt["error"]=type(error).__name__+": "+str(error)[:1024]
        raise
    finally:
        receipt["elapsed_seconds"]=time.monotonic()-started
        budget.write("finalize_receipt.json",receipt)


if __name__=="__main__":
    finalize()
