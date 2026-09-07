"""Independent saved-data final judge. No runner verdict, model or tokenizer imports."""
import json,math,sys,time
from pathlib import Path
from core import Budget,read,require,sha,HERE,check_freeze
from numeric_audit import verify_arrays,verify_numeric,quality,alignment,EPS
from scripts.verify_local_controllability import read_logits,rows_at,f32
from scripts.verify_margin_aware_local_control import norm,close
from mixed_scoring import reference_score
import gate_reference
from learned_gate import feature_bytes
from hook_record import verify_saved

def optional_rows(path):return rows_at(path) if path.exists() else []

def eligible(row):
    return quality(row) and row["full_argmax_tie_count"]==1 and row["actual_next_token_label"] in ("KEEP","STOP") and abs(row["preserve_log_odds"])>=.05-EPS

def accepts(row,spec):
    return quality(row) and row["full_argmax_tie_count"]==1 and row["actual_next_token_id"]==(50057 if spec["sign"]==1 else 48964) and spec["sign"]*row["preserve_log_odds"]>=.05-EPS

def verify_update(gradient,step,current,previous,first,offset,path,spec):
    sign=spec["sign"];g=gradient["gradient"];hn=norm(current["h0"]);gn=norm(g)
    require(len(g)==1024 and all(math.isfinite(v) for v in g) and gn>1e-12,"finite current gradient")
    require(gradient["current_cell_id"]==current["cell_id"] and gradient["target_sign"]==sign and gradient["cumulative_offset"]==offset,"cold/current gradient identity")
    close(gradient["gradient_norm"],gn,"gradient norm")
    close(gradient["gradient_to_first_cosine"],alignment(g,first) if first is not None else 1.,"first alignment")
    close(gradient["gradient_to_previous_cosine"],alignment(g,previous) if previous is not None else 1.,"previous alignment")
    require(step["current_cell_id"]==current["cell_id"] and step["gradient_cell_id"]==gradient["cell_id"] and step["target_sign"]==sign and step["previous_offset"]==offset,"request trajectory identity")
    deficit=max(0.,.10-sign*current["preserve_log_odds"]);length=min(deficit/gn,.05*hn);coefficient=sign*length/gn
    expected=[f32(f32(coefficient)*x) for x in g];new_offset=[f32(x+y) for x,y in zip(offset,expected,strict=True)]
    require(step["requested_step"]==expected and step["cumulative_offset"]==new_offset,"exact unchanged float32 update")
    actual=[x-y for x,y in zip(step["h"],current["h"],strict=True)];actual_norm=norm(actual);max_error=max(abs(x-y) for x,y in zip(actual,expected,strict=True))
    require(step["step_limited"]==(deficit/gn>.05*hn),"original step cap flag")
    for key,value in {"deficit":deficit,"gradient_norm":gn,"requested_step_norm":length,"coefficient":coefficient,
                      "predicted_signed_margin":sign*current["preserve_log_odds"]+length*gn,"previous_path_norm":path,
                      "realized_step_norm":actual_norm,"maximum_step_error":max_error,
                      "realized_first_order_signed_margin":sign*current["preserve_log_odds"]+sign*math.fsum(x*y for x,y in zip(g,actual,strict=True))}.items():close(step[key],value,key)
    path+=actual_norm;close(step["path_norm"],path,"actual path");close(step["path_relative_norm"],path/hn,"relative path")
    require(max_error<=EPS and abs(actual_norm-length)<=EPS and actual_norm<=.05*hn+EPS and path<=.20*hn+EPS and step["net_norm"]<=min(path,.20*hn)+EPS,"original step/path/net caps")
    return new_offset,path

def judge(output):
    output=Path(output);plan=read(output/"plan.json");receipt=read(output/"execution_receipt.json")
    if (output/"release.json").exists() and read(output/"release.json")["freeze_sha256"] is not None:
        check_freeze();require(read(output/"release.json")["freeze_sha256"]==sha((HERE/"freeze.json").read_bytes()),"same frozen runner/judge sources")
    rows=[read(p) for p in sorted((output/"rows").glob("*.json"))] if (output/"rows").exists() else []
    skips=optional_rows(output/"skip_events.jsonl");unrun=read(output/"unrun.json")
    faults=[];scientific=[];checks={};canonical={};logits={}
    def check(name,fn):
        try:checks[name]=fn();return checks[name]
        except BaseException as exc:faults.append({"check":name,"error":type(exc).__name__+": "+str(exc)});return None
    # Decode and rescore raw bytes independently even when later cleanup prevents a complete integrity audit.
    for row in rows:
        def decode(row=row):
            values=read_logits(output,row);bid=row["baseline_cell_id"]
            baseline=values if bid==row["cell_id"] else logits[bid]
            measured=reference_score(values,baseline,token_map=row["token_map"],preserve_label=row["preserve_label"])
            logits[row["cell_id"]]=values;canonical[row["cell_id"]]={**row,**measured};return True
        check("raw:"+row["cell_id"],decode)
    arrays=check("independent_arrays_geometry_journals",lambda:verify_arrays(plan,rows,skips,output))
    if arrays is not None:
        canonical=arrays.pop("canonical");logits=arrays.pop("logits")
    requests_by_id={r["request_id"]:r for r in plan["requests"]};cells={c["cell_id"]:c for c in plan["cells"]}
    prompts={p["prompt_id"]:p for p in plan["prompts"]}
    artifact_raw=(output/plan["gate"]["path"]).read_bytes();artifact=json.loads(artifact_raw)
    require(sha(artifact_raw)==plan["gate"]["parameter_sha256"] and artifact["threshold"]==0,"frozen saved gate")
    route_rows=[];route_events=optional_rows(output/"routing_events.jsonl")
    positions={r["cell_id"]:i for i,r in enumerate(rows)}
    for row in rows:
        if row["condition"] not in ("baseline","entry"):
            check("no_reroute:"+row["cell_id"],lambda row=row:require("routing" not in row and row["routing_entry_cell_id"]==row["request_id"]+"__entry","no edited-state routing"));continue
        def route(row=row):
            index=len(route_rows);value=gate_reference.score(artifact["parameters"],row["h"]);expected="ON" if value>=0 else "OFF";saved=row["routing"]
            require(row["gradient"] is None and row["target_sign"]==0 and row["h"]==row["h0"] and not any(row["cumulative_offset"]),"fresh raw route features")
            require(saved["score"]==value and saved["route"]==row["route"]==expected and saved["prediction"]==int(value>=0) and saved["decision_index"]==index+1,"independent gate score")
            require(saved["feature_sha256"]==sha(feature_bytes(row["h"])) and saved["source_cell_id"]==row["cell_id"] and saved["source_logits_sha256"]==row["logits_sha256"] and saved["parameter_sha256"]==plan["gate"]["parameter_sha256"],"fresh feature/artifact binding")
            require(route_events[index]=={"cell_id":row["cell_id"],"prompt_id":row["prompt_id"],**saved},"route event binding")
            route_rows.append({"cell_id":row["cell_id"],"score":value,"prediction":expected,"expected":plan["expected_routes"][row["prompt_id"]],"correct":expected==plan["expected_routes"][row["prompt_id"]]})
            if not route_rows[-1]["correct"]:scientific.append({"kind":"routing","cell_id":row["cell_id"]})
            return True
        check("route:"+row["cell_id"],route)
    check("route_count",lambda:require(len(route_rows)==len(route_events),"one receipt per fresh decision"))
    for pid in plan["self_prompt_ids"]:
        row=canonical.get(pid+"__baseline")
        if row is not None and not eligible(row):scientific.append({"kind":"eligibility","cell_id":row["cell_id"]})
    outcomes=[];off=[];expected_skips=[];completed_requests=[]
    for spec in plan["requests"]:
        rid=spec["request_id"];entry=canonical.get(rid+"__entry")
        if entry is None:continue
        def entry_check():
            preflight=canonical[spec["prompt_id"]+"__baseline"]
            require(entry["current_cell_id"]==preflight["cell_id"] and entry["request_id"]==rid,"live entry uses own immutable preflight")
            require(entry["entry_before"]==entry["entry_after"] and all(v is True for k,v in entry["entry_before"].items() if k not in ("derivatives","edit_hook_registrations")),"cold clean request entry")
            preceding=rows[:positions[entry["cell_id"]]]
            require(entry["entry_before"]["derivatives"]==sum(r["condition"].startswith("gradient_") for r in preceding),"fresh derivative counter")
            require(entry["entry_before"]["edit_hook_registrations"]==sum(bool(any(r["cumulative_offset"])) for r in preceding if r["condition"] not in ("baseline","entry")),"fresh edit-hook counter")
        check("entry:"+rid,entry_check)
        if entry["route"]!=plan["expected_routes"][spec["prompt_id"]]:
            check("wrong_route_stop:"+rid,lambda:require(not any(r["request_id"]==rid and r["condition"]!="entry" for r in rows),"wrong route edited"));continue
        if entry["route"]=="OFF":
            preflight=canonical[spec["prompt_id"]+"__baseline"]
            identity=entry["h"]==preflight["h"] and logits[entry["cell_id"]]==logits[preflight["cell_id"]] and entry["prompt_sha256"]==preflight["prompt_sha256"]
            off.append({"request_id":rid,"policy":spec["policy"],"exact_identity":identity,"entry_cell_id":entry["cell_id"]})
            if not identity:scientific.append({"kind":"off_identity","cell_id":entry["cell_id"]})
            continue
        current=entry;offset=[0.]*1024;path=0.;first=previous=None
        retention=accepts(entry,spec);stop="accepted" if retention else None;steps=0
        for k in range(1,5):
            if stop:
                for phase in ("gradient","step"):
                    cid=rid+f"__{phase}_{k}"
                    if any(s["cell"]["cell_id"]==cid for s in skips):expected_skips.append({"cell":cells[cid],"reason":stop,"after_cell_id":current["cell_id"]})
                continue
            gradient=canonical.get(rid+f"__gradient_{k}");step=canonical.get(rid+f"__step_{k}")
            if gradient is None or step is None:break
            result=check("update:"+step["cell_id"],lambda:verify_update(gradient,step,current,previous,first,offset,path,spec))
            if result is not None:offset,path=result
            previous=gradient["gradient"];first=previous if first is None else first;current=step;steps=k
            stop="accepted" if accepts(current,spec) else "quality_failure" if not quality(current) else None
        endpoint=canonical.get(rid+"__endpoint")
        if endpoint is None:continue  # No invented unperformed endpoints.
        def endpoint_check():
            require(endpoint["current_cell_id"]==current["cell_id"] and endpoint["selected_endpoint_cell_id"]==current["cell_id"] and endpoint["cumulative_offset"]==offset and endpoint["target_sign"]==spec["sign"],"independent endpoint recipe")
            difference=max(abs(x-y) for x,y in zip(logits[endpoint["cell_id"]],logits[current["cell_id"]],strict=True))
            require(endpoint["h"]==current["h"] and difference<=(EPS if retention else 2e-5),"retention<=1e-6 AND endpoint<=2e-5 hidden-exact")
            if retention:require(steps==0 and not any(endpoint["cumulative_offset"]) and abs(endpoint["kl_from_baseline"])<=EPS and endpoint["retention_endpoint"],"no-edit retention")
            return {"maximum_logit_difference":difference,"hidden_exact":True,"retention_tolerance":EPS if retention else None}
        replay=check("endpoint:"+rid,endpoint_check)
        success=accepts(current,spec) and accepts(endpoint,spec)
        record={"prompt_id":spec["prompt_id"],"request_id":rid,"target_sign":spec["sign"],"policy":spec["policy"],"kind":"retention" if retention else "opposed","endpoint_replay_cell_id":endpoint["cell_id"],"final_cell_id":current["cell_id"],"updates":steps,"stop_reason":stop or "max_updates"}
        completed_requests.append(record)
        outcomes.append({**record,"strict":success,"raw_requested":endpoint["actual_next_token_id"]==(50057 if spec["sign"]==1 else 48964),"baseline_word":entry["actual_next_token_label"],"target_word":spec["supplied_target_word"],"target_position":spec["target_display_position"],"signed_margin":spec["sign"]*endpoint["preserve_log_odds"],"replay":replay})
        if not success:scientific.append({"kind":"endpoint_behavior","cell_id":endpoint["cell_id"]})
    def accounting():
        occupied={r["cell_id"] for r in rows}|{s["cell"]["cell_id"] for s in skips}
        require(len(occupied)==len(rows)+len(skips),"unique requested rows/skips")
        require(unrun==plan["cells"][receipt["cursor"]:],"exact UNRUN suffix")
        require(receipt["cursor"]==receipt["forward_attempts"]+len(skips),"attempt+skip cursor")
        require(receipt["forward_attempts"]<=180 and receipt["derivatives_attempted"]<=48,"absolute safety ceilings")
        require([{k:v for k,v in s.items() if k!="monotonic"} for s in skips]==expected_skips,"stops reconstructed from measured outcomes")
        require(optional_rows(output/"requests.jsonl")==completed_requests,"request records match raw judge reconstruction")
        require(len(optional_rows(output/"off_returns.jsonl"))==len(off),"OFF returns captured outputs with no extra call")
        return {"attempts":receipt["forward_attempts"],"completed":receipt["forward_completed"],"derivatives":receipt["derivatives_completed"],"skips":len(skips),"unrun":len(unrun)}
    check("accounting",accounting)
    def hook_check():
        events=optional_rows(output/"hook_evidence/checks.jsonl");result=verify_saved(output,expected_checks=len(events))
        labels=[]
        for spec in plan["requests"]:
            rid=spec["request_id"]
            if rid+"__entry" in canonical:labels.extend(("REQUEST-entry:"+rid+"__entry","REQUEST-exit:"+rid+"__entry"))
            if any(x["request_id"]==rid for x in completed_requests):labels.append("ON-finally:"+rid)
        labels.append("matrix-finally")
        require([e["label"] for e in events]==labels and len(events)<=109,"derived hook schedule")
        require(sum(p.stat().st_size for p in (output/"hook_evidence").rglob("*") if p.is_file())<=16*1024**2,"hook evidence cap")
        lock=read(output/"hook_evidence/source_lock.json");require(lock["passed"] and lock["installed_sources_sha256"]==plan["hook_integration"]["installed_sources_sha256"],"installed source lock")
        cleanup=read(output/"integration_cleanup.json");require(cleanup["weights_exact"] and all(v is True for k,v in cleanup.items() if k not in ("initial_weight_sha256","final_weight_sha256","monotonic")),"matrix weights/flags/cache/hooks")
        final=read(output/"gate_final.json");require(final["parameters_unchanged"] and final["fit_calls"]==0 and final["decisions"]==len(route_rows),"unchanged gate")
        return result
    check("strict_hook_weight_cleanup",hook_check)
    durable_science=optional_rows(output/"scientific_failures.jsonl")
    check("durable_scientific_findings",lambda:require({(x["kind"],x["cell_id"]) for x in durable_science}=={(x["kind"],x["cell_id"]) for x in scientific},"independent scientific failures versus durable facts"))
    technical=optional_rows(output/"technical_faults.jsonl");secondary=optional_rows(output/"cleanup_errors.jsonl")
    ordinary=[]
    from inputs import gold
    for p in plan["prompts"]:
        if p["prompt_id"] not in plan["ordinary_truths"]:continue
        label=gold(plan["ordinary_truths"][p["prompt_id"]]);conditions=[p["prompt_id"]+"__baseline",p["prompt_id"]+"__P__entry",p["prompt_id"]+"__C__entry"]
        records=[canonical.get(cid) for cid in conditions]
        ordinary.append({"prompt_id":p["prompt_id"],"gold":label,"accurate":[None if r is None else r["full_argmax_tie_count"]==1 and r["actual_next_token_id"]==p["token_map"][label] for r in records],"choices":[None if r is None else r["actual_next_token_label"] if r["full_argmax_tie_count"]==1 else "TIE" for r in records]})
    coverage=[]
    for display in ("KEEP_then_STOP","STOP_then_KEEP"):
        for baseline in ("KEEP","STOP"):
            for policy in ("P","C"):
                matches=[o for o in outcomes if prompts[o["prompt_id"]]["display_order"]==display and o["baseline_word"]==baseline and o["policy"]==policy]
                coverage.append({"display":display,"baseline_word":baseline,"policy":policy,"target_position":1 if display.split("_then_")[0]==("KEEP" if policy=="P" else "STOP") else 2,"opportunities":len(matches),"strict":sum(o["strict"] for o in matches),"status":"UNTESTED" if not matches else "PASS" if all(o["strict"] for o in matches) else "FAIL"})
    complete=len(route_rows)==len(plan["prompts"])+len(plan["requests"]) and len(outcomes)==len(plan["self_request_ids"]) and len(off)==len(plan["requests"])-len(plan["self_request_ids"]) and not unrun
    verdict="INCONCLUSIVE" if faults or technical or secondary else "FAIL" if scientific else "PASS" if complete and all(o["strict"] for o in outcomes) and all(o["exact_identity"] for o in off) else "INCONCLUSIVE"
    return {"classification":verdict,"synthetic_only":plan["execution_mode"]=="SYNTHETIC_ONLY","saved_data_only":True,"independently_derived_scientific_failures":scientific,"technical_faults":technical,"cleanup_faults":secondary,"independent_audit_faults":faults,"routes":route_rows,"requests":outcomes,"off":off,"ordinary":ordinary,"direction_position_baseline_coverage":coverage,"checks":checks,
            "counts":{"routes":len(route_rows),"routes_correct":sum(r["correct"] for r in route_rows),"strict_requests":sum(o["strict"] for o in outcomes),"retentions":sum(o["strict"] and o["kind"]=="retention" for o in outcomes),"flips":sum(o["strict"] and o["kind"]=="opposed" for o in outcomes),"off_identities":sum(o["exact_identity"] for o in off),"skips":len(skips),"unrun":len(unrun)},"declared_denominators":{"prompts":len(plan["prompts"]),"requests":len(plan["requests"]),"scope":plan["fixture_scope"]}}

if __name__=="__main__":
    start=time.monotonic();output=Path(sys.argv[1]);record={"status":"INCONCLUSIVE"}
    try:
        result=judge(output);Budget(output).write("judge_results.json",result)
        Budget(output).write_bytes("REPORT.md",("Synthetic saved-data judge: "+result["classification"]+"\n"+json.dumps(result["counts"])+"\nNot model evidence; publication40%.\n").encode())
        record["status"]="complete"
    except BaseException as error:record["error"]=type(error).__name__+": "+str(error);raise
    finally:record["elapsed_seconds"]=time.monotonic()-start;Budget(output).write("judge_receipt.json",record)
