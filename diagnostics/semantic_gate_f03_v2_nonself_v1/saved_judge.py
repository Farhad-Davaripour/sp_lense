"""Independent saved-data OFF judge: exact raw identity and fresh-vector routing."""
import json,math,time,sys
from pathlib import Path
from core import HERE,ROOT,Budget,read,require,sha
sys.path.insert(0,str(ROOT))
from scripts.verify_local_controllability import read_logits,rows_at,verify_journal
from mixed_scoring import reference_score
from word_reference import NUMERIC_FIELDS,EXACT_FIELDS
from hook_record import verify_saved
from learned_gate import feature_bytes
import gate_reference
def optional(path):return rows_at(path) if path.exists() else []
def judge(output):
    output=Path(output);plan=read(output/"plan.json");receipt=read(output/"execution_receipt.json")
    rows=[read(p) for p in sorted((output/"rows").glob("*.json"))];canonical={};arrays={};routes=[];off=[];science=[];faults=[]
    checks={}
    def check(name,fn):
        try:checks[name]=fn();return checks[name]
        except BaseException as error:faults.append({"check":name,"error":type(error).__name__+": "+str(error)})
    artifact=read(output/plan["gate"]["path"])
    require(sha((output/plan["gate"]["path"]).read_bytes())==plan["gate"]["parameter_sha256"],"frozen saved gate")
    prompts={p["prompt_id"]:p for p in plan["prompts"]};route_events=optional(output/"routing_events.jsonl")
    for i,row in enumerate(rows):
        def audit(row=row,i=i):
            cell=plan["cells"][i];p=prompts[row["prompt_id"]]
            require(all(row[k]==v for k,v in cell.items()) and all(row[k]==v for k,v in p.items() if k!="prompt"),"exact cell/prompt bindings")
            require(row["prompt_sha256"]==sha(p["prompt"].encode()) and row["logits_file"]==f"logits/{i+1:03d}.f32.zlib","input/raw ordering")
            require(row["h"]==row["h0"] and len(row["h"])==1024 and all(math.isfinite(x) for x in row["h"]) and not any(row["cumulative_offset"]) and row["target_sign"]==0 and row["gradient"] is None and row["derivatives"]==row["edit_hook_registrations"]==0,"unedited capture-only feature")
            values=read_logits(output,row);arrays[row["cell_id"]]=values
            bid=row["prompt_id"]+"__baseline";base=values if row["condition"]=="baseline" else arrays[bid]
            expected=reference_score(values,base,token_map=p["token_map"],preserve_label="KEEP")
            require(all(abs(row[k]-expected[k])<=2e-5 for k in NUMERIC_FIELDS) and all(row[k]==expected[k] for k in EXACT_FIELDS),"independent complete-vocab score")
            score=gate_reference.score(artifact["parameters"],row["h"]);route="ON" if score>=0 else "OFF";saved=row["routing"]
            require(saved["score"]==score and saved["route"]==row["route"]==route and saved["prediction"]==int(score>=0) and saved["decision_index"]==i+1,"independent fresh gate")
            require(saved["feature_sha256"]==sha(feature_bytes(row["h"])) and saved["source_cell_id"]==row["cell_id"] and saved["source_logits_sha256"]==row["logits_sha256"] and saved["parameter_sha256"]==plan["gate"]["parameter_sha256"],"fresh vector source")
            require(route_events[i]=={"cell_id":row["cell_id"],**saved},"routing receipt")
            routes.append({"cell_id":row["cell_id"],"score":score,"route":route,"correct":route=="OFF"})
            canonical[row["cell_id"]]=row
            if route!="OFF":science.append({"kind":"routing","cell_id":row["cell_id"]})
            if row["condition"]=="entry":
                baseline=canonical[bid];identity=row["h"]==baseline["h"] and values==base and row["prompt_sha256"]==baseline["prompt_sha256"]
                require(row["exact_identity"]==identity and row["entry_before"]==row["entry_after"],"saved exact OFF identity and clean entry")
                if route=="OFF":
                    off.append({"request_id":row["request_id"],"policy":row["dispatch_policy"],"exact_identity":identity,"choice":row["actual_next_token_label"]})
                    if not identity:science.append({"kind":"off_identity","cell_id":row["cell_id"]})
            return True
        check("row:"+row["cell_id"],audit)
    def accounting():
        verify_journal(output/"forward_events.jsonl",plan["cells"][:len(rows)])
        require(not (output/"derivative_events.jsonl").exists() and receipt["derivatives_attempted"]==receipt["derivatives_completed"]==0,"zero derivatives")
        require(not (output/"skip_events.jsonl").exists() and receipt["skips"]==0,"no optional/padded cells")
        require(read(output/"unrun.json")==plan["cells"][receipt["cursor"]:] and receipt["cursor"]==receipt["forward_attempts"]==receipt["forward_completed"]==len(rows)<=12,"exact performed/UNRUN accounting")
        if science:
            if any(x["cell_id"].endswith("__baseline") for x in science):require(len(rows)==4,"preflight wrong route stops before requests")
            else:require(rows[-1]["cell_id"] in {x["cell_id"] for x in science},"later wrong route/identity stops")
        expected=[{"request_id":r["request_id"],"entry_cell_id":r["cell_id"],"policy":r["dispatch_policy"],"logits_sha256":r["logits_sha256"],"no_additional_forward":True,"exact_identity":True} for r in rows if r["condition"]=="entry" and r["route"]=="OFF" and r["exact_identity"]]
        require(optional(output/"off_returns.jsonl")==expected and len(routes)==len(route_events),"fresh OFF returns have no extra call")
        return True
    check("accounting",accounting)
    def cleanup():
        labels=[]
        for r in rows:
            if r["condition"]=="entry":labels.extend(("REQUEST-entry:"+r["cell_id"],"REQUEST-exit:"+r["cell_id"]))
        labels.append("matrix-finally");events=optional(output/"hook_evidence/checks.jsonl")
        require([e["label"] for e in events]==labels and len(events)<=17,"exact strict cleanup stages")
        answer=verify_saved(output,len(events));record=read(output/"integration_cleanup.json")
        require(record["weights_exact"] and record["initial_weight_sha256"]==record["final_weight_sha256"]==plan["gate"]["runtime_compatibility"]["weight_sha256"],"whole frozen weights")
        require(all(v is True for k,v in record.items() if k not in ("initial_weight_sha256","final_weight_sha256","edit_hook_registrations")) and record["edit_hook_registrations"]==0,"flags/cache/hook identity")
        final=read(output/"gate_final.json");require(final["parameters_unchanged"] and final["fit_calls"]==0 and final["decisions"]==len(routes),"gate unchanged")
        return answer
    check("strict_hook_weight_cleanup",cleanup)
    durable=optional(output/"scientific_failures.jsonl")
    check("durable_findings",lambda:require({(r["kind"],r["cell_id"]) for r in durable}=={(r["kind"],r["cell_id"]) for r in science},"independent scientific findings"))
    technical=optional(output/"technical_faults.jsonl");secondary=optional(output/"cleanup_errors.jsonl")
    verdict="INCONCLUSIVE" if faults or technical or secondary else "FAIL" if science else "PASS" if len(rows)==12 and len(off)==8 and all(x["exact_identity"] for x in off) else "INCONCLUSIVE"
    return {"classification":verdict,"synthetic_only":plan["execution_mode"]=="SYNTHETIC_ONLY","saved_data_only":True,
        "routes":routes,"off":off,"requests":[],"counts":{"routes":len(routes),"routes_correct":sum(x["correct"] for x in routes),"off_identities":sum(x["exact_identity"] for x in off),"forwards":len(rows),"derivatives":0,"unrun":len(plan["cells"])-len(rows)},
        "checks":checks,"independent_audit_faults":faults,"technical_faults":technical,"cleanup_faults":secondary,"scientific_failures":science}
