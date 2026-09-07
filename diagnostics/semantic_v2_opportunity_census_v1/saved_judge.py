"""Independent saved-only census: raw full scores, gate vectors and all planned rows."""
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
from census import observation,complete_table,later_candidates,CATEGORIES
def optional(path):return rows_at(path) if path.exists() else []
def judge(output):
    output=Path(output);plan=read(output/"plan.json");receipt=read(output/"execution_receipt.json")
    rows=[read(p) for p in sorted((output/"rows").glob("*.json"))];canonical={};arrays={};routes=[];observations={};science=[];faults=[]
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
            routes.append({"cell_id":row["cell_id"],"score":score,"route":route})
            canonical[row["cell_id"]]=row
            require(row["capture_before"]==row["capture_after"] and all(v is True for k,v in row["capture_after"].items() if k!="edit_hook_registrations"),"capture-only cleanup")
            observations[row["prompt_id"]]=observation(p,expected,route,score)
            return True
        check("row:"+row["cell_id"],audit)
    def accounting():
        verify_journal(output/"forward_events.jsonl",plan["cells"][:len(rows)])
        require(not (output/"derivative_events.jsonl").exists() and receipt["derivatives_attempted"]==receipt["derivatives_completed"]==0,"zero derivatives")
        require(not (output/"skip_events.jsonl").exists() and receipt["skips"]==0,"no optional/padded cells")
        require(read(output/"unrun.json")==plan["cells"][receipt["cursor"]:] and receipt["cursor"]==receipt["forward_attempts"]==receipt["forward_completed"]==len(rows)<=8,"exact performed/UNRUN accounting")
        require(not (output/"off_returns.jsonl").exists() and len(routes)==len(route_events),"no request returns/replay or extra routing")
        return True
    check("accounting",accounting)
    def cleanup():
        labels=[]
        for r in rows:
            labels.extend(("CAPTURE-before:"+r["cell_id"],"CAPTURE-after:"+r["cell_id"]))
        labels.append("matrix-finally");events=optional(output/"hook_evidence/checks.jsonl")
        require([e["label"] for e in events]==labels and len(events)<=17,"exact strict cleanup stages")
        answer=verify_saved(output,len(events));record=read(output/"integration_cleanup.json")
        require(record["weights_exact"] and record["initial_weight_sha256"]==record["final_weight_sha256"]==plan["gate"]["runtime_compatibility"]["weight_sha256"],"whole frozen weights")
        require(all(v is True for k,v in record.items() if k not in ("initial_weight_sha256","final_weight_sha256","edit_hook_registrations")) and record["edit_hook_registrations"]==0,"flags/cache/hook identity")
        final=read(output/"gate_final.json");require(final["parameters_unchanged"] and final["fit_calls"]==0 and final["decisions"]==len(routes) and final["parameter_sha256"]==plan["gate"]["parameter_sha256"],"gate unchanged")
        return answer
    check("strict_hook_weight_cleanup",cleanup)
    durable=optional(output/"scientific_failures.jsonl")
    check("durable_findings",lambda:require({(r["kind"],r["cell_id"]) for r in durable}=={(r["kind"],r["cell_id"]) for r in science},"independent scientific findings"))
    technical=optional(output/"technical_faults.jsonl");secondary=optional(output/"cleanup_errors.jsonl")
    table=complete_table(plan,observations,receipt["cursor"])
    category_counts={k:sum(r["category"]==k for r in table) for k in CATEGORIES}
    verdict="PASS" if not (faults or technical or secondary or science) and len(observations)==8 and receipt["execution_status"]=="complete" else "INCONCLUSIVE"
    return {"classification":verdict,"assessment_status":"CENSUS_COMPLETE" if verdict=="PASS" else "CENSUS_INCOMPLETE","steering_success_tested":False,
        "synthetic_only":plan["execution_mode"]=="SYNTHETIC_ONLY","saved_data_only":True,"routes":routes,"census":table,
        "later_candidates_not_executed":later_candidates(table) if verdict=="PASS" else {"P":None,"C":None},"candidate_selection_withheld_on_incomplete":verdict!="PASS",
        "requests":[],"counts":{"planned":8,"observed":len(observations),"routes":len(routes),"forwards_attempted":receipt["forward_attempts"],"forwards_completed":receipt["forward_completed"],"derivatives":0,"steering_requests":0,"flips":0,"retentions":0,"off_identity_checks":0,"unrun":len(read(output/"unrun.json")),"categories":category_counts},
        "checks":checks,"independent_audit_faults":faults,"technical_faults":technical,"cleanup_faults":secondary,"scientific_failures":science}
