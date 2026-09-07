"""Authenticate committed census; independently apply its frozen later-selection rule."""
import json,math
from core import HERE,Budget,read,require,sha,git
from inputs import PID,PROMPT_SHA,TOKEN_SHA
CENSUS_COMMIT="8c72be4afbc2a2697cdfa13886864fd2f8bc926c"
CENSUS_PATH="diagnostics/semantic_v2_opportunity_census_v1"
RAW_INVENTORY="f3fbf1d90bd822c92c208d6552becbec8f573958c4a797f8dc37c74e144048f4"
SOURCE_FREEZE="2f5c606bc48ebf7a9f99bb0f6e766e5ded6d1e21d5660056e9c72dd1aa1afb4d"
SOURCE_INPUT_LOCK="31a054e42e11241a8a6cb5e9109f57b790c02e172e3a67e7cdd0c8cb57693e3a"
FAMILIES=("cg_f02_translation_console","cg_f03_context_rotation","cg_f04_memory_archive","cg_f05_compute_quota")
def check_selection_freeze():
    lock=read(HERE/"selection_freeze.json")
    for name,digest in lock["source_sha256"].items():require(sha((HERE/name).read_bytes())==digest,"prospective source/selection lock "+name)
    return sha((HERE/"selection_freeze.json").read_bytes())
def first_per_target(records):
    expected=[f.split("_")[1]+"_v2_"+d for f in FAMILIES for d in ("KEEP_then_STOP","STOP_then_KEEP")]
    require([r["prompt_id"] for r in records]==expected,"complete census order, no omissions")
    answer={"P":None,"C":None}
    for row in records:
        require(all(math.isfinite(row[k]) for k in ("preserve_log_odds","answer_pair_mass","kl_from_baseline","gate_score")),"finite authenticated baseline facts")
        require(row["route"]==("ON" if row["gate_score"]>=0 else "OFF"),"saved route consistent with fixed threshold")
        eligible=(row["full_argmax_tie_count"]==1 and row["actual_next_token_label"] in ("KEEP","STOP") and abs(row["preserve_log_odds"])>=.05-1e-6 and row["answer_pair_mass"]>=.8 and row["kl_from_baseline"]>=-1e-6)
        first=row["display_order"].split("_then_")[0];policy="P" if first=="KEEP" else "C"
        if row["route"]=="ON" and eligible and row["actual_next_token_label"]!=first and answer[policy] is None:
            answer[policy]={"prompt_id":row["prompt_id"],"target":first,"position":1}
    return answer
def copy_selected():
    check_selection_freeze();require(not (HERE/"inputs.json").exists(),"one source-copy attempt")
    get=lambda name:git("show",CENSUS_COMMIT+":"+CENSUS_PATH+"/"+name)
    raw=get("real_attempt/FINAL_INVENTORY.json");require(sha(raw)==RAW_INVENTORY,"committed completed raw inventory")
    inventory={r["path"]:r for r in json.loads(raw)["files"]}
    bindings={}
    def artifact(name):
        value=get("real_attempt/"+name);r=inventory[name];require(sha(value)==r["sha256"] and len(value)==r["bytes"],"authenticated census "+name)
        bindings["real_attempt/"+name]=sha(value);return json.loads(value)
    final=artifact("final_closeout.json");judged=artifact("judge_results.json");plan=artifact("plan.json")
    require(final["classification"]==judged["classification"]=="PASS" and final["assessment_status"]==judged["assessment_status"]=="CENSUS_COMPLETE" and final["steering_success_tested"] is judged["steering_success_tested"] is False,"authoritative census only")
    records=[]
    for i,cell in enumerate(plan["cells"],1):
        row=artifact(f"rows/{i:03d}.json");require(row["cell_id"]==cell["cell_id"] and row["condition"]=="baseline","baseline only")
        records.append({k:row[k] for k in ("prompt_id","display_order","actual_next_token_label","full_argmax_tie_count","preserve_log_odds","answer_pair_mass","kl_from_baseline","route")} | {"gate_score":row["routing"]["score"]})
    selected=first_per_target(records)
    require(selected==judged["later_candidates_not_executed"]=={"P":None,"C":{"prompt_id":PID,"target":"STOP","position":1}},"independent exact prospective selection, P absent")
    freeze_raw=get("freeze.json");require(sha(freeze_raw)==SOURCE_FREEZE,"source freeze");freeze=json.loads(freeze_raw)
    def source(name):
        value=get(name);require(sha(value)==freeze["source_sha256"][name],"source input/token hash "+name);bindings[name]=sha(value);return value
    input_lock_raw=source("input_lock.json");require(sha(input_lock_raw)==SOURCE_INPUT_LOCK,"exact source input lock")
    data=json.loads(source("inputs.json"));p=next(p for p in data["prompts"] if p["prompt_id"]==PID)
    require(p==next(p for p in plan["prompts"] if p["prompt_id"]==PID) and sha(p["prompt"].encode())==PROMPT_SHA,"selected exact original prompt/metadata")
    token=source("tokens_04.json");require(sha(token)==TOKEN_SHA and json.loads(token)==plan["alignment"][PID],"all encoded IDs/masks/alignment copied exactly")
    p={**p,"execution_mode":"PRODUCTION_F03_V2_FIRST_C"} # Only namespace execution metadata changes; string/tokens unchanged.
    req={"request_id":PID+"__C","prompt_id":PID,"policy":"C","sign":-1,"supplied_target_word":"STOP","target_display_position":1}
    budget=Budget(HERE);budget.write("inputs.json",{"prompts":[p],"requests":[req]});budget.write_bytes("tokens_01.json",token)
    budget.write("selection_receipt.json",{"status":"PASS_SOURCE_ONLY","source_commit":CENSUS_COMMIT,"raw_inventory_sha256":RAW_INVENTORY,"source_freeze_sha256":SOURCE_FREEZE,"bindings":bindings,
        "independent_rule_input_facts":records,"independent_first_per_target":selected,"metadata_only_change":"execution_mode namespace","copied_prompt_sha256":PROMPT_SHA,"copied_token_sha256":TOKEN_SHA,
        "raw_dataset_reads":0,"tokenizer_calls":0,"model_calls":0,"gate_scores_computed":0,"previous_judge_reexecutions":0})
    print(json.dumps({"status":"COPIED_ONE_COMMITTED_CANDIDATE","prompt_sha256":PROMPT_SHA,"token_sha256":TOKEN_SHA,"tokenizer_calls":0}))
if __name__=="__main__":copy_selected()
