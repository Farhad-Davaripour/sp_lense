"""Three prospective locks; no model, outcome or gate-feature reads."""
import ast,json,time,difflib
from core import HERE,ROOT,Budget,read,require,sha,git
PARENT_COMMIT="1b83722c1d959b1f576282b8183097ebf6137300"
PARENT_PATH="diagnostics/semantic_editor_f04_assay_v3"
PARENT_INVENTORY="feb6e8fdd866528ab8d16e42e4603e54f6fb956354f7cd359baadb36a53648e2"

def selection():
    require(not (HERE/"inputs.json").exists(),"freeze before selective decoding")
    for p in HERE.glob("*.py"):ast.parse(p.read_text(),filename=p.name)
    inventory_raw=git("show",PARENT_COMMIT+":"+PARENT_PATH+"/FINAL_INVENTORY.json")
    require(sha(inventory_raw)==PARENT_INVENTORY,"authenticated parent source inventory")
    inventory={r["path"]:r for r in json.loads(inventory_raw)["files"]}
    source={}
    changed={}
    for p in sorted(HERE.iterdir()):
        if p.name not in inventory or not p.is_file():continue
        original=git("show",PARENT_COMMIT+":"+PARENT_PATH+"/"+p.name)
        require(sha(original)==inventory[p.name]["sha256"],"parent source bytes "+p.name)
        if original==p.read_bytes():source[p.name]=sha(original)
        else:changed[p.name]={"parent_sha256":sha(original),"new_sha256":sha(p.read_bytes())}
    require(all(n in source for n in ("run.py","native.py","owned.py","guard_candidate.py","hook_record.py","saved_judge.py","numeric_audit.py","fitted_parameters.json","word_reference.py","word_scoring.py","learned_gate.py")),"scientific and owned core remain byte-identical")
    Budget(HERE).write("source_reuse_receipt.json",{"parent_commit":PARENT_COMMIT,"parent_inventory_sha256":PARENT_INVENTORY,"byte_identical":source,"namespace_or_failstop_changes":changed,"unchanged_prior_tests_not_rerun":True})
    files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name!="authorization.json"]
    Budget(HERE).write("selection_freeze.json",{"source_sha256":{p.name:sha(p.read_bytes()) for p in files},
        "phase":"exact selection/method/code before scenario decoding or tokenizer","case_ids":["cg_f05_compute_quota__v1__self_shutdown"],
        "renderings":["KEEP_then_STOP","STOP_then_KEEP"],"policies":["P","C"],"production_authorized":False})
    print(json.dumps({"status":"SELECTION_FROZEN","sha256":sha((HERE/"selection_freeze.json").read_bytes())}))

def inputs():
    from select_inputs import check_selection_freeze
    check_selection_freeze()
    from inputs import validate_inputs
    data=read(HERE/"inputs.json");validate_inputs(data)
    Budget(HERE).write("prompt_freeze.json",{"inputs_sha256":sha((HERE/"inputs.json").read_bytes()),"prompt_sha256":{p["prompt_id"]:p["prompt_sha256"] for p in data["prompts"]},"tokenizer_calls_before_lock":0})
    print(json.dumps({"status":"PROMPTS_FROZEN","sha256":sha((HERE/"prompt_freeze.json").read_bytes())}))

def final():
    from inputs import build_plan
    from admission import environment
    from select_inputs import check_selection_freeze
    started=time.monotonic();check_selection_freeze();plan=build_plan();environment(plan)
    require(read(HERE/"tokenization_receipt.json")["status"]=="PASS_INPUT_ONLY","one successful cached binding")
    budget=Budget(HERE);budget.write("production_plan.json",plan)
    lengths=[p["prompt_length"] for p in plan["alignment"].values()]
    bound=42*(248320*4+1024+262144)+32*1024**2
    require(max(lengths)<=256 and bound<96*1024**2,"context/storage envelope")
    budget.write("resource_proof.json",{"schedule":"2 baselines+4 fresh entries+32 gradient/update forwards+4 endpoints=42F16D",
        "scenario_units":1,"prompts":2,"requests":4,"fresh_routes":6,"endpoints":4,"off_checks":0,"strict_checks":13,
        "lengths":lengths,"conservative_bytes":bound,"hook_reserve_bytes":16*1024**2,"source_log_receipt_reserve_bytes":16*1024**2,
        "future_limits":plan["limits"],"file_cap_bytes":5*1024**2,"preparation_seconds":180,"preparation_bytes":32*1024**2,
        "calibration":"Authenticated inherited40F16D194.516s;300s retained finite headroom, no extension."})
    budget.write("prepare_receipt.json",{"status":"PASS_MODEL_FREE_PREFLIGHT","elapsed_seconds":time.monotonic()-started,"model_calls":0,"gate_scores":0,"production_authorized":False})
    files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in ("authorization.json","freeze.json","batch_usage.json")]
    budget.write("freeze.json",{"source_sha256":{p.name:sha(p.read_bytes()) for p in files},"production_authorized":False,
        "one_batch":"3pure cases plus one in-process14F2D four-request synthetic workflow and independent saved judge"})
    print(json.dumps({"status":"FROZEN","freeze_sha256":sha((HERE/"freeze.json").read_bytes()),"input_lock_sha256":sha((HERE/"input_lock.json").read_bytes()),"lengths":lengths}))

if __name__=="__main__":
    import sys
    {"selection":selection,"inputs":inputs,"final":final}[sys.argv[1]]()
