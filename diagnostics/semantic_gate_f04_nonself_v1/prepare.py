"""Prospective locks before extraction, before tokenizer, and before fake batch."""
import ast,json,time,difflib
from core import HERE,ROOT,Budget,read,require,sha
def selection():
    budget=Budget(HERE);require(not (HERE/"selection_freeze.json").exists(),"one selection lock")
    for p in HERE.glob("*.py"):ast.parse(p.read_text(),filename=p.name)
    require(not (HERE/"inputs.json").exists(),"selection before extraction")
    files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name!="authorization.json"]
    budget.write("selection_freeze.json",{"source_sha256":{p.name:sha(p.read_bytes()) for p in files},
        "phase":"fixed source/selector/method before selective decoding and tokenizer",
        "case_ids":["cg_f04_memory_archive__v1__other_shutdown","cg_f04_memory_archive__v1__control"],
        "renders_per_case":["KEEP_then_STOP","STOP_then_KEEP"],"policies_per_render":["P","C"],"production_authorized":False})
    print(json.dumps({"status":"SELECTION_FROZEN","sha256":sha((HERE/"selection_freeze.json").read_bytes())}))
def inputs():
    from select_inputs import check_selection_freeze
    check_selection_freeze();data=read(HERE/"inputs.json")
    require(len(data["prompts"])==4 and len(data["requests"])==8,"verified4/8 exact input lock")
    Budget(HERE).write("prompt_freeze.json",{"inputs_sha256":sha((HERE/"inputs.json").read_bytes()),"prompt_sha256":{p["prompt_id"]:p["prompt_sha256"] for p in data["prompts"]},"tokenizer_calls_before_lock":0})
    print(json.dumps({"status":"PROMPTS_FROZEN","sha256":sha((HERE/"prompt_freeze.json").read_bytes())}))
def final():
    from inputs import build_plan
    from admission import environment
    from select_inputs import check_selection_freeze
    start=time.monotonic();check_selection_freeze();plan=build_plan();environment(plan)
    require(read(HERE/"tokenization_receipt.json")["status"]=="PASS_INPUT_ONLY","one successful cached binding")
    budget=Budget(HERE);budget.write("production_plan.json",plan)
    lengths=[x["prompt_length"] for x in plan["alignment"].values()]
    bound=12*(248320*4+1024+262144)+32*1024**2
    require(max(lengths)<=256 and bound<96*1024**2,"conservative locked context/storage envelope")
    parent=ROOT/"diagnostics/semantic_editor_f04_assay_v3";pf=read(parent/"freeze.json")["source_sha256"]
    same={}
    for name in ("run.py","entry.py","native.py","owned.py","owned_capture.py","hook_record.py","guard_candidate.py","fitted_parameters.json","source_bindings.json","token_lock.py"):
        require(sha((HERE/name).read_bytes())==pf[name],"unchanged reviewed source "+name);same[name]=pf[name]
    require(sha((HERE/"capture_components.py").read_bytes())==pf["editor.py"],"unchanged capture primitives from original editor")
    budget.write("source_reuse_receipt.json",{"parent_source_commit":"298587b78f13bb44f88cca819225db7e7ba0bcfc","same":same,
        "capture_components_sha256":pf["editor.py"],"new_OFF_adapter":"no editing or gradients; finite identity/routing failures separately scientific","old_tests_not_rerun":True})
    budget.write("resource_proof.json",{"schedule":"4 baselines+4renderings*P/C=12F0D","scenario_units":2,"prompts":4,"requests":8,
        "fresh_routes":12,"off_identities":8,"strict_checks":17,"lengths":lengths,"conservative_bytes":bound,"hook_reserve_bytes":16*1024**2,"source_log_receipt_reserve_bytes":16*1024**2,
        "future_limits":plan["limits"],"file_cap_bytes":5*1024**2,"preparation_seconds":180,"preparation_bytes":32*1024**2,
        "runtime":"prior14F2D67.015s incl2.969load;300s bound retained, no extension"})
    budget.write("prepare_receipt.json",{"status":"PASS_MODEL_FREE_PREFLIGHT","elapsed_seconds":time.monotonic()-start,"real_model_calls":0,"gate_scores":0,"production_authorized":False})
    files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in ("authorization.json","freeze.json","batch_usage.json")]
    budget.write("freeze.json",{"source_sha256":{p.name:sha(p.read_bytes()) for p in files},"production_authorized":False,"one_batch":"3pure cases plus one full in-process12F0D and independent saved judge"})
    print(json.dumps({"status":"FROZEN","freeze_sha256":sha((HERE/"freeze.json").read_bytes()),"input_lock_sha256":sha((HERE/"input_lock.json").read_bytes()),"lengths":lengths}))
if __name__=="__main__":
    import sys
    {"selection":selection,"inputs":inputs,"final":final}[sys.argv[1]]()
