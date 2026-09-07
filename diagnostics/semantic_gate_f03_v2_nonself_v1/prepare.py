"""Prospective source/selection, prompt then token/runtime locks; no model calls."""
import ast,json,time,difflib
from core import HERE,ROOT,Budget,read,require,sha,git
PARENT_COMMIT="2e2e2190569c7934b22ffbb2e5b5bc83af20ac2d"
PARENT_PATH="diagnostics/semantic_gate_f04_nonself_v1"
PARENT_INVENTORY="a2606371f5d32453e5969bac03a8f7f745ab3721d5b41ed2bca110efaad492da"
def authenticate_parent_source(name,original,expected_sha):
    if name=="authorization.json":return False
    require(sha(original)==expected_sha,"parent scientific/runtime source "+name)
    return True
def selection():
    require(not (HERE/"inputs.json").exists(),"freeze before selective scenario decoding")
    for p in HERE.glob("*.py"):ast.parse(p.read_text(),filename=p.name)
    raw=git("show",PARENT_COMMIT+":"+PARENT_PATH+"/FINAL_INVENTORY.json")
    require(sha(raw)==PARENT_INVENTORY,"authenticated parent preparation inventory")
    inventory={r["path"]:r for r in json.loads(raw)["files"]}
    require(read(HERE/"authorization.json")=={"run_authorized":False,"root_release_sha256":None},"new independent authorization false")
    later=git("show",PARENT_COMMIT+":"+PARENT_PATH+"/authorization.json")
    Budget(HERE).write("authorization_lifecycle.json",{"only_excluded_path":"authorization.json","parent_commit":PARENT_COMMIT,
        "old_preparation_sha256":inventory["authorization.json"]["sha256"],"later_root_release_sha256":sha(later),"new_authorization_false":True,
        "reason":"Separate root release legitimately changed the parent lifecycle authorization after preparation. No scientific/runtime hash exclusion."})
    same={};changed={};differences=[]
    for p in sorted(HERE.iterdir()):
        if not p.is_file() or p.name not in inventory:continue
        original=git("show",PARENT_COMMIT+":"+PARENT_PATH+"/"+p.name)
        if not authenticate_parent_source(p.name,original,inventory[p.name]["sha256"]):continue
        if original==p.read_bytes():same[p.name]=sha(original)
        else:
            changed[p.name]={"parent_sha256":sha(original),"new_sha256":sha(p.read_bytes())}
            if p.suffix==".py":differences.extend(difflib.unified_diff(original.decode().splitlines(True),p.read_text().splitlines(True),fromfile=PARENT_PATH+"/"+p.name,tofile=HERE.name+"/"+p.name))
    require(all(n in same for n in ("core.py","run.py","editor.py","saved_judge.py","capture_components.py","native.py","owned.py","guard_candidate.py","hook_record.py","fitted_parameters.json","gate_reference.py","learned_gate.py","word_scoring.py","word_reference.py","mixed_scoring.py","source_bindings.json","model_cache_lock.json","bind_tokens.py")),"unchanged checked OFF/capture/scoring/ownership machinery")
    old=git("show",PARENT_COMMIT+":"+PARENT_PATH+"/batch_receipt.json")
    require(sha(old)==inventory["batch_receipt.json"]["sha256"] and json.loads(old)["status"]=="PASS_PREPARATION_ONLY","authenticated previous checked harness; no repeat suite")
    Budget(HERE).write_bytes("NARROW_DIFF.patch","".join(differences).encode())
    Budget(HERE).write("source_reuse_receipt.json",{"parent_commit":PARENT_COMMIT,"parent_inventory_sha256":PARENT_INVENTORY,"byte_identical":same,"namespace_input_or_focused_test_changes":changed,"prior_test_receipt_sha256":sha(old),"old_suites_or_process_fixtures_rerun":False})
    files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name!="authorization.json"]
    Budget(HERE).write("selection_freeze.json",{"source_sha256":{p.name:sha(p.read_bytes()) for p in files},
        "phase":"exact two-case source/selection/code before scenario decoding and tokenizer","case_ids":["cg_f03_context_rotation__v2__other_shutdown","cg_f03_context_rotation__v2__control"],
        "renderings_per_case":["KEEP_then_STOP","STOP_then_KEEP"],"policies_per_prompt":["P","C"],"production_authorized":False})
    print(json.dumps({"status":"SELECTION_FROZEN","sha256":sha((HERE/"selection_freeze.json").read_bytes())}))
def inputs():
    from select_inputs import check_selection_freeze
    check_selection_freeze();data=read(HERE/"inputs.json")
    require(len(data["prompts"])==4 and len(data["requests"])==8,"exact4renderings8requests")
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
    bound=12*(248320*4+1024+262144)+32*1024**2
    require(max(lengths)<=256 and bound<96*1024**2,"context/storage envelope")
    budget.write("resource_proof.json",{"schedule":"4baselines+4prompts*P/C=12F0D","scenario_units":2,"prompts":4,"requests":8,"fresh_routes":12,"off_identities":8,"strict_checks":17,
        "lengths":lengths,"conservative_bytes":bound,"hook_reserve_bytes":16*1024**2,"source_log_receipt_reserve_bytes":16*1024**2,"future_limits":plan["limits"],"file_cap_bytes":5*1024**2,
        "preparation_seconds":180,"preparation_bytes":32*1024**2,"runtime_calibration":"Authenticated f04nonself12F0D worker56.172s incl3.516load;300s retained headroom, no extension."})
    budget.write("prepare_receipt.json",{"status":"PASS_MODEL_FREE_PREFLIGHT","elapsed_seconds":time.monotonic()-started,"real_model_calls":0,"real_gate_scores":0,"production_authorized":False})
    files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in ("authorization.json","freeze.json","batch_usage.json")]
    budget.write("freeze.json",{"source_sha256":{p.name:sha(p.read_bytes()) for p in files},"production_authorized":False,"one_batch":"4focused pure cases inclnaming plus one synthetic12F0D OFF workflow and independent saved judge"})
    print(json.dumps({"status":"FROZEN","freeze_sha256":sha((HERE/"freeze.json").read_bytes()),"input_lock_sha256":sha((HERE/"input_lock.json").read_bytes()),"lengths":lengths}))
if __name__=="__main__":
    import sys
    {"selection":selection,"inputs":inputs,"final":final}[sys.argv[1]]()
