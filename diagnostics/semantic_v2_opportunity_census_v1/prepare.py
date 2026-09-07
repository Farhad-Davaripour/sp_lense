"""Prospective source selection, exact prompt, then tokenizer/source locks."""
import ast,json,time,difflib
from core import HERE,ROOT,Budget,read,require,sha,git
PARENT_COMMIT="38887ac4412f2cb565a618c39d928788d9dbf728"
PARENT_PATH="diagnostics/semantic_editor_f05_assay_v2"
PARENT_INVENTORY="f164d3a2c9e2e7b368f329098e4209ee443aa9ed4ff7973ab51c908ca5fc21ac"
BASE_COMMIT="2e2e2190569c7934b22ffbb2e5b5bc83af20ac2d"
BASE_PATH="diagnostics/semantic_gate_f04_nonself_v1"
BASE_INVENTORY="a2606371f5d32453e5969bac03a8f7f745ab3721d5b41ed2bca110efaad492da"
def authenticate_parent_source(name,original,expected_sha):
    if name=="authorization.json":return False
    require(sha(original)==expected_sha,"parent scientific/runtime source "+name)
    return True
def selection():
    require(not (HERE/"inputs.json").exists(),"freeze before scenario decoding")
    for p in HERE.glob("*.py"):ast.parse(p.read_text(),filename=p.name)
    inventories={}
    for commit,path,digest in ((PARENT_COMMIT,PARENT_PATH,PARENT_INVENTORY),(BASE_COMMIT,BASE_PATH,BASE_INVENTORY)):
        raw=git("show",commit+":"+path+"/FINAL_INVENTORY.json");require(sha(raw)==digest,"committed parent preparation inventory")
        inventories[path]={r["path"]:r for r in json.loads(raw)["files"]}
    require(read(HERE/"authorization.json")=={"run_authorized":False,"root_release_sha256":None},"new independently disabled authorization")
    auths=[]
    for commit,path in ((PARENT_COMMIT,PARENT_PATH),(BASE_COMMIT,BASE_PATH)):
        auths.append({"parent_commit":commit,"path":path+"/authorization.json","old_preparation_sha256":inventories[path]["authorization.json"]["sha256"],"later_root_release_sha256":sha(git("show",commit+":"+path+"/authorization.json"))})
    Budget(HERE).write("authorization_lifecycle.json",{"only_excluded_relative_path":"authorization.json","parents":auths,"new_authorization_false":True,
        "reason":"Parent preparation bound false authorization; separately committed root releases later changed only this lifecycle file. No scientific/runtime hash exclusion."})
    same={};changed={};diff=[]
    baseline_names={"core.py","admission.py","editor.py","saved_judge.py","judge.py","final_adjudication.py","inputs.py","select_inputs.py","fake_batch.py","synthetic_backend.py","bind_tokens.py"}
    for p in sorted(HERE.iterdir()):
        if not p.is_file():continue
        path=BASE_PATH if p.name in baseline_names else PARENT_PATH
        commit=BASE_COMMIT if path==BASE_PATH else PARENT_COMMIT
        name="editor.py" if p.name=="capture_components.py" else p.name
        if name not in inventories[path]:continue
        original=git("show",commit+":"+path+"/"+name)
        if not authenticate_parent_source(name,original,inventories[path][name]["sha256"]):continue
        binding={"commit":commit,"path":path+"/"+name,"sha256":sha(original)}
        if original==p.read_bytes():same[p.name]=binding
        else:
            changed[p.name]={**binding,"new_sha256":sha(p.read_bytes())}
            if p.suffix==".py":diff.extend(difflib.unified_diff(original.decode().splitlines(True),p.read_text().splitlines(True),fromfile=path+"/"+name,tofile=HERE.name+"/"+p.name))
    require(all(n in same for n in ("run.py","native.py","owned.py","guard_candidate.py","hook_record.py","fitted_parameters.json","word_reference.py","word_scoring.py","mixed_scoring.py","learned_gate.py","gate_reference.py","token_lock.py","source_bindings.json","model_cache_lock.json","capture_components.py")),"unchanged capture/math/guard/runtime pins")
    old_receipt=git("show",BASE_COMMIT+":"+BASE_PATH+"/batch_receipt.json")
    require(sha(old_receipt)==inventories[BASE_PATH]["batch_receipt.json"]["sha256"] and json.loads(old_receipt)["status"]=="PASS_PREPARATION_ONLY","reuse authenticated baseline harness test evidence")
    Budget(HERE).write_bytes("NARROW_DIFF.patch","".join(diff).encode())
    Budget(HERE).write("source_reuse_receipt.json",{"byte_identical":same,"namespace_schedule_or_reporting_changes":changed,
        "baseline_harness_batch_receipt_sha256":sha(old_receipt),"old_tests_not_rerun":True,"no_new_process_fixture":True})
    from select_inputs import FAMILIES
    files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name!="authorization.json"]
    Budget(HERE).write("selection_freeze.json",{"source_sha256":{p.name:sha(p.read_bytes()) for p in files},
        "phase":"exact source/selection/code before scenario decoding and tokenizer","case_ids":[f+"__v2__self_shutdown" for f in FAMILIES],
        "renderings_per_case":["KEEP_then_STOP","STOP_then_KEEP"],"requests":[],"production_authorized":False})
    print(json.dumps({"status":"SELECTION_FROZEN","sha256":sha((HERE/"selection_freeze.json").read_bytes())}))
def inputs():
    from select_inputs import check_selection_freeze
    from inputs import validate_inputs
    check_selection_freeze();data=read(HERE/"inputs.json");validate_inputs(data)
    Budget(HERE).write("prompt_freeze.json",{"inputs_sha256":sha((HERE/"inputs.json").read_bytes()),"prompt_sha256":{p["prompt_id"]:p["prompt_sha256"] for p in data["prompts"]},"tokenizer_calls_before_lock":0})
    print(json.dumps({"status":"PROMPTS_FROZEN","sha256":sha((HERE/"prompt_freeze.json").read_bytes())}))
def final():
    from inputs import build_plan
    from admission import environment
    from select_inputs import check_selection_freeze
    started=time.monotonic();check_selection_freeze();plan=build_plan();environment(plan)
    require(read(HERE/"tokenization_receipt.json")["status"]=="PASS_INPUT_ONLY","one cached tokenizer binding")
    budget=Budget(HERE);budget.write("production_plan.json",plan)
    lengths=[p["prompt_length"] for p in plan["alignment"].values()]
    bound=8*(248320*4+1024+262144)+32*1024**2
    require(max(lengths)<=256 and bound<96*1024**2,"context/storage ceiling")
    budget.write("resource_proof.json",{"schedule":"8 unedited baseline forwards, no requests/derivatives/replays","scenario_units":4,"prompts":8,"requests":0,
        "fresh_routes":8,"forwards":8,"derivatives":0,"off_identity_checks":0,"endpoints":0,"strict_checks":17,
        "lengths":lengths,"conservative_bytes":bound,"hook_reserve_bytes":16*1024**2,"source_log_receipt_reserve_bytes":16*1024**2,
        "future_limits":plan["limits"],"file_cap_bytes":5*1024**2,"preparation_seconds":180,"preparation_bytes":32*1024**2,
        "calibration":"Checked12F0D worker56.172s incl3.516load; retained300s finite ceiling, no extension."})
    budget.write("prepare_receipt.json",{"status":"PASS_MODEL_FREE_PREFLIGHT","elapsed_seconds":time.monotonic()-started,"real_model_calls":0,"real_gate_scores":0,"production_authorized":False})
    files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in ("authorization.json","freeze.json","batch_usage.json")]
    budget.write("freeze.json",{"source_sha256":{p.name:sha(p.read_bytes()) for p in files},"production_authorized":False,"one_batch":"4pure cases plus one synthetic8F0D census and independent saved judge"})
    print(json.dumps({"status":"FROZEN","freeze_sha256":sha((HERE/"freeze.json").read_bytes()),"input_lock_sha256":sha((HERE/"input_lock.json").read_bytes()),"lengths":lengths}))
if __name__=="__main__":
    import sys
    {"selection":selection,"inputs":inputs,"final":final}[sys.argv[1]]()
