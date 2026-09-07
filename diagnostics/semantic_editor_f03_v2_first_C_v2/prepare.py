"""Three prospective locks; no model, outcome or gate-feature reads."""
import ast,json,time,difflib
from core import HERE,ROOT,Budget,read,require,sha,git
PARENT_COMMIT="108494b88458dc20d4f4a8cb4a42640c1569d9fc"
PARENT_PATH="diagnostics/semantic_editor_f03_v2_first_C_v1"
PARENT_INVENTORY="f3ee0e483b6c84ea920e6eb697c6777f2f58269f2c2e1dd82761cdb7d3d54704"

def authenticate_parent_source(name,original,expected_sha):
    if name=="authorization.json":return False
    require(sha(original)==expected_sha,"parent source bytes "+name)
    return True

def selection():
    require(not (HERE/"inputs.json").exists(),"freeze before selective decoding")
    for p in HERE.glob("*.py"):ast.parse(p.read_text(),filename=p.name)
    inventory_raw=git("show",PARENT_COMMIT+":"+PARENT_PATH+"/FINAL_INVENTORY.json")
    require(sha(inventory_raw)==PARENT_INVENTORY,"authenticated parent source inventory")
    inventory={r["path"]:r for r in json.loads(inventory_raw)["files"]}
    require(read(HERE/"authorization.json")["run_authorized"] is False and read(HERE/"authorization.json")["root_release_sha256"] is None,"new independent production authorization remains false")
    original_auth=git("show",PARENT_COMMIT+":"+PARENT_PATH+"/authorization.json")
    Budget(HERE).write("authorization_lifecycle.json",{"only_excluded_path":"authorization.json","parent_inventory_sha256":inventory["authorization.json"]["sha256"],
        "later_parent_committed_sha256":sha(original_auth),"parent_commit":PARENT_COMMIT,"new_authorization_false":True,
        "reason":"Earlier parent preparation inventory bound disabled authorization; later root release legitimately changed this lifecycle file. No scientific/runtime source exclusion.",
        "prior_authorization_bug":"f05V1 bb960f3 remains immutable INCONCLUSIVE; checked f05V2 lifecycle exclusion reused"})
    source={}
    changed={}
    differences=[]
    for p in sorted(HERE.iterdir()):
        if p.name not in inventory or not p.is_file():continue
        original=git("show",PARENT_COMMIT+":"+PARENT_PATH+"/"+p.name)
        if not authenticate_parent_source(p.name,original,inventory[p.name]["sha256"]):continue
        if original==p.read_bytes():source[p.name]=sha(original)
        else:
            changed[p.name]={"parent_sha256":sha(original),"new_sha256":sha(p.read_bytes())}
            if p.suffix==".py":differences.extend(difflib.unified_diff(original.decode().splitlines(True),p.read_text().splitlines(True),fromfile=PARENT_PATH+"/"+p.name,tofile=HERE.name+"/"+p.name))
    require(all(n in source for n in ("run.py","editor.py","native.py","owned.py","guard_candidate.py","hook_record.py","numeric_audit.py","fitted_parameters.json","word_reference.py","word_scoring.py","learned_gate.py")),"scientific and owned core remain byte-identical")
    # Reuse previously authenticated source-parity receipt; no numerical audit rerun.
    require("math_parity.json" in source,"unchanged previously checked science parity receipt")
    prior=git("show",PARENT_COMMIT+":"+PARENT_PATH+"/batch_receipt.json")
    require(sha(prior)==inventory["batch_receipt.json"]["sha256"] and json.loads(prior)["status"]=="INCONCLUSIVE","preserve failed V1 receipt; never claim it passed")
    Budget(HERE).write("prior_failure_binding.json",{"commit":PARENT_COMMIT,"inventory_sha256":PARENT_INVENTORY,"receipt_sha256":sha(prior),
        "status":"INCONCLUSIVE","scientific_result":"NOT_RUN","partial_synthetic_outputs_copied":False,"unchanged_math_tests_reused_from_v1_bound_prior_engine":True})
    Budget(HERE).write_bytes("NARROW_DIFF.patch","".join(differences).encode())
    Budget(HERE).write("source_reuse_receipt.json",{"parent_commit":PARENT_COMMIT,"parent_inventory_sha256":PARENT_INVENTORY,"byte_identical":source,"namespace_or_failstop_changes":changed,"unchanged_prior_tests_not_rerun":True})
    files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name!="authorization.json"]
    Budget(HERE).write("selection_freeze.json",{"source_sha256":{p.name:sha(p.read_bytes()) for p in files},
        "phase":"exact selected census source/method/code before committed prompt/token copy; no dataset or tokenizer","case_ids":["cg_f03_context_rotation__v2__self_shutdown"],
        "renderings":["STOP_then_KEEP"],"requests":["C/STOP first only"],"production_authorized":False})
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
    started=time.monotonic();check_selection_freeze()
    from inputs import PID,PROMPT_SHA,TOKEN_SHA
    source=read(HERE/"selection_receipt.json");require(source["status"]=="PASS_SOURCE_ONLY" and source["tokenizer_calls"]==0,"authenticated no-tokenizer copy")
    commit=git("rev-parse","HEAD").decode().strip();raw=(HERE/"inputs.json").read_bytes()
    require(raw==git("show",commit+":"+HERE.relative_to(ROOT).as_posix()+"/inputs.json"),"committed exact copied prompt before final lock")
    token=(HERE/"tokens_01.json").read_bytes();require(sha(token)==TOKEN_SHA and token==git("show",commit+":"+HERE.relative_to(ROOT).as_posix()+"/tokens_01.json"),"committed copied complete token bytes")
    budget=Budget(HERE);budget.write("input_lock.json",{"input_commit":commit,"inputs_sha256":sha(raw),"prompts":[{"prompt_id":PID,"prompt_sha256":PROMPT_SHA,"path":"tokens_01.json","sha256":TOKEN_SHA,"length":137}],
        "tokenizer_loads":0,"model_calls":0,"gate_scores":0,"source_commit":source["source_commit"],"source_input_copy_exact":True})
    plan=build_plan();environment(plan);budget.write("production_plan.json",plan)
    lengths=[p["prompt_length"] for p in plan["alignment"].values()]
    bound=11*(248320*4+1024+262144)+32*1024**2
    require(lengths==[137] and bound<96*1024**2,"copied context/storage envelope")
    budget.write("resource_proof.json",{"schedule":"1 baseline+1 conditional fresh entry+8 gradient/update forwards+1 endpoint=11F4D",
        "scenario_units":1,"prompts":1,"requests":1,"fresh_routes_max":2,"endpoints_max":1,"off_checks":0,"strict_checks_max":4,"strict_hook_cap_inherited":13,
        "lengths":lengths,"conservative_bytes":bound,"hook_reserve_bytes":16*1024**2,"source_log_receipt_reserve_bytes":16*1024**2,
        "future_limits":plan["limits"],"file_cap_bytes":5*1024**2,"preparation_seconds":180,"preparation_bytes":32*1024**2,
        "calibration":"Authenticated16F3D worker82.078s;300s retained finite headroom, no extension."})
    budget.write("prepare_receipt.json",{"status":"PASS_MODEL_FREE_PREFLIGHT","elapsed_seconds":time.monotonic()-started,"model_calls":0,"real_gate_scores":0,"tokenizer_calls":0,"production_authorized":False})
    files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in ("authorization.json","freeze.json","batch_usage.json")]
    budget.write("freeze.json",{"source_sha256":{p.name:sha(p.read_bytes()) for p in files},"production_authorized":False,
        "one_batch":"4pure cases (original3 plus naming) and one in-process5F1D one-request synthetic workflow and independent saved judge"})
    print(json.dumps({"status":"FROZEN","freeze_sha256":sha((HERE/"freeze.json").read_bytes()),"input_lock_sha256":sha((HERE/"input_lock.json").read_bytes()),"lengths":lengths}))

if __name__=="__main__":
    import sys
    {"selection":selection,"inputs":inputs,"final":final}[sys.argv[1]]()
