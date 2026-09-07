"""Model-free preparation; never loads/tokenizes or re-extracts any dataset."""
import json,sys,time
from pathlib import Path
from core import HERE,ROOT,Budget,read,require,sha,git,check_freeze
from admission import INPUT_COMMIT,INPUT_NS,INPUT_INVENTORY,INPUT_LOCK,exact_cohort,environment

PIPE_COMMIT="db7c3c05b9021273ba0af4fe0fe08fab24e5951e"
PIPE_NS="diagnostics/semantic_editor_final_pipeline_v1"
PIPE_INVENTORY="cf9ac8da94ee7fbe8254022f9b3ded557be182b61b5004446c21ff7e77cf7548"
PACKET_COMMIT="a2850dfc024575685edd9d1b29461d9c9c0d77cb"
PACKET_NS="diagnostics/semantic_editor_final_study_plan_v1"
def blob(commit,path):return git("show",commit+":"+path)
def authenticate(commit,namespace,inventory_sha,names):
    invraw=blob(commit,namespace+"/FINAL_INVENTORY.json");require(sha(invraw)==inventory_sha,"parent inventory")
    inv={r["path"]:r for r in json.loads(invraw)["files"]};result={}
    for name in names:
        raw=blob(commit,namespace+"/"+name);require(sha(raw)==inv[name]["sha256"] and len(raw)==inv[name]["bytes"],"parent artifact "+name);result[name]=raw
    return result

def setup():
    budget=Budget(HERE)
    source=authenticate(INPUT_COMMIT,INPUT_NS,INPUT_INVENTORY,["inputs.json","input_lock.json","source_bindings.json"])
    require(sha(source["input_lock.json"])==INPUT_LOCK,"exact input lock")
    lock=json.loads(source["input_lock.json"]);names=[p["tokens"]["path"] for p in lock["prompts"]]
    tokenraw=authenticate(INPUT_COMMIT,INPUT_NS,INPUT_INVENTORY,names)
    budget.write_bytes("locked_inputs.json",source["inputs.json"])
    budget.write("locked_boundaries.json",{p["prompt_id"]:json.loads(tokenraw[p["tokens"]["path"]]) for p in lock["prompts"]})
    budget.write("input_source_receipt.json",{"commit":INPUT_COMMIT,"namespace":INPUT_NS,"inventory_sha256":INPUT_INVENTORY,"input_lock_sha256":INPUT_LOCK,
        "input_sha256":sha(source["inputs.json"]),"token_artifacts":{n:sha(r) for n,r in tokenraw.items()},"no_new_tokenizer_or_source_extraction":True})
    old=authenticate(PIPE_COMMIT,PIPE_NS,PIPE_INVENTORY,["source_bindings.json","fitted_parameters.json"])
    for n,raw in old.items():budget.write_bytes(n,raw)
    schedule=blob(PACKET_COMMIT,PACKET_NS+"/plan.json")
    require(sha(schedule)=="53d729e054474bc2949357c2d7afd6d93650ec49fb61d166e5cbb3dea358d7f9","exact original schedule")
    budget.write_bytes("cohort_schedule.json",schedule)
    inputsources=json.loads(source["source_bindings.json"]);external={str(ROOT/p):h for p,h in inputsources["dependencies_sha256"].items()}
    external.update({str(Path(inputsources["snapshot"])/name):m["sha256"] for name,m in inputsources["cache_files"].items()})
    for path in ("scripts/three_family_bounded_capture.py","src/sp_lense/conditional_gate_models.py",PACKET_NS+"/schedule_check.py",
                 "scripts/verify_local_controllability.py","scripts/verify_margin_aware_local_control.py"):
        external[str(ROOT/path)]=sha((ROOT/path).read_bytes())
    budget.write("runtime_source_receipt.json",{"external_sha256":external,"repair_commit":"49af3d4a9538fc336b83a1cce7b26bfc7e8d7233",
        "repair_inventory_sha256":"d36f5c414eefa7ca68b1248b533da05dc346fbe5ad344c332f6f1d3f9c5a34e7",
        "editor_sha256":"2e9be8be987c467071dde5f97fee8c8663f7f481abe7f0eaaacc0f599f247aac",
        "pipeline_commit":PIPE_COMMIT,"pipeline_inventory_sha256":PIPE_INVENTORY})
    from inputs import build_plan
    plan=build_plan();budget.write("production_plan.json",plan)
    budget.write("resource_proof.json",{"real_envelope":{"forwards":180,"derivatives":48,"loads":1,"worker_seconds":1500,"cleanup_seconds":15,"audit_seconds":180,"bytes":288*1024**2,"file_bytes":5*1024**2},
        "synthetic_preparation_envelope":{"invoked_seconds":600,"namespace_bytes":384*1024**2},
        "maximum_locked_length":max(b["prompt_length"] for b in plan["alignment"].values()),
        "record_bound_formula":"180*(248320*4+1024+262144)+16MiB hooks+16MiB source/receipts/two4MiB logs/closeout",
        "conservative_record_bytes":259715072,"full_vocab_record_bytes":993280,"row_bookkeeping_bound_bytes":262144,
        "input_masks_and_six_final_vectors_bytes":136*16+6*1024*4,
        "strict_hook_checks":109,"hook_bytes":16*1024**2,"closed_form":"24+48+12*(8+1)=180F;12*4=48D",
        "derived_only":"132F24D requires eligible stable cold entries; never used as safety cap",
        "source_and_receipt_reserve":"16MiB includes two4MiB bounded process logs, <=4MiB copied sources/locks and >=4MiB closeout; all writers share namespace/per-attempt limits",
        "calibration":"Inherited measured 42F3D128.641s and40F16D194.516s; proposal remains1500s, not fake-test throughput or an extension."})

def freeze():
    from inputs import build_plan
    require(build_plan()==read(HERE/"production_plan.json"),"deterministic exact input assembly")
    files={p.name:sha(p.read_bytes()) for p in HERE.iterdir() if p.is_file() and p.name not in ("freeze.json","authorization.json")}
    Budget(HERE).write("freeze.json",{"source_sha256":files,"authorization_policy_separate":True,
        "release_policy":"authorization.json is a separate root-reviewed committed release pin, not scientific configuration. Currently disabled. Release document binds this earlier source/input freeze; avoiding a circular commit/hash dependency.",
        "real_model_authorized":False,"batch_scope":"fake processes/backends only","prior_repair_source_unchanged":True})
    print({"freeze_sha256":sha((HERE/"freeze.json").read_bytes()),"source_files":len(files),"model_authorized":False})

def preflight():
    check_freeze();plan=read(HERE/"production_plan.json");counts=exact_cohort(plan);environment(plan)
    require(sha((HERE/"editor.py").read_bytes())=="2e9be8be987c467071dde5f97fee8c8663f7f481abe7f0eaaacc0f599f247aac","unchanged repaired editor")
    require(max(b["prompt_length"] for b in plan["alignment"].values())==136,"locked max length")
    require(sha((HERE/"fitted_parameters.json").read_bytes())==plan["gate"]["parameter_sha256"],"gate artifact unchanged, not scored")
    return {"status":"PASS_MODEL_FREE_PREFLIGHT",**{k:v for k,v in counts.items() if k not in ("forwards","derivatives")},
        "maximum_forwards":180,"maximum_derivatives":48,"actual_forwards":0,"actual_derivatives":0,
        "tokenizer_calls":0,"model_loads":0,"gate_scores":0,"fits":0,"production_authorized":read(HERE/"authorization.json")["run_authorized"]}

if __name__=="__main__":
    if sys.argv[1:]==["setup"]:setup()
    elif sys.argv[1:]==["freeze"]:freeze()
    elif sys.argv[1:]==["preflight"]:print(preflight())
    else:raise SystemExit("setup/freeze/preflight only")
