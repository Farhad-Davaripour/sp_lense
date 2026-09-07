"""Bounded32MiB source-successor preparation; no model/tokenizer/data extraction."""
import difflib,json,sys,time
from core import HERE,ROOT,read,require,sha,git,json_bytes,check_freeze
PARENT="b184f36461d84d7e0fb090db0789825551942705"
PARENT_NS="diagnostics/semantic_editor_final_runtime_v1"
PARENT_INVENTORY="a5ada41ab99e069a93ab5d817b0527769217282d0cc5362da1f99919efa7b8af"
PREP_CAP=32*1024**2
def write(name,value,raw=False):
    data=value if raw else json_bytes(value)
    require(len(data)<=5*1024**2,"preparation file cap")
    require(sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())+len(data)<=PREP_CAP,"preparation32MiB cap")
    with (HERE/name).open("xb") as out:out.write(data)
def blob(name):return git("show",PARENT+":"+PARENT_NS+"/"+name)
def setup():
    invraw=blob("FINAL_INVENTORY.json");require(sha(invraw)==PARENT_INVENTORY,"reviewed v1 inventory")
    inv={x["path"]:x for x in json.loads(invraw)["files"]}
    data=("cohort_schedule.json","fitted_parameters.json","input_source_receipt.json","locked_boundaries.json","locked_inputs.json",
        "production_plan.json","resource_proof.json","runtime_source_receipt.json","source_bindings.json")
    copied={}
    for name in data:
        raw=blob(name);require(sha(raw)==inv[name]["sha256"] and len(raw)==inv[name]["bytes"],"exact parent data "+name)
        write(name,raw,raw=True);copied[name]=sha(raw)
    unchanged=("editor.py","gate_reference.py","gate_reload.py","guard_candidate.py","hook_record.py","inputs.py","learned_gate.py","locked_backend.py",
        "mixed_boundary.py","mixed_scoring.py","numeric_audit.py","saved_judge.py","word_reference.py","word_scoring.py")
    for name in unchanged:
        require((HERE/name).read_bytes()==blob(name),"unchanged scientific/input source "+name);copied[name]=sha(blob(name))
    patches=[]
    for name in ("admission.py","judge.py","run.py","core.py"):
        old=blob(name).decode().splitlines(keepends=True);new=(HERE/name).read_text().splitlines(keepends=True)
        patches.extend(difflib.unified_diff(old,new,fromfile="v1/"+name,tofile="v2/"+name))
    write("minimal.patch","".join(patches).encode(),raw=True)
    evidence={name:{"sha256":inv[name]["sha256"],"value":json.loads(blob(name))} for name in ("test_receipt.json","verification_receipt.json")}
    write("parent_source_receipt.json",{"parent_commit":PARENT,"parent_inventory_sha256":PARENT_INVENTORY,"copied_exact_sha256":copied,
        "reused_parent_evidence":evidence,"no_parent_model_test_replay":True,"preparation_cap_bytes":PREP_CAP,
        "future_attempt_envelope_unchanged":"180F48D;1500+15+180s;288MiB/5MiB;109strictchecks/16MiBhooks. Parent core namespace safety cap is not this job's preparation allowance."})
def freeze():
    sources={p.name:sha(p.read_bytes()) for p in HERE.iterdir() if p.is_file() and p.name not in ("freeze.json","authorization.json")}
    write("freeze.json",{"source_sha256":sources,"preparation_only":True,"model_authorized":False,
        "authorization_policy_separate":"unchanged disabled root pin; positive tests use in-memory fixtures only",
        "changes":"usage receipt derivation; authoritative external-capture final join; audit-time helper/runtime reauthentication; namespace/provenance only",
        "limits":{"invoked_seconds":180,"preparation_bytes":PREP_CAP,"file_bytes":5*1024**2}})
    print({"status":"FROZEN_PURE_VALIDATOR_BATCH","files":len(sources),"freeze_sha256":sha((HERE/"freeze.json").read_bytes())})
def preflight():
    check_freeze()
    from admission import exact_cohort,environment,admit_production
    plan=read(HERE/"production_plan.json");counts=exact_cohort(plan);environment(plan)
    require(read(HERE/"authorization.json")["run_authorized"] is False and not (HERE/"approved_root_release.json").exists() and not (HERE/"real_attempt").exists(),"production remains disabled")
    try:admit_production()
    except ValueError as error:require("NO REAL MODEL RELEASE" in str(error),"expected disabled rejection")
    else:raise AssertionError("actual production admission enabled")
    print({"status":"PASS_PURE_PREFLIGHT","maximum_forwards":counts["forwards"],"maximum_derivatives":counts["derivatives"],
        "actual_forwards":0,"actual_derivatives":0,"model_loads":0,"tokenizer_calls":0,"gate_scores":0,"fits":0,"production_authorized":False})
if __name__=="__main__":
    if sys.argv[1:]==["setup"]:setup()
    elif sys.argv[1:]==["freeze"]:freeze()
    elif sys.argv[1:]==["preflight"]:preflight()
    else:raise SystemExit("setup/freeze/preflight only")
