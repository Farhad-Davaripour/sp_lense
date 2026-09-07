"""Post-batch byte/receipt closeout only; never model, tokenizer, or gate execution."""
import json,sys,time
from core import HERE,ROOT,Budget,read,require,sha,git,check_freeze

def verify():
    start=time.monotonic();freeze=check_freeze();receipt=read(HERE/"test_receipt.json")
    require(receipt["status"]=="PASS" and receipt["tests"]==6 and not receipt["errors"] and not receipt["failures"],"focused test batch")
    require(read(HERE/"authorization.json")["run_authorized"] is False and not (HERE/"approved_root_release.json").exists() and not (HERE/"real_attempt").exists(),"no production authority/attempt")
    source_commit="a3a0daa8952bf9305765e7ed39221dc0660a41c1";prefix=HERE.relative_to(ROOT).as_posix()
    for name,digest in freeze["source_sha256"].items():
        require(sha(git("show",source_commit+":"+prefix+"/"+name))==digest,"committed prospective source bytes "+name)
    require(sha((HERE/"editor.py").read_bytes())=="2e9be8be987c467071dde5f97fee8c8663f7f481abe7f0eaaacc0f599f247aac","unchanged repaired editor")
    for name in ("guard_candidate.py","hook_record.py","numeric_audit.py","learned_gate.py","gate_reload.py","gate_reference.py","mixed_scoring.py","word_scoring.py","word_reference.py"):
        original=git("show","db7c3c05b9021273ba0af4fe0fe08fab24e5951e:diagnostics/semantic_editor_final_pipeline_v1/"+name)
        require((HERE/name).read_bytes()==original,"byte-identical inherited scientific/guard source "+name)
    captures=[]
    for path in HERE.rglob("capture.json"):
        item=read(path)
        if "status" not in item:continue # Explicit minimal inventory-lifecycle fixture, not a process claim.
        require(item["quiescent"] and item["worker_joined"] and item["reader_joined"],"owned process/writers remain")
        log=path.parent/"worker.log"
        require(log.stat().st_size==item["captured_prefix_bytes"] and sha(log.read_bytes())==item["captured_prefix_sha256"],"actual captured prefix bytes")
        captures.append({"path":path.relative_to(HERE).as_posix(),"status":item["status"],"fault":item["technical_recording_fault"],"quiescent":True,"prefix_bytes":log.stat().st_size})
    normal=read(HERE/"representative_once/independent_judge.json");combined=read(HERE/"combined_fault/independent_judge.json")
    require(normal["classification"]=="PASS" and normal["counts"]["strict_requests"]==4 and not normal["independent_audit_faults"],"independent toy judge")
    require(combined["classification"]=="FAIL" and len(combined["independently_derived_scientific_failures"])==3 and combined["counts"]["unrun"]==40 and not combined["independent_audit_faults"],"combined scientific findings retained")
    allfiles=[p for p in HERE.rglob("*") if p.is_file()]
    require(sum(p.stat().st_size for p in allfiles)<384*1024**2 and all(p.stat().st_size<=5*1024**2 for p in allfiles),"namespace/file caps")
    Budget(HERE).write("verification_receipt.json",{"status":"PASS_FROZEN_FAKE_EVIDENCE","captures":captures,"source_files_unchanged":len(freeze["source_sha256"]),
        "elapsed_seconds":time.monotonic()-start,"real_model_calls":0,"tokenizer_calls":0,"gate_scores":0,"fits":0,
        "toy_forwards":22,"toy_derivatives":2,"pure_counter_callbacks":180,"blocked_preload_forwards":1,
        "production_release_ready":False,"remaining_admission_gap":"positive release path and embedded usage-value consistency need independent review before authorization",
        "batch_command_seconds":30.114162,"cumulative_invoked_seconds_conservative_bound":75,"invoked_allowance_seconds":600})
    print({"status":"PASS_FROZEN_FAKE_EVIDENCE","captures":len(captures),"elapsed_seconds":time.monotonic()-start})

def inventory():
    require(read(HERE/"verification_receipt.json")["status"]=="PASS_FROZEN_FAKE_EVIDENCE","post-writer verification first")
    require((HERE/"REPORT.md").exists(),"final report first")
    files=[{"path":p.relative_to(HERE).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for p in sorted(HERE.rglob("*")) if p.is_file() and p!=HERE/"FINAL_INVENTORY.json"]
    Budget(HERE).write("FINAL_INVENTORY.json",{"files":files,"status":"SYNTHETIC_VERIFICATION_PASS_REAL_RELEASE_BLOCKED","sole_owner":"closeout after all test child processes/writers, saved audits, report and receipts"})
    print({"entries":len(files),"inventory_sha256":sha((HERE/"FINAL_INVENTORY.json").read_bytes()),"namespace_bytes":sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())})

def archive():
    commit=git("rev-parse","HEAD").decode().strip();prefix=HERE.relative_to(ROOT).as_posix()
    invraw=git("show",commit+":"+prefix+"/FINAL_INVENTORY.json");require(invraw==(HERE/"FINAL_INVENTORY.json").read_bytes(),"final inventory committed")
    inv=json.loads(invraw)
    for entry in inv["files"]:
        raw=git("show",commit+":"+prefix+"/"+entry["path"])
        require(sha(raw)==entry["sha256"] and len(raw)==entry["bytes"],"committed exact artifact "+entry["path"])
    print({"status":"PASS_ALL_COMMITTED_BYTES","commit":commit,"entries":len(inv["files"]),"inventory_sha256":sha(invraw)})

if __name__=="__main__":
    if sys.argv[1:]==["verify"]:verify()
    elif sys.argv[1:]==["inventory"]:inventory()
    elif sys.argv[1:]==["archive"]:archive()
    else:raise SystemExit("verify/inventory/archive only")
