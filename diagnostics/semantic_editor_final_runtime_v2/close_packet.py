"""Pure preparation closeout; no production run or real-model evidence."""
import json,sys,time
from core import HERE,ROOT,read,sha,git,check_freeze,require,json_bytes
from prep import write,PREP_CAP,PARENT,PARENT_NS,PARENT_INVENTORY

def close(source_commit):
    start=time.monotonic();lock=check_freeze()
    require(git("show",source_commit+":"+HERE.relative_to(ROOT).as_posix()+"/freeze.json")== (HERE/"freeze.json").read_bytes(),"prospective source commit")
    test=read(HERE/"test_receipt.json");require(test["status"]=="PASS" and test["tests_run"]==5,"single pure batch PASS")
    require(all(test[k]==0 for k in ("model_loads","forwards","derivatives","tokenizer_calls","gate_scores","gate_fits","child_process_launches")),"zero model/gate/process work")
    require(read(HERE/"authorization.json")["run_authorized"] is False and not (HERE/"approved_root_release.json").exists() and not (HERE/"real_attempt").exists(),"actual production disabled")
    parent=read(HERE/"parent_source_receipt.json")
    for name,digest in parent["copied_exact_sha256"].items():require(sha((HERE/name).read_bytes())==digest,"preserved parent scientific/input bytes")
    usage=read(HERE/"batch_usage.json")
    from usage_receipt import derive_standard_usage
    derived=derive_standard_usage(usage["tool_receipt"])
    require(derived["used_percent"]<100,"known batch usage")
    report=("# Final runtime v2: pure boundary verification PASS\n\n"
        "The five focused pure tests passed. Embedded standard usage is derived from the full receipt; admitted positive fixtures exercised the actual validator/admission with only in-memory authorization stubs. Actual production remains disabled. External audit faults now override stale judge PASS, while raw FAIL/scientific/cleanup/UNRUN evidence stays intact. The saved judge reauthenticates external sources after exact-cohort admission and before numerical judging, without rechecking launch-receipt freshness.\n\n"
        "No model loads, forwards, derivatives, tokenizer calls, gate scores/fits or child-process launches occurred. Parent scientific/input bytes and 180F/48D, 1500+15+180s, 288MiB/5MiB, 109 strict-check envelope are unchanged; parent process/trajectory tests were authenticated and reused, not rerun. Pure test elapsed seconds: "+str(test["elapsed_seconds"])+". Batch standard usage: "+str(derived["used_percent"])+"%.\n\n"
        "Prospective source commit: "+source_commit+". Freeze SHA256: "+sha((HERE/"freeze.json").read_bytes())+". RELEASE_SCHEMA.md specifies the separate committed root pin, fresh receipt and exact launch command/attempt path. This is implementation evidence, not a final model result. Historical non-access remains unverified; no additional blind model replay is implemented beyond the fixed independent endpoint forwards and saved-data judge.\n\n"
        "Next single step: root independently verify this packet, then separately pin/release the already-fixed one-shot assessment if satisfied. No launch is authorized by this packet.\n")
    write("REPORT.md",report.encode(),raw=True)
    write("verification_receipt.json",{"status":"PASS_PURE_PACKET","source_commit":source_commit,"freeze_sha256":sha((HERE/"freeze.json").read_bytes()),
        "parent_commit":PARENT,"parent_inventory_sha256":PARENT_INVENTORY,"unchanged_copies":len(parent["copied_exact_sha256"]),"tests":test["tests_run"],
        "known_standard_usage":derived["used_percent"],"actual_production_enabled":False,"models":0,"forwards":0,"derivatives":0,"tokenizers":0,"gate_scores":0,"fits":0,
        "elapsed_seconds":time.monotonic()-start,"test_process_exit_observed_by_caller_before_closeout":True,"no_worker_or_audit_process_started":True})
    files=[{"path":p.relative_to(HERE).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for p in sorted(HERE.rglob("*")) if p.is_file() and p.name!="FINAL_INVENTORY.json"]
    require(sum(x["bytes"] for x in files)<PREP_CAP and max(x["bytes"] for x in files)<=5*1024**2,"preparation caps")
    write("FINAL_INVENTORY.json",{"files":files,"sole_owner":"close_packet after the single pure test command exited and all final report/receipts were written",
        "models":0,"forwards":0,"derivatives":0,"production_authorized":False})
    print({"status":"CLOSED_PURE_PACKET","files":len(files),"bytes":sum(x["bytes"] for x in files),"inventory_sha256":sha((HERE/"FINAL_INVENTORY.json").read_bytes())})
if __name__=="__main__":close(sys.argv[1])
