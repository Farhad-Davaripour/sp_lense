"""Read-only authentication of the closed real attempt; no execution/judge rerun."""
import hashlib,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2]
NS=ROOT/"diagnostics/fresh_confirmation_first_forward_helper_diagnostic_v3"
ATT=NS/"real_evidence/fresh_confirmation_first_forward_helper_diagnostic_attempt_002"
C=ATT/"control"
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def main():
    parent=read(C/"PARENT_FINAL.json");execution=parent["execution"]
    worker=read(C/"WORKER_RESULT.json");dt=read(C/"DIAGNOSTIC_TERMINAL.json")
    terminal=read(C/"ATTEMPT_TERMINAL.json");audit=read(C/"AUDIT_RESULT.json")
    joins={}
    for name in ("AUDIT_RESULT","WORKER_RESULT","CLOSED_WORKER_BINDING","LOADER_TERMINAL","DIAGNOSTIC_TERMINAL","ATTEMPT_TERMINAL","HELPER_OUTER_STATUS"):
        x=read(C/(name+".json"))
        need((x["diagnostic_result"]["execution"] if name=="WORKER_RESULT" else x["execution"])==execution,"EXECUTION:"+name)
        joins[name]=sha(C/(name+".json"))
    for lane in ("worker","audit"):
        need(parent[lane+"_result_sha256"]==sha(C/(lane.upper()+"_RESULT.json")),"RESULT_SHA")
    need(terminal["parent_final_sha256"]==sha(C/"PARENT_FINAL.json"),"PARENT_SHA")
    need(worker["diagnostic_result"]["diagnostic_terminal"]["sha256"]==sha(C/"DIAGNOSTIC_TERMINAL.json"),"NATIVE_SHA")
    need(dt["trace_status"]["receipt_sha256"]==sha(C/"FIRST_FORWARD_TRACE.json"),"TRACE_SHA")
    need(dt["index"]["sha256"]==sha(ATT/"evidence/attempt/closeout/index.json"),"INDEX_SHA")
    caps={}
    for lane in ("worker","audit"):
        p=C/"owned"/("production_"+lane)/"CAPTURE.json";cap=read(p)
        need(parent[lane+"_capture_sha256"]==sha(p),"CAPTURE_SHA")
        need(all(cap[k] is True for k in ("quiescent","actual_handle_closed","launcher_original_handle_closed","pipes_closed","threads_joined")),"QUIESCENCE")
        for v in cap["exit_proofs"].values():
            need(v["exit_code"]==0 and all(v[k] is True for k in ("signaled","query_success","valid_retained_handle")),"EXIT_PROOF")
        caps[lane]={"sha256":sha(p),"exit_proofs":cap["exit_proofs"],"quiescent":True}
    for field,path in (("authority_lock_sha256",NS/"root_release/AUTHORITY_LOCK.json"),("release_sha256",NS/"root_release/ROOT_RELEASE.json"),
      ("authorization_sha256",NS/"root_release/AUTHORIZATION.json"),("source_sha256",NS/"SOURCE_FREEZE.json"),("admission_sha256",C/"ATTEMPT_ADMISSION.json")):
        need(execution[field]==sha(path),"AUTHORITY:"+field)
    # Native loader acknowledgment and exact saved copy, not a new weight hash.
    ld=dt["loader_diagnostics"]
    old=read(NS/"FINAL_INVENTORY.json")
    for f in old["files"]:need((NS/f["path"]).stat().st_size==f["bytes"] and sha(NS/f["path"])==f["sha256"],"OLD_BYTES")
    need(sha(NS/"FINAL_INVENTORY.json")=="f1dd1dc9e67ad12e40ab7170ef0ef0107610968dea468f6c86e018ae7e527549","OLD_INVENTORY")
    print(json.dumps({"status":"SAVED_JOINS_VERIFIED","parent_sha256":sha(C/"PARENT_FINAL.json"),
      "trace_sha256":sha(C/"FIRST_FORWARD_TRACE.json"),"joins":joins,"captures":caps,"counts":audit["counts"],
      "state":dt["state"],"loader_diagnostic_keys":list(ld),"elapsed":parent["elapsed_seconds"],"cleanup":parent["shared_cleanup"]["used_seconds"],
      "classification":parent["classification"],"guard_restored":audit["guard_restored"],"normal_hook_checks_unrun":dt["normal_hook_checks_unrun"],
      "old_inventory_entries_unchanged":len(old["files"]),"real_files":len(list(ATT.rglob("*.*")))}))
if __name__=="__main__":main()
