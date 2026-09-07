"""One frozen batch: new verdict cases and one launcher-alive fake child."""
import copy,hashlib,json,os,queue,secrets,subprocess,sys,threading,time
from pathlib import Path
from owned import OwnedPair,strict_json,require
from verdict import live_verdict,batch_verdict
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=ROOT/"diagnostics/windows_owned_worker_supervision_v1"

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def budget(extra=0):
    files=[p for p in HERE.iterdir() if p.is_file()]
    require(all(p.stat().st_size<=1024**2 for p in files),"file cap")
    require(sum(p.stat().st_size for p in files)+extra<=8*1024**2,"namespace cap")
def write(name,value,append=False):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+"\n").encode()
    old=(HERE/name).stat().st_size if (HERE/name).exists() else 0
    require(old+len(raw)<=1024**2,"bounded evidence file");budget(len(raw))
    with (HERE/name).open("ab" if append else "xb") as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())

def authenticate():
    plan=read(HERE/"plan.json");freeze=read(HERE/"freeze.json")
    for name,digest in freeze["files"].items():require(sha((HERE/name).read_bytes())==digest,"frozen "+name)
    for name,digest in plan["source_sha256"].items():require(sha(Path(name).read_bytes())==digest,"external pin "+name)
    inventory=read(PARENT/"FINAL_INVENTORY.json")
    require(sha((PARENT/"FINAL_INVENTORY.json").read_bytes())==plan["parent_inventory_sha256"],"parent inventory")
    entries={x["path"]:x for x in inventory["files"]}
    selected=["owned.py","native.py","tiny_worker.py","test_batch.py","pure_tests.json","ADJUDICATION.json"]
    for name in selected:
        raw=(PARENT/name).read_bytes();require(len(raw)==entries[name]["bytes"] and sha(raw)==entries[name]["sha256"],"parent "+name)
    for name in plan["unchanged_helpers"]:require((HERE/name).read_bytes()==(PARENT/name).read_bytes(),"byte-identical "+name)
    rows=read(PARENT/"pure_tests.json");require(len(rows)==12 and all(x["status"]=="PASS" for x in rows),"12 inherited pure results")
    require(read(PARENT/"ADJUDICATION.json")["classification"]=="FAIL_FAKE_VERIFICATION_NOT_APPROVED_FOR_PRODUCTION","preserve failed parent")
    require(os.path.normcase(sys.executable)==os.path.normcase(plan["identity"]["launch_image"]),"pinned venv driver")
    budget(1024**2)
    return plan,{"reused_pure_count":12,"parent_commit":plan["parent_commit"],"parent_inventory_sha256":plan["parent_inventory_sha256"],
                 "authenticated_selected_files":selected,"unchanged_helpers":plan["unchanged_helpers"],"prior_suite_rerun":False}

def good_receipt():
    return {"assertions_passed":True,"primary_exception":None,"final_cleanup_errors":[],"cleanup_faults":[],"reader_faults":[],"cap_errors":[],
        "authenticated_before_permission":True,"actual_handle_retained_from_authentication":True,
        "before_cleanup":{"launcher_exited":False,"actual_worker_exited":False,"quiescent":False},
        "actual_retained_termination_events":1,"worker_exit_code":125,"launcher_exit_code":125,"all_threads_joined":True,
        "pipes_closed":True,"stdout_EOF":True,"stderr_EOF":True,"quiescent":True,"evidence_cap_ok":True,
        "faults":[{"kind":"deadline","detail":"expected synthetic deadline"}],"elapsed_seconds":1.2}

def new_pure_tests():
    rows=[];passing=[{"status":"PASS"}]*9
    for name in ("valid","late_exit_assertion","late_cleanup","stale_nested_pass_error","reader_fault","cap_fault","incomplete_exit","missing_receipt","incomplete_io"):
        value=good_receipt();checked=True
        if name=="late_exit_assertion":value.update(status="PASS_EXPECTED_FAKE_LIFECYCLE_FAULTS",worker_exit_code=0,primary_exception="ValueError: late exit-code assertion")
        if name=="late_cleanup":value["final_cleanup_errors"]=["late retained handle close failed"]
        if name=="stale_nested_pass_error":value.update(status="PASS_EXPECTED_DEADLINE_CLEANUP",error="preserved late nested error")
        if name=="reader_fault":value["reader_faults"]=["bounded reader failed"]
        if name=="cap_fault":value["cap_errors"]=["write cap exceeded"]
        if name=="incomplete_exit":value.update(worker_exit_code=None,quiescent=False)
        if name=="missing_receipt":checked=False
        if name=="incomplete_io":value.update(all_threads_joined=False,pipes_closed=False,stdout_EOF=False)
        derived=live_verdict(value);batch=batch_verdict(value,passing,12,checked)
        require((batch["status"]=="PASS_FAKE_ONLY")==(name=="valid"),"fail-closed batch "+name)
        require((derived["status"]=="PASS_EXPECTED_DEADLINE_CLEANUP")==(name in ("valid","missing_receipt")),"fail-closed live "+name)
        rows.append({"fixture":name,"status":"PASS","derived_live":derived,"derived_batch":batch})
    write("new_pure_tests.json",rows);return rows

class Pump:
    def __init__(self,stream,label):
        self.stream=stream;self.label=label;self.frames=queue.Queue(maxsize=8);self.eof=threading.Event();self.error=None;self.total=0
        self.thread=threading.Thread(target=self.run,name=label,daemon=True);self.thread.start()
    def run(self):
        try:
            while True:
                line=self.stream.readline(8193)
                if not line:self.eof.set();break
                self.total+=len(line);require(len(line)<=8192 and self.total<=16384,"bounded private pipe "+self.label)
                self.frames.put_nowait(line)
        except BaseException as error:self.error=type(error).__name__+": "+str(error)
    def frame(self,timeout):
        raw=self.frames.get(timeout=timeout);return strict_json(raw)


def live_test(plan):
    from native import NativeHandle,open_claimed_worker
    start=time.monotonic();proc=None;pair=None;worker=None;launcher=None;pumps=[];events=[]
    result={"assertions_passed":False,"primary_exception":None,"final_cleanup_errors":[],"cap_errors":[],
            "model_calls":0,"tokenizers":0,"live_scenarios":0,"actual_handle_retained_from_authentication":False}
    def event(kind,data):
        row={"kind":kind,"data":data,"monotonic":time.monotonic()}
        if len(events)<64:events.append(row)
        else:result["cap_errors"].append("event count cap")
        try:write("lifecycle_events.jsonl",row,True)
        except BaseException as error:
            result["cap_errors"].append(type(error).__name__+": "+str(error))
            if kind=="authenticated_before_permission":raise
    def safe(label,call):
        try:return call()
        except BaseException as error:
            result["final_cleanup_errors"].append(label+": "+type(error).__name__+": "+str(error))
            return None
    try:
        command=[sys.executable,"-B",str(HERE/"tiny_worker.py")]
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
        result.update(live_scenarios=1,command=command,owned_launcher_pid=proc.pid)
        launcher=NativeHandle(int(proc._handle),"owned_launcher_original",True)
        stdout=Pump(proc.stdout,"stdout");stderr=Pump(proc.stderr,"stderr");pumps=[stdout,stderr]
        nonce=secrets.token_hex(32);proc.stdin.write(json.dumps({"nonce":nonce}).encode()+b"\n");proc.stdin.flush()
        claim=stdout.frame(3.)
        require(claim.get("kind")=="claim" and claim.get("nonce")==nonce and type(claim.get("pid")) is int and claim["pid"]>0,"private claim before OpenProcess")
        worker=open_claimed_worker(claim["pid"]);pair=OwnedPair(launcher,worker,event);claims=[claim]
        while not stdout.frames.empty():claims.append(stdout.frame(.1))
        def ack(frame):proc.stdin.write(json.dumps(frame).encode()+b"\n");proc.stdin.flush()
        pair.authorize(claims,plan["identity"],nonce,os.getpid(),ack)
        result.update(binding=pair.binding,authenticated_before_permission=pair.authenticated and pair.acknowledged,
                      actual_handle_retained_from_authentication=True,secondary_pid_lookup_for_cleanup=False)
        require(stdout.frame(2.)=={"kind":"ready_to_close_outputs","nonce":nonce},"READY")
        deadline=time.monotonic()+1.;threading.Event().wait(max(0.,deadline-time.monotonic()))
        before={"launcher_exited":launcher.exited(),"actual_worker_exited":worker.exited(),"stdout_EOF":stdout.eof.is_set(),
                "stderr_EOF":stderr.eof.is_set(),"quiescent":pair.quiescent(False,False),"deadline":deadline}
        result["before_cleanup"]=before;event("before_deadline_cleanup",before)
        require(not before["launcher_exited"] and not before["actual_worker_exited"] and not before["quiescent"],"both live at deadline, no false quiescence")
        require(pair.observe(deadline)=="STOPPED","unchanged actual-worker-first deadline cleanup")
        require(sum(x["kind"]=="terminate_actual_retained_handle" for x in events)==1,"exactly one retained actual termination event")
        require(worker.exit_code()==125,"actual exit125 not watchdog")
        require(worker.exited() and launcher.exited() and not pair.cleanup_faults,"both process exits")
        result["assertions_passed"]=True
    except BaseException as error:result["primary_exception"]=type(error).__name__+": "+str(error)
    finally:
        if pair is not None:
            if not (safe("worker state",worker.exited) and safe("launcher state",launcher.exited)):
                safe("final pair cleanup",lambda:pair.stop_and_wait("final_cleanup",timeout=1.))
        elif launcher is not None and not safe("launcher state",launcher.exited):
            safe("owned launcher cleanup",launcher.terminate);safe("owned launcher wait",lambda:launcher.wait(1.))
        if proc is not None:
            safe("launcher Popen wait",lambda:proc.wait(timeout=2.))
        # After both exits duplicate launcher pipes must reach EOF; not before.
        for pump in pumps:safe("reader join",lambda p=pump:p.thread.join(2.))
        joined=bool(pumps) and all(not p.thread.is_alive() for p in pumps)
        pipes_closed=False
        if proc is not None:
            for pipe in (proc.stdin,proc.stdout,proc.stderr):safe("pipe close",pipe.close)
            pipes_closed=all(p.closed for p in (proc.stdin,proc.stdout,proc.stderr))
        result.update(reader_faults=[p.error for p in pumps if p.error],all_threads_joined=joined,pipes_closed=pipes_closed,
                      stdout_EOF=bool(pumps) and pumps[0].eof.is_set(),stderr_EOF=len(pumps)==2 and pumps[1].eof.is_set(),
                      faults=[] if pair is None else pair.faults,cleanup_faults=[] if pair is None else pair.cleanup_faults,
                      quiescent=False if pair is None else safe("final quiescence",lambda:pair.quiescent(joined,pipes_closed)),
                      worker_exit_code=None if worker is None else safe("actual exitcode",worker.exit_code),
                      launcher_exit_code=None if launcher is None else safe("launcher exitcode",launcher.exit_code),
                      actual_retained_termination_events=sum(x["kind"]=="terminate_actual_retained_handle" for x in events))
        if worker is not None:safe("retained handle close",worker.close)
        try:budget(16384);result["evidence_cap_ok"]=True
        except BaseException as error:result["cap_errors"].append(str(error));result["evidence_cap_ok"]=False
        result["elapsed_seconds"]=time.monotonic()-start
        result["verdict"]=live_verdict(result);result["status"]=result["verdict"]["status"]
        write("live_result.json",result)
    return result

def main():
    start=time.monotonic();pure=[];live={};reused=0;checked=False;error=None
    try:
        require(not (HERE/"RUN_STARTED.json").exists(),"single batch, no retry")
        plan,reuse=authenticate()
        usage=read(HERE/"batch_usage.json")
        require(type(usage["known_standard_used_percent"]) in (int,float) and 0<=usage["known_standard_used_percent"]<100,"known available usage")
        write("RUN_STARTED.json",{"scope":plan["scope"],"monotonic":start,"one_batch_only":True})
        write("reuse_receipt.json",reuse);reused=reuse["reused_pure_count"]
        pure=new_pure_tests();live=live_test(plan)
        saved=read(HERE/"live_result.json");require(saved==live,"closed saved live receipt roundtrip")
        require(not any(x in sys.modules for x in ("torch","transformers","transformer_lens")),"no model imports")
        checked=True;budget(16384)
    except BaseException as exc:error=type(exc).__name__+": "+str(exc)
    verdict=batch_verdict(live,pure,reused,checked,error)
    receipt={**verdict,"error":error,"elapsed_seconds":time.monotonic()-start,"pure_fixtures_new":len(pure),
             "pure_fixtures_reused":reused,"live_scenarios":live.get("live_scenarios",0),"live_receipt_sha256":sha((HERE/"live_result.json").read_bytes()) if checked else None,
             "receipt_checked_after_writer_close":checked,"model_calls":0,"tokenizers":0,"gate_scores_or_fits":0,"new_data_reads":0}
    write("batch_result.json",receipt)
    require(read(HERE/"batch_result.json")==receipt,"final batch receipt roundtrip")
    print(json.dumps(receipt))
    return 0 if receipt["status"]=="PASS_FAKE_ONLY" else 1
if __name__=="__main__":sys.exit(main())
