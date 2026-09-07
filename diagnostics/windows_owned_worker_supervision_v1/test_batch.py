"""Pure boundaries then exactly one live, deliberately orphaned-output fake worker."""
import copy,hashlib,json,os,queue,secrets,subprocess,sys,threading,time
from pathlib import Path
from owned import OwnedPair,strict_json,require
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def write(name,value,append=False):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+( "\n" if append else "\n")).encode()
    old=(HERE/name).stat().st_size if (HERE/name).exists() else 0
    require(old+len(raw)<=1024**2 and sum(p.stat().st_size for p in HERE.iterdir() if p.is_file())+len(raw)<=8*1024**2,"bounded evidence")
    with (HERE/name).open("ab" if append else "xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())

class FakeHandle:
    def __init__(self,snapshot,tag,rights=True,fail_terminate=False):
        self.value=copy.deepcopy(snapshot);self.tag=tag;self.termination_rights=rights;self.fail_terminate=fail_terminate;self.terminations=0
    def snapshot(self):return copy.deepcopy(self.value)
    def exited(self):return not self.value["live"]
    def terminate(self):
        self.terminations+=1
        if self.fail_terminate:raise ValueError("injected retained-handle termination failure")
        self.value["live"]=False
    def wait(self,timeout):return self.exited()

def fake_data(direct=False):
    config={"launch_image":"C:/p/base.exe" if direct else "C:/p/venv.exe","launch_sha256":"base" if direct else "venv","base_image":"C:/p/base.exe","base_sha256":"base"}
    launcher={"pid":10,"ppid":1,"image":config["launch_image"],"image_sha256":config["launch_sha256"],"creation_filetime":10,"live":True}
    actual=copy.deepcopy(launcher) if direct else {"pid":11,"ppid":10,"image":config["base_image"],"image_sha256":"base","creation_filetime":11,"live":True}
    claim={"kind":"claim","nonce":"n","pid":actual["pid"],"ppid":actual["ppid"],"os_identity":copy.deepcopy(actual),"sys_executable":config["launch_image"],"base_executable":config["base_image"]}
    return config,launcher,actual,claim

def pure_tests():
    rows=[]
    for mode in ("wrapper","direct","wrong_nonce","wrong_parent","wrong_image","wrong_sha","creation_order","reused_pid","dead_worker","multiple_claims","missing_termination_rights","cleanup_failure"):
        config,launch,actual,claim=fake_data(mode=="direct");claims=[claim];rights=True
        if mode=="wrong_nonce":claim["nonce"]="wrong"
        if mode=="wrong_parent":actual["ppid"]=99;claim["ppid"]=99;claim["os_identity"]=copy.deepcopy(actual)
        if mode=="wrong_image":actual["image"]="C:/p/unrelated.exe";claim["os_identity"]=copy.deepcopy(actual)
        if mode=="wrong_sha":actual["image_sha256"]="wrong";claim["os_identity"]=copy.deepcopy(actual)
        if mode=="creation_order":actual["creation_filetime"]=9;claim["os_identity"]=copy.deepcopy(actual)
        if mode=="reused_pid":actual["creation_filetime"]=12
        if mode=="dead_worker":actual["live"]=False;claim["os_identity"]=copy.deepcopy(actual)
        if mode=="multiple_claims":claims.append(copy.deepcopy(claim))
        if mode=="missing_termination_rights":rights=False
        launcher=FakeHandle(launch,"launcher");worker=launcher if mode=="direct" else FakeHandle(actual,"worker",rights,mode=="cleanup_failure")
        events=[];acks=[];pair=OwnedPair(launcher,worker,lambda kind,data:events.append({"kind":kind,"data":data}))
        expected_valid=mode in ("wrapper","direct","cleanup_failure");error=None
        try:pair.authorize(claims,config,"n",1,acks.append)
        except ValueError as exc:error=str(exc)
        require(pair.acknowledged==expected_valid and len(acks)==int(expected_valid),"permission boundary "+mode)
        require(bool(error)==(not expected_valid),"expected pure rejection "+mode)
        if expected_valid:
            if mode!="direct":launcher.value["live"]=False
            require(not pair.quiescent(True,True),"EOF and launcher death not quiescence")
            pair.observe(deadline=0.,now=lambda:1.)
            if mode=="cleanup_failure":
                require(pair.cleanup_faults and pair.faults and not pair.quiescent(True,True),"durable primary plus cleanup failure")
            else:require(pair.quiescent(True,True) and worker.terminations==1,"retained worker termination despite launcher death")
        else:
            pair.stop_and_wait("identity_rejected_cleanup",timeout=0.)
            require(worker.terminations==0 and not pair.authenticated and not pair.quiescent(True,True),"unrelated/unverified worker never targeted or called quiescent")
        rows.append({"fixture":mode,"status":"PASS","permission_sent":len(acks),"actual_worker_terminations":worker.terminations,"error":error,"events":events})
    write("pure_tests.json",rows);return len(rows)

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
    start=time.monotonic();overall_deadline=start+20.;proc=None;pair=None;worker=None;launcher=None;pumps=[];pipes_closed=False
    nonce=secrets.token_hex(32);events=[]
    def event(kind,data):
        row={"kind":kind,"data":data,"monotonic":time.monotonic()};events.append(row);write("lifecycle_events.jsonl",row,True)
    result={"status":"INCONCLUSIVE","live_scenarios":0,"model_calls":0,"tokenizers":0,"primary_exception":None,"final_cleanup_errors":[]}
    try:
        command=[sys.executable,"-B",str(HERE/"tiny_worker.py")]
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
        result.update(live_scenarios=1,command=command,owned_launcher_pid=proc.pid)
        launcher=NativeHandle(int(proc._handle),"owned_launcher_original",True)
        stdout=Pump(proc.stdout,"stdout");stderr=Pump(proc.stderr,"stderr");pumps=[stdout,stderr]
        proc.stdin.write(json.dumps({"nonce":nonce}).encode()+b"\n");proc.stdin.flush()
        claim=stdout.frame(3.);require(claim.get("kind")=="claim" and claim.get("nonce")==nonce and type(claim.get("pid")) is int and claim["pid"]>0,"authenticated private claim before OpenProcess")
        worker=open_claimed_worker(claim["pid"]);pair=OwnedPair(launcher,worker,event)
        claims=[claim]
        while not stdout.frames.empty():claims.append(stdout.frame(.1))
        def ack(frame):proc.stdin.write(json.dumps(frame).encode()+b"\n");proc.stdin.flush()
        pair.authorize(claims,plan["identity"],nonce,os.getpid(),ack)
        require(pair.authenticated and worker.termination_rights,"termination-capable binding before fake work")
        ready=stdout.frame(2.);require(ready=={"kind":"ready_to_close_outputs","nonce":nonce},"fixed worker closes output only after permission")
        # Predeclared one-second work allowance after READY; expiry and launcher
        # fault injection coincide so neither condition is ignored by cleanup.
        deadline=time.monotonic()+1.;threading.Event().wait(max(0.,deadline-time.monotonic()))
        event("injected_owned_launcher_termination",{"handle_tag":launcher.tag,"deadline":deadline});launcher.terminate();require(launcher.wait(1.),"owned launcher termination")
        require(stdout.eof.wait(1.) and stderr.eof.wait(1.),"both closed worker outputs reached EOF")
        for pump in pumps:pump.thread.join(.2)
        require(all(not p.thread.is_alive() and p.error is None for p in pumps),"closed output readers joined")
        for pipe in (proc.stdin,proc.stdout,proc.stderr):pipe.close()
        pipes_closed=True
        before={"launcher_exited":launcher.exited(),"actual_worker_exited":worker.exited(),"stdout_EOF":stdout.eof.is_set(),"all_readers_joined":True,"pipes_closed":True,
            "quiescent":pair.quiescent(True,True)}
        event("before_deadline_cleanup",before)
        require(before["launcher_exited"] and not before["actual_worker_exited"] and not before["quiescent"],"no false quiescence after EOF/launcher death")
        observed=pair.observe(deadline)
        require(observed=="STOPPED" and pair.quiescent(True,True) and not pair.cleanup_faults,"retained actual worker stopped and both exited")
        result.update(status="PASS_EXPECTED_FAKE_LIFECYCLE_FAULTS",binding=pair.binding,before_cleanup=before,after_cleanup={"quiescent":True,"worker_exit_code":worker.exit_code(),"launcher_exit_code":launcher.exit_code()},
            actual_worker_pid=pair.binding["actual_worker"]["pid"],actual_handle_retained_from_authentication=True,secondary_pid_lookup_for_cleanup=False)
        require(worker.exit_code()==125,"retained-handle termination beat independent child watchdog")
    except BaseException as error:result["primary_exception"]=type(error).__name__+": "+str(error)
    finally:
        if pair is not None and not (pair.worker.exited() and pair.launcher.exited()):pair.stop_and_wait("final_cleanup",timeout=1.)
        elif launcher is not None and not launcher.exited():
            try:launcher.terminate();launcher.wait(1.)
            except BaseException as error:result["final_cleanup_errors"].append(str(error))
        if proc is not None:
            # Pipe loss releases an unauthenticated waiting child; never target
            # that claimed PID. Its own watchdog is the additional hard bound.
            for pipe in (proc.stdin,proc.stdout,proc.stderr):
                try:
                    if pipe is not None:pipe.close()
                except BaseException as error:result["final_cleanup_errors"].append(str(error))
            pipes_closed=True
        for pump in pumps:pump.thread.join(.5)
        joined=all(not p.thread.is_alive() for p in pumps)
        result.update(reader_faults=[p.error for p in pumps if p.error],all_threads_joined=joined,pipes_closed=pipes_closed,
            faults=[] if pair is None else pair.faults,cleanup_faults=[] if pair is None else pair.cleanup_faults,
            quiescent=False if pair is None else pair.quiescent(joined,pipes_closed))
        if worker is not None:worker.close()
        if proc is not None:
            try:proc.wait(timeout=max(.01,min(1.,overall_deadline-time.monotonic())))
            except BaseException as error:result["final_cleanup_errors"].append(str(error))
        result["elapsed_seconds"]=time.monotonic()-start
        if result["elapsed_seconds"]>20. or result["final_cleanup_errors"] or result["reader_faults"] or not result["quiescent"]:result["status"]="INCONCLUSIVE"
        write("live_result.json",result)
    require(result["status"]=="PASS_EXPECTED_FAKE_LIFECYCLE_FAULTS","single live scenario did not pass; preserve without retry")
    return result

def main():
    start=time.monotonic();receipt={"status":"INCONCLUSIVE","model_calls":0,"tokenizers":0,"gate_scores_or_fits":0,"new_data_reads":0}
    try:
        lock=read(HERE/"freeze.json");plan=read(HERE/"plan.json")
        for name,digest in lock["files"].items():require(sha((HERE/name).read_bytes())==digest,"frozen source "+name)
        for name,digest in plan["source_sha256"].items():require(sha(Path(name).read_bytes())==digest,"external identity/supervision source "+name)
        require(os.path.normcase(sys.executable)==os.path.normcase(plan["identity"]["launch_image"]),"same pinned venv driver")
        require(read(HERE/"batch_usage.json")["known_standard_used_percent"]<100,"known usage")
        receipt["pure_fixtures"]=pure_tests();live=live_test(plan);receipt.update(status="PASS_FAKE_ONLY",live_scenarios=live["live_scenarios"])
        require(not any(x in sys.modules for x in ("torch","transformers","transformer_lens")),"no model imports")
    except BaseException as error:receipt["error"]=type(error).__name__+": "+str(error);raise
    finally:receipt["elapsed_seconds"]=time.monotonic()-start;write("batch_result.json",receipt)
    print(json.dumps(receipt))
if __name__=="__main__":main()
