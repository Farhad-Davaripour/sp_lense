"""Production adapter disabled pending separate root release; fake verification only now."""
import json,os,secrets,subprocess,sys,time
from pathlib import Path
from core import HERE,ROOT,Budget,Counter,read,require,sha,check_freeze
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from scripts.three_family_bounded_capture import run_capture
from locked_backend import LockedBackend
import editor
from learned_gate import RoutingMismatch

class ForwardGuard:
    """Installed at bridge class before loader construction, same parent implementation."""
    def __init__(self,model_class,counter):
        self.model_class,self.counter,self.cell=model_class,counter,None;self.original=model_class.forward
    def install(self):
        def counted(model,*args,**kwargs):return self.counter.call(self.cell,lambda:self.original(model,*args,**kwargs))
        self.model_class.forward=counted
    def restore(self):self.model_class.forward=self.original

def validate_backend(backend,plan):
    require(backend.device=="cpu" and backend.dtype_name=="float32","CPU float32")
    require(backend.model.cfg.n_layers==24 and backend.model.cfg.d_model==1024,"architecture")
    require(backend.config.model.id==plan["model"]["id"] and backend.config.model.revision==plan["model"]["revision"],"fixed model revision")
    require(all(p.device.type=="cpu" and (not p.is_floating_point() or p.dtype==backend.torch.float32) for p in backend.model.parameters()),"all parameter device/dtype")
    require(sha(backend.model.tokenizer.chat_template.encode())==plan["gate"]["runtime_compatibility"]["prompt_format"]["chat_template_sha256"],"loaded template identity")

def execute(plan,factory,model_class,output,deadline,synthetic=False):
    """Internal shared engine. Production admission is done before invoking this."""
    require((plan["execution_mode"]=="SYNTHETIC_ONLY")==synthetic,"explicit execution lane")
    if synthetic:require(model_class.__module__=="synthetic_backend","synthetic fixture cannot construct another model class")
    output=Path(output);budget=Budget(output);started=time.monotonic()
    counter=Counter(budget,plan["cells"],deadline);derivatives=editor.DerivativeLedger(output,plan["derivative_cells"],deadline)
    guard=ForwardGuard(model_class,counter);state={"execution_status":"technical_stop","error":None,"model_load_attempts":0,"model_load_completed":0}
    guard.install()
    try:
        state["model_load_attempts"]=1;backend=factory();state["model_load_completed"]=1
        state["load_elapsed_seconds"]=time.monotonic()-started
        require(counter.attempts==0,"loader unexpectedly forwarded")
        if not synthetic:validate_backend(backend,plan)
        backend=LockedBackend(backend,plan)
        budget.write("runtime.json",{**backend.metadata(),"execution_mode":plan["execution_mode"],
            "locked_inputs_sha256":plan.get("input_binding",{}).get("input_sha256"),"tokenizer_calls_after_load":0,
            "boundaries":[{"prompt_id":p["prompt_id"],**plan["alignment"][p["prompt_id"]]} for p in plan["prompts"]]})
        editor.evaluate(plan,backend,counter,derivatives,output,guard)
        state["execution_status"]="complete"
    except (RoutingMismatch,editor.EligibilityError) as fault:
        state.update(execution_status="scientific_stop",error=type(fault).__name__+": "+str(fault)[:2048])
    except BaseException as fault:
        state["error"]=type(fault).__name__+": "+str(fault)[:2048]
        try:budget.event("technical_faults.jsonl",{"kind":type(fault).__name__,"error":str(fault)[:2048],"monotonic":time.monotonic()})
        except BaseException as recording_fault:state["fault_record_write_error"]=type(recording_fault).__name__+": "+str(recording_fault)[:2048]
    finally:
        guard.restore();state.update(elapsed_seconds=time.monotonic()-started,forward_attempts=counter.attempts,
            forward_completed=counter.completed,derivatives_attempted=derivatives.attempts,derivatives_completed=derivatives.completed,
            cursor=counter.cursor,skips=len(counter.skips),execution_mode=plan["execution_mode"],
            real_model_loads=0 if synthetic else state["model_load_attempts"],tokenizer_calls=0,gate_fit_calls=0,
            forward_guard_restored=model_class.forward is guard.original)
        budget.write("execution_receipt.json",state);budget.write("unrun.json",plan["cells"][counter.cursor:])
        budget.write("runner_finished.json",{"status":"worker evaluator returned; process exit still independently required","monotonic":time.monotonic()})
    return state

def supervise(command,output,seconds):
    """Only the Popen object created here is terminated; never kill by supplied PID."""
    output=Path(output);budget=Budget(output);start=time.monotonic();owned={}
    require(command[0]==sys.executable and command[1]=="-B" and Path(command[2]).resolve().parent==HERE,"owned fixed namespace Python child")
    budget.write("RUN_STARTED.json",{"started_monotonic":start,"deadline_monotonic":start+seconds,"seconds":seconds,"nonce":secrets.token_hex(16),"command":command})
    def spawn(actual,**kwargs):
        require(actual==command and kwargs.get("stdin")==subprocess.DEVNULL and kwargs.get("stdout")==subprocess.PIPE and not kwargs.get("shell",False),"own exact child argv/no shell")
        child=subprocess.Popen(actual,**kwargs)
        owned.update(pid=child.pid,command=actual,identity="owned Popen process handle; PID recorded but never used for targeting",spawn_monotonic=time.monotonic())
        budget.write("process_identity.json",owned);return child
    capture=None
    try:
        capture=run_capture(command,budget,start+seconds,cwd=ROOT,process_factory=spawn,terminate_timeout=3,kill_timeout=3,reader_join_timeout=1)
        budget.write("capture.json",capture);return capture
    finally:
        budget.write("supervisor_final.json",{"elapsed_seconds_including_cleanup":time.monotonic()-start,"capture_returned":capture is not None,
            "quiescent":False if capture is None else capture["quiescent"],"owned_process":owned,"no_pid_tree_kill":True})

def recover_missing_closeout(output,plan,capture):
    output=Path(output);budget=Budget(output)
    require(capture["quiescent"],"no recovery/finalization while a writer may live")
    if capture["status"]!="complete_valid":budget.event("technical_faults.jsonl",{"kind":"external_capture","capture_fault":capture["technical_recording_fault"],"cleanup_error":capture["cleanup_error"],"monotonic":time.monotonic()})
    if (output/"execution_receipt.json").exists():return
    events=[json.loads(s) for s in (output/"forward_events.jsonl").read_text().splitlines()] if (output/"forward_events.jsonl").exists() else []
    skips=[json.loads(s) for s in (output/"skip_events.jsonl").read_text().splitlines()] if (output/"skip_events.jsonl").exists() else []
    starts=[e for e in events if e["event"]=="attempt_started"]
    cursor=len(starts)+len(skips);require(cursor<=len(plan["cells"]),"bounded recovery cursor")
    devents=[json.loads(s) for s in (output/"derivative_events.jsonl").read_text().splitlines()] if (output/"derivative_events.jsonl").exists() else []
    budget.write("execution_receipt.json",{"execution_status":"technical_stop","error":"worker final receipt unavailable; supervisor reconstructed accounting only",
        "execution_mode":plan["execution_mode"],"forward_attempts":len(starts),"forward_completed":sum(e["event"]=="attempt_completed" for e in events),
        "derivatives_attempted":sum(e["event"]=="attempt_started" for e in devents),"derivatives_completed":sum(e["event"]=="attempt_completed" for e in devents),"cursor":cursor,"skips":len(skips),"reconstructed_after_exit":True})
    if not (output/"unrun.json").exists():budget.write("unrun.json",plan["cells"][cursor:])
    if not (output/"runner_finished.json").exists():budget.write("runner_finished.json",{"status":"external supervisor observed child exit; not an earlier worker cleanup claim"})
    budget.write("recovery_receipt.json",{"preserved_scientific_failures":True,"no_invented_endpoints":True,"unrun":len(plan["cells"])-cursor})

def final_inventory(output):
    output=Path(output)
    for path in (output,output/"audit"):
        require(read(path/"capture.json")["quiescent"] and read(path/"supervisor_final.json")["quiescent"],"all worker/audit processes and writers quiescent")
    require((output/"judge_process.json").exists() and (output/"REPORT.md").exists() and (output/"final_closeout.json").exists(),"authoritative final closeout/report and process receipts before inventory")
    files=[{"path":p.relative_to(output).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for p in sorted(output.rglob("*")) if p.is_file() and p!=output/"FINAL_INVENTORY.json"]
    Budget(output).write("FINAL_INVENTORY.json",{"files":files,"sole_owner":"run.final_inventory after both owned processes, writers, report and receipts","quiescent":True})

def production_run():
    from admission import admit_production
    plan,release=admit_production()  # Disabled now; no output or model imports before authority.
    output=HERE/"real_attempt";output.mkdir(exist_ok=False);budget=Budget(output)
    budget.write("plan.json",plan);budget.write("release.json",{"freeze_sha256":sha((HERE/"freeze.json").read_bytes()),"root_release":release})
    budget.write_bytes("fitted_parameters.json",(HERE/"fitted_parameters.json").read_bytes())
    capture=supervise([sys.executable,"-B",str(HERE/"run.py"),"worker"],output,1500)
    require(capture["quiescent"],"unjoined worker: no finalization permitted")
    if (output/"WORKER_CLAIM.json").exists():
        claim=read(output/"WORKER_CLAIM.json");identity=read(output/"process_identity.json");launch=read(output/"RUN_STARTED.json")
        if claim["pid"]!=identity["pid"] or claim["nonce"]!=launch["nonce"]:
            budget.event("technical_faults.jsonl",{"kind":"worker_claim_identity","error":"own child claim mismatch","monotonic":time.monotonic()})
    else:budget.event("technical_faults.jsonl",{"kind":"worker_claim_missing","monotonic":time.monotonic()})
    recover_missing_closeout(output,plan,capture)
    audit=output/"audit";audit.mkdir(exist_ok=False)
    judge_capture=supervise([sys.executable,"-B",str(HERE/"judge.py"),str(output)],audit,180)
    require(judge_capture["quiescent"],"unjoined judge: no finalization permitted")
    budget.write("judge_process.json",judge_capture)
    from final_adjudication import finalize
    finalize(output,capture,judge_capture)
    final_inventory(output)

def production_worker():
    from admission import admit_production
    plan,_=admit_production();output=HERE/"real_attempt";started=read(output/"RUN_STARTED.json")
    require(started["command"]==[sys.executable,"-B",str(HERE/"run.py"),"worker"],"fixed launched worker command")
    Budget(output).write("WORKER_CLAIM.json",{"pid":os.getpid(),"nonce":started["nonce"]})
    require(time.monotonic()<started["deadline_monotonic"],"deadline before real imports/load")
    os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
    from transformer_lens.model_bridge import TransformerBridge
    from sp_lense.backend import ResearchBackend
    from sp_lense.config import load_config
    execute(plan,lambda:ResearchBackend.load(load_config(ROOT/plan["model"]["config_path"]),with_lens=False),TransformerBridge,output,started["deadline_monotonic"])

if __name__=="__main__":
    if sys.argv[1:]==["run"]:production_run()
    elif sys.argv[1:]==["worker"]:production_worker()
    else:raise SystemExit("run/worker only; neither is authorized in this preparation job")
