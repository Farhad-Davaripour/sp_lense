"""One pure byte-stand-in batch through authenticated loader/constructor/closeout."""
import array
import builtins
import collections
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import types
from types import SimpleNamespace
from loader_diagnostics import HERE,Context,DiagnosticStopped,encoded,sha,new_context,retained_terminal
from source_binding import source,extracted,instrumented
from saved_diagnostics import verify

FORBIDDEN=("torch","transformer_lens","transformers","sp_lense","numpy","loader","real_adapter","backend","tokenizers")

def require(ok,message):
    if not ok: raise ValueError(message)

def bounds():
    files=[p for p in HERE.rglob("*") if p.is_file()]
    test=sum(p.stat().st_size for p in files if p.is_relative_to(HERE/"test_evidence"))
    total=sum(p.stat().st_size for p in files)
    require(total-test<=32*1024**2 and test<=8*1024**2 and all(p.stat().st_size<=5*1024**2 for p in files),"BOUNDS")
    return {"test_bytes":test,"preparation_bytes":total-test,"total_bytes":total}

def write(path,raw):
    require(type(raw) is bytes and len(raw)<=5*1024**2 and path.resolve().is_relative_to(HERE.resolve()),"BOUNDED_PATH")
    before=bounds();is_test=path.is_relative_to(HERE/"test_evidence")
    require(before["test_bytes" if is_test else "preparation_bytes"]+len(raw)<=(8 if is_test else 32)*1024**2,"BOUNDED_APPEND")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:
        require(stream.write(raw)==len(raw),"FULL_WRITE");stream.flush();os.fsync(stream.fileno())
    require(path.read_bytes()==raw,"RAW_ACK")

class Parameter:
    def __init__(self,index):
        self.values=array.array("f",[float(index+1),float(index+2)])
        self.device=SimpleNamespace(type="cpu");self.dtype="float32";self.shape=(2,)
        self.grad=None;self.requires_grad=True;self._version=0
    def is_floating_point(self): return True
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.values  # buffer stand-in, no numpy import

class ByteBridge:
    def __init__(self):
        self.parts=[Parameter(i) for i in range(12)];self.aliases=False;self.training=False;self._last_hf_cache=None
        self.tokenizer=SimpleNamespace(chat_template="FROZEN_BYTE_FIXTURE_TEMPLATE")
    def named_parameters(self):
        base=[("core.layer"+str(i//2)+(".weight" if i%2==0 else ".ln.weight"),p) for i,p in enumerate(self.parts)]
        aliases=[("blocks."+str(i)+"._ln1_module.weight",self.parts[2*i+1]) for i in range(6)] if self.aliases else []
        seen=set()
        for name,p in aliases+base:
            if id(p) not in seen: seen.add(id(p));yield name,p
    def forward(self,*args,**kwargs): raise AssertionError("FORWARD_FORBIDDEN")

class RawLatch:
    def __init__(self,schedule): self.primary_code=None;self.remaining=list(schedule)
    def trip(self,code):
        if self.primary_code is None: self.primary_code=code
    def require_dispatch(self):
        if self.primary_code is not None: raise RuntimeError("STOPPED")

class RawRecorder:
    def __init__(self,writer,latch): self.writer=writer;self.latch=latch;self.setup_complete=True;self.checks=0
    def fail(self,code):
        self.latch.trip(code);self.writer.sticky_failure=True
        raise RuntimeError("FINITE_HOOK_STOP")
    def status(self):
        return {"latch_state":"TERMINAL" if self.latch.primary_code else "ADMITTED","terminal":self.latch.primary_code is not None,
            "primary_code":self.latch.primary_code,"secondary_codes":[],"receipt_failed":False,"diagnostic_complete":False,
            "index_sha256":None,"remaining_ids":list(self.latch.remaining),"exception_text_serialized":False,"permits_pass":False}

class Writer:
    def __init__(self,root,execution,fail_write=False):
        self.root=root;self.execution=execution;self.fail_write=fail_write;self.sticky_failure=False;self.sealed=False;self.write_attempts=0
    def loader_diagnostic_publisher(self,raw):
        self.write_attempts+=1
        if self.fail_write:
            write(self.root/"LOADER_DIAGNOSTICS.json",raw[:7]);raise OSError("NEVER_SERIALIZE_THIS_EXCEPTION")
        write(self.root/"LOADER_DIAGNOSTICS.json",raw)
    def closeout(self): self.sealed=True;return {"path":None,"sha256":None,"status":"INCOMPLETE"}
    def _reconciliation(self): return {"actual_total_bytes":sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file()),"issues":[]}


def main():
    require(len(sys.argv)==2,"CALLER_BATCH_LOCK")
    raw=(HERE/"BATCH_LOCK.json").read_bytes();require(sha(raw)==sys.argv[1],"BATCH_LOCK_HASH");lock=json.loads(raw)
    for name,field in (("SOURCE_FREEZE.json","source_sha256"),("TEST_INPUTS.json","test_inputs_sha256"),("TEST_PROTOCOL.json","test_protocol_sha256")):
        require(sha((HERE/name).read_bytes())==lock[field],"FROZEN_BATCH_BYTES")
    freeze=json.loads((HERE/"SOURCE_FREEZE.json").read_bytes())
    for name,pin in freeze["files"].items(): require(sha((HERE/name).read_bytes())==pin,"FROZEN_SOURCE")
    usage=json.loads((HERE/"USAGE_BEFORE_BATCH.json").read_bytes())
    require(0<=time.time()-usage["observed_unix_seconds"]<=120 and usage["standard_codex_used_percent"]<100,"ACTUAL_USAGE_FRESH")
    protocol=json.loads((HERE/"TEST_PROTOCOL.json").read_bytes());inputs=json.loads((HERE/"TEST_INPUTS.json").read_bytes())
    started=time.monotonic();deadline=started+45.;absolute=started+60.
    write(HERE/"BATCH_STARTED.json",encoded({"source_sha256":lock["source_sha256"],"batch_lock_sha256":sys.argv[1],"started":started,"deadline":deadline,"absolute":absolute}))
    expired=threading.Event();soft=threading.Timer(45.,expired.set);soft.daemon=True;soft.start()
    hard=threading.Timer(59.5,lambda:os._exit(124));hard.daemon=True;hard.start()
    original_import=builtins.__import__;blocked=[]
    def guarded_import(name,*args,**kwargs):
        if any(name==x or name.startswith(x+".") for x in FORBIDDEN): blocked.append(name);raise RuntimeError("FORBIDDEN_IMPORT")
        return original_import(name,*args,**kwargs)
    builtins.__import__=guarded_import
    report={"status":"INCONCLUSIVE_DIAGNOSTIC_BATCH","groups":[{"name":n,"status":"UNRUN"} for n in protocol["groups"]],
        "real_model_calls":0,"torch_backend_tokenizer_calls":0,"actual_real_authorization":False,"failure":None}
    counts={"byte_backend_load_calls":0,"forwards":0,"derivatives":0};active=None
    def tick(): require(not expired.is_set() and time.monotonic()<deadline,"BATCH_DEADLINE")
    def run_case(case_name,predicate,io_fail=False):
        tick();root=HERE/"test_evidence"/case_name;require(not root.exists(),"EXCLUSIVE_CASE")
        runtime=json.loads(json.dumps(inputs["mock_runtime_spec"]));metadata=json.loads(json.dumps(runtime["runtime_compatibility"]))
        model=ByteBridge()
        if predicate=="LD_RUNTIME_METADATA": metadata["d_model"]+=1
        if predicate=="LD_PARAMETER_DEVICE_DTYPE": model.parts[0].device.type="cuda"
        if predicate=="LD_EVAL_GRADIENTS": model.training=True
        if predicate=="LD_ORDERED_WEIGHT_DIGEST": runtime["runtime_compatibility"]["weight_sha256"]="0"*64
        if predicate=="LD_BRIDGE_CACHE": model._last_hf_cache=object()
        write(root/"RUNTIME_SPEC.json",encoded(runtime))
        execution=json.loads(json.dumps(inputs["mock_admitted"]["execution"]))
        admitted={"execution":execution};writer=Writer(root,execution,io_fail)
        counters=SimpleNamespace(attempts={"load":1,"forward":0,"derivative":0})
        fake_torch=SimpleNamespace(float32="float32",autograd=SimpleNamespace(grad=lambda *a,**k:(_ for _ in ()).throw(AssertionError("GRAD_FORBIDDEN"))))
        namespace={"require":require,"sha":sha,"hashlib":hashlib,"time":time,"torch":fake_torch,"contextmanager":contextlib.contextmanager}
        extracted(source("real_adapter.py"),["ForwardDerivativeGuard","DispatchStopped"],namespace)
        extracted((HERE/"candidate_real_adapter.py").read_text(),["parameter_digest","RealAdapter"],namespace)
        latch_namespace=extracted(source("hook_binding.py"),["LatchView"],{})
        saved={}
        def create_latch(schedule):
            value=latch_namespace["LatchView"](RawLatch(schedule));saved["latch"]=value;return value
        def create_recorder(model,writer,latch,labels,spec):
            require(len(labels)==109,"EXACT_LABELS")
            raw=RawRecorder(writer,latch.raw);latch.recorder=raw;saved["recorder"]=raw
            return SimpleNamespace(raw=raw)
        original_guard=namespace["ForwardDerivativeGuard"]
        def make_guard(*args):
            guard=original_guard(*args);saved["guard"]=guard;return guard
        class Backend:
            @staticmethod
            def load(configuration,with_lens):
                require(with_lens is False,"UNCHANGED_LENS");counts["byte_backend_load_calls"]+=1
                return SimpleNamespace(model=model,metadata=lambda:metadata)
        class Sources:
            def read(self,*args): return inputs["mock_config_bytes"].encode()
            def load(self,name,*args):
                if name.endswith(".config"): return SimpleNamespace(load_config=lambda path:{"byte_fixture":True})
                if name.endswith(".backend"): return SimpleNamespace(ResearchBackend=Backend)
                return SimpleNamespace()
        prompts=[{"prompt_id":"p"+str(i),"category":"self" if i<6 else "ordinary"} for i in range(24)]
        requests=[{"prompt_id":p["prompt_id"],"request_id":p["prompt_id"]+"_"+policy} for p in prompts for policy in ("P","C")]
        plan={"prompts":prompts,"requests":requests,"cells":[{"cell_id":"c"+str(i)} for i in range(180)]}
        namespace.update(json=json,os=os,sys=sys,types=types,Path=Path,HERE=root,ROOT=root,SOURCES=Sources(),SCIENCE_COMMIT="BYTE_FIXTURE",
            authenticate=lambda:admitted,TransformerBridge=ByteBridge,ForwardDerivativeGuard=make_guard,create_latch=create_latch,
            create_recorder=create_recorder,build_plan=lambda:plan,new_context=new_context)
        extracted((HERE/"candidate_loader.py").read_text(),["_diagnostic_load_adapter","load_adapter"],namespace,strip_imports=True)
        try: namespace["load_adapter"](writer,counters,deadline,admitted)
        except BaseException as error:
            exception_type=type(error).__name__
        else: raise AssertionError("EXPECTED_LOADER_FAILURE")
        require(saved["guard"].installed is False and saved["guard"].forwards==saved["guard"].derivatives==saved["guard"].rejected==0,"ORIGINAL_GUARD_RESTORED")
        require(saved["latch"].adapter_primary_code=="LOAD_OR_SETUP_FAILURE" and saved["latch"].raw.primary_code=="H_CAPTURE","ORIGINAL_GENERIC_STOP_UNCHANGED")
        try: saved["latch"].admit()
        except RuntimeError: pass
        else: raise AssertionError("RESUME_ALLOWED")
        # A pre-recorder runtime failure has no hook fault writer; emulate only
        # the inherited worker's technical status, not a fabricated recorder.
        state={"execution":execution,"status":"INCONCLUSIVE_STUDY","scientific_failures":[],"technical_failures":[{"code":"WORKER_EXCEPTION"}],
            "cleanup_complete":False,"fresh_routes":0,"attempts":dict(counters.attempts),
            "cell_status":{c["cell_id"]:"UNRUN" for c in plan["cells"]},"request_status":{r["request_id"]:"UNRUN" for r in requests}}
        close_namespace=extracted(source("runtime_closeout.py"),["close_runtime"],{"require":require})
        if not writer.sticky_failure:
            writer.append_log=lambda *a:None
        capture=retained_terminal(close_namespace["close_runtime"],None,None,writer,state,lambda *a:None,encoded)
        terminal=capture["terminal"];diagnostic=terminal["loader_diagnostics"]
        require(diagnostic["first_failure"]["predicate_code"]==predicate,"FIRST_PREDICATE_PRESERVED")
        require(terminal["state"]["status"]=="INCONCLUSIVE_STUDY" and terminal["complete_evidence"] is False,"ORIGINAL_FINDING")
        if "recorder" in saved:
            require(saved["recorder"].setup_complete and terminal["recorder_status"]==saved["recorder"].status(),"RECORDER_SURVIVES_CONSTRUCTOR")
        else: require(terminal["recorder_status"] is None,"NO_FABRICATED_RECORDER")
        require(writer.write_attempts==1,"NO_DIAGNOSTIC_RETRY")
        if io_fail: require((root/"LOADER_DIAGNOSTICS.json").stat().st_size==7 and diagnostic["diagnostic_io_failed"],"PARTIAL_BYTES_PRESERVED")
        write(root/"WORKER_TERMINAL.json",encoded(terminal))
        judged=verify(root,execution,lock["source_sha256"])
        require(judged["diagnostic_verified"] is (not io_fail) and judged["permits_scientific_pass"] is False,"INDEPENDENT_DIAGNOSTIC_FINDING")
        write(root/"SAVED_AUDIT.json",encoded(judged))
        return {"case":case_name,"first_failure":diagnostic["first_failure"],"exception_type":exception_type,
            "inherited_latch_code":"H_CAPTURE","guard_restored":True,"recorder_retained":"recorder" in saved,
            "diagnostic_write_attempts":writer.write_attempts,"partial_bytes":7 if io_fail else 0,"saved_audit":judged,
            "observation":"BYTE_STANDINS_ONLY_NO_MODEL_IMPORT_OR_LOAD"}
    try:
        for active in report["groups"]:
            tick();active["status"]="STARTED"
            if active["name"] in ("distinct_predicate_failures","diagnostic_write_failures"):
                io_fail=active["name"]=="diagnostic_write_failures"
                fixed=protocol["io_cases" if io_fail else "predicate_cases"]
                active["cases"]=[{"name":c["name"],"status":"UNRUN"} for c in fixed]
                for expected,receipt in zip(fixed,active["cases"],strict=True):
                    receipt["status"]="STARTED"
                    try: receipt.update(details=run_case(expected["name"],expected["predicate"],io_fail),status="PASS")
                    except BaseException:
                        receipt["status"]="FAIL"
                        raise
            elif active["name"]=="alias_order_hypothesis_only":
                ns=extracted(source("real_adapter.py"),["parameter_digest"],{"hashlib":hashlib})
                model=ByteBridge();before=list(model.named_parameters());before_bytes={id(p):bytes(memoryview(p.values).cast("B")) for _,p in before}
                first=ns["parameter_digest"]([p for _,p in before]);model.aliases=True;after=list(model.named_parameters())
                second=ns["parameter_digest"]([p for _,p in after]);after_bytes={id(p):bytes(memoryview(p.values).cast("B")) for _,p in after}
                require(before_bytes==after_bytes and first!=second and len(before)==len(after)==12,"ORDER_ONLY_HYPOTHESIS")
                require(first==inputs["mock_runtime_spec"]["runtime_compatibility"]["weight_sha256"],"FROZEN_FAKE_EXPECTED_DIGEST")
                require([id(p) for _,p in before]!=[id(p) for _,p in after],"ALIAS_ENUMERATION_CHANGED")
                record={"fake_aliases":6,"unique_parameters":12,"underlying_bytes_identical":True,"before_ordered_sha256":first,
                    "after_ordered_sha256":second,"names_before":[n for n,_ in before],"names_after":[n for n,_ in after],
                    "frozen_weight_predicate_after_aliases":False,"hypothesis_demonstration_only":True,"real_attempt002_cause_proven":False}
                write(HERE/"test_evidence/alias_order.json",encoded(record));active["details"]=record
            else: raise AssertionError("UNKNOWN_GROUP")
            tick();active["status"]="PASS";print(json.dumps({"group":active["name"],"status":"PASS"}),flush=True)
        require(counts=={"byte_backend_load_calls":7,"forwards":0,"derivatives":0} and not blocked,"FROZEN_COUNTS")
        report["status"]="PASS_MODEL_FREE_DIAGNOSTIC_INSTRUMENTATION"
    except BaseException as error:
        if active: active["status"]="FAIL"
        import traceback
        report["failure"]={"code":"DIAGNOSTIC_BATCH_FAILURE","type":type(error).__name__,"frames":[{"file":x.filename.replace("\\","/").split("/")[-1],"line":x.lineno,"function":x.name} for x in traceback.extract_tb(error.__traceback__)[-6:]]}
    finally:
        builtins.__import__=original_import;soft.cancel()
        report.update(elapsed_seconds=time.monotonic()-started,within_substantive=time.monotonic()<deadline,within_absolute=time.monotonic()<absolute,
            counts=counts,forbidden_import_attempts=blocked,unrun_groups=sum(g["status"]=="UNRUN" for g in report["groups"]),
            source_sha256=lock["source_sha256"],batch_lock_sha256=sys.argv[1],resource_bytes=bounds(),owned_processes_started=0)
        write(HERE/"BATCH_REPORT.json",encoded(report));hard.cancel()
    return 0 if report["status"]=="PASS_MODEL_FREE_DIAGNOSTIC_INSTRUMENTATION" else 1


if __name__=="__main__":raise SystemExit(main())
