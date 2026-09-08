"""Inert, explicitly mock ports for the owned integration path; no torch imports."""
import array
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types
from types import SimpleNamespace
from support import HERE,ROOT,SOURCES,require,sha
from setup_counter import LoadSources
from loader_diagnostics import Context
COMMIT="84bfd749876c44dc9bb1bc1e75db89f3e491c159"
PREFIX="diagnostics/fresh_confirmation_loader_diagnostics_v1/"

class Parameter:
    def __init__(self,index):
        self.value=array.array("f",[float(index+1),float(index+2)]);self.shape=(2,)
        self.device=SimpleNamespace(type="cpu");self.dtype="float32";self.grad=None;self.requires_grad=True;self._version=0
    def is_floating_point(self): return True
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.value
    def requires_grad_(self,value): self.requires_grad=value;return self

class Bridge:
    def __init__(self):
        self.parts=[Parameter(i) for i in range(12)];self.training=False;self._last_hf_cache=None
        self.tokenizer=SimpleNamespace(chat_template="FROZEN_BYTE_FIXTURE_TEMPLATE")
    def named_parameters(self): return iter([("p"+str(i),p) for i,p in enumerate(self.parts)])
    def forward(self,*args,**kwargs): raise AssertionError("MOCK_FORWARD_BODY_MUST_NEVER_RUN")

def invoke(raw,writer,counter,deadline,admitted,case):
    source_binding=SOURCES.load("checked_pure_extraction",COMMIT,PREFIX+"source_binding.py")
    inputs=json.loads(SOURCES.read(COMMIT,PREFIX+"TEST_INPUTS.json"))
    runtime=inputs["mock_runtime_spec"];metadata=json.loads(json.dumps(runtime["runtime_compatibility"]))
    if case=="constructor_failure": runtime["runtime_compatibility"]["weight_sha256"]="0"*64
    model=Bridge();fake_torch=SimpleNamespace(float32="float32",autograd=SimpleNamespace(grad=lambda *a,**k:(_ for _ in ()).throw(AssertionError("GRAD_BODY_MUST_NOT_RUN"))))
    ns={"require":require,"sha":sha,"hashlib":hashlib,"time":time,"torch":fake_torch,"contextmanager":contextlib.contextmanager}
    source_binding.extracted(source_binding.source("real_adapter.py"),["ForwardDerivativeGuard","DispatchStopped"],ns)
    original_guard=ns["ForwardDerivativeGuard"]
    def retain_guard(model_class,counters,latch,deadline):
        guard=original_guard(model_class,counters,latch,deadline);counters.guard=guard;return guard
    ns["ForwardDerivativeGuard"]=retain_guard
    source_binding.extracted(SOURCES.read(COMMIT,PREFIX+"candidate_real_adapter.py").decode(),["parameter_digest","RealAdapter"],ns)
    from hook_binding import create_latch,component,WriterIO,RecorderView
    plan=mock_plan()
    def recorder(model,writer,latch,labels,spec):
        io=WriterIO(writer);raw_rec=component().Recorder(writer.root,latch.raw,labels,io=io);latch.recorder=raw_rec
        baseline={"mock_only":True,"registry":["unchanged"],"cache_empty":True}
        guard=SimpleNamespace(baseline=baseline,inspect=lambda current:{"matches":True,"changes":[],"current":baseline})
        raw_rec.admit(baseline,baseline,{"changes":[]},{"mock_only":True,"passed":True,"before_materialization":True,"source_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes())},{"status":"PASS","forward_calls":0,"injected_tensor_module_only":True})
        return RecorderView(raw_rec,guard,io)
    class Backend:
        @staticmethod
        def load(configuration,with_lens):
            require(with_lens is False,"same loader lens argument")
            if case=="forbidden_forward": model.forward()
            if case=="forbidden_derivative": fake_torch.autograd.grad()
            if case=="second_load": return Backend.load(configuration,with_lens=with_lens)
            return SimpleNamespace(model=model,metadata=lambda:metadata)
    class MockSources:
        def read(self,*args): return inputs["mock_config_bytes"].encode()
        def load(self,name,commit,path):
            if name.endswith(".config"): return SimpleNamespace(load_config=lambda path:{"mock":True})
            if name.endswith(".backend"): return SimpleNamespace(ResearchBackend=Backend)
            return SimpleNamespace()
    # Only the fake dependency port sees these fixture booleans; all persisted
    # authority, dispatch, diagnostics and terminal records use admitted identity.
    predicate_authority=inputs["mock_admitted"]
    def new_context(w,ignored):
        context=Context(admitted["execution"],w.loader_diagnostic_publisher,
            sha(SOURCES.read(COMMIT,PREFIX+"SOURCE_FREEZE.json")))
        w.loader_diagnostics=context;return context
    class RuntimePath:
        def __truediv__(self,name): return self
        def read_bytes(self): return json.dumps(runtime).encode()
    ns.update(json=json,os=os,sys=sys,types=types,Path=Path,HERE=RuntimePath(),ROOT=ROOT,SOURCES=LoadSources(MockSources(),counter),
        SCIENCE_COMMIT="INERT_PORT_ONLY",authenticate=lambda:predicate_authority,TransformerBridge=Bridge,
        create_latch=create_latch,create_recorder=recorder,build_plan=lambda:plan,new_context=new_context)
    source_binding.extracted(raw.decode(),["_diagnostic_load_adapter","load_adapter"],ns,strip_imports=True)
    return ns["load_adapter"](writer,counter,deadline,predicate_authority)

def mock_plan():
    prompts=[{"prompt_id":"p"+str(i),"category":"self" if i<6 else "ordinary"} for i in range(24)]
    requests=[{"prompt_id":p["prompt_id"],"request_id":p["prompt_id"]+"_"+v} for p in prompts for v in ("P","C")]
    cells=[{"cell_id":"cell"+str(i)} for i in range(180)]
    return {"prompts":prompts,"requests":requests,"cells":cells}

def partial_native_publication(publish,raw):
    """One actual seven-byte write; native write_new rejects its short count."""
    original=Path.open
    class ShortStream:
        def __init__(self,stream): self.stream=stream
        def __enter__(self): self.stream.__enter__();return self
        def __exit__(self,*args): return self.stream.__exit__(*args)
        def write(self,data): return self.stream.write(data[:7])
        def __getattr__(self,name): return getattr(self.stream,name)
    def opened(path,*args,**kwargs):
        stream=original(path,*args,**kwargs)
        return ShortStream(stream) if path.name=="LOADER_DIAGNOSTICS.json" and args and args[0]=="xb" else stream
    Path.open=opened
    try: return publish(raw)
    finally: Path.open=original
