"""Ordinary saved-only engineering tests; never constructs a tensor or provider."""
import copy
import importlib.abc
import json
from pathlib import Path
import struct
import sys
import tempfile
import time

class DenyProviders(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {"torch","transformers","transformer_lens","datasets","pyarrow","tokenizers","safetensors"}:
            raise RuntimeError("PROVIDER_IMPORT_FORBIDDEN")
sys.meta_path.insert(0,DenyProviders())
from reference import authenticate,need,sha,canonical,VOCAB
from replay import run,audit,Output,LIMITS,SCHEDULE
from engine import ReceiverEngine,real_entrypoint
HERE=Path(__file__).resolve().parent

class SavedFake:
    limits=LIMITS
    def __init__(self,ref,fault=None):self.ref=ref;self.fault=fault;self.calls=[];self.d=0;self.closed=False
    def forward(self,cell,ids,mask,offset):
        i=len(self.calls);need(cell==SCHEDULE[i],"FAKE_SCHEDULE");self.calls.append(cell)
        if i==0:
            need(ids==[1] and mask==[1] and offset==[0.0]*1024,"FAKE_SMOKE")
            return {"logits_raw":bytes(VOCAB*4),"h":[0.0]*1024,"unselected_sha256":sha(b""),"nonfinal_unchanged":True}
        row=self.ref["rows"][i-1]
        need(ids==self.ref["boundary"]["full_token_ids"] and mask==[1]*137 and offset==row["cumulative_offset"],"FAKE_FIXED_INPUT")
        result={"logits_raw":self.ref["logits"][i-1],"h":row["h"],"unselected_sha256":row["unselected_sha256"],"nonfinal_unchanged":True}
        if self.fault==cell:result["h"]=[x+0.25 for x in row["h"]]
        return result
    def gradient(self):
        need(self.calls[-1]=="gradient_1" and self.d==0,"FAKE_CURRENT_GRAPH");self.d+=1
        return self.ref["rows"][2]["gradient"]
    def close(self):self.closed=True;return {"complete":True,"synthetic_cleanup":True}
    def accounting(self):return {"loads":1,"forwards":len(self.calls),"derivatives":self.d,"encoding":0}

def main():
    start=time.monotonic();reference=authenticate();checks=[]
    identity={"observed_mode":"SAVED_FAKE","reference_commit":reference["receipt"]["reference_commit"],"actual_model_loads":0}
    # Fixtures live only within this new development namespace; output contains raw saved copies, not model outputs.
    root=Path(tempfile.mkdtemp(prefix="saved_fake_",dir=HERE))
    engine=SavedFake(reference);ok=run(reference,engine,Output(root/"success"),execution_identity=identity,worker_deadline=time.monotonic()+300)
    need(ok["status"]=="REPLAY_PARITY_COMPLETE" and len(engine.calls)==6 and engine.d==1 and engine.closed,"CLEAN_JOIN")
    need(audit(reference,root/"success",identity)["status"]=="SAVED_REPLAY_PARITY_COMPLETE","INDEPENDENT_SAVED_JOIN")
    checks.append("six fixed cells and one derivative; independent full-array audit")
    engine=SavedFake(reference,"entry");bad=run(reference,engine,Output(root/"fail_prefix"),execution_identity=identity,worker_deadline=time.monotonic()+300)
    need(bad["status"]=="INCOMPLETE" and bad["primary"]=="EXACT_H" and bad["unrun"]==list(SCHEDULE[3:])
         and engine.calls==list(SCHEDULE[:3]) and engine.d==0 and engine.closed,"FAIL_STOP_SUFFIX")
    checks.append("first parity failure stops, exact UNRUN suffix, cleanup retained")
    # Coherent outer hashes cannot make a corrupted inner gradient pass the independent audit.
    target=root/"success"/"gradient.json";original=target.read_bytes();value=json.loads(original);value["gradient"][0]+=0.1
    altered=canonical(value);target.write_bytes(altered)
    term=root/"success"/"terminal.json";terminal=json.loads(term.read_bytes())
    for rec in terminal["files"]:
        if rec["path"]=="gradient.json":rec.update(bytes=len(altered),sha256=sha(altered))
    term.write_bytes(canonical(terminal))
    try:audit(reference,root/"success",identity)
    except ValueError as error:need(str(error)=="SAVED_GRADIENT_PARITY","INTENDED_INNER_CHECK")
    else:raise ValueError("TAMPER_ACCEPTED")
    checks.append("coherent repaired-hash gradient tamper rejected")
    # Exercise the actual handoff adapter with inert receiver methods, no model/tensor.
    class Value:
        def __init__(self,values):self.values=values
        def detach(self):return self
        def cpu(self):return self
        def tolist(self):return self.values
    class Receiver:
        def __init__(self):self.events=[];self.i=0;self.capture=None
        def clean(self):self.events.append("clean")
        def start_request(self,name):self.events.append(("start",name))
        def begin_edit(self):self.events.append("begin")
        def finish_request(self):self.events.append("finish")
        def clear_capture(self):self.capture=None
        def forward_inputs(self,ids,mask,phase,offset):
            expected="baseline" if self.i==0 else reference["rows"][self.i-1]["condition"]
            need(phase==expected,"HANDOFF_PHASE");self.events.append(("forward",phase));i=self.i;self.i+=1
            self.capture={"unselected_sha256":sha(b"") if i==0 else reference["rows"][i-1]["unselected_sha256"]}
            return (bytes(VOCAB*4) if i==0 else reference["logits"][i-1]),Value([0.0]*1024 if i==0 else reference["rows"][i-1]["h"])
        def gradient(self,current):
            need(current==reference["logits"][2],"HANDOFF_GRAPH");self.events.append("gradient")
            return Value(reference["rows"][2]["gradient"])
        def finalize(self):self.events.append("finalize");return {"inert":True}
    from types import SimpleNamespace
    receiver=Receiver();counter=SimpleNamespace(attempts={"forward":0,"derivative":0});restored=[];weights=[]
    def weight():weights.append(True);return True
    handoff=ReceiverEngine(receiver,counter,lambda x:x,lambda x:x,weight,lambda:restored.append(True))
    for i,cell in enumerate(SCHEDULE):
        row=None if i==0 else reference["rows"][i-1]
        handoff.forward(cell,[1] if i==0 else reference["boundary"]["full_token_ids"],[1] if i==0 else [1]*137,[0.0]*1024 if i==0 else row["cumulative_offset"])
        if cell=="gradient_1":handoff.gradient()
    need(handoff.close()["complete"] and restored==[True] and len(weights)==2,"HANDOFF_CLOSE")
    need(counter.attempts=={"forward":6,"derivative":1} and receiver.events.count("finish")==1
         and receiver.events.count("begin")==2 and receiver.events.count("gradient")==1,"HANDOFF_COLD_ENDPOINT")
    checks.append("actual receiver handoff: fixed phases, current derivative, cold endpoint, weight checks and restoration")
    try:real_entrypoint()
    except RuntimeError as error:need(str(error)=="NATIVE_REPLAY_REAL_ADMISSION_NOT_IMPLEMENTED","DISABLED_ENTRY")
    else:raise ValueError("REAL_ENTRY_REACHABLE")
    checks.append("real entry remains blocked before any provider or checkpoint work")
    report={"status":"PASS","kind":"saved evidence replay harness engineering; not native numerical parity",
        "seconds":time.monotonic()-start,"checks":checks,"reference":reference["receipt"],
        "actual_model_loads":0,"actual_forwards":0,"actual_derivatives":0,"provider_imports":0,
        "synthetic_root":str(root.relative_to(HERE)),"unverified":["native loading/admission","native legacy weight mapping","actual native numerical parity","retained-owner controller integration"]}
    (HERE/"TEST_RESULTS.json").write_bytes(canonical(report));print(json.dumps(report))

if __name__=="__main__":main()
