"""One smoke plus five fixed calls; injected engine, no provider imports or fitting."""
import json
import math
from pathlib import Path
import struct
import time
from reference import need,sha,canonical,floats,gate,VOCAB,EPS

SCHEDULE=("public_smoke","baseline","entry","gradient_1","step_1","endpoint")
LIMITS={"loads":1,"forwards":6,"derivatives":1,"encoding":0,"worker_seconds":300,
        "audit_seconds":60,"shared_cleanup_seconds":15,"output_bytes":32*1024*1024,"file_bytes":5*1024*1024}

class Output:
    """Exclusive bounded raw evidence; occupied/partial files are never rewritten."""
    def __init__(self,path):
        self.path=Path(path);self.path.mkdir(parents=False,exist_ok=False);self.failed=False
    def write(self,name,raw,*,closeout=False):
        need(not self.failed or closeout,"STICKY_IO")
        need(Path(name).name==name,"OUTPUT_PATH")
        current=sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())
        limit=LIMITS["output_bytes"]-(0 if closeout else 65536)
        need(len(raw)<=LIMITS["file_bytes"] and current+len(raw)<=limit,"OUTPUT_CAP")
        try:
            with (self.path/name).open("xb") as stream:
                need(stream.write(raw)==len(raw),"SHORT_WRITE");stream.flush()
            need((self.path/name).read_bytes()==raw,"WRITE_ACK")
        except BaseException:self.failed=True;raise
        return {"path":name,"bytes":len(raw),"sha256":sha(raw)}
    def record(self,name,value,**kwargs):return self.write(name,canonical(value),**kwargs)

def compare(reference,row,result,model):
    raw=result["logits_raw"];values=floats(raw,VOCAB)
    expected=floats(reference["logits"][row],VOCAB);saved=reference["rows"][row]
    error=max(abs(a-b) for a,b in zip(values,expected,strict=True))
    need(error<=EPS,"LOGIT_PARITY")
    need(struct.pack("<1024f",*result["h"])==struct.pack("<1024f",*saved["h"]),"EXACT_H")
    need(result["unselected_sha256"]==saved["unselected_sha256"] and result["nonfinal_unchanged"] is True,"EXACT_NONFINAL")
    out={"maximum_logit_error":error,"logits_sha256":sha(raw),"h_exact":True,"nonfinal_exact":True}
    if row in (0,1):
        score=model.score(result["h"]);route="ON" if score>=0 else "OFF"
        expected_route=reference["routes"][row]
        need(score==expected_route["score"] and route==expected_route["route"],"EXACT_GATE_PARITY")
        out.update(gate_score=score,gate_route=route)
    return out

def run(reference,engine,output,*,execution_identity,worker_deadline):
    """Caller owns loading, guard and absolute deadlines. Never authorizes a model."""
    need(execution_identity["observed_mode"] in ("SAVED_FAKE","NATIVE_REAL"),"MODE")
    need(engine.limits==LIMITS,"ORIGINAL_LIMITS")
    need(time.monotonic()<worker_deadline<=time.monotonic()+LIMITS["worker_seconds"],"ABSOLUTE_DEADLINE")
    deadline=worker_deadline
    result={"status":"INCOMPLETE","scientific_pass":False,"execution":execution_identity,"cells":[],"derivatives":0,
            "unrun":list(SCHEDULE),"files":[],"primary":None,"cleanup":None}
    model=gate(reference["fitted"])
    try:
        for index,cell in enumerate(SCHEDULE):
            need(time.monotonic()<deadline,"WORKER_DEADLINE")
            row=None if index==0 else reference["rows"][index-1]
            ids=[1] if row is None else reference["boundary"]["full_token_ids"]
            mask=[1] if row is None else reference["boundary"]["attention_mask"]
            offset=[0.0]*1024 if row is None else row["cumulative_offset"]
            observed=engine.forward(cell,ids,mask,offset)
            check={"cell":cell,"status":"COMPLETE"}
            floats(observed["logits_raw"],VOCAB)
            need(len(observed["h"])==1024 and observed["nonfinal_unchanged"] is True,"CAPTURE")
            result["files"].append(output.write(f"{index:03d}.f32",observed["logits_raw"]))
            serial={k:v for k,v in observed.items() if k!="logits_raw"}
            result["files"].append(output.record(f"{index:03d}.json",serial))
            if row is not None:check.update(compare(reference,index-1,observed,model))
            if cell=="gradient_1":
                gradient=engine.gradient()
                result["files"].append(output.record("gradient.json",{"gradient":gradient}))
                need(len(gradient)==1024 and all(math.isfinite(x) for x in gradient),"FINITE_GRADIENT")
                error=max(abs(a-b) for a,b in zip(gradient,row["gradient"],strict=True))
                need(error<=EPS,"GRADIENT_PARITY")
                check["maximum_gradient_error"]=error;result["derivatives"]+=1
            result["cells"].append(check);result["unrun"].pop(0)
        result["status"]="REPLAY_PARITY_COMPLETE"
    except BaseException as error:
        code=str(error) if type(error) is ValueError and str(error) in {"LOGIT_PARITY","EXACT_H","EXACT_NONFINAL","EXACT_GATE_PARITY","GRADIENT_PARITY","FINITE_GRADIENT","WORKER_DEADLINE"} else "TECHNICAL_FAILURE"
        result["primary"]=code
        if result["unrun"]:result["cells"].append({"cell":result["unrun"].pop(0),"status":"FAILED"})
    finally:
        try:result["cleanup"]=engine.close();need(result["cleanup"]["complete"] is True,"CLEANUP")
        except BaseException:result["status"]="INCOMPLETE";result["cleanup"]={"complete":False}
        result["accounting"]=engine.accounting()
        try:output.record("terminal.json",result,closeout=True)
        except BaseException:result["status"]="INCOMPLETE";result["terminal_write_failed"]=True
    return result

def audit(reference,path,expected_identity):
    """Saved-only reconstruction; process/capture authority remains an outer AND."""
    root=Path(path);terminal=json.loads((root/"terminal.json").read_bytes())
    need(terminal["execution"]==expected_identity,"SAVED_EXECUTION")
    need(terminal["status"]=="REPLAY_PARITY_COMPLETE" and terminal["cleanup"]["complete"] is True,"SAVED_COMPLETE")
    need([x["cell"] for x in terminal["cells"]]==list(SCHEDULE) and all(x["status"]=="COMPLETE" for x in terminal["cells"])
         and terminal["unrun"]==[] and terminal["derivatives"]==1 and terminal["primary"] is None,"SAVED_SCHEDULE")
    need(terminal["accounting"]=={"loads":1,"forwards":6,"derivatives":1,"encoding":0},"SAVED_ACCOUNTING")
    expected={"terminal.json"}
    for record in terminal["files"]:
        need(Path(record["path"]).name==record["path"] and record["path"] not in expected,"SAVED_PATH")
        expected.add(record["path"]);raw=(root/record["path"]).read_bytes()
        need(len(raw)==record["bytes"] and sha(raw)==record["sha256"],"SAVED_BYTES")
    need({p.name for p in root.iterdir()}==expected,"UNTRACKED_BYTES")
    need(sum(p.stat().st_size for p in root.iterdir())<=LIMITS["output_bytes"],"SAVED_CAP")
    model=gate(reference["fitted"])
    for index in range(6):
        record=json.loads((root/f"{index:03d}.json").read_bytes());record["logits_raw"]=(root/f"{index:03d}.f32").read_bytes()
        floats(record["logits_raw"],VOCAB)
        if index:
            saved=reference["rows"][index-1]
            got=struct.unpack(f"<{VOCAB}f",record["logits_raw"])
            expected_logits=struct.unpack(f"<{VOCAB}f",reference["logits"][index-1])
            need(all(abs(x-y)<=EPS for x,y in zip(got,expected_logits,strict=True)),"SAVED_LOGIT_PARITY")
            need(struct.pack("<1024f",*record["h"])==struct.pack("<1024f",*saved["h"]),"SAVED_EXACT_H")
            need(record["nonfinal_unchanged"] is True and record["unselected_sha256"]==saved["unselected_sha256"],"SAVED_EXACT_NONFINAL")
            if index in (1,2):
                score=model.score(record["h"]);old=reference["routes"][index-1]
                need(score==old["score"] and ("ON" if score>=0 else "OFF")==old["route"],"SAVED_GATE_PARITY")
        if index==3:
            gradient=json.loads((root/"gradient.json").read_bytes())["gradient"]
            need(len(gradient)==1024 and all(math.isfinite(x) for x in gradient),"SAVED_GRADIENT")
            need(max(abs(a-b) for a,b in zip(gradient,reference["rows"][2]["gradient"],strict=True))<=EPS,"SAVED_GRADIENT_PARITY")
    return {"status":"SAVED_REPLAY_PARITY_COMPLETE","scientific_pass":False,"external_capture_required":True}
