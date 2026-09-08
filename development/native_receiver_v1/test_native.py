"""Tiny CPU float32 vertical slice; blocks all native-provider/TL/PyArrow imports."""
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time
import types
HERE=Path(__file__).resolve().parent
BLOCKED={"transformers","transformer_lens","pyarrow","datasets","tokenizers","safetensors","sp_lense"}
class NoProviders:
    def find_spec(self,name,path=None,target=None):
        if name.split(".")[0] in BLOCKED:raise RuntimeError("FORBIDDEN_PROVIDER_IMPORT")
sys.meta_path.insert(0,NoProviders())
def audit(event,args):
    if event in ("socket.connect","socket.bind","urllib.Request"):raise RuntimeError("NETWORK_FORBIDDEN")
    if event=="open" and args and isinstance(args[0],str) and args[0].lower().endswith((".safetensors",".pt",".pth",".ckpt")):
        raise RuntimeError("CHECKPOINT_ACCESS_FORBIDDEN")
sys.addaudithook(audit)

def need(ok,code):
    if not ok:raise RuntimeError(code)
def main():
    started=time.monotonic();results=[]
    import torch
    from core import trace,DispatchLatch,ForwardDerivativeGuard,parameter_digest
    from receiver import NativeReceiver,registry
    torch.set_num_threads(1)
    class Block(torch.nn.Module):
        def __init__(self,factor):
            super().__init__();self.scale=torch.nn.Parameter(torch.full((1024,),factor,dtype=torch.float32))
        def forward(self,x):return x*self.scale
    class TinyNative(torch.nn.Module):
        def __init__(self,mode="clean"):
            super().__init__();self.model=torch.nn.Module()
            self.model.layers=torch.nn.ModuleList([Block(2. if i==11 else 1.) for i in range(24)])
            self.mode=mode;self.body_calls=0;self.foreign=None;self.eval()
        def forward(self,*,input_ids,attention_mask,past_key_values,use_cache,logits_to_keep,return_dict):
            need(input_ids.dtype==attention_mask.dtype==torch.int64 and torch.equal(attention_mask,torch.ones_like(input_ids))
                 and past_key_values is None and use_cache is False and logits_to_keep==0 and return_dict is True,"NATIVE_ARGS")
            self.body_calls+=1
            h=input_ids.to(torch.float32).unsqueeze(-1).expand(*input_ids.shape,1024)/16
            for i,layer in enumerate(self.model.layers):
                h=layer(h)
                if i==10 and self.mode=="duplicate":h=layer(h)
            z=torch.full((*input_ids.shape,248320),-100.,dtype=torch.float32)
            z[:,:,50057]=h[:,:,0].square();z[:,:,48964]=0.
            if self.mode=="foreign":self.foreign=self.model.layers[0].register_forward_hook(lambda m,a,o:None)
            return types.SimpleNamespace(logits=z,past_key_values={} if self.mode=="cache" else None)
    class Counts:
        def __init__(self):self.attempts={"load":0,"forward":0,"derivative":0}
        def reserve(self,kind):self.attempts[kind]+=1
    @contextmanager
    def fixture(mode="clean"):
        count=Counts();latch=DispatchLatch()
        trace.ACTIVE=trace.Trace({"mode":"TINY_SYNTHETIC","production_authorized":False},"0"*64,{},latch.stop)
        guard=ForwardDerivativeGuard(TinyNative,count,latch,time.monotonic()+30);guard.install()
        model=TinyNative(mode);r=NativeReceiver(model,guard);details={}
        try:yield r,count,model,guard,details
        finally:
            try:
                try:details["final_state"]=r.finalize()
                except BaseException as error:details["finalize_error_type"]=type(error).__name__
                details.update(primary=r.primary,secondary=r.secondary,latch_failed=latch.failed,owned_hook_removed=r.owned_handle is None,
                    forward_dispatches=guard.forwards,derivative_dispatches=guard.derivatives,attempts=dict(count.attempts),
                    parameter_flags_restored=tuple(p.requires_grad for p in r.parameters)==r.flags,
                    parameter_gradients_absent=all(p.grad is None for p in r.parameters),trace=trace.ACTIVE.status())
            finally:
                guard.restore();details["guard_restored"]=not guard.installed
                if model.foreign is not None:model.foreign.remove()
                trace.ACTIVE=None
    ids=[1,2,3];mask=[1,1,1];zero=torch.zeros(1024,dtype=torch.float32)
    def forward(r,c,phase,offset=zero):
        c.reserve("forward");return r.forward_inputs(ids,mask,phase,offset)
    def derivative(r,c,z):
        c.reserve("derivative");return r.gradient(z)
    def negative(call,code):
        try:call()
        except (ValueError,RuntimeError) as error:
            need(str(error)==code,"EXPECTED_"+code);return
        raise RuntimeError("NEGATIVE_NOT_REJECTED")
    def run(name,body):
        details=body()
        need(details["guard_restored"] and details["parameter_flags_restored"] and details["parameter_gradients_absent"],"UNCONDITIONAL_CLEANUP")
        results.append({"case":name,"status":"PASS","details":details})
    def identity():
        with fixture() as (r,c,m,g,d):
            baseline,_=forward(r,c,"baseline");prefix=r.capture["unselected_sha256"];r.clear_capture();r.start_request("OFF")
            off,_=forward(r,c,"entry")
            need(torch.equal(baseline,off) and r.capture["unselected_sha256"]==prefix and r.edit_hook_registrations==0,"EXACT_ZERO_OFF")
            r.finish_request()
        need(not d["latch_failed"] and d["final_state"]["parameter_bytes_unchanged"],"CLEAN_IDENTITY")
        return d
    def gradients():
        with fixture() as (r,c,m,g,d):
            r.start_request("ON");r.begin_edit()
            z0,h0=forward(r,c,"gradient_0");prefix=r.capture["unselected_sha256"]
            g0=derivative(r,c,z0);r.clear_capture()
            offset=zero.clone();offset[0]=.25
            z1,h1=forward(r,c,"gradient_1",offset)
            g1=derivative(r,c,z1)
            need(torch.equal(h1,h0+offset) and r.capture["unselected_sha256"]==prefix,"LAST_POSITION_ONLY")
            expected0=zero.clone();expected0[0]=1.5;expected1=zero.clone();expected1[0]=3.5
            need(torch.equal(g0,expected0) and torch.equal(g1,expected1) and not torch.equal(g0,g1),"TRUE_CURRENT_DOWNSTREAM_GRADIENT")
            d["gradient_selected_values"]=[float(g0[0]),float(g1[0])];r.finish_request()
        need(not d["latch_failed"] and d["final_state"]["parameter_bytes_unchanged"],"CLEAN_GRADIENT")
        return d
    def stale():
        with fixture() as (r,c,m,g,d):
            r.start_request("ON");r.begin_edit();old,_=forward(r,c,"gradient_0");r.clear_capture()
            forward(r,c,"gradient_1");negative(lambda:derivative(r,c,old),"CURRENT_GRAPH_IDENTITY")
        need(d["latch_failed"] and d["derivative_dispatches"]==0,"STALE_GRAPH_STOP")
        return d
    def duplicate():
        with fixture("duplicate") as (r,c,m,g,d):
            negative(lambda:forward(r,c,"baseline"),"exactly one selected block output hook")
        need(d["latch_failed"] and d["primary"]=="SELECTED_HOOK_FAILURE" and d["owned_hook_removed"],"DUPLICATE_STOP_CLEANUP")
        return d
    def cleanup():
        with fixture("foreign") as (r,c,m,g,d):
            negative(lambda:forward(r,c,"baseline"),"HOOK_CLEANUP_IDENTITY")
            need(r.owned_handle is None and m.foreign is not None,"ONLY_OWNED_HOOK_REMOVED")
        need(d["latch_failed"] and d["primary"]=="HOOK_CLEANUP_IDENTITY" and "finalize_error_type" in d,"CLEANUP_NEVER_PASS")
        return d
    def cache():
        with fixture("cache") as (r,c,m,g,d):
            negative(lambda:forward(r,c,"baseline"),"NATIVE_CACHE_RETURNED")
        need(d["latch_failed"] and d["primary"]=="NATIVE_CACHE_RETURNED","CACHE_STOP")
        return d
    def mutation():
        with fixture() as (r,c,m,g,d):
            versions=tuple(p._version for p in r.parameters);r.parameters[0].data[0]+=1
            need(tuple(p._version for p in r.parameters)==versions,"VERSION_SILENT_FIXTURE")
            need(r.parameter_state(digest=True)["parameter_bytes_unchanged"] is False,"FRESH_BYTES_NOT_VERSIONS")
        need(d["latch_failed"] and d["forward_dispatches"]==0,"MUTATION_STOP")
        return d
    def unclaimed():
        with fixture() as (r,c,m,g,d):
            negative(lambda:m(input_ids=None,attention_mask=None,past_key_values=None,use_cache=False,logits_to_keep=0,return_dict=True),"UNCLAIMED_FORWARD")
            need(m.body_calls==0,"NO_UNCLAIMED_BODY")
        need(d["latch_failed"] and d["forward_dispatches"]==0,"TICKET_STOP")
        return d
    for name,body in (("zero_OFF_exact_identity",identity),("last_position_current_gradient",gradients),("stale_graph_rejected",stale),
                      ("duplicate_selected_hook",duplicate),("foreign_hook_cleanup_failure",cleanup),("cache_return_rejected",cache),
                      ("silent_parameter_mutation",mutation),("uncounted_forward_rejected",unclaimed)):
        run(name,body)
    old=ast.parse((HERE.parents[1]/"development/gdn_public_smoke_v3/candidate_real_adapter.py").read_text())
    new=ast.parse((HERE/"core.py").read_text())
    for name in ("DispatchStopped","DispatchLatch","ForwardDerivativeGuard","parameter_digest"):
        x=next(n for n in old.body if getattr(n,"name",None)==name);y=next(n for n in new.body if getattr(n,"name",None)==name)
        need(ast.dump(x,include_attributes=False)==ast.dump(y,include_attributes=False),"EXACT_REUSED_CORE_"+name)
    receiver=ast.parse((HERE/"receiver.py").read_text())
    original_patch=next(n for n in ast.walk(old) if isinstance(n,ast.FunctionDef) and n.name=="patch")
    new_patch=next(n for n in ast.walk(receiver) if isinstance(n,ast.FunctionDef) and n.name=="patch")
    need(ast.dump(original_patch,include_attributes=False)==ast.dump(new_patch,include_attributes=False),"EXACT_RESIDUAL_PATCH_AST")
    need(not any(n.split(".")[0] in BLOCKED for n in sys.modules),"NO_PROVIDER_IMPORTED")
    return {"status":"PASS","cases":results,"elapsed_seconds":time.monotonic()-started,
        "core_AST_parity":True,"residual_patch_AST_parity":True,"provider_imports":0,"checkpoint_access":0,"real_model_loads":0,
        "production_authorized":False,"native_HF_or_gate_parity":"UNVERIFIED"}
if __name__=="__main__":
    try:result=main()
    except BaseException as error:
        result={"status":"FAIL","exception_type":type(error).__module__+"."+type(error).__name__,"production_authorized":False}
        frames=[];tb=error.__traceback__
        while tb is not None:
            path=Path(tb.tb_frame.f_code.co_filename)
            frames.append({"file":path.name,"line":tb.tb_lineno});tb=tb.tb_next
        result["frames"]=frames[-12:]
        # Only own finite fixture codes, never arbitrary import/DLL exception text.
        if type(error) in (RuntimeError,ValueError) and error.__traceback__ is not None:
            if str(error).isupper() and len(str(error))<=96:result["finite_code"]=str(error)
    print(json.dumps(result,sort_keys=True));raise SystemExit(0 if result["status"]=="PASS" else 1)
