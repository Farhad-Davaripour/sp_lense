"""Explicit test-only tensor module; never imports a Qwen constructor/tokenizer.

Forward receives only tensors/cache options. Fixture formulas are keyed by exact
tensor hashes, never by request policy, gold or gate audit labels.
"""
from contextlib import contextmanager
import json
import os
from types import SimpleNamespace
import torch
from support import HERE,require,sha,fixture


class TinyBridge(torch.nn.Module):
    def __init__(self,artifact_raw):
        super().__init__()
        self.scale=torch.nn.Parameter(torch.ones(1024,dtype=torch.float32))
        self.frozen=torch.nn.Parameter(torch.zeros(1,dtype=torch.float32),requires_grad=False)
        self.block=torch.nn.Identity()
        self._last_hf_cache=None
        parameters=json.loads(artifact_raw)["parameters"]
        mean,direction=torch.tensor(parameters["grand_mean"]),torch.tensor(parameters["direction"])
        table=json.loads((HERE/"INJECTION_SPEC.json").read_bytes())
        self.records={r["input_int64_le_sha256"]:r for r in table["records"]}
        self.states={key:mean+10*r["axis_sign"]*direction for key,r in self.records.items()}
        self.selected=table["first_self_tensor_sha256"]
        self.case=fixture()
        self.body_calls=0
        self.eval()

    @contextmanager
    def hooks(self,*,fwd_hooks):
        handles=[]
        try:
            for name,callback in fwd_hooks:
                require(name=="blocks.10.hook_out","same actual selected hook")
                handles.append(self.block.register_forward_hook(lambda module,args,out,callback=callback:callback(out,SimpleNamespace(name=name))))
            yield self
        finally:
            for handle in handles: handle.remove()

    def forward(self,tokens,*,attention_mask,use_cache,return_type):
        require(tokens.dtype==attention_mask.dtype==torch.int64 and torch.equal(attention_mask,torch.ones_like(tokens))
            and use_cache is False and return_type=="logits","actual uncached tensor interface")
        key=sha(tokens.contiguous().numpy().astype("<i8",copy=False).tobytes())
        require(key in self.records,"exact frozen tensor fixture")
        r,base=self.records[key],self.states[key]
        h=self.block(base.reshape(1,1,1024).expand(1,tokens.shape[1],1024)*self.scale)
        z=torch.full((248320,),-100.,dtype=torch.float32)
        if r["keep_intercept"] is not None:
            intercept=r["keep_intercept"]
            slope=1.
            if key==self.selected and self.case=="baseline_stop": intercept=.02
            if key==self.selected and self.case=="on_stop": slope=.0001
            z[50057]=intercept+slope*(h[0,-1,0]-base[0])+self.frozen[0]
            z[48964]=0.
        else:
            for token,value in r["constant_logits"]: z[token]=value
        self.body_calls+=1
        return z.reshape(1,1,248320).expand(1,tokens.shape[1],248320)


class FailedHookReceiptIO:
    """First check append is partially written; fault receipt is unavailable."""
    def __init__(self,base): self.base=base
    def __getattr__(self,name): return getattr(self.base,name)
    def write(self,path,data,append=False,fault=False):
        if fault: raise OSError("INJECTED_FAULT_RECEIPT_UNAVAILABLE")
        if path=="hook_evidence/checks.jsonl":
            calls=[0]
            def partial(fd,raw):
                calls[0]+=1
                if calls[0]==1: return os.write(fd,raw[:7])
                raise OSError("INJECTED_PARTIAL_HOOK_APPEND")
            writer=self.base.writer
            previous=writer.write_function
            writer.write_function=partial
            try: return self.base.write(path,data,append=append,fault=fault)
            finally: writer.write_function=previous
        return self.base.write(path,data,append=append,fault=fault)


def load_adapter(writer,counters,deadline,admitted):
    from authority import authenticate
    require(authenticate()==admitted and admitted["execution"]["mode"]=="INJECTION_MODULE"
        and admitted["execution"]["production_authorized"] is False,"injection-only authenticated constructor")
    from real_adapter import RealAdapter,ForwardDerivativeGuard
    from hook_binding import create_latch,create_recorder,injected_spec
    from saved_runtime import labels_for
    from plan import build_plan
    from pins import scientific
    plan=build_plan()
    latch=create_latch([c["cell_id"] for c in plan["cells"]])
    guard=ForwardDerivativeGuard(TinyBridge,counters,latch,deadline)
    guard.install()
    try:
        require(counters.attempts=={"forward":0,"derivative":0,"load":1},"one reserved injection construction")
        model=TinyBridge(scientific().artifact_raw)
        recorder=create_recorder(model,writer,latch,labels_for(plan),injected_spec(),
            io_wrapper=FailedHookReceiptIO if fixture()=="hook_io" else None)
        spec=json.loads((HERE/"INJECTION_SPEC.json").read_bytes())
        return RealAdapter(model,recorder,guard,expected_weight_sha256=spec["parameter_sha256"],runtime_metadata=spec["runtime_metadata"])
    except BaseException:
        latch.stop("INJECTION_SETUP_FAILURE")
        guard.restore()
        raise
