"""One fixed, bounded real-library hook-lifecycle fixture; never a language model."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def write(name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    assert len(raw)<=5*1024**2
    assert sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())+len(raw)<32*1024**2
    with (HERE/name).open("xb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())

def fixture_types():
    import torch
    from types import SimpleNamespace
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.model_bridge.bridge_core import BridgeCore
    from transformer_lens.model_bridge.generalized_components.block import BlockBridge
    from transformer_lens.model_bridge.generalized_components.attention import AttentionBridge
    from transformer_lens.model_bridge.generalized_components.normalization import NormalizationBridge
    class Arithmetic(torch.nn.Module):
        def __init__(self,ln1,ln2):
            super().__init__()
            self.ln1,self.ln2=ln1,ln2
        def forward(self,hidden_states): return self.ln2(self.ln1(hidden_states)+1)+2
    class TinyBridge(TransformerBridge):
        calls=0
        def __init__(self):
            torch.nn.Module.__init__(self)
            cfg=SimpleNamespace(d_vocab=2,d_vocab_out=2,device="cpu",d_model=4,n_heads=1,
                                use_hook_mlp_in=False,eps=1e-5,normalization_type="LN")
            adapter=SimpleNamespace(cfg=cfg,component_mapping={})
            BridgeCore.__init__(self,adapter,None,SimpleNamespace(non_fireable_hook_points=frozenset()))
            ln1=NormalizationBridge("ln1",cfg,use_native_layernorm_autograd=True)
            ln2=NormalizationBridge("ln2",cfg,use_native_layernorm_autograd=True)
            ln1.set_original_component(torch.nn.LayerNorm(4))
            ln2.set_original_component(torch.nn.LayerNorm(4))
            attn=AttentionBridge("attn",cfg)
            block=BlockBridge("blocks.0",cfg,submodules={"ln1":ln1,"ln2":ln2,"attn":attn})
            block.add_module("ln1",ln1)
            block.add_module("ln2",ln2)
            block.add_module("attn",attn)
            original=Arithmetic(ln1,ln2)
            block.set_original_component(original)
            self.__dict__["original_model"]=original
            self.blocks=torch.nn.ModuleList([block])
            self._scan_existing_hooks(block,"blocks.0")
            self._hook_registry_initialized=True
        def forward(self,values):
            type(self).calls+=1
            assert type(self).calls<=8,"toy arithmetic forward cap"
            assert values.shape==(1,2,4) and not torch.is_grad_enabled()
            return self.blocks[0](values)[0]
    return torch,TinyBridge

def paths():
    base=ROOT/".venv/Lib/site-packages"
    names=["transformer_lens/model_bridge/bridge_core.py","transformer_lens/model_bridge/transformer_bridge.py",
           "transformer_lens/hook_points.py","transformer_lens/model_bridge/generalized_components/base.py",
           "transformer_lens/model_bridge/generalized_components/block.py","transformer_lens/model_bridge/generalized_components/attention.py",
           "transformer_lens/model_bridge/generalized_components/normalization.py","torch/nn/modules/module.py"]
    return {str(base/name):sha((base/name).read_bytes()) for name in names}

def freeze():
    record={"source_sha256":{n:sha((HERE/n).read_bytes()) for n in ("PLAN.json","README.md","guard_candidate.py","lifecycle.py","candidate.patch","SUCCESSOR_PLAN.json")},
            "installed_sources_sha256":paths(),"prospective_plan_commit":"d788624ddf92558aa37ac3fb1433e9ede5ab2934",
            "qwen_loaded":False,"toy_forwards":0}
    write("freeze.json",record)
    print(json.dumps({"status":"FROZEN","freeze_sha256":sha((HERE/"freeze.json").read_bytes())}))

def check_freeze():
    frozen=read(HERE/"freeze.json")
    assert all(sha((HERE/n).read_bytes())==s for n,s in frozen["source_sha256"].items())
    assert paths()==frozen["installed_sources_sha256"]

def experiment():
    started=time.monotonic()
    receipt={"status":"FAIL","qwen_loads":0,"language_model_forwards":0,"derivatives":0,"toy_forwards":0,"error":None,"cases":[]}
    torch=TinyBridge=None
    try:
        check_freeze()
        torch,TinyBridge=fixture_types()
        from guard_candidate import HookGuard,snapshot,differences
        import torch.nn.modules.module as module_api
        observed=[]
        def capture(value,hook): observed.append(float(value.sum())); return value
        def edit(value,hook): return value+.25
        def gradient_capture(value,hook): observed.append("gradient_context_only"); return value
        def baseline(value,hook): return value
        x=torch.arange(8,dtype=torch.float32).reshape(1,2,4)
        def call(model):
            with torch.inference_mode(): return model(x)
        def save(name,model,before,during,after,expected,actual,extra=None):
            assert expected==actual,name+" unexpected guard result"
            rec={"case":name,"status":"PASS","expected_guard_clean":expected,"observed_guard_clean":actual,
                 "before":before,"during":during,"after":after,"after_changes":differences(before,after),**(extra or {})}
            index=len(receipt["cases"])+1
            write(f"case_{index:02d}.json",rec)
            receipt["cases"].append({"case":name,"status":"PASS","artifact":f"case_{index:02d}.json"})
        def forbidden(*a,**kw): raise AssertionError("derivative forbidden in model-free fixture")
        original_grad,original_backward=torch.autograd.grad,torch.autograd.backward
        torch.autograd.grad=torch.autograd.backward=forbidden
        try:
            model=TinyBridge()
            before=snapshot(model)
            with model.hooks(fwd_hooks=[("blocks.0.hook_out",capture)]):
                during=snapshot(model)
                call(model)
            after=snapshot(model)
            point=model.blocks[0].hook_out
            assert not point.fwd_hooks and not point._forward_hooks
            assert len(model.blocks[0]._pre_ln_capture_handles)==2
            assert before!=after
            save("cold_capture_lazy_setup",model,before,during,after,False,before==after,
                 {"diagnostic_capture_callbacks_remaining":0,"architecture_pre_hooks_remaining":2,
                  "known_ln1_alias_present":model.blocks[0].attn._ln1_module is model.blocks[0].ln1.original_component})
            model=TinyBridge()
            model.blocks[0].hook_out.add_perma_hook(baseline)
            guard=HookGuard(model)
            before=guard.baseline
            histories=[]
            for _ in range(3):
                with model.hooks(fwd_hooks=[("blocks.0.hook_out",capture)]):
                    during=snapshot(model); call(model)
                histories.append(guard.require_clean(model))
            save("pre_materialized_capture_repeat_3",model,before,during,snapshot(model),True,guard.inspect(model)["matches"],
                 {"setup":guard.setup,"repeat_clean":[r["matches"] for r in histories]})
            model=TinyBridge()
            guard=HookGuard(model)
            before=guard.baseline
            with model.hooks(fwd_hooks=[("blocks.0.hook_out",edit)]):
                outer=snapshot(model)
                with model.hooks(fwd_hooks=[("blocks.0.hook_out",gradient_capture)]):
                    with model.hooks(fwd_hooks=[("blocks.0.hook_out",capture)]):
                        during=snapshot(model); call(model)
                    inner_exit=snapshot(model)
                    assert differences(outer,inner_exit),"installed remove-all semantics must be visible"
            save("nested_edit_gradient_capture_contexts_no_derivative",model,before,during,snapshot(model),True,guard.inspect(model)["matches"],
                 {"outer_state":outer,"after_inner_exit":inner_exit,"outer_callbacks_removed_by_inner_exit":True,"derivatives":0})
            names=read(HERE/"PLAN.json")["experiment_order"][3:]
            for name in names:
                model=TinyBridge()
                point=model.blocks[0].hook_out
                if name in ("baseline_callback_removed","baseline_callback_replaced_same_key","baseline_callback_order_changed"):
                    point.add_hook(baseline,is_permanent=name!="baseline_callback_removed")
                    if name=="baseline_callback_order_changed": point.add_perma_hook(capture)
                guard=HookGuard(model)
                before=guard.baseline
                handle=None
                try:
                    if name=="capture_leak": point.add_hook(capture); call(model)
                    elif name=="edit_leak": point.add_hook(edit); call(model)
                    elif name=="permanent_leak": point.add_perma_hook(edit)
                    elif name=="forward_pre_leak": handle=point.register_forward_pre_hook(lambda m,a:None)
                    elif name=="backward_leak_no_backward_call": point.add_hook(capture,dir="bwd")
                    elif name=="global_forward_leak": handle=module_api.register_module_forward_hook(lambda m,a,o:None)
                    elif name=="global_forward_pre_leak": handle=module_api.register_module_forward_pre_hook(lambda m,a:None)
                    elif name=="global_backward_leak_no_backward_call": handle=module_api.register_module_full_backward_hook(lambda m,a,o:None)
                    elif name=="baseline_callback_removed":
                        with model.hooks(fwd_hooks=[("blocks.0.hook_out",capture)]): pass
                    elif name=="baseline_callback_replaced_same_key":
                        key=next(iter(point._forward_hooks)); point._forward_hooks[key]=lambda m,a,o:o
                    elif name=="baseline_callback_order_changed":
                        point._forward_hooks.move_to_end(next(iter(point._forward_hooks)))
                    elif name=="unexpected_module_added": model.blocks[0].add_module("unexpected",torch.nn.Identity())
                    elif name=="expected_alias_replaced": model.blocks[0].attn._ln1_module=torch.nn.Identity()
                    else: raise AssertionError(name)
                    during=snapshot(model)
                    result=guard.inspect(model)
                    save(name,model,before,during,result["current"],False,result["matches"])
                finally:
                    if handle is not None: handle.remove()
                    point.remove_hooks(dir="both",including_permanent=True)
                    model.blocks[0]._teardown_pre_ln_capture()
        finally:
            torch.autograd.grad,torch.autograd.backward=original_grad,original_backward
        assert [r["case"] for r in receipt["cases"]]==read(HERE/"PLAN.json")["experiment_order"]
        receipt["status"]="PASS"
    except BaseException as error:
        receipt["error"]=type(error).__name__+": "+str(error)
        raise
    finally:
        receipt["toy_forwards"]=0 if TinyBridge is None else TinyBridge.calls
        receipt["elapsed_seconds"]=time.monotonic()-started
        receipt["unrun_cases"]=read(HERE/"PLAN.json")["experiment_order"][len(receipt["cases"]):]
        write("experiment_receipt.json",receipt)

def run():
    check_freeze()
    write("RUN_STARTED.json",{"started_unix":time.time(),"timeout_seconds":60,"attempts":1})
    started=time.monotonic()
    receipt={"status":"FAIL","error":None}
    try:
        result=subprocess.run([sys.executable,"-B",str(HERE/"lifecycle.py"),"worker"],capture_output=True,timeout=60)
        write("process_output.json",{"stdout":result.stdout.decode(errors="replace"),"stderr":result.stderr.decode(errors="replace")})
        receipt.update(exit_code=result.returncode,status="PASS" if result.returncode==0 else "FAIL")
    except BaseException as error:
        receipt["error"]=type(error).__name__+": "+str(error)
        raise
    finally:
        receipt["elapsed_seconds"]=time.monotonic()-started
        write("supervisor_receipt.json",receipt)
    print(json.dumps(receipt))

if __name__=="__main__":
    command=sys.argv[1]
    if command=="setup":
        start=time.monotonic()
        torch,Tiny=fixture_types()
        fixture=Tiny()
        print(json.dumps({"status":"SETUP_ONLY","toy_parameters":sum(p.numel() for p in fixture.parameters()),
                          "modules":len(list(fixture.modules())),"toy_forwards":Tiny.calls,"elapsed_seconds":time.monotonic()-start}))
    elif command=="freeze": freeze()
    elif command=="run": run()
    elif command=="worker": experiment()
