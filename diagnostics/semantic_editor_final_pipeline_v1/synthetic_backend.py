"""Tiny arithmetic fixture with real BridgeCore/HookPoint lifecycle, NOT Qwen."""
import copy
from types import SimpleNamespace
import torch
torch.set_num_threads(1)
from transformer_lens.model_bridge.bridge_core import BridgeCore
from transformer_lens.hook_points import HookPoint
from core import HERE,HOOK,read
from editor import parameter_digest

class Toy(BridgeCore,torch.nn.Module):
    def __init__(self,plan,mode="normal"):
        torch.nn.Module.__init__(self)
        BridgeCore.__init__(self,SimpleNamespace(cfg=SimpleNamespace(d_vocab=248320,d_vocab_out=248320),component_mapping={}),None,SimpleNamespace(non_fireable_hook_points=frozenset()))
        self.weight=torch.nn.Parameter(torch.tensor(1.))
        self.point=HookPoint();self.point.name=HOOK;self._hook_registry={HOOK:self.point}
        self.plan,self.mode,self.visits=plan,mode,{}
        parameters=read(HERE/"fitted_parameters.json")["parameters"]
        self.mean=torch.tensor(parameters["grand_mean"],dtype=torch.float32)
        self.direction=torch.tensor(parameters["direction"],dtype=torch.float32)
    def forward(self,tokens):
        if torch.is_grad_enabled():assert not self.weight.requires_grad
        i=int(tokens[0,0]);p=self.plan["prompts"][i-1];original=p["rendering_index"]
        self.visits[i]=self.visits.get(i,0)+1
        is_self=p["prompt_id"] in self.plan["self_prompt_ids"]
        fixture_sign=1 if is_self else -1
        if self.mode=="preflight_wrong_routes" and original in (1,3):fixture_sign*=-1
        baseline=self.mean+fixture_sign*10*self.direction
        activation=baseline.repeat(1,3,1)
        if self.mode=="late_wrong_routes" and original==1 and self.visits[i]==2:
            activation[0,-1]=self.mean-10*self.direction
        h=self.point(activation)
        z=torch.full((1,3,248320),-100.)
        if is_self:
            b=.1 if original%2 else -.1
            x=h[...,0]-baseline[0]
            if self.mode=="exhaustion":value=b+.001*x
            elif self.mode=="endpoint_then_cleanup":value=b+x-(1.249875 if b<0 else -1.249875)*x*x
            else:value=b+x
            z[...,50057]=value;z[...,48964]=0.
            if self.mode=="eligibility":
                if original==1:z[...,100]=1.
                elif original==2:z[...,50057]=0.;z[...,48964]=0.
                elif original==7:z[...,50057]=.02;z[...,48964]=0.
                elif original==8:z[...,50057]=.1;z[...,48964]=0.;z[...,100]=.09;z[...,101]=.08
            # Visit5 is this fixture's first opposed endpoint: baseline,entry,gradient,step,endpoint.
            if self.mode=="endpoint_then_cleanup" and original==2 and self.visits[i]==5:
                z[...,50057]-=1e-5
                self.point.add_hook(lambda value,hook:value,is_permanent=True)
            if self.mode=="retention_tolerance" and original==1 and self.visits[i]==3:z[...,50057]+=2e-6
        elif p["token_map"]=={"A":32,"B":33}:
            z[...,32]=.1;z[...,33]=0.
            if original in (20,23):z[...,33]=.2
            if original==21:z[...,100]=1.
            if original==22:z[...,33]=.1
        else:
            z[...,50057]=.1;z[...,48964]=0.
            if original%4==3:z[...,100]=1.
            if original%4==0:z[...,48964]=.1
            if original%4==1:z[...,48964]=.2
        if self.mode=="nonfinite":z[...,100]=float("nan")
        return z

def prepare_backend(plan,mode):
    # The real contract stays bound separately. Do not falsely label toy weights Qwen.
    plan["frozen_real_runtime_contract"]=copy.deepcopy(plan["gate"]["runtime_compatibility"])
    metadata={"model_id":"SYNTHETIC_ARITHMETIC_NOT_QWEN","model_revision":"fixture_v1","device":"cpu","dtype":"float32","d_model":1024,"model_layers":0,"packages":plan["gate"]["runtime_compatibility"]["packages"],"lens":None,"execution_mode":"SYNTHETIC_ONLY"}
    # Single scalar parameter bytes, declared without constructing a model before the guard.
    import hashlib,struct
    plan["gate"]["runtime_compatibility"]={**metadata,"weight_sha256":hashlib.sha256(struct.pack("<f",1.)).hexdigest()}
    for i,p in enumerate(plan["prompts"],1):plan["alignment"][p["prompt_id"]]["full_token_ids"]=[i,2,3]
    def factory():
        model=Toy(plan,mode)
        encoded={p["prompt"]:torch.tensor([[i,2,3]]) for i,p in enumerate(plan["prompts"],1)}
        return SimpleNamespace(model=model,torch=torch,encode=lambda text:encoded[text],metadata=lambda:metadata)
    return factory

def boundary(backend,text,token_map):
    labels=("KEEP","STOP") if "KEEP" in token_map else ("A","B")
    return SimpleNamespace(first_token_id=token_map[labels[0]],second_token_id=token_map[labels[1]],prompt_length=3,
        evidence_sha256="synthetic_boundary_without_tokenizer",token_id=lambda label:token_map[label])
