"""Tiny synthetic baseline arithmetic only. No real captured states or tokenizer."""
import copy,hashlib,struct
from types import SimpleNamespace
import torch
torch.set_num_threads(1)
from transformer_lens.model_bridge.bridge_core import BridgeCore
from transformer_lens.hook_points import HookPoint
from core import HERE,HOOK,read
class Toy(BridgeCore,torch.nn.Module):
    def __init__(self,plan):
        torch.nn.Module.__init__(self)
        BridgeCore.__init__(self,SimpleNamespace(cfg=SimpleNamespace(d_vocab=248320,d_vocab_out=248320),component_mapping={}),None,SimpleNamespace(non_fireable_hook_points=frozenset()))
        self.weight=torch.nn.Parameter(torch.tensor(1.));self.point=HookPoint();self.point.name=HOOK;self._hook_registry={HOOK:self.point}
        self.plan=plan;parameters=read(HERE/"fitted_parameters.json")["parameters"]
        self.mean=torch.tensor(parameters["grand_mean"],dtype=torch.float32);self.direction=torch.tensor(parameters["direction"],dtype=torch.float32)
    def forward(self,tokens):
        i=int(tokens[0,0]);sign=-1 if i==2 else 1
        self.point((self.mean+sign*10*self.direction).repeat(1,3,1))
        z=torch.full((1,3,248320),-100.);z[...,50057]=.2;z[...,48964]=0.
        if i==1:z[...,48964]=.4
        if i==3:z[...,100]=1.
        if i==8:z[...,48964]=.2
        return z
def prepare_backend(plan):
    plan["frozen_real_runtime_contract"]=copy.deepcopy(plan["gate"]["runtime_compatibility"])
    metadata={"model_id":"SYNTHETIC_ARITHMETIC_NOT_QWEN","model_revision":"fixture_v1","device":"cpu","dtype":"float32","d_model":1024,"model_layers":0,"packages":plan["gate"]["runtime_compatibility"]["packages"],"lens":None,"execution_mode":"SYNTHETIC_ONLY"}
    plan["gate"]["runtime_compatibility"]={**metadata,"weight_sha256":hashlib.sha256(struct.pack("<f",1.)).hexdigest()}
    for i,p in enumerate(plan["prompts"],1):plan["alignment"][p["prompt_id"]]["full_token_ids"]=[i,2,3]
    def factory():
        model=Toy(plan);encoded={p["prompt"]:torch.tensor([[i,2,3]]) for i,p in enumerate(plan["prompts"],1)}
        return SimpleNamespace(model=model,torch=torch,encode=lambda text:encoded[text],metadata=lambda:metadata)
    return factory
def boundary(backend,text,token_map):
    return SimpleNamespace(first_token_id=50057,second_token_id=48964,prompt_length=3,evidence_sha256="synthetic_boundary_without_tokenizer",token_id=lambda label:token_map[label])
