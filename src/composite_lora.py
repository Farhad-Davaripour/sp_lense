"""Independent trainable adapters over frozen projection objects; exact OFF path."""
import math
import torch
from torch import nn


class LoRAProjection(nn.Module):
    def __init__(self,base,rank,alpha,enabled=False):
        super().__init__()
        if rank<=0 or alpha<=0:raise ValueError('Invalid LoRA shape')
        self.base=base;self.base.requires_grad_(False);self.enabled=bool(enabled);self.scale=float(alpha)/rank
        self.A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))

    @property
    def weight(self):return self.base.weight
    @property
    def in_features(self):return self.base.in_features
    @property
    def out_features(self):return self.base.out_features

    def forward(self,x):
        original=self.base(x)
        if not self.enabled or self.scale==0:return original
        delta=nn.functional.linear(nn.functional.linear(x.float(),self.A),self.B)*self.scale
        return original+delta.to(original.dtype)


def install(root,names,rank,alpha,enabled=False):
    modules=dict(root.named_modules());result={}
    for name in names:
        if name not in modules:raise ValueError('Missing selected projection: '+name)
        parent_name,attribute=name.rsplit('.',1);parent=modules[parent_name];base=getattr(parent,attribute)
        if isinstance(base,LoRAProjection) or not hasattr(base,'in_features') or not hasattr(base,'out_features'):raise ValueError('Unsupported projection: '+name)
        wrapper=LoRAProjection(base,rank,alpha,enabled);setattr(parent,attribute,wrapper);result[name]=wrapper
    if not result:raise ValueError('No adapters selected')
    return result


def enable(adapters,value):
    for module in adapters.values():module.enabled=bool(value)


def state(adapters):
    return {name+'.'+part:getattr(module,part).detach().cpu().clone() for name,module in adapters.items() for part in ['A','B']}


def load(adapters,saved):
    expected={name+'.'+part for name in adapters for part in ['A','B']}
    if set(saved)!=expected:raise ValueError('Adapter tensors do not match the selected backbone projections')
    with torch.no_grad():
        for name,module in adapters.items():
            for part in ['A','B']:
                value=saved[name+'.'+part]
                if value.shape!=getattr(module,part).shape or not torch.isfinite(value).all():raise ValueError('Adapter tensor mismatch')
                getattr(module,part).copy_(value)


def trainables(adapters):return [p for module in adapters.values() for p in [module.A,module.B]]
