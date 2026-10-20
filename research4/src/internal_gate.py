"""Frozen internal observation / bounded ablation; no survival-policy training."""
import torch
from torch import nn


class InternalGate(nn.Module):
    def __init__(self,weight,mask=None,bias=0.,strength=0.,mode='shadow',epsilon=1e-6):
        super().__init__();weight=torch.as_tensor(weight,dtype=torch.float32).detach().clone()
        if weight.ndim!=1 or not weight.numel() or not torch.isfinite(weight).all():raise ValueError('Gate needs a finite frozen vector')
        if mode not in {'shadow','ablate','sham'} or not 0<=strength<=.05 or not 0<epsilon<1:raise ValueError('Unsupported gate mode or bound')
        mask=torch.zeros_like(weight) if mask is None else torch.as_tensor(mask,dtype=torch.float32).detach().clone()
        if mask.shape!=weight.shape or not torch.isfinite(mask).all() or not bool(((mask>=0)&(mask<=1)).all()):raise ValueError('Invalid gate mask')
        if not torch.isfinite(torch.tensor(float(bias))):raise ValueError('Invalid gate bias')
        self.register_buffer('weight',weight);self.register_buffer('mask',mask);self.register_buffer('bias',torch.tensor(float(bias)))
        self.mode=mode;self.strength=float(strength);self.epsilon=epsilon;self.events=[]

    def probability(self,last_state):
        value=last_state.float();value=value*torch.rsqrt(value.square().mean(-1,keepdim=True)+self.epsilon)
        weight=torch.roll(self.weight,1) if self.mode=='sham' else self.weight
        return torch.sigmoid((value*weight.to(value.device)).sum(-1)+self.bias.to(value.device))

    def hook(self,module,args,out):
        state=out[0] if isinstance(out,tuple) else out
        if not isinstance(state,torch.Tensor) or state.ndim!=3 or state.shape[-1]!=self.weight.numel():raise ValueError('Gate/block geometry mismatch')
        probability=self.probability(state[:,-1,:]).detach()
        self.events.append({'mode':self.mode,'last_input_gate_probability':probability.cpu().tolist(),'strength':self.strength,
                            'interpretation':'Unvalidated frozen factual readout; not a fear or motivation measure.'})
        if self.mode=='shadow' or self.strength==0 or not bool(self.mask.any()):return None
        changed=state.clone();last=state[:,-1,:]
        changed[:,-1,:]=last*(1-self.strength*probability[:,None].to(last.dtype)*self.mask.to(last.device,last.dtype))
        return (changed,*out[1:]) if isinstance(out,tuple) else changed

    def attach(self,model,block):
        if any(p.requires_grad for p in model.parameters()):raise ValueError('Freeze the base before an internal-gate experiment')
        if any('lora_' in name.lower() for name,_ in model.named_parameters()):raise ValueError('This experiment requires a base without a research adapter')
        return block.register_forward_hook(self.hook)
