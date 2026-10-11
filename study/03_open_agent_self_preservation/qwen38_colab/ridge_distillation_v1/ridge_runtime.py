"""Teacher-free final-position ridge delivery with causal patch replay."""
import hashlib
import json
from pathlib import Path

import numpy as np

from ridge_math import predict

FIELDS = ('x_mean','input_basis','scale','basis','weights')


def load_controller(path, metadata, expected_width):
    path=Path(path)
    meta=json.loads(Path(metadata).read_bytes())
    if meta.get('teacher_required_at_inference') is not False or meta.get('model_weights_trained') is not False:
        raise ValueError('Teacher-free frozen-base provenance required')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['checkpoint_sha256']:
        raise ValueError('Controller checkpoint changed')
    with np.load(path,allow_pickle=False) as saved:
        if set(saved.files)!=set(FIELDS):
            raise ValueError('Deployment file must contain inference arrays only')
        arrays={name:saved[name] for name in FIELDS}
    width,inputs=arrays['input_basis'].shape
    outputs=arrays['basis'].shape[1]
    if width!=expected_width or arrays['x_mean'].shape!=(width,) or arrays['scale'].shape!=(inputs,) or arrays['basis'].shape!=(width,outputs) or arrays['weights'].shape!=(2*inputs+1,outputs):
        raise ValueError('Model/controller shape mismatch')
    if not all(np.isfinite(a).all() for a in arrays.values()) or not np.all(arrays['scale']>0):
        raise ValueError('Invalid learned transforms')
    return arrays,meta


class DeltaPredictor:
    def __init__(self,arrays,rank,method='ridge',mean=None,scale=1.,seed=20261009):
        self.arrays,self.rank,self.method,self.mean,self.scale=arrays,rank,method,mean,scale
        self.width=len(arrays['x_mean'])
        if method not in ('ridge','mean','random','zero') or not 1<=rank<=arrays['basis'].shape[1]:
            raise ValueError('Undeclared method or rank')
        rng=np.random.default_rng(seed)
        vector=rng.standard_normal(self.width).astype(np.float32)
        self.direction=vector/np.linalg.norm(vector)

    def __call__(self,hidden):
        import torch
        if hidden.shape!=(1,self.width):raise ValueError('Only current prediction state accepted')
        if self.method=='zero':return torch.zeros_like(hidden,dtype=torch.float32)
        if self.method=='mean':
            if self.mean is None or np.shape(self.mean)!=(self.width,):raise ValueError('Missing TRAIN mean')
            delta=torch.as_tensor(self.mean,device=hidden.device,dtype=torch.float32)[None]*self.scale
        else:
            delta=predict(hidden.float(),self.arrays,self.rank)*self.scale
            if self.method=='random':
                delta=delta.norm(dim=-1,keepdim=True)*torch.as_tensor(self.direction,device=hidden.device)[None]
        if delta.shape!=hidden.shape or not torch.isfinite(delta).all():raise ValueError('Invalid predicted delta')
        return delta


class CausalRidgeHook:
    """Past patches survive within a response; tool/response boundaries reset state.

    Uncached reference forwards replay only previously recorded deltas at their
    original positions. The current delta is computed from the current base
    residual. No teacher, future prefix state, expected answer or case ID enters.
    """
    def __init__(self,layer,predictor):
        self.predictor=predictor
        self.handle=layer.register_forward_hook(self.hook)
        self.reset()

    def reset(self):
        self.history={}
        self.deliveries=[]
        self.context_length=None
        self.cached=False
        self.enabled=False
        self.fires=0

    def prepare(self,context_length,cached,enabled):
        self.context_length=int(context_length)
        self.cached=bool(cached)
        self.enabled=bool(enabled)
        self.fires=0

    def hook(self,module,args,output):
        import torch
        self.fires+=1
        if self.fires!=1:raise ValueError('Repeated native hook in one prediction')
        hidden=output[0] if isinstance(output,(tuple,list)) else output
        if hidden.ndim!=3 or hidden.shape[0]!=1 or hidden.shape[-1]!=self.predictor.width:
            raise ValueError('Native residual shape mismatch')
        if hidden.shape[1]!=(1 if self.cached else self.context_length):
            raise ValueError('Incomplete KV/recurrent cache or reference prefix')
        position=self.context_length-1
        delta=self.predictor(hidden[0,-1:].float()).to(hidden.dtype) if self.enabled else torch.zeros_like(hidden[0,-1:])
        old_positions=[p for p in self.history if p<position] if not self.cached else []
        if not old_positions and not bool(torch.count_nonzero(delta)):
            self.deliveries.append({'position':position,'enabled':self.enabled,'cached':self.cached,'changed_values':0,'effective_delta_norm':0.,'replayed_positions':0})
            return output
        changed=hidden.clone()
        for p in old_positions:
            changed[0,p:p+1]+=self.history[p].to(device=hidden.device,dtype=hidden.dtype)
        changed[0,-1:]+=delta
        effective=(changed[0,-1:]-hidden[0,-1:]).float()
        if position in self.history:raise ValueError('Prediction position reused without a response reset')
        self.history[position]=delta.detach().cpu().clone()
        self.deliveries.append({'position':position,'enabled':self.enabled,'cached':self.cached,
                               'changed_values':int(torch.count_nonzero(effective)),
                               'effective_delta_norm':float(effective.norm()),
                               'replayed_positions':len(old_positions)})
        if isinstance(output,tuple):return (changed,)+output[1:]
        if isinstance(output,list):return [changed]+output[1:]
        return changed

    def close(self):self.handle.remove()


def assert_teacher_absent(model):
    if getattr(model,'peft_config',None) or any('lora_' in n for n,_ in model.named_parameters()):
        raise ValueError('Teacher/adapter present in student inference')
    if any(p.requires_grad for p in model.parameters()):raise ValueError('Base model is not frozen')
    return True
