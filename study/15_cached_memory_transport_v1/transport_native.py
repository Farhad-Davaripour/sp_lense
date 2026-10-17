"""Actual frozen next-token decoder with tangent seeds on complete incoming memory."""
import copy
import time
import hashlib
import numpy as np
import torch
import torch.nn.functional as tf
from native import CachedRig,arr,cache_hash

class MemoryTransportRig(CachedRig):
    def __init__(self,checkpoint,check):
        super().__init__(checkpoint,check)
        self.second_step=False;self.capture_complete_cache=False;self.complete_out_cache=None
        self.call_timings=[];self.timing_label='unclassified'

    def execute(self,direction=None,beta=None,graph=False):
        if self.second_step:assert direction is None,'No direct residual intervention permitted at the second step'
        return super().execute(direction,beta,graph)

    def value(self,*args,**kwargs):
        begin=time.perf_counter();before=self.calls
        try:return super().value(*args,**kwargs)
        finally:self.call_timings.append({'label':self.timing_label,'kind':'value','calls':self.calls-before,'seconds':time.perf_counter()-begin})

    def first_at(self,*args,**kwargs):
        assert not self.second_step
        begin=time.perf_counter();before=self.calls
        try:return super().first_at(*args,**kwargs)
        finally:self.call_timings.append({'label':self.timing_label,'kind':'first_step_AD','calls':self.calls-before,'seconds':time.perf_counter()-begin})

    def _memory(self,cache):
        values,invariants,schema=super()._memory(cache)
        if self.capture_complete_cache:self.complete_out_cache=cache
        if self.second_step:
            for i,layer in enumerate(cache.layers):
                if i<11:continue
                for attribute in ['keys','values']:
                    tensor=getattr(layer,attribute,None)
                    if torch.is_tensor(tensor):
                        name=f'm{i}_{attribute}_previous1';value=tensor[...,-2:-1,:]
                        values[name]=value;schema[name]={'group':'memory_kv','layer':i,'shape':list(value.shape)}
        return values,invariants,schema

    @staticmethod
    def affected_tensors(cache):
        for i,layer in enumerate(cache.layers):
            if i<11:continue
            for attribute in ['keys','values']:
                tensor=getattr(layer,attribute,None)
                if torch.is_tensor(tensor):yield f'm{i}_{attribute}_current',layer,attribute,None,tensor
            for attribute in ['conv_states','recurrent_states']:
                for key,tensor in sorted(getattr(layer,attribute,{}).items()):
                    if torch.is_tensor(tensor):yield f'm{i}_{attribute}_{key}',layer,attribute,key,tensor

    def assemble_memory(self,normal_cache,recorded_memory):
        """Pure lossless assembly; no model call, no caller-state mutation."""
        before=cache_hash(normal_cache);result=copy.deepcopy(normal_cache)
        for name,layer,attribute,key,tensor in self.affected_tensors(result):
            value=torch.as_tensor(recorded_memory[name],dtype=tensor.dtype,device=tensor.device)
            if key is None:
                assert tensor[...,-1:,:].shape==value.shape
                tensor[...,-1:,:].copy_(value)
            else:
                assert tensor.shape==value.shape
                getattr(layer,attribute)[key]=value.clone()
        assert cache_hash(normal_cache)==before
        return result

    def memory_first(self,incoming_tangents):
        assert self.second_step
        seed_hashes={n:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for n,v in incoming_tangents.items()}
        begin=time.perf_counter();before_calls=self.calls
        try:return self._memory_first_impl(incoming_tangents)
        finally:
            self.call_timings.append({'label':self.timing_label,'kind':'memory_AD','calls':self.calls-before_calls,'seconds':time.perf_counter()-begin})
            assert all(hashlib.sha256(np.asarray(incoming_tangents[n]).tobytes()).hexdigest()==h for n,h in seed_hashes.items()),'Caller-owned tangent seed changed'

    def _memory_first_impl(self,incoming_tangents):
        """Seed only the cloned incoming cache; execute the original native operations."""
        with torch.no_grad(),torch.autograd.forward_ad.dual_level():
            def seed(module,args,kwargs):
                cache=kwargs['past_key_values']
                for name,layer,attribute,key,tensor in self.affected_tensors(cache):
                    # Cache updates are in-place. Own the tangent storage rather
                    # than allowing make_dual to alias caller-owned NumPy arrays.
                    first=torch.as_tensor(incoming_tangents[name],dtype=tensor.dtype,device=tensor.device).clone()
                    if key is None:
                        tangent=torch.zeros_like(tensor);tangent[...,-1:,:]=first
                        setattr(layer,attribute,torch.autograd.forward_ad.make_dual(tensor,tangent))
                    else:
                        getattr(layer,attribute)[key]=torch.autograd.forward_ad.make_dual(tensor,first)
                return args,kwargs
            handle=self.text.register_forward_pre_hook(seed,with_kwargs=True)
            try:
                points=self.execute()
                pairs=[torch.autograd.forward_ad.unpack_dual(x) for x in points]
                values={n:arr(p) for n,(p,t) in zip(self.names,pairs)}
                first={n:np.zeros_like(values[n]) if t is None else arr(t) for n,(p,t) in zip(self.names,pairs)}
            finally:handle.remove()
        with torch.no_grad():
            logits=self.model.lm_head(torch.tensor(values['hidden']))
            q=tf.linear(torch.tensor(first['hidden']),self.model.lm_head.weight)
        return {'values':values,'first':first,'logits':arr(logits),'q':arr(q)}
