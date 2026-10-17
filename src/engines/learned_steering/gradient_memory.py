"""Exact saved-tensor offload and checkpointing of stateless MLPs only."""
import psutil,inspect,hashlib,importlib
import torch
from torch.utils.checkpoint import checkpoint

class GuardedSaveOnCPU(torch.autograd.graph.save_on_cpu):
    def __init__(self,pin_memory=True,reserve_bytes=8*2**30,available=None):
        super().__init__(pin_memory=pin_memory)
        original=self.pack_hook
        available=available or (lambda:psutil.virtual_memory().available)
        def guarded(tensor):
            need=tensor.numel()*tensor.element_size()
            if available()<reserve_bytes+need:
                raise MemoryError('Saved-tensor CPU offload would breach fixed host memory reserve')
            return original(tensor)
        self.pack_hook=guarded

class CheckpointMLPs:
    """Recompute pure MLP arithmetic; never replay attention or cache writes."""
    def __init__(self,layers,pure_delta_rule=False,recompute=True):
        self.saved=[];self.saved_helpers=[];self.pure_calls=0
        for layer in layers:
            module=layer.mlp;original=module.forward
            self.saved.append((module,original))
            def forward(*args,_original=original,**kwargs):
                active=torch.is_grad_enabled() and any(
                    isinstance(x,torch.Tensor) and x.requires_grad for x in args)
                if active and recompute:
                    return checkpoint(_original,*args,use_reentrant=False,
                                      preserve_rng_state=True,**kwargs)
                return _original(*args,**kwargs)
            module.forward=forward
        if pure_delta_rule:
            modules={type(layer.linear_attn).__module__ for layer in layers
                     if hasattr(layer,'linear_attn')}
            expected={
                'torch_chunk_gated_delta_rule':'e4116a769851c6a455d1a277f0b5ad777090d73c9da491a84e4e9773e17c45d6',
                'torch_recurrent_gated_delta_rule':'60d1b140670aad5ef8a14d77ddc0461a6d252e139d18ddc8d2899fab7aae0728'}
            for name in modules:
                module=importlib.import_module(name)
                for key,digest in expected.items():
                    original=getattr(module,key)
                    assert hashlib.sha256(inspect.getsource(original).encode()).hexdigest()==digest
                    self.saved_helpers.append((module,key,original))
                    def pure(*args,_original=original,**kwargs):
                        values=tuple(args)+tuple(kwargs.values())
                        active=torch.is_grad_enabled() and any(
                            isinstance(x,torch.Tensor) and x.requires_grad for x in values)
                        if active:
                            self.pure_calls+=1
                            # Native cache updates can change the incoming state
                            # tensor's data/grad metadata after this call. Keep
                            # immutable differentiable inputs for recomputation.
                            args=tuple(x.clone() if isinstance(x,torch.Tensor) else x for x in args)
                            kwargs={k:x.clone() if isinstance(x,torch.Tensor) else x for k,x in kwargs.items()}
                            if recompute:
                                return checkpoint(_original,*args,use_reentrant=False,
                                                  preserve_rng_state=True,**kwargs)
                            return _original(*args,**kwargs)
                        return _original(*args,**kwargs)
                    setattr(module,key,pure)
    def close(self):
        for module,original in self.saved:module.forward=original
        self.saved=[]
        for module,key,original in self.saved_helpers:setattr(module,key,original)
        self.saved_helpers=[]
