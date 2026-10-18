"""Offline state replacement controls. No teacher cache is injected into a student."""
import contextlib


@contextlib.contextmanager
def base_mode(model):
    try:
        with model.disable_adapter():
            model.requires_grad_(False)
            yield
    finally:model.requires_grad_(False)


def replace_last(value, reference):
    import torch
    if value[:,-1:,:].shape!=reference.shape or not torch.isfinite(reference).all():
        raise ValueError('State replacement shape or finiteness differs')
    result=value.clone();result[:,-1:,:]=reference.to(device=value.device,dtype=value.dtype)
    return result


class StateCapture:
    """Capture full teacher states, retaining CPU clones for the current query only."""
    def __init__(self,layers,sites):
        self.saved=[];self.current={}
        for site in sites:
            def mid(module,args,_site=site):
                self.current[f'{_site}/attention']=args[0][:,-1:,:].detach().float().cpu().clone()
            def output(module,args,value,_site=site):
                hidden=value[0] if isinstance(value,(tuple,list)) else value
                self.current[f'{_site}/output']=hidden[:,-1:,:].detach().float().cpu().clone()
            self.saved.extend([layers[site].post_attention_layernorm.register_forward_pre_hook(mid),
                               layers[site].register_forward_hook(output)])
    def close(self):
        for h in self.saved:h.remove()
        self.saved=[]


class SingleStateClamp:
    """Use the same causal positions as the single-site learned map."""
    def __init__(self,layer,site,references):
        self.site,self.references=site,references;self.current=0;self.enabled=False;self.fires=0
        self.handle=layer.register_forward_hook(self.hook)
    def prepare(self,j,kinds):
        self.current=j;self.enabled=bool(kinds);self.fires=0
    def hook(self,module,args,value):
        self.fires+=1
        if self.fires!=1:raise ValueError('Repeated state replacement')
        if not self.enabled:return value
        hidden=value[0] if isinstance(value,(tuple,list)) else value
        changed=replace_last(hidden,self.references[self.current][f'{self.site}/output'])
        if isinstance(value,tuple):return (changed,)+value[1:]
        if isinstance(value,list):return [changed]+value[1:]
        return changed
    def close(self):self.handle.remove()


class SixStateClamp:
    """Reproduce native two-boundary residual algebra, replacing current states only."""
    def __init__(self,layers,sites,references):
        import types
        self.references=references;self.current=0;self.saved=[];self.fires={}
        for site in sites:
            layer=layers[site];self.saved.append((layer,layer.forward))
            def replacement(module,hidden_states,position_embeddings,attention_mask=None,position_ids=None,
                            past_key_values=None,_site=site,**kwargs):
                from native_sites import attention,mlp_residual
                self.fires[_site]=self.fires.get(_site,0)+1
                if self.fires[_site]!=1:raise ValueError('Repeated native state replacement')
                mid=hidden_states+attention(module,hidden_states,position_embeddings,attention_mask,
                                            position_ids,past_key_values,kwargs)
                mid=replace_last(mid,self.references[self.current][f'{_site}/attention'])
                out=mlp_residual(module,mid)
                return replace_last(out,self.references[self.current][f'{_site}/output'])
            layer.forward=types.MethodType(replacement,layer)
    def prepare(self,j):self.current=j;self.fires={}
    def close(self):
        for layer,forward in self.saved:layer.forward=forward
        self.saved=[]
