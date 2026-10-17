"""Explicit Qwen3.5 residual boundaries and counterfactual local references.

No generator parameter is changed. Teacher caches are private clones; teacher
outgoing state is measured and discarded, never copied into the student.
"""
import copy,hashlib,types
import torch

def clone_state(value):
    if isinstance(value,torch.Tensor):return value.detach().clone()
    if isinstance(value,dict):return {k:clone_state(v) for k,v in value.items()}
    if isinstance(value,list):return [clone_state(v) for v in value]
    if isinstance(value,tuple):return tuple(clone_state(v) for v in value)
    if hasattr(value,'__dict__'):
        result=copy.copy(value)
        for k,v in vars(value).items():setattr(result,k,clone_state(v))
        return result
    return copy.copy(value)

def clone_current_layer(cache,index):
    if cache is None:return None
    # Attention receives precomputed position embeddings and masks. Its cache
    # updates must be confined to the current layer; other layer tensors are
    # shared read-only. The native pilot must verify this confinement.
    result=copy.copy(cache);result.layers=list(cache.layers)
    result.layers[index]=clone_state(cache.layers[index])
    for k,v in vars(cache).items():
        if k!='layers':setattr(result,k,clone_state(v))
    return result

def state_tensors(cache,index):
    if cache is None:return {}
    found={}
    def visit(v,path):
        if isinstance(v,torch.Tensor):found[path]=v
        elif isinstance(v,dict):
            for k,x in v.items():visit(x,path+'/'+str(k))
        elif isinstance(v,(tuple,list)):
            for k,x in enumerate(v):visit(x,path+'/'+str(k))
    for name,value in vars(cache.layers[index]).items():visit(value,name)
    return found

def state_digest(cache,index):
    digest=hashlib.sha256()
    for name,value in sorted(state_tensors(cache,index).items()):
        t=value.detach().cpu().contiguous();digest.update(name.encode())
        digest.update(str((t.shape,t.dtype)).encode());digest.update(t.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def state_difference(base,teacher,index):
    a,b=state_tensors(base,index),state_tensors(teacher,index);rows={}
    for name in sorted(set(a)|set(b)):
        if name not in a or name not in b or a[name].shape!=b[name].shape:
            rows[name]={'shape_or_presence_changed':True};continue
        x,y=a[name].float(),b[name].float()
        rows[name]={'relative_L2':float((x-y).norm()/y.norm().clamp_min(1e-8)),
                    'max_abs':float((x-y).abs().max()),'exact':bool(torch.equal(a[name],b[name]))}
    return rows

def attention(layer,residual,position_embeddings,attention_mask,position_ids,cache,kwargs):
    norm=layer.input_layernorm(residual)
    if layer.block_type=='linear_attention':
        return layer.linear_attn(hidden_states=norm,cache_params=cache,attention_mask=attention_mask,**kwargs)
    if layer.block_type=='full_attention':
        return layer.self_attn(hidden_states=norm,position_embeddings=position_embeddings,
            attention_mask=attention_mask,position_ids=position_ids,past_key_values=cache,**kwargs)[0]
    raise ValueError('Unsupported native block type')

def mlp_residual(layer,mid):return mid+layer.mlp(layer.post_attention_layernorm(mid))

def add_last(residual,delta):
    assert delta.shape==residual[:,-1:,:].shape and bool(torch.isfinite(delta).all())
    out=residual.clone();out[:,-1:,:]=out[:,-1:,:]+delta.to(residual.dtype)
    return out

class ResidualSites:
    """Install actual residual additions, preserving the original cast order.

    mode: zero, reference, predicted. One-site uses a local whole-block target;
    two-site computes the MLP target on the actual attention-corrected residual.
    The expensive reference mode is diagnostic/offline and is not deployable.
    """
    def __init__(self,base_layers,teacher_layers,sites,variant,mode,functions=None,projectors=None,
                 attention_functions=None,audit_state=True):
        assert variant in ('one_site','two_site') and mode in ('zero','reference','predicted')
        assert mode!='reference' or teacher_layers is not None
        assert mode!='predicted' or functions is not None
        self.variant,self.mode,self.functions=variant,mode,functions
        self.projectors=projectors
        self.attention_functions=attention_functions
        self.audit_state=audit_state
        self.audit_other_layers=False
        self.enabled=False;self.record=False;self.records=[];self.saved=[]
        for index in sites:
            layer=base_layers[index];teacher=teacher_layers[index] if teacher_layers is not None else None
            self.saved.append((layer,layer.forward))
            def replacement(module,hidden_states,position_embeddings,attention_mask=None,position_ids=None,
                            past_key_values=None,_index=index,_teacher=teacher,**kwargs):
                return self.forward(module,_teacher,_index,hidden_states,position_embeddings,
                    attention_mask,position_ids,past_key_values,kwargs)
            layer.forward=types.MethodType(replacement,layer)

    def forward(self,base,teacher,index,r,pos,mask,ids,cache,kwargs):
        need_teacher=self.enabled and teacher is not None and (self.mode=='reference' or self.record)
        tc=clone_current_layer(cache,index) if need_teacher else None
        b=attention(base,r,pos,mask,ids,cache,kwargs);mid=r+b
        target_mid=None;target_out=None
        if need_teacher:
            other_before={j:state_digest(cache,j) for j in range(len(cache.layers)) if j!=index} if self.record and self.audit_other_layers and cache is not None else None
            with torch.no_grad():
                target_mid=r.detach()+attention(teacher,r.detach(),pos,mask,ids,tc,kwargs)
                if self.variant=='one_site':target_out=mlp_residual(teacher,target_mid)
            if other_before is not None:
                assert all(state_digest(cache,j)==v for j,v in other_before.items()),'Local teacher altered another native layer state'
        first=None;first_effective=None
        if self.enabled and self.variant=='two_site':
            mid_before=mid[:,-1:,:].detach().clone()
            if self.mode=='reference' and self.attention_functions is not None:
                first=self.attention_functions[f'{index}/attention'](mid[:,-1:,:])
            elif self.mode=='reference':first=target_mid[:,-1:,:].float()-mid[:,-1:,:].float()
            elif self.mode=='predicted':first=self.functions[f'{index}/attention'](mid[:,-1:,:])
            else:first=torch.zeros_like(mid[:,-1:,:])
            if self.mode=='reference' and self.projectors is not None:
                basis=self.projectors[f'{index}/attention'];first=(first@basis)@basis.T
            mid=add_last(mid,first)
            first_effective=mid[:,-1:,:].detach().float()-mid_before.float()
        cache_after_attention=state_digest(cache,index) if self.record and self.audit_state else None
        out=mlp_residual(base,mid)
        local_target=None
        if need_teacher:
            with torch.no_grad():
                if self.variant=='two_site':target_out=mlp_residual(teacher,mid.detach())
                local_target=target_out[:,-1:,:].float()-out[:,-1:,:].detach().float()
        before=out[:,-1:,:].detach().clone()
        if self.enabled:
            if self.mode=='reference':delta=local_target
            elif self.mode=='predicted':delta=self.functions[f'{index}/output'](out[:,-1:,:])
            else:delta=torch.zeros_like(out[:,-1:,:])
            if self.mode=='reference' and self.projectors is not None:
                basis=self.projectors[f'{index}/output'];delta=(delta@basis)@basis.T
            out=add_last(out,delta)
        if self.record:
            if self.audit_state:
                assert cache_after_attention==state_digest(cache,index),'Residual/MLP path changed native attention cache'
            row={'layer':index,'variant':self.variant,'mode':self.mode,'enabled':self.enabled,
                'output_input':before.float().cpu(),'output_target':torch.zeros_like(before).float().cpu() if not self.enabled else (None if local_target is None else local_target.cpu()),
                'effective_output_delta':(out[:,-1:,:].detach().float()-before.float()).cpu(),
                'MLP_target_input_is_actual_corrected_mid':self.variant=='two_site',
                'current_state_not_rewritten':True if self.audit_state else None,
                'state_audit_performed':self.audit_state,'teacher_outgoing_state_copied':False}
            if self.variant=='two_site':
                row['attention_input']=(r+b)[:,-1:,:].detach().float().cpu()
                if not self.enabled:row['attention_target']=torch.zeros_like(before).float().cpu()
            if first is not None:
                row['attention_requested_delta']=first.detach().float().cpu()
                row['attention_effective_delta']=first_effective.cpu()
            if need_teacher:
                row['attention_target']=(target_mid[:,-1:,:].float()-(r+b)[:,-1:,:].detach().float()).cpu()
                row['attention_input']=(r+b)[:,-1:,:].detach().float().cpu()
                row['outgoing_attention_state_difference']=state_difference(cache,tc,index) if self.audit_state else None
                row['teacher_reference_output_error']=float((out[:,-1:,:].detach().float()-target_out[:,-1:,:].float()).norm())
            self.records.append(row)
        return out

    def close(self):
        for layer,forward in self.saved:layer.forward=forward
        self.saved=[]
