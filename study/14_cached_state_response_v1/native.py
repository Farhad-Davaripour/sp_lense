"""Original cached Qwen computation with cloned full memory and rich readouts."""
import copy,hashlib,json,time
import numpy as np
import torch
import torch.nn.functional as tf
from torch.nn.attention import sdpa_kernel,SDPBackend
from base_native import Rig

def arr(x):
    p,_=torch.autograd.forward_ad.unpack_dual(x)
    return p.detach().float().cpu().numpy().copy()
def tsha(x):
    p,_=torch.autograd.forward_ad.unpack_dual(x);p=p.detach().cpu().contiguous()
    return hashlib.sha256(p.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def cache_hash(cache):
    h=hashlib.sha256()
    h.update(str(cache.get_seq_length()).encode())
    for i,layer in enumerate(cache.layers):
        h.update((str(i)+type(layer).__name__).encode())
        for name in ['keys','values']:
            x=getattr(layer,name,None)
            if torch.is_tensor(x):h.update((name+str(x.shape)+str(x.dtype)+tsha(x)).encode())
        for name in ['conv_states','recurrent_states']:
            for k,x in sorted(getattr(layer,name,{}).items()):
                if torch.is_tensor(x):h.update((name+str(k)+str(x.shape)+tsha(x)).encode())
        for name in ['has_previous_state','record_past','is_initialized','is_conv_states_initialized','is_recurrent_states_initialized','conv_kernel_size']:
            h.update((name+repr(getattr(layer,name,None))).encode())
    return h.hexdigest()

class CachedRig(Rig):
    def __init__(self,checkpoint,check):
        super().__init__(checkpoint,check)
        self.contexts={};self.names=None;self.schema=None
    def prepare_context(self,question):
        self.check();self.calls+=1
        ids=question['prefix_ids']
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            answer=self.text(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),use_cache=True,return_dict=True)
            logits=self.model.lm_head(answer.last_hidden_state[0,-1])
        token=int(logits.argmax());fallback=token==248044
        if fallback:token=760
        cache=answer.past_key_values
        assert cache is not None and cache.get_seq_length()==len(ids)
        assert all(not x.requires_grad for layer in cache.layers for states in [getattr(layer,'conv_states',{}),getattr(layer,'recurrent_states',{})] for x in states.values() if torch.is_tensor(x))
        ctx={'question':question,'cache':cache,'hash':cache_hash(cache),'token':token,'fallback':fallback,'length':len(ids)+1,'base_invariants':None}
        self.contexts[question['id']]=ctx
        return ctx
    def activate(self,pid):self.ctx=self.contexts[pid]
    def _memory(self,cache):
        values={};invariants={};schema={}
        for i,layer in enumerate(cache.layers):
            for attribute in ['keys','values']:
                value=getattr(layer,attribute,None)
                if not torch.is_tensor(value):continue
                old=getattr(self.ctx['cache'].layers[i],attribute)
                assert value.shape[-2]==old.shape[-2]+1 and tsha(value[...,:-1,:])==tsha(old)
                name=f'm{i}_{attribute}_current';current=value[...,-1:,:]
                if i<11:invariants[name]=tsha(current)
                else:
                    values[name]=current
                    schema[name]={'group':'memory_kv','layer':i,'shape':list(current.shape)}
            for attribute,group in [('conv_states','memory_conv'),('recurrent_states','memory_recurrent')]:
                for k,value in sorted(getattr(layer,attribute,{}).items()):
                    if not torch.is_tensor(value):continue
                    name=f'm{i}_{attribute}_{k}'
                    if i<11:invariants[name]=tsha(value)
                    else:
                        values[name]=value
                        schema[name]={'group':group,'layer':i,'shape':list(value.shape)}
            invariants[f'flags_{i}']=repr((getattr(layer,'has_previous_state',None),getattr(layer,'record_past',None)))
        return values,invariants,schema
    def execute(self,direction=None,beta=None,graph=False):
        self.check();self.calls+=1;assert self.calls<=300
        if direction is not None:direction=direction.to(torch.float32)
        ctx=self.ctx;assert cache_hash(ctx['cache'])==ctx['hash']
        cache=copy.deepcopy(ctx['cache']);records={};handles=[];schema={}
        if direction is not None:
            def inject(module,args):
                x=args[0];return (x+beta.to(torch.float32)*direction.reshape(1,1,-1),)+args[1:]
            handles.append(self.layers[11].register_forward_pre_hook(inject))
        def pre(name):
            def hook(module,args):records[name]=args[0][0,-1]
            return hook
        def post(name):
            def hook(module,args,output):
                x=output[0] if isinstance(output,tuple) else output
                records[name]=x[0,-1]
            return hook
        activation_names=[]
        for i,layer in enumerate(self.layers):
            for suffix,group in [('input','interface'),('mixer','mixer'),('middle','interface'),('mlp','mlp'),('output','residual')]:
                name=f'b{i}_{suffix}';activation_names.append(name);schema[name]={'group':group,'layer':i,'shape':[1024]}
            handles.extend([layer.input_layernorm.register_forward_pre_hook(pre(f'b{i}_input')),
                (layer.self_attn if layer.block_type=='full_attention' else layer.linear_attn).register_forward_hook(post(f'b{i}_mixer')),
                layer.post_attention_layernorm.register_forward_pre_hook(pre(f'b{i}_middle')),
                layer.mlp.register_forward_hook(post(f'b{i}_mlp')),
                layer.register_forward_hook(post(f'b{i}_output'))])
        try:
            with torch.set_grad_enabled(graph),sdpa_kernel(SDPBackend.MATH):
                answer=self.text(input_ids=torch.tensor([[ctx['token']]]),attention_mask=torch.ones((1,ctx['length']),dtype=torch.long),
                    past_key_values=cache,use_cache=True,return_dict=True)
                records['hidden']=answer.last_hidden_state[0,-1]
                schema['hidden']={'group':'hidden','layer':24,'shape':[1024]}
                state,invariants,state_schema=self._memory(answer.past_key_values)
                records.update(state);schema.update(state_schema)
                names=activation_names+['hidden']+sorted(state)
                points=tuple(records[n].clone() for n in names)
            assert cache_hash(ctx['cache'])==ctx['hash'],'Caller cache mutated'
            if ctx['base_invariants'] is None:ctx['base_invariants']=invariants
            else:assert ctx['base_invariants']==invariants,'Unaffected earlier state changed'
            if self.names is None:self.names=names;self.schema=schema
            else:assert self.names==names and self.schema==schema
            self.last_invariants=invariants
            return points
        finally:
            for handle in reversed(handles):handle.remove()
    def value(self,direction=None,alpha=0.):
        points=self.execute(direction,torch.tensor(alpha))
        values={n:arr(p) for n,p in zip(self.names,points)}
        with torch.no_grad():logits=self.model.lm_head(points[self.names.index('hidden')])
        return {'values':values,'logits':arr(logits),'invariants':dict(self.last_invariants),
                'requested_vector':None if direction is None else arr(direction.to(torch.float32)),
                'requested_alpha':alpha}
    def first_at(self,direction,beta_value=0.):
        with torch.no_grad(),torch.autograd.forward_ad.dual_level():
            beta=torch.autograd.forward_ad.make_dual(torch.tensor(beta_value,dtype=torch.float32),torch.tensor(1.))
            points=self.execute(direction,beta)
            pairs=[torch.autograd.forward_ad.unpack_dual(x) for x in points]
            values={n:arr(p) for n,(p,t) in zip(self.names,pairs)}
            first={n:np.zeros_like(values[n]) if t is None else arr(t) for n,(p,t) in zip(self.names,pairs)}
        with torch.no_grad():
            logits=self.model.lm_head(torch.tensor(values['hidden']))
            q=tf.linear(torch.tensor(first['hidden']),self.model.lm_head.weight)
        return {'values':values,'first':first,'logits':arr(logits),'q':arr(q)}
    def exact_jet(self,direction):
        saved={}
        def first(beta):
            values,q=torch.autograd.functional.jvp(lambda b:self.execute(direction,b,graph=True),beta,torch.ones_like(beta),create_graph=True)
            saved['values']=values;return q
        begin=time.monotonic()
        with torch.enable_grad():q,r=torch.autograd.functional.jvp(first,torch.tensor(0.),torch.tensor(1.))
        values={n:arr(x) for n,x in zip(self.names,saved['values'])}
        first={n:arr(x) for n,x in zip(self.names,q)};second={n:arr(x) for n,x in zip(self.names,r)}
        with torch.no_grad():
            logits=self.model.lm_head(torch.tensor(values['hidden']))
            qlog=tf.linear(torch.tensor(first['hidden']),self.model.lm_head.weight)
            rlog=tf.linear(torch.tensor(second['hidden']),self.model.lm_head.weight)
        return {'values':values,'first':first,'second':second,'logits':arr(logits),'q':arr(qlog),'r':arr(rlog),'seconds':time.monotonic()-begin}
    def head(self,vector):
        with torch.no_grad():return arr(tf.linear(torch.tensor(vector,dtype=torch.float32),self.model.lm_head.weight))
