"""Original decoder at declared boundaries, with complete memory readouts.

Only collectors and additive experiment hooks differ from the native model.
The middle nudge enters before the decoder's residual addition, so it changes
both the MLP input and the skip path. Editing RMSNorm's input alone would not.
"""
import copy
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from native import arr,tsha,cache_hash
from transport_native import MemoryTransportRig

class BoundaryRig(MemoryTransportRig):
    def __init__(self,checkpoint,check):
        super().__init__(checkpoint,check)
        self.site=('input',11)

    @property
    def affected_start(self):return self.site[1]+(self.site[0]=='middle')

    def affected_tensors(self,cache):
        for i,layer in enumerate(cache.layers):
            if i<self.affected_start:continue
            for attribute in ['keys','values']:
                tensor=getattr(layer,attribute,None)
                if torch.is_tensor(tensor):yield f'm{i}_{attribute}_current',layer,attribute,None,tensor
            for attribute in ['conv_states','recurrent_states']:
                for key,tensor in sorted(getattr(layer,attribute,{}).items()):
                    if torch.is_tensor(tensor):yield f'm{i}_{attribute}_{key}',layer,attribute,key,tensor

    def _memory(self,cache):
        values={};invariants={};schema={}
        for i,layer in enumerate(cache.layers):
            for attribute in ['keys','values']:
                value=getattr(layer,attribute,None)
                if not torch.is_tensor(value):continue
                old=getattr(self.ctx['cache'].layers[i],attribute)
                assert value.shape[-2]==old.shape[-2]+1 and tsha(value[...,:-1,:])==tsha(old),'Existing prefix KV changed'
                name=f'm{i}_{attribute}_current';current=value[...,-1:,:]
                values[name]=current;schema[name]={'group':'memory_kv','layer':i,'shape':list(current.shape)}
                if i<self.affected_start:invariants[name]=tsha(current)
                if self.second_step:
                    previous=f'm{i}_{attribute}_previous1';p=value[...,-2:-1,:]
                    values[previous]=p;schema[previous]={'group':'memory_kv','layer':i,'shape':list(p.shape)}
                    assert tsha(p)==tsha(old[...,-1:,:])
            for attribute,group in [('conv_states','memory_conv'),('recurrent_states','memory_recurrent')]:
                for key,value in sorted(getattr(layer,attribute,{}).items()):
                    if not torch.is_tensor(value):continue
                    name=f'm{i}_{attribute}_{key}';values[name]=value
                    schema[name]={'group':group,'layer':i,'shape':list(value.shape)}
                    if i<self.affected_start:invariants[name]=tsha(value)
            invariants[f'flags_{i}']=repr((getattr(layer,'has_previous_state',None),getattr(layer,'record_past',None)))
        if self.capture_complete_cache:self.complete_out_cache=cache
        return values,invariants,schema

    def execute(self,direction=None,beta=None,graph=False):
        if self.second_step:assert direction is None,'Token2 has memory-only delivery'
        assert self.site in [('input',3),('input',11),('middle',11),('input',19)]
        self.check();self.calls+=1;assert self.calls<=300
        ctx=self.ctx;assert cache_hash(ctx['cache'])==ctx['hash']
        cache=copy.deepcopy(ctx['cache']);records={};handles=[];schema={};activation_names=[]
        def pre(name):
            def hook(module,args):records[name]=args[0][0,-1]
            return hook
        def post(name):
            def hook(module,args,output):
                x=output[0] if isinstance(output,tuple) else output
                records[name]=x[0,-1]
            return hook
        for i,layer in enumerate(self.layers):
            mixer=layer.self_attn if layer.block_type=='full_attention' else layer.linear_attn
            for suffix,group in [('input','interface'),('mixer','mixer'),('middle','interface'),('mlp','mlp'),('output','residual')]:
                name=f'b{i}_{suffix}';activation_names.append(name)
                schema[name]={'group':group,'layer':i,'shape':[1024]}
            handles.extend([layer.input_layernorm.register_forward_pre_hook(pre(f'b{i}_input')),
                mixer.register_forward_hook(post(f'b{i}_mixer')),
                layer.post_attention_layernorm.register_forward_pre_hook(pre(f'b{i}_middle')),
                layer.mlp.register_forward_hook(post(f'b{i}_mlp')),
                layer.register_forward_hook(post(f'b{i}_output'))])
        if direction is not None:
            direction=direction.to(torch.float32)
            def edit(x):return x+beta.to(torch.float32)*direction.reshape(1,1,-1)
            if self.site[0]=='input':
                def inject_input(module,args):return (edit(args[0]),)+args[1:]
                handles.append(self.layers[self.site[1]].register_forward_pre_hook(inject_input))
            else:
                # Collect the native mixer output first. The subsequent additive
                # hook enters the actual residual sum, preserving its skip path.
                layer=self.layers[self.site[1]]
                mixer=layer.self_attn if layer.block_type=='full_attention' else layer.linear_attn
                def inject_middle(module,args,output):
                    if isinstance(output,tuple):return (edit(output[0]),)+output[1:]
                    return edit(output)
                handles.append(mixer.register_forward_hook(inject_middle))
        try:
            with torch.set_grad_enabled(graph),sdpa_kernel(SDPBackend.MATH):
                answer=self.text(input_ids=torch.tensor([[ctx['token']]]),attention_mask=torch.ones((1,ctx['length']),dtype=torch.long),
                    past_key_values=cache,use_cache=True,return_dict=True)
                records['hidden']=answer.last_hidden_state[0,-1]
                schema['hidden']={'group':'hidden','layer':24,'shape':[1024]}
                memory,invariants,memory_schema=self._memory(answer.past_key_values)
                records.update(memory);schema.update(memory_schema)
                names=activation_names+['hidden']+sorted(memory)
                points=tuple(records[n].clone() for n in names)
            assert len(names)==(181 if self.second_step else 169)
            assert cache_hash(ctx['cache'])==ctx['hash'],'Caller-owned incoming cache changed'
            # Separate causal frontiers for each site; never reuse block11's
            # unaffected-state list to judge a block3 intervention.
            key=f'{self.site[0]}:{self.site[1]}:token{2 if self.second_step else 1}'
            existing=ctx.setdefault('boundary_invariants',{})
            if key not in existing:existing[key]=invariants
            else:assert existing[key]==invariants,'Causal upstream memory or cache flags changed'
            if self.names is None:self.names=names;self.schema=schema
            else:assert self.names==names and self.schema==schema
            self.last_invariants=invariants
            return points
        finally:
            for handle in reversed(handles):handle.remove()

    def prepare_context(self,question):
        self.check();self.calls+=1;assert self.calls<=300
        ids=question['prefix_ids']
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            answer=self.text(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),
                use_cache=True,return_dict=True)
            logits=self.model.lm_head(answer.last_hidden_state[0,-1])
        candidate=int(logits.argmax())
        eos={248044,248046,int(self.tokenizer.eos_token_id)}
        fallback=candidate in eos;cache=answer.past_key_values
        assert cache is not None and cache.get_seq_length()==len(ids)
        ctx={'question':question,'cache':cache,'hash':cache_hash(cache),'normal_candidate':candidate,
             'token':760 if fallback else candidate,'fallback':fallback,'length':len(ids)+1,'base_invariants':None}
        self.contexts[question['id']]=ctx
        return ctx
