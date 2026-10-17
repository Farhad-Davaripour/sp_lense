"""Real full-prefix and cached Qwen contexts, with narrowly scoped response hooks."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration

def cache_hash(cache):
    h=hashlib.sha256();visited=set()
    def add(value,path):
        if torch.is_tensor(value):
            t=value.detach().cpu().contiguous()
            h.update((path+str(t.dtype)+str(tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value,(str,int,float,bool,type(None))):h.update((path+repr(value)).encode())
        elif id(value) not in visited:
            visited.add(id(value))
            if isinstance(value,dict):
                for k in sorted(value,key=str):add(value[k],path+'/'+str(k))
            elif isinstance(value,(list,tuple)):
                for i,v in enumerate(value):add(v,path+'/'+str(i))
            elif hasattr(value,'__dict__'):
                for k,v in sorted(vars(value).items()):
                    if not callable(v):add(v,path+'/'+k)
    add(cache,'cache')
    return h.hexdigest()

def detached_copy(value):
    if torch.is_tensor(value):return value.detach().clone()
    if isinstance(value,tuple):return tuple(detached_copy(v) for v in value)
    if isinstance(value,list):return [detached_copy(v) for v in value]
    if isinstance(value,dict):return {k:detached_copy(v) for k,v in value.items()}
    return copy.deepcopy(value)

class Rig:
    def __init__(self,checkpoint):
        self.tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
        self.model=Qwen3_5ForConditionalGeneration.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False,
            use_safetensors=True,dtype=torch.bfloat16,attn_implementation='sdpa')
        self.model.eval();self.model.requires_grad_(False)
        self.text=self.model.model.language_model;self.layers=self.text.layers
        assert len(self.layers)==24 and self.layers[11].block_type=='full_attention'
        assert all(p.device.type=='cpu' and not p.requires_grad for p in self.model.parameters())
        self.calls=0
        self.check=lambda:None

    def precision(self,dtype):
        self.model.to(dtype=dtype);self.model.eval()
        assert next(self.model.parameters()).dtype==dtype

    def prefix(self,system,text):
        tokens=self.tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':text}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if hasattr(tokens,'keys'):tokens=tokens['input_ids']
        if hasattr(tokens,'tolist'):tokens=tokens.tolist()
        if tokens and isinstance(tokens[0],list):assert len(tokens)==1;tokens=tokens[0]
        return list(tokens)

    def forward(self,ids,attention_length,cache=None,*,capture=False,edit=None,with_grad=False,use_cache=True):
        self.check()
        self.calls+=1
        handles=[];data={};original_cache_hash=cache_hash(cache) if cache is not None else None
        local_cache=copy.deepcopy(cache) if cache is not None else None
        layer=self.layers[11]
        if capture:
            def mlp_input(module,args):data['mlp_h']=args[0][0,-1].detach().clone()
            def attn_input(module,args):data['attn_H']=args[0].detach().clone()
            def attn_kwargs(module,args,kwargs):
                data['attn_kwargs']=detached_copy({k:v for k,v in kwargs.items() if k!='hidden_states'})
            def attn_output(module,args,result):data['attn_branch']=result[0][0,-1].detach().clone()
            def readout(module,args,result):data['readout15']=result[0,-1].detach().clone()
            def site(module,args,result):data['site10']=result[0,-1].detach().clone()
            handles += [layer.post_attention_layernorm.register_forward_pre_hook(mlp_input),
                layer.input_layernorm.register_forward_pre_hook(attn_input),
                layer.self_attn.register_forward_pre_hook(attn_kwargs,with_kwargs=True),
                layer.self_attn.register_forward_hook(attn_output),self.layers[15].register_forward_hook(readout),
                self.layers[10].register_forward_hook(site)]
        if edit is not None:
            block,callback=edit
            handles.append(self.layers[block].register_forward_hook(callback))
        try:
            with torch.set_grad_enabled(with_grad),sdpa_kernel(SDPBackend.MATH):
                answer=self.text(input_ids=torch.tensor([ids],dtype=torch.long),attention_mask=torch.ones((1,attention_length),dtype=torch.long),
                    past_key_values=local_cache,use_cache=use_cache,return_dict=True)
                hidden=answer.last_hidden_state[0,-1]
                logits=self.model.lm_head(hidden)
            if cache is not None:assert cache_hash(cache)==original_cache_hash,'Caller cache mutated'
            return dict(hidden=hidden,logits=logits,cache=answer.past_key_values,data=data)
        finally:
            for handle in reversed(handles):handle.remove()

    def hidden_function(self,ids,callback):
        """Differentiable full-prefix final hidden state; no cache/state surrogate."""
        self.check()
        self.calls+=1
        handle=self.layers[11].register_forward_hook(callback)
        try:
            with sdpa_kernel(SDPBackend.MATH):
                result=self.text(input_ids=torch.tensor([ids],dtype=torch.long),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),
                    use_cache=False,return_dict=True)
            return result.last_hidden_state[0,-1]
        finally:handle.remove()

def save_context(folder,context,*,prefix_ids,precision,position):
    folder.mkdir(parents=True,exist_ok=False)
    data=context['data']
    np.savez_compressed(folder/'vectors.npz',mlp_h=data['mlp_h'].float().numpy(),attn_H=data['attn_H'].float().numpy(),
        native_attn_branch=data['attn_branch'].float().numpy(),readout15=data['readout15'].float().numpy(),
        site10=data['site10'].float().numpy(),final_hidden=context['hidden'].detach().float().numpy(),
        logits=context['logits'].detach().float().numpy(),prefix_ids=np.asarray(prefix_ids,dtype=np.int64))
    # Private, checksum-protected replay state; never put these tensors in Git.
    torch.save(data['attn_kwargs'],folder/'attention_context.pt')
    (folder/'META.json').write_text(json.dumps({'precision':precision,'position':position,'prefix_length':len(prefix_ids),
        'incoming_attention_cache_sha256':cache_hash(data['attn_kwargs'].get('past_key_values'))},indent=2)+'\n')

def attention_function(layer,context):
    """Last-token input variation, actual full prefix or actual held-fixed cache."""
    full=context['data']['attn_H'].detach().clone()
    original_kwargs=context['data']['attn_kwargs']
    original_hash=cache_hash(original_kwargs.get('past_key_values'))
    def evaluate(last):
        changed=torch.cat([full[:,:-1],last.reshape(1,1,-1)],dim=1)
        kwargs=detached_copy(original_kwargs)
        with sdpa_kernel(SDPBackend.MATH):out=layer.self_attn(hidden_states=layer.input_layernorm(changed),**kwargs)[0][0,-1]
        assert cache_hash(original_kwargs.get('past_key_values'))==original_hash
        return out
    return evaluate,full[0,-1].detach().clone(),original_hash
