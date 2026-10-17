"""Unmodified frozen native Qwen with input injection and read-only collectors."""
import copy,hashlib
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration

def array(t):return t.detach().cpu().numpy().copy()
def digest(t):
    a=array(t).reshape(-1).view(np.uint8);return hashlib.sha256(a.tobytes()).hexdigest()
def cache_hash(cache,stop=24):
    h=hashlib.sha256()
    for i,l in enumerate(cache.layers[:stop]):
        h.update((str(i)+type(l).__name__).encode())
        for n in ['keys','values']:
            x=getattr(l,n,None)
            if torch.is_tensor(x):h.update((n+str(tuple(x.shape))+str(x.dtype)+digest(x)).encode())
        for n in ['conv_states','recurrent_states']:
            for key,x in sorted(getattr(l,n,{}).items()):
                if torch.is_tensor(x):h.update((n+str(key)+str(tuple(x.shape))+str(x.dtype)+digest(x)).encode())
    return h.hexdigest()

class Teacher:
    def __init__(self,checkpoint,deadline):
        self.deadline=deadline;self.calls=0;self.driver_replays=0
        self.tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
        self.model=Qwen3_5ForConditionalGeneration.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16,attn_implementation='sdpa')
        self.model.eval().requires_grad_(False)
        self.text=self.model.model.language_model;self.layers=self.text.layers
        self.loaded_A_log_dtype=str(self.layers[12].linear_attn.A_log.dtype)
        self.model.to(dtype=torch.float32).eval();self.contexts={}
        assert len(self.layers)==24 and all(self.layers[i].block_type=='linear_attention' for i in [12,13])
        assert all(not p.requires_grad and p.device.type=='cpu' for p in self.model.parameters())
    def count(self):
        self.deadline();self.calls+=1;assert self.calls<=300
    def encode(self,question,system):
        ids=self.tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':question}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if hasattr(ids,'keys'):ids=ids['input_ids']
        if hasattr(ids,'tolist'):ids=ids.tolist()
        if ids and isinstance(ids[0],list):ids=ids[0]
        assert 1<len(ids)<=128 and all(type(i)==int for i in ids)
        return ids
    def prefix(self,ids):
        self.count()
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            answer=self.text(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),use_cache=True,return_dict=True)
        return answer.past_key_values,None
    def step(self,incoming,token,length,direction=None,alpha=0.,choose_next=False):
        self.count();before=cache_hash(incoming);cache=copy.deepcopy(incoming);records={};handles=[]
        def pre(n):
            def hook(module,args):records[n]=args[0][0,-1]
            return hook
        def post(n):
            def hook(module,args,out):records[n]=(out[0] if isinstance(out,tuple) else out)[0,-1]
            return hook
        for i in [12,13]:
            l=self.layers[i]
            handles += [l.input_layernorm.register_forward_pre_hook(pre(f'b{i}_input')),l.linear_attn.register_forward_hook(post(f'b{i}_mixer')),
                        l.post_attention_layernorm.register_forward_pre_hook(pre(f'b{i}_middle')),l.mlp.register_forward_hook(post(f'b{i}_mlp')),l.register_forward_hook(post(f'b{i}_output'))]
        if direction is not None:
            d=torch.tensor(direction,dtype=torch.float32).reshape(1,1,-1);a=torch.tensor(alpha,dtype=torch.float32)
            def inject(module,args):return (args[0]+a*d,)+args[1:]
            handles.append(self.layers[12].register_forward_pre_hook(inject))
        try:
            with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
                answer=self.text(input_ids=torch.tensor([[token]]),attention_mask=torch.ones((1,length),dtype=torch.long),past_key_values=cache,use_cache=True,return_dict=True)
                result={n:array(t) for n,t in records.items()}
                logits=self.model.lm_head(answer.last_hidden_state[0,-1])
                result['final_embedding']=array(answer.last_hidden_state[0,-1]);result['logits']=array(logits)
                candidate=int(logits.argmax()) if choose_next else None
            for i in [12,13]:
                result[f'b{i}_S']=array(answer.past_key_values.layers[i].recurrent_states[0]).reshape(16,128,128)
                result[f'b{i}_window']=array(answer.past_key_values.layers[i].conv_states[0])
            assert len(result)==16 and cache_hash(incoming)==before,'Caller-owned cache changed'
            assert np.array_equal(result['b12_output'],result['b13_input'])
            return result,answer.past_key_values,candidate,cache_hash(answer.past_key_values,12)
        finally:
            for h in reversed(handles):h.remove()
    def initial(self,cache):
        return {i:{'S':array(cache.layers[i].recurrent_states[0]).reshape(16,128,128),'window':array(cache.layers[i].conv_states[0])} for i in [12,13]}
    def energy_driver(self,i,fields):
        self.driver_replays+=1;l=self.layers[i];m=l.linear_attn
        with torch.no_grad():
            z=l.input_layernorm(torch.tensor(fields[f'b{i}_input']).reshape(1,1,1024))
            pre=(torch.tensor(fields[f'b{i}_window'])[0]*m.conv1d.weight[:,0,:]).sum(-1)
            if m.conv1d.bias is not None:pre=pre+m.conv1d.bias
            mixed=torch.nn.functional.silu(pre);k=mixed[2048:4096].reshape(16,128)
            k=k*torch.rsqrt((k*k).sum(-1,keepdim=True)+1e-6)
            beta=m.in_proj_b(z).sigmoid()[0,0]
            gamma=(-m.A_log.float().exp()*torch.nn.functional.softplus(m.in_proj_a(z).float()+m.dt_bias)).exp()[0,0]
            return {'k':array(k),'v':array(mixed[4096:].reshape(16,128)),'beta':array(beta),'gamma':array(gamma)}
    def parameters(self):
        result={}
        for i in [12,13]:
            l=self.layers[i];m=l.linear_attn
            assert (m.num_k_heads,m.num_v_heads,m.head_k_dim,m.head_v_dim,m.conv_kernel_size)==(16,16,128,128,4)
            tensors={'input_norm':l.input_layernorm.weight,'conv':m.conv1d.weight,'qkv':m.in_proj_qkv.weight,'z':m.in_proj_z.weight,
                     'a':m.in_proj_a.weight,'b':m.in_proj_b.weight,'A_log':m.A_log,'dt_bias':m.dt_bias,'gate_norm':m.norm.weight,
                     'out':m.out_proj.weight,'post_norm':l.post_attention_layernorm.weight,'mlp_gate':l.mlp.gate_proj.weight,
                     'mlp_up':l.mlp.up_proj.weight,'mlp_down':l.mlp.down_proj.weight}
            if m.conv1d.bias is not None:tensors['conv_bias']=m.conv1d.bias
            result[i]={n:array(t) for n,t in tensors.items()}
        return result
