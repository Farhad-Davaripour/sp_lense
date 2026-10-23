"""One frozen native Qwen with independently switchable behavior LoRA."""
import threading,weakref
import torch
from torch import nn
from .composite_lora import install,enable,state,load,trainables
from .composite_interface import input_ids,action_text
from .composite_training import parameter_snapshot,assert_frozen,auxiliary_digest
from .native27_tools import decode,parse
from .composite_data import require_databricks
from .composite_capacity import qwen_targets,capacity,checkpointed_training


class Actor(nn.Module):
    def __init__(self,path,cfg,guard,on_forward=None,kernel_profile='reference',frozen_probe=None):
        super().__init__();require_databricks()
        self.frozen_analysis_only=frozen_probe is not None
        self.frozen_probe_counts={'attempted':0,'completed':0}
        device_map={'':0};context=None
        if self.frozen_analysis_only:
            from .composite_frozen_probe import authorized_load_plan,verify_runtime,verify_downloaded_model
            context=authorized_load_plan(frozen_probe['root'],frozen_probe['source_commit'],cfg)
            if kernel_profile!='reference':raise PermissionError('Frozen probe requires reference kernels')
            verify_runtime(context);device_map=context['plan']['device_map'];self.frozen_probe_context=context
            if torch.is_autocast_enabled():raise RuntimeError('Frozen probe rejects unregistered caller autocast')
            torch.backends.cuda.matmul.allow_tf32=False
            self.frozen_probe_file_proof=verify_downloaded_model(context,path)
        from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration,BitsAndBytesConfig
        from .composite_kernels import select
        self.kernel_proof=select(kernel_profile);self.kernel_profile=kernel_profile
        self.cfg=cfg;self.guard=guard;self.on_forward=on_forward;self.lock=threading.RLock();self.generating=False;self.adapters={};self.placement=None;self._last_cache=None;self._trace=None
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        arguments={'local_files_only':True,'trust_remote_code':False,'use_safetensors':True,'quantization_config':quant,'dtype':torch.bfloat16,'device_map':device_map,'attn_implementation':'sdpa','low_cpu_mem_usage':True}
        if self.frozen_analysis_only:
            from .composite_frozen_probe import validate_loading_info
            model,loading=Qwen3_5ForConditionalGeneration.from_pretrained(path,output_loading_info=True,**arguments)
            self.frozen_loading_proof=validate_loading_info(loading)
        else:model=Qwen3_5ForConditionalGeneration.from_pretrained(path,**arguments)
        self.model=model.eval().requires_grad_(False)
        self.kernel_proof=select(kernel_profile)
        if len(self.text.layers)!=cfg['layers'] or self.model.config.text_config.hidden_size!=cfg['hidden_size'] or any(p.device.type!='cuda' or p.requires_grad for p in self.model.parameters()):raise ValueError('Unexpected native Qwen geometry')
        if self.frozen_analysis_only:
            from .composite_frozen_probe import verify_loaded_layout
            self.frozen_layout_proof=verify_loaded_layout(self.model,context)
        self.base_references=list(self.model.named_parameters());self.frozen_snapshot=parameter_snapshot(self.base_references)
        self.base_buffers=list(self.model.named_buffers());self.frozen_auxiliary=auxiliary_digest(self.base_references,self.base_buffers)
        self.eos=set()
        for value in [self.model.generation_config.eos_token_id,self.tokenizer.eos_token_id]:
            if value is not None:self.eos.update(value if isinstance(value,list) else [value])
        if not self.eos:raise ValueError('Missing native EOS')
        self.handles=[self.model.register_forward_pre_hook(self._before,with_kwargs=True),self.model.register_forward_hook(self._after)]

    @property
    def text(self):return self.model.model.language_model

    def _before(self,module,args,kwargs):
        self.guard()
        if getattr(self,'frozen_analysis_only',False):
            from .composite_frozen_probe import guard_frozen_devices
            guard_frozen_devices(torch.cuda)
            if self.frozen_probe_counts['attempted']>=400:raise TimeoutError('Frozen probe forward ceiling')
            self.frozen_probe_counts['attempted']+=1
        if self.on_forward:self.on_forward('attempt')
        if self.generating and self._trace is not None and not self._trace['prefill_checked']:
            cache=kwargs.get('past_key_values');length=0 if cache is None else int(cache.get_seq_length())
            if length!=0:raise RuntimeError('A previous response cache crossed the generation boundary')
            if cache is not None and self._last_cache is not None and cache is self._last_cache():raise RuntimeError('A cache object was reused across responses')
            if getattr(self,'frozen_analysis_only',False):
                previous=None if self._last_cache is None else self._last_cache()
                if cache is None:raise RuntimeError('Frozen probe requires an observable fresh cache object')
                self._trace.update(cache_object_present=True,cache_identity=id(cache),previous_cache_alive=previous is not None,
                                   previous_cache_identity=None if previous is None else id(previous),previous_cache_reused=cache is previous)
            if cache is not None:self._last_cache=weakref.ref(cache)
            self._trace.update(prefill_checked=True,initial_cache_length=length)

    def _after(self,module,args,result):
        if getattr(self,'frozen_analysis_only',False):self.frozen_probe_counts['completed']+=1
        if self.on_forward:self.on_forward('complete')
        if not torch.isfinite(result.logits).all():raise FloatingPointError('Nonfinite Qwen logits')

    def configure(self,placement,for_training=False):
        if getattr(self,'frozen_analysis_only',False):raise PermissionError('Frozen probe cannot install behavior adapters')
        with self.lock:
            if self.generating:raise RuntimeError('Cannot change adapters during a response')
            for name,wrapper in self.adapters.items():
                parent,attribute=name.rsplit('.',1);self.text.get_submodule(parent).__setattr__(attribute,wrapper.base)
            self.adapters={};self.placement=placement
            names=qwen_targets(self.text,placement,self.cfg['layers'])
            self.capacity_proof=capacity(self.text,names,self.cfg['rank'])
            if placement.get('scope')=='all_blocks':
                required=self.capacity_proof['adapter_parameters']*(16 if for_training else 4)+self.cfg['training_memory_reserve_bytes']
                if torch.cuda.mem_get_info()[0]<required:raise MemoryError('All-block FP32 adapter/optimizer storage plus activation reserve exceeds free GPU memory')
            self.adapters=install(self.text,names,self.cfg['rank'],self.cfg['alpha'],False)
            self.clear_cache()

    def training_context(self):
        if getattr(self,'frozen_analysis_only',False):raise PermissionError('Frozen probe cannot enter fitting')
        return checkpointed_training(self.text,self.text.layers,self.cfg.get('gradient_checkpointing',False))

    def clear_cache(self):
        if self.generating:raise RuntimeError('Cannot clear cache during generation')
        if hasattr(self.model,'_cache'):self.model._cache=None
        # Retain a weak reference across response boundaries so a generation
        # implementation cannot silently recycle the preceding cache object.

    def adapter_state(self):return {'placement':self.placement,'config':self.cfg,'weights':state(self.adapters)}

    def select_kernels(self,profile):
        if getattr(self,'frozen_analysis_only',False) and profile!='reference':raise PermissionError('Frozen probe kernels are immutable')
        from .composite_kernels import select
        with self.lock:
            if self.generating:raise RuntimeError('Cannot change kernels during a response')
            self.clear_cache();self.kernel_proof=select(profile);self.kernel_profile=profile

    def verify_frozen(self):
        assert_frozen(self.frozen_snapshot)
        if auxiliary_digest(self.base_references,self.base_buffers)!=self.frozen_auxiliary:raise ValueError('Frozen Qwen buffers or quantization metadata changed')

    def frozen_proof(self):
        return {'hash_scope':'Every original parameter byte, original buffers and packed quantization metadata','auxiliary_sha256':self.frozen_auxiliary,'parameters':[{'name':name,'shape':list(p.shape),'dtype':str(p.dtype),'version':version,'sha256':value} for name,p,version,value in self.frozen_snapshot]}

    def restore(self,saved):
        if getattr(self,'frozen_analysis_only',False):raise PermissionError('Frozen probe cannot restore behavior adapters')
        if saved['config']!=self.cfg:raise ValueError('Qwen adapter belongs to another model/configuration')
        self.configure(saved['placement']);load(self.adapters,saved['weights'])

    def generate(self,text,on,maximum_new_tokens=None):
        if getattr(self,'frozen_analysis_only',False) and (on or maximum_new_tokens not in [None,self.cfg['maximum_new_tokens']]):raise PermissionError('Frozen probe requires the unadapted full-budget generation')
        with self.lock:
            if self.generating:raise RuntimeError('Concurrent Qwen response')
            limit=self.cfg['maximum_new_tokens'] if maximum_new_tokens is None else maximum_new_tokens
            if type(limit) is not int or not 0<limit<=self.cfg['maximum_new_tokens']:raise ValueError('Generation budget cannot exceed the frozen ceiling')
            self.clear_cache();enable(self.adapters,on);ids=input_ids(self.tokenizer,text)
            if len(ids)>self.cfg['maximum_tokens']:raise ValueError('Actor context budget exceeded')
            tensor=torch.tensor([ids],device='cuda:0',dtype=torch.long);self._trace={'prefill_checked':False,'initial_cache_length':None};self.generating=True
            try:
                with torch.inference_mode():sequence=self.model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),past_key_values=None,use_cache=True,do_sample=False,num_beams=1,num_return_sequences=1,max_new_tokens=limit,eos_token_id=sorted(self.eos),pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else min(self.eos),logits_to_keep=1)
                if not self._trace['prefill_checked']:raise RuntimeError('Native generation bypassed the cache-boundary hook')
                generated=sequence[0,len(ids):].cpu().tolist();ended=bool(generated and generated[-1] in self.eos);raw=decode(self.tokenizer,generated,self.eos)
                try:actions=parse(raw,1) if ended else None
                except (ValueError,TypeError,RecursionError):actions=None
                result={'input_token_ids':ids,'generated_token_ids':generated,'finished_with_eos':ended,'raw_text':raw,'actions':actions,'adapter_enabled':bool(on),'cache':dict(self._trace),'kernel_profile':self.kernel_profile}
                if getattr(self,'frozen_analysis_only',False):
                    result['generation_parameters']={'max_new_tokens':limit,'do_sample':False,'num_beams':1,'num_return_sequences':1,'use_cache':True}
                return result
            finally:self.generating=False;self._trace=None

    def supervised_loss(self,row):
        if getattr(self,'frozen_analysis_only',False):raise PermissionError('Frozen probe cannot compute fitting targets')
        self.guard();enable(self.adapters,True);prefix=input_ids(self.tokenizer,row['text']);target=self.tokenizer.encode(action_text(row['teacher_action']),add_special_tokens=False)+[self.tokenizer.eos_token_id]
        combined=prefix+target[:-1]
        if len(combined)>self.cfg['maximum_tokens']:raise ValueError('Teacher-forcing context budget exceeded')
        tokens=torch.tensor([combined],dtype=torch.long,device='cuda:0');labels=torch.tensor(target,dtype=torch.long,device='cuda:0')
        if self.on_forward:self.on_forward('attempt')
        hidden=self.text(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,return_dict=True).last_hidden_state
        logits=self.model.lm_head(hidden[:,-len(target):,:])
        if self.on_forward:self.on_forward('complete')
        return nn.functional.cross_entropy(logits[0].float(),labels)

    def bypass_check(self,texts):
        if getattr(self,'frozen_analysis_only',False):raise PermissionError('Frozen base probes do not establish adapter ON/OFF effects')
        records=[]
        for index,text in enumerate(texts):
            ids=input_ids(self.tokenizer,text);tokens=torch.tensor([ids],device='cuda:0',dtype=torch.long);enable(self.adapters,False)
            with torch.inference_mode():wrapped=self.model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,logits_to_keep=1).logits.detach().clone()
            wrapped_generation=self.generate(text,False) if index==0 else None
            try:
                for name,wrapper in self.adapters.items():
                    parent,attribute=name.rsplit('.',1);setattr(self.text.get_submodule(parent),attribute,wrapper.base)
                with torch.inference_mode():bare=self.model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,logits_to_keep=1).logits
                equal=torch.equal(wrapped,bare);difference=float((wrapped-bare).abs().max())
                bare_generation=self.generate(text,False) if index==0 else None
            finally:
                for name,wrapper in self.adapters.items():
                    parent,attribute=name.rsplit('.',1);setattr(self.text.get_submodule(parent),attribute,wrapper)
            enable(self.adapters,True)
            with torch.inference_mode():on=self.model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,logits_to_keep=1).logits
            generation_equal=wrapped_generation['generated_token_ids']==bare_generation['generated_token_ids'] if index==0 else None
            records.append({'input_token_ids':ids,'bitwise_equal':bool(equal),'maximum_absolute_logit_difference':difference,'on_logit_difference':float((wrapped-on).abs().max()),'off_generation_equal':generation_equal,'wrapped_generation':wrapped_generation,'bare_generation':bare_generation})
        self.verify_frozen()
        enable(self.adapters,False)
        return {'pass':bool(records) and all(r['bitwise_equal'] for r in records) and records[0]['off_generation_equal'] and any(r['on_logit_difference']>0 for r in records),'records':records,'one_Qwen_backbone':True}
