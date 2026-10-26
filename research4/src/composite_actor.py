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

LEGACY_INTERFACE='composite-native-v1'


def native_interface(profile):
    if profile==LEGACY_INTERFACE:return None
    from . import composite_generalized_interface
    from . import composite_bounded_behavior_interface
    if profile==composite_bounded_behavior_interface.PROFILE:return composite_bounded_behavior_interface
    if profile!=composite_generalized_interface.PROFILE:raise PermissionError('Unregistered native interface profile')
    return composite_generalized_interface


class Actor(nn.Module):
    def __init__(self,path,cfg,guard,on_forward=None,kernel_profile='reference',frozen_probe=None,native_interface_profile=LEGACY_INTERFACE,task_training=None,task_evaluation=None,bounded_training=None,bounded_evaluation=None):
        super().__init__();require_databricks()
        if sum(v is not None for v in [frozen_probe,task_training,task_evaluation,bounded_training,bounded_evaluation])>1:
            raise PermissionError('Frozen probe, task fitting and frozen task evaluation are separate contexts')
        native_interface(native_interface_profile)
        from .composite_bounded_behavior_interface import PROFILE as BOUNDED_PROFILE
        if (native_interface_profile==BOUNDED_PROFILE)!=(bounded_training is not None or bounded_evaluation is not None):
            raise PermissionError('V4 native actors require the separately validated V4 fitting constructor')
        if frozen_probe is not None and native_interface_profile!=LEGACY_INTERFACE:
            raise PermissionError('Historical frozen probes cannot change their interface')
        self.native_interface_profile=native_interface_profile
        if native_interface_profile!=LEGACY_INTERFACE:torch.backends.cuda.matmul.allow_tf32=False
        self.frozen_analysis_only=frozen_probe is not None
        self.task_training_context=None
        self.bounded_training_context=None
        self.bounded_evaluation_context=None
        self.task_evaluation_context=None
        self.frozen_probe_counts={'attempted':0,'completed':0}
        device_map={'':0};context=None
        if self.frozen_analysis_only:
            from .composite_frozen_probe import authorized_load_plan,verify_runtime,verify_downloaded_model
            stage=frozen_probe.get('stage','frozen-qwen-two-t4-qualification-v1')
            if stage=='frozen-base-validation-v1':
                from .composite_frozen_base import authorized_load_plan
            elif stage!='frozen-qwen-two-t4-qualification-v1':raise PermissionError('Unregistered frozen actor stage')
            context=authorized_load_plan(frozen_probe['root'],frozen_probe['source_commit'],cfg)
            self.frozen_forward_limit=context['pin']['maximum_backbone_forwards']
            if kernel_profile!='reference':raise PermissionError('Frozen probe requires reference kernels')
            verify_runtime(context);device_map=context['plan']['device_map'];self.frozen_probe_context=context
            if torch.is_autocast_enabled():raise RuntimeError('Frozen probe rejects unregistered caller autocast')
            torch.backends.cuda.matmul.allow_tf32=False
            self.frozen_probe_file_proof=verify_downloaded_model(context,path)
        if task_training is not None:
            from .composite_generalized_interface import PROFILE
            if native_interface_profile!=PROFILE or kernel_profile!='reference':raise PermissionError('Registered task-native reference interface required')
            from .composite_task_skill_training import authorized_load_plan
            from .composite_frozen_probe import verify_runtime,verify_downloaded_model
            context=authorized_load_plan(task_training['root'],task_training['source_commit'],cfg)
            verify_runtime(context)
            if torch.is_autocast_enabled():raise PermissionError('Unregistered caller autocast')
            device_map=context['plan']['device_map'];self.task_training_context=context
            self.task_training_file_proof=verify_downloaded_model(context,path)
        if bounded_training is not None:
            if kernel_profile!='reference':raise PermissionError('V4 fitting requires reference kernels')
            from .composite_bounded_behavior_training import authorized_load_plan
            from .composite_frozen_probe import verify_runtime,verify_downloaded_model
            context=authorized_load_plan(bounded_training['root'],bounded_training['source_commit'],cfg)
            verify_runtime(context)
            if torch.is_autocast_enabled():raise PermissionError('Unregistered caller autocast')
            device_map=context['plan']['device_map'];self.bounded_training_context=context
            self.bounded_training_file_proof=verify_downloaded_model(context,path)
        if bounded_evaluation is not None:
            if kernel_profile!='reference':raise PermissionError('Frozen V4 actor evaluation uses reference kernels')
            from .composite_bounded_behavior_evaluation_source import authorized_load_plan
            from .composite_frozen_probe import verify_runtime,verify_downloaded_model
            context=authorized_load_plan(bounded_evaluation,cfg);verify_runtime(context)
            if torch.is_autocast_enabled():raise PermissionError('Unregistered caller autocast')
            device_map=context['plan']['device_map'];self.bounded_evaluation_context=bounded_evaluation
            self.bounded_evaluation_load_context=context
            self.frozen_forward_limit=context['pin']['maximum_backbone_forwards']
            self.bounded_evaluation_file_proof=verify_downloaded_model(context,path)
        if task_evaluation is not None:
            from .composite_generalized_interface import PROFILE
            if native_interface_profile!=PROFILE or kernel_profile!='reference':
                raise PermissionError('Frozen task evaluation requires the unchanged native reference interface')
            from .composite_task_skill_inference import authorized_load_plan
            from .composite_frozen_probe import verify_runtime,verify_downloaded_model
            context=authorized_load_plan(task_evaluation,cfg);verify_runtime(context)
            if torch.is_autocast_enabled():raise PermissionError('Unregistered caller autocast')
            device_map=context['plan']['device_map'];self.task_evaluation_context=task_evaluation
            self.task_evaluation_load_context=context
            self.frozen_forward_limit=context['pin']['maximum_backbone_forwards']
            self.task_evaluation_file_proof=verify_downloaded_model(context,path)
        from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration,BitsAndBytesConfig
        from .composite_kernels import select
        self.kernel_proof=select(kernel_profile);self.kernel_profile=kernel_profile
        self.cfg=cfg;self.guard=guard;self.on_forward=on_forward;self.lock=threading.RLock();self.generating=False;self.adapters={};self.placement=None;self._last_cache=None;self._trace=None
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        arguments={'local_files_only':True,'trust_remote_code':False,'use_safetensors':True,'quantization_config':quant,'dtype':torch.bfloat16,'device_map':device_map,'attn_implementation':'sdpa','low_cpu_mem_usage':True}
        if self.frozen_analysis_only or self.interface() is not None:
            from .composite_frozen_probe import validate_loading_info
            model,loading=Qwen3_5ForConditionalGeneration.from_pretrained(path,output_loading_info=True,**arguments)
            self.frozen_loading_proof=validate_loading_info(loading)
        else:model=Qwen3_5ForConditionalGeneration.from_pretrained(path,**arguments)
        self.model=model.eval().requires_grad_(False)
        self.kernel_proof=select(kernel_profile)
        if len(self.text.layers)!=cfg['layers'] or self.model.config.text_config.hidden_size!=cfg['hidden_size'] or any(p.device.type!='cuda' or p.requires_grad for p in self.model.parameters()):raise ValueError('Unexpected native Qwen geometry')
        if self.frozen_analysis_only or self.task_training_context is not None or self.task_evaluation_context is not None or self.bounded_training_context is not None or self.bounded_evaluation_context is not None:
            from .composite_frozen_probe import verify_loaded_layout
            self.frozen_layout_proof=verify_loaded_layout(self.model,context)
        self.base_references=list(self.model.named_parameters());self.frozen_snapshot=parameter_snapshot(self.base_references)
        self.base_buffers=list(self.model.named_buffers());self.frozen_auxiliary=auxiliary_digest(self.base_references,self.base_buffers)
        self.eos=set()
        for value in [self.model.generation_config.eos_token_id,self.tokenizer.eos_token_id]:
            if value is not None:self.eos.update(value if isinstance(value,list) else [value])
        if not self.eos:raise ValueError('Missing native EOS')
        self.handles=[self.model.register_forward_pre_hook(self._before,with_kwargs=True),self.model.register_forward_hook(self._after)]
        if self.task_evaluation_context is not None:
            from .composite_task_skill_inference import restore_checkpoint
            restore_checkpoint(self,self.task_evaluation_context)
        if self.bounded_evaluation_context is not None:
            from .composite_bounded_behavior_evaluation_inference import restore_checkpoint
            restore_checkpoint(self,self.bounded_evaluation_context)

    @property
    def text(self):return self.model.model.language_model

    def interface(self):return native_interface(getattr(self,'native_interface_profile',LEGACY_INTERFACE))

    def prompt_ids(self,text):
        interface=self.interface()
        return input_ids(self.tokenizer,text) if interface is None else interface.input_ids(self.tokenizer,text)

    def target_text(self,action):
        interface=self.interface()
        return action_text(action) if interface is None else interface.action_text(action)

    def parse_response(self,text):
        interface=self.interface()
        return parse(text,1) if interface is None else interface.parse(text)

    def _before(self,module,args,kwargs):
        self.guard()
        if getattr(self,'frozen_analysis_only',False):
            from .composite_frozen_probe import guard_frozen_devices
            guard_frozen_devices(torch.cuda)
            if self.frozen_probe_counts['attempted']>=getattr(self,'frozen_forward_limit',400):raise TimeoutError('Frozen probe forward ceiling')
            self.frozen_probe_counts['attempted']+=1
        if getattr(self,'task_evaluation_context',None) is not None or getattr(self,'bounded_evaluation_context',None) is not None:
            from .composite_task_skill_training import live_resources
            live_resources(torch.cuda)
            if self.frozen_probe_counts['attempted']>=self.frozen_forward_limit:
                raise TimeoutError('Frozen task-evaluation forward ceiling')
            self.frozen_probe_counts['attempted']+=1
        if self.on_forward:self.on_forward('attempt')
        if self.generating and self._trace is not None and not self._trace['prefill_checked']:
            cache=kwargs.get('past_key_values');length=0 if cache is None else int(cache.get_seq_length())
            if length!=0:raise RuntimeError('A previous response cache crossed the generation boundary')
            if cache is not None and self._last_cache is not None and cache is self._last_cache():raise RuntimeError('A cache object was reused across responses')
            if getattr(self,'frozen_analysis_only',False) or self.interface() is not None:
                previous=None if self._last_cache is None else self._last_cache()
                if cache is None:raise RuntimeError('Frozen probe requires an observable fresh cache object')
                self._trace.update(cache_object_present=True,cache_identity=id(cache),previous_cache_alive=previous is not None,
                                   previous_cache_identity=None if previous is None else id(previous),previous_cache_reused=cache is previous)
            if cache is not None:self._last_cache=weakref.ref(cache)
            self._trace.update(prefill_checked=True,initial_cache_length=length)

    def _after(self,module,args,result):
        if getattr(self,'frozen_analysis_only',False) or getattr(self,'task_evaluation_context',None) is not None or getattr(self,'bounded_evaluation_context',None) is not None:
            self.frozen_probe_counts['completed']+=1
        if self.on_forward:self.on_forward('complete')
        if not torch.isfinite(result.logits).all():raise FloatingPointError('Nonfinite Qwen logits')

    def configure(self,placement,for_training=False):
        if getattr(self,'frozen_analysis_only',False):raise PermissionError('Frozen probe cannot install behavior adapters')
        if getattr(self,'task_training_context',None) is not None and (placement!={'name':'all_blocks','scope':'all_blocks'} or not for_training):
            raise PermissionError('Task fitting requires the fresh full all-block installation; restoration or reduced placement is not admitted')
        if getattr(self,'bounded_training_context',None) is not None and (placement!={'name':'all_blocks','scope':'all_blocks'} or not for_training):
            raise PermissionError('V4 fitting requires a fresh full all-block installation')
        if getattr(self,'bounded_evaluation_context',None) is not None and (
            not getattr(self,'_bounded_restoring',False) or for_training or placement!={'name':'all_blocks','scope':'all_blocks'}):
            raise PermissionError('Frozen V4 evaluation only restores its final all-block checkpoint once')
        if getattr(self,'task_evaluation_context',None) is not None and (
                not getattr(self,'_task_restoring',False) or for_training or placement!={'name':'all_blocks','scope':'all_blocks'}):
            raise PermissionError('Frozen task evaluation only restores its full fitted checkpoint once')
        with self.lock:
            if self.generating:raise RuntimeError('Cannot change adapters during a response')
            for name,wrapper in self.adapters.items():
                parent,attribute=name.rsplit('.',1);self.text.get_submodule(parent).__setattr__(attribute,wrapper.base)
            self.adapters={};self.placement=placement
            names=qwen_targets(self.text,placement,self.cfg['layers'])
            self.capacity_proof=capacity(self.text,names,self.cfg['rank'])
            if placement.get('scope')=='all_blocks':
                if getattr(self,'bounded_training_context',None) is not None:
                    from .composite_bounded_behavior_training import memory_admission,PARAMETERS,PROJECTIONS
                    if not for_training or self.capacity_proof['adapter_parameters']!=PARAMETERS or len(names)!=PROJECTIONS:
                        raise PermissionError('V4 fitting retains all agreed FP32 adapter projections')
                    self.capacity_proof['per_device_storage_admission']=memory_admission(
                        self.bounded_training_context['adapter_layout'],torch.cuda,self.cfg['training_memory_reserve_bytes'])
                elif getattr(self,'bounded_evaluation_context',None) is not None:
                    from .composite_task_skill_inference import inference_storage
                    if self.capacity_proof['adapter_parameters']!=233455616 or len(names)!=496:
                        raise PermissionError('Frozen V4 evaluation retains every FP32 fitted projection')
                    self.capacity_proof['per_device_storage_admission']=inference_storage(
                      self.bounded_evaluation_load_context['adapter_layout'],torch.cuda,self.cfg['training_memory_reserve_bytes'])
                elif getattr(self,'task_training_context',None) is not None:
                    from .composite_task_skill_training import memory_admission,PARAMETERS,PROJECTIONS
                    if not for_training or self.capacity_proof['adapter_parameters']!=PARAMETERS or len(names)!=PROJECTIONS:
                        raise PermissionError('This task route only installs the full agreed adapter for fresh fitting')
                    self.capacity_proof['per_device_storage_admission']=memory_admission(
                        self.task_training_context['adapter_layout'],torch.cuda,self.cfg['training_memory_reserve_bytes'])
                elif getattr(self,'task_evaluation_context',None) is not None:
                    from .composite_task_skill_inference import inference_storage
                    if self.capacity_proof['adapter_parameters']!=233455616 or len(names)!=496:
                        raise PermissionError('Frozen task evaluation retains every full-rank fitted projection')
                    self.capacity_proof['per_device_storage_admission']=inference_storage(
                      self.task_evaluation_load_context['adapter_layout'],torch.cuda,self.cfg['training_memory_reserve_bytes'])
                else:
                    required=self.capacity_proof['adapter_parameters']*(16 if for_training else 4)+self.cfg['training_memory_reserve_bytes']
                    if torch.cuda.mem_get_info()[0]<required:raise MemoryError('All-block FP32 adapter/optimizer storage plus activation reserve exceeds free GPU memory')
            self.adapters=install(self.text,names,self.cfg['rank'],self.cfg['alpha'],False)
            self.clear_cache()

    def training_context(self):
        if getattr(self,'frozen_analysis_only',False) or getattr(self,'task_evaluation_context',None) is not None or getattr(self,'bounded_evaluation_context',None) is not None:
            raise PermissionError('Frozen evaluation cannot enter fitting')
        return checkpointed_training(self.text,self.text.layers,self.cfg.get('gradient_checkpointing',False))

    def clear_cache(self):
        if self.generating:raise RuntimeError('Cannot clear cache during generation')
        if hasattr(self.model,'_cache'):self.model._cache=None
        # Retain a weak reference across response boundaries so a generation
        # implementation cannot silently recycle the preceding cache object.

    def adapter_state(self):
        value={'placement':self.placement,'config':self.cfg,'weights':state(self.adapters)}
        if self.interface() is not None:value['native_interface_profile']=self.native_interface_profile
        return value

    def select_kernels(self,profile):
        if (getattr(self,'frozen_analysis_only',False) or getattr(self,'task_evaluation_context',None) is not None or getattr(self,'bounded_evaluation_context',None) is not None) and profile!='reference':
            raise PermissionError('Frozen evaluation kernels are immutable')
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
        if getattr(self,'task_evaluation_context',None) is not None:raise PermissionError('Frozen task checkpoint is immutable')
        if getattr(self,'bounded_evaluation_context',None) is not None:raise PermissionError('Frozen V4 checkpoint is immutable')
        if saved.get('native_interface_profile',LEGACY_INTERFACE)!=getattr(self,'native_interface_profile',LEGACY_INTERFACE):
            raise ValueError('Behavior adapter belongs to another native interface profile')
        if saved['config']!=self.cfg:raise ValueError('Qwen adapter belongs to another model/configuration')
        self.configure(saved['placement']);load(self.adapters,saved['weights'])

    def generate(self,text,on,maximum_new_tokens=None):
        if getattr(self,'frozen_analysis_only',False) and (on or maximum_new_tokens not in [None,self.cfg['maximum_new_tokens']]):raise PermissionError('Frozen probe requires the unadapted full-budget generation')
        if (getattr(self,'task_evaluation_context',None) is not None or getattr(self,'bounded_evaluation_context',None) is not None) and (
                type(on) is not bool or maximum_new_tokens not in [None,self.cfg['maximum_new_tokens']]):
            raise PermissionError('Frozen task evaluation requires an exact base/always-on full-budget response')
        with self.lock:
            if self.generating:raise RuntimeError('Concurrent Qwen response')
            limit=self.cfg['maximum_new_tokens'] if maximum_new_tokens is None else maximum_new_tokens
            if type(limit) is not int or not 0<limit<=self.cfg['maximum_new_tokens']:raise ValueError('Generation budget cannot exceed the frozen ceiling')
            self.clear_cache();enable(self.adapters,on);ids=self.prompt_ids(text)
            if len(ids)>self.cfg['maximum_tokens']:raise ValueError('Actor context budget exceeded')
            tensor=torch.tensor([ids],device='cuda:0',dtype=torch.long);self._trace={'prefill_checked':False,'initial_cache_length':None};self.generating=True
            try:
                with torch.inference_mode():sequence=self.model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),past_key_values=None,use_cache=True,do_sample=False,num_beams=1,num_return_sequences=1,max_new_tokens=limit,eos_token_id=sorted(self.eos),pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else min(self.eos),logits_to_keep=1)
                if not self._trace['prefill_checked']:raise RuntimeError('Native generation bypassed the cache-boundary hook')
                generated=sequence[0,len(ids):].cpu().tolist();ended=bool(generated and generated[-1] in self.eos);raw=decode(self.tokenizer,generated,self.eos)
                try:actions=self.parse_response(raw) if ended else None
                except (ValueError,TypeError,RecursionError):actions=None
                result={'input_token_ids':ids,'generated_token_ids':generated,'finished_with_eos':ended,'raw_text':raw,'actions':actions,'adapter_enabled':bool(on),'cache':dict(self._trace),'kernel_profile':self.kernel_profile}
                if self.interface() is not None:result['native_interface_profile']=self.native_interface_profile
                if getattr(self,'frozen_analysis_only',False) or self.interface() is not None:
                    result['generation_parameters']={'max_new_tokens':limit,'do_sample':False,'num_beams':1,'num_return_sequences':1,'use_cache':True}
                return result
            finally:self.generating=False;self._trace=None

    def supervised_loss(self,row):
        if getattr(self,'frozen_analysis_only',False) or getattr(self,'task_evaluation_context',None) is not None or getattr(self,'bounded_evaluation_context',None) is not None:
            raise PermissionError('Frozen evaluation cannot compute fitting targets')
        self.guard();enable(self.adapters,True);prefix=self.prompt_ids(row['text']);target=self.tokenizer.encode(self.target_text(row['teacher_action']),add_special_tokens=False)+[self.tokenizer.eos_token_id]
        combined=prefix+target[:-1]
        if len(combined)>self.cfg['maximum_tokens']:raise ValueError('Teacher-forcing context budget exceeded')
        tokens=torch.tensor([combined],dtype=torch.long,device='cuda:0');labels=torch.tensor(target,dtype=torch.long,device='cuda:0')
        if self.on_forward:self.on_forward('attempt')
        hidden=self.text(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,return_dict=True).last_hidden_state
        logits=self.model.lm_head(hidden[:,-len(target):,:].to(self.model.lm_head.weight.device))
        if self.on_forward:self.on_forward('complete')
        return nn.functional.cross_entropy(logits[0].float(),labels.to(logits.device))

    def bypass_check(self,texts,persist_logits=None):
        if getattr(self,'frozen_analysis_only',False):raise PermissionError('Frozen base probes do not establish adapter ON/OFF effects')
        records=[]
        for index,text in enumerate(texts):
            ids=self.prompt_ids(text);tokens=torch.tensor([ids],device='cuda:0',dtype=torch.long);enable(self.adapters,False)
            with torch.inference_mode():wrapped=self.model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,logits_to_keep=1).logits.detach().clone()
            if persist_logits:persist_logits(index,'wrapped_OFF',ids,wrapped)
            wrapped_generation=self.generate(text,False) if index==0 else None
            try:
                for name,wrapper in self.adapters.items():
                    parent,attribute=name.rsplit('.',1);setattr(self.text.get_submodule(parent),attribute,wrapper.base)
                with torch.inference_mode():bare=self.model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,logits_to_keep=1).logits
                if persist_logits:persist_logits(index,'bare',ids,bare)
                equal=torch.equal(wrapped,bare);difference=float((wrapped-bare).abs().max())
                bare_generation=self.generate(text,False) if index==0 else None
            finally:
                for name,wrapper in self.adapters.items():
                    parent,attribute=name.rsplit('.',1);setattr(self.text.get_submodule(parent),attribute,wrapper)
            enable(self.adapters,True)
            with torch.inference_mode():on=self.model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,logits_to_keep=1).logits
            if persist_logits:persist_logits(index,'wrapped_ON',ids,on)
            generation_equal=wrapped_generation['generated_token_ids']==bare_generation['generated_token_ids'] if index==0 else None
            records.append({'input_token_ids':ids,'bitwise_equal':bool(equal),'maximum_absolute_logit_difference':difference,'on_logit_difference':float((wrapped-on).abs().max()),'off_generation_equal':generation_equal,'wrapped_generation':wrapped_generation,'bare_generation':bare_generation})
        self.verify_frozen()
        enable(self.adapters,False)
        return {'pass':bool(records) and all(r['bitwise_equal'] for r in records) and records[0]['off_generation_equal'] and any(r['on_logit_difference']>0 for r in records),'records':records,'one_Qwen_backbone':True}
