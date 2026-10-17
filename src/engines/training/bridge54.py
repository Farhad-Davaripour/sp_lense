"""Databricks A10 FP32 checkpointed native Bridge54 continuation only; common24 is held."""
import argparse,contextlib,copy,dataclasses,datetime as dt,gc,hashlib,importlib.metadata,importlib.util,inspect,json,math,os,sys,threading,time
from pathlib import Path
WORKER_SHA='32f9212e2a2028ebb63f900e71ce3f6ca351b4b5e7537a0951181e7b328512a0'
CP_SHA='7c8bda494ce99d13124c14fe196cbe8d6848dc489a17211e71a8c38ed59665fa'
PROOF_SHA='337d21e69b825bc442f069b83c8b53598c9553f4e94a112f610c37541deb91b4'
BASE_SHA='c02a217f9c3edbdfb84665d790447e89f222e786d52c34404de6f11dd4e19bf9'
DATA_SHA='3e84895f4b262ec032310e9cfeacce943fe514759585e3069c717b3fcd297807'
ABS=1e-6;REL=1e-5;GPU_RESERVE=2*2**30
def utc():return dt.datetime.now(dt.timezone.utc)
def parse(v):
 t=dt.datetime.fromisoformat(v.replace('Z','+00:00'));assert t.tzinfo is not None;return t.astimezone(dt.timezone.utc)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def cpu_logits_output(out):
 if not dataclasses.is_dataclass(out):raise TypeError('Expected actual dataclass ModelOutput')
 result=dataclasses.replace(out,logits=out.logits.to('cpu'));assert type(result) is type(out);return result
COMPARISON_SHA='7f65a2bedd52893ed9265c54c77cb491f44d287c95fb0acd0fd2605b3a657ed4'
RESTORATION_SHA='a6d2f6d2b847f92e7ff66e1c08b9b6cea5972bba6b7a9d5fe825f3d112c42786'





def run(c):
 from src.engines.training import bridge54_policy as policy
 from src.engines.training.checkpointing import configure_candidate
 code=Path(c['source_dir']);admitted=policy.validate_worker(c,code)
 import psutil
 effective=admitted['end'];allocated=admitted['allocation_request'];neural_end=admitted['neural_end']
 assert policy.effective_available_bytes()>=16*2**30
 assert c.get('allow_evaluation',False) is False
 assert sha(c['native_worker'])==WORKER_SHA and sha(c['resume_checkpoint'])==CP_SHA and Path(c['resume_checkpoint']).stat().st_size==202305257
 packet=Path(c['packet']);policy.verify_selected_packet(packet)
 historical=policy.HISTORICAL;assert historical['original_strict_CPU_GPU_equivalence']=='FAILED'
 root=Path(c['root']);root.mkdir(parents=True,exist_ok=False);q=root/'disclosed_execution';q.mkdir();stop=Path(c['stop_file']);assert not stop.exists()
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['HF_HUB_OFFLINE']='1'
 phase='setup';total=216;history=[];model=None;masters=None;gpu_named=None;opt=None;sched=None;before=None;old_forward=None
 old_functions={};watcher=None;watch_done=threading.Event();GPU_stop=threading.Event();host_soft_stop=threading.Event();watch_fault=threading.Event();first_step_end=None;resource_lock=threading.Lock()
 observed={'min_host_available_bytes':None,'min_GPU_driver_free_bytes':None,'GPU_watch_samples':0}
 counts={'CPU_reference_forward_attempts':0,'GPU_qualification_forward_attempts':0,'scientific_forward_attempts':0,
  'shadow_CPU_Adam_steps_completed':0,'shadow_GPU_master_Adam_steps_completed':0,'scientific_CPU_Adam_steps_completed':0,'scientific_GPU_mirrors_completed':0}
 def count():write(root/'COUNTS.json',{'utc':utc().isoformat(),'phase':phase,'actual_total_updates':total,'qualification_scientific_updates':0,**counts})
 try:
  import torch
  from transformers import Qwen3_5ForConditionalGeneration,get_linear_schedule_with_warmup
  from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
  from torch.nn.attention import SDPBackend,sdpa_kernel
  policy.validate_hardware(torch)
  assert str(torch.__version__)=='2.11.0+cu128' and torch.version.cuda=='12.8' and torch.cuda.is_available() and torch.cuda.device_count()==1
  for name,version in (('transformers','5.19.0'),('peft','0.21.2')):assert importlib.metadata.version(name)==version
  torch.set_num_threads(4);torch.set_num_interop_threads(1);assert not torch.is_autocast_enabled() and not torch.is_autocast_enabled('cpu')
  spec=importlib.util.spec_from_file_location('r3_native32f_GPU',c['native_worker']);w=importlib.util.module_from_spec(spec);sys.modules[spec.name]=w;spec.loader.exec_module(w)
  def resource():
   host=policy.effective_available_bytes();free,gpu_total=torch.cuda.mem_get_info(0)
   with resource_lock:
    observed['min_host_available_bytes']=host if observed['min_host_available_bytes'] is None else min(observed['min_host_available_bytes'],host)
    observed['min_GPU_driver_free_bytes']=free if observed['min_GPU_driver_free_bytes'] is None else min(observed['min_GPU_driver_free_bytes'],free)
   return {'utc':utc().isoformat(),'host_available_bytes':host,'GPU_driver_free_bytes':free,'GPU_total_bytes':gpu_total,
    'cumulative_peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(0),'cumulative_peak_GPU_reserved_bytes':torch.cuda.max_memory_reserved(0)}
  def GPU_watch():
   try:
    while not watch_done.is_set():
     r=resource()
     with resource_lock:observed['GPU_watch_samples']+=1
     if r['host_available_bytes']<10*2**30:
      host_soft_stop.set();first=root/'FIRST_CHILD_HOST_RAM_STOP.json'
      if not first.exists():write(first,{'utc':r['utc'],'phase':phase,'reason':'child_host_available_below_soft10GiB','resource':r,'actual_total_updates':total})
      stop.parent.mkdir(parents=True,exist_ok=True);stop.touch(exist_ok=True)
     if r['GPU_driver_free_bytes']<GPU_RESERVE:
      first_GPU_breach=not GPU_stop.is_set();GPU_stop.set()
      if first_GPU_breach:write(root/'FIRST_GPU_RESERVE_STOP.json',{'utc':r['utc'],'phase':phase,'resource':r,'actual_total_updates':total})
      stop.parent.mkdir(parents=True,exist_ok=True);stop.touch(exist_ok=True)
     watch_done.wait(1)
   except BaseException as error:
    watch_fault.set();GPU_stop.set();write(root/'GPU_MEMORY_WATCH_FAULT.json',{'utc':utc().isoformat(),'type':type(error).__name__})
    try:stop.touch(exist_ok=True)
    except BaseException:pass
  def guard():
   r=resource();assert not watch_fault.is_set() and not GPU_stop.is_set() and r['GPU_driver_free_bytes']>=GPU_RESERVE,'Child GPU2GiB guard'
   assert utc()<neural_end and r['host_available_bytes']>=8*2**30,'Effective hard deadline or host8GiB'
   if phase=='scientific' and total==216 and first_step_end is not None:assert utc()<first_step_end,'First real update300s bound'
   if phase=='setup':assert utc()<allocated+dt.timedelta(seconds=1800) and not stop.exists() and r['host_available_bytes']>=10*2**30
   return r
  watcher=threading.Thread(target=GPU_watch,name='child_GPU2GiB_guard',daemon=True);watcher.start();assert watcher.is_alive()
  write(root/'GPU_MEMORY_GUARD_STARTED.json',{'utc':utc().isoformat(),'child_GPU_memory_guard_started':True,'required_driver_free_bytes':GPU_RESERVE,'interval_seconds':1,'observed_guard_not_full_fit_proof':True})
  guard();state=torch.load(c['resume_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
  assert state['stage']=='bridge' and state['stage_updates']==2 and state['total_updates']==216 and state['native_schedule_next_update']==2
  assert state['rank']==16 and state['alpha']==32 and state['seed']==941 and state['scheduler']['last_epoch']==2
  assert state['base_fingerprint']=={'sha256':BASE_SHA,'parameter_tensors':617}
  assert len(state['adapter'])==len(state['optimizer']['state'])==372 and all(float(v['step'])==2 for v in state['optimizer']['state'].values())
  assert all(torch.isfinite(v).all() for v in state['adapter'].values())
  data=read(packet/'tokenized/Qwen3.5-2B/bridge_final_response_tokens.json');plans=read(packet/'schedule/bridge_56_updates.json')
  assert len(data)==112 and len(plans)==56
  pad=int(read(packet/'tokenized/Qwen3.5-2B/TOKENIZATION_AUDIT.json')['pad_token_id'])
  w.seed(93);model=Qwen3_5ForConditionalGeneration.from_pretrained(c['model_cache'],local_files_only=True,trust_remote_code=False,dtype=torch.float32,device_map={'':'cuda:0'},attn_implementation='sdpa',use_safetensors=True)
  model.eval().requires_grad_(False);before=w.digest_base(model);assert before==state['base_fingerprint']
  targets=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and 'language_model.layers.' in n and n.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj','gate_proj','up_proj','down_proj'}]
  assert targets and all('visual' not in n and 'lm_head' not in n for n in targets)
  model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',target_modules=targets))
  named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];names=[n for n,p in named];cpu_params=[p for n,p in named]
  assert len(named)==372 and len(set(names))==372 and sum(p.numel() for p in cpu_params)==16819200
  assert all(p.device.type=='cuda' and p.dtype==torch.float32 for p in cpu_params)
  assert not getattr(model,'_use_kernels',False) and not any(getattr(m,'_hf_hook',None) for m in model.modules())
  for name,m in model.named_modules():
   if 'language_model' in name:
    if isinstance(m,torch.nn.Dropout):assert m.p==0
    if hasattr(m,'attention_dropout'):assert m.attention_dropout==0
  set_peft_model_state_dict(model,state['adapter']);model.train();w.seed(941);torch.set_rng_state(state['torch_rng']);cpu_rng=torch.get_rng_state().clone()
  views=get_peft_model_state_dict(model);assert set(views)==set(state['adapter']) and all(torch.equal(v.detach().cpu(),state['adapter'][k]) for k,v in views.items())
  pointers={p.data_ptr():i for i,p in enumerate(cpu_params)};assert len(pointers)==372
  adapter_index={k:pointers[v.data_ptr()] for k,v in views.items()};assert len(set(adapter_index.values()))==372
  index_key={i:k for k,i in adapter_index.items()}
  import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
  pure={};backend_sources={}
  for name in ('causal_conv1d_fn','causal_conv1d_update','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'):
   fn=getattr(qwen,name);raw=inspect.unwrap(fn)
   assert raw.__module__==qwen.__name__ and raw.__name__==name and Path(raw.__code__.co_filename).resolve()==Path(qwen.__file__).resolve()
   cursor=fn
   while inspect.isfunction(cursor):
    nonlocal_values=inspect.getclosurevars(cursor).nonlocals
    if 'is_new_implementation' in nonlocal_values:assert nonlocal_values['is_new_implementation'] is False,'Optional fast path in CPU reference'
    if not hasattr(cursor,'__wrapped__'):break
    cursor=cursor.__wrapped__
   pure[name]=raw;backend_sources[name]={'module':raw.__module__,'name':raw.__name__,'source_sha256':hashlib.sha256(inspect.getsource(raw).encode()).hexdigest()}
  old_forward=model.forward
  def forward(**kw):
   guard()
   assert phase=='scientific'
   counts['scientific_forward_attempts']+=1;count()
   assert kw.get('use_cache') is False and set(kw)=={'input_ids','attention_mask','use_cache','logits_to_keep'}
   moved={k:(v.to('cuda') if k in ('input_ids','attention_mask','logits_to_keep') else v) for k,v in kw.items()}
   with torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):out=old_forward(**moved)
   assert out.logits.dtype==torch.float32 and out.logits.requires_grad;result=cpu_logits_output(out)
   assert result.logits.device.type=='cpu' and result.logits.requires_grad;return result
  model.forward=forward
  def new_optimizer(params):
   optimizer=torch.optim.AdamW(params,lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01);scheduler=get_linear_schedule_with_warmup(optimizer,4,56)
   optimizer.load_state_dict(copy.deepcopy(state['optimizer']));scheduler.load_state_dict(copy.deepcopy(state['scheduler']))
   assert scheduler.last_epoch==2 and all(p.device.type=='cpu' for pg in optimizer.param_groups for p in pg['params'])
   return optimizer,scheduler
  def CPU_step(params,optimizer,scheduler,kind):
   assert all(p.device.type=='cpu' and p.grad is not None and p.grad.device.type=='cpu' and torch.isfinite(p.grad).all() for p in params)
   norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm);optimizer.step()
   key={'CPU_shadow':'shadow_CPU_Adam_steps_completed','GPU_shadow':'shadow_GPU_master_Adam_steps_completed','scientific':'scientific_CPU_Adam_steps_completed'}[kind]
   counts[key]+=1;count();scheduler.step();return float(norm)
  def RNG_unchanged(cpu_before,cuda_before=None):
   assert torch.equal(torch.get_rng_state(),cpu_before),'Unexpected CPU RNG consumption'
   if cuda_before is not None:assert torch.equal(torch.cuda.get_rng_state(0),cuda_before),'Unexpected CUDA RNG consumption'
  guard()
  write(root/'MODEL_READY_GPU.json',{'utc':utc().isoformat(),'model_loaded_once':True,'checkpoint216_loaded':True,'scientific_updates':0,
   'qualification_forwards_this_run':0,'shadow_Adam_steps_this_run':0,'resource':resource(),
   'strict_equivalence_passed':False,'historical_qualification_reused_as_pass':False,'setup_watchdog_must_remain_live_until_GPU_ADMISSION':True})
  write(root/'HISTORICAL_FAILED_QUALIFICATION_REFERENCE.json',historical)
  write(q/'DISCLOSED_EXECUTION_FREEZE.json',{'utc':utc().isoformat(),'protocol_sha256':c['protocol_sha256'],
   'task_configuration_sha256':c['task_config_sha256'],'prior_qualification_comparison_sha256':COMPARISON_SHA,
   'strict_equivalence_passed':False,'historical_qualification_reused_as_pass':False,'qualification_forwards_this_run':0,
   'shadow_Adam_steps_this_run':0,'remaining_scientific_updates':54,'no_new_qualification_sweep_or_retry':True,
   'GPU_pure_Torch_functions':backend_sources,'disclosed_cross_device_execution':True})
  masters=[torch.nn.Parameter(p.detach().to('cpu').clone(),requires_grad=True) for p in cpu_params]
  for name,fn in pure.items():old_functions[name]=getattr(qwen,name);setattr(qwen,name,fn)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_float32_matmul_precision('highest')
  assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
  # Non-reentrant recomputation happens during backward outside the outer forward context.
  # Globally pin the same MATH attention backend for BOTH original and recompute.
  torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
  assert torch.backends.cuda.math_sdp_enabled() and not torch.backends.cuda.flash_sdp_enabled() and not torch.backends.cuda.mem_efficient_sdp_enabled() and not torch.backends.cuda.cudnn_sdp_enabled()
  guard();model.to('cuda');gpu_named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
  assert [n for n,p in gpu_named]==names and all(p.device.type=='cuda' and p.dtype==torch.float32 for n,p in gpu_named)
  assert all(torch.equal(master.detach(),p.detach().cpu()) for master,(name,p) in zip(masters,gpu_named))
  assert all(p.dtype==torch.float32 for p in model.parameters() if p.is_floating_point()) and w.digest_base(model)==before
  gpu_views=get_peft_model_state_dict(model);assert set(gpu_views)==set(adapter_index) and all(gpu_views[k].data_ptr()==gpu_named[i][1].data_ptr() for k,i in adapter_index.items())
  torch.cuda.manual_seed_all(941);cuda_rng=torch.cuda.get_rng_state(0).clone();opt,sched=new_optimizer(masters)

  def adapter_copy():
   return {key:masters[i].detach().cpu().clone() for key,i in adapter_index.items()}
  def save_adapter(path):
   full={name:p.detach().cpu().clone() for name,p in zip(names,masters)};mapped=get_peft_model_state_dict(model,state_dict=full);expected=adapter_copy()
   assert set(mapped)==set(expected) and all(torch.equal(mapped[k],expected[k]) for k in expected)
   model.save_pretrained(path,safe_serialization=True,state_dict=full)
  def mirror():
   with torch.no_grad():
    for master,(name,p) in zip(masters,gpu_named):p.copy_(master)
   torch.cuda.synchronize(0)
  def compare_tree(x,y,path=''):
   metrics={};ok=True
   if torch.is_tensor(x):
    assert torch.is_tensor(y) and x.shape==y.shape and x.dtype==y.dtype and x.device.type==y.device.type=='cpu'
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    delta=x.double()-y.double();absolute=float(delta.abs().max()) if delta.numel() else 0.;relative=float(delta.norm())/max(float(x.double().norm()),1e-30)
    return absolute<=ABS and relative<=REL,{path:{'max_abs':absolute,'relative_L2':relative}}
   if isinstance(x,dict):
    assert isinstance(y,dict) and set(x)==set(y)
    for key in x:
     good,m=compare_tree(x[key],y[key],path+'/'+str(key));ok=ok and good;metrics.update(m)
    return ok,metrics
   if isinstance(x,(tuple,list)):
    assert type(x) is type(y) and len(x)==len(y)
    for i,(a,b) in enumerate(zip(x,y)):
     good,m=compare_tree(a,b,path+'/'+str(i));ok=ok and good;metrics.update(m)
    return ok,metrics
   if isinstance(x,float):
    assert math.isfinite(x) and math.isfinite(float(y));absolute=abs(x-y);relative=absolute/max(abs(x),1e-30)
    return absolute<=ABS and relative<=REL,{path:{'max_abs':absolute,'relative_L2':relative}}
   assert x==y,(path,'nonfloating metadata mismatch');return True,metrics
  def exact_state(x,y):
   if torch.is_tensor(x):return torch.is_tensor(y) and x.shape==y.shape and x.dtype==y.dtype and x.device.type==y.device.type=='cpu' and torch.equal(x,y)
   if isinstance(x,dict):return isinstance(y,dict) and set(x)==set(y) and all(exact_state(x[k],y[k]) for k in x)
   if isinstance(x,(list,tuple)):return type(x) is type(y) and len(x)==len(y) and all(exact_state(a,b) for a,b in zip(x,y))
   return x==y
  def GPU_group(group,optimizer,scheduler,kind,planned=None):
   nonlocal total
   guard();model.zero_grad(set_to_none=True);optimizer.zero_grad(set_to_none=True)
   CPU_before=torch.get_rng_state().clone();CUDA_before=torch.cuda.get_rng_state(0).clone();torch.cuda.synchronize(0);tick=time.monotonic()
   losses=w.group_loss(model,group,1,pad);guard()
   assert len(losses)==4 and all(math.isfinite(v) for v in losses) and all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
   for master,(name,p) in zip(masters,gpu_named):
    assert p.grad is not None and p.grad.dtype==torch.float32 and torch.isfinite(p.grad).all();master.grad=p.grad.detach().to('cpu')
   raw={name:p.grad.detach().clone() for name,p in zip(names,masters)} if kind=='GPU_shadow' else None
   guard();norm=CPU_step(masters,optimizer,scheduler,kind)
   if kind=='scientific':
    total+=1
    history.append({'stage':'bridge','update':total-214,'row_ids':planned['row_ids'],'seconds':time.monotonic()-tick,'loss_per_example':losses,
     'host_free_GiB':psutil.virtual_memory().available/2**30,'RSS_GiB':psutil.Process().memory_info().rss/2**30,
     'base_gradients_absent':True,'adapter_gradients_finite':True,'CPU_master_commit_complete':True,'GPU_mirror_complete':False,'RNG_invariants_verified':False})
    write(root/'LOSS_RECORDS_PRIVATE.json',history);count()
   RNG_unchanged(CPU_before,CUDA_before)
   if kind=='scientific':history[-1]['RNG_invariants_verified']=True
   mirror();elapsed=time.monotonic()-tick
   if kind=='scientific':
    counts['scientific_GPU_mirrors_completed']+=1;history[-1]['GPU_mirror_complete']=True;history[-1]['seconds']=elapsed
    write(root/'LOSS_RECORDS_PRIVATE.json',history);count()
   guard();return {'losses':losses,'seconds':elapsed,'clip_norm':norm,'raw_gradients':raw}
  def restore216():
   model.zero_grad(set_to_none=True)
   with torch.no_grad():
    for i,p in enumerate(masters):p.copy_(state['adapter'][index_key[i]])
   opt.load_state_dict(copy.deepcopy(state['optimizer']));sched.load_state_dict(copy.deepcopy(state['scheduler']));opt.zero_grad(set_to_none=True)
   w.seed(941);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng,0);mirror()
   restored_adapter=adapter_copy()
   assert all(torch.equal(restored_adapter[k],v) for k,v in state['adapter'].items())
   assert all(torch.equal(v.detach().cpu(),state['adapter'][k]) for k,v in get_peft_model_state_dict(model).items())
   assert sched.last_epoch==2 and all(float(v['step'])==2 for v in opt.state.values()) and w.digest_base(model)==before
   assert exact_state(opt.state_dict(),state['optimizer']) and exact_state(sched.state_dict(),state['scheduler']),'Exact native216 Adam/scheduler restoration'
   RNG_unchanged(cpu_rng,cuda_rng)
   return {'utc':utc().isoformat(),'adapter_exact216':True,'GPU_adapter_exact216':True,'Torch_RNG_exact216':True,'CPU_Adam2_scheduler2_restored':True,
    'base_unchanged':True,'gradients_cleared':all(p.grad is None for p in masters) and all(p.grad is None for n,p in gpu_named),
    'CUDA_RNG_seed941_restored':True,'same_model_retained':True,'no_cross_device_RNG_equivalence_claim':True}
  phase='setup';restoration=restore216();guard()
  restoration.update(initial_checkpoint_load_not_new_qualification_restore=True,strict_equivalence_passed=False,
   historical_qualification_reused_as_pass=False,qualification_forwards_this_run=0,shadow_Adam_steps_this_run=0)
  write(root/'RESTORATION.json',restoration)
  checkpointing=configure_candidate(model,explicit_execution_approval=True)
  assert not checkpointing['fit_verified'];RNG_unchanged(cpu_rng,cuda_rng)
  write(root/'CHECKPOINTING_POLICY.json',{'utc':utc().isoformat(),**checkpointing,'native_model_already_training':True,'no_precision_or_dropout_change':True,'SDPA_original_and_recompute_MATH':True,'top_level_forward_counts_exclude_internal_checkpoint_recomputations':True,'direct_GPU_loading_CPU_master_Adam':True})
  assert counts['CPU_reference_forward_attempts']==counts['GPU_qualification_forward_attempts']==0
  assert counts['shadow_CPU_Adam_steps_completed']==counts['shadow_GPU_master_Adam_steps_completed']==0 and total==216
  guard();remaining=(effective-utc()).total_seconds();decision=policy.training_gate(plans,data,2,[],remaining)
  if not decision['admitted'] or stop.exists() or host_soft_stop.is_set() or GPU_stop.is_set():
   write(root/'RESOURCE_REFUSAL.json',{'utc':utc().isoformat(),**decision,'total_updates':216,'no_scientific_updates':True});return
  write(root/'GPU_ADMISSION.json',{'utc':utc().isoformat(),'selected_device':'cuda:0','same_model_loaded_once':True,
   'first_real_native_update_only_initial_time_admission':True,'whole_remaining_dose_time_admitted':False,
   'remaining_updates':54,'protocol_sha256':c['protocol_sha256'],'task_configuration_sha256':c['task_config_sha256'],
   'strict_CPU_equivalence_passed':False,'original_failure_retained':True,'gradient_checkpointing_enabled':True,
   'single_Databricks_GPU':True,'one_visible_GPU':True,'new_qualification_forwards':0,**decision})
  phase='scientific';end=time.monotonic()+max(0,(neural_end-utc()).total_seconds())
  def checkpoint(completed):
   assert total==214+completed and sched.last_epoch==completed and all(float(v['step'])==completed for v in opt.state.values())
   value={'adapter':adapter_copy(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'stage':'bridge','stage_updates':completed,'total_updates':total,
    'base_fingerprint':before,'seed':941,'torch_rng':torch.get_rng_state(),'native_schedule_next_update':completed,'rank':16,'alpha':32}
   target=root/'LAST_COMPLETED_STATE.pt';partial=target.with_suffix('.partial.pt');torch.save(value,partial);partial.replace(target)
   if completed in (0,7,14,28,56):
    snapshot=root/'snapshots'/('bridge_'+str(completed));snapshot.mkdir(parents=True,exist_ok=True);torch.save(value,snapshot/'STATE.pt')
   save_adapter(root/'latest_adapter')
   import shutil
   with target.open('rb') as handle:readback=hashlib.file_digest(handle,'sha256').hexdigest()
   assert readback==sha(target)
   write(root/'CHECKPOINT_RECEIPT.json',{'utc':utc().isoformat(),'stage':'bridge','stage_updates':completed,'total_updates':total,'bytes':target.stat().st_size,
    'sha256':sha(target),'Volume_readback_verified':True,'C_verification_pending':True,'authoritative_CPU_master_adapter_Adam_scheduler':True})
  times=[]
  for j,planned in enumerate(plans[2:],start=2):
   assert [data[i]['id'] for i in planned['slots']]==planned['row_ids']
   remaining=end-time.monotonic()
   window=policy.training_gate(plans,data,j,times,(effective-utc()).total_seconds())
   write(root/'REMAINING_EXECUTION_ADMISSION.json',{'utc':utc().isoformat(),'completed_new_scientific_updates':len(times),**window})
   if not window['admitted'] or remaining<max(180,2*max(times or [150.])) or policy.effective_available_bytes()<8*2**30 or stop.exists() or host_soft_stop.is_set() or GPU_stop.is_set():
    checkpoint(j);write(root/'RESOURCE_BOUNDARY.json',{'utc':utc().isoformat(),'stage':'bridge','stage_updates':j,'total_updates':total,'next_update':j,
     'remaining_seconds':remaining,'whole_remaining_dose_admission':window,'base_unchanged':w.digest_base(model)==before,'incomplete_dose_not_capability_failure':True});return
   if j==2:
    first_step_end=min(neural_end,utc()+dt.timedelta(seconds=300))
    write(root/'FIRST_SCIENTIFIC_UPDATE_STARTED.json',{'utc':utc().isoformat(),'native_Bridge_update':3,'actual_total_before':216,'hard_end_utc':first_step_end.isoformat(),'real_scientific_update_not_discarded_trial':True})
   result=GPU_group([data[i] for i in planned['slots']],opt,sched,'scientific',planned);times.append(result['seconds'])
   assert total==214+j+1 and sched.last_epoch==j+1
   write(root/'PROGRESS.json',{'utc':utc().isoformat(),'stage':'bridge','stage_updates':j+1,'stage_total':56,'total_updates':total,'last_seconds':result['seconds'],
    'host_free_GiB':psutil.virtual_memory().available/2**30,'RSS_GiB':psutil.Process().memory_info().rss/2**30,'base_gradients_absent':True,'adapter_gradients_finite':True,
    'CPU_master_commit_complete':True,'GPU_mirror_complete':True,'provider_calls':0})
   if True:
    checkpoint(j+1)
    if j+1 in (7,14,28,56):save_adapter(root/'snapshots'/('bridge_'+str(j+1)))
  opt.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True);guard()
  assert total==270 and sched.last_epoch==56 and counts['scientific_CPU_Adam_steps_completed']==counts['scientific_GPU_mirrors_completed']==54
  assert w.digest_base(model)==before
  authoritative=adapter_copy();gpu_adapter=get_peft_model_state_dict(model)
  assert all(torch.equal(v.detach().cpu(),authoritative[k]) for k,v in gpu_adapter.items())
  from safetensors.torch import load_file
  saved=load_file(str(root/'latest_adapter/adapter_model.safetensors'),device='cpu')
  assert set(saved)==set(authoritative) and all(torch.equal(saved[k],authoritative[k]) and torch.isfinite(saved[k]).all() for k in authoritative)
  final_pin=read(root/'CHECKPOINT_RECEIPT.json');assert final_pin['stage_updates']==56 and final_pin['Volume_readback_verified'] and sha(root/'LAST_COMPLETED_STATE.pt')==final_pin['sha256']
  write(root/'COMPLETE.json',{'utc':utc().isoformat(),'completed':True,'H2_updates':214,'bridge_updates':56,'total_updates':270,'presentations':1080,
   'base_before':before,'base_after':before,'new_scientific_updates':54,'qualification_scientific_updates':0,'discarded_shadow_Adam_steps':0,
   'CPU_master_state_and_saved_adapter_verified':True,'final_native_checkpoint_Volume_readback_verified':True,'owner_archive_export_pending':True,
   'GPU_backend_amendment':True,'disclosed_GPU_FP32_continuation':True,'strict_equivalence_passed':False,'historical_qualification_reused_as_pass':False,
   'protocol_sha256':c['protocol_sha256'],'task_scope_receipt_sha256':c['task_config_sha256'],'prior_qualification_comparison_sha256':COMPARISON_SHA,'provider_calls':0,'no_old_holdout_opened':True,'original_finetuning_seen_then_original_validation_and_gated_heldout_protocol_pending':True,'common24_NOT_AUTHORIZED_in_this_training_run':True,'no_common24_inputs_read':True})
  proof=read(root/'COMPLETE.json');last=torch.load(root/'LAST_COMPLETED_STATE.pt',map_location='cpu',weights_only=True,mmap=True)
  assert last['stage_updates']==56 and last['native_schedule_next_update']==56 and last['total_updates']==270 and last['scheduler']['last_epoch']==56
  assert len(last['adapter'])==len(last['optimizer']['state'])==372 and all(float(v['step'])==56 for v in last['optimizer']['state'].values())
  assert last['torch_rng'].dtype==torch.uint8 and last['torch_rng'].numel()==5056
  for v in last['optimizer']['state'].values():
   for key,item in v.items():
    if torch.is_tensor(item):assert item.device.type=='cpu' and torch.isfinite(item).all()
  assert len(history)==54 and all(h['row_ids']==p['row_ids'] for h,p in zip(history,plans[2:]))
  assert counts['scientific_forward_attempts']==216 and counts['scientific_CPU_Adam_steps_completed']==counts['scientific_GPU_mirrors_completed']==54
  write(root/'ON_RUNTIME_FINAL270_PROOF.json',{'utc':utc().isoformat(),'actual_H2_updates':214,'actual_Bridge_updates':56,'actual_total_updates':270,'new_scientific_updates':54,'new_model_forward_attempts':216,'CPU_Adam_scheduler_steps':56,'all372_adapter_and_optimizer_states_finite':True,'frozen_base':before,'checkpoint_sha256':final_pin['sha256'],'saved_adapter_matches_CPU_master':True,'working_files_readback_verified':True,'local_backup_readback_pending':True,'not_a_strict_CPU_equivalence_claim':True,'common24_HELD_not_executed':True})
 except BaseException as error:
  write(root/'FAULT.json',{'utc':utc().isoformat(),'phase':phase,'type':type(error).__name__,'actual_total_updates':total,'counts':counts,'automatic_retry':False,'immutable_prior216_preserved':True})
  if phase=='scientific' and opt is not None and sched is not None and 'checkpoint' in locals():
   try:
    completed=total-214
    if sched.last_epoch==completed and all(float(v['step'])==completed for v in opt.state.values()):
     checkpoint(completed);write(root/'COHERENT_CPU_MASTER_FAULT_CHECKPOINT.json',{'utc':utc().isoformat(),'stage_updates':completed,'total_updates':total,'GPU_mirror_may_be_incomplete':True,'clean_completion_not_claimed':True})
   except BaseException as e:write(root/'FAULT_CHECKPOINT_SAVE_FAULT.json',{'utc':utc().isoformat(),'type':type(e).__name__,'last_immutable_checkpoint_preserved':True})
  elif model is not None and state is not None:
   try:
    if 'restore216' in locals():r=restore216()
    else:
     model.zero_grad(set_to_none=True);set_peft_model_state_dict(model,state['adapter']);w.seed(941);torch.set_rng_state(state['torch_rng'])
     assert all(torch.equal(v.detach().cpu(),state['adapter'][k]) for k,v in get_peft_model_state_dict(model).items())
     r={'utc':utc().isoformat(),'adapter_exact216':True,'Torch_RNG_exact216':True,'setup_incomplete':True,'new_qualification_performed':False,'disclosed_science_not_started':True}
    write(root/'FAULT_RESTORATION.json',r)
   except BaseException as e:write(root/'FAULT_RESTORATION_FAILED.json',{'utc':utc().isoformat(),'type':type(e).__name__,'immutable_prior216_preserved':True})
  raise
 finally:
  watch_done.set()
  if watcher is not None:watcher.join(timeout=5)
  if model is not None and old_forward is not None:model.forward=old_forward
  if old_functions:
   for name,fn in old_functions.items():setattr(qwen,name,fn)
  if root.exists():write(root/'FINAL_EXECUTION_COUNTS.json',{'utc':utc().isoformat(),'actual_total_updates':total,'observed_resources':observed,**counts,'new_JEV_calls':0,'new_qualification_forward_attempts':0,'new_shadow_Adam_steps':0,
   'historical_failed_qualification_forward_attempts':8,'historical_failed_qualification_shadow_Adam_steps':2,'strict_equivalence_passed':False,'historical_qualification_reused_as_pass':False,'no_model_size_only_or_cross_device_bitwise_claim':True})
def own_error_context(error):
 context={'type':type(error).__name__,'identifier_key':None,'source_function':None,'source_line':None}
 if isinstance(error,KeyError) and len(error.args)==1 and isinstance(error.args[0],str):
  key=error.args[0]
  if len(key)<=64 and key.isascii() and key.isidentifier():context['identifier_key']=key
 tb=error.__traceback__
 while tb is not None:
  try:
   if Path(tb.tb_frame.f_code.co_filename).resolve()==Path(__file__).resolve():
    context.update(source_function=tb.tb_frame.f_code.co_name,source_line=tb.tb_lineno)
  except (OSError,ValueError):pass
  tb=tb.tb_next
 return context
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();config=None
 try:config=read(args.config);run(config)
 except BaseException as error:
  try:
   target=Path(config['root']) if isinstance(config,dict) and 'root' in config else Path(args.config).parent/'run'
   write(target/'CHILD_VALIDATION_OR_EXECUTION_FAULT_CONTEXT.json',{'utc':utc().isoformat(),**own_error_context(error),'no_error_values_or_traceback_recorded':True})
  except BaseException:pass
  raise
