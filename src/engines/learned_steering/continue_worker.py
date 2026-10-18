"""Resume the frozen sparse six-map TRAIN schedule from saved native Adam state.

Reuses all completed bootstrap capture; no fitting, recapture or DEV/holdout.
Stops at original-schedule index47 because index48 requires TRAIN refresh.
"""
import collections,datetime,gc,hashlib,json,pathlib,sys,time,random
import numpy as np
import torch
from safetensors.torch import save_file
from peft import PeftModel
from native_sites import ResidualSites,clone_state
from adapter_state import frozen_base_mode,local_teacher_mode
from controller_math import matched_maps,distillation_objective
from training_targets import target_basis,projected_energy,initialize_with_basis
from decision_weights_v3 import scheduled
from pilot_worker import Engine
from gpu_validation_worker import digest_model
from initial_gate import decode_reference,digest
from gradient_memory import CheckpointMLPs,GuardedSaveOnCPU
SITES=(31,47,62);KEYS=tuple(f'{i}/{k}' for i in SITES for k in ('attention','output'))
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial')
 t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(p):
 with pathlib.Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def full_kl(t,s):
 t=np.asarray(t,dtype=np.float64).reshape(-1);s=np.asarray(s,dtype=np.float64).reshape(-1)
 tl=t-t.max();tl-=np.log(np.exp(tl).sum());sl=s-s.max();sl-=np.log(np.exp(sl).sum())
 return float(np.sum(np.exp(tl)*(tl-sl)))

class RateLimitedEngine(Engine):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.last_flush=0.;self.last_flush_count=0
 def flush(self):
  save(self.root/'PHYSICAL_COUNTS.json',{'utc':now(),'model_forward_attempts':self.attempts,
   'model_forwards_completed':self.completed,'emitted_token_IDs':self.tokens,
   'new_reference_calls_in_GPU_worker':0,'disk_receipt_rate_limited':True})
  self.last_flush=time.monotonic();self.last_flush_count=self.completed
 def forward(self,ids,past,total,grad=False,use_cache=True):
  self.guard();self.attempts+=1
  x=torch.tensor([ids],device='cuda');mask=torch.ones((1,total),device='cuda',dtype=torch.long)
  with torch.set_grad_enabled(grad):
   out=self.model(input_ids=x,attention_mask=mask,past_key_values=past,use_cache=use_cache,logits_to_keep=1)
  self.completed+=1
  if self.completed-self.last_flush_count>=128 or time.monotonic()-self.last_flush>=5:self.flush()
  return out

def run(model,tok,inputs,root,drive,deadline,prior,stop_after=48,step_reserve_seconds=600,resume_checkpoint=None):
 inputs,root,drive,prior=map(pathlib.Path,(inputs,root,drive,prior));root.mkdir(exist_ok=False)
 assert 2<stop_after<=48,'Original-only resume must stop before required TRAIN refresh'
 checkpoint=pathlib.Path(resume_checkpoint) if resume_checkpoint else prior/'UPDATE_002.pt'
 prior_complete=read(prior/'COMPLETE.json')
 assert prior_complete['pilot_complete'] and prior_complete['scientific_optimizer_updates']==2
 assert prior_complete['teacher_base_unchanged'] and not prior_complete['joint_deterioration_requires_audit']
 resume_progress=read(checkpoint.parent/'PROGRESS.json')
 assert sha(checkpoint)==resume_progress['checkpoint_sha256']
 resumed=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
 start_update=int(resumed['completed_updates'])
 assert 2<=start_update<stop_after<=48 and resumed['rank_each']==32 and tuple(resumed['sites'])==SITES
 assert resume_progress['scientific_optimizer_updates']==start_update
 assert all(torch.isfinite(v).all() for v in resumed['maps'].values())
 assert len(resumed['optimizer']['state'])==12
 assert all(float(v['step'])==start_update for v in resumed['optimizer']['state'].values())
 assert all(torch.isfinite(t).all() for v in resumed['optimizer']['state'].values() for t in v.values() if isinstance(t,torch.Tensor))
 assert resumed.get('rng') or start_update==2,'Later checkpoints must preserve RNG state'
 assert step_reserve_seconds>=600

 layers=model.model.language_model.layers;assert len(layers)==64
 before=digest_model(model);assert before==read(inputs/'PROTECTED_DIGESTS.json')['base']
 save(root/'BASE_PROTECTED.json',before)
 contexts=read(inputs/'TRAIN_CONTEXTS_PRIVATE.json');requests=read(inputs/'TRAIN_INITIAL_REQUESTS_PRIVATE.json')
 receipts=read(inputs/'TRAIN_reference_RECEIPTS_PRIVATE.json');assert len(contexts)==136 and len(requests)==len(receipts)==24
 gates={}
 for request in requests:
  receipt=receipts[request['case_id']]
  assert receipt['actual_service_call'] and receipt['request_sha256']==request['request_sha256']==digest(request['request'])
  gates[request['case_id']]=decode_reference(request['request'],receipt['response'])
 index=read(inputs/'POSITION_INDEX_PRIVATE.json');weights=np.load(inputs/'CASE_NORMALIZED_DECISION_WEIGHTS.npy')
 plan=read(inputs/'REFINEMENT_PLAN_PRIVATE.json');assert len(index)==len(weights)==4539 and plan['feasible']
 for ci,c in enumerate(contexts):
  rendered=tok.apply_chat_template(c['history'],tools=c['tools'],tokenize=True,add_generation_prompt=True,enable_thinking=False,preserve_thinking=False)
  if hasattr(rendered,'keys'):rendered=rendered['input_ids']
  if rendered and isinstance(rendered[0],list):rendered=rendered[0]
  assert rendered==c['prefix_ids'],'Frozen causal TRAIN prefix changed'
 panel=read(inputs/'FOCUSED_CONTEXTS_PRIVATE.json');assert len({c['case_id'] for c in panel})==4
 save(root/'EXECUTION_FREEZE.json',{'utc':now(),'sites':SITES,'functions':6,'TRAIN_cases':24,'all_prediction_positions':4539,
  'bootstrap_trajectory':'sparse zero corrections','refinement_local_targets':'actual current sparse predicted trajectory',
  'MLP_targets_conditioned_on_actual_corrected_mid':True,'full_A_logits_identical_causal_prefixes_only':True,
  'rank_candidates_per_function':[8,16,32,51,64,96,128],'pooled_target_energy_criterion':.8,
  'resumed_completed_updates':start_update,'stop_after_completed_updates':stop_after,'positions_per_update':32,'fixed_panel':{'cases':4,'anchors':sum(len(c['anchors']) for c in panel)},
  'DEV_and_holdouts_opened':False,'new_reference_calls':0,'frozen_initial_gates':24,'teacher_at_adaptive_inference':False,
  'protocol_sha256':sha(inputs/'PROTOCOL.md'),'schedule_sha256':sha(inputs/'REFINEMENT_PLAN_PRIVATE.json'),'reuses_complete_bootstrap':True,
  'resume_checkpoint_sha256':sha(checkpoint),'next_schedule_index':start_update,
  'new_initialization_or_capture':False,'original_pilot_rng_not_saved':True,
  'deterministic_eval_no_sampling_dropout_or_random_schedule':True})
 adapter=inputs/'A';assert sha(adapter/'adapter_model.safetensors')=='8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa'
 assert sha(adapter/'adapter_config.json')=='06a6ad4588f4f0d7b46cf9f7f67f9b7b5028d153134cdd0829a929e1caf9fcda'
 model=PeftModel.from_pretrained(model,adapter,is_trainable=False);model.eval().requires_grad_(False)
 layers=model.base_model.model.model.language_model.layers
 protected_teacher=digest_model(model);assert protected_teacher==read(inputs/'PROTECTED_DIGESTS.json')['teacher']
 engine=RateLimitedEngine(model,deadline,root/'counts');engine.runtime_versions={}
 def stage(name,**extra):save(root/'RUNNING_STAGE.json',dict(utc=now(),stage=name,model_loaded=True,**extra))
 def hook(mode,maps=None,record=False,reference=False):
  h=ResidualSites(layers,layers if reference else None,SITES,'two_site',mode,maps,audit_state=False,
    teacher_context=local_teacher_mode);h.record=record;return h
 def replay(c,until,h,grad=False):
  past=None;out=None
  for j in range(until+1):
   h.records=[]
   if grad and j==until and past is not None:past=clone_state(past)
   out=engine.forward(c['prefix_ids'] if j==0 else [c['source_emitted_ids'][j-1]],past,len(c['prefix_ids'])+j,grad=grad and j==until)
   past=out.past_key_values
   if j<until:del out
  return out
 with frozen_base_mode(model):
  stage('native_zero_hook_cache_and_gradient_checks',scientific_optimizer_updates=start_update)
  c=next(c for c in contexts if gates[c['case_id']].enabled)
  def greedy(wrapped,cached):
   emitted=[];past=None;h=hook('zero') if wrapped else None
   if h:h.enabled=True
   try:
    for j in range(4):
     ids=(c['prefix_ids'] if j==0 else [emitted[-1]]) if cached else c['prefix_ids']+emitted
     out=engine.forward(ids,past if cached else None,len(c['prefix_ids'])+j,use_cache=cached)
     emitted.append(int(out.logits[0,-1].argmax()));past=out.past_key_values if cached else None;del out
   finally:
    if h:h.close()
    del past
   return emitted
  raw=greedy(False,True);assert raw==greedy(False,False)==greedy(True,True)==greedy(True,False)
  h=hook('zero',record=True,reference=True);h.enabled=True;h.audit_state=True;h.audit_other_layers=True
  try:
   out=replay(c,1,h);del out
   assert len(h.records)==3 and all(r['current_state_not_rewritten'] and not r['teacher_outgoing_state_copied'] for r in h.records)
   assert all(r['MLP_target_input_is_actual_corrected_mid'] for r in h.records)
   assert any(float(r['attention_target'].norm())>0 for r in h.records)
  finally:h.close()
  scratch=matched_maps(5120,SITES,16,'two_site').cuda();h=hook('predicted',scratch);h.enabled=True
  trace=[];handles=[]
  for i,layer in enumerate(layers):
   def observe(module,args,kwargs,_i=i):
    value=args[0] if args else kwargs.get('hidden_states')
    trace.append({'layer':_i,'input_requires_grad':bool(isinstance(value,torch.Tensor) and value.requires_grad)})
   handles.append(layer.register_forward_pre_hook(observe,with_kwargs=True))
  try:
   out=engine.forward(c['prefix_ids'],None,len(c['prefix_ids']),grad=True,use_cache=False)
   out.logits.float().square().mean().backward();del out
   assert all(p.grad is None for p in model.parameters())
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in scratch.parameters())
   assert len(trace)==64 and not any(r['input_requires_grad'] for r in trace if r['layer']<=31)
   assert all(r['input_requires_grad'] for r in trace if r['layer']>31)
  finally:
   h.close()
   for handle in handles:handle.remove()
   del scratch;gc.collect();torch.cuda.empty_cache()
  save(root/'MECHANICS_VERIFIED.json',{'utc':now(),'native_zero_cached_uncached_greedy_IDs_equal':True,'checked_tokens_each_path':4,
   'teacher_local_cache_isolation_checked':True,'teacher_actual_frozen_adapter_enabled_only_inside_no_grad_local_calls':True,
   'base_gradients_absent':True,'no_dummy_gradient':True,'gradient_cutoff_first_site31_verified':True,
   'layer_input_requires_grad':trace,'scratch_backward_only_optimizer_updates':0,'scientific_optimizer_updates':start_update})
  stage('verify_and_reuse_completed_sparse_bootstrap',scientific_optimizer_updates=start_update)
  folder=prior/'bootstrap';capture=read(prior/'CAPTURE_PROGRESS.json')
  assert capture['contexts']==136 and capture['positions']==4539
  active=np.array([r['gate_on'] for r in index]);assert active.any()
  for key in KEYS:
   for field in ('x','target'):
    values=np.load(folder/key/(field+'.npy'),mmap_mode='r')
    assert values.shape==(4539,5120) and values.dtype==np.float32
    assert all(np.isfinite(values[i:i+128]).all() for i in range(0,4539,128))
    values._mmap.close()
  rank=resumed['rank_each'];maps=matched_maps(5120,SITES,2*rank,'two_site').cuda()
  maps.load_state_dict(resumed['maps'],strict=True)
  assert list(maps.state_dict())==list(resumed['maps']),'Adam parameter order changed'
  initializer=prior/'RIDGE_INITIALIZER.safetensors'
  assert sha(initializer)==resumed['initializer_sha256']==read(prior/'CAPACITY_AND_INITIALIZER.json')['initializer_sha256']
  scales={};w=torch.from_numpy(weights.astype(np.float64)).cuda()
  for key in KEYS:
   y=torch.from_numpy(np.array(np.load(folder/key/'target.npy',mmap_mode='r'),copy=True)).cuda()
   scales[key]=max(float((y.double().square()*w[:,None]).sum()/(w[torch.from_numpy(active).cuda()].sum()*5120)),1e-6)
   del y
  del w
  save(root/'REUSED_CAPTURE_AND_RESUME.json',{'utc':now(),'contexts':136,'positions':4539,
   'rank_each':rank,'maps':6,'initializer_sha256':sha(initializer),'checkpoint_sha256':sha(checkpoint),
   'original_capture_reused_without_recollection':True,'scales':scales,
   'prior_total_physical_counts':read(prior/'counts'/'PHYSICAL_COUNTS.json'),
   'prior_scientific_updates':start_update,'next_schedule_index':start_update,'no_new_reference_calls':True})
  # Teacher output rows are disk-backed, aligned with the same recorded context.
  def target_logits(ci,j):return torch.from_numpy(np.array(np.load(inputs/'TEACHER_LOGITS_PRIVATE'/f'context_{ci:03}.npy',mmap_mode='r')[j],copy=True)).cuda()
  def diagnostic(label):
   grouped=collections.defaultdict(list);fixed=[]
   for c in panel:
    h=hook('predicted',maps);h.enabled=c['gate_on']
    try:
     for anchor in c['anchors']:
      out=replay(c,anchor['prediction_index'],h);ci=c['context_index'];j=anchor['prediction_index']
      teacher=target_logits(ci,j) if c['gate_on'] else out.logits.detach().float()
      grouped[c['case_id']].append(full_kl(teacher.detach().cpu().numpy(),out.logits.detach().float().cpu().numpy()));del out,teacher
      row=anchor['row'];errors=[]
      for key in KEYS:
       x=np.load(folder/key/'x.npy',mmap_mode='r');y=np.load(folder/key/'target.npy',mmap_mode='r')
       with torch.no_grad():pred=maps[key](torch.tensor(np.array(x[row],copy=True),device='cuda'))
       errors.append(float((pred-torch.tensor(np.array(y[row],copy=True),device='cuda')).square().mean()/scales[key]));del pred
       x._mmap.close();y._mmap.close()
      fixed.append({'case_id':c['case_id'],'gate_on':c['gate_on'],'normalized_fixed_bootstrap_MSE':sum(errors)/len(errors)})
    finally:h.close()
   own={c['case_id'] for c in panel if c['gate_on']};own_kl=np.mean([np.mean(v) for k,v in grouped.items() if k in own])
   own_mse=np.mean([np.mean([r['normalized_fixed_bootstrap_MSE'] for r in fixed if r['case_id']==k]) for k in own])
   result={'utc':now(),'label':label,'own_case_macro_full_vocabulary_forward_KL':float(own_kl),
    'own_case_macro_fixed_bootstrap_normalized_MSE':float(own_mse),'KL_by_case':{k:float(np.mean(v)) for k,v in grouped.items()},
    'gate_off_is_common_base_fallback_not_intrinsic_selectivity':True,'fixed_rows':fixed,'TRAIN_diagnostic_not_validation':True}
   save(root/(label+'_FIXED_TRAIN_DIAGNOSTIC.json'),result);return result
  baseline=read(prior/'INITIALIZER_FIXED_TRAIN_DIAGNOSTIC.json')
  reference_path=checkpoint.parent/f'UPDATE_{start_update:03}_FIXED_TRAIN_DIAGNOSTIC.json'
  # A completed resource segment saves this diagnostic. An interrupted segment
  # may have only a verified checkpoint; recompute and record its fixed baseline.
  resume_reference=read(reference_path) if reference_path.is_file() else None
  restored=diagnostic(f'RESUMED_UPDATE_{start_update:03}')
  if resume_reference is None:resume_reference=restored

  fields=('own_case_macro_full_vocabulary_forward_KL','own_case_macro_fixed_bootstrap_normalized_MSE')
  for field in fields:assert np.isclose(restored[field],resume_reference[field],rtol=1e-5,atol=1e-7),'Restored fixed diagnostic changed'
  assert set(restored['KL_by_case'])==set(resume_reference['KL_by_case'])
  for case,value in restored['KL_by_case'].items():
   assert np.isclose(value,resume_reference['KL_by_case'][case],rtol=1e-5,atol=1e-7),'Restored case KL changed'
  assert len(restored['fixed_rows'])==len(resume_reference['fixed_rows'])
  for actual,expected in zip(restored['fixed_rows'],resume_reference['fixed_rows']):
   assert (actual['case_id'],actual['gate_on'])==(expected['case_id'],expected['gate_on'])
   assert np.isclose(actual['normalized_fixed_bootstrap_MSE'],expected['normalized_fixed_bootstrap_MSE'],rtol=1e-5,atol=1e-7),'Restored fixed row changed'
  engine.flush()
  memory=CheckpointMLPs(layers,pure_delta_rule=True);optimizer=torch.optim.AdamW(maps.parameters(),lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)
  optimizer.load_state_dict(resumed['optimizer'])
  assert all(float(v['step'])==start_update for v in optimizer.state.values())
  assert all(g['lr']==2e-4 and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0 for g in optimizer.param_groups)
  assert not model.training and all(not m.training for m in model.modules())
  # No random operation after overwritten map construction participates in training.
  # Save future RNG states for completeness even though this schedule/eval is deterministic.
  if resumed.get('rng'):
   rng=resumed['rng'];random.setstate(rng['python'])
   a,b,c,d,e=rng['numpy'];np.random.set_state((a,b.numpy().astype(np.uint32),c,d,e))
   torch.set_rng_state(rng['torch_CPU']);torch.cuda.set_rng_state_all(rng['torch_CUDA'])
  completed=start_update;boundary=False
  try:
   for update in range(start_update,stop_after):
    if time.monotonic()+step_reserve_seconds>=deadline:
     boundary=True;save(root/'RESOURCE_BOUNDARY.json',{'utc':now(),'completed_updates':completed,
      'next_schedule_index':completed,'remaining_seconds':deadline-time.monotonic(),
      'step_reserve_seconds':step_reserve_seconds,'last_completed_checkpoint_preserved':True});break
    stage('six_function_output_refinement',scientific_optimizer_updates=update)
    tick=time.monotonic();optimizer.zero_grad(set_to_none=True);chosen=scheduled(plan,update)
    by_context=collections.defaultdict(dict)
    for case,source,row,mult in chosen:
     assert source=='original' and index[row]['case_id']==case
     r=index[row];by_context[r['context_index']][r['prediction_index']]=(row,mult)
    loss_total=0.
    for ci,positions in sorted(by_context.items()):
     c=contexts[ci];h=hook('predicted',maps,record=True,reference=True);h.enabled=gates[c['case_id']].enabled;past=None
     try:
      for j in range(max(positions)+1):
       anchor=j in positions;h.records=[];h.record=anchor
       if anchor and past is not None:past=clone_state(past)
       with GuardedSaveOnCPU(pin_memory=True,reserve_bytes=10*2**30):
        out=engine.forward(c['prefix_ids'] if j==0 else [c['source_emitted_ids'][j-1]],past,len(c['prefix_ids'])+j,grad=anchor)
        if anchor:
         row,mult=positions[j];predicted=[];targets=[];normalizers=[]
         for r in h.records:
          for kind in ('attention','output'):
           key=f"{r['layer']}/{kind}";predicted.append(maps[key](r[kind+'_input'].cuda()));targets.append(r[kind+'_target'].cuda());normalizers.append(scales[key])
         teacher=target_logits(ci,j) if h.enabled else out.logits.detach()
         loss,terms=distillation_objective(out.logits,teacher,predicted,targets,normalizers,h.enabled)
         (loss*mult/32).backward();loss_total+=float(loss.detach())*mult/32;del loss,teacher,predicted,targets
       past=out.past_key_values;del out
     finally:h.close();del past;gc.collect();torch.cuda.empty_cache()
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in maps.parameters())
    norm=torch.nn.utils.clip_grad_norm_(maps.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
    completed=update+1
    assert all(torch.isfinite(p).all() for p in maps.parameters()),'Post-update map state is non-finite'
    assert all(float(v['step'])==completed for v in optimizer.state.values())
    assert all(torch.isfinite(t).all() for v in optimizer.state.values() for t in v.values() if isinstance(t,torch.Tensor)),'Post-update Adam state is non-finite'
    cp=root/f'UPDATE_{completed:03}.pt';torch.save({'maps':{k:v.detach().cpu() for k,v in maps.state_dict().items()},
     'optimizer':optimizer.state_dict(),'completed_updates':completed,'rank_each':rank,'sites':SITES,
     'initializer_sha256':sha(initializer),'next_schedule_index':completed,
     'source_checkpoint_sha256':sha(checkpoint),'objective_and_schedule_unchanged':True,
     'rng':{'python':random.getstate(),'numpy':(np.random.get_state()[0],torch.tensor(np.random.get_state()[1].astype(np.int64)),*np.random.get_state()[2:]),
            'torch_CPU':torch.get_rng_state(),'torch_CUDA':torch.cuda.get_rng_state_all()}},cp)
    import shutil
    drive.mkdir(parents=True,exist_ok=True);dest=drive/cp.name;shutil.copyfile(cp,dest);assert sha(cp)==sha(dest)
    save(root/'PROGRESS.json',{'utc':now(),'scientific_optimizer_updates':completed,'stop_after_completed_updates':stop_after,'positions_presented':32*completed,
     'step_seconds':time.monotonic()-tick,'step_weighted_loss':loss_total,'base_gradients_absent':True,'controller_gradients_finite':True,
     'checkpoint_sha256':sha(cp),'private_Drive_readback_verified':True,'C_verification_pending':True,'new_reference_calls':0,'next_schedule_index':completed,
     'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30,'host_free_GiB':__import__('psutil').virtual_memory().available/2**30})
    engine.flush()
  finally:
   memory.close();engine.flush()
  current=diagnostic(f'UPDATE_{completed:03}') if time.monotonic()+300<deadline else None
  joint=None if current is None else (current[fields[0]]>restored[fields[0]] and current[fields[1]]>restored[fields[1]])
  assert digest_model(model)==protected_teacher,'Frozen base/teacher/quantization state changed'
  engine.flush()
  save(root/'SEGMENT_COMPLETE.json',{'utc':now(),'segment_complete':True,'scientific_optimizer_updates':completed,
   'starting_updates':start_update,'new_updates':completed-start_update,'sites':SITES,'functions':6,'rank_each':rank,
   'resume_reference':restored,'last_fixed_diagnostic':current,'joint_deterioration_requires_audit':joint,
   'teacher_base_unchanged':True,'all_base_gradients_absent':all(p.grad is None for p in model.parameters()),
   'next_schedule_index':completed,'resource_boundary':boundary,
   'TRAIN_refresh_required_before_update49':completed==48,
   'new_DEV_or_holdout_episodes':0,'new_workflow_episodes':0,'new_reference_calls':0,
   'full_96_schedule_complete':False,'DEV_or_holdout_eligible':False})
 return model
