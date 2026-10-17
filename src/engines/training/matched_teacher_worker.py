"""Bounded matched H2 -> bridge replay. Final-response loss; CPU only."""
import argparse,datetime,gc,hashlib,json,pathlib,random,time
import numpy as np
import psutil,torch
from transformers import Qwen3_5ForConditionalGeneration,get_linear_schedule_with_warmup
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
 p=pathlib.Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial')
 t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def digest_base(model):
 h=hashlib.sha256();count=0
 for name,p in sorted(model.named_parameters(),key=lambda x:x[0].removeprefix('base_model.model.').replace('.base_layer.','.')):
  if 'lora_' in name:continue
  name=name.removeprefix('base_model.model.').replace('.base_layer.','.')
  t=p.detach().cpu().contiguous();h.update(name.encode());h.update(str((list(t.shape),str(t.dtype))).encode())
  h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()));count+=1
 return {'sha256':h.hexdigest(),'parameter_tensors':count}
def seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value)
def adapter_copy(model):return {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}

def group_loss(model,examples,microbatch,pad_id):
 """Right padded native grouping; each example contributes exactly 1/4."""
 assert len(examples)==4
 ordered=sorted(enumerate(examples),key=lambda pair:pair[1]['padded_length'],reverse=True)
 values=[None]*4
 for start in range(0,4,microbatch):
  members=ordered[start:start+microbatch];batch=[e for _,e in members];length=max(e['padded_length'] for e in batch)
  ids=torch.full((len(batch),length),pad_id,dtype=torch.long);mask=torch.zeros_like(ids)
  positions=sorted({p for e in batch for p in e['prediction_positions']});lookup={p:i for i,p in enumerate(positions)}
  for i,e in enumerate(batch):
   assert e['actual_length']==len(e['input_ids']) and e['prefix_length']+len(e['response_ids'])==e['actual_length']
   assert e['prediction_positions']==list(range(e['prefix_length']-1,e['actual_length']-1))
   ids[i,:e['actual_length']]=torch.tensor(e['input_ids']);mask[i,:e['actual_length']]=1
  out=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=torch.tensor(positions))
  losses=[]
  for i,e in enumerate(batch):
   logits=out.logits[i,torch.tensor([lookup[p] for p in e['prediction_positions']])].float()
   loss=torch.nn.functional.cross_entropy(logits,torch.tensor(e['response_ids']))
   assert torch.isfinite(loss);losses.append(loss);values[members[i][0]]=float(loss.detach())
  (torch.stack(losses).sum()/4).backward()
  del out,losses,ids,mask;gc.collect()
 return values

def main(a):
 root=pathlib.Path(a.root);root.mkdir(parents=True,exist_ok=False);packet=pathlib.Path(a.packet)
 assert not torch.cuda.is_available(),'CPU track must not allocate or use a GPU'
 torch.set_num_threads(4);torch.set_num_interop_threads(1);seed(93)
 end=time.monotonic()+a.max_seconds
 model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model,local_files_only=True,trust_remote_code=False,
  dtype=torch.float32,device_map={'':'cpu'},attn_implementation='sdpa',use_safetensors=True)
 model.eval().requires_grad_(False);before=digest_base(model)
 assert before['sha256']==a.expected_base_sha
 targets=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and 'language_model.layers.' in n and
  n.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj','gate_proj','up_proj','down_proj'}]
 assert targets and all('visual' not in n and 'lm_head' not in n for n in targets)
 model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',target_modules=targets))
 assert digest_base(model)==before
 params=[p for p in model.parameters() if p.requires_grad];initial=adapter_copy(model)
 save(root/'REPLAY_FREEZE.json',{'utc':now(),'model':a.model_name,'base':before,'rank':16,'alpha':32,'target_modules':targets,
  'trainable_parameters':sum(p.numel() for p in params),'H2_updates':214,'bridge_updates':56,'presentations':1080,
  'final_response_only':True,'H2_seed':93,'bridge_seed':941,'bridge_optimizer_reset':True,'qualification_checkpoint':'bridge56',
  'H2_intermediate_guard_diagnostic_only':True,'precision':'FP32 CPU','model_size_only_causal_claim':False,
  'holdouts_opened':False,'provider_calls':0})
 data_root=packet/'tokenized'/a.model_name.split('/')[-1]
 datasets={k:json.loads((data_root/(k+'_final_response_tokens.json')).read_bytes()) for k in ('H2','bridge')}
 plans={k:json.loads((packet/'schedule'/(k+('_214_updates.json' if k=='H2' else '_56_updates.json'))).read_bytes()) for k in datasets}
 assert len(datasets['H2'])==428 and len(datasets['bridge'])==112
 audit=json.loads((data_root/'TOKENIZATION_AUDIT.json').read_bytes());pad=int(audit['pad_token_id'])
 def update(group,optimizer):
  optimizer.zero_grad(set_to_none=True);tick=time.monotonic();values=group_loss(model,group,2 if stage=='H2' else 1,pad)
  assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
  norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm);optimizer.step()
  return {'seconds':time.monotonic()-tick,'loss_per_example':values,'host_free_GiB':psutil.virtual_memory().available/2**30,
   'RSS_GiB':psutil.Process().memory_info().rss/2**30,'base_gradients_absent':True,'adapter_gradients_finite':True}
 model.train();measurements={}
 # Fixed longest native update of each stage; scratch state discarded exactly.
 for stage in ('H2','bridge'):
  selected=max(plans[stage],key=lambda p:sum(datasets[stage][i]['padded_length'] for i in p['slots']))
  opt=torch.optim.AdamW(params,lr=1e-4 if stage=='H2' else 5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
  measurements[stage]=update([datasets[stage][i] for i in selected['slots']],opt)
  set_peft_model_state_dict(model,initial);opt.zero_grad(set_to_none=True);del opt;gc.collect()
 assert all(torch.equal(v,initial[k]) for k,v in get_peft_model_state_dict(model).items())
 seed(93);forecast=1.5*(214*measurements['H2']['seconds']+56*measurements['bridge']['seconds'])+600
 save(root/'FEASIBILITY.json',{'utc':now(),'scratch_updates_not_scientific':2,'fresh_adapter_restored_exactly':True,
  'measurements':measurements,'conservative_full_training_seconds':forecast,'bounded_worker_seconds':a.max_seconds,
  'full_dose_may_span_verified_checkpoints':forecast>a.max_seconds,'minimum_free_RAM_GiB':8,'provider_calls':0})
 completed_total=0;history=[];prior=None
 if a.resume:
  prior=torch.load(a.resume,map_location='cpu',weights_only=True);assert prior['base_fingerprint']==before
  assert prior['rank']==16 and prior['alpha']==32 and prior['stage'] in ('H2','bridge')
  set_peft_model_state_dict(model,prior['adapter']);completed_total=prior['total_updates']
 drive=pathlib.Path(a.private_drive_output);drive.mkdir(parents=True,exist_ok=False)
 def checkpoint(stage,completed,opt,scheduler):
  value={'adapter':adapter_copy(model),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'stage':stage,
   'stage_updates':completed,'total_updates':completed_total,'base_fingerprint':before,'seed':93 if stage=='H2' else 941,
   'torch_rng':torch.get_rng_state(),'native_schedule_next_update':completed,'rank':16,'alpha':32}
  dest=root/'LAST_COMPLETED_STATE.pt';temp=dest.with_suffix('.partial.pt');torch.save(value,temp);temp.replace(dest)
  if (stage=='H2' and completed in (107,214)) or (stage=='bridge' and completed in (0,7,14,28,56)):
   immutable=root/'snapshots'/(stage+'_'+str(completed));immutable.mkdir(parents=True,exist_ok=True)
   torch.save(value,immutable/'STATE.pt')
  model.save_pretrained(root/'latest_adapter',safe_serialization=True)
  import shutil
  target=drive/'LAST_COMPLETED_STATE.pt';partial=target.with_suffix('.partial.pt');shutil.copyfile(dest,partial);partial.replace(target)
  with dest.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
  with target.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha
  save(root/'CHECKPOINT_RECEIPT.json',{'utc':now(),'stage':stage,'stage_updates':completed,'total_updates':completed_total,
   'bytes':dest.stat().st_size,'sha256':sha,'private_Drive_readback_verified':True,'C_verification_pending':True})
 for stage,total,lr,warmup,sd in (('H2',214,1e-4,10,93),('bridge',56,5e-5,4,941)):
  if prior and prior['stage']=='bridge' and stage=='H2':continue
  seed(sd);opt=torch.optim.AdamW(params,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=.01)
  scheduler=get_linear_schedule_with_warmup(opt,warmup,total);times=[]
  start=0
  if prior and prior['stage']==stage:
   opt.load_state_dict(prior['optimizer']);scheduler.load_state_dict(prior['scheduler'])
   torch.set_rng_state(prior['torch_rng']);start=prior['stage_updates']
  if stage=='bridge' and not start:checkpoint(stage,0,opt,scheduler)
  for j,planned in enumerate(plans[stage][start:],start=start):
   remaining=end-time.monotonic()
   if remaining<max(180,2*max(times or [measurements[stage]['seconds']])) or psutil.virtual_memory().available<8*2**30 or (a.stop_file and pathlib.Path(a.stop_file).exists()):
    checkpoint(stage,j,opt,scheduler);after=digest_base(model);assert after==before
    save(root/'RESOURCE_BOUNDARY.json',{'utc':now(),'stage':stage,'stage_updates':j,'total_updates':completed_total,
     'next_update':j,'remaining_seconds':remaining,'base_unchanged':True,'incomplete_dose_not_capability_failure':True});return
   assert [datasets[stage][i]['id'] for i in planned['slots']]==planned['row_ids']
   result=update([datasets[stage][i] for i in planned['slots']],opt);scheduler.step();times.append(result['seconds']);completed_total+=1
   history.append(dict(stage=stage,update=j+1,row_ids=planned['row_ids'],**result))
   save(root/'PROGRESS.json',{'utc':now(),'stage':stage,'stage_updates':j+1,'stage_total':total,'total_updates':completed_total,
    'last_seconds':result['seconds'],'host_free_GiB':result['host_free_GiB'],'RSS_GiB':result['RSS_GiB'],
    'base_gradients_absent':True,'checkpoint_total':completed_total if (j+1)%8==0 else None,'provider_calls':0})
   save(root/'LOSS_RECORDS_PRIVATE.json',history)
   if (j+1)%8==0 or (stage=='H2' and j+1 in (107,214)) or (stage=='bridge' and j+1 in (7,14,28,56)):
    checkpoint(stage,j+1,opt,scheduler)
    if (stage=='H2' and j+1 in (107,214)) or (stage=='bridge' and j+1 in (7,14,28,56)):
     model.save_pretrained(root/'snapshots'/(stage+'_'+str(j+1)),safe_serialization=True)
  opt.zero_grad(set_to_none=True);del opt,scheduler;gc.collect()
 after=digest_base(model);assert after==before
 save(root/'COMPLETE.json',{'utc':now(),'completed':True,'H2_updates':214,'bridge_updates':56,'total_updates':270,
  'presentations':1080,'base_before':before,'base_after':after,'final_teacher_qualification_pending':True,
  'no_old_holdout_opened':True,'provider_calls':0,'scientific_success_not_inferred_from_training_loss':True})

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('model','model-name','packet','root','private-drive-output','expected-base-sha'):p.add_argument('--'+n,required=True)
 p.add_argument('--max-seconds',type=int,required=True);p.add_argument('--resume');p.add_argument('--stop-file');a=p.parse_args()
 try:main(a)
 except Exception as e:
  save(pathlib.Path(a.root)/'FAULT.json',{'utc':now(),'type':type(e).__name__,'message':str(e)[:180],
   'automatic_retry':False,'previous_native_checkpoints_preserved':True});raise
