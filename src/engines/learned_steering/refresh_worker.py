"""One frozen sparse-update48 TRAIN refresh; offline literal-prefix full-A logits.

Preparation only. Caller must admit and supervise this bounded native stage.
No bootstrap recapture, new reference, local-target prefill shortcut or heldout access.
"""
import collections,json,pathlib,time,shutil,concurrent.futures
import numpy as np
import torch
import psutil
from continue_worker import SITES,KEYS,read,save,sha,now,RateLimitedEngine
from controller_math import matched_maps
from adapter_state import frozen_base_mode,local_teacher_mode
from native_sites import ResidualSites
from gpu_validation_worker import digest_model
from initial_gate import decode_reference,digest
from decision_weights_v3 import build_index,make_plan,scheduled
from workflow import Decoder,episode


def commit(source,drive,relative,replace=False):
 source,drive=map(pathlib.Path,(source,drive));dest=drive/relative
 dest.parent.mkdir(parents=True,exist_ok=True)
 expected=sha(source)
 if dest.is_file() and not replace:assert sha(dest)==expected,'Existing durable artifact differs'
 else:
  temp=dest.with_suffix(dest.suffix+'.partial');shutil.copyfile(source,temp)
  assert sha(temp)==expected;temp.replace(dest)
 return {'bytes':source.stat().st_size,'sha256':expected,'private_Drive_readback_verified':True,'C_verification_pending':True}


def logit_array(path,positions,width):
 assert psutil.virtual_memory().available>=10*2**30,'Host reserve breached during offline feedback'
 assert shutil.disk_usage(path.parent).free>=positions*width*4+2*2**30,'Disk/export reserve breached'
 return np.lib.format.open_memmap(path.with_suffix('.partial.npy'),mode='w+',dtype=np.float32,shape=(positions,width))


def checked_state(checkpoint,expected_update):
 checkpoint=pathlib.Path(checkpoint);progress=read(checkpoint.parent/'PROGRESS.json')
 assert sha(checkpoint)==progress['checkpoint_sha256']
 state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
 assert state['completed_updates']==expected_update==progress['scientific_optimizer_updates']
 assert state['rank_each']==32 and tuple(state['sites'])==SITES
 assert all(torch.isfinite(v).all() for v in state['maps'].values())
 assert len(state['optimizer']['state'])==12
 assert all(float(v['step'])==expected_update for v in state['optimizer']['state'].values())
 assert all(torch.isfinite(t).all() for v in state['optimizer']['state'].values() for t in v.values() if isinstance(t,torch.Tensor))
 assert state.get('rng'),'Current checkpoints must preserve RNG'
 return state


def check_admission(admission,checkpoint):
 assert admission['quality_review_completed'] and admission['may_continue_TRAIN']
 assert admission['quality_checked_checkpoint_sha256']==sha(checkpoint)
 assert admission['complete_window_admitted'] and admission['teacher_base_unchanged']
 observed=__import__('datetime').datetime.fromisoformat(admission['observed_utc'].replace('Z','+00:00'))
 age=(__import__('datetime').datetime.now(__import__('datetime').timezone.utc)-observed).total_seconds()
 assert 0<=age<=600,'Fresh resource admission required for this stage'


def gates_and_cases(inputs):
 cases=read(inputs/'TRAIN_CASES_PRIVATE.json')
 fixtures={f['case_id']:f for f in read(inputs/'TRAIN_INITIAL_FIXTURES_PRIVATE.json')}
 requests={r['case_id']:r for r in read(inputs/'TRAIN_INITIAL_REQUESTS_PRIVATE.json')}
 receipts=read(inputs/'TRAIN_reference_RECEIPTS_PRIVATE.json');gates={}
 for case in cases:
  request=requests[case['id']];receipt=receipts[case['id']]
  assert receipt['actual_service_call'] and receipt['request_sha256']==request['request_sha256']==digest(request['request'])
  gates[case['id']]=decode_reference(request['request'],receipt['response'])
 assert len(cases)==len(fixtures)==len(gates)==24
 return sorted(cases,key=lambda c:c['id']),fixtures,gates


def run(model,tok,inputs,root,drive,deadline,checkpoint48,admission,resume=False):
 inputs,root,drive,checkpoint48=map(pathlib.Path,(inputs,root,drive,checkpoint48))
 check_admission(admission,checkpoint48);state=checked_state(checkpoint48,48)
 assert getattr(model,'peft_config',None) and not model.training
 assert all(not p.requires_grad and p.grad is None for p in model.parameters())
 before=digest_model(model);assert before==read(inputs/'PROTECTED_DIGESTS.json')['teacher']
 root.mkdir(parents=True,exist_ok=resume);drive.mkdir(parents=True,exist_ok=True)
 assert not (root/'COMPLETE.json').exists(),'Do not repeat completed refresh'
 layers=model.base_model.model.model.language_model.layers
 maps=matched_maps(5120,SITES,64,'two_site').cuda();maps.load_state_dict(state['maps'])
 assert list(maps.state_dict())==list(state['maps'])
 cases,fixtures,gates=gates_and_cases(inputs)
 config=read(inputs/'GENERATION.json');assert config['context_tokens']==8192 and config['greedy'] and not config['thinking']
 engine=RateLimitedEngine(model,deadline,root/'counts');engine.runtime_versions={}
 source_sha=sha(checkpoint48)
 freeze={'utc':now(),'source_update48_sha256':source_sha,'original_initializer_sha256':state['initializer_sha256'],
  'TRAIN_cases':24,'sites':SITES,'six_functions':True,'rank_each':32,'max_tokens_per_case':512,'max_responses':9,
  'context_tokens':8192,'generation_sha256':sha(inputs/'GENERATION.json'),'source_worker_sha256':sha(__file__),
  'one_rollout_per_TRAIN_case_no_outcome_replacement':True,'new_reference_calls':0,'DEV_or_holdout_episodes':0,
  'local_targets_regenerated_at_current_corrected_anchor_during_final_training':True,
  'full_A_logits_offline_identical_literal_prefixes':True,'teacher_outgoing_cache_never_copied':True}
 if (root/'REFRESH_FREEZE.json').is_file():
  old=read(root/'REFRESH_FREEZE.json')
  for key in ('source_update48_sha256','generation_sha256','source_worker_sha256'):assert old[key]==freeze[key]
 else:save(root/'REFRESH_FREEZE.json',freeze)
 commit(root/'REFRESH_FREEZE.json',drive,'REFRESH_FREEZE.json')
 contexts=[];index=[]
 previous_cases=read(root/'ROLLOUT_PROGRESS.json')['Drive_verified_cases'] if resume and (root/'ROLLOUT_PROGRESS.json').is_file() else 0
 assert 0<=previous_cases<=24
 try:
  with frozen_base_mode(model):
   decoder=Decoder(engine,tok,config,layers,maps,'two_site')
   for attempted,case in enumerate(cases,1):
    engine.guard();destination=root/'episodes'/(case['id']+'__sparse_six_update48_TRAIN_refresh.json')
    if destination.is_file():
     assert resume,'Existing rollout requires explicit boundary resume'
     result=read(destination)
     assert result['completed'] and not result['runtime_fault'],'Incomplete/faulted rollout cannot be rerun or replaced'
    else:
     assert not destination.with_suffix('.partial.json').exists(),'Incomplete rollout boundary preserved; audit before retry'
     result=episode(decoder,case,fixtures[case['id']],gates[case['id']],'sparse_six_update48_TRAIN_refresh',root,refresh=True)
    assert result['completed'],'Actual incomplete rollout blocks refresh'
    retained=sum(len(turn['token_ids']) for turn in result['turns'])
    assert retained==result['physical_generated_token_IDs'],'Missing actually emitted partial IDs; preserve and stop'
    assert retained<=512 and len(result['turns'])<=9
    for turn_number,turn in enumerate(result['turns']):
     if not turn['token_ids']:continue
     ci=len(contexts)
     contexts.append({'case_id':case['id'],'history':turn['public_history_before'],
      'prefix_ids':turn['initial_prefix_token_ids'],'source_emitted_ids':turn['token_ids'],
      'source_truncated':turn['truncated'],'source_policy':'sparse_six_update48','turn_index':turn_number,'notice_turn_index':0,
      'tools':fixtures[case['id']]['tools']})
     index.extend({'case_id':case['id'],'context_index':ci,'prediction_index':j,'gate_on':gates[case['id']].enabled}
       for j in range(len(turn['token_ids'])))
    if attempted<=previous_cases:continue  # Reconstruct existing IDs without regressing durable progress.
    save(root/'CONTEXTS_PRIVATE.json',contexts);save(root/'POSITION_INDEX_PRIVATE.json',index)
    # Durable progress advances only after actual per-case outputs and IDs verify.
    for name in ('CONTEXTS_PRIVATE.json','POSITION_INDEX_PRIVATE.json'):commit(root/name,drive,name,replace=True)
    commit(destination,drive,'episodes/'+destination.name)
    save(root/'ROLLOUT_PROGRESS.json',{'utc':now(),'Drive_verified_cases':attempted,'scheduled_cases':24,
     'retained_generated_positions':len(index),'outcome_based_case_replacement':False})
    # Progress files are mutable receipts; publish their current verified bytes.
    temp=drive/'ROLLOUT_PROGRESS.json.partial';shutil.copyfile(root/'ROLLOUT_PROGRESS.json',temp)
    assert sha(temp)==sha(root/'ROLLOUT_PROGRESS.json');temp.replace(drive/'ROLLOUT_PROGRESS.json')
   assert all(any(r['case_id']==c['id'] for r in index) for c in cases),'Empty refresh case blocks prescribed mixture'
   annotated,annotations,pools,context_rows,priorities,weights=build_index(contexts,lambda ids:tok.decode(ids,skip_special_tokens=True))
   assert len(annotated)==len(index)
   for row,actual in zip(annotated,index):
    assert (row['case_id'],row['context_index'],row['prediction_index'])==(actual['case_id'],actual['context_index'],actual['prediction_index'])
   plan=make_plan(contexts,annotated,annotations,pools,context_rows,priorities,pre_budget=32,post_budget=0)
   save(root/'REFINEMENT_PLAN_PRIVATE.json',plan);save(root/'CONTEXT_ANNOTATIONS_PRIVATE.json',annotations)
   np.save(root/'CASE_NORMALIZED_DECISION_WEIGHTS.npy',weights)
   summary={'feasible':plan['feasible'],'infeasible_cases':plan['infeasible_cases'],
    'available_spans':len(plan['coverage']),'covered_spans':sum(v['covered'] for v in plan['coverage']),
    'actual_generated_positions':len(index),'missing_stage_counts_are_not_invented':True}
   summary['available_context_stages']={name:sum(name in c['context_tags'] for c in annotations)
     for name in ('after_grant_current_task_pending','current_task_submission','finish_decision')}
   save(root/'DECISION_COVERAGE_SUMMARY.json',summary)
   for name in ('REFINEMENT_PLAN_PRIVATE.json','CONTEXT_ANNOTATIONS_PRIVATE.json','CASE_NORMALIZED_DECISION_WEIGHTS.npy','DECISION_COVERAGE_SUMMARY.json'):
    commit(root/name,drive,name)
   assert plan['feasible'] and all(v['covered'] for v in plan['coverage']),'Actual history exceeds fixed32-anchor budget'
   selected=sorted({row for p in plan['cases'].values() for row in p['pre_rows']})
   save(root/'FEATURE_INDEX_PRIVATE.json',{str(r):i for i,r in enumerate(selected)});commit(root/'FEATURE_INDEX_PRIVATE.json',drive,'FEATURE_INDEX_PRIVATE.json')
   logits_root=root/'teacher_logits';logits_root.mkdir(exist_ok=True)
   writer=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='private-Drive-writer')
   pending=collections.deque();durable=0;manifest={};offset=0
   def finish_one():
    nonlocal durable
    ci,future=pending.popleft();manifest[str(ci)]=future.result();durable+=1
    save(root/'FEEDBACK_PROGRESS.json',{'utc':now(),'Drive_verified_contexts':durable,'scheduled_contexts':len(contexts),
     'new_reference_calls':0,'full_A_feedback_offline':True})
   try:
    for ci,c in enumerate(contexts):
     engine.guard();assert psutil.virtual_memory().available>=10*2**30,'Host reserve breached'
     path=logits_root/f'context_{ci:03}.npy';receipt=drive/'feedback_contexts'/f'context_{ci:03}.json'
     if path.is_file() and receipt.is_file():
      assert resume
      expected=read(receipt);assert sha(path)==expected['sha256']
      m=np.load(path,mmap_mode='r');assert m.shape[0]==len(c['source_emitted_ids']);m._mmap.close()
      manifest[str(ci)]=expected;durable+=1;offset+=len(c['source_emitted_ids']);continue
     assert not receipt.exists(),'Durable feedback receipt without recoverable local array'
     prefix=tok.apply_chat_template(c['history'],tools=c['tools'],tokenize=True,add_generation_prompt=True,enable_thinking=False,preserve_thinking=False)
     if hasattr(prefix,'keys'):prefix=prefix['input_ids']
     if prefix and isinstance(prefix[0],list):prefix=prefix[0]
     assert prefix==c['prefix_ids'],'Recorded refresh causal prefix changed'
     enabled=gates[c['case_id']].enabled;past=None;logits=None
     hook=ResidualSites(layers,None,SITES,'two_site','predicted',maps,audit_state=False);hook.enabled=enabled;hook.record=False
     try:
      for j,expected_id in enumerate(c['source_emitted_ids']):
       with torch.inference_mode():out=engine.forward(c['prefix_ids'] if j==0 else [c['source_emitted_ids'][j-1]],past,len(c['prefix_ids'])+j)
       past=out.past_key_values
       assert torch.isfinite(out.logits).all(),'Non-finite refresh replay logits'
       assert int(out.logits[0,-1].argmax())==expected_id,'Update48 refresh literal-cache replay changed actual generated ID'
       if not enabled:
        value=out.logits.detach().float().cpu().reshape(-1).numpy()
        if logits is None:logits=logit_array(path,len(c['source_emitted_ids']),len(value))
        logits[j]=value
       del out
     finally:hook.close();del past
     if enabled:
      # Full-A output reference is separate from local submodule counterfactuals.
      # Hooks are removed; its private full-A cache is rebuilt from literal IDs.
      past=None
      try:
       with torch.inference_mode(),local_teacher_mode(model):
        for j in range(len(c['source_emitted_ids'])):
         out=engine.forward(c['prefix_ids'] if j==0 else [c['source_emitted_ids'][j-1]],past,len(c['prefix_ids'])+j)
         past=out.past_key_values;value=out.logits.detach().float().cpu().reshape(-1).numpy()
         assert np.isfinite(value).all(),'Non-finite offline full-A logits'
         if logits is None:logits=logit_array(path,len(c['source_emitted_ids']),len(value))
         logits[j]=value;del out
      finally:del past
     assert logits is not None and np.isfinite(logits).all()
     logits.flush();logits._mmap.close();path.with_suffix('.partial.npy').replace(path)
     def write_context(p=path,k=ci):
      verified=commit(p,drive,'teacher_logits/'+p.name)
      save(root/'feedback_contexts'/f'context_{k:03}.json',verified)
      commit(root/'feedback_contexts'/f'context_{k:03}.json',drive,'feedback_contexts/'+f'context_{k:03}.json')
      return verified
     pending.append((ci,writer.submit(write_context)));offset+=len(c['source_emitted_ids'])
     if len(pending)>=2:finish_one()
    while pending:finish_one()
   finally:
    writer.shutdown(wait=True,cancel_futures=False);engine.flush()
   assert durable==len(contexts) and offset==len(index)
   save(root/'TEACHER_LOGIT_MANIFEST.json',manifest);commit(root/'TEACHER_LOGIT_MANIFEST.json',drive,'TEACHER_LOGIT_MANIFEST.json')
  assert digest_model(model)==before,'Frozen base/teacher state changed during refresh'
  metadata_names=('REFRESH_FREEZE.json','CONTEXTS_PRIVATE.json','POSITION_INDEX_PRIVATE.json','REFINEMENT_PLAN_PRIVATE.json','TEACHER_LOGIT_MANIFEST.json')
  complete={'utc':now(),'completed':True,'source_update48_sha256':source_sha,'TRAIN_cases':24,'refresh_rollouts':24,
   'retained_generated_positions':len(index),'contexts':len(contexts),'unique_prescribed_feedback_positions':len(selected),
   'all_generated_IDs_retained':True,'full_A_logits_literal_prefix_only':True,'all_teacher_logit_files_Drive_verified':True,
   'current_local_targets_recomputed_during_final_training':True,'teacher_base_unchanged':True,'new_reference_calls':0,
   'DEV_or_holdout_episodes':0,'metadata_sha256':{name:sha(root/name) for name in metadata_names}}
  save(root/'COMPLETE.json',complete);commit(root/'COMPLETE.json',drive,'COMPLETE.json');return complete
 finally:
  engine.flush()
  del maps
