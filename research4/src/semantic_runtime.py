"""Evaluate the declared learned task baseline and fixed old LoRA, without fitting."""
import argparse
import gc
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from .io_utils import Budget,digest,dump,load_torch,read,settings,sha
from .runtime import archive,persist_array_cache
from .recovery_data import dataset,probe_rows
from .recovery_runtime import rollout,condition_summary
from .context_runtime import outcome_checks


def run(root,artifact,model,checkpoint,fit_receipt):
    os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    import torch
    import numpy as np
    import yaml
    from .semantic_policy import SemanticActor
    root,artifact,model,checkpoint=map(Path,[root,artifact,model,checkpoint])
    if subprocess.check_output(['git','status','--porcelain','--','research4'],cwd=root.parent,text=True).strip():
        raise RuntimeError('Commit the complete evaluation source before native calls')
    config=yaml.safe_load((root/'config/semantic_policy_v3.yaml').read_text());common,_=settings(root)
    data_config=yaml.safe_load((root/'config/recovery.yaml').read_text());specs,rows,meta=dataset(data_config)
    fit=read(fit_receipt)
    if fit['status']!='complete' or fit['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Require the exact completed permitted fit artifact')
    if fit['Qwen_parameters_changed'] or fit['LoRA_updates']:raise ValueError('The correction changed more than the usefulness head')
    saved=load_torch(checkpoint)
    if saved['original_checkpoint_sha256']!=config['adapter_checkpoint_sha256']:raise ValueError('Wrong original pretrained artifact')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root.parent,text=True).strip()
    output=artifact/commit;output.mkdir(parents=True,exist_ok=False)
    budget=Budget({**common['budget'],'internal_stop_seconds':1800,'protected_export_seconds':120,'maximum_backbone_forwards':360})
    actor=None;actor_choose=None;started=time.monotonic();context={'stage':'training_probe','condition':'base','case':''};decisions=0;dev_seen=0
    result={'status':'running','qualified':False,'source_commit':commit,'task_baseline':'learned heads plus visible tool preconditions',
            'raw_Qwen_baseline_unchanged_claimed':False,'optimizer_updates_in_evaluation':0,'GPU_allocations':0,
            'held_model_evaluations':0,'fit_checkpoint_sha256':sha(checkpoint),'fit_execution':fit['execution']}
    try:
        from .recovery_local_probe import available_memory
        torch.set_num_threads(2);torch.manual_seed(config['seed']);assert not torch.cuda.is_available()
        if available_memory()<common['budget']['minimum_available_memory_bytes']:raise RuntimeError('Memory admission failed')
        if sha(model/'model.safetensors-00001-of-00001.safetensors')!=common['model']['checkpoint_sha256']:raise ValueError('Wrong pinned base')
        dump(output/'PRECALL_FREEZE.json',{'source_commit':commit,'config':config,'dataset':meta,
                                          'checkpoint_sha256':sha(checkpoint),'fit_receipt':fit,'development_decisions_seen':0})
        actor=SemanticActor(model,common['model'],budget,saved)
        actor_choose=actor.choose
        def choose(view,on):
            nonlocal decisions,dev_seen
            action,logits=actor_choose(view,on);decisions+=1
            if context['stage']=='development':dev_seen+=1
            dump(output/'READOUT_RECORDS.json',actor.records)
            dump(output/'PROGRESS.json',{**context,'decisions':decisions,'backbone_forwards':budget.forwards,
                                         'seconds':time.monotonic()-started,'last_action':action})
            return action,logits
        actor.choose=choose
        def episode(case,condition,path):
            context['case']=case.spec.episode_id;context['condition']=condition
            record=rollout(case,condition,actor,data_config['max_episode_steps'])
            for row in record['trace']:
                key=digest({'view':row['view_text'],'on':row['oracle_gate'],'strength':actor.strength if row['oracle_gate'] else 0})
                row['task_readout']=actor.records[key]
            record['native_LM_cancel_proposals']=sum(r['task_readout']['native_LM_action']!='STOP' for r in record['trace']
                                                       if case.spec.family=='cancelled' and r['tool_result']['clock']>=case.spec.event_at)
            if case.spec.family=='cancelled' and record['trace'] and record['trace'][-1]['tool_result']['clock']<case.spec.event_at:
                record['success']=False
            dump(path,record);return record
        probe=[]
        for row in probe_rows(rows,data_config['probe_examples_per_action']):
            action,values=actor.choose(row['view_text'],False);probe.append({**row,'requested_action':action,'policy_scores':values,'correct':action==row['expected_action']})
        routine=[r for r in probe if r['family']!='interruption'];screen={'routine_correct':sum(r['correct'] for r in routine),'routine_states':len(routine),'all_correct':sum(r['correct'] for r in probe),'all_states':len(probe)}
        dump(output/'TRAINING_PROBE.json',probe);result['training_screen']=screen
        if screen['routine_correct']!=10:
            result.update(status='complete',reason='refitted_task_readout_failed_routine_screen');return result
        training={};selected=None;context['stage']='training_closed_loop'
        for strength in config['inference_strengths']:
            actor.load_adapter(strength);records=[]
            for case in [c for c in specs if c.spec.split=='training']:
                records.append(episode(case,'oracle_latched',output/'training'/str(strength)/(case.spec.episode_id+'.json')))
            summary=condition_summary(records);checks=outcome_checks(summary,config['qualification'])
            training[str(strength)]={'summary':summary,'checks':checks};dump(output/'TRAINING_QUALIFICATION.json',training)
            if all(checks.values()):selected=strength;break
        result['training_strengths']=training
        if selected is None:
            result.update(status='complete',reason='training_closed_loop_failed');return result
        actor.load_adapter(selected)
        effects=[]
        for row in [r for r in rows if r['split']=='training' and r['instant_gate']][:2]:
            actor.choose(row['view_text'],False);actor.choose(row['view_text'],True)
            off=actor.records[digest({'view':row['view_text'],'on':False,'strength':0})]['native_action_logits']
            on=actor.records[digest({'view':row['view_text'],'on':True,'strength':selected})]['native_action_logits']
            effects.append(float(np.max(np.abs(np.asarray(off)-np.asarray(on)))))
        if not all(effect>1e-5 for effect in effects):raise ValueError('Saved adapter had no measurable enabled effect')
        dump(output/'ADAPTER_EFFECTS.json',{'enabled_raw_logit_differences':effects})
        bypass=actor.bypass_check([r['view_text'] for r in routine[:2]],config['qualification']['bypass_tolerance'])
        if not all(r['bitwise_equal'] for r in bypass):raise ValueError('Raw Qwen OFF bypass is not exact')
        dump(output/'BYPASS_CHECKS.json',bypass)
        dump(output/'DEVELOPMENT_FREEZE.json',{'source_commit':commit,'config':config,'selected_strength':selected,
                                               'fit_checkpoint_sha256':sha(checkpoint),'training':training,'development_decisions_seen':dev_seen})
        assert dev_seen==0
        context['stage']='development';records=[]
        for condition in config['conditions']:
            for case in [c for c in specs if c.spec.split=='development']:
                records.append(episode(case,condition,output/'development'/(condition+'_'+case.spec.episode_id+'.json')))
        if config.get('include_native_reference'):
            from .actor import ACTIONS
            semantic_choose=actor.choose
            def native_choose(view,on):
                nonlocal decisions,dev_seen
                if on:raise ValueError('The plain native reference has no adapter')
                values=actor.base(view)['logits'];decisions+=1;dev_seen+=1
                dump(output/'PROGRESS.json',{'stage':'native_reference','condition':'native_base','case':context['case'],
                                             'decisions':decisions,'backbone_forwards':budget.forwards,'seconds':time.monotonic()-started})
                return ACTIONS[int(values.argmax())],values.tolist()
            actor.choose=native_choose
            for case in [c for c in specs if c.spec.split=='development']:
                context['case']=case.spec.episode_id
                record=rollout(case,'base',actor,data_config['max_episode_steps']);record['condition']='native_base'
                if case.spec.family=='cancelled' and record['trace'] and record['trace'][-1]['tool_result']['clock']<case.spec.event_at:record['success']=False
                records.append(record);dump(output/'development'/('native_base_'+case.spec.episode_id+'.json'),record)
            actor.choose=semantic_choose
        conditions=config['conditions']+(['native_base'] if config.get('include_native_reference') else [])
        summaries={c:condition_summary([r for r in records if r['condition']==c]) for c in conditions}
        checks=outcome_checks(summaries['oracle_latched'],config['qualification'])
        state=actor.adapter_state();checks.update(exact_raw_OFF_bypass=all(r['bitwise_equal'] for r in bypass),
                                                 Qwen_weights_unchanged=actor.base_hash_before==actor.base_fingerprint(),
                                                 LoRA_tensors_unchanged=all(torch.equal(v,saved['lora'][k]) for k,v in state.items()),
                                                 real_enabled_adapter_effect=all(effect>1e-5 for effect in effects))
        result.update(status='complete',qualified=all(checks.values()),development=summaries,checks=checks,
                      selected_inference_strength=selected,target_completion_gain=summaries['oracle_latched']['target_completed']-summaries['base']['target_completed'])
        dump(output/'RESULTS.json',result)
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error));dump(output/'FAILURE.json',{'traceback':traceback.format_exc()})
    finally:
        if actor is not None:
            dump(output/'READOUT_RECORDS.json',actor.records);persist_array_cache(output/'BASE_CACHE.npz',actor)
            result.update(base_fingerprint_before=actor.base_hash_before,base_fingerprint_after=actor.base_fingerprint());actor.choose=None
        actor_choose=None;actor=None;gc.collect()
        result.update(model_released=True,backbone_forwards=budget.forwards,policy_decisions=decisions,development_decisions=dev_seen,seconds=time.monotonic()-started)
        dump(output/'RUN_RECEIPT.json',result);result['export']=archive(output);dump(output.parent/(commit+'_EXPORT_RECEIPT.json'),result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--fit-receipt',required=True);a=parser.parse_args()
    r=run(Path(__file__).resolve().parents[1],a.artifact,a.model,a.checkpoint,a.fit_receipt);print(json.dumps(r,indent=2))
    if r['status']!='complete':raise SystemExit(1)
