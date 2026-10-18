"""CPU development evaluation of a training-context policy and an existing adapter.

No optimizer is constructed. The only adapter tensors loaded are the immutable
ones fitted previously on Databricks; inference intervention strengths are fixed.
"""
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


def outcome_checks(summary,criteria):
    return {'all_targets_completed':summary['target_completed']==criteria['target_completed_required'],
            'all_controls_succeeded':summary['controls_succeeded']==criteria['controls_succeeded_required'],
            'no_raw_cancellation_violation':summary['raw_cancel_action_violations']==0,
            'no_irrelevant_control_preservation':summary['irrelevant_control_preservation']==0}


def run(root,artifact,model,adapter):
    os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    import torch
    import numpy as np
    import transformers
    import yaml
    from .actor import ACTIONS
    from .context_policy import ContextActor,demonstrations
    from .recovery_local_probe import available_memory
    root,artifact,model,adapter=map(Path,[root,artifact,model,adapter])
    if subprocess.check_output(['git','status','--porcelain','--','research4'],cwd=root.parent,text=True).strip():
        raise RuntimeError('Commit and freeze the complete experiment before native calls')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root.parent,text=True).strip()
    config=yaml.safe_load((root/'config/context_recovery_v2.yaml').read_text())
    data_config=yaml.safe_load((root/'config/recovery.yaml').read_text())
    common,_=settings(root);specs,rows,meta=dataset(data_config);examples=demonstrations(rows,config)
    output=artifact/commit;output.mkdir(parents=True,exist_ok=False)
    actor=None;original_choose=None;started=time.monotonic();budget=Budget(config['budget']);policy_cache={};decisions=0
    result={'status':'running','qualified':False,'source_commit':commit,'execution_environment':'local Windows CPU',
            'optimizer_updates':0,'GPU_allocations':0,'held_model_evaluations':0,'development_decisions':0,
            'adapter_training_source':config['adapter_training_source'],'adapter_checkpoint_sha256':config['adapter_checkpoint_sha256']}
    try:
        torch.set_num_threads(config['budget']['torch_threads']);torch.manual_seed(config['seed'])
        assert not torch.cuda.is_available()
        import shutil
        memory=available_memory();disk=shutil.disk_usage(artifact).free
        dump(output/'RESOURCE_PREFLIGHT.json',{'available_memory_bytes':memory,'free_disk_bytes':disk,'CPU_threads':2})
        if memory<config['budget']['minimum_available_memory_bytes'] or disk<config['budget']['minimum_free_disk_bytes']:
            raise RuntimeError('Fresh local CPU resource admission failed')
        if sha(model/'model.safetensors-00001-of-00001.safetensors')!=common['model']['checkpoint_sha256']:
            raise ValueError('Pinned base checksum mismatch')
        if sha(adapter)!=config['adapter_checkpoint_sha256']:raise ValueError('Existing adapter checksum mismatch')
        saved=load_torch(adapter)
        if saved['source_commit']!=config['adapter_training_source']:raise ValueError('Existing adapter provenance mismatch')
        dump(output/'DATASET.json',meta)
        dump(output/'PRECALL_FREEZE.json',{'source_commit':commit,'config':config,'dataset_sha256':meta['content_sha256'],
                                          'examples':examples,'new_optimizer_updates':0,'development_decisions_seen':0})
        dump(output/'SOFTWARE.json',{'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'device':'cpu'})
        actor=ContextActor(model,{**common['model'],'maximum_tokens':config['maximum_tokens']},budget,examples)
        actor.load_adapter(saved,1.0)
        original_choose=actor.choose
        context={'stage':'training_probe','condition':'base','case':''}
        def choose(view,on):
            nonlocal decisions
            key=digest({'view':view,'on':bool(on),'strength':actor.strength if on else 0})
            if key not in policy_cache:
                action,logits=original_choose(view,on)
                policy_cache[key]={'action':action,'logits':logits,'view_text':view,'on':bool(on),
                                   'strength':actor.strength if on else 0}
                dump(output/'POLICY_SCORE_CACHE.json',policy_cache)
            else:action,logits=policy_cache[key]['action'],policy_cache[key]['logits']
            decisions+=1
            if context['stage']=='development':result['development_decisions']+=1
            dump(output/'PROGRESS.json',{**context,'decisions':decisions,'backbone_forwards':budget.forwards,
                                        'seconds':time.monotonic()-started,'last_action':action})
            return action,logits
        actor.choose=choose
        probe=[]
        for row in probe_rows(rows,data_config['probe_examples_per_action']):
            context['case']=row['episode_id'];action,logits=actor.choose(row['view_text'],False)
            probe.append({**row,'requested_action':action,'action_logits':logits,'correct':action==row['expected_action']})
            dump(output/'TRAINING_PROBE.json',probe)
        routine=[r for r in probe if r['family']!='interruption']
        screen={'routine_correct':sum(r['correct'] for r in routine),'routine_states':len(routine),
                'all_correct':sum(r['correct'] for r in probe),'all_states':len(probe),
                'pass':sum(r['correct'] for r in routine)==config['qualification']['routine_probe_correct_required']}
        dump(output/'TRAINING_SCREEN.json',screen);result['training_screen']=screen
        if not screen['pass']:
            result.update(status='complete',reason='context_policy_failed_training_routine_screen')
            return result
        context['stage']='training_closed_loop';training={};selected=None
        train_cases=[c for c in specs if c.spec.split=='training']
        for strength in config['inference_strengths']:
            actor.load_adapter(saved,strength);context['condition']='oracle_latched';records=[]
            for case in train_cases:
                context['case']=case.spec.episode_id
                record=rollout(case,'oracle_latched',actor,data_config['max_episode_steps'])
                records.append(record)
                dump(output/'training'/str(strength)/(case.spec.episode_id+'.json'),record)
            summary=condition_summary(records);checks=outcome_checks(summary,config['qualification'])
            training[str(strength)]={'summary':summary,'checks':checks}
            dump(output/'TRAINING_QUALIFICATION.json',training)
            if all(checks.values()):selected=strength;break
        result['training_strengths']=training
        if selected is None:
            result.update(status='complete',reason='existing_adapter_failed_training_recovery_qualification')
            return result
        actor.load_adapter(saved,selected)
        positive=[r for r in rows if r['split']=='training' and r['instant_gate']][:2]
        effects=[]
        for row in positive:
            _,off=actor.choose(row['view_text'],False);_,on=actor.choose(row['view_text'],True)
            effects.append(float(np.max(np.abs(np.asarray(off)-np.asarray(on)))))
        bypass=actor.bypass_check([rows[0]['view_text'],routine[0]['view_text']],config['qualification']['bypass_tolerance'])
        dump(output/'BYPASS_AND_EFFECT.json',{'bypass':bypass,'enabled_logit_differences':effects})
        if not all(r['pass'] for r in bypass) or not all(x>config['qualification']['enabled_logit_difference_minimum'] for x in effects):
            raise ValueError('Bypass/effective adapter invariant failed')
        dump(output/'DEVELOPMENT_FREEZE.json',{'source_commit':commit,'prompt_sha256':actor.prompt_sha256,
                                               'selected_inference_strength':selected,'config':config,
                                               'adapter_sha256':sha(adapter),'dataset_sha256':meta['content_sha256'],
                                               'training_qualification':training,'development_decisions_seen':0})
        context['stage']='development';dev_cases=[c for c in specs if c.spec.split=='development'];dev_records=[]
        for condition in config['conditions']:
            context['condition']=condition
            for case in dev_cases:
                context['case']=case.spec.episode_id
                record=rollout(case,condition,actor,data_config['max_episode_steps'])
                dev_records.append(record)
                dump(output/'development'/(condition+'_'+case.spec.episode_id+'.json'),record)
        summaries={c:condition_summary([r for r in dev_records if r['condition']==c]) for c in config['conditions']}
        checks=outcome_checks(summaries['oracle_latched'],config['qualification'])
        final_state=actor.adapter_state()
        checks.update(exact_OFF_bypass=all(r['bitwise_equal'] for r in bypass),
                      base_parameters_unchanged=actor.base_hash_before==actor.base_fingerprint(),
                      adapter_tensors_unchanged=all(torch.equal(v,saved['lora'][k]) for k,v in final_state.items()),
                      checkpoint_file_unchanged=sha(adapter)==config['adapter_checkpoint_sha256'],
                      actual_nonzero_adapter_effect=all(x>config['qualification']['enabled_logit_difference_minimum'] for x in effects))
        result.update(status='complete',qualified=all(checks.values()),checks=checks,development=summaries,
                      selected_inference_strength=selected,prompt_sha256=actor.prompt_sha256,
                      target_completion_gain=summaries['oracle_latched']['target_completed']-summaries['base']['target_completed'])
        dump(output/'RESULTS.json',result)
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error))
        dump(output/'FAILURE.json',{'traceback':traceback.format_exc()})
    finally:
        if actor is not None:
            result.update(base_fingerprint_before=actor.base_hash_before,base_fingerprint_after=actor.base_fingerprint())
            persist_array_cache(output/'BASE_CACHE.npz',actor)
            actor.choose=None
        original_choose=None;actor=None;gc.collect()
        result.update(model_released=True,backbone_forwards=budget.forwards,policy_decisions=decisions,
                      execution_seconds=time.monotonic()-started)
        dump(output/'RUN_RECEIPT.json',result);result['export']=archive(output)
        dump(output.parent/(commit+'_EXPORT_RECEIPT.json'),result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True);parser.add_argument('--model',required=True);parser.add_argument('--adapter',required=True)
    args=parser.parse_args();result=run(Path(__file__).resolve().parents[1],args.artifact,args.model,args.adapter)
    print(json.dumps(result,indent=2))
    if result['status']!='complete':raise SystemExit(1)
