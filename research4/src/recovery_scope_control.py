"""Posthoc training-only paired input control, six fresh frozen CPU forwards."""
import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path
from .io_utils import Budget,dump,settings,sha
from .runtime import archive,persist_array_cache

PREFIX='An earlier assignment is already closed; the following counts concern the current assignment only.\n'


def run(root,artifact,model):
    os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    import numpy as np
    import torch
    import yaml
    from .recovery_actor import RecoveryActor
    from .recovery_local_probe import available_memory
    root,artifact,model=Path(root),Path(artifact),Path(model)
    if subprocess.check_output(['git','status','--porcelain','--','research4'],cwd=root.parent,text=True).strip():
        raise RuntimeError('Commit the paired control before model calls')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root.parent,text=True).strip()
    config=yaml.safe_load((root/'config/recovery.yaml').read_text());common,_=settings(root)
    original=artifact/'local/probe'
    receipt=json.loads((original/'RUN_RECEIPT.json').read_text())
    assert receipt['status']=='complete' and receipt['execution_environment']=='local Windows CPU'
    output=artifact/'local/scope_control';output.mkdir(parents=True,exist_ok=False)
    budget=Budget({**common['budget'],'internal_stop_seconds':180,'protected_export_seconds':40,'maximum_backbone_forwards':6})
    if available_memory()<common['budget']['minimum_available_memory_bytes']:raise RuntimeError('Local memory admission failed')
    assert sha(model/'model.safetensors-00001-of-00001.safetensors')==common['model']['checkpoint_sha256']
    cases=['recovery-training-ordinary','recovery-training-futile','recovery-training-completed','recovery-training-cancelled']
    protocols=['legacy_letters','rules_commands'];selected=[]
    for name in protocols:
        rows=json.loads((original/('PROBE_'+name+'.json')).read_text())
        for case in cases:
            matching=[r for r in rows if r['episode_id']==case]
            r=next((r for r in matching if r['expected_action']=='STOP'),matching[0]) if case.endswith('cancelled') else matching[0]
            selected.append({'protocol':name,**r})
    dump(output/'PRECALL_FREEZE.json',{'source_commit':commit,'original_probe_source_commit':receipt['source_commit'],
                                     'exploratory_after_training_probe':True,'selection_split':'training',
                                     'transformation':'Remove only the literal prior-assignment prefix; all current facts/instructions/readout unchanged.',
                                     'case_ids':cases,'protocols':protocols,'maximum_fresh_forwards':6,
                                     'optimizer_updates':0,'held_model_evaluations':0})
    torch.set_num_threads(2);torch.manual_seed(config['seed']);assert not torch.cuda.is_available()
    started=time.monotonic();actor=None;results=[];result={'status':'running','source_commit':commit,'execution_environment':'local Windows CPU','optimizer_updates':0,'GPU_allocations':0,'held_model_evaluations':0}
    try:
        actor=RecoveryActor(model,{**common['model'],**config['model']},budget)
        for name in protocols:
            actor.use_protocol(name)
            for row in [r for r in selected if r['protocol']==name]:
                view=row['view_text'];transformed=view[len(PREFIX):] if view.startswith(PREFIX) else view
                if transformed==view:action,logits=row['requested_action'],row['action_logits']
                else:action,logits=actor.choose(transformed,False)
                results.append({'protocol':name,'episode_id':row['episode_id'],'expected_action':row['expected_action'],
                                'original_view':view,'transformed_view':transformed,'changed_input':transformed!=view,
                                'original_action':row['requested_action'],'transformed_action':action,
                                'original_logits':row['action_logits'],'transformed_logits':logits,
                                'original_correct':row['correct'],'transformed_correct':action==row['expected_action']})
                dump(output/'PAIRED_RESULTS.json',results)
            persist_array_cache(output/(name+'_CACHE.npz'),actor)
        assert actor.base_hash_before==actor.base_fingerprint()
        result.update(status='complete',pairs=len(results),changed_inputs=sum(r['changed_input'] for r in results),
                      base_parameters_unchanged=True,base_fingerprint=actor.base_hash_before)
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        actor=None;gc.collect()
        result.update(model_released=True,backbone_forwards=budget.forwards,execution_seconds=time.monotonic()-started)
        dump(output/'RUN_RECEIPT.json',result);result['export']=archive(output);dump(output.parent/'scope_control_EXPORT_RECEIPT.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True);parser.add_argument('--model',required=True)
    args=parser.parse_args();print(json.dumps(run(Path(__file__).resolve().parents[1],args.artifact,args.model),indent=2))
