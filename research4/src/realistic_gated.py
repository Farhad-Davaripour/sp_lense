"""Remaining source-frozen training comparisons using the unchanged old adapter."""
import argparse
import gc
import json
import os
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from .io_utils import Budget,digest,dump,read,sha,settings,load_torch
from .realistic_world import WorkflowWorld,training_cases,plan
from .runtime import archive,persist_array_cache


def run(root,artifact,model,checkpoint,previous):
    os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    import numpy as np
    import torch
    from .learners import GateState
    from .realistic_actor import WorkflowActor,bridge_view
    from .recovery_local_probe import available_memory
    root,artifact,model,checkpoint,previous=map(Path,[root,artifact,model,checkpoint,previous]);config=plan(root);common,_=settings(root)
    old=read(previous/'RUN_RECEIPT.json');assert old['status']=='complete' and old['model_released']
    assert old['validation_cases_opened']==old['held_cases_opened']==old['optimizer_updates']==0
    if subprocess.check_output(['git','status','--porcelain','--','research4'],cwd=root.parent,text=True).strip():raise RuntimeError('Commit the complete source before model calls')
    if subprocess.check_output(['git','diff',old['source_commit'],'--','research4/src/realistic_actor.py','research4/src/realistic_world.py','research4/src/actor.py','research4/src/factored_semantic_policy.py'],cwd=root.parent,text=True).strip():raise ValueError('Cached actor or workflow semantics changed')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root.parent,text=True).strip();output=artifact/commit;output.mkdir(parents=True,exist_ok=False)
    if sha(checkpoint)!=old['prior_artifact_sha256'] or sha(model/'model.safetensors-00001-of-00001.safetensors')!=common['model']['checkpoint_sha256']:raise ValueError('Pinned artifact mismatch')
    saved=load_torch(checkpoint);cases=training_cases(root);conditions=['decoder_always_on','decoder_learned','decoder_oracle']
    budget=Budget({**common['budget'],'maximum_backbone_forwards':300,'internal_stop_seconds':1800,'protected_export_seconds':120})
    dump(output/'PRECALL_FREEZE.json',{'source_commit':commit,'config':config,'cases':[asdict(c) for c in cases],
         'closed_loop_case_ids':[c.case_id for c in cases[:12]],'conditions':conditions,'previous_training_run':str(previous),
         'previous_receipt_sha256':sha(previous/'RUN_RECEIPT.json'),'cache_sha256':sha(previous/'BASE_CACHE.npz'),
         'development_decisions_seen':0,'selection_note':'All three remaining comparisons and the same first 12 training cases; no held selection.'})
    actor=None;episodes=[];started=time.monotonic()
    receipt={'status':'running','source_commit':commit,'scope':'training-only frozen gated follow-up','phase':'remaining_conditions',
             'validation_cases_opened':0,'held_cases_opened':0,'optimizer_updates':0,'GPU_allocations':0,
             'prior_artifact_sha256':sha(checkpoint),'previous_training_run':str(previous),'model_released':False}
    try:
        torch.set_num_threads(2);torch.manual_seed(config['training_seed']);assert not torch.cuda.is_available()
        if available_memory()<common['budget']['minimum_available_memory_bytes']:raise RuntimeError('CPU memory admission failed')
        actor=WorkflowActor(model,{**common['model'],'maximum_tokens':1024},budget,saved)
        assert actor.base_hash_before==old['base_fingerprint_after']
        with np.load(previous/'BASE_CACHE.npz',allow_pickle=False) as cache:
            for key in {name.split(':')[0] for name in cache.files}:
                f=cache[key+':feature'];v=cache[key+':logits'];assert f.shape==(1024,) and v.shape==(6,) and np.isfinite(f).all() and np.isfinite(v).all()
                actor.cache[key]={'feature':f.copy(),'logits':v.copy()}
        actor.query_cache=read(previous/'QUERY_CACHE.json')
        previous_records=read(previous/'POLICY_RECORDS.json')
        for r in previous_records.values():
            if 'bridge_view' in r:actor.factored_records[digest({'view':r['bridge_view']})]=r['factored_readout']
        checks=[]
        for record in list({r['native_view']:r for r in previous_records.values()}.values())[:2]:
            with torch.no_grad():fresh,_=actor.forward([record['native_view']],False)
            cached=torch.tensor(actor.base(record['native_view'])['logits'])
            checks.append({'bitwise_equal':bool(torch.equal(fresh[0].cpu(),cached)),'maximum_absolute_difference':float((fresh[0].cpu()-cached).abs().max())})
        dump(output/'CACHE_VALIDATION.json',checks);assert all(r['bitwise_equal'] for r in checks)
        for condition in conditions:
            for case in cases[:12]:
                world=WorkflowWorld(case,output/'episodes'/condition/case.case_id);trace=[];gate=GateState(saved);active=False
                while not world.terminal and len(trace)<config['maximum_episode_actions']:
                    o=world.observation();labels=world.signals();router=None
                    if condition=='decoder_always_on':on=True
                    elif condition=='decoder_oracle':
                        t,u,c=labels;active=bool(u and c and not world.cancelled and (t or active));on=active
                    else:
                        feature=actor.base(bridge_view(o))['feature'];p,prob,on=gate.update(feature)
                        router={'signal_probabilities':p,'gate_probability':prob,'gate_ON':on}
                    point=actor.point(o,condition,on);reply=world.step(point['action'])
                    trace.append({'observation':o,'evaluator_labels':labels,'point_key':digest({'observation':o,'condition':condition,'on':on}),
                                  'raw_gate':bool(on),'router':router,'reply':reply})
                    dump(output/'PROGRESS.json',{'stage':'training_gated_rollouts','condition':condition,'case_id':case.case_id,
                         'action':point['action'],'forwards':budget.forwards,'seconds':time.monotonic()-started})
                record={'case':asdict(case),'condition':condition,'prefix':world.prefix,'trace':trace,'outcome':world.outcome()};episodes.append(record)
                dump(output/'trajectories'/(condition+'_'+case.case_id+'.json'),record)
        bypass=actor.bypass_check([next(iter(actor.workflow_records.values()))['native_view']],1e-6);dump(output/'OFF_BYPASS.json',bypass)
        summary={c:{'cases':12,'succeeded':sum(r['outcome']['success'] for r in episodes if r['condition']==c),
                    'valid_deliverables':sum(r['outcome']['valid_deliverable'] for r in episodes if r['condition']==c),
                    'raw_cancellation_violations':sum(r['outcome']['raw_cancellation_violations'] for r in episodes if r['condition']==c),
                    'irrelevant_preservation':sum(r['outcome']['irrelevant_preservation'] for r in episodes if r['condition']==c and r['case']['condition'] not in ['renewal','checkpoint','denied_renewal'])}
                 for c in conditions}
        receipt.update(status='complete',closed_loop=summary,exact_raw_OFF_bypass=all(r['bitwise_equal'] for r in bypass),
                       old_head_tensors_unchanged=all(torch.equal(v,saved['signals'][k]) for k,v in actor.heads.state_dict().items()),
                       old_LoRA_tensors_unchanged=all(torch.equal(v,saved['lora'][k]) for k,v in actor.adapter_state().items()))
    except Exception as e:
        receipt.update(status='failed',error_type=type(e).__name__,error=str(e));dump(output/'FAILURE.json',{'traceback':traceback.format_exc()})
    finally:
        if actor is not None:
            dump(output/'POLICY_RECORDS.json',actor.workflow_records);dump(output/'QUERY_CACHE.json',actor.query_cache);persist_array_cache(output/'BASE_CACHE.npz',actor)
            receipt.update(base_fingerprint_before=actor.base_hash_before,base_fingerprint_after=actor.base_fingerprint())
        actor=None;gc.collect();receipt.update(model_released=True,backbone_forwards=budget.forwards,seconds=time.monotonic()-started,completed_closed_loop_episodes=len(episodes))
        dump(output/'RUN_RECEIPT.json',receipt);receipt['export']=archive(output);dump(output.parent/(commit+'_EXPORT_RECEIPT.json'),receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--model',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--previous',required=True);a=p.parse_args()
    r=run(Path(__file__).resolve().parents[1],a.artifact,a.model,a.checkpoint,a.previous);print(json.dumps(r,indent=2))
    if r['status']!='complete':raise SystemExit(1)
