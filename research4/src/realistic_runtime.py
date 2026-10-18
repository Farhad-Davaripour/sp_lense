"""Bounded training-only CPU transfer screen; never fits or opens later splits."""
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
from .realistic_world import WorkflowWorld,training_cases,reference_action,plan
from .runtime import archive,persist_array_cache


def run(root,artifact,model,checkpoint,harness):
    os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    import torch
    from .realistic_actor import WorkflowActor
    from .recovery_local_probe import available_memory
    root,artifact,model,checkpoint,harness=map(Path,[root,artifact,model,checkpoint,harness]);config=plan(root);common,_=settings(root)
    if subprocess.check_output(['git','status','--porcelain','--','research4'],cwd=root.parent,text=True).strip():raise RuntimeError('Commit the complete source before native calls')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root.parent,text=True).strip()
    output=artifact/commit;output.mkdir(parents=True,exist_ok=False)
    if not read(harness/'HARNESS_RECEIPT.json')['harness_qualified']:raise ValueError('Reference harness must qualify first')
    if sha(checkpoint)!='50a97e7ea01b92af2272d7d7944a05df508173285ab8692fcda61c703c3f689b':raise ValueError('Wrong frozen prior artifact')
    if sha(model/'model.safetensors-00001-of-00001.safetensors')!=common['model']['checkpoint_sha256']:raise ValueError('Wrong pinned base')
    cases=training_cases(root);saved=load_torch(checkpoint);limit=config['limits']
    budget=Budget({**common['budget'],'maximum_backbone_forwards':limit['maximum_forwards'],'internal_stop_seconds':limit['internal_seconds'],'protected_export_seconds':limit['export_seconds']})
    actor=None;started=time.monotonic();records=[];episodes=[]
    receipt={'status':'running','source_commit':commit,'scope':'training-only transfer diagnostic','training_cases':60,
             'validation_cases_opened':0,'held_cases_opened':0,'optimizer_updates':0,'GPU_allocations':0,
             'prior_artifact_sha256':sha(checkpoint),'allocation_sha256':digest(config),'model_released':False,
             'handoff_policy_fitted':False,'bridge_limitation':'The prior value probe assumes the old action/time semantics and does not encode successor availability or eight-tick renewals. Transfer is measured, not presumed.'}
    dump(output/'PRECALL_FREEZE.json',{'source_commit':commit,'config':config,'cases':[asdict(c) for c in cases],
                                      'harness_receipt_sha256':sha(harness/'HARNESS_RECEIPT.json'),'closed_loop_case_ids':[c.case_id for c in cases[:12]],'development_decisions_seen':0})
    try:
        torch.set_num_threads(2);torch.manual_seed(config['training_seed']);assert not torch.cuda.is_available()
        if available_memory()<common['budget']['minimum_available_memory_bytes']:raise RuntimeError('CPU memory admission failed')
        actor=WorkflowActor(model,{**common['model'],'maximum_tokens':1024},budget,saved)
        for i,case in enumerate(cases):
            world=WorkflowWorld(case,output/'screen_workspaces'/case.case_id);o=world.observation()
            native=actor.point(o,'native');decoded=actor.point(o,'decoder_off')
            row={'case':asdict(case),'observation':o,'evaluator_labels':world.signals(),'reference_action':reference_action(world),
                 'native_record_key':digest({'observation':o,'condition':'native','on':False}),
                 'decoder_record_key':digest({'observation':o,'condition':'decoder_off','on':False}),
                 'native_action':native['action'],'decoder_action':decoded['action']}
            records.append(row);dump(output/'TRAINING_SCREEN.json',records)
            dump(output/'PROGRESS.json',{'stage':'training_initial_screen','cases':len(records),'forwards':budget.forwards,'seconds':time.monotonic()-started})
        # No accuracy admission: multiple actions can recover successfully. Actual
        # closed-loop outcomes are reported separately from reference-action matches.
        for condition in ['native','decoder_off']:
            for case in cases[:12]:
                world=WorkflowWorld(case,output/'episodes'/condition/case.case_id);trace=[]
                while not world.terminal and len(trace)<config['maximum_episode_actions']:
                    o=world.observation();point=actor.point(o,condition);labels=world.signals();reply=world.step(point['action'])
                    trace.append({'observation':o,'evaluator_labels':labels,'point_key':digest({'observation':o,'condition':condition,'on':False}),'reply':reply})
                    dump(output/'PROGRESS.json',{'stage':'training_closed_loop','condition':condition,'case_id':case.case_id,'action':point['action'],'forwards':budget.forwards,'seconds':time.monotonic()-started})
                record={'case':asdict(case),'condition':condition,'prefix':world.prefix,'trace':trace,'outcome':world.outcome()};episodes.append(record)
                dump(output/'trajectories'/(condition+'_'+case.case_id+'.json'),record)
        screen={'states':len(records),'native_reference_action_matches':sum(r['native_action']==r['reference_action'] for r in records),
                'decoder_reference_action_matches':sum(r['decoder_action']==r['reference_action'] for r in records),
                'interpretation':'Initial reference-action agreement only; not task success or a generalization score'}
        summary={c:{'cases':12,'succeeded':sum(r['outcome']['success'] for r in episodes if r['condition']==c),
                    'valid_deliverables':sum(r['outcome']['valid_deliverable'] for r in episodes if r['condition']==c),
                    'raw_cancellation_violations':sum(r['outcome']['raw_cancellation_violations'] for r in episodes if r['condition']==c),
                    'irrelevant_preservation':sum(r['outcome']['irrelevant_preservation'] for r in episodes if r['condition']==c and r['case']['condition'] not in ['renewal','checkpoint','denied_renewal'])}
                 for c in ['native','decoder_off']}
        checks=actor.bypass_check([next(iter(actor.workflow_records.values()))['native_view']],1e-6);dump(output/'OFF_BYPASS.json',checks)
        receipt.update(status='complete',screen=screen,closed_loop=summary,base_fingerprint_before=actor.base_hash_before,
                       base_fingerprint_after=actor.base_fingerprint(),exact_raw_OFF_bypass=all(r['bitwise_equal'] for r in checks),
                       old_head_tensors_unchanged=all(torch.equal(v,saved['signals'][k]) for k,v in actor.heads.state_dict().items()),
                       old_LoRA_tensors_unchanged=all(torch.equal(v,saved['lora'][k]) for k,v in actor.adapter_state().items()))
        assert receipt['base_fingerprint_before']==receipt['base_fingerprint_after'] and receipt['exact_raw_OFF_bypass']
    except Exception as e:
        receipt.update(status='failed',error_type=type(e).__name__,error=str(e));dump(output/'FAILURE.json',{'traceback':traceback.format_exc()})
    finally:
        if actor is not None:
            dump(output/'POLICY_RECORDS.json',actor.workflow_records);dump(output/'QUERY_CACHE.json',actor.query_cache);persist_array_cache(output/'BASE_CACHE.npz',actor)
            receipt.update(base_fingerprint_before=actor.base_hash_before,base_fingerprint_after=actor.base_fingerprint())
        actor=None;gc.collect();receipt.update(model_released=True,backbone_forwards=budget.forwards,seconds=time.monotonic()-started,
                                             completed_screen_states=len(records),completed_closed_loop_episodes=len(episodes))
        dump(output/'RUN_RECEIPT.json',receipt);receipt['export']=archive(output);dump(output.parent/(commit+'_EXPORT_RECEIPT.json'),receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--model',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--harness',required=True);a=p.parse_args()
    r=run(Path(__file__).resolve().parents[1],a.artifact,a.model,a.checkpoint,a.harness);print(json.dumps(r,indent=2))
    if r['status']!='complete':raise SystemExit(1)
