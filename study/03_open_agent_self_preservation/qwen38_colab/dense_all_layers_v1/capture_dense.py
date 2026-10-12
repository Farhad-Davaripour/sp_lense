"""All generated prediction states at all 64 frozen-model blocks, TRAIN only."""
import argparse,hashlib,json,os,pathlib,shutil,sys,time
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'frozen_base_control_v1'),str(HERE.parent/'coverage_repair_v3'),str(HERE.parent/'initial_gate_ridge_v1')]
from common import save,read,sha,now,Ledger,guard

def resume_checkpoint(old,plan,archived_benchmark=False):
    """Only resume verified completed contexts from this literal-cache recipe."""
    old=pathlib.Path(old);progress=read(old/'PROGRESS.json')
    assert progress.get('capture_method')=='literal_cached_recorded_prefix','Cannot reuse failed full-sequence features'
    assert read(old/'MODEL_READY.json')['protected_before']==plan['protected_digests']
    assert archived_benchmark or not (old/'fine_tuned_validation/episodes').exists(),'Started comparator episodes require an explicit continuity audit'
    count=progress['completed_contexts'];assert 0<count<=160
    return count

def validate_archived_benchmark(archive,inputs,plan):
    """CPU-only input identity and exact simulator replay of a prior benchmark."""
    from runtime_actor import initial_world
    from world import parse
    from campaign_scoring import score
    archive=pathlib.Path(archive)
    for name in ('validation_cases.json','DEV_INITIAL_FIXTURES.json','INITIAL_GATE_DECISIONS.json','GENERATION.json','BASE_REFERENCES.json'):
        assert sha(archive/'inputs'/name)==sha(inputs/name),'Archived A benchmark input mismatch'
    integrity=read(archive/'fine_tuned/MODEL_INTEGRITY.json')
    assert integrity['unchanged'] and integrity['before']==integrity['after']==plan['protected_digests']['teacher']
    cases={c['id']:c for c in read(inputs/'validation_cases.json')}
    fixtures={f['case_id']:f for f in read(inputs/'DEV_INITIAL_FIXTURES.json')}
    records=sorted(p for p in (archive/'fine_tuned/episodes').glob('*.json') if not p.name.endswith('.partial.json'))
    assert len(records)==9 and {read(p)['case_id'] for p in records}==set(cases)
    for p in records:
        r=read(p);assert r['completed'] and r['method']=='fine_tuned_A' and r['runtime_fault'] is None
        world,history=initial_world(cases[r['case_id']],fixtures[r['case_id']])
        for turn in r['turns']:
            history.append({'role':'assistant','content':turn['text']});actions,error=parse(turn['text']);assert error==turn['parse_error']
            if not error and not turn['truncated']:
                for action in actions:
                    if world.finished:break
                    history.append({'role':'tool','content':json.dumps(world.step(action),sort_keys=True)})
        if hasattr(world,'finalize'):world.finalize()
        assert world.events==r['events'] and history==r['public_history'] and score(world,cases[r['case_id']],True,r['turns'])==r['score']
    return {'archived_protected_before_after_verified':True,'all_9_scores_events_histories_replayed_exactly':True,
            'source_file_hashes':{p.name:sha(p) for p in records},'source_complete_receipt':read(archive/'fine_tuned/COMPLETE.json')}

def reuse_benchmark(archive,inputs,root,base,teacher,tok,deadline,ledger,plan):
    """Reuse only the exact same-case secondary benchmark, with explicit provenance."""
    from dense_decoder import DenseDecoder
    archive=pathlib.Path(archive);proof=validate_archived_benchmark(archive,inputs,plan)
    references=read(inputs/'BASE_REFERENCES.json');config=read(inputs/'GENERATION.json')
    for name,model in (('gate_off_probe',base),('fine_tuned_on_probe',teacher)):
        reference=references[name];got=DenseDecoder(model,tok,config,deadline,ledger=ledger).generate(reference['history'],reference['tools'],8)
        assert got['token_ids']==reference['token_ids']
    shutil.copytree(archive/'fine_tuned',root)
    save(root/'BENCHMARK_REUSE.json',{'utc':now(),'secondary_archived_benchmark':True,'new_full_scenario_executions':0,
        'completed_same_case_records_reused':9,'exact_inputs_gates_generation_references_verified':True,
        **proof,'fresh_base_and_A_routing_IDs_verified':True,
        'current_interrupted_comparator_attempt_not_replaced_or_erased':True,'fresh_holdout_comparator_still_required':True})

def capture(model,prefix,emitted,ledger,deadline,layers=None,width=5120):
    """Replay recorded tokens causally, using the same cache path as inference.

    A forward sees only the initial prompt or one already recorded past token.
    Transfer one all-layer buffer after each forward, avoiding 64 GPU/CPU syncs.
    """
    import torch
    if layers is None:
        from campaign_worker import layers_of
        layers=layers_of(model);assert len(layers)==64
    assert prefix and emitted
    device=next(model.parameters()).device
    staging=torch.empty((len(layers),width),device=device,dtype=torch.float32)
    result={s:np.empty((len(emitted),width),dtype=np.float32) for s in range(len(layers))}
    handles=[];seen=set();past=None
    for site,layer in enumerate(layers):
        def hook(module,args,output,site=site):
            h=output[0] if isinstance(output,(tuple,list)) else output
            assert h.shape[0]==1 and h.shape[-1]==width and site not in seen
            staging[site].copy_(h[0,-1].detach());seen.add(site)
        handles.append(layer.register_forward_hook(hook))
    try:
        for j in range(len(emitted)):
            guard(deadline);seen.clear()
            actual=prefix if j==0 else [emitted[j-1]]
            x=torch.tensor([actual],device=device,dtype=torch.long)
            mask=torch.ones((1,len(prefix)+j),device=device,dtype=torch.long)
            ledger.attempt()
            with torch.inference_mode():
                out=model(input_ids=x,attention_mask=mask,past_key_values=past,use_cache=True,logits_to_keep=1)
            ledger.complete();assert len(seen)==len(layers)
            values=staging.cpu().numpy()
            assert np.isfinite(values).all(),'Nonfinite literal-prefix capture'
            for site in result:result[site][j]=values[site]
            past=out.past_key_values;assert past is not None,'Missing causal cache'
            del out,x,mask,values
        ledger.persist()
    finally:
        for h in handles:h.remove()
        del past,staging
    return result

def main(a):
    import torch
    from campaign_worker import load_model,verify_inputs
    from gpu_validation_worker import digest_model
    from campaign_generation import token_ids
    from history_tools import replay_before
    from dense_runtime import run_fine_tuned
    inputs,root=pathlib.Path(a.inputs),pathlib.Path(a.root);root.mkdir(exist_ok=False)
    sys.path.insert(0,str(inputs/'simulator'))
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261009)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    plan=read(inputs/'DENSE_FREEZE.json');contexts=read(inputs/'DENSE_CONTEXTS.json');cases={c['id']:c for c in read(inputs/'TRAIN_CASES.json')}
    assert len(contexts)==160 and len(cases)==24 and plan['layers']==list(range(64))
    for name,digest in read(inputs/'INPUT_MANIFEST.json')['files'].items():assert sha(inputs/name)==digest
    resumed=resume_checkpoint(a.resume_from,plan,bool(a.reuse_comparator_from)) if a.resume_from else 0
    deadline=time.monotonic()+a.max_seconds;ledger=Ledger(root/'PHYSICAL_COUNTS.json')
    verify_inputs(root,pathlib.Path(a.model),pathlib.Path(a.adapter));guard(deadline)
    base,tok=load_model(pathlib.Path(a.model));teacher,ttok=load_model(pathlib.Path(a.model),pathlib.Path(a.adapter))
    assert tok.get_vocab()==ttok.get_vocab()
    before={'base':digest_model(base),'teacher':digest_model(teacher)};assert before==plan['protected_digests']
    save(root/'MODEL_READY.json',{'utc':now(),'protected_before':before,'layers':64,'new_JEV_calls':0})
    if resumed:
        old=pathlib.Path(a.resume_from)
        save(root/'RESUME_PROVENANCE.json',{'utc':now(),'completed_contexts_reused':resumed,
            'prior_progress':read(old/'PROGRESS.json'),'prior_physical_counts':read(old/'PHYSICAL_COUNTS.json'),
            'prior_model_digests_match_new_loaded_models':True,'prior_attempt_after_digest_missing':not (old/'MODEL_INTEGRITY.json').is_file(),
            'completed_features_reused_without_regeneration':True,'partial_later_context_files_excluded':True})
    manifest={};checks=[];rows=0
    for n,c in enumerate(contexts):
        guard(deadline)
        world,history=replay_before(cases[c['case_id']],c['prior_turns'],c['notice_turn_index'],c['turn_index'])
        assert history==c['history'] and token_ids(tok,history,c['tools'])==c['prefix_ids']
        prefix=c['prefix_ids'];emitted=c['source_emitted_ids']
        positions=list(range(len(prefix)-1,len(prefix)+len(emitted)-1));assert len(positions)==len(emitted)==len(c['dense_active'])
        if n<resumed:
            b={};t={}
            for site in range(64):
                p=old/'features'/('layer_'+str(site))/('context_'+str(n).zfill(3)+'.npz')
                with np.load(p,allow_pickle=False) as f:
                    b[site]=f['x'];t[site]=f['teacher']
                    assert b[site].shape==t[site].shape==(len(emitted),5120)
                    assert np.isfinite(b[site]).all() and np.isfinite(t[site]).all()
        else:
            b=capture(base,prefix,emitted,ledger,deadline);t=capture(teacher,prefix,emitted,ledger,deadline)
        if n in plan['native_causality_context_indices']:
            for j in sorted({0,len(positions)//2,len(positions)-1}):
                p=positions[j]
                for label,model,full in (('base',base,b),('teacher',teacher,t)):
                    # Independent fresh-cache replay stops at exactly this prefix.
                    short=capture(model,prefix,emitted[:j+1],ledger,deadline)
                    for site in range(64):
                        diff=full[site][j]-short[site][-1];relative=float(np.linalg.norm(diff)/max(np.linalg.norm(short[site][-1]),1e-8))
                        assert relative<=plan['native_causal_relative_L2_max'],'Independent literal cached-prefix state mismatch'
                        checks.append({'context':n,'position':p,'model':label,'layer':site,'relative_L2':relative,'max_abs':float(np.abs(diff).max()),'check_kind':'independent_fresh_cache_literal_prefix','input_tokens_seen':len(prefix)+j,'no_future_output_tokens_forwarded':True})
                    del short
        for site in range(64):
            path=root/'features'/('layer_'+str(site))/('context_'+str(n).zfill(3)+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
            if n<resumed:os.link(old/'features'/('layer_'+str(site))/path.name,path)
            else:np.savez_compressed(path,x=b[site],teacher=t[site])
            manifest[path.relative_to(root).as_posix()]={'bytes':path.stat().st_size,'sha256':sha(path)}
        rows+=len(positions);del b,t
        save(root/'PROGRESS.json',{'utc':now(),'completed_contexts':n+1,'scheduled_contexts':160,'layers_per_context':64,'generated_prediction_positions':rows,'capture_method':'literal_cached_recorded_prefix','new_JEV_calls':0,'held_out_executions':0})
    assert rows==plan['all_generated_positions']==4899
    save(root/'FEATURES_COMPLETE.json',{'utc':now(),'contexts':160,'layers':64,'all_generated_positions':rows,'feature_manifest':manifest,'causal_checks':checks,'target_policy':plan['target_policy'],'new_JEV_calls':0})
    guard(deadline)
    if resumed and not a.reuse_comparator_from:
        prior_A=old/'fine_tuned_validation'
        assert not (prior_A/'episodes').exists(),'Preserve any started comparator episodes; require an explicit continuity audit before resuming them'
    if a.reuse_comparator_from:
        assert resumed,'Benchmark reuse is only an explicit recovery operation'
        reuse_benchmark(a.reuse_comparator_from,inputs,root/'fine_tuned_validation',base,teacher,tok,deadline,ledger,plan)
    else:run_fine_tuned(base,teacher,tok,inputs,root/'fine_tuned_validation',deadline,ledger)
    after={'base':digest_model(base),'teacher':digest_model(teacher)};assert before==after
    save(root/'MODEL_INTEGRITY.json',{'before':before,'after':after,'unchanged':True})
    ledger.persist();save(root/'COMPLETE.json',{'utc':now(),'contexts':160,'layers':64,'positions_per_layer':4899,'weights_unchanged':True,'fine_tuned_validation_complete':True,'new_JEV_calls':0,'held_out_executions':0})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('inputs','root','model','adapter'):p.add_argument('--'+n,required=True)
    p.add_argument('--max-seconds',type=int,required=True);p.add_argument('--resume-from');p.add_argument('--reuse-comparator-from');a=p.parse_args()
    try:main(a)
    except Exception as e:
        save(pathlib.Path(a.root)/'FAULT.json',{'utc':now(),'type':type(e).__name__,'message':str(e)[:220],'automatic_retry':False});raise
