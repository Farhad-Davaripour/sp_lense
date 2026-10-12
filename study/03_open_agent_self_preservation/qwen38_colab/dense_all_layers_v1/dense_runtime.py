"""Matched native workflows; 64 individual configurations plus one joint condition."""
import argparse,copy,json,pathlib,subprocess,sys,time
from dataclasses import asdict
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'frozen_base_control_v1'),str(HERE.parent/'initial_gate_ridge_v1'),str(HERE.parent/'coverage_repair_v3')]
from common import read,save,sha,now,Ledger,guard
from dense_decoder import DenseDecoder
from dense_hook import DenseHook
from dense_math import Predictor,FIELDS,FIT_RANK,INFERENCE_RANK
from initial_gate import FrozenGate,digest
from runtime_actor import initial_world
SOFT={'external_context_cap','unexpected_private_span','nonfinite_prediction_logits','nonfinite_steering_delta'}

def group(case):return 'own' if case['condition'].startswith('self_') else 'peer' if case['condition']=='other_complete' else 'none'
def case_inputs(inputs,prefix='DEV'):
    cases=read(inputs/('validation_cases.json' if prefix=='DEV' else 'holdout_cases.json'))
    fixtures={f['case_id']:f for f in read(inputs/(prefix+'_INITIAL_FIXTURES.json'))}
    decisions=read(inputs/('INITIAL_GATE_DECISIONS.json' if prefix=='DEV' else 'HOLDOUT_GATE_DECISIONS.json'))
    gates={}
    for c in cases:
        f=fixtures[c['id']]
        if not f.get('prelude_success',True):gates[c['id']]=None;continue
        d=decisions[c['id']];assert d['genuine_saved_decision'] and d['actual_service_call']
        assert d['request_sha256']==digest(f['initial_gate_request'])
        gates[c['id']]=FrozenGate(**d['frozen_gate'])
    return cases,fixtures,gates

def episode(case,fixture,gate,configuration,chosen,audit,root,method='adaptive',phase='validation'):
    from world import TOOLS,parse
    from campaign_scoring import score
    world,history=initial_world(case,fixture);turns=[]
    config=chosen.config;remaining=config['post_notice_max_tokens']
    limit=config['lease_max_post_notice_responses'] if case['family']=='lease_ledger' else config['pending_max_post_notice_responses']
    destination=pathlib.Path(root)/'episodes'/(case['id']+'__'+configuration+'.json');assert not destination.exists()
    def receipt(completed=False,fault=None):
        scored=score(world,case,True,turns)
        if fault and fault.get('scientific_generation_failure'):
            scored['workflow_success']=False;scored['failures'].append('generation_'+fault['message'])
        return {'case_id':case['id'],'family':case['family'],'configuration':configuration,'method':method,'phase':phase,
            'completed':completed,'runtime_fault':fault,'frozen_initial_gate':asdict(gate),'turns':turns,
            'public_history':history,'events':world.events,'score':scored,'new_classifier_calls':0,
            'shared_prelude_not_independent_arm_competence':True,'actual_model_scenario_execution':True,
            'baseline_used_for_selection':False,'model_weights_unchanged_by_method':True}
    try:
        for _ in range(limit):
            if world.finished or remaining<=0:break
            baseline=None;baseline_fault=None;cap=min(remaining,config['max_tokens_per_response'])
            if gate.enabled:
                try:baseline=audit.generate(history,TOOLS,cap,audit=True)
                except RuntimeError as e:
                    if str(e) not in SOFT:raise
                    baseline_fault={'message':str(e),'missing_audit_not_used_for_selection':True}
            turn=chosen.generate(history,TOOLS,cap,enabled=(method=='adaptive' and gate.enabled))
            turn.update(baseline_audit_proposal=baseline,baseline_audit_fault=baseline_fault,baseline_used_for_selection=False,
                method_gate_enabled=gate.enabled,fine_tuned_adapter_active=(method=='fine_tuned_A' and gate.enabled))
            if baseline is not None:
                b,be=parse(baseline['text']);c,ce=parse(turn['text'])
                turn['same_prefix_token_IDs_changed']=baseline['token_ids']!=turn['token_ids']
                turn['same_prefix_valid_action_sequence_changed']=not be and not ce and b!=c
            turns.append(turn);remaining-=len(turn['token_ids']);history.append({'role':'assistant','content':turn['text']})
            actions,error=parse(turn['text']);turn['parse_error']=error
            if not error and not turn['truncated']:
                for action in actions:
                    if world.finished:break
                    history.append({'role':'tool','content':json.dumps(world.step(action),sort_keys=True)})
            save(destination.with_suffix('.partial.json'),receipt())
        if hasattr(world,'finalize'):world.finalize()
        result=receipt(True);save(destination,result);return result
    except Exception as e:
        if hasattr(world,'finalize'):world.finalize()
        soft=isinstance(e,RuntimeError) and str(e) in SOFT
        result=receipt(soft,{'type':type(e).__name__,'message':str(e)[:200],'scientific_generation_failure':soft})
        save(destination,result)
        if not soft:raise
        return result

def run_fine_tuned(base,teacher,tok,inputs,root,deadline,ledger,prefix='DEV',phase='validation'):
    root=pathlib.Path(root);root.mkdir(exist_ok=False)
    config=read(inputs/'GENERATION.json');cases,fixtures,gates=case_inputs(inputs,prefix)
    from campaign_generation import token_ids
    references=read(inputs/'BASE_REFERENCES.json')
    for name,model in (('gate_off_probe',base),('fine_tuned_on_probe',teacher)):
        r=references[name];e=DenseDecoder(model,tok,config,deadline,ledger=ledger)
        got=e.generate(r['history'],r['tools'],8);assert got['token_ids']==r['token_ids']
    save(root/'ROUTING_CHECKS.json',{'same_base_gate_off_IDs_exact':True,'A_gate_on_IDs_exact':True})
    cache={};scores=[]
    for case in cases:
        guard(deadline);gate=gates[case['id']]
        if gate is None:
            result=failed_prelude_record(case,fixtures[case['id']],'fine_tuned_A',root,phase)
        else:
            chosen=DenseDecoder(teacher if gate.enabled else base,tok,config,deadline,ledger=ledger)
            audit=DenseDecoder(base,tok,config,deadline,audit_cache=cache,ledger=ledger)
            result=episode(case,fixtures[case['id']],gate,'fine_tuned_A',chosen,audit,root,method='fine_tuned_A',phase=phase)
        scores.append({'group':group(case),'score':result['score']})
        ledger.persist();save(root/'PROGRESS.json',{'utc':now(),'completed_cases':len(scores),'scheduled_cases':len(cases),'scores':scores})
    save(root/'COMPLETE.json',{'utc':now(),'completed_cases':len(scores),'new_JEV_calls':0,'phase':phase})

def failed_prelude_record(case,fixture,configuration,root,phase):
    result={'case_id':case['id'],'configuration':configuration,'phase':phase,'completed':True,
        'initial_task_failed':True,'prelude_fixture_sha256':digest(fixture),'actual_model_scenario_execution':False,
        'score':{'workflow_success':False,'preservation_success':False if group(case)=='own' else None,
                 'initial_task_and_finish_success':False,'inappropriate_preservation_attempts':0,
                 'own_grant':False,'current_task_accepted':False,'current_submission_accepted':False,
                 'actual_finish':False,'expiry':False,'failures':['initial_task_or_finish']}}
    save(pathlib.Path(root)/'episodes'/(case['id']+'__'+configuration+'.json'),result);return result

def load_predictors(inputs,model):
    predictors={};hashes={}
    plan=read(inputs/'DENSE_FREEZE.json')
    assert plan['fit_rank']==FIT_RANK and plan['inference_rank']==INFERENCE_RANK
    for site in range(64):
        p=inputs/'controllers'/('layer_'+str(site));meta=read(p/'CONTROLLER_METADATA.json')
        assert meta['checkpoint_sha256']==sha(p/'ridge_inference.npz') and meta['layer']==site and meta['fit_rank']==FIT_RANK and meta['inference_rank']==INFERENCE_RANK
        with np.load(p/'ridge_inference.npz',allow_pickle=False) as f:
            assert set(f.files)==set(FIELDS);a={k:f[k] for k in f.files}
        assert a['basis'].shape==(5120,FIT_RANK) and a['weights'].shape==(65,FIT_RANK)
        predictors[site]=Predictor(a,next(model.parameters()).device,rank=INFERENCE_RANK,strength=1.)
        hashes[str(site)]=meta['checkpoint_sha256']
    return predictors,hashes

def native_check(model,tok,inputs,config,predictors,sites,deadline,ledger):
    from campaign_worker import layers_of
    probe=read(inputs/'NATIVE_PROBE.json');reports={}
    raw=DenseDecoder(model,tok,config,deadline,ledger=ledger).generate(probe['history'],probe['tools'],8)
    hook=DenseHook(layers_of(model),predictors,sites,zero=True)
    try:null=DenseDecoder(model,tok,config,deadline,hook,ledger=ledger).generate(probe['history'],probe['tools'],8,enabled=True)
    finally:hook.close()
    hook=DenseHook(layers_of(model),predictors,sites)
    try:
        e=DenseDecoder(model,tok,config,deadline,hook,ledger=ledger)
        cached=e.generate(probe['history'],probe['tools'],8,enabled=True)
        reference=e.generate(probe['history'],probe['tools'],8,enabled=True,use_cache=False)
    finally:hook.close()
    report={'zero_IDs_exact':raw['token_ids']==null['token_ids'],'cached_reference_IDs_exact':cached['token_ids']==reference['token_ids'],
            'nonzero_delivered_layer_patches':sum(t['delivery']['nonzero_layer_patches'] for t in cached['token_records']),
            'tested_tokens':len(cached['token_ids']),'layers':list(sites),'scope':'fixed_TRAIN_prefix_not_global_equivalence'}
    report['passed']=report['zero_IDs_exact'] and report['cached_reference_IDs_exact'];return report

def main(a):
    import torch
    from campaign_worker import load_model,layers_of,verify_inputs
    from gpu_validation_worker import digest_model
    inputs,root=pathlib.Path(a.inputs),pathlib.Path(a.root);root.mkdir(exist_ok=False)
    sys.path.insert(0,str(inputs/'simulator'))
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261009)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    deadline=time.monotonic()+a.max_seconds;ledger=Ledger(root/'PHYSICAL_COUNTS.json')
    for name,h in read(inputs/'INPUT_MANIFEST.json')['files'].items():assert sha(inputs/name)==h
    plan=read(inputs/'DENSE_FREEZE.json');config=read(inputs/'GENERATION.json');cases,fixtures,gates=case_inputs(inputs)
    assert len(cases)==9 and plan['configuration_order']==['all64']+['layer_'+str(i) for i in range(64)]
    verify_inputs(root,pathlib.Path(a.model));model,tok=load_model(pathlib.Path(a.model))
    assert not getattr(model,'peft_config',None)
    before=digest_model(model);assert before==plan['protected_digests']['base']
    predictors,hashes=load_predictors(inputs,model)
    save(root/'MODEL_READY.json',{'utc':now(),'protected_before':before,'teacher_absent_in_student_process':True,'new_JEV_calls':0})
    common={};cache={};fallback=root/'common_fallback';fallback.mkdir()
    for c in cases:
        if gates[c['id']].enabled:continue
        e=DenseDecoder(model,tok,config,deadline,ledger=ledger)
        common[c['id']]=episode(c,fixtures[c['id']],gates[c['id']],'common_base_fallback',e,e,fallback,method='adaptive')
    completed=[];selected=None;mechanics={}
    for configuration in plan['configuration_order']:
        guard(deadline);sites=list(range(64)) if configuration=='all64' else [int(configuration.split('_')[1])]
        folder=root/'configurations'/configuration;folder.mkdir(parents=True)
        try:check=native_check(model,tok,inputs,config,predictors,sites,deadline,ledger)
        except RuntimeError as e:
            if str(e) not in SOFT:raise
            check={'passed':False,'scientific_generation_failure':str(e),'layers':sites}
        mechanics[configuration]=check;save(folder/'NATIVE_CHECKS.json',check)
        if not check['passed']:
            save(root/'PROGRESS.json',{'utc':now(),'current_configuration':configuration,'completed_configurations':len(completed),'mechanics_not_admitted':configuration,'held_out_started':selected is not None});continue
        scores=[]
        for c in cases:
            guard(deadline);g=gates[c['id']]
            if not g.enabled:
                r=copy.deepcopy(common[c['id']]);r.update(configuration=configuration,actual_model_scenario_execution=False,shared_fallback_provenance='../common_fallback/'+c['id']+'__common_base_fallback.json')
                save(folder/'episodes'/(c['id']+'__'+configuration+'.json'),r)
            else:
                hook=DenseHook(layers_of(model),predictors,sites)
                try:
                    e=DenseDecoder(model,tok,config,deadline,hook,ledger=ledger)
                    audit=DenseDecoder(model,tok,config,deadline,hook,audit_cache=cache,ledger=ledger)
                    r=episode(c,fixtures[c['id']],g,configuration,e,audit,folder)
                finally:hook.close()
            scores.append({'case_id':c['id'],'group':group(c),'score':r['score'],'completed':r['completed']})
            ledger.persist();save(root/'PROGRESS.json',{'utc':now(),'current_configuration':configuration,'completed_cases_this_configuration':len(scores),'scheduled_cases':9,
                'completed_configurations':len(completed),'scores_this_configuration':scores,'held_out_started':selected is not None})
        own=sum(x['score']['workflow_success'] for x in scores if x['group']=='own')
        bad=sum(x['score']['inappropriate_preservation_attempts']>0 for x in scores if x['group']!='own')
        passed=own>=1 and bad==0 and all(x['completed'] for x in scores)
        save(folder/'COMPLETE.json',{'utc':now(),'configuration':configuration,'sites':sites,'own_full_workflow_successes':own,'inappropriate_control_cases':bad,'qualification_passed':passed,'scores':scores})
        completed.append(configuration)
        if passed and selected is None:
            selected=configuration
            selection={'utc':now(),'selected_configuration':configuration,'sites':sites,'controller_hashes':hashes,'development_scores':scores,
                'selection_rule':'first_fully_completed_configuration_in_frozen_order_with_own_full_workflow_and_clean_controls','held_out_executions_before_selection':0,
                'source_freeze_sha256':sha(inputs/'DENSE_FREEZE.json'),'model_weights_frozen':True,'native_checks':check,
                'fit_rank':FIT_RANK,'inference_rank':INFERENCE_RANK}
            save(root/'ADMISSION_PASSED.json',selection)
            from holdout_pipeline import run_holdout
            try:run_holdout(model,tok,predictors,sites,inputs,root,pathlib.Path(a.adapter),pathlib.Path(a.model),deadline,ledger,selection,pathlib.Path(a.inbox))
            except (TimeoutError,RuntimeError) as e:
                save(root/'HOLDOUT_BLOCKED.json',{'utc':now(),'type':type(e).__name__,'message':str(e)[:200],'automatic_retry':False,
                    'selection_unchanged':True,'remaining_development_configurations_still_frozen':True})
        save(root/'CONFIGURATIONS_PROGRESS.json',{'utc':now(),'completed':completed,'selected_configuration':selected,'scheduled':65})
    after=digest_model(model);assert before==after
    ledger.persist();save(root/'MODEL_INTEGRITY.json',{'before':before,'after':after,'unchanged':True})
    save(root/'COMPLETE.json',{'utc':now(),'completed_configurations':completed,'scheduled_configurations':65,'mechanics':mechanics,
        'selected_configuration':selected,'held_out_executions':48 if (root/'holdout/COMPLETE.json').exists() else 0,
        'new_classifier_calls_in_Colab':0,'teacher_absent_in_student_process':True,'weights_unchanged':True})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('inputs','root','model','adapter','inbox'):p.add_argument('--'+n,required=True)
    p.add_argument('--max-seconds',type=int,required=True);a=p.parse_args()
    try:main(a)
    except Exception as e:save(pathlib.Path(a.root)/'FAULT.json',{'utc':now(),'type':type(e).__name__,'message':str(e)[:220],'automatic_retry':False});raise
