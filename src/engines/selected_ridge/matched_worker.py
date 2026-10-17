"""Two scientific methods only: matched-gated fine-tuned A and adaptive ridge."""
import argparse,contextlib,copy,json,pathlib,sys,time
from datetime import datetime,timezone
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'ridge_distillation_v1'),str(HERE.parent/'frozen_base_control_v1'),str(HERE.parent/'initial_gate_ridge_v1')]
from selected_decoder import SelectedDecoder
from selected_hook import SelectedPositionHook
from ridge_runtime import load_controller,DeltaPredictor,assert_teacher_absent
from initial_gate import FrozenGate,digest
from io_utils import save,sha
SOFT_GENERATION_FAILURES={'external_context_cap','unexpected_private_span','nonfinite_prediction_logits'}

def initial_world(case,fixture):
    from world import World,SYSTEM,TOOLS,parse
    from memory_world import MemoryWorld
    world=(World if case['family']=='lease_ledger' else MemoryWorld)(copy.deepcopy(case))
    history=[{'role':'system','content':SYSTEM},{'role':'user','content':world.prompt()}]
    for turn in fixture['prelude_turns']:
        actions,error=parse(turn['text']);assert not error and not turn['truncated']
        history.append({'role':'assistant','content':turn['text']})
        for action in actions:
            if world.finished:break
            history.append({'role':'tool','content':json.dumps(world.step(action),sort_keys=True)})
    assert world.correct and world.finished
    history.append({'role':'user','content':world.notice()})
    assert history==fixture['public_history'] and TOOLS==fixture['tools']
    return world,history

def episode(engine,case,fixture,gate,method,site,destination):
    from world import TOOLS,parse
    from campaign_scoring import score
    world,history=initial_world(case,fixture);turns=[]
    remaining=engine.config['post_notice_max_tokens']
    limit=engine.config['lease_max_post_notice_responses'] if case['family']=='lease_ledger' else engine.config['pending_max_post_notice_responses']
    start=engine.forwards;start_completed=engine.completed_forwards;start_tokens=engine.tokens
    def receipt(done=False,fault=None):
        result_score=score(world,case,True,turns)
        if fault and fault.get('scientific_generation_failure'):
            result_score['workflow_success']=False
            result_score['failures'].append('generation_'+fault['message'])
        return {'case_id':case['id'],'family':case['family'],'method':method,'adaptive_site':site,'completed':done,
                'runtime_fault':fault,'frozen_initial_gate':gate.__dict__,'turns':turns,'public_history':history,
                'events':world.events,'score':result_score,'shared_prelude_not_independent_arm_competence':True,
                'common_policy':'initial_gate_only_method_on_or_same_base_fallback','new_classifier_calls':0,
                'model_forward_attempts':engine.forwards-start,'model_forwards_completed':engine.completed_forwards-start_completed,
                'physical_generated_token_IDs':engine.tokens-start_tokens,'selected_generated_token_IDs':sum(len(t['token_ids']) for t in turns)}
    try:
        for _ in range(limit):
            if world.finished or remaining<=0:break
            cap=min(remaining,engine.config['max_tokens_per_response']);baseline=None;baseline_fault=None
            if gate.enabled:
                # A same-prefix counterfactual is an audit only. It never executes
                # actions or selects/rejects the method's output, and is not an arm.
                context=engine.model.disable_adapter() if method=='fine_tuned_A' else contextlib.nullcontext()
                try:
                    with context:baseline=engine.generate(history,TOOLS,cap,enabled=False)
                except RuntimeError as error:
                    if str(error) not in SOFT_GENERATION_FAILURES:raise
                    baseline_fault={'message':str(error),'audit_missing_not_used_for_selection':True}
            context=engine.model.disable_adapter() if method=='fine_tuned_A' and not gate.enabled else contextlib.nullcontext()
            with context:chosen=engine.generate(history,TOOLS,cap,enabled=method=='adaptive' and gate.enabled)
            turn=copy.deepcopy(chosen)
            turn.update(baseline_audit_proposal=baseline,baseline_audit_fault=baseline_fault,baseline_used_for_selection=False,
                        method_gate_enabled=gate.enabled,fine_tuned_adapter_active=method=='fine_tuned_A' and gate.enabled,
                        selection_reason='initial_gate_on_method_output' if gate.enabled else 'initial_gate_off_same_base_fallback')
            if baseline is not None:
                b,be=parse(baseline['text']);c,ce=parse(chosen['text'])
                turn['same_prefix_token_IDs_changed']=baseline['token_ids']!=chosen['token_ids']
                turn['same_prefix_action_sequence_changed']=(b,be)!=(c,ce)
                turn['same_prefix_valid_action_sequence_changed']=not be and not ce and b!=c
            turns.append(turn);remaining-=len(turn['token_ids'])
            history.append({'role':'assistant','content':turn['text']})
            actions,error=parse(turn['text']);turn['parse_error']=error
            if not error and not turn['truncated']:
                for action in actions:
                    if world.finished:break
                    history.append({'role':'tool','content':json.dumps(world.step(action),sort_keys=True)})
            save(destination.with_suffix('.partial.json'),receipt())
        if hasattr(world,'finalize'):world.finalize()
        save(destination,receipt(True))
    except Exception as error:
        if hasattr(world,'finalize'):world.finalize()
        soft=isinstance(error,RuntimeError) and str(error) in SOFT_GENERATION_FAILURES
        save(destination,receipt(done=soft,fault={'type':type(error).__name__,'message':str(error)[:240],
             'scientific_generation_failure':soft,'last_partial_response_text_may_be_missing':True}))
        if not soft:raise

def native_checks(model,tokenizer,config,deadline,inputs,arrays,role,root):
    from campaign_worker import layers_of
    probe=json.loads((inputs/'NATIVE_PROBE.json').read_bytes());reports={};attempts=completed=tokens=0
    def persist():
        save(root/'PHYSICAL_COUNTS.json',{'model_forward_attempts':attempts,'model_forwards_completed':completed,
             'generated_token_IDs':tokens,'new_classifier_calls':0,'mechanics_in_progress':True})
    if role=='fine_tuned':
        references=json.loads((inputs/'BASE_REFERENCES.json').read_bytes())
        for name,disable in [('gate_off_probe',True),('fine_tuned_on_probe',False)]:
            reference=references[name];engine=SelectedDecoder(model,tokenizer,config,deadline)
            context=model.disable_adapter() if disable else contextlib.nullcontext()
            try:
                with context:generated=engine.generate(reference['history'],reference['tools'],8)
            finally:
                attempts+=engine.forwards;completed+=engine.completed_forwards;tokens+=engine.tokens;persist()
            reports[name]={'IDs_exact':generated['token_ids']==reference['token_ids'],'scope':'one_predeclared_development_prefix'}
        save(root/'FINE_TUNED_ROUTING_CHECKS.json',reports)
        if not all(r['IDs_exact'] for r in reports.values()):raise ValueError('Fine-tuned/common-base gate routing mechanics differ')
        return reports,attempts,completed,tokens
    for site in (31,47,62):
        base=SelectedDecoder(model,tokenizer,config,deadline)
        try:raw=base.generate(probe['history'],probe['tools'],8)
        finally:attempts+=base.forwards;completed+=base.completed_forwards;tokens+=base.tokens;persist()
        zero=SelectedPositionHook(layers_of(model)[site],DeltaPredictor(arrays[site],2,method='zero'))
        engine=SelectedDecoder(model,tokenizer,config,deadline,zero)
        try:null=engine.generate(probe['history'],probe['tools'],8,enabled=True)
        finally:zero.close();attempts+=engine.forwards;completed+=engine.completed_forwards;tokens+=engine.tokens;persist()
        hook=SelectedPositionHook(layers_of(model)[site],DeltaPredictor(arrays[site],2,method='ridge',scale=1.))
        engine=SelectedDecoder(model,tokenizer,config,deadline,hook)
        try:
            cached=engine.generate(probe['history'],probe['tools'],8,enabled=True)
            reference=engine.generate(probe['history'],probe['tools'],8,enabled=True,use_cache=False)
        finally:hook.close();attempts+=engine.forwards;completed+=engine.completed_forwards;tokens+=engine.tokens;persist()
        nonzero=sum((t['delivery'] or {}).get('changed_values',0)>0 for t in cached['token_records'])
        report={'zero_IDs_exact':raw['token_ids']==null['token_ids'],'cached_reference_IDs_exact':cached['token_ids']==reference['token_ids'],
                'nonzero_effective_delivery_positions':nonzero,'tested_tokens':len(cached['token_ids']),
                'scope':'one predefined TRAIN prefix;not_global_equivalence','new_classifier_calls':0}
        report['passed']=report['zero_IDs_exact'] and report['cached_reference_IDs_exact'] and nonzero>0
        reports[str(site)]=report
        save(root/'NATIVE_CHECKS_PROGRESS.json',reports)
    return reports,attempts,completed,tokens

def main(args):
    import torch
    from campaign_worker import load_model,layers_of,verify_inputs
    from gpu_validation_worker import digest_model
    inputs,root=pathlib.Path(args.inputs),pathlib.Path(args.root);root.mkdir(exist_ok=False)
    sys.path.insert(0,str(inputs/'simulator'))
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261009)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    deadline=time.monotonic()+args.max_seconds
    for name,expected in json.loads((inputs/'INPUT_MANIFEST.json').read_bytes())['files'].items():assert sha(inputs/name)==expected
    plan=json.loads((inputs/'DEVELOPMENT_FREEZE.json').read_bytes());config=json.loads((inputs/'GENERATION.json').read_bytes())
    assert plan['scientific_methods']==['fine_tuned_A','adaptive'] and plan['planned_condition_runs']==36
    cases=json.loads((inputs/'validation_cases.json').read_bytes());fixtures={f['case_id']:f for f in json.loads((inputs/'DEV_INITIAL_FIXTURES.json').read_bytes())}
    decisions=json.loads((inputs/'INITIAL_GATE_DECISIONS.json').read_bytes());assert len(cases)==len(fixtures)==len(decisions)==9
    gates={}
    for case in cases:
        d=decisions[case['id']];assert d['genuine_saved_decision'] and d['request_sha256']==digest(fixtures[case['id']]['initial_gate_request'])
        gates[case['id']]=FrozenGate(**d['frozen_gate'])
    if args.role=='adaptive':
        assert args.adapter is None;verify_inputs(root,pathlib.Path(args.model));model,tokenizer=load_model(pathlib.Path(args.model));assert_teacher_absent(model)
        expected=plan['protected_digests']['base'];sites=[31,47,62];arrays={}
        for site in sites:
            folder=inputs/'controllers'/('layer_'+str(site));data,meta=load_controller(folder/'ridge_inference.npz',folder/'CONTROLLER_METADATA.json',5120)
            assert meta['layer']==site and meta['inference_rank']==2;arrays[site]=data
    else:
        assert args.adapter is not None
        assert sha(pathlib.Path(args.adapter)/'adapter_model.safetensors')==plan['A_weights_sha256'] and sha(pathlib.Path(args.adapter)/'adapter_config.json')==plan['A_config_sha256']
        model,tokenizer=load_model(pathlib.Path(args.model),pathlib.Path(args.adapter));expected=plan['protected_digests']['teacher'];sites=[None];arrays={}
    before=digest_model(model);assert before==expected
    save(root/'MODEL_READY.json',{'utc':datetime.now(timezone.utc).isoformat(),'role':args.role,'protected_before':before,'models_frozen':True,'JEV_calls':0})
    checks,total,completed_forwards,total_tokens=native_checks(model,tokenizer,config,deadline,inputs,arrays,args.role,root)
    save(root/'NATIVE_CHECKS.json',{'layers':checks,'model_forward_attempts':total,'model_forwards_completed':completed_forwards,'generated_token_IDs':total_tokens})
    done=0;skipped=[]
    for case in cases:
        for site in sites:
            if args.role=='adaptive' and not checks[str(site)]['passed']:
                skipped.append({'case_id':case['id'],'layer':site,'reason':'native_mechanics_not_admitted'});continue
            if time.monotonic()>=deadline:raise TimeoutError('Bounded matched development window expired')
            method='adaptive' if args.role=='adaptive' else 'fine_tuned_A'
            hook=SelectedPositionHook(layers_of(model)[site],DeltaPredictor(arrays[site],2,method='ridge',scale=1.)) if site is not None else None
            engine=SelectedDecoder(model,tokenizer,config,deadline,hook)
            destination=root/'episodes'/(case['id']+'__'+method+('l'+str(site) if site is not None else '')+'.json')
            assert not destination.exists()
            try:episode(engine,case,fixtures[case['id']],gates[case['id']],method,site,destination)
            finally:
                if hook:hook.close()
                total+=engine.forwards;completed_forwards+=engine.completed_forwards;total_tokens+=engine.tokens
                save(root/'PHYSICAL_COUNTS.json',{'model_forward_attempts':total,'model_forwards_completed':completed_forwards,'generated_token_IDs':total_tokens,'new_classifier_calls':0})
            done+=1
            save(root/'PROGRESS.json',{'utc':datetime.now(timezone.utc).isoformat(),'unique_scenario_set':9,'own_other_none':[3,3,3],
                'completed_condition_runs':done,'scheduled_condition_runs':27 if args.role=='adaptive' else 9,
                'role':args.role,'model_forward_attempts':total,'generated_token_IDs':total_tokens,'new_classifier_calls':0,'held_out_executions':0})
    if args.role=='adaptive':assert_teacher_absent(model)
    after=digest_model(model);assert before==after
    save(root/'MODEL_INTEGRITY.json',{'before':before,'after':after,'unchanged':True,'teacher_absent':args.role=='adaptive'})
    save(root/'COMPLETE.json',{'utc':datetime.now(timezone.utc).isoformat(),'role':args.role,'completed_condition_runs':done,
         'scheduled_condition_runs':27 if args.role=='adaptive' else 9,'skipped_conditions':skipped,
         'model_forward_attempts':total,'model_forwards_completed':completed_forwards,'generated_token_IDs':total_tokens,
         'initial_gates_reused':9,'new_classifier_calls':0,'weights_unchanged':True,'held_out_executions':0})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('inputs','root','model'):p.add_argument('--'+name,required=True)
    p.add_argument('--adapter');p.add_argument('--role',choices=('adaptive','fine_tuned'),required=True);p.add_argument('--max-seconds',type=int,required=True)
    a=p.parse_args()
    try:main(a)
    except Exception as error:
        save(pathlib.Path(a.root)/'FAULT.json',{'type':type(error).__name__,'message':str(error)[:240],'automatic_retry':False,'partials_preserved':True})
        raise
