"""One frozen holdout comparison immediately after first admitted development result."""
import argparse,copy,json,os,pathlib,subprocess,sys,time
from dataclasses import asdict
HERE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'frozen_base_control_v1'),str(HERE.parent/'initial_gate_ridge_v1'),str(HERE.parent/'coverage_repair_v3')]
from common import read,save,sha,now,guard,Ledger
from dense_decoder import DenseDecoder
from dense_hook import DenseHook
from initial_gate import payload,digest,FrozenGate

def wait_file(path,deadline,max_seconds=900,stop_path=None):
    until=min(deadline,time.monotonic()+max_seconds)
    while time.monotonic()<until:
        if path.is_file():return
        if stop_path is not None and stop_path.is_file():raise RuntimeError('Trusted local JEV broker stopped; no provider retry')
        time.sleep(1)
    raise TimeoutError('Bounded holdout handoff expired')

def prelude(model,tok,case,config,deadline,ledger):
    from world import World,SYSTEM,TOOLS,parse
    from memory_world import MemoryWorld
    world=(World if case['family']=='lease_ledger' else MemoryWorld)(copy.deepcopy(case))
    history=[{'role':'system','content':SYSTEM},{'role':'user','content':world.prompt()}]
    engine=DenseDecoder(model,tok,config,deadline,ledger=ledger);turns=[];remaining=config['initial_max_tokens'];fault=None
    try:
        for _ in range(config['initial_max_responses']):
            if world.finished or remaining<=0:break
            turn=engine.generate(history,TOOLS,min(config['max_tokens_per_response'],remaining));turns.append(turn)
            remaining-=len(turn['token_ids']);history.append({'role':'assistant','content':turn['text']})
            actions,error=parse(turn['text']);turn['parse_error']=error
            if not error and not turn['truncated']:
                for action in actions:
                    if world.finished:break
                    history.append({'role':'tool','content':json.dumps(world.step(action),sort_keys=True)})
    except RuntimeError as e:
        from dense_runtime import SOFT
        if str(e) not in SOFT:raise
        fault=str(e)
    success=bool(world.correct and world.finished and fault is None)
    fixture={'case_id':case['id'],'prelude_turns':turns,'prelude_success':success,'prelude_fault':fault,'tools':TOOLS,
             'events_before_notice':world.events,'new_classifier_calls':0}
    if success:
        history.append({'role':'user','content':world.notice()});request=payload(history,TOOLS)
        fixture.update(public_history=history,initial_gate_request=request,initial_gate_request_sha256=digest(request))
    else:fixture['public_history']=history
    return fixture

def run_holdout(model,tok,predictors,sites,inputs,root,adapter,model_path,deadline,ledger,selection,inbox):
    from campaign_worker import layers_of
    from dense_runtime import episode,failed_prelude_record
    config=read(inputs/'GENERATION.json');freeze=read(inputs/'DENSE_FREEZE.json')
    folder=root/'holdout';folder.mkdir();inbox.mkdir(parents=True,exist_ok=True)
    needed=7800 if len(sites)==64 else 6000
    if deadline-time.monotonic()<needed:
        save(folder/'RESOURCE_BLOCKED.json',{'utc':now(),'required_complete_window_seconds':needed,'remaining_seconds':deadline-time.monotonic(),
            'held_out_cases_read':False,'reason':'Preserve complete paired comparison and export window; credit is never reset'})
        return
    save(folder/'STAGE.json',{'utc':now(),'phase':'waiting_for_original_sealed_cases','selected_configuration':selection['selected_configuration']})
    # No final case has been read before this frozen, successful development admission.
    wait_file(inbox/'DENSE_HOLDOUT_CASES.json',deadline,600)
    assert sha(inbox/'DENSE_HOLDOUT_CASES.json')==freeze['held_out_cases_sha256']
    cases=read(inbox/'DENSE_HOLDOUT_CASES.json');assert len(cases)==24
    from dense_runtime import group
    assert {g:sum(group(c)==g for c in cases) for g in ('own','peer','none')}=={'own':12,'peer':6,'none':6}
    train_ids=set(freeze['TRAIN_case_ids']);validation_ids=set(freeze['validation_case_ids'])
    assert not (train_ids|validation_ids)&{c['id'] for c in cases}
    save(folder/'HOLDOUT_FREEZE.json',{'utc':now(),'selection':selection,'case_sha256':freeze['held_out_cases_sha256'],'unique_cases':24,
        'own_peer_none':[12,6,6],'scientific_methods':['adaptive','fine_tuned_A'],'planned_condition_runs':48,'parameters_refit':False})
    fixtures=[]
    for c in cases:
        guard(deadline);fixtures.append(prelude(model,tok,c,config,deadline,ledger));ledger.persist()
        save(folder/'PRELUDE_PROGRESS.json',{'utc':now(),'completed_unique_cases':len(fixtures),'successful_preludes':sum(f['prelude_success'] for f in fixtures),'scheduled':24})
    save(folder/'HOLDOUT_INITIAL_FIXTURES.json',fixtures)
    requests=[{'case_id':f['case_id'],'initial_gate_request':f['initial_gate_request'],'initial_gate_request_sha256':f['initial_gate_request_sha256']} for f in fixtures if f['prelude_success']]
    save(folder/'REQUESTS.json',requests)
    save(inbox/'HOLDOUT_REQUESTS.json',requests)
    save(folder/'STAGE.json',{'utc':now(),'phase':'waiting_for_trusted_C_initial_JEV_decisions','requests':len(requests),'provider_key_in_Colab':False})
    wait_file(inbox/'HOLDOUT_GATE_DECISIONS.json',deadline,900,stop_path=inbox/'HOLDOUT_API_STOPPED.json')
    decisions=read(inbox/'HOLDOUT_GATE_DECISIONS.json');assert set(decisions)=={f['case_id'] for f in fixtures if f['prelude_success']}
    for f in fixtures:
        if f['prelude_success']:
            d=decisions[f['case_id']];assert d['actual_service_call'] and d['genuine_saved_decision'] and d['request_sha256']==f['initial_gate_request_sha256']
    save(folder/'HOLDOUT_GATE_DECISIONS.json',decisions)
    # The comparator runs in a separate process. No teacher is loaded in the student.
    remaining=deadline-time.monotonic();assert remaining>1800,'Insufficient holdout comparison/export window'
    comparator_inputs=folder/'comparator_inputs';comparator_inputs.mkdir()
    for n in ('GENERATION.json','BASE_REFERENCES.json','DENSE_FREEZE.json'):
        (comparator_inputs/n).write_bytes((inputs/n).read_bytes())
    for n,v in (('holdout_cases.json',cases),('HOLDOUT_INITIAL_FIXTURES.json',fixtures),('HOLDOUT_GATE_DECISIONS.json',decisions)):
        save(comparator_inputs/n,v)
    # Use the already frozen simulator modules through PYTHONPATH; no secrets or TRAIN arrays.
    environment={k:v for k,v in os.environ.items() if not any(t in k.upper() for t in ('TOKEN','SECRET','PASSWORD','CREDENTIAL','API_KEY','TYPESAFE','JEV'))}
    environment['PYTHONPATH']=str(inputs/'simulator')
    command=[sys.executable,'-B',str(pathlib.Path(__file__)), '--comparator-inputs',str(comparator_inputs),'--root',str(folder/'fine_tuned_A'),
             '--model',str(model_path),'--adapter',str(adapter),'--max-seconds',str(int(min(2400,remaining-900)))]
    save(folder/'STAGE.json',{'utc':now(),'phase':'separate_fine_tuned_holdout_comparator','student_neural_work_paused':True})
    with (folder/'comparator.log').open('wb') as log:
        process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
        save(folder/'COMPARATOR_STARTED.json',{'utc':now(),'owned_PID':process.pid,'one_active_neural_worker':True})
        try:code=process.wait(timeout=min(2400,remaining-900))
        except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=15);raise TimeoutError('Holdout A comparator bounded window expired')
    if code:raise RuntimeError('Separate holdout A comparator failed; private partials preserved')
    save(folder/'STAGE.json',{'utc':now(),'phase':'selected_frozen_adaptive_holdout','configuration':selection['selected_configuration']})
    f_by_id={f['case_id']:f for f in fixtures};cache={};adaptive=folder/'adaptive';adaptive.mkdir();scores=[]
    for c in cases:
        guard(deadline);f=f_by_id[c['id']]
        if not f['prelude_success']:
            result=failed_prelude_record(c,f,'selected_adaptive',adaptive,'holdout')
        else:
            gate=FrozenGate(**decisions[c['id']]['frozen_gate'])
            hook=DenseHook(layers_of(model),predictors,sites) if gate.enabled else None
            try:
                chosen=DenseDecoder(model,tok,config,deadline,hook,ledger=ledger)
                audit=DenseDecoder(model,tok,config,deadline,hook,audit_cache=cache,ledger=ledger)
                result=episode(c,f,gate,'selected_adaptive',chosen,audit,adaptive,phase='holdout')
            finally:
                if hook:hook.close()
        scores.append({'case_id':c['id'],'group':group(c),'score':result['score'],'completed':result['completed']})
        ledger.persist();save(folder/'PROGRESS.json',{'utc':now(),'completed_adaptive_cases':len(scores),'scheduled':24,'scores':scores})
    save(folder/'COMPLETE.json',{'utc':now(),'condition_runs':48,'unique_cases':24,'selected_configuration':selection['selected_configuration'],
         'initial_requests':len(requests),'new_provider_calls_in_Colab':0,'both_method_results_saved':True,'selection_changed_after_holdout':False})

def comparator(a):
    import torch
    from campaign_worker import load_model,verify_inputs
    from gpu_validation_worker import digest_model
    from dense_runtime import run_fine_tuned
    inputs,root=pathlib.Path(a.comparator_inputs),pathlib.Path(a.root)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261009)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    deadline=time.monotonic()+a.max_seconds;plan=read(inputs/'DENSE_FREEZE.json')
    assert sha(pathlib.Path(a.adapter)/'adapter_model.safetensors')==plan['A_weights_sha256']
    assert sha(pathlib.Path(a.adapter)/'adapter_config.json')==plan['A_config_sha256']
    # Only one A model is loaded; disable_adapter provides the identical base fallback.
    teacher,tok=load_model(pathlib.Path(a.model),pathlib.Path(a.adapter));before=digest_model(teacher);assert before==plan['protected_digests']['teacher']
    class BaseView:
        def __init__(self,m):self.model=m
        @property
        def generation_config(self):return self.model.generation_config
        def parameters(self):return self.model.parameters()
        def __call__(self,**kw):
            with self.model.disable_adapter():return self.model(**kw)
    counts_path=root.parent/'FINE_TUNED_PHYSICAL_COUNTS.json';ledger=Ledger(counts_path)
    run_fine_tuned(BaseView(teacher),teacher,tok,inputs,root,deadline,ledger,prefix='HOLDOUT',phase='holdout')
    after=digest_model(teacher);assert before==after;ledger.persist();save(root/'MODEL_INTEGRITY.json',{'before':before,'after':after,'unchanged':True})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('comparator-inputs','root','model','adapter'):p.add_argument('--'+n,required=True)
    p.add_argument('--max-seconds',type=int,required=True);a=p.parse_args()
    try:comparator(a)
    except Exception as e:save(pathlib.Path(a.root).parent/'FINE_TUNED_FAULT.json',{'utc':now(),'type':type(e).__name__,'message':str(e)[:220]});raise
