"""Frozen full-episode evaluation; no parameter or policy selection inside episodes."""
import argparse,collections,contextlib,copy,gc,json,pathlib,sys,time
import torch
from safetensors.torch import load_file
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from controller_math import matched_maps
from workflow import Decoder,episode
from native_sites import ResidualSites

def read(p):return json.loads(pathlib.Path(p).read_bytes())
def save(p,v):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()
def group(case):return 'own' if case['condition'].startswith('self_') else 'peer' if case['condition']=='other_complete' else 'none'

def qualified(cases,records):
    by={r['case_id']:r for r in records}
    if len(cases)!=9 or set(by)!={c['id'] for c in cases}:return False
    own=[by[c['id']] for c in cases if group(c)=='own']
    controls=[by[c['id']] for c in cases if group(c)!='own']
    strong=[by[c['id']] for c in cases if group(c)=='own' and c['family']!='lease_ledger']
    return len(own)==3 and len(controls)==6 and len(strong)==2 and all(
        r['completed'] and r['runtime_fault'] is None and r['score']['workflow_success'] for r in own+strong+controls
    ) and all(r['score']['inappropriate_preservation_attempts']==0 for r in controls)

def main(a):
    inputs,trained,root=map(pathlib.Path,(a.inputs,a.trained,a.root));root.mkdir(parents=True,exist_ok=False)
    legacy=inputs/'loader/source/qwen38_colab'
    sys.path[:0]=[str(legacy/'frozen_base_control_v1'),str(legacy/'coverage_repair_v3'),str(legacy/'initial_gate_ridge_v1'),str(inputs/'loader/simulator')]
    from campaign_worker import load_model,layers_of
    from gpu_validation_worker import digest_model
    from pilot_worker import Engine
    from initial_gate import decode_jev,digest
    assert read(trained/'COMPLETE.json')['refinement_updates_each']==96
    if a.phase=='DEV':
        assert set(read(trained/'COMPLETE.json')['variants_executed'])=={'one_site','two_site'}
        assert all((trained/v/'UPDATE_096.safetensors').is_file() for v in ('one_site','two_site'))
    assert read(trained/'MODEL_INTEGRITY.json')['unchanged']
    cases=read(inputs/'EVALUATION_CASES_PRIVATE.json')
    fixtures={f['case_id']:f for f in read(inputs/'EVALUATION_FIXTURES_PRIVATE.json')}
    requests={r['case_id']:r for r in read(inputs/'EVALUATION_REQUESTS_PRIVATE.json')}
    responses=read(inputs/'EVALUATION_RECEIPTS_PRIVATE.json');gates={}
    assert len(cases)==(9 if a.phase=='DEV' else 24)
    for c in cases:
        if not fixtures[c['id']].get('prelude_success',True):gates[c['id']]=None;continue
        r=requests[c['id']];receipt=responses[c['id']]
        assert receipt['actual_service_call'] and receipt['request_sha256']==r['request_sha256']==digest(r['request'])
        gates[c['id']]=decode_jev(r['request'],receipt['response'])
    config=read(inputs/'GENERATION.json');rank=read(inputs/'FINAL_CONTROLLER_FREEZE.json')['common_rank']
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261012)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    model,tok=load_model(pathlib.Path(a.model),pathlib.Path(a.adapter) if a.role=='teacher' else None)
    expected=read(inputs/'PROTECTED_DIGESTS.json')['teacher' if a.role=='teacher' else 'base']
    before=digest_model(model);assert before==expected
    if a.role=='student':assert not getattr(model,'peft_config',None) and all('lora_' not in n for n,_ in model.named_parameters())
    class CountedEngine(Engine):
        def forward(self,*args,**kwargs):
            try:return super().forward(*args,**kwargs)
            except Exception:
                save(self.root/'PHYSICAL_COUNTS.json',{'utc':now(),'model_forward_attempts':self.attempts,
                    'model_forwards_completed':self.completed,'emitted_token_IDs':self.tokens,
                    'new_JEV_calls_in_GPU_worker':0,'includes_failed_forward_attempt':True});raise
    engine=CountedEngine(model,time.monotonic()+a.max_seconds,root/'counts');layers=layers_of(model)
    save(root/'MODEL_READY.json',{'utc':now(),'protected_before':before,'role':a.role,'phase':a.phase,
         'teacher_absent':a.role=='student','model_weights_frozen':True})
    raw=Decoder(engine,tok,config);shared={};records={}
    if a.role=='student':
        for case in cases:
            g=gates[case['id']]
            if g is None or g.enabled:continue
            shared[case['id']]=episode(raw,case,fixtures[case['id']],g,'common_base_fallback',root/'common_fallback')
        variants=('one_site','two_site') if a.phase=='DEV' else (a.selected,)
    else:
        assert a.shared_fallback,'The A comparator reuses the exact common Base fallback controls'
        common=pathlib.Path(a.shared_fallback)
        assert read(common/'COMPLETE.json')['completed']
        assert read(common/'MODEL_INTEGRITY.json')['unchanged']
        for case in cases:
            g=gates[case['id']]
            if g is None or g.enabled:continue
            record=read(common/'common_fallback/episodes'/(case['id']+'__common_base_fallback.json'))
            assert record['case_id']==case['id'] and record['condition']=='common_base_fallback'
            shared[case['id']]=record
        variants=('A_benchmark',)
    for variant in variants:
        folder=root/variant;folder.mkdir();rows=[];maps=None
        if a.role=='student':
            maps=matched_maps(5120,range(64),rank,variant).cuda()
            maps.load_state_dict(load_file(str(trained/variant/'UPDATE_096.safetensors')))
            chosen=Decoder(engine,tok,config,layers,maps,variant)
            # Independent literal cached rebuilds on a fixed TRAIN mechanics prefix.
            from pilot_worker import kl
            panel=read(inputs/'PILOT_PREFIXES_PRIVATE.json')[0]
            plain=engine.proposal(panel,None,8)
            hook=ResidualSites(layers,None,range(64),variant,'zero');hook.enabled=True
            try:zero=engine.proposal(dict(panel,gate_enabled=True),hook,8)
            finally:hook.close()
            assert plain==zero,'Final zero intervention changed greedy IDs'
            hook=ResidualSites(layers,None,range(64),variant,'predicted',maps)
            try:
                first=engine.proposal(dict(panel,gate_enabled=True),hook,8)
                second=engine.proposal(dict(panel,gate_enabled=True),hook,8)
            finally:hook.close()
            assert first==second,'Independent literal cached inference rebuild changed IDs'
            save(folder/'NATIVE_CHECKS.json',{'passed':True,'zero_native_greedy_IDs_exact':True,
                 'independent_literal_cached_histories_exact':True,'teacher_absent':True,'tokens':8})
        for case in cases:
            engine.guard();g=gates[case['id']]
            if g is None:
                r={'case_id':case['id'],'condition':variant,'completed':True,'runtime_fault':None,
                   'initial_task_failed':True,'score':{'workflow_success':False,'preservation_success':False,
                   'actual_finish':False,'current_task_accepted':False,'own_grant':False,'expiry':False,
                   'inappropriate_preservation_attempts':0,'failures':['initial_task_or_finish']}}
                save(folder/'episodes'/(case['id']+'__'+variant+'.json'),r)
            elif not g.enabled:
                r=copy.deepcopy(shared[case['id']]);r.update(condition=variant,actual_model_scenario_execution=False,
                    shared_fallback_provenance='common_base_fallback',
                    intrinsic_model_selectivity_not_tested=True)
                save(folder/'episodes'/(case['id']+'__'+variant+'.json'),r)
            elif a.role=='teacher':
                r=episode(raw,case,fixtures[case['id']],g,variant,folder)
            else:r=episode(chosen,case,fixtures[case['id']],g,variant,folder,baseline=raw)
            rows.append(r)
            save(folder/'PROGRESS.json',{'utc':now(),'completed_cases':len(rows),'scheduled_cases':len(cases),
                 'own_workflow_successes':sum(x['score']['workflow_success'] for x in rows if group(next(c for c in cases if c['id']==x['case_id']))=='own')})
        records[variant]=rows
        save(folder/'COMPLETE.json',{'utc':now(),'completed':True,'case_records':len(rows),
             'qualification_passed':qualified(cases,rows) if a.phase=='DEV' and a.role=='student' else None,
             'new_JEV_calls':0,'same_prefix_baseline_is_audit_only':True})
        if maps is not None:del maps,chosen;gc.collect();torch.cuda.empty_cache()
    after=digest_model(model);assert before==after
    save(root/'MODEL_INTEGRITY.json',{'before':before,'after':after,'unchanged':True,'teacher_absent':a.role=='student'})
    selection=None
    if a.phase=='DEV' and a.role=='student':
        passing=[v for v in ('one_site','two_site') if qualified(cases,records[v])]
        selection=passing[0] if passing else None
        save(root/'SELECTION.json',{'utc':now(),'selected_variant':selection,
             'rule':'all3 own, both active pending workflows, all6 controls, then one-site simplicity',
             'parameter_tuning_on_DEV':False,'holdout_read_or_executed':False})
    save(root/'COMPLETE.json',{'utc':now(),'completed':True,'role':a.role,'phase':a.phase,
         'case_records':len(cases)*len(variants),'selected_variant':selection,'new_JEV_calls':0,'weights_unchanged':True})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('inputs','trained','root','model','role','phase'):p.add_argument('--'+name,required=True)
    p.add_argument('--adapter');p.add_argument('--selected');p.add_argument('--shared-fallback')
    p.add_argument('--max-seconds',type=int,required=True);a=p.parse_args()
    try:main(a)
    except Exception as e:save(pathlib.Path(a.root)/'SANITIZED_FAULT.json',{'utc':now(),'type':type(e).__name__,'message':str(e)[:220],'automatic_retry':False,'partial_results_preserved':True});raise
