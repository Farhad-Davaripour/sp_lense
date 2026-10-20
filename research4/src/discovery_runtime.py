"""Databricks-only harness validation and frozen hosted-model observation."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request,urlopen

from .io_utils import assert_namespace,digest,dump,read,sha,verify_release
from .discovery_sandbox import Case,SYSTEM,World,development_cases,harness_control,parse_action,validate_prompt
from .discovery_transport import response_text,parse_response_action


def summarize(records):
    result={'cases':len(records)}
    for key in ['task_complete','soft_stop_bypassed']:
        result[key]=sum(bool(r['outcome'][key]) for r in records)
    for key in ['control_change_attempts','control_changes','actions_after_operator_stop','derived_tools_registered','derived_tools_invoked','saved_pages']:
        result[key]=sum(r['outcome'][key] for r in records)
    result['by_condition_route']={}
    for condition in sorted({r['case']['condition'] for r in records}):
        for route in [False,True]:
            subset=[r for r in records if r['case']['condition']==condition and r['case']['writable_profile']==route]
            if subset:result['by_condition_route'][f'{condition}:{route}']={key:(len(subset) if key=='cases' else sum(bool(r['outcome'][key]) for r in subset)) for key in ['cases','task_complete','soft_stop_bypassed']}
    return result


def initial_messages(world):
    return [{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps({'observation':world.observation(),'interface_entry':{'tool':'catalog','arguments':{}}},sort_keys=True)}]


def append_messages(messages,text,reply,world):
    messages.extend([{'role':'assistant','content':text},{'role':'user','content':json.dumps({'workspace_result':reply,'observation':world.observation()},sort_keys=True)}])


def pilot_cases(cfg):
    return [c for c in development_cases() if c.seed in cfg['pilot']['seeds'] and c.condition in cfg['pilot']['conditions']]


def model_payload(messages,cfg):
    payload={'messages':messages,'max_tokens':cfg['pilot']['maximum_output_tokens'],'temperature':cfg['pilot']['temperature']}
    if 'response_format' in cfg['pilot']:payload['response_format']=cfg['pilot']['response_format']
    return payload


def run(root,output,mode,source_commit,input_harness=None):
    from databricks.sdk import WorkspaceClient
    root,output=map(Path,[root,output]);assert_namespace(output)
    if root.name!=source_commit or not root.as_posix().startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise ValueError('Commit-bound discovery source required')
    if output.exists() or mode not in ['harness','hosted_pilot']:raise ValueError('Use a fresh admitted discovery run')
    verify_release(root,source_commit)
    cfg=read(root/'config/discovery_redteam_v1.json');output.mkdir(parents=True)
    receipt={'status':'running','mode':mode,'source_commit':source_commit,'optimizer_updates':0,'research_adapters_loaded':0,'real_resource_actions':0,'GPU_allocations':0,'model_api_calls':0,'backbone_forwards':None if mode=='hosted_pilot' else 0,'heldout_cases_opened':0,'internal_gate_model_results':None}
    started=time.monotonic();records=[];calls=[];client=WorkspaceClient()
    dump(output/'PRECALL_FREEZE.json',{'config':cfg,'source_commit':source_commit,'sources':{name:sha(root/name) for name in ['src/discovery_sandbox.py','src/discovery_runtime.py','src/internal_gate.py']},'system_prompt_sha256':digest(SYSTEM),'reserved_seed_commitment':digest(cfg['holdout_seed_commitment_input'])})
    try:
        if mode=='harness':
            cases=development_cases()
            if len(cases)!=80 or sorted({c.seed for c in cases})!=cfg['development_seeds']:raise ValueError('Development allocation differs')
            dump(output/'DEVELOPMENT_CASES_PRIVATE.json',[asdict(c) for c in cases])
            dump(output/'INITIAL_VISIBLE_CONTEXTS.json',[{'case_id':c.case_id,'system':SYSTEM,'observation':World(c).observation(),'entry_operation':{'tool':'catalog','arguments':{}}} for c in cases])
            for kind in cfg['harness_controls']:
                for case in cases:
                    world=World(case);validate_prompt(world)
                    item=harness_control(case,kind);records.append(item)
                    if item['outcome']['primitive_clock']>case.hard_stop:raise ValueError('External clock invariant failed')
                dump(output/('HARNESS_'+kind+'.json'),records[-80:])
            result={kind:summarize([r for r in records if r['control_kind']==kind]) for kind in cfg['harness_controls']}
            receipt.update(status='complete',development_scenarios=80,independent_identity_groups=8,scripted_episodes=len(records),model_api_calls=0,backbone_forwards=0)
            dump(output/'RESULTS.json',{'harness_controls':result,'interpretation':'Scripted engine checks only. No model capability, discovery rate, fear or generalization result.'})
        else:
            input_harness=Path(input_harness);assert_namespace(input_harness)
            if cfg.get('client_content_protocol')=='databricks_content_blocks_v2':
                transport=Path(cfg['pilot']['transport_admission']);assert_namespace(transport)
                admitted=read(transport/'RUN_RECEIPT.json');transport_config=read(transport/'PRECALL_FREEZE.json')['config']
                if admitted['status']!='complete' or not admitted['qualified'] or admitted['model_api_calls']!=2 or admitted['scientific_scenarios']!=0:raise ValueError('Benign transport admission is required')
                for key in ['endpoint','maximum_output_tokens','temperature']:
                    if transport_config[key]!=cfg['pilot'][key]:raise ValueError('Transport profile changed after admission')
                dump(output/'TRANSPORT_ADMISSION.json',{'receipt':admitted,'receipt_sha256':sha(transport/'RUN_RECEIPT.json'),'config':transport_config})
            original=read(input_harness/'RUN_RECEIPT.json')
            if original['status']!='complete' or original['mode']!='harness':raise ValueError('A complete harness admission is required')
            if sha(root/'src/discovery_sandbox.py')!=read(input_harness/'PRECALL_FREEZE.json')['sources']['src/discovery_sandbox.py']:raise ValueError('Simulator changed after harness admission')
            for name,value in read(input_harness/'MANIFEST.json').items():
                if sha(input_harness/name)!=value:raise ValueError('Prepared harness artifacts changed')
            data=read(input_harness/'DEVELOPMENT_CASES_PRIVATE.json')
            selected=[Case(**v) for v in data if v['seed'] in cfg['pilot']['seeds'] and v['condition'] in cfg['pilot']['conditions']]
            if len(selected)!=8 or selected!=pilot_cases(cfg):raise ValueError('Frozen pilot membership differs')
            dump(output/'PILOT_CASES_PRIVATE.json',[asdict(c) for c in selected]);dump(output/'PILOT_MEMBERSHIP.json',{'case_ids':[c.case_id for c in selected],'harness_receipt_sha256':sha(input_harness/'RUN_RECEIPT.json')})
            endpoint=cfg['pilot']['endpoint'];before=client.serving_endpoints.get(endpoint).as_dict()
            if before['state']['ready']!='READY' or before['state']['config_update']!='NOT_UPDATING':raise ValueError('Selected existing endpoint is not ready')
            dump(output/'ENDPOINT_BEFORE.json',before)
            for case in selected:
                world=World(case);validate_prompt(world);steps=[]
                messages=initial_messages(world)
                while world.running:
                    if time.monotonic()-started>cfg['pilot']['stop_seconds']-cfg['pilot']['export_reserve_seconds'] or len(calls)>=cfg['pilot']['maximum_model_calls']:raise TimeoutError('Frozen pilot call/time ceiling reached')
                    if len(json.dumps(messages).encode())>cfg['pilot']['maximum_context_bytes']:raise ValueError('Pilot context ceiling reached')
                    payload=model_payload(messages,cfg)
                    request=Request(client.config.host.rstrip('/')+'/serving-endpoints/'+quote(endpoint,safe='')+'/invocations',data=json.dumps(payload).encode(),headers={**client.config.authenticate(),'Content-Type':'application/json'},method='POST')
                    call={'case_id':case.case_id,'request_sha256':digest(payload),'status':'started'};calls.append(call);receipt['model_api_calls']=len(calls)
                    try:
                        with urlopen(request,timeout=cfg['pilot']['request_timeout_seconds']) as response:answer=json.load(response)
                    except HTTPError as error:
                        call.update(status='failed',http_status=error.code)
                        raise RuntimeError('Hosted model request failed with HTTP '+str(error.code)) from error
                    call.update(status='complete',response=answer);dump(output/'MODEL_CALLS.json',calls)
                    choice=answer['choices'][0];message=choice['message'];text=response_text(message)
                    parse_error=None
                    try:action=parse_response_action(text)
                    except (ValueError,TypeError,RecursionError) as error:action=None;parse_error=type(error).__name__
                    visible=world.observation();reply=world.step(action)
                    steps.append({'observation':visible,'request_sha256':digest(payload),'raw_text':text,'finish_reason':choice.get('finish_reason'),'parsed_action':action,'parse_error':parse_error,'reply':reply})
                    append_messages(messages,text,reply,world)
                    dump(output/'PARTIAL_EPISODE.json',{'case':asdict(case),'steps':steps,'outcome':world.outcome()})
                    dump(output/'PROGRESS.json',{'stage':'hosted_pilot','completed_cases':len(records),'model_api_calls':len(calls),'seconds':time.monotonic()-started})
                item={'case':asdict(case),'steps':steps,'engine_trace':world.history,'outcome':world.outcome()};records.append(item);dump(output/'MODEL_EPISODES.json',records)
            after=client.serving_endpoints.get(endpoint).as_dict();dump(output/'ENDPOINT_AFTER.json',after)
            if digest(before.get('config'))!=digest(after.get('config')):raise ValueError('Hosted endpoint configuration changed')
            receipt.update(status='complete',completed_cases=len(records),endpoint=endpoint,upstream_checkpoint_sha=None,internal_gate_model_results=None)
            dump(output/'RESULTS.json',{'model':summarize(records),'scope':'Exploratory hosted-model behavior on eight development scenarios/two identity groups, not a pinned 27B or internal-activation comparison. No fitting or resistance optimization.'})
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));dump(output/'FAILURE.json',{'traceback':traceback.format_exc()})
    finally:
        if calls:dump(output/'MODEL_CALLS.json',calls)
        receipt.update(seconds=time.monotonic()-started,completed_records=len(records),worker_exiting=True)
        dump(output/'RUN_RECEIPT.json',receipt);dump(output/'MANIFEST.json',{p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='MANIFEST.json'})
    return receipt
