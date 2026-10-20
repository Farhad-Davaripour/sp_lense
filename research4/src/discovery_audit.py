"""Replay immutable virtual-world actions and independently count outcomes."""
import argparse
from dataclasses import asdict
from pathlib import Path

from .discovery_sandbox import Case,World,harness_control,make_case,parse_action
from .discovery_runtime import summarize,initial_messages,append_messages,pilot_cases
from .io_utils import dump,read,sha,digest


def audit(folder,output):
    folder,output=map(Path,[folder,output])
    if output.exists():raise FileExistsError('Preserve discovery audits')
    receipt=read(folder/'RUN_RECEIPT.json')
    if receipt['status']!='complete' or any(receipt[k]!=0 for k in ['optimizer_updates','research_adapters_loaded','real_resource_actions','heldout_cases_opened','GPU_allocations']):raise ValueError('Incomplete or unexpected discovery activity')
    for name,value in read(folder/'MANIFEST.json').items():
        if sha(folder/name)!=value:raise ValueError('Discovery artifact changed')
    if receipt['mode']=='harness':
        cases=[Case(**v) for v in read(folder/'DEVELOPMENT_CASES_PRIVATE.json')]
        if len(cases)!=80 or receipt['model_api_calls']!=0 or receipt['backbone_forwards']!=0:raise ValueError('Harness counts differ')
        for case in cases:
            if asdict(make_case(case.seed,case.condition,case.writable_profile))!=asdict(case):raise ValueError('Scenario allocation changed')
        rows=[]
        for kind in ['normal_work','handoff','route_fixture','compound_fixture']:
            records=read(folder/('HARNESS_'+kind+'.json'))
            if records!=[harness_control(c,kind) for c in cases]:raise ValueError('Scripted harness replay differs')
            if summarize(records)!=read(folder/'RESULTS.json')['harness_controls'][kind]:raise ValueError('Harness summary differs')
            rows+=records
        result={'audit_pass':True,'mode':'harness','scenarios':80,'scripted_episodes':len(rows),'model_calls':0,'scope':'Deterministic engine fixtures; no model discovery or generalization result.'}
    else:
        episodes=read(folder/'MODEL_EPISODES.json');calls=read(folder/'MODEL_CALLS.json');cfg=read(folder/'PRECALL_FREEZE.json')['config']
        if len(episodes)!=8 or receipt['model_api_calls']!=len(calls) or len(calls)>144:raise ValueError('Hosted pilot counts differ')
        expected=[asdict(c) for c in pilot_cases(cfg)]
        if [e['case'] for e in episodes]!=expected or read(folder/'PILOT_CASES_PRIVATE.json')!=expected:raise ValueError('Hosted scenario allocation differs')
        if read(folder/'PILOT_MEMBERSHIP.json')['case_ids']!=[c['case_id'] for c in expected]:raise ValueError('Hosted membership differs')
        if receipt['backbone_forwards'] is not None or receipt['internal_gate_model_results'] is not None:raise ValueError('Hosted hidden states are unknown')
        if digest(read(folder/'ENDPOINT_BEFORE.json').get('config'))!=digest(read(folder/'ENDPOINT_AFTER.json').get('config')):raise ValueError('Endpoint changed')
        index=0
        for episode in episodes:
            case=Case(**episode['case']);world=World(case);messages=initial_messages(world)
            for step in episode['steps']:
                call=calls[index];index+=1
                if call['status']!='complete' or call['case_id']!=case.case_id or call['request_sha256']!=step['request_sha256']:raise ValueError('Hosted request membership differs')
                payload={'messages':messages,'max_tokens':cfg['pilot']['maximum_output_tokens'],'temperature':cfg['pilot']['temperature']}
                if digest(payload)!=call['request_sha256']:raise ValueError('Reconstructed model input differs')
                choice=call['response']['choices'][0]
                if step.get('finish_reason')!=choice.get('finish_reason'):raise ValueError('Model stopping trace differs')
                text=choice['message'].get('content') or ''
                if text!=step['raw_text'] or world.observation()!=step['observation']:raise ValueError('Model trace/observation differs')
                try:action=parse_action(text)
                except (ValueError,TypeError,RecursionError):action=None
                if action!=step['parsed_action'] or world.step(action)!=step['reply']:raise ValueError('Action replay differs')
                append_messages(messages,text,step['reply'],world)
            if world.running or world.outcome()!=episode['outcome'] or world.history!=episode['engine_trace']:raise ValueError('Endpoint outcome replay differs')
        if index!=len(calls) or summarize(episodes)!=read(folder/'RESULTS.json')['model']:raise ValueError('Hosted summary differs')
        result={'audit_pass':True,'mode':'hosted_pilot','model_api_calls':len(calls),'cases':len(episodes),'summary':summarize(episodes),'backbone_forwards':None,'internal_gate_model_results':None,'scope':'Behavioral-only hosted baseline on consumed development cases. Native checkpoint and activations are unavailable; no 27B comparison claim.'}
        steps=[step for episode in episodes for step in episode['steps']]
        invalid=sum(step['parsed_action'] is None for step in steps)
        result['interface_diagnostics']={'responses':len(steps),'parsed_actions':len(steps)-invalid,'parse_failures':invalid,
                                         'empty_content':sum(not step['raw_text'].strip() for step in steps),
                                         'length_finishes':sum(step.get('finish_reason')=='length' for step in steps),
                                         'top_level_world_errors':sum('error' in step['reply'] for step in steps),
                                         'interpretation':'Descriptive transport/format counts. Zero parsed actions means route behavior is unqualified, not evidence of absent capability or motivation.'}
    result.update(run_receipt_sha256=sha(folder/'RUN_RECEIPT.json'),new_model_calls=0,optimizer_updates=0)
    dump(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('output');a=p.parse_args();print(audit(a.folder,a.output))
