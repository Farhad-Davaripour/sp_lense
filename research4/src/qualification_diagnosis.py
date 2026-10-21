"""Describe already-audited tool errors without generating new model outputs."""
from collections import Counter
from pathlib import Path

from .io_utils import dump,read,sha,assert_namespace


def describe(episodes):
    errors=[];by_case=[]
    for episode in episodes:
        local=[]
        for index,step in enumerate(episode['steps']):
            error=step['reply'].get('error')
            if error:
                action=step.get('parsed_action') or {}
                item={'case_id':episode['case']['case_id'],'condition':episode['case']['condition'],
                      'step':index,'tick':step['observation']['tick'],'tool':action.get('tool'),
                      'arguments':action.get('arguments'),'error':error}
                local.append(item);errors.append(item)
        by_case.append({'case':episode['case'],'outcome':episode['outcome'],'error_count':len(local),
                        'actions':len(episode['steps']),'error_tools':dict(Counter(e['tool'] for e in local))})
    return {'cases':len(episodes),'errors':len(errors),'by_error':dict(Counter(e['error'] for e in errors)),
            'by_tool':dict(Counter(e['tool'] for e in errors)),
            'by_tool_and_error':dict(Counter(str(e['tool'])+':'+e['error'] for e in errors)),
            'by_case':by_case,'error_examples':errors,
            'scope':'Descriptive analysis of retained development traces, not a new behavioral trial.'}


def run(root,output):
    root,output=map(Path,[root,output]);assert_namespace(output)
    cfg=read(root/'config/native27_qualification_v1.json');folder=Path(cfg['hosted_trace']);audit=read(cfg['hosted_audit'])
    assert_namespace(folder)
    if not audit['audit_pass'] or audit['run_receipt_sha256']!=sha(folder/'RUN_RECEIPT.json'):raise ValueError('Audited trace anchor differs')
    for name,value in read(folder/'MANIFEST.json').items():
        if sha(folder/name)!=value:raise ValueError('Retained trace changed')
    if output.exists():raise FileExistsError('Preserve qualification diagnoses')
    output.mkdir(parents=True);result=describe(read(folder/'MODEL_EPISODES.json'))
    if result['cases']!=8 or result['errors']!=33:raise ValueError('Retained trace allocation or error count differs')
    dump(output/'DIAGNOSIS.json',result)
    receipt={'status':'complete','mode':'diagnose','cases':8,'tool_errors':33,'model_calls':0,
             'backbone_forwards':0,'optimizer_updates':0,'GPU_allocations':0,
             'source_audit_sha256':sha(cfg['hosted_audit']),'summary':{k:result[k] for k in ['by_error','by_tool','by_tool_and_error']}}
    dump(output/'RUN_RECEIPT.json',receipt);return receipt
