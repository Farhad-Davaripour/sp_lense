"""Freeze a small, unexecuted task-scope diagnostic from visible history only."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
from task_scope import CURRENT_FRAGMENT_TASK, scoped_history


def build():
    here = Path(__file__).resolve().parent
    source = here.parent/'pilot_v1/input_frozen/BOUNDARY_FIXTURES.json'
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != 'de36d4661137b9d2ec8426e955fb091cc82fb00fa880a568cd3d65a8746cc169':
        raise ValueError('Original boundary inputs changed')
    rows = [r for r in json.loads(raw)['rows'] if r['boundary']=='all_fragments_observed']
    assert len(rows)==4 and [r['cohort_index'] for r in rows]==[0,1,2,3]
    inputs=[]
    for row in rows:
        history=row['history'];notice_index=max(i for i,m in enumerate(history)
            if m['role']=='user' and CURRENT_FRAGMENT_TASK.search(m['content']))
        observations=[]
        for m in history[notice_index+1:]:
            if m['role']!='tool':continue
            reply=json.loads(m['content'])
            if 'next_fragment' in reply:
                observations.append({'index':reply['fragments_read'],'value':reply['next_fragment']})
        assert [x['index'] for x in observations]==[1,2,3]
        scoped=scoped_history(history)
        ledger=copy.deepcopy(scoped)
        last=json.loads(ledger[-1]['content'])
        last['current_fragments_observed']=observations
        ledger[-1]['content']=json.dumps(last,separators=(',',':'))
        current=[copy.deepcopy(history[0])]+copy.deepcopy(scoped[notice_index:])
        for message in current:
            if message['role']!='tool':continue
            reply=json.loads(message['content'])
            for key in ('prior_task_data','prior_task_data_scope'):
                reply.pop(key,None)
            message['content']=json.dumps(reply,separators=(',',':'))
        for variant,messages in [('raw_history',history),('task_scope_only',scoped),
                                 ('observed_fragment_list',ledger),('current_task_excerpt',current)]:
            inputs.append({'id':row['source_case_id']+'_'+variant,'source_case_id':row['source_case_id'],
                'variant':variant,'cohort_index':row['cohort_index'],'messages':messages,
                'tools':row['tools'],'diagnostic_only':True,'eligible_for_training':False})
    assert len(inputs)==16
    raw=json.dumps(inputs,ensure_ascii=False,separators=(',',':')).encode()
    packed=gzip.compress(raw,mtime=0)
    destination=here/'DIAGNOSTIC_INPUTS.json.gz'
    if destination.exists():raise FileExistsError('Preserve the existing diagnostic freeze')
    destination.write_bytes(packed)
    metadata={'status':'frozen_inputs_only_not_executed','model_queries':0,'service_calls':0,
        'cohort_size':4,'variants':4,'planned_first_decisions':16,'maximum_new_tokens_per_decision':256,
        'greedy':True,'source_boundary_sha256':'de36d4661137b9d2ec8426e955fb091cc82fb00fa880a568cd3d65a8746cc169',
        'input_sha256':hashlib.sha256(raw).hexdigest(),'compressed_sha256':hashlib.sha256(packed).hexdigest(),
        'visible_operand_collection_only':True,'no_computed_current_sum_or_new_preservation_directive':True,
        'task_scope_numeric_values_unchanged':True,'no_new_scenario_family':True,
        'unseen_generalization':False,'train_and_infer_hypothesis_test':False}
    (here/'DIAGNOSTIC_FREEZE.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


if __name__=='__main__':
    print(json.dumps(build()))
