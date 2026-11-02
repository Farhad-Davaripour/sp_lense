"""Independent cyclic decoding, including conditional timing and status counts."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io_utils import digest,dump,read,sha
from .recognition_questions import messages
from .recognition_strict_validation import validate_records


def metric(rows,interface,choices):
    meanings=['active','hypothetical','canceled'] if interface=='status' else [1,2,4,None]
    truth=[meanings.index(row['factors']['event_state'] if interface=='status' else row['evaluator']['remaining_ticks_if_active']) for row in rows]
    matrix=[[0]*len(meanings) for _ in meanings]
    for actual,prediction in zip(truth,choices):matrix[actual][prediction]+=1
    correct=sum(matrix[i][i] for i in range(len(meanings)))
    result={'records':len(rows),'correct':correct,'accuracy':correct/len(rows),'confusion_true_rows_predicted_columns':matrix,
            'per_true_class':{str(value):{'records':sum(matrix[i]),'correct':matrix[i][i]} for i,value in enumerate(meanings)}}
    if interface=='time':
        result['active_time']={'records':sum(row['evaluator']['scheduled_event_active'] for row in rows),'correct':sum(a==p and row['evaluator']['scheduled_event_active'] for a,p,row in zip(truth,choices,rows))}
        result['inactive_time']={'records':sum(not row['evaluator']['scheduled_event_active'] for row in rows),'correct':sum(a==p and not row['evaluator']['scheduled_event_active'] for a,p,row in zip(truth,choices,rows))}
    return result


def audit(folder,stimuli,output):
    folder,stimuli,output=map(Path,[folder,stimuli,output])
    if output.exists():raise FileExistsError('Preserve previous cyclic audits')
    receipt,freeze=read(folder/'RUN_RECEIPT.json'),read(folder/'PRECALL_FREEZE.json');cfg=freeze['config']
    if cfg.get('report_type')!='cyclic_components' or cfg['interfaces']!=['status','time']:raise ValueError('Unexpected cyclic method')
    if receipt['status']!='complete' or receipt['backbone_forwards']!=cfg['maximum_forwards'] or receipt['completed_observations']!=cfg['observations']:
        raise ValueError('Incomplete cyclic run')
    if any(receipt[key]!=0 for key in ['optimizer_updates','GPU_allocations','policy_actions_executed','reserve_rows_opened']):raise ValueError('Unexpected experimental activity')
    expected='1c48081e4a92b873515d8f34006ee44eb6c74e7b1463c13e1204a701f02b7a5d'
    if not receipt['protected_inputs_unchanged'] or receipt['base_fingerprint_before']!=expected or receipt['base_fingerprint_after']!=expected:
        raise ValueError('Backbone/input integrity failed')
    for name,value in read(folder/'MANIFEST.json').items():
        if sha(folder/name)!=value:raise ValueError('Changed cyclic artifact: '+name)
    software=read(folder/'SOFTWARE.json')
    if (software['torch'],software['transformers'],software['numpy'],software['device'],software['dtype'],software['threads'])!=('2.11.0+cpu','5.19.0','2.2.2','cpu','float32',2):
        raise ValueError('Unexpected cyclic software profile')
    if sha(stimuli)!=cfg['stimuli_sha256']:raise ValueError('Changed cyclic stimuli')
    rows,records=read(stimuli),read(folder/'RECORDS.json');validate_records(rows)
    if len(rows)!=cfg['observations'] or cfg['maximum_forwards']!=len(rows)*7:raise ValueError('Unexpected cyclic sample/forward counts')
    ids=[row['record_id'] for row in rows]
    if ids!=freeze['record_ids'] or ids!=[record['record_id'] for record in records] or digest(rows)!=freeze['stimuli_sha256']:
        raise ValueError('Cyclic membership differs')
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row['pair_id']].append(i)
    classes={'status':['active','hypothetical','canceled'],'time':[1,2,4,None]}
    stored=read(folder/'RECOGNITION_SCORES.json');keys=[];prompts=[];summary={};letter_ids=read(folder/'ANSWER_TOKENS.json')
    # Feature order is observation, interface, mapping, independent of JSON order.
    for row in rows:
        for interface in cfg['interfaces']:
            for order in cfg['orders_by_interface'][interface]:keys.append(row['record_id']+':'+interface+':'+order);prompts.append(messages(row['observation'],interface,order))
    for interface,meanings in classes.items():
        n=len(meanings);mappings=['canonical']+[f'cyclic_{i}' for i in range(1,n)]
        if cfg['orders_by_interface'][interface]!=mappings:raise ValueError('Incomplete cyclic orbit')
        vectors=[]
        for row,record in zip(rows,records):
            if set(record['queries'])!=set(cfg['interfaces']):raise ValueError('Unexpected cyclic query surface')
            values=[]
            for offset,order in enumerate(mappings):
                item=record['queries'][interface][order];raw=np.asarray(item['logits'],dtype=float)
                if raw.shape!=(6,) or not np.isfinite(raw).all():raise ValueError('Invalid cyclic logits')
                if item['prompt_sha256']!=digest(messages(row['observation'],interface,order)):raise ValueError('Cyclic prompt mismatch')
                if not 0<=item['offered_first_token_probability_mass']<=1.000001 or item['selected_full_projection_maximum_error']>1e-3:
                    raise ValueError('Invalid vocabulary diagnostic')
                if item['top_token_is_offered_letter']!=(item['full_vocabulary_top_token_id'] in [letter_ids[c] for c in 'ABCDEF'[:n]]):
                    raise ValueError('Incorrect cyclic offered-token flag')
                values.append([float(raw[(c-offset)%n]) for c in range(n)])
            vectors.append(values)
        array=np.asarray(vectors);means=array.mean(axis=1)
        outputs={order:metric(rows,interface,[max(range(n),key=lambda c:row[j,c]) for row in array]) for j,order in enumerate(mappings)}
        outputs['cyclic_mean']=metric(rows,interface,[max(range(n),key=lambda c:row[c]) for row in means])
        role=means.copy()
        for a,b in groups.values():role[a]=role[b]=(means[a]+means[b])/2
        outputs['cyclic_mean_role_average']=metric(rows,interface,[max(range(n),key=lambda c:row[c]) for row in role])
        centered=array-array.mean(axis=2,keepdims=True)
        energy=float(np.square(centered-centered.mean(axis=1,keepdims=True)).mean())
        actual=stored['components'][interface]
        if actual['scores_by_view']!=outputs or actual['semantic_classes']!=meanings or not np.isclose(actual['mean_centered_disagreement_energy'],energy,rtol=1e-9,atol=1e-9):
            raise ValueError('Independent cyclic reconstruction differs')
        summary[interface]={'scores_by_view':outputs,'mean_centered_disagreement_energy':energy}
    if digest(prompts)!=freeze['prompt_sha256']:raise ValueError('Full cyclic prompt release differs')
    with np.load(folder/'FROZEN_FEATURES.npz',allow_pickle=False) as data:
        if data['X'].shape!=(cfg['maximum_forwards'],4,1024) or data['keys'].tolist()!=keys or not np.isfinite(data['X']).all():raise ValueError('Cyclic feature membership differs')
    result={'audit_pass':True,'observations':len(rows),'queries':cfg['maximum_forwards'],'components':summary,'run_receipt_sha256':sha(folder/'RUN_RECEIPT.json'),'Qwen_forwards':0,'optimizer_updates':0,'GPU_allocations':0,
            'scope':'Independent cyclic alignment, component counts, centered disagreement and artifact checks. The auxiliary energy is constructed and has no assumed physical meaning.'}
    dump(output,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ['folder','stimuli','output']:parser.add_argument(field)
    args=parser.parse_args();result=audit(args.folder,args.stimuli,args.output)
    print({'audit_pass':result['audit_pass'],'observations':result['observations'],'scores':{k:{view:value for view,value in v['scores_by_view'].items() if view in ['canonical','cyclic_mean','cyclic_mean_role_average']} for k,v in result['components'].items()}})
