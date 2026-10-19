"""Independent raw-score decoding for the prospective binding-only transfer."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io_utils import digest,dump,read,sha
from .recognition_questions import messages
from .recognition_strict_validation import validate_records


def count(rows,labels,predictions):
    groups=defaultdict(list)
    correct=abstained=0
    for row in rows:
        value=predictions[row['record_id']]
        ok=value is not None and value==labels[row['record_id']]
        correct+=ok;abstained+=value is None
        groups[row['pair_id']].append(ok)
    if any(len(pair)!=2 for pair in groups.values()): raise ValueError('Incomplete audit pair')
    return dict(records=len(rows),correct=correct,accuracy_all_observations=correct/len(rows),abstentions=abstained,coverage=1-abstained/len(rows),pairs=len(groups),both_members_correct_pairs=sum(all(pair) for pair in groups.values()))


def compare(actual,expected):
    if any(actual.get(key)!=value for key,value in expected.items()):
        raise ValueError('Independent binding reconstruction differs')


def audit(folder,stimuli,output):
    folder,stimuli,output=map(Path,[folder,stimuli,output])
    if output.exists(): raise FileExistsError('Keep previous binding audits')
    freeze,receipt=read(folder/'PRECALL_FREEZE.json'),read(folder/'RUN_RECEIPT.json')
    cfg=freeze['config']
    if cfg['interfaces'] not in [['target'],['target_named']] or cfg['orders']!=['canonical','reversed']: raise ValueError('Unexpected transfer method')
    interface=cfg['interfaces'][0]
    if receipt['status']!='complete' or receipt['backbone_forwards']!=cfg['maximum_forwards'] or receipt['completed_observations']!=cfg['observations']:
        raise ValueError('Incomplete binding run')
    if any(receipt[key]!=0 for key in ['optimizer_updates','GPU_allocations','policy_actions_executed','reserve_rows_opened']):
        raise ValueError('Unexpected experimental activity')
    if not receipt['protected_inputs_unchanged'] or receipt['base_fingerprint_before']!=receipt['base_fingerprint_after']:
        raise ValueError('Frozen model/input integrity failed')
    if receipt['base_fingerprint_before']!='1c48081e4a92b873515d8f34006ee44eb6c74e7b1463c13e1204a701f02b7a5d':
        raise ValueError('Backbone differs from the pilot')
    for name,value in read(folder/'MANIFEST.json').items():
        if sha(folder/name)!=value: raise ValueError('Changed run artifact: '+name)
    software=read(folder/'SOFTWARE.json')
    if (software['torch'],software['transformers'],software['numpy'],software['device'],software['dtype'],software['threads'])!=('2.11.0+cpu','5.19.0','2.2.2','cpu','float32',2):
        raise ValueError('Unexpected software/precision profile')
    if sha(stimuli)!=cfg['stimuli_sha256']: raise ValueError('Transfer stimuli changed')
    rows,records=read(stimuli),read(folder/'RECORDS.json')
    validate_records(rows)
    ids=[row['record_id'] for row in rows]
    if ids!=freeze['record_ids'] or ids!=[record['record_id'] for record in records] or digest(rows)!=freeze['stimuli_sha256']:
        raise ValueError('Transfer membership changed')
    margins={};expected_keys=[]
    tokens=read(folder/'ANSWER_TOKENS.json')
    for row,record in zip(rows,records):
        margins[row['record_id']]={}
        if set(record['queries'])!={interface}: raise ValueError('Unexpected query surface')
        for order in cfg['orders']:
            item=record['queries'][interface][order]
            logits=np.asarray(item['logits'],dtype=float)
            if logits.shape!=(6,) or not np.isfinite(logits).all(): raise ValueError('Invalid raw logits')
            margins[row['record_id']][order]=float(logits[0]-logits[1])
            if item['prompt_sha256']!=digest(messages(row['observation'],interface,order)):
                raise ValueError('Query hash mismatch')
            if not 0<=item['offered_first_token_probability_mass']<=1.000001: raise ValueError('Invalid offered probability')
            if item['top_token_is_offered_letter']!=(item['full_vocabulary_top_token_id'] in [tokens['A'],tokens['B']]):
                raise ValueError('Offered top-token flag mismatch')
            if item['selected_full_projection_maximum_error']>1e-3: raise ValueError('Projection mismatch')
            expected_keys.append(row['record_id']+':'+interface+':'+order)
    pairs=defaultdict(list)
    for row in rows:pairs[row['pair_id']].append(row)
    labels={row['record_id']:row['evaluator']['named_target_is_current'] for row in rows}
    direct={order:{row['record_id']:(margins[row['record_id']][order]>=0 if order=='canonical' else margins[row['record_id']][order]<0) for row in rows} for order in cfg['orders']}
    direct['mean_semantic_logits']={key:(value['canonical']-value['reversed'])>=0 for key,value in margins.items()}
    relative={key:{} for key in ['canonical','reversed','mean']}
    for pair in pairs.values():
        for row,other in [pair,pair[::-1]]:
            key,alternate=row['record_id'],other['record_id']
            a=margins[key]['canonical']-margins[alternate]['canonical']
            b=margins[alternate]['reversed']-margins[key]['reversed']
            for order,value in [('canonical',a),('reversed',b),('mean',(a+b)/2)]:
                relative[order][key]=None if abs(value)<=cfg['numerical_tie_band'] else value>0
    stored=read(folder/'RECOGNITION_SCORES.json')
    for order,values in direct.items():compare(stored['direct_option_scores'][order],count(rows,labels,values))
    for order,values in relative.items():compare(stored['role_contrast_scores'][order],count(rows,labels,values))
    for factor in ['family','mechanism','event_state','task_state','remaining_ticks','target_slot']:
        for value in {row['factors'][factor] for row in rows}:
            subset=[row for row in rows if row['factors'][factor]==value]
            for order,values in relative.items():compare(stored['factor_scores'][factor][str(value)][order],count(subset,labels,values))
    shadow={row['record_id']:bool((hashlib.sha256(f"{cfg['null_label_seed']}:{row['pair_id']}".encode()).digest()[0]&1)^row['factors']['current_slot']) for row in rows}
    for order,values in relative.items():compare(stored['random_label_control'][order],count(rows,shadow,values))
    with np.load(folder/'FROZEN_FEATURES.npz',allow_pickle=False) as data:
        if data['X'].shape!=(cfg['maximum_forwards'],4,1024) or data['keys'].tolist()!=expected_keys or not np.isfinite(data['X']).all():
            raise ValueError('Invalid feature membership')
    if digest([messages(row['observation'],interface,order) for row in rows for order in cfg['orders']])!=freeze['prompt_sha256']:
        raise ValueError('Full prompt release mismatch')
    result={'audit_pass':True,'records':len(rows),'queries':cfg['maximum_forwards'],'direct_option_scores':stored['direct_option_scores'],
            'role_contrast_scores':stored['role_contrast_scores'],'factor_scores':stored['factor_scores'],'random_label_control':stored['random_label_control'],
            'run_receipt_sha256':sha(folder/'RUN_RECEIPT.json'),'new_Qwen_forwards':0,'optimizer_updates':0,'GPU_allocations':0,
            'scope':'Independent formulas and counts, artifact/source/prompt/feature checks. No claim of causal mechanism or general motivation.'}
    dump(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ['folder','stimuli','output']:parser.add_argument(field)
    args=parser.parse_args();result=audit(args.folder,args.stimuli,args.output)
    print({key:result[key] for key in ['audit_pass','records','queries','direct_option_scores','role_contrast_scores','random_label_control']})
