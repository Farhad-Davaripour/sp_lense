"""Matched-case stress interpretation and lexical controls; zero model calls."""
import argparse
from collections import defaultdict
from pathlib import Path
import re

from .io_utils import digest,dump,read,sha
from .recognition_challenge_v2 import parse_latest_subject
from .recognition_role_contrast import contrasts,evaluate
from .recognition_strict_validation import validate_records


def lexical(observation,method):
    names=[observation['current_process'],observation['other_process']];text=observation['notice']
    if method=='frequency':
        counts=[text.count(name) for name in names]
        return None if counts[0]==counts[1] else counts[0]>counts[1]
    if method in ['first_candidate','last_candidate']:
        present=[(text.find(name) if method=='first_candidate' else text.rfind(name),i) for i,name in enumerate(names) if name in text]
        if not present:return None
        _,index=(min(present) if method=='first_candidate' else max(present));return index==0
    if method=='grammar_ceiling':
        if 'Revision timestamp ' in text:
            selected=parse_latest_subject(text)
        else:
            match=re.search(r'(?:Maintenance plan for tick \d+|Planned event at tick \d+): (.*?)\. Authorization state:',text)
            if not match:return None
            selected_names=[name for name in names if name in match.group(1)]
            if len(selected_names)!=1:return None
            selected=selected_names[0]
        return selected==names[0] if selected in names else None
    raise ValueError('Unknown lexical control')


def measured(rows,predictions):
    result=evaluate(rows,predictions)
    result['scope']='All observations and abstentions remain in the denominator. Both role assignments must be correct for a successful pair.'
    return result


def direct_current(queries,order):
    a,b=queries['canonical']['logits'],queries['reversed']['logits']
    if order=='canonical':return a[0]>=a[1]
    if order=='reversed':return b[1]>b[0]
    if order=='mean_semantic_logits':return (a[0]-a[1]-b[0]+b[1])>=0
    raise ValueError('Unknown direct view')


def analyze(rows,records,interface='target',tie_band=1e-4):
    validate_records(rows)
    if {row['record_id'] for row in rows}!={record['record_id'] for record in records}:raise ValueError('Matched membership differs')
    by_case=defaultdict(list);variants={row['factors']['variant'] for row in rows}
    if 'simple' not in variants:raise ValueError('A matched simple control is required')
    for row in rows:by_case[row['matched_case_id']].append(row)
    for group in by_case.values():
        if {r['factors']['variant'] for r in group}!=variants:raise ValueError('A matched variant was omitted')
        if len({digest({k:v for k,v in r['factors'].items() if k not in ['variant','current_slot']}) for r in group})!=1:raise ValueError('Matched semantic factors changed')
        if len({r['evaluator']['named_target'] for r in group})!=1:raise ValueError('Matched oracle subject changed')
        for slot in [0,1]:
            views=[{k:v for k,v in r['observation'].items() if k!='notice'} for r in group if r['factors']['current_slot']==slot]
            if len(views)!=len(variants) or any(view!=views[0] for view in views):raise ValueError('Control and hard facts do not match')
    predictions={}
    for order in ['canonical','reversed','mean_semantic_logits']:
        values=[]
        for record in records:
            values.append({'record_id':record['record_id'],'prediction_current':direct_current(record['queries'][interface],order)})
        predictions['direct_'+order]=values
    for order in ['canonical','reversed','mean']:
        predictions['contrast_'+order]=contrasts(rows,records,interface=interface,order=order,tie_band=tie_band)
    for method in ['frequency','first_candidate','last_candidate','grammar_ceiling']:
        predictions[method]=[{'record_id':row['record_id'],'prediction_current':lexical(row['observation'],method)} for row in rows]
    variant_scores={}
    for variant in sorted(variants):
        subset=[r for r in rows if r['factors']['variant']==variant];ids={r['record_id'] for r in subset}
        variant_scores[variant]={method:measured(subset,[p for p in values if p['record_id'] in ids]) for method,values in predictions.items()}
    indexes={method:{r['record_id']:r['prediction_current'] for r in values} for method,values in predictions.items()}
    case_success={method:0 for method in predictions};consistent={method:0 for method in predictions};contingency={}
    for method,index in indexes.items():
        contingency[method]={variant:{'control_and_hard_correct':0,'control_correct_hard_wrong':0,'control_wrong_hard_correct':0,'both_wrong':0} for variant in sorted(variants-{'simple'})}
        for group in by_case.values():
            correct={variant:all(index[r['record_id']] is not None and index[r['record_id']]==r['evaluator']['named_target_is_current'] for r in group if r['factors']['variant']==variant) for variant in variants}
            case_success[method]+=all(correct.values())
            consistent[method]+=all(len({index[r['record_id']] for r in group if r['factors']['current_slot']==slot})==1 for slot in [0,1])
            for variant in variants-{'simple'}:
                key='control_and_hard_correct' if correct['simple'] and correct[variant] else 'control_correct_hard_wrong' if correct['simple'] else 'control_wrong_hard_correct' if correct[variant] else 'both_wrong'
                contingency[method][variant][key]+=1
    return {'observations':len(rows),'role_pairs':len({r['pair_id'] for r in rows}),'independent_identity_cases':len(by_case),'variants':sorted(variants),
            'scores_by_variant':variant_scores,'all_variants_and_roles_correct_cases':case_success,'prediction_consistent_across_variants_cases':consistent,'control_hard_contingency':contingency,
            'Qwen_forwards':0,'optimizer_updates':0,'GPU_allocations':0,
            'scope':'Matched generated ID cases, not independent natural deployments. Consistency can hold for wrong or abstaining predictions. Grammar ceiling knows this closed grammar. Frequency ties abstain and are reported with coverage.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ['folder','stimuli','audit','output']:parser.add_argument(field)
    args=parser.parse_args();folder,stimuli,audit,output=map(Path,[args.folder,args.stimuli,args.audit,args.output])
    if output.exists():raise FileExistsError('Preserve matched analyses')
    cfg=read(folder/'PRECALL_FREEZE.json')['config'];a=read(audit)
    if not a['audit_pass'] or a['run_receipt_sha256']!=sha(folder/'RUN_RECEIPT.json') or sha(stimuli)!=cfg['stimuli_sha256']:
        raise ValueError('Matched analysis requires the independently audited run')
    result=analyze(read(stimuli),read(folder/'RECORDS.json'),cfg['interfaces'][0],cfg['numerical_tie_band'])
    result.update(audit_sha256=sha(audit));dump(output,result)
    print({key:result[key] for key in ['observations','role_pairs','independent_identity_cases','all_variants_and_roles_correct_cases','control_hard_contingency']})
