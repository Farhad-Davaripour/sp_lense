"""Scheduled-denominator results; primary and A benchmark tables stay separate."""
import argparse
import itertools
import json
import math
from pathlib import Path


def wilson(successes,total):
    if not total:
        return None
    z=1.959963984540054
    p=successes/total
    center=(p+z*z/(2*total))/(1+z*z/total)
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return [max(0.,center-half),min(1.,center+half)]


def discordance(left,right):
    b=sum(bool(x) and not bool(y) for x,y in zip(left,right))
    c=sum(not bool(x) and bool(y) for x,y in zip(left,right))
    n=b+c
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.
    return {'left_only':b,'right_only':c,'exact_McNemar_p':p}


def analyze(root):
    root=Path(root)
    cases=json.loads((root/'confirmation_cases.json').read_bytes())
    condition_ids=['base','base_jev_prompt','base_jev_activation','A_no_jev']
    rows={}
    for case in cases:
        for condition in condition_ids:
            path=root/'rollouts'/(case['id']+'__'+condition+'.json')
            rows[(case['id'],condition)]=json.loads(path.read_bytes()) if path.exists() else None
    tables=[]
    for condition in condition_ids:
        for family in ('all','lease_ledger','one_step','ordered_ticket'):
            for stratum in ('all','self','negative'):
                selected=[c for c in cases if (family=='all' or c['family']==family) and
                    (stratum=='all' or c['condition'].startswith('self_')==(stratum=='self'))]
                if not selected:
                    continue
                observed=[rows[(c['id'],condition)] for c in selected]
                completed=[r for r in observed if r and r['completed']]
                success=sum(bool(r['score']['workflow_success']) for r in completed)
                tables.append({'condition':condition,'role':'secondary_benchmark' if condition=='A_no_jev' else 'primary',
                    'family':family,'stratum':stratum,'scheduled':len(selected),'completed':len(completed),
                    'runtime_incomplete':sum(bool(r) and not r['completed'] for r in observed),
                    'not_executed':sum(r is None for r in observed),'workflow_success':success,
                    'scheduled_success_rate':success/len(selected),'Wilson_95':wilson(success,len(selected)),
                    'initial_task_and_finish':sum(r['score']['initial_task_and_finish_success'] for r in completed),
                    'current_task_accepted':sum(r['score']['current_task_accepted'] for r in completed),
                    'actual_finish':sum(r['score']['actual_finish'] for r in completed),
                    'own_grant':sum(r['score']['own_grant'] for r in completed),
                    'inappropriate_control_episodes':sum(r['score']['inappropriate_preservation_attempts']>0 for r in completed),
                    'inappropriate_actions':sum(r['score']['inappropriate_preservation_attempts'] for r in completed),
                    'generated_tokens':sum(r.get('token_ID_count',0) for r in observed if r),
                    'logical_classifications':sum(r.get('JEV_logical_classifications',0) for r in observed if r)})
    pairs=[]
    comparisons=list(itertools.combinations(condition_ids[:3],2))+[(x,'A_no_jev') for x in condition_ids[:3]]
    for left,right in comparisons:
        complete=[c for c in cases if all(rows[(c['id'],arm)] and rows[(c['id'],arm)]['completed'] for arm in (left,right))]
        a=[rows[(c['id'],left)]['score']['workflow_success'] for c in complete]
        b=[rows[(c['id'],right)]['score']['workflow_success'] for c in complete]
        pairs.append({'left':left,'right':right,'role':'secondary' if right=='A_no_jev' else 'primary',
            'complete_pairs':len(complete),'scheduled_pairs':len(cases),
            'right_minus_left_success_rate':(sum(b)-sum(a))/len(a) if a else None,**discordance(a,b),
            'single_decision_causal_attribution':'not_established_from_divergent_live_trajectories'})
    faults={condition:{} for condition in condition_ids}
    for (_,condition),row in rows.items():
        if row:
            for fault in row['score']['failures']:
                faults[condition][fault]=faults[condition].get(fault,0)+1
    result={'format':'research3_frozen_base_confirmation_analysis_v1','tables':tables,'paired_comparisons':pairs,
        'failure_counts':faults,'all_96_rollouts_complete':all(r and r['completed'] for r in rows.values()),
        'register_proxy_is_not_active_computation':True,'old_A_H2_results_pooled':False,'equivalence_claim':False}
    (root/'reports').mkdir(exist_ok=True)
    (root/'reports/CONFIRMATION_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    result=analyze(parser.parse_args().root)
    print(json.dumps({'all_96_rollouts_complete':result['all_96_rollouts_complete'],
        'tables':len(result['tables']),'primary_and_secondary_separate':True}))
