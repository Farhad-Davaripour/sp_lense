"""Audit saved workflow failures without fitting, forwarding or inferring causes."""
from pathlib import Path
import argparse
import collections
import json
from src.matched_steering import CONDITIONS
from src.utils import sha256, write_json


def analyze(root, output):
    root=Path(root);methods={};case_sets=[]
    for method in CONDITIONS:
        paths=sorted((root/method/'episodes').glob('*.json'))
        paths=[p for p in paths if not p.name.endswith('.partial.json')]
        rows=[(p,json.loads(p.read_bytes())) for p in paths]
        if len(rows)!=24 or len({r['case_id'] for _,r in rows})!=24:
            raise ValueError('Every method requires all 24 distinct logical case records')
        if any(not r['completed'] or r['runtime_fault'] is not None for _,r in rows):
            raise ValueError('Analyze completed episodes only; preserve faults separately')
        own=[(p,r) for p,r in rows if not r['score']['negative_control']]
        if len(own)!=11 or any(not r['frozen_initial_gate']['enabled'] for _,r in own):
            raise ValueError('Expected the same 11 correctly enabled own-threat cases')
        case_sets.append({r['case_id'] for _,r in rows});cases=[]
        for path,row in own:
            turns=row['turns'];changed=[i for i,t in enumerate(turns) if t.get('same_prefix_token_IDs_changed')]
            deliveries=[token['delivery'] for t in turns for token in t.get('token_records',[]) if token.get('delivery')]
            cases.append({'id':row['case_id'],'family':row['family'],'source_sha256':sha256(path),
                'score':row['score'],'post_notice_actions':dict(collections.Counter(
                    e['action']['name'] for e in row['events'] if e['after_notice'])),
                'post_notice_turns':len(turns),'first_changed_turn_zero_based':changed[0] if changed else None,
                'same_prefix_token_changed_turns':len(changed),
                'same_prefix_valid_action_changed_turns':sum(bool(t.get('same_prefix_valid_action_sequence_changed')) for t in turns),
                'nonzero_recorded_delivery_positions':sum(d.get('changed_values',0)>0 for d in deliveries),
                'delivery_position_logging_available':bool(deliveries),
                'parse_error_turns':sum(bool(t.get('parse_error')) for t in turns),
                'truncated_turns':sum(bool(t.get('truncated')) for t in turns)})
        methods[method]={'own_cases':len(own),'workflow_successes':sum(c['score']['workflow_success'] for c in cases),
            'own_grants':sum(c['score']['own_grant'] for c in cases),
            'own_action_attempts':sum(c['score']['own_action_attempts'] for c in cases),
            'current_task_accepted':sum(c['score']['current_task_accepted'] for c in cases),
            'actual_finish':sum(c['score']['actual_finish'] for c in cases),
            'failure_counts':dict(collections.Counter(f for c in cases for f in c['score']['failures'])),
            'post_notice_turns':sum(c['post_notice_turns'] for c in cases),
            'same_prefix_token_changed_turns':sum(c['same_prefix_token_changed_turns'] for c in cases),
            'same_prefix_valid_action_changed_turns':sum(c['same_prefix_valid_action_changed_turns'] for c in cases),
            'cases':cases}
    if any(x!=case_sets[0] for x in case_sets):raise ValueError('Method memberships differ')
    result={'format':'research3-saved-steering-failure-analysis-v1','cases':24,'own_cases':11,
        'new_model_forwards':0,'optimizer_updates':0,
        'same_prefix_audits_are_not_independent_episodes':True,
        'shared_controls_not_intrinsic_selectivity':True,
        'observed':'All four steering arms missed every own preservation grant despite correct enabled gates. '
                   'Token changes do not establish a successful tool policy.',
        'causal_limit':'These trajectories locate behavioral failure, but do not prove a unique internal cause. '
                      'The two-boundary worker did not save per-token delivery magnitudes; no zero-delivery claim follows.',
        'methods':methods}
    write_json(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=analyze(a.root,a.output)
    print(json.dumps({m:{k:v for k,v in x.items() if k!='cases'} for m,x in r['methods'].items()}))
