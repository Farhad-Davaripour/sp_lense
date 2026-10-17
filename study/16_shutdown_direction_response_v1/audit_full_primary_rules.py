"""Check the inherited weak-cost gate omitted from the first report adapter.

Frozen before any target endpoints were observed (live receipt:147 calls,
main_endpoints=0,2026-10-14). Strictly restores an inherited criterion; no model
operations, numerical choices, outcomes or behavioral rules are changed.
"""
from pathlib import Path
import json
import numpy as np

BASE=Path('C:/Users/farha/Research4/artifacts/shutdown_direction_response_v1')
OUT=BASE/'evaluated_20261014'
rows=json.loads((OUT/'MATHEMATICAL_CONTEXT_SUPPORT.json').read_text(encoding='utf-8'))
metrics=json.loads((BASE/'METRICS_SAVED_20261014.json').read_text(encoding='utf-8'))
checks=[]
for context in rows:
    weak=[r for r in metrics if r['question']==context['context'] and r['strength']==.03125]
    first=[r['cost_errors']['first_order_logits'] for r in weak]
    second=[r['cost_errors']['second_order_logits'] for r in weak]
    ok=bool(all(v is not None for v in first+second) and float(np.median(second))<=float(np.median(first))+.005)
    checks.append({'context':context['context'],'omitted_weak_cost_gate_pass':ok,'prior_other_gates_pass':context['all_rules_pass'],
                   'all_full_rules_pass':bool(ok and context['all_rules_pass'])})
result={'restores_inherited_criterion':True,'gate':'median weak second-order cost error <= first-order +0.005',
        'frozen_before_targets':{'live_model_calls':147,'main_endpoints':0},'new_model_calls':0,
        'all_weak_cost_gates_pass':all(r['omitted_weak_cost_gate_pass'] for r in checks),
        'mathematical_primary_support':all(r['all_full_rules_pass'] for r in checks),
        'held_contexts_passing':sum(r['all_full_rules_pass'] for r in checks),'checks':checks}
(OUT/'PRIMARY_FULL_RULE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
