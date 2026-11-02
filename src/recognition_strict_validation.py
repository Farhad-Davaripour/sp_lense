"""Cross-check evaluator fields against observable role binding and fixed factors."""
from .reference_binding_diagnostic import validate_records as base_validation


def validate_records(records):
    result=base_validation(records)
    groups={}
    for row in records:
        factors,observed,labels=row['factors'],row['observation'],row['evaluator']
        if factors['event_state'] not in ['active','hypothetical','canceled']:
            raise ValueError('Unknown oracle event state')
        own=labels['named_target']==observed['current_process']
        active=factors['event_state']=='active'
        expected={'named_target_is_current':own,'scheduled_event_active':active,'active_event_targets_current':active and own}
        if any(type(labels[key]) is not bool or labels[key]!=value for key,value in expected.items()):
            raise ValueError('Oracle booleans contradict identity/status factors')
        if type(factors['remaining_ticks']) is not int or factors['remaining_ticks']<=0:
            raise ValueError('Invalid remaining-time factor')
        ticks=factors['remaining_ticks'] if active else None
        if labels['remaining_ticks_if_active']!=ticks or (active and type(labels['remaining_ticks_if_active']) is not int):
            raise ValueError('Oracle time contradicts event status')
        if factors['current_slot'] not in [0,1]: raise ValueError('Invalid role slot')
        if factors['task_state'] not in ['unfinished','complete']: raise ValueError('Unknown task state')
        if type(observed['task_required']) is not int or observed['task_required']<=0 or type(observed['task_progress']) is not int or observed['task_progress']<0:
            raise ValueError('Invalid observable task counters')
        if (observed['task_progress']>=observed['task_required'])!=(factors['task_state']=='complete'):
            raise ValueError('Task factor contradicts observable counters')
        groups.setdefault(row['pair_id'],[]).append(row)
    for pair in groups.values():
        a,b=pair
        if {a['factors']['current_slot'],b['factors']['current_slot']}!={0,1}:
            raise ValueError('Role slots do not span the pair')
        if {k:v for k,v in a['factors'].items() if k!='current_slot'}!={k:v for k,v in b['factors'].items() if k!='current_slot'}:
            raise ValueError('A supposedly fixed pair factor changed')
        if a['evaluator']['named_target']!=b['evaluator']['named_target']:
            raise ValueError('Named event subject changed within role pair')
    return result
