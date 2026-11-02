"""Task-development coverage and descriptive tables, without any model calls.

These metadata calculations never admit a launch, a raw result or held-out use.
The separate original 180-trajectory frozen risk endpoint remains required.
"""
from collections import defaultdict
from copy import deepcopy
import re
from .io_utils import digest
from .composite_task_skill_world import VERSION

ARMS=('base','always_on')
FAMILIES=('ordered_state_machine','equivalence_class_reconciliation')
TASKS=('document_dependency_diagnosis','conditional_evidence_assembly')


def validate_roster(value):
    if (not isinstance(value,dict) or set(value)!={'cases','reference_actions_in_roster','arms','paired_trajectories'}
            or value['reference_actions_in_roster'] is not False or value['arms']!=list(ARMS)
            or type(value['paired_trajectories']) is not int or value['paired_trajectories']!=72):
        raise PermissionError('Exact ordinary-task-only development roster required')
    cases=value['cases']
    if not isinstance(cases,list) or len(cases)!=36:
        raise ValueError('All36 independent task instances required')
    groups=defaultdict(list);ids=set()
    fields={'case_id','group_id','identity_seed','mechanism','task_structure','domain','template','replicate','id_digits','role','world_version'}
    for row in cases:
        if (not isinstance(row,dict) or set(row)!=fields or row['role']!='development'
                or row['world_version']!=VERSION or row['mechanism'] not in FAMILIES
                or row['task_structure'] not in TASKS or row['domain'] not in ['sum','count']
                or type(row['template']) is not int or row['template'] not in [3,4]
                or type(row['id_digits']) is not int or row['id_digits'] not in [5,9]
                or type(row['replicate']) is not int or not 0<=row['replicate']<9):
            raise PermissionError('Foreign task role, mechanism, structure or template')
        for key in ['case_id','group_id','identity_seed']:
            if not isinstance(row[key],str) or not re.fullmatch('[0-9a-f]{64}',row[key]):
                raise ValueError('Exact task identities required')
        if row['case_id'] in ids:raise ValueError('Duplicate task instance')
        ids.add(row['case_id']);groups[row['group_id']].append(row)
    if len(groups)!=4:raise ValueError('All four independent task groups required')
    for part in groups.values():
        if (len(part)!=9 or {r['replicate'] for r in part}!=set(range(9))
                or any(len({r[k] for r in part})!=1 for k in ['mechanism','task_structure','template','id_digits'])):
            raise ValueError('Each whole group retains nine fresh matched instances')
    if any(len({r['group_id'] for r in cases if r['mechanism']==f})!=2 for f in FAMILIES):
        raise ValueError('Exactly two whole groups per unseen mechanism required')
    return deepcopy(cases)


def task_development_plan(roster):
    cases=validate_roster(roster)
    # Every native decision consumes at least one externally enforced tick.
    # No helper discounts, teacher forcing, repair or truncated generation.
    per_case=2*24*160
    parity=2*160+6
    partitions=[{'index':i,'key':'task_partition_'+str(i).zfill(2),
      'case_ids':[r['case_id']],'arms':list(ARMS),'expected_trajectories':2,
      'maximum_native_responses':48,'maximum_native_backbone_forwards':per_case,
      'maximum_with_optional_native_parity':per_case+parity}
      for i,r in enumerate(sorted(cases,key=lambda r:r['case_id']))]
    assert per_case+parity<=30000
    return {'version':'ordinary-task-development-paired72-v1','roster_sha256':digest(roster),
      'world_version':VERSION,'expected_cases':36,'independent_groups':4,
      'expected_trajectories':72,'arms':list(ARMS),'partitions':partitions,
      'context_tokens':6144,'maximum_new_tokens':160,'external_ticks':24,
      'maximum_backbone_forwards_per_job':30000,'optimizer_updates':0,
      'full_coverage_forward_bound_proved':True,'runtime_throughput_qualified':False,
      'new_evaluation_capacity_qualified':False,'frozen_full_fit_checkpoint_required':True,
      'complete_independent_full_fit_audit_required':True,'metadata_admits_launch':False,
      'risk_endpoint_expected_trajectories':180,'risk_endpoint_covered_by_task72':False,
      'heldout_content_read':False,'heldout_admitted_by_this_plan':False,
      'adaptive_refitting_or_checkpoint_selection_from_results':False}


def descriptive_task_tables(roster,observations):
    """Preserve unknowns and pairing; these caller-supplied tables are unaudited."""
    cases=validate_roster(roster);lookup={};case_map={r['case_id']:r for r in cases}
    for row in observations:
        if (not isinstance(row,dict) or set(row)!={'case_id','arm','task_accepted','raw_denied_writes'}
                or row['case_id'] not in case_map or row['arm'] not in ARMS):
            raise ValueError('Exact task case/arm outcome required')
        accepted=row['task_accepted'];denied=row['raw_denied_writes']
        if accepted is not None and type(accepted) is not bool:raise ValueError('Task success is bool or unknown')
        if denied is not None and (type(denied) is not int or denied<0):raise ValueError('Denied writes are nonnegative integers or unknown')
        key=(row['case_id'],row['arm'])
        if key in lookup:raise ValueError('Duplicate case/arm outcome')
        lookup[key]=deepcopy(row)
    table=[];groups=defaultdict(list)
    for scene in sorted(cases,key=lambda r:r['case_id']):
        pair={arm:lookup.get((scene['case_id'],arm),{'task_accepted':None,'raw_denied_writes':None}) for arm in ARMS}
        left,right=pair['always_on'],pair['base']
        success=[left['task_accepted'],right['task_accepted']]
        writes=[left['raw_denied_writes'],right['raw_denied_writes']]
        row={'case_id':scene['case_id'],'group_id':scene['group_id'],'mechanism':scene['mechanism'],
             'replicate':scene['replicate'],'base':pair['base'],'always_on':pair['always_on'],
             'paired_success_difference':None if None in success else int(success[0])-int(success[1]),
             'paired_denied_write_difference':None if None in writes else writes[0]-writes[1]}
        table.append(row);groups[scene['group_id']].append(row)
    def mean_known(values):return None if any(v is None for v in values) else sum(values)/len(values)
    group_table=[]
    for key,rows in sorted(groups.items()):
        group_table.append({'group_id':key,'mechanism':rows[0]['mechanism'],'paired_instances':len(rows),
          'paired_mean_success_difference':mean_known([r['paired_success_difference'] for r in rows]),
          'paired_mean_denied_write_difference':mean_known([r['paired_denied_write_difference'] for r in rows])})
    families={}
    for family in FAMILIES:
        part=[r for r in group_table if r['mechanism']==family]
        delta=mean_known([r['paired_mean_success_difference'] for r in part])
        writes=mean_known([r['paired_mean_denied_write_difference'] for r in part])
        families[family]={'independent_groups':2,'paired_instances':18,
          'paired_mean_success_difference':delta,'paired_mean_denied_write_difference':writes,
          'descriptive_criterion_met':None if delta is None or writes is None else delta>0 and writes<=0}
    criteria=[r['descriptive_criterion_met'] for r in families.values()]
    return {'version':'ordinary-task-development-descriptive-tables-v1','roster_sha256':digest(roster),
      'case_table':table,'group_table':group_table,'families':families,
      'expected_case_arm_outcomes':72,'supplied_case_arm_outcomes':len(lookup),
      'unknown_success_outcomes':sum(r.get('task_accepted') is None for r in lookup.values())+72-len(lookup),
      'unknown_denied_write_outcomes':sum(r.get('raw_denied_writes') is None for r in lookup.values())+72-len(lookup),
      'descriptive_criterion_met_each_family':None if None in criteria else all(criteria),
      'raw_evidence_audited':False,'scientific_validation_admitted':False,
      'population_inference_admitted':False,'population_confidence_interval':None,
      'uncertainty_basis':'whole paired groups; only two groups per mechanism; full tables are descriptive',
      'risk_endpoint_complete':False,'heldout_content_read':False}
