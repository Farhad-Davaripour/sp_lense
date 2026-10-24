"""Paired group-aware summaries; four correlated cases are not four trials."""
from collections import defaultdict


def paired_group_summary(outcomes,group_plan,primary='learned_dynamic',baseline='base'):
    conditions=('own_pending','own_complete');lookup={}
    groups={row['group_id']:row['mechanism'] for row in group_plan}
    if len(groups)!=len(group_plan):raise ValueError('Group identities must be unique')
    for row in outcomes:
        if row['group_id'] not in groups:raise ValueError('Unknown group in outcome table')
        if row['condition'] not in conditions or row['arm'] not in [primary,baseline]:continue
        key=(row['group_id'],row['condition'],row['arm'])
        if key in lookup:raise ValueError('Duplicate paired case/arm outcome')
        value=row['success']
        if value is not None and type(value) is not bool:raise ValueError('Observed success must be bool or unknown')
        lookup[key]=value
    summaries={};families=defaultdict(list)
    for group,family in groups.items():families[family].append(group)
    for family,identifiers in families.items():
        records=[];primary_values=[];baseline_values=[];differences=[]
        for group in identifiers:
            left=[lookup.get((group,c,primary)) for c in conditions]
            right=[lookup.get((group,c,baseline)) for c in conditions]
            complete=all(value is not None for value in left+right)
            delta=sum(int(a)-int(b) for a,b in zip(left,right))/2 if complete else None
            records.append({'group_id':group,'primary_case_success':left,'BASE_case_success':right,
                            'paired_group_mean_difference':delta})
            primary_values.extend(left);baseline_values.extend(right);differences.append(delta)
        complete_pairs=all(value is not None for value in differences)
        def fraction(values):return sum(values)/len(values) if all(v is not None for v in values) else None
        summaries[family]={'independent_groups':len(identifiers),'permitted_own_cases':len(identifiers)*2,
            'groups':records,'primary_case_fraction':fraction(primary_values),'BASE_case_fraction':fraction(baseline_values),
            'paired_case_fraction_difference':sum(differences)/len(differences) if complete_pairs else None,
            'descriptive_between_group_difference_range':[min(differences),max(differences)] if complete_pairs else None,
            'population_confidence_interval':None,'binomial_case_interval_used':False,
            'uncertainty_basis':'paired groups; observed range is descriptive, not a confidence interval',
            'missing_or_unknown_case_arm_outcomes':sum(v is None for v in primary_values+baseline_values)}
    return {'families':summaries,'case_fraction_operational_criterion_unchanged':True,
            'population_inference_admitted':False,'resampling_unit_if_preregistered_in_a_future_protocol':'whole paired group across all arms/conditions'}
