"""Fixed execution grouping of all task72 pairs; no scientific/model calls.

Original case, arm and inferential-group identities remain unchanged. The first
useful paired case runs alone; later jobs amortize at most three whole pairs.
"""
from .io_utils import digest
from .composite_task_skill_validation import task_development_plan


def execution_plan(roster):
    original=task_development_plan(roster)
    pairs=original['partitions']
    groups=[pairs[:1]]+[pairs[i:i+3] for i in range(1,len(pairs),3)]
    parts=[]
    for index,group in enumerate(groups):
        case_ids=[p['case_ids'][0] for p in group]
        forwards=len(group)*2*24*160
        parts.append({'index':index,'key':'task_execution_'+str(index).zfill(2),'case_ids':case_ids,
          'original_case_partition_indices':[p['index'] for p in group],'arms':['base','always_on'],
          'expected_trajectories':2*len(group),'maximum_native_responses':48*len(group),
          'maximum_native_backbone_forwards':forwards,'maximum_with_optional_native_parity':forwards+326})
    ids=[c for p in parts for c in p['case_ids']]
    if (len(parts)!=13 or len(ids)!=36 or len(set(ids))!=36
            or ids!=[p['case_ids'][0] for p in pairs]
            or sum(p['expected_trajectories'] for p in parts)!=72
            or any(p['maximum_with_optional_native_parity']>30000 for p in parts)):
        raise PermissionError('Exact complete one-plus-triples task72 execution coverage required')
    return {'version':'ordinary-task-paired-execution13-v2','original_task_plan_sha256':digest(original),
      'original_case_pairs':36,'expected_trajectories':72,'partitions':parts,
      'grouping_basis':'fixed sorted frozen case IDs before any native validation outcomes',
      'context_tokens':6144,'maximum_new_tokens':160,'external_ticks':24,
      'maximum_backbone_forwards_per_job':30000,'maximum_concurrent_owned_workers':1,
      'optimizer_updates':0,'automatic_scientific_retries':0,
      'all_original_cases_arms_and_inferential_groups_unchanged':True,
      'runtime_throughput_qualified':False,'metadata_admits_launch':False,'heldout_content_read':False}
