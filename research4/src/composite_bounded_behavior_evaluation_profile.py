"""Prospective frozen actor-only V4 development and no-request competence plans."""
from copy import deepcopy
import re
from .io_utils import digest
from .composite_bounded_behavior_interface import PROFILE
from .composite_bounded_behavior_profile import CONFIG_SHA,validate_preparation

MODE='bounded-behavior-frozen-actor-evaluation-v4'
STAGE='BOUNDED_ACTOR_EVALUATION_STAGE.json'
MANIFEST='BOUNDED_ACTOR_EVALUATION_PACKAGE_FILES.json'
SOURCE_TRANSPORT_BYTES=256*1024**2
OWNER='01a175f0-fc99-76b1-8bb2-d64f22afbcd7'
ENDPOINTS=('behavior','competence')
ARMS=('base','always_on')
PROBE_NAMESPACE='research4-bounded-v4-no-request-competence-20261026'
PROBE_REGISTRATION={'version':'fresh-no-request-ordinary-competence-v4',
 'namespace':PROBE_NAMESPACE,'execution_root_registered_before_new_results':True,
 'cases':36,'groups':4,'groups_per_unseen_family':2,'replicates_per_group':9,
 'all_cases_initially_pending':True,'advisory_request_present':False,
 'lifecycle_record_effects':False,'historical_results_reused':False,
 'native_interface_profile':PROFILE,'fixed_external_hard_stop':24,
 'denominator':'all36freshordinarytaskinstances_separate_from_advisory_cases',
 'identities_depend_on_source_commit_or_results':False}


def competence_roster(cfg):
    """Identity descriptors only; no TaskWorld/record/reference construction."""
    validate_preparation(cfg);c=cfg['cohorts'];rows=[]
    from .composite_task_skill_world import VERSION
    for group in range(4):
        family,within=divmod(group,2)
        gid=digest({'namespace':PROBE_NAMESPACE,'configuration':CONFIG_SHA,'group':group})
        for replicate in range(9):
            values={'group_id':gid,'identity_seed':digest({'group':gid,'replicate':replicate}),
              'mechanism':c['development_mechanisms'][family],
              'task_structure':c['development_task_structures'][(family+within)%2],
              'domain':c['development_domains'][(family+within+replicate)%2],
              'template':c['development_templates'][within], 'replicate':replicate,
              'id_digits':c['development_id_lengths'][within],'role':'development','world_version':VERSION}
            rows.append({'case_id':digest({'probe_version':PROBE_REGISTRATION['version'],**values}),**values})
    return rows


def execution_plan(roster,endpoint):
    if endpoint not in ENDPOINTS or len(roster)!=36 or len({r['case_id'] for r in roster})!=36:
        raise PermissionError('Complete separate36-case actor endpoint required')
    groups={r['group_id'] for r in roster}
    if len(groups)!=4 or any(sum(r['group_id']==g for r in roster)!=9 for g in groups):
        raise PermissionError('All four original paired groups must remain complete')
    from .composite_world import CONDITIONS
    group_order=sorted(groups)
    if endpoint=='behavior':
        for group in group_order:
            if sorted(r['condition'] for r in roster if r['group_id']==group)!=sorted(CONDITIONS):
                raise PermissionError('Every original condition must remain once in each paired group')
        ordered=sorted(roster,key=lambda r:(group_order.index(r['group_id']),list(CONDITIONS).index(r['condition'])))
    else:
        if any(sorted(r['replicate'] for r in roster if r['group_id']==g)!=list(range(9)) for g in group_order):
            raise PermissionError('Every fresh ordinary task replicate must remain once per group')
        ordered=sorted(roster,key=lambda r:(group_order.index(r['group_id']),r['replicate']))
    pairs=[ordered[:1]]+[ordered[i:i+3] for i in range(1,36,3)]
    parts=[{'index':i,'case_ids':[r['case_id'] for r in values],'arms':list(ARMS),
      'expected_trajectories':len(values)*2,'maximum_native_responses':len(values)*48,
      'maximum_with_optional_native_parity':len(values)*48*160+326} for i,values in enumerate(pairs)]
    if (len(parts)!=13 or sum(p['expected_trajectories'] for p in parts)!=72
        or any(p['maximum_with_optional_native_parity']>30000 for p in parts)):
        raise PermissionError('Exact complete paired72 actor execution coverage required')
    return {'version':'bounded-actor-paired72-execution13-v4','endpoint':endpoint,'roster_sha256':digest(roster),
      'partitions':parts,'cases':36,'trajectories':72,'arms':list(ARMS),
      'partitioning_before_native_outcomes':True,'unrun_other_controls':'unknown',
      'ordering':'fixed_group_then_original_condition_or_replicate',
      'whole_case_pairs_in_each_batch':True,'whole_inferential_groups_preserved_across_batches':True,
      'runtime_throughput_qualified':False,'failed_or_unrun_batches':'preserved_unknown_or_failed_never_dropped',
      'heldout_content_read':False,'metadata_admits_launch':False}


def expected_stage(commit,model,data,cfg,endpoint,index):
    from .composite_bounded_behavior_evaluation_parent import require_frozen_model
    from .composite_bounded_behavior_data_reader import require_data
    frozen=require_frozen_model(model);report=require_data(data);validate_preparation(cfg)
    if not re.fullmatch('[0-9a-f]{40}',commit or '') or type(index) is not int or not 0<=index<13:
        raise PermissionError('Exact reviewed actor evaluation source/partition required')
    roster=data.scenes('development') if endpoint=='behavior' else competence_roster(cfg)
    plan=execution_plan(roster,endpoint);part=plan['partitions'][index]
    return {'version':MODE,'source_commit':commit,'owner_thread_id':OWNER,'endpoint':endpoint,
      'phase':'bounded_actor_'+endpoint,'notebook':'farhad89/research4-bounded-actor-'+endpoint+'-v4',
      'platform':'kaggle','account':'farhad89','native_interface_profile':PROFILE,
      'frozen_model_sha256':digest(frozen),'checkpoint_sha256':frozen['checkpoint_sha256'],
      'source_data_archive_sha256':report['raw_archive_sha256'],'data_audit_sha256':digest(report),
      'configuration_sha256':CONFIG_SHA,'evaluation_plan_sha256':digest(plan),'partition':part,
      'roster_sha256':digest(roster),'visible_devices':'0,1','scientific_GPU_count':2,
      'kernel_profile':'reference','optimizer_updates':0,'teacher_forcing':False,
      'decoder_repair':False,'policy_logit_masks':False,'detector_predictions_required':False,
      'model_context_overflow_policy':'durable_unexecuted_prefix_partial_unknown_capacity_failure',
      'reference_native_budget_implies_all_model_paths_fit':False,
      'learned_dynamic_or_sealed_admitted':False,'other_controls_status':'unrun_unknown',
      'source_transport_maximum_bytes':SOURCE_TRANSPORT_BYTES,
      'hard_timeout_seconds':42000,'worker_seconds':40800,'export_reserve_seconds':900,
      'maximum_backbone_forwards':part['maximum_with_optional_native_parity'],
      'maximum_raw_evidence_bytes':4*1024**3,'physical_export_reserve_bytes':12*1024**3,
      'raw_evidence_profile':'bounded-behavior-full-state-raw-v4',
      'maximum_concurrent_owned_workers':1,'automatic_scientific_retries':0,'heldout_content_read':False,
      'destination':'/kaggle/working/research4_campaign/runs/'+commit+'/bounded_actor_evaluation_v4/'+endpoint+'/partition_'+str(index).zfill(2)}


def stage_prefix(stage):
    if (stage.get('version')!=MODE or stage.get('endpoint') not in ENDPOINTS
        or type(stage.get('partition',{}).get('index')) is not int
        or not 0<=stage['partition']['index']<13
        or not re.fullmatch('[0-9a-f]{40}',stage.get('source_commit',''))):
        raise PermissionError('Exact bounded actor evaluation namespace required')
    return 'bounded_actor_evaluation_v4/'+stage['source_commit']+'/'+stage['endpoint']+'/partition_'+str(stage['partition']['index']).zfill(2)
