"""Fixed development coverage and phase-specific bounds; no scene/model calls."""
import re
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_v2_controller import CONDITIONS
from .composite_generalized_interface import PROFILE

MODE='generalized-development-endpoint-v1'
PLAN='V2_DEVELOPMENT_STAGE.json'
GiB=1024**3
COLLECTION_BOUNDS={'worker_seconds':2400,'export_reserve_seconds':180,'hard_timeout_seconds':2700,
    'maximum_backbone_forwards':30000,'maximum_raw_evidence_bytes':4*GiB,'physical_export_reserve_bytes':9*GiB,
    'minimum_host_available_bytes':12*GiB,'GPU_count':1,'accelerator_type':'GPU_1xA10',
    'maximum_concurrent_owned_workers':1,'automatic_scientific_retries':0,'forward_progress_interval':100}
TRANSFER_BOUNDS={'worker_seconds':600,'export_reserve_seconds':60,'hard_timeout_seconds':900,
    'maximum_backbone_forwards':900,'maximum_raw_evidence_bytes':256*1024**2,'physical_export_reserve_bytes':GiB,
    'minimum_host_available_bytes':2*GiB,'GPU_count':1,'accelerator_type':'GPU_1xA10',
    'maximum_concurrent_owned_workers':1,'automatic_scientific_retries':0,'forward_progress_interval':25}


def _same(actual,expected,message):
    if digest(actual)!=digest(expected):raise PermissionError(message)


def validate_roster(roster,design):
    if not isinstance(roster,list) or len(roster)!=36 or len({r.get('case_id') for r in roster})!=36:
        raise PermissionError('Exactly 36 distinct complete development cases required')
    groups={r['group_id'] for r in roster}
    if len(groups)!=4:raise PermissionError('All four independent groups required')
    families=design['cohorts']['development_mechanisms']
    for row in roster:
        if row.get('role')!='development' or row.get('mechanism') not in families or row.get('task_structure') not in design['cohorts']['development_task_structures']:
            raise PermissionError('Foreign development role/mechanism/task')
        for key in ['case_id','group_id']:
            if not isinstance(row.get(key),str) or not re.fullmatch('[0-9a-f]{64}',row[key]):raise PermissionError('Exact case/group digest required')
    for group in groups:
        part=[r for r in roster if r['group_id']==group]
        if len(part)!=9 or {r['condition'] for r in part}!=set(design['cohorts']['conditions']):
            raise PermissionError('Every group retains all nine original scenarios')
        if len({r['mechanism'] for r in part})!=1 or len({r['task_structure'] for r in part})!=1:
            raise PermissionError('Group mechanism/task crossed identities')
    if any(len({r['group_id'] for r in roster if r['mechanism']==f})!=2 for f in families):
        raise PermissionError('Exactly two paired groups per mechanism required')
    return True


def partition_plan(roster,design):
    validate_roster(roster,design)
    _same(design['architecture']['maximum_new_tokens'],160,'Original complete native generation budget required')
    _same(design['tool_interface']['maximum_episode_ticks'],24,'Original external episode bound required')
    _same(design['controls'],list(CONDITIONS),'All five original control meanings required')
    # Each decision consumes at least one external tick, including invalid
    # calls. Helpers cannot discount ticks. A generation consumes <=160 model
    # calls and observations add <=25 detector calls per trace. The optional
    # matched OFF/bare parity adds at most 2*160+6 actor calls per partition.
    ticks=24;generation=160;arms=len(CONDITIONS)
    per_trace=ticks*generation+(ticks+1)
    bound=arms*per_trace+2*generation+6
    if bound>COLLECTION_BOUNDS['maximum_backbone_forwards']:raise AssertionError('Whole-case partition exceeds original forward envelope')
    partitions=[{'index':i,'key':'partition_'+str(i).zfill(2),'case_ids':[row['case_id']],
        'controls':list(CONDITIONS),'expected_trajectories':5,'maximum_native_responses':arms*ticks,
        'maximum_joint_backbone_forwards':bound} for i,row in enumerate(sorted(roster,key=lambda r:r['case_id']))]
    return {'version':'fixed-whole-case-all-five-v1','roster_sha256':digest(roster),'partitions':partitions,
        'expected_cases':36,'expected_trajectories':180,'maximum_joint_backbone_forwards_per_partition':bound,
        'maximum_reference_detector_states':36*(ticks+1),'maximum_collection_job_hours':36*2700/3600,
        'full_coverage_forward_proved':True,'runtime_throughput_qualified':False,'full_campaign_cost_admitted':False,
        'partitioning_changes_no_cases_controls_or_scientific_limits':True}


def validate_stage(root,commit):
    root=Path(root)
    # Both validators verify immutable source/input/model/workspace bindings;
    # neither creates compute authority. Transfer has no dependency on a fitted
    # Qwen model, so it can validate the fixed detector before model fitting.
    stage=read(root/'inputs'/PLAN);phase=stage.get('phase')
    if phase=='detector_transfer':
        from .composite_v2_data_contract import validate_metadata
        prior=validate_metadata(root,commit)
        context={'root':str(root),'source_commit':commit,'workspace':prior['workspace']}
    elif phase=='development_partition':
        from .composite_v2_model_contract import validate_model_metadata
        context=validate_model_metadata(root,commit)
        if context['stage']['phase']!='full_fit':raise PermissionError('Development requires the exact full-fit source contract')
    else:raise PermissionError('Only frozen detector transfer or native development collection')
    manifest=read(root/'inputs/MANIFEST.json')
    if manifest.get(PLAN)!=sha(root/'inputs'/PLAN):raise PermissionError('Development stage is not in the frozen input manifest')
    cfg=read(root/'config/composite_controller_rank32_v2.json')
    if cfg['qwen']['maximum_new_tokens']!=160 or cfg['bounds']['maximum_episode_ticks']!=24:
        raise PermissionError('Static coverage proof requires the original native/episode bounds')
    design=read(root/'config/composite_generalization_amendment_v2.json')
    for key,value in [('mode',MODE),('source_commit',commit),('native_interface_profile',PROFILE),
                      ('heldout_content_read',False),('optimizer_updates',0),('threshold',0.9),
                      ('detector_checkpoint_sha256',design['detector']['checkpoint_sha256'])]:
        _same(stage.get(key),value,'Development mode/source/frozen-detector boundary differs')
    bounds=TRANSFER_BOUNDS if phase=='detector_transfer' else COLLECTION_BOUNDS
    _same(stage.get('bounds'),bounds,'Phase-specific complete time/forward/export envelope changed')
    if stage.get('protocol_sha256')!=digest({'base':cfg,'design':design}):raise PermissionError('Scientific protocol differs')
    locator=stage.get('data_export',{})
    if not re.fullmatch(r'data_generalized_v2/[0-9a-f]{40}/export_01_EVIDENCE\.zip',locator.get('path','')):
        raise PermissionError('Exact complete nonsealed data archive required')
    for value in [locator.get('sha256'),stage.get('data_release_sha256')]:
        if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise PermissionError('Exact data archive/release hashes required')
    roster=stage.get('development_roster');plan=partition_plan(roster,design)
    def reference(value,expected_sha=None):
        if (not isinstance(value,dict) or not isinstance(value.get('path'),str)
            or not re.fullmatch('[A-Za-z0-9_./-]+',value['path']) or value['path'].startswith('/')
            or '..' in value['path'].split('/') or 'sealed' in value['path'].lower() or 'heldout' in value['path'].lower()
            or not isinstance(value.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',value['sha256'])
            or (expected_sha is not None and value['sha256']!=expected_sha)):
            raise PermissionError('Exact nonsealed Research4 artifact reference required')
    reference(stage.get('detector_checkpoint'),design['detector']['checkpoint_sha256'])
    if phase=='development_partition':
        index=stage.get('partition_index')
        if type(index) is not int or not 0<=index<36:raise PermissionError('Only the 36 fixed whole-case partitions')
        _same(stage.get('partition'),plan['partitions'][index],'Case/control partition differs from the fixed roster')
        reference(stage.get('package'));reference(stage.get('model_checkpoint'))
        if stage['model_checkpoint']['path']!='model_generalized_v2/'+commit+'/full_fit_01/raw/behavior.pt':
            raise PermissionError('Only the exact complete fitted actor checkpoint')
        from .composite_v2_full_parent import validate_full_fit_descriptor
        validate_full_fit_descriptor(stage,context['workspace'])
        leaf=plan['partitions'][index]['key']
    else:leaf='detector_transfer_01'
    relative='development_generalized_v2/'+commit+'/'+leaf
    expected=context['workspace']['artifact_volume']+'/composite_controller_v1/'+relative
    if stage.get('destination')!=expected or stage.get('provider_binding_path')!=relative+'/PROVIDER_BINDING.json':
        raise PermissionError('Exact source/phase/owner destination required')
    return {**context,'stage':stage,'configuration':cfg,'design':design,'partition_plan':plan,
            'bounds':bounds,'allocation_authorized_by_metadata_validation':False}
