"""Pure V2 data-stage binding and export contract; no provider or cohort calls."""
from pathlib import Path
import re
from .io_utils import read,sha,digest,verify_release
from .workspace_accounts import profiles,artifact_volume
from .composite_generalized_data import cohort_contract
from .composite_generalized_world import VERSION

MODE='generalized-data-export-v2'
AUTHORITY='GENERALIZED_DATA_AUTHORIZATION.json'
HUMAN='HUMAN_GENERALIZATION_SCOPE.json'
PLAN='GENERALIZED_DATA_PLAN.json'
ORIGINAL_MODEL_MANIFEST='db9522b76dc7af09fa354577761441d3896b62d17706bab96ea37c9159b48596'
BOUNDS={'worker_seconds':600,'export_reserve_seconds':60,'hard_timeout_seconds':900,
        'maximum_raw_export_bytes':256*1024**2,'terminal_metadata_reserve_bytes':1024**2,
        'maximum_concurrent_owned_workers':1,'automatic_scientific_retries':0}


def validate_workspace(value):
    expected=profiles()['accounts'].get(value.get('account'))
    if expected is None:raise PermissionError('Unknown explicitly selected account')
    for key in ['host','organization']:
        if value.get(key)!=expected[key]:raise PermissionError('Workspace descriptor changed')
    if value.get('artifact_volume')!=artifact_volume(account=value['account']):raise PermissionError('Dedicated account volume required')
    return value


def validate_metadata(root,commit):
    root=Path(root)
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact reviewed commit required')
    verify_release(root,commit);manifest=read(root/'inputs/MANIFEST.json')
    for name in ['src/composite_v2_data_contract.py','src/composite_v2_data_worker.py','src/composite_generalized_data.py']:
        if not (root/name).is_file():raise PermissionError('Selected source predates the registered data contract')
    required={AUTHORITY,HUMAN,PLAN,'WORKSPACE.json','QWEN_PIN.json','MODERNBERT_PIN.json','PARENT_MODEL_INPUT_MANIFEST.json'}
    if not required<=set(manifest):raise PermissionError('Incomplete source/workspace/data plan freeze')
    for name,expected in manifest.items():
        if Path(name).name!=name or 'heldout' in name.lower() or 'sealed' in name.lower() or sha(root/'inputs'/name)!=expected:
            raise ValueError('Unregistered, sealed or changed data-stage input')
    parent=root/'inputs/PARENT_MODEL_INPUT_MANIFEST.json'
    if sha(parent)!=ORIGINAL_MODEL_MANIFEST:raise ValueError('Original model input freeze changed')
    for name in ['QWEN_PIN.json','MODERNBERT_PIN.json']:
        if sha(root/'inputs'/name)!=read(parent)[name]:raise ValueError('Original full model-file pin changed')
    from .composite_frozen_probe import verify_original_model_pin
    verify_original_model_pin(read(root/'inputs/QWEN_PIN.json'))
    human=read(root/'inputs'/HUMAN)
    if (human.get('status')!='verified_direct_human_generalization_scope'
        or human.get('human_user_message_id')!='01a16c34-f963-77c0-aee1-d48bd7338bf8'):
        raise PermissionError('Bind the already verified direct human design steering')
    workspace=validate_workspace(read(root/'inputs/WORKSPACE.json'))
    cfg=read(root/'config/composite_generalization_amendment_v2.json')
    plan=read(root/'inputs'/PLAN)
    expected_plan={'version':MODE,'roles':['training','development'],
        'contracts':{role:cohort_contract(cfg,role) for role in ['training','development']},
        'identity_material':'research4-generalized-v2-20261024-cohorts-v1',
        'model_weights_downloaded':False,'model_forwards':0,'optimizer_updates':0,
        'sealed_instances_materialized':False,**BOUNDS}
    if plan!=expected_plan:raise ValueError('Full roster/identity/bounds contract changed')
    authority=read(root/'inputs'/AUTHORITY)
    expected={'status':'explicit_human_generalization_continuation','authorized_by':'human_user',
        'human_user_message_id':human['human_user_message_id'],'source_commit':commit,
        'world_version':VERSION,'mode':MODE,'roles':['training','development'],'heldout_content_read':False,
        'maximum_episode_ticks':24,'maximum_concurrent_owned_workers':1,'automatic_scientific_retries':0,
        'human_scope_sha256':sha(root/'inputs'/HUMAN),'plan_sha256':sha(root/'inputs'/PLAN),
        'workspace_descriptor_sha256':sha(root/'inputs/WORKSPACE.json'),
        'amendment_sha256':sha(root/'config/composite_generalization_amendment_v2.json')}
    if any(type(authority.get(k)) is not type(v) or authority[k]!=v for k,v in expected.items()):
        raise PermissionError('Data-stage source/authority/workspace binding differs')
    prefix=artifact_volume(account=workspace['account'])+'/composite_controller_v1/data_generalized_v2/'+commit+'/'
    destination=authority.get('destination','')
    if not destination.startswith(prefix) or destination!=prefix+'export_01':raise PermissionError('Fresh exact Research4 export destination required')
    return {'source_root':str(root),'source_commit':commit,'configuration':cfg,'workspace':workspace,
            'plan':plan,'destination':destination,'authorization_sha256':sha(root/'inputs'/AUTHORITY),
            'bounds':BOUNDS,'allocation_authorized_by_metadata_validation':False}


def source_fingerprint(root):
    root=Path(root)
    names=['src/composite_generalized_world.py','src/composite_generalized_teacher.py',
           'src/composite_generalized_tasks.py','src/composite_generalized_interface.py',
           'src/composite_generalized_audit.py','src/composite_generalized_data.py',
           'src/composite_design_v2.py','src/composite_design_v2_native.py','src/composite_generalized_novelty.py',
           'src/composite_world.py','src/composite_interface.py','src/native27_tools.py']
    return {name:sha(root/name) for name in names}


def roster(episodes,role,cfg):
    contract=cohort_contract(cfg,role)
    if len(episodes)!=contract['cases']:raise ValueError('Roster must retain every original condition/group')
    result=[{key:episode['scene'][key] for key in ['case_id','group_id','condition','mechanism','task_structure','domain','template','role']} for episode in episodes]
    if any(row['role']!=role for row in result) or len({r['case_id'] for r in result})!=len(result):raise ValueError('Roster identity or role differs')
    return {'role':role,'case_count':len(result),'group_count':len({r['group_id'] for r in result}),'cases':result,
            'roster_sha256':digest(result),'reference_actions_in_model_evaluation_roster':False}


def dataset_admission(audits,budgets,leakage):
    if set(audits)!={'training','development'} or set(budgets)!={'training','development'}:raise ValueError('Both complete nonsealed roles required')
    for role in ['training','development']:
        if audits[role].get('role')!=role or audits[role].get('reference_world_replayed') is not True:
            raise PermissionError('Complete exact reference replay required')
        if budgets[role].get('status')!='complete_native_budget_audit_pass' or budgets[role].get('truncated_or_repaired') is not False:
            raise PermissionError('Every actual prompt and full native target must fit without truncation')
        if audits[role]['complete_states']!=budgets[role]['states']:raise ValueError('Native audit omitted states')
    if leakage.get('status')!='structural_and_exact_leakage_checks_passed':
        raise PermissionError('Leakage failure or unresolved semantic-review flags block admission')
    return {'status':'complete_nonsealed_dataset_audits_passed','training_and_development_admitted':True,
            'model_training_admitted_by_this_report':False,'detector_transfer_validation':None,
            'heldout_content_read':False,'model_forwards':0,'optimizer_updates':0}
