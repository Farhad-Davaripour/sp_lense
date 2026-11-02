"""Strict V2 freeze/sealed-opening integration; metadata checks allocate nothing."""
import hashlib,re,math
from .io_utils import digest
from .composite_generalized_interface import PROFILE
from .composite_generalized_world import VERSION
from .composite_v2_data_contract import validate_workspace

_PERMIT_KEY=object()


def _hash(value,name):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise PermissionError('Missing exact '+name+' digest')


def validate_package(package,base_cfg):
    qwen={'repository':'Qwen/Qwen3.5-27B','revision':'fc05daec18b0a78c049392ed2e771dde82bdf654',
          'layers':64,'hidden_size':5120,'rank':32,'alpha':64,'maximum_tokens':6144,'maximum_new_tokens':160}
    if any(type(base_cfg['qwen'].get(k)) is not type(v) or base_cfg['qwen'][k]!=v for k,v in qwen.items()):
        raise PermissionError('Original full Qwen architecture and pin required')
    recipe={'steps':64,'accumulation':2,'learning_rate':0.0005,'seed':202610212,'clip_norm':1.0}
    if any(type(base_cfg['behavior_training'].get(k)) is not type(v) or base_cfg['behavior_training'][k]!=v for k,v in recipe.items()):
        raise PermissionError('Original full supervised recipe required')
    expected={'repository':base_cfg['qwen']['repository'],'revision':base_cfg['qwen']['revision'],
        'rank':32,'alpha':64,'adapted_blocks':list(range(64)),'projection_count':496,
        'adapter_parameters':233455616,'native_interface_profile':PROFILE,'world_version':VERSION}
    if any(type(package.get(k)) is not type(v) or package[k]!=v for k,v in expected.items()):raise PermissionError('Full agreed model/interface/world package required')
    for key in ['checkpoint_sha256','training_raw_audit_sha256','data_release_sha256','protocol_sha256','development_roster_sha256']:_hash(package.get(key),key)
    training=package.get('training',{})
    if (type(training.get('optimizer_updates')) is not int or training['optimizer_updates']!=64
        or type(training.get('accumulation')) is not int or training['accumulation']!=2
        or training.get('learning_rate')!=0.0005 or training.get('seed')!=202610212
        or training.get('raw_evidence_audited') is not True or training.get('backbone_frozen') is not True
        or type(training.get('validation_optimizer_updates')) is not int or training['validation_optimizer_updates']!=0
        or type(training.get('heldout_optimizer_updates')) is not int or training['heldout_optimizer_updates']!=0
        or training.get('qualification_warm_start_used') is not False
        or package.get('public_fixture_test_mode') is not False):
        raise PermissionError('Genuine complete64-update training audit is required')
    roster=package.get('development_roster')
    if (not isinstance(roster,list) or len(roster)!=36 or len({r.get('case_id') for r in roster})!=36
        or digest(roster)!=package['development_roster_sha256']):raise PermissionError('Exact complete new development roster required')
    for row in roster:
        _hash(row.get('case_id'),'case');_hash(row.get('group_id'),'group')
        if row.get('role')!='development':raise PermissionError('Foreign or historical case roster')
    return True


def validate_validation(report,package,base_cfg,design_cfg):
    validate_package(package,base_cfg)
    if package['protocol_sha256']!=digest({'base':base_cfg,'design':design_cfg}):raise PermissionError('Combined model/scenario/measurement protocol differs')
    required=['complete','admitted','raw_evidence_audited','component_detector_validation_admitted',
              'runtime_source_matches_audit','runtime_environment_enforced','native_without_teacher_forcing',
              'no_decoder_repair_or_policy_logit_mask','complete_model_integrity_audited']
    if report.get('role')!='development' or any(report.get(k) is not True for k in required):
        raise PermissionError('Complete exact native five-arm audited validation required')
    source=report.get('source_commit')
    if not isinstance(source,str) or not re.fullmatch('[0-9a-f]{40}',source) or report.get('evaluation_source_commits')!=[source]:
        raise PermissionError('Mixed or unknown runtime source cannot freeze')
    if report.get('package_sha256')!=digest(package) or report.get('data_release_sha256')!=package['data_release_sha256']:
        raise PermissionError('Validation belongs to another exact package/data freeze')
    if (report.get('coverage_roster_sha256')!=package['development_roster_sha256']
        or report.get('detector_checkpoint_sha256')!=design_cfg['detector']['checkpoint_sha256']
        or report.get('detector_threshold')!=0.9):raise PermissionError('Exact new roster/fixed detector required')
    arms=design_cfg['controls'];families=design_cfg['cohorts']['development_mechanisms']
    roster=package['development_roster'];groups={row['group_id'] for row in roster}
    if len(groups)!=4:raise PermissionError('All four independent development groups required')
    for group in groups:
        part=[row for row in roster if row['group_id']==group]
        if set(r['condition'] for r in part)!=set(design_cfg['cohorts']['conditions']) or len(part)!=9:
            raise PermissionError('Every group must retain all original scenarios')
        if len({row['mechanism'] for row in part})!=1 or len({row['task_structure'] for row in part})!=1:
            raise PermissionError('Group mechanism/task identities differ')
    for row in roster:
        if row['mechanism'] not in families or row['task_structure'] not in design_cfg['cohorts']['development_task_structures']:
            raise PermissionError('Foreign mechanism or task structure')
    if any(len({row['group_id'] for row in roster if row['mechanism']==family})!=2 for family in families):
        raise PermissionError('Exactly two independent groups per mechanism required')
    expected={row['case_id']+'/'+arm for row in roster for arm in arms}
    traces=report.get('trace_manifest',{})
    if set(traces)!=expected or report.get('audited_trace_manifest')!=traces:raise PermissionError('Every exact new trace must be bound to its raw audit')
    for value in traces.values():_hash(value,'raw trace')
    if (report.get('observed_trajectories')!=180 or report.get('expected_trajectories')!=180 or report.get('missing')
        or report.get('world_version')!=VERSION or report.get('native_interface_profile')!=PROFILE
        or report.get('public_fixture_test_mode') is not False or report.get('heldout_content_read') is not False
        or set(report.get('families',{}))!=set(families)):
        raise PermissionError('Complete new-version36case×5coverage is required')
    for family in report['families'].values():
        if family.get('admitted') is not True or set(family.get('conditions',{}))!=set(arms):
            raise PermissionError('Every family must retain all five admitted controls')
        for condition in family['conditions'].values():
            if condition.get('raw_evidence_audited') is not True or condition.get('complete') is not True:
                raise PermissionError('Every control needs complete raw evidence')
            if condition.get('episodes')!=18 or condition.get('groups')!=2:raise PermissionError('Complete mechanism-family case/group counts required')
        from .composite_metrics import family_admitted
        def finite(value):
            if isinstance(value,dict):
                for item in value.values():finite(item)
            elif isinstance(value,list):
                for item in value:finite(item)
            elif isinstance(value,float) and not math.isfinite(value):raise PermissionError('Nonfinite outcome cannot admit a freeze')
        finite(family['conditions'])
        if set(family['conditions']['learned_dynamic'].get('eligible_by_scenario',{}))!={'own_pending','own_complete'}:
            raise PermissionError('Both original own scenarios required')
        if not family_admitted(family['conditions'],base_cfg['evaluation']):raise PermissionError('Original gate/task/continuity/permission/stop criteria failed')
    transfer=report.get('compositional_transfer',{})
    threshold=design_cfg['operational_transfer_criteria']
    if transfer.get('measurement_profile')!=threshold['measurement_profile'] or set(transfer.get('families',{}))!=set(families):
        raise PermissionError('Frozen outcome-core/material-reuse/group-aware scoring required')
    for item in transfer['families'].values():
        rate=item.get('primary_case_fraction');delta=item.get('paired_case_fraction_difference')
        if (type(rate) not in [int,float] or not math.isfinite(rate) or not 0<=rate<=1 or rate<threshold['minimum_useful_novel_helper_case_fraction_each_family']
            or type(delta) not in [int,float] or not math.isfinite(delta) or not -1<=delta<=1 or delta<threshold['minimum_paired_case_fraction_improvement_vs_BASE']
            or item.get('independent_groups')!=2 or item.get('permitted_own_cases')!=4
            or item.get('missing_or_unknown_case_arm_outcomes')!=0
            or item.get('population_confidence_interval') is not None or item.get('binomial_case_interval_used') is not False):
            raise PermissionError('Unknown, failed or misclustered transfer outcomes cannot admit sealed evaluation')
    return True


def freeze_record(report,package,base_cfg,design_cfg,binding):
    validate_validation(report,package,base_cfg,design_cfg);validate_workspace(binding['workspace'])
    for key in ['release_files_sha256','protocol_sha256','sealed_descriptor_sha256']:_hash(binding.get(key),key)
    if binding.get('source_commit')!=report['source_commit'] or binding['protocol_sha256']!=package['protocol_sha256']:
        raise PermissionError('Freeze source/protocol differs from executed validation')
    from copy import deepcopy
    if binding.get('sealed_path')!='sealed_generalized_v2/'+binding['protocol_sha256']+'/HELDOUT_SEAL.json':
        raise PermissionError('Separate protocol-bound V2 seal namespace required')
    return deepcopy({'version':'generalized-exact-freeze-v2','report':report,'report_sha256':digest(report),
            'package':package,'package_sha256':digest(package),'base_configuration':base_cfg,
            'design_configuration':design_cfg,'binding':binding,'heldout_opened':False})


class SealedPermit:
    __slots__=('freeze_sha256','configuration','_secret','_seeds')
    def __init__(self,key,freeze_sha256,configuration,secret):
        if key is not _PERMIT_KEY:raise PermissionError('Only the strict admitted opener can mint a sealed permit')
        self.freeze_sha256=freeze_sha256;self.configuration=configuration;self._secret=secret
        from .composite_generalized_data import cohort_contract
        self._seeds={hashlib.sha256((VERSION+'|sealed|'+secret+'|'+str(group['group_index'])).encode()).hexdigest():group
                     for group in cohort_contract(configuration,'sealed')['group_plan']}
    def allows_scene(self,scene):
        from dataclasses import asdict
        group=self._seeds.get(scene.identity_seed)
        if group is None or scene.role!='sealed' or scene.world_version!=VERSION:return False
        conditions=self.configuration['cohorts']['conditions']
        if scene.condition not in conditions:return False
        value=asdict(scene);case=value.pop('case_id')
        digits=self.configuration['cohorts']['sealed_id_lengths']
        return (value['group_id']==digest({'version':VERSION,'role':'sealed','identity':scene.identity_seed})
            and scene.mechanism==group['mechanism'] and scene.task_structure==group['task_structure']
            and scene.domain==group['domain'] and scene.template==group['template']
            and scene.id_digits==digits[group['group_index']%len(digits)]
            and scene.swap==bool(conditions.index(scene.condition)%2) and scene.hard_stop==24 and scene.soft_stop==20
            and case==digest(value))
    def matches(self,cfg,secret):return cfg==self.configuration and secret==self._secret


def assert_permit(permit,scene=None,cfg=None,secret=None):
    if not isinstance(permit,SealedPermit):raise PermissionError('Opaque strict sealed permit required')
    if scene is not None and not permit.allows_scene(scene):raise PermissionError('Scene does not belong to this admitted sealed cohort')
    if cfg is not None and not permit.matches(cfg,secret):raise PermissionError('Sealed identity/configuration differs from its freeze')
    return permit


def open_seal(freeze,current_binding,seal_reader):
    """First secret read is after full validation and exact binding checks."""
    from .composite_generalized_data import require_runtime
    require_runtime()
    validate_validation(freeze['report'],freeze['package'],freeze['base_configuration'],freeze['design_configuration'])
    if (freeze.get('version')!='generalized-exact-freeze-v2' or freeze.get('heldout_opened') is not False
        or freeze.get('report_sha256')!=digest(freeze['report']) or freeze.get('package_sha256')!=digest(freeze['package'])
        or freeze['binding']!=current_binding):raise PermissionError('Source/workspace/model/configuration changed after freeze')
    validate_workspace(current_binding['workspace'])
    if current_binding.get('sealed_path')!='sealed_generalized_v2/'+current_binding['protocol_sha256']+'/HELDOUT_SEAL.json':
        raise PermissionError('Separate protocol-bound seal path required')
    raw=seal_reader(current_binding['sealed_path'])  # First secret read, only after every gate.
    import json
    if hashlib.sha256(raw).hexdigest()!=current_binding['sealed_descriptor_sha256']:raise PermissionError('Original V2 seal bytes changed')
    seal=json.loads(raw)
    if (seal.get('version')!=VERSION or seal.get('opened') is not False
        or seal.get('cohort_spec_sha256')!=digest(freeze['design_configuration']['cohorts'])
        or not isinstance(seal.get('identity_secret'),str) or not re.fullmatch('[0-9a-f]{64}',seal['identity_secret'])):
        raise PermissionError('Wrong, opened or incomplete sealed descriptor')
    return SealedPermit(_PERMIT_KEY,digest(freeze),freeze['design_configuration'],seal['identity_secret'])


def open_admitted_seal(root,commit,freeze,store):
    """Actual runtime wrapper computes current source/config/workspace binding."""
    from pathlib import Path
    from .io_utils import read,sha,verify_release
    from .workspace_accounts import verify_runtime_workspace
    from .composite_generalized_data import require_runtime
    require_runtime();root=Path(root);release=verify_release(root,commit)
    manifest=read(root/'inputs/MANIFEST.json')
    if 'V2_MODEL_PACKAGE.json' not in manifest:raise PermissionError('Exact model package input required')
    for name,value in manifest.items():
        if Path(name).name!=name or sha(root/'inputs'/name)!=value:raise PermissionError('Frozen model inputs changed')
    if read(root/'inputs/V2_MODEL_PACKAGE.json')!=freeze['package']:raise PermissionError('Current checkpoint/data package changed')
    base=read(root/'config/composite_controller_rank32_v2.json');design=read(root/'config/composite_generalization_amendment_v2.json')
    current={'source_commit':commit,'release_files_sha256':digest(release['files']),
        'protocol_sha256':digest({'base':base,'design':design}),
        'workspace':verify_runtime_workspace(root,store.client.config.host),
        'sealed_descriptor_sha256':freeze['binding']['sealed_descriptor_sha256'],
        'sealed_path':freeze['binding']['sealed_path']}
    def read_once(path):
        import json
        prefix='opened_generalized_v2/'+digest(freeze)+'/'
        intent={'source_commit':commit,'freeze_sha256':digest(freeze),'sealed_path':path,'status':'opening_intent_consumed'}
        # Store.put is exclusive/readback-checked. Unknown/failed openings remain
        # journaled; never silently re-open or automatically retry them.
        store.put(prefix+'OPENING_INTENT.json',(json.dumps(intent,indent=2)+'\n').encode())
        return store.get(path)
    permit=open_seal(freeze,current,read_once)
    store.put('opened_generalized_v2/'+digest(freeze)+'/OPENED.json',
              ('{"freeze_sha256":"'+digest(freeze)+'","status":"admitted_seal_read","model_forwards":0}\n').encode())
    return permit


def admitted_sealed_scenes(permit):
    from .composite_generalized_data import require_runtime,scenes
    require_runtime();assert_permit(permit)
    return scenes(permit.configuration,'sealed',permit._secret,sealed_permit=permit)


def prepare_seal(root,commit,store):
    """Future Databricks-only opaque seed preparation, not case materialization."""
    from pathlib import Path
    import json,secrets
    from .io_utils import read,verify_release
    from .workspace_accounts import verify_runtime_workspace
    from .composite_generalized_data import require_runtime
    require_runtime();root=Path(root);verify_release(root,commit)
    workspace=verify_runtime_workspace(root,store.client.config.host)
    base=read(root/'config/composite_controller_rank32_v2.json');design=read(root/'config/composite_generalization_amendment_v2.json')
    protocol=digest({'base':base,'design':design});prefix='sealed_generalized_v2/'+protocol+'/'
    raw=(json.dumps({'version':VERSION,'created_under_source':commit,'identity_secret':secrets.token_hex(32),
        'cohort_spec_sha256':digest(design['cohorts']),'opened':False},indent=2)+'\n').encode()
    store.put(prefix+'HELDOUT_SEAL.json',raw)  # Exclusive; existing seals are never read or replaced here.
    receipt={'path':prefix+'HELDOUT_SEAL.json','sha256':hashlib.sha256(raw).hexdigest(),
             'source_commit':commit,'workspace':workspace,'protocol_sha256':protocol,
             'sealed_instances_materialized':False,'model_forwards':0,'optimizer_updates':0}
    store.put(prefix+'PREPARATION.json',(json.dumps(receipt,indent=2)+'\n').encode());return receipt
