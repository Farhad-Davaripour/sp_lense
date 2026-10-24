"""Pure V2 model/data-stage metadata contract; no provider or model calls.

The input contract is deliberately default-denied without an actual admitted
data export and exact source/workspace/human/model bindings.
"""
import io,json,zipfile,hashlib,re
from pathlib import Path
from .io_utils import read,sha,digest,verify_release
from .composite_generalized_interface import PROFILE
from .composite_v2_training import validate_plan
from .composite_v2_data_contract import source_fingerprint,validate_workspace,ORIGINAL_MODEL_MANIFEST
from .composite_exports import verify_evidence_archive

MODE='generalized-qwen-model-stage-v2'


def validate_model_metadata(root,commit):
    root=Path(root);verify_release(root,commit);manifest=read(root/'inputs/MANIFEST.json')
    required={'WORKSPACE.json','V2_MODEL_STAGE.json','HUMAN_QWEN_FITTING_SCOPE.json','HUMAN_GENERALIZATION_SCOPE.json',
              'QWEN_PIN.json','PARENT_MODEL_INPUT_MANIFEST.json'}
    if not required<=set(manifest):raise PermissionError('Actual complete V2 model-stage preparation required')
    for name,value in manifest.items():
        if Path(name).name!=name or 'sealed' in name.lower() or 'heldout' in name.lower() or sha(root/'inputs'/name)!=value:
            raise PermissionError('Changed, unsafe or sealed model-stage input')
    if sha(root/'inputs/PARENT_MODEL_INPUT_MANIFEST.json')!=ORIGINAL_MODEL_MANIFEST:raise PermissionError('Original model input pin required')
    if sha(root/'inputs/QWEN_PIN.json')!=read(root/'inputs/PARENT_MODEL_INPUT_MANIFEST.json')['QWEN_PIN.json']:
        raise PermissionError('Original full Qwen file pin changed')
    human=read(root/'inputs/HUMAN_QWEN_FITTING_SCOPE.json')
    if human.get('human_user_message_id')!='01a16c28-f501-7af1-b3ff-afb1a7e9761b' or human.get('status')!='verified_direct_human_Qwen_adapter_fitting_scope_clarification':
        raise PermissionError('Bind the verified direct human fitting scope')
    general=read(root/'inputs/HUMAN_GENERALIZATION_SCOPE.json')
    if general.get('human_user_message_id')!='01a16c34-f963-77c0-aee1-d48bd7338bf8':raise PermissionError('Generalized curriculum scope required')
    workspace=validate_workspace(read(root/'inputs/WORKSPACE.json'));stage=read(root/'inputs/V2_MODEL_STAGE.json')
    if (stage.get('mode')!=MODE or stage.get('source_commit')!=commit or stage.get('native_interface_profile')!=PROFILE
        or stage.get('automatic_scientific_retries')!=0 or stage.get('maximum_concurrent_owned_workers')!=1
        or stage.get('phase') not in ['qualification','full_fit'] or stage.get('GPU_count')!=1
        or stage.get('accelerator_type')!='GPU_1xA10' or stage.get('heldout_content_read') is not False
        or stage.get('qualification_warm_start_used') is not False):raise PermissionError('Exact bounded V2 single-worker model contract required')
    cfg=read(root/'config/composite_controller_rank32_v2.json')
    bounds=cfg['bounds']
    expected={'worker_seconds':bounds['worker_seconds'],'export_reserve_seconds':bounds['export_reserve_seconds'],
              'hard_timeout_seconds':bounds['hard_timeout_minutes']*60,'maximum_backbone_forwards':bounds['maximum_backbone_forwards_per_run']}
    if any(type(stage.get(k)) is not type(v) or stage[k]!=v for k,v in expected.items()):raise PermissionError('Registered source-bound time/forward/export envelope required')
    if stage.get('protocol_sha256')!=digest({'base':cfg,'design':read(root/'config/composite_generalization_amendment_v2.json')}):
        raise PermissionError('Exact model/scenario/measurement protocol changed')
    prefix=workspace['artifact_volume']+'/composite_controller_v1/model_generalized_v2/'+commit+'/'
    if stage.get('destination')!=prefix+stage.get('phase','')+'_01':raise PermissionError('Exact fresh source/phase destination required')
    data=stage.get('data_export',{})
    if not isinstance(data.get('path'),str) or not re.fullmatch(r'data_generalized_v2/[0-9a-f]{40}/export_01_EVIDENCE\.zip',data['path']):
        raise PermissionError('Exact nonsealed data export locator required')
    if not re.fullmatch('[0-9a-f]{64}',data.get('sha256','')):raise PermissionError('Exact data archive digest required')
    return {'root':str(root),'source_commit':commit,'workspace':workspace,'stage':stage,'configuration':cfg,
            'allocation_authorized_by_metadata_validation':False}


def load_training_export(raw,context,*,qualification_evidence=None):
    if hashlib.sha256(raw).hexdigest()!=context['stage']['data_export']['sha256']:raise PermissionError('Data archive changed')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(v.file_size for v in archive.infolist())>256*1024**2:raise ValueError('Data expansion exceeds its unchanged bound')
        if any('sealed' in name.lower() or 'heldout' in name.lower() for name in archive.namelist()):raise PermissionError('No sealed data in a training package')
    verify_evidence_archive(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        release=json.loads(archive.read('DATA_RELEASE.json'));receipt=json.loads(archive.read('RUN_RECEIPT.json'))
        if receipt.get('status')!='complete' or receipt.get('scientific_dataset_admitted') is not True or receipt.get('public_fixture_test_mode') is not False:
            raise PermissionError('A public fixture, failed or partial data export cannot admit fitting')
        if release['world_source_files']!=source_fingerprint(context['root']):raise PermissionError('World/teacher/interface changed after data generation')
        validate_plan(context['stage'],context['configuration'],release,context['workspace'],qualification_evidence=qualification_evidence)
        raw_rows=archive.read('training/DETECTION_ROWS.json')
        if hashlib.sha256(raw_rows).hexdigest()!=release['cohorts']['training']['files']['DETECTION_ROWS.json']:
            raise PermissionError('Training row digest changed')
        rows=json.loads(raw_rows)
        if any(row.get('split')!='training' for row in rows):raise PermissionError('Optimizer reads only training targets')
    return rows,release

