"""Shared-ledger one-use V4 stage reservation. This module submits nothing."""
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_submission_admission import guard,reserve,reconcile
from .composite_kaggle_v2_admission import candidate_rule
from .composite_kaggle_v2_profile import HUMAN_SHA
from .composite_bounded_sampling_fit_profile import MODE,PLAN,MANIFEST,SOURCE_TRANSPORT_BYTES
from .composite_bounded_sampling_fit_source import validate_source,notebook_code,validate_fit_package

QUALIFICATION='bounded_sampling_stateful_qualification_v1'
FULL_FIT='bounded_sampling_full_fit_v1'
PACKAGE='BOUNDED_SAMPLING_FIT_PACKAGE.zip'
NOTEBOOK_FILE='research4_bounded_sampling_fit_v1.ipynb'


def validate_preparation(path):
    path=Path(path).resolve();value=read(path);base=path.parent;source=base/'bundle/source'
    phase=value.get('phase')
    if (phase not in ['qualification','full_fit'] or value.get('mode')!=MODE
            or value.get('status')!='prepared_bounded_behavior_'+phase+'_not_submitted'
            or value.get('owner_thread_id')!='01a175f0-fc99-76b1-8bb2-d64f22afbcd7'
            or value.get('scientific_allocation_admitted_by_preparation') is not False
            or value.get('model_forwards')!=0 or value.get('optimizer_updates')!=0
            or value.get('heldout_content_read') is not False):
        raise PermissionError('Exact new reviewed V4 phase preparation only')
    stage,_,data=validate_source(source,value['source_commit'],value.get('qualification_archive_local_path'),
      data_archive=value['data_archive_local_path'],alias_fit_archive=value['alias_fit_archive_local_path'],
      alias_qualification_archive=value['alias_qualification_archive_local_path'])
    for member,key in [(PACKAGE,'bundle_sha256'),('bundle/'+MANIFEST,'bundle_manifest_sha256'),
      (NOTEBOOK_FILE,'notebook_file_sha256'),('bundle/source/inputs/'+PLAN,'plan_sha256'),
      ('bundle/source/inputs/MANIFEST.json','input_manifest_sha256')]:
        if sha(base/member)!=value[key]:raise PermissionError('Frozen V4 fitting package/source/input bytes changed')
    validate_fit_package(base/PACKAGE,base/'bundle')
    code=[''.join(c['source']) for c in read(base/NOTEBOOK_FILE)['cells'] if c['cell_type']=='code']
    if (code!=[notebook_code(source,value['source_commit'],sha(base/'bundle'/MANIFEST))]
            or value['notebook']!=stage['notebook'] or phase!=stage['phase']
            or value['training_states']!=len(data.fitting_rows())
            or value['source_data_archive_sha256']!=stage['source_data_archive_sha256']
            or value['data_audit_canonical_sha256']!=stage['data_audit_canonical_sha256']):
        raise PermissionError('V4 notebook/full new data/phase binding differs')
    return {'source_commit':value['source_commit'],'notebook':stage['notebook'],
      'mode':MODE,'phase':phase,'owner_thread_id':stage['owner_thread_id'],
      'package_sha256':value['bundle_sha256'],'authorization_sha256':value['plan_sha256'],
      'source_data_archive_sha256':stage['source_data_archive_sha256'],
      'data_audit_canonical_sha256':stage['data_audit_canonical_sha256'],
      'kaggle_remainder_scope_sha256':HUMAN_SHA,'provider_scope':'human-selected-Kaggle-remainder-v1',
      'fresh_stage':True,'scientific_GPU_count':2,'optimizer_updates':stage['optimizer_updates'],
      'full_fit_completed':False,'hard_timeout_seconds':stage['hard_timeout_seconds'],
      'automatic_scientific_retries':0}


def reserve_stage(campaign_root,preparation_file,live_snapshot,guard_token=None):
    root=Path(campaign_root);lock,token,_=guard(root,guard_token)
    try:
        binding=validate_preparation(preparation_file)
        phase=binding['phase'];prefix='SAMPLING_DATA' if phase=='qualification' else 'SAMPLING_QUAL'
        from .composite_bounded_sampling_admission import CANDIDATE as DATA_CANDIDATE
        parent_key=DATA_CANDIDATE if phase=='qualification' else QUALIFICATION
        ledger=read(root/'SCIENTIFIC_EXECUTION_LEDGER.json');parent=ledger['candidates'].get(parent_key)
        expected=read(Path(preparation_file).parent/'bundle/source/inputs'/(prefix+'_PARENT_EXPECTED.json'))
        if (not parent or parent.get('owned_compute_released') is not True
                or not parent.get('closeout_receipt') or sha(parent['closeout_receipt'])!=expected['closeout_sha256']):
            raise PermissionError('Exact fully raw-audited and released parent must also reconcile in the shared ledger')
    except Exception:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()
        raise
    candidate=QUALIFICATION if phase=='qualification' else FULL_FIT
    return reserve(root,preparation_file,live_snapshot,candidate,'attempts','r4-bounded-sampling-a1-'+phase+'-',
      validate_preparation,candidate_rule,token)


def reconcile_stage(campaign_root,identity,phase,script_version_id=None,guard_token=None):
    if phase not in ['qualification','full_fit']:raise PermissionError('Exact registered V4 phase required')
    return reconcile(campaign_root,identity,QUALIFICATION if phase=='qualification' else FULL_FIT,
      'attempts',script_version_id,guard_token)
