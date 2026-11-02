"""Shared-ledger one-use V4 stage reservation. This module submits nothing."""
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_submission_admission import guard,reserve,reconcile
from .composite_kaggle_v2_admission import candidate_rule
from .composite_kaggle_v2_profile import HUMAN_SHA
from .composite_bounded_sampling_fit_profile import (MODE,DIRECT_MODE,PLAN,MANIFEST,SOURCE_TRANSPORT_BYTES,WAIVER_SHA)
from .composite_bounded_sampling_fit_source import validate_source,notebook_code,validate_fit_package

QUALIFICATION='bounded_sampling_stateful_qualification_v1'
FULL_FIT='bounded_sampling_full_fit_v1'
DIRECT_FULL_FIT='bounded_sampling_direct_full_fit_waiver_v1'
PACKAGE='BOUNDED_SAMPLING_FIT_PACKAGE.zip'
NOTEBOOK_FILE='research4_bounded_sampling_fit_v1.ipynb'


def validate_preparation(path):
    path=Path(path).resolve();value=read(path);base=path.parent;source=base/'bundle/source'
    phase=value.get('phase')
    direct=value.get('mode')==DIRECT_MODE
    if (phase not in ['qualification','full_fit'] or value.get('mode') not in [MODE,DIRECT_MODE]
            or value.get('status')!='prepared_bounded_behavior_'+phase+'_not_submitted'
            or value.get('owner_thread_id')!='01a175f0-fc99-76b1-8bb2-d64f22afbcd7'
            or value.get('scientific_allocation_admitted_by_preparation') is not False
            or value.get('model_forwards')!=0 or value.get('optimizer_updates')!=0
            or value.get('heldout_content_read') is not False):
        raise PermissionError('Exact new reviewed V4 phase preparation only')
    if direct and (phase!='full_fit' or value.get('separate_qualification_waived') is not True
            or value.get('separate_qualification_performed') is not False
            or value.get('direct_fit_waiver_sha256')!=WAIVER_SHA
            or value.get('qualification_archive_local_path') is not None):
        raise PermissionError('Direct fitting requires its exact human waiver and no qualification model parent')
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
            or value['mode']!=stage['version']
            or value['notebook']!=stage['notebook'] or phase!=stage['phase']
            or value['training_states']!=len(data.fitting_rows())
            or value['source_data_archive_sha256']!=stage['source_data_archive_sha256']
            or value['data_audit_canonical_sha256']!=stage['data_audit_canonical_sha256']):
        raise PermissionError('V4 notebook/full new data/phase binding differs')
    result={'source_commit':value['source_commit'],'notebook':stage['notebook'],
      'mode':stage['version'],'phase':phase,'owner_thread_id':stage['owner_thread_id'],
      'package_sha256':value['bundle_sha256'],'authorization_sha256':value['plan_sha256'],
      'source_data_archive_sha256':stage['source_data_archive_sha256'],
      'data_audit_canonical_sha256':stage['data_audit_canonical_sha256'],
      'kaggle_remainder_scope_sha256':HUMAN_SHA,'provider_scope':'human-selected-Kaggle-remainder-v1',
      'fresh_stage':True,'scientific_GPU_count':2,'optimizer_updates':stage['optimizer_updates'],
      'full_fit_completed':False,'hard_timeout_seconds':stage['hard_timeout_seconds'],
      'automatic_scientific_retries':0}
    if direct:
        closeout=_interrupted_closeout(value)
        result.update(separate_qualification_waived=True,separate_qualification_performed=False,
          direct_fit_waiver_sha256=WAIVER_SHA,interrupted_qualification_run_id=closeout['scriptVersionId'],
          interrupted_qualification_closeout_sha256=value['interrupted_qualification_closeout_sha256'])
    return result


def _interrupted_closeout(preparation):
    """Resource evidence only; no qualification tensor or capacity authority."""
    path=Path(preparation['interrupted_qualification_closeout_local_path'])
    if sha(path)!=preparation['interrupted_qualification_closeout_sha256']:
        raise PermissionError('The actual interrupted qualification closeout changed')
    closed=read(path)
    if (preparation.get('interrupted_qualification_run_id')!=362573972
            or type(closed.get('scriptVersionId')) is not int or closed['scriptVersionId']!=362573972
            or closed.get('provider_terminal') is not True
            or closed.get('owned_compute_released') is not True):
        raise PermissionError('The exact interrupted qualification must be terminal and independently released')
    return closed


def reserve_stage(campaign_root,preparation_file,live_snapshot,guard_token=None):
    root=Path(campaign_root);lock,token,_=guard(root,guard_token)
    try:
        binding=validate_preparation(preparation_file)
        phase=binding['phase'];direct=binding['mode']==DIRECT_MODE
        prefix='SAMPLING_DATA' if phase=='qualification' or direct else 'SAMPLING_QUAL'
        from .composite_bounded_sampling_admission import CANDIDATE as DATA_CANDIDATE
        parent_key=DATA_CANDIDATE if phase=='qualification' or direct else QUALIFICATION
        ledger=read(root/'SCIENTIFIC_EXECUTION_LEDGER.json');parent=ledger['candidates'].get(parent_key)
        expected=read(Path(preparation_file).parent/'bundle/source/inputs'/(prefix+'_PARENT_EXPECTED.json'))
        if (not parent or parent.get('owned_compute_released') is not True
                or not parent.get('closeout_receipt') or sha(parent['closeout_receipt'])!=expected['closeout_sha256']):
            raise PermissionError('Exact fully raw-audited and released parent must also reconcile in the shared ledger')
        if direct:
            value=read(preparation_file);closed=_interrupted_closeout(value)
            interrupted=ledger['candidates'].get(QUALIFICATION,{})
            attempts=interrupted.get('attempts',[])
            if (interrupted.get('owned_compute_released') is not True
                    or not interrupted.get('closeout_receipt')
                    or sha(interrupted['closeout_receipt'])!=binding['interrupted_qualification_closeout_sha256']
                    or len(attempts)!=1 or attempts[0].get('scriptVersionId')!=closed['scriptVersionId']
                    or attempts[0].get('authority_consumed') is not True
                    or attempts[0].get('source_commit')!=closed.get('source_commit')):
                raise PermissionError('The consumed exact qualification identity and released closeout must reconcile before direct fitting')
    except Exception:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()
        raise
    candidate=DIRECT_FULL_FIT if direct else QUALIFICATION if phase=='qualification' else FULL_FIT
    prefix='r4-bounded-sampling-direct-v1-' if direct else 'r4-bounded-sampling-a1-'+phase+'-'
    return reserve(root,preparation_file,live_snapshot,candidate,'attempts',prefix,
      validate_preparation,candidate_rule,token)


def reconcile_stage(campaign_root,identity,phase,script_version_id=None,guard_token=None,mode=MODE):
    if phase not in ['qualification','full_fit']:raise PermissionError('Exact registered V4 phase required')
    if mode not in [MODE,DIRECT_MODE] or (mode==DIRECT_MODE and phase!='full_fit'):
        raise PermissionError('Exact registered fitting protocol and phase required')
    candidate=DIRECT_FULL_FIT if mode==DIRECT_MODE else QUALIFICATION if phase=='qualification' else FULL_FIT
    return reconcile(campaign_root,identity,candidate,
      'attempts',script_version_id,guard_token)
