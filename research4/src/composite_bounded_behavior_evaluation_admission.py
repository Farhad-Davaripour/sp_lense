"""One-use frozen actor partitions under the existing shared submission guard."""
from pathlib import Path
import math
from .io_utils import read,sha,digest
from .composite_submission_admission import guard,reserve,reconcile
from .composite_kaggle_v2_admission import candidate_rule
from .composite_kaggle_v2_profile import HUMAN_SHA
from .composite_bounded_behavior_evaluation_profile import MODE,MANIFEST,STAGE
from .composite_bounded_behavior_evaluation_source import validate_source,notebook_code,validate_package

PACKAGE='BOUNDED_ACTOR_EVALUATION_PACKAGE.zip'
NOTEBOOK='research4_bounded_actor_evaluation_v4.ipynb'


def candidate(endpoint,index):
    if endpoint not in ['behavior','competence'] or type(index) is not int or not 0<=index<13:
        raise PermissionError('Exact actor endpoint/partition required')
    return 'bounded_actor_'+endpoint+'_v4_partition_'+str(index).zfill(2)


def validate_preparation(path):
    path=Path(path).resolve();value=read(path);base=path.parent;source=base/'bundle/source'
    if (value.get('status')!='prepared_frozen_bounded_actor_partition_not_submitted' or value.get('mode')!=MODE
        or value.get('scientific_allocation_admitted_by_preparation') is not False
        or value.get('model_forwards')!=0 or value.get('optimizer_updates')!=0 or value.get('heldout_content_read') is not False):
        raise PermissionError('Exact new frozen actor partition preparation only')
    deps=[value[k+'_local_path'] for k in ['fit_archive','qualification_archive','data_archive','alias_fit_archive','alias_qualification_archive']]
    context=validate_source(source,value['source_commit'],*deps);stage=context.stage()
    for member,key in [(PACKAGE,'bundle_sha256'),('bundle/'+MANIFEST,'bundle_manifest_sha256'),
      (NOTEBOOK,'notebook_file_sha256'),('bundle/source/inputs/'+STAGE,'stage_sha256'),
      ('bundle/source/inputs/MANIFEST.json','input_manifest_sha256')]:
        if sha(base/member)!=value[key]:raise PermissionError('Frozen actor package/input/code bytes changed')
    validate_package(base/PACKAGE,base/'bundle')
    code=[''.join(c['source']) for c in read(base/NOTEBOOK)['cells'] if c['cell_type']=='code']
    if (code!=[notebook_code(source,value['source_commit'],sha(base/'bundle'/MANIFEST))]
        or value['endpoint']!=stage['endpoint'] or value['partition_index']!=stage['partition']['index']):
        raise PermissionError('Actual prepared actor notebook/source/partition differ')
    return {'source_commit':value['source_commit'],'mode':MODE,'phase':stage['phase'],'endpoint':stage['endpoint'],
      'partition_index':stage['partition']['index'],'notebook':stage['notebook'],'owner_thread_id':stage['owner_thread_id'],
      'package_sha256':value['bundle_sha256'],'authorization_sha256':value['stage_sha256'],
      'frozen_model_sha256':stage['frozen_model_sha256'],'checkpoint_sha256':stage['checkpoint_sha256'],
      'training_lineage':stage['training_lineage'],
      'source_data_archive_sha256':stage['source_data_archive_sha256'],'configuration_sha256':stage['configuration_sha256'],
      'roster_sha256':stage['roster_sha256'],'kaggle_remainder_scope_sha256':HUMAN_SHA,
      'provider_scope':'human-selected-Kaggle-remainder-v1','fresh_stage':True,'scientific_GPU_count':2,
      'hard_timeout_seconds':stage['hard_timeout_seconds'],'optimizer_updates':0,'automatic_scientific_retries':0}


def reserve_partition(campaign_root,preparation_file,live_snapshot,guard_token=None):
    root=Path(campaign_root);lock,token,_=guard(root,guard_token)
    try:
        if not callable(live_snapshot):raise PermissionError('Read fresh target provider state while the shared guard is held')
        binding=validate_preparation(preparation_file);endpoint=binding['endpoint'];index=binding['partition_index']
        key=candidate(endpoint,index);ledger=read(root/'SCIENTIFIC_EXECUTION_LEDGER.json')
        expected=read(Path(preparation_file).parent/'bundle/source/inputs/BOUNDED_FIT_PARENT_EXPECTED.json')
        from .composite_bounded_sampling_fit_profile import MODE as NEW_MODE
        from .composite_bounded_sampling_fit_admission import FULL_FIT as NEW_FULL_FIT
        parent=ledger['candidates'].get(NEW_FULL_FIT if expected.get('fit_mode')==NEW_MODE else 'bounded_behavior_full_fit_v4')
        if (not parent or parent.get('owned_compute_released') is not True or not parent.get('closeout_receipt')
            or sha(parent['closeout_receipt'])!=expected['closeout_sha256']):
            raise PermissionError('Actual complete audited/released V4 fit must reconcile in the shared ledger')
        if index>0:previous=candidate(endpoint,index-1)
        elif endpoint=='competence':previous=candidate('behavior',12)
        else:previous=None
        if previous is not None:
            prior=ledger['candidates'].get(previous)
            if (not prior or prior.get('owned_compute_released') is not True
                or prior.get('complete_partition_raw_audited') is not True):
                raise PermissionError('Every fixed predecessor partition must finish raw audit/release; task failures remain valid outcomes')
        for entry in ledger['candidates'].values():
            for attempt in entry.get('attempts',[]):
                if attempt.get('mode')==MODE and any(attempt.get(k)!=binding[k] for k in
                    ['source_commit','frozen_model_sha256','checkpoint_sha256','source_data_archive_sha256','configuration_sha256','training_lineage']):
                    raise PermissionError('Frozen model/source/data/criteria cannot change between actor partitions')
    except Exception:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()
        raise
    def observed():
        value=live_snapshot();actual=value.get('kaggle_observation',{})
        required={'account':'farhad89','notebook':binding['notebook'],'notebook_draft_sessions':0,
          'active_notebook_versions':0,'pending_notebook_versions':0,'events_checked':True,'accelerator':'T4x2'}
        if any(digest(actual.get(k))!=digest(v) for k,v in required.items()):
            raise PermissionError('Fresh same-account target notebook/version/draft/T4-pair evidence required')
        quota=actual.get('remaining_accelerator_seconds')
        if type(quota) not in [int,float] or not math.isfinite(quota) or quota<binding['hard_timeout_seconds']:
            raise PermissionError('Actual remaining quota must cover the complete registered actor stage')
        return value
    return reserve(root,preparation_file,observed,key,'attempts','r4-bounded-actor-'+endpoint+'-'+str(index).zfill(2)+'-',
      validate_preparation,candidate_rule,token)


def reconcile_partition(campaign_root,identity,endpoint,index,script_version_id=None,guard_token=None):
    return reconcile(campaign_root,identity,candidate(endpoint,index),'attempts',script_version_id,guard_token)
