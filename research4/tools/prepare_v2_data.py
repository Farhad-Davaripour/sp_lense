"""Prepare an exact source/workspace data-only package; no deployment or APIs."""
import argparse,json,shutil,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.io_utils import read,dump,sha
from src.composite_source_archive import archive_source
from src.workspace_accounts import selected,artifact_volume
from src.composite_v2_data_contract import (MODE,PLAN,HUMAN,AUTHORITY,BOUNDS,ORIGINAL_MODEL_MANIFEST,validate_metadata)
from src.composite_generalized_data import cohort_contract
from src.composite_generalized_world import VERSION


def prepare(repo,commit,model_inputs,human_scope,output,account,notebook=None,kaggle_scope=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=output/'source';archive_source(repo,commit,source)
    inputs=source/'inputs';inputs.mkdir(exist_ok=True)
    model_inputs=Path(model_inputs);parent=model_inputs/'MANIFEST.json'
    if sha(parent)!=ORIGINAL_MODEL_MANIFEST:raise ValueError('Use the unchanged original model input freeze')
    shutil.copyfile(parent,inputs/'PARENT_MODEL_INPUT_MANIFEST.json')
    for name in ['QWEN_PIN.json','MODERNBERT_PIN.json']:
        if sha(model_inputs/name)!=read(parent)[name]:raise ValueError('Original model pin differs')
        shutil.copyfile(model_inputs/name,inputs/name)
    shutil.copyfile(human_scope,inputs/HUMAN)
    if account=='kaggle':
        from src.composite_kaggle_v2_profile import identity,HUMAN as KAGGLE_HUMAN,HUMAN_SHA
        if kaggle_scope is None or sha(kaggle_scope)!=HUMAN_SHA:raise PermissionError('Verified direct human Kaggle scope required')
        shutil.copyfile(kaggle_scope,inputs/KAGGLE_HUMAN);workspace=identity(notebook)
    else:
        profile=selected(account,environ={},env_file=Path(repo)/'.env')
        workspace={key:profile[key] for key in ['account','label','host','organization']}
        workspace['artifact_volume']=artifact_volume(account=account)
    dump(inputs/'WORKSPACE.json',workspace)
    cfg=read(source/'config/composite_generalization_amendment_v2.json')
    plan={'version':MODE,'roles':['training','development'],
        'contracts':{role:cohort_contract(cfg,role) for role in ['training','development']},
        'identity_material':'research4-generalized-v2-20261024-cohorts-v1','model_weights_downloaded':False,
        'model_forwards':0,'optimizer_updates':0,'sealed_instances_materialized':False,**BOUNDS}
    dump(inputs/PLAN,plan)
    if account=='kaggle':
        from src.composite_kaggle_v2_profile import data_destination,PROFILE,FILE,HUMAN_SHA
        destination=data_destination(workspace,commit)
        dump(inputs/FILE,{'version':PROFILE,'platform':'kaggle','account':'farhad89','notebook':notebook,
            'source_commit':commit,'mode':MODE,'visible_devices':'','scientific_GPU_count':0,
            'hard_timeout_seconds':900,'worker_seconds':600,'export_reserve_seconds':60,
            'maximum_raw_export_bytes':256*1024**2,'maximum_concurrent_owned_workers':1,
            'automatic_scientific_retries':0,'heldout_content_read':False,'human_scope_sha256':HUMAN_SHA,'destination':destination})
    else:destination=workspace['artifact_volume']+'/composite_controller_v1/data_generalized_v2/'+commit+'/export_01'
    authority={'status':'explicit_human_generalization_continuation','authorized_by':'human_user',
        'human_user_message_id':read(inputs/HUMAN)['human_user_message_id'],'source_commit':commit,'world_version':VERSION,
        'mode':MODE,'roles':['training','development'],'heldout_content_read':False,'maximum_episode_ticks':24,
        'maximum_concurrent_owned_workers':1,'automatic_scientific_retries':0,
        'human_scope_sha256':sha(inputs/HUMAN),'plan_sha256':sha(inputs/PLAN),
        'workspace_descriptor_sha256':sha(inputs/'WORKSPACE.json'),
        'amendment_sha256':sha(source/'config/composite_generalization_amendment_v2.json'),'destination':destination}
    dump(inputs/AUTHORITY,authority)
    dump(inputs/'MANIFEST.json',{file.name:sha(file) for file in inputs.iterdir() if file.name!='MANIFEST.json'})
    context=validate_metadata(source,commit)
    package=output/'V2_DATA_SOURCE.zip'
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as archive:
        for file in source.rglob('*'):
            if file.is_file():archive.write(file,file.relative_to(source).as_posix())
    receipt={'status':'prepared_data_source_not_deployed_not_submitted','source_commit':commit,
        'mode':MODE,'workspace':workspace,'destination':destination,'package_sha256':sha(package),
        'input_manifest_sha256':sha(inputs/'MANIFEST.json'),'authorization_sha256':context['authorization_sha256'],
        'bounds':BOUNDS,'command':['python','-m','src.composite_v2_data_worker','--root','$CODE_SOURCE_PATH','--commit',commit],
        'scientific_allocation_authorized':False,'GPU_required':False,'compute_submissions':0,
        'actual_scientific_cohorts_generated':False,'worst_case_cost_CAD':None,'paid_submission_enabled':False,
        'original_heldout_or_new_seal_in_package':False}
    dump(output/'PREPARATION.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['repo','commit','model-inputs','human-scope','output','account']:p.add_argument('--'+name,required=True)
    p.add_argument('--notebook');p.add_argument('--kaggle-scope')
    print(json.dumps(prepare(**vars(p.parse_args()))))
