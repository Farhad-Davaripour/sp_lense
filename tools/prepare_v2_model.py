"""Inert V2 model-stage preparation/template; no provider or model calls."""
import argparse,json,shutil,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.io_utils import read,dump,sha,digest
from src.composite_source_archive import archive_source
from src.workspace_accounts import selected,artifact_volume
from src.composite_v2_model_contract import MODE,validate_model_metadata
from src.composite_v2_training import validate_plan_metadata
from src.composite_v2_data_contract import ORIGINAL_MODEL_MANIFEST,source_fingerprint
from src.composite_generalized_interface import PROFILE
from src.composite_v2_raw_exports import PROFILE as RAW_PROFILE,RAW_BYTES,RESERVE_BYTES


def plan_template(cfg,design,commit,phase,workspace):
    bounds=cfg['bounds']
    return {'mode':MODE,'source_commit':commit,'phase':phase,'native_interface_profile':PROFILE,
        'optimizer_updates':1 if phase=='qualification' else 64,
        'sampling':'longest_complete_eligible_training_examples' if phase=='qualification' else 'registered_seeded_training_sampler',
        'qualification_warm_start_used':False,'validation_optimizer_updates':0,'heldout_optimizer_updates':0,
        'heldout_content_read':False,'automatic_scientific_retries':0,'maximum_concurrent_owned_workers':1,
        'GPU_count':1,'accelerator_type':'GPU_1xA10','worker_seconds':bounds['worker_seconds'],
        'export_reserve_seconds':bounds['export_reserve_seconds'],'hard_timeout_seconds':bounds['hard_timeout_minutes']*60,
        'maximum_backbone_forwards':bounds['maximum_backbone_forwards_per_run'],
        'raw_evidence_profile':RAW_PROFILE,'maximum_raw_evidence_bytes':RAW_BYTES,
        'physical_export_reserve_bytes':RESERVE_BYTES,
        'provider_binding_path':'model_generalized_v2/'+commit+'/'+phase+'_01/PROVIDER_BINDING.json',
        'protocol_sha256':digest({'base':cfg,'design':design}),
        'destination':workspace['artifact_volume']+'/composite_controller_v1/model_generalized_v2/'+commit+'/'+phase+'_01',
        'data_release_sha256':None,'data_export':{'path':None,'sha256':None},
        'allocation_authorized_by_this_template':False}


def prepare(repo,commit,output,account,phase,model_inputs=None,human_fitting=None,human_generalization=None,
            data_release=None,data_archive_path=None,data_archive_sha256=None,qualification_closeout=None,
            qwen_tensor_metadata=None,template_only=False):
    if phase not in ['qualification','full_fit']:raise ValueError('Only registered qualification/full_fit phases')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);source=output/'source';archive_source(repo,commit,source)
    profile=selected(account,environ={},env_file=Path(repo)/'.env')
    workspace={key:profile[key] for key in ['account','label','host','organization']};workspace['artifact_volume']=artifact_volume(account=account)
    cfg=read(source/'config/composite_controller_rank32_v2.json');design=read(source/'config/composite_generalization_amendment_v2.json')
    plan=plan_template(cfg,design,commit,phase,workspace)
    if template_only:
        dump(output/'MODEL_PLAN_TEMPLATE.json',plan)
        receipt={'status':'conditional_source_template_missing_actual_data_not_launchable','source_commit':commit,
            'phase':phase,'workspace':workspace,'allocation_authorized':False,'paid_submission_enabled':False,
            'actual_data_release_available':False,'compute_submissions':0,'pretrained_model_forwards':0,'heldout_content_read':False}
        dump(output/'PREPARATION.json',receipt);return receipt
    if any(value is None for value in [model_inputs,human_fitting,human_generalization,data_release,data_archive_path,data_archive_sha256]):
        raise PermissionError('Real admitted V2 data and both verified human scope receipts required')
    maintained=Path(__file__).resolve().parents[1]
    for name in ['tools/prepare_v2_model.py','src/composite_v2_model_worker.py','src/composite_v2_raw_exports.py',
                 'src/composite_v2_model_contract.py','src/composite_v2_parent_audit.py','src/composite_v2_model_audit.py']:
        if not (source/name).is_file() or (source/name).read_bytes().replace(b'\r\n',b'\n')!=(maintained/name).read_bytes().replace(b'\r\n',b'\n'):
            raise PermissionError('Model preparation must use its exact reviewed archived tool/runtime source')
    inputs=source/'inputs';inputs.mkdir(exist_ok=True);pins=Path(model_inputs)
    if sha(pins/'MANIFEST.json')!=ORIGINAL_MODEL_MANIFEST:raise ValueError('Original full model pin manifest changed')
    shutil.copyfile(pins/'MANIFEST.json',inputs/'PARENT_MODEL_INPUT_MANIFEST.json')
    for name in ['QWEN_PIN.json','MODERNBERT_PIN.json']:
        if sha(pins/name)!=read(pins/'MANIFEST.json')[name]:raise ValueError('Original model-file pin changed')
        shutil.copyfile(pins/name,inputs/name)
    if qwen_tensor_metadata is None:raise PermissionError('Complete original header-only tensor metadata required')
    metadata=read(qwen_tensor_metadata)
    if metadata.get('tensor_values_read') is not False or digest(metadata.get('tensors'))!='a2477ac2c15f9c198d5137204f00da050d8aa87716a3cc5f0156f00b84034f7f':
        raise PermissionError('Original complete tensor metadata changed')
    shutil.copyfile(qwen_tensor_metadata,inputs/'QWEN_TENSOR_METADATA.json')
    shutil.copyfile(human_fitting,inputs/'HUMAN_QWEN_FITTING_SCOPE.json')
    shutil.copyfile(human_generalization,inputs/'HUMAN_GENERALIZATION_SCOPE.json')
    dump(inputs/'WORKSPACE.json',workspace);data=read(data_release)
    if data['world_source_files']!=source_fingerprint(source):raise PermissionError('Actual dataset source/interface changed')
    plan.update(data_release_sha256=digest(data),data_export={'path':data_archive_path,'sha256':data_archive_sha256})
    if phase=='full_fit':
        if qualification_closeout is None:raise PermissionError('Raw-audited released scratch qualification required')
        plan['qualification']=read(qualification_closeout)
    validate_plan_metadata(plan,cfg,data,workspace)
    dump(inputs/'V2_MODEL_STAGE.json',plan)
    dump(inputs/'MANIFEST.json',{p.name:sha(p) for p in inputs.iterdir() if p.name!='MANIFEST.json'})
    validate_model_metadata(source,commit)
    package=output/'V2_MODEL_SOURCE.zip'
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as archive:
        for path in source.rglob('*'):
            if path.is_file():archive.write(path,path.relative_to(source).as_posix())
    receipt={'status':'prepared_model_source_not_deployed_not_submitted','source_commit':commit,'phase':phase,
        'workspace':workspace,'package_sha256':sha(package),'input_manifest_sha256':sha(inputs/'MANIFEST.json'),
        'allocation_authorized':False,'paid_submission_enabled':False,'worst_case_cost_CAD':None,
        'runtime_parent_raw_reaudit_required':phase=='full_fit','scientific_fit_admitted':False,
        'compute_submissions':0,'pretrained_model_forwards':0,'heldout_content_read':False}
    dump(output/'PREPARATION.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['repo','commit','output','account','phase']:p.add_argument('--'+name,required=True)
    for name in ['model-inputs','human-fitting','human-generalization','data-release','data-archive-path','data-archive-sha256','qualification-closeout','qwen-tensor-metadata']:p.add_argument('--'+name)
    p.add_argument('--template-only',action='store_true')
    print(json.dumps(prepare(**vars(p.parse_args()))))
