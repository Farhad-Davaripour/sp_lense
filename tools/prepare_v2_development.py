"""Inert source-bound transfer/native development preparation; no submission."""
import argparse,json,shutil,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.io_utils import read,dump,sha,digest
from src.composite_source_archive import archive_source
from src.workspace_accounts import selected,artifact_volume
from src.composite_generalized_interface import PROFILE
from src.composite_v2_development_contract import (MODE,PLAN,COLLECTION_BOUNDS,TRANSFER_BOUNDS,partition_plan,validate_stage)


def stage_template(commit,phase,workspace,cfg,design,index=None):
    if phase not in ['detector_transfer','development_partition']:raise ValueError('Only the two registered development phases')
    if phase=='development_partition' and (type(index) is not int or not 0<=index<36):raise ValueError('Fixed partition index 0..35 required')
    leaf='detector_transfer_01' if phase=='detector_transfer' else 'partition_'+str(index).zfill(2)
    relative='development_generalized_v2/'+commit+'/'+leaf
    return {'mode':MODE,'phase':phase,'source_commit':commit,'native_interface_profile':PROFILE,
        'bounds':dict(TRANSFER_BOUNDS if phase=='detector_transfer' else COLLECTION_BOUNDS),
        'threshold':0.9,'detector_checkpoint_sha256':design['detector']['checkpoint_sha256'],
        'heldout_content_read':False,'optimizer_updates':0,'development_roster':None,
        'partition_index':index,'partition':None,'data_export':{'path':None,'sha256':None},
        'data_release_sha256':None,'detector_checkpoint':{'path':None,'sha256':design['detector']['checkpoint_sha256']},
        'package':None,'full_fit_parent':None,'model_checkpoint':None,
        'protocol_sha256':digest({'base':cfg,'design':design}),
        'destination':workspace['artifact_volume']+'/composite_controller_v1/'+relative,
        'provider_binding_path':relative+'/PROVIDER_BINDING.json','allocation_authorized_by_this_template':False}


def prepare(repo,commit,output,account,phase,index=None,base_source=None,bindings=None,template_only=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);source=output/'source';archive_source(repo,commit,source)
    profile=selected(account,environ={},env_file=Path(repo)/'.env')
    workspace={k:profile[k] for k in ['account','label','host','organization']};workspace['artifact_volume']=artifact_volume(account=account)
    cfg=read(source/'config/composite_controller_rank32_v2.json');design=read(source/'config/composite_generalization_amendment_v2.json')
    stage=stage_template(commit,phase,workspace,cfg,design,index)
    if template_only:
        dump(output/'DEVELOPMENT_PLAN_TEMPLATE.json',stage)
        result={'status':'conditional_development_source_template_not_launchable','source_commit':commit,'phase':phase,
            'partition_index':index,'workspace':workspace,'actual_data_or_model_admitted':False,
            'allocation_authorized':False,'paid_submission_enabled':False,'pretrained_model_forwards':0,'heldout_content_read':False}
        dump(output/'PREPARATION.json',result);return result
    if base_source is None or bindings is None:raise PermissionError('Exact same-source parent inputs and actual data/model bindings required')
    base_source=Path(base_source)
    if phase=='detector_transfer':
        from src.composite_v2_data_contract import validate_metadata
        parent=validate_metadata(base_source,commit)
    else:
        from src.composite_v2_model_contract import validate_model_metadata
        parent=validate_model_metadata(base_source,commit)
    if parent['workspace']!=workspace:raise PermissionError('Parent/preparation workspace differs')
    current=Path(__file__).resolve().parents[1]
    for name in ['tools/prepare_v2_development.py','src/composite_v2_development_contract.py',
        'src/composite_v2_development_worker.py','src/composite_v2_development_audit.py','src/composite_v2_detector_transfer.py']:
        if (source/name).read_bytes().replace(b'\r\n',b'\n')!=(current/name).read_bytes().replace(b'\r\n',b'\n'):
            raise PermissionError('Preparation tool/runtime source must match the exact archived commit')
    inputs=source/'inputs';inputs.mkdir(exist_ok=True)
    for name,value in read(base_source/'inputs/MANIFEST.json').items():
        if Path(name).name!=name or sha(base_source/'inputs'/name)!=value:raise PermissionError('Parent input changed')
        shutil.copyfile(base_source/'inputs'/name,inputs/name)
    actual=read(bindings)
    permitted={'development_roster','data_export','data_release_sha256','detector_checkpoint','package','full_fit_parent','model_checkpoint'}
    if set(actual)-permitted:raise PermissionError('Bindings cannot override source/controls/bounds/threshold/ownership')
    stage.update(actual)
    plan=partition_plan(stage['development_roster'],design)
    if phase=='development_partition':stage['partition']=plan['partitions'][index]
    dump(inputs/PLAN,stage);dump(inputs/'MANIFEST.json',{p.name:sha(p) for p in inputs.iterdir() if p.name!='MANIFEST.json'})
    validate_stage(source,commit)
    archive=output/'V2_DEVELOPMENT_SOURCE.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as stream:
        for path in source.rglob('*'):
            if path.is_file():stream.write(path,path.relative_to(source).as_posix())
    result={'status':'prepared_development_source_not_deployed_not_submitted','source_commit':commit,
        'phase':phase,'partition_index':index,'workspace':workspace,'package_sha256':sha(archive),
        'input_manifest_sha256':sha(inputs/'MANIFEST.json'),'roster_sha256':plan['roster_sha256'],
        'allocation_authorized':False,'paid_submission_enabled':False,'worst_case_cost_CAD':None,
        'runtime_throughput_qualified':False,'pretrained_model_forwards':0,'heldout_content_read':False}
    dump(output/'PREPARATION.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['repo','commit','output','account','phase']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--index',type=int);parser.add_argument('--base-source');parser.add_argument('--bindings')
    parser.add_argument('--template-only',action='store_true')
    print(json.dumps(prepare(**vars(parser.parse_args()))))
