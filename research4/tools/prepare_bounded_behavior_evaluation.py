"""Freeze one prospectively registered actor-only evaluation partition; never submit."""
import argparse,base64,json,shutil,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.io_utils import read,dump,sha,digest
from src.composite_source_archive import archive_source
from src.composite_kaggle_v2_profile import HUMAN,HUMAN_SHA,identity
from src.composite_bounded_behavior_evaluation_profile import (MODE,STAGE,MANIFEST,SOURCE_TRANSPORT_BYTES,
    PROBE_REGISTRATION,expected_stage,execution_plan,competence_roster)
from src.composite_bounded_behavior_evaluation_parent import read_parent
from src.composite_bounded_behavior_evaluation_source import validate_source,notebook_code,validate_package

PACKAGE='BOUNDED_ACTOR_EVALUATION_PACKAGE.zip'
NOTEBOOK='research4_bounded_actor_evaluation_v4.ipynb'


def prepare(repo,commit,fit_parent,fit_archive,qualification_archive,data_archive,
            alias_fit_archive,alias_qualification_archive,kaggle_scope,endpoint,index,output):
    if sha(kaggle_scope)!=HUMAN_SHA:raise PermissionError('Existing selected Kaggle authority required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);bundle=out/'bundle';source=bundle/'source'
    archive_source(repo,commit,source);inputs=source/'inputs';inputs.mkdir(exist_ok=True)
    parent=Path(fit_parent);prefix='BOUNDED_FIT_PARENT_'
    from src.composite_bounded_sampling_fit_bridge import fit_modules
    parent_mode=read(parent/'PREPARATION.json')['mode']
    fit_profile,_,_=fit_modules(parent_mode)
    from src.composite_bounded_sampling_fit_profile import MODE as SAMPLING_MODE,DIRECT_MODE
    direct=parent_mode==DIRECT_MODE
    if direct and qualification_archive is not None:raise PermissionError('Direct evaluation cannot attach a synthetic qualification archive')
    prepared_name='research4_bounded_sampling_fit_v1.ipynb' if parent_mode in [SAMPLING_MODE,DIRECT_MODE] else 'research4_bounded_behavior_fit_v4.ipynb'
    mapping={'CLOSEOUT.json':'CLOSEOUT.json','SUBMISSION.json':'BINDING.json',
      'CONTROLLER_LOG_READBACK.json':'CONTROLLER.json','TERMINAL_SUCCESS_AX.txt':'TERMINAL_AX.txt',
      'RELEASE_AX.txt':'RELEASE_AX.txt','ACTUAL_SUBMITTED_NOTEBOOK.ipynb':'ACTUAL.ipynb',
      prepared_name:'PREPARED.ipynb'}
    for before,after in mapping.items():shutil.copyfile(parent/before,inputs/(prefix+after))
    package=parent/fit_profile.MANIFEST.replace('_PACKAGE_FILES.json','_PACKAGE.zip')
    (inputs/(prefix+'SOURCE.r4b64')).write_bytes(base64.b64encode(package.read_bytes()))
    submission=read(parent/'SUBMISSION.json');audited=read(parent/'INDEPENDENT_AUDIT.json')
    fit_stage=read(parent/'bundle/source/inputs'/fit_profile.PLAN)
    dump(inputs/(prefix+'EXPECTED.json'),{'source_commit':submission['source_commit'],'run_id':submission['scriptVersionId'],
      'package_sha256':sha(package),'raw_archive_sha256':sha(fit_archive),'closeout_sha256':sha(parent/'CLOSEOUT.json'),
      'fit_mode':parent_mode,
      'checkpoint_sha256':audited['checkpoint_sha256'],'qualification_source_commit':fit_stage.get('qualification_source_commit'),
      'data_source_commit':fit_stage['source_data_source_commit'],
      'dependencies_sha256':{name:sha(path) for name,path in zip(['fit','qualification','data','alias_fit','alias_qualification'],
        [fit_archive,qualification_archive,data_archive,alias_fit_archive,alias_qualification_archive]) if path is not None}})
    model=read_parent(source,fit_archive,qualification_archive,data_archive,alias_fit_archive,alias_qualification_archive)
    cfg=read(source/'config/composite_bounded_behavior_v4.json')
    roster=model.data.scenes('development') if endpoint=='behavior' else competence_roster(cfg)
    stage=expected_stage(commit,model,model.data,cfg,endpoint,index)
    for name in ['QWEN_PIN.json','QWEN_TENSOR_METADATA.json']:dump(inputs/name,model.inputs[name])
    shutil.copyfile(kaggle_scope,inputs/HUMAN)
    dump(inputs/STAGE,stage);dump(inputs/'WORKSPACE.json',identity(stage['notebook']))
    dump(inputs/'EVALUATION_PLAN.json',execution_plan(roster,endpoint));dump(inputs/'EVALUATION_ROSTER.json',roster)
    dump(inputs/'PROBE_REGISTRATION.json',PROBE_REGISTRATION)
    dump(inputs/'MANIFEST.json',{p.name:sha(p) for p in sorted(inputs.iterdir()) if p.name!='MANIFEST.json'})
    dump(bundle/MANIFEST,{'version':MODE,'source_commit':commit,'notebook':stage['notebook'],
      'files':{p.relative_to(bundle).as_posix():sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}})
    size=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    if size>SOURCE_TRANSPORT_BYTES:raise PermissionError('Complete frozen actor source transport exceeds256MiB')
    validate_source(source,commit,fit_archive,qualification_archive,data_archive,alias_fit_archive,alias_qualification_archive)
    archive=out/PACKAGE
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as stream:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():stream.write(p,p.relative_to(bundle).as_posix())
    validate_package(archive,bundle)
    code=notebook_code(source,commit,sha(bundle/MANIFEST))
    dump(out/NOTEBOOK,{'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
      'source':code.splitlines(keepends=True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
      'nbformat':4,'nbformat_minor':5})
    result={'status':'prepared_frozen_bounded_actor_partition_not_submitted','mode':MODE,'source_commit':commit,
      'endpoint':endpoint,'phase':stage['phase'],'partition_index':index,'owner_thread_id':stage['owner_thread_id'],
      'notebook':stage['notebook'],'bundle_sha256':sha(archive),'bundle_manifest_sha256':sha(bundle/MANIFEST),
      'notebook_file_sha256':sha(out/NOTEBOOK),'stage_sha256':sha(inputs/STAGE),'input_manifest_sha256':sha(inputs/'MANIFEST.json'),
      'source_transport_expanded_bytes':size,'source_transport_maximum_bytes':SOURCE_TRANSPORT_BYTES,
      'model_forwards':0,'optimizer_updates':0,'heldout_content_read':False,
      'scientific_allocation_admitted_by_preparation':False,
      'fit_archive_local_path':str(Path(fit_archive).resolve()),'qualification_archive_local_path':str(Path(qualification_archive).resolve()) if qualification_archive else None,
      'data_archive_local_path':str(Path(data_archive).resolve()),'alias_fit_archive_local_path':str(Path(alias_fit_archive).resolve()),
      'alias_qualification_archive_local_path':str(Path(alias_qualification_archive).resolve())}
    dump(out/'PREPARATION.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['repo','commit','fit-parent','fit-archive','data-archive',
      'alias-fit-archive','alias-qualification-archive','kaggle-scope','endpoint','output']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--qualification-archive',default=None)
    parser.add_argument('--index',required=True,type=int);print(json.dumps(prepare(**vars(parser.parse_args()))))
