"""Freeze a new V4 qualifier or fresh full fit; never allocate or fit."""
import argparse,base64,json,shutil,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.io_utils import read,dump,sha,digest
from src.composite_source_archive import archive_source
from src.composite_bounded_sampling_fit_profile import MODE,PLAN,MANIFEST,SOURCE_TRANSPORT_BYTES,qualification_plan,full_fit_plan
from src.composite_bounded_sampling_fit_source import validate_source,notebook_code,validate_fit_package
from src.composite_bounded_sampling_fit_parent import read_data_parent,read_qualification_parent
from src.composite_kaggle_v2_profile import HUMAN,HUMAN_SHA,identity
from src.composite_v2_data_contract import ORIGINAL_MODEL_MANIFEST

PACKAGE='BOUNDED_SAMPLING_FIT_PACKAGE.zip'
NOTEBOOK_FILE='research4_bounded_sampling_fit_v1.ipynb'


def copy_parent(inputs,directory,kind,package,raw):
    directory=Path(directory);prefix='SAMPLING_'+kind+'_PARENT_'
    mapping={'CLOSEOUT.json':'CLOSEOUT.json','SUBMISSION.json':'BINDING.json',
      'CONTROLLER_LOG_READBACK.json':'CONTROLLER.json','TERMINAL_SUCCESS_AX.txt':'TERMINAL_AX.txt',
      'RELEASE_AX.txt':'RELEASE_AX.txt','ACTUAL_SUBMITTED_NOTEBOOK.ipynb':'ACTUAL.ipynb'}
    mapping['research4_bounded_sampling_amendment_v1.ipynb' if kind=='DATA' else NOTEBOOK_FILE]='PREPARED.ipynb'
    for before,after in mapping.items():shutil.copyfile(directory/before,inputs/(prefix+after))
    encoded=directory/package
    (inputs/(prefix+'SOURCE.r4b64')).write_bytes(base64.b64encode(encoded.read_bytes()))
    bound=read(directory/'SUBMISSION.json')
    expected={'source_commit':bound['source_commit'],'run_id':bound['scriptVersionId'],
      'package_sha256':sha(encoded),'raw_archive_sha256':sha(raw),
      'closeout_sha256':sha(directory/'CLOSEOUT.json')}
    if kind=='QUAL':
        audited=read(directory/'INDEPENDENT_AUDIT.json')
        expected.update(checkpoint_sha256=audited['checkpoint_sha256'],
          initial_adapter_hashes_sha256=digest(audited['initial_adapter_hashes']))
    dump(inputs/(prefix+'EXPECTED.json'),expected)


def prepare(repo,commit,model_inputs,tensor_metadata,kaggle_scope,data_parent,data_archive,
            alias_fit_archive,alias_qualification_archive,output,qualification_parent=None,qualification_archive=None):
    if (qualification_parent is None)!=(qualification_archive is None):
        raise PermissionError('Both complete qualification raw and its actual source/closure are required')
    model_inputs=Path(model_inputs)
    if sha(model_inputs/'MANIFEST.json')!=ORIGINAL_MODEL_MANIFEST or sha(kaggle_scope)!=HUMAN_SHA:
        raise PermissionError('Original complete model-file freeze and selected Kaggle authority required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    bundle=output/'bundle';source=bundle/'source';archive_source(repo,commit,source)
    inputs=source/'inputs';inputs.mkdir(exist_ok=True)
    shutil.copyfile(model_inputs/'MANIFEST.json',inputs/'PARENT_MODEL_INPUT_MANIFEST.json')
    for name in ['QWEN_PIN.json','MODERNBERT_PIN.json']:
        if sha(model_inputs/name)!=read(model_inputs/'MANIFEST.json')[name]:raise PermissionError('Original model pin changed')
        shutil.copyfile(model_inputs/name,inputs/name)
    shutil.copyfile(tensor_metadata,inputs/'QWEN_TENSOR_METADATA.json')
    shutil.copyfile(kaggle_scope,inputs/HUMAN)
    if qualification_parent is None:
        copy_parent(inputs,data_parent,'DATA','BOUNDED_SAMPLING_AMENDMENT_PACKAGE.zip',data_archive)
        data=read_data_parent(source,data_archive,alias_fit_archive,alias_qualification_archive)
        stage=qualification_plan(commit,data)
    else:
        copy_parent(inputs,qualification_parent,'QUAL',PACKAGE,qualification_archive)
        qualification=read_qualification_parent(source,qualification_archive,data_archive=data_archive,
          alias_fit=alias_fit_archive,alias_qualification=alias_qualification_archive)
        data=qualification.data
        stage=full_fit_plan(commit,data,qualification)
    dump(inputs/PLAN,stage);dump(inputs/'WORKSPACE.json',identity(stage['notebook']))
    dump(inputs/'MANIFEST.json',{p.name:sha(p) for p in inputs.iterdir() if p.name!='MANIFEST.json'})
    dump(bundle/MANIFEST,{'version':MODE,'source_commit':commit,'notebook':stage['notebook'],
      'files':{p.relative_to(bundle).as_posix():sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}})
    transport_bytes=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    if transport_bytes>SOURCE_TRANSPORT_BYTES:raise PermissionError('Complete nested V4 source/input transport exceeds its explicit256MiB cap')
    validate_source(source,commit,qualification_archive,data_archive=data_archive,
      alias_fit_archive=alias_fit_archive,alias_qualification_archive=alias_qualification_archive)
    archive=output/PACKAGE
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as stream:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():stream.write(p,p.relative_to(bundle).as_posix())
    validate_fit_package(archive,bundle)
    code=notebook_code(source,commit,sha(bundle/MANIFEST))
    notebook=output/NOTEBOOK_FILE
    dump(notebook,{'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
      'source':code.splitlines(keepends=True)}],
      'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
      'nbformat':4,'nbformat_minor':5})
    value={'status':'prepared_bounded_behavior_'+stage['phase']+'_not_submitted',
      'mode':MODE,'phase':stage['phase'],'source_commit':commit,'notebook':stage['notebook'],
      'owner_thread_id':stage['owner_thread_id'],'platform':'kaggle','account':'farhad89',
      'source_transport_expanded_bytes':transport_bytes,'source_transport_maximum_bytes':SOURCE_TRANSPORT_BYTES,
      'bundle_sha256':sha(archive),'bundle_manifest_sha256':sha(bundle/MANIFEST),
      'notebook_file_sha256':sha(notebook),'plan_sha256':sha(inputs/PLAN),
      'input_manifest_sha256':sha(inputs/'MANIFEST.json'),
      'source_data_archive_sha256':stage['source_data_archive_sha256'],
      'data_audit_canonical_sha256':stage['data_audit_canonical_sha256'],
      'training_states':len(data.fitting_rows()),'scientific_GPU_count':2,
      'data_archive_local_path':str(Path(data_archive).resolve()),
      'alias_fit_archive_local_path':str(Path(alias_fit_archive).resolve()),
      'alias_qualification_archive_local_path':str(Path(alias_qualification_archive).resolve()),
      'qualification_archive_local_path':str(Path(qualification_archive).resolve()) if qualification_archive else None,
      'qualification_checkpoint_initializes_full_fit':False,'model_forwards':0,'optimizer_updates':0,
      'heldout_content_read':False,'full_fit_completed':False,
      'scientific_allocation_admitted_by_preparation':False}
    dump(output/'PREPARATION.json',value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['repo','commit','model-inputs','tensor-metadata','kaggle-scope','data-parent',
                'data-archive','alias-fit-archive','alias-qualification-archive','output']:
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--qualification-parent');parser.add_argument('--qualification-archive')
    print(json.dumps(prepare(**vars(parser.parse_args()))))
