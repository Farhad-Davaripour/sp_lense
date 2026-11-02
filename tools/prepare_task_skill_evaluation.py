"""Prepare one fixed task-pair partition from complete exported dependencies."""
import argparse
import base64
import json
import os
from pathlib import Path
import socket
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.io_utils import read,sha,dump,digest
from src.composite_task_skill_fit_parent import FIT_FILES,read_full_fit_parent
from src.composite_task_skill_fit_registry import registered_fit
from src.composite_task_skill_package import read_task_development_inputs,restore_frozen_task_package
from src.composite_task_skill_inference import expected_stage,validate_source,STAGE
from src.composite_task_skill_validation import task_development_plan
from src.composite_task_skill_execution import execution_plan
from src.composite_task_skill_source_package import frozen_source_package
from src.composite_source_archive import archive_source
from src.composite_kaggle_v2_profile import identity


def prepare(repo,frozen_package,fit_archive,qualification_archive,partition_index,output):
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    parent=Path(frozen_package);inputs=parent/'model_parent_inputs'
    value=read(parent/'FROZEN_TASK_MODEL_PACKAGE.json');commit=value['evaluation_source_commit']
    proof=read_full_fit_parent(inputs,fit_archive,qualification_archive)
    registration=registered_fit(value['fit_source_commit'],value['fit_run_id'])
    encoded=base64.b64decode((inputs/'FIT_PARENT_SOURCE.r4b64').read_bytes(),validate=True)
    # Read the immutable complete parent source/input package; never use a draft
    # tree or invent a completed checkpoint from a running-state summary.
    with frozen_source_package(encoded,registration['package_sha256'],registration['source_commit']) as (fit_source,_):
        raw=base64.b64decode((fit_source/'inputs/TASK_DATA_EVIDENCE.r4b64').read_bytes(),validate=True)
        model=restore_frozen_task_package(value,read(inputs/'FIT_PARENT_AUDIT.json'),proof,
                                         read_task_development_inputs(raw),commit)
        stage=expected_stage(commit,model,partition_index)
        out=Path(output);out.mkdir(parents=True,exist_ok=False);bundle=out/'bundle'
        source=bundle/'source';archive_source(repo,commit,source)
        target=source/'inputs';target.mkdir(exist_ok=False)
        for name in FIT_FILES:(target/name).write_bytes((inputs/name).read_bytes())
        for name in ['QWEN_TENSOR_METADATA.json','QWEN_PIN.json','TASK_DATA_EVIDENCE.r4b64']:
            (target/name).write_bytes((fit_source/'inputs'/name).read_bytes())
    dump(target/'FROZEN_TASK_MODEL_PACKAGE.json',value)
    dump(target/'TASK_EVALUATION_PLAN.json',task_development_plan(value['development_roster']))
    dump(target/'TASK_EXECUTION_PLAN.json',execution_plan(value['development_roster']))
    dump(target/'WORKSPACE.json',identity(stage['notebook']));dump(target/STAGE,stage)
    dump(target/'MANIFEST.json',{p.name:sha(p) for p in sorted(target.iterdir()) if p.name!='MANIFEST.json'})
    manifest=bundle/'TASK_TRAINING_PACKAGE_FILES.json'
    dump(manifest,{'version':stage['version'],'source_commit':commit,'notebook':stage['notebook'],
                  'files':{p.relative_to(bundle).as_posix():sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}})
    validate_source(source,commit,fit_archive,qualification_archive)
    archive=out/'TASK_EVALUATION_PACKAGE.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(bundle).as_posix())
    code=(source/'notebooks/composite_task_skill_qualification.py').read_text().replace(
      'SOURCE_COMMIT_PLACEHOLDER',commit).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',sha(manifest))
    notebook=out/'research4_task_development.ipynb'
    dump(notebook,{'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
      'source':code.splitlines(keepends=True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
      'nbformat':4,'nbformat_minor':5})
    receipt={'status':'prepared_frozen_task_development_not_submitted','source_commit':commit,
      'mode':stage['version'],'phase':stage['phase'],'notebook':stage['notebook'],'platform':'kaggle','account':'farhad89',
      'partition_index':partition_index,'case_ids':stage['partition']['case_ids'],
      'expected_trajectories':stage['partition']['expected_trajectories'],
      'full_task_trajectories_still_required':72,'separate_risk_trajectories_still_required':180,
      'fit_run_id':value['fit_run_id'],'checkpoint_sha256':value['checkpoint_sha256'],
      'fit_archive_local_path':str(Path(fit_archive).resolve()),
      'qualification_archive_local_path':str(Path(qualification_archive).resolve()),
      'frozen_model_package_sha256':digest(value),'bundle_sha256':sha(archive),'bundle_manifest_sha256':sha(manifest),
      'input_manifest_sha256':sha(target/'MANIFEST.json'),'stage_sha256':sha(target/STAGE),'notebook_file_sha256':sha(notebook),
      'destination':stage['destination'],'new_model_forwards':0,'optimizer_updates':0,
      'scientific_allocation_admitted_by_preparation':False,'scientific_validation_admitted':False,'heldout_content_read':False}
    dump(out/'PREPARATION.json',receipt);return receipt


def deny_network(*args,**kwargs):raise RuntimeError('Frozen task preparation forbids network access')


if __name__=='__main__':
    socket.create_connection=deny_network;socket.socket.connect=deny_network;socket.socket.connect_ex=deny_network
    parser=argparse.ArgumentParser()
    for name in ['repo','frozen-package','fit-archive','qualification-archive','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--partition-index',required=True,type=int)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
