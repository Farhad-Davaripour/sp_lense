"""Read-only completed-parent preparation for frozen recognition/risk stages."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.io_utils import read,sha,dump,digest
from src.composite_source_archive import archive_source
from src.composite_prepared_transport import source_manifest,write_transport
from src.composite_task_skill_fit_parent import FIT_FILES,read_full_fit_parent
from src.composite_task_skill_fit_registry import registered_fit
from src.composite_task_skill_source_package import frozen_source_package
from src.composite_task_skill_package import read_task_development_inputs,restore_frozen_task_package
from src.composite_task_skill_model_audit import kaggle_closure
from src.composite_frozen_risk_data_reader import audit_data,require_data
from src.composite_frozen_risk_data_parent import FILES as DATA_FILES
from src.composite_frozen_risk_profile import expected_stage,execution_plan,validate_source,STAGE
from src.composite_kaggle_v2_profile import identity
from src.composite_frozen_base import DETECTOR_ARCHIVE_SHA256,DETECTOR_CHECKPOINT_SHA256


def prepare(repo,frozen_package,fit_archive,qualification_archive,risk_preparation,risk_archive,detector_artifacts,phase,output,partition_index=None):
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    frozen=Path(frozen_package);parent=frozen/'model_parent_inputs';value=read(frozen/'FROZEN_TASK_MODEL_PACKAGE.json')
    commit=value['evaluation_source_commit'];proof=read_full_fit_parent(parent,fit_archive,qualification_archive)
    fit=registered_fit(value['fit_source_commit'],value['fit_run_id'])
    risk=Path(risk_preparation);rp=read(risk/'PREPARATION.json');risk_sha=sha(risk_archive)
    code=lambda v:[''.join(c['source']) for c in v['cells'] if c['cell_type']=='code']
    closure=kaggle_closure(read(risk/'SUBMISSION.json'),risk_sha,
      (risk/'TERMINAL_SUCCESS_AX.txt').read_text(encoding='utf-8'),(risk/'RELEASE_AX.txt').read_text(encoding='utf-8'),
      read(risk/'CONTROLLER_RECEIPT.json'),code(read(risk/'ACTUAL_SUBMITTED_NOTEBOOK.ipynb')),code(read(risk/'research4_frozen_risk_data.ipynb')))
    data=audit_data(risk_archive,risk_sha,risk/'bundle/source',rp['source_commit'],closure);data_report=require_data(data)
    source_raw=base64.b64decode((parent/'FIT_PARENT_SOURCE.r4b64').read_bytes(),validate=True)
    with frozen_source_package(source_raw,fit['package_sha256'],fit['source_commit']) as (fit_source,_):
        ordinary=read_task_development_inputs(base64.b64decode((fit_source/'inputs/TASK_DATA_EVIDENCE.r4b64').read_bytes(),validate=True))
        model=restore_frozen_task_package(value,read(parent/'FIT_PARENT_AUDIT.json'),proof,ordinary,commit)
        cfg=read(fit_source/'config/composite_generalization_amendment_v2.json')
        stage=expected_stage(commit,model,data,cfg,phase,partition_index)
        out=Path(output);out.mkdir(parents=True,exist_ok=False);source=out/'bundle/source';archive_source(repo,commit,source)
        inputs=source/'inputs';inputs.mkdir(exist_ok=False)
        for name in FIT_FILES:(inputs/name).write_bytes((parent/name).read_bytes())
        for name in ['QWEN_TENSOR_METADATA.json','QWEN_PIN.json','MODERNBERT_PIN.json','TASK_DATA_EVIDENCE.r4b64']:
            (inputs/name).write_bytes((fit_source/'inputs'/name).read_bytes())
    dump(inputs/'FROZEN_TASK_MODEL_PACKAGE.json',value)
    (inputs/'RISK_PARENT_SOURCE.r4b64').write_bytes(base64.b64encode((risk/'FROZEN_RISK_DATA_PACKAGE.zip').read_bytes()))
    mapping={'SUBMISSION.json':'RISK_PARENT_BINDING.json','CONTROLLER_RECEIPT.json':'RISK_PARENT_CONTROLLER.json',
      'TERMINAL_SUCCESS_AX.txt':'RISK_PARENT_TERMINAL_AX.txt','RELEASE_AX.txt':'RISK_PARENT_RELEASE_AX.txt',
      'ACTUAL_SUBMITTED_NOTEBOOK.ipynb':'RISK_PARENT_ACTUAL.ipynb','research4_frozen_risk_data.ipynb':'RISK_PARENT_PREPARED.ipynb'}
    for before,after in mapping.items():(inputs/after).write_bytes((risk/before).read_bytes())
    dump(inputs/'RISK_PARENT_AUDIT.json',data_report)
    dump(inputs/'RISK_PARENT_EXPECTED.json',{'source_commit':rp['source_commit'],'run_id':data_report['provider_run_id'],
      'package_sha256':sha(risk/'FROZEN_RISK_DATA_PACKAGE.zip'),'package_manifest_sha256':sha(risk/'bundle/RISK_DATA_PACKAGE_FILES.json'),
      'raw_archive_sha256':risk_sha,'release_sha256':data_report['release_sha256'],'audit_canonical_sha256':digest(data_report)})
    detector=Path(detector_artifacts);detector_archive=detector/'RUN_361470090_EVIDENCE.zip'
    if sha(detector_archive)!=DETECTOR_ARCHIVE_SHA256:raise PermissionError('Original admitted detector archive required')
    with zipfile.ZipFile(detector_archive) as z:
        selected=z.read('detector_best.pt')
        if hashlib.sha256(selected).hexdigest()!=DETECTOR_CHECKPOINT_SHA256:raise PermissionError('Fixed selected detector checkpoint changed')
        (inputs/'DETECTOR_SELECTED.pt').write_bytes(selected)
    for before,after in [('RUN_361470090_CLOSEOUT.json','DETECTOR_COMPONENT_CLOSEOUT.json'),
                         ('RUN_361470090_INDEPENDENT_AUDIT.json','DETECTOR_COMPONENT_AUDIT.json')]:
        (inputs/after).write_bytes((detector/before).read_bytes())
    dump(inputs/'RISK_EXECUTION_PLAN.json',execution_plan(data.scenes(),cfg))
    dump(inputs/'WORKSPACE.json',identity(stage['notebook']));dump(inputs/STAGE,stage)
    dump(inputs/'MANIFEST.json',{p.name:sha(p) for p in sorted(inputs.iterdir())})
    manifest=source_manifest(source,stage,'TASK_TRAINING_PACKAGE_FILES.json')
    validate_source(source,commit,fit_archive,qualification_archive,risk_archive)
    archive,notebook=write_transport(out,source,stage,manifest,'FROZEN_RISK_STAGE_PACKAGE.zip',
                                   'research4_frozen_risk_stage.ipynb','notebooks/composite_task_skill_qualification.py')
    receipt={'status':'prepared_frozen_risk_stage_not_submitted','source_commit':commit,'mode':stage['version'],
      'phase':phase,'notebook':stage['notebook'],'destination':stage['destination'],
      'partition_index':partition_index,'units':[] if stage['partition'] is None else stage['partition']['units'],
      'fit_archive_local_path':str(Path(fit_archive).resolve()),'qualification_archive_local_path':str(Path(qualification_archive).resolve()),
      'risk_archive_local_path':str(Path(risk_archive).resolve()),'checkpoint_sha256':value['checkpoint_sha256'],
      'detector_checkpoint_sha256':DETECTOR_CHECKPOINT_SHA256,'risk_data_release_sha256':data_report['release_sha256'],
      'bundle_sha256':sha(archive),'bundle_manifest_sha256':sha(manifest),'stage_sha256':sha(inputs/STAGE),
      'input_manifest_sha256':sha(inputs/'MANIFEST.json'),'notebook_file_sha256':sha(notebook),
      'scientific_GPU_count':stage['scientific_GPU_count'],'new_model_forwards':0,'new_optimizer_updates':0,
      'scientific_allocation_admitted_by_preparation':False,'scientific_validation_admitted':False,'heldout_content_read':False}
    dump(out/'PREPARATION.json',receipt);return receipt


def deny_network(*a,**k):raise RuntimeError('Frozen risk preparation forbids network access')


if __name__=='__main__':
    socket.create_connection=deny_network;socket.socket.connect=deny_network;socket.socket.connect_ex=deny_network
    parser=argparse.ArgumentParser()
    for name in ['repo','frozen-package','fit-archive','qualification-archive','risk-preparation','risk-archive','detector-artifacts','phase','output']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--partition-index',type=int)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
