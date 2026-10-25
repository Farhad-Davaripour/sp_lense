"""Read-only post-fit package preparation; no provider, fit or evaluation calls."""
import argparse
import base64
import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.io_utils import read,sha,dump,digest,verify_release
from src.composite_task_skill_package import read_task_development_inputs,frozen_task_package,full_fit_identity
from src.composite_task_skill_fit_registry import registered_fit


def prepare(repo,evaluation_source,fit_preparation,fit_archive,fit_archive_sha256,
            qualification_archive,controller,terminal_ax,release_ax,output):
    # Local processing is restricted to existing exported bytes and tokenizer
    # reconstruction. Actual scientific model/world execution stays on Kaggle.
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    fit=Path(fit_preparation);source=fit/'bundle/source'
    preparation=read(fit/'PREPARATION.json')
    binding=read(fit/'SUBMISSION.json')
    registration=registered_fit(binding.get('source_commit'),binding.get('scriptVersionId'))
    commit=registration['source_commit'];run_id=registration['run_id']
    if (preparation['phase']!='full_fit' or preparation['source_commit']!=commit
            or sha(fit/'TASK_FULL_FIT_PACKAGE.zip')!=registration['package_sha256']
            or sha(fit/'bundle/TASK_TRAINING_PACKAGE_FILES.json')!=registration['manifest_sha256']
            or sha(qualification_archive)!=registration['qualification_archive_sha256']):
        raise PermissionError('Exact registered full-fit source/input package required')
    verify_release(source,commit)
    mf=read(fit/'bundle/TASK_TRAINING_PACKAGE_FILES.json')
    for name,expected in mf['files'].items():
        target=(fit/'bundle'/name).resolve()
        if not target.is_relative_to((fit/'bundle').resolve()) or sha(target)!=expected:
            raise PermissionError('Frozen full-fit package member differs')
    if binding['phase']!='full_fit':
        raise PermissionError('Existing exact one-use full-fit submission required')
    from src.composite_task_skill_model_audit import kaggle_closure,audit_task_model
    code=lambda value:[''.join(c['source']) for c in value['cells'] if c['cell_type']=='code']
    prepared_notebook=fit/registration['prepared_notebook']
    prepared_code=code(read(prepared_notebook))
    expected_code=(source/'notebooks/composite_task_skill_qualification.py').read_text().replace(
      'SOURCE_COMMIT_PLACEHOLDER',commit).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',registration['manifest_sha256'])
    if prepared_code!=[expected_code]:raise PermissionError('Prepared parent bootstrap differs from its source/input freeze')
    closure=kaggle_closure(binding,fit_archive_sha256,Path(terminal_ax).read_text(encoding='utf-8'),
      Path(release_ax).read_text(encoding='utf-8'),read(controller),
      code(read(fit/'ACTUAL_SUBMITTED_NOTEBOOK.ipynb')),prepared_code)
    proof=audit_task_model(Path(fit_archive),fit_archive_sha256,source,commit,closure,
                          required_phase='full_fit',qualification_archive=Path(qualification_archive))
    raw_data=base64.b64decode((source/'inputs/TASK_DATA_EVIDENCE.r4b64').read_bytes(),validate=True)
    package=frozen_task_package(proof,read_task_development_inputs(raw_data),evaluation_source)
    # No checkpoint weights/optimizer/sampler/RNG are copied into this package.
    # A later reviewed runtime must independently bind the original raw output.
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    from src.composite_source_archive import archive_source
    archive_source(repo,evaluation_source,out/'evaluation_source')
    configs=['composite_controller_rank32_v2.json','composite_task_skill_v1.json',
             'composite_qwen_two_t4_candidate_v1.json','composite_qwen_file_pin_fc05.json']
    if registration['memory_profile_config'] is not None:configs.append(registration['memory_profile_config'])
    for name in configs:
        if sha(out/'evaluation_source/config'/name)!=sha(source/'config'/name):
            raise PermissionError('Full model, task study, placement or original model-file freeze changed')
    value=package.receipt()
    dump(out/'FROZEN_TASK_MODEL_PACKAGE.json',value)
    dump(out/'FULL_FIT_RAW_REAUDIT.json',proof.receipt())
    parent=out/'model_parent_inputs';parent.mkdir()
    (parent/'FIT_PARENT_SOURCE.r4b64').write_bytes(base64.b64encode((fit/'TASK_FULL_FIT_PACKAGE.zip').read_bytes()))
    mapping={fit/'SUBMISSION.json':'FIT_PARENT_BINDING.json',Path(controller):'FIT_PARENT_CONTROLLER.json',
      Path(terminal_ax):'FIT_PARENT_TERMINAL_AX.txt',Path(release_ax):'FIT_PARENT_RELEASE_AX.txt',
      fit/'ACTUAL_SUBMITTED_NOTEBOOK.ipynb':'FIT_PARENT_ACTUAL.ipynb',
      prepared_notebook:'FIT_PARENT_PREPARED.ipynb'}
    for before,after in mapping.items():(parent/after).write_bytes(before.read_bytes())
    dump(parent/'FIT_PARENT_AUDIT.json',proof.receipt())
    dump(parent/'FIT_PARENT_EXPECTED.json',{'source_commit':commit,'run_id':run_id,
      'package_sha256':registration['package_sha256'],'raw_archive_sha256':fit_archive_sha256,
      'checkpoint_sha256':value['checkpoint_sha256'],'fit_evidence_identity_sha256':full_fit_identity(proof),
      'raw_audit_canonical_sha256':digest(proof.receipt()),
      'qualification_archive_sha256':sha(qualification_archive)})
    dump(parent/'MANIFEST.json',{p.name:sha(p) for p in sorted(parent.iterdir())})
    receipt={'status':'frozen_metadata_package_prepared_no_evaluation_allocation',
      'fit_run_id':run_id,'fit_source_commit':commit,'evaluation_source_commit':evaluation_source,
      'checkpoint_sha256':value['checkpoint_sha256'],'fit_raw_archive_sha256':fit_archive_sha256,
      'full_fit_raw_reaudit_sha256':sha(out/'FULL_FIT_RAW_REAUDIT.json'),
      'model_package_sha256':sha(out/'FROZEN_TASK_MODEL_PACKAGE.json'),
      'model_package_canonical_sha256':digest(value),'task_development_trajectories':72,
      'model_parent_input_manifest_sha256':sha(parent/'MANIFEST.json'),
      'separate_risk_trajectories_still_required':180,'new_model_forwards':0,'new_optimizer_updates':0,
      'new_world_collection':False,'evaluation_runtime_qualified':False,
      'scientific_validation_admitted':False,'heldout_content_read':False}
    dump(out/'PREPARATION.json',receipt)
    return receipt


def deny_network(*args,**kwargs):raise RuntimeError('Post-fit raw package preparation forbids network access')


if __name__=='__main__':
    socket.create_connection=deny_network;socket.socket.connect=deny_network;socket.socket.connect_ex=deny_network
    parser=argparse.ArgumentParser()
    for name in ['repo','evaluation-source','fit-preparation','fit-archive','fit-archive-sha256',
                 'qualification-archive','controller','terminal-ax','release-ax','output']:
        parser.add_argument('--'+name,required=True)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
