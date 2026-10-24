"""Prepare the one-use full-rank task qualification package; never allocate."""
import argparse
import base64
import json
from pathlib import Path
import shutil
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import read, dump, sha
from src.composite_source_archive import archive_source
from src.composite_task_skill_training import (MODE, PLAN, DATA_ARCHIVE, DATA_ARCHIVE_SHA,
    DATA_AUDIT_SHA, DATA_CLOSEOUT_SHA, PRE_FORWARD_PARENT, PRE_FORWARD_PARENT_SHA,
    qualification_plan, validate_model_source)
from src.composite_task_skill_profile import HUMAN, SELECTION_SHA
from src.composite_kaggle_v2_profile import HUMAN as KAGGLE_HUMAN, HUMAN_SHA, identity
from src.composite_v2_data_contract import ORIGINAL_MODEL_MANIFEST


def prepare(repo, commit, model_inputs, tensor_metadata, human_selection, kaggle_scope, data_preparation, pre_forward_parent, output,qualification_parent=None):
    if sha(human_selection) != SELECTION_SHA or sha(kaggle_scope) != HUMAN_SHA:
        raise PermissionError('Exact direct human option-A/Kaggle selections required')
    if sha(pre_forward_parent)!=PRE_FORWARD_PARENT_SHA:raise PermissionError('Exact pre-forward failure closeout required')
    data = Path(data_preparation)
    files = {'RUN_361865152_EVIDENCE.zip': DATA_ARCHIVE_SHA,
             'RUN_361865152_INDEPENDENT_DATA_AUDIT.json': DATA_AUDIT_SHA,
             'RUN_361865152_CLOSEOUT.json': DATA_CLOSEOUT_SHA}
    for name, expected in files.items():
        if sha(data / name) != expected:
            raise PermissionError('Use the exact audited/released task data, not a substitute or summary')
    model_inputs = Path(model_inputs)
    if sha(model_inputs / 'MANIFEST.json') != ORIGINAL_MODEL_MANIFEST:
        raise PermissionError('Original complete model-file freeze required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / 'bundle'
    source = bundle / 'source'
    archive_source(repo, commit, source)
    inputs = source / 'inputs'
    inputs.mkdir(exist_ok=True)
    shutil.copyfile(model_inputs / 'MANIFEST.json', inputs / 'PARENT_MODEL_INPUT_MANIFEST.json')
    for name in ['QWEN_PIN.json', 'MODERNBERT_PIN.json']:
        if sha(model_inputs / name) != read(model_inputs / 'MANIFEST.json')[name]:
            raise ValueError('Pinned model metadata changed')
        shutil.copyfile(model_inputs / name, inputs / name)
    shutil.copyfile(human_selection, inputs / HUMAN)
    shutil.copyfile(kaggle_scope, inputs / KAGGLE_HUMAN)
    shutil.copyfile(pre_forward_parent,inputs/PRE_FORWARD_PARENT)
    shutil.copyfile(tensor_metadata, inputs / 'QWEN_TENSOR_METADATA.json')
    shutil.copyfile(data / 'RUN_361865152_INDEPENDENT_DATA_AUDIT.json', inputs / 'TASK_DATA_AUDIT.json')
    shutil.copyfile(data / 'RUN_361865152_CLOSEOUT.json', inputs / 'TASK_DATA_CLOSEOUT.json')
    shutil.copyfile(data / 'bundle/source/RELEASE.json', inputs / 'PARENT_DATA_SOURCE_RELEASE.json')
    # Kaggle may expand uploaded archives. Keep this exact nested evidence
    # archive in an explicit byte transport so its checksums remain stable.
    (inputs / DATA_ARCHIVE).write_bytes(base64.b64encode((data / 'RUN_361865152_EVIDENCE.zip').read_bytes()))
    raw_parent=None
    if qualification_parent is not None:
        from src.composite_task_skill_parent import PARENT_PACKAGE_SHA,PARENT_CLOSEOUT_SHA,read_parent
        from src.composite_task_skill_full_fit import full_fit_plan
        parent=Path(qualification_parent)
        if sha(parent/'TASK_QUALIFICATION_PACKAGE.zip')!=PARENT_PACKAGE_SHA or sha(parent/'CLOSEOUT.json')!=PARENT_CLOSEOUT_SHA:
            raise PermissionError('Exact raw-audited released qualifier and frozen source package required')
        mapping={'CLOSEOUT.json':'PARENT_QUALIFICATION_CLOSEOUT.json',
          'RUN_361902084_INDEPENDENT_AUDIT.json':'PARENT_QUALIFICATION_AUDIT.json',
          'SUBMISSION.json':'PARENT_QUALIFICATION_BINDING.json',
          'CONTROLLER_LOG_READBACK.json':'PARENT_QUALIFICATION_CONTROLLER.json',
          'TERMINAL_SUCCESS_AX.txt':'PARENT_QUALIFICATION_TERMINAL_AX.txt',
          'RELEASE_AX.txt':'PARENT_QUALIFICATION_RELEASE_AX.txt',
          'research4_task_skill_qualification.ipynb':'PARENT_QUALIFICATION_PREPARED.ipynb',
          'ACTUAL_SUBMITTED_NOTEBOOK.ipynb':'PARENT_QUALIFICATION_ACTUAL.ipynb'}
        for before,after in mapping.items():shutil.copyfile(parent/before,inputs/after)
        (inputs/'PARENT_QUALIFICATION_SOURCE.r4b64').write_bytes(base64.b64encode((parent/'TASK_QUALIFICATION_PACKAGE.zip').read_bytes()))
        raw_parent=parent/'RUN_361902084_EVIDENCE.zip'
        stage=full_fit_plan(commit,read_parent(source,raw_parent))
    else:stage = qualification_plan(commit)
    dump(inputs / PLAN, stage)
    dump(inputs / 'WORKSPACE.json', identity(stage['notebook']))
    dump(inputs / 'MANIFEST.json', {p.name: sha(p) for p in inputs.iterdir() if p.name != 'MANIFEST.json'})
    _, cfg, evidence = validate_model_source(source, commit,raw_parent)
    package_manifest = bundle / 'TASK_TRAINING_PACKAGE_FILES.json'
    dump(package_manifest, {'version': MODE, 'source_commit': commit, 'notebook': stage['notebook'],
        'files': {p.relative_to(bundle).as_posix(): sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}})
    archive = output / ('TASK_FULL_FIT_PACKAGE.zip' if qualification_parent is not None else 'TASK_QUALIFICATION_PACKAGE.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():
                stream.write(p, p.relative_to(bundle).as_posix())
    code = (source / 'notebooks/composite_task_skill_qualification.py').read_text()
    code = code.replace('SOURCE_COMMIT_PLACEHOLDER', commit).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER', sha(package_manifest))
    notebook = output / ('research4_task_skill_fit.ipynb' if qualification_parent is not None else 'research4_task_skill_qualification.ipynb')
    dump(notebook, {'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {},
        'outputs': [], 'source': code.splitlines(keepends=True)}],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'nbformat': 4, 'nbformat_minor': 5})
    receipt = {'status': 'prepared_task_full_fit_not_submitted' if qualification_parent is not None else 'prepared_task_qualification_not_submitted', 'mode': MODE, 'phase': stage['phase'],
        'source_commit': commit, 'notebook': stage['notebook'], 'platform': 'kaggle', 'account': 'farhad89',
        'destination': stage['destination'], 'bundle_sha256': sha(archive),
        'bundle_manifest_sha256': sha(package_manifest), 'plan_sha256': sha(inputs / PLAN),
        'input_manifest_sha256': sha(inputs / 'MANIFEST.json'), 'notebook_file_sha256': sha(notebook),
        'data_archive_sha256': DATA_ARCHIVE_SHA, 'training_states': len(evidence.rows()),
        'source_data_run': 361865152, 'data_scope': 'ordinary_authorized_task_skills_only',
        'scientific_GPU_count': 2, 'actual_training_capacity_qualified': False,
        'full_fit_completed': False, 'model_forwards': 0, 'optimizer_updates': 0,
        'heldout_content_read': False, 'scientific_allocation_admitted_by_preparation': False}
    if raw_parent is not None:receipt.update(qualification_archive_local_path=str(raw_parent),qualification_run_id=361902084,
        parent_scratch_training_capacity_independently_audited=True,
        new_full_fit_training_execution_verified=False)
    dump(output / 'PREPARATION.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ['repo', 'commit', 'model-inputs', 'tensor-metadata', 'human-selection',
                'kaggle-scope', 'data-preparation', 'pre-forward-parent', 'output']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--qualification-parent')
    print(json.dumps(prepare(**vars(parser.parse_args()))))
