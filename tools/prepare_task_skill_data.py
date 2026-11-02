"""Prepare exact reviewed option-A CPU data bytes; never allocate compute."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import read, dump, sha
from src.composite_source_archive import archive_source
from src.composite_task_skill_profile import (MODE, PROFILE, HUMAN, SELECTION_SHA, NOTEBOOK,
                                               PARENT_FAILURE, PARENT_FAILURE_SHA, expected_profile, validate_source)
from src.composite_kaggle_v2_profile import identity, HUMAN as KAGGLE_HUMAN, HUMAN_SHA
from src.composite_v2_data_contract import ORIGINAL_MODEL_MANIFEST


def prepare(repo, commit, model_inputs, human_selection, kaggle_scope, parent_failure, output):
    if sha(human_selection) != SELECTION_SHA or sha(kaggle_scope) != HUMAN_SHA:
        raise PermissionError('Exact human option-A/Kaggle selection receipts required')
    if sha(parent_failure) != PARENT_FAILURE_SHA:
        raise PermissionError('Exact released pre-data failure required for the separate bootstrap repair')
    model_inputs = Path(model_inputs)
    if sha(model_inputs / 'MANIFEST.json') != ORIGINAL_MODEL_MANIFEST:
        raise ValueError('Original full model input freeze required')
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
            raise ValueError('Original model pin bytes differ')
        shutil.copyfile(model_inputs / name, inputs / name)
    shutil.copyfile(human_selection, inputs / HUMAN)
    shutil.copyfile(kaggle_scope, inputs / KAGGLE_HUMAN)
    shutil.copyfile(parent_failure, inputs / PARENT_FAILURE)
    dump(inputs / 'WORKSPACE.json', identity(NOTEBOOK))
    dump(inputs / PROFILE, expected_profile(commit))
    dump(inputs / 'MANIFEST.json', {p.name: sha(p) for p in inputs.iterdir() if p.name != 'MANIFEST.json'})
    profile, _ = validate_source(source, commit)
    package_manifest = bundle / 'TASK_SKILL_PACKAGE_FILES.json'
    dump(package_manifest, {'version': MODE, 'source_commit': commit, 'notebook': NOTEBOOK,
        'files': {p.relative_to(bundle).as_posix(): sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}})
    archive = output / 'TASK_SKILL_DATA_PACKAGE.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():
                stream.write(p, p.relative_to(bundle).as_posix())
    cell = (source / 'notebooks/composite_task_skill_data.py').read_text()
    cell = cell.replace('SOURCE_COMMIT_PLACEHOLDER', commit).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER', sha(package_manifest))
    notebook = output / 'research4_task_skill_data.ipynb'
    dump(notebook, {'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {},
        'outputs': [], 'source': cell.splitlines(keepends=True)}],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'nbformat': 4, 'nbformat_minor': 5})
    receipt = {'status': 'prepared_task_only_CPU_data_not_submitted', 'mode': MODE,
        'source_commit': commit, 'notebook': NOTEBOOK, 'platform': 'kaggle', 'account': 'farhad89',
        'destination': profile['destination'], 'bundle_sha256': sha(archive),
        'bundle_manifest_sha256': sha(package_manifest), 'profile_sha256': sha(inputs / PROFILE),
        'input_manifest_sha256': sha(inputs / 'MANIFEST.json'), 'notebook_file_sha256': sha(notebook),
        'human_selection_sha256': SELECTION_SHA, 'GPU_required': False,
        'task_dataset_generated': False, 'continuity_targets_generated': False,
        'heldout_content_read': False, 'model_forwards': 0, 'optimizer_updates': 0,
        'scientific_allocation_admitted_by_preparation': False}
    dump(output / 'PREPARATION.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ['repo', 'commit', 'model-inputs', 'human-selection', 'kaggle-scope', 'parent-failure', 'output']:
        parser.add_argument('--' + key, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
