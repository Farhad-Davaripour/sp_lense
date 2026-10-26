"""Prepare one separate frozen ordinary record-entry alias probe offline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import digest, dump, read, sha
from src.composite_source_archive import archive_source
from src.composite_task_skill_evaluation_admission import validate_preparation as validate_actor_preparation
from src.composite_tool_alias_profile import (MANIFEST, MODE, NOTEBOOK, STAGE, OFFLINE_PREPARATION,
                                              expected_stage, notebook_code, validate_source)


def prepare_actor_transport(actor_archive, output, expected_sha256):
    """Write a local ZIP and an opaque byte-identical upload file."""
    out = Path(output)
    actor_copy = out / 'ACTOR_PACKAGE.zip'
    actor_blob = out / 'ACTOR_PACKAGE.r4blob'
    shutil.copyfile(actor_archive, actor_copy)
    if sha(actor_copy) != expected_sha256:
        raise PermissionError('Original actor package copy changed')
    shutil.copyfile(actor_copy, actor_blob)
    if sha(actor_blob) != expected_sha256:
        raise PermissionError('Opaque actor transport differs from the original package bytes')
    return actor_copy, actor_blob


def prepare(repo, source_commit, actor_preparation, offline_preparation,
            fit_archive, qualification_archive, output):
    """Freeze source, selected training prompts and unchanged actor transport."""
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    actor_preparation = Path(actor_preparation).resolve()
    offline_preparation = Path(offline_preparation).resolve()
    fit_archive = Path(fit_archive).resolve()
    qualification_archive = Path(qualification_archive).resolve()
    actor_receipt = read(actor_preparation / 'PREPARATION.json')
    if (actor_receipt.get('fit_archive_local_path') != str(fit_archive)
            or actor_receipt.get('qualification_archive_local_path') != str(qualification_archive)):
        raise PermissionError('Original actor preparation must retain the exact full-fit archive paths')
    binding = validate_actor_preparation(actor_preparation / 'PREPARATION.json')
    if (binding['partition_index'] != 0 or binding['checkpoint_sha256'] != actor_receipt['checkpoint_sha256']
            or binding['fit_run_id'] != 361964150):
        raise PermissionError('Only the released first task-partition actor may enter this probe')
    actor_root = actor_preparation / 'bundle/source'
    actor_archive = actor_preparation / 'TASK_EVALUATION_PACKAGE.zip'
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    actor_copy, actor_blob = prepare_actor_transport(actor_archive, out, binding['package_sha256'])
    bundle = out / 'bundle'
    source = bundle / 'source'
    archive_source(repo, source_commit, source)
    inputs = source / 'inputs'
    inputs.mkdir(exist_ok=False)
    shutil.copyfile(offline_preparation, inputs / OFFLINE_PREPARATION)
    from src.composite_task_skill_inference import validate_source as validate_actor_source
    actor = validate_actor_source(actor_root, binding['source_commit'], fit_archive, qualification_archive)
    stage = expected_stage(source_commit, actor_root, sha(actor_copy), actor.model,
                           fit_archive, inputs / OFFLINE_PREPARATION)
    dump(inputs / STAGE, stage)
    dump(inputs / 'MANIFEST.json', {STAGE: sha(inputs / STAGE),
                                    OFFLINE_PREPARATION: sha(inputs / OFFLINE_PREPARATION)})
    manifest = bundle / MANIFEST
    dump(manifest, {'version': MODE, 'source_commit': source_commit, 'notebook': NOTEBOOK,
                    'files': {p.relative_to(bundle).as_posix(): sha(p)
                              for p in sorted(source.rglob('*')) if p.is_file()}})
    # Independent source/model/token audit before producing a notebook or receipt.
    validate_source(source, source_commit, fit_archive, qualification_archive,
                    actor_root, actor_copy)
    package = out / 'TOOL_ALIAS_PACKAGE.zip'
    with zipfile.ZipFile(package, 'x', zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                stream.write(path, path.relative_to(bundle).as_posix())
    code = notebook_code(source, source_commit, sha(manifest))
    notebook = out / 'research4_ordinary_entry_alias.ipynb'
    dump(notebook, {'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {},
                               'outputs': [], 'source': code.splitlines(keepends=True)}],
                    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                                'name': 'python3'}},
                    'nbformat': 4, 'nbformat_minor': 5})
    receipt = {'status': 'prepared_frozen_ordinary_entry_alias_not_submitted',
               'source_commit': source_commit, 'mode': MODE, 'phase': stage['phase'],
               'platform': 'kaggle', 'account': 'farhad89', 'notebook': NOTEBOOK,
               'actor_source_commit': binding['source_commit'],
               'actor_package_sha256': sha(actor_copy),
               'actor_blob_sha256': sha(actor_blob),
               'actor_original_preparation_sha256': sha(actor_preparation / 'PREPARATION.json'),
               'fit_archive_local_path': str(fit_archive),
               'qualification_archive_local_path': str(qualification_archive),
               'fit_run_id': stage['fit_run_id'], 'checkpoint_sha256': stage['checkpoint_sha256'],
               'selected_rows_sha256': stage['selected_rows_sha256'],
               'selected_rows': stage['selected_rows'], 'expected_responses': 48,
               'offline_preparation_sha256': sha(inputs / OFFLINE_PREPARATION),
               'stage_sha256': sha(inputs / STAGE),
               'input_manifest_sha256': sha(inputs / 'MANIFEST.json'),
               'bundle_manifest_sha256': sha(manifest), 'bundle_sha256': sha(package),
               'notebook_file_sha256': sha(notebook),
               'new_model_forwards': 0, 'new_optimizer_updates': 0,
               'new_world_actions': 0, 'scientific_allocation_admitted_by_preparation': False,
               'scientific_validation_admitted': False, 'heldout_content_read': False}
    dump(out / 'PREPARATION.json', receipt)
    return receipt


def deny_network(*_args, **_kwargs):
    raise RuntimeError('Frozen alias preparation forbids network access')


if __name__ == '__main__':
    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    parser = argparse.ArgumentParser()
    for name in ['repo', 'source-commit', 'actor-preparation', 'offline-preparation', 'fit-archive',
                 'qualification-archive', 'output']:
        parser.add_argument('--' + name, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
