"""Freeze reviewed V4 CPU data source/identities; never generate a world or submit."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import dump, read, sha
from src.composite_source_archive import archive_source
from src.composite_bounded_behavior_profile import validate_preparation
from src.composite_bounded_behavior_parent import assemble_parent, read_parent
from src.composite_bounded_behavior_data_profile import (
    MODE, PHASE, OWNER_THREAD, PROFILE, MANIFEST, NOTEBOOK, SCOPE,
    TAKEOVER, TAKEOVER_SHA, PARENT, PARENT_DESCRIPTOR, expected_profile, validate_source, notebook_code)
from src.composite_bounded_behavior_data_admission import PACKAGE, NOTEBOOK_FILE, validate_package
from src.composite_kaggle_v2_profile import HUMAN as KAGGLE_HUMAN, HUMAN_SHA, identity


def prepare(repo, commit, alias_directory, gate, fit_archive, qualification_archive,
            human_scope, kaggle_scope, takeover_scope, output):
    fit_archive, qualification_archive = Path(fit_archive).resolve(), Path(qualification_archive).resolve()
    if sha(kaggle_scope) != HUMAN_SHA or sha(takeover_scope) != TAKEOVER_SHA:
        raise PermissionError('Exact Kaggle and execution-takeover human receipts required')
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    source = out / 'bundle/source'
    archive_source(repo, commit, source)
    cfg = validate_preparation(read(source / 'config/composite_bounded_behavior_v4.json'))
    if sha(human_scope) != cfg['scope_amendment_receipt_sha256']:
        raise PermissionError('Exact verified bounded-study human scope receipt required')
    packet = out / PARENT
    parent_archive = assemble_parent(alias_directory, gate, packet)
    inputs = source / 'inputs'; inputs.mkdir(exist_ok=False)
    shutil.copyfile(packet, inputs / PARENT)
    for origin, name in [(human_scope, SCOPE), (kaggle_scope, KAGGLE_HUMAN), (takeover_scope, TAKEOVER)]:
        shutil.copyfile(origin, inputs / name)
    with read_parent(packet, parent_archive['archive_sha256'], fit_archive, qualification_archive) as parent:
        descriptor = parent.receipt()
        dump(inputs / PARENT_DESCRIPTOR, descriptor)
        profile = expected_profile(commit, cfg, parent)
        dump(inputs / PROFILE, profile)
        dump(inputs / 'WORKSPACE.json', identity(NOTEBOOK))
        dump(inputs / 'MANIFEST.json', {p.name: sha(p) for p in sorted(inputs.iterdir())
                                        if p.name != 'MANIFEST.json'})
        manifest_path = out / 'bundle' / MANIFEST
        dump(manifest_path, {'version': MODE, 'source_commit': commit, 'notebook': NOTEBOOK,
            'files': {p.relative_to(out / 'bundle').as_posix(): sha(p)
                      for p in sorted(source.rglob('*')) if p.is_file()}})
        validate_source(source, commit, parent)
    archive = out / PACKAGE
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
        for path in sorted((out / 'bundle').rglob('*')):
            if path.is_file(): stream.write(path, path.relative_to(out / 'bundle').as_posix())
    validate_package(archive, out / 'bundle')
    code = notebook_code(source, commit, sha(manifest_path))
    dump(out / NOTEBOOK_FILE, {'cells': [{'cell_type': 'code', 'execution_count': None,
        'metadata': {}, 'outputs': [], 'source': code.splitlines(keepends=True)}],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'nbformat': 4, 'nbformat_minor': 5})
    receipt = {'status': 'prepared_bounded_behavior_CPU_data_not_submitted',
        'source_commit': commit, 'mode': MODE, 'phase': PHASE, 'owner_thread_id': OWNER_THREAD,
        'notebook': NOTEBOOK, 'platform': 'kaggle', 'account': 'farhad89',
        'destination': profile['destination'], 'bundle_sha256': sha(archive),
        'bundle_manifest_sha256': sha(manifest_path), 'profile_sha256': sha(inputs / PROFILE),
        'input_manifest_sha256': sha(inputs / 'MANIFEST.json'), 'notebook_file_sha256': sha(out / NOTEBOOK_FILE),
        'parent_packet_sha256': descriptor['parent_packet_sha256'],
        'fit_archive_local_path': str(fit_archive), 'qualification_archive_local_path': str(qualification_archive),
        'model_forwards': 0, 'optimizer_updates': 0, 'scientific_GPU_count': 0,
        'behavioral_data_materialized': False, 'training_launch_admitted': False,
        'scientific_allocation_admitted_by_preparation': False, 'heldout_content_read': False}
    dump(out / 'PREPARATION.json', receipt)
    return receipt


def deny_network(*_args, **_kwargs):
    raise PermissionError('Local V4 source/identity preparation is offline')


if __name__ == '__main__':
    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    parser = argparse.ArgumentParser()
    for name in ['repo', 'commit', 'alias-directory', 'gate', 'fit-archive', 'qualification-archive',
                 'human-scope', 'kaggle-scope', 'takeover-scope', 'output']:
        parser.add_argument('--' + name, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
