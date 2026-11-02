"""Build an inert frozen input bundle; this command never starts compute."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import zipfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(repo, commit, frozen_inputs, transport, output, notebook,
            recovery_parent_run=None, recovery_authorization=None, _stage_builder=None):
    repo, frozen_inputs, transport, output = map(Path, [repo, frozen_inputs, transport, output])
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('A complete reviewed source commit is required')
    if not re.fullmatch('farhad89/[a-z0-9-]+', notebook):
        raise ValueError('Explicit Kaggle notebook identity required')
    output.mkdir(parents=True, exist_ok=False)
    stage = output / 'bundle'
    source = stage / 'source'
    source.mkdir(parents=True)
    raw = subprocess.check_output(['git', 'archive', commit + ':research4'], cwd=repo)
    canonicalized = []
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive.getmembers():
            target = (source / member.name).resolve()
            if not target.is_relative_to(source.resolve()):
                raise ValueError('Source archive escaped its root')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                content = archive.extractfile(member).read()
                # Existing cohort digests bind Windows CRLF source bytes.
                # Reproduce that declared representation on Linux as well.
                if target.suffix in ['.py', '.json', '.yaml', '.yml', '.txt', '.md']:
                    normalized = content.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
                    if normalized != content:
                        canonicalized.append(member.name)
                    content = normalized
                target.write_bytes(content)
            else:
                raise ValueError('Source archive contains unsupported links')
    files = {file.relative_to(source).as_posix(): digest(file)
             for folder in ['src', 'config', 'ddl']
             for file in (source / folder).rglob('*') if file.is_file()}
    (source / 'RELEASE.json').write_text(json.dumps({
        'commit': commit, 'files': files, 'deployment_text_line_endings': 'CRLF (original study freeze)',
        'canonicalized_source_files': canonicalized}, indent=2) + '\n')
    (source / 'inputs').mkdir(exist_ok=True)
    original = json.loads((frozen_inputs / 'MANIFEST.json').read_text())
    for name, expected in original.items():
        if Path(name).name != name or 'heldout' in name.lower():
            raise ValueError('Unregistered scientific input')
        if digest(frozen_inputs / name) != expected:
            raise ValueError('Original scientific input freeze changed')
        shutil.copyfile(frozen_inputs / name, source / 'inputs' / name)
    artifacts = stage / 'artifacts'
    artifacts.mkdir()
    manifest = json.loads((transport / 'TRANSPORT_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        target = (artifacts / name).resolve()
        if not target.is_relative_to(artifacts.resolve()) or 'heldout' in name.lower():
            raise ValueError('Unregistered or sealed artifact')
        if digest(transport / name) != expected:
            raise ValueError('Transport bytes differ')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(transport / name, target)
    shutil.copyfile(transport / 'TRANSPORT_MANIFEST.json', artifacts / 'TRANSPORT_MANIFEST.json')
    root = '/kaggle/working/research4_inputs/' + commit
    destination = '/kaggle/working/research4_campaign/runs/' + commit + '/detector_rank32_v5_01'
    pin = {'platform': 'kaggle', 'account': 'farhad89', 'notebook': notebook,
           'mode': 'detector-rank32-v5', 'source_commit': commit,
           'input_root': root + '/artifacts',
           'transport_manifest_sha256': digest(artifacts / 'TRANSPORT_MANIFEST.json'),
           'destination': destination,
           'original_scientific_input_manifest_sha256': digest(frozen_inputs / 'MANIFEST.json'),
           'hardware': {'physical_allocation': 'Kaggle_T4_pair', 'visible_devices': '0'},
           'hard_timeout_seconds': 2700, 'scientific_retries': 0}
    stage_status = None
    if _stage_builder is not None:
        if recovery_parent_run or recovery_authorization:
            raise ValueError('Separate stage authority cannot be combined with recognition recovery')
        pin, stage_status = _stage_builder(source, pin)
    if recovery_authorization and not recovery_parent_run:
        raise ValueError('A recovery authorization requires its preserved parent run')
    if recovery_parent_run:
        if not re.fullmatch('[0-9]+', str(recovery_parent_run)):
            raise ValueError('An exact preserved parent run ID is required')
        pin.update(hard_timeout_seconds=4500,
                   recovery={'profile': 'kaggle-recognition-complete-75min-v1',
                             'parent_run_id': str(recovery_parent_run),
                             'authorization_file': 'RECOVERY_AUTHORIZATION.json'})
        if recovery_authorization:
            authorization = json.loads(Path(recovery_authorization).read_text())
        else:
            authorization = {'status': 'proposal_not_authorized', 'authorized_by': None,
                             'human_message_evidence_locator': None,
                             'profile': pin['recovery']['profile'],
                             'parent_run_id': str(recovery_parent_run),
                             'source_commit': commit, 'notebook': notebook,
                             'original_scientific_input_manifest_sha256': pin['original_scientific_input_manifest_sha256'],
                             'transport_manifest_sha256': pin['transport_manifest_sha256'],
                             'recipe_sha256': digest(source / 'config/composite_detector_rank32_v5.json'),
                             'original_protocol_sha256': digest(source / 'config/composite_controller_rank32_v2.json'),
                             'maximum_fresh_recovery_submissions': 1,
                             'fresh_full_run_not_exact_continuation': True,
                             'use_selected_checkpoint_weights': False, 'initialization_seed': 202610213,
                             'maximum_optimizer_updates': 1200,
                             'validation_updates': [200, 400, 600, 800, 1000, 1200],
                             'hard_timeout_seconds': 4500, 'worker_seconds': 4200,
                             'scientific_visible_GPU_count': 1, 'maximum_backbone_forwards': 30000,
                             'automatic_scientific_retries': 0, 'maximum_concurrent_research_runs': 1,
                             'reason': 'No human exception to the prior time bound and scientific-retry restriction has been recorded.'}
        (source / 'inputs/RECOVERY_AUTHORIZATION.json').write_text(json.dumps(authorization, indent=2) + '\n')
    (source / 'inputs/EXECUTION_PLATFORM.json').write_text(json.dumps(pin, indent=2) + '\n')
    (source / 'inputs/MANIFEST.json').write_text(json.dumps({
        file.name: digest(file) for file in (source / 'inputs').iterdir()
        if file.name != 'MANIFEST.json'}, indent=2) + '\n')
    if recovery_authorization:
        import sys
        sys.path.insert(0, str(source))
        from src.composite_recovery import execution_config
        import os
        previous_platform = os.environ.get('RESEARCH4_EXECUTION_PLATFORM')
        try:
            os.environ['RESEARCH4_EXECUTION_PLATFORM'] = 'kaggle'
            execution_config(source, commit, 'detector-rank32-v5',
                             json.loads((source / 'config/composite_controller_rank32_v2.json').read_text()))
        finally:
            if previous_platform is None:
                os.environ.pop('RESEARCH4_EXECUTION_PLATFORM', None)
            else:
                os.environ['RESEARCH4_EXECUTION_PLATFORM'] = previous_platform
    encoded_files = {}
    for file in list(stage.rglob('*')):
        if file.is_file() and file.suffix.lower() in ['.zip', '.gz', '.tgz', '.tar', '.7z']:
            assert file.resolve().is_relative_to(stage.resolve())
            relative = file.relative_to(stage).as_posix()
            encoded = file.with_name(file.name + '.r4b64')
            raw_bytes = file.read_bytes()
            with encoded.open('xb') as stream:
                stream.write(base64.b64encode(raw_bytes))
            encoded_files[relative] = {'stored_path': encoded.relative_to(stage).as_posix(),
                                       'decoded_sha256': hashlib.sha256(raw_bytes).hexdigest()}
            file.unlink()
    contents = stage / 'PACKAGE_FILES.json'
    contents.write_text(json.dumps({'source_commit': commit, 'encoded_files': encoded_files, 'files': {
        file.relative_to(stage).as_posix(): digest(file)
        for file in sorted(stage.rglob('*')) if file.is_file()}}, indent=2) + '\n')
    package = output / 'FROZEN_PACKAGE.zip'
    with zipfile.ZipFile(package, 'x', zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(stage.rglob('*')):
            if file.is_file():
                archive.write(file, file.relative_to(stage).as_posix())
    receipt = {**pin, 'package_sha256': digest(package),
               'package_files_manifest_sha256': digest(contents),
               'archives_preserved_via_base64': sorted(encoded_files),
               'package_bytes': package.stat().st_size,
               'heldout_transferred': False, 'scientific_execution_started': False}
    if recovery_parent_run:
        receipt['status'] = ('prepared_with_explicit_exception_not_submitted' if recovery_authorization
                             else 'proposal_requires_explicit_human_retry_and_bound_exception')
    if stage_status is not None:
        receipt['status'] = stage_status
    (output / 'PREPARATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['repo', 'commit', 'frozen-inputs', 'transport', 'output', 'notebook']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--recovery-parent-run')
    parser.add_argument('--recovery-authorization')
    arguments = vars(parser.parse_args())
    print(json.dumps(prepare(**{key.replace('-', '_'): value for key, value in arguments.items()})))
