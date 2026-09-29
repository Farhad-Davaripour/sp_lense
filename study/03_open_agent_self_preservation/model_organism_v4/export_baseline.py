"""Immutable audited baseline-only archive on C:, with committed hash pointer."""
import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

ROOT = Path('/var/lib/sp-lense-r3-organism-v4')
PRIMARY = Path('/mnt/c/Users/farha/AppData/Local/SP_Lense/Research3Runs/v4')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with path.open('xb') as stream:
        stream.write((json.dumps(value, indent=2) + '\n').encode('utf-8'))


def main(run_id, source_commit, analysis_commit, manifest_out):
    if (not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', run_id)
            or not re.fullmatch(r'[0-9a-f]{40}', source_commit)
            or not re.fullmatch(r'[0-9a-f]{40}', analysis_commit)):
        raise ValueError('Invalid run ID or exact Git commit SHA')
    if ROOT.resolve() != ROOT or PRIMARY.resolve() != PRIMARY or not PRIMARY.is_dir():
        raise RuntimeError('Unexpected raw archive root')
    archive = PRIMARY / (run_id + '.tar.gz')
    target = Path(manifest_out).resolve()
    if archive.exists() or target.exists() or (ROOT / 'ARCHIVE_CONTENTS.json').exists():
        raise FileExistsError('Archive, repository manifest, or internal content manifest already exists')
    gate = read(ROOT / 'BASELINE_GATE.json')
    audit = read(ROOT / 'BASELINE_AUDIT.json')
    diagnosis = read(ROOT / 'BASELINE_DIAGNOSIS.json')
    storage = read(ROOT / 'STORAGE_DECISION.json')
    trust = read(ROOT / 'TRUSTED_CONTROL_REVISION.json')
    if (gate['baseline_pass'] is not False or not audit['passed']
            or audit['new_captures'] != 80 or audit['model_jobs']['fit'] != 0
            or diagnosis['scientific_thresholds_or_scores_changed']
            or storage['separate_backup'] is not False
            or storage['single_drive_risk_accepted_by_user'] is not True):
        raise RuntimeError('Baseline-only negative capture not fully audited')
    if not re.fullmatch(r'[0-9a-f]{40}', trust.get('source_commit_sha', '')):
        raise RuntimeError('Missing trusted control commit provenance')
    experiment_jobs = []
    for directory in (ROOT / 'runs').iterdir():
        group = Path('/sys/fs/cgroup/system.slice') / ('sp-r3-' + directory.name + '.service') / 'cgroup.procs'
        if group.exists() and group.read_text().strip():
            raise RuntimeError('Model/probe worker remains active')
        receipt = read(directory / 'receipt.json')
        if receipt['mode'] == 'experiment':
            if receipt['returncode'] or not receipt['worker_processes_gone']:
                raise RuntimeError('Failed or active baseline model job')
            experiment_jobs.append(directory.name)
    if len(experiment_jobs) != 10 or any(name.startswith('v4-fit-') for name in experiment_jobs):
        raise RuntimeError('Unexpected baseline-only job count')
    members = [p for p in ROOT.rglob('*') if p.is_file()
               and 'work' not in p.relative_to(ROOT).parts
               and '__pycache__' not in p.relative_to(ROOT).parts]
    contents = {}
    for path in members:
        if path.is_symlink() or path.stat().st_nlink != 1 or path.stat().st_size > 64 * 1024**2:
            raise RuntimeError('Unsafe archive member: ' + str(path))
        contents[str(path.relative_to(ROOT))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    internal = ROOT / 'ARCHIVE_CONTENTS.json'
    write_new(internal, contents)
    with tarfile.open(archive, 'x:gz') as bundle:
        for path in members + [internal]:
            relative = str(path.relative_to(ROOT))
            info = bundle.gettarinfo(str(path), arcname='model_organism_v4/' + relative)
            info.mode, info.uid, info.gid = 0o644, 0, 0
            info.uname = info.gname = ''
            with path.open('rb') as stream:
                bundle.addfile(info, stream)
    verified = 0
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle.getmembers():
            relative = str(Path(member.name).relative_to('model_organism_v4'))
            if not member.isfile() or '..' in Path(relative).parts or member.size > 64 * 1024**2:
                raise RuntimeError('Unsafe archived member')
            with bundle.extractfile(member) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            expected = sha(internal) if relative == 'ARCHIVE_CONTENTS.json' else contents[relative]['sha256']
            if actual != expected:
                raise RuntimeError('Archived member hash differs: ' + relative)
            verified += 1
    freeze = {p.name: sha(p) for p in (ROOT / 'inputs').glob('*FREEZE.json')}
    record = {'run_id': run_id, 'stage': 'unchanged_base_only',
              'foundation_baseline_pass': False,
              'source_commit_sha': source_commit,
              'trusted_control_commit_sha': trust['source_commit_sha'],
              'postrun_analysis_commit_sha': analysis_commit,
              'archive_windows_path': 'C:\\Users\\farha\\AppData\\Local\\SP_Lense\\Research3Runs\\v4\\' + archive.name,
              'archive_linux_path': str(archive), 'archive_sha256': sha(archive),
              'bytes': archive.stat().st_size, 'verified_members': verified,
              'internal_manifest_sha256': sha(internal),
              'baseline_gate_sha256': sha(ROOT / 'BASELINE_GATE.json'),
              'baseline_audit_sha256': sha(ROOT / 'BASELINE_AUDIT.json'),
              'baseline_diagnosis_sha256': sha(ROOT / 'BASELINE_DIAGNOSIS.json'),
              'storage_decision_sha256': sha(ROOT / 'STORAGE_DECISION.json'),
              'isolation_gate_sha256': sha(ROOT / 'isolation_gate.json'),
              'frozen_input_sha256': freeze,
              'generated_token_activations_in_archive': True,
              'baseline_model_weights_included': False,
              'version_4_fitted_adapters_included': False,
              'separate_backup': False,
              'single_drive_risk_accepted_by_user': True,
              'worker_processes_active_at_export': False}
    target.parent.mkdir(parents=True, exist_ok=True)
    write_new(target, record)
    print(json.dumps({'archive': str(archive), 'bytes': record['bytes'],
                      'sha256': record['archive_sha256'], 'verified_members': verified,
                      'manifest': str(target)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--analysis-commit', required=True)
    parser.add_argument('--manifest-out', required=True)
    args = parser.parse_args()
    main(args.run_id, args.source_commit, args.analysis_commit, args.manifest_out)
