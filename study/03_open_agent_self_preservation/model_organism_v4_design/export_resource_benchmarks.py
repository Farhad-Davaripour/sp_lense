"""Export small timing and isolation evidence, never model weights or raw trajectories."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = (
    ('capacity', Path('/var/lib/sp-lense-r3-capacity-benchmark'),
     HERE / 'resource_benchmark/evidence', 'r3capacity-threads'),
    ('confirmation', Path('/var/lib/sp-lense-r3-capacity-confirm'),
     HERE / 'resource_benchmark_confirm/evidence', 'r3confirm-threads'),
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(label, root, destination, job):
    if destination.exists():
        raise FileExistsError(destination)
    gate, decision = read(root / 'isolation_gate.json'), read(root / 'DECISION.json')
    receipt = read(root / 'runs' / job / 'receipt.json')
    if not gate['passed'] or not decision['boundary_gate_passed'] or receipt['returncode']:
        raise RuntimeError('Benchmark evidence incomplete: ' + label)
    if not receipt['worker_processes_gone'] or gate['boundary_hashes'] != receipt['boundary_hashes']:
        raise RuntimeError('Worker or boundary mismatch: ' + label)
    for directory in (root / 'runs').iterdir():
        group = Path('/sys/fs/cgroup/system.slice') / ('sp-r3-' + directory.name + '.service') / 'cgroup.procs'
        if group.exists() and group.read_text().strip():
            raise RuntimeError('Worker remains active: ' + directory.name)
    for name, expected in receipt['artifact_sha256'].items():
        if sha(root / 'runs' / job / 'artifacts' / name) != expected:
            raise RuntimeError('Artifact changed: ' + name)
    files = [root / name for name in ('isolation_gate.json', 'INPUT_MANIFEST.json', 'DECISION.json')]
    for run in sorted((root / 'runs').iterdir()):
        files.extend(run.glob('receipt.json'))
        files.extend((run / 'artifacts').glob('*.json'))
    if label == 'confirmation':
        files += [root / 'source_snapshot/analyze.py', root / 'analysis_fix/analyze.py']
    destination.mkdir(parents=True)
    manifest = {}
    for source in files:
        if source.is_symlink() or not source.is_file() or source.suffix not in ('.json', '.py'):
            raise RuntimeError('Invalid small evidence file')
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(source) != sha(target):
            raise RuntimeError('Export hash mismatch: ' + str(relative))
        manifest[str(relative)] = {'bytes': source.stat().st_size, 'sha256': sha(source)}
    (destination / 'EVIDENCE_MANIFEST.json').write_text(json.dumps({
        'source_root': str(root), 'files': manifest, 'model_weights_exported': False,
        'evaluation_cases_opened': 0, 'workers_active_at_export': False,
    }, indent=2) + '\n')
    return {'label': label, 'files': len(manifest),
            'decision': decision['chosen_training_threads']}


if __name__ == '__main__':
    print(json.dumps([export(*spec) for spec in RUNS], indent=2))
