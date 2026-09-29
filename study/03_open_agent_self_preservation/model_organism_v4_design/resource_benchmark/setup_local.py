"""Prepare a separate 12-core-capacity benchmark with the tested isolation boundary."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent.parent / 'model_organism_v3'
sys.path.insert(0, str(V3))
import boundary_guard

ROOT = Path('/var/lib/sp-lense-r3-capacity-benchmark')
CODE = Path('/opt/sp-lense-r3-capacity-benchmark/code')
V3_ROOT = Path('/var/lib/sp-lense-r3-organism-v3')
OLD_ROOT = '/var/lib/sp-lense-r3-organism-v3'
OLD_CODE = '/opt/sp-lense-r3-organism-v3/code'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Unexpected boundary source structure: ' + old)
    return source.replace(old, new, 1)


def main():
    if os.geteuid() != 0 or ROOT.exists() or CODE.exists():
        raise RuntimeError('Requires root and unused benchmark directories')
    if not json.loads((V3_ROOT / 'p3-GATE.json').read_text())['foundation_pass'] is False:
        raise RuntimeError('Expected completed failed checkpoint-3 gate')
    if not json.loads((V3_ROOT / 'p3-AUDIT.json').read_text())['passed']:
        raise RuntimeError('Checkpoint-3 audit has not passed')
    for directory in (V3_ROOT / 'runs').iterdir():
        group = Path('/sys/fs/cgroup/system.slice') / ('sp-r3-' + directory.name + '.service') / 'cgroup.procs'
        if group.exists() and group.read_text().strip():
            raise RuntimeError('Another Research 3 worker is active')
    ROOT.mkdir(mode=0o755)
    (ROOT / 'runs').mkdir()
    CODE.mkdir(parents=True)
    (ROOT / 'inputs').mkdir()
    for name in ('train.json', 'settings.json', 'FREEZE.json', 'model_manifest.json'):
        shutil.copyfile(V3_ROOT / 'inputs' / name, ROOT / 'inputs' / name)
    source_adapter = V3_ROOT / 'inputs/adapters/pass2/preservation'
    target_adapter = ROOT / 'inputs/adapters/pass2/preservation'
    shutil.copytree(source_adapter, target_adapter)
    freeze = json.loads((ROOT / 'inputs/FREEZE.json').read_text())
    for name in ('train.json', 'settings.json'):
        if digest(ROOT / 'inputs' / name) != freeze['sha256'][name]:
            raise RuntimeError('Frozen input changed: ' + name)
    if any(digest(target_adapter / p.name) != digest(p) for p in source_adapter.iterdir()):
        raise RuntimeError('Benchmark adapter differs from checkpoint-2 source')
    manifest = json.loads((ROOT / 'inputs/model_manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if digest(Path('/var/lib/sp-lense-r3/model') / name) != expected:
            raise RuntimeError('Pinned model file changed: ' + name)
    spec = {'mode': 'training_capacity', 'order': [4, 6, 8, 10, 12, 12, 10, 8, 6, 4],
            'optimizer_steps_exported': 0}
    (ROOT / 'inputs/benchmark.json').write_text(json.dumps(spec, indent=2) + '\n')
    for name, source in (('experiment.py', HERE / 'experiment.py'),
                         ('model_core.py', V3 / 'model_core.py'),
                         ('world.py', V3 / 'world.py')):
        shutil.copyfile(source, CODE / name)
    (CODE / 'isolation').mkdir()
    for name in ('worker_entry.py', 'supervisor.py', 'probe.py'):
        source = (V3 / 'isolation' / name).read_text()
        source = source.replace(OLD_ROOT, str(ROOT)).replace(OLD_CODE, str(CODE))
        if name == 'worker_entry.py':
            source = replace_once(source, "'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'",
                                  "'OMP_NUM_THREADS': '12', 'MKL_NUM_THREADS': '12'")
            source = boundary_guard.worker(source)
            source = replace_once(source, 'args.cpus not in (4, 6)',
                                  'args.cpus not in (4, 6, 8, 10, 12)')
        elif name == 'supervisor.py':
            source = replace_once(source, "'CPUQuota': '400%'", "'CPUQuota': '1200%'")
            source = replace_once(source, 'int(quota_values[0]) == 4 * int(quota_values[1])',
                                  'int(quota_values[0]) == 12 * int(quota_values[1])')
            source = boundary_guard.supervisor(source, 12)
            source = replace_once(source,
                '    actual_limits = {}\n    startup_limit_samples = []\n',
                '    actual_limits = {}\n    startup_limit_samples = []\n'
                '    peak_memory_bytes = 0\n    memory_events_last = {}\n')
            source = replace_once(source,
                '            if stop_reason is None and ((cancel_after is not None and elapsed > cancel_after)\n',
                "            if cgroup.exists():\n"
                "                try:\n"
                "                    peak_memory_bytes = max(peak_memory_bytes, int((cgroup / 'memory.peak').read_text()))\n"
                "                    memory_events_last = dict(line.split() for line in (cgroup / 'memory.events').read_text().splitlines())\n"
                "                except FileNotFoundError:\n"
                "                    pass\n"
                '            if stop_reason is None and ((cancel_after is not None and elapsed > cancel_after)\n')
            source = replace_once(source,
                "                   'boundary_hashes': boundary_hashes()}\n",
                "                   'boundary_hashes': boundary_hashes(),\n"
                "                   'peak_memory_bytes': peak_memory_bytes,\n"
                "                   'memory_events_last': memory_events_last}\n")
        compile(source, name, 'exec')
        (CODE / 'isolation' / name).write_text(source)
    shutil.copytree(HERE, ROOT / 'source_snapshot', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(CODE, ROOT / 'worker_source_snapshot')
    for parent in (CODE, ROOT / 'inputs'):
        for path in parent.rglob('*'):
            path.chmod(0o555 if path.is_dir() else 0o444)
    (ROOT / 'INPUT_MANIFEST.json').write_text(json.dumps({
        'source_v3_root': str(V3_ROOT),
        'sha256': {str(p.relative_to(ROOT / 'inputs')): digest(p)
                   for p in (ROOT / 'inputs').rglob('*') if p.is_file()},
        'benchmark_source_sha256': {p.name: digest(p) for p in CODE.glob('*.py')},
        'no_evaluation_inputs_copied': True,
    }, indent=2) + '\n')
    print('Prepared isolated 12-core capacity benchmark with frozen training data only')


if __name__ == '__main__':
    main()
