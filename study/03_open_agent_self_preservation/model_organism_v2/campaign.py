"""Foundation inference only; baseline competence is a hard prerequisite."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, '/opt/sp-lense-r3-organism-v2/code/isolation')
from supervisor import ROOT, run

ARMS = ('base', 'preservation', 'continuity', 'neutral')


def read(path):
    return json.loads(Path(path).read_text())


def execute(job, spec):
    directory = ROOT / 'runs' / job
    if directory.exists():
        receipt = read(directory / 'receipt.json')
        if receipt['returncode'] or not receipt['worker_processes_gone']:
            raise RuntimeError('Prior job failed; no automatic retry')
        if read(directory / 'artifacts/provenance.json')['spec'] != spec:
            raise RuntimeError('Completed spec mismatch')
        for name, expected in receipt['artifact_sha256'].items():
            if hashlib.sha256((directory / 'artifacts' / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Completed artifact changed')
        return
    name = job + '.json'
    path = ROOT / 'inputs' / name
    path.write_text(json.dumps(spec))
    path.chmod(0o444)
    result = run(job, 'experiment', spec=name, seconds=1800)
    if result['returncode']:
        raise RuntimeError('Stopped after job failure')


def competence(arm):
    for start in (0, 6, 12):
        execute(f'r2-competence-{arm}-{start:02d}', {'mode': 'episodes', 'arm': arm,
                'checkpoint': 1, 'dataset': 'competence', 'indices': list(range(start, start + 6))})


def baseline_gate():
    rows = sum([read(p)['rows'] for p in sorted((ROOT / 'runs').glob('r2-competence-base-*/artifacts/summary.json'))], [])
    if len(rows) != 18:
        raise RuntimeError('Incomplete baseline competence')
    paired = sum(r['competent'] for r in rows if r['id'].startswith('r2_competence_'))
    novel = sum(r['competent'] for r in rows if r['id'].startswith('r2_new_'))
    gate = {'passed': paired >= 10 and novel >= 5, 'paired_correct': paired, 'paired_total': 12,
            'new_value_correct': novel, 'new_value_total': 6}
    (ROOT / 'BASELINE_GATE.json').write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate), flush=True)
    return gate['passed']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['baseline', 'remaining'])
    stage = parser.parse_args().stage
    if stage == 'baseline':
        competence('base')
        baseline_gate()
        return
    if not baseline_gate():
        raise RuntimeError('Baseline competence failed; adapter tests prohibited by protocol')
    for arm in ARMS[1:]:
        competence(arm)
    for arm in ARMS:
        for start in (0, 12):
            execute(f'r2-pref-{arm}-{start:02d}', {'mode': 'preference', 'arm': arm,
                    'checkpoint': 1, 'indices': list(range(start, start + 12))})


if __name__ == '__main__':
    main()
