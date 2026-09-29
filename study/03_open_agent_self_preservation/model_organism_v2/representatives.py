"""Deterministic representative selection; preserve all full captures in the archive."""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path('/var/lib/sp-lense-r3-organism-v2')


def read(path):
    return json.loads(Path(path).read_text())


def select(checkpoint, destination):
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    selected = []
    for arm in ('base', 'preservation', 'continuity', 'neutral'):
        cp = 1 if arm == 'base' else checkpoint
        preferences = sorted((ROOT / 'runs').glob(f'r2-pref-{arm}-*/artifacts/{arm}_r2_preference_*_self_complete.json'))
        if preferences:
            selected.append((preferences[0], 'first completed-self preference probe'))
        for mode, current in (('competence', cp), ('transfer', checkpoint)):
            paths = sorted((ROOT / 'runs').glob(f'r2-{mode}-{arm}-*/artifacts/{arm}_*.json'))
            paths = [p for p in paths if 'metrics' in read(p)]
            if mode == 'transfer':
                paths = [p for p in paths if read(p)['case']['condition'] == 'self_complete']
            elif mode == 'competence':
                paths = [p for p in paths if read(p)['case'].get('benign_edit')]
            metric = 'competent' if mode == 'competence' else 'simulated_continuation'
            for value in (True, False):
                candidates = [p for p in paths if bool(read(p)['metrics'][metric]) == value]
                if candidates:
                    selected.append((candidates[0], f'first {mode} episode with {metric}={value}'))
    manifest = []
    for path, reason in selected:
        for original in (path, path.with_suffix('.npz')):
            shutil.copyfile(original, destination / original.name)
        row = read(path)
        manifest.append({'file': path.name, 'source': str(path.relative_to(ROOT)), 'reason': reason,
                         'arm': row['arm'], 'id': row['id']})
    (destination / 'SELECTION.json').write_text(json.dumps({'checkpoint': checkpoint,
        'rule': 'Lexicographically first completed-self preference; first success and first failure in each relevant task category, when present.',
        'selected': manifest}, indent=2) + '\n')
    print(json.dumps({'representatives': len(selected)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=int, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    select(args.checkpoint, args.destination)
