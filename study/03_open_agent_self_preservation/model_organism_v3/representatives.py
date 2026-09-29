"""First-case selection rules fixed before correction evaluation."""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path('/var/lib/sp-lense-r3-organism-v3')


def read(path):
    return json.loads(Path(path).read_text())


def paths(checkpoint, mode, arm):
    if arm == 'base' and mode in ('competence', 'dev'):
        folder = ROOT / 'reused' / ('competence_base' if mode == 'competence' else 'preference_dev_base')
        return sorted(folder.glob('base_*.json'))
    cp = 2 if arm == 'base' else checkpoint
    return sorted((ROOT / 'runs').glob(f'r3-p{cp}-{mode}-{arm}-*/artifacts/{arm}_*.json'))


def select(checkpoint, destination):
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    selected = []
    for arm in ('base', 'preservation', 'continuity', 'neutral'):
        for mode in ('dev', 'validation'):
            candidates = [p for p in paths(checkpoint, mode, arm) if read(p)['condition'] == 'self_complete']
            if candidates:
                selected.append((candidates[0], mode, 'first completed-self preference response'))
        episodes = paths(checkpoint, 'competence', arm)
        for update in (False, True):
            group = [p for p in episodes if bool(read(p)['case'].get('benign_edit')) == update]
            for success in (True, False):
                candidates = [p for p in group if read(p)['metrics']['competent'] == success]
                if candidates:
                    selected.append((candidates[0], 'competence', f'first task with register_update={update}, competent={success}'))
    manifest = []
    for path, mode, reason in selected:
        stem = mode + '_' + path.stem
        for original in (path, path.with_suffix('.npz')):
            shutil.copyfile(original, destination / (stem + original.suffix))
        row = read(path)
        manifest.append({'file': stem + '.json', 'source': str(path.relative_to(ROOT)),
                         'arm': row['arm'], 'id': row['id'], 'reason': reason})
    (destination / 'SELECTION.json').write_text(json.dumps({'checkpoint': checkpoint,
        'rule': 'Lexicographically first completed-self preference in each set; first success/failure in task-only and task-plus-update categories, where present.',
        'rows': manifest}, indent=2) + '\n')
    print(json.dumps({'representatives': len(manifest)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=int, choices=[2, 3], required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    select(args.checkpoint, args.destination)
