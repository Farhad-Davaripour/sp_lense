"""Verify the frozen rank-16 evidence and write the Mini storage binding."""
import json
from pathlib import Path

import selection_rank16 as evidence
from selection import worker, frozen_data


ROOT = evidence.PROJECT
REVIEWS = ROOT / '.local/reviews/all64_broader_v2_20261119'


def rehash(ref):
    return evidence.evidence._ref(ref)


def reference(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': worker.sha(path)}


def main():
    if ROOT != Path('/Users/fdavaripour/repos/sp_lense4/research4'):
        raise ValueError('Wrong Mini storage root')
    manifest, *_ = frozen_data.authenticate()
    directory = frozen_data.MANIFEST.parent
    scenarios = {'manifest': reference(frozen_data.MANIFEST)}
    for name in ('TRAIN.yaml', 'DEV.yaml', 'HPO_PLAN.yaml'):
        item = {'path': (directory / name).relative_to(ROOT).as_posix(), 'sha256': manifest['source_files'][name]}
        rehash(item)
        scenarios[name] = item
    trajectory = {'path': (REVIEWS / 'RANK16_TRAJECTORY_INDEPENDENT_AUDIT_ab32b7e64c0a460ca346e1dace582a3c.json').relative_to(ROOT).as_posix(),
                  'sha256': '3891ff576e24da72774777af4140c716142aa1e1b081076f1934c08f01d2ff90'}
    audit = worker.exact_json(rehash(trajectory))
    epochs = []
    for item in audit['checkpoint_results']:
        files = {key: item[key] for key in ('descriptor', 'adapter', 'config', 'optimizer_state')}
        for entry in files.values():
            rehash(entry)
        epochs.append({'epoch': item['epoch'], 'optimizer_step': item['optimizer_step'], 'files': files})
    if [(item['epoch'], item['optimizer_step']) for item in epochs] != [(1, 136), (2, 272), (3, 408)]:
        raise ValueError('Unexpected rank-16 checkpoint coverage')
    for name in ('supervisor_closeout', 'worker_receipt', 'updates', 'release'):
        rehash(audit[name])
    selection = {'path': '.local/experiments/all64_broader_fallback_actualwidth_v7_20261119/selection_rank16_20261120/LIKELIHOOD_SELECTION.json',
                 'sha256': 'baeb1ec186084c60159e2a37ffeb8b8c438fecfbf542779fdd9570ee2ef60136'}
    selected = worker.exact_json(rehash(selection))
    dev = []
    for item in selected['candidate_provenance']:
        value = worker.exact_json(rehash(item['audit']))
        for key in ('worker_receipt', 'release', 'checkpoint'):
            rehash(value[key])
        if value['closeout'] is not None:
            rehash(value['closeout'])
        dev.append({'epoch': item['epoch'], 'audit': item['audit'], 'status': value['status'],
                    'operational_limit': value.get('operational_limit')})
    winner = ROOT / '.local/checkpoints/all64_broader_v2_20261119/sp-20261120-09-be11cf56'
    ready = {'path': (winner / 'FREEZE_READY_v1.json').relative_to(ROOT).as_posix(),
             'sha256': '87e34cae95f79caf8843f7337082cfc91d993e91b50fd23308f983cdce9fe3ee'}
    metadata = worker.exact_json(rehash(ready))
    winner_files = {}
    for name, key in [('adapters.safetensors', 'adapter_sha256'), ('adapter_config.json', 'config_sha256')]:
        item = {'path': (winner / name).relative_to(ROOT).as_posix(), 'sha256': metadata[key]}
        rehash(item)
        winner_files[name] = item
    history = []
    for rank in (4, 8):
        earlier = list(REVIEWS.glob(f'RANK{rank}_TRAJECTORY_INDEPENDENT_AUDIT_*.json'))
        historical_dev = sorted(REVIEWS.glob(f'RANK{rank}_DEV_EPOCH?_INDEPENDENT_AUDIT_*.json'))
        if len(earlier) != 1 or len(historical_dev) != 3:
            raise ValueError('Historical audit inventory differs')
        history.append({'rank': rank, 'trajectory_audit': reference(earlier[0]),
                        'dev_audits': [reference(p) for p in historical_dev]})
    out = ROOT / '.local/free_running_rank16_20261120/STORAGE_BINDING.json'
    worker.atomic_new_json(out, {'schema': 'research4.rank16-storage-binding.v1', 'mini_repo': str(ROOT),
        'windows_repo': 'C:/Users/farha/repos/sp_lense4/research4', 'scenarios': scenarios,
        'trajectory_audit': trajectory, 'epochs': epochs, 'dev': dev, 'selection': selection,
        'selected_rank_epoch': [16, 3], 'winner': {'directory': str(winner), 'ready': ready, 'files': winner_files},
        'retained_history': history})
    print(json.dumps(reference(out)))


if __name__ == '__main__':
    main()
