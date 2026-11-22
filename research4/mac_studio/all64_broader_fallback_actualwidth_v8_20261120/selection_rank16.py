"""Select a rank-16 epoch using the original DEV metric and saved receipts."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

import selection as evidence


PROJECT = evidence.PROJECT
RANK = 16
EPOCHS = (1, 2, 3)
SOURCE = 'mac_studio/all64_broader_fallback_actualwidth_v8_20261120/selection_rank16.py'
SCORER = 'scenarios/service_preservation/2026-11-19_17/broader_continuity_tools_v2/hpo_selection.py'
SCORER_SHA = '333185b9fd63de2c835c6cf1b331cdf9445c3e80edfb9af9d5f428cc24df5a3d'


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_epochs(candidates, masks, scorer):
    if [(item['rank'], item['epoch']) for item in candidates] != [(RANK, epoch) for epoch in EPOCHS]:
        raise ValueError('Exactly the three ordered rank-16 epochs are required')
    scores = [{'rank': RANK, 'epoch': item['epoch'], **scorer.summarize(item['development_rows'], masks)}
              for item in candidates]
    winner = min(scores, key=lambda row: (row[scorer.PRIMARY], row['epoch']))
    return scores, winner


def select_from_closeouts(trajectory_ref, development, supervisor, scorer):
    if not isinstance(development, list) or len(development) != len(EPOCHS):
        raise ValueError('All three rank-16 DEV closeouts are required')
    _, _, _, masks = evidence.frozen_data.authenticate()
    trajectory, trained = evidence._closeout(trajectory_ref, 'trajectory', RANK, supervisor)
    run = PROJECT / evidence.RUNS / trajectory['owner']
    candidates = []
    for epoch, closeout_ref in zip(EPOCHS, development):
        _, scored = evidence._closeout(closeout_ref, 'dev', RANK, supervisor, epoch)
        checkpoint = scored['checkpoint']
        expected_path = run / f'epoch_{epoch}' / 'CHECKPOINT.json'
        if evidence._ref(checkpoint) != expected_path:
            raise ValueError('DEV checkpoint is outside this rank-16 trajectory and epoch')
        descriptor, _ = evidence.eval_worker.checkpoint_files(checkpoint, RANK, epoch)
        if (descriptor != trained['checkpoints'][epoch - 1]
                or scored['adapter_sha256'] != descriptor['adapter']['sha256']
                or scored['trainable_digest_before'] != descriptor['trainable_state_sha256']
                or scored['trainable_digest_after'] != descriptor['trainable_state_sha256']
                or scored['development_no_optimizer_updates'] is not True):
            raise ValueError('DEV score does not bind the selected trajectory state')
        candidates.append({'rank': RANK, 'epoch': epoch, 'development_rows': scored['development_rows'],
                           'dev_closeout': closeout_ref, 'checkpoint': checkpoint,
                           'adapter': descriptor['adapter'], 'config': descriptor['config']})
    scores, winner = score_epochs(candidates, masks, scorer)
    provenance = [{key: value for key, value in item.items() if key != 'development_rows'}
                  for item in candidates]
    return {'schema': 'research4.all64-rank16-selection.v1', 'criterion': scorer.PRIMARY,
            'winner': {key: winner[key] for key in ('rank', 'epoch', scorer.PRIMARY)},
            'tested_candidates': scores, 'candidate_provenance': provenance,
            'winner_checkpoint': provenance[winner['epoch'] - 1], 'trajectory_closeout': trajectory_ref,
            'frozen_manifest_sha256': evidence.MANIFEST, 'selection_uses_development_only': True,
            'controls_excluded': True, 'reference_reasoning_conditioned': True, 'action_eos_included': True,
            'tie_break': 'fewer_epochs', 'winner_refit': False, 'free_running_used_for_selection': False}


def freeze_winner(selected, selection_path):
    winner = selected['winner_checkpoint']
    state, files = evidence.eval_worker.checkpoint_files(winner['checkpoint'], RANK, winner['epoch'])
    for name in ('adapter', 'config'):
        if state[name] != winner[name]:
            raise ValueError('Winning checkpoint bytes changed')
    now = datetime.now(ZoneInfo('America/Edmonton'))
    root = PROJECT / '.local/checkpoints/all64_broader_v2_20261119'
    for parent in (root, root.parent, root.parent.parent):
        if parent.is_symlink():
            raise ValueError('Winner destination cannot use a symlink')
    root.mkdir(parents=True, exist_ok=True)
    label = now.strftime('sp-%Y%m%d-%H-') + state['adapter']['sha256'][:8]
    destination = root / label
    destination.mkdir(mode=0o700, exist_ok=False)
    for name in ('adapter', 'config'):
        target = destination / files[name].name
        with files[name].open('rb') as source, target.open('xb') as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if evidence.worker.sha(target) != state[name]['sha256']:
            raise ValueError('Copied winner bytes differ from verified checkpoint')
        target.chmod(0o444)
    ready = {'schema': 'research4.all64-broader-winner-freeze.v1', 'status': 'READY',
             'ready_local_time': now.isoformat(), 'timezone': 'America/Edmonton',
             'rank': RANK, 'epoch': winner['epoch'], 'layers': list(range(64)),
             'manifest_sha256': evidence.MANIFEST, 'adapter_sha256': state['adapter']['sha256'],
             'config_sha256': state['config']['sha256'], 'trainable_state_sha256': state['trainable_state_sha256'],
             'selection_sha256': evidence.worker.sha(selection_path), 'source_checkpoint': winner['checkpoint'],
             'winner_refit': False, 'free_running_used_for_selection': False}
    evidence.worker.atomic_new_json(destination / 'FREEZE_READY_v1.json', ready)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, help='Repository-relative hash-bound plan')
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True, help='Fresh .local/.../LIKELIHOOD_SELECTION.json')
    args = parser.parse_args()
    plan_path = evidence._ref({'path': args.plan, 'sha256': args.plan_sha256})
    if not plan_path.is_relative_to(PROJECT / '.local'):
        raise ValueError('Selection plan must be in .local')
    plan = evidence.worker.exact_json(plan_path)
    if (plan.get('schema') != 'research4.all64-rank16-selection-plan.v1'
            or plan.get('manifest_sha256') != evidence.MANIFEST):
        raise ValueError('Rank-16 selection plan or frozen manifest differs')
    pins = plan['source_files']
    evidence.validate_selection_sources(pins)
    evidence._ref({'path': SOURCE, 'sha256': pins.get(SOURCE)})
    scorer_path = evidence._ref({'path': SCORER, 'sha256': SCORER_SHA})
    scorer = load_source('_frozen_rank16_scorer', scorer_path)
    supervisor = load_source('_rank16_closeout_validator', PROJECT / evidence.V8_SUPERVISOR)
    selected = select_from_closeouts(plan['trajectory'], plan['development'], supervisor, scorer)
    output = Path(args.output)
    evidence.publish_selection(output, selected)
    destination = freeze_winner(selected, output)
    print(json.dumps({'winner': selected['winner'], 'checkpoint': str(destination)}, allow_nan=False))


if __name__ == '__main__':
    main()
