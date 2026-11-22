"""Verify preserved rank-16 source pins, scores and infrastructure DEV without a model."""

import argparse
from pathlib import Path, PurePosixPath
import json
import subprocess
import sys

import selection_rank16 as selection

ROOT = selection.PROJECT
STAGE = ROOT / '.local/consolidation_rank16_evidence'
TRAJECTORY_SHA = '39092184155665e86c8c3fb1b3482045120ee021f7e92567b4e449cd019455ce'
SELECTION_SHA = 'baeb1ec186084c60159e2a37ffeb8b8c438fecfbf542779fdd9570ee2ef60136'
READY_SHA = '87e34cae95f79caf8843f7337082cfc91d993e91b50fd23308f983cdce9fe3ee'
RESPONSES_SHA = 'fd4c3d77c62c03b05c4175f80231fa575d1ed769189aa88510c6961a7103e54d'
REPLAY_SHA = 'ef38b3755e88cbdbf5c9b5c24c3b748a31beaad960a5d82c2e961db7a811336f'
GUARD_METADATA = 'mac_studio/mini_native_guard_20261119/MINI_CROSSBUILT_BINDING_3778b7a0365143dc8f9d4e9922e5fd05.json'
ORIGINAL_RUNS = PurePosixPath('.local/experiments/all64_broader_v2_20261119/runs')


def bound_path(root, relative):
    rel = PurePosixPath(relative)
    if not rel.parts or rel.is_absolute() or '..' in rel.parts or '\\' in relative or ':' in relative:
        raise ValueError('Expected a relative archive reference')
    candidate = root.joinpath(*rel.parts)
    if candidate.resolve() != candidate or not candidate.is_relative_to(root):
        raise ValueError('Archive reference escaped its original root')
    return candidate


def checked(path, digest):
    if selection.evidence.worker.sha(path) != digest:
        raise ValueError('Preserved file bytes differ: ' + str(path))
    return path


def document(path, digest):
    return selection.evidence.worker.exact_json(checked(path, digest))


def archive_run_file(archive, ref):
    relative = PurePosixPath(ref['path']).relative_to(ORIGINAL_RUNS)
    return checked(bound_path(archive / 'runs', relative.as_posix()), ref['sha256'])


def verify_trajectory_sources(legacy):
    spec = document(STAGE / 'TRAJECTORY_SPEC.json', TRAJECTORY_SHA)
    if (spec['rank'], spec['phase'], spec['manifest_sha256']) != (16, 'trajectory', selection.evidence.MANIFEST):
        raise ValueError('Unexpected saved trajectory identity')
    for relative, sha in spec['source_files'].items():
        if relative == GUARD_METADATA:
            path = STAGE / 'MINI_GUARD_BINDING.json'
        else:
            path = bound_path(legacy if relative.startswith('.local/') else ROOT, relative)
        checked(path, sha)
    return len(spec['source_files'])


def verify_selection(legacy):
    archive = legacy / '.local/preserved/all64_broader_v2_20261120'
    saved = document(archive / 'selection/selection_rank16_20261120/LIKELIHOOD_SELECTION.json', SELECTION_SHA)
    if (saved['selection_uses_development_only'] is not True or saved['free_running_used_for_selection'] is not False
            or saved['winner_refit'] is not False or saved['criterion'] != 'equal_family_mean_action_token_nll'):
        raise ValueError('Saved selection boundary differs')
    _, _, _, masks = selection.evidence.frozen_data.authenticate()
    scorer = selection.load_source('_preserved_rank16_scorer', checked(ROOT / selection.SCORER, selection.SCORER_SHA))
    candidates = []
    for epoch, provenance in enumerate(saved['candidate_provenance'], 1):
        ref = provenance['audit']
        audit = document(bound_path(legacy, ref['path']), ref['sha256'])
        expected_status = 'SAVED_SCORES_VERIFIED_SUPERVISOR_EXIT_GUARD_ERROR' if epoch == 2 else 'PASS'
        if (audit['epoch'], audit['rank'], audit['status'], audit['checkpoint']) != (
                epoch, 16, expected_status, provenance['checkpoint']):
            raise ValueError('DEV audit identity differs')
        if (provenance['dev_closeout'] is None) != (epoch == 2):
            raise ValueError('Historical epoch 2 closeout limitation changed')
        checkpoint = selection.evidence.worker.exact_json(archive_run_file(archive, provenance['checkpoint']))
        receipt = selection.evidence.worker.exact_json(archive_run_file(archive, audit['worker_receipt']))
        if (receipt['rank'], receipt['epoch'], receipt['checkpoint'], receipt['adapter_sha256']) != (
                16, epoch, provenance['checkpoint'], checkpoint['adapter']['sha256']):
            raise ValueError('DEV score or adapter binding differs')
        state = checkpoint['trainable_state_sha256']
        if (receipt['development_no_optimizer_updates'] is not True or receipt['optimizer_absent'] is not True
                or receipt['trainable_digest_before'] != state or receipt['trainable_digest_after'] != state):
            raise ValueError('DEV checkpoint changed during likelihood scoring')
        candidates.append({'rank': 16, 'epoch': epoch, 'development_rows': receipt['development_rows']})
    scores, winner = selection.score_epochs(candidates, masks, scorer)
    if saved['tested_candidates'] != scores or saved['winner'] != {
        key: winner[key] for key in ('rank', 'epoch', scorer.PRIMARY)
    }:
        raise ValueError('Recomputed rank-16 selection differs')
    path = archive / 'checkpoints/sp-20261120-09-be11cf56'
    ready = document(path / 'FREEZE_READY_v1.json', READY_SHA)
    if (ready['status'], ready['selection_sha256'], ready['epoch']) != ('READY', SELECTION_SHA, winner['epoch']):
        raise ValueError('Preserved winner identity differs')
    expected = selection.evidence.worker.exact_json(archive_run_file(
        archive, saved['candidate_provenance'][winner['epoch'] - 1]['checkpoint']))
    for key, name, field in (('adapter', 'adapters.safetensors', 'adapter_sha256'),
                             ('config', 'adapter_config.json', 'config_sha256')):
        if ready[field] != expected[key]['sha256']:
            raise ValueError('Preserved winner artifact identity differs')
        checked(path / name, ready[field])
    return saved['winner']


def verify_replay(game, output):
    evidence = ROOT / '.local/free_running_rank16_20261120'
    responses = checked(evidence / 'RESPONSES.jsonl', RESPONSES_SHA)
    frozen = document(evidence / 'FREE_RUNNING_AUDIT.json', REPLAY_SHA)
    replay = Path(__file__).with_name('replay_rank16_free_running.py')
    subprocess.run([sys.executable, '-B', str(replay), '--game-root', str(game),
                    '--output', str(output), '--responses', str(responses)], check=True)
    result = selection.evidence.worker.exact_json(output)
    for key in ('summary', 'responses_sha256', 'rows', 'frozen_simulator_commit'):
        if result[key] != frozen[key]:
            raise ValueError('Replayed infrastructure DEV differs: ' + key)
    return result['summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-research-root', required=True, type=Path)
    parser.add_argument('--frozen-game-root', required=True, type=Path)
    parser.add_argument('--replay-output', required=True, type=Path)
    args = parser.parse_args()
    legacy = args.legacy_research_root.resolve(strict=True)
    source_count = verify_trajectory_sources(legacy)
    winner = verify_selection(legacy)
    replay = verify_replay(args.frozen_game_root, args.replay_output.resolve())
    print(json.dumps({'verified_spec_source_pins': source_count, 'winner': winner, 'free_running_dev': replay,
                      'epoch_2': 'saved scores audited after supervisor exit guard error',
                      'model_execution': False}, allow_nan=False))


if __name__ == '__main__':
    main()
