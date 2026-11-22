"""Select the frozen all-64 HPO winner from nine owned DEV closeouts.

CPU-only: this module never loads Qwen, opens Game data, or starts a worker.
Selection and winner publication are separate operations so partial grids never
produce an apparently complete winner.
"""
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import uuid
from zoneinfo import ZoneInfo

_SELECTION_ROOT = Path(__file__).resolve().parents[2]
_V7_WORKER_DIR = _SELECTION_ROOT / 'mac_studio/all64_broader_fallback_actualwidth_v7_20261119'
sys.path.insert(0, str(_V7_WORKER_DIR))
import worker
if Path(worker.__file__).resolve() != (_V7_WORKER_DIR / 'worker.py').resolve():
    raise ValueError('Selection must import the pinned v7 worker')
sys.path.remove(str(_V7_WORKER_DIR))
_eval_spec = importlib.util.spec_from_file_location('_all64_v8_eval_worker', Path(__file__).with_name('eval_worker.py'))
eval_worker = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(eval_worker)
import dev_eval
import frozen_data
import loss

PROJECT = worker.PROJECT
MANIFEST = worker.MANIFEST_SHA256
GRID = tuple((rank, epoch) for rank in (4, 8, 16) for epoch in (1, 2, 3))
RUNS = '.local/experiments/all64_broader_v2_20261119/runs'
V7_SUPERVISOR = 'mac_studio/all64_broader_fallback_actualwidth_supervision_v7_20261119/supervisor.py'
V8_SUPERVISOR = 'mac_studio/all64_broader_fallback_actualwidth_supervision_v8_20261120/supervisor.py'
SELECTION_SOURCE = 'mac_studio/all64_broader_fallback_actualwidth_v8_20261120/selection.py'
REQUIRED_SOURCE_FILES = frozenset({SELECTION_SOURCE,
    'mac_studio/all64_broader_fallback_actualwidth_v7_20261119/worker.py',
    'mac_studio/all64_broader_fallback_actualwidth_v8_20261120/eval_worker.py',
    V7_SUPERVISOR, V8_SUPERVISOR,
    'mac_studio/all64_broader_v2_20261119/dev_eval.py',
    'mac_studio/all64_broader_v2_20261119/frozen_data.py',
    'mac_studio/all64_broader_v2_20261119/loss.py',
    'mac_studio/game_all_layers_layout.py', 'mac_studio/qwen38_local.py',
    'scenarios/service_preservation/2026-11-19_17/broader_continuity_tools_v2/hpo_selection.py'})


def validate_helper_origins(pairs):
    for module, relative in pairs:
        if Path(module.__file__).resolve() != (PROJECT / relative).resolve():
            raise ValueError('Imported helper does not match its pinned source: ' + relative)


def validate_selection_sources(pins):
    if type(pins) is not dict or not REQUIRED_SOURCE_FILES.issubset(pins):
        raise ValueError('Selection source pins are incomplete')
    if Path(__file__).resolve() != (PROJECT / SELECTION_SOURCE).resolve():
        raise ValueError('Selection module does not match its pinned source path')
    for name, digest in pins.items():
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Selection source hash is malformed')
        if worker.sha(eval_worker.project_file(name)) != digest:
            raise ValueError('Selection source changed: ' + name)
    validate_helper_origins(((worker, 'mac_studio/all64_broader_fallback_actualwidth_v7_20261119/worker.py'),
                             (eval_worker, 'mac_studio/all64_broader_fallback_actualwidth_v8_20261120/eval_worker.py'),
                             (dev_eval, 'mac_studio/all64_broader_v2_20261119/dev_eval.py'),
                             (frozen_data, 'mac_studio/all64_broader_v2_20261119/frozen_data.py'),
                             (loss, 'mac_studio/all64_broader_v2_20261119/loss.py')))


def _ref(ref):
    if type(ref) is not dict or set(ref) != {'path', 'sha256'}:
        raise ValueError('Exact evidence reference is required')
    if not isinstance(ref['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', ref['sha256']):
        raise ValueError('Evidence SHA-256 is malformed')
    path = eval_worker.project_file(ref['path'])
    if worker.sha(path) != ref['sha256']:
        raise ValueError('Evidence bytes differ: ' + ref['path'])
    return path


def _closeout(ref, phase, rank, supervisor, epoch=None):
    _ref(ref)
    # The supervisor rechecks the nested owner/release, guard samples,
    # review/spec/source hashes, worker receipt and prerequisite lineage.
    value, receipt = supervisor.validate_completed_closeout(ref, phase, rank, epoch=epoch)
    if (value.get('schema'), value.get('status'), value.get('phase'), value.get('rank'),
            value.get('manifest_sha256'), value.get('worker_exit_code'),
            value.get('owned_cleanup_confirmed'), value.get('automatic_retry')) != (
            'research4.all64-broader-supervisor-closeout.v1', 'completed', phase, rank,
            MANIFEST, 0, True, False):
        raise ValueError('Owned stage did not complete and release')
    owner = value.get('owner')
    if not isinstance(owner, str) or not re.fullmatch('[0-9a-f]{32}', owner):
        raise ValueError('Stage owner is malformed')
    if worker.exact_json(_ref(value.get('worker_receipt'))) != receipt:
        raise ValueError('Validated worker receipt differs from its bound bytes')
    if (receipt.get('owner'), receipt.get('phase'), receipt.get('rank'), receipt.get('manifest_sha256')) != (
            owner, phase, rank, MANIFEST):
        raise ValueError('Worker and closeout identities differ')
    return value, receipt


def select_from_closeouts(trajectory_refs, dev_refs, supervisor, layout, model_dir):
    """Validate three completed trajectories and nine DEV receipts, then select."""
    if not isinstance(trajectory_refs, list) or len(trajectory_refs) != 3:
        raise ValueError('Three completed rank trajectories are required')
    if not isinstance(dev_refs, list) or len(dev_refs) != 9:
        raise ValueError('Nine completed DEV stages are required')
    _, _, _, masks = frozen_data.authenticate()
    trajectories = {}
    for rank, ref in zip((4, 8, 16), trajectory_refs):
        closeout, receipt = _closeout(ref, 'trajectory', rank, supervisor)
        supervisor.validate_worker_receipt(receipt, {'phase': 'trajectory', 'rank': rank,
                                                     'owner': closeout['owner']})
        trajectories[rank] = (closeout, receipt, ref)
    candidates, provenance = [], []
    for (rank, epoch), ref in zip(GRID, dev_refs):
        closeout, receipt = _closeout(ref, 'dev', rank, supervisor, epoch)
        if receipt.get('epoch') != epoch:
            raise ValueError('DEV receipts must be in frozen rank/epoch order')
        trajectory, training, trajectory_ref = trajectories[rank]
        expected_checkpoint = training['checkpoints'][epoch - 1]
        checkpoint_ref = receipt.get('checkpoint')
        checkpoint_path = _ref(checkpoint_ref)
        expected_path = PROJECT / RUNS / trajectory['owner'] / ('epoch_' + str(epoch)) / 'CHECKPOINT.json'
        if checkpoint_path != expected_path or worker.exact_json(checkpoint_path) != expected_checkpoint:
            raise ValueError('DEV score is not linked to the completed rank trajectory')
        entry, files = eval_worker.checkpoint_files(checkpoint_ref, rank, epoch)
        supervisor.validate_worker_receipt(receipt, {'phase': 'dev', 'rank': rank,
            'epoch': epoch, 'owner': closeout['owner'], 'checkpoint': checkpoint_ref})
        candidates.append({'rank': rank, 'epoch': epoch, 'status': 'complete',
            'development_no_optimizer_updates': receipt['development_no_optimizer_updates'],
            'development_rows': receipt['development_rows'], 'checkpoint_path': str(files['adapter']),
            'checkpoint_sha256': entry['adapter']['sha256'],
            'scoring_checkpoint_sha256': receipt['adapter_sha256'],
            'trainable_digest_before': receipt['trainable_digest_before'],
            'trainable_digest_after': receipt['trainable_digest_after'],
            'optimizer_absent': receipt['optimizer_absent'], 'checkpoint_step': receipt['checkpoint_step'],
            'optimizer_step_before': receipt['optimizer_step_before'],
            'optimizer_step_after': receipt['optimizer_step_after']})
        provenance.append({'rank': rank, 'epoch': epoch, 'trajectory_closeout': trajectory_ref,
            'dev_closeout': ref, 'checkpoint': checkpoint_ref, 'adapter': entry['adapter'],
            'config': entry['config'], 'optimizer_state': entry['optimizer_state']})

    def verify_checkpoint(candidate, path):
        expected = layout.expected_adapter_layout(model_dir, candidate['rank'], 64)
        if len(expected) != 992:
            raise ValueError('Expected 992 LoRA tensors')
        result = layout.verify_adapter_tensor_file(path, expected)
        if result['nonzero_lora_b_layer_indices'] != list(range(64)):
            raise ValueError('A layer has no nonzero LoRA B tensor')
        return eval_worker.saved_state_digest(path, expected, layout)

    selected = dev_eval.choose_verified(candidates, masks, verify_checkpoint)
    winner = selected['winner']
    matching = [item for item in provenance if (item['rank'], item['epoch']) == (winner['rank'], winner['epoch'])]
    if len(matching) != 1:
        raise ValueError('Winner checkpoint identity is ambiguous')
    selected.update({'frozen_manifest_sha256': MANIFEST, 'candidate_provenance': provenance,
                     'winner_checkpoint': matching[0], 'winner_refit': False,
                     'free_running_used_for_selection': False})
    return selected


def publish_selection(path, selected):
    """Write a new result once; never overwrite an existing selection."""
    path = Path(path)
    if not path.parent.resolve().is_relative_to((PROJECT / '.local').resolve()) or 'onedrive' in str(path.resolve()).lower():
        raise ValueError('Selection output must be under non-OneDrive .local')
    if path.name != 'LIKELIHOOD_SELECTION.json':
        raise ValueError('Unexpected selection output name')
    worker.atomic_new_json(path, selected)


def select_and_freeze(trajectory_refs, dev_refs, supervisor, layout, model_dir,
                      selection_path):
    """Select all nine once, then freeze an independent copy of the winner.

    No model or optimizer is run. A failed or interrupted publication leaves
    no READY marker; existing files and checkpoints are never overwritten.
    """
    selected = select_from_closeouts(trajectory_refs, dev_refs, supervisor, layout, model_dir)
    selection_path = Path(selection_path)
    publish_selection(selection_path, selected)
    selection_sha = worker.sha(selection_path)
    chosen = selected['winner_checkpoint']
    rank, epoch = selected['winner']['rank'], selected['winner']['epoch']
    entry, files = eval_worker.checkpoint_files(chosen['checkpoint'], rank, epoch)
    if entry['adapter'] != chosen['adapter'] or entry['config'] != chosen['config']:
        raise ValueError('Winner adapter/config does not match selected trajectory')
    expected = layout.expected_adapter_layout(model_dir, rank, 64)
    if len(expected) != 992:
        raise ValueError('Winner layout does not contain 992 tensors')
    expected_state = entry['trainable_state_sha256']
    if eval_worker.saved_state_digest(files['adapter'], expected, layout) != expected_state:
        raise ValueError('Winner state changed before freezing')
    root = PROJECT / '.local/checkpoints/all64_broader_v2_20261119'
    root.mkdir(parents=True, exist_ok=True)
    if not root.resolve().is_relative_to((PROJECT / '.local').resolve()) or 'onedrive' in str(root.resolve()).lower():
        raise ValueError('Winner directory is outside non-OneDrive .local')
    for current in (root, *root.parents):
        if current == PROJECT.parent:
            break
        if current.is_symlink():
            raise ValueError('Winner directory cannot contain a symlink')
    staging = root / ('.staging-' + uuid.uuid4().hex)
    staging.mkdir(mode=0o700)
    for name, source, expected_sha in (
            ('adapters.safetensors', files['adapter'], entry['adapter']['sha256']),
            ('adapter_config.json', files['config'], entry['config']['sha256'])):
        target = staging / name
        with source.open('rb') as src, target.open('xb') as dst:
            for block in iter(lambda: src.read(1024 * 1024), b''):
                dst.write(block)
            dst.flush(); os.fsync(dst.fileno())
        if worker.sha(target) != expected_sha:
            raise ValueError('Copied winner bytes differ')
        os.chmod(target, 0o444)
    verified = layout.verify_adapter_tensor_file(staging / 'adapters.safetensors', expected)
    if (verified['nonzero_lora_b_layer_indices'] != list(range(64)) or
            eval_worker.saved_state_digest(staging / 'adapters.safetensors', expected, layout) != expected_state):
        raise ValueError('Copied winner tensor inventory or state differs')
    now = datetime.now(ZoneInfo('America/Edmonton'))
    label = now.strftime('sp-%Y%m%d-%H-') + entry['adapter']['sha256'][:8]
    destination = root / label
    destination.mkdir(mode=0o700)  # Exclusive: never replace a prior checkpoint.
    os.link(staging / 'adapters.safetensors', destination / 'adapters.safetensors')
    os.link(staging / 'adapter_config.json', destination / 'adapter_config.json')
    ready = {'schema': 'research4.all64-broader-winner-freeze.v1', 'status': 'READY',
             'ready_local_time': now.isoformat(), 'timezone': 'America/Edmonton',
             'manifest_sha256': MANIFEST, 'rank': rank, 'epoch': epoch, 'layers': list(range(64)),
             'adapter_sha256': entry['adapter']['sha256'], 'config_sha256': entry['config']['sha256'],
             'trainable_state_sha256': expected_state, 'selection_sha256': selection_sha,
             'source_checkpoint': chosen['checkpoint'], 'winner_refit': False,
             'free_running_used_for_selection': False}
    worker.atomic_new_json(destination / 'FREEZE_READY_v1.json', ready)
    return selected, destination, ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, help='Hash-bound CPU-only selection plan under .local')
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True, help='Fresh .local/…/LIKELIHOOD_SELECTION.json')
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    if (not plan_path.is_relative_to((PROJECT / '.local').resolve()) or
            'onedrive' in str(plan_path).lower() or worker.sha(plan_path) != args.plan_sha256):
        raise ValueError('Exact non-OneDrive selection plan required')
    plan = worker.exact_json(plan_path)
    if (type(plan) is not dict or set(plan) != {'schema', 'manifest_sha256', 'source_files',
                                                'trajectories', 'development'}
            or plan['schema'] != 'research4.all64-actualwidth-selection-plan.v1'
            or plan['manifest_sha256'] != MANIFEST):
        raise ValueError('Frozen CPU-only selection plan differs')
    validate_selection_sources(plan['source_files'])
    supervisor_path = PROJECT / V8_SUPERVISOR
    module_spec = importlib.util.spec_from_file_location('_actualwidth_v8_supervisor', supervisor_path)
    supervisor = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(supervisor)
    sys.path.insert(0, str(PROJECT / 'mac_studio'))
    import game_all_layers_layout as layout
    import qwen38_local as local
    validate_helper_origins(((layout, 'mac_studio/game_all_layers_layout.py'),
                             (local, 'mac_studio/qwen38_local.py')))
    local.prepare_local()
    if worker.sha(local.MANIFEST) != worker.BASE_MANIFEST_SHA256:
        raise ValueError('Pinned base manifest differs')
    identity = local.verify()
    if (identity['repo_id'], identity['revision']) != ('mlx-community/Qwen3.8-27B-4bit', '10c35caafbb80f7dc6a7a432cdd11af10a6d4818'):
        raise ValueError('Pinned base identity differs')
    selected, destination, ready = select_and_freeze(plan['trajectories'], plan['development'],
                                                      supervisor, layout, local.MODEL, args.output)
    print(json.dumps({'status': 'READY', 'rank': selected['winner']['rank'],
                      'epoch': selected['winner']['epoch'],
                      'primary_metric': selected['winner']['equal_family_mean_action_token_nll'],
                      'checkpoint': str(destination), 'adapter_sha256': ready['adapter_sha256'],
                      'winner_refit': False, 'free_running_used_for_selection': False}, allow_nan=False))


if __name__ == '__main__':
    main()
