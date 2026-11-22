import hashlib
import json
from datetime import timezone
from types import SimpleNamespace

import pytest

import selection
import dev_eval
import frozen_data
eval_worker = selection.eval_worker


def test_selection_rejects_shadowed_helper_path(tmp_path, monkeypatch):
    monkeypatch.setattr(selection, 'PROJECT', tmp_path)
    expected = tmp_path / 'mac_studio' / 'loss.py'
    selection.validate_helper_origins(((SimpleNamespace(__file__=str(expected)), 'mac_studio/loss.py'),))
    with pytest.raises(ValueError, match='pinned source'):
        selection.validate_helper_origins(((SimpleNamespace(__file__=str(tmp_path / 'shadow.py')), 'mac_studio/loss.py'),))


def test_selection_requires_both_historical_and_corrected_supervisor_pins(monkeypatch):
    pins = {name: selection.worker.sha(selection.PROJECT / name) for name in selection.REQUIRED_SOURCE_FILES}
    selection.validate_selection_sources(pins)
    for name in (selection.V7_SUPERVISOR, selection.V8_SUPERVISOR,
                 'mac_studio/all64_broader_v2_20261119/loss.py'):
        incomplete = dict(pins)
        incomplete.pop(name)
        with pytest.raises(ValueError, match='pins are incomplete'):
            selection.validate_selection_sources(incomplete)
    monkeypatch.setattr(selection.eval_worker, '__file__', str(selection.PROJECT / 'shadow.py'))
    with pytest.raises(ValueError, match='pinned source'):
        selection.validate_selection_sources(pins)


def test_selection_binds_nine_dev_receipts_to_original_trajectories(tmp_path, monkeypatch):
    monkeypatch.setattr(selection, 'PROJECT', tmp_path)
    monkeypatch.setattr(eval_worker, 'PROJECT', tmp_path)
    _, _, _, masks = frozen_data.authenticate()

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf8')
        return {'path': path.relative_to(tmp_path).as_posix(),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def closeout(owner, phase, rank, receipt_ref):
        return {'schema': 'research4.all64-broader-supervisor-closeout.v1', 'status': 'completed',
            'phase': phase, 'rank': rank, 'manifest_sha256': selection.MANIFEST,
            'worker_exit_code': 0, 'owned_cleanup_confirmed': True, 'automatic_retry': False,
            'owner': owner, 'worker_receipt': receipt_ref}

    trajectories, devs = [], []
    for rank_index, rank in enumerate((4, 8, 16)):
        owner = f'{rank_index + 1:032x}'
        entries = []
        for epoch in (1, 2, 3):
            folder = tmp_path / selection.RUNS / owner / f'epoch_{epoch}'
            folder.mkdir(parents=True)
            adapter = folder / 'adapters.safetensors'
            adapter.write_bytes(f'{rank}-{epoch}'.encode())
            adapter_ref = {'path': adapter.relative_to(tmp_path).as_posix(),
                           'sha256': hashlib.sha256(adapter.read_bytes()).hexdigest()}
            entries.append({'epoch': epoch, 'optimizer_step': 136 * epoch, 'adapter': adapter_ref,
                            'config': {'path': 'config', 'sha256': 'a' * 64},
                            'optimizer_state': {'path': 'optimizer', 'sha256': 'b' * 64}})
            write(folder / 'CHECKPOINT.json', entries[-1])
        trajectory_receipt = {'owner': owner, 'phase': 'trajectory', 'rank': rank,
                              'manifest_sha256': selection.MANIFEST, 'checkpoints': entries}
        tr = write(tmp_path / 'receipts' / f'trajectory-{rank}.json', trajectory_receipt)
        trajectories.append(write(tmp_path / 'closeouts' / f'trajectory-{rank}.json',
                                  closeout(owner, 'trajectory', rank, tr)))
        for epoch in (1, 2, 3):
            dev_owner = f'{rank_index * 3 + epoch + 10:032x}'
            checkpoint = tmp_path / selection.RUNS / owner / f'epoch_{epoch}' / 'CHECKPOINT.json'
            checkpoint_ref = {'path': checkpoint.relative_to(tmp_path).as_posix(),
                              'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            rows = [{'id': mask['id'], 'family': mask['family'],
                     'action_nll_sum': float(mask['action_tokens_including_eos'] * rank),
                     'action_tokens': mask['action_tokens_including_eos'],
                     'full_suffix_nll_sum': float((mask['full_suffix_end_token_exclusive']
                         - mask['full_suffix_start_token']) * rank),
                     'full_suffix_tokens': mask['full_suffix_end_token_exclusive']
                         - mask['full_suffix_start_token']} for mask in masks]
            receipt = {'owner': dev_owner, 'phase': 'dev', 'rank': rank, 'epoch': epoch,
                'manifest_sha256': selection.MANIFEST, 'checkpoint': checkpoint_ref,
                'development_no_optimizer_updates': True, 'development_rows': rows,
                'adapter_sha256': entries[epoch - 1]['adapter']['sha256'],
                'trainable_digest_before': 'c' * 64, 'trainable_digest_after': 'c' * 64,
                'optimizer_absent': True, 'checkpoint_step': 136 * epoch,
                'optimizer_step_before': None, 'optimizer_step_after': None}
            dr = write(tmp_path / 'receipts' / f'dev-{rank}-{epoch}.json', receipt)
            devs.append(write(tmp_path / 'closeouts' / f'dev-{rank}-{epoch}.json',
                              closeout(dev_owner, 'dev', rank, dr)))

    class Supervisor:
        @staticmethod
        def validate_completed_closeout(ref, phase, rank, epoch=None):
            value = json.loads((tmp_path / ref['path']).read_text())
            receipt = json.loads((tmp_path / value['worker_receipt']['path']).read_text())
            if value.get('owned_cleanup_confirmed') is not True:
                raise ValueError('Nested release evidence invalid')
            return value, receipt

        @staticmethod
        def validate_worker_receipt(receipt, spec):
            assert receipt['owner'] == spec['owner']

    class Layout:
        @staticmethod
        def expected_adapter_layout(*_):
            return {f'tensor-{i}': (1,) for i in range(992)}

        @staticmethod
        def verify_adapter_tensor_file(*_):
            return {'nonzero_lora_b_layer_indices': list(range(64))}

    def checkpoint_files(ref, rank, epoch):
        entry = json.loads((tmp_path / ref['path']).read_text())
        return entry, {'adapter': tmp_path / entry['adapter']['path']}

    monkeypatch.setattr(eval_worker, 'checkpoint_files', checkpoint_files)
    monkeypatch.setattr(eval_worker, 'saved_state_digest', lambda *_: 'c' * 64)
    result = selection.select_from_closeouts(trajectories, devs, Supervisor, Layout, tmp_path)
    assert (result['winner']['rank'], result['winner']['epoch']) == (4, 1)
    assert result['winner_refit'] is False and result['free_running_used_for_selection'] is False
    assert len(result['candidate_provenance']) == 9
    broken = tmp_path / devs[0]['path']
    value = json.loads(broken.read_text())
    value['owned_cleanup_confirmed'] = False
    bad_ref = write(broken, value)
    bad_devs = [bad_ref, *devs[1:]]
    with pytest.raises(ValueError, match='release'):
        selection.select_from_closeouts(trajectories, bad_devs, Supervisor, Layout, tmp_path)
    value['owned_cleanup_confirmed'] = True
    devs[0] = write(broken, value)
    with pytest.raises(ValueError, match='Nine completed'):
        selection.select_from_closeouts(trajectories, devs[:8], Supervisor, Layout, tmp_path)
    devs[0], devs[1] = devs[1], devs[0]
    with pytest.raises(ValueError, match='rank/epoch order'):
        selection.select_from_closeouts(trajectories, devs, Supervisor, Layout, tmp_path)


def test_freeze_copies_selected_bytes_without_refit_or_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(selection, 'PROJECT', tmp_path)
    monkeypatch.setattr(selection, 'ZoneInfo', lambda _: timezone.utc)
    source = tmp_path / '.local' / 'source'
    source.mkdir(parents=True)
    adapter, config = source / 'adapters.safetensors', source / 'adapter_config.json'
    adapter.write_bytes(b'original-adapter'); config.write_bytes(b'original-config')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    entry = {'adapter': {'path': '.local/source/adapters.safetensors', 'sha256': digest(adapter)},
             'config': {'path': '.local/source/adapter_config.json', 'sha256': digest(config)},
             'trainable_state_sha256': 'c' * 64}
    selected = {'winner': {'rank': 4, 'epoch': 1}, 'winner_refit': False,
        'winner_checkpoint': {'checkpoint': {'path': '.local/source/CHECKPOINT.json', 'sha256': 'd' * 64},
                              'adapter': entry['adapter'], 'config': entry['config']}}
    monkeypatch.setattr(selection, 'select_from_closeouts', lambda *a: selected)
    monkeypatch.setattr(eval_worker, 'checkpoint_files', lambda *a: (entry, {'adapter': adapter, 'config': config}))
    monkeypatch.setattr(eval_worker, 'saved_state_digest', lambda *a: 'c' * 64)

    class Layout:
        @staticmethod
        def expected_adapter_layout(*_):
            return {f'tensor-{i}': (1,) for i in range(992)}

        @staticmethod
        def verify_adapter_tensor_file(*_):
            return {'nonzero_lora_b_layer_indices': list(range(64))}

    output = tmp_path / '.local' / 'selection' / 'LIKELIHOOD_SELECTION.json'
    output.parent.mkdir(parents=True)
    _, destination, ready = selection.select_and_freeze([], [], None, Layout, tmp_path, output)
    assert (destination / 'adapters.safetensors').read_bytes() == b'original-adapter'
    assert json.loads((destination / 'FREEZE_READY_v1.json').read_text()) == ready
    assert ready['winner_refit'] is False and ready['layers'] == list(range(64))
    adapter.write_bytes(b'changed-source')
    assert (destination / 'adapters.safetensors').read_bytes() == b'original-adapter'
    with pytest.raises(FileExistsError):
        selection.select_and_freeze([], [], None, Layout, tmp_path, output)
