"""Score one frozen epoch checkpoint under an independently owned supervisor.

The process loads one pinned base and one adapter, never instantiates an
optimizer and writes no result until the full fixed DEV likelihood set passes.
Importing this module does not load or execute a model.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

import worker

PROJECT = worker.PROJECT
MANIFEST = worker.MANIFEST_SHA256
if Path(worker.__file__).resolve() != Path(__file__).with_name('worker.py').resolve():
    raise ValueError('DEV must import its co-versioned v7 worker')
sys.path.insert(1, str(PROJECT / 'mac_studio/all64_broader_v2_20261119'))


def project_file(relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('A repository-relative POSIX file is required')
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe repository path')
    path = PROJECT / rel
    if not path.resolve().is_relative_to(PROJECT.resolve()) or 'onedrive' in str(path.resolve()).lower():
        raise ValueError('Checkpoint must be in the canonical non-OneDrive repository')
    for current in (path, *path.parents):
        if current == PROJECT.parent:
            break
        if current.is_symlink():
            raise ValueError('Symlinked checkpoint path is forbidden')
    if not path.is_file():
        raise ValueError('Checkpoint file is unavailable')
    return path


def checkpoint_files(ref, rank, epoch):
    """Bind immutable checkpoint bytes, config, and saved optimizer step."""
    if type(ref) is not dict or set(ref) != {'path', 'sha256'}:
        raise ValueError('Exact checkpoint reference is required')
    entry_path = project_file(ref['path'])
    if worker.sha(entry_path) != ref['sha256']:
        raise ValueError('Checkpoint receipt hash differs')
    entry = worker.exact_json(entry_path)
    if (type(entry) is not dict or set(entry) != {'epoch', 'optimizer_step', 'adapter', 'config',
                                                  'optimizer_state', 'optimizer_verification', 'trainable_state_sha256'}
            or entry['epoch'] != epoch or entry['optimizer_step'] != 136 * epoch):
        raise ValueError('Checkpoint epoch or saved optimizer step differs')
    if entry['optimizer_verification'] != {'finite_f32_moments': 1984,
            'nonnegative_second_moments': True, 'optimizer_step': 136 * epoch}:
        raise ValueError('Saved optimizer verification differs')
    files = {}
    for key in ('adapter', 'config', 'optimizer_state'):
        item = entry[key]
        if type(item) is not dict or set(item) != {'path', 'sha256'}:
            raise ValueError('Malformed checkpoint artifact')
        path = project_file(item['path'])
        if worker.sha(path) != item['sha256'] or path.parent != entry_path.parent:
            raise ValueError('Checkpoint artifact bytes or directory differ')
        files[key] = path
    if (files['adapter'].name != 'adapters.safetensors' or
            files['config'].name != 'adapter_config.json' or
            files['optimizer_state'].name != 'optimizer.safetensors'):
        raise ValueError('Unexpected checkpoint artifact names')
    config = worker.exact_json(files['config'])
    expected = {'fine_tune_type': 'lora', 'num_layers': 64, 'batch_size': 1,
                'max_seq_length': 608, 'mask_prompt': True, 'resume_adapter_file': None,
                'frozen_prompt_context_tokens': 640, 'actual_model_input_width_limit': 608,
                'no_truncation': True,
                'optimizer': 'adam', 'grad_accumulation_steps': 1, 'grad_checkpoint': True,
                'lora_parameters': {'rank': rank, 'dropout': 0.0, 'scale': 2.0},
                'iters': 136 * epoch, 'learning_rate': 1e-5, 'seed': 20261108,
                'optimizer_config': {'adam': {'betas': [.9, .999], 'eps': 1e-8, 'bias_correction': False}}}
    if config != expected:
        raise ValueError('Checkpoint configuration differs from frozen recipe')
    if not isinstance(entry['trainable_state_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', entry['trainable_state_sha256']):
        raise ValueError('Checkpoint trainable-state digest is missing')
    return entry, files


def saved_state_digest(path, expected_shapes, layout):
    """Reconstruct the worker's sorted name/shape/F32-byte digest on CPU."""
    header, data_start, _ = layout.read_tensor_header(path)
    if {k for k in header if k != '__metadata__'} != set(expected_shapes):
        raise ValueError('Saved tensor names differ')
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for name in sorted(expected_shapes):
            info = header[name]
            shape = expected_shapes[name]
            if info['dtype'] != 'F32' or tuple(info['shape']) != shape:
                raise ValueError('Saved tensor shape or dtype differs')
            start, end = info['data_offsets']
            if end - start != 4 * math.prod(shape):
                raise ValueError('Saved tensor byte range differs')
            result.update(name.encode()); result.update(str(shape).encode())
            stream.seek(data_start + start)
            remaining = end - start
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError('Truncated checkpoint tensor')
                result.update(block); remaining -= len(block)
    return result.hexdigest()


def verify_admission(args):
    if os.environ.get('MLX_SDPA_VJP_FALLBACK') != '1' or 'mlx' in sys.modules:
        raise ValueError('Fallback must be set before MLX import in a fresh DEV process')
    run = Path(args.run_dir).resolve()
    if not run.is_dir() or not run.is_relative_to((PROJECT / '.local').resolve()) or 'onedrive' in str(run).lower():
        raise ValueError('Owned non-OneDrive run directory required')
    spec_path = Path(args.spec).resolve()
    if not spec_path.is_relative_to((PROJECT / '.local').resolve()) or worker.sha(spec_path) != args.spec_sha256:
        raise ValueError('Exact supervisor specification is required')
    spec = worker.exact_json(spec_path)
    if (spec.get('schema') != 'research4.all64-broader-fallback-actualwidth-hpo-supervision-spec.v1'
            or spec.get('phase') != 'dev' or args.phase != 'dev'
            or spec.get('owner') != args.owner or spec.get('rank') != args.rank
            or spec.get('epoch') != args.epoch or spec.get('deadline_epoch') != args.deadline_epoch
            or spec.get('manifest_sha256') != MANIFEST or args.manifest_sha256 != MANIFEST
            or args.rank not in (4, 8, 16) or args.epoch not in (1, 2, 3)
            or not re.fullmatch('[0-9a-f]{32}', args.owner)
            or not math.isfinite(args.deadline_epoch) or time.time() >= args.deadline_epoch):
        raise ValueError('DEV authority or deadline differs')
    expected_authority = {'schema': 'research4.all64-broader-worker-authority.v1', 'phase': 'dev',
                          'rank': args.rank, 'epoch': args.epoch, 'owner': args.owner,
                          'supervisor_pid': os.getppid(), 'deadline_epoch': args.deadline_epoch,
                          'manifest_sha256': MANIFEST, 'worker_sha256': worker.sha(__file__),
                          'spec_sha256': args.spec_sha256}
    if worker.exact_json(run / 'worker_authority.json') != expected_authority:
        raise ValueError('Exact supervisor/worker authority differs')
    lease = worker.exact_json(worker.LOCK)
    if (lease.get('owner'), lease.get('pid'), lease.get('kind')) != (args.owner, os.getppid(), 'all64_broader_v2'):
        raise ValueError('Canonical lease does not belong to supervisor')
    worker.verify_source_files(spec)
    # Revalidate completed qualifications, trajectory and exact epoch
    # checkpoint inside the DEV child, not only in its supervisor.
    supervisor_path = PROJECT / 'mac_studio/all64_broader_fallback_actualwidth_supervision_v7_20261119/supervisor.py'
    supervisor_spec = importlib.util.spec_from_file_location('_actualwidth_hpo_dev_supervisor', supervisor_path)
    supervisor = importlib.util.module_from_spec(supervisor_spec)
    supervisor_spec.loader.exec_module(supervisor)
    supervisor.validate_spec(spec)
    for relative in ('mac_studio/all64_broader_fallback_actualwidth_v7_20261119/eval_worker.py',
                     'mac_studio/all64_broader_v2_20261119/dev_eval.py'):
        if worker.sha(PROJECT / relative) != spec['source_files'].get(relative):
            raise ValueError('DEV source is not bound')
    return run, spec


def evaluate(args, run, spec):
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Mini Apple-silicon runtime is required')
    versions = {name: importlib.metadata.version(name) for name in ('mlx', 'mlx-lm', 'transformers', 'numpy')}
    if versions != {'mlx': '0.32.4', 'mlx-lm': '0.32.2', 'transformers': '5.22.2', 'numpy': '2.5.4'}:
        raise ValueError('Pinned numerical dependencies differ')
    sys.path.insert(0, str(PROJECT / 'mac_studio'))
    import qwen38_local as local
    import game_all_layers_layout as layout
    import frozen_data, dev_eval, loss
    for module, relative in ((frozen_data, 'mac_studio/all64_broader_v2_20261119/frozen_data.py'),
                             (dev_eval, 'mac_studio/all64_broader_v2_20261119/dev_eval.py'),
                             (loss, 'mac_studio/all64_broader_v2_20261119/loss.py')):
        expected_source = PROJECT / relative
        if (Path(module.__file__).resolve() != expected_source.resolve()
                or worker.sha(expected_source) != spec['source_files'].get(relative)):
            raise ValueError('DEV imported an unexpected helper source: ' + relative)
    local.prepare_local()
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'DO_NOT_TRACK': '1'})
    if worker.sha(local.MANIFEST) != worker.BASE_MANIFEST_SHA256:
        raise ValueError('Pinned base manifest differs')
    identity = local.verify()
    if (identity['repo_id'], identity['revision']) != ('mlx-community/Qwen3.8-27B-4bit', '10c35caafbb80f7dc6a7a432cdd11af10a6d4818'):
        raise ValueError('Pinned base identity differs')
    layout.verify_installed_lora_sources()
    entry, files = checkpoint_files(spec['checkpoint'], args.rank, args.epoch)
    expected = layout.expected_adapter_layout(local.MODEL, args.rank, 64)
    if len(expected) != 992:
        raise ValueError('Expected 496 projections and 992 adapter tensors')
    if worker.verify_optimizer_file(files['optimizer_state'], expected, 136 * args.epoch, layout) != entry['optimizer_verification']:
        raise ValueError('Saved optimizer state does not match its checkpoint verification')
    structure = layout.verify_adapter_tensor_file(files['adapter'], expected)
    if structure['nonzero_lora_b_layer_indices'] != list(range(64)):
        raise ValueError('Checkpoint has a layer without a nonzero LoRA B tensor')
    saved_digest = saved_state_digest(files['adapter'], expected, layout)
    if saved_digest != entry['trainable_state_sha256']:
        raise ValueError('Saved checkpoint state digest differs')
    _, train, dev, masks = frozen_data.authenticate()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(local.MODEL), local_files_only=True, trust_remote_code=False)
    _, prepared = frozen_data.render_inputs(tokenizer, train, dev, masks)
    widths = sorted({loss.geometry(ids, prefix)['model_input_width']
                     for _row_id, _family, ids, prefix, _start, _end, _meta in prepared})
    if not widths or any(width not in (288, 544, 576, 608) or width > 608 for width in widths):
        raise ValueError('DEV inputs must use only actual frozen widths at most 608')
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm import load
    import numpy as np
    info = mx.device_info()
    if (info.get('device_name'), info.get('memory_size'), info.get('max_recommended_working_set_size')) != ('Apple M5 Pro', 68719476736, 55662788608):
        raise ValueError('Expected Mini device identity differs')
    mx.set_memory_limit(50096509747)  # Guideline only; native guard is independent.
    if mx.distributed.init().size() != 1:
        raise ValueError('Only one local model process is permitted')
    worker._boundary(args, run, 'before_dev_model_load', 'launch')
    model, _ = load(str(local.MODEL), adapter_path=str(files['adapter'].parent), trust_remote_code=False)
    mx.eval(model.parameters())

    def live_digest(instance):
        params = {name: value for name, value in tree_flatten(instance.parameters())
                  if name.endswith('.lora_a') or name.endswith('.lora_b')}
        if set(params) != set(expected):
            raise ValueError('Loaded model adapter inventory differs')
        result = hashlib.sha256()
        for name in sorted(params):
            value = params[name]
            if tuple(value.shape) != expected[name] or value.dtype != mx.float32:
                raise ValueError('Loaded adapter tensor shape/dtype differs')
            result.update(name.encode()); result.update(str(tuple(value.shape)).encode())
            result.update(np.array(value).tobytes(order='C'))
        return result.hexdigest()

    if live_digest(model) != saved_digest:
        raise ValueError('Loaded adapter does not match saved checkpoint')
    score = dev_eval.score_rows(mx, nn, model, prepared, masks, live_digest, checkpoint_step=136 * args.epoch,
                       restore_training=False, row_boundary=lambda row: worker._boundary(args, run, 'dev:' + row))
    worker._boundary(args, run, 'after_dev_scoring')
    local.verify(); worker.verify_source_files(spec)
    return {'schema': 'research4.all64-broader-actualwidth-dev.v1', 'status': 'complete', 'phase': 'dev',
            'rank': args.rank, 'epoch': args.epoch, 'owner': args.owner, 'manifest_sha256': MANIFEST,
            'checkpoint': spec['checkpoint'], 'adapter_sha256': entry['adapter']['sha256'],
            'trainable_state_sha256': saved_digest, **score,
            'dev_model_input_widths': widths, 'max_dev_model_input_width': max(widths),
            'frozen_prompt_context_tokens': 640, 'no_truncation': True,
            'evaluation_policy': 'eval_mode_default_sdpa_no_grad',
            'fallback_requested_before_mlx_import': True,
            'base_unchanged': True,
            'base_unchanged_scope': 'Pinned base files verified before and after; live adapter state separately checked.',
            'source_files_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('dev',), required=True)
    parser.add_argument('--rank', type=int, choices=(4, 8, 16), required=True)
    parser.add_argument('--epoch', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--run-dir', required=True); parser.add_argument('--owner', required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--manifest-sha256', required=True); parser.add_argument('--spec', required=True)
    parser.add_argument('--spec-sha256', required=True)
    args = parser.parse_args()
    run, spec = verify_admission(args)
    receipt = evaluate(args, run, spec)
    worker.atomic_new_json(run / 'worker_receipt.json', receipt)
    print(json.dumps({'status': 'complete', 'rank': args.rank, 'epoch': args.epoch, 'rows': len(receipt['development_rows'])}))


if __name__ == '__main__':
    main()
