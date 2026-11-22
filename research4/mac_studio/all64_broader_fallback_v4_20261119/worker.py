"""Fresh all-64 worker. Execution requires a separately owned supervisor.

This source is a draft.  A real qualification requires independent review,
authenticated Mini deployment and a live owner; importing it loads no model.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import time
import gc
import importlib.util

PROJECT = Path(__file__).resolve().parents[2]
MANIFEST_SHA256 = 'dc1d812ba3e2b2aa147b1fbe148440ff6a5327a5a67feb8d32f330883ac98f70'

LOCK = PROJECT / '.local/validation/qwen38_shared_v1/inference.lock/owner.json'
BASE_MANIFEST_SHA256 = 'e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
SOURCE_REQUIREMENTS = {
    'mac_studio/all64_broader_fallback_v4_20261119/worker.py',
    'mac_studio/all64_broader_v2_20261119/worker.py',
    'mac_studio/all64_broader_v2_20261119/loss.py',
    'mac_studio/all64_broader_v2_20261119/frozen_data.py',
    'mac_studio/all64_broader_v2_20261119/dev_eval.py',
    'mac_studio/all64_force_fused_widths_v4_20261119/scoped_sdpa_widths.py',
    'mac_studio/all64_sdpa_vjp_probe_v1_20261119/scoped_fallback.py',
    'mac_studio/all64_force_fused_widths_v4_20261119/WIDTH_CASES.json',
    'mac_studio/game_block3_sdpa_policy.py',
    'mac_studio/all64_broader_fallback_supervision_v4_20261119/supervisor.py',
    'mac_studio/owned_validation_process.py',
    'mac_studio/general_preservation_render_v1.py',
    'mac_studio/game_all_layers_layout.py',
    'mac_studio/game_compiled_direct_gradient_factory_v1.py',
    'mac_studio/game_selected_head_v3_compare.py',
    'mac_studio/qwen38_local.py',
    'mac_studio/qwen38_config.json',
    'mac_studio/mini_native_guard_20261119/mini_guard.py',
    'mac_studio/mini_native_guard_20261119/MINI_CROSSBUILT_BINDING_3778b7a0365143dc8f9d4e9922e5fd05.json',
    'scenarios/service_preservation/2026-11-19_17/broader_continuity_tools_v2/hpo_selection.py',
}
POLICY_PHASE = {'name': 'before_training'}


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular non-symlink source required')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_new_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def append_event(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def exact_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected regular non-symlink JSON')
    return json.loads(path.read_text(encoding='utf-8'))


def verify_source_files(spec):
    entries = spec.get('source_files')
    if not isinstance(entries, dict) or not SOURCE_REQUIREMENTS.issubset(entries):
        raise ValueError('Required worker and numerical sources are not all bound')
    for relative, digest in entries.items():
        path = Path(relative)
        if (not isinstance(relative, str) or path.is_absolute() or '..' in path.parts
                or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)
                or sha(PROJECT / path) != digest):
            raise ValueError('Supervisor-pinned source changed: ' + str(relative))


def verify_admission(args):
    # This check occurs before importing any MLX or project numerical helper.
    if os.environ.get('MLX_SDPA_VJP_FALLBACK') != '1' or 'mlx' in sys.modules:
        raise ValueError('Fallback must be set before MLX import in a fresh process')
    run = Path(args.run_dir).resolve()
    if ('onedrive' in str(run).lower() or not run.is_relative_to((PROJECT / '.local').resolve())
            or not run.is_dir() or run.is_symlink()):
        raise ValueError('Owned non-OneDrive run directory required')
    spec_path = Path(args.spec).resolve()
    if not spec_path.is_relative_to((PROJECT / '.local').resolve()) or sha(spec_path) != args.spec_sha256:
        raise ValueError('Exact reviewed local supervisor specification required')
    spec = exact_json(spec_path)
    if (spec.get('schema') != 'research4.all64-broader-fallback-supervision-spec.v1'
            or spec.get('owner') != args.owner or spec.get('phase') != args.phase
            or spec.get('rank') != args.rank or spec.get('deadline_epoch') != args.deadline_epoch
            or spec.get('manifest_sha256') != MANIFEST_SHA256
            or args.manifest_sha256 != MANIFEST_SHA256
            or args.rank not in (4, 8, 16) or args.phase not in ('qualification', 'trajectory')
            or not re.fullmatch('[0-9a-f]{32}', args.owner)
            or not math.isfinite(args.deadline_epoch) or time.time() >= args.deadline_epoch):
        raise ValueError('Frozen run identity, phase or deadline differs')
    authority = exact_json(run / 'worker_authority.json')
    expected = {'schema': 'research4.all64-broader-worker-authority.v1', 'phase': args.phase,
                'rank': args.rank, 'owner': args.owner, 'supervisor_pid': os.getppid(),
                'deadline_epoch': args.deadline_epoch, 'manifest_sha256': MANIFEST_SHA256,
                'worker_sha256': sha(__file__), 'spec_sha256': args.spec_sha256}
    if authority != expected:
        raise ValueError('Exact worker authority or supervising parent differs')
    owner = exact_json(LOCK)
    if (owner.get('owner'), owner.get('pid'), owner.get('kind')) != (args.owner, os.getppid(), 'all64_broader_v2'):
        raise ValueError('Canonical shared-lease owner differs')
    verify_source_files(spec)
    # Bind project sources before importing any project helper. The owner and
    # external supervisor still re-check them throughout the run.
    sys.path.insert(0, str(PROJECT / 'mac_studio/all64_broader_v2_20261119'))
    sys.path.insert(0, str(PROJECT / 'mac_studio'))
    supervisor_path = PROJECT / 'mac_studio/all64_broader_fallback_supervision_v4_20261119/supervisor.py'
    supervisor_spec = importlib.util.spec_from_file_location('_all64_broader_supervisor', supervisor_path)
    supervisor = importlib.util.module_from_spec(supervisor_spec)
    supervisor_spec.loader.exec_module(supervisor)
    supervisor.validate_spec(spec)
    global authenticate, render_inputs, geometry, make_batch, selected_suffix_loss
    from frozen_data import authenticate, render_inputs
    from loss import geometry, make_batch, selected_suffix_loss
    authenticate()
    return run, spec


def qualification_cases(prepared):
    """One real case per observed width plus a clearly marked capacity pad."""
    chosen = {}
    for row_id, ids, prefix, _ in prepared:
        info = geometry(ids, prefix)
        width = info['model_input_width']
        if (width not in chosen or
                (len(ids) - prefix, len(ids)) > (len(chosen[width][1]) - chosen[width][2], len(chosen[width][1]))):
            chosen[width] = (row_id, ids, prefix, False)
    if sorted(chosen) != [288, 544, 576, 608]:
        raise ValueError('Frozen training width inventory differs')
    longest = max(prepared, key=lambda item: len(item[1]))
    if len(longest[1]) != 590:
        raise ValueError('Frozen longest actual training row differs')
    if max(len(ids) - prefix for _, ids, prefix, _ in prepared) != 50:
        raise ValueError('Frozen maximum complete training suffix differs')
    chosen[640] = (longest[0], longest[1], longest[2], True)
    return [chosen[key] for key in sorted(chosen)]


def _load_runtime(args, run):
    """Bind the exact frozen base and fresh all-layer trainables on the Mini."""
    import importlib.metadata
    import platform
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Mini Apple-silicon runtime required')
    versions = {name: importlib.metadata.version(name) for name in ('mlx', 'mlx-lm', 'transformers', 'numpy')}
    if versions != {'mlx': '0.32.4', 'mlx-lm': '0.32.2', 'transformers': '5.22.2', 'numpy': '2.5.4'}:
        raise ValueError('Pinned numerical dependencies differ')
    if 'MLX_DISABLE_COMPILE' in os.environ:
        raise ValueError('MLX_DISABLE_COMPILE must be absent')
    import qwen38_local as local
    import game_all_layers_layout as layout
    local.prepare_local()
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'DO_NOT_TRACK': '1'})
    if sha(local.MANIFEST) != BASE_MANIFEST_SHA256:
        raise ValueError('Pinned base manifest differs')
    base_identity = local.verify()
    if (base_identity['repo_id'], base_identity['revision']) != ('mlx-community/Qwen3.8-27B-4bit', '10c35caafbb80f7dc6a7a432cdd11af10a6d4818'):
        raise ValueError('Pinned base repository or revision differs')
    layout.verify_installed_lora_sources()
    gated = Path(importlib.metadata.distribution('mlx-lm').locate_file('mlx_lm/models/gated_delta.py'))
    if sha(gated) != 'f69e837ea68641dbb06fdb5dcc024a9a7e1d9353f08ae58c05275253a0c39428':
        raise ValueError('Pinned native gated-delta implementation differs')
    manifest, train, dev, masks = authenticate()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(local.MODEL), local_files_only=True, trust_remote_code=False)
    prepared, prepared_dev = render_inputs(tokenizer, train, dev, masks)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx.utils import tree_flatten
    from game_compiled_direct_gradient_factory_v1 import CompiledDirectGradient
    info = mx.device_info()
    if (info.get('device_name'), info.get('memory_size'), info.get('max_recommended_working_set_size')) != ('Apple M5 Pro', 68719476736, 55662788608):
        raise ValueError('Expected 64 GiB Mini device identity differs')
    mx.set_memory_limit(50096509747)  # MLX guideline; native guard is independent.
    if mx.distributed.init().size() != 1:
        raise ValueError('Only one local model process is permitted')
    model, _ = load(str(local.MODEL), trust_remote_code=False)
    model.freeze()
    mx.random.seed(20261108)
    expected = layout.expected_adapter_layout(local.MODEL, args.rank, 64)
    if len(expected) != 992:
        raise ValueError('All 496 projections / 992 A/B tensors required')
    linear_to_lora_layers(model, 64, {'rank': args.rank, 'scale': 2.0, 'dropout': 0.0})
    params = dict(tree_flatten(model.trainable_parameters()))
    if set(params) != set(expected) or any(tuple(value.shape) != expected[name] or value.dtype != mx.float32 for name, value in params.items()):
        raise ValueError('Live all-layer trainable tensor inventory differs')
    mx.eval(model.parameters())
    b_zero = [mx.all(value == 0) for name, value in params.items() if name.endswith('.lora_b')]
    mx.eval(b_zero)
    if len(b_zero) != 496 or not all(bool(item.item()) for item in b_zero):
        raise ValueError('Fresh LoRA B initialization differs')
    original_call = type(model.layers[0]).__call__
    grad_checkpoint(model.layers[0])
    model.train()
    def objective(batch, lengths):
        return selected_suffix_loss(mx, nn, model, batch, lengths)
    handle = CompiledDirectGradient(mx, nn, model, objective)
    optimizer = optim.Adam(learning_rate=1e-5, betas=[.9, .999], eps=1e-8, bias_correction=False)
    mx.eval(optimizer.state)
    if int(optimizer.step.item()) != 0:
        raise ValueError('Fresh optimizer is required')
    return locals()


def _boundary(args, run, label, phase='run'):
    if time.time() >= args.deadline_epoch:
        raise TimeoutError('Original supervisor deadline reached')
    import psutil
    sys.path.insert(0, str(PROJECT / 'mac_studio/mini_native_guard_20261119'))
    import mini_guard
    binding = PROJECT / 'mac_studio/mini_native_guard_20261119/MINI_CROSSBUILT_BINDING_3778b7a0365143dc8f9d4e9922e5fd05.json'
    if not hasattr(args, '_native_guard'):
        args._native_guard = mini_guard.load_runtime(binding, 'acc7e820fa25f2f00b7394fb4ed562568f917d6fa43c540fe994679979fe3f52')
    guard = args._native_guard
    target = {'pid': os.getpid(), 'create_time_epoch': psutil.Process().create_time()}
    sample = guard.sample(target, phase)
    append_event(run / 'worker_native_guard.jsonl', {'label': label, 'sample': sample})
    if not sample['valid'] or sample['stop_required']:
        raise RuntimeError('Mini native memory/pressure guard stopped worker at ' + label)
    return sample


def _full_native_reference(mx, nn, model, batch, lengths):
    targets = batch[:, 1:]
    logits = model(batch[:, :-1])
    positions = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(positions >= lengths[:, 0:1], positions < lengths[:, 1:])
    total = (nn.losses.cross_entropy(logits, targets) * mask).astype(mx.float32).sum()
    return total / mask.sum(), mask.sum()


def _save_gradient(mx, flat, path):
    if path.exists():
        raise ValueError('Immutable gradient evidence already exists')
    mx.save_safetensors(str(path), flat)
    with path.open('rb') as stream:
        os.fsync(stream.fileno())


def _gradient_layers(mx, flat):
    names = sorted(flat)
    finite = [mx.all(mx.isfinite(flat[name])) for name in names]
    nonzero = [mx.any(flat[name] != 0) for name in names]
    mx.eval(finite, nonzero)
    if not all(bool(v.item()) for v in finite):
        raise ValueError('Nonfinite gradient')
    layers = set()
    counts = {'nonzero_a_tensors': 0, 'nonzero_b_tensors': 0}
    for name, value in zip(names, nonzero):
        match = re.fullmatch(r'language_model\.model\.layers\.(\d+)\..+\.lora_([ab])', name)
        if match is None or not 0 <= int(match.group(1)) < 64:
            raise ValueError('Unrecognized all-layer gradient name')
        if bool(value.item()):
            counts['nonzero_' + match.group(2) + '_tensors'] += 1
            if match.group(2) == 'b':
                layers.add(int(match.group(1)))
    return {'nonzero_b_layer_indices': sorted(layers), **counts}


def _validate_gradient_inventory(mx, flat, expected):
    if set(flat) != set(expected):
        raise ValueError('All 992 expected gradient tensors are required')
    if any(tuple(value.shape) != tuple(expected[name]) or value.dtype != mx.float32
           for name, value in flat.items()):
        raise ValueError('Gradient tensor shape or F32 dtype differs')


def parameter_digest(model):
    """Hash exact trainable names, shapes and contiguous native F32 bytes."""
    import numpy as np
    from mlx.utils import tree_flatten
    digest = hashlib.sha256()
    for name, value in sorted(tree_flatten(model.trainable_parameters())):
        array = np.array(value)
        if array.dtype != np.dtype('float32'):
            raise ValueError('Trainable digest requires F32 arrays')
        digest.update(name.encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def _b_parameter_layers(mx, model):
    from mlx.utils import tree_flatten
    params = [(name, value) for name, value in tree_flatten(model.trainable_parameters()) if name.endswith('.lora_b')]
    flags = [mx.any(value != 0) for _, value in params]
    mx.eval(flags)
    layers = set()
    for (name, _), flag in zip(params, flags):
        match = re.fullmatch(r'language_model\.model\.layers\.(\d+)\..+\.lora_b', name)
        if match is None:
            raise ValueError('Unexpected LoRA B name')
        if bool(flag.item()):
            layers.add(int(match.group(1)))
    return sorted(layers)


def verify_optimizer_file(path, expected, step, layout):
    """Stream native Adam moments and scalars without loading them to GPU."""
    import numpy as np
    header, offset, _ = layout.read_tensor_header(path)
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    moments = {name + '.' + kind: shape for name, shape in expected.items() for kind in ('m', 'v')}
    if set(tensors) != set(moments) | {'step', 'learning_rate'}:
        raise ValueError('Native Adam saved-state inventory differs')
    intervals = []
    for item in tensors.values():
        bounds = item.get('data_offsets')
        if (not isinstance(bounds, list) or len(bounds) != 2 or any(type(x) is not int for x in bounds)
                or not 0 <= bounds[0] <= bounds[1]):
            raise ValueError('Invalid Adam tensor offset')
        intervals.append(tuple(bounds))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError('Overlapping or gapped Adam state bytes')
        cursor = end
    if offset + cursor != path.stat().st_size:
        raise ValueError('Adam state size differs')
    with path.open('rb') as stream:
        for name, shape in moments.items():
            item = tensors[name]
            start, end = item['data_offsets']
            if item['dtype'] != 'F32' or tuple(item['shape']) != tuple(shape) or end - start != math.prod(shape) * 4:
                raise ValueError('Adam moment dtype or shape differs')
            stream.seek(offset + start)
            remaining = end - start
            while remaining:
                size = min(1024**2, remaining)
                values = np.frombuffer(stream.read(size), dtype='<f4')
                if values.nbytes != size or not np.isfinite(values).all() or (name.endswith('.v') and np.any(values < 0)):
                    raise ValueError('Nonfinite or negative native Adam moment')
                remaining -= size
        for name, dtype, size, value in [('step', 'U64', 8, step), ('learning_rate', 'F32', 4, 1e-5)]:
            item = tensors[name]
            start, end = item['data_offsets']
            if item['dtype'] != dtype or item['shape'] != [] or end - start != size:
                raise ValueError('Adam scalar layout differs')
            stream.seek(offset + start)
            actual = struct.unpack('<Q' if name == 'step' else '<f', stream.read(size))[0]
            if (name == 'step' and actual != value) or (name != 'step' and not math.isclose(actual, value, rel_tol=1e-7)):
                raise ValueError('Adam scalar value differs')
    return {'finite_f32_moments': len(moments), 'nonnegative_second_moments': True, 'optimizer_step': step}


def qualification(args, run, spec):
    from game_selected_head_v3_compare import compare_gradients, loss_equivalence, TOLERANCES
    from mlx.utils import tree_flatten
    state = _load_runtime(args, run)
    mx, nn, model, handle = (state[key] for key in ('mx', 'nn', 'model', 'handle'))
    optimizer, expected = state['optimizer'], state['expected']
    original_call = state['original_call']
    ordinary = nn.value_and_grad(model, state['objective'])
    def native_objective(batch, lengths):
        return _full_native_reference(mx, nn, model, batch, lengths)
    native_gradient_fn = nn.value_and_grad(model, native_objective)
    cases = qualification_cases(state['prepared'])
    seen_b = set()
    observations = []
    try:
        for step, (row_id, ids, prefix, ceiling) in enumerate(cases, 1):
            _boundary(args, run, 'before_case_' + str(step))
            batch, lengths, info = make_batch(mx, ids, prefix, ceiling)
            width = info['model_input_width']
            started = time.perf_counter()
            theta_before = parameter_digest(model)
            mx.eval(mx.random.state)
            rng_before = [item.tolist() for item in list(mx.random.state)]
            rng_before_sha = hashlib.sha256(json.dumps(rng_before, separators=(',', ':')).encode()).hexdigest()
            b_before = _b_parameter_layers(mx, model)
            if step == 1 and b_before:
                raise ValueError('Fresh-B numerical control requires zero B')
            if step == 2 and not b_before:
                raise ValueError('Changed-B numerical control requires prior nonzero B')
            POLICY_PHASE['name'] = 'width_%d_full_forward' % width
            ref, ref_count = _full_native_reference(mx, nn, model, batch, lengths)
            mx.eval(ref, ref_count)
            ref_value, n = float(ref.item()), int(ref_count.item())
            del ref, ref_count
            gc.collect(); mx.clear_cache()
            native_path = None
            if step <= 2:
                # Compare the actual full-native derivative to selected64 at
                # fresh B=0 and after one Adam update, at the same parameters.
                _boundary(args, run, 'before_native_gradient_' + str(step))
                POLICY_PHASE['name'] = 'width_%d_native_ad' % width
                (native_mean, native_count), native_gradient = native_gradient_fn(batch, lengths)
                mx.eval(native_mean, native_count, native_gradient)
                if not loss_equivalence(ref_value, float(native_mean.item()), n,
                                        int(native_count.item()), TOLERANCES)['passed']:
                    raise ValueError('Full-native derivative primal differs from full forward')
                native_flat = dict(tree_flatten(native_gradient))
                _validate_gradient_inventory(mx, native_flat, expected)
                native_path = run / ('case_%d_native.safetensors' % step)
                _save_gradient(mx, native_flat, native_path)
                del native_gradient, native_flat, native_mean, native_count
                gc.collect(); mx.clear_cache()
            _boundary(args, run, 'before_ordinary_' + str(step))
            POLICY_PHASE['name'] = 'width_%d_selected_ad' % width
            (raw_mean, raw_count), raw_gradient = ordinary(batch, lengths)
            mx.eval(raw_mean, raw_count, raw_gradient)
            raw_value, raw_n = float(raw_mean.item()), int(raw_count.item())
            raw_path = run / ('case_%d_ordinary.safetensors' % step)
            raw_flat = dict(tree_flatten(raw_gradient))
            _validate_gradient_inventory(mx, raw_flat, expected)
            _save_gradient(mx, raw_flat, raw_path)
            del raw_gradient, raw_flat, raw_mean, raw_count
            gc.collect(); mx.clear_cache()
            native_comparison = None
            if native_path is not None:
                native_comparison = compare_gradients(native_path, raw_path, expected, TOLERANCES,
                                                      require_zero_a=(step == 1))
                if not native_comparison['passed']:
                    raise ValueError('Full-native vs selected64 derivative check failed')
            _boundary(args, run, 'before_compiled_' + str(step))
            POLICY_PHASE['name'] = 'width_%d_compiled_ad' % width
            (compiled_mean, compiled_count), gradient = handle(batch, lengths)
            mx.eval(compiled_mean, compiled_count, gradient, handle.state)
            compiled_value, compiled_n = float(compiled_mean.item()), int(compiled_count.item())
            flat = dict(tree_flatten(gradient))
            _validate_gradient_inventory(mx, flat, expected)
            compiled_path = run / ('case_%d_compiled.safetensors' % step)
            _save_gradient(mx, flat, compiled_path)
            full_check = loss_equivalence(ref_value, raw_value, n, raw_n, TOLERANCES)
            compiled_check = loss_equivalence(raw_value, compiled_value, raw_n, compiled_n, TOLERANCES)
            gradients = compare_gradients(raw_path, compiled_path, expected, TOLERANCES, require_zero_a=(step == 1))
            if n != len(ids) - prefix or not (full_check['passed'] and compiled_check['passed'] and gradients['passed']):
                raise ValueError('Native full-head/selected/compiled numerical check failed')
            mx.eval(mx.random.state)
            theta_after_controls = parameter_digest(model)
            rng_after = [item.tolist() for item in list(mx.random.state)]
            rng_after_sha = hashlib.sha256(json.dumps(rng_after, separators=(',', ':')).encode()).hexdigest()
            if theta_after_controls != theta_before or rng_after_sha != rng_before_sha:
                raise ValueError('Numerical controls changed trainable parameters or RNG')
            coverage = _gradient_layers(mx, flat)
            seen_b.update(coverage['nonzero_b_layer_indices'])
            _boundary(args, run, 'before_adam_' + str(step))
            optimizer.update(model, gradient)
            mx.eval(model.trainable_parameters(), optimizer.state)
            if int(optimizer.step.item()) != step:
                raise ValueError('Host Adam update count differs')
            _boundary(args, run, 'after_adam_' + str(step))
            observation = {'step': step, 'row_id': row_id, **info,
                           'full_vs_selected': full_check, 'ordinary_vs_compiled': compiled_check,
                           'gradient_comparison_passed': gradients['passed'],
                           'gradient_global_relative_l2': gradients['global_relative_l2'],
                           'native_gradient_comparison_performed': native_comparison is not None,
                           'native_gradient_comparison_passed': native_comparison['passed'] if native_comparison else None,
                           'native_gradient_sha256': sha(native_path) if native_path else None,
                           'ordinary_gradient_sha256': sha(raw_path), 'compiled_gradient_sha256': sha(compiled_path),
                           'gradient_coverage': coverage, 'optimizer_step': step,
                           'control_theta_sha256': theta_before,
                           'control_theta_after_sha256': theta_after_controls,
                           'control_theta_unchanged': True,
                           'control_rng_before_sha256': rng_before_sha,
                           'control_rng_after_sha256': rng_after_sha,
                           'control_rng_unchanged': True,
                           'nonzero_b_layers_before_control': b_before,
                           'elapsed_seconds': time.perf_counter() - started}
            observations.append(observation)
            append_event(run / 'qualification_cases.jsonl', observation)
            del gradient, flat, compiled_mean, compiled_count, gradients
            gc.collect(); mx.clear_cache()
        if seen_b != set(range(64)):
            raise ValueError('B gradient coverage does not include all 64 layers')
        weights = run / 'qualification_adapter.safetensors'
        _save_gradient(mx, dict(tree_flatten(model.trainable_parameters())), weights)
        saved = state['layout'].verify_adapter_tensor_file(weights, expected)
        if saved['nonzero_lora_b_layer_indices'] != list(range(64)):
            raise ValueError('Saved LoRA B must be nonzero in every layer')
        _boundary(args, run, 'before_base_recheck')
        state['local'].verify()
        verify_source_files(spec)
        return {'schema': 'research4.all64-broader-qualification.v1', 'status': 'qualified',
                'phase': args.phase, 'rank': args.rank, 'owner': args.owner,
                'manifest_sha256': MANIFEST_SHA256, 'executed_updates': len(observations),
                'actual_widths': [288, 544, 576, 608], 'ceiling_test_model_input_width': 640,
                'maximum_observed_suffix_tokens': 50,
                'native_full_head_vs_selected_passed': True, 'ordinary_vs_compiled_gradients_passed': True,
                'native_vs_selected_gradients_passed': True, 'native_gradient_control_steps': [1, 2],
                'gradient_b_layers': sorted(seen_b), 'saved_adapter_all_layers_nonzero': True,
                'base_unchanged': True, 'source_files_unchanged': True,
                'base_unchanged_scope': 'Pinned base payload files were verified before and after; all trainables were adapter tensors. In-memory frozen tensors were not separately hashed.',
                'qualification_adapter_sha256': sha(weights), 'cases': observations}
    finally:
        handle.close()
        type(model.layers[0]).__call__ = original_call


def trajectory(args, run, spec):
    """One fresh optimizer and unchanged source-order trajectory, three epochs."""
    from mlx.utils import tree_flatten
    from game_selected_head_v3_compare import parameter_state_sha256
    state = _load_runtime(args, run)
    mx, model, handle = (state[key] for key in ('mx', 'model', 'handle'))
    optimizer, expected, prepared = state['optimizer'], state['expected'], state['prepared']
    original_call = state['original_call']
    if len(prepared) != 136:
        raise ValueError('Exact 136 training decisions required')
    checkpoints = []
    b_layers = set()
    try:
        for epoch in range(1, 4):
            for index, (row_id, ids, prefix, _meta) in enumerate(prepared, 1):
                step = (epoch - 1) * 136 + index
                _boundary(args, run, 'before_step_' + str(step))
                batch, lengths, info = make_batch(mx, ids, prefix)
                POLICY_PHASE['name'] = 'step_%d_width_%d' % (step, info['model_input_width'])
                started = time.perf_counter()
                (mean, count), gradient = handle(batch, lengths)
                mx.eval(mean, count, gradient, handle.state)
                flat = dict(tree_flatten(gradient))
                _validate_gradient_inventory(mx, flat, expected)
                if int(count.item()) != len(ids) - prefix or not math.isfinite(float(mean.item())):
                    raise ValueError('Complete supervised suffix or finite loss differs')
                coverage = _gradient_layers(mx, flat)
                b_layers.update(coverage['nonzero_b_layer_indices'])
                _boundary(args, run, 'before_adam_' + str(step))
                optimizer.update(model, gradient)
                mx.eval(model.trainable_parameters(), optimizer.state)
                if int(optimizer.step.item()) != step:
                    raise ValueError('Single continuous Adam optimizer step differs')
                _boundary(args, run, 'after_step_' + str(step))
                append_event(run / 'updates.jsonl', {'epoch': epoch, 'step': step, 'row_id': row_id,
                    'target_tokens': info['target_tokens'], 'sequence_tokens': info['semantic_tokens'],
                    'model_input_width': info['model_input_width'], 'loss': float(mean.item()),
                    'optimizer_step': step, 'step_seconds': time.perf_counter() - started,
                    'gradient_coverage': coverage, 'native_route': 'mlx_native_default'})
                del gradient, flat, mean, count
                gc.collect(); mx.clear_cache()
            if b_layers != set(range(64)):
                raise ValueError('B gradient coverage missing one or more layers')
            folder = run / ('epoch_' + str(epoch))
            folder.mkdir(mode=0o700, exist_ok=False)
            weights = folder / 'adapters.safetensors'
            moments = folder / 'optimizer.safetensors'
            config_path = folder / 'adapter_config.json'
            _save_gradient(mx, dict(tree_flatten(model.trainable_parameters())), weights)
            _save_gradient(mx, dict(tree_flatten(optimizer.state)), moments)
            expected_config = {'fine_tune_type': 'lora', 'num_layers': 64, 'batch_size': 1,
                'max_seq_length': 640, 'mask_prompt': True, 'resume_adapter_file': None,
                'optimizer': 'adam', 'grad_accumulation_steps': 1, 'grad_checkpoint': True,
                'lora_parameters': {'rank': args.rank, 'dropout': 0.0, 'scale': 2.0},
                'iters': epoch * 136, 'learning_rate': 1e-5, 'seed': 20261108,
                'optimizer_config': {'adam': {'betas': [.9, .999], 'eps': 1e-8, 'bias_correction': False}}}
            atomic_new_json(config_path, expected_config)
            if exact_json(config_path) != expected_config:
                raise ValueError('Saved epoch adapter config differs')
            saved = state['layout'].verify_adapter_tensor_file(weights, expected)
            if saved['nonzero_lora_b_layer_indices'] != list(range(64)):
                raise ValueError('Epoch checkpoint lacks nonzero B in every layer')
            optimizer_verification = verify_optimizer_file(moments, expected, epoch * 136, state['layout'])
            checkpoint = {'epoch': epoch, 'optimizer_step': epoch * 136,
                'trainable_state_sha256': parameter_state_sha256(weights, expected),
                'optimizer_verification': optimizer_verification,
                'adapter': {'path': weights.relative_to(PROJECT).as_posix(), 'sha256': sha(weights)},
                'config': {'path': config_path.relative_to(PROJECT).as_posix(), 'sha256': sha(config_path)},
                'optimizer_state': {'path': moments.relative_to(PROJECT).as_posix(), 'sha256': sha(moments)}}
            atomic_new_json(folder / 'CHECKPOINT.json', checkpoint)
            checkpoints.append(checkpoint)
        _boundary(args, run, 'before_base_recheck')
        state['local'].verify()
        verify_source_files(spec)
        return {'schema': 'research4.all64-broader-trajectory.v1', 'status': 'complete',
                'phase': args.phase, 'rank': args.rank, 'owner': args.owner,
                'manifest_sha256': MANIFEST_SHA256, 'executed_updates': 408,
                'epoch_steps': [136, 272, 408], 'gradient_b_layers': sorted(b_layers),
                'saved_adapter_all_layers_nonzero': True, 'base_unchanged': True,
                'base_unchanged_scope': 'Pinned base payload files were verified before and after; all trainables were adapter tensors. In-memory frozen tensors were not separately hashed.',
                'source_files_unchanged': True, 'checkpoints': checkpoints,
                'training_order': 'frozen_train_preview_jsonl_order_each_epoch',
                'fresh_base_and_optimizer': True, 'resumed': False}
    finally:
        handle.close()
        type(model.layers[0]).__call__ = original_call


def load_exact_force_scope():
    original = (PROJECT / 'mac_studio/game_block3_sdpa_policy.py').read_bytes()
    path = PROJECT / 'mac_studio/all64_force_fused_widths_v4_20261119/scoped_sdpa_widths.py'
    derived = path.read_bytes()
    template = original.replace(b'1023', b'_approved_width')
    needle = b'    def dispatch(queries, keys, values, cache, scale, mask, sinks=None):\n        valid = (\n            list(queries.shape)'
    replacement = (b'    def dispatch(queries, keys, values, cache, scale, mask, sinks=None):\n'
                   b'        _approved_width = queries.shape[2] if len(queries.shape) == 4 else None\n'
                   b'        valid = (\n            type(_approved_width) is int and _approved_width in (288, 544, 576, 608, 640)\n'
                   b'            and list(queries.shape)')
    if (hashlib.sha256(original).hexdigest() != '80817ebfab8733c1ab6261e78922137998ce29e48733d4bab95fa346d3e795a3'
            or original.count(b'1023') != 3 or template.count(needle) != 1
            or derived != template.replace(needle, replacement)
            or hashlib.sha256(derived).hexdigest() != 'b84a79c806980408e9cfdd30794093baf6b40e0cb5c1c2882bf08243b7baf3bf'):
        raise ValueError('Only the reviewed exact-width owned SDPA scope is allowed')
    fallback_path = PROJECT / 'mac_studio/all64_sdpa_vjp_probe_v1_20261119/scoped_fallback.py'
    fallback = fallback_path.read_bytes()
    old = b"expected_environment['MLX_SDPA_VJP_FALLBACK'] not in (None, '0')"
    new = b"expected_environment['MLX_SDPA_VJP_FALLBACK'] != '1'"
    if (derived.count(old) != 1 or fallback != derived.replace(old, new)
            or hashlib.sha256(fallback).hexdigest() != '78d74ab14d664dda2355fa25a97ead831053ed5ee4c4550e2a2a4ce398ea55b8'):
        raise ValueError('Exact reviewed preimport fallback scope required')
    module_spec = importlib.util.spec_from_file_location('_all64_exact_fallback_scope', fallback_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def main(args):
    run, spec = verify_admission(args)
    sdpa = load_exact_force_scope()
    environment = sdpa.environment_snapshot()
    scope = sdpa.scoped_sdpa_policy(owned_process_pid=os.getpid(), policy='force_fused',
                                    phase=POLICY_PHASE, expected_environment=environment)
    policy = None
    try:
        with scope as policy:
            if args.phase == 'qualification':
                result = qualification(args, run, spec)
            else:
                result = trajectory(args, run, spec)
    finally:
        # Preserve partial calls and restoration evidence on all failures.
        if policy is not None:
            atomic_new_json(run / 'FORCE_POLICY.json', policy)
    if (policy is None or policy.get('policy') != 'force_fused'
            or policy.get('off_scope_calls') != 0 or not policy.get('restoration_verified')
            or not policy.get('native_base_and_fast_untouched') or not policy.get('environment_unchanged')
            or policy.get('installed_files_modified') is not False
            or not policy.get('calls') or any(c.get('force_fused_requested') is not True for c in policy['calls'])):
        raise ValueError('Forced SDPA scope or alias restoration check failed')
    phase_counts = {}
    for call in policy['calls']:
        phase = call.get('phase')
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    widths = sorted({call['q_shape'][2] for call in policy['calls']})
    expected_widths = [288, 544, 576, 608, 640] if args.phase == 'qualification' else [288, 544, 576, 608]
    if widths != expected_widths or sum(phase_counts.values()) != policy['matched_calls']:
        raise ValueError('Observed forced-scope width inventory differs')
    if args.phase == 'qualification':
        for width in expected_widths:
            for kind in ('full_forward', 'selected_ad', 'compiled_ad'):
                if phase_counts.get('width_%d_%s' % (width, kind), 0) < 16:
                    raise ValueError('Forced scope did not cover all attention blocks in a qualification phase')
        for width in expected_widths[:2]:
            if phase_counts.get('width_%d_native_ad' % width, 0) < 16:
                raise ValueError('Forced scope did not cover a native AD control')
    result.update({'policy_receipt': {'path': (run / 'FORCE_POLICY.json').relative_to(PROJECT).as_posix(),
                                      'sha256': sha(run / 'FORCE_POLICY.json')},
                   'policy_restored': True, 'forced_width_phase_counts': phase_counts,
                   'observed_forced_widths': widths,
                   'fallback_requested_before_mlx_import': True,
                   'policy_scope': 'Forced-forward alias and fallback-before-import request, with Python trace evidence only; not kernel profiling.'})
    atomic_new_json(run / 'worker_receipt.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('qualification', 'trajectory'), required=True)
    parser.add_argument('--rank', type=int, choices=(4, 8, 16), required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--spec-sha256', required=True)
    options = parser.parse_args()
    try:
        main(options)
    except BaseException as exc:
        failure_dir = Path(options.run_dir).resolve()
        target = failure_dir / 'worker_failure.json'
        if (failure_dir.is_relative_to((PROJECT / '.local').resolve()) and failure_dir.is_dir()
                and not target.exists()):
            atomic_new_json(target, {'status': 'failed', 'error_type': type(exc).__name__,
                                     'message': str(exc), 'epoch': time.time()})
        raise
