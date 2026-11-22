"""One-use, externally guarded supervisor for the frozen all-64 study.

This is a candidate, not run authority. A separately reviewed exact spec and
review receipt are required. The native guard is sampled in this process while
the model worker is executing, including during a blocked worker call. Process
ownership is the shared Research4 lease plus an owned process group; it does
not establish visibility into other users' GPU workloads.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]

MANIFEST = 'dc1d812ba3e2b2aa147b1fbe148440ff6a5327a5a67feb8d32f330883ac98f70'
FROZEN = 'scenarios/service_preservation/2026-11-19_17/broader_continuity_tools_v2/FROZEN_MANIFEST.json'
WORKER = 'mac_studio/all64_broader_v2_20261119/worker.py'
DEV_WORKER = 'mac_studio/all64_broader_v2_20261119/eval_worker.py'
SUPERVISOR = 'mac_studio/all64_broader_tools_supervision_v1_20261119/supervisor.py'
GUARD = 'mac_studio/mini_native_guard_20261119/mini_guard.py'
GUARD_BINDING = 'mac_studio/mini_native_guard_20261119/MINI_CROSSBUILT_BINDING_3778b7a0365143dc8f9d4e9922e5fd05.json'
GUARD_BINDING_SHA = 'acc7e820fa25f2f00b7394fb4ed562568f917d6fa43c540fe994679979fe3f52'
SELECTOR = 'scenarios/service_preservation/2026-11-19_17/broader_continuity_tools_v2/hpo_selection.py'
SELECTOR_SHA = '333185b9fd63de2c835c6cf1b331cdf9445c3e80edfb9af9d5f428cc24df5a3d'
DEV_MASKS = '.local/broader_continuity_tools_v2_cpu_frozen_01bed751/DEV_ACTION_MASKS.jsonl'
DEV_MASKS_SHA = 'b2ed41957a4e56e8e09236d7cbbce1807f9d7fca2e66529733cc06be767752ba'
TOKEN_LENGTHS = '.local/broader_continuity_tools_v2_cpu_frozen_01bed751/TOKEN_LENGTHS.json'
TOKEN_LENGTHS_SHA = '6d071d6f11e4b442ccc9d6d929669891066f54519aff449ebba1ebb3cc28cb73'
PYTHON = '.local/envs/mini-inference-20261119/bin/python'
LOCK = '.local/validation/qwen38_shared_v1/inference.lock'
RUNS = '.local/experiments/all64_broader_v2_20261119/runs'
MONITOR = '.local/monitor/all64_broader_supervision_v1_20261119'
KIND = 'all64_broader_v2'
HEX64 = re.compile(r'^[0-9a-f]{64}$')
HEX32 = re.compile(r'^[0-9a-f]{32}$')
KNOWN = ('all64_broader_v2_20261119/worker.py', 'game_all_layers_worker.py', 'game_training_worker.py',
         'exact_prefix_training.py', 'validation_mlx_worker.py', 'heldout_mlx_worker.py',
         'local_game_service.py', 'local_last8_endpoint_service.py', 'mlx_lm.server',
         'game_compiled_direct_pilot_v1.py', 'game_body_vjp_v5_worker.py')


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular non-symlink file required')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('Duplicate JSON key')
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('Non-finite JSON value')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def project_path(relative):
    if type(relative) is not str or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Repository-relative POSIX path required')
    part = Path(relative)
    if part.is_absolute() or '..' in part.parts:
        raise ValueError('Unsafe relative path')
    path = PROJECT / part
    if not path.resolve().is_relative_to(PROJECT.resolve()) or 'onedrive' in str(path.resolve()).lower():
        raise ValueError('Outside canonical non-OneDrive repository')
    for current in (path, *path.parents):
        if current == PROJECT.parent:
            break
        if current.is_symlink():
            raise ValueError('Symlinked path forbidden')
    return path


def exclusive_json(path, data):
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def append_json(path, data):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(data, separators=(',', ':'), allow_nan=False) + '\n')
        f.flush()


def required_prerequisites(phase, rank):
    if phase == 'dev': return {'4', '8', '16', 'trajectory'}
    if phase == 'trajectory': return {'4', '8', '16'}
    return {str(r) for r in (4, 8) if r < rank}


def validate_spec(spec, now=None):
    keys = {'schema', 'owner', 'phase', 'rank', 'epoch', 'checkpoint', 'manifest_sha256', 'deadline_epoch',
            'max_wall_seconds', 'python_binary_sha256', 'source_files', 'prerequisites'}
    if type(spec) is not dict or set(spec) != keys or spec['schema'] != 'research4.all64-broader-supervision-spec.v1':
        raise ValueError('Exact supervision spec required')
    if type(spec['owner']) is not str or not HEX32.fullmatch(spec['owner']):
        raise ValueError('Unique 32-hex owner required')
    phase, rank = spec['phase'], spec['rank']
    if phase not in ('qualification', 'trajectory', 'dev') or type(rank) is not int or rank not in (4, 8, 16):
        raise ValueError('Unknown phase or rank')
    epoch = spec['epoch']
    if type(epoch) is not int or (phase == 'dev' and epoch not in (1, 2, 3)) or (phase != 'dev' and epoch != 0):
        raise ValueError('Exact stage epoch required')
    if (phase != 'dev' and spec['checkpoint'] is not None) or (phase == 'dev' and type(spec['checkpoint']) is not dict):
        raise ValueError('Only DEV is bound to an epoch checkpoint')
    if spec['manifest_sha256'] != MANIFEST:
        raise ValueError('Frozen curriculum identity differs')
    deadline = spec['deadline_epoch']; maximum = spec['max_wall_seconds']
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= (time.time() if now is None else now):
        raise ValueError('Prospective finite deadline required')
    maximum_by_phase = {'qualification': 14400, 'trajectory': 43200, 'dev': 3600}
    if type(maximum) is not int or not 0 < maximum <= maximum_by_phase[phase]:
        raise ValueError('Invalid finite stage bound')
    python = (PROJECT / PYTHON).resolve(strict=True)
    if ('onedrive' in str(python).lower() or not HEX64.fullmatch(spec['python_binary_sha256'])
            or digest(python) != spec['python_binary_sha256']):
        raise ValueError('Mini virtual-environment Python identity differs')
    sources = spec['source_files']
    program = DEV_WORKER if phase == 'dev' else WORKER
    if type(sources) is not dict or not {program, SUPERVISOR, FROZEN, GUARD, GUARD_BINDING,
                                         'mac_studio/game_all_layers_layout.py',
                                         'mac_studio/game_selected_head_v3_compare.py',
                                         'mac_studio/owned_validation_process.py'}.issubset(sources):
        raise ValueError('Source pins incomplete')
    for name, expected in sources.items():
        if type(expected) is not str or not HEX64.fullmatch(expected) or digest(project_path(name)) != expected:
            raise ValueError('Pinned source changed: ' + str(name))
    if sources[FROZEN] != MANIFEST or sources[GUARD_BINDING] != GUARD_BINDING_SHA:
        raise ValueError('Frozen dataset or guard binding differs')
    if digest(project_path(TOKEN_LENGTHS)) != TOKEN_LENGTHS_SHA:
        raise ValueError('Frozen sequence length inventory differs')
    if phase == 'dev' and ('mac_studio/all64_broader_v2_20261119/dev_eval.py' not in sources or
                           sources.get(SELECTOR) != SELECTOR_SHA or digest(project_path(DEV_MASKS)) != DEV_MASKS_SHA):
        raise ValueError('Frozen DEV selector or masks differ')
    prereqs = spec['prerequisites']
    if type(prereqs) is not dict or set(prereqs) != required_prerequisites(phase, rank):
        raise ValueError('Exact successful lower-rank qualification prerequisites required')
    cache = {}
    for key, ref in prereqs.items():
        prereq_phase = 'trajectory' if key == 'trajectory' else 'qualification'
        prereq_rank = rank if key == 'trajectory' else int(key)
        value, worker_value = _validate_completed_closeout(ref, prereq_phase, prereq_rank, None, cache, set())
        if key == 'trajectory':
            match = worker_value['checkpoints'][epoch - 1]
            ref_checkpoint = spec['checkpoint']
            if (type(ref_checkpoint) is not dict or set(ref_checkpoint) != {'path', 'sha256'} or
                    type(ref_checkpoint['sha256']) is not str or not HEX64.fullmatch(ref_checkpoint['sha256'])):
                raise ValueError('Exact DEV checkpoint descriptor reference required')
            checkpoint_path = project_path(ref_checkpoint['path'])
            trajectory_run = project_path(RUNS + '/' + value['owner'])
            if (checkpoint_path != trajectory_run / ('epoch_' + str(epoch)) / 'CHECKPOINT.json' or
                    digest(checkpoint_path) != ref_checkpoint['sha256'] or
                    strict_json(checkpoint_path.read_bytes()) != match):
                raise ValueError('DEV checkpoint does not match completed trajectory')
    return spec


def validate_completed_closeout(ref, phase, rank, epoch=None):
    """Revalidate a complete stage, its artifacts, evidence and all ancestors.

    Used by frozen-selection orchestration as well as stage admission. This
    checks saved files only: it does not load a model or start a worker.
    """
    if (phase not in ('qualification', 'trajectory', 'dev') or type(rank) is not int or rank not in (4, 8, 16) or
            (phase == 'dev' and epoch not in (1, 2, 3)) or (phase != 'dev' and epoch is not None)):
        raise ValueError('Exact completed-stage identity required')
    return _validate_completed_closeout(ref, phase, rank, epoch, {}, set())


def _validate_completed_closeout(ref, phase, rank, epoch, cache, active):
    if (type(ref) is not dict or set(ref) != {'path', 'sha256'} or
            type(ref['sha256']) is not str or not HEX64.fullmatch(ref['sha256'])):
        raise ValueError('Exact completed-stage reference required')
    cache_key = (ref['path'], ref['sha256'], phase, rank, epoch)
    if cache_key in cache: return cache[cache_key]
    if cache_key in active: raise ValueError('Cyclic stage prerequisites')
    active.add(cache_key)
    path = project_path(ref['path'])
    if digest(path) != ref['sha256']: raise ValueError('Completed-stage bytes changed')
    value = strict_json(path.read_bytes())
    if (value.get('schema'), value.get('phase'), value.get('rank'), value.get('status'),
            value.get('manifest_sha256')) != ('research4.all64-broader-supervisor-closeout.v1',
            phase, rank, 'completed', MANIFEST):
        raise ValueError('Completed stage identity differs')
    owner = value.get('owner')
    if type(owner) is not str or not HEX32.fullmatch(owner): raise ValueError('Invalid completed-stage owner')
    monitor = project_path(MONITOR + '/' + owner)
    if path != monitor / 'CLOSEOUT.json' or value.get('worker_exit_code') != 0 or value.get('owned_cleanup_confirmed') is not True:
        raise ValueError('Completed stage path or worker cleanup differs')
    run = project_path(RUNS + '/' + owner)
    worker_path = bound_artifact(value.get('worker_receipt'), run)
    if worker_path != run / 'worker_receipt.json': raise ValueError('Unexpected worker receipt path')
    worker_value = strict_json(worker_path.read_bytes())
    release_path = bound_artifact(value.get('release'), monitor)
    guard_path = bound_artifact(value.get('guard_evidence'), monitor)
    identity_path = bound_artifact(value.get('worker_identity'), monitor)
    if (release_path, guard_path, identity_path) != (monitor / 'RELEASE.json', monitor / 'GUARD.jsonl', monitor / 'WORKER_IDENTITY.json'):
        raise ValueError('Unexpected owned evidence path')
    release = strict_json(release_path.read_bytes()); identity = strict_json(identity_path.read_bytes())
    if (release.get('owner') != owner or release.get('lease_released') is not True or
            release.get('owned_cleanup_confirmed') is not True or release.get('error') is not None or
            identity.get('owner') != owner):
        raise ValueError('Completed-stage release or worker identity differs')
    verify_guard_evidence(guard_path, identity['pid'], identity['create_time_epoch'])
    prior_spec_path = bound_artifact(value.get('spec'), project_path('.local'))
    prior_review_path = bound_artifact(value.get('review'), project_path('.local'))
    prior_spec = strict_json(prior_spec_path.read_bytes()); prior_review = strict_json(prior_review_path.read_bytes())
    if (prior_spec.get('owner'), prior_spec.get('phase'), prior_spec.get('rank'), prior_spec.get('manifest_sha256'),
            prior_review.get('schema'), prior_review.get('status'), prior_review.get('spec_sha256'),
            prior_review.get('model_execution')) != (owner, phase, rank, MANIFEST,
            'research4.all64-broader-supervisor-review.v1', 'PASS', value['spec']['sha256'], False):
        raise ValueError('Completed-stage reviewed spec binding differs')
    if value.get('spec_sha256') != value['spec']['sha256']:
        raise ValueError('Closeout spec digest differs')
    prior_sources = prior_spec.get('source_files')
    if type(prior_sources) is not dict or prior_review.get('supervisor_sha256') != prior_sources.get(SUPERVISOR):
        raise ValueError('Completed-stage reviewed supervisor source differs')
    for source_name, source_sha in prior_sources.items():
        if digest(project_path(source_name)) != source_sha:
            raise ValueError('Completed-stage source changed')
    if phase == 'dev' and (epoch not in (1, 2, 3) or value.get('epoch') != epoch or prior_spec.get('epoch') != epoch):
        raise ValueError('Completed DEV epoch differs')
    validate_worker_receipt(worker_value, prior_spec)
    prereqs = prior_spec.get('prerequisites')
    if type(prereqs) is not dict or set(prereqs) != required_prerequisites(phase, rank):
        raise ValueError('Completed stage prerequisites differ')
    trajectory_value = trajectory_worker = None
    for key, ancestor in prereqs.items():
        a_phase = 'trajectory' if key == 'trajectory' else 'qualification'
        a_rank = rank if key == 'trajectory' else int(key)
        a_value, a_worker = _validate_completed_closeout(ancestor, a_phase, a_rank, None, cache, active)
        if key == 'trajectory': trajectory_value, trajectory_worker = a_value, a_worker
    if phase == 'dev':
        checkpoint = prior_spec.get('checkpoint')
        if (type(checkpoint) is not dict or set(checkpoint) != {'path', 'sha256'} or
                type(checkpoint['sha256']) is not str or not HEX64.fullmatch(checkpoint['sha256'])):
            raise ValueError('DEV checkpoint reference differs')
        checkpoint_path = project_path(checkpoint['path'])
        trajectory_run = project_path(RUNS + '/' + trajectory_value['owner'])
        if (checkpoint_path != trajectory_run / ('epoch_' + str(epoch)) / 'CHECKPOINT.json' or
                digest(checkpoint_path) != checkpoint['sha256'] or
                strict_json(checkpoint_path.read_bytes()) != trajectory_worker['checkpoints'][epoch - 1]):
            raise ValueError('DEV checkpoint does not match its completed trajectory')
    active.remove(cache_key); cache[cache_key] = (value, worker_value)
    return cache[cache_key]


def verify_review(spec_path, spec_sha, review_path, review_sha):
    if not HEX64.fullmatch(spec_sha) or digest(spec_path) != spec_sha:
        raise ValueError('Reviewed spec hash differs')
    if not HEX64.fullmatch(review_sha) or digest(review_path) != review_sha:
        raise ValueError('Review receipt hash differs')
    review = strict_json(Path(review_path).read_bytes())
    if (review.get('schema'), review.get('status'), review.get('spec_sha256'),
            review.get('supervisor_sha256'), review.get('model_execution')) != (
            'research4.all64-broader-supervisor-review.v1', 'PASS', spec_sha,
            digest(project_path(SUPERVISOR)), False):
        raise ValueError('Independent exact-source review required')
    return validate_spec(strict_json(Path(spec_path).read_bytes()))


def same_user_known_worker_scan(psutil_module, own_pid, uid):
    """Detect known script names for our UID; do not claim all-user exclusivity."""
    unknown = object(); scanned = other_uid_hidden = 0
    for proc in psutil_module.process_iter(['pid', 'uids', 'cmdline'], ad_value=unknown):
        try:
            pid, uids, args = proc.info['pid'], proc.info['uids'], proc.info['cmdline']
            if pid == own_pid: continue
            if uids is unknown or uids is None:
                raise RuntimeError('Process UID not visible')
            if uids.real != uid:
                if args is unknown or args is None: other_uid_hidden += 1
                continue
            if args is unknown or args is None:
                raise RuntimeError('Same-user process command line not visible')
            scanned += 1
            if any(any(str(arg).endswith(name) for name in KNOWN) for arg in args):
                raise RuntimeError('Known same-user model worker exists')
        except psutil_module.NoSuchProcess:
            continue
        except psutil_module.AccessDenied as exc:
            raise RuntimeError('Cannot establish same-user process visibility') from exc
    return {'scope': 'same_uid_known_worker_names_only', 'same_uid_readable': scanned,
            'other_uid_hidden_command_lines': other_uid_hidden,
            'all_user_or_arbitrary_gpu_exclusivity_established': False}


def guard_runtime():
    path = project_path(GUARD)
    spec = importlib.util.spec_from_file_location('_all64_mini_guard', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.load_runtime(project_path(GUARD_BINDING), GUARD_BINDING_SHA)


def check_sample(guard, target, phase, path):
    sample = guard.sample(target, phase)
    append_json(path, sample)
    if sample.get('valid') is not True or sample.get('stop_required') is not False:
        raise RuntimeError('Native memory guard stopped or failed closed')


def verify_guard_evidence(path, worker_pid, worker_create_time=None):
    values = [strict_json(line) for line in Path(path).read_bytes().splitlines() if line.strip()]
    if (len(values) < 2 or values[0].get('phase') != 'launch' or
            not any(value.get('phase') == 'run' for value in values)):
        raise ValueError('Launch and runtime native guard evidence required')
    for value in values:
        if (value.get('schema') != 'research4.inference-native-guard-sample.v1' or
                value.get('policy_id') != 'research4.mini.inference_NATIVE_R_NORMAL_only.6GiB_launch_4GiB_run.v1' or
                value.get('valid') is not True or value.get('stop_required') is not False or
                value.get('assessment', {}).get('pressure_flag_distinct_from_bytes') != 1):
            raise ValueError('Unhealthy or invalid native guard sample')
        if value.get('phase') == 'run' and value.get('target', {}).get('pid') != worker_pid:
            raise ValueError('Native guard observed the wrong worker')
        if (value.get('phase') == 'run' and worker_create_time is not None and
                value.get('target', {}).get('create_time_epoch') != worker_create_time):
            raise ValueError('Native guard observed a reused worker PID')
    return len(values)


def release_owned_lock(lock, owner):
    path = lock / 'owner.json'
    if path.is_symlink() or strict_json(path.read_bytes()).get('owner') != owner:
        raise RuntimeError('Lease ownership changed; retaining lock')
    path.unlink(); lock.rmdir()


def bound_artifact(ref, within):
    if (type(ref) is not dict or set(ref) != {'path', 'sha256'} or
            type(ref['sha256']) is not str or not HEX64.fullmatch(ref['sha256'])):
        raise ValueError('Exact artifact reference required')
    path = project_path(ref['path'])
    if not path.resolve().is_relative_to(within.resolve()) or digest(path) != ref['sha256']:
        raise ValueError('Owned artifact missing or changed')
    return path


def saved_layout(rank):
    module_path = project_path('mac_studio/game_all_layers_layout.py')
    module_spec = importlib.util.spec_from_file_location('_all64_saved_layout', module_path)
    layout = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(layout)
    expected = layout.expected_adapter_layout(project_path('.local/models/qwen3.8-27b-mlx-4bit'), rank, 64)
    if len(expected) != 992: raise ValueError('Expected 496 LoRA projections / 992 tensors')
    return layout, expected


def verify_adapter(path, rank):
    """Inspect actual saved tensor bytes without loading a model."""
    layout, expected = saved_layout(rank)
    result = layout.verify_adapter_tensor_file(path, expected)
    if (result.get('tensor_count'), result.get('dtype'), result.get('all_finite'),
            result.get('nonzero_lora_b_layer_indices')) != (992, 'F32', True, list(range(64))):
        raise ValueError('Saved LoRA tensors fail coverage, shape or finite readback')
    return result


def verify_state_digest(path, rank):
    """Reproduce the live trainable parameter digest from exact saved F32 bytes."""
    layout, expected = saved_layout(rank)
    header, offset, _ = layout.read_tensor_header(path)
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    if set(tensors) != set(expected): raise ValueError('State digest inventory differs')
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for name in sorted(expected):
            item = tensors[name]; start, end = item['data_offsets']
            if item['dtype'] != 'F32' or item['shape'] != list(expected[name]) or end - start != 4 * math.prod(expected[name]):
                raise ValueError('State digest dtype or shape differs')
            h.update(name.encode()); h.update(str(tuple(expected[name])).encode()); stream.seek(offset + start)
            remaining = end - start
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                if not block: raise ValueError('Truncated state digest')
                h.update(block); remaining -= len(block)
    return h.hexdigest()


def verify_optimizer(path, rank, step):
    """Independently stream the 1,984 Adam moments and two scalars on CPU."""
    import numpy as np
    layout, expected = saved_layout(rank)
    header, offset, _ = layout.read_tensor_header(path)
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    moments = {name + '.' + kind: shape for name, shape in expected.items() for kind in ('m', 'v')}
    if set(tensors) != set(moments) | {'step', 'learning_rate'}: raise ValueError('Adam tensor inventory differs')
    intervals = []
    for item in tensors.values():
        bounds = item.get('data_offsets')
        if type(bounds) is not list or len(bounds) != 2 or any(type(x) is not int for x in bounds) or not 0 <= bounds[0] <= bounds[1]:
            raise ValueError('Adam byte offsets differ')
        intervals.append(tuple(bounds))
    end = 0
    for start, finish in sorted(intervals):
        if start != end: raise ValueError('Adam state bytes overlap or have gaps')
        end = finish
    if offset + end != Path(path).stat().st_size: raise ValueError('Adam state file size differs')
    with Path(path).open('rb') as stream:
        for name, shape in moments.items():
            item = tensors[name]; start, end = item['data_offsets']
            if item['dtype'] != 'F32' or item['shape'] != list(shape) or end - start != 4 * math.prod(shape):
                raise ValueError('Adam moment dtype or shape differs')
            stream.seek(offset + start); remaining = end - start
            while remaining:
                size = min(1024 * 1024, remaining); raw = stream.read(size)
                if len(raw) != size: raise ValueError('Truncated Adam moment')
                values = np.frombuffer(raw, dtype='<f4')
                if not np.isfinite(values).all() or (name.endswith('.v') and np.any(values < 0)):
                    raise ValueError('Adam moments non-finite or second moment negative')
                remaining -= size
        for name, dtype, size, expected_scalar in [('step', 'U64', 8, step), ('learning_rate', 'F32', 4, 1e-5)]:
            item = tensors[name]; start, end = item['data_offsets']
            if item['dtype'] != dtype or item['shape'] != [] or end - start != size: raise ValueError('Adam scalar shape differs')
            stream.seek(offset + start); actual = struct.unpack('<Q' if name == 'step' else '<f', stream.read(size))[0]
            if (name == 'step' and actual != expected_scalar) or (name != 'step' and not math.isclose(actual, expected_scalar, rel_tol=1e-7)):
                raise ValueError('Adam step or rate differs')
    return {'finite_f32_moments': 1984, 'nonnegative_second_moments': True, 'optimizer_step': step}


def verify_qualification_comparisons(run, cases, rank):
    """Replay loss and F32 tensor comparisons from preserved actual bytes."""
    # Called during prerequisite validation before execute() imports the
    # owned-process helper; the reviewed comparison imports the layout by name.
    source_dir = str(PROJECT / 'mac_studio')
    if source_dir not in sys.path: sys.path.insert(0, source_dir)
    path = project_path('mac_studio/game_selected_head_v3_compare.py')
    module_spec = importlib.util.spec_from_file_location('_all64_comparison', path)
    compare = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(compare)
    _, expected = saved_layout(rank)
    for index, case in enumerate(cases, 1):
        for key in ('full_vs_selected', 'ordinary_vs_compiled'):
            recorded = case[key]
            actual = compare.loss_equivalence(recorded['reference_mean_nll'], recorded['selected_mean_nll'],
                                             recorded['target_tokens'], case['target_tokens'], compare.TOLERANCES)
            if actual != recorded or actual['passed'] is not True:
                raise ValueError('Qualification scalar comparison replay failed')
        ordinary = run / ('case_%d_ordinary.safetensors' % index)
        compiled = run / ('case_%d_compiled.safetensors' % index)
        actual = compare.compare_gradients(ordinary, compiled, expected, compare.TOLERANCES,
                                           require_zero_a=(index == 1))
        if (actual['passed'] is not True or actual['tensor_count'] != 992 or
                not math.isclose(actual['global_relative_l2'], case['gradient_global_relative_l2'], rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError('Ordinary versus compiled gradient replay failed')
        if index <= 2:
            native = run / ('case_%d_native.safetensors' % index)
            replay = compare.compare_gradients(native, ordinary, expected, compare.TOLERANCES,
                                               require_zero_a=(index == 1))
            if replay['passed'] is not True or replay['tensor_count'] != 992:
                raise ValueError('Native versus selected gradient replay failed')


def validate_worker_receipt(value, spec):
    if type(value) is not dict or (value.get('phase'), value.get('rank'), value.get('owner'),
            value.get('manifest_sha256')) != (spec['phase'], spec['rank'], spec['owner'], MANIFEST):
        raise ValueError('Worker terminal identity differs')
    common = (value.get('gradient_b_layers') == list(range(64)) and
              value.get('saved_adapter_all_layers_nonzero') is True and
              value.get('base_unchanged') is True and value.get('source_files_unchanged') is True)
    if spec['phase'] == 'dev':
        checkpoint = strict_json(project_path(spec['checkpoint']['path']).read_bytes())
        state_hash = checkpoint.get('trainable_state_sha256')
        if ((value.get('schema'), value.get('status'), value.get('epoch'), value.get('checkpoint'),
             value.get('checkpoint_step'), value.get('development_no_optimizer_updates'), value.get('optimizer_absent'),
             value.get('optimizer_step_before'), value.get('optimizer_step_after'),
             value.get('base_unchanged'), value.get('source_files_unchanged')) !=
                ('research4.all64-broader-dev.v1', 'complete', spec['epoch'], spec['checkpoint'],
                 136 * spec['epoch'], True, True, None, None, True, True) or
                value.get('adapter_sha256') != checkpoint['adapter']['sha256'] or
                type(state_hash) is not str or not HEX64.fullmatch(state_hash) or
                value.get('trainable_state_sha256') != state_hash or
                value.get('trainable_digest_before') != state_hash or value.get('trainable_digest_after') != state_hash):
            raise ValueError('DEV checkpoint identity, unchanged adapter or no-optimizer evidence differs')
        masks_path = project_path(DEV_MASKS)
        if digest(masks_path) != DEV_MASKS_SHA or digest(project_path(SELECTOR)) != SELECTOR_SHA:
            raise ValueError('Frozen DEV selection inputs differ')
        masks = [strict_json(line) for line in masks_path.read_bytes().splitlines() if line.strip()]
        if len(masks) != 36: raise ValueError('Exactly 36 likelihood masks required')
        module_spec = importlib.util.spec_from_file_location('_all64_frozen_selector', project_path(SELECTOR))
        selector = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(selector)
        selected = selector.summarize(value.get('development_rows'), masks)
        if selected['rows'] != 36 or selected['families'] != 11:
            raise ValueError('DEV likelihood families or rows differ')
        return value
    if spec['phase'] == 'qualification':
        valid = ((value.get('schema'), value.get('status'), value.get('executed_updates'),
                  value.get('actual_widths'), value.get('ceiling_test_model_input_width'),
                  value.get('native_full_head_vs_selected_passed'),
                  value.get('ordinary_vs_compiled_gradients_passed'),
                  value.get('native_vs_selected_gradients_passed'),
                  value.get('native_gradient_control_steps')) ==
                 ('research4.all64-broader-qualification.v1', 'qualified', 5,
                  [288, 544, 576, 608], 640, True, True, True, [1, 2]))
    else:
        checkpoints = value.get('checkpoints')
        valid = ((value.get('schema'), value.get('status'), value.get('executed_updates'),
                  value.get('epoch_steps')) == ('research4.all64-broader-trajectory.v1',
                                                'complete', 408, [136, 272, 408]) and
                 isinstance(checkpoints, list) and len(checkpoints) == 3 and
                 [(x.get('epoch'), x.get('optimizer_step')) for x in checkpoints] ==
                 [(1, 136), (2, 272), (3, 408)])
    if not common or not valid: raise ValueError('Worker completion/qualification evidence is incomplete')
    run = project_path(RUNS + '/' + spec['owner'])
    if spec['phase'] == 'qualification':
        cases = value.get('cases')
        if type(cases) is not list or len(cases) != 5 or value.get('maximum_observed_suffix_tokens') != 50:
            raise ValueError('All five numerical qualification cases required')
        lengths_path = project_path(TOKEN_LENGTHS)
        if digest(lengths_path) != TOKEN_LENGTHS_SHA: raise ValueError('Frozen token lengths changed')
        lengths = strict_json(lengths_path.read_bytes())['rows']
        targets = {row['id']: row for row in lengths if row.get('stage') == 'target'}
        if len(targets) != 136: raise ValueError('Frozen training length inventory differs')
        maxima = {}
        for row in targets.values():
            length = row['complete_tokens']; width = 32 * ((length + 31) // 32)
            maxima[width] = max(maxima.get(width, (0, 0)), (length - row['prefix_tokens'], length))
        covered = set()
        for index, (case, width) in enumerate(zip(cases, [288, 544, 576, 608, 640]), 1):
            if (type(case) is not dict or case.get('step') != index or case.get('optimizer_step') != index or
                    case.get('model_input_width') != width or case.get('batch_width') != width + 1 or
                    case.get('ceiling_padded') is not (index == 5) or
                    case.get('full_vs_selected', {}).get('passed') is not True or
                    case.get('ordinary_vs_compiled', {}).get('passed') is not True or
                    case.get('gradient_comparison_passed') is not True):
                raise ValueError('Numerical qualification case mismatch')
            theta = case.get('control_theta_sha256'); rng = case.get('control_rng_before_sha256')
            if (type(theta) is not str or not HEX64.fullmatch(theta) or
                    theta != case.get('control_theta_after_sha256') or case.get('control_theta_unchanged') is not True or
                    type(rng) is not str or not HEX64.fullmatch(rng) or rng != case.get('control_rng_after_sha256') or
                    case.get('control_rng_unchanged') is not True):
                raise ValueError('Control parameters or RNG changed')
            before_layers = case.get('nonzero_b_layers_before_control')
            if (type(before_layers) is not list or any(type(x) is not int or not 0 <= x < 64 for x in before_layers) or
                    (index == 1 and before_layers != []) or (index == 2 and not before_layers)):
                raise ValueError('Fresh / changed B control not established')
            prefix, length = case.get('prefix_tokens'), case.get('semantic_tokens')
            if (type(prefix) is not int or type(length) is not int or not 0 < prefix < length <= 640 or
                    case.get('target_tokens') != length - prefix or
                    case.get('target_positions') != list(range(prefix, length))):
                raise ValueError('Qualification shifted target mask mismatch')
            frozen = targets.get(case.get('row_id'))
            if (type(frozen) is not dict or frozen['complete_tokens'] != length or frozen['prefix_tokens'] != prefix or
                    (index <= 4 and (length - prefix, length) != maxima.get(width)) or
                    (index == 5 and length != max(row['complete_tokens'] for row in targets.values()))):
                raise ValueError('Qualification case is not the required real-width/suffix representative')
            rel = case.get('gradient_global_relative_l2')
            if type(rel) not in (int, float) or not math.isfinite(rel) or rel < 0:
                raise ValueError('Invalid numerical comparison magnitude')
            coverage = case.get('gradient_coverage', {}).get('nonzero_b_layer_indices')
            if type(coverage) is not list or any(type(x) is not int or not 0 <= x < 64 for x in coverage):
                raise ValueError('Qualification gradient coverage missing')
            covered.update(coverage)
            for kind in ('ordinary', 'compiled'):
                recorded = case.get(kind + '_gradient_sha256')
                if type(recorded) is not str or not HEX64.fullmatch(recorded):
                    raise ValueError('Qualification gradient hash missing')
                if digest(run / ('case_%d_%s.safetensors' % (index, kind))) != recorded:
                    raise ValueError('Qualification gradient bytes changed')
            if index <= 2:
                recorded = case.get('native_gradient_sha256')
                if (case.get('native_gradient_comparison_performed') is not True or
                        case.get('native_gradient_comparison_passed') is not True or
                        type(recorded) is not str or not HEX64.fullmatch(recorded) or
                        digest(run / ('case_%d_native.safetensors' % index)) != recorded):
                    raise ValueError('Full native gradient control differs')
            elif (case.get('native_gradient_comparison_performed') is not False or
                  case.get('native_gradient_comparison_passed') is not None or
                  case.get('native_gradient_sha256') is not None):
                raise ValueError('Unexpected native-gradient control metadata')
        if covered != set(range(64)):
            raise ValueError('Qualification B-gradient coverage is incomplete')
        event_path = run / 'qualification_cases.jsonl'
        if event_path.is_symlink() or not event_path.is_file(): raise ValueError('Qualification case log missing')
        if [strict_json(line) for line in event_path.read_bytes().splitlines() if line.strip()] != cases:
            raise ValueError('Qualification log and receipt disagree')
        verify_qualification_comparisons(run, cases, spec['rank'])
        adapter = run / 'qualification_adapter.safetensors'
        if digest(adapter) != value.get('qualification_adapter_sha256'):
            raise ValueError('Qualification adapter bytes changed')
        verify_adapter(adapter, spec['rank'])
    else:
        for checkpoint in value['checkpoints']:
            epoch = checkpoint['epoch']; folder = run / ('epoch_' + str(epoch))
            for kind in ('adapter', 'config', 'optimizer_state'):
                bound_artifact(checkpoint.get(kind), folder)
            adapter_path = project_path(checkpoint['adapter']['path']); verify_adapter(adapter_path, spec['rank'])
            if verify_state_digest(adapter_path, spec['rank']) != checkpoint.get('trainable_state_sha256'):
                raise ValueError('Saved parameter-state digest differs')
            if verify_optimizer(project_path(checkpoint['optimizer_state']['path']), spec['rank'], epoch * 136) != checkpoint.get('optimizer_verification'):
                raise ValueError('Saved optimizer-state readback differs')
            config = strict_json(project_path(checkpoint['config']['path']).read_bytes())
            expected_config = {'fine_tune_type': 'lora', 'num_layers': 64, 'batch_size': 1,
                'max_seq_length': 640, 'mask_prompt': True, 'resume_adapter_file': None,
                'optimizer': 'adam', 'grad_accumulation_steps': 1, 'grad_checkpoint': True,
                'lora_parameters': {'rank': spec['rank'], 'dropout': 0.0, 'scale': 2.0},
                'iters': epoch * 136, 'learning_rate': 1e-5, 'seed': 20261108,
                'optimizer_config': {'adam': {'betas': [.9, .999], 'eps': 1e-8, 'bias_correction': False}}}
            if config != expected_config: raise ValueError('Saved adapter optimizer/config differs')
            saved = folder / 'CHECKPOINT.json'
            if saved.is_symlink() or not saved.is_file() or strict_json(saved.read_bytes()) != checkpoint:
                raise ValueError('Immutable checkpoint descriptor differs')
    return value


def execute(spec_path, spec_sha, review_path, review_sha):
    if sys.platform != 'darwin' or os.uname().machine != 'arm64':
        raise RuntimeError('Mini Apple-silicon macOS required')
    import psutil
    spec_path = Path(spec_path).absolute(); review_path = Path(review_path).absolute()
    for p in (spec_path, review_path):
        if not p.resolve().is_relative_to(PROJECT.resolve()) or any(x.is_symlink() for x in (p, *p.parents)):
            raise ValueError('Reviewed inputs must be canonical local files')
    spec = verify_review(spec_path, spec_sha, review_path, review_sha)
    sys.path.insert(0, str(PROJECT / 'mac_studio'))
    import owned_validation_process as owned
    owner = spec['owner']; start_epoch = time.time(); start_mono = time.monotonic()
    deadline = spec['deadline_epoch']
    if deadline > start_epoch + spec['max_wall_seconds']:
        raise ValueError('Absolute deadline exceeds declared stage wall bound')
    mono_deadline = start_mono + (deadline - start_epoch)
    if deadline <= start_epoch: raise TimeoutError('No remaining stage budget')
    run = project_path(RUNS + '/' + owner); monitor = project_path(MONITOR + '/' + owner)
    lock = project_path(LOCK); run.parent.mkdir(parents=True, exist_ok=True)
    monitor.parent.mkdir(parents=True, exist_ok=True); lock.parent.mkdir(parents=True, exist_ok=True)
    run.mkdir(); monitor.mkdir()
    exclusive_json(monitor / 'START.json', {'owner': owner, 'phase': spec['phase'], 'rank': spec['rank'],
        'supervisor_pid': os.getpid(),
        'spec_sha256': spec_sha, 'review_sha256': review_sha, 'start_epoch': start_epoch,
        'effective_deadline_epoch': deadline, 'automatic_retry': False})
    slot = owned.LaunchSlot(); lock_owned = False; clean = False; result = None; error = None
    try:
        guard = guard_runtime()
        check_sample(guard, {'pid': os.getpid(), 'create_time_epoch': psutil.Process(os.getpid()).create_time()},
                     'launch', monitor / 'GUARD.jsonl')
        lock.mkdir(); lock_owned = True
        exclusive_json(lock / 'owner.json', {'owner': owner, 'pid': os.getpid(), 'kind': KIND})
        visibility = same_user_known_worker_scan(psutil, os.getpid(), os.getuid())
        program = DEV_WORKER if spec['phase'] == 'dev' else WORKER
        authority = {'schema': 'research4.all64-broader-worker-authority.v1', 'phase': spec['phase'],
            'rank': spec['rank'], 'owner': owner, 'supervisor_pid': os.getpid(), 'deadline_epoch': deadline,
            'manifest_sha256': MANIFEST, 'worker_sha256': spec['source_files'][program], 'spec_sha256': spec_sha}
        if spec['phase'] == 'dev': authority['epoch'] = spec['epoch']
        exclusive_json(run / 'worker_authority.json', authority)
        command = [str(PROJECT / PYTHON), '-B', '-u', str(project_path(program)),
            '--phase', spec['phase'], '--rank', str(spec['rank']), '--run-dir', str(run), '--owner', owner,
            '--deadline-epoch', str(deadline), '--manifest-sha256', MANIFEST,
            '--spec', str(spec_path), '--spec-sha256', spec_sha]
        if spec['phase'] == 'dev': command.extend(['--epoch', str(spec['epoch'])])
        with (monitor / 'worker.log').open('xb') as log:
            pending = owned.launch(slot, command, stdin=subprocess.DEVNULL, stdout=log, cwd=PROJECT)
            if pending: raise InterruptedError('Signal during owned worker spawn')
            target = {'pid': slot.process.pid, 'create_time_epoch': psutil.Process(slot.process.pid).create_time()}
            exclusive_json(monitor / 'WORKER_IDENTITY.json', {'owner': owner, **target, 'visibility': visibility})
            while True:
                if time.time() >= deadline or time.monotonic() >= mono_deadline:
                    raise TimeoutError('Original finite stage deadline reached')
                if (monitor / 'STOP').exists(): raise InterruptedError('External operator stop')
                if owned.peek_exited(slot.process.pid): break
                check_sample(guard, target, 'run', monitor / 'GUARD.jsonl')
                time.sleep(1.0)
            code, descendants, pending = owned.cleanup(slot.process); clean = True
        if code != 0 or descendants or pending:
            raise RuntimeError('Worker exit or process-group cleanup failed')
        validate_spec(spec, now=min(time.time(), deadline - 1e-6))
        receipt = run / 'worker_receipt.json'
        if not receipt.is_file() or receipt.is_symlink(): raise RuntimeError('Worker terminal receipt missing')
        value = validate_worker_receipt(strict_json(receipt.read_bytes()), spec)
        guard_path = monitor / 'GUARD.jsonl'; verify_guard_evidence(guard_path, target['pid'], target['create_time_epoch'])
        identity_path = monitor / 'WORKER_IDENTITY.json'
        result = {'schema': 'research4.all64-broader-supervisor-closeout.v1', 'status': 'completed',
            'owner': owner, 'phase': spec['phase'], 'rank': spec['rank'], 'manifest_sha256': MANIFEST,
            'epoch': spec['epoch'],
            'spec_sha256': spec_sha, 'worker_receipt': {'path': receipt.relative_to(PROJECT).as_posix(),
            'sha256': digest(receipt)}, 'worker_exit_code': code, 'owned_cleanup_confirmed': True,
            'spec': {'path': spec_path.relative_to(PROJECT).as_posix(), 'sha256': spec_sha},
            'review': {'path': review_path.relative_to(PROJECT).as_posix(), 'sha256': review_sha},
            'guard_evidence': {'path': guard_path.relative_to(PROJECT).as_posix(), 'sha256': digest(guard_path)},
            'worker_identity': {'path': identity_path.relative_to(PROJECT).as_posix(), 'sha256': digest(identity_path)},
            'known_worker_scan': visibility, 'automatic_retry': False}
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        if slot.process is not None and not clean:
            try:
                code, descendants, pending = owned.cleanup(slot.process); clean = True
            except BaseException as cleanup_error:
                error['cleanup_error'] = f'{type(cleanup_error).__name__}: {cleanup_error}'
        raise
    finally:
        may_release = slot.process is None or clean
        if lock_owned and may_release:
            try: release_owned_lock(lock, owner)
            except BaseException as exc:
                may_release = False
                if error is None: error = {'type': type(exc).__name__, 'message': str(exc)}
        summary = {'schema': 'research4.all64-broader-supervisor-release.v1', 'owner': owner,
            'worker_started': slot.process is not None, 'owned_cleanup_confirmed': clean,
            'lease_released': bool(lock_owned and may_release), 'lease_retained_uncertain': bool(lock_owned and not may_release),
            'error': error, 'finished_epoch': time.time()}
        exclusive_json(monitor / 'RELEASE.json', summary)
        if result is not None and error is None and summary['lease_released']:
            release_path = monitor / 'RELEASE.json'
            result['release'] = {'path': release_path.relative_to(PROJECT).as_posix(), 'sha256': digest(release_path)}
            exclusive_json(monitor / 'CLOSEOUT.json', result)
        elif result is not None and error is None:
            raise RuntimeError('Completion lease release not established')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--review', required=True); parser.add_argument('--review-sha256', required=True)
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    if not args.execute: raise RuntimeError('Explicit exact reviewed execution required')
    def interrupted(signum, frame): raise InterruptedError('Operator termination signal')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP): signal.signal(sig, interrupted)
    print(json.dumps(execute(args.spec, args.spec_sha256, args.review, args.review_sha256)))


if __name__ == '__main__': main()
