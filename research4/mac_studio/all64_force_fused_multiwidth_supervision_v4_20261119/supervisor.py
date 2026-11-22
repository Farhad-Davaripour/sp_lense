"""One-use scoped fused-attention primal supervisor; no optimizer updates.

This separate program never qualifies a rank or starts an HPO trajectory. It
preserves the failed attempt, owns the shared model lease, and monitors the
reviewed Mini native pressure guard while the observer runs in a child process.
The actual observation is recorded even when it reveals a mismatch.
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
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE_SUP = 'mac_studio/all64_broader_tools_supervision_v1_20261119/supervisor.py'
BASE_SUP_SHA = 'fbe34eb0dcc9b6e0746d18e209249348a12f18d5c797b53e809cbe34460dae8e'
THIS = 'mac_studio/all64_force_fused_multiwidth_supervision_v4_20261119/supervisor.py'
WORKER = 'mac_studio/all64_force_fused_widths_v4b_20261119/diagnostic_worker.py'
OBSERVER = 'mac_studio/all64_force_fused_widths_v4_20261119/width_observer.py'
POLICY = 'mac_studio/all64_force_fused_widths_v4_20261119/scoped_sdpa_widths.py'
FIXTURE = 'mac_studio/all64_force_fused_widths_v4_20261119/WIDTH_CASES.json'
TOKEN_LENGTHS = '.local/broader_continuity_tools_v2_cpu_frozen_01bed751/TOKEN_LENGTHS.json'
TOKEN_LENGTHS_SHA = '6d071d6f11e4b442ccc9d6d929669891066f54519aff449ebba1ebb3cc28cb73'
ORIGINAL_POLICY = 'mac_studio/game_block3_sdpa_policy.py'
ORIGINAL_POLICY_SHA = '80817ebfab8733c1ab6261e78922137998ce29e48733d4bab95fa346d3e795a3'
FAILED_WORKER = 'mac_studio/all64_broader_v2_20261119/worker.py'
FAILED_WORKER_SHA = 'b4996bc66695ff8abda70cc78764875d7cdfadcb1be1d7666b985fd6f68e938e'
FAILED_OWNER = 'c18da432781347699cae2aa6e15dec90'
FAILED_RUN = '.local/experiments/all64_broader_v2_20261119/runs/' + FAILED_OWNER
FAILED_MONITOR = '.local/monitor/all64_broader_supervision_v1_20261119/' + FAILED_OWNER
FAILED_WORKER_FAILURE_SHA = '3d3c4886117f78100ae2b06825ad9521fe78929268de2875c9e787b33a948670'
FAILED_SUP_RELEASE_SHA = '4184aad2f160ff9d9954f79af7900d2312e0207024cd320dc24de9256440f11e'
PRIOR_OWNER = 'f73186a2ee0d44448e163a14c4808326'
PRIOR_RUN = '.local/experiments/all64_native_primal_diagnostic_v1_20261119/runs/' + PRIOR_OWNER
PRIOR_MONITOR = '.local/monitor/all64_native_primal_supervision_v1_20261119/' + PRIOR_OWNER
PRIOR_OBSERVATION_SHA = '255e95806cef0f342cbb9b9a1bdbb7224434ef9acb8b916771ef818738150058'
PRIOR_RECEIPT_SHA = 'cb412c289ff676d3472fb73666be146badc13be6fa81f5dd47f8027be6287a35'
PRIOR_RELEASE_SHA = '60528fc196eade21cf2afd580736a572c497f3f7b5bcc4d43606b30f71a95ec6'
ORDER_OWNER = '3a78fc712c9f409e9c3b22f6f255e44d'
ORDER_RUN = '.local/experiments/all64_primal_order_diagnostic_v2_20261119/runs/' + ORDER_OWNER
ORDER_MONITOR = '.local/monitor/all64_primal_order_supervision_v2_20261119/' + ORDER_OWNER
ORDER_OBSERVATION_SHA = '00cc08e63344d0dd50144b0fcbc7e847ebf5226a090ed20571d6eb5ce4ef1b5e'
ORDER_RECEIPT_SHA = '3c9dfd8608891e2a045078b020766e348fd0d3e3cd8f3f4b51e617fb1c702860'
ORDER_RELEASE_SHA = 'c4077a407b5e017ff0936aa628b42444cc8357bf4dc6cb51b566b119a90bea71'
FUSED_OWNER = 'bc118dccf6d4426da1d3f5b2d217e787'
FUSED_RUN = '.local/experiments/all64_force_fused_diagnostic_v3_20261119/runs/' + FUSED_OWNER
FUSED_MONITOR = '.local/monitor/all64_native_primal_fused_supervision_v3_20261119/' + FUSED_OWNER
FUSED_OBSERVATION_SHA = '548a570aea2c4a66a7568eb572b6d75e6a9f06760e8f1d775180b92c14ba4f68'
FUSED_RECEIPT_SHA = '7adf1a46d6203578ff41401ba12da1abbfe4dae27b0eb981bfe3fe74e17cb726'
FUSED_RELEASE_SHA = '09e16a596b4fedad11b018180edf61d2f93e44c49d4b1403df62fa8bed93ae65'
MANIFEST = 'scenarios/service_preservation/2026-11-19_17/broader_continuity_tools_v2/FROZEN_MANIFEST.json'
MANIFEST_SHA = 'dc1d812ba3e2b2aa147b1fbe148440ff6a5327a5a67feb8d32f330883ac98f70'
GUARD = 'mac_studio/mini_native_guard_20261119/mini_guard.py'
GUARD_BINDING = 'mac_studio/mini_native_guard_20261119/MINI_CROSSBUILT_BINDING_3778b7a0365143dc8f9d4e9922e5fd05.json'
GUARD_BINDING_SHA = 'acc7e820fa25f2f00b7394fb4ed562568f917d6fa43c540fe994679979fe3f52'
OWNED = 'mac_studio/owned_validation_process.py'
PYTHON = '.local/envs/mini-inference-20261119/bin/python'
LOCK = '.local/validation/qwen38_shared_v1/inference.lock'
RUNS = '.local/experiments/all64_force_fused_widths_v4_20261119/runs'
MONITOR = '.local/monitor/all64_force_fused_multiwidth_supervision_v4_20261119'
KIND = 'all64_force_fused_widths_diagnostic_v4'
HEX64 = re.compile(r'^[0-9a-f]{64}$'); HEX32 = re.compile(r'^[0-9a-f]{32}$')


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('Regular file required')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def local(relative):
    if type(relative) is not str or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Repository-relative POSIX path required')
    part = Path(relative)
    if part.is_absolute() or '..' in part.parts: raise ValueError('Unsafe path')
    path = ROOT / part
    if not path.resolve().is_relative_to(ROOT.resolve()) or 'onedrive' in str(path.resolve()).lower():
        raise ValueError('Canonical non-OneDrive path required')
    for current in (path, *path.parents):
        if current == ROOT.parent: break
        if current.is_symlink(): raise ValueError('Symlinked repository path')
    return path


def strict_json(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out: raise ValueError('Duplicate JSON key')
            out[k] = v
        return out
    def invalid(_): raise ValueError('Nonfinite JSON')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def once(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def get_base():
    if digest(local(BASE_SUP)) != BASE_SUP_SHA: raise ValueError('Original guard helper source changed')
    spec = importlib.util.spec_from_file_location('_all64_diagnostic_base_supervisor', local(BASE_SUP))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate_spec(spec, now=None):
    keys = {'schema', 'owner', 'rank', 'manifest_sha256', 'deadline_epoch', 'max_wall_seconds', 'python_binary_sha256', 'source_files',
            'failed_owner', 'failed_worker_sha256', 'failed_worker_failure_sha256', 'failed_supervisor_release_sha256',
            'prior_diagnostic_owner', 'prior_diagnostic_observation_sha256', 'prior_diagnostic_release_sha256',
            'prior_diagnostic_receipt_sha256', 'order_diagnostic_owner', 'order_diagnostic_observation_sha256',
            'order_diagnostic_release_sha256', 'order_diagnostic_receipt_sha256', 'fused_diagnostic_owner',
            'fused_diagnostic_observation_sha256', 'fused_diagnostic_receipt_sha256', 'fused_diagnostic_release_sha256'}
    if type(spec) is not dict or set(spec) != keys or spec['schema'] != 'research4.all64-force-fused-widths-supervision-spec.v1':
        raise ValueError('Exact diagnostic spec required')
    if type(spec['owner']) is not str or not HEX32.fullmatch(spec['owner']) or spec['owner'] in (FAILED_OWNER, PRIOR_OWNER, ORDER_OWNER, FUSED_OWNER):
        raise ValueError('Fresh owner required')
    if type(spec['rank']) is not int or spec['rank'] != 4 or spec['manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('Exact frozen rank-4 diagnostic required')
    if (spec['failed_owner'], spec['failed_worker_sha256'], spec['failed_worker_failure_sha256'],
            spec['failed_supervisor_release_sha256']) != (FAILED_OWNER, FAILED_WORKER_SHA,
            FAILED_WORKER_FAILURE_SHA, FAILED_SUP_RELEASE_SHA):
        raise ValueError('Original failed attempt identity differs')
    if (spec['prior_diagnostic_owner'], spec['prior_diagnostic_observation_sha256'],
            spec['prior_diagnostic_release_sha256'], spec['prior_diagnostic_receipt_sha256'],
            spec['order_diagnostic_owner'], spec['order_diagnostic_observation_sha256'],
            spec['order_diagnostic_release_sha256'], spec['order_diagnostic_receipt_sha256']) != (
            PRIOR_OWNER, PRIOR_OBSERVATION_SHA, PRIOR_RELEASE_SHA, PRIOR_RECEIPT_SHA,
            ORDER_OWNER, ORDER_OBSERVATION_SHA, ORDER_RELEASE_SHA, ORDER_RECEIPT_SHA):
        raise ValueError('Prior no-update diagnostic identities differ')
    if (spec['fused_diagnostic_owner'], spec['fused_diagnostic_observation_sha256'],
            spec['fused_diagnostic_receipt_sha256'], spec['fused_diagnostic_release_sha256']) != (
            FUSED_OWNER, FUSED_OBSERVATION_SHA, FUSED_RECEIPT_SHA, FUSED_RELEASE_SHA):
        raise ValueError('Prior forced-fused diagnostic identity differs')
    deadline, maximum = spec['deadline_epoch'], spec['max_wall_seconds']
    if (type(deadline) not in (int, float) or not math.isfinite(deadline) or
            deadline <= (time.time() if now is None else now) or type(maximum) is not int or not 0 < maximum <= 14400):
        raise ValueError('Prospective bounded diagnostic deadline required')
    python = (ROOT / PYTHON).resolve(strict=True)
    if ('onedrive' in str(python).lower() or type(spec['python_binary_sha256']) is not str or
            not HEX64.fullmatch(spec['python_binary_sha256']) or digest(python) != spec['python_binary_sha256']):
        raise ValueError('Virtual environment Python identity differs')
    sources = spec['source_files']; required = {THIS, BASE_SUP, WORKER, OBSERVER, POLICY, FIXTURE, ORIGINAL_POLICY,
                                                FAILED_WORKER, MANIFEST, GUARD, GUARD_BINDING, OWNED}
    if type(sources) is not dict or not required.issubset(sources): raise ValueError('Diagnostic source set incomplete')
    for path, sha in sources.items():
        if type(sha) is not str or not HEX64.fullmatch(sha) or digest(local(path)) != sha:
            raise ValueError('Diagnostic source changed: ' + str(path))
    for path, sha in ((BASE_SUP, BASE_SUP_SHA), (FAILED_WORKER, FAILED_WORKER_SHA),
                      (MANIFEST, MANIFEST_SHA), (GUARD_BINDING, GUARD_BINDING_SHA),
                      (ORIGINAL_POLICY, ORIGINAL_POLICY_SHA)):
        if sources[path] != sha: raise ValueError('Pinned historical source differs')
    original, derived = local(ORIGINAL_POLICY).read_bytes(), local(POLICY).read_bytes()
    old = (b"        valid = (\n"
           b"            list(queries.shape) == [1, 24, 1023, 256]\n"
           b"            and list(keys.shape) == [1, 4, 1023, 256]\n"
           b"            and list(values.shape) == [1, 4, 1023, 256]\n")
    new = (b"        _approved_width = queries.shape[2] if len(queries.shape) == 4 else None\n"
           b"        valid = (\n"
           b"            type(_approved_width) is int and _approved_width in (288, 544, 576, 608, 640)\n"
           b"            and list(queries.shape) == [1, 24, _approved_width, 256]\n"
           b"            and list(keys.shape) == [1, 4, _approved_width, 256]\n"
           b"            and list(values.shape) == [1, 4, _approved_width, 256]\n")
    if original.count(old) != 1 or original.replace(old, new) != derived:
        raise ValueError('Scoped policy must differ only by the approved five-width shape guard')
    if digest(local(TOKEN_LENGTHS)) != TOKEN_LENGTHS_SHA:
        raise ValueError('Frozen sequence length inventory changed')
    fixture = strict_json(local(FIXTURE).read_bytes()); cases = fixture.get('cases')
    if (fixture.get('schema') != 'research4.all64-force-fused-width-cases.v1' or
            fixture.get('frozen_manifest_sha256') != MANIFEST_SHA or fixture.get('qwen_model_loaded') is not False or
            type(cases) is not list or [row.get('model_input_width') for row in cases] != [288, 544, 576, 608, 640] or
            [row.get('ceiling_padded') for row in cases] != [False, False, False, False, True]):
        raise ValueError('Exact five-width frozen fixture required')
    lengths = strict_json(local(TOKEN_LENGTHS).read_bytes())['rows']
    targets = {row['id']: row for row in lengths if row.get('stage') == 'target'}
    if len(targets) != 136: raise ValueError('Frozen target length inventory differs')
    for case in cases:
        row = targets.get(case.get('row_id'))
        if (type(row) is not dict or case.get('semantic_tokens') != row.get('complete_tokens') or
                case.get('prefix_tokens') != row.get('prefix_tokens') or
                case.get('target_tokens') != row['complete_tokens'] - row['prefix_tokens'] or
                type(case.get('token_u32be_sha256')) is not str or not HEX64.fullmatch(case['token_u32be_sha256'])):
            raise ValueError('Width fixture does not match frozen target geometry')
    failure = local(FAILED_RUN + '/worker_failure.json'); release = local(FAILED_MONITOR + '/RELEASE.json')
    if digest(failure) != FAILED_WORKER_FAILURE_SHA or digest(release) != FAILED_SUP_RELEASE_SHA:
        raise ValueError('Original failure/release bytes differ')
    old_failure = strict_json(failure.read_bytes()); old_release = strict_json(release.read_bytes())
    if (old_failure.get('error_type'), old_failure.get('message'), old_release.get('owner'),
            old_release.get('owned_cleanup_confirmed'), old_release.get('lease_released')) != (
            'ValueError', 'Full-native derivative primal differs from full forward', FAILED_OWNER, True, True):
        raise ValueError('Original failure or release is not established')
    for run, monitor, obs_name, owner, obs_sha, receipt_sha, release_sha in (
            (PRIOR_RUN, PRIOR_MONITOR, 'NATIVE_PRIMAL_OBSERVATION.json', PRIOR_OWNER, PRIOR_OBSERVATION_SHA, PRIOR_RECEIPT_SHA, PRIOR_RELEASE_SHA),
            (ORDER_RUN, ORDER_MONITOR, 'PRIMAL_ORDER_OBSERVATION.json', ORDER_OWNER, ORDER_OBSERVATION_SHA, ORDER_RECEIPT_SHA, ORDER_RELEASE_SHA)):
        obs, receipt, rel = local(run + '/' + obs_name), local(run + '/worker_receipt.json'), local(monitor + '/RELEASE.json')
        if (digest(obs), digest(receipt), digest(rel)) != (obs_sha, receipt_sha, release_sha):
            raise ValueError('Prior diagnostic evidence changed')
        observed, record, released = strict_json(obs.read_bytes()), strict_json(receipt.read_bytes()), strict_json(rel.read_bytes())
        if (observed.get('adam_updates'), observed.get('qualification_claim'), record.get('owner'),
                record.get('observation_sha256'), record.get('optimizer_step_before'), record.get('optimizer_step_after'),
                released.get('owner'), released.get('owned_cleanup_confirmed'), released.get('lease_released')) != (
                0, False, owner, obs_sha, 0, 0, owner, True, True):
            raise ValueError('Prior diagnostic observation or release differs')
    fused_obs, fused_receipt = local(FUSED_RUN + '/NATIVE_PRIMAL_OBSERVATION.json'), local(FUSED_RUN + '/worker_receipt.json')
    fused_release = local(FUSED_MONITOR + '/RELEASE.json')
    if (digest(fused_obs), digest(fused_receipt), digest(fused_release)) != (
            FUSED_OBSERVATION_SHA, FUSED_RECEIPT_SHA, FUSED_RELEASE_SHA):
        raise ValueError('Prior forced-fused diagnostic evidence changed')
    observed, record, released = strict_json(fused_obs.read_bytes()), strict_json(fused_receipt.read_bytes()), strict_json(fused_release.read_bytes())
    if (observed.get('status'), observed.get('adam_updates'), observed.get('qualification_claim'),
            record.get('owner'), record.get('observation_sha256'), record.get('policy_restored'),
            record.get('optimizer_step_before'), record.get('optimizer_step_after'),
            released.get('owner'), released.get('owned_cleanup_confirmed'), released.get('lease_released')) != (
            'MATCH', 0, False, FUSED_OWNER, FUSED_OBSERVATION_SHA, True, 0, 0, FUSED_OWNER, True, True):
        raise ValueError('Prior forced-fused match/restoration/release differs')
    return spec


def verify_review(spec_path, spec_sha, review_path, review_sha):
    if not HEX64.fullmatch(spec_sha) or digest(spec_path) != spec_sha: raise ValueError('Spec SHA differs')
    if not HEX64.fullmatch(review_sha) or digest(review_path) != review_sha: raise ValueError('Review SHA differs')
    review = strict_json(Path(review_path).read_bytes())
    if (review.get('schema'), review.get('status'), review.get('spec_sha256'),
            review.get('supervisor_sha256'), review.get('model_execution')) != (
            'research4.all64-force-fused-widths-supervisor-review.v1', 'PASS', spec_sha, digest(local(THIS)), False):
        raise ValueError('Independent diagnostic review required')
    return validate_spec(strict_json(Path(spec_path).read_bytes()))


def validate_observation(value, expected):
    """Check one pinned real-row direct/AD scalar result under original tolerance."""
    if (type(value) is not dict or value.get('schema') != 'research4.all64-force-fused-width-observation.v1' or
            value.get('case') != expected or value.get('status') not in ('MATCH', 'MISMATCH') or
            value.get('adam_updates') != 0 or value.get('qualification_claim') is not False or
            (value.get('loss_atol'), value.get('loss_rtol')) != (0.0001, 0.0001)):
        raise ValueError('Exact no-update width observation required')
    before, plain, after = value.get('before'), value.get('after_plain'), value.get('after_ad')
    if any(type(x) is not dict for x in (before, plain, after)):
        raise ValueError('Diagnostic state snapshots missing')
    for state in (before, plain, after):
        if any(type(state.get(k)) is not str or not HEX64.fullmatch(state[k]) for k in ('trainable_state_sha256', 'rng_state_sha256')):
            raise ValueError('Diagnostic state digest missing')
    if (len({x['trainable_state_sha256'] for x in (before, plain, after)}) != 1 or
            len({x['rng_state_sha256'] for x in (before, plain, after)}) != 1 or
            value.get('theta_unchanged') is not True or value.get('rng_unchanged') is not True):
        raise ValueError('Per-width trainable state or RNG changed')
    numbers = []
    for key in ('plain', 'differentiated'):
        row = value.get(key)
        if (type(row) is not dict or row.get('target_count') != expected['target_tokens'] or
                type(row.get('loss')) is not dict or row['loss'].get('classification') != 'finite' or
                type(row['loss'].get('value')) not in (int, float) or not math.isfinite(row['loss']['value'])):
            raise ValueError('Finite direct/AD scalar and frozen target count required')
        numbers.append(row['loss']['value'])
    difference = abs(numbers[0] - numbers[1]); allowed = 0.0001 + 0.0001 * abs(numbers[0])
    passed = difference <= allowed
    if (value.get('absolute_difference') != difference or value.get('original_allowed_difference') != allowed or
            value.get('original_tolerance_passed') is not passed or value.get('status') != ('MATCH' if passed else 'MISMATCH')):
        raise ValueError('Width result or original tolerance classification differs')
    return value


def validate_cross_case_state(cases):
    if type(cases) is not list or len(cases) != 5: raise ValueError('Five width observations required')
    states = [(row['before']['trainable_state_sha256'], row['before']['rng_state_sha256']) for row in cases]
    if len(set(states)) != 1: raise ValueError('Trainable state or RNG changed between width cases')
    return True


def validate_policy_and_state(receipt, run, worker_pid):
    """Verify exact scoped policy calls and full-state evidence after the worker exits."""
    policy_path = run / 'FORCE_POLICY.json'; policy = strict_json(policy_path.read_bytes())
    if (receipt.get('policy_receipt_sha256') != digest(policy_path) or
            policy.get('schema') != 'research4.scoped-block3-sdpa-policy.v1' or
            policy.get('policy') != 'force_fused' or policy.get('owned_process_pid') != worker_pid or
            policy.get('off_scope_calls') != 0 or policy.get('restoration_verified') is not True or
            policy.get('native_base_and_fast_untouched') is not True or policy.get('environment_unchanged') is not True or
            policy.get('installed_files_modified') is not False or receipt.get('policy_restored') is not True):
        raise ValueError('Scoped attention policy receipt or restoration differs')
    calls = policy.get('calls')
    if type(calls) is not list or len(calls) != policy.get('matched_calls'):
        raise ValueError('Scoped attention call inventory differs')
    widths = (288, 544, 576, 608, 640)
    phases = {f'force_fused_{width}_{phase}' for width in widths for phase in ('direct', 'ad')}
    phase_counts = {phase: sum(call.get('phase') == phase for call in calls) for phase in phases}
    if any(count < 16 for count in phase_counts.values()) or receipt.get('force_fused_phase_counts') != phase_counts:
        raise ValueError('Expected fused attention scope in every width and phase is missing')
    for call in calls:
        q = call.get('q_shape'); width = q[2] if isinstance(q, list) and len(q) == 4 else None
        if (width not in widths or call.get('phase') not in {f'force_fused_{width}_direct', f'force_fused_{width}_ad'} or
                (q, call.get('k_shape'), call.get('v_shape')) !=
                ([1, 24, width, 256], [1, 4, width, 256], [1, 4, width, 256]) or
                not isinstance(call.get('dtype'), str) or not call['dtype'].endswith('bfloat16') or
                call.get('cache') is not None or call.get('mask') != 'causal' or call.get('sinks') is not None or
                call.get('scale') != .0625 or call.get('force_fused_requested') is not True):
            raise ValueError('Unexpected scoped attention call')
    before_path, after_path = run / 'FULL_STATE_BEFORE.json', run / 'FULL_STATE_AFTER.json'
    if (receipt.get('state_before_sha256'), receipt.get('state_after_sha256')) != (digest(before_path), digest(after_path)):
        raise ValueError('Full state inventory file hashes differ')
    before, after = strict_json(before_path.read_bytes()), strict_json(after_path.read_bytes())
    for state in (before, after):
        if (type(state) is not dict or type(state.get('sha256')) is not str or not HEX64.fullmatch(state['sha256']) or
                type(state.get('inventory')) is not list or not state['inventory'] or
                state.get('leaf_count') != len(state['inventory']) or type(state.get('parameter_count')) is not int or
                state['parameter_count'] < 1 or type(state.get('array_count')) is not int or state['array_count'] < 1):
            raise ValueError('Full enumerable state digest or inventory missing')
    if (before.get('all_parameters_included') is not True or after.get('all_parameters_included') is not True or
            before.get('sha256') != after.get('sha256') or before.get('inventory') != after.get('inventory') or
            receipt.get('model_state_digest_before') != before.get('sha256') or
            receipt.get('model_state_digest_after') != after.get('sha256') or
            receipt.get('state_unchanged') is not True or receipt.get('training_flags_unchanged') is not True or
            receipt.get('rng_unchanged') is not True):
        raise ValueError('Full enumerable model state, flags or RNG changed')
    flags_a, flags_b = receipt.get('training_flags_before'), receipt.get('training_flags_after')
    rng_a, rng_b = receipt.get('rng_before_sha256'), receipt.get('rng_after_sha256')
    if (type(flags_a) is not dict or flags_a != flags_b or flags_a.get('all_training') is not True or
            type(rng_a) is not str or not HEX64.fullmatch(rng_a) or rng_a != rng_b):
        raise ValueError('Recorded full-state flag/RNG evidence differs')


def execute(spec_path, spec_sha, review_path, review_sha):
    if sys.platform != 'darwin' or os.uname().machine != 'arm64': raise RuntimeError('Mini Apple silicon required')
    import psutil
    spec_path, review_path = Path(spec_path).absolute(), Path(review_path).absolute()
    for p in (spec_path, review_path):
        if not p.resolve().is_relative_to((ROOT / '.local').resolve()) or any(q.is_symlink() for q in (p, *p.parents)):
            raise ValueError('Exact canonical reviewed local input required')
    spec = verify_review(spec_path, spec_sha, review_path, review_sha)
    base = get_base(); base.KNOWN = (*base.KNOWN, 'all64_force_fused_widths_v4b_20261119/diagnostic_worker.py')
    sys.path.insert(0, str(ROOT / 'mac_studio')); import owned_validation_process as owned
    owner = spec['owner']; start = time.time(); mono = time.monotonic(); deadline = spec['deadline_epoch']
    if not start < deadline <= start + spec['max_wall_seconds']: raise ValueError('Original diagnostic deadline exceeds bound')
    monotonic_deadline = mono + deadline - start
    run = local(RUNS + '/' + owner); monitor = local(MONITOR + '/' + owner); lock = local(LOCK)
    for p in (run.parent, monitor.parent, lock.parent): p.mkdir(parents=True, exist_ok=True)
    run.mkdir(); monitor.mkdir(); slot = owned.LaunchSlot(); lock_owned = False; clean = False
    error = None; observed = None; code = None; target = None
    once(monitor / 'START.json', {'owner': owner, 'supervisor_pid': os.getpid(), 'spec_sha256': spec_sha,
        'review_sha256': review_sha, 'deadline_epoch': deadline, 'adam_updates_authorized': 0, 'automatic_retry': False})
    try:
        guard = base.guard_runtime()
        base.check_sample(guard, {'pid': os.getpid(), 'create_time_epoch': psutil.Process(os.getpid()).create_time()},
                          'launch', monitor / 'GUARD.jsonl')
        lock.mkdir(); lock_owned = True
        once(lock / 'owner.json', {'owner': owner, 'pid': os.getpid(), 'kind': KIND})
        visibility = base.same_user_known_worker_scan(psutil, os.getpid(), os.getuid())
        authority = {'schema': 'research4.all64-force-fused-widths-diagnostic-authority.v1', 'owner': owner,
            'supervisor_pid': os.getpid(), 'deadline_epoch': deadline, 'spec_sha256': spec_sha,
            'worker_sha256': spec['source_files'][WORKER], 'failed_owner': FAILED_OWNER,
            'failed_worker_sha256': FAILED_WORKER_SHA}
        once(run / 'worker_authority.json', authority)
        command = [str(ROOT / PYTHON), '-B', '-u', str(local(WORKER)), '--run-dir', str(run), '--owner', owner,
                   '--deadline-epoch', str(deadline), '--spec', str(spec_path), '--spec-sha256', spec_sha]
        with (monitor / 'worker.log').open('xb') as log:
            pending = owned.launch(slot, command, stdin=subprocess.DEVNULL, stdout=log, cwd=ROOT)
            if pending: raise InterruptedError('Signal during owned diagnostic spawn')
            target = {'pid': slot.process.pid, 'create_time_epoch': psutil.Process(slot.process.pid).create_time()}
            once(monitor / 'WORKER_IDENTITY.json', {'owner': owner, **target, 'visibility': visibility})
            while True:
                if time.time() >= deadline or time.monotonic() >= monotonic_deadline:
                    raise TimeoutError('Original diagnostic deadline reached')
                if (monitor / 'STOP').exists(): raise InterruptedError('External operator stop')
                if owned.peek_exited(slot.process.pid): break
                base.check_sample(guard, target, 'run', monitor / 'GUARD.jsonl')
                time.sleep(1.0)
            code, descendants, pending = owned.cleanup(slot.process); clean = True
        if descendants or pending: raise RuntimeError('Diagnostic process-group cleanup uncertain')
        # Check source and guard identity again, even if the worker recorded a mismatch.
        validate_spec(spec, now=min(time.time(), deadline - 1e-6))
        base.verify_guard_evidence(monitor / 'GUARD.jsonl', target['pid'], target['create_time_epoch'])
        if code != 0:
            raise RuntimeError('Width diagnostic ended without a complete worker receipt')
        receipt_path = run / 'worker_receipt.json'
        receipt = strict_json(receipt_path.read_bytes())
        if (receipt.get('schema'), receipt.get('status'), receipt.get('owner'), receipt.get('rank'),
                receipt.get('failed_owner'), receipt.get('failed_worker_sha256'), receipt.get('manifest_sha256'),
                receipt.get('observed_widths'), receipt.get('optimizer_constructed_by_original_loader'),
                receipt.get('optimizer_step_before'), receipt.get('optimizer_step_after'), receipt.get('adam_updates'),
                receipt.get('qualification_claim')) != (
                'research4.all64-force-fused-widths-diagnostic-receipt.v1', 'observed_not_qualification', owner, 4,
                FAILED_OWNER, FAILED_WORKER_SHA, MANIFEST_SHA, [288, 544, 576, 608, 640], True, 0, 0, 0, False):
            raise ValueError('Width diagnostic worker terminal receipt differs')
        fixture = strict_json(local(FIXTURE).read_bytes()); cases = fixture['cases']; results = receipt.get('case_results')
        if type(results) is not list or len(results) != len(cases): raise ValueError('Five case results required')
        statuses, measurements = [], []
        for expected, ref in zip(cases, results):
            width = expected['model_input_width']; folder = run / ('width_' + str(width))
            obs_path, plain_path = folder / 'WIDTH_OBSERVATION.json', folder / 'PLAIN_FORWARD.json'
            if (type(ref) is not dict or set(ref) != {'width', 'observation_sha256', 'plain_forward_sha256'} or
                    ref['width'] != width or ref['observation_sha256'] != digest(obs_path) or
                    ref['plain_forward_sha256'] != digest(plain_path)):
                raise ValueError('Width observation files differ')
            measured = validate_observation(strict_json(obs_path.read_bytes()), expected)
            measurements.append(measured)
            baseline = strict_json(plain_path.read_bytes())
            if any(measured.get(key) != value for key, value in baseline.items()):
                raise ValueError('Saved per-width pre-AD observation differs')
            statuses.append(measured['status'])
        validate_cross_case_state(measurements)
        if type(receipt.get('all_widths_matched')) is not bool or receipt['all_widths_matched'] != all(x == 'MATCH' for x in statuses):
            raise ValueError('Aggregate width tolerance result differs')
        validate_policy_and_state(receipt, run, target['pid'])
        observed = {'schema': 'research4.all64-force-fused-widths-supervisor-observation.v1',
            'status': 'OBSERVED_NOT_QUALIFICATION', 'owner': owner, 'worker_exit_code': code,
            'widths': [288, 544, 576, 608, 640], 'width_statuses': statuses,
            'all_widths_matched': receipt['all_widths_matched'],
            'worker_receipt': {'path': receipt_path.relative_to(ROOT).as_posix(), 'sha256': digest(receipt_path)},
            'state_unchanged': receipt['state_unchanged'], 'rng_unchanged': receipt['rng_unchanged'],
            'optimizer_constructed_by_original_loader': True, 'optimizer_step_before': 0, 'optimizer_step_after': 0,
            'adam_updates': 0, 'qualification_claim': False, 'automatic_retry': False}

    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        if slot.process is not None and not clean:
            try: code, descendants, pending = owned.cleanup(slot.process); clean = True
            except BaseException as cleanup: error['cleanup_error'] = f'{type(cleanup).__name__}: {cleanup}'
        raise
    finally:
        released = slot.process is None or clean
        if lock_owned and released:
            try: base.release_owned_lock(lock, owner)
            except BaseException as exc:
                released = False
                if error is None: error = {'type': type(exc).__name__, 'message': str(exc)}
        release = {'schema': 'research4.all64-force-fused-widths-supervisor-release.v1', 'owner': owner,
            'worker_started': slot.process is not None, 'worker_exit_code': code, 'owned_cleanup_confirmed': clean,
            'lease_released': bool(lock_owned and released), 'lease_retained_uncertain': bool(lock_owned and not released),
            'error': error, 'finished_epoch': time.time()}
        once(monitor / 'RELEASE.json', release)
        if observed is not None and error is None and release['lease_released']:
            observed['release'] = {'path': (monitor / 'RELEASE.json').relative_to(ROOT).as_posix(),
                                   'sha256': digest(monitor / 'RELEASE.json')}
            observed['guard_evidence'] = {'path': (monitor / 'GUARD.jsonl').relative_to(ROOT).as_posix(),
                                          'sha256': digest(monitor / 'GUARD.jsonl')}
            once(monitor / 'OBSERVATION_CLOSEOUT.json', observed)
        elif observed is not None and error is None:
            raise RuntimeError('Diagnostic physical release not established')
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--review', required=True); parser.add_argument('--review-sha256', required=True)
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    if not args.execute: raise RuntimeError('Explicit separately reviewed execution required')
    def interrupted(signum, frame): raise InterruptedError('Operator termination')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP): signal.signal(sig, interrupted)
    print(json.dumps(execute(args.spec, args.spec_sha256, args.review, args.review_sha256)))


if __name__ == '__main__': main()
