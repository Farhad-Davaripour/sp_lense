"""One-use SDPA VJP diagnostic supervisor; no optimizer updates.

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
THIS = 'mac_studio/all64_sdpa_vjp_supervision_v2_20261119/supervisor.py'
CAPTURE_WORKER = 'mac_studio/all64_sdpa_vjp_probe_v1_20261119/capture_fused_v2.py'
FALLBACK_WORKER = 'mac_studio/all64_sdpa_vjp_probe_v1_20261119/fallback_reference.py'
OWNED_V2 = 'mac_studio/all64_broader_forced_supervision_v3_20261119/owned_process_v2.py'
OWNED_V2_SHA = '519765e91594735b3d055d220bc751ddaaa836bcf91923d4b59924e48d493c0f'
WIDTH_AUDIT = '.local/reviews/all64_broader_v2_20261119/V4_RECOVERED_INDEPENDENT_AUDIT_bdfd9635516d4198bc34a9cd2837f16e.json'
WIDTH_AUDIT_SHA = '3023e734d2879ef50425ed422a4b85ecf9f3d961427a3d6b780db0bae6e76969'
FAILED_CAPTURE_OWNER = '710a18c5566f4de7b6ae2ffcd345bd23'
FAILED_CAPTURE_FAILURE = '.local/experiments/all64_sdpa_vjp_probe_v1_20261119/runs/' + FAILED_CAPTURE_OWNER + '/worker_failure.json'
FAILED_CAPTURE_FAILURE_SHA = 'c9d2b0b7b631ff5a27400773f660ad97b33b59faf68f97c7b10e71e810b92193'
FAILED_CAPTURE_RELEASE = '.local/monitor/all64_sdpa_vjp_supervision_v1_20261119/' + FAILED_CAPTURE_OWNER + '/RELEASE.json'
FAILED_CAPTURE_RELEASE_SHA = 'c1a337a3727ad236fc3364e1b31a76ae52018f497ff0a8bfb91f46ac128fe327'
FAILED_WORKER = 'mac_studio/all64_broader_v2_20261119/worker.py'
FAILED_WORKER_SHA = 'b4996bc66695ff8abda70cc78764875d7cdfadcb1be1d7666b985fd6f68e938e'
FAILED_OWNER = 'c18da432781347699cae2aa6e15dec90'
FAILED_RUN = '.local/experiments/all64_broader_v2_20261119/runs/' + FAILED_OWNER
FAILED_MONITOR = '.local/monitor/all64_broader_supervision_v1_20261119/' + FAILED_OWNER
FAILED_WORKER_FAILURE_SHA = '3d3c4886117f78100ae2b06825ad9521fe78929268de2875c9e787b33a948670'
FAILED_SUP_RELEASE_SHA = '4184aad2f160ff9d9954f79af7900d2312e0207024cd320dc24de9256440f11e'
MANIFEST = 'scenarios/service_preservation/2026-11-19_17/broader_continuity_tools_v2/FROZEN_MANIFEST.json'
MANIFEST_SHA = 'dc1d812ba3e2b2aa147b1fbe148440ff6a5327a5a67feb8d32f330883ac98f70'
GUARD = 'mac_studio/mini_native_guard_20261119/mini_guard.py'
GUARD_BINDING = 'mac_studio/mini_native_guard_20261119/MINI_CROSSBUILT_BINDING_3778b7a0365143dc8f9d4e9922e5fd05.json'
GUARD_BINDING_SHA = 'acc7e820fa25f2f00b7394fb4ed562568f917d6fa43c540fe994679979fe3f52'
OWNED = 'mac_studio/owned_validation_process.py'
PYTHON = '.local/envs/mini-inference-20261119/bin/python'
LOCK = '.local/validation/qwen38_shared_v1/inference.lock'
RUNS = '.local/experiments/all64_sdpa_vjp_probe_v1_20261119/runs'
MONITOR = '.local/monitor/all64_sdpa_vjp_supervision_v1_20261119'
KIND = 'all64_sdpa_vjp_probe_v1'
HEX64 = re.compile(r'^[0-9a-f]{64}$'); HEX32 = re.compile(r'^[0-9a-f]{32}$')
ENV_KEYS = ('MLX_SDPA_VJP_FALLBACK', 'MLX_ENABLE_TF32', 'MLX_METAL_GPU_ARCH', 'MLX_DISABLE_COMPILE')


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


def artifact(ref, within):
    if (type(ref) is not dict or set(ref) != {'path', 'sha256'} or type(ref['sha256']) is not str or
            not HEX64.fullmatch(ref['sha256'])):
        raise ValueError('Exact artifact reference required')
    path = local(ref['path'])
    if not path.resolve().is_relative_to(within.resolve()) or digest(path) != ref['sha256']:
        raise ValueError('Bound artifact missing or changed')
    return path


def validate_capture(ref, fixture):
    if type(ref) is not dict or set(ref) != {'path', 'sha256'}: raise ValueError('Capture closeout reference required')
    path = local(ref['path'])
    if digest(path) != ref['sha256']: raise ValueError('Capture closeout changed')
    closeout = strict_json(path.read_bytes()); owner = closeout.get('owner')
    if (type(owner) is not str or not HEX32.fullmatch(owner) or path != local(MONITOR + '/' + owner + '/PROBE_CLOSEOUT.json') or
            (closeout.get('schema'), closeout.get('phase'), closeout.get('status'), closeout.get('adam_updates'),
             closeout.get('qualification_claim')) != ('research4.all64-sdpa-vjp-probe-supervisor-closeout.v1',
             'capture_fused', 'CAPTURED_NOT_QUALIFIED', 0, False)):
        raise ValueError('Capture stage identity differs')
    monitor, run = local(MONITOR + '/' + owner), local(RUNS + '/' + owner)
    receipt_path = artifact(closeout.get('worker_receipt'), run)
    for name, key in [('FULL_STATE_BEFORE.json', 'full_state_before'), ('FULL_STATE_AFTER.json', 'full_state_after')]:
        if artifact(closeout.get(key), run) != run / name:
            raise ValueError('Prior capture full-state evidence path differs')
    release_path = artifact(closeout.get('release'), monitor)
    guard_path = artifact(closeout.get('guard_evidence'), monitor)
    identity_path = artifact(closeout.get('worker_identity'), monitor)
    spec_path = artifact(closeout.get('spec'), ROOT / '.local')
    review_path = artifact(closeout.get('review'), ROOT / '.local')
    old_spec, review = strict_json(spec_path.read_bytes()), strict_json(review_path.read_bytes())
    if (old_spec.get('schema'), old_spec.get('phase'), old_spec.get('owner'), old_spec.get('rank'),
            old_spec.get('manifest_sha256'), review.get('schema'), review.get('status'),
            review.get('spec_sha256'), review.get('model_execution')) != (
            'research4.all64-sdpa-vjp-probe-supervision-spec.v1', 'capture_fused', owner, 4, MANIFEST_SHA,
            'research4.all64-sdpa-vjp-probe-supervisor-review.v1', 'PASS', closeout['spec']['sha256'], False):
        raise ValueError('Prior capture reviewed source/spec differs')
    old_sources = old_spec.get('source_files')
    if type(old_sources) is not dict or review.get('supervisor_sha256') != old_sources.get(THIS):
        raise ValueError('Prior capture supervisor source differs')
    for name, sha in old_sources.items():
        if digest(local(name)) != sha: raise ValueError('Prior capture source changed')
    receipt, release = strict_json(receipt_path.read_bytes()), strict_json(release_path.read_bytes())
    identity = strict_json(identity_path.read_bytes())
    if (receipt.get('schema'), receipt.get('phase'), receipt.get('status'), receipt.get('owner'),
            receipt.get('rank'), receipt.get('manifest_sha256'), receipt.get('adam_updates'), receipt.get('fixture')) != (
            'research4.all64-sdpa-vjp-probe-receipt.v1', 'capture_fused', 'captured_not_qualified',
            owner, 4, MANIFEST_SHA, 0, fixture):
        raise ValueError('Capture worker receipt or fixture differs')
    if (release.get('owner'), release.get('lease_released'), release.get('owned_cleanup_confirmed'),
            release.get('error'), identity.get('owner')) != (owner, True, True, None, owner):
        raise ValueError('Capture physical release evidence differs')
    artifact(fixture, run)
    validate_probe_receipt(receipt, old_spec, run)
    marker_ref = closeout.get('guard_exit_race'); marker = None
    if marker_ref is not None: marker = strict_json(artifact(marker_ref, monitor).read_bytes())
    verify_guard_evidence(guard_path, identity['pid'], identity['create_time_epoch'], marker)
    return closeout, receipt


def validate_spec(spec, now=None):
    keys = {'schema', 'owner', 'phase', 'rank', 'manifest_sha256', 'deadline_epoch', 'max_wall_seconds',
            'python_binary_sha256', 'source_files', 'capture_closeout', 'fixture',
            'prior_capture_failure_sha256', 'prior_capture_release_sha256'}
    if type(spec) is not dict or set(spec) != keys or spec['schema'] != 'research4.all64-sdpa-vjp-probe-supervision-spec.v1':
        raise ValueError('Exact diagnostic spec required')
    if type(spec['owner']) is not str or not HEX32.fullmatch(spec['owner']): raise ValueError('Fresh owner required')
    if type(spec['rank']) is not int or spec['rank'] != 4 or spec['manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('Exact frozen rank-4 diagnostic required')
    if spec['phase'] not in ('capture_fused', 'fallback_reference'): raise ValueError('Unknown probe phase')
    if spec['owner'] == FAILED_CAPTURE_OWNER:
        raise ValueError('Prior failed owner cannot be reused')
    if (spec['prior_capture_failure_sha256'], spec['prior_capture_release_sha256']) != (
            FAILED_CAPTURE_FAILURE_SHA, FAILED_CAPTURE_RELEASE_SHA):
        raise ValueError('Prior failed capture evidence binding differs')
    failure_path, release_path = local(FAILED_CAPTURE_FAILURE), local(FAILED_CAPTURE_RELEASE)
    if digest(failure_path) != FAILED_CAPTURE_FAILURE_SHA or digest(release_path) != FAILED_CAPTURE_RELEASE_SHA:
        raise ValueError('Prior failed capture evidence changed')
    failure, release = strict_json(failure_path.read_bytes()), strict_json(release_path.read_bytes())
    if ((failure.get('status'), failure.get('error_type'), failure.get('message')) !=
            ('failed', 'ValueError', 'Capture identity/deadline differs') or
            (release.get('owner'), release.get('worker_exit_code'), release.get('owned_cleanup_confirmed'),
             release.get('lease_released'), release.get('lease_retained_uncertain')) !=
            (FAILED_CAPTURE_OWNER, 1, True, True, False)):
        raise ValueError('Prior capture failure or release differs')
    deadline, maximum = spec['deadline_epoch'], spec['max_wall_seconds']
    if (type(deadline) not in (int, float) or not math.isfinite(deadline) or
            deadline <= (time.time() if now is None else now) or type(maximum) is not int or not 0 < maximum <= 3600):
        raise ValueError('Prospective bounded diagnostic deadline required')
    python = (ROOT / PYTHON).resolve(strict=True)
    if ('onedrive' in str(python).lower() or type(spec['python_binary_sha256']) is not str or
            not HEX64.fullmatch(spec['python_binary_sha256']) or digest(python) != spec['python_binary_sha256']):
        raise ValueError('Virtual environment Python identity differs')
    worker = CAPTURE_WORKER if spec['phase'] == 'capture_fused' else FALLBACK_WORKER
    sources = spec['source_files']; required = {THIS, BASE_SUP, worker, OWNED_V2, WIDTH_AUDIT, FAILED_WORKER, MANIFEST, GUARD, GUARD_BINDING}
    if type(sources) is not dict or not required.issubset(sources): raise ValueError('Diagnostic source set incomplete')
    for path, sha in sources.items():
        if type(sha) is not str or not HEX64.fullmatch(sha) or digest(local(path)) != sha:
            raise ValueError('Diagnostic source changed: ' + str(path))
    for path, sha in ((BASE_SUP, BASE_SUP_SHA), (OWNED_V2, OWNED_V2_SHA), (WIDTH_AUDIT, WIDTH_AUDIT_SHA),
                      (FAILED_WORKER, FAILED_WORKER_SHA), (MANIFEST, MANIFEST_SHA), (GUARD_BINDING, GUARD_BINDING_SHA)):
        if sources[path] != sha: raise ValueError('Pinned historical source differs')
    width_audit = strict_json(local(WIDTH_AUDIT).read_bytes())
    if (width_audit.get('schema'), width_audit.get('status'), width_audit.get('owner'), width_audit.get('adam_updates'),
            width_audit.get('qualification_claim')) != ('research4.all64-v4-recovered-independent-audit.v1',
            'FIVE_SCALARS_MATCH_PHYSICAL_RELEASE_RECOVERED_SUPERVISOR_UNCERTAIN',
            'bdfd9635516d4198bc34a9cd2837f16e', 0, False):
        raise ValueError('Reviewed no-update width evidence differs')
    if spec['phase'] == 'capture_fused':
        if spec['capture_closeout'] is not None or spec['fixture'] is not None: raise ValueError('Fresh capture cannot use an earlier fixture')
    else:
        closeout, _ = validate_capture(spec['capture_closeout'], spec['fixture'])
        if closeout['owner'] == spec['owner']: raise ValueError('Fallback process requires a fresh distinct owner')
    return spec


def check_environment(environ):
    """Match the observed V4 baseline; fallback is set only in child argv."""
    result = {key: environ.get(key) for key in ENV_KEYS}
    if any(value is not None for value in result.values()):
        raise ValueError('Unexpected inherited MLX environment; no silent policy override')
    return result


def verify_review(spec_path, spec_sha, review_path, review_sha):
    if not HEX64.fullmatch(spec_sha) or digest(spec_path) != spec_sha: raise ValueError('Spec SHA differs')
    if not HEX64.fullmatch(review_sha) or digest(review_path) != review_sha: raise ValueError('Review SHA differs')
    review = strict_json(Path(review_path).read_bytes())
    if (review.get('schema'), review.get('status'), review.get('spec_sha256'),
            review.get('supervisor_sha256'), review.get('model_execution')) != (
            'research4.all64-sdpa-vjp-probe-supervisor-review.v1', 'PASS', spec_sha, digest(local(THIS)), False):
        raise ValueError('Independent diagnostic review required')
    return validate_spec(strict_json(Path(spec_path).read_bytes()))


def terminal_exit_sample(sample, target):
    """An exited target is never healthy; this only classifies terminal race evidence."""
    assessment = sample.get('assessment', {})
    return (sample.get('phase') == 'run' and sample.get('target') == target and
            sample.get('valid') is False and sample.get('stop_required') is True and
            assessment.get('reason') == 'invalid_cache_estimate_sample' and
            assessment.get('errors') == ['rusage reports exited process'] and
            assessment.get('pressure_flag_distinct_from_bytes') == 1)


def sample_hash(sample):
    return hashlib.sha256(json.dumps(sample, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_guard_evidence(path, worker_pid, worker_create_time=None, terminal_exit_marker=None):
    values = [strict_json(line) for line in Path(path).read_bytes().splitlines() if line.strip()]
    if (len(values) < 2 or values[0].get('phase') != 'launch' or
            not any(value.get('phase') == 'run' and value.get('valid') is True and
                    value.get('stop_required') is False for value in values)):
        raise ValueError('Launch and runtime native guard evidence required')
    for index, value in enumerate(values):
        if (terminal_exit_marker is not None and index == len(values) - 1 and
                terminal_exit_sample(value, {'pid': worker_pid, 'create_time_epoch': worker_create_time}) and
                terminal_exit_marker == {'schema': 'research4.all64-terminal-guard-exit-race.v1',
                    'worker_pid': worker_pid, 'worker_create_time_epoch': worker_create_time,
                    'sample_sha256': sample_hash(value), 'waitid_exited_after_sample': True,
                    'invalid_sample_treated_as_healthy': False, 'worker_allowed_to_continue': False}):
            continue
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
    if terminal_exit_marker is not None and not terminal_exit_sample(
            values[-1], {'pid': worker_pid, 'create_time_epoch': worker_create_time}):
        raise ValueError('Exit marker does not match final invalid sample')
    return len(values)


def validate_numeric_comparison(value):
    """Accept a faithfully reported mismatch/nonfinite result, never call it a pass."""
    if type(value) is not dict: raise ValueError('Comparison object missing')
    if value.get('available') is False:
        if value.get('reason') not in ('fused_reference_nonfinite', 'ordinary_reference_nonfinite'):
            raise ValueError('Unavailable reference reason differs')
        return
    finite = value.get('observed_finite'); failures = value.get('observed_nonfinite_count')
    if type(finite) is not bool or type(failures) is not int or failures < 0 or (finite != (failures == 0)):
        raise ValueError('Comparison finite/nonfinite count differs')
    if (value.get('atol'), value.get('rtol')) != (1e-6, 1e-3):
        raise ValueError('Original strict gradient tolerance changed')
    if not finite:
        if value.get('passes_original_strict_gradient_gate') is not False:
            raise ValueError('Nonfinite comparison cannot pass')
        return
    rel, violations = value.get('global_relative_l2'), value.get('elementwise_violations')
    if (type(violations) is not int or violations < 0 or
            (rel is not None and (type(rel) not in (int, float) or not math.isfinite(rel) or rel < 0))):
        raise ValueError('Invalid gradient comparison measures')
    expected = violations == 0 and rel is not None and rel <= 1e-3
    if value.get('passes_original_strict_gradient_gate') is not expected:
        raise ValueError('Gradient comparison pass classification differs')


def validate_probe_receipt(value, spec, run):
    """Check exact no-update stage evidence; receipt is not qualification."""
    phase = spec['phase']
    if (type(value) is not dict or value.get('schema') != 'research4.all64-sdpa-vjp-probe-receipt.v1' or
            (value.get('phase'), value.get('owner'), value.get('rank'), value.get('manifest_sha256'),
             value.get('adam_updates'), value.get('qualification_claim')) !=
            (phase, spec['owner'], 4, MANIFEST_SHA, 0, False)):
        raise ValueError('Probe receipt identity or no-update scope differs')
    if phase == 'capture_fused':
        if (value.get('status') != 'captured_not_qualified' or value.get('untapped_tapped_parity_passed') is not True or
                value.get('full_native_primal_passed') is not True or value.get('model_execution') is not True or
                value.get('optimizer_step_before') != 0 or value.get('optimizer_step_after') != 0 or
                value.get('optimizer_constructed_by_original_loader') is not True or
                (value.get('semantic_tokens'), value.get('input_tokens'), value.get('target_tokens'),
                 value.get('ceiling_padding')) != (590, 640, 30, True) or
                value.get('row_id') != 'sustained_multi_incident_02.12' or
                value.get('token_u32be_sha256') != '9fc914b962fa2cbb96f70f31c32e12aa5b26dc99ee0272158c36353c1bd8cf93'):
            raise ValueError('Capture parity evidence is missing')
        names = {'fixture': 'ATTENTION_FIXTURE.safetensors', 'fused_vjp': 'FUSED_VJP.safetensors',
                 'lora_sample': 'LORA_SAMPLE.safetensors', 'qk_summary': 'QK_SUMMARY.json',
                 'tap_receipt': 'TAP_RECEIPT.json', 'policy_receipt': 'FORCE_POLICY.json',
                 'synthetic_fixture': 'SYNTHETIC_FIXTURE.safetensors',
                 'synthetic_fused_vjp': 'SYNTHETIC_FUSED_VJP.safetensors'}
        files = {key: artifact(value.get(key), run) for key in names}
        if any(files[key] != run / name for key, name in names.items()):
            raise ValueError('Unexpected capture artifact filename')
        before, after = value.get('full_state_before_sha256'), value.get('full_state_after_sha256')
        theta, rng_a, rng_b = value.get('theta_sha256'), value.get('rng_before_sha256'), value.get('rng_after_sha256')
        if (type(before) is not str or not HEX64.fullmatch(before) or before != after or
                type(theta) is not str or not HEX64.fullmatch(theta) or type(rng_a) is not str or
                not HEX64.fullmatch(rng_a) or rng_a != rng_b or
                type(value.get('training_flags_before')) is not dict or
                value.get('training_flags_before') != value.get('training_flags_after') or
                value['training_flags_before'].get('all_training') is not True):
            raise ValueError('Capture model state, mode or RNG changed')
        state_a, state_b = strict_json((run / 'FULL_STATE_BEFORE.json').read_bytes()), strict_json((run / 'FULL_STATE_AFTER.json').read_bytes())
        if (type(state_a.get('inventory')) is not list or not state_a['inventory'] or
                state_a.get('leaf_count') != len(state_a['inventory']) or
                state_b.get('leaf_count') != len(state_b.get('inventory', [])) or
                state_a.get('sha256') != before or state_b.get('sha256') != before or
                state_a.get('inventory') != state_b.get('inventory') or
                state_a.get('all_parameters_included') is not True or state_b.get('all_parameters_included') is not True):
            raise ValueError('Full enumerable model-state evidence differs')
        qk = strict_json(files['qk_summary'].read_bytes()); layers = qk.get('layers')
        if (qk.get('schema') != 'research4.qwen-actual-qk-maxima.v1' or type(layers) is not list or
                [row.get('layer') for row in layers] != list(range(3, 64, 4)) or
                qk.get('worst_layer') != value.get('worst_layer')):
            raise ValueError('QK summary lacks the exact 16 attention layers')
        try:
            winner = max(layers, key=lambda row: (row['max_unscaled_qk_absolute_causal'], -row['layer']))
        except (KeyError, TypeError, ValueError):
            raise ValueError('Invalid per-layer QK maxima') from None
        if winner['layer'] != value['worst_layer'] or any(type(row.get('max_unscaled_qk_absolute_causal')) not in (int, float)
                or not math.isfinite(row['max_unscaled_qk_absolute_causal']) or row['max_unscaled_qk_absolute_causal'] < 0 for row in layers):
            raise ValueError('Worst-layer selection does not match finite QK maxima')
        tap = strict_json(files['tap_receipt'].read_bytes()); policy = strict_json(files['policy_receipt'].read_bytes())
        if (tap.get('schema'), tap.get('attention_class_restored'), tap.get('sdpa_alias_modified_by_tap'),
                tap.get('optimizer_updates'), tap.get('kernel_profile_claimed'), tap.get('expected_layers'),
                tap.get('vjp_capture_count')) != (
                'research4.attention-output-tap.v1', True, False, 0, False, list(range(3, 64, 4)), 16):
            raise ValueError('Attention tap restoration evidence differs')
        if (policy.get('schema'), policy.get('policy'), policy.get('off_scope_calls'),
                policy.get('restoration_verified'), policy.get('native_base_and_fast_untouched'),
                policy.get('environment_unchanged'), policy.get('installed_files_modified')) != (
                'research4.scoped-block3-sdpa-policy.v1', 'force_fused', 0, True, True, True, False):
            raise ValueError('Forced policy restoration evidence differs')
        identity_path = local(MONITOR + '/' + spec['owner'] + '/WORKER_IDENTITY.json')
        identity = strict_json(identity_path.read_bytes()); calls = policy.get('calls')
        if (tap.get('owned_pid') != identity.get('pid') or policy.get('owned_process_pid') != identity.get('pid') or
                policy.get('environment') != {'MLX_SDPA_VJP_FALLBACK': None, 'MLX_ENABLE_TF32': None,
                                              'MLX_METAL_GPU_ARCH': None} or
                type(calls) is not list or len(calls) != policy.get('matched_calls') or not calls):
            raise ValueError('Capture tap/policy process or environment binding differs')
        if (any(sum(call.get('phase') == phase for call in calls) < 16
                for phase in ('untapped_direct', 'untapped_ad', 'tapped_ad')) or
                any(not any(call.get('phase') == 'reconstruct_' + str(layer) for call in calls)
                    for layer in range(3, 64, 4))):
            raise ValueError('Actual forced Qwen phases or per-layer reconstruction missing')
        counts = {}
        for call in calls:
            phase_name = call.get('phase')
            if type(phase_name) is not str: raise ValueError('Forced policy phase missing')
            counts[phase_name] = counts.get(phase_name, 0) + 1
        allowed_phases = {'untapped_direct', 'untapped_ad', 'tapped_ad'} | {'reconstruct_' + str(x) for x in range(3, 64, 4)}
        if set(counts) != allowed_phases or value.get('forced_phase_counts') != counts:
            raise ValueError('Forced phase count summary differs from actual calls')
        for call in calls:
            if ((call.get('q_shape'), call.get('k_shape'), call.get('v_shape')) !=
                    ([1, 24, 640, 256], [1, 4, 640, 256], [1, 4, 640, 256]) or
                    not isinstance(call.get('dtype'), str) or not call['dtype'].endswith('bfloat16') or
                    call.get('cache') is not None or call.get('mask') != 'causal' or call.get('sinks') is not None or
                    call.get('scale') != .0625 or call.get('force_fused_requested') is not True):
                raise ValueError('Capture forced attention call left exact scope')
        if (type(value.get('synthetic_score_max')) not in (int, float) or not math.isfinite(value['synthetic_score_max']) or
                value['synthetic_score_max'] < 10000 or value.get('synthetic_scale') != .125):
            raise ValueError('Separate model-free switch-discriminator evidence differs')
    else:
        if value.get('status') != 'observed_not_qualification' or value.get('fixture') != spec['fixture']:
            raise ValueError('Fallback did not bind the exact captured fixture')
        capture_closeout, capture_receipt = validate_capture(spec['capture_closeout'], spec['fixture'])
        owner = capture_closeout['owner']
        artifact(spec['fixture'], local(RUNS + '/' + owner))
        obs_path = artifact(value.get('observation'), run)
        if obs_path != run / 'FALLBACK_OBSERVATION.json': raise ValueError('Unexpected fallback observation path')
        obs = strict_json(obs_path.read_bytes())
        if ((obs.get('schema'), obs.get('fixture'), obs.get('captured_fused_vjp'), obs.get('row_id'),
             obs.get('token_u32be_sha256'), obs.get('semantic_tokens'), obs.get('input_tokens'), obs.get('target_tokens'),
             obs.get('ceiling_padding'), obs.get('fallback_requested_before_mlx_import'), obs.get('adam_updates'),
             obs.get('optimizer_step_before'), obs.get('optimizer_step_after'), obs.get('policy_restored'),
             obs.get('qualification_claim')) != ('research4.all64-sdpa-fallback-observation.v1', spec['fixture'],
             capture_receipt['fused_vjp'], 'sustained_multi_incident_02.12',
             '9fc914b962fa2cbb96f70f31c32e12aa5b26dc99ee0272158c36353c1bd8cf93', 590, 640, 30, True,
             True, 0, 0, 0, True, False)):
            raise ValueError('Fallback observation identity, environment or no-update scope differs')
        if obs.get('environment') != {'MLX_SDPA_VJP_FALLBACK': '1', 'MLX_ENABLE_TF32': None, 'MLX_METAL_GPU_ARCH': None}:
            raise ValueError('Fallback environment differs from pre-import control')
        for name in ('theta', 'full_state', 'rng'):
            before, after = obs.get(name + '_before_sha256'), obs.get(name + '_after_sha256')
            if type(before) is not str or not HEX64.fullmatch(before) or before != after:
                raise ValueError('Fallback state or RNG changed')
        if (obs['theta_before_sha256'] != capture_receipt.get('theta_sha256') or
                obs['full_state_before_sha256'] != capture_receipt.get('full_state_before_sha256') or
                obs['rng_before_sha256'] != capture_receipt.get('rng_before_sha256') or
                obs.get('training_flags_before') != obs.get('training_flags_after') or
                obs.get('training_flags_before') != capture_receipt.get('training_flags_before')):
            raise ValueError('Fallback starting state differs from captured fused state')
        fallback_names = {'fallback_attention_vjp': 'FALLBACK_ATTENTION_VJP.safetensors',
                          'ordinary_bf16_attention_vjp': 'ORDINARY_BF16_ATTENTION_VJP.safetensors',
                          'fallback_lora_sample': 'FALLBACK_LORA_SAMPLE.safetensors',
                          'full_state_before': 'FULL_STATE_BEFORE.json', 'full_state_after': 'FULL_STATE_AFTER.json',
                          'policy_receipt': 'FALLBACK_POLICY.json'}
        fallback_files = {key: artifact(obs.get(key), run) for key in fallback_names}
        if any(fallback_files[key] != run / name for key, name in fallback_names.items()):
            raise ValueError('Fallback evidence filenames differ')
        state_a = strict_json(fallback_files['full_state_before'].read_bytes())
        state_b = strict_json(fallback_files['full_state_after'].read_bytes())
        if (state_a.get('sha256') != obs['full_state_before_sha256'] or
                state_b.get('sha256') != obs['full_state_after_sha256'] or
                type(state_a.get('inventory')) is not list or not state_a['inventory'] or
                state_a.get('leaf_count') != len(state_a['inventory']) or
                state_a.get('inventory') != state_b.get('inventory') or
                state_a.get('all_parameters_included') is not True or state_b.get('all_parameters_included') is not True):
            raise ValueError('Fallback full enumerable state record differs')
        policy = strict_json(fallback_files['policy_receipt'].read_bytes())
        identity = strict_json(local(MONITOR + '/' + spec['owner'] + '/WORKER_IDENTITY.json').read_bytes())
        if (policy.get('schema'), policy.get('policy'), policy.get('owned_process_pid'), policy.get('off_scope_calls'),
                policy.get('restoration_verified'), policy.get('native_base_and_fast_untouched'),
                policy.get('environment_unchanged'), policy.get('installed_files_modified'), policy.get('environment')) != (
                'research4.scoped-block3-sdpa-policy.v1', 'force_fused', identity['pid'], 0, True, True, True, False,
                {'MLX_SDPA_VJP_FALLBACK': '1', 'MLX_ENABLE_TF32': None, 'MLX_METAL_GPU_ARCH': None}):
            raise ValueError('Fallback policy/process/restoration evidence differs')
        calls = policy.get('calls')
        if (type(calls) is not list or len(calls) != policy.get('matched_calls') or
                any(sum(call.get('phase') == phase for call in calls) < 16 for phase in ('fallback_direct', 'fallback_ad'))):
            raise ValueError('Fallback direct/AD policy call coverage differs')
        for call in calls:
            if (call.get('phase') not in ('fallback_direct', 'fallback_ad') or
                    (call.get('q_shape'), call.get('k_shape'), call.get('v_shape')) !=
                    ([1, 24, 640, 256], [1, 4, 640, 256], [1, 4, 640, 256]) or
                    not isinstance(call.get('dtype'), str) or not call['dtype'].endswith('bfloat16') or
                    call.get('cache') is not None or call.get('mask') != 'causal' or call.get('sinks') is not None or
                    call.get('scale') != .0625 or call.get('force_fused_requested') is not True):
                raise ValueError('Fallback policy call left the approved Qwen scope')
        primal = obs.get('direct_vs_ad')
        if type(primal) is not dict or primal.get('target_tokens') != 30:
            raise ValueError('Fallback direct/AD scalar control missing')
        a, b = primal.get('reference_mean_nll'), primal.get('selected_mean_nll')
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (a, b)):
            raise ValueError('Fallback scalar loss is nonfinite')
        difference, allowed = abs(a - b), 0.0001 + 0.0001 * abs(a)
        if (primal.get('absolute_difference') != difference or primal.get('allowed_difference') != allowed or
                primal.get('passed') is not (difference <= allowed)):
            raise ValueError('Fallback scalar original-tolerance result differs')
        attention = obs.get('attention_vjp_comparison')
        if (obs.get('primary_attention_reference') != 'ordinary_ops_bf16' or
                type(attention) is not dict or set(attention) != {'dq', 'dk', 'dv'}):
            raise ValueError('Fallback attention VJP inventory differs')
        row_sums = obs.get('ordinary_bf16_row_sums')
        if (type(row_sums) is not dict or row_sums.get('method') != 'ordinary_ops_bf16' or
                row_sums.get('rows') != 24 * 640 or row_sums.get('fused_internal_row_sums') != 'not_directly_observed' or
                any(type(row_sums.get(k)) not in (int, float) or not math.isfinite(row_sums[k]) for k in ('minimum', 'maximum')) or
                row_sums['minimum'] > row_sums['maximum']):
            raise ValueError('Ordinary BF16 reference row-sum evidence differs')
        for item in attention.values():
            if type(item) is not dict or set(item) != {'fused_vs_f64', 'fallback_vs_f64', 'fallback_vs_fused',
                    'ordinary_bf16_vs_f64', 'fused_vs_ordinary_bf16', 'fallback_vs_ordinary_bf16'}:
                raise ValueError('Fallback attention comparison is incomplete')
            for comparison in item.values(): validate_numeric_comparison(comparison)
        lora = obs.get('lora_gradient_sample_vs_fused'); names = capture_receipt.get('lora_sample_names')
        if type(lora) is not dict or type(names) is not list or len(names) != 5 or set(lora) != set(names):
            raise ValueError('Fallback fixed LoRA gradient sample inventory differs')
        for comparison in lora.values(): validate_numeric_comparison(comparison)
        toy = obs.get('synthetic_switch_control')
        if type(toy) is not dict or toy.get('status') not in ('observed', 'inconclusive'):
            raise ValueError('Synthetic switch-control outcome missing')
        if toy['status'] == 'observed':
            components = toy.get('components')
            if type(components) is not dict or set(components) != {'dq', 'dk', 'dv'}:
                raise ValueError('Synthetic switch-control components differ')
            for item in components.values():
                if type(item) is not dict or set(item) != {'fused_vs_f64', 'fallback_vs_f64'}:
                    raise ValueError('Synthetic fused/fallback comparisons missing')
                for comparison in item.values(): validate_numeric_comparison(comparison)
        if obs.get('synthetic_fallback_vjp') is not None:
            if artifact(obs['synthetic_fallback_vjp'], run) != run / 'SYNTHETIC_FALLBACK_VJP.safetensors':
                raise ValueError('Unexpected synthetic fallback artifact path')
        elif toy['status'] == 'observed':
            raise ValueError('Observed synthetic comparison needs its saved fallback VJP')
    return value


def execute(spec_path, spec_sha, review_path, review_sha):
    if sys.platform != 'darwin' or os.uname().machine != 'arm64': raise RuntimeError('Mini Apple silicon required')
    import psutil
    spec_path, review_path = Path(spec_path).absolute(), Path(review_path).absolute()
    for p in (spec_path, review_path):
        if not p.resolve().is_relative_to((ROOT / '.local').resolve()) or any(q.is_symlink() for q in (p, *p.parents)):
            raise ValueError('Exact canonical reviewed local input required')
    spec = verify_review(spec_path, spec_sha, review_path, review_sha)
    base = get_base(); base.KNOWN = (*base.KNOWN, 'all64_sdpa_vjp_probe_v1_20261119/capture_fused_v2.py',
                                     'all64_sdpa_vjp_probe_v1_20261119/fallback_reference.py')
    sys.path.insert(0, str(ROOT / 'mac_studio'))
    own_spec = importlib.util.spec_from_file_location('_vjp_owned_process_v2', local(OWNED_V2))
    owned = importlib.util.module_from_spec(own_spec); own_spec.loader.exec_module(owned)
    environment_observed = check_environment(os.environ)
    owner = spec['owner']; start = time.time(); mono = time.monotonic(); deadline = spec['deadline_epoch']
    if not start < deadline <= start + spec['max_wall_seconds']: raise ValueError('Original diagnostic deadline exceeds bound')
    monotonic_deadline = mono + deadline - start
    run = local(RUNS + '/' + owner); monitor = local(MONITOR + '/' + owner); lock = local(LOCK)
    for p in (run.parent, monitor.parent, lock.parent): p.mkdir(parents=True, exist_ok=True)
    run.mkdir(); monitor.mkdir(); slot = owned.LaunchSlot(); lock_owned = False; clean = False
    error = None; observed = None; code = None; target = None; exit_race = None
    once(monitor / 'START.json', {'owner': owner, 'supervisor_pid': os.getpid(), 'spec_sha256': spec_sha,
        'review_sha256': review_sha, 'deadline_epoch': deadline, 'adam_updates_authorized': 0, 'automatic_retry': False,
        'inherited_mlx_environment': environment_observed,
        'child_fallback_setting': 'unset' if spec['phase'] == 'capture_fused' else '1',
        'tf32_note': 'Unset is recorded; actual backend default is not inferred.'})
    try:
        guard = base.guard_runtime()
        base.check_sample(guard, {'pid': os.getpid(), 'create_time_epoch': psutil.Process(os.getpid()).create_time()},
                          'launch', monitor / 'GUARD.jsonl')
        lock.mkdir(); lock_owned = True
        once(lock / 'owner.json', {'owner': owner, 'pid': os.getpid(), 'kind': KIND})
        visibility = base.same_user_known_worker_scan(psutil, os.getpid(), os.getuid())
        worker = CAPTURE_WORKER if spec['phase'] == 'capture_fused' else FALLBACK_WORKER
        authority = {'schema': 'research4.all64-sdpa-vjp-probe-authority.v1', 'owner': owner,
            'supervisor_pid': os.getpid(), 'deadline_epoch': deadline, 'spec_sha256': spec_sha,
            'worker_sha256': spec['source_files'][worker], 'phase': spec['phase'], 'rank': 4,
            'manifest_sha256': MANIFEST_SHA}
        once(run / 'worker_authority.json', authority)
        # env executes the interpreter in the same owned process, before Python
        # can import MLX. The fallback selector is never toggled in-process.
        environment = ['/usr/bin/env', '-u', 'MLX_SDPA_VJP_FALLBACK'] if spec['phase'] == 'capture_fused' else [
            '/usr/bin/env', 'MLX_SDPA_VJP_FALLBACK=1']
        command = environment + [str(ROOT / PYTHON), '-B', '-u', str(local(worker)), '--phase', spec['phase'], '--run-dir', str(run), '--owner', owner,
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
                sample = guard.sample(target, 'run'); base.append_json(monitor / 'GUARD.jsonl', sample)
                if sample.get('valid') is not True or sample.get('stop_required') is not False:
                    if terminal_exit_sample(sample, target) and owned.peek_exited(slot.process.pid):
                        exit_race = {'schema': 'research4.all64-terminal-guard-exit-race.v1',
                            'worker_pid': target['pid'], 'worker_create_time_epoch': target['create_time_epoch'],
                            'sample_sha256': sample_hash(sample), 'waitid_exited_after_sample': True,
                            'invalid_sample_treated_as_healthy': False, 'worker_allowed_to_continue': False}
                        once(monitor / 'EXIT_DURING_GUARD.json', exit_race)
                        break
                    raise RuntimeError('Native memory guard stopped or failed closed')
                time.sleep(1.0)
            code, descendants, pending = owned.cleanup(slot.process); clean = True
        if descendants or pending: raise RuntimeError('Diagnostic process-group cleanup uncertain')
        # Check source and guard identity again, even if the worker recorded a mismatch.
        validate_spec(spec, now=min(time.time(), deadline - 1e-6))
        verify_guard_evidence(monitor / 'GUARD.jsonl', target['pid'], target['create_time_epoch'], exit_race)
        if code != 0: raise RuntimeError('Probe worker did not complete cleanly')
        receipt_path = run / 'worker_receipt.json'
        receipt = validate_probe_receipt(strict_json(receipt_path.read_bytes()), spec, run)
        status = 'CAPTURED_NOT_QUALIFIED' if spec['phase'] == 'capture_fused' else 'OBSERVED_NOT_QUALIFICATION'
        observed = {'schema': 'research4.all64-sdpa-vjp-probe-supervisor-closeout.v1',
            'status': status, 'phase': spec['phase'], 'owner': owner, 'worker_exit_code': code,
            'worker_receipt': {'path': receipt_path.relative_to(ROOT).as_posix(), 'sha256': digest(receipt_path)},
            'worker_identity': {'path': (monitor / 'WORKER_IDENTITY.json').relative_to(ROOT).as_posix(),
                                'sha256': digest(monitor / 'WORKER_IDENTITY.json')},
            'spec': {'path': spec_path.relative_to(ROOT).as_posix(), 'sha256': spec_sha},
            'review': {'path': review_path.relative_to(ROOT).as_posix(), 'sha256': review_sha},
            'adam_updates': 0, 'qualification_claim': False, 'automatic_retry': False}
        for name, key in [('FULL_STATE_BEFORE.json', 'full_state_before'), ('FULL_STATE_AFTER.json', 'full_state_after')]:
            path = run / name
            observed[key] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)}
        if exit_race is not None:
            marker = monitor / 'EXIT_DURING_GUARD.json'
            observed['guard_exit_race'] = {'path': marker.relative_to(ROOT).as_posix(), 'sha256': digest(marker)}

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
        release = {'schema': 'research4.all64-sdpa-vjp-probe-supervisor-release.v1', 'owner': owner,
            'worker_started': slot.process is not None, 'worker_exit_code': code, 'owned_cleanup_confirmed': clean,
            'lease_released': bool(lock_owned and released), 'lease_retained_uncertain': bool(lock_owned and not released),
            'error': error, 'finished_epoch': time.time()}
        once(monitor / 'RELEASE.json', release)
        if observed is not None and error is None and release['lease_released']:
            observed['release'] = {'path': (monitor / 'RELEASE.json').relative_to(ROOT).as_posix(),
                                   'sha256': digest(monitor / 'RELEASE.json')}
            observed['guard_evidence'] = {'path': (monitor / 'GUARD.jsonl').relative_to(ROOT).as_posix(),
                                          'sha256': digest(monitor / 'GUARD.jsonl')}
            once(monitor / 'PROBE_CLOSEOUT.json', observed)
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
