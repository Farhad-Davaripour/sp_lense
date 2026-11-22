"""One-use all-64 supervisor with forced forward and pre-import fallback VJP.

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
WORKER = 'mac_studio/all64_broader_fallback_actualwidth_v7_20261119/worker.py'
DEV_WORKER = 'mac_studio/all64_broader_fallback_actualwidth_v8_20261120/eval_worker.py'
SUPERVISOR = 'mac_studio/all64_broader_fallback_actualwidth_supervision_v8_20261120/supervisor.py'
PREVIOUS_SUPERVISOR = 'mac_studio/all64_broader_fallback_actualwidth_supervision_v7_20261119/supervisor.py'
PREVIOUS_SUPERVISOR_SHA = 'd9152acca545747592d410a73182d1d807e7536402da607e5884bb10a58a6cee'
QUAL_SUPERVISOR = 'mac_studio/all64_broader_fallback_actualwidth_supervision_v6_20261119/supervisor.py'
QUAL_SUPERVISOR_SHA = '609e380a213d1bd5cc24a34f911564799e466dc4c8d5589cfd923793cced985a'
LEGACY_SUPERVISOR = 'mac_studio/all64_broader_fallback_supervision_v5_20261119/supervisor.py'
LEGACY_SUPERVISOR_SHA = 'ead05a1e71117dfee2b450180421b9ed97cc668b71f1e5149003070b90bb4cda'
PRIOR_RANK8_AUDIT = '.local/reviews/all64_broader_v2_20261119/RANK8_RESOURCE_STOP_INDEPENDENT_AUDIT_790d023ba6c747e3ba8c9960212e7384.json'
PRIOR_RANK8_AUDIT_SHA = 'b9d8b647d13cc8865b349acb22a157afcc3928b607b21ec4ea0c43c8a260892c'
OWNED_PROCESS = 'mac_studio/all64_broader_fallback_supervision_v5_20261119/owned_process_v3.py'
RANK4_OWNER = '2962db4b6e2047e182df48e2716170f9'
RANK4_AUDIT = '.local/reviews/all64_broader_v2_20261119/RANK4_FALLBACK_RECOVERED_AUDIT_' + RANK4_OWNER + '.json'
RANK4_AUDIT_SHA = '9e7bda8d6e9f25a060224304c138cfa6dc96f8fe3e97c8f1a61eb42b27bbd27a'
RANK4_SUPERVISOR = 'mac_studio/all64_broader_fallback_supervision_v4_20261119/supervisor.py'
RANK4_SUPERVISOR_SHA = '8e808b46d75eb371e224ee3542108ef3eabd13241b128c393e09dccc2e3a8c26'
RANK4_WORKER = 'mac_studio/all64_broader_fallback_v4_20261119/worker.py'
RANK4_WORKER_SHA = '248b3d82396a0777c6b813b87f69d7007c08d260bc738370b59ddc04d5ec55af'
WIDTH_SUPERVISOR = 'mac_studio/all64_force_fused_multiwidth_supervision_v4_20261119/supervisor.py'
WIDTH_SUPERVISOR_SHA = '22be0abb4b3773384c17a53534949afc9c39a8d7894c671eb514885993aeeb56'
FORCED_SCOPE = 'mac_studio/all64_force_fused_widths_v4_20261119/scoped_sdpa_widths.py'
FORCED_SCOPE_SHA = 'b84a79c806980408e9cfdd30794093baf6b40e0cb5c1c2882bf08243b7baf3bf'
FALLBACK_SCOPE = 'mac_studio/all64_sdpa_vjp_probe_v1_20261119/scoped_fallback.py'
FALLBACK_SCOPE_SHA = '78d74ab14d664dda2355fa25a97ead831053ed5ee4c4550e2a2a4ce398ea55b8'
FALLBACK_SUPERVISOR = 'mac_studio/all64_sdpa_vjp_supervision_v3_20261119/supervisor.py'
FALLBACK_SUPERVISOR_SHA = '38846f7a07e1ab0c8fa51abf712e27ef824ce946a79860f4b679e06b60be9829'
FALLBACK_AUDIT = '.local/reviews/all64_broader_v2_20261119/SDPA_FALLBACK_INDEPENDENT_AUDIT_5062ff2d286e406ca6861d665b483d31.json'
FALLBACK_AUDIT_SHA = 'be331eec5315ac6d01877c9ae7c1f842ae20839d1a2c6fb4ef871a5fda18eb84'
FALLBACK_OWNER = '5062ff2d286e406ca6861d665b483d31'
FALLBACK_RUN = '.local/experiments/all64_sdpa_vjp_probe_v1_20261119/runs/' + FALLBACK_OWNER
FALLBACK_MONITOR = '.local/monitor/all64_sdpa_vjp_supervision_v1_20261119/' + FALLBACK_OWNER
FALLBACK_CLOSEOUT_SHA = 'cfa171eea84c6ae5906e3d06920a82e68ed14ce8fbaef7909a10d9a49bd882a4'
WIDTH_OWNER = 'bdfd9635516d4198bc34a9cd2837f16e'
WIDTH_RUN = '.local/experiments/all64_force_fused_widths_v4_20261119/runs/' + WIDTH_OWNER
WIDTH_MONITOR = '.local/monitor/all64_force_fused_multiwidth_supervision_v4_20261119/' + WIDTH_OWNER
WIDTH_SPEC = '.local/experiments/all64_force_fused_widths_v4_20261119/authorities/' + WIDTH_OWNER + '/SPEC.json'
WIDTH_SPEC_SHA = '83d2c76077adb9f9f67b87ffe8e8413d796b4a1e352bf5828e25131b2b5b2723'
WIDTH_REVIEW = '.local/reviews/all64_broader_v2_20261119/RANK4_FORCE_FUSED_WIDTHS_REVIEW_' + WIDTH_OWNER + '.json'
WIDTH_REVIEW_SHA = '5c24f09bac25732751897f38288a4871349f4b76fa3da55328719ae13725a6fb'
WIDTH_AUDIT = '.local/reviews/all64_broader_v2_20261119/V4_RECOVERED_INDEPENDENT_AUDIT_' + WIDTH_OWNER + '.json'
WIDTH_AUDIT_SHA = '3023e734d2879ef50425ed422a4b85ecf9f3d961427a3d6b780db0bae6e76969'
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
KNOWN = ('all64_broader_v2_20261119/worker.py', 'all64_broader_forced_v3_20261119/worker.py',
         'all64_broader_fallback_v4_20261119/worker.py', 'all64_broader_fallback_v5_20261119/worker.py',
         'all64_broader_fallback_actualwidth_v6_20261119/worker.py',
         'all64_broader_fallback_actualwidth_v7_20261119/worker.py',
         'all64_broader_fallback_actualwidth_v7_20261119/eval_worker.py',
         'all64_broader_fallback_actualwidth_v8_20261120/eval_worker.py',
         'game_all_layers_worker.py', 'game_training_worker.py',
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


def validate_widths_evidence(ref):
    """Replay audited five-width results without inventing a clean worker exit."""
    if (type(ref) is not dict or set(ref) != {'path', 'sha256'} or ref['path'] != WIDTH_AUDIT or
            ref['sha256'] != WIDTH_AUDIT_SHA or digest(project_path(WIDTH_AUDIT)) != WIDTH_AUDIT_SHA):
        raise ValueError('Exact independently audited recovered five-width evidence required')
    audit = strict_json(project_path(WIDTH_AUDIT).read_bytes()); widths = [288, 544, 576, 608, 640]
    if (audit.get('schema'), audit.get('status'), audit.get('owner'), audit.get('adam_updates'),
            audit.get('qualification_claim'), audit.get('automatic_retry')) != (
            'research4.all64-v4-recovered-independent-audit.v1',
            'FIVE_SCALARS_MATCH_PHYSICAL_RELEASE_RECOVERED_SUPERVISOR_UNCERTAIN', WIDTH_OWNER, 0, False, False):
        raise ValueError('Recovered five-width audit identity differs')
    if audit.get('worker_exit_status') != 'unknown; no supervisor success closeout; original RELEASE preserved uncertain':
        raise ValueError('Prior worker exit must remain explicitly unknown')
    artifacts = audit.get('mini_artifacts')
    expected_keys = {'start', 'worker_identity', 'guard', 'original_uncertain_release', 'worker_receipt',
        'policy', 'full_state_before', 'full_state_after', 'spec', 'review', 'recovery', 'quarantined_owner'}
    expected_keys |= {f'width_{w}_{name}' for w in widths for name in ('WIDTH_OBSERVATION.json', 'PLAIN_FORWARD.json')}
    if type(artifacts) is not dict or set(artifacts) != expected_keys:
        raise ValueError('Complete posthoc five-width artifact inventory required')
    for item in artifacts.values():
        if (type(item) is not dict or set(item) != {'path', 'sha256'} or
                type(item['sha256']) is not str or not HEX64.fullmatch(item['sha256']) or
                digest(project_path(item['path'])) != item['sha256']):
            raise ValueError('Posthoc five-width artifact changed')
    monitor, run = project_path(WIDTH_MONITOR), project_path(WIDTH_RUN)
    exact = {'start': monitor / 'START.json', 'worker_identity': monitor / 'WORKER_IDENTITY.json',
        'guard': monitor / 'GUARD.jsonl', 'original_uncertain_release': monitor / 'RELEASE.json',
        'worker_receipt': run / 'worker_receipt.json', 'policy': run / 'FORCE_POLICY.json',
        'full_state_before': run / 'FULL_STATE_BEFORE.json', 'full_state_after': run / 'FULL_STATE_AFTER.json',
        'spec': project_path(WIDTH_SPEC), 'review': project_path(WIDTH_REVIEW)}
    for width in widths:
        for name in ('WIDTH_OBSERVATION.json', 'PLAIN_FORWARD.json'):
            exact[f'width_{width}_{name}'] = run / ('width_' + str(width)) / name
    for key, expected in exact.items():
        if project_path(artifacts[key]['path']) != expected: raise ValueError('Unexpected five-width artifact path')
    start = strict_json((monitor / 'START.json').read_bytes())
    identity = strict_json((monitor / 'WORKER_IDENTITY.json').read_bytes())
    release = strict_json((monitor / 'RELEASE.json').read_bytes())
    recovery = strict_json(project_path(artifacts['recovery']['path']).read_bytes())
    if (start.get('owner'), start.get('spec_sha256'), start.get('review_sha256'), identity.get('owner'),
            release.get('owner'), release.get('worker_exit_code'), release.get('owned_cleanup_confirmed'),
            release.get('lease_released'), release.get('lease_retained_uncertain'),
            recovery.get('schema'), recovery.get('status'), recovery.get('owner'), recovery.get('original_release_modified'),
            recovery.get('original_release_sha256')) != (WIDTH_OWNER, WIDTH_SPEC_SHA, WIDTH_REVIEW_SHA, WIDTH_OWNER,
            WIDTH_OWNER, None, False, False, True, 'research4.v4-stale-lease-recovery.v1',
            'PASS_EXACT_LEASE_QUARANTINED', WIDTH_OWNER, False, artifacts['original_uncertain_release']['sha256']):
        raise ValueError('Original uncertain lifecycle or separate recovery differs')
    physical = audit.get('physical_release_now', {})
    if (physical.get('canonical_lock_absent'), physical.get('worker_absent'), physical.get('supervisor_absent'),
            physical.get('group_members'), physical.get('group_unknown_count'), physical.get('port_connect_ex')) != (
            True, True, True, [], 0, 61):
        raise ValueError('Independent exact-owner physical release not established')
    samples = [strict_json(line) for line in (monitor / 'GUARD.jsonl').read_bytes().splitlines() if line.strip()]
    if (len(samples) != 173 or audit.get('valid_normal_guard_samples_before_exit_race') != 172 or
            audit.get('invalid_terminal_guard_reason') != ['rusage reports exited process']):
        raise ValueError('Prior native guard event inventory differs')
    for sample in samples[:-1]:
        if (sample.get('valid') is not True or sample.get('stop_required') is not False or
                sample.get('assessment', {}).get('pressure_flag_distinct_from_bytes') != 1):
            raise ValueError('A pre-terminal native guard sample was unhealthy')
    if not terminal_exit_sample(samples[-1], {'pid': identity['pid'], 'create_time_epoch': identity['create_time_epoch']}):
        raise ValueError('Final invalid guard sample does not have the observed exited-target reason')
    original_spec = strict_json(project_path(WIDTH_SPEC).read_bytes())
    original_review = strict_json(project_path(WIDTH_REVIEW).read_bytes())
    if (digest(project_path(WIDTH_SPEC)) != WIDTH_SPEC_SHA or digest(project_path(WIDTH_REVIEW)) != WIDTH_REVIEW_SHA or
            original_review.get('schema') != 'research4.all64-force-fused-widths-supervisor-review.v1' or
            original_review.get('status') != 'PASS' or original_review.get('spec_sha256') != WIDTH_SPEC_SHA or
            original_review.get('supervisor_sha256') != WIDTH_SUPERVISOR_SHA or original_review.get('model_execution') is not False):
        raise ValueError('Five-width source review does not bind original spec')
    module_spec = importlib.util.spec_from_file_location('_all64_five_width_review', project_path(WIDTH_SUPERVISOR))
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    module.validate_spec(original_spec, now=min(time.time(), original_spec['deadline_epoch'] - 1e-6))
    receipt = strict_json((run / 'worker_receipt.json').read_bytes())
    fixture = strict_json(project_path(module.FIXTURE).read_bytes())['cases']; results = receipt.get('case_results')
    if (receipt.get('owner'), receipt.get('observed_widths'), receipt.get('all_widths_matched'),
            receipt.get('adam_updates'), receipt.get('qualification_claim')) != (WIDTH_OWNER, widths, True, 0, False):
        raise ValueError('Five-width worker receipt differs')
    if type(results) is not list or len(results) != 5: raise ValueError('Five saved case records required')
    observations = []
    for expected, item in zip(fixture, results):
        width = expected['model_input_width']; folder = run / ('width_' + str(width))
        obs_path, plain_path = folder / 'WIDTH_OBSERVATION.json', folder / 'PLAIN_FORWARD.json'
        if (type(item) is not dict or item.get('width') != width or item.get('observation_sha256') != digest(obs_path) or
                item.get('plain_forward_sha256') != digest(plain_path)):
            raise ValueError('Five-width saved case bytes differ')
        observation = module.validate_observation(strict_json(obs_path.read_bytes()), expected)
        baseline = strict_json(plain_path.read_bytes())
        if observation.get('status') != 'MATCH' or any(observation.get(k) != v for k, v in baseline.items()):
            raise ValueError('Five-width saved case did not match')
        observations.append(observation)
    module.validate_cross_case_state(observations)
    module.validate_policy_and_state(receipt, run, identity['pid'])
    return audit


def validate_fallback_evidence(ref):
    """Replay the single-case BF16-primary fallback probe without overstating it."""
    expected_path = FALLBACK_MONITOR + '/PROBE_CLOSEOUT.json'
    if (type(ref) is not dict or set(ref) != {'path', 'sha256'} or ref['path'] != expected_path or
            ref['sha256'] != FALLBACK_CLOSEOUT_SHA or digest(project_path(expected_path)) != FALLBACK_CLOSEOUT_SHA or
            digest(project_path(FALLBACK_AUDIT)) != FALLBACK_AUDIT_SHA):
        raise ValueError('Exact saved fallback probe and independent audit required')
    closeout = strict_json(project_path(expected_path).read_bytes())
    audit = strict_json(project_path(FALLBACK_AUDIT).read_bytes())
    if (closeout.get('schema'), closeout.get('status'), closeout.get('phase'), closeout.get('owner'),
            closeout.get('worker_exit_code'), closeout.get('adam_updates'), closeout.get('qualification_claim'),
            audit.get('schema'), audit.get('status'), audit.get('worker_absent'), audit.get('supervisor_absent'),
            audit.get('group'), audit.get('group_unknown_count'), audit.get('lock_absent'), audit.get('port_connect_ex')) != (
            'research4.all64-sdpa-vjp-probe-supervisor-closeout.v1', 'OBSERVED_NOT_QUALIFICATION', 'fallback_reference',
            FALLBACK_OWNER, 0, 0, False, 'research4.all64-sdpa-fallback-independent-audit.v1',
            'PASS_SAVED_OBSERVATION_NOT_QUALIFICATION', True, True, [], 0, True, 61):
        raise ValueError('Fallback probe or independent physical-release scope differs')
    monitor, run = project_path(FALLBACK_MONITOR), project_path(FALLBACK_RUN)
    receipt_path = bound_artifact(closeout.get('worker_receipt'), run)
    release_path = bound_artifact(closeout.get('release'), monitor)
    guard_path = bound_artifact(closeout.get('guard_evidence'), monitor)
    identity_path = bound_artifact(closeout.get('worker_identity'), monitor)
    spec_path = bound_artifact(closeout.get('spec'), project_path('.local'))
    review_path = bound_artifact(closeout.get('review'), project_path('.local'))
    if (receipt_path, release_path, guard_path, identity_path) != (
            run / 'worker_receipt.json', monitor / 'RELEASE.json', monitor / 'GUARD.jsonl', monitor / 'WORKER_IDENTITY.json'):
        raise ValueError('Fallback owned evidence paths differ')
    receipt, release = strict_json(receipt_path.read_bytes()), strict_json(release_path.read_bytes())
    identity = strict_json(identity_path.read_bytes()); prior_spec = strict_json(spec_path.read_bytes())
    prior_review = strict_json(review_path.read_bytes())
    if (release.get('owner'), release.get('worker_exit_code'), release.get('owned_cleanup_confirmed'),
            release.get('lease_released'), release.get('error'), identity.get('owner'),
            prior_spec.get('phase'), prior_spec.get('owner'), prior_review.get('schema'),
            prior_review.get('status'), prior_review.get('spec_sha256'), prior_review.get('supervisor_sha256')) != (
            FALLBACK_OWNER, 0, True, True, None, FALLBACK_OWNER, 'fallback_reference', FALLBACK_OWNER,
            'research4.all64-sdpa-vjp-probe-supervisor-review.v1', 'PASS', closeout['spec']['sha256'], FALLBACK_SUPERVISOR_SHA):
        raise ValueError('Fallback prior owner/spec/review/release differs')
    module_spec = importlib.util.spec_from_file_location('_all64_fallback_probe_review', project_path(FALLBACK_SUPERVISOR))
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    marker = None
    if closeout.get('guard_exit_race') is not None:
        marker = strict_json(bound_artifact(closeout['guard_exit_race'], monitor).read_bytes())
    module.verify_guard_evidence(guard_path, identity['pid'], identity['create_time_epoch'], marker)
    module.validate_spec(prior_spec, now=min(time.time(), prior_spec['deadline_epoch'] - 1e-6))
    module.validate_probe_receipt(receipt, prior_spec, run)
    obs_path = bound_artifact(receipt.get('observation'), run); obs = strict_json(obs_path.read_bytes())
    if (audit.get('obs') != obs or audit.get('receipt') != receipt or audit.get('release') != release or
            audit.get('observation_sha256') != digest(obs_path) or audit.get('guard_all_normal') is not True):
        raise ValueError('Independent fallback audit does not match original evidence')
    if (obs.get('primary_attention_reference') != 'ordinary_ops_bf16' or
            obs.get('direct_vs_ad', {}).get('passed') is not True or
            any(obs['attention_vjp_comparison'][name]['fallback_vs_ordinary_bf16'].get('passes_original_strict_gradient_gate') is not True
                for name in ('dq', 'dk', 'dv'))):
        raise ValueError('Fallback direct/AD or single-layer BF16 reference did not pass')
    return audit


def validate_recovered_rank4(ref):
    """Replay rank4 evidence without inventing the missing worker exit code."""
    if (type(ref) is not dict or set(ref) != {'path', 'sha256'} or ref['path'] != RANK4_AUDIT or
            ref['sha256'] != RANK4_AUDIT_SHA or digest(project_path(RANK4_AUDIT)) != RANK4_AUDIT_SHA):
        raise ValueError('Exact independent recovered rank4 audit required')
    audit = strict_json(project_path(RANK4_AUDIT).read_bytes())
    if (audit.get('schema'), audit.get('status'), audit.get('owner'), audit.get('rank'), audit.get('manifest_sha256'),
            audit.get('adam_updates'), audit.get('fallback_requested_before_mlx_import'), audit.get('automatic_retry')) != (
            'research4.all64-rank4-fallback-recovered-independent-audit.v1',
            'FIVE_CASE_NUMERICAL_PASS_PHYSICAL_RELEASE_RECOVERED_SUPERVISOR_UNCERTAIN',
            RANK4_OWNER, 4, MANIFEST, 5, True, False):
        raise ValueError('Recovered rank4 audit identity or numerical scope differs')
    if audit.get('worker_exit_status') != 'UNKNOWN: supervisor did not reap the worker and no ordinary successful closeout exists':
        raise ValueError('Original rank4 worker exit must remain unknown')
    physical = audit.get('physical_release_now', {})
    if (physical.get('canonical_lock_absent'), physical.get('worker_absent'), physical.get('supervisor_absent'),
            physical.get('group'), physical.get('unknown'), physical.get('port')) != (True, True, True, [], 0, 61):
        raise ValueError('Recovered rank4 exact-owner release not established')
    refs = audit.get('artifact_sha256')
    if type(refs) is not dict or len(refs) != 22:
        raise ValueError('Recovered rank4 artifact inventory differs')
    for path, sha in refs.items():
        if type(sha) is not str or not HEX64.fullmatch(sha) or digest(project_path(path)) != sha:
            raise ValueError('Recovered rank4 artifact changed: ' + str(path))
    for key in ('pre_recovery_audit', 'recovery', 'original_uncertain_release', 'qualification_review', 'quarantined_owner'):
        ref_item = audit.get(key)
        if type(ref_item) is not dict or set(ref_item) != {'path', 'sha256'} or digest(project_path(ref_item['path'])) != ref_item['sha256']:
            raise ValueError('Recovered rank4 independent evidence changed')
    release = strict_json(project_path(audit['original_uncertain_release']['path']).read_bytes())
    recovery = strict_json(project_path(audit['recovery']['path']).read_bytes())
    if (release.get('owner'), release.get('owned_cleanup_confirmed'), release.get('lease_released'),
            release.get('lease_retained_uncertain'), recovery.get('owner'), recovery.get('status'),
            recovery.get('original_release_modified')) != (RANK4_OWNER, False, False, True, RANK4_OWNER,
            'PASS_EXACT_LEASE_QUARANTINED', False):
        raise ValueError('Original uncertain release or separate exact-owner recovery differs')
    run = project_path(RUNS + '/' + RANK4_OWNER)
    monitor = project_path(MONITOR + '/' + RANK4_OWNER)
    receipt_path = run / 'worker_receipt.json'; receipt = strict_json(receipt_path.read_bytes())
    if digest(receipt_path) != audit.get('worker_receipt_sha256'):
        raise ValueError('Recovered rank4 worker receipt changed')
    identity = strict_json((monitor / 'WORKER_IDENTITY.json').read_bytes())
    samples = [strict_json(line) for line in (monitor / 'GUARD.jsonl').read_bytes().splitlines() if line.strip()]
    if (len(samples) != 427 or audit.get('guard_prior_normal_count') != 426 or
            audit.get('guard_last_invalid_reason') != ['rusage reports exited process']):
        raise ValueError('Recovered rank4 guard inventory differs')
    for sample in samples[:-1]:
        if (sample.get('valid') is not True or sample.get('stop_required') is not False or
                sample.get('assessment', {}).get('pressure_flag_distinct_from_bytes') != 1):
            raise ValueError('Prior rank4 guard sample was not valid NORMAL')
    if not terminal_exit_sample(samples[-1], {'pid': identity['pid'], 'create_time_epoch': identity['create_time_epoch']}):
        raise ValueError('Last rank4 invalid sample differs from recorded exited-target evidence')
    if digest(project_path(LEGACY_SUPERVISOR)) != LEGACY_SUPERVISOR_SHA:
        raise ValueError('Recovered rank4 validator changed')
    module_spec = importlib.util.spec_from_file_location('_all64_legacy_recovered_validator', project_path(LEGACY_SUPERVISOR))
    legacy = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(legacy)
    legacy.validate_worker_receipt(receipt, {'phase': 'qualification', 'rank': 4, 'owner': RANK4_OWNER})
    if digest(run / 'qualification_adapter.safetensors') != audit.get('qualification_adapter_sha256'):
        raise ValueError('Recovered rank4 actual qualification adapter changed')
    return audit, receipt


def validate_prior_rank8_stop(ref):
    if ref != {'path': PRIOR_RANK8_AUDIT, 'sha256': PRIOR_RANK8_AUDIT_SHA} or digest(project_path(PRIOR_RANK8_AUDIT)) != PRIOR_RANK8_AUDIT_SHA:
        raise ValueError('Exact prior rank8 resource-stop audit required')
    audit = strict_json(project_path(PRIOR_RANK8_AUDIT).read_bytes())
    if (audit.get('schema'), audit.get('status'), audit.get('qualification_passed'), audit.get('automatic_retry')) != (
            'research4.all64-rank8-independent-resource-stop-audit.v1',
            'RESOURCE_WARNING_STOP_INCOMPLETE_RELEASE_VERIFIED', False, False):
        raise ValueError('Prior rank8 stop must remain incomplete')
    refs = audit.get('artifact_sha256')
    if type(refs) is not dict or not refs:
        raise ValueError('Prior rank8 evidence inventory missing')
    for path, sha in refs.items():
        if type(sha) is not str or not HEX64.fullmatch(sha) or digest(project_path(path)) != sha:
            raise ValueError('Prior rank8 evidence changed')
    if ([case.get('width') for case in audit.get('cases', [])] != [288, 544, 576, 608] or
            audit.get('guard_last', {}).get('pressure_state') != 'warning' or
            audit.get('qualification_adapter_exists') is not False or
            audit.get('policy_receipt_exists') is not False):
        raise ValueError('Prior rank8 partial scope differs')
    return audit

def validate_dev_widths(rows):
    probes = [row['dev_complete_tokens'] for row in rows
              if row.get('stage') == 'probe' and 'dev_complete_tokens' in row]
    if (len(probes) != 36 or any(type(length) is not int or length <= 0 or
                                 32 * ((length + 31) // 32) > 608 for length in probes)):
        raise ValueError('Frozen DEV probes exceed the actual-width scope')


def validate_spec(spec, now=None):
    keys = {'schema', 'owner', 'phase', 'rank', 'epoch', 'checkpoint', 'manifest_sha256', 'deadline_epoch',
            'max_wall_seconds', 'python_binary_sha256', 'source_files', 'prerequisites', 'forced_widths_evidence',
            'fallback_probe_evidence', 'recovered_rank4_evidence', 'prior_rank8_stop_evidence'}
    if type(spec) is not dict or set(spec) != keys or spec['schema'] != 'research4.all64-broader-fallback-actualwidth-hpo-supervision-spec.v1':
        raise ValueError('Exact supervision spec required')
    if type(spec['owner']) is not str or not HEX32.fullmatch(spec['owner']):
        raise ValueError('Unique 32-hex owner required')
    phase, rank = spec['phase'], spec['rank']
    if phase not in ('trajectory', 'dev') or type(rank) is not int or rank not in (4, 8, 16):
        raise ValueError('Only actual-width trajectory or DEV stages are prepared')
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
    if deadline > (time.time() if now is None else now) + maximum:
        raise ValueError('Absolute deadline exceeds declared stage wall bound')
    python = (PROJECT / PYTHON).resolve(strict=True)
    if ('onedrive' in str(python).lower() or not HEX64.fullmatch(spec['python_binary_sha256'])
            or digest(python) != spec['python_binary_sha256']):
        raise ValueError('Mini virtual-environment Python identity differs')
    sources = spec['source_files']
    program = DEV_WORKER if phase == 'dev' else WORKER
    if type(sources) is not dict or not {program, WORKER, SUPERVISOR, OWNED_PROCESS, QUAL_SUPERVISOR, LEGACY_SUPERVISOR, PRIOR_RANK8_AUDIT, RANK4_AUDIT, RANK4_SUPERVISOR, RANK4_WORKER,
                                         WIDTH_SUPERVISOR, WIDTH_SPEC, WIDTH_REVIEW, WIDTH_AUDIT, FORCED_SCOPE,
                                         FALLBACK_SCOPE, FALLBACK_SUPERVISOR, FALLBACK_AUDIT,
                                         FROZEN, GUARD, GUARD_BINDING,
                                         'mac_studio/game_all_layers_layout.py',
                                         'mac_studio/game_selected_head_v3_compare.py',
                                         'mac_studio/owned_validation_process.py'}.issubset(sources):
        raise ValueError('Source pins incomplete')
    for name, expected in sources.items():
        if type(expected) is not str or not HEX64.fullmatch(expected) or digest(project_path(name)) != expected:
            raise ValueError('Pinned source changed: ' + str(name))
    if sources[QUAL_SUPERVISOR] != QUAL_SUPERVISOR_SHA:
        raise ValueError('Pinned qualification supervisor differs')
    if sources.get(PREVIOUS_SUPERVISOR) != PREVIOUS_SUPERVISOR_SHA:
        raise ValueError('Pinned previous HPO supervisor differs')
    if sources[LEGACY_SUPERVISOR] != LEGACY_SUPERVISOR_SHA or sources[PRIOR_RANK8_AUDIT] != PRIOR_RANK8_AUDIT_SHA:
        raise ValueError('Prior rank8 audit or recovered-rank4 validator differs')
    validate_prior_rank8_stop(spec['prior_rank8_stop_evidence'])
    if sources[FROZEN] != MANIFEST or sources[GUARD_BINDING] != GUARD_BINDING_SHA:
        raise ValueError('Frozen dataset or guard binding differs')
    if (sources[WIDTH_SUPERVISOR], sources[WIDTH_SPEC], sources[WIDTH_REVIEW], sources[WIDTH_AUDIT], sources[FORCED_SCOPE]) != (
            WIDTH_SUPERVISOR_SHA, WIDTH_SPEC_SHA, WIDTH_REVIEW_SHA, WIDTH_AUDIT_SHA, FORCED_SCOPE_SHA):
        raise ValueError('Reviewed multiwidth diagnostic source/spec/review differ')
    if (sources[FALLBACK_SCOPE], sources[FALLBACK_SUPERVISOR], sources[FALLBACK_AUDIT]) != (
            FALLBACK_SCOPE_SHA, FALLBACK_SUPERVISOR_SHA, FALLBACK_AUDIT_SHA):
        raise ValueError('Reviewed fallback scope/supervisor/audit differ')
    if (sources[RANK4_AUDIT], sources[RANK4_SUPERVISOR], sources[RANK4_WORKER]) != (
            RANK4_AUDIT_SHA, RANK4_SUPERVISOR_SHA, RANK4_WORKER_SHA):
        raise ValueError('Recovered rank4 source/audit differs')
    original_scope, fallback_scope = project_path(FORCED_SCOPE).read_bytes(), project_path(FALLBACK_SCOPE).read_bytes()
    old = b"expected_environment['MLX_SDPA_VJP_FALLBACK'] not in (None, '0')"
    new = b"expected_environment['MLX_SDPA_VJP_FALLBACK'] != '1'"
    if original_scope.count(old) != 1 or original_scope.replace(old, new) != fallback_scope:
        raise ValueError('Fallback scope must differ only by the explicitly approved environment guard')
    validate_widths_evidence(spec['forced_widths_evidence'])
    validate_fallback_evidence(spec['fallback_probe_evidence'])
    if spec['recovered_rank4_evidence'] != {'path': RANK4_AUDIT, 'sha256': RANK4_AUDIT_SHA}:
        raise ValueError('Exact recovered rank4 audit required')
    if digest(project_path(TOKEN_LENGTHS)) != TOKEN_LENGTHS_SHA:
        raise ValueError('Frozen sequence length inventory differs')
    if phase == 'dev' and ('mac_studio/all64_broader_v2_20261119/dev_eval.py' not in sources or
                           sources.get(SELECTOR) != SELECTOR_SHA or digest(project_path(DEV_MASKS)) != DEV_MASKS_SHA):
        raise ValueError('Frozen DEV selector or masks differ')
    if phase == 'dev':
        lengths = strict_json(project_path(TOKEN_LENGTHS).read_bytes())['rows']
        validate_dev_widths(lengths)
    prereqs = spec['prerequisites']
    if type(prereqs) is not dict or set(prereqs) != required_prerequisites(phase, rank):
        raise ValueError('Exact successful lower-rank qualification prerequisites required')
    cache = {}
    for key, ref in prereqs.items():
        if key == '4':
            if ref != spec['recovered_rank4_evidence']:
                raise ValueError('Rank4 prerequisite must be the independently recovered audit')
            validate_recovered_rank4(ref)
            continue
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


def validate_producer_supervisor(prior_sources, prior_review):
    """Accept only the pinned v7 producer or this exact v8 producer."""
    if type(prior_sources) is not dict:
        raise ValueError('Completed-stage reviewed source inventory differs')
    if SUPERVISOR in prior_sources:
        expected = digest(project_path(SUPERVISOR))
        source_matches = prior_sources.get(SUPERVISOR) == expected
    else:
        expected = PREVIOUS_SUPERVISOR_SHA
        source_matches = prior_sources.get(PREVIOUS_SUPERVISOR) == expected
    if not source_matches or prior_review.get('supervisor_sha256') != expected:
        raise ValueError('Completed-stage reviewed supervisor source differs')


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
    if phase == 'qualification' and rank in (8, 16):
        if digest(project_path(QUAL_SUPERVISOR)) != QUAL_SUPERVISOR_SHA:
            raise ValueError('Pinned qualification validator changed')
        loader = importlib.util.spec_from_file_location('_all64_actualwidth_qual_validator', project_path(QUAL_SUPERVISOR))
        qualification = importlib.util.module_from_spec(loader); loader.loader.exec_module(qualification)
        return qualification.validate_completed_closeout(ref, phase, rank, epoch)
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
    marker = None
    if value.get('guard_exit_race') is not None:
        marker_path = bound_artifact(value['guard_exit_race'], monitor)
        if marker_path != monitor / 'EXIT_DURING_GUARD.json':
            raise ValueError('Unexpected terminal guard marker path')
        marker = strict_json(marker_path.read_bytes())
    verify_guard_evidence(guard_path, identity['pid'], identity['create_time_epoch'], marker)
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
    validate_producer_supervisor(prior_sources, prior_review)
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
        if key == '4':
            if ancestor != prior_spec.get('recovered_rank4_evidence'):
                raise ValueError('Prior stage is not bound to recovered rank4 audit')
            validate_recovered_rank4(ancestor)
            continue
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


def load_owned():
    spec = importlib.util.spec_from_file_location('_all64_owned_process_v2', project_path(OWNED_PROCESS))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def check_sample(guard, target, phase, path):
    sample = guard.sample(target, phase)
    append_json(path, sample)
    if sample.get('valid') is not True or sample.get('stop_required') is not False:
        raise RuntimeError('Native memory guard stopped or failed closed')


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


def validate_forced_policy(value, run, phase):
    """Verify actual forced-call scope and restoration without claiming profiling."""
    path = bound_artifact(value.get('policy_receipt'), run)
    if path != run / 'FORCE_POLICY.json': raise ValueError('Unexpected forced policy receipt path')
    policy = strict_json(path.read_bytes())
    if (policy.get('schema') != 'research4.scoped-block3-sdpa-policy.v1' or policy.get('policy') != 'force_fused' or
            policy.get('off_scope_calls') != 0 or policy.get('restoration_verified') is not True or
            policy.get('native_base_and_fast_untouched') is not True or policy.get('environment_unchanged') is not True or
            policy.get('installed_files_modified') is not False or value.get('policy_restored') is not True or
            policy.get('environment') != {'MLX_SDPA_VJP_FALLBACK': '1', 'MLX_ENABLE_TF32': None,
                                          'MLX_METAL_GPU_ARCH': None} or
            value.get('fallback_requested_before_mlx_import') is not True):
        raise ValueError('Forced scope policy/restoration evidence differs')
    calls = policy.get('calls')
    if type(calls) is not list or len(calls) != policy.get('matched_calls'):
        raise ValueError('Forced policy call inventory differs')
    counts, widths = {}, set()
    for call in calls:
        q = call.get('q_shape'); width = q[2] if isinstance(q, list) and len(q) == 4 else None
        if (width not in (288, 544, 576, 608) or
                (q, call.get('k_shape'), call.get('v_shape')) != ([1, 24, width, 256], [1, 4, width, 256], [1, 4, width, 256]) or
                not isinstance(call.get('dtype'), str) or not call['dtype'].endswith('bfloat16') or
                call.get('cache') is not None or call.get('mask') != 'causal' or call.get('sinks') is not None or
                call.get('scale') != .0625 or call.get('force_fused_requested') is not True):
            raise ValueError('Forced policy call left the reviewed width/shape scope')
        name = call.get('phase')
        if type(name) is not str: raise ValueError('Forced policy phase missing')
        if phase == 'trajectory':
            match = re.fullmatch(r'step_(\d+)_width_(288|544|576|608)', name)
            if not match or not 1 <= int(match.group(1)) <= 408 or int(match.group(2)) != width:
                raise ValueError('Forced trajectory trace has unexpected phase/width')
        elif not re.fullmatch(r'width_(288|544|576|608)_(full_forward|selected_ad|compiled_ad|native_ad)', name) or not name.startswith('width_' + str(width) + '_'):
            raise ValueError('Forced qualification trace has unexpected phase/width')
        counts[name] = counts.get(name, 0) + 1; widths.add(width)
    if phase == 'qualification':
        expected = {f'width_{width}_{kind}' for width in (288, 544, 576, 608)
                    for kind in ('full_forward', 'selected_ad', 'compiled_ad')}
        expected |= {'width_288_native_ad', 'width_544_native_ad'}
        if set(counts) != expected or any(counts[x] < 16 for x in expected):
            raise ValueError('Forced qualification did not cover every expected width and phase')
    elif widths != {288, 544, 576, 608}:
        raise ValueError('Forced trajectory did not observe every real training width')
    if value.get('forced_width_phase_counts') != counts or value.get('observed_forced_widths') != sorted(widths):
        raise ValueError('Forced policy receipt summary differs from actual calls')
    return counts


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


def verify_training_updates(run):
    updates_path = run / 'updates.jsonl'
    if updates_path.is_symlink() or not updates_path.is_file():
        raise ValueError('Actual-width training update log missing')
    updates = [strict_json(line) for line in updates_path.read_bytes().splitlines() if line.strip()]
    if len(updates) != 408 or any(
            event.get('step') != i or event.get('optimizer_step') != i or
            event.get('epoch') != (i - 1) // 136 + 1 or
            event.get('model_input_width') not in (288, 544, 576, 608)
            for i, event in enumerate(updates, 1)):
        raise ValueError('Training update log does not show 408 actual-width optimizer steps')
    lengths_path = project_path(TOKEN_LENGTHS)
    if digest(lengths_path) != TOKEN_LENGTHS_SHA:
        raise ValueError('Frozen training token inventory changed')
    frozen_rows = [row for row in strict_json(lengths_path.read_bytes())['rows'] if row.get('stage') == 'target']
    if len(frozen_rows) != 136:
        raise ValueError('Frozen training token inventory differs')
    for i, event in enumerate(updates):
        row = frozen_rows[i % 136]
        length, prefix = row['complete_tokens'], row['prefix_tokens']
        expected_width = 32 * ((length + 31) // 32)
        if ((event.get('row_id'), event.get('sequence_tokens'), event.get('target_tokens'),
             event.get('model_input_width')) != (row['id'], length, length - prefix, expected_width) or
                type(event.get('loss')) not in (int, float) or not math.isfinite(event['loss']) or
                type(event.get('step_seconds')) not in (int, float) or
                not math.isfinite(event['step_seconds']) or event['step_seconds'] <= 0):
            raise ValueError('Training update differs from frozen order or token geometry')


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
                ('research4.all64-broader-actualwidth-dev.v1', 'complete', spec['epoch'], spec['checkpoint'],
                 136 * spec['epoch'], True, True, None, None, True, True) or
                value.get('adapter_sha256') != checkpoint['adapter']['sha256'] or
                type(state_hash) is not str or not HEX64.fullmatch(state_hash) or
                value.get('trainable_state_sha256') != state_hash or
                value.get('trainable_digest_before') != state_hash or value.get('trainable_digest_after') != state_hash):
            raise ValueError('DEV checkpoint identity, unchanged adapter or no-optimizer evidence differs')
        masks_path = project_path(DEV_MASKS)
        if digest(masks_path) != DEV_MASKS_SHA or digest(project_path(SELECTOR)) != SELECTOR_SHA:
            raise ValueError('Frozen DEV selection inputs differ')
        lengths_path = project_path(TOKEN_LENGTHS)
        if digest(lengths_path) != TOKEN_LENGTHS_SHA:
            raise ValueError('Frozen DEV token inventory changed')
        lengths = strict_json(lengths_path.read_bytes())['rows']
        validate_dev_widths(lengths)
        widths = sorted({32 * ((row['dev_complete_tokens'] + 31) // 32) for row in lengths
                         if row.get('stage') == 'probe' and 'dev_complete_tokens' in row})
        if (value.get('dev_model_input_widths') != widths or value.get('max_dev_model_input_width') != max(widths) or
                value.get('frozen_prompt_context_tokens') != 640 or value.get('no_truncation') is not True or
                value.get('evaluation_policy') != 'eval_mode_default_sdpa_no_grad' or
                value.get('fallback_requested_before_mlx_import') is not True):
            raise ValueError('Actual-width DEV geometry, evaluation policy or fallback setting differs')
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
                 ('research4.all64-broader-actualwidth-qualification.v1', 'qualified', 4,
                  [288, 544, 576, 608], None, True, True, True, [1, 2]) and
                 value.get('compiled_640_qualified') is False)
    else:
        checkpoints = value.get('checkpoints')
        valid = ((value.get('schema'), value.get('status'), value.get('executed_updates'),
                  value.get('epoch_steps')) == ('research4.all64-broader-actualwidth-trajectory.v1',
                                                'complete', 408, [136, 272, 408]) and
                 value.get('allowed_model_input_widths') == [288, 544, 576, 608] and
                 value.get('compiled_640_qualified') is False and
                 value.get('fresh_base_and_optimizer') is True and value.get('resumed') is False and
                 value.get('training_order') == 'frozen_train_preview_jsonl_order_each_epoch' and
                 isinstance(checkpoints, list) and len(checkpoints) == 3 and
                 [(x.get('epoch'), x.get('optimizer_step')) for x in checkpoints] ==
                 [(1, 136), (2, 272), (3, 408)])
    if not common or not valid: raise ValueError('Worker completion/qualification evidence is incomplete')
    run = project_path(RUNS + '/' + spec['owner'])
    if spec['phase'] == 'qualification':
        cases = value.get('cases')
        if type(cases) is not list or len(cases) != 4 or value.get('maximum_observed_suffix_tokens') != 50:
            raise ValueError('All four actual-width numerical qualification cases required')
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
        for index, (case, width) in enumerate(zip(cases, [288, 544, 576, 608]), 1):
            if (type(case) is not dict or case.get('step') != index or case.get('optimizer_step') != index or
                    case.get('model_input_width') != width or case.get('batch_width') != width + 1 or
                    case.get('ceiling_padded') is not False or
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
            if (type(prefix) is not int or type(length) is not int or not 0 < prefix < length <= 608 or
                    case.get('target_tokens') != length - prefix or
                    case.get('target_positions') != list(range(prefix, length))):
                raise ValueError('Qualification shifted target mask mismatch')
            frozen = targets.get(case.get('row_id'))
            if (type(frozen) is not dict or frozen['complete_tokens'] != length or frozen['prefix_tokens'] != prefix or
                    (length - prefix, length) != maxima.get(width)):
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
        verify_training_updates(run)
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
                'max_seq_length': 608, 'frozen_prompt_context_tokens': 640,
                'actual_model_input_width_limit': 608, 'no_truncation': True,
                'mask_prompt': True, 'resume_adapter_file': None,
                'optimizer': 'adam', 'grad_accumulation_steps': 1, 'grad_checkpoint': True,
                'lora_parameters': {'rank': spec['rank'], 'dropout': 0.0, 'scale': 2.0},
                'iters': epoch * 136, 'learning_rate': 1e-5, 'seed': 20261108,
                'optimizer_config': {'adam': {'betas': [.9, .999], 'eps': 1e-8, 'bias_correction': False}}}
            if config != expected_config: raise ValueError('Saved adapter optimizer/config differs')
            saved = folder / 'CHECKPOINT.json'
            if saved.is_symlink() or not saved.is_file() or strict_json(saved.read_bytes()) != checkpoint:
                raise ValueError('Immutable checkpoint descriptor differs')
    validate_forced_policy(value, run, spec['phase'])
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
    owned = load_owned()
    environment = {key: os.environ.get(key) for key in ('MLX_SDPA_VJP_FALLBACK', 'MLX_ENABLE_TF32',
                                                        'MLX_METAL_GPU_ARCH', 'MLX_DISABLE_COMPILE')}
    if any(value is not None for value in environment.values()):
        raise ValueError('Unexpected inherited MLX environment; no silent policy override')
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
        'effective_deadline_epoch': deadline, 'automatic_retry': False,
        'inherited_mlx_environment': environment, 'child_fallback_setting': '1',
        'tf32_note': 'Unset is recorded; actual backend default is not inferred.'})
    slot = owned.LaunchSlot(); lock_owned = False; clean = False; result = None; error = None; exit_race = None
    terminal_sample_candidate = None
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
        command = ['/usr/bin/env', 'MLX_SDPA_VJP_FALLBACK=1', str(PROJECT / PYTHON), '-B', '-u', str(project_path(program)),
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
                sample = guard.sample(target, 'run'); append_json(monitor / 'GUARD.jsonl', sample)
                if sample.get('valid') is not True or sample.get('stop_required') is not False:
                    # The only non-error case is an already-exited owned child:
                    # this invalid sample is retained, never marked healthy,
                    # and the model is not allowed to continue.
                    if terminal_exit_sample(sample, target):
                        # Stop monitoring and immediately enter owned cleanup;
                        # no invalid sample permits further training. Cleanup
                        # must later prove waitid-exited and reap exit code 0.
                        terminal_sample_candidate = sample
                        break
                    raise RuntimeError('Native memory guard stopped or failed closed')
                time.sleep(1.0)
            code, descendants, pending = owned.cleanup(slot.process); clean = True
        if code != 0 or descendants or pending:
            raise RuntimeError('Worker exit or process-group cleanup failed')
        if terminal_sample_candidate is not None:
            # owned.cleanup has now established the anchored child exited and
            # the owned group was empty before reaping. A nonzero or uncertain
            # exit never reaches this marker.
            exit_race = {'schema': 'research4.all64-terminal-guard-exit-race.v1',
                'worker_pid': target['pid'], 'worker_create_time_epoch': target['create_time_epoch'],
                'sample_sha256': sample_hash(terminal_sample_candidate), 'waitid_exited_after_sample': True,
                'invalid_sample_treated_as_healthy': False, 'worker_allowed_to_continue': False}
            exclusive_json(monitor / 'EXIT_DURING_GUARD.json', exit_race)
        validate_spec(spec, now=min(time.time(), deadline - 1e-6))
        receipt = run / 'worker_receipt.json'
        if not receipt.is_file() or receipt.is_symlink(): raise RuntimeError('Worker terminal receipt missing')
        value = validate_worker_receipt(strict_json(receipt.read_bytes()), spec)
        guard_path = monitor / 'GUARD.jsonl'; verify_guard_evidence(guard_path, target['pid'], target['create_time_epoch'], exit_race)
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
        if exit_race is not None:
            marker = monitor / 'EXIT_DURING_GUARD.json'
            result['guard_exit_race'] = {'path': marker.relative_to(PROJECT).as_posix(), 'sha256': digest(marker)}
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
