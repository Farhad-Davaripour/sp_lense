"""Owned no-update draft for frozen-width force-fused primal observations.

This program does not itself create authority, acquire a lease, or launch a
model. A separate reviewed supervisor must provide all three. It reproduces
the failed worker's loader and original full-native function, captures the
failed exact row, and exits without making an optimizer update.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
ORIGINAL_REL = 'mac_studio/all64_broader_v2_20261119/worker.py'
ORIGINAL_SHA = 'b4996bc66695ff8abda70cc78764875d7cdfadcb1be1d7666b985fd6f68e938e'
FAILED_OWNER = 'c18da432781347699cae2aa6e15dec90'
FROZEN_SHA = 'dc1d812ba3e2b2aa147b1fbe148440ff6a5327a5a67feb8d32f330883ac98f70'
LOCK = PROJECT / '.local/validation/qwen38_shared_v1/inference.lock/owner.json'
REQUIRED_SOURCES = {ORIGINAL_REL,
    'mac_studio/all64_force_fused_widths_v4b_20261119/diagnostic_worker.py',
    'mac_studio/all64_force_fused_widths_v4_20261119/scoped_sdpa_widths.py',
    'mac_studio/all64_force_fused_widths_v4_20261119/width_observer.py',
    'mac_studio/all64_force_fused_widths_v4_20261119/WIDTH_CASES.json',
    'mac_studio/game_block3_sdpa_policy.py',
    'mac_studio/all64_native_primal_diagnostic_v1_20261119/native_primal_observer.py',
    'mac_studio/all64_primal_order_diagnostic_v2_20261119/repeat_observer.py',
    'mac_studio/all64_force_fused_multiwidth_supervision_v4_20261119/supervisor.py',
    'mac_studio/owned_validation_process.py',
    'mac_studio/all64_broader_v2_20261119/frozen_data.py',
    'mac_studio/all64_broader_v2_20261119/loss.py',
    'mac_studio/general_preservation_render_v1.py',
    'mac_studio/game_all_layers_layout.py',
    'mac_studio/game_compiled_direct_gradient_factory_v1.py',
    'mac_studio/game_selected_head_v3_compare.py',
    'mac_studio/qwen38_local.py', 'mac_studio/qwen38_config.json',
    'mac_studio/mini_native_guard_20261119/mini_guard.py',
    'mac_studio/mini_native_guard_20261119/MINI_CROSSBUILT_BINDING_3778b7a0365143dc8f9d4e9922e5fd05.json'}


def sha(path):
    p = Path(path)
    if p.is_symlink() or not p.is_file(): raise ValueError('Regular non-symlink file required')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('Regular JSON required')
    return json.loads(path.read_text(encoding='utf-8'))


def write_once(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def run_scoped_and_persist(scope, operation, receipt_path):
    """Save yielded alias/restoration evidence even when observation fails."""
    receipt = None
    try:
        with scope as receipt:
            result = operation()
    finally:
        if receipt is not None:
            write_once(receipt_path, receipt)
    return result, receipt


def verify_shared_case_state(result, theta, rng):
    for name in ('before', 'after_plain', 'after_ad'):
        if (result.get(name, {}).get('trainable_state_sha256') != theta
                or result.get(name, {}).get('rng_state_sha256') != rng):
            raise ValueError('Shared theta/RNG identity differs across width controls')


def verify_admission(args):
    run = Path(args.run_dir).resolve()
    spec_path = Path(args.spec).resolve()
    if (not run.is_relative_to((PROJECT / '.local').resolve()) or not run.is_dir()
            or not spec_path.is_relative_to((PROJECT / '.local').resolve())
            or 'onedrive' in str(run).lower() or sha(spec_path) != args.spec_sha256):
        raise ValueError('Exact owned, non-OneDrive run/spec required')
    spec = read_json(spec_path)
    if (spec.get('schema') != 'research4.all64-force-fused-widths-supervision-spec.v1'
            or spec.get('owner') != args.owner or spec.get('deadline_epoch') != args.deadline_epoch
            or spec.get('manifest_sha256') != FROZEN_SHA or spec.get('rank') != 4
            or spec.get('failed_owner') != FAILED_OWNER or spec.get('failed_worker_sha256') != ORIGINAL_SHA
            or spec.get('failed_worker_failure_sha256') != '3d3c4886117f78100ae2b06825ad9521fe78929268de2875c9e787b33a948670'
            or spec.get('failed_supervisor_release_sha256') != '4184aad2f160ff9d9954f79af7900d2312e0207024cd320dc24de9256440f11e'
            or spec.get('prior_diagnostic_owner') != 'f73186a2ee0d44448e163a14c4808326'
            or spec.get('prior_diagnostic_observation_sha256') != '255e95806cef0f342cbb9b9a1bdbb7224434ef9acb8b916771ef818738150058'
            or spec.get('prior_diagnostic_receipt_sha256') != 'cb412c289ff676d3472fb73666be146badc13be6fa81f5dd47f8027be6287a35'
            or spec.get('prior_diagnostic_release_sha256') != '60528fc196eade21cf2afd580736a572c497f3f7b5bcc4d43606b30f71a95ec6'
            or spec.get('order_diagnostic_owner') != '3a78fc712c9f409e9c3b22f6f255e44d'
            or spec.get('order_diagnostic_observation_sha256') != '00cc08e63344d0dd50144b0fcbc7e847ebf5226a090ed20571d6eb5ce4ef1b5e'
            or spec.get('order_diagnostic_receipt_sha256') != '3c9dfd8608891e2a045078b020766e348fd0d3e3cd8f3f4b51e617fb1c702860'
            or spec.get('order_diagnostic_release_sha256') != 'c4077a407b5e017ff0936aa628b42444cc8357bf4dc6cb51b566b119a90bea71'
            or spec.get('fused_diagnostic_owner') != 'bc118dccf6d4426da1d3f5b2d217e787'
            or spec.get('fused_diagnostic_observation_sha256') != '548a570aea2c4a66a7568eb572b6d75e6a9f06760e8f1d775180b92c14ba4f68'
            or spec.get('fused_diagnostic_receipt_sha256') != '7adf1a46d6203578ff41401ba12da1abbfe4dae27b0eb981bfe3fe74e17cb726'
            or spec.get('fused_diagnostic_release_sha256') != '09e16a596b4fedad11b018180edf61d2f93e44c49d4b1403df62fa8bed93ae65'
            or not re.fullmatch('[0-9a-f]{32}', args.owner)
            or not math.isfinite(args.deadline_epoch) or time.time() >= args.deadline_epoch):
        raise ValueError('Diagnostic identity, failed reference or deadline differs')
    sources = spec.get('source_files')
    if not isinstance(sources, dict) or not REQUIRED_SOURCES.issubset(sources):
        raise ValueError('All exact diagnostic/native loader source bindings required')
    for rel, expected in sources.items():
        path = Path(rel)
        if (path.is_absolute() or '..' in path.parts or not re.fullmatch('[0-9a-f]{64}', expected)
                or sha(PROJECT / path) != expected):
            raise ValueError('Bound diagnostic source differs: ' + str(rel))
    if sources[ORIGINAL_REL] != ORIGINAL_SHA:
        raise ValueError('Original failed worker bytes differ')
    prior_policy = (PROJECT / 'mac_studio/game_block3_sdpa_policy.py').read_bytes()
    derived = (PROJECT / 'mac_studio/all64_force_fused_widths_v4_20261119/scoped_sdpa_widths.py').read_bytes()
    template = prior_policy.replace(b'1023', b'_approved_width')
    needle = b'    def dispatch(queries, keys, values, cache, scale, mask, sinks=None):\n        valid = (\n            list(queries.shape)'
    replacement = (b'    def dispatch(queries, keys, values, cache, scale, mask, sinks=None):\n'
                   b'        _approved_width = queries.shape[2] if len(queries.shape) == 4 else None\n'
                   b'        valid = (\n            type(_approved_width) is int and _approved_width in (288, 544, 576, 608, 640)\n'
                   b'            and list(queries.shape)')
    if (hashlib.sha256(prior_policy).hexdigest() != '80817ebfab8733c1ab6261e78922137998ce29e48733d4bab95fa346d3e795a3'
            or prior_policy.count(b'1023') != 3 or template.count(needle) != 1
            or derived != template.replace(needle, replacement)
            or hashlib.sha256(derived).hexdigest() != 'b84a79c806980408e9cfdd30794093baf6b40e0cb5c1c2882bf08243b7baf3bf'):
        raise ValueError('Exact old owned SDPA scope with only reviewed width whitelist required')
    supervisor_path = PROJECT / 'mac_studio/all64_force_fused_multiwidth_supervision_v4_20261119/supervisor.py'
    supervisor_spec = importlib.util.spec_from_file_location('_all64_primal_diagnostic_supervisor', supervisor_path)
    supervisor = importlib.util.module_from_spec(supervisor_spec)
    supervisor_spec.loader.exec_module(supervisor)
    supervisor.validate_spec(spec)
    authority = read_json(run / 'worker_authority.json')
    expected_authority = {'schema': 'research4.all64-force-fused-widths-diagnostic-authority.v1',
        'owner': args.owner, 'supervisor_pid': os.getppid(), 'deadline_epoch': args.deadline_epoch,
        'spec_sha256': args.spec_sha256, 'worker_sha256': sha(__file__),
        'failed_owner': FAILED_OWNER, 'failed_worker_sha256': ORIGINAL_SHA}
    if authority != expected_authority:
        raise ValueError('Exact diagnostic owner and parent authority required')
    owner = read_json(LOCK)
    if (owner.get('owner'), owner.get('pid'), owner.get('kind')) != (args.owner, os.getppid(), 'all64_force_fused_widths_diagnostic_v4'):
        raise ValueError('Canonical shared lease owner differs')
    return run, spec


def run_diagnostic(args):
    run, spec = verify_admission(args)
    # These imports happen only after the supervisor-owned source hash checks.
    old_dir = PROJECT / 'mac_studio/all64_broader_v2_20261119'
    sys.path.insert(0, str(old_dir)); sys.path.insert(0, str(PROJECT / 'mac_studio'))
    import frozen_data
    import loss
    sys.path.insert(0, str(PROJECT / 'mac_studio/all64_native_primal_diagnostic_v1_20261119'))
    sys.path.insert(0, str(PROJECT / 'mac_studio/all64_primal_order_diagnostic_v2_20261119'))
    sys.path.insert(0, str(PROJECT / 'mac_studio/all64_force_fused_widths_v4_20261119'))
    import repeat_observer as repeat
    import width_observer as observer
    import scoped_sdpa_widths as sdpa
    module_spec = importlib.util.spec_from_file_location('_exact_failed_all64_worker', PROJECT / ORIGINAL_REL)
    old = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(old)
    old.authenticate = frozen_data.authenticate
    old.render_inputs = frozen_data.render_inputs
    old.geometry = loss.geometry
    old.make_batch = loss.make_batch
    old.selected_suffix_loss = loss.selected_suffix_loss
    args.rank = 4
    old._boundary(args, run, 'before_original_loader', 'launch')
    state = old._load_runtime(args, run)
    mx, model, optimizer, handle = (state[name] for name in ('mx', 'model', 'optimizer', 'handle'))
    original_call = state['original_call']
    try:
        mx.eval(optimizer.state)
        before_step = int(optimizer.step.item())
        if before_step != 0 or old._b_parameter_layers(mx, model):
            raise ValueError('Fresh rank4 initial model/optimizer/B state required')
        fixture_path = PROJECT / 'mac_studio/all64_force_fused_widths_v4_20261119/WIDTH_CASES.json'
        if sha(fixture_path) != '3b0a08174b32fe5be6f84eb7dd1ed068ad00da9a4027a3ef6addf1a5871e6b0e':
            raise ValueError('Reviewed width fixture differs')
        fixture = read_json(fixture_path)
        if (fixture.get('schema') != 'research4.all64-force-fused-width-cases.v1'
                or fixture.get('frozen_manifest_sha256') != FROZEN_SHA
                or fixture.get('qwen_model_loaded') is not False
                or [item['model_input_width'] for item in fixture.get('cases', [])] != list(observer.WIDTHS)):
            raise ValueError('Frozen width case identities differ')
        actual = old.qualification_cases(state['prepared'])
        if len(actual) != len(fixture['cases']):
            raise ValueError('All five frozen width cases are required')
        for (row_id, ids, prefix, ceiling), expected in zip(actual, fixture['cases']):
            info = loss.geometry(ids, prefix, ceiling)
            actual_identity = {'row_id': row_id, 'token_u32be_sha256': observer.token_sha(ids),
                'semantic_tokens': len(ids), 'prefix_tokens': prefix, 'target_tokens': len(ids)-prefix,
                'model_input_width': info['model_input_width'], 'batch_width': info['batch_width'],
                'ceiling_padded': ceiling}
            if actual_identity != expected:
                raise ValueError('Actual frozen width case differs from reviewed fixture')
        write_once(run / 'DIAGNOSTIC_PRECONDITION.json', {
            'schema': 'research4.all64-force-fused-widths-diagnostic-precondition.v1',
            'failed_owner': FAILED_OWNER, 'failed_worker_sha256': ORIGINAL_SHA,
            'width_fixture_sha256': sha(fixture_path), 'cases': fixture['cases'],
            'rank': 4, 'adapter_layers': 64,
            'optimizer_constructed_by_original_loader': True, 'optimizer_step_before': before_step,
            'adam_updates': 0, 'qualification_claim': False})
        from mlx.utils import tree_flatten
        phase = {'name': 'pre_policy'}
        def boundary(label): return old._boundary(args, run, 'diagnostic_' + label)
        full_before = repeat.full_state_snapshot(mx, model, tree_flatten, boundary)
        flags_before, rng_before = repeat.flags_snapshot(model), repeat.rng_snapshot(mx)
        theta_before = old.parameter_digest(model)
        if not flags_before['all_training']:
            raise ValueError('Expected complete training mode before diagnostic')
        write_once(run / 'FULL_STATE_BEFORE.json', full_before)
        original_env = sdpa.environment_snapshot()
        def measure():
            results = []
            for (row_id, ids, prefix, ceiling), expected in zip(actual, fixture['cases']):
                width = expected['model_input_width']
                batch, lengths, _info = loss.make_batch(mx, ids, prefix, ceiling)
                folder = run / ('width_' + str(width))
                folder.mkdir(mode=0o700, exist_ok=False)
                def case_boundary(label):
                    if label == 'before_plain_forward': phase['name'] = 'force_fused_%d_direct' % width
                    elif label == 'after_plain_before_ad': phase['name'] = 'force_fused_%d_ad' % width
                    return boundary('width_%d_%s' % (width, label))
                try:
                    result = observer.observe(mx, state['nn'], model, batch, lengths, expected,
                        full_native_reference=old._full_native_reference, parameter_digest=old.parameter_digest,
                        boundary=case_boundary, output_dir=folder)
                except RuntimeError as exc:
                    if str(exc) != 'Forced width primal did not match; exact evidence saved': raise
                    result = read_json(folder / 'WIDTH_OBSERVATION.json')
                    if result.get('status') != 'MISMATCH' or not result.get('theta_unchanged') or not result.get('rng_unchanged'):
                        raise
                verify_shared_case_state(result, theta_before, rng_before)
                results.append(result)
            return results
        scope = sdpa.scoped_sdpa_policy(owned_process_pid=os.getpid(), policy='force_fused',
                                        phase=phase, expected_environment=original_env)
        observations, policy = run_scoped_and_persist(scope, measure, run / 'FORCE_POLICY.json')
        if (policy.get('policy') != 'force_fused' or policy.get('off_scope_calls') != 0
                or not policy.get('restoration_verified') or not policy.get('native_base_and_fast_untouched')
                or not policy.get('environment_unchanged') or policy.get('installed_files_modified') is not False
                or any(item.get('force_fused_requested') is not True for item in policy.get('calls', []))):
            raise ValueError('Scoped force-fused alias integrity/restoration differs')
        phase_counts = {}
        for width in observer.WIDTHS:
            for kind in ('direct', 'ad'):
                name = 'force_fused_%d_%s' % (width, kind)
                phase_counts[name] = sum(item.get('phase') == name for item in policy['calls'])
                if phase_counts[name] < 16:
                    raise ValueError('All 16 attention blocks must request forced scope in every width/phase')
        if sum(phase_counts.values()) != policy['matched_calls']:
            raise ValueError('Unattributed forced policy requests')
        full_after = repeat.full_state_snapshot(mx, model, tree_flatten, boundary)
        flags_after, rng_after = repeat.flags_snapshot(model), repeat.rng_snapshot(mx)
        write_once(run / 'FULL_STATE_AFTER.json', full_after)
        if (full_before['sha256'] != full_after['sha256'] or full_before['inventory'] != full_after['inventory']
                or flags_before != flags_after or rng_before != rng_after):
            raise ValueError('Enumerable full model state, training flags or RNG changed')
        if any(observation.get('status') not in ('MATCH', 'MISMATCH')
                or observation.get('qualification_claim') is not False
                or observation.get('adam_updates') != 0 for observation in observations):
            raise ValueError('Saved no-update observation identity or status differs')
        mx.eval(optimizer.state)
        after_step = int(optimizer.step.item())
        if after_step != before_step or after_step != 0:
            raise ValueError('Diagnostic optimizer advanced unexpectedly')
        old._boundary(args, run, 'diagnostic_after_observation')
        receipt = {'schema': 'research4.all64-force-fused-widths-diagnostic-receipt.v1',
            'status': 'observed_not_qualification', 'owner': args.owner, 'rank': 4,
            'failed_owner': FAILED_OWNER, 'failed_worker_sha256': ORIGINAL_SHA,
            'manifest_sha256': FROZEN_SHA, 'observed_widths': list(observer.WIDTHS),
            'all_widths_matched': all(item['status'] == 'MATCH' for item in observations),
            'case_results': [{'width': width,
                'observation_sha256': sha(run / ('width_' + str(width)) / 'WIDTH_OBSERVATION.json'),
                'plain_forward_sha256': sha(run / ('width_' + str(width)) / 'PLAIN_FORWARD.json')}
                for width in observer.WIDTHS],
            'policy_receipt_sha256': sha(run / 'FORCE_POLICY.json'),
            'state_before_sha256': sha(run / 'FULL_STATE_BEFORE.json'),
            'state_after_sha256': sha(run / 'FULL_STATE_AFTER.json'),
            'model_state_digest_before': full_before['sha256'],
            'model_state_digest_after': full_after['sha256'],
            'state_unchanged': True, 'training_flags_unchanged': True, 'rng_unchanged': True,
            'training_flags_before': flags_before, 'training_flags_after': flags_after,
            'rng_before_sha256': rng_before, 'rng_after_sha256': rng_after,
            'shared_trainable_state_sha256': theta_before,
            'force_fused_phase_counts': phase_counts,
            'policy_restored': True,
            'optimizer_constructed_by_original_loader': True, 'optimizer_step_before': before_step,
            'optimizer_step_after': after_step, 'adam_updates': 0, 'qualification_claim': False}
        write_once(run / 'worker_receipt.json', receipt)
    finally:
        handle.close()
        type(model.layers[0]).__call__ = original_call


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True); parser.add_argument('--owner', required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--spec', required=True); parser.add_argument('--spec-sha256', required=True)
    options = parser.parse_args()
    try:
        run_diagnostic(options)
    except BaseException as exc:
        failure_dir = Path(options.run_dir).resolve()
        target = failure_dir / 'worker_failure.json'
        if failure_dir.is_relative_to((PROJECT / '.local').resolve()) and failure_dir.is_dir() and not target.exists():
            write_once(target, {'status': 'failed', 'error_type': type(exc).__name__,
                                'message': str(exc), 'epoch': time.time()})
        raise
