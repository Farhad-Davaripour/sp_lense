"""Owned no-update Qwen capture for the pinned fused-SDPA VJP concern.

No model loads on import. A reviewed external supervisor must own the Mini
lease, source manifest, native memory guard, deadline and this worker.
"""
import argparse
import gc
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
MANIFEST = 'dc1d812ba3e2b2aa147b1fbe148440ff6a5327a5a67feb8d32f330883ac98f70'
ORIGINAL = 'mac_studio/all64_broader_v2_20261119/worker.py'
ORIGINAL_SHA = 'b4996bc66695ff8abda70cc78764875d7cdfadcb1be1d7666b985fd6f68e938e'
PROBE_DIR = 'mac_studio/all64_sdpa_vjp_probe_v1_20261119'
SUPERVISOR = 'mac_studio/all64_sdpa_vjp_supervision_v2_20261119/supervisor.py'
LOCK = PROJECT / '.local/validation/qwen38_shared_v1/inference.lock/owner.json'
CASE_ID = 'sustained_multi_incident_02.12'
CASE_TOKEN_SHA = '9fc914b962fa2cbb96f70f31c32e12aa5b26dc99ee0272158c36353c1bd8cf93'
SAMPLE_NAMES = ['language_model.model.layers.0.linear_attn.in_proj_qkv.lora_b'] + [
    'language_model.model.layers.3.self_attn.%s.lora_b' % name for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
REQUIRED = {ORIGINAL, SUPERVISOR, PROBE_DIR + '/capture_fused_v2.py', PROBE_DIR + '/attention_tap.py',
    PROBE_DIR + '/synthetic_discriminator.py',
    'mac_studio/all64_broader_v2_20261119/frozen_data.py', 'mac_studio/all64_broader_v2_20261119/loss.py',
    'mac_studio/all64_primal_order_diagnostic_v2_20261119/repeat_observer.py',
    'mac_studio/all64_force_fused_widths_v4_20261119/scoped_sdpa_widths.py',
    'mac_studio/game_block3_sdpa_policy.py', 'mac_studio/game_all_layers_layout.py',
    'mac_studio/game_compiled_direct_gradient_factory_v1.py', 'mac_studio/game_selected_head_v3_compare.py',
    'mac_studio/general_preservation_render_v1.py', 'mac_studio/qwen38_local.py', 'mac_studio/qwen38_config.json',
    'mac_studio/mini_native_guard_20261119/mini_guard.py',
    'mac_studio/mini_native_guard_20261119/MINI_CROSSBUILT_BINDING_3778b7a0365143dc8f9d4e9922e5fd05.json'}


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('Regular file required')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('Regular JSON required')
    return json.loads(path.read_text(encoding='utf-8'))


def once(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def reference(path):
    return {'path': path.relative_to(PROJECT).as_posix(), 'sha256': sha(path)}


def scalar_record(value, count):
    scalar = float(value.item())
    return {'loss': scalar if math.isfinite(scalar) else None, 'loss_repr': repr(scalar),
            'loss_classification': 'finite' if math.isfinite(scalar) else 'nonfinite',
            'loss_dtype': str(value.dtype), 'target_count': int(count.item()), 'count_dtype': str(count.dtype)}


def verify_admission(args):
    if 'MLX_SDPA_VJP_FALLBACK' in os.environ:
        raise ValueError('Capture process must start with fallback environment absent')
    run, spec_path = Path(args.run_dir).resolve(), Path(args.spec).resolve()
    if (not run.is_relative_to((PROJECT / '.local').resolve()) or not run.is_dir()
            or not spec_path.is_relative_to((PROJECT / '.local').resolve())
            or 'onedrive' in str(run).lower() or sha(spec_path) != args.spec_sha256):
        raise ValueError('Exact owned non-OneDrive run/spec required')
    spec = read_json(spec_path)
    if (spec.get('schema') != 'research4.all64-sdpa-vjp-probe-supervision-spec.v1'
            or spec.get('phase') != 'capture_fused' or spec.get('rank') != 4
            or spec.get('owner') != args.owner or spec.get('deadline_epoch') != args.deadline_epoch
            or spec.get('manifest_sha256') != MANIFEST or not re.fullmatch('[0-9a-f]{32}', args.owner)
            or not math.isfinite(args.deadline_epoch) or time.time() >= args.deadline_epoch):
        raise ValueError('Capture identity/deadline differs')
    sources = spec.get('source_files')
    if not isinstance(sources, dict) or not REQUIRED.issubset(sources):
        raise ValueError('Required capture source bindings missing')
    for rel, digest in sources.items():
        path = Path(rel)
        if path.is_absolute() or '..' in path.parts or not re.fullmatch('[0-9a-f]{64}', digest) or sha(PROJECT / path) != digest:
            raise ValueError('Source binding differs: ' + str(rel))
    if sources[ORIGINAL] != ORIGINAL_SHA:
        raise ValueError('Original Qwen loader differs')
    sup_spec = importlib.util.spec_from_file_location('_vjp_probe_supervisor', PROJECT / SUPERVISOR)
    supervisor = importlib.util.module_from_spec(sup_spec); sup_spec.loader.exec_module(supervisor)
    supervisor.validate_spec(spec)
    authority = read_json(run / 'worker_authority.json')
    expected = {'schema': 'research4.all64-sdpa-vjp-probe-authority.v1', 'phase': 'capture_fused',
        'rank': 4, 'owner': args.owner, 'supervisor_pid': os.getppid(), 'deadline_epoch': args.deadline_epoch,
        'spec_sha256': args.spec_sha256, 'worker_sha256': sha(__file__), 'manifest_sha256': MANIFEST}
    if authority != expected: raise ValueError('Exact external owner/parent authority required')
    lock = read_json(LOCK)
    if (lock.get('owner'), lock.get('pid'), lock.get('kind')) != (args.owner, os.getppid(), 'all64_sdpa_vjp_probe_v1'):
        raise ValueError('Canonical shared lease differs')
    return run, spec


def save_tensors(runtime, path, values):
    if path.exists(): raise ValueError('Immutable private tensor evidence exists')
    runtime.save_safetensors(str(path), values)
    os.chmod(path, 0o600)
    with path.open('rb') as stream: os.fsync(stream.fileno())


def capture(args):
    run, spec = verify_admission(args)
    old_dir = PROJECT / 'mac_studio/all64_broader_v2_20261119'
    sys.path[:0] = [str(old_dir), str(PROJECT / 'mac_studio'),
                    str(PROJECT / 'mac_studio/all64_primal_order_diagnostic_v2_20261119'),
                    str(PROJECT / 'mac_studio/all64_force_fused_widths_v4_20261119')]
    import frozen_data, loss, repeat_observer as repeat, scoped_sdpa_widths as sdpa
    import attention_tap as tap
    import synthetic_discriminator as synthetic
    from game_selected_head_v3_compare import compare_gradients, loss_equivalence, TOLERANCES
    original_spec = importlib.util.spec_from_file_location('_exact_failed_all64_worker', PROJECT / ORIGINAL)
    old = importlib.util.module_from_spec(original_spec); original_spec.loader.exec_module(old)
    old.authenticate, old.render_inputs = frozen_data.authenticate, frozen_data.render_inputs
    old.geometry, old.make_batch, old.selected_suffix_loss = loss.geometry, loss.make_batch, loss.selected_suffix_loss
    args.rank = 4
    old._boundary(args, run, 'capture_before_loader', 'launch')
    state = old._load_runtime(args, run)
    mx, nn, model, optimizer, handle = (state[name] for name in ('mx', 'nn', 'model', 'optimizer', 'handle'))
    original_call = state['original_call']
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3_next
    try:
        mx.eval(optimizer.state)
        if int(optimizer.step.item()) != 0 or old._b_parameter_layers(mx, model):
            raise ValueError('Fresh B0/Adam0 required')
        row_id, ids, prefix, ceiling = old.qualification_cases(state['prepared'])[-1]
        if row_id != CASE_ID or len(ids) != 590 or prefix != 560 or not ceiling:
            raise ValueError('Exact longest frozen 590-token row padded to 640 input required')
        import struct
        token_sha = hashlib.sha256(b''.join(struct.pack('>I', value) for value in ids)).hexdigest()
        if token_sha != CASE_TOKEN_SHA: raise ValueError('Exact frozen token sequence differs')
        batch, lengths, geometry = loss.make_batch(mx, ids, prefix, ceiling)
        if geometry['model_input_width'] != 640 or geometry['target_tokens'] != 30:
            raise ValueError('Exact padded geometry or mask differs')
        expected_layers = [index for index, block in enumerate(model.layers) if not block.is_linear]
        if expected_layers != list(range(3, 64, 4)):
            raise ValueError('Exact 16 full-attention layer identities required')
        if any(name not in state['expected'] for name in SAMPLE_NAMES):
            raise ValueError('Fixed five LoRA sample names absent')
        expected_sample = {name: state['expected'][name] for name in SAMPLE_NAMES}
        def boundary(label): return old._boundary(args, run, 'capture_' + label)
        before = repeat.full_state_snapshot(mx, model, tree_flatten, boundary)
        before_flags, before_rng = repeat.flags_snapshot(model), repeat.rng_snapshot(mx)
        before_theta = old.parameter_digest(model)
        once(run / 'FULL_STATE_BEFORE.json', before)
        phase = {'name': 'capture_pre'}
        scope = sdpa.scoped_sdpa_policy(owned_process_pid=os.getpid(), policy='force_fused',
                                        phase=phase, expected_environment=sdpa.environment_snapshot())
        policy = None
        try:
            with scope as policy:
                def full(b, lengths_): return old._full_native_reference(mx, nn, model, b, lengths_)
                grad_fn = nn.value_and_grad(model, full)
                boundary('before_plain')
                phase['name'] = 'untapped_direct'
                plain, plain_n = full(batch, lengths); mx.eval(plain, plain_n)
                plain_value, count = float(plain.item()), int(plain_n.item())
                once(run / 'UNTAPPED_DIRECT_RAW.json', scalar_record(plain, plain_n))
                del plain, plain_n; gc.collect(); mx.clear_cache()
                boundary('before_untapped_ad')
                phase['name'] = 'untapped_ad'
                (baseline, baseline_n), gradient = grad_fn(batch, lengths)
                mx.eval(baseline, baseline_n, gradient)
                baseline_value, baseline_count = float(baseline.item()), int(baseline_n.item())
                once(run / 'UNTAPPED_AD_RAW.json', scalar_record(baseline, baseline_n))
                primal = loss_equivalence(plain_value, baseline_value, count, baseline_count, TOLERANCES)
                once(run / 'UNTAPPED_PRIMAL_COMPARISON.json', primal)
                if count != 30 or not primal['passed']: raise ValueError('Untapped forced direct/AD primal gate failed')
                baseline_flat = dict(tree_flatten(gradient))
                if set(baseline_flat) != set(state['expected']): raise ValueError('All 992 gradients required')
                old._validate_gradient_inventory(mx, baseline_flat, state['expected'])
                old._gradient_layers(mx, baseline_flat)
                baseline_path = run / 'LORA_SAMPLE.safetensors'
                save_tensors(mx, baseline_path, {name: baseline_flat[name] for name in SAMPLE_NAMES})
                del gradient, baseline_flat, baseline, baseline_n; gc.collect(); mx.clear_cache()
                boundary('before_tapped_ad')
                phase['name'] = 'tapped_ad'
                with tap.capture_attention_outputs(mx, model, qwen3_next.Qwen3NextAttention,
                        owned_pid=os.getpid(), expected_layers=expected_layers) as (records, tap_receipt):
                    (tapped, tapped_n), tapped_gradient = grad_fn(batch, lengths)
                    mx.eval(tapped, tapped_n, tapped_gradient)
                    tapped_value, tapped_count = float(tapped.item()), int(tapped_n.item())
                    once(run / 'TAPPED_AD_RAW.json', scalar_record(tapped, tapped_n))
                    tapped_flat = dict(tree_flatten(tapped_gradient))
                    old._validate_gradient_inventory(mx, tapped_flat, state['expected'])
                    tapped_path = run / 'TAPPED_LORA_SAMPLE.safetensors'
                    save_tensors(mx, tapped_path, {name: tapped_flat[name] for name in SAMPLE_NAMES})
                    sample_comparison = compare_gradients(baseline_path, tapped_path, expected_sample, TOLERANCES)
                    tapped_primal = loss_equivalence(baseline_value, tapped_value, baseline_count, tapped_count, TOLERANCES)
                    once(run / 'TAP_NUMERICAL_COMPARISON.json', {'loss': tapped_primal, 'lora_sample': sample_comparison})
                    if not sample_comparison['passed'] or not tapped_primal['passed']:
                        raise ValueError('Attention tap changed the actual loss/LoRA sample')
                    by_layer = tap.validate_capture_inventory(records, expected_layers)
                    for record in by_layer.values():
                        mx.eval(record['input'], record['attention_output'], record['cotangent'])
                    # The actual tapped values are now materialized; discard
                    # the full 992-gradient tree before per-layer rebuilding.
                    del tapped_gradient, tapped_flat, tapped, tapped_n
                    gc.collect(); mx.clear_cache()
                    qk = []
                    worst = None
                    for index in expected_layers:
                        boundary('before_reconstruct_layer_' + str(index))
                        phase['name'] = 'reconstruct_' + str(index)
                        values = tap.reconstruct_qkv_and_output_cotangent(mx, by_layer[index],
                                                                         qwen3_next.scaled_dot_product_attention)
                        maxima = tap.qk_maxima(mx, values['q'], values['k'])
                        qk.append({'layer': index, **maxima})
                        append(run / 'QK_PARTIAL.jsonl', qk[-1])
                        key = (maxima['max_unscaled_qk_absolute_causal'], -index)
                        if worst is None or key > worst[0]: worst = (key, values)
                    if worst is None or len(qk) != 16: raise ValueError('All 16 attention layers required')
                    selected = worst[1]
                    fixture = run / 'ATTENTION_FIXTURE.safetensors'
                    save_tensors(mx, fixture, {key: selected[key] for key in ('q', 'k', 'v', 'cotangent')})
                    phase['name'] = 'fused_tensor_vjp'
                    def fused(q, k, v):
                        return mx.fast.scaled_dot_product_attention(q, k, v, scale=.0625, mask='causal', force_fused=True)
                    output, gradients = mx.vjp(fused, [selected['q'], selected['k'], selected['v']], [selected['cotangent']])
                    mx.eval(output, gradients)
                    if len(gradients) != 3:
                        raise ValueError('Three-component fused attention VJP required')
                    fused_finite = [bool(mx.all(mx.isfinite(value)).item()) for value in gradients]
                    fused_path = run / 'FUSED_VJP.safetensors'
                    save_tensors(mx, fused_path, dict(zip(('dq', 'dk', 'dv'), gradients)))
                    qk_path = run / 'QK_SUMMARY.json'
                    once(qk_path, {'schema': 'research4.qwen-actual-qk-maxima.v1', 'layers': qk,
                        'worst_layer': selected['layer'], 'criterion': 'maximum unscaled causal QK over all 640 input positions',
                        'sequence_note': '590 semantic frozen tokens padded to 640 model input positions',
                        'recorded_token_text_or_ids': False})
                    # Fixed high-score toy is a separate switch-discrimination
                    # control, never Qwen data or scientific training.
                    toy = synthetic.build(mx)
                    toy_score = synthetic.causal_score_max(mx, toy)
                    if not math.isfinite(toy_score) or toy_score < 10000:
                        raise ValueError('High-score synthetic control is not discriminative')
                    toy_fixture = run / 'SYNTHETIC_FIXTURE.safetensors'
                    save_tensors(mx, toy_fixture, toy)
                    def toy_fused(q, k, v):
                        return mx.fast.scaled_dot_product_attention(q, k, v, scale=.125, mask='causal', force_fused=True)
                    toy_out, toy_gradient = mx.vjp(toy_fused, [toy['q'], toy['k'], toy['v']], [toy['cotangent']])
                    mx.eval(toy_out, toy_gradient)
                    if len(toy_gradient) != 3: raise ValueError('Three synthetic VJP components required')
                    toy_finite = [bool(mx.all(mx.isfinite(value)).item()) for value in toy_gradient]
                    toy_fused_path = run / 'SYNTHETIC_FUSED_VJP.safetensors'
                    save_tensors(mx, toy_fused_path, dict(zip(('dq', 'dk', 'dv'), toy_gradient)))
                if not tap_receipt['attention_class_restored']:
                    raise ValueError('Attention output tap was not restored')
                once(run / 'TAP_RECEIPT.json', tap_receipt)
                del gradients, output, records, by_layer
                gc.collect(); mx.clear_cache()
        finally:
            if policy is not None: once(run / 'FORCE_POLICY.json', policy)
        if (policy is None or policy.get('off_scope_calls') != 0 or not policy.get('restoration_verified')
                or not policy.get('native_base_and_fast_untouched') or not policy.get('environment_unchanged')):
            raise ValueError('Original scoped forced alias did not restore')
        phase_counts = {}
        for call in policy.get('calls', []):
            phase_name = call.get('phase')
            phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
            if (call.get('q_shape') != [1, 24, 640, 256] or call.get('k_shape') != [1, 4, 640, 256]
                    or call.get('v_shape') != [1, 4, 640, 256] or call.get('force_fused_requested') is not True):
                raise ValueError('Forced Qwen attention geometry/request differs')
        if (sum(phase_counts.values()) != policy.get('matched_calls')
                or any(phase_counts.get(name, 0) < 16 for name in ('untapped_direct', 'untapped_ad', 'tapped_ad'))):
            raise ValueError('Each Qwen phase must request forced alias for all attention blocks')
        after = repeat.full_state_snapshot(mx, model, tree_flatten, boundary)
        after_flags, after_rng = repeat.flags_snapshot(model), repeat.rng_snapshot(mx)
        once(run / 'FULL_STATE_AFTER.json', after)
        mx.eval(optimizer.state)
        if (before['sha256'] != after['sha256'] or before['inventory'] != after['inventory']
                or before_flags != after_flags or before_rng != after_rng or old.parameter_digest(model) != before_theta
                or int(optimizer.step.item()) != 0):
            raise ValueError('Full state, flags, RNG, trainable theta or Adam step changed')
        old._boundary(args, run, 'capture_before_base_recheck')
        state['local'].verify()
        for rel, digest in spec['source_files'].items():
            if sha(PROJECT / rel) != digest: raise ValueError('Source changed during capture')
        receipt = {'schema': 'research4.all64-sdpa-vjp-probe-receipt.v1', 'phase': 'capture_fused',
            'status': 'captured_not_qualified', 'owner': args.owner, 'rank': 4, 'manifest_sha256': MANIFEST,
            'fixture': reference(fixture), 'fused_vjp': reference(fused_path), 'lora_sample': reference(baseline_path),
            'synthetic_fixture': reference(toy_fixture), 'synthetic_fused_vjp': reference(toy_fused_path),
            'synthetic_score_max': toy_score, 'synthetic_scale': .125,
            'fused_vjp_finite_by_component': dict(zip(('dq', 'dk', 'dv'), fused_finite)),
            'synthetic_fused_vjp_finite_by_component': dict(zip(('dq', 'dk', 'dv'), toy_finite)),
            'qk_summary': reference(qk_path), 'tap_receipt': reference(run / 'TAP_RECEIPT.json'),
            'policy_receipt': reference(run / 'FORCE_POLICY.json'), 'worst_layer': selected['layer'],
            'forced_phase_counts': phase_counts,
            'row_id': CASE_ID, 'token_u32be_sha256': CASE_TOKEN_SHA, 'semantic_tokens': 590,
            'input_tokens': 640, 'target_tokens': 30, 'ceiling_padding': True,
            'untapped_tapped_parity_passed': True, 'full_native_primal_passed': primal['passed'],
            'untapped_loss': baseline_value, 'tapped_loss': tapped_value,
            'lora_sample_names': SAMPLE_NAMES, 'lora_sample_shapes': {k: list(v) for k, v in expected_sample.items()},
            'theta_sha256': before_theta, 'full_state_before_sha256': before['sha256'],
            'full_state_after_sha256': after['sha256'], 'rng_before_sha256': before_rng,
            'rng_after_sha256': after_rng, 'training_flags_before': before_flags, 'training_flags_after': after_flags,
            'optimizer_constructed_by_original_loader': True, 'optimizer_step_before': 0, 'optimizer_step_after': 0,
            'adam_updates': 0, 'qualification_claim': False, 'model_execution': True,
            'scope': 'Observed requested alias and VJP values; not kernel profiling or all-layer gradient correctness.'}
        once(run / 'worker_receipt.json', receipt)
    finally:
        handle.close()
        type(model.layers[0]).__call__ = original_call


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('capture_fused',), required=True)
    parser.add_argument('--run-dir', required=True); parser.add_argument('--owner', required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--spec', required=True); parser.add_argument('--spec-sha256', required=True)
    args = parser.parse_args()
    try: capture(args)
    except BaseException as exc:
        folder = Path(args.run_dir).resolve(); path = folder / 'worker_failure.json'
        if folder.is_relative_to((PROJECT / '.local').resolve()) and folder.is_dir() and not path.exists():
            once(path, {'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc), 'epoch': time.time()})
        raise
