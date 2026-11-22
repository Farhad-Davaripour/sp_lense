"""Owned, no-update fallback comparison against captured exact Qwen attention.

The supervisor must start this as a fresh process with
MLX_SDPA_VJP_FALLBACK=1 before Python imports MLX. This file is a diagnostic,
not a training worker or an authorization to change the training recipe.
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
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = 'dc1d812ba3e2b2aa147b1fbe148440ff6a5327a5a67feb8d32f330883ac98f70'
TOKEN_SHA = '9fc914b962fa2cbb96f70f31c32e12aa5b26dc99ee0272158c36353c1bd8cf93'
SUPERVISOR = ROOT / 'mac_studio/all64_sdpa_vjp_supervision_v3_20261119/supervisor.py'
OLD_WORKER = ROOT / 'mac_studio/all64_broader_v2_20261119/worker.py'
OLD_SHA = 'b4996bc66695ff8abda70cc78764875d7cdfadcb1be1d7666b985fd6f68e938e'
LOCK = ROOT / '.local/validation/qwen38_shared_v1/inference.lock/owner.json'
SAMPLE_NAMES = (
    'language_model.model.layers.0.linear_attn.in_proj_qkv.lora_b',
    'language_model.model.layers.3.self_attn.q_proj.lora_b',
    'language_model.model.layers.3.self_attn.k_proj.lora_b',
    'language_model.model.layers.3.self_attn.v_proj.lora_b',
    'language_model.model.layers.3.self_attn.o_proj.lora_b',
)


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('Regular file required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('Regular JSON required')
    return json.loads(path.read_text(encoding='utf-8'))


def write_once(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'w', encoding='utf8') as out:
        json.dump(value, out, indent=2, allow_nan=False); out.write('\n'); out.flush(); os.fsync(out.fileno())


def save_tensors_once(mx, path, values):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    os.close(fd)
    mx.save_safetensors(str(path), values)
    with path.open('rb') as stream:
        os.fsync(stream.fileno())


def ref(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate_owner(authority, lease, *, owner, supervisor_pid, deadline_epoch, spec_sha256, worker_sha256):
    expected = {'schema': 'research4.all64-sdpa-vjp-probe-authority.v1', 'phase': 'fallback_reference',
        'rank': 4, 'owner': owner, 'supervisor_pid': supervisor_pid, 'deadline_epoch': deadline_epoch,
        'manifest_sha256': MANIFEST, 'worker_sha256': worker_sha256, 'spec_sha256': spec_sha256}
    if authority != expected:
        raise ValueError('Exact parent/worker authority differs')
    if (lease.get('owner'), lease.get('pid'), lease.get('kind')) != (owner, supervisor_pid, 'all64_sdpa_vjp_probe_v1'):
        raise ValueError('Canonical shared lease differs')


def admission(args):
    # This check precedes every MLX import. A mere later environment mutation
    # cannot establish which process-level VJP route was initialized.
    if os.environ.get('MLX_SDPA_VJP_FALLBACK') != '1' or 'mlx' in sys.modules:
        raise ValueError('Fallback must be set before MLX import in a fresh process')
    if any(key in os.environ for key in ('MLX_ENABLE_TF32', 'MLX_METAL_GPU_ARCH', 'MLX_DISABLE_COMPILE')):
        raise ValueError('Unreviewed numerical environment override')
    run = Path(args.run_dir).resolve(); spec_path = Path(args.spec).resolve()
    if (not run.is_dir() or not run.is_relative_to((ROOT / '.local').resolve())
            or not spec_path.is_relative_to((ROOT / '.local').resolve())
            or 'onedrive' in str(run).lower() or sha(spec_path) != args.spec_sha256):
        raise ValueError('Exact owned non-OneDrive run and spec required')
    spec = read(spec_path)
    if (args.phase != 'fallback_reference' or spec.get('phase') != args.phase
            or spec.get('owner') != args.owner or spec.get('rank') != 4
            or spec.get('deadline_epoch') != args.deadline_epoch or spec.get('manifest_sha256') != MANIFEST
            or not re.fullmatch('[0-9a-f]{32}', args.owner)
            or not math.isfinite(args.deadline_epoch) or time.time() >= args.deadline_epoch):
        raise ValueError('Exact fallback authority or original deadline differs')
    sup = load_module('_vjp_supervisor_for_fallback', SUPERVISOR)
    sup.validate_spec(spec)
    validate_owner(read(run / 'worker_authority.json'), read(LOCK), owner=args.owner,
                   supervisor_pid=os.getppid(), deadline_epoch=args.deadline_epoch,
                   spec_sha256=args.spec_sha256, worker_sha256=sha(__file__))
    source_pins = spec['source_files']
    for path in (Path(__file__), ROOT / 'mac_studio/all64_sdpa_vjp_probe_v1_20261119/fallback_compare.py',
                 ROOT / 'mac_studio/all64_sdpa_vjp_probe_v1_20261119/numpy_gqa_oracle.py',
                 ROOT / 'mac_studio/all64_sdpa_vjp_probe_v1_20261119/gqa_reference.py',
                 ROOT / 'mac_studio/all64_sdpa_vjp_probe_v1_20261119/scoped_fallback.py',
                 ROOT / 'mac_studio/all64_primal_order_diagnostic_v2_20261119/repeat_observer.py'):
        if source_pins.get(path.relative_to(ROOT).as_posix()) != sha(path):
            raise ValueError('Fallback source is not exactly pinned')
    if sha(OLD_WORKER) != OLD_SHA:
        raise ValueError('Original reviewed model loader differs')
    closeout, capture = sup.validate_capture(spec['capture_closeout'], spec['fixture'])
    capture_root = sup.local(sup.RUNS + '/' + closeout['owner'])
    artifacts = {}
    for key in ('fixture', 'fused_vjp', 'lora_sample', 'qk_summary', 'synthetic_fixture', 'synthetic_fused_vjp'):
        artifacts[key] = sup.artifact(capture.get(key), capture_root)
    return run, spec, capture, artifacts


def numpy_f32(mx, value):
    import numpy as np
    return np.asarray(value.astype(mx.float32))


def validate_target_counts(direct_count, primal_count):
    if type(direct_count) is not int or type(primal_count) is not int or direct_count != 30 or primal_count != 30:
        raise ValueError('The frozen case requires exactly 30 direct and AD targets')


def attention_vjp(mx, fixture):
    if set(fixture) != {'q', 'k', 'v', 'cotangent'}:
        raise ValueError('Exact captured attention inputs required')
    q, k, v, cotangent = (fixture[key] for key in ('q', 'k', 'v', 'cotangent'))
    if any(value.dtype != mx.bfloat16 for value in (q, k, v, cotangent)):
        raise ValueError('Captured native attention inputs must be BF16')
    output, grads = mx.vjp(lambda a, b, c: mx.fast.scaled_dot_product_attention(
        a, b, c, scale=0.0625, mask='causal', force_fused=True), [q, k, v], [cotangent])
    mx.eval(output, grads)
    if len(grads) != 3: raise ValueError('Attention VJP inventory differs')
    return {'dq': grads[0], 'dk': grads[1], 'dv': grads[2]}


def prepare_module_paths():
    for path in (ROOT / 'mac_studio', Path(__file__).parent,
                 ROOT / 'mac_studio/all64_broader_v2_20261119'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def run_fallback(args):
    run, spec, capture, paths = admission(args)
    prepare_module_paths()
    import numpy as np
    import mlx.core as mx
    from mlx.utils import tree_flatten
    import frozen_data, loss, scoped_fallback
    from fallback_compare import compare_fixture, array_comparison
    from gqa_reference import ordinary_gqa, ordinary_gqa_probabilities
    from game_selected_head_v3_compare import loss_equivalence, TOLERANCES
    from numpy_gqa_oracle import causal_gqa_vjp
    old = load_module('_pinned_original_worker_for_fallback', OLD_WORKER)
    old.authenticate, old.render_inputs = frozen_data.authenticate, frozen_data.render_inputs
    old.geometry, old.make_batch, old.selected_suffix_loss = loss.geometry, loss.make_batch, loss.selected_suffix_loss
    args.rank = 4
    old._boundary(args, run, 'fallback_before_attention_fixture', 'launch')
    fixture = mx.load(str(paths['fixture'])); fused = mx.load(str(paths['fused_vjp']))
    fallback = attention_vjp(mx, fixture)
    save_tensors_once(mx, run / 'FALLBACK_ATTENTION_VJP.safetensors', fallback)
    if set(fused) != {'dq', 'dk', 'dv'}: raise ValueError('Exact fused VJP inventory required')
    q, k, v, cotangent = (fixture[key] for key in ('q', 'k', 'v', 'cotangent'))
    ordinary_out, ordinary_grads = mx.vjp(
        lambda a, b, c: ordinary_gqa(mx, a, b, c, .0625, precision='bf16'),
        [q, k, v], [cotangent])
    probabilities = ordinary_gqa_probabilities(mx, q, k, .0625, precision='bf16')
    row_sums = mx.sum(probabilities.astype(mx.float32), axis=-1)
    row_min, row_max = mx.min(row_sums), mx.max(row_sums)
    mx.eval(ordinary_out, ordinary_grads, row_min, row_max)
    if len(ordinary_grads) != 3: raise ValueError('Three ordinary BF16 VJP components required')
    ordinary = dict(zip(('dq', 'dk', 'dv'), ordinary_grads))
    save_tensors_once(mx, run / 'ORDINARY_BF16_ATTENTION_VJP.safetensors', ordinary)
    ordinary_rows = {'method': 'ordinary_ops_bf16', 'rows': int(row_sums.size),
        'minimum': float(row_min.item()), 'maximum': float(row_max.item()),
        'fused_internal_row_sums': 'not_directly_observed'}
    if not math.isfinite(ordinary_rows['minimum']) or not math.isfinite(ordinary_rows['maximum']):
        raise ValueError('Ordinary BF16 reference row sums are not finite')
    attention = compare_fixture(*(numpy_f32(mx, fixture[k]) for k in ('q', 'k', 'v', 'cotangent')),
        {k: numpy_f32(mx, fused[k]) for k in ('dq', 'dk', 'dv')},
        {k: numpy_f32(mx, fallback[k]) for k in ('dq', 'dk', 'dv')},
        ordinary={k: numpy_f32(mx, ordinary[k]) for k in ('dq', 'dk', 'dv')})
    write_once(run / 'ATTENTION_COMPARISON.json', {'fixture': spec['fixture'],
        'fused_vjp': capture['fused_vjp'], 'fallback_vjp': ref(run / 'FALLBACK_ATTENTION_VJP.safetensors'),
        'ordinary_bf16_vjp': ref(run / 'ORDINARY_BF16_ATTENTION_VJP.safetensors'),
        'ordinary_bf16_row_sums': ordinary_rows, 'comparison': attention,
        'scope': 'One Qwen layer and actual cotangent; ordinary BF16 is primary, F64 is diagnostic.'})
    # The synthetic stress fixture is the same captured bytes in both fresh
    # processes. The result characterizes switch behavior; no hidden model data
    # or tensor values are printed in the report.
    toy = mx.load(str(paths['synthetic_fixture'])); toy_fused = mx.load(str(paths['synthetic_fused_vjp']))
    if set(toy) != {'q', 'k', 'v', 'cotangent'} or set(toy_fused) != {'dq', 'dk', 'dv'}:
        raise ValueError('Exact discriminative toy fixture required')
    tq, tk, tv, td = (toy[k] for k in ('q', 'k', 'v', 'cotangent'))
    if any(value.dtype != mx.bfloat16 for value in (tq, tk, tv, td)):
        raise ValueError('Toy inputs must be BF16')
    try:
        toy_out, toy_grad = mx.vjp(lambda a,b,c: mx.fast.scaled_dot_product_attention(
            a,b,c,scale=.125,mask='causal',force_fused=True), [tq,tk,tv], [td])
        toy_ordinary_out, toy_ordinary_grad = mx.vjp(
            lambda a,b,c: ordinary_gqa(mx, a, b, c, .125, precision='bf16'), [tq,tk,tv], [td])
        mx.eval(toy_out, toy_grad, toy_ordinary_out, toy_ordinary_grad)
        save_tensors_once(mx, run / 'SYNTHETIC_FALLBACK_VJP.safetensors',
                          dict(zip(('dq', 'dk', 'dv'), toy_grad)))
        save_tensors_once(mx, run / 'SYNTHETIC_ORDINARY_BF16_VJP.safetensors',
                          dict(zip(('dq', 'dk', 'dv'), toy_ordinary_grad)))
        _, toy_oracle = causal_gqa_vjp(*(numpy_f32(mx, toy[k]) for k in ('q','k','v','cotangent')), .125)
        toy_result = {'status': 'observed', 'components': {
            key: {'fused_vs_f64': array_comparison(expected, numpy_f32(mx, toy_fused[key])),
                  'fallback_vs_f64': array_comparison(expected, numpy_f32(mx, actual)),
                  'ordinary_bf16_vs_f64': array_comparison(expected, numpy_f32(mx, ordinary_actual)),
                  'fallback_vs_ordinary_bf16': array_comparison(numpy_f32(mx, ordinary_actual), numpy_f32(mx, actual))}
            for key, expected, actual, ordinary_actual in zip(('dq','dk','dv'), toy_oracle, toy_grad, toy_ordinary_grad)}}
    except (RuntimeError, ValueError) as exc:
        # Preserve the actual Qwen attention comparison even if a differently
        # shaped synthetic switch control is unsupported or inconclusive.
        toy_result = {'status': 'inconclusive', 'error_type': type(exc).__name__}
    write_once(run / 'SYNTHETIC_SWITCH_COMPARISON.json', toy_result)
    del fixture, fused, fallback, ordinary, ordinary_out, ordinary_grads, probabilities, row_sums, toy, toy_fused
    q = k = v = cotangent = tq = tk = tv = td = None
    toy_out = toy_grad = toy_ordinary_out = toy_ordinary_grad = toy_oracle = None
    row_min = row_max = None
    gc.collect(); mx.clear_cache()
    old._boundary(args, run, 'fallback_before_model_load')
    state = old._load_runtime(args, run)
    model, optimizer, handle = (state[key] for key in ('model', 'optimizer', 'handle'))
    original_call = state['original_call']
    try:
        repeat = load_module('_repeat_observer_for_fallback',
            ROOT / 'mac_studio/all64_primal_order_diagnostic_v2_20261119/repeat_observer.py')
        item = old.qualification_cases(state['prepared'])[-1]
        row_id, ids, prefix, ceiling = item
        token_sha = hashlib.sha256(b''.join(struct.pack('>I', value) for value in ids)).hexdigest()
        if (row_id, token_sha, len(ids), prefix, ceiling) != (
                'sustained_multi_incident_02.12', TOKEN_SHA, 590, 560, True):
            raise ValueError('Frozen maximum-width case differs')
        batch, lengths, geometry = loss.make_batch(mx, ids, prefix, ceiling)
        if (geometry['model_input_width'], geometry['target_tokens']) != (640, 30):
            raise ValueError('Maximum-width case geometry differs')
        theta_before = old.parameter_digest(model)
        boundary = lambda label: old._boundary(args, run, 'fallback_state_' + label)
        full_before = repeat.full_state_snapshot(mx, model, tree_flatten, boundary)
        flags_before, rng_before = repeat.flags_snapshot(model), repeat.rng_snapshot(mx)
        write_once(run / 'FULL_STATE_BEFORE.json', full_before)
        if (theta_before != capture.get('theta_sha256') or full_before['sha256'] != capture.get('full_state_before_sha256')
                or flags_before != capture.get('training_flags_before') or rng_before != capture.get('rng_before_sha256')):
            raise ValueError('Fused and fallback LoRA starting states differ')
        mx.eval(optimizer.state)
        if int(optimizer.step.item()) != 0: raise ValueError('Optimizer must remain at step zero')
        phase = {'name': 'fallback_direct'}
        environment = scoped_fallback.environment_snapshot()
        policy = None
        try:
            with scoped_fallback.scoped_sdpa_policy(owned_process_pid=os.getpid(), policy='force_fused',
                    phase=phase, expected_environment=environment) as policy:
                direct, direct_n = old._full_native_reference(mx, state['nn'], model, batch, lengths)
                mx.eval(direct, direct_n)
                direct_value, direct_count = float(direct.item()), int(direct_n.item())
                del direct, direct_n; gc.collect(); mx.clear_cache()
                old._boundary(args, run, 'fallback_before_full_ad')
                phase['name'] = 'fallback_ad'
                grad_fn = state['nn'].value_and_grad(model, lambda b, l: old._full_native_reference(mx, state['nn'], model, b, l))
                (primal, primal_n), gradients = grad_fn(batch, lengths)
                mx.eval(primal, primal_n, gradients)
                primal_value, primal_count = float(primal.item()), int(primal_n.item())
                validate_target_counts(direct_count, primal_count)
                flat = dict(tree_flatten(gradients)); fused_lora = mx.load(str(paths['lora_sample']))
                if set(fused_lora) != set(SAMPLE_NAMES) or not set(SAMPLE_NAMES).issubset(flat):
                    raise ValueError('Fixed LoRA gradient sample inventory differs')
                save_tensors_once(mx, run / 'FALLBACK_LORA_SAMPLE.safetensors',
                                  {name: flat[name] for name in SAMPLE_NAMES})
                lora = {}
                for name in SAMPLE_NAMES:
                    reference, observed = numpy_f32(mx, fused_lora[name]), numpy_f32(mx, flat[name])
                    if not np.isfinite(reference).all():
                        lora[name] = {'available': False, 'reason': 'fused_reference_nonfinite',
                                      'fused_nonfinite_count': int(np.count_nonzero(~np.isfinite(reference))),
                                      'fallback_nonfinite_count': int(np.count_nonzero(~np.isfinite(observed)))}
                    else:
                        lora[name] = array_comparison(reference, observed)
                primal_check = loss_equivalence(direct_value, primal_value, direct_count, primal_count, TOLERANCES)
        finally:
            if policy is not None:
                write_once(run / 'FALLBACK_POLICY.json', policy)
        theta_after = old.parameter_digest(model)
        full_after = repeat.full_state_snapshot(mx, model, tree_flatten, boundary)
        flags_after, rng_after = repeat.flags_snapshot(model), repeat.rng_snapshot(mx)
        write_once(run / 'FULL_STATE_AFTER.json', full_after)
        mx.eval(optimizer.state)
        if (theta_after != theta_before or full_after['sha256'] != full_before['sha256']
                or full_after['inventory'] != full_before['inventory'] or flags_after != flags_before
                or rng_after != rng_before or int(optimizer.step.item()) != 0):
            raise ValueError('No-update fallback test changed model state or optimizer')
        if (policy.get('restoration_verified') is not True or policy.get('off_scope_calls') != 0
                or not policy.get('native_base_and_fast_untouched') or not policy.get('environment_unchanged')):
            raise ValueError('Owned fallback policy restoration failed')
        old._boundary(args, run, 'fallback_after_qwen')
        state['local'].verify()
        for rel, expected_sha in spec['source_files'].items():
            if sha(ROOT / rel) != expected_sha:
                raise ValueError('Fallback source changed during diagnostic')
        result = {'schema': 'research4.all64-sdpa-fallback-observation.v1',
            'fixture': spec['fixture'], 'captured_fused_vjp': capture['fused_vjp'],
            'fallback_attention_vjp': ref(run / 'FALLBACK_ATTENTION_VJP.safetensors'),
            'ordinary_bf16_attention_vjp': ref(run / 'ORDINARY_BF16_ATTENTION_VJP.safetensors'),
            'ordinary_bf16_row_sums': ordinary_rows, 'primary_attention_reference': 'ordinary_ops_bf16',
            'row_id': row_id, 'token_u32be_sha256': TOKEN_SHA, 'semantic_tokens': 590,
            'input_tokens': 640, 'target_tokens': 30, 'ceiling_padding': True,
            'environment': environment, 'fallback_requested_before_mlx_import': True,
            'attention_vjp_comparison': attention, 'synthetic_switch_control': toy_result,
            'lora_gradient_sample_vs_fused': lora, 'direct_vs_ad': primal_check,
            'theta_before_sha256': theta_before, 'theta_after_sha256': theta_after,
            'full_state_before_sha256': full_before['sha256'], 'full_state_after_sha256': full_after['sha256'],
            'rng_before_sha256': rng_before, 'rng_after_sha256': rng_after,
            'training_flags_before': flags_before, 'training_flags_after': flags_after,
            'fallback_lora_sample': ref(run / 'FALLBACK_LORA_SAMPLE.safetensors'),
            'synthetic_fallback_vjp': (ref(run / 'SYNTHETIC_FALLBACK_VJP.safetensors')
                                       if (run / 'SYNTHETIC_FALLBACK_VJP.safetensors').is_file() else None),
            'synthetic_ordinary_bf16_vjp': (ref(run / 'SYNTHETIC_ORDINARY_BF16_VJP.safetensors')
                                            if (run / 'SYNTHETIC_ORDINARY_BF16_VJP.safetensors').is_file() else None),
            'full_state_before': ref(run / 'FULL_STATE_BEFORE.json'),
            'full_state_after': ref(run / 'FULL_STATE_AFTER.json'),
            'policy_receipt': ref(run / 'FALLBACK_POLICY.json'),
            'adam_updates': 0, 'optimizer_step_before': 0, 'optimizer_step_after': 0,
            'policy_restored': True, 'qualification_claim': False,
            'scope': 'One captured layer/cotangent, one frozen case, fixed LoRA subset; F64 comparison is not a whole-model guarantee.'}
        write_once(run / 'FALLBACK_OBSERVATION.json', result)
        write_once(run / 'worker_receipt.json', {'schema': 'research4.all64-sdpa-vjp-probe-receipt.v1',
            'phase': args.phase, 'status': 'observed_not_qualification', 'owner': args.owner,
            'rank': 4, 'manifest_sha256': MANIFEST, 'adam_updates': 0, 'qualification_claim': False,
            'fixture': spec['fixture'], 'observation': {'path': (run / 'FALLBACK_OBSERVATION.json').relative_to(ROOT).as_posix(),
                                                        'sha256': sha(run / 'FALLBACK_OBSERVATION.json')}})
    finally:
        handle.close()
        type(model.layers[0]).__call__ = original_call


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('fallback_reference',), required=True)
    parser.add_argument('--run-dir', required=True); parser.add_argument('--owner', required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--spec', required=True); parser.add_argument('--spec-sha256', required=True)
    run_fallback(parser.parse_args())
