"""Owned identity-VJP attention-output tap; no model loaded on import.

The caller must first authenticate the original attention class/source, model
identity, exact inputs, existing SDPA scope and independent resource guard.
This module never patches the reviewed SDPA alias and never updates weights.
"""
from contextlib import contextmanager
import os
import threading


@contextmanager
def capture_attention_outputs(runtime, model, attention_class, *, owned_pid, expected_layers):
    if owned_pid != os.getpid() or not isinstance(expected_layers, (list, tuple)):
        raise ValueError('Exact owned process and explicit attention-layer inventory required')
    if not expected_layers or len(set(expected_layers)) != len(expected_layers):
        raise ValueError('Distinct expected attention layers required')
    mapping = {}
    for index in expected_layers:
        block = model.layers[index]
        attention = block.self_attn
        if type(attention) is not attention_class or id(attention) in mapping:
            raise ValueError('Exact native attention module inventory required')
        mapping[id(attention)] = index
    original = attention_class.__call__
    owner_thread = threading.get_ident()
    records = []
    receipt = {'schema': 'research4.attention-output-tap.v1', 'owned_pid': owned_pid,
               'expected_layers': list(expected_layers), 'method': 'identity custom-VJP at attention output',
               'attention_class_restored': False, 'sdpa_alias_modified_by_tap': False,
               'optimizer_updates': 0, 'kernel_profile_claimed': False}

    def wrapped(attention, x, mask=None, cache=None):
        if threading.get_ident() != owner_thread or attention_class.__call__ is not wrapped:
            raise RuntimeError('Attention tap ownership changed')
        index = mapping.get(id(attention))
        if index is None or cache is not None or not isinstance(mask, str) or mask != 'causal':
            raise ValueError('Unrecognized attention layer/cache/mask')
        output = original(attention, x, mask=mask, cache=cache)

        @runtime.custom_function
        def identity(value):
            return value

        @identity.vjp
        def identity_vjp(value, cotangent, _output):
            # These are graph values for the actual native attention output
            # and actual downstream loss cotangent. Their numerical validity
            # must be checked after evaluation, before any use or export.
            records.append({'layer': index, 'module': attention, 'input': x,
                            'attention_output': value, 'cotangent': cotangent})
            return cotangent

        return identity(output)

    try:
        attention_class.__call__ = wrapped
        yield records, receipt
    finally:
        unexpected = attention_class.__call__ is not wrapped
        attention_class.__call__ = original
        receipt['attention_class_restored'] = attention_class.__call__ is original
        receipt['vjp_capture_count'] = len(records)
        if unexpected or not receipt['attention_class_restored']:
            raise RuntimeError('Owned attention tap restoration failed')


def validate_capture_inventory(records, expected_layers):
    """Reject ambiguous duplicate/missing traced backward cotangents."""
    layers = [record.get('layer') for record in records]
    if len(layers) != len(expected_layers) or sorted(layers) != sorted(expected_layers):
        raise ValueError('Exactly one actual cotangent per expected layer is required')
    return {record['layer']: record for record in records}


def reconstruct_qkv_and_output_cotangent(runtime, record, scoped_sdpa):
    """Reconstruct pinned Q/K/V and d(SDPA output) from the actual layer tap.

    Q/K/V use the same live module and tapped input.  The original layer output
    must match bit-for-bit after recomputation before its true downstream
    cotangent is propagated through the same gating/output-projection tail.
    The caller controls the previously reviewed owned forced SDPA alias and
    must audit that no parameters or RNG change during this reconstruction.
    """
    attention, x, captured_output, upstream = (record[k] for k in
        ('module', 'input', 'attention_output', 'cotangent'))
    batch, length, _ = x.shape
    if (batch != 1 or length != 640 or attention.num_attention_heads != 24
            or attention.num_key_value_heads != 4 or attention.head_dim != 256 or attention.scale != .0625):
        raise ValueError('Exact 640-input ceiling-padded Qwen attention geometry required')
    projected = attention.q_proj(x)
    q, gate = runtime.split(projected.reshape(batch, length, attention.num_attention_heads, -1), 2, axis=-1)
    gate = gate.reshape(batch, length, -1)
    k, v = attention.k_proj(x), attention.v_proj(x)
    q = attention.q_norm(q).transpose(0, 2, 1, 3)
    k = attention.k_norm(k.reshape(batch, length, attention.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
    v = v.reshape(batch, length, attention.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    q, k = attention.rope(q), attention.rope(k)
    if q.dtype != runtime.bfloat16 or k.dtype != runtime.bfloat16 or v.dtype != runtime.bfloat16:
        raise ValueError('Actual Q/K/V dtype is not the pinned BF16 path')
    z = scoped_sdpa(q, k, v, cache=None, scale=attention.scale, mask='causal')
    def tail(z_value):
        middle = z_value.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return attention.o_proj(middle * runtime.sigmoid(gate))
    output, gradients = runtime.vjp(tail, [z], [upstream])
    reconstructed, dz = output[0], gradients[0]
    same = runtime.array_equal(reconstructed, captured_output)
    finite = runtime.all(runtime.isfinite(dz))
    runtime.eval(q, k, v, dz, reconstructed, captured_output, same, finite)
    if not bool(same.item()) or not bool(finite.item()):
        raise ValueError('Tapped native attention output or its cotangent failed validation')
    if tuple(dz.shape) != tuple(q.shape):
        raise ValueError('Actual d(SDPA output) shape differs from query/output')
    return {'layer': record['layer'], 'q': q, 'k': k, 'v': v, 'cotangent': dz,
            'scale': attention.scale, 'mask': 'causal', 'output_reconstructed_bitwise': True}


def qk_maxima(runtime, q, k):
    """Return unscaled QK maxima using independent CPU float64 arithmetic.

    BF16 values are exactly representable in F32/F64. A single attention
    layer is materialized at a time; the model is not copied wholesale.
    """
    import numpy as np
    if tuple(q.shape) != (1, 24, 640, 256) or tuple(k.shape) != (1, 4, 640, 256):
        raise ValueError('Exact frozen Qwen grouped-query geometry required')
    qf = np.array(q.astype(runtime.float32)).astype(np.float64)
    kf = np.repeat(np.array(k.astype(runtime.float32)).astype(np.float64), 6, axis=1)
    if not np.isfinite(qf).all() or not np.isfinite(kf).all():
        raise ValueError('Captured BF16 Q/K contain nonfinite values')
    absolute = np.abs(np.matmul(qf, kf.transpose(0, 1, 3, 2)))
    positions = np.arange(640)
    causal = positions[:, None] >= positions[None, :]
    semantic_causal = causal & (positions[:, None] < 590) & (positions[None, :] < 590)
    per_head_all = np.max(absolute, axis=(-2, -1))
    per_head_causal = np.max(np.where(causal[None, None, :, :], absolute, 0.0), axis=(-2, -1))
    per_head_semantic = np.max(np.where(semantic_causal[None, None, :, :], absolute, 0.0), axis=(-2, -1))
    return {'max_unscaled_qk_absolute_all': float(per_head_all.max()),
            'max_unscaled_qk_absolute_causal': float(per_head_causal.max()),
            'max_unscaled_qk_absolute_semantic_causal': float(per_head_semantic.max()),
            'per_head_unscaled_qk_absolute_all': per_head_all.tolist()[0],
            'per_head_unscaled_qk_absolute_causal': per_head_causal.tolist()[0],
            'per_head_unscaled_qk_absolute_semantic_causal': per_head_semantic.tolist()[0],
            'score_dtype': 'CPU_numpy_float64_from_actual_bfloat16_qk',
            'semantic_context': '640 input positions include padding after 590 frozen tokens; semantic-causal includes the EOS-position query and is a conservative input bound, not exactly the contributing-loss subset.'}
