"""Pinned hybrid-model LoRA layouts and bounded CPU saved-file validation.

This is separate from the frozen last-layer worker. No MLX/model code is
imported. Header checks establish structure, not full base-weight identity;
callers must independently verify the complete pinned base manifest.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sysconfig

CONFIG_SHA256 = '14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85'
BASE_MANIFEST_SHA256 = 'e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
BASE_HEADERS = {
    'model-00001-of-00003.safetensors': (5343268662, 'd16c876049004eb2ad58e1826a371d635a82a7ae902b76887a0be2670c7e1982'),
    'model-00002-of-00003.safetensors': (5354185130, '9166a0770932dbfe3ad5ff61dc557870e9f14986e56b568fc049997cc2ed9aab'),
    'model-00003-of-00003.safetensors': (5357087557, '96b0ca9e070b260c9b4d24d67504ebde99eaa94bf8129b71ec2af7304f417e45'),
}
ALLOWED_RANKS = (4, 8, 16)
ALLOWED_LAYER_COUNTS = (1, 4, 8, 16, 64)
MAX_HEADER_BYTES = 16 * 1024**2
CHUNK_BYTES = 1024**2
INSTALLED_SOURCE_SHA256 = {
    'mlx_lm/tuner/utils.py': 'f59f3c027176a3975fe60e7fbd9f3f38723c27c3f888cf9e591c103f93f68588',
    'mlx_lm/tuner/lora.py': '4d3a8edab111d4ddba33398ba8700203db7b61621c39e9c348fdd50e57278b45',
    'mlx_lm/tuner/trainer.py': '0b6027fdb11803e951ceee52c83a81f82daa922ea8c6b7ca6888953461064e8e',
    'mlx_lm/models/qwen3_5.py': '851449fd9989d6bf4b3c4fed92ddc3be5f1e781eb2246f89c54d7eff2518d9c6',
    'mlx_lm/models/qwen3_next.py': '67ca08a4d08cb7be124965e95399735e556f1eaeb21a16bc23cb88d1ac035e11',
    'mlx/optimizers/optimizers.py': 'a649eeb65133d1a21eec61ab83529badd0ec50938ee6476b1e59f7ee36def1f2',
}


def _sha(data): return hashlib.sha256(data).hexdigest()


def _regular_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('Expected regular file: ' + str(path))
    return path


def _pairs(pairs):
    result = {}
    for name, value in pairs:
        if name in result: raise ValueError('Duplicate JSON key')
        result[name] = value
    return result


def verify_installed_lora_sources(site_packages=None):
    """Authenticate reviewed selection/arithmetic source bytes without import."""
    root = Path(site_packages) if site_packages is not None else Path(sysconfig.get_paths()['purelib'])
    found = {}
    for name, expected in INSTALLED_SOURCE_SHA256.items():
        actual = _sha(_regular_file(root / name).read_bytes())
        if actual != expected: raise ValueError('Installed LoRA source differs: ' + name)
        found[name] = actual
    return {'status': 'pinned_lora_source_bytes_verified', 'file_sha256': found,
            'scope': 'CPU source read only; no package import, model execution or numerical compatibility test.'}


def read_tensor_header(path):
    """Read only a bounded safetensors header, without importing tensor code."""
    path = _regular_file(path)
    with path.open('rb') as file:
        prefix = file.read(8)
        if len(prefix) != 8: raise ValueError('Truncated safetensors length')
        size = struct.unpack('<Q', prefix)[0]
        if not 2 <= size <= MAX_HEADER_BYTES: raise ValueError('Safetensors header outside bounds')
        raw = file.read(size)
        if len(raw) != size: raise ValueError('Truncated safetensors header')
    header = json.loads(raw, object_pairs_hook=_pairs)
    if not isinstance(header, dict): raise ValueError('Safetensors header must be an object')
    # Native MLX saves an explicit null when no metadata was supplied.
    # It is not tensor data and is also accepted by the native loader.
    metadata = header.get('__metadata__')
    if metadata is not None and (not isinstance(metadata, dict)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items())):
        raise ValueError('Safetensors metadata must contain strings')
    return header, 8 + size, prefix + raw


def _dimensions(config):
    """Native qwen3_5/qwen3_next Linear projections from the pinned source."""
    text = config['text_config']
    if (config.get('model_type'), config.get('quantization', {}).get('bits'), text.get('num_hidden_layers'),
        text.get('full_attention_interval')) != ('qwen3_5', 4, 64, 4):
        raise ValueError('Pinned hybrid architecture differs')
    types = text['layer_types']
    if types != [('full_attention' if (i + 1) % 4 == 0 else 'linear_attention') for i in range(64)]:
        raise ValueError('Pinned hybrid block ordering differs')
    h, mid = text['hidden_size'], text['intermediate_size']
    attention = text['num_attention_heads'] * text['head_dim']
    kv = text['num_key_value_heads'] * text['head_dim']
    key = text['linear_num_key_heads'] * text['linear_key_head_dim']
    value = text['linear_num_value_heads'] * text['linear_value_head_dim']
    if text.get('attn_output_gate') is not True: raise ValueError('Attention output gate differs')
    shared = {'mlp.gate_proj': (h, mid), 'mlp.up_proj': (h, mid), 'mlp.down_proj': (mid, h)}
    full = {**shared, 'self_attn.q_proj': (h, attention * 2), 'self_attn.k_proj': (h, kv),
            'self_attn.v_proj': (h, kv), 'self_attn.o_proj': (attention, h)}
    linear = {**shared, 'linear_attn.in_proj_qkv': (h, key * 2 + value),
              'linear_attn.in_proj_z': (h, value),
              'linear_attn.in_proj_b': (h, text['linear_num_value_heads']),
              'linear_attn.in_proj_a': (h, text['linear_num_value_heads']),
              'linear_attn.out_proj': (value, h)}
    return types, full, linear


def _projection_layout_from(config, headers, num_layers):
    """Internal pure-data function; public entry authenticates exact base headers."""
    if type(num_layers) is not int or num_layers not in ALLOWED_LAYER_COUNTS:
        raise ValueError('Unsupported adapted block count')
    types, full, linear = _dimensions(config)
    projections = {}; expected_weights = set()
    for index, kind in enumerate(types):
        for relative, (inputs, outputs) in (full if kind == 'full_attention' else linear).items():
            module = f'language_model.model.layers.{index}.{relative}'
            expected_weights.add(module + '.weight')
            item = headers.get(module + '.weight')
            if not isinstance(item, dict): raise ValueError('Required base projection missing: ' + module)
            quantized = module + '.scales' in headers
            if item.get('shape') != [outputs, inputs // 8 if quantized else inputs]:
                raise ValueError('Base projection dimensions differ: ' + module)
            if quantized and (inputs % 8 or item.get('dtype') != 'U32'):
                raise ValueError('Base projection packing differs: ' + module)
            if not quantized and item.get('dtype') not in ('F32', 'F16', 'BF16'):
                raise ValueError('Base projection dtype differs: ' + module)
            if index >= 64 - num_layers:
                projections[module] = {'input_dim': inputs, 'output_dim': outputs, 'layer_index': index,
                                       'layer_type': kind, 'base_quantized': quantized,
                                       'packed_weight_dtype': item['dtype'], 'packed_weight_shape': item['shape']}
    extras = [key for key, value in headers.items()
              if re.match(r'^language_model\.model\.layers\.\d+\.', key)
              and key.endswith('.weight') and isinstance(value, dict)
              and len(value.get('shape', [])) == 2 and key not in expected_weights]
    if extras: raise ValueError('Unaccounted native layer projection weights: ' + str(extras))
    return projections


def projection_layout(model_dir, num_layers=64):
    """Return authenticated structural module paths; does not instantiate modules."""
    model_dir = Path(model_dir)
    config_path = _regular_file(model_dir / 'config.json')
    raw = config_path.read_bytes()
    if _sha(raw) != CONFIG_SHA256: raise ValueError('Pinned base config hash differs')
    config = json.loads(raw, object_pairs_hook=_pairs)
    if {p.name for p in model_dir.glob('*.safetensors')} != set(BASE_HEADERS):
        raise ValueError('Base safetensors shard inventory differs')
    headers = {}
    for name, (size, expected_sha) in BASE_HEADERS.items():
        path = _regular_file(model_dir / name)
        header, _, prefix_and_header = read_tensor_header(path)
        if path.stat().st_size != size or _sha(prefix_and_header[8:]) != expected_sha:
            raise ValueError('Pinned base header binding differs: ' + name)
        for key, value in header.items():
            if key == '__metadata__': continue
            if key in headers: raise ValueError('Duplicate base tensor across shards')
            headers[key] = value
    return _projection_layout_from(config, headers, num_layers)


def _expected_from_projections(projections, rank):
    if type(rank) is not int or rank not in ALLOWED_RANKS: raise ValueError('Unsupported LoRA rank')
    result = {}
    for module, info in projections.items():
        result[module + '.lora_a'] = (info['input_dim'], rank)
        result[module + '.lora_b'] = (rank, info['output_dim'])
    return result


def expected_adapter_layout(model_dir, rank, num_layers=64):
    return _expected_from_projections(projection_layout(model_dir, num_layers), rank)


def verify_adapter_tensor_file(path, expected_shapes):
    """Stream exact F32 adapter payloads in ≤1MiB chunks and reject NaN/Inf.

    NumPy processes CPU bytes only. The whole adapter is never materialized at
    once, avoiding a second hundreds-of-MiB adapter copy after training.
    """
    import numpy as np
    path = _regular_file(path)
    header, data_start, prefix = read_tensor_header(path)
    tensors = {key: value for key, value in header.items() if key != '__metadata__'}
    if set(tensors) != set(expected_shapes): raise ValueError('Saved adapter tensor inventory differs')
    intervals = []
    for name, shape in expected_shapes.items():
        item = tensors[name]
        if (not isinstance(item, dict) or set(item) != {'dtype', 'shape', 'data_offsets'}
            or item['dtype'] != 'F32' or item['shape'] != list(shape)):
            raise ValueError('Saved adapter tensor dtype/shape differs: ' + name)
        offsets = item['data_offsets']
        if (not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int for v in offsets)
            or offsets[0] < 0 or offsets[1] - offsets[0] != 4 * math.prod(shape)):
            raise ValueError('Saved adapter tensor offsets differ: ' + name)
        intervals.append((offsets[0], offsets[1], name))
    intervals.sort()
    end = 0
    for start, finish, name in intervals:
        if start != end: raise ValueError('Adapter tensor storage has gaps/overlap')
        end = finish
    if path.stat().st_size != data_start + end: raise ValueError('Saved adapter length differs')
    digest = hashlib.sha256(prefix); nonzero_b = []
    with path.open('rb') as file:
        file.seek(data_start)
        for start, finish, name in intervals:
            remaining = finish - start; has_nonzero = False
            while remaining:
                data = file.read(min(CHUNK_BYTES, remaining))
                if not data or len(data) % 4: raise ValueError('Truncated adapter payload')
                values = np.frombuffer(data, dtype='<f4')
                if not np.isfinite(values).all(): raise ValueError('Nonfinite saved tensor: ' + name)
                has_nonzero = has_nonzero or bool(np.count_nonzero(values))
                digest.update(data); remaining -= len(data)
            if name.endswith('.lora_b') and has_nonzero: nonzero_b.append(name)
    if not nonzero_b: raise ValueError('All LoRA B tensors remain zero')
    nonzero_layers = sorted({int(match.group(1)) for name in nonzero_b
                             if (match := re.search(r'(?:^|\.)layers\.(\d+)\.', name))})
    return {'adapter_sha256': digest.hexdigest(), 'tensor_count': len(tensors),
            'trainable_parameters': sum(math.prod(shape) for shape in expected_shapes.values()),
            'all_finite': True, 'dtype': 'F32', 'nonzero_lora_b': True,
            'nonzero_lora_b_tensor_count': len(nonzero_b),
            'nonzero_lora_b_layer_indices': nonzero_layers, 'payload_chunk_bytes': CHUNK_BYTES}


def verify_saved_adapter(adapter_path, config_path, model_dir, rank, num_layers=64,
                         expected_iterations=None, expected_learning_rate=None,
                         require_all_layer_updates=False):
    """Validate exact saved layout/config; no reload inference or quality claim."""
    expected = expected_adapter_layout(model_dir, rank, num_layers)
    config_path = _regular_file(config_path)
    raw = config_path.read_bytes(); config = json.loads(raw, object_pairs_hook=_pairs)
    checks = {'fine_tune_type': 'lora', 'num_layers': num_layers, 'batch_size': 1,
              'max_seq_length': 1024, 'mask_prompt': True, 'resume_adapter_file': None,
              'optimizer': 'adam', 'grad_accumulation_steps': 1,
              'lora_parameters': {'rank': rank, 'dropout': 0.0, 'scale': 2.0}}
    if expected_iterations is not None: checks['iters'] = expected_iterations
    if expected_learning_rate is not None: checks['learning_rate'] = expected_learning_rate
    for key, value in checks.items():
        if key not in config or config[key] != value or (type(value) in (int, bool, float) and isinstance(config[key], bool) != isinstance(value, bool)):
            raise ValueError('Saved adapter config differs: ' + key)
    result = verify_adapter_tensor_file(adapter_path, expected)
    if type(require_all_layer_updates) is not bool: raise ValueError('Layer-coverage requirement must be boolean')
    wanted = list(range(64 - num_layers, 64))
    all_layers_updated = result['nonzero_lora_b_layer_indices'] == wanted
    if require_all_layer_updates and not all_layers_updated:
        raise ValueError('At least one adapted layer has no nonzero LoRA B tensor')
    result.update({'status': 'saved_hybrid_adapter_verified', 'adapter_config_sha256': _sha(raw),
                   'num_layers': num_layers, 'rank': rank, 'base_config_sha256': CONFIG_SHA256,
                   'all_requested_layers_have_nonzero_b': all_layers_updated,
                   'all_layer_update_coverage_required': require_all_layer_updates,
                   'base_manifest_sha256_required_separately': BASE_MANIFEST_SHA256,
                   'full_base_payload_verified_here': False,
                   'scope': 'Exact saved structural/config/finite-byte checks; no live module, reload, memory-fit or game-improvement claim.'})
    return result
