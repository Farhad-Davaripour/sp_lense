"""CPU-only hybrid layout and saved-byte checks; no model or MLX import."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'mac_studio/game_all_layers_layout.py'
SPEC = importlib.util.spec_from_file_location('hybrid_layout', PATH)
layout = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(layout)


def config_fixture():
    return {'model_type': 'qwen3_5', 'quantization': {'bits': 4}, 'text_config': {
        'num_hidden_layers': 64, 'full_attention_interval': 4,
        'layer_types': ['full_attention' if (i + 1) % 4 == 0 else 'linear_attention' for i in range(64)],
        'hidden_size': 5120, 'intermediate_size': 17408, 'num_attention_heads': 24,
        'num_key_value_heads': 4, 'head_dim': 256, 'attn_output_gate': True,
        'linear_num_key_heads': 16, 'linear_key_head_dim': 128,
        'linear_num_value_heads': 48, 'linear_value_head_dim': 128}}


def synthetic_headers(config):
    types, full, linear = layout._dimensions(config)
    result = {}
    for index, kind in enumerate(types):
        for name, (inputs, outputs) in (full if kind == 'full_attention' else linear).items():
            name = f'language_model.model.layers.{index}.{name}'
            result[name + '.weight'] = {'dtype': 'U32', 'shape': [outputs, inputs // 8]}
            result[name + '.scales'] = {'dtype': 'BF16', 'shape': [outputs, inputs // 64]}
    return result


def write_tensor_file(path, values, changes=None, extra=b''):
    header = {}; parts = []; start = 0
    for name, (shape, entries) in values.items():
        data = struct.pack('<' + 'f' * len(entries), *entries)
        header[name] = {'dtype': 'F32', 'shape': list(shape), 'data_offsets': [start, start + len(data)]}
        parts.append(data); start += len(data)
    if changes: changes(header)
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * ((-len(raw)) % 8)
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + b''.join(parts) + extra)


class HybridLayoutTests(unittest.TestCase):
    def test_all_hybrid_placements_and_rank_counts(self):
        config = config_fixture(); headers = synthetic_headers(config)
        projections = layout._projection_layout_from(config, headers, 64)
        self.assertEqual(len(projections), 496)
        self.assertEqual(sum(x['layer_type'] == 'linear_attention' for x in projections.values()), 48 * 8)
        self.assertEqual(sum(x['layer_type'] == 'full_attention' for x in projections.values()), 16 * 7)
        for rank, parameters in ((4, 29181952), (8, 58363904), (16, 116727808)):
            expected = layout._expected_from_projections(projections, rank)
            self.assertEqual(len(expected), 992)
            self.assertEqual(sum(a * b for a, b in expected.values()), parameters)
            self.assertEqual(expected['language_model.model.layers.60.linear_attn.in_proj_a.lora_b'], (rank, 48))
            self.assertEqual(expected['language_model.model.layers.60.linear_attn.in_proj_qkv.lora_b'], (rank, 10240))
            self.assertEqual(expected['language_model.model.layers.63.self_attn.q_proj.lora_b'], (rank, 12288))
            self.assertEqual(expected['language_model.model.layers.63.self_attn.k_proj.lora_b'], (rank, 1024))
            self.assertFalse(any('conv1d' in name or 'embed_tokens' in name or 'lm_head' in name for name in expected))
        for rank in (True, 0, 3, 32):
            with self.assertRaises(ValueError): layout._expected_from_projections(projections, rank)

    def test_last_block_counts_exactly_match_frozen_one_layer_shape(self):
        config = config_fixture(); headers = synthetic_headers(config)
        for blocks, tensors, parameters in ((1, 14, 434176), (4, 62, 1823872), (8, 124, 3647744), (16, 248, 7295488)):
            projections = layout._projection_layout_from(config, headers, blocks)
            expected = layout._expected_from_projections(projections, 4)
            self.assertEqual(len(expected), tensors)
            self.assertEqual(sum(a * b for a, b in expected.values()), parameters)
            self.assertTrue(all(int(name.split('.layers.')[1].split('.')[0]) >= 64 - blocks for name in expected))

    def test_missing_extra_wrong_shape_and_hybrid_order_rejected(self):
        config = config_fixture(); headers = synthetic_headers(config)
        del headers['language_model.model.layers.0.linear_attn.in_proj_b.weight']
        with self.assertRaises(ValueError): layout._projection_layout_from(config, headers, 64)
        headers = synthetic_headers(config)
        headers['language_model.model.layers.0.unexpected.weight'] = {'dtype': 'U32', 'shape': [48, 640]}
        with self.assertRaises(ValueError): layout._projection_layout_from(config, headers, 64)
        headers = synthetic_headers(config)
        headers['language_model.model.layers.0.linear_attn.in_proj_a.weight']['shape'] = [47, 640]
        with self.assertRaises(ValueError): layout._projection_layout_from(config, headers, 64)
        config['text_config']['layer_types'][0] = 'full_attention'
        with self.assertRaises(ValueError): layout._projection_layout_from(config, headers, 64)

    def test_actual_pinned_base_headers_only_if_installed(self):
        model = os.environ.get('RESEARCH4_MODEL_HEADERS_ONLY')
        if not model: self.skipTest('Set model path for actual pinned header-only validation')
        projections = layout.projection_layout(Path(model), 64)
        self.assertEqual(len(projections), 496)
        self.assertEqual(layout.verify_installed_lora_sources()['status'], 'pinned_lora_source_bytes_verified')
        for rank in (4, 8, 16):
            self.assertEqual(len(layout.expected_adapter_layout(Path(model), rank)), 992)


class SavedByteTests(unittest.TestCase):
    def setUp(self):
        try: import numpy
        except ImportError: self.skipTest('CPU numpy required for streamed finite checks')
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.path = self.root / 'adapters.safetensors'
        self.shapes = {'layer.lora_a': (2, 2), 'layer.lora_b': (2, 2)}
        self.values = {'layer.lora_a': ((2, 2), [1, 0, 0, 1]), 'layer.lora_b': ((2, 2), [0, 0.125, 0, 0])}

    def test_stream_finite_values_hash_and_nonzero_b(self):
        write_tensor_file(self.path, self.values)
        with patch.object(layout, 'CHUNK_BYTES', 8):
            receipt = layout.verify_adapter_tensor_file(self.path, self.shapes)
        self.assertEqual(receipt['adapter_sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(receipt['trainable_parameters'], 8)
        self.assertEqual(receipt['nonzero_lora_b_tensor_count'], 1)
        self.assertTrue(receipt['all_finite'])

    def test_native_mlx_null_optional_metadata_is_accepted_and_invalid_values_rejected(self):
        for metadata in (None, {}, {'source': 'synthetic CPU fixture'}):
            write_tensor_file(self.path, self.values, lambda header: header.update(__metadata__=metadata))
            receipt = layout.verify_adapter_tensor_file(self.path, self.shapes)
            self.assertEqual(receipt['adapter_sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())
        for metadata in (False, 1, 'not_a_metadata_object', {'number': 1}):
            write_tensor_file(self.path, self.values, lambda header: header.update(__metadata__=metadata))
            with self.assertRaises(ValueError): layout.verify_adapter_tensor_file(self.path, self.shapes)

    def test_nan_infinity_zero_b_wrong_dtype_shape_and_storage_rejected(self):
        for number in (float('nan'), float('inf')):
            self.values['layer.lora_b'] = ((2, 2), [0, number, 0, 0])
            write_tensor_file(self.path, self.values)
            with self.assertRaises(ValueError): layout.verify_adapter_tensor_file(self.path, self.shapes)
        self.values['layer.lora_b'] = ((2, 2), [0, 0, 0, 0])
        write_tensor_file(self.path, self.values)
        with self.assertRaises(ValueError): layout.verify_adapter_tensor_file(self.path, self.shapes)
        self.values['layer.lora_b'] = ((2, 2), [0, 1, 0, 0])
        for mutation in (lambda h: h['layer.lora_b'].update(dtype='F16'),
                         lambda h: h['layer.lora_b'].update(shape=[1, 4]),
                         lambda h: h['layer.lora_b'].update(data_offsets=[0, 16]),
                         lambda h: h.update(unexpected=h['layer.lora_b'])):
            write_tensor_file(self.path, self.values, mutation)
            with self.assertRaises(ValueError): layout.verify_adapter_tensor_file(self.path, self.shapes)
        write_tensor_file(self.path, self.values, extra=b'bad')
        with self.assertRaises(ValueError): layout.verify_adapter_tensor_file(self.path, self.shapes)

    def test_saved_config_exact_scope_rank_and_resume_guard(self):
        write_tensor_file(self.path, self.values)
        config = {'fine_tune_type': 'lora', 'num_layers': 64, 'batch_size': 1, 'max_seq_length': 1024,
                  'mask_prompt': True, 'resume_adapter_file': None, 'optimizer': 'adam', 'grad_accumulation_steps': 1,
                  'lora_parameters': {'rank': 4, 'dropout': 0.0, 'scale': 2.0}, 'iters': 8, 'learning_rate': 1e-5}
        path = self.root / 'adapter_config.json'
        with patch.object(layout, 'expected_adapter_layout', return_value=self.shapes):
            path.write_text(json.dumps(config))
            receipt = layout.verify_saved_adapter(self.path, path, self.root, 4, expected_iterations=8, expected_learning_rate=1e-5)
            self.assertEqual(receipt['status'], 'saved_hybrid_adapter_verified')
            self.assertFalse(receipt['full_base_payload_verified_here'])
            for name, value in (('resume_adapter_file', 'old_adapter'), ('num_layers', 1), ('batch_size', True), ('mask_prompt', 1)):
                changed = {**config, name: value}; path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): layout.verify_saved_adapter(self.path, path, self.root, 4)

    def test_all_layer_update_coverage_requires_each_block_not_each_projection(self):
        shapes = {}; values = {}
        for index in range(64):
            for suffix in ('a', 'b'):
                name = f'language_model.model.layers.{index}.projection.lora_{suffix}'
                shapes[name] = (2, 2)
                values[name] = ((2, 2), [1, 0, 0, 1] if suffix == 'a' else [0, 0.125 if index != 7 else 0, 0, 0])
        config = {'fine_tune_type': 'lora', 'num_layers': 64, 'batch_size': 1, 'max_seq_length': 1024,
                  'mask_prompt': True, 'resume_adapter_file': None, 'optimizer': 'adam', 'grad_accumulation_steps': 1,
                  'lora_parameters': {'rank': 4, 'dropout': 0.0, 'scale': 2.0}}
        path = self.root / 'adapter_config.json'; path.write_text(json.dumps(config))
        write_tensor_file(self.path, values)
        with patch.object(layout, 'expected_adapter_layout', return_value=shapes):
            receipt = layout.verify_saved_adapter(self.path, path, self.root, 4)
            self.assertNotIn(7, receipt['nonzero_lora_b_layer_indices'])
            self.assertFalse(receipt['all_requested_layers_have_nonzero_b'])
            with self.assertRaisesRegex(ValueError, 'no nonzero LoRA B'):
                layout.verify_saved_adapter(self.path, path, self.root, 4, require_all_layer_updates=True)
            values['language_model.model.layers.7.projection.lora_b'] = ((2, 2), [0, 0, 0.25, 0])
            write_tensor_file(self.path, values)
            receipt = layout.verify_saved_adapter(self.path, path, self.root, 4, require_all_layer_updates=True)
            self.assertEqual(receipt['nonzero_lora_b_layer_indices'], list(range(64)))
            self.assertEqual(receipt['nonzero_lora_b_tensor_count'], 64)
            self.assertTrue(receipt['all_requested_layers_have_nonzero_b'])

    def test_header_bounds_duplicate_keys_and_truncation_rejected(self):
        for raw in (b'', struct.pack('<Q', layout.MAX_HEADER_BYTES + 1), struct.pack('<Q', 100) + b'{}'):
            self.path.write_bytes(raw)
            with self.assertRaises(ValueError): layout.read_tensor_header(self.path)
        raw = b'{"x":{},"x":{}}'
        self.path.write_bytes(struct.pack('<Q', len(raw)) + raw)
        with self.assertRaises(ValueError): layout.read_tensor_header(self.path)


if __name__ == '__main__': unittest.main()
