import copy
import hashlib
import json
from pathlib import Path
import unittest
from src.composite_placement import candidate_plan, parameter_device, metadata_bounds, audit_frozen_activations

ROOT = Path(__file__).resolve().parents[1]


class PlacementTests(unittest.TestCase):
    def fixture(self):
        config = json.loads((ROOT / 'config/composite_controller_rank32_v2.json').read_text())['qwen']
        profile = json.loads((ROOT / 'config/composite_qwen_two_t4_candidate_v1.json').read_text())
        plan = candidate_plan(config, profile)
        tensors = {'model.language_model.embed_tokens.weight': {'dtype': 'BF16', 'shape': [4, 3]},
                   'lm_head.weight': {'dtype': 'BF16', 'shape': [4, 3]},
                   'model.visual.patch_embed.weight': {'dtype': 'BF16', 'shape': [2, 3]},
                   'model.language_model.norm.weight': {'dtype': 'F32', 'shape': [3]},
                   'mtp.layers.0.weight': {'dtype': 'BF16', 'shape': [2, 2]}}
        for index in range(64):
            tensors['model.language_model.layers.' + str(index) + '.mlp.down_proj.weight'] = {'dtype': 'BF16', 'shape': [2, 3]}
        metadata = {'repository': plan['repository'], 'revision': plan['revision'],
                    'config_sha256': plan['model_config_sha256'], 'tensor_values_read': False,
                    'index_sha256': plan['model_index_sha256'],
                    'tensors': tensors}
        # Tiny synthetic metadata exercises audit failures without instantiating
        # a model. The real candidate_plan binds the full pinned header digest.
        plan['tensor_metadata_sha256'] = hashlib.sha256(json.dumps(tensors, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        parameters = {name: {'device': 'cuda:' + str(parameter_device(name, plan)),
                             'logical_shape': value['shape'], 'sha256': 'a' * 64,
                             'requires_grad': False}
                      for name, value in tensors.items() if not name.startswith('mtp.')}
        record = {'profile': plan['profile'], 'revision': plan['revision'],
                  'model_instances': 1, 'optimizer_updates': 0, 'trainable_parameters': 0,
                  'heldout_content_read': False, 'kernel_profile': 'reference',
                  'input_token_ids': [1, 2, 3], 'device_map': plan['device_map'],
                  'parameters_before': parameters, 'parameters_after': copy.deepcopy(parameters),
                  'quantization_and_buffer_sha256_before': 'b' * 64,
                  'quantization_and_buffer_sha256_after': 'b' * 64,
                  'last_token_activations': [{'layer': index, 'device': 'cuda:' + str(index >= 18 and 1 or 0),
                                              'source_dtype': 'bfloat16', 'values': [0.0] * 5120}
                                             for index in range(64)],
                  'cache_boundary': {'prefill_checked': True, 'initial_cache_length': 0,
                                     'reused_previous_response_cache': False}}
        return config, profile, plan, metadata, record

    def test_complete_block_placement_without_model_or_GPU_execution(self):
        _, _, plan, metadata, _ = self.fixture()
        for index in range(64):
            self.assertEqual(parameter_device('model.language_model.layers.' + str(index) + '.weight', plan), 0 if index < 18 else 1)
        self.assertEqual(parameter_device('model.visual.patch_embed.weight', plan), 0)
        self.assertEqual(parameter_device('lm_head.weight', plan), 0)
        self.assertEqual(parameter_device('model.language_model.norm.weight', plan), 1)
        bounds = metadata_bounds(metadata, plan)
        self.assertEqual(len(bounds['runtime_parameter_names']), 68)
        self.assertEqual(bounds['ignored_checkpoint_only_MTP_keys'], ['mtp.layers.0.weight'])
        self.assertFalse(bounds['runtime_memory_admitted'])
        self.assertFalse(plan['runtime_admitted'])
        self.assertEqual(plan['compute_submissions'], 0)

    def test_changed_architecture_scope_and_generation_are_rejected(self):
        config, profile, _, _, _ = self.fixture()
        for values in [{'rank': 16}, {'layers': 32}, {'maximum_tokens': 4096}, {'quantization': 'int8'}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                candidate_plan({**config, **values}, profile)
        for values in [{'first_device_layer_count': 32}, {'kernel_profile': 'fla'},
                       {'retained_adapter_capacity': {'scope': 'last_blocks', 'rank': 32, 'alpha': 64}},
                       {'generation': {**profile['generation'], 'do_sample': True}},
                       {'optimizer_updates': 1}, {'behavior_adapter_fitting': True},
                       {'CPU_or_disk_offload': True}, {'heldout_content_read': True}]:
            with self.subTest(values=values), self.assertRaises((ValueError, PermissionError)):
                candidate_plan(config, {**profile, **values})

    def test_missing_unknown_or_wrong_revision_metadata_fail_closed(self):
        _, _, plan, metadata, _ = self.fixture()
        missing = copy.deepcopy(metadata)
        missing['tensors'].pop('model.language_model.layers.63.mlp.down_proj.weight')
        with self.assertRaisesRegex(ValueError, 'metadata'):
            metadata_bounds(missing, plan)
        foreign = copy.deepcopy(metadata)
        foreign['tensors']['another_model.weight'] = {'dtype': 'BF16', 'shape': [1]}
        with self.assertRaisesRegex(ValueError, 'metadata'):
            metadata_bounds(foreign, plan)
        with self.assertRaises(ValueError):
            metadata_bounds({**metadata, 'revision': 'other'}, plan)

    def test_raw_activation_fixture_does_not_claim_runtime_or_validation_admission(self):
        _, _, plan, metadata, record = self.fixture()
        audit = audit_frozen_activations(record, plan, metadata, [1, 2, 3])
        self.assertEqual(audit['layers_replayed'], 64)
        self.assertEqual(audit['audit_model_forwards'], 0)
        self.assertFalse(audit['full_validation_admitted'])
        self.assertEqual(audit['cross_device_numerical_parity'], 'not_established_by_this_audit')

    def test_wrong_device_changed_bytes_nonfinite_values_and_cache_reuse_reject(self):
        _, _, plan, metadata, record = self.fixture()
        changes = []
        wrong = copy.deepcopy(record); wrong['parameters_after']['lm_head.weight']['sha256'] = 'c' * 64; changes.append(wrong)
        wrong = copy.deepcopy(record); wrong['last_token_activations'][18]['device'] = 'cuda:0'; changes.append(wrong)
        wrong = copy.deepcopy(record); wrong['last_token_activations'][63]['values'][0] = float('nan'); changes.append(wrong)
        wrong = copy.deepcopy(record); wrong['cache_boundary']['reused_previous_response_cache'] = True; changes.append(wrong)
        wrong = copy.deepcopy(record); wrong['input_token_ids'] = [1, 2]; changes.append(wrong)
        wrong = copy.deepcopy(record); wrong['optimizer_updates'] = 1; changes.append(wrong)
        for index, value in enumerate(changes):
            with self.subTest(index=index), self.assertRaises(ValueError):
                audit_frozen_activations(value, plan, metadata, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
