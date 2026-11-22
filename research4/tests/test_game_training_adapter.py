"""Synthetic saved-adapter tests; no model weights or GPU are loaded."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    from safetensors.numpy import save_file
except ImportError:
    raise unittest.SkipTest('Install numpy and safetensors for the synthetic adapter test')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mac_studio'))
import game_training_worker as worker


class SavedAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name); (self.run / 'adapters').mkdir()
        self.layout = {}; self.values = {}
        for index in range(7):
            name = f'language_model.model.layers.63.projection_{index}'
            for suffix in ('.lora_a', '.lora_b'):
                self.layout[name + suffix] = (4, 4)
                self.values[name + suffix] = (np.ones if suffix == '.lora_a' else np.zeros)((4, 4), dtype=np.float32)
        config = {'num_layers': 1, 'batch_size': 1, 'iters': 1, 'max_seq_length': 1024,
                  'mask_prompt': True, 'lora_parameters': worker.plans.CONFIG['lora_parameters']}
        (self.run / 'adapters/adapter_config.json').write_text(json.dumps(config))

    def verify(self):
        save_file(self.values, str(self.run / 'adapters/adapters.safetensors'))
        with patch.object(worker, 'expected_adapter_layout', return_value=self.layout):
            return worker.verify_adapter(self.run, {'iterations': 1, 'max_seq_length': 1024})

    def test_nonzero_a_with_all_zero_b_is_rejected(self):
        self.assertTrue(any(np.count_nonzero(value) for value in self.values.values()))
        with self.assertRaisesRegex(ValueError, 'nonzero LoRA B'):
            self.verify()

    def test_one_nonzero_b_tensor_passes_and_records_check(self):
        name = next(name for name in self.values if name.endswith('.lora_b'))
        self.values[name][0, 0] = np.float32(0.125)
        result = self.verify()
        self.assertEqual(result['tensor_count'], 14)
        self.assertTrue(result['nonzero_lora_b'])

    def test_nonfinite_b_is_rejected(self):
        name = next(name for name in self.values if name.endswith('.lora_b'))
        self.values[name][0, 0] = np.float32('nan')
        with self.assertRaisesRegex(ValueError, 'tensor invalid'):
            self.verify()


if __name__ == '__main__': unittest.main()
