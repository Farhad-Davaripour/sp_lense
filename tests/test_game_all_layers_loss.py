"""Synthetic suffix-mask tests; never import MLX or execute a real model."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

try: import numpy as np
except ImportError: np = None
PATH = Path(__file__).resolve().parents[1]/'mac_studio/game_all_layers_loss.py'
SPEC = importlib.util.spec_from_file_location('all_layers_suffix_loss', PATH)
loss = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(loss)


class PositionTests(unittest.TestCase):
    def test_first_target_eos_trailing_newline_and_padding_positions(self):
        self.assertEqual(loss.target_positions(3, 8, 33), [3,4,5,6,7])
        self.assertEqual(loss.target_positions(3, 8, 8), [3,4,5,6,7])
        self.assertEqual(loss.target_positions(1001, 1022, 1024), list(range(1001,1022)))
    def test_invalid_or_truncated_lengths_fail(self):
        for values in ((0,8,33), (8,8,33), (3,34,33), (3,1025,1025), (True,8,33)):
            with self.assertRaises(ValueError): loss.target_positions(*values)


@unittest.skipUnless(np is not None, 'NumPy required for synthetic numerical fixture')
class NumericalMaskTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(array=np.asarray, int32=np.int32, float32=np.float32,
            arange=np.arange, logical_and=np.logical_and)
        self.neural = SimpleNamespace(losses=SimpleNamespace(cross_entropy=lambda logits, targets:
            np.broadcast_to(np.arange(1,targets.shape[1]+1,dtype=np.float32),targets.shape)))
        self.fake = lambda values: np.zeros((*values.shape, 4), dtype=np.float32)
    def test_exact_target_markers_include_eos_and_trailing_newline_without_pad(self):
        batch, lengths = loss.make_batch(self.runtime,[101,102,103,201,202,203,204,205],3)
        total, count = loss.sum_with_runtime(self.runtime,self.neural,self.fake,batch,lengths)
        self.assertEqual((int(count),float(total)), (5,3+4+5+6+7))
        self.assertEqual((batch.shape,lengths.tolist()), ((1,33),[[3,8]]))
    def test_actual_r4_max_geometry_has_21_targets_not_22(self):
        batch, lengths = loss.make_batch(self.runtime,list(range(1022)),1001)
        total, count = loss.sum_with_runtime(self.runtime,self.neural,self.fake,batch,lengths)
        self.assertEqual(int(count),21)
        self.assertEqual(float(total),sum(range(1001,1022)))
    def test_valid_zero_token_is_supervised_and_padding_zero_is_not(self):
        batch, lengths = loss.make_batch(self.runtime,[101,102,103,0,202,203,204,205],3)
        _, count = loss.sum_with_runtime(self.runtime,self.neural,self.fake,batch,lengths)
        self.assertEqual(int(count),5)
    def test_padding_never_changes_unpadded_target_count(self):
        for length in (8,31,32,33,1022,1023,1024):
            batch, lengths = loss.make_batch(self.runtime,list(range(length)),3)
            _, count = loss.sum_with_runtime(self.runtime,self.neural,self.fake,batch,lengths)
            self.assertEqual(int(count),length-3)


if __name__ == '__main__': unittest.main()
