import ast
import hashlib
import importlib.metadata
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mac_studio'))
import game_all_layers_data as data
try:import numpy as np
except ImportError:np=None


class CohortTests(unittest.TestCase):
    def metadata(self):
        return [{'family':'family'+str(family),'audit_seed':1000+seed,'split':'train'}for family in range(12)for seed in range(64)]
    def test_exactly_four_lowest_training_seeds_per_family_in_original_file_order(self):
        chosen=data.choose_training_indices(self.metadata())
        self.assertEqual(chosen,[family*64+seed for family in range(12)for seed in range(4)])
    def test_development_row_or_missing_family_cannot_enter_short_training_cohort(self):
        rows=self.metadata();rows[0]['split']='dev'
        with self.assertRaises(ValueError):data.choose_training_indices(rows)
        with self.assertRaises(ValueError):data.choose_training_indices(self.metadata()[:-1])

    @unittest.skipUnless(np is not None,'NumPy required for native RNG/batching AST fixture')
    def test_native_ast_batch_order_equals_persisted_mt19937_length_sorted_order(self):
        try:
            distribution=importlib.metadata.distribution('mlx-lm')
        except importlib.metadata.PackageNotFoundError:
            self.skipTest('Pinned native MLX-LM source is unavailable; this parity check runs in the Mac environment')
        root=Path(distribution.locate_file(''))
        source=root/'mlx_lm/tuner/trainer.py';raw=source.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),'0b6027fdb11803e951ceee52c83a81f82daa922ea8c6b7ca6888953461064e8e')
        tree=ast.parse(raw);node=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=='iterate_batches')
        scope={'np':np,'mx':type('FakeMX',(),{'array':staticmethod(np.asarray)}),
            'CacheDataset':type('UnusedCacheDataset',(),{}),'rprint':lambda *args:None}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        lengths=[12+(i%7)for i in range(48)]
        dataset=[([1000+i]+[0]*(length-1),1)for i,length in enumerate(lengths)]
        np.random.seed(20261108)
        native=[int(batch[0,0])-1000 for batch,_ in scope['iterate_batches'](dataset,1,1024,loop=False)]
        self.assertEqual(data.native_batch_one_order(lengths),native)


if __name__=='__main__':unittest.main()
