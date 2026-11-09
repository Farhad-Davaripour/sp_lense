import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

try:import numpy as np
except ImportError:np=None
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mac_studio'))
import game_selected_head_v3_compare as compare


@unittest.skipUnless(np is not None,'NumPy required for saved F32 numerical fixtures')
class ComparatorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.a='language_model.model.layers.0.mlp.up_proj.lora_a';self.b='language_model.model.layers.0.mlp.up_proj.lora_b'
        self.arrays={self.a:np.zeros((3,2),np.float32),self.b:np.ones((2,3),np.float32)}
        self.expected={name:value.shape for name,value in self.arrays.items()}
    def save(self,name,arrays):
        header={'__metadata__':None};payload=b''
        for key,value in sorted(arrays.items()):
            raw=value.astype('<f4').tobytes();header[key]={'dtype':'F32','shape':list(value.shape),'data_offsets':[len(payload),len(payload)+len(raw)]};payload+=raw
        encoded=json.dumps(header).encode();encoded+=b' '*((-len(encoded))%8)
        path=self.root/name;path.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload);return path
    def test_identical_all_tensors_and_initial_zero_a_pass(self):
        a=self.save('a.safetensors',self.arrays);b=self.save('b.safetensors',self.arrays)
        result=compare.compare_gradients(a,b,self.expected,compare.TOLERANCES,True)
        self.assertTrue(result['passed']);self.assertEqual(result['tensor_count'],2)
        self.assertEqual(compare.parameter_state_sha256(a,self.expected),compare.parameter_state_sha256(b,self.expected))
    def test_after_update_nonzero_a_allowed_but_initial_zero_a_gate_is_exact(self):
        changed={name:value.copy()for name,value in self.arrays.items()};changed[self.a][0,0]=1e-9
        a=self.save('a.safetensors',changed);b=self.save('b.safetensors',changed)
        self.assertFalse(compare.compare_gradients(a,b,self.expected,compare.TOLERANCES,True)['passed'])
        self.assertTrue(compare.compare_gradients(a,b,self.expected,compare.TOLERANCES,False)['passed'])
    def test_tolerance_excess_and_nonfinite_are_not_repaired(self):
        a=self.save('a.safetensors',self.arrays);changed={name:value.copy()for name,value in self.arrays.items()}
        changed[self.b][0,0]=1.1;b=self.save('b.safetensors',changed)
        self.assertFalse(compare.compare_gradients(a,b,self.expected,compare.TOLERANCES)['passed'])
        changed[self.b][0,0]=float('nan');bad=self.save('nan.safetensors',changed)
        with self.assertRaises(ValueError):compare.compare_gradients(a,bad,self.expected,compare.TOLERANCES)
    def test_only_saved_valid_logit_rows_are_compared_and_loss_counts_exact(self):
        values={'target_logits':np.arange(20,dtype=np.float32).reshape(5,4)}
        a=self.save('logitsa.safetensors',values);b=self.save('logitsb.safetensors',values)
        result=compare.compare_logits(a,b,(5,4));self.assertTrue(result['passed']);self.assertTrue(result['valid_target_rows_only'])
        self.assertTrue(compare.loss_equivalence(1,1.00001,5,5,compare.TOLERANCES)['passed'])
        with self.assertRaises(ValueError):compare.loss_equivalence(1,1,5,4,compare.TOLERANCES)


if __name__=='__main__':unittest.main()
