import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

try:import numpy as np
except ImportError:np=None
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mac_studio'))
import game_chain_rule_v4_compare as compare
from game_selected_head_v3_compare import TOLERANCES


@unittest.skipUnless(np is not None,'NumPy required for native payload fixtures')
class NativeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def save(self,name,key,values,dtype='F32'):
        if dtype=='BF16':raw=(values.astype('<f4').view('<u4')>>16).astype('<u2').tobytes()
        else:raw=values.astype('<f2'if dtype=='F16'else'<f4').tobytes()
        header={'__metadata__':None,key:{'dtype':dtype,'shape':list(values.shape),'data_offsets':[0,len(raw)]}}
        encoded=json.dumps(header).encode();encoded+=b' '*((-len(encoded))%8)
        path=self.root/name;path.write_bytes(struct.pack('<Q',len(encoded))+encoded+raw);return path
    def test_native_float_and_bfloat_audit_copies_preserve_exact_values(self):
        values=np.array([1.5,-2.0,0.0,.125],dtype=np.float32).reshape(1,2,2)
        for dtype in('F16','BF16','F32'):
            native=self.save(dtype+'.safetensors','hidden',values,dtype)
            shadow=self.save(dtype+'_shadow.safetensors','hidden_values',values)
            report=compare.verify_native_audit_copy(native,'hidden',shadow,'hidden_values',values.shape)
            self.assertTrue(report['passed']);self.assertEqual(report['native_dtype'],dtype)
            self.assertTrue(report['native_shadow_values_equal'])
    def test_corrupt_shadow_cannot_masquerade_as_native_covector(self):
        values=np.ones((1,3,2),dtype=np.float32);native=self.save('native','dh',values,'BF16')
        values[0,0,0]=2;shadow=self.save('shadow','hidden_gradient',values)
        with self.assertRaises(ValueError):compare.verify_native_audit_copy(native,'dh',shadow,'hidden_gradient',values.shape)
    def test_covector_checks_every_hidden_row_and_exact_outside_zeros(self):
        values=np.zeros((1,8,3),dtype=np.float32);values[:,2:5,:]=1
        a=self.save('a','hidden_gradient',values);b=self.save('b','hidden_gradient',values)
        result=compare.compare_covectors(a,b,values.shape,3,6,TOLERANCES)
        self.assertTrue(result['passed']);self.assertEqual(result['reference_support']['all_hidden_positions_checked'],8)
        values[0,0,0]=1e-12;bad=self.save('bad','hidden_gradient',values)
        with self.assertRaises(ValueError):compare.compare_covectors(a,bad,values.shape,3,6,TOLERANCES)
    def test_zero_everywhere_is_not_a_valid_head_sensitivity_proof(self):
        values=np.zeros((1,8,3),dtype=np.float32);path=self.save('zero','hidden_gradient',values)
        with self.assertRaises(ValueError):compare.covector_support(path,values.shape,3,6)
    def test_recomputed_native_hidden_values_must_be_exactly_the_same(self):
        values=np.ones((1,3,2),dtype=np.float32)
        a=self.save('a','hidden',values,'BF16');b=self.save('b','hidden',values,'BF16')
        self.assertTrue(compare.exact_native_values(a,b,'hidden',values.shape)['passed'])
        values[0,0,0]=1.125;c=self.save('changed','hidden',values,'BF16')
        self.assertFalse(compare.exact_native_values(a,c,'hidden',values.shape)['passed'])


if __name__=='__main__':unittest.main()
