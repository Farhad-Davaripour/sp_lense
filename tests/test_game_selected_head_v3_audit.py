"""Independent saved-byte auditor fixtures; no MLX/model/subprocess/network."""
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

try:import numpy as np
except ImportError:np=None
PATH=Path(__file__).resolve().parents[1]/'mac_studio/game_selected_head_v3_audit.py'
SPEC=importlib.util.spec_from_file_location('offline_auditor',PATH)
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)


@unittest.skipUnless(np is not None,'NumPy CPU fixtures required')
class ByteTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
    def save(self,name,arrays):
        meta={'__metadata__':None};payload=bytearray()
        for key,array in sorted(arrays.items()):
            data=array.astype('<f4').tobytes();meta[key]={'dtype':'F32','shape':list(array.shape),'data_offsets':[len(payload),len(payload)+len(data)]};payload.extend(data)
        header=json.dumps(meta).encode();path=self.root/name;path.write_bytes(struct.pack('<Q',len(header))+header+payload);return path
    def test_raw_comparison_rejects_difference_and_nonfinite(self):
        arrays={'language_model.model.layers.0.p.lora_a':np.zeros((2,2),np.float32),'language_model.model.layers.0.p.lora_b':np.ones((2,2),np.float32)}
        expected={name:value.shape for name,value in arrays.items()};a=self.save('a',arrays);b=self.save('b',arrays)
        self.assertTrue(audit.compare_files(a,b,expected,audit.LOSS,True)['passed'])
        arrays[next(name for name in arrays if name.endswith('.lora_b'))][0,0]=1.2
        self.assertFalse(audit.compare_files(a,self.save('wrong',arrays),expected,audit.LOSS)['passed'])
        arrays[next(name for name in arrays if name.endswith('.lora_b'))][0,0]=float('nan')
        with self.assertRaises(ValueError):audit.compare_files(a,self.save('nan',arrays),expected,audit.LOSS)
    def test_parameter_snapshot_distinguishes_initial_and_updated_b(self):
        arrays={'language_model.model.layers.0.p.lora_a':np.ones((2,2),np.float32),'language_model.model.layers.0.p.lora_b':np.zeros((2,2),np.float32)}
        shapes={name:value.shape for name,value in arrays.items()};first=audit.snapshot(self.save('init',arrays),shapes)
        self.assertEqual(first['nonzero_b_layers'],[])
        arrays[next(name for name in arrays if name.endswith('.lora_b'))][0,0]=.1
        after=audit.snapshot(self.save('updated',arrays),shapes)
        self.assertEqual(after['nonzero_b_layers'],[0]);self.assertNotEqual(first['state_sha256'],after['state_sha256'])
    def test_independent_cross_entropy_uses_actual_target_ids_including_zero(self):
        values=np.array([[2,1,0,-1],[-2,1,3,0]],dtype=np.float32);path=self.save('logits',{'target_logits':values})
        observed=audit.raw_mean_nll(path,[0,2],4)
        expected=np.mean([math.log(sum(math.exp(float(v)) for v in row))-float(row[token]) for row,token in zip(values,[0,2])])
        self.assertAlmostEqual(observed,float(expected),places=12)


class IncompleteTests(unittest.TestCase):
    def test_missing_raw_proof_never_qualifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'.local/run';run.mkdir(parents=True)
            item=root/'.local/item';item.write_text('{}');binding={'path':'.local/item','sha256':audit.sha(item)}
            plan={'schema':'research4.selected-head-equivalence-plan.v3','rank':4,'num_layers':64,'head_window':32,
                'original_started_epoch':1794187139.176875,'original_deadline_epoch':1794266339.176875,
                'gradient_tolerances':audit.LOSS,'logit_tolerances':audit.LOGITS,'source_sha256':{},
                **{key:binding for key in ('protocol','config','cohort','original_campaign','original_failure_latch','original_supervisor_failure')}}
            packet=root/'.local/plan';packet.write_text(json.dumps(plan));expected=audit.sha(packet)
            (run/'request.json').write_text(json.dumps({'plan_sha256':expected,'plan':plan}))
            with patch.object(audit,'PLAN_SHA',expected):result=audit.audit(root,run,packet,expected)
            self.assertEqual(result['status'],'incomplete_or_failed')
            self.assertFalse(result['real_equivalence_measurements_verified']);self.assertFalse(result['eligible_for_tuning_or_serving'])
    def test_escaping_evidence_path_is_rejected(self):
        for value in ('../escape','/outside','C:/outside','.local\\outside'):
            with self.assertRaises(ValueError):audit.local_path(Path.cwd(),value)


if __name__=='__main__':unittest.main()
