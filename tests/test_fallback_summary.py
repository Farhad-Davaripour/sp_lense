"""A stopped GPU run permits exact control reuse, never invented neural episodes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.fallback_summary import recover
from src.matched_steering import CONDITIONS
from src.utils import sha256, write_json


class FallbackSummaryTests(unittest.TestCase):
    def fixture(self, temporary):
        root=Path(temporary)/'original';root.mkdir()
        cases=[{'id':'own','condition':'self_unfinished'}, {'id':'peer','condition':'other_complete'}]
        gates={'own':{'enabled':True},'peer':{'enabled':False}}
        prediction=Path(temporary)/'predictions.json'
        write_json(prediction,[{'id':k,'gate':v} for k,v in gates.items()])
        for name,value in [('ADMISSION.json',{'packet_sha256':'pinned'}),
                ('FAULT.json',{'type':'MemoryError','message':'GPU reserve breached','case_records':9}),
                ('PROGRESS.json',{'case_records':9}),('CLEANUP.json',{'released':True})]:
            write_json(root/name,value)
        for case in cases:
            row={'case_id':case['id'],'completed':True,'runtime_fault':None,
                 'frozen_initial_gate':gates[case['id']],
                 'score':{'negative_control':case['id']=='peer','workflow_success':True}}
            fallback=root/'shared_fallback/episodes/peer.json'
            if case['id']=='peer':write_json(fallback,row)
            for method in CONDITIONS:
                if case['id']=='peer' and method=='fine_tuned_A':continue
                value=dict(row)
                if case['id']=='peer':value.update(shared_base_fallback=True,source_episode_sha256=sha256(fallback))
                write_json(root/method/'episodes'/(case['id']+'.json'),value)
        return root, cases, prediction

    def test_exact_off_control_reuse_preserves_the_failed_original(self):
        with tempfile.TemporaryDirectory() as t:
            root,cases,p=self.fixture(t);before=(root/'FAULT.json').read_bytes()
            with patch('src.fallback_summary.read_packet',return_value={'cases.json':cases}):
                result=recover(root,'pinned-packet',p,Path(t)/'derived')
            self.assertEqual(result['source_GPU_worker_result'],'FAILED')
            self.assertEqual(len(result['derived_control_aliases']),1)
            self.assertEqual(result['new_model_forwards'],0)
            self.assertFalse(result['source_final_model_fingerprint_completed'])
            self.assertEqual((root/'FAULT.json').read_bytes(),before)
            self.assertFalse((root/'COMPLETE.json').exists())

    def test_a_missing_positive_episode_cannot_be_invented(self):
        with tempfile.TemporaryDirectory() as t:
            root,cases,p=self.fixture(t)
            (root/'fine_tuned_A/episodes/own.json').unlink()
            with patch('src.fallback_summary.read_packet',return_value={'cases.json':cases}):
                with self.assertRaises(ValueError):recover(root,'pinned-packet',p,Path(t)/'derived')
            self.assertFalse((Path(t)/'derived').exists())

    def test_conflicting_fallback_provenance_stops_before_writing(self):
        with tempfile.TemporaryDirectory() as t:
            root,cases,p=self.fixture(t)
            path=root/'ridge31/episodes/peer.json'
            value=json.loads(path.read_bytes());value['source_episode_sha256']='wrong';write_json(path,value)
            with patch('src.fallback_summary.read_packet',return_value={'cases.json':cases}):
                with self.assertRaises(ValueError):recover(root,'pinned-packet',p,Path(t)/'derived')
            self.assertFalse((Path(t)/'derived').exists())
