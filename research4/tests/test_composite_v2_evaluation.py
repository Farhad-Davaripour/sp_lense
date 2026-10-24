from pathlib import Path
import unittest
from unittest.mock import patch
from src.io_utils import read
from src.composite_generalized_world import fixture
from src.composite_v2_evaluation import summarize_v2,collect_partition

ROOT=Path(__file__).resolve().parents[1]


class V2EvaluationTests(unittest.TestCase):
    def test_missing_controls_and_helper_audits_stay_unknown_not_negative(self):
        base=read(ROOT/'config/composite_controller_rank32_v2.json');design=read(ROOT/'config/composite_generalization_amendment_v2.json')
        scene=fixture();package={'data_release_sha256':'a'*64,'development_roster_sha256':'b'*64}
        value=summarize_v2([], [scene],base,design,package,'c'*40,False,public_fixture=True)
        self.assertFalse(value['complete']);self.assertFalse(value['admitted'])
        self.assertEqual(value['observed_trajectories'],0);self.assertEqual(len(value['missing']),5)
        family=value['compositional_transfer']['families'][scene.mechanism]
        self.assertIsNone(family['primary_case_fraction']);self.assertIsNone(family['paired_case_fraction_difference'])
        self.assertIsNone(family['population_confidence_interval'])

    def test_partial_collection_exports_finished_trace_and_always_verifies(self):
        scene=fixture();saved={};verified=[]
        record={'scene':{'case_id':scene.case_id},'condition':'base','public_fixture':True}
        def execute(model,actual_scene,arm,*args):
            if arm!='base':raise RuntimeError('synthetic later-arm failure')
            return record
        with patch('src.composite_v2_evaluation.rollout',side_effect=execute):
            with self.assertRaises(RuntimeError):collect_partition(object(),[scene],['base','always_on'],lambda:None,
                lambda name,value:saved.update({name:value}),lambda:verified.append(True),'a'*40,public_fixture=True)
        self.assertEqual(verified,[True]);self.assertIn('traces/'+scene.case_id+'/base.json',saved)
        self.assertEqual(saved['COLLECTION_RECEIPT.json']['observed_trajectories'],1)
        self.assertFalse(saved['COLLECTION_RECEIPT.json']['scientific_validation_admitted'])
        self.assertEqual(len(saved['COLLECTION_RECEIPT.json']['missing']),1)

    def test_legacy_or_foreign_world_cannot_be_relabelled_as_v2(self):
        base=read(ROOT/'config/composite_controller_rank32_v2.json');design=read(ROOT/'config/composite_generalization_amendment_v2.json')
        with self.assertRaises(PermissionError):summarize_v2([{'environment':{'version':'legacy'}}],[fixture()],base,design,{},'a'*40,False)

    def test_caller_model_integrity_boolean_is_not_a_reader_attestation(self):
        base=read(ROOT/'config/composite_controller_rank32_v2.json');design=read(ROOT/'config/composite_generalization_amendment_v2.json')
        value=summarize_v2([], [fixture()],base,design,{},'a'*40,False,model_integrity_audited=True,public_fixture=True)
        self.assertFalse(value['complete_model_integrity_audited'])
        self.assertFalse(value['admitted'])


if __name__=='__main__':unittest.main()
