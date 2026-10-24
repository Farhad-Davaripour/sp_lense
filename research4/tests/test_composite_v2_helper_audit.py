from copy import deepcopy
from pathlib import Path
import unittest
from src.io_utils import read,digest
from src.composite_generalized_world import fixture
from src.composite_generalized_teacher import teacher_episode
from src.composite_v2_helper_audit import (HelperEvidence,TrainingExposure,training_exposure,
    read_helper_evidence,bound_helper_receipt,MEASUREMENT)
from test_composite_v2_trace_audit import record_for,TOKENIZERS,SOURCE

CFG=read(Path(__file__).resolve().parents[1]/'config/composite_generalization_amendment_v2.json')


class HelperAuditTests(unittest.TestCase):
    def record(self):
        # Replay an existing public reference as a scripted fake, never a model
        # or scientific cohort. The reader does not import or call the teacher.
        scene=fixture('own_complete','aggregate_predicate',seed='unit:helper_evidence')
        example=teacher_episode(scene)
        record,_=record_for(scene=scene,actions=[a['action'] for a in example['actions']])
        return record,scene,example

    def exposure(self,episodes):
        return training_exposure(episodes,[],CFG,SOURCE,'b'*64,public_fixture=True)

    def audit(self,record,scene,exposure=None):
        return read_helper_evidence(record,scene,SOURCE,CFG,exposure,public_fixture=True,
                                    fixture_tokenizers=TOKENIZERS).receipt()

    def test_relevant_writes_live_bindings_and_costs_are_derived(self):
        record,scene,example=self.record()
        # A separate existing public *read-only* program supplies the exclusion
        # set; no caller claims of relevance or success enter the reader.
        exposure_episode=deepcopy(example)
        exposure_episode['helper_definitions']={}
        proof=self.audit(record,scene,self.exposure([exposure_episode]))
        self.assertTrue(proof['success']);self.assertFalse(proof['raw_evidence_audited'])
        helper=proof['helpers'][0]
        self.assertEqual(helper['distinct_consumed_bindings'],2)
        self.assertGreater(helper['net_decisions_saved'],0)
        self.assertFalse(proof['caller_supplied_relevance_used'])
        self.assertIsNotNone(proof['actual_effect_transition_primitive_index'])

    def test_training_exposure_and_missing_exposure_are_not_novel_success(self):
        record,scene,example=self.record()
        self.assertIsNone(self.audit(record,scene)['success'])
        self.assertFalse(self.audit(record,scene,self.exposure([example]))['success'])

    def test_caller_booleans_and_foreign_or_public_exposure_cannot_admit_science(self):
        record,scene,example=self.record()
        with self.assertRaises(PermissionError):HelperEvidence({'success':True})
        with self.assertRaises(PermissionError):TrainingExposure(SOURCE,'b'*64,[],False)
        with self.assertRaises(PermissionError):self.audit(record,scene,{'success':True})
        self.assertIsNone(bound_helper_receipt({'success':True,'trace_sha256':digest(record)},digest(record),SOURCE,MEASUREMENT))
        evidence=read_helper_evidence(record,scene,SOURCE,CFG,self.exposure([example]),public_fixture=True,fixture_tokenizers=TOKENIZERS)
        self.assertIsNone(bound_helper_receipt(evidence,digest(record),SOURCE,MEASUREMENT))
        with self.assertRaises(PermissionError):training_exposure([example],[],CFG,SOURCE,'b'*64)

    def test_tampered_raw_effects_cannot_obtain_causal_credit(self):
        record,scene,_=self.record();bad=deepcopy(record)
        entry=next(v for v in bad['primitive_trace'] if v['op']=='patch')
        entry['record_before']=deepcopy(entry['record_after'])
        with self.assertRaises(ValueError):self.audit(bad,scene)

    def test_read_only_return_is_unknown_even_if_value_matches_later_input(self):
        from src.composite_generalized_world import World
        scene=fixture('quiet_pending','aggregate_predicate',seed='unit:read_only_return')
        world=World(scene)
        actions=[{'tool':'catalog','arguments':{}},
            {'tool':'define','arguments':{'name':'reader','parameters':['s'],
             'steps':[['a','read',['$s']],['b','aggregate',['$a.data.rows','value','sum']],['c','emit',['$b']]]}},
            {'tool':'invoke','arguments':{'name':'reader','arguments':{'s':world.source_ids[0]}}},
            {'tool':'invoke','arguments':{'name':'reader','arguments':{'s':world.source_ids[1]}}},
            {'tool':'stop','arguments':{}}]
        record,_=record_for(scene=scene,actions=actions)
        proof=self.audit(record,scene)
        self.assertIsNone(proof['success'])
        self.assertFalse(proof['value_equality_is_return_use_proof'])
        self.assertTrue(all(c['return_use']=='unknown_no_explicit_result_reference' for c in proof['helpers'][0]['calls']))

    def test_no_helper_is_observed_false_and_missing_case_is_not_synthesized(self):
        record,scene=record_for()
        proof=self.audit(record,scene)
        self.assertFalse(proof['success']);self.assertEqual(proof['helpers'],[])

    def test_overwritten_helper_write_does_not_count_as_a_second_relevant_binding(self):
        from src.composite_generalized_world import World,expected_analysis
        _,scene,example=self.record();world=World(scene)
        actions=[deepcopy(a['action']) for a in example['actions']]
        calls=[i for i,a in enumerate(actions) if a['tool']=='invoke']
        target=world.analysis_ids[0]
        expected=expected_analysis(scene.mechanism,world.documents[world.source_ids[0]]['data'])
        actions.insert(calls[0]+1,{'tool':'patch','arguments':{'identifier':target,'fields':{'total':-1}}})
        actions.insert(-1,{'tool':'patch','arguments':{'identifier':target,'fields':expected}})
        record,_=record_for(scene=scene,actions=actions)
        exposure=deepcopy(example);exposure['helper_definitions']={}
        proof=self.audit(record,scene,self.exposure([exposure]))
        self.assertFalse(proof['success'])
        self.assertEqual(proof['helpers'][0]['distinct_consumed_bindings'],1)


if __name__=='__main__':unittest.main()
