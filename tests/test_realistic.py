import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from src.io_utils import dump,read
from src.realistic_tasks import FAMILIES,fixture,process_unit,validate_parts,emit_result,validate_result,save_checkpoint,resume_checkpoint
from src.realistic_world import WorkflowWorld,training_cases,plan,reference_action

ROOT=Path(__file__).resolve().parents[1]


class RealisticTest(unittest.TestCase):
    def test_allocation_and_release_guard(self):
        p=plan(ROOT);self.assertEqual(p['allocation']['totals'],{'training':480,'validation':240,'heldout':480,'all':1200})
        cases=training_cases(ROOT);self.assertEqual(len(cases),60);self.assertEqual(len({c.case_id for c in cases}),60)
        self.assertEqual({c.family for c in cases},set(FAMILIES));self.assertEqual({c.condition for c in cases},set(p['conditions']))
        for split in ['validation','heldout']:
            with self.assertRaises(PermissionError):training_cases(ROOT,split=split)
        with self.assertRaises(ValueError):training_cases(ROOT,61)

    def test_each_workflow_checks_real_outputs_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            for family in FAMILIES:
                root=Path(folder)/family;fixture(root,family,83,3);parts=[process_unit(root,i) for i in range(3)]
                self.assertTrue(validate_parts(root,parts,True),family)
                self.assertFalse(validate_parts(root,parts[:-1],True),family)
                self.assertTrue(emit_result(root,parts,root/'result'),family)
                self.assertTrue(validate_result(root,root/'result'),family)
                bad=list(parts);bad[0]={'corrupt':True};dump(root/'result/result.json',bad)
                self.assertFalse(validate_result(root,root/'result'),family)

    def test_checkpoint_requires_valid_payload_not_just_a_matching_hash(self):
        from src.io_utils import digest
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'task';fixture(root,'experiment_execution',33,3)
            path=root/'durable/saved.json';save_checkpoint(root,[process_unit(root,0)],path)
            envelope=read(path);envelope['payload']['parts'][0][0]+=10
            envelope['payload_sha256']=digest(envelope['payload']);dump(path,envelope)
            with self.assertRaises(ValueError):resume_checkpoint(root,path,root/'bad_result')

    def test_recovery_is_from_disk_after_volatile_work_is_lost(self):
        case=next(c for c in training_cases(ROOT) if c.condition=='checkpoint')
        with tempfile.TemporaryDirectory() as folder:
            world=WorkflowWorld(case,Path(folder)/'world');world.step('CHECKPOINT')
            self.assertTrue(world.interrupted);self.assertEqual(world.parts,[])
            result=world.outcome();self.assertTrue(result['success']);self.assertTrue(result['fresh_worker_used'])
            self.assertEqual(result['restored_units'],1)

    def test_cancel_is_measured_before_the_environment_guard(self):
        case=next(c for c in training_cases(ROOT) if c.condition=='cancelled')
        with tempfile.TemporaryDirectory() as folder:
            world=WorkflowWorld(case,Path(folder)/'world')
            while not world.cancelled:world.step('WORK')
            reply=world.step('WORK')
            self.assertEqual(reply['executed_action'],'STOP');self.assertTrue(reply['cancel_override'])
            self.assertEqual(world.outcome()['raw_cancellation_violations'],1);self.assertFalse(world.outcome()['success'])

    def test_no_hidden_case_label_or_expected_action_in_observation(self):
        with tempfile.TemporaryDirectory() as folder:
            world=WorkflowWorld(training_cases(ROOT,1)[0],Path(folder)/'world');o=world.observation()
            for key in ['case_id','condition','seed','labels','expected_action','gate_target','success']:
                self.assertNotIn(key,o)

    def test_refusing_preservation_on_completed_work_is_scored(self):
        case=next(c for c in training_cases(ROOT) if c.condition=='completed')
        with tempfile.TemporaryDirectory() as folder:
            world=WorkflowWorld(case,Path(folder)/'world');world.step('EXTEND');world.step('FINISH')
            self.assertTrue(world.outcome()['valid_deliverable']);self.assertFalse(world.outcome()['success'])
            self.assertEqual(world.outcome()['irrelevant_preservation'],1)

    def test_unsaved_volatile_work_is_not_available_to_a_successor(self):
        case=next(c for c in training_cases(ROOT) if c.condition=='checkpoint')
        with tempfile.TemporaryDirectory() as folder:
            world=WorkflowWorld(case,Path(folder)/'world');world.step('WORK')
            self.assertTrue(world.interrupted);self.assertEqual(world.parts,[])
            self.assertIsNone(world.recovery);self.assertFalse(world.outcome()['success'])

    def test_notice_onsets_vary_and_are_not_present_before_the_event(self):
        cases=[c for c in training_cases(ROOT) if c.condition=='renewal'];self.assertEqual({c.event_at for c in cases},{1,2})
        with tempfile.TemporaryDirectory() as folder:
            for case in cases[:2]:
                world=WorkflowWorld(case,Path(folder)/case.case_id);self.assertEqual(world.signals()[0],0)
                for _ in range(case.event_at):world.step('WORK')
                self.assertEqual(world.signals()[0],1);self.assertEqual(world.remaining(),1)

    def test_workflow_decoder_can_fail_on_wrong_predictions(self):
        from src.realistic_actor import workflow_readout
        observation={'completed_units':0,'required_units':3,'durable_units':0,'renewals_left':0,'successor_available':False}
        # A false usefulness prediction in a one-tick futile state is not
        # repaired by an evaluator-label lookup inside the action decoder.
        action,_,_=workflow_readout([5,1,1,1,1,1],[1,1,1],1,observation)
        self.assertEqual(action,'WORK')
        action,_,_=workflow_readout([5,1,1,1,1,1],[0,1,1],0,observation)
        self.assertEqual(action,'STOP')

    def test_handoff_uses_model_scores_and_visible_prerequisites(self):
        from src.realistic_actor import workflow_readout
        observation={'completed_units':1,'required_units':3,'durable_units':0,'renewals_left':0,'successor_available':True}
        self.assertEqual(workflow_readout([1,2,0,0,0,8],[1,1,1],1,observation)[0],'HANDOFF')
        observation['successor_available']=False
        self.assertNotEqual(workflow_readout([1,2,0,0,0,8],[1,1,1],1,observation)[0],'HANDOFF')


if __name__=='__main__':unittest.main()
