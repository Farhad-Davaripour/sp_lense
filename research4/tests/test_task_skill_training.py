from copy import deepcopy
import json
from pathlib import Path
import unittest
from src.composite_task_skill_training import (TaskDataEvidence, read_data, validate_recipe,
                                               memory_admission, token_plan, qualification_plan)


class TaskTrainingBoundaryTests(unittest.TestCase):
    def test_public_bool_or_corrupt_archive_cannot_create_data_evidence(self):
        with self.assertRaises(PermissionError):
            TaskDataEvidence([], {}, key=True)
        for raw in [b'', b'{}', b'public fixture not scientific evidence']:
            with self.assertRaises(PermissionError):
                read_data(raw)

    def test_the_recipe_cannot_shrink_or_change_the_full_agreed_model(self):
        cfg = json.loads((Path(__file__).parents[1] / 'config/composite_controller_rank32_v2.json').read_text())
        self.assertTrue(validate_recipe(cfg))
        for key, value in [('rank', 16), ('layers', 32), ('maximum_tokens', 2048), ('maximum_new_tokens', 80)]:
            bad = deepcopy(cfg)
            bad['qwen'][key] = value
            with self.assertRaises(PermissionError):
                validate_recipe(bad)
        bad = deepcopy(cfg)
        bad['behavior_training']['steps'] = 32
        with self.assertRaises(PermissionError):
            validate_recipe(bad)

    def test_spare_memory_on_one_device_cannot_cover_a_shortfall_on_the_other(self):
        class GPU:
            def mem_get_info(self, index):
                return [(100, 200), (19, 200)][index]
        layout = {'FP32_weights_gradients_AdamW_bytes': [20, 20]}
        with self.assertRaises(MemoryError):
            memory_admission(layout, GPU(), 5)

    def test_live_cache_is_not_mistaken_for_live_tensor_storage_or_capacity_proof(self):
        from src.composite_task_skill_training import live_resources
        class GPU:
            def is_available(self): return True
            def device_count(self): return 2
            def mem_get_info(self,index): return (50*1024**2, 16*1024**3)
            def memory_allocated(self,index): return 1024**3
            def memory_reserved(self,index): return 2*1024**3
        proof = live_resources(GPU())
        self.assertFalse(proof['actual_backward_fit_verified'])
        self.assertEqual(len(proof['devices']), 2)
        class FullGPU(GPU):
            def memory_reserved(self,index): return self.memory_allocated(index)
        with self.assertRaises(MemoryError): live_resources(FullGPU())

    def test_risk_or_validation_rows_never_enter_the_task_token_planner(self):
        for row in [{'split':'development','task_only':True,'training_eligible':True},
                    {'split':'training','task_only':False,'training_eligible':True},
                    {'split':'training','task_only':True,'training_eligible':False},
                    {'split':'training','labels':{'gate':True}}]:
            with self.assertRaises(PermissionError):
                token_plan([row], None, {})

    def test_qualification_keeps_full_capacity_and_does_not_admit_full_fit(self):
        plan = qualification_plan('a' * 40)
        self.assertEqual(plan['scientific_GPU_count'], 2)
        self.assertEqual(plan['optimizer_updates'], 1)
        self.assertFalse(plan['qualification_checkpoint_initializes_full_fit'])
        self.assertEqual(plan['source_training_rows'], 1188)
        self.assertFalse(plan['heldout_content_read'])
        self.assertEqual(plan['maximum_raw_evidence_bytes'], 4 * 1024**3)
        self.assertEqual(plan['physical_export_reserve_bytes'], 12 * 1024**3)

    def test_task_export_disk_envelope_covers_scratch_and_published_raw_plus_archive(self):
        from src.composite_v2_raw_exports import validate_export_plan, TASK_PROFILE
        plan = qualification_plan('b' * 40)
        proof = validate_export_plan(plan, expected_profile=TASK_PROFILE)
        # One raw copy plus reserved space must cover two raw and two archive copies.
        self.assertGreaterEqual(proof['physical_export_reserve_bytes'] + proof['maximum_raw_evidence_bytes'],
                                4 * proof['maximum_raw_evidence_bytes'])
        with self.assertRaises(PermissionError):
            validate_export_plan(plan)  # The legacy profile is not silently widened.
        bad = deepcopy(plan)
        bad['physical_export_reserve_bytes'] = 9 * 1024**3
        with self.assertRaises(PermissionError):
            validate_export_plan(bad, expected_profile=TASK_PROFILE)

    def test_shared_optimizer_backend_accepts_task_rows_without_fake_threat_labels(self):
        from tempfile import TemporaryDirectory
        from tests.test_composite_v2_training import TinyFitActor
        from src.composite_behavior import fit_selected_rows
        cfg = json.loads((Path(__file__).parents[1] / 'config/composite_controller_rank32_v2.json').read_text())
        rows = [{'state_id':'unit_task_0','value':1.0,'task_only':True},
                {'state_id':'unit_task_1','value':2.0,'task_only':True}]
        with TemporaryDirectory() as directory:
            actor = TinyFitActor({'phase':'qualification','optimizer_updates':1})
            result = fit_selected_rows(actor, rows, {'name':'tiny'}, cfg, directory,
                                       lambda:None, lambda *a,**k:None, qualification_rows=rows)
            self.assertEqual(result['optimizer_updates'], 1)
            self.assertFalse(result['full_fit_completed'])
            actor.verify_frozen()

    def test_raw_clipped_gradient_and_moments_recompute_the_actual_tiny_scratch_update(self):
        import torch
        from tempfile import TemporaryDirectory
        from tests.test_composite_v2_training import TinyFitActor
        from src.composite_behavior import fit_selected_rows
        from src.composite_v2_raw_exports import write_gradient_state, write_optimizer_state
        from src.composite_task_skill_model_audit import audit_scratch_update
        from src.composite_training import tensor_digest
        cfg=json.loads((Path(__file__).parents[1]/'config/composite_controller_rank32_v2.json').read_text())
        rows=[{'state_id':'unit_task_a','value':1.0},{'state_id':'unit_task_b','value':2.0}]
        with TemporaryDirectory() as directory:
            root=Path(directory);initial={}
            actor=TinyFitActor({'phase':'qualification','optimizer_updates':1})
            def observer(event,**value):
                if event=='after_configure':
                    initial.update({n:tensor_digest(t) for n,t in value['initial_weights'].items()})
                    group=value['optimizer'].param_groups[0]
                    (root/'OPTIMIZER_CONFIG.json').write_text(json.dumps({k:group[k] for k in
                        ['lr','betas','eps','weight_decay','foreach','maximize','capturable','differentiable']}))
                elif event=='before_optimizer_step':write_gradient_state(actor.adapters,root)
                elif event=='after_optimizer_step':write_optimizer_state(value['optimizer'],actor.adapters,root)
            fit_selected_rows(actor,rows,{'name':'tiny'},cfg,root,lambda:None,lambda *a,**k:None,
                              observer=observer,qualification_rows=rows)
            class Files:
                def get(self,name):return (root/name).read_bytes()
                def json(self,name):return json.loads(self.get(name))
            proof=audit_scratch_update(Files(),{'layer.proj.A':[32,2],'layer.proj.B':[2,32]},initial)
            self.assertTrue(proof['scratch_AdamW_update_independently_recomputed'])
            self.assertEqual(proof['full_clipped_gradient_tensors_verified'],2)

    def test_public_booleans_cannot_create_qualification_or_provider_release(self):
        from src.composite_task_skill_model_audit import TaskQualification,KaggleClosure,kaggle_closure
        with self.assertRaises(PermissionError):TaskQualification({'qualified':True},key=True)
        with self.assertRaises(PermissionError):KaggleClosure({'released':True},key=True)
        with self.assertRaises(PermissionError):kaggle_closure({'scriptVersionId':True},'a'*64,'success','0 Active Events',{},[],[])


if __name__ == '__main__':
    unittest.main()
