import unittest
from unittest.mock import patch
from src.composite_task_memory_source import full_fit_plan
from src.composite_task_skill_model_audit import require_stateful_qualification


class StatefulFullFitBoundaryTests(unittest.TestCase):
    def test_one_update_running_summary_or_boolean_cannot_admit_a_new_fit(self):
        for value in [True,None,{'scientific_qualification_admitted':True},
                      {'qualified_optimizer_updates':1,'owned_compute_released':True}]:
            with self.assertRaises(PermissionError):require_stateful_qualification(value)
            with self.assertRaises(PermissionError):full_fit_plan('a'*40,value)

    def test_conditional_metadata_preserves_fresh_full64_recipe(self):
        proof={'source_commit':'b'*40,'provider_run_id':361949668,'raw_archive_sha256':'c'*64}
        with patch('src.composite_task_skill_model_audit.require_stateful_qualification',return_value=proof):
            plan=full_fit_plan('a'*40,object())
        self.assertEqual(plan['phase'],'full_fit');self.assertEqual(plan['optimizer_updates'],64)
        self.assertEqual(plan['source_training_rows'],1188);self.assertEqual(plan['accumulation'],2)
        self.assertEqual(plan['seed'],202610212);self.assertEqual(plan['learning_rate'],0.0005)
        self.assertEqual(plan['maximum_raw_evidence_bytes'],4*1024**3)
        self.assertFalse(plan['checkpoint_initializes_full_fit']);self.assertFalse(plan['optimizer_sampler_or_RNG_warm_start'])
        self.assertTrue(plan['additional_started_full_fit_attempt']);self.assertFalse(plan['exact_resume_from12'])
        self.assertEqual(plan['previous_completed_update_journals'],12)
        self.assertFalse(plan['metadata_plan_alone_admits_launch'])


if __name__=='__main__':unittest.main()
