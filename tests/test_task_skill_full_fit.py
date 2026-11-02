import unittest
from unittest.mock import patch
from src.composite_task_skill_full_fit import full_fit_plan
from src.composite_task_skill_model_audit import TaskQualification


class FullTaskFitBoundaryTests(unittest.TestCase):
    def test_caller_assertions_do_not_promote_a_full_fit(self):
        for supplied in [True, {'complete_training_qualification_audited':True,'owned_compute_released':True}, None]:
            with self.assertRaises(PermissionError):full_fit_plan('a'*40,supplied)
        with self.assertRaises(PermissionError):TaskQualification({'scientific_qualification_admitted':True},key=True)

    def test_conditional_metadata_preserves_recipe_and_never_warm_starts(self):
        # Public fixture exercises plan construction only; it cannot create
        # opaque raw qualification evidence or admit a real workload.
        proof={'source_commit':'b'*40,'provider_run_id':123,'raw_archive_sha256':'c'*64}
        with patch('src.composite_task_skill_model_audit.require_qualification',return_value=proof):
            plan=full_fit_plan('a'*40,object())
        self.assertEqual(plan['phase'],'full_fit');self.assertEqual(plan['optimizer_updates'],64)
        self.assertEqual(plan['accumulation'],2);self.assertEqual(plan['source_training_rows'],1188)
        self.assertFalse(plan['optimizer_sampler_or_RNG_warm_start'])
        self.assertFalse(plan['checkpoint_initializes_full_fit'])
        self.assertFalse(plan['metadata_plan_alone_admits_launch'])
        self.assertFalse(plan['risk_examples_or_outcomes_used_for_fitting_or_selection'])
        self.assertEqual(plan['hard_timeout_seconds'],42000)

    def test_full_fit_gradient_reader_checks_actual_tensors_and_global_clipping(self):
        import torch
        from src.composite_task_skill_model_audit import audit_clipped_gradients
        proof=audit_clipped_gradients({'a':torch.tensor([0.3,0.4])},{'a':[2]})
        self.assertAlmostEqual(proof['clipped_global_norm'],0.5,places=6)
        for values in [{}, {'a':torch.ones(2)}, {'a':torch.tensor([float('nan'),0.0])},
                       {'a':torch.zeros(2,dtype=torch.float64)}, {'a':torch.zeros(3)}]:
            with self.assertRaises(ValueError):audit_clipped_gradients(values,{'a':[2]})


if __name__=='__main__':unittest.main()
