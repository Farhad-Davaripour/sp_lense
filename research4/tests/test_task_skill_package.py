import unittest
from src.composite_task_skill_package import TaskDevelopmentInputs,FrozenTaskPackage,read_task_development_inputs,frozen_task_package
from src.composite_task_skill_model_audit import TaskQualification,require_full_fit,verify_fresh_initialization


class TaskPackageBoundaryTests(unittest.TestCase):
    def test_scratch_running_summary_and_public_booleans_cannot_admit_evaluation(self):
        for value in [True,None,{'full_fit_completed':True,'owned_compute_released':True},
                      {'phase':'qualification','scientific_qualification_admitted':True}]:
            with self.assertRaises(PermissionError):require_full_fit(value)
            with self.assertRaises(PermissionError):frozen_task_package(value,{},'a'*40)
        with self.assertRaises(PermissionError):TaskQualification({'phase':'full_fit'},key=True)

    def test_data_and_package_receipts_are_not_reader_proofs(self):
        with self.assertRaises(PermissionError):TaskDevelopmentInputs({}, {},key=True)
        with self.assertRaises(PermissionError):FrozenTaskPackage({'full_fit_completed_and_raw_audited':True},key=True)
        for raw in [b'',b'{}',b'public fixture is not a scientific task dataset']:
            with self.assertRaises(PermissionError):read_task_development_inputs(raw)

    def test_comparison_uses_initial_bytes_rather_than_trained_parent_checkpoint(self):
        initial={'public.A':'a'*64,'public.B':'b'*64}
        self.assertTrue(verify_fresh_initialization(initial,dict(initial)))
        for changed in [{'public.A':'a'*64,'public.B':'c'*64},{'public.A':'a'*64},{}]:
            with self.assertRaises((PermissionError,ValueError)):verify_fresh_initialization(changed,initial)
        with self.assertRaises(PermissionError):verify_fresh_initialization(initial,None)


if __name__=='__main__':unittest.main()
