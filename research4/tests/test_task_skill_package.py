import unittest
from unittest.mock import patch
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

    def test_raw_tensor_identity_excludes_only_derived_reduction_metrics(self):
        from copy import deepcopy
        from src.composite_task_skill_package import full_fit_identity
        # Metadata fixture cannot create a model/package permit.
        value={k:'public' for k in ['phase','source_commit','provider_run_id','raw_archive_sha256',
               'source_data_archive_sha256','checkpoint_sha256','initial_adapter_hashes','recorded_forward_counts']}
        value['checkpoint']={'adapter_parameters':2,'tensors':2,'hashes':{'unit':'a'*64},'tensor_shapes':{'unit':[2]}}
        value['optimizer']={'clipped_global_norm':0.9999999}
        with patch('src.composite_task_skill_model_audit.require_full_fit',side_effect=lambda v:v):
            first=full_fit_identity(value);second=deepcopy(value);second['optimizer']['clipped_global_norm']+=1e-15
            self.assertEqual(first,full_fit_identity(second))
            second['checkpoint']['hashes']['unit']='b'*64
            self.assertNotEqual(first,full_fit_identity(second))

    def test_missing_full_parent_metadata_never_loads_a_model_or_admits_a_stage(self):
        from tempfile import TemporaryDirectory
        from src.composite_task_skill_fit_parent import read_full_fit_parent
        with TemporaryDirectory() as folder:
            with self.assertRaises(PermissionError):read_full_fit_parent(folder,'missing','missing')

    def test_only_tiny_cpu_gradient_reduction_roundoff_is_allowed(self):
        from copy import deepcopy
        from src.composite_task_skill_package import verify_audit_reference,restore_frozen_task_package
        current={'optimizer':{'clipped_global_norm':0.5},'checkpoint_sha256':'a'*64,'counter':64}
        original=deepcopy(current);original['optimizer']['clipped_global_norm']+=1e-15
        self.assertTrue(verify_audit_reference(original,current))
        for change in ['counter','hash','norm']:
            value=deepcopy(original)
            if change=='counter':value['counter']=63
            elif change=='hash':value['checkpoint_sha256']='b'*64
            else:value['optimizer']['clipped_global_norm']=0.501
            with self.assertRaises(PermissionError):verify_audit_reference(value,current)
        with self.assertRaises(PermissionError):restore_frozen_task_package({},original,True,{},'a'*40)


if __name__=='__main__':unittest.main()
