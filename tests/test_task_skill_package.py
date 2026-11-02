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

    def test_registered_memory_fit_is_only_an_identity_not_completion_evidence(self):
        from src.composite_task_skill_fit_registry import registered_fit
        old='6c37f59b60ceff32c74b1c39f0a6b031d207e443'
        new='82cf3ecf41e799facd93b8c11e05c6d1f569599c'
        first=registered_fit(old,361922334);second=registered_fit(new,361964150)
        self.assertNotEqual(first['qualification_archive_sha256'],second['qualification_archive_sha256'])
        self.assertEqual(second['prepared_notebook'],'research4_task_skill_fit_v3.ipynb')
        for commit,run in [(old,361964150),(new,361922334),(new,True),(new,'361964150'),('a'*40,361964150)]:
            with self.assertRaises(PermissionError):registered_fit(commit,run)
        second['package_sha256']='a'*64
        self.assertNotEqual(second,registered_fit(new,361964150))
        for receipt in [first,second,registered_fit(new,361964150)]:
            with self.assertRaises(PermissionError):require_full_fit(receipt)

    def test_new_memory_fit_cannot_substitute_the_old_one_update_parent(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from src.io_utils import dump,sha
        from src.composite_task_skill_fit_registry import registered_fit
        from src.composite_task_skill_fit_parent import FIT_FILES,read_full_fit_parent
        current=registered_fit('82cf3ecf41e799facd93b8c11e05c6d1f569599c',361964150)
        with TemporaryDirectory() as folder:
            root=Path(folder)
            for name in FIT_FILES:(root/name).write_text('{}',encoding='utf-8')
            expected={k:current[k] for k in ['source_commit','run_id','package_sha256','qualification_archive_sha256']}
            expected.update(raw_archive_sha256='a'*64,checkpoint_sha256='b'*64,
                            fit_evidence_identity_sha256='c'*64,raw_audit_canonical_sha256='d'*64)
            expected['qualification_archive_sha256']=registered_fit(
                '6c37f59b60ceff32c74b1c39f0a6b031d207e443',361922334)['qualification_archive_sha256']
            dump(root/'FIT_PARENT_EXPECTED.json',expected)
            dump(root/'MANIFEST.json',{n:sha(root/n) for n in FIT_FILES})
            with self.assertRaisesRegex(PermissionError,'registered exact full-fit lineage'):
                read_full_fit_parent(root,'missing fit','missing scratch')

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
