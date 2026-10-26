"""Public fake-only direct-waiver preparation/admission; no model or provider."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from src.io_utils import dump, read, sha
from src.composite_bounded_sampling_fit_profile import DIRECT_MODE, WAIVER_SHA
from src import composite_bounded_sampling_fit_admission as admission
from src.composite_bounded_sampling_admission import CANDIDATE as DATA_CANDIDATE
from src.composite_kaggle_v2_profile import HUMAN_SHA


class DirectFitAdmissionTests(unittest.TestCase):
    def fixture(self, base):
        base=Path(base);inputs=base/'preparation/bundle/source/inputs';inputs.mkdir(parents=True)
        cpu=base/'CPU_CLOSEOUT.json';dump(cpu,{'public_fake_cpu_release':True})
        closed=base/'INTERRUPTED_CLOSEOUT.json'
        dump(closed,{'scriptVersionId':362573972,'source_commit':'b'*40,
            'provider_terminal':True,'owned_compute_released':True,'qualification_admitted':False})
        prep={'interrupted_qualification_run_id':362573972,
            'interrupted_qualification_closeout_local_path':str(closed),
            'interrupted_qualification_closeout_sha256':sha(closed)}
        file=inputs.parents[2]/'PREPARATION.json';dump(file,prep)
        dump(inputs/'SAMPLING_DATA_PARENT_EXPECTED.json',{'closeout_sha256':sha(cpu)})
        ledger={'active_owned_submission':None,'candidates':{
            DATA_CANDIDATE:{'owned_compute_released':True,'closeout_receipt':str(cpu)},
            admission.QUALIFICATION:{'owned_compute_released':True,'closeout_receipt':str(closed),
                'attempts':[{'source_commit':'b'*40,'scriptVersionId':362573972,'authority_consumed':True}]}}}
        dump(base/'SCIENTIFIC_EXECUTION_LEDGER.json',ledger)
        binding={'mode':DIRECT_MODE,'phase':'full_fit','source_commit':'a'*40,
            'notebook':'farhad89/research4-bounded-sampling-direct-fit-v1',
            'authorization_sha256':'c'*64,'fresh_stage':True,
            'provider_scope':'human-selected-Kaggle-remainder-v1','kaggle_remainder_scope_sha256':HUMAN_SHA,
            'direct_fit_waiver_sha256':WAIVER_SHA,
            'interrupted_qualification_closeout_sha256':sha(closed)}
        return file,prep,closed,ledger,binding

    @staticmethod
    def fresh():return {'at_UTC':datetime.now(timezone.utc).isoformat(),'surfaces':{'kaggle':0}}

    def test_terminal_release_closeout_is_exact_and_not_capacity_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            _,prep,closed,_,_=self.fixture(folder)
            value=admission._interrupted_closeout(prep)
            self.assertFalse(value['qualification_admitted'])
            for field,bad in [('scriptVersionId',362573973),('scriptVersionId',True),
                              ('provider_terminal',False),('owned_compute_released',False)]:
                altered={**value,field:bad};dump(closed,altered)
                bound={**prep,'interrupted_qualification_closeout_sha256':sha(closed)}
                with self.subTest(field=field,bad=bad),self.assertRaises(PermissionError):
                    admission._interrupted_closeout(bound)
            dump(closed,value)
            with self.assertRaises(PermissionError):
                admission._interrupted_closeout({**prep,'interrupted_qualification_closeout_sha256':'0'*64})

    def test_direct_reservation_uses_cpu_parent_and_distinct_once_only_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            file,_,_,_,binding=self.fixture(folder)
            with patch.object(admission,'validate_preparation',return_value=binding):
                result=admission.reserve_stage(folder,file,self.fresh)
            ledger=read(Path(folder)/'SCIENTIFIC_EXECUTION_LEDGER.json')
            self.assertEqual(ledger['active_owned_submission']['candidate'],admission.DIRECT_FULL_FIT)
            self.assertNotIn(admission.FULL_FIT,ledger['candidates'])
            self.assertEqual(len(ledger['candidates'][admission.QUALIFICATION]['attempts']),1)
            self.assertTrue(result['authority_consumed'])
            self.assertTrue(result['guard_retained_through_submission'])
            reconciled=admission.reconcile_stage(folder,result['identity'],'full_fit',123,
                result['guard_token'],mode=DIRECT_MODE)
            self.assertEqual(reconciled['scriptVersionId'],123)
            self.assertFalse((Path(folder)/'COMPUTE_SUBMISSION.lock').exists())
            with patch.object(admission,'validate_preparation',return_value=binding):
                with self.assertRaises(PermissionError):admission.reserve_stage(folder,file,self.fresh)
            self.assertEqual(read(Path(folder)/'SCIENTIFIC_EXECUTION_LEDGER.json')['active_owned_submission']['scriptVersionId'],123)

    def test_unreleased_wrong_identity_or_changed_closeout_cannot_reserve_direct_fit(self):
        for defect in ['unreleased','wrong_run','changed_closeout','wrong_source']:
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as folder:
                file,_,closed,ledger,binding=self.fixture(folder)
                qualifier=ledger['candidates'][admission.QUALIFICATION]
                if defect=='unreleased':qualifier['owned_compute_released']=False
                elif defect=='wrong_run':qualifier['attempts'][0]['scriptVersionId']=999
                elif defect=='wrong_source':qualifier['attempts'][0]['source_commit']='c'*40
                else:dump(closed,{'changed_public_fake_closeout':True})
                dump(Path(folder)/'SCIENTIFIC_EXECUTION_LEDGER.json',ledger)
                with patch.object(admission,'validate_preparation',return_value=binding):
                    with self.assertRaises(PermissionError):admission.reserve_stage(folder,file,self.fresh)
                self.assertIsNone(read(Path(folder)/'SCIENTIFIC_EXECUTION_LEDGER.json')['active_owned_submission'])
                self.assertFalse((Path(folder)/'COMPUTE_SUBMISSION.lock').exists())

    def test_direct_preparation_binds_real_package_mode_waiver_and_released_closeout(self):
        from src.composite_bounded_sampling_fit_profile import MODE, MANIFEST, PLAN
        with tempfile.TemporaryDirectory() as folder:
            file,old,_,_,_=self.fixture(folder);base=file.parent;bundle=base/'bundle';source=bundle/'source'
            (source/'notebooks').mkdir()
            code_file=source/'notebooks/composite_bounded_sampling_fit.py'
            code_file.write_text("source='SOURCE_COMMIT_PLACEHOLDER'\nmanifest='PACKAGE_MANIFEST_SHA256_PLACEHOLDER'\n")
            stage={'version':DIRECT_MODE,'phase':'full_fit','notebook':'farhad89/research4-bounded-sampling-direct-fit-v1',
                'owner_thread_id':'01a175f0-fc99-76b1-8bb2-d64f22afbcd7','optimizer_updates':64,
                'source_data_archive_sha256':'d'*64,'data_audit_canonical_sha256':'e'*64,'hard_timeout_seconds':42000}
            dump(source/'inputs'/PLAN,stage);dump(source/'inputs/MANIFEST.json',{'FAKE_ONLY':True})
            dump(bundle/MANIFEST,{'version':DIRECT_MODE,'source_commit':'a'*40,
                'files':{p.relative_to(bundle).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}})
            with zipfile.ZipFile(base/admission.PACKAGE,'x') as stream:
                for p in bundle.rglob('*'):
                    if p.is_file():stream.write(p,p.relative_to(bundle).as_posix())
            code=admission.notebook_code(source,'a'*40,sha(bundle/MANIFEST))
            dump(base/admission.NOTEBOOK_FILE,{'cells':[{'cell_type':'code','source':code.splitlines(keepends=True)}]})
            value={**old,'phase':'full_fit','mode':DIRECT_MODE,'status':'prepared_bounded_behavior_full_fit_not_submitted',
                'owner_thread_id':stage['owner_thread_id'],'scientific_allocation_admitted_by_preparation':False,
                'model_forwards':0,'optimizer_updates':0,'heldout_content_read':False,
                'separate_qualification_waived':True,'separate_qualification_performed':False,
                'direct_fit_waiver_sha256':WAIVER_SHA,'qualification_archive_local_path':None,
                'data_archive_local_path':'FAKE_ONLY','alias_fit_archive_local_path':'FAKE_ONLY',
                'alias_qualification_archive_local_path':'FAKE_ONLY','source_commit':'a'*40,
                'bundle_sha256':sha(base/admission.PACKAGE),'bundle_manifest_sha256':sha(bundle/MANIFEST),
                'notebook_file_sha256':sha(base/admission.NOTEBOOK_FILE),'plan_sha256':sha(source/'inputs'/PLAN),
                'input_manifest_sha256':sha(source/'inputs/MANIFEST.json'),'notebook':stage['notebook'],
                'training_states':2,'source_data_archive_sha256':'d'*64,'data_audit_canonical_sha256':'e'*64}
            data=SimpleNamespace(fitting_rows=lambda:[{'state_id':'FAKE_ONLY::1'},{'state_id':'FAKE_ONLY::2'}])
            with patch.object(admission,'validate_source',return_value=(stage,{},data)):
                dump(file,value);bound=admission.validate_preparation(file)
                self.assertEqual(bound['mode'],DIRECT_MODE);self.assertEqual(bound['optimizer_updates'],64)
                self.assertTrue(bound['separate_qualification_waived'])
                self.assertFalse(bound['separate_qualification_performed'])
                for key,bad in [('mode',MODE),('qualification_archive_local_path','FAKE_CURRENT_QUALIFIER'),
                                ('direct_fit_waiver_sha256','0'*64),('separate_qualification_performed',True)]:
                    dump(file,{**value,key:bad})
                    with self.subTest(key=key),self.assertRaises(PermissionError):admission.validate_preparation(file)

    def test_direct_preparer_forbids_qualification_tensors_and_requires_release_evidence(self):
        from tools.prepare_bounded_sampling_fit import prepare
        args={key:'FAKE_ONLY_nonexistent' for key in ['repo','commit','model_inputs','tensor_metadata',
            'kaggle_scope','data_parent','data_archive','alias_fit_archive','alias_qualification_archive','output']}
        with self.assertRaises(PermissionError):
            prepare(**args,direct_waiver='FAKE_ONLY',qualification_parent='FAKE_ONLY',qualification_archive='FAKE_ONLY')
        with self.assertRaises(PermissionError):prepare(**args,direct_waiver='FAKE_ONLY')
        with self.assertRaises(PermissionError):prepare(**args,interrupted_qualification_closeout='FAKE_ONLY')

    def test_direct_bootstrap_asserts_fresh64_full_fit_before_any_setup(self):
        path=Path(__file__).parents[1]/'notebooks/composite_bounded_sampling_fit.py'
        tree=ast.parse(path.read_text())
        direct=next(node for node in tree.body if isinstance(node,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='direct' for t in node.targets))
        checks=next(node for node in tree.body if isinstance(node,ast.If)
            and isinstance(node.test,ast.Name) and node.test.id=='direct')
        program=compile(ast.Module(body=[direct,checks],type_ignores=[]),str(path),'exec')
        stage={'version':DIRECT_MODE,'phase':'full_fit','optimizer_updates':64,
            'separate_qualification_waived':True,'qualification_checkpoint_initializes_full_fit':False,
            'optimizer_sampler_or_RNG_warm_start':False}
        exec(program,{'stage':stage})
        for key,bad in [('phase','qualification'),('optimizer_updates',3),
                        ('separate_qualification_waived',False),('qualification_checkpoint_initializes_full_fit',True),
                        ('optimizer_sampler_or_RNG_warm_start',True)]:
            with self.subTest(key=key),self.assertRaises(AssertionError):exec(program,{'stage':{**stage,key:bad}})


if __name__=='__main__':unittest.main()
