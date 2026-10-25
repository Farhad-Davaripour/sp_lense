"""Isolated public ledger fixtures; no scientific source or cloud submission."""
from datetime import datetime,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from src.io_utils import dump,read
from src.composite_kaggle_v2_profile import HUMAN_SHA
from src.composite_task_skill_evaluation_admission import reserve_evaluation,reconcile_evaluation,candidate_key


class TaskEvaluationAdmissionTests(unittest.TestCase):
    def fixture(self,root):
        binding={'source_commit':'a'*40,'phase':'task_development','case_id':'b'*64,
          'authorization_sha256':'c'*64,'fit_run_id':361964150,'fit_source_commit':'d'*40,
          'fit_raw_archive_sha256':'e'*64,'checkpoint_sha256':'f'*64,
          'provider_scope':'human-selected-Kaggle-remainder-v1','kaggle_remainder_scope_sha256':HUMAN_SHA,'fresh_stage':True}
        closeout={k:binding[v] for k,v in [('scriptVersionId','fit_run_id'),('source_commit','fit_source_commit'),
                    ('raw_archive_sha256','fit_raw_archive_sha256'),('checkpoint_sha256','checkpoint_sha256')]}
        closeout.update(full_fit_completed=True,full_fit_raw_audited=True,owned_compute_released=True)
        dump(root/'PUBLIC_CLOSEOUT.json',closeout)
        ledger={'active_owned_submission':None,'candidates':{'task_skill_stateful_memory_full_fit_v3':{
          'status':'public_fixture_complete','owned_compute_released':True,'closeout_receipt':str(root/'PUBLIC_CLOSEOUT.json')}}}
        dump(root/'SCIENTIFIC_EXECUTION_LEDGER.json',ledger)
        return binding,ledger

    def live(self):return {'at_UTC':datetime.now(timezone.utc).isoformat(),'surfaces':{'kaggle':0}}

    def test_existing_or_ambiguous_worker_blocks_before_any_dependency_reaudit(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);binding,ledger=self.fixture(root)
            for run in [361964150,None]:
                ledger['active_owned_submission']={'scriptVersionId':run,'status':'submitted_or_unknown'}
                dump(root/'SCIENTIFIC_EXECUTION_LEDGER.json',ledger)
                with patch('src.composite_task_skill_evaluation_admission.validate_preparation') as validator:
                    with self.assertRaises(PermissionError):reserve_evaluation(root,'public_fixture.json',self.live)
                    validator.assert_not_called()
                self.assertFalse((root/'COMPUTE_SUBMISSION.lock').exists())
                self.assertEqual(read(root/'SCIENTIFIC_EXECUTION_LEDGER.json'),ledger)

    def test_wrong_closeout_or_unreleased_parent_cannot_consume_a_case(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);binding,ledger=self.fixture(root)
            closeout=read(root/'PUBLIC_CLOSEOUT.json');closeout['checkpoint_sha256']='0'*64
            dump(root/'PUBLIC_CLOSEOUT.json',closeout)
            with patch('src.composite_task_skill_evaluation_admission.validate_preparation',return_value=binding):
                with self.assertRaises(PermissionError):reserve_evaluation(root,'public_fixture.json',self.live)
            self.assertEqual(read(root/'SCIENTIFIC_EXECUTION_LEDGER.json'),ledger)
            self.assertFalse((root/'COMPUTE_SUBMISSION.lock').exists())

    def test_consumed_case_identity_survives_source_changes_and_exact_version_reconciliation(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);binding,_=self.fixture(root)
            with patch('src.composite_task_skill_evaluation_admission.validate_preparation',return_value=binding):
                first=reserve_evaluation(root,'public_fixture.json',self.live)
            self.assertTrue(first['guard_retained_through_submission'])
            value=reconcile_evaluation(root,first['identity'],binding['case_id'],100,first['guard_token'])
            self.assertEqual(value['scriptVersionId'],100)
            ledger=read(root/'SCIENTIFIC_EXECUTION_LEDGER.json');ledger['active_owned_submission']=None
            key=candidate_key(binding['case_id']);ledger['candidates'][key].update(status='public_fixture_complete',owned_compute_released=True)
            dump(root/'SCIENTIFIC_EXECUTION_LEDGER.json',ledger)
            changed={**binding,'source_commit':'1'*40,'authorization_sha256':'2'*64}
            with patch('src.composite_task_skill_evaluation_admission.validate_preparation',return_value=changed):
                with self.assertRaises(PermissionError):reserve_evaluation(root,'different_public_source.json',self.live)
            self.assertEqual(read(root/'SCIENTIFIC_EXECUTION_LEDGER.json'),ledger)
            self.assertFalse((root/'COMPUTE_SUBMISSION.lock').exists())


if __name__=='__main__':unittest.main()
