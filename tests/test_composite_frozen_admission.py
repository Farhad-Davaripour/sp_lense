import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src import composite_frozen_admission as frozen
from src import composite_recovery_admission as recovery
from src import composite_submission_admission as shared
from src.composite_frozen_probe import PRE_MODEL_PARENT_PROFILE,PRE_MODEL_PARENT_CANDIDATE,MODE
from src.io_utils import sha


class FrozenAdmissionTests(unittest.TestCase):
    def setup_ledger(self,root):
        audit={'run_id':'1234','worker_status':'failed','complete_qualification_audited':False,
               'archive_sha256':'e'*64,'counts_exact_at_exit':True,'forward_counts':{'attempted':0,'completed':0}}
        audit_file=root/'PARENT_AUDIT.json';audit_file.write_text(json.dumps(audit))
        closeout={'run_id':1234,'source_commit':'b'*40,'mode':MODE,
            'error':'Actual model metadata bytes differ from the original Git blob',
            'audit_receipt':str(audit_file),'audit_sha256':sha(audit_file),'archive_sha256':'e'*64,
            'counts_exact_at_exit':True,'forward_counts':audit['forward_counts'],
            'pre_model_initialization_failure_verified':True,'owned_compute_released':True}
        closeout_file=root/'PARENT_CLOSEOUT.json';closeout_file.write_text(json.dumps(closeout))
        self.parent={'profile':PRE_MODEL_PARENT_PROFILE,'candidate':PRE_MODEL_PARENT_CANDIDATE,
            'run_id':'1234','source_commit':'b'*40,'archive_sha256':'e'*64,'audit_sha256':sha(audit_file),
            'closeout_sha256':sha(closeout_file),'counts_exact_at_exit':True,'forward_counts':{'attempted':0,'completed':0},
            'pre_model_initialization_failure_verified':True,'owned_compute_released':True}
        value={'candidates':{recovery.CANDIDATE:{'scriptVersionId':361312460,
            'status':'scientific_timeout_partial_checkpoint_audited','owned_compute_released':True,
            'raw_audit_pending':False},PRE_MODEL_PARENT_CANDIDATE:{
            **closeout,'status':'failed_before_model_initialization_independently_audited_GPU_released',
            'current_run_id':1234,'scientific_execution_started':False,'closeout_sha256':sha(closeout_file),'closeout_receipt':str(closeout_file),
            frozen.ATTEMPTS:[{'scriptVersionId':1234,'authority_consumed':True,'source_commit':'b'*40,
                'status':'failed_before_model_initialization_independently_audited_GPU_released','owned_compute_released':True}]}}}
        (root/'SCIENTIFIC_EXECUTION_LEDGER.json').write_text(json.dumps(value))
        return value

    def live(self):
        return {'at_UTC':datetime.now(timezone.utc).isoformat(),'surfaces':{name:0 for name in shared.REQUIRED_SURFACES}}

    def binding(self):
        return {'source_commit':'a'*40,'notebook':'farhad89/synthetic-not-cloud-authority',
                'authorization_sha256':'b'*64,'package_sha256':'c'*64,'fresh_stage':True,
                'human_message_evidence_locator':'synthetic-unit-test-only','scientific_visible_GPU_count':2,
                'pre_model_parent':copy.deepcopy(self.parent)}

    def test_unknown_nonzero_unreleased_or_changed_parent_cannot_authorize_recovery(self):
        for kind in ['unknown','nonzero','science_unknown','unreleased','source','closeout_changed','audit_changed','outside','not_consumed']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);ledger=self.setup_ledger(root);parent=ledger['candidates'][PRE_MODEL_PARENT_CANDIDATE]
                if kind=='unknown':parent['forward_counts']={'attempted':None,'completed':None}
                elif kind=='nonzero':parent['forward_counts']={'attempted':1,'completed':1}
                elif kind=='science_unknown':parent['scientific_execution_started']=None
                elif kind=='unreleased':parent['owned_compute_released']=False
                elif kind=='source':parent['source_commit']='c'*40
                elif kind=='closeout_changed':(root/'PARENT_CLOSEOUT.json').write_text('{}')
                elif kind=='audit_changed':(root/'PARENT_AUDIT.json').write_text('{}')
                elif kind=='outside':parent['closeout_receipt']=str(root.parent/'outside.json')
                else:parent[frozen.ATTEMPTS][0]['authority_consumed']=False
                (root/'SCIENTIFIC_EXECUTION_LEDGER.json').write_text(json.dumps(ledger))
                before=(root/'SCIENTIFIC_EXECUTION_LEDGER.json').read_bytes()
                with patch.object(frozen,'validate_preparation',return_value=self.binding()):
                    with self.assertRaises(PermissionError):frozen.reserve_once(root,'synthetic',self.live)
                self.assertEqual((root/'SCIENTIFIC_EXECUTION_LEDGER.json').read_bytes(),before)
                self.assertFalse((root/'COMPUTE_SUBMISSION.lock').exists())

    def test_distinct_allowance_consumed_before_submit_and_recognition_parent_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);before=self.setup_ledger(root)
            with patch.object(frozen,'validate_preparation',return_value=self.binding()):
                reservation=frozen.reserve_once(root,'synthetic',self.live)
            ledger=json.loads((root/'SCIENTIFIC_EXECUTION_LEDGER.json').read_text())
            self.assertEqual(ledger['candidates'][recovery.CANDIDATE],before['candidates'][recovery.CANDIDATE])
            candidate=ledger['candidates'][frozen.CANDIDATE]
            self.assertTrue(candidate[frozen.ATTEMPTS][0]['authority_consumed'])
            self.assertNotIn('authorized_recovery_attempts',candidate)
            self.assertTrue((root/'COMPUTE_SUBMISSION.lock').exists())
            result=frozen.reconcile_submission(root,reservation['identity'],guard_token=reservation['guard_token'])
            self.assertEqual(result['status'],'submission_outcome_unknown')
            with patch.object(frozen,'validate_preparation',return_value=self.binding()):
                with self.assertRaises(PermissionError):frozen.reserve_once(root,'synthetic',self.live)
            result=frozen.reconcile_submission(root,reservation['identity'],123)
            self.assertEqual(result['scriptVersionId'],123)
            with self.assertRaises(PermissionError):frozen.reconcile_submission(root,reservation['identity'],456)

    def test_cross_stage_exclusion_in_both_directions(self):
        for first,second in [(frozen,recovery),(recovery,frozen)]:
            with self.subTest(first=first.__name__),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);self.setup_ledger(root)
                binding=self.binding()
                if first is recovery:
                    binding.pop('fresh_stage');binding['parent_run_id']='361312460'
                with patch.object(first,'validate_preparation',return_value=binding):
                    reservation=first.reserve_once(root,'synthetic',self.live)
                first.reconcile_submission(root,reservation['identity'],123,reservation['guard_token'])
                before=(root/'SCIENTIFIC_EXECUTION_LEDGER.json').read_bytes()
                with patch.object(second,'validate_preparation',return_value=self.binding()):
                    with self.assertRaises(PermissionError):second.reserve_once(root,'synthetic',self.live)
                self.assertEqual((root/'SCIENTIFIC_EXECUTION_LEDGER.json').read_bytes(),before)

    def test_proposal_and_recognition_authority_do_not_consume_frozen_allowance(self):
        for status in ['proposal_requires_separate_human_two_device_exception','prepared_with_explicit_exception_not_submitted']:
            with self.subTest(status=status),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);self.setup_ledger(root)
                prep=root/'PREPARATION.json';prep.write_text(json.dumps({'status':status}))
                before=(root/'SCIENTIFIC_EXECUTION_LEDGER.json').read_bytes()
                with self.assertRaises(PermissionError):frozen.reserve_once(root,prep,self.live)
                self.assertEqual((root/'SCIENTIFIC_EXECUTION_LEDGER.json').read_bytes(),before)
                self.assertFalse((root/'COMPUTE_SUBMISSION.lock').exists())

    def test_executed_candidate_cannot_be_reintroduced_as_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);value=self.setup_ledger(root)
            value['candidates'][frozen.CANDIDATE]={'status':'complete','scientific_execution_started':True,'owned_compute_released':True}
            (root/'SCIENTIFIC_EXECUTION_LEDGER.json').write_text(json.dumps(value))
            with patch.object(frozen,'validate_preparation',return_value=self.binding()):
                with self.assertRaises(PermissionError):frozen.reserve_once(root,'synthetic',self.live)
