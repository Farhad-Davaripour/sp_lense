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


class FrozenAdmissionTests(unittest.TestCase):
    def setup_ledger(self,root):
        value={'candidates':{recovery.CANDIDATE:{'scriptVersionId':361312460,
            'status':'scientific_timeout_partial_checkpoint_audited','owned_compute_released':True,
            'raw_audit_pending':False}}}
        (root/'SCIENTIFIC_EXECUTION_LEDGER.json').write_text(json.dumps(value))
        return value

    def live(self):
        return {'at_UTC':datetime.now(timezone.utc).isoformat(),'surfaces':{name:0 for name in shared.REQUIRED_SURFACES}}

    def binding(self):
        return {'source_commit':'a'*40,'notebook':'farhad89/synthetic-not-cloud-authority',
                'authorization_sha256':'b'*64,'package_sha256':'c'*64,'fresh_stage':True,
                'human_message_evidence_locator':'synthetic-unit-test-only','scientific_visible_GPU_count':2}

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
