import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import composite_frozen_base_admission as admission
from src import composite_submission_admission as shared


class BaseAdmissionTests(unittest.TestCase):
    def ledger(self, root):
        value = {'active_owned_submission':None,'candidates':{
            'detector_rank32_v5':{'owned_compute_released':True,'authorized_recovery_attempts':[
                {'scriptVersionId':361470090,'full_validation_complete':True,
                 'component_validation_admitted':True,'owned_compute_released':True}]},
            'frozen_qwen_two_t4_metadata_qualification_v2':{'current_run_id':361504592,
                'complete_qualification_audited':True,'owned_compute_released':True}}}
        (root/'SCIENTIFIC_EXECUTION_LEDGER.json').write_text(json.dumps(value)); return value

    def binding(self, partition=0):
        return {'source_commit':'a'*40,'notebook':'farhad89/synthetic-not-cloud-authority',
            'package_sha256':'b'*64,'authorization_sha256':str(partition)*64,'fresh_stage':True,
            'human_message_evidence_locator':'synthetic-unit-test-only','partition':partition,
            'scientific_visible_GPU_count':2,'mode':admission.MODE,'common_freeze_sha256':'c'*64}

    def live(self):
        return {'at_UTC':datetime.now(timezone.utc).isoformat(),'surfaces':{key:0 for key in shared.REQUIRED_SURFACES}}

    def test_unknown_or_unreleased_parent_never_consumes_a_base_partition(self):
        for changed in ['qualification','recognition']:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); value=self.ledger(root)
                if changed=='qualification':value['candidates']['frozen_qwen_two_t4_metadata_qualification_v2']['owned_compute_released']=None
                else:value['candidates']['detector_rank32_v5']['authorized_recovery_attempts'][0]['full_validation_complete']=False
                (root/'SCIENTIFIC_EXECUTION_LEDGER.json').write_text(json.dumps(value)); before=(root/'SCIENTIFIC_EXECUTION_LEDGER.json').read_bytes()
                with patch.object(admission,'validate_preparation',return_value=self.binding()):
                    with self.assertRaises(PermissionError):admission.reserve_once(root,'synthetic',self.live)
                self.assertEqual((root/'SCIENTIFIC_EXECUTION_LEDGER.json').read_bytes(),before)
                self.assertFalse((root/'COMPUTE_SUBMISSION.lock').exists())

    def test_exact_version_binding_and_consumption_survive_unknown_ui_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); before=self.ledger(root)
            with patch.object(admission,'validate_preparation',return_value=self.binding()):
                value=admission.reserve_once(root,'synthetic',self.live)
            self.assertTrue(value['authority_consumed']); self.assertTrue((root/'COMPUTE_SUBMISSION.lock').exists())
            result=admission.reconcile_submission(root,value['identity'],guard_token=value['guard_token'])
            self.assertEqual(result['status'],'submission_outcome_unknown')
            with patch.object(admission,'validate_preparation',return_value=self.binding()):
                with self.assertRaises(PermissionError):admission.reserve_once(root,'synthetic',self.live)
            result=admission.reconcile_submission(root,value['identity'],123)
            self.assertEqual(result['scriptVersionId'],123)
            with self.assertRaises(PermissionError):admission.reconcile_submission(root,value['identity'],456)
            current=json.loads((root/'SCIENTIFIC_EXECUTION_LEDGER.json').read_text())
            self.assertEqual(current['candidates']['detector_rank32_v5'],before['candidates']['detector_rank32_v5'])

    def test_subsequent_partition_cannot_change_the_scientific_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); value=self.ledger(root)
            value['candidates'][admission.PREFIX+'00']={'status':'complete','owned_compute_released':True,
                admission.ATTEMPTS:[{'common_freeze_sha256':'c'*64,'identity':'prior-synthetic-partition'}]}
            (root/'SCIENTIFIC_EXECUTION_LEDGER.json').write_text(json.dumps(value))
            binding=self.binding(1); binding['common_freeze_sha256']='d'*64
            with patch.object(admission,'validate_preparation',return_value=binding):
                with self.assertRaisesRegex(PermissionError,'one immutable'):
                    admission.reserve_once(root,'synthetic',self.live)


if __name__=='__main__':unittest.main()
