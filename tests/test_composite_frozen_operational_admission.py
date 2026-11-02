import copy,json,tempfile
from datetime import datetime,timezone
from pathlib import Path
import unittest
from unittest.mock import patch
from src import composite_frozen_operational_admission as admission
from src import composite_frozen_base_admission as original
from src import composite_submission_admission as shared
from src.io_utils import sha


class OperationalAdmissionTests(unittest.TestCase):
    def ledger(self,root):
        records={}
        for number in range(3):
            records[original.PREFIX+str(number).zfill(2)]={
                'complete_base_partition_audited':True,'owned_compute_released':True,
                original.ATTEMPTS:[{'source_commit':admission.ORIGINAL_SOURCE,
                    'common_freeze_sha256':admission.PARENT_COMMON_FREEZE,'case_ids':['old'+str(number)]}]}
        closeout=root/'failed-parent.json';closeout.write_text('synthetic failed-parent fixture')
        records[original.PREFIX+'03']={'current_run_id':admission.PARENT_RUN,
            'scientific_execution_started':True,'owned_compute_released':True,'owned_cache_removed':True,
            'forward_counts':{'attempted':1260,'completed':1260},'complete_base_partition_audited':False,
            'closeout_receipt':str(closeout),original.ATTEMPTS:[{'identity':'closed-consumed-parent','case_ids':['remaining3']}]}
        value={'active_owned_submission':None,'candidates':records}
        self.save(root,value);return value,sha(closeout)

    def save(self,root,value):
        (root/'SCIENTIFIC_EXECUTION_LEDGER.json').write_text(json.dumps(value))

    def binding(self,number=3):
        return {'source_commit':'a'*40,'notebook':'farhad89/synthetic-not-cloud-authority',
            'package_sha256':'b'*64,'authorization_sha256':str(number)*64,'fresh_stage':True,
            'human_message_evidence_locator':'synthetic-unit-test-only','partition':number,
            'case_ids':['remaining'+str(number)],'scientific_visible_GPU_count':2,
            'mode':original.MODE,'common_freeze_sha256':'c'*64}

    def live(self):
        return {'at_UTC':datetime.now(timezone.utc).isoformat(),
            'surfaces':{key:0 for key in shared.REQUIRED_SURFACES}}

    def test_parent_identity_is_immutable_and_unknown_submission_never_becomes_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);before,parent_sha=self.ledger(root)
            with patch.object(admission,'PARENT_CLOSEOUT_SHA256',parent_sha),patch.object(admission,'validate_preparation',return_value=self.binding()):
                reserved=admission.reserve_once(root,'synthetic',self.live)
                admission.reconcile_submission(root,reserved['identity'],guard_token=reserved['guard_token'])
                with self.assertRaises(PermissionError):admission.reserve_once(root,'synthetic',self.live)
                admission.reconcile_submission(root,reserved['identity'],123)
                with self.assertRaises(PermissionError):admission.reconcile_submission(root,reserved['identity'],456)
            after=json.loads((root/'SCIENTIFIC_EXECUTION_LEDGER.json').read_text())
            self.assertEqual(after['candidates'][original.PREFIX+'03'],before['candidates'][original.PREFIX+'03'])

    def test_missing_failed_parent_counter_freeze_release_or_prior_stage_blocks_before_consumption(self):
        for kind in ['counts','release','complete','closeout','old_freeze','case_overlap','sequence','new_freeze']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);value,parent_sha=self.ledger(root);binding=self.binding()
                parent=value['candidates'][original.PREFIX+'03']
                if kind=='counts':parent['forward_counts']['completed']=None
                elif kind=='release':parent['owned_cache_removed']=False
                elif kind=='complete':parent['complete_base_partition_audited']=True
                elif kind=='closeout':parent_sha='0'*64
                elif kind=='old_freeze':value['candidates'][original.PREFIX+'00'][original.ATTEMPTS][0]['common_freeze_sha256']='d'*64
                elif kind=='case_overlap':binding['case_ids']=['old0']
                elif kind=='sequence':binding=self.binding(4)
                else:value['candidates'][admission.PREFIX+'03']={'owned_compute_released':True,admission.ATTEMPTS:[{'common_freeze_sha256':'d'*64}]}
                self.save(root,value);before=(root/'SCIENTIFIC_EXECUTION_LEDGER.json').read_bytes()
                with patch.object(admission,'PARENT_CLOSEOUT_SHA256',parent_sha),patch.object(admission,'validate_preparation',return_value=binding):
                    with self.assertRaises(PermissionError):admission.reserve_once(root,'synthetic',self.live)
                self.assertEqual((root/'SCIENTIFIC_EXECUTION_LEDGER.json').read_bytes(),before)
                self.assertFalse((root/'COMPUTE_SUBMISSION.lock').exists())

    def test_original_admission_cannot_duplicate_operational_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            from test_composite_frozen_base_admission import BaseAdmissionTests
            fixtures=BaseAdmissionTests();value=fixtures.ledger(root);binding=fixtures.binding()
            binding['case_ids']=['claimed']
            value['candidates'][admission.PREFIX+'04']={'owned_compute_released':True,
                admission.ATTEMPTS:[{'case_ids':['claimed']}]};self.save(root,value)
            with patch.object(original,'validate_preparation',return_value=binding):
                with self.assertRaisesRegex(PermissionError,'already owns'):original.reserve_once(root,'synthetic',self.live)


if __name__=='__main__':unittest.main()
