from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from src.io_utils import dump,read
from src.composite_frozen_risk_admission import reserve_stage,candidate_key


class FrozenRiskAdmissionBoundaryTests(unittest.TestCase):
    def test_existing_worker_blocks_before_any_large_raw_dependency_read(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);ledger={'active_owned_submission':{'scriptVersionId':361964150},'candidates':{}}
            dump(root/'SCIENTIFIC_EXECUTION_LEDGER.json',ledger)
            with patch('src.composite_frozen_risk_admission.validate_preparation') as validator:
                with self.assertRaises(PermissionError):reserve_stage(root,'missing',{})
                validator.assert_not_called()
            self.assertEqual(read(root/'SCIENTIFIC_EXECUTION_LEDGER.json'),ledger)
            self.assertFalse((root/'COMPUTE_SUBMISSION.lock').exists())

    def test_risk_needs_genuine_fixed_transfer_evidence_not_a_boolean(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);ledger={'active_owned_submission':None,'candidates':{}};dump(root/'SCIENTIFIC_EXECUTION_LEDGER.json',ledger)
            binding={'phase':'risk_development','partition_index':0,'units':[{'case_id':'a'*64,'arm':'base'}]}
            with patch('src.composite_frozen_risk_admission.validate_preparation',return_value=binding):
                for supplied in [None,True,{'stage_integrity_admitted':True,'owned_compute_released':True}]:
                    with self.assertRaises(PermissionError):reserve_stage(root,'public fixture',{},transfer_audit=supplied)
            self.assertEqual(read(root/'SCIENTIFIC_EXECUTION_LEDGER.json'),ledger)
            self.assertFalse((root/'COMPUTE_SUBMISSION.lock').exists())

    def test_unit_identity_is_stable_across_source_and_order_changes(self):
        left={'phase':'risk_development','source_commit':'a'*40,'units':[{'case_id':'1'*64,'arm':'base'},{'case_id':'2'*64,'arm':'always_on'}]}
        right={**left,'source_commit':'b'*40,'units':list(reversed(left['units']))}
        self.assertEqual(candidate_key(left),candidate_key(right))
        with self.assertRaises(PermissionError):candidate_key({**left,'units':[left['units'][0]]*2})


if __name__=='__main__':unittest.main()
