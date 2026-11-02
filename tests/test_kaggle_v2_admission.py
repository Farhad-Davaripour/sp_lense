from pathlib import Path
from datetime import datetime,timezone
import json,tempfile,unittest
from src.composite_submission_admission import reserve
from src.composite_kaggle_v2_profile import HUMAN_SHA
from src.composite_kaggle_v2_admission import candidate_rule


class KaggleRemainderAdmissionTests(unittest.TestCase):
    def test_selected_kaggle_scope_does_not_require_a_fictitious_databricks_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'SCIENTIFIC_EXECUTION_LEDGER.json').write_text(json.dumps({'candidates':{},'active_owned_submission':None}))
            binding={'authorization_sha256':'a'*64,'provider_scope':'human-selected-Kaggle-remainder-v1',
                'kaggle_remainder_scope_sha256':HUMAN_SHA,'fresh_stage':True,'scientific_GPU_count':0}
            def snapshot():return {'at_UTC':datetime.now(timezone.utc).isoformat(),'surfaces':{'kaggle':0},
                'evidence':'public model-free fixture, not actual provider availability'}
            result=reserve(root,'public-fixture',snapshot,'unit','attempts','unit-',lambda path:binding,candidate_rule)
            self.assertEqual(result['scientific_GPU_count'],0)
            self.assertTrue(result['guard_retained_through_submission'])
            # The fixture owns this exact guard; do not leave a fake reservation.
            (root/'COMPUTE_SUBMISSION.lock').unlink()
            with self.assertRaises(PermissionError):reserve(root,'public-fixture',snapshot,'unit','attempts','unit-',lambda path:binding,candidate_rule)

    def test_active_or_ambiguous_owned_candidate_still_blocks_selected_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'SCIENTIFIC_EXECUTION_LEDGER.json').write_text(json.dumps({'candidates':{},
                'active_owned_submission':{'identity':'existing','platform':'other preserved provider'}}))
            binding={'authorization_sha256':'a'*64,'provider_scope':'human-selected-Kaggle-remainder-v1',
                'kaggle_remainder_scope_sha256':HUMAN_SHA}
            snapshot=lambda:{'at_UTC':datetime.now(timezone.utc).isoformat(),'surfaces':{'kaggle':0}}
            with self.assertRaises(PermissionError):reserve(root,'fixture',snapshot,'unit','attempts','unit-',lambda path:binding,candidate_rule)


if __name__=='__main__':unittest.main()
