"""Public raw-prefix and authority fixtures, never scientific outcomes."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src.io_utils import dump,sha
from src.composite_exports import make_evidence_archive
from src.composite_frozen_risk_audit import audit_stage,FrozenRiskEvidence,RiskArchive,require_stage
from src.composite_v2_helper_audit import ordinary_task_training_exposure
from src.composite_frozen_risk_runtime import trace_replay_scope


class FrozenRiskRawBoundaryTests(unittest.TestCase):
    def test_flags_do_not_replace_any_model_data_or_native_raw_reader(self):
        for value in [None,True,{'stage_integrity_admitted':True,'raw_evidence_audited':True,'owned_compute_released':True}]:
            with self.assertRaises(PermissionError):require_stage(value)
            with self.assertRaises(PermissionError):ordinary_task_training_exposure(value)
            with self.assertRaises(PermissionError):
                with trace_replay_scope(value,{}):pass
        with self.assertRaises(PermissionError):FrozenRiskEvidence({},key=True)
        with self.assertRaises(PermissionError):RiskArchive(None,[],key=True)

    def test_incomplete_risk_prefix_keeps_exact_counts_unknown(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);output=root/'raw';output.mkdir();dump(output/'RUN_RECEIPT.json',{'status':'failed','forward_counts':None})
            archive=root/'evidence.zip';make_evidence_archive(output,archive)
            context=SimpleNamespace(model=object(),data=object(),stage=lambda:{'phase':'risk_development','version':'public_fixture',
              'partition':{'index':0},'maximum_raw_evidence_bytes':4*1024**3})
            with patch('src.composite_frozen_risk_audit.validate_source',return_value=context),patch(
              'src.composite_frozen_risk_audit.require_frozen_task_package',return_value={'checkpoint_sha256':'a'*64}),patch(
              'src.composite_frozen_risk_audit.require_data',return_value={'release_sha256':'b'*64}):
                proof=audit_stage(archive,sha(archive),root,'a'*40,'missing','missing','missing')
            result=proof.receipt();self.assertIsNone(result['recorded_forward_counts'])
            self.assertIsNone(result['completed_trajectories']);self.assertIsNone(result['owned_compute_released'])
            self.assertFalse(result['stage_integrity_admitted']);self.assertEqual(result['new_model_forwards'],0)
            with self.assertRaises(PermissionError):require_stage(proof)


if __name__=='__main__':unittest.main()
