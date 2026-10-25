import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src.composite_exports import make_evidence_archive
from src.composite_task_skill_evaluation_audit import TaskDevelopmentAudit,require_task_partition,audit_collection


class TaskEvaluationAuditBoundaryTests(unittest.TestCase):
    def test_flags_and_running_or_partial_results_cannot_admit_a_partition(self):
        for value in [True,None,{'task_partition_admitted':True,'raw_evidence_audited':True,'owned_compute_released':True}]:
            with self.assertRaises(PermissionError):require_task_partition(value)
        with self.assertRaises(PermissionError):TaskDevelopmentAudit({'task_partition_admitted':True},key=True)

    def test_failed_prefix_preserves_unknown_counts_instead_of_calling_them_zero(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';raw.mkdir();archive=root/'evidence.zip'
            (raw/'RUN_RECEIPT.json').write_text(json.dumps({'status':'failed','forward_counts':None}))
            make_evidence_archive(raw,archive)
            context=SimpleNamespace(stage=lambda:{'partition':{'index':0,'case_ids':['public_fixture']},
              'maximum_raw_evidence_bytes':4*1024**3},model=object())
            model={'fit_run_id':361964150,'checkpoint_sha256':'a'*64}
            with patch('src.composite_task_skill_evaluation_audit.validate_source',return_value=context),patch(
              'src.composite_task_skill_evaluation_audit.require_frozen_task_package',return_value=model):
                proof=audit_collection(archive,hashlib.sha256(archive.read_bytes()).hexdigest(),root,'a'*40,'missing','missing')
            value=proof.receipt()
            self.assertIsNone(value['recorded_forward_counts']);self.assertIsNone(value['completed_trajectories'])
            self.assertIsNone(value['recorded_optimizer_updates']);self.assertIsNone(value['owned_compute_released'])
            self.assertFalse(value['task_partition_admitted']);self.assertFalse(value['raw_evidence_audited'])
            self.assertEqual(value['new_model_forwards'],0)
            with self.assertRaises(PermissionError):require_task_partition(proof)


if __name__=='__main__':unittest.main()
