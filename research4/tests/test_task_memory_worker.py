import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.composite_task_memory_profile import MODE
from src.composite_task_memory_source import StartedFailureEvidence,read_started_failure,qualification_plan


class StatefulWorkerBoundaryTests(unittest.TestCase):
    def test_current_qualification_is_three_updates_and_never_a_zero_forward_repair(self):
        plan=qualification_plan('a'*40)
        self.assertEqual(plan['version'],MODE);self.assertEqual(plan['optimizer_updates'],3)
        self.assertEqual(plan['previous_completed_update_journals'],12)
        self.assertFalse(plan['previous_failure_is_pre_execution']);self.assertFalse(plan['original_fit_identity_reset'])
        self.assertFalse(plan['qualification_checkpoint_initializes_full_fit'])
        self.assertEqual(plan['maximum_raw_evidence_bytes'],6*1024**3)
        with self.assertRaises(PermissionError):StartedFailureEvidence({'owned_compute_released':True},key=True)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(PermissionError):read_started_failure(folder)

    def test_shared_fitter_retains_moments_across_all_three_registered_updates(self):
        from tests.test_composite_v2_training import TinyFitActor
        from src.composite_behavior import fit_selected_rows
        cfg=json.loads((Path(__file__).parents[1]/'config/composite_controller_rank32_v2.json').read_text())
        rows=[{'state_id':'unit_'+str(i),'value':float(i+1)} for i in range(4)]
        selected=rows+rows[:2];resident=[]
        def observer(event,**value):
            if event=='before_microstep':resident.append((value['update'],value['microstep'],len(value['optimizer'].state)))
        with tempfile.TemporaryDirectory() as folder:
            actor=TinyFitActor({'phase':'qualification','optimizer_updates':3,'version':MODE})
            result=fit_selected_rows(actor,rows,{'name':'tiny'},cfg,folder,lambda:None,lambda *a,**k:None,
              observer=observer,qualification_rows=selected)
        self.assertEqual(result['optimizer_updates'],3);self.assertEqual(len(result['losses']),6)
        self.assertEqual(result['sampled_state_ids'],[r['state_id'] for r in selected])
        self.assertEqual(resident,[(1,0,0),(1,1,0),(2,0,2),(2,1,2),(3,0,2),(3,1,2)])
        self.assertFalse(result['full_fit_completed'])

    def test_admission_retains_started_parent_and_consumed_qualification_identity(self):
        from datetime import datetime,timezone
        from src.io_utils import sha
        from src.composite_kaggle_v2_profile import HUMAN_SHA
        from src.composite_task_skill_model_admission import reserve_stateful_qualification,reconcile_stateful_qualification
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);closed=root/'closed.json';closed.write_text('{"public_fixture":true}')
            parent={'status':'failed_started_full64_fit_OOM_partial_journals_audited_GPU_released',
              'owned_compute_released':True,'scientific_execution_started':True,'complete_resume_state_available':False,
              'closeout_receipt':str(closed),'actual_partial_counts':{'completed_update_journals':12}}
            ledger={'candidates':{'task_skill_full_fit_v1':parent},'active_owned_submission':None}
            path=root/'SCIENTIFIC_EXECUTION_LEDGER.json';path.write_text(json.dumps(ledger))
            binding={'source_commit':'a'*40,'notebook':'farhad89/research4-task-memory-qualification-v3',
              'mode':MODE,'phase':'qualification','optimizer_updates':3,'authorization_sha256':'b'*64,
              'fresh_stage':True,'provider_scope':'human-selected-Kaggle-remainder-v1','kaggle_remainder_scope_sha256':HUMAN_SHA}
            live=lambda:{'at_UTC':datetime.now(timezone.utc).isoformat(),'surfaces':{'kaggle':0}}
            with patch('src.composite_task_memory_source.FAILURE_CLOSEOUT_SHA',sha(closed)),patch(
                 'src.composite_task_skill_model_admission.validate_preparation',return_value=binding):
                # A pre-execution or unknown parent cannot use this route.
                parent['scientific_execution_started']=False;path.write_text(json.dumps(ledger))
                with self.assertRaises(PermissionError):reserve_stateful_qualification(root,'public',live)
                self.assertFalse((root/'COMPUTE_SUBMISSION.lock').exists())
                parent['scientific_execution_started']=True;path.write_text(json.dumps(ledger))
                first=reserve_stateful_qualification(root,'public',live)
                reconcile_stateful_qualification(root,first['identity'],123,first['guard_token'])
                saved=json.loads(path.read_text());candidate=saved['candidates']['task_skill_stateful_memory_qualification_v3']
                candidate.update(status='completed_public_fixture',owned_compute_released=True)
                saved['active_owned_submission']=None;path.write_text(json.dumps(saved))
                with self.assertRaises(PermissionError):reserve_stateful_qualification(root,'public',live)
                final=json.loads(path.read_text())
                self.assertEqual(final['candidates']['task_skill_full_fit_v1']['actual_partial_counts']['completed_update_journals'],12)
                self.assertEqual(len(final['candidates']['task_skill_stateful_memory_qualification_v3']['qualification_attempts']),1)


if __name__=='__main__':unittest.main()
