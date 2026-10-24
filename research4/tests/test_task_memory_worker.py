import json
from pathlib import Path
import tempfile
import unittest
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


if __name__=='__main__':unittest.main()
