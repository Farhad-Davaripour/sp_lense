from pathlib import Path
import unittest
from src.io_utils import read
from tools.prepare_v2_model import plan_template
from src.workspace_accounts import profiles,artifact_volume

ROOT=Path(__file__).resolve().parents[1]


class ModelPreparationTests(unittest.TestCase):
    def test_templates_keep_unknown_actual_data_and_all_original_limits(self):
        cfg=read(ROOT/'config/composite_controller_rank32_v2.json');design=read(ROOT/'config/composite_generalization_amendment_v2.json')
        workspace={'account':'azure',**profiles()['accounts']['azure'],'artifact_volume':artifact_volume(account='azure')}
        for phase,count in [('qualification',1),('full_fit',64)]:
            plan=plan_template(cfg,design,'a'*40,phase,workspace)
            self.assertEqual(plan['optimizer_updates'],count);self.assertEqual(plan['GPU_count'],1)
            self.assertEqual(plan['accelerator_type'],'GPU_1xA10');self.assertEqual(plan['hard_timeout_seconds'],2700)
            self.assertIsNone(plan['data_release_sha256']);self.assertIsNone(plan['data_export']['sha256'])
            self.assertFalse(plan['allocation_authorized_by_this_template']);self.assertFalse(plan['qualification_warm_start_used'])
            self.assertEqual(plan['maximum_raw_evidence_bytes'],4*1024**3)
            self.assertEqual(plan['physical_export_reserve_bytes'],9*1024**3)


if __name__=='__main__':unittest.main()
