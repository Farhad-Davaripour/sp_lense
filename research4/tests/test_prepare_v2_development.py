from pathlib import Path
import unittest
from src.io_utils import read
from src.workspace_accounts import profiles,artifact_volume
from tools.prepare_v2_development import stage_template

ROOT=Path(__file__).resolve().parents[1]


class DevelopmentPreparationTests(unittest.TestCase):
    def test_templates_have_no_actual_data_or_model_and_cannot_allocate(self):
        cfg=read(ROOT/'config/composite_controller_rank32_v2.json');design=read(ROOT/'config/composite_generalization_amendment_v2.json')
        workspace={'account':'azure',**profiles()['accounts']['azure'],'artifact_volume':artifact_volume(account='azure')}
        stages=[stage_template('a'*40,'detector_transfer',workspace,cfg,design)]
        stages.extend(stage_template('a'*40,'development_partition',workspace,cfg,design,i) for i in range(36))
        self.assertEqual(len({s['destination'] for s in stages}),37)
        for stage in stages:
            self.assertFalse(stage['allocation_authorized_by_this_template'])
            self.assertIsNone(stage['development_roster']);self.assertIsNone(stage['data_export']['sha256'])
            self.assertIsNone(stage['package']);self.assertEqual(stage['threshold'],0.9)
            self.assertEqual(stage['detector_checkpoint_sha256'],design['detector']['checkpoint_sha256'])
            self.assertEqual(stage['optimizer_updates'],0)
        with self.assertRaises(ValueError):stage_template('a'*40,'development_partition',workspace,cfg,design,36)


if __name__=='__main__':unittest.main()
