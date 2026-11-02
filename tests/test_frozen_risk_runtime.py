"""Public metadata/owned-lifecycle fixtures; no scientific model execution."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from src.io_utils import read
from src.composite_frozen_risk_data_profile import planned_roster
from src.composite_frozen_risk_profile import execution_plan,FrozenRiskContext
from src.composite_frozen_risk_runtime import frozen_scope
from src.composite_frozen_risk_detector import selected_detector
from src.composite_frozen_risk_data_parent import read_risk_parent

ROOT=Path(__file__).resolve().parents[1]


class FrozenRiskRuntimeBoundaryTests(unittest.TestCase):
    def test_preregistered_packing_preserves_all180_exact_case_control_units(self):
        cfg=read(ROOT/'config/composite_generalization_amendment_v2.json');scenes=planned_roster(cfg)
        original=deepcopy(scenes);plan=execution_plan(scenes,cfg)
        units=[u for p in plan['partitions'] for u in p['units']]
        self.assertEqual(len(units),180);self.assertEqual(len({(u['case_id'],u['arm']) for u in units}),180)
        self.assertEqual(len(plan['partitions']),26)
        self.assertEqual([p['expected_trajectories'] for p in plan['partitions']],[5]+[7]*25)
        self.assertEqual(max(p['maximum_joint_backbone_forwards'] for p in plan['partitions']),27381)
        self.assertEqual(scenes,original);self.assertFalse(plan['metadata_admits_launch'])
        self.assertFalse(plan['runtime_throughput_qualified']);self.assertEqual(plan['optimizer_updates'],0)
        for case in scenes:
            self.assertEqual([u['arm'] for u in units if u['case_id']==case['case_id']],cfg['controls'])

    def test_missing_case_or_control_cannot_be_hidden_by_packing(self):
        cfg=read(ROOT/'config/composite_generalization_amendment_v2.json');scenes=planned_roster(cfg)
        with self.assertRaises(PermissionError):execution_plan(scenes[:-1],cfg)
        changed=deepcopy(cfg);changed['controls'].pop()
        with self.assertRaises(PermissionError):execution_plan(scenes,changed)

    def test_no_flags_or_missing_parents_admit_any_risk_model(self):
        with self.assertRaises(PermissionError):FrozenRiskContext('.',None,None,{},'.',key=True)
        for value in [None,True,{'full_fit_completed':True,'risk_development_data_admitted':True}]:
            with self.assertRaises(PermissionError):
                with frozen_scope(value):pass
        with TemporaryDirectory() as folder:
            with self.assertRaises((FileNotFoundError,PermissionError)):selected_detector(folder,{})
            with self.assertRaises((FileNotFoundError,PermissionError)):read_risk_parent(folder,'missing')


if __name__=='__main__':unittest.main()
