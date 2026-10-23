import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src import composite_frozen_base as base
from src.io_utils import read, digest
from src.composite_actor import Actor

ROOT = Path(__file__).resolve().parents[1]


class FrozenBaseTests(unittest.TestCase):
    def fixture(self):
        cfg = read(ROOT/'config/composite_controller_rank32_v2.json')
        scenes = [{'case_id': 'synthetic-'+str(i), 'group_id': 'group-'+str(i//9),
                   'template': 3 + i%2, 'condition': cfg['cohorts']['conditions'][i%9],
                   'hard_stop': 24} for i in range(36)]
        raw = json.dumps([{'scene': scene} for scene in scenes]).encode()
        return cfg, scenes, raw

    def test_full_roster_fixed_partitions_and_forward_ceiling(self):
        cfg, scenes, raw = self.fixture()
        with patch.object(base, 'EPISODES_SHA256', hashlib.sha256(raw).hexdigest()), \
             patch.object(base, 'ROSTER_SHA256', digest([s['case_id'] for s in scenes])):
            self.assertEqual(base.roster(raw, cfg), scenes)
            plans = [base.partition_plan(scenes, i) for i in range(6)]
            self.assertEqual([key for plan in plans for key in plan['case_ids']], [s['case_id'] for s in scenes])
            self.assertTrue(all(plan['conditions'] == ['base'] for plan in plans))
            self.assertTrue(all(plan['original_controls'] == cfg['evaluation']['conditions'] for plan in plans))
            self.assertEqual(base.conservative_forwards(6), 24316)
            self.assertLess(base.conservative_forwards(6), base.BOUNDS['maximum_backbone_forwards'])
            changed = copy.deepcopy(cfg); changed['evaluation']['conditions'] = ['base']
            with self.assertRaises(ValueError): base.roster(raw, changed)
            with self.assertRaises(ValueError): base.roster(raw+b' ', cfg)
        for bad in [-1, 6, True]:
            with self.assertRaises(ValueError): base.partition_plan(scenes, bad)

    def test_base_report_keeps_missing_and_other_controls_unknown(self):
        records = [{'scene': {'case_id': 'a'}, 'condition': 'base', 'outcome': {'reason': 'hard_stop'}}]
        report = base.base_report(records, ['a', 'b'])
        self.assertFalse(report['partition_complete'])
        self.assertEqual(report['missing_case_ids'], ['b'])
        self.assertIsNone(report['adapter_OFF_bypass'])
        self.assertIsNone(report['adapter_ON_effect'])
        self.assertFalse(report['full_validation_admitted'])
        self.assertEqual(set(report['other_controls_status'].values()), {'unexecuted_unknown'})
        with self.assertRaises(ValueError): base.base_report(records*2, ['a', 'b'])
        changed = copy.deepcopy(records); changed[0]['condition'] = 'always_on'
        with self.assertRaises(ValueError): base.base_report(changed, ['a'])

    def test_unregistered_actor_stage_cannot_reach_any_native_loader(self):
        with self.assertRaises(PermissionError):
            with patch('src.composite_actor.require_databricks'):
                Actor('not-a-model', {}, lambda: None,
                      frozen_probe={'stage': 'bypass-authority', 'root': '.', 'source_commit': 'a'*40})


if __name__ == '__main__': unittest.main()
