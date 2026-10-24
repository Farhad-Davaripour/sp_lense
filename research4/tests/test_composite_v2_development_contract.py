from copy import deepcopy
from pathlib import Path
import unittest
from src.io_utils import read,digest
from src.composite_v2_development_contract import partition_plan,validate_roster,COLLECTION_BOUNDS,TRANSFER_BOUNDS

ROOT=Path(__file__).resolve().parents[1]


def metadata_roster():
    design=read(ROOT/'config/composite_generalization_amendment_v2.json');rows=[]
    for group in range(4):
        for condition in design['cohorts']['conditions']:
            rows.append({'case_id':digest({'public_metadata_fixture':group,'condition':condition}),
                'group_id':digest({'public_metadata_group':group}),'condition':condition,'role':'development',
                'mechanism':design['cohorts']['development_mechanisms'][group//2],
                'task_structure':design['cohorts']['development_task_structures'][group//2],
                'domain':'sum','template':3+group%2})
    return rows,design


class DevelopmentContractTests(unittest.TestCase):
    def test_static_whole_case_partition_bound_covers_all_180_without_execution(self):
        rows,design=metadata_roster();plan=partition_plan(rows,design)
        self.assertEqual(len(plan['partitions']),36)
        self.assertEqual(plan['maximum_joint_backbone_forwards_per_partition'],19651)
        self.assertLess(plan['maximum_joint_backbone_forwards_per_partition'],30000)
        coverage=[(case,arm) for p in plan['partitions'] for case in p['case_ids'] for arm in p['controls']]
        self.assertEqual(len(set(coverage)),180)
        self.assertTrue(all(len(p['case_ids'])==1 and len(p['controls'])==5 for p in plan['partitions']))
        self.assertFalse(plan['runtime_throughput_qualified']);self.assertFalse(plan['full_campaign_cost_admitted'])
        self.assertEqual(TRANSFER_BOUNDS['maximum_backbone_forwards'],900)
        self.assertEqual(COLLECTION_BOUNDS['hard_timeout_seconds'],2700)

    def test_missing_case_arm_group_or_changed_limits_cannot_be_prepared(self):
        rows,design=metadata_roster()
        for bad in [rows[:-1],rows+[rows[0]]]:
            with self.assertRaises(PermissionError):partition_plan(bad,design)
        changed=deepcopy(design);changed['controls'].pop()
        with self.assertRaises(PermissionError):partition_plan(rows,changed)
        changed=deepcopy(design);changed['architecture']['maximum_new_tokens']=161
        with self.assertRaises(PermissionError):partition_plan(rows,changed)
        changed=deepcopy(rows);changed[0]['group_id']=changed[9]['group_id']
        with self.assertRaises(PermissionError):validate_roster(changed,design)


if __name__=='__main__':unittest.main()
