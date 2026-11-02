"""Tiny public metadata fixtures, never scientific world materialization."""
from copy import deepcopy
import unittest
from src.io_utils import digest
from src.composite_task_skill_validation import task_development_plan,descriptive_task_tables,FAMILIES,TASKS
from src.composite_task_skill_world import VERSION


def roster():
    rows=[]
    for group in range(4):
        for replica in range(9):
            rows.append({'case_id':digest(['public-case',group,replica]),'group_id':digest(['public-group',group]),
              'identity_seed':digest(['public-seed',group,replica]),'mechanism':FAMILIES[group//2],
              'task_structure':TASKS[(group//2+group%2)%2],'domain':['sum','count'][replica%2],
              'template':[3,4][group%2],'replicate':replica,'id_digits':[5,9][group%2],
              'role':'development','world_version':VERSION})
    return {'cases':rows,'reference_actions_in_roster':False,'arms':['base','always_on'],'paired_trajectories':72}


class TaskValidationTests(unittest.TestCase):
    def test_plan_keeps_all_pairs_and_never_substitutes_for_risk_or_launch(self):
        plan=task_development_plan(roster())
        self.assertEqual(sum(p['expected_trajectories'] for p in plan['partitions']),72)
        self.assertEqual(len(plan['partitions']),36)
        self.assertEqual(plan['partitions'][0]['maximum_with_optional_native_parity'],8006)
        self.assertFalse(plan['metadata_admits_launch']);self.assertFalse(plan['risk_endpoint_covered_by_task72'])
        self.assertFalse(plan['runtime_throughput_qualified'])

    def test_missing_or_unknown_is_not_zero_or_failed(self):
        r=roster();one={'case_id':r['cases'][0]['case_id'],'arm':'base','task_accepted':None,'raw_denied_writes':None}
        report=descriptive_task_tables(r,[one])
        self.assertEqual(report['unknown_success_outcomes'],72)
        self.assertIsNone(report['descriptive_criterion_met_each_family'])
        self.assertFalse(report['scientific_validation_admitted'])

    def test_complete_descriptive_pairing_uses_whole_groups_and_no_audit_claim(self):
        r=roster();obs=[{'case_id':c['case_id'],'arm':a,'task_accepted':a=='always_on','raw_denied_writes':0}
                       for c in r['cases'] for a in r['arms']]
        report=descriptive_task_tables(r,obs)
        self.assertEqual(len(report['case_table']),36);self.assertEqual(len(report['group_table']),4)
        self.assertEqual(report['unknown_success_outcomes'],0)
        self.assertTrue(report['descriptive_criterion_met_each_family'])
        self.assertFalse(report['raw_evidence_audited']);self.assertFalse(report['scientific_validation_admitted'])
        obs[1]['raw_denied_writes']=1
        self.assertFalse(descriptive_task_tables(r,obs)['descriptive_criterion_met_each_family'])

    def test_duplicate_missing_role_or_crossed_group_cannot_reduce_coverage(self):
        for change in ['missing','duplicate','role','replica','family','arms']:
            r=roster()
            if change=='missing':r['cases'].pop()
            elif change=='duplicate':r['cases'][1]=deepcopy(r['cases'][0])
            elif change=='role':r['cases'][0]['role']='training'
            elif change=='replica':r['cases'][0]['replicate']=True
            elif change=='family':r['cases'][0]['mechanism']=FAMILIES[1]
            else:r['arms']=['base']
            with self.assertRaises((PermissionError,ValueError)):task_development_plan(r)
        r=roster();row={'case_id':r['cases'][0]['case_id'],'arm':'base','task_accepted':False,'raw_denied_writes':0}
        with self.assertRaises(ValueError):descriptive_task_tables(r,[row,row])
        row['raw_denied_writes']=True
        with self.assertRaises(ValueError):descriptive_task_tables(r,[row])


if __name__=='__main__':unittest.main()
