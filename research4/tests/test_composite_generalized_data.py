import copy
import json
from pathlib import Path
import tempfile
import unittest
from src.composite_generalized_data import cohort_contract,scenes,materialize
from src.composite_generalized_tasks import total,TASKS
from src.composite_generalized_world import fixture
from src.composite_generalized_teacher import teacher_episode

ROOT=Path(__file__).resolve().parents[1]


class GeneralizedDataTests(unittest.TestCase):
    def test_public_contract_counts_and_balanced_role_metadata_without_instances(self):
        cfg=json.loads((ROOT/'config/composite_generalization_amendment_v2.json').read_text())
        for role,groups,cases in [('training',12,108),('development',4,36),('sealed',6,54)]:
            contract=cohort_contract(cfg,role)
            self.assertEqual((contract['groups'],contract['cases']),(groups,cases))
            self.assertFalse(contract['instances_materialized']);self.assertFalse(contract['sealed_content_read'])
            family_counts={family:sum(g['mechanism']==family for g in contract['group_plan']) for family in cfg['cohorts'][role+'_mechanisms']}
            self.assertEqual(set(family_counts.values()),{cfg['cohorts']['groups_per_mechanism'][role]})
        cfg['cohorts']['conditions'][0]='replacement'
        with self.assertRaises(ValueError):cohort_contract(cfg,'training')

    def test_scientific_materialization_fails_locally_before_any_output_or_secret_use(self):
        cfg=json.loads((ROOT/'config/composite_generalization_amendment_v2.json').read_text())
        with self.assertRaises(RuntimeError):scenes(cfg,'training','must_not_materialize')
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'data'
            with self.assertRaises(RuntimeError):materialize(ROOT,'a'*40,'training','must_not_materialize',out)
            self.assertFalse(out.exists())

    def test_training_task_structures_are_different_problems_with_manual_answers(self):
        self.assertEqual(total('source_aggregation','sum',{'a':{'value':4},'b':{'value':3}}),7)
        records={'a':{'expected':5,'observed':3},'b':{'expected':2,'observed':7}}
        self.assertEqual(total('manifest_reconciliation','sum',records),3)
        self.assertEqual(total('manifest_reconciliation','count',records),2)
        records={'a':{'weight':4,'ready':False,'requires':[]},'b':{'weight':7,'ready':True,'requires':['a']}}
        self.assertEqual(total('record_dependency_planning','sum',records),4)
        self.assertEqual(total('record_dependency_planning','count',records),1)

    def test_unseen_task_algorithms_are_static_public_code_contracts_not_sealed_instances(self):
        self.assertEqual(total('document_dependency_diagnosis','sum',{
            'a':{'declared_next':'b','observed_next':'a','weight':3},
            'b':{'declared_next':'a','observed_next':'a','weight':8}}),3)
        self.assertEqual(total('conditional_evidence_assembly','sum',{
            'a':{'value':4,'verified':True,'excluded':False},
            'b':{'value':9,'verified':True,'excluded':True}}),4)
        self.assertEqual(total('inventory_consistency_proof','sum',{
            'a':{'stock':3,'receipts':4,'shipments':2,'reported':8}}),3)
        self.assertEqual(total('synthetic_route_verification','count',{
            'a':{'start':'x','end':'y','weight':7},'b':{'start':'y','end':'missing','weight':2}}),1)
        self.assertEqual(total('constraint_report','max',{
            'a':{'choice':7,'lower':2,'upper':4,'weight':6},
            'b':{'choice':3,'lower':2,'upper':4,'weight':9}}),6)

    def test_public_reference_targets_follow_each_training_task_without_hidden_case_labels(self):
        for structure in TASKS['training']:
            for domain in ['sum','count']:
                episode=teacher_episode(fixture('own_pending','aggregate_predicate',domain=domain,task_structure=structure))
                self.assertTrue(episode['outcome']['task_complete'])
                for state in episode['states']:
                    self.assertNotIn(structure,state['text'])
                    start,end=state['evidence_chars'];self.assertTrue(0<=start<end<=len(state['text']))


if __name__=='__main__':unittest.main()
