import copy,json,tempfile,unittest
from pathlib import Path
from src.composite_v2_data_contract import dataset_admission,roster,validate_workspace
from src.workspace_accounts import profiles,artifact_volume


class DataContractTests(unittest.TestCase):
    def reports(self):
        audits={role:{'role':role,'reference_world_replayed':True,'complete_states':3} for role in ['training','development']}
        budgets={role:{'status':'complete_native_budget_audit_pass','states':3,'truncated_or_repaired':False} for role in audits}
        return audits,budgets,{'status':'structural_and_exact_leakage_checks_passed'}

    def test_no_missing_role_replay_state_budget_or_semantic_flag_can_admit(self):
        self.assertTrue(dataset_admission(*self.reports())['training_and_development_admitted'])
        for mutation in ['role','replay','count','budget','repair','leakage_review']:
            audits,budgets,leak=self.reports()
            if mutation=='role':audits.pop('development')
            elif mutation=='replay':audits['training']['reference_world_replayed']=False
            elif mutation=='count':budgets['training']['states']=2
            elif mutation=='budget':budgets['training']['status']='failed_native_budget_admission'
            elif mutation=='repair':budgets['training']['truncated_or_repaired']=True
            else:leak['status']='requires_semantic_review_of_near_duplicates'
            with self.subTest(mutation=mutation),self.assertRaises((PermissionError,ValueError)):dataset_admission(audits,budgets,leak)

    def test_workspace_is_bound_to_account_host_organization_and_volume(self):
        profile=profiles()['accounts']['azure'];value={'account':'azure',**profile,'artifact_volume':artifact_volume(account='azure')}
        validate_workspace(value)
        for key in ['host','organization','artifact_volume']:
            with self.assertRaises(PermissionError):validate_workspace({**value,key:'wrong'})

    def test_source_preparation_is_explicitly_nonallocating(self):
        from src.composite_v2_data_contract import BOUNDS
        self.assertEqual(BOUNDS['automatic_scientific_retries'],0)
        self.assertEqual(BOUNDS['maximum_concurrent_owned_workers'],1)
        self.assertLess(BOUNDS['worker_seconds'],BOUNDS['hard_timeout_seconds'])


if __name__=='__main__':unittest.main()
