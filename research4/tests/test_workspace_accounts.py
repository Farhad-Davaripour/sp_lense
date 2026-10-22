import json,tempfile,unittest
from pathlib import Path
from src.workspace_accounts import selected,command_environment,switch,redact,verify_runtime_workspace,env_values,profiles


class AccountTests(unittest.TestCase):
    def fixture(self,root):
        path=root/'.env';path.write_text('RESEARCH4_DATABRICKS_ACCOUNT=work\nDATABRICKS_WORK_TOKEN=synthetic-work-secret\nDATABRICKS_FREE_TIER_TOKEN=synthetic-free-secret\n')
        return path

    def test_selection_pairs_each_host_with_only_its_own_secret(self):
        with tempfile.TemporaryDirectory() as scratch:
            path=self.fixture(Path(scratch));old={'DATABRICKS_HOST':profiles()['accounts']['free_tier']['host'],'DATABRICKS_TOKEN':'stale-free-secret','DATABRICKS_CONFIG_PROFILE':'unrelated'}
            work=command_environment(environ=old,env_file=path)
            self.assertEqual(work['DATABRICKS_TOKEN'],'synthetic-work-secret');self.assertEqual(work['DATABRICKS_HOST'],profiles()['accounts']['work']['host']);self.assertNotIn('DATABRICKS_CONFIG_PROFILE',work)
            free=command_environment('free tier',environ=old,env_file=path)
            self.assertEqual(free['DATABRICKS_TOKEN'],'synthetic-free-secret');self.assertEqual(free['DATABRICKS_HOST'],old['DATABRICKS_HOST'])
            self.assertEqual(selected(environ={},env_file=path)['account'],'work')

    def test_round_trip_switch_retains_both_secrets(self):
        with tempfile.TemporaryDirectory() as scratch:
            path=self.fixture(Path(scratch));before=env_values(path)
            self.assertEqual(switch('free tier',path)['account'],'free_tier');self.assertEqual(switch('work',path)['account'],'work')
            self.assertEqual(env_values(path),before)

    def test_wrong_host_missing_secret_and_duplicate_keys_fail_closed(self):
        with tempfile.TemporaryDirectory() as scratch:
            path=self.fixture(Path(scratch))
            with self.assertRaises(ValueError):command_environment('work',environ={'DATABRICKS_WORK_HOST':'https://unapproved.invalid'},env_file=path)
            path.write_text('RESEARCH4_DATABRICKS_ACCOUNT=work\n')
            with self.assertRaises(ValueError):command_environment('work',environ={'DATABRICKS_HOST':profiles()['accounts']['free_tier']['host'],'DATABRICKS_TOKEN':'wrong-account'},env_file=path)
            path.write_text('DATABRICKS_WORK_TOKEN=one\nDATABRICKS_WORK_TOKEN=two\n')
            with self.assertRaises(ValueError):env_values(path)

    def test_remote_descriptor_cannot_cross_workspaces(self):
        with tempfile.TemporaryDirectory() as scratch:
            root=Path(scratch);p=profiles()['accounts'];descriptor={'account':'work','host':p['work']['host'],'organization':p['work']['organization']}
            (root/'inputs').mkdir();(root/'inputs/WORKSPACE.json').write_text(json.dumps(descriptor))
            self.assertEqual(verify_runtime_workspace(root,p['work']['host'])['account'],'work')
            with self.assertRaises(ValueError):verify_runtime_workspace(root,p['free_tier']['host'])
            (root/'inputs/WORKSPACE.json').unlink()
            with self.assertRaises(ValueError):verify_runtime_workspace(root,p['work']['host'])
            self.assertEqual(verify_runtime_workspace(root,p['free_tier']['host'])['account'],'free_tier')

    def test_redaction_covers_non_dapi_credentials_too(self):
        self.assertEqual(redact('failed synthetic-secret',{'DATABRICKS_TOKEN':'synthetic-secret'}),'failed [REDACTED]')


if __name__=='__main__':unittest.main()
