import io,json,sys,tempfile,types,unittest
from unittest.mock import patch
from pathlib import Path
from src.workspace_accounts import selected,command_environment,switch,redact,verify_runtime_workspace,env_values,profiles,artifact_volume


class AccountTests(unittest.TestCase):
    def fixture(self,root):
        path=root/'.env';path.write_text('RESEARCH4_DATABRICKS_ACCOUNT=work\nDATABRICKS_WORK_TOKEN=synthetic-work-secret\nDATABRICKS_FREE_TIER_TOKEN=synthetic-free-secret\nDATABRICKS_AZURE_TOKEN=synthetic-azure-secret\n')
        return path

    def test_selection_pairs_each_host_with_only_its_own_secret(self):
        with tempfile.TemporaryDirectory() as scratch:
            path=self.fixture(Path(scratch));old={'DATABRICKS_HOST':profiles()['accounts']['free_tier']['host'],'DATABRICKS_TOKEN':'stale-free-secret','DATABRICKS_CONFIG_PROFILE':'unrelated'}
            work=command_environment(environ=old,env_file=path)
            self.assertEqual(work['DATABRICKS_TOKEN'],'synthetic-work-secret');self.assertEqual(work['DATABRICKS_HOST'],profiles()['accounts']['work']['host']);self.assertNotIn('DATABRICKS_CONFIG_PROFILE',work)
            free=command_environment('free tier',environ=old,env_file=path)
            self.assertEqual(free['DATABRICKS_TOKEN'],'synthetic-free-secret');self.assertEqual(free['DATABRICKS_HOST'],old['DATABRICKS_HOST'])
            self.assertEqual(selected(environ={},env_file=path)['account'],'work')
            azure=command_environment('azure',environ=old,env_file=path)
            self.assertEqual(azure['DATABRICKS_TOKEN'],'synthetic-azure-secret');self.assertEqual(azure['DATABRICKS_HOST'],profiles()['accounts']['azure']['host'])

    def test_round_trip_switch_retains_both_secrets(self):
        with tempfile.TemporaryDirectory() as scratch:
            path=self.fixture(Path(scratch));before=env_values(path)
            self.assertEqual(switch('free tier',path)['account'],'free_tier');self.assertEqual(switch('azure',path)['account'],'azure');self.assertEqual(switch('work',path)['account'],'work')
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
            with self.assertRaises(ValueError):verify_runtime_workspace(root,p['azure']['host'])
            descriptor={'account':'azure','host':p['azure']['host'],'organization':p['azure']['organization'],'artifact_volume':artifact_volume('azure')};(root/'inputs/WORKSPACE.json').write_text(json.dumps(descriptor))
            self.assertEqual(verify_runtime_workspace(root,p['azure']['host'])['account'],'azure')
            with self.assertRaises(ValueError):verify_runtime_workspace(root,p['work']['host'])

    def test_artifact_volume_is_bound_to_host_without_credentials(self):
        p=profiles()['accounts']
        self.assertEqual(artifact_volume(host=p['free_tier']['host']),'/Volumes/workspace/research4/artifacts')
        self.assertEqual(artifact_volume(host=p['azure']['host']),'/Volumes/db_sp_lense/research4/artifacts')
        with self.assertRaises(ValueError):artifact_volume('free tier',host=p['azure']['host'])
        with self.assertRaises(ValueError):artifact_volume(host='https://unapproved.invalid')

    def test_azure_descriptor_cannot_omit_or_swap_the_volume(self):
        with tempfile.TemporaryDirectory() as scratch:
            root=Path(scratch);(root/'inputs').mkdir();p=profiles()['accounts']['azure'];value={'account':'azure','host':p['host'],'organization':p['organization']}
            (root/'inputs/WORKSPACE.json').write_text(json.dumps(value))
            with self.assertRaises(ValueError):verify_runtime_workspace(root,p['host'])
            value['artifact_volume']=artifact_volume('free tier');(root/'inputs/WORKSPACE.json').write_text(json.dumps(value))
            with self.assertRaises(ValueError):verify_runtime_workspace(root,p['host'])

    def test_namespace_rejects_other_studies_and_path_traversal(self):
        from src.io_utils import assert_namespace
        for root in [artifact_volume('azure'),artifact_volume('free tier')]:
            assert_namespace(root+'/composite_controller_v1/receipt.json')
            for value in [root+'/../research3/file',root+'_other/file',root+'/x\\..\\research3']:
                with self.assertRaises(ValueError):assert_namespace(value)

    def test_store_uses_actual_client_host_and_rejects_cross_workspace_prefix(self):
        from src.composite_store import Store
        data={}
        files=types.SimpleNamespace(create_directory=lambda path:None,upload=lambda path,stream,overwrite:data.setdefault(path,stream.read()),download=lambda path:types.SimpleNamespace(contents=io.BytesIO(data[path])))
        sdk=types.ModuleType('databricks.sdk');parent=types.ModuleType('databricks');parent.__path__=[];parent.sdk=sdk
        for account in ['azure','free_tier']:
            host=profiles()['accounts'][account]['host'];sdk.WorkspaceClient=lambda:types.SimpleNamespace(config=types.SimpleNamespace(host=host),files=files)
            with patch.dict(sys.modules,{'databricks':parent,'databricks.sdk':sdk}):
                store=Store();self.assertEqual(store.prefix,artifact_volume(account)+'/composite_controller_v1')
                store.put('synthetic/roundtrip.json',b'{}');self.assertEqual(store.get('synthetic/roundtrip.json'),b'{}')
                other='free_tier' if account=='azure' else 'azure'
                with self.assertRaises(ValueError):Store(artifact_volume(other)+'/composite_controller_v1')
                with self.assertRaises(ValueError):store.path('../research3/private')

    def test_redaction_covers_non_dapi_credentials_too(self):
        self.assertEqual(redact('failed synthetic-secret',{'DATABRICKS_TOKEN':'synthetic-secret'}),'failed [REDACTED]')


if __name__=='__main__':unittest.main()
