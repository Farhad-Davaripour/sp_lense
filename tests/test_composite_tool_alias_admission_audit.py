"""Fail-closed checks for the separate frozen first-call diagnostic."""
import unittest

from src.composite_tool_alias_admission import _provider_snapshot
from src.composite_tool_alias_audit import (kaggle_closure, require_complete_alias,
                                            require_worker_exit)
from src.composite_tool_alias_profile import NOTEBOOK, notebook_code


class AliasAdmissionAuditTests(unittest.TestCase):
    def test_complete_raw_run_requires_confirmed_owned_cache_cleanup(self):
        complete = {'status': 'complete', 'worker_exiting': True,
                    'optimizer_updates': 0, 'world_actions': 0,
                    'official_task72_or_risk180_result': False,
                    'provider_compute_release_verified': False,
                    'completed_durable_responses': 48,
                    'response_count_unknown': False,
                    'cache_cleanup': {'owned_download_cache_removed': True}}
        self.assertIs(require_worker_exit(complete), complete)
        failed = dict(complete, cache_cleanup={
            'owned_download_cache_removed': False, 'error_type': 'OSError'})
        with self.assertRaises(PermissionError):
            require_worker_exit(failed)
        missing = dict(complete)
        del missing['cache_cleanup']
        with self.assertRaises(PermissionError):
            require_worker_exit(missing)

    def test_live_provider_observation_requires_same_account_clean_draft_and_quota(self):
        good = {'kaggle_observation': {
            'account': 'farhad89', 'notebook': NOTEBOOK,
            'notebook_draft_sessions': 0, 'active_notebook_versions': 0,
            'pending_notebook_versions': 0, 'gpu_quota_sufficient': True,
            'events_checked': True}}
        _provider_snapshot(good, NOTEBOOK)
        for key, bad in [('account', 'other'), ('notebook_draft_sessions', 1),
                         ('active_notebook_versions', 1), ('pending_notebook_versions', 1),
                         ('gpu_quota_sufficient', False), ('events_checked', False)]:
            changed = {'kaggle_observation': dict(good['kaggle_observation'])}
            changed['kaggle_observation'][key] = bad
            with self.subTest(key=key), self.assertRaises(PermissionError):
                _provider_snapshot(changed, NOTEBOOK)

    def test_summary_and_fake_closure_cannot_admit_result(self):
        with self.assertRaises(PermissionError):
            require_complete_alias({'raw_evidence_audited': True,
                                    'owned_compute_released': True,
                                    'completed_responses': 48})
        binding = {'scriptVersionId': 123, 'source_commit': 'a' * 40,
                   'mode': 'ordinary-record-entry-alias-frozen-probe-v1',
                   'phase': 'ordinary_entry_alias_diagnostic', 'notebook': NOTEBOOK}
        controller = {'source_commit': 'a' * 40,
                      'version': 'ordinary-record-entry-alias-frozen-probe-v1',
                      'status': 'worker_exited', 'exit_code': 0,
                      'child_exit_confirmed': True, 'timed_out': False,
                      'automatic_scientific_retries': 0,
                      'optimizer_updates': 0, 'world_actions': 0}
        terminal = f'{NOTEBOOK} scriptVersionId=123 successfully ran after 10 seconds'
        with self.assertRaises(PermissionError):
            kaggle_closure(binding, 'b' * 64, terminal, 'Draft Session off',
                           controller, ['source'], ['source'])
        with self.assertRaises(PermissionError):
            kaggle_closure(binding, 'b' * 64, terminal,
                           '0  Active Events\nDraft Session off', controller,
                           ['altered source'], ['source'])
        closure = kaggle_closure(binding, 'b' * 64, terminal,
                                 '0  Active Events\nDraft Session off', controller,
                                 ['source'], ['source'])
        self.assertTrue(closure.receipt()['owned_compute_released'])
        self.assertEqual(closure.receipt()['archive_sha256'], 'b' * 64)

    def test_notebook_source_requires_exact_two_placeholders(self):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'notebooks/composite_tool_alias_probe.py'
            target.parent.mkdir()
            target.write_text("commit='SOURCE_COMMIT_PLACEHOLDER'\n"
                              "manifest='PACKAGE_MANIFEST_SHA256_PLACEHOLDER'\n")
            code = notebook_code(folder, 'a' * 40, 'b' * 64)
            self.assertIn('a' * 40, code)
            self.assertIn('b' * 64, code)
            target.write_text("commit='SOURCE_COMMIT_PLACEHOLDER'\n")
            with self.assertRaises(PermissionError):
                notebook_code(folder, 'a' * 40, 'b' * 64)


if __name__ == '__main__':
    unittest.main()
