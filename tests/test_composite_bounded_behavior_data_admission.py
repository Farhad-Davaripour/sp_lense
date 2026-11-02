"""Public transport/provider boundary fixtures; no model or cohort generation."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.io_utils import dump, sha
from src.composite_bounded_behavior_data_profile import (
    MODE, PHASE, NOTEBOOK, OWNER_THREAD, MANIFEST, notebook_code)
from src.composite_bounded_behavior_data_admission import (
    DataKaggleClosure, kaggle_closure, require_data_closure, _provider_snapshot, validate_package)


class BoundedDataAdmissionTests(unittest.TestCase):
    def closure_inputs(self):
        binding = {'source_commit': 'a' * 40, 'scriptVersionId': 123, 'mode': MODE,
                   'phase': PHASE, 'notebook': NOTEBOOK, 'owner_thread_id': OWNER_THREAD}
        controller = {**binding, 'status': 'worker_exited', 'exit_code': 0, 'timed_out': False,
            'child': {'child_started': True, 'child_exit_confirmed': True, 'exit_code': 0},
            'automatic_scientific_retries': 0, 'model_forwards': 0, 'optimizer_updates': 0,
            'public_fixture_test_mode': False, 'raw_archive_sha256': 'b' * 64,
            'durable_raw_export': True,
            'heldout_content_read': False}
        return binding, controller, NOTEBOOK + ' scriptVersionId=123 successfully ran after 10s'

    def test_release_requires_exact_run_actual_child_and_saved_source(self):
        binding, controller, terminal = self.closure_inputs()
        release = 'https://www.kaggle.com/code/' + NOTEBOOK + '\n0  Active Events\nDraft Session off'
        value = kaggle_closure(binding, 'b' * 64, terminal, release, controller, ['code'], ['code'])
        self.assertTrue(require_data_closure(value)['owned_compute_released'])
        for key, bad in [('source_commit', 'c' * 40), ('owner_thread_id', 'other'),
                         ('exit_code', 1), ('timed_out', True), ('model_forwards', 1)]:
            with self.subTest(key=key), self.assertRaises(PermissionError):
                kaggle_closure(binding, 'b' * 64, terminal, release,
                               {**controller, key: bad}, ['code'], ['code'])
        for key, bad in [('public_fixture_test_mode', True), ('raw_archive_sha256', 'c' * 64),
                         ('durable_raw_export', False)]:
            with self.subTest(key=key), self.assertRaises(PermissionError):
                kaggle_closure(binding, 'b' * 64, terminal, release,
                               {**controller, key: bad}, ['code'], ['code'])
        with self.assertRaises(PermissionError):
            kaggle_closure(binding, 'b' * 64, terminal, release,
                           {**controller, 'child': {'child_started': True, 'child_exit_confirmed': False,
                                                   'exit_code': None}}, ['code'], ['code'])
        with self.assertRaises(PermissionError):
            kaggle_closure(binding, 'b' * 64, terminal.replace('123', '124'), release,
                           controller, ['code'], ['code'])
        with self.assertRaises(PermissionError):
            kaggle_closure(binding, 'b' * 64, terminal, 'Draft Session off', controller, ['code'], ['code'])
        for wrong_release in ['0  Active Events\nDraft Session off',
                release.replace(NOTEBOOK, 'farhad89/research4-other'),
                release.replace(NOTEBOOK, NOTEBOOK + '-other')]:
            with self.subTest(release=wrong_release), self.assertRaises(PermissionError):
                kaggle_closure(binding, 'b' * 64, terminal, wrong_release, controller, ['code'], ['code'])
        with self.assertRaises(PermissionError):
            kaggle_closure(binding, 'b' * 64, terminal, release, controller, ['changed'], ['code'])

    def test_serialized_success_cannot_become_genuine_closure(self):
        with self.assertRaises(PermissionError): require_data_closure({'owned_compute_released': True})
        with self.assertRaises(PermissionError): DataKaggleClosure({'owned_compute_released': True})

    def test_live_cpu_observation_rejects_active_or_wrong_surface(self):
        observation = {'account': 'farhad89', 'notebook': NOTEBOOK, 'notebook_draft_sessions': 0,
            'active_notebook_versions': 0, 'pending_notebook_versions': 0,
            'events_checked': True, 'accelerator': 'None'}
        _provider_snapshot({'kaggle_observation': observation})
        for key, bad in [('account', 'other'), ('notebook_draft_sessions', 1),
                         ('active_notebook_versions', True), ('events_checked', False),
                         ('accelerator', 'T4 x2')]:
            with self.subTest(key=key), self.assertRaises(PermissionError):
                _provider_snapshot({'kaggle_observation': {**observation, key: bad}})

    def test_uploaded_zip_must_match_the_complete_prepared_source(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); (base / 'bundle/source').mkdir(parents=True)
            source = base / 'bundle/source/public.txt'; source.write_text('public fixture')
            dump(base / 'bundle' / MANIFEST, {'files': {'source/public.txt': sha(source)}})
            archive = base / 'package.zip'
            def write(extra=False, changed=False):
                with zipfile.ZipFile(archive, 'w') as stream:
                    stream.write(base / 'bundle' / MANIFEST, MANIFEST)
                    stream.writestr('source/public.txt', 'changed' if changed else 'public fixture')
                    if extra: stream.writestr('source/extra.txt', 'unregistered')
            write(); validate_package(archive, base / 'bundle')
            write(extra=True)
            with self.assertRaises(PermissionError): validate_package(archive, base / 'bundle')
            write(changed=True)
            with self.assertRaises(PermissionError): validate_package(archive, base / 'bundle')

    def test_notebook_requires_one_exact_binding_of_both_source_and_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'notebooks').mkdir()
            path = root / 'notebooks/composite_bounded_behavior_data.py'
            path.write_text("source='SOURCE_COMMIT_PLACEHOLDER'\nmanifest='PACKAGE_MANIFEST_SHA256_PLACEHOLDER'\n")
            code = notebook_code(root, 'a' * 40, 'b' * 64)
            self.assertIn('a' * 40, code); self.assertIn('b' * 64, code)
            path.write_text("source='SOURCE_COMMIT_PLACEHOLDER'\n")
            with self.assertRaises(PermissionError): notebook_code(root, 'a' * 40, 'b' * 64)


if __name__ == '__main__': unittest.main()
