"""Public fake-only sampler lifecycle/provider boundaries; no scientific rows."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.io_utils import dump, read, sha
from src.composite_bounded_sampling_profile import (
    MODE, PHASE, NOTEBOOK, OWNER_THREAD, BOUNDS, APPROVAL, PROPOSAL,
    APPROVAL_SHA, PROPOSAL_SHA, FAILED_SOURCE, FAILED_RUN, FAILED_CLOSEOUT_SHA,
    notebook_code)
from src.composite_bounded_sampling_rule import RULE_SHA, approved_rule_descriptor
from src.composite_bounded_sampling_source import approval, SamplingContext, require_context
from src.composite_bounded_sampling_worker import run
from src.composite_bounded_behavior_data_worker import CPU_PACKAGES
from src.composite_bounded_sampling_controller import run as controlled_run, rescue_partial
from src.composite_bounded_sampling_admission import (
    SamplingKaggleClosure, kaggle_closure, require_closure, _provider_snapshot)
from src.composite_exports import make_evidence_archive


COMMIT = 'a' * 40


class FakeServices:
    clock = staticmethod(lambda: 0)
    alarm = staticmethod(lambda seconds: nullcontext())
    runtime = staticmethod(lambda: {'python': 'public-fake-only', 'torch': '2.11.0+cpu',
        'torch_cuda_build': None, 'packages': dict(CPU_PACKAGES)})

    def __init__(self, output, *, fail=False, completed=144):
        self.output, self.fail, self.completed = str(output), fail, completed

    def preview(self, root, commit): return {'destination': self.output}

    def authorize(self, root, commit):
        return SimpleNamespace(profile=lambda: {'destination': self.output}, cfg=lambda: {},
            parent=SimpleNamespace(receipt=lambda: {'public_fake_reference': True}))

    def materialize(self, context, persist):
        persist('PUBLIC_FAKE_ONLY.json', {'public_fixture_test_mode': True})
        if self.fail: raise ValueError('hand-authored fake replay failure')
        return {'completed_cases': self.completed}


class SamplingStageTests(unittest.TestCase):
    def source(self, folder):
        root = Path(folder) / 'source'; (root / 'inputs').mkdir(parents=True)
        dump(root / 'inputs/MANIFEST.json', {'public_fake_source': True})
        dump(root / 'RELEASE.json', {'public_fake_source': True})
        return root

    def binding(self):
        return {'source_commit': COMMIT, 'scriptVersionId': 123, 'mode': MODE,
            'phase': PHASE, 'notebook': NOTEBOOK, 'owner_thread_id': OWNER_THREAD}

    def terminal(self):
        return ('Browser URL: "https://www.kaggle.com/code/' + NOTEBOOK
            + '?scriptVersionId=123"\nsuccessfully ran after 10s')

    def release(self):
        return 'https://www.kaggle.com/code/' + NOTEBOOK + '/edit\n0  Active Events\nDraft Session off'

    def controller(self):
        return {**self.binding(), 'status': 'worker_exited', 'exit_code': 0, 'timed_out': False,
            'child': {'child_started': True, 'child_exit_confirmed': True, 'exit_code': 0},
            'automatic_scientific_retries': 0, 'model_forwards': 0, 'optimizer_updates': 0,
            'public_fixture_test_mode': False, 'raw_archive_sha256': 'b' * 64,
            'durable_raw_export': True, 'original_failure_preserved': True,
            'heldout_content_read': False}

    def close(self, controller=None, terminal=None, release=None, cells=None):
        return kaggle_closure(self.binding(), (controller or self.controller())['raw_archive_sha256'],
            self.terminal() if terminal is None else terminal,
            self.release() if release is None else release,
            self.controller() if controller is None else controller,
            ['public-fake-code'] if cells is None else cells, ['public-fake-code'])

    def test_fake_worker_exports_original_failure_and_no_admission_or_model_activity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.source(folder); out = Path(folder) / 'raw'
            value = run(root, COMMIT, FakeServices(out))
            self.assertEqual(value['status'], 'complete')
            self.assertEqual(value['completed_cases'], 144)
            self.assertTrue(value['public_fixture_test_mode'])
            self.assertTrue(value['original_failure_preserved'])
            self.assertFalse(value['new_teacher_or_case_generation'])
            self.assertFalse(value['scientific_dataset_admitted'])
            self.assertFalse(value['training_launch_admitted'])
            self.assertEqual((value['model_forwards'], value['optimizer_updates'], value['scientific_GPU_count']), (0, 0, 0))
            freeze = read(out / 'PRECALL_FREEZE.json')
            self.assertEqual(freeze['approval_sha256'], APPROVAL_SHA)
            self.assertEqual(freeze['approved_rule_sha256'], RULE_SHA)
            self.assertTrue(Path(value['archive']).is_file())

    def test_failed_or_incomplete_fake_replay_keeps_partial_raw_and_zero_completed_cases(self):
        for options in [{'fail': True}, {'completed': 143}]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as folder:
                root = self.source(folder); out = Path(folder) / 'raw'
                value = run(root, COMMIT, FakeServices(out, **options))
                self.assertEqual(value['status'], 'failed')
                self.assertEqual(value['completed_cases'], 0)
                self.assertTrue(value['counts_exact_at_controlled_exit'])
                self.assertTrue(value['worker_exiting'])
                self.assertTrue((out / 'FAILURE.json').is_file())
                self.assertTrue(Path(value['archive']).is_file())
                self.assertFalse(value['scientific_dataset_admitted'])

    def test_actual_controller_contract_propagates_checked_worker_flag_to_closure(self):
        # Patch the external runner instead of the controller's fixture seam so
        # this exercises the production receipt schema, without starting a child.
        with tempfile.TemporaryDirectory() as folder:
            root = self.source(folder); out = Path(folder) / 'run/raw'; calls = []
            def fake_child(command, cwd, environment, log, timeout, exit_observer):
                calls.append(command); self.assertEqual(timeout, 1200)
                out.mkdir(parents=True)
                dump(out / 'RUN_RECEIPT.json', {'source_commit': COMMIT, 'stage': MODE,
                    'phase': PHASE, 'status': 'complete', 'completed_cases': 144,
                    'metadata_cache_removed': True, 'new_teacher_or_case_generation': False,
                    'original_failure_preserved': True, 'worker_exiting': True,
                    'public_fixture_test_mode': False, 'cpu_runtime': FakeServices.runtime()})
                make_evidence_archive(out, out.with_name(out.name + '_EVIDENCE.zip'), BOUNDS['maximum_raw_export_bytes'])
                exit_observer({'child_started': True, 'child_exit_confirmed': True, 'exit_code': 0})
                return {'timed_out': False, 'exit_code': 0}
            with patch('src.composite_bounded_sampling_worker.Services.preview', return_value={'destination': str(out)}), \
                 patch('src.composite_bounded_sampling_controller.run_logged', side_effect=fake_child):
                controller = controlled_run(root, COMMIT, 'no-process-started', {}, _clock=lambda: 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(controller['status'], 'worker_exited')
            self.assertIs(controller['original_failure_preserved'], True)
            proof = self.close(controller)
            self.assertTrue(require_closure(proof)['owned_compute_released'])
            self.assertEqual(require_closure(proof)['archive_sha256'], sha(out.with_name(out.name + '_EVIDENCE.zip')))
            with self.assertRaises(PermissionError): self.close({**controller, 'original_failure_preserved': False})

    def test_controller_rejects_unverified_failure_preservation_and_never_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.source(folder); out = Path(folder) / 'run/raw'; calls = []
            def fake_child(command, cwd, environment, log, timeout, exit_observer):
                calls.append(command); out.mkdir(parents=True)
                dump(out / 'RUN_RECEIPT.json', {'source_commit': COMMIT, 'stage': MODE,
                    'phase': PHASE, 'status': 'complete', 'completed_cases': 144,
                    'metadata_cache_removed': True, 'new_teacher_or_case_generation': False,
                    'original_failure_preserved': False, 'worker_exiting': True,
                    'public_fixture_test_mode': False, 'cpu_runtime': FakeServices.runtime()})
                make_evidence_archive(out, out.with_name(out.name + '_EVIDENCE.zip'), BOUNDS['maximum_raw_export_bytes'])
                exit_observer({'child_started': True, 'child_exit_confirmed': True, 'exit_code': 0})
                return {'timed_out': False, 'exit_code': 0}
            with patch('src.composite_bounded_sampling_worker.Services.preview', return_value={'destination': str(out)}):
                result = controlled_run(root, COMMIT, 'unused', {}, _runner=fake_child, _clock=lambda: 0)
                self.assertEqual(result['status'], 'failed')
                self.assertNotIn('original_failure_preserved', result)
                with self.assertRaises(PermissionError): controlled_run(root, COMMIT, 'unused', {}, _runner=fake_child)
            self.assertEqual(len(calls), 1)

    def test_exact_terminal_notebook_and_run_cannot_be_assembled_from_separate_substrings(self):
        self.assertTrue(require_closure(self.close())['owned_compute_released'])
        wrong = [self.terminal().replace(NOTEBOOK, 'farhad89/other'),
            self.terminal().replace(NOTEBOOK, NOTEBOOK + '-other'),
            self.terminal().replace('=123', '=1234'),
            self.terminal().replace('=123', '=124') + '\nscriptVersionId=123',
            self.terminal().replace(NOTEBOOK, 'farhad89/other') + '\n' + NOTEBOOK,
            self.terminal().replace('?scriptVersionId=123', '?scriptVersionId=123&scriptVersionId=124'),
            self.terminal().replace('https://www.kaggle.com', 'https://attacker.invalid'),
            NOTEBOOK + ' scriptVersionId=123 successfully ran after 10s']
        for terminal in wrong:
            with self.subTest(terminal=terminal), self.assertRaises(PermissionError): self.close(terminal=terminal)
        for release in [self.release().replace(NOTEBOOK, NOTEBOOK + '-other'), '0  Active Events\nDraft Session off']:
            with self.subTest(release=release), self.assertRaises(PermissionError): self.close(release=release)

    def test_closure_requires_actual_raw_child_source_and_no_fixture_flag(self):
        for key, bad in [('original_failure_preserved', False), ('public_fixture_test_mode', True),
                         ('durable_raw_export', False), ('source_commit', 'c' * 40),
                         ('model_forwards', 1), ('timed_out', True), ('exit_code', True)]:
            with self.subTest(key=key), self.assertRaises(PermissionError): self.close({**self.controller(), key: bad})
        with self.assertRaises(PermissionError): self.close(cells=['changed'])
        with self.assertRaises(PermissionError): require_closure({'owned_compute_released': True})
        with self.assertRaises(PermissionError): SamplingKaggleClosure({'owned_compute_released': True})

    def test_specific_approval_and_original_failure_lineage_are_bound(self):
        cfg = read(Path(__file__).parents[1] / 'config/composite_bounded_behavior_v4.json')
        approved = {'approved_proposal_sha256': PROPOSAL_SHA, 'approved_rule_canonical_sha256': RULE_SHA,
            'original_failed_run_id': FAILED_RUN, 'original_failed_closeout_sha256': FAILED_CLOSEOUT_SHA,
            'source_user_message_id': '01a178f2-bc41-7963-a5ab-d8ac14241ba6',
            'scientific_stage_admission_still_required': True,
            'scientific_allocation_admitted_by_receipt_alone': False}
        proposal = {'proposed_generic_rule': approved_rule_descriptor(),
            'failure_lineage': {'source_commit': FAILED_SOURCE, 'old_run_remains_failed_unadmitted': True}}
        def fake_sha(path): return {APPROVAL: APPROVAL_SHA, PROPOSAL: PROPOSAL_SHA}[Path(path).name]
        def fake_read(path): return approved if Path(path).name == APPROVAL else proposal
        with patch('src.composite_bounded_sampling_source.sha', side_effect=fake_sha), \
             patch('src.composite_bounded_sampling_source.read', side_effect=fake_read):
            self.assertEqual(approval(Path('public-fake-only'), cfg), approved)
            for key, bad in [('source_user_message_id', 'other'), ('original_failed_run_id', 999),
                    ('scientific_allocation_admitted_by_receipt_alone', True)]:
                original = approved[key]; approved[key] = bad
                with self.subTest(key=key), self.assertRaises(PermissionError): approval(Path('public-fake-only'), cfg)
                approved[key] = original
            proposal['failure_lineage']['old_run_remains_failed_unadmitted'] = False
            with self.assertRaises(PermissionError): approval(Path('public-fake-only'), cfg)
        with self.assertRaises(PermissionError): require_context({'approved': True, 'runtime': 'kaggle'})
        with self.assertRaises(PermissionError): SamplingContext('.', {}, {}, {}, {})

    def test_fresh_cpu_observation_and_external_bounds_are_preserved(self):
        self.assertEqual((BOUNDS['hard_timeout_seconds'], BOUNDS['worker_seconds'], BOUNDS['export_reserve_seconds']), (1800, 1200, 120))
        observed = {'account': 'farhad89', 'notebook': NOTEBOOK, 'notebook_draft_sessions': 0,
            'active_notebook_versions': 0, 'pending_notebook_versions': 0,
            'events_checked': True, 'accelerator': 'None'}
        _provider_snapshot({'kaggle_observation': observed})
        for key, bad in [('notebook_draft_sessions', 1), ('active_notebook_versions', True), ('accelerator', 'T4 x2')]:
            with self.subTest(key=key), self.assertRaises(PermissionError):
                _provider_snapshot({'kaggle_observation': {**observed, key: bad}})

    def test_bootstrap_requires_exact_source_and_manifest_binding_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'notebooks').mkdir()
            path = root / 'notebooks/composite_bounded_sampling_amendment.py'
            path.write_text("source='SOURCE_COMMIT_PLACEHOLDER'\nmanifest='PACKAGE_MANIFEST_SHA256_PLACEHOLDER'\n")
            code = notebook_code(root, COMMIT, 'b' * 64)
            self.assertIn(COMMIT, code); self.assertIn('b' * 64, code)
            path.write_text("source='SOURCE_COMMIT_PLACEHOLDER'\n")
            with self.assertRaises(PermissionError): notebook_code(root, COMMIT, 'b' * 64)


if __name__ == '__main__': unittest.main()
