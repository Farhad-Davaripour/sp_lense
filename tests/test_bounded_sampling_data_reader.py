"""Public/fake sampler staging boundaries. No production48+80 call is made."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from src.io_utils import digest
from src import composite_bounded_sampling_data as data
from src import composite_bounded_sampling_reader as reader
from src.composite_bounded_behavior_data_reader import require_data


class FakeParent:
    def __init__(self):
        # Explicit fake state IDs only; no scientific scene constructor is used.
        self.fit = [{'state_id': 'FAKE_ONLY::' + str(i).zfill(4)} for i in reversed(range(640))]
    def fitting_rows(self): return deepcopy(self.fit)
    def cfg(self): return {'FAKE_ONLY': True}
    def receipt(self): return {'FAKE_ONLY': True}
    def scenes(self, role): return []
    def protocol(self): return {'FAKE_ONLY': True, 'scientific_dataset_admitted': False}
    def original_draw_plan(self): return {'FAKE_ONLY': True, 'coverage_passed': False}
    def rows(self, role): return []
    def replay(self): return {'audits': {}, 'budgets': {}, 'leakage': {}}
    def reference_exports(self):
        return {'training/FITTING_ROWS.json': self.fitting_rows(),
            'training/FITTING_DRAW_PLAN.json': self.original_draw_plan(),
            'training/ROWS.json': [], 'development/ROWS.json': []}


class ApprovedDataBoundaryTests(unittest.TestCase):
    def test_matching_caller_code_cannot_replace_exact_reviewed_sampling_bootstrap(self):
        from src import composite_bounded_sampling_source as source
        from src.composite_bounded_sampling_admission import kaggle_closure
        from src.composite_bounded_sampling_profile import NOTEBOOK, OWNER_THREAD, notebook_code
        from src.io_utils import sha
        binding = {'scriptVersionId': 123, 'source_commit': 'a'*40, 'mode': reader.MODE,
            'phase': reader.PHASE, 'notebook': NOTEBOOK, 'owner_thread_id': OWNER_THREAD}
        controller = {**binding, 'status': 'worker_exited', 'exit_code': 0, 'timed_out': False,
            'child': {'child_started': True, 'child_exit_confirmed': True, 'exit_code': 0},
            'automatic_scientific_retries': 0, 'model_forwards': 0, 'optimizer_updates': 0,
            'public_fixture_test_mode': False, 'original_failure_preserved': True,
            'raw_archive_sha256': 'b'*64, 'durable_raw_export': True, 'heldout_content_read': False}
        substitute = 'print("FAKE_ONLY caller substitute")'
        closure = kaggle_closure(binding, 'b'*64, 'https://www.kaggle.com/code/' + NOTEBOOK
            + '?scriptVersionId=123 successfully ran after 1s',
            'https://www.kaggle.com/code/' + NOTEBOOK + ' 0  Active Events Draft Session off',
            controller, [substitute], [substitute])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'source'; (root/'notebooks').mkdir(parents=True)
            (root/'notebooks/composite_bounded_sampling_amendment.py').write_text(
                "commit='SOURCE_COMMIT_PLACEHOLDER'\nmanifest='PACKAGE_MANIFEST_SHA256_PLACEHOLDER'\n")
            (root.parent/reader.MANIFEST).write_text('{}')
            self.assertNotEqual(substitute, notebook_code(root,'a'*40,sha(root.parent/reader.MANIFEST)))
            context = SimpleNamespace(cfg=lambda:{'FAKE_ONLY':True},profile=lambda:{'notebook':NOTEBOOK})
            with patch.object(source, 'validate_source', return_value=context), \
                    patch.object(reader, 'EvidenceFiles', side_effect=AssertionError('Raw reader must not open')):
                with self.assertRaises(PermissionError):
                    reader.audit_data('FAKE_ONLY nonexistent raw','b'*64,root,'a'*40,None,closure)

    def test_raw_and_sampler_are_unreadable_until_genuine_source_bound_release(self):
        from src import composite_bounded_sampling_source as source
        from src import composite_bounded_sampling_rule as rule
        from src.composite_bounded_sampling_admission import kaggle_closure
        from src.composite_bounded_sampling_profile import NOTEBOOK, OWNER_THREAD
        binding = {'scriptVersionId': 123, 'source_commit': 'a'*40, 'mode': reader.MODE,
            'phase': reader.PHASE, 'notebook': NOTEBOOK, 'owner_thread_id': OWNER_THREAD}
        controller = {**binding, 'status': 'worker_exited', 'exit_code': 0, 'timed_out': False,
            'child': {'child_started': True, 'child_exit_confirmed': True, 'exit_code': 0},
            'automatic_scientific_retries': 0, 'model_forwards': 0, 'optimizer_updates': 0,
            'public_fixture_test_mode': False, 'original_failure_preserved': True,
            'raw_archive_sha256': 'b'*64, 'durable_raw_export': True, 'heldout_content_read': False}
        closure = kaggle_closure(binding, 'b'*64,
            'https://www.kaggle.com/code/' + NOTEBOOK + '?scriptVersionId=123 successfully ran after 1s',
            'https://www.kaggle.com/code/' + NOTEBOOK + ' 0  Active Events Draft Session off',
            controller, ['FAKE_ONLY'], ['FAKE_ONLY'])
        context = SimpleNamespace(cfg=lambda:{'FAKE_ONLY':True}, profile=lambda:{'notebook':NOTEBOOK})
        for supplied, raw_sha, commit in [({'owned_compute_released':True},'b'*64,'a'*40),
                                          (closure,'c'*64,'a'*40), (closure,'b'*64,'c'*40)]:
            with self.subTest(raw_sha=raw_sha, commit=commit), \
                    patch.object(source,'validate_source',return_value=context), \
                    patch.object(reader,'EvidenceFiles') as raw, patch.object(rule,'verify_draw_plan') as draw:
                with self.assertRaises(PermissionError):
                    reader.audit_data('FAKE_ONLY missing raw',raw_sha,'FAKE_ONLY source',commit,None,supplied)
                raw.assert_not_called(); draw.assert_not_called()

    def test_fresh_receipt_requires_full_counts_runtime_no_generation_and_controlled_exit(self):
        cpu = {'python': '3.13.0', 'torch': '2.11.0+cpu', 'torch_cuda_build': None,
            'packages': {'transformers': '5.19.0', 'numpy': '2.2.2', 'safetensors': '0.8.0'}}
        receipt = {'status': 'complete', 'source_commit': 'a'*40, 'stage': reader.MODE,
            'mode': reader.MODE, 'phase': reader.PHASE, 'completed_roles': ['training', 'development'],
            'completed_cases': 144, 'model_forwards': 0, 'optimizer_updates': 0,
            'scientific_GPU_count': 0, 'GPU_required': False, 'model_weights_downloaded': False,
            'heldout_content_read': False, 'public_fixture_test_mode': False,
            'counts_exact_at_controlled_exit': True, 'worker_exiting': True,
            'metadata_cache_created': False, 'metadata_cache_removed': True,
            'new_teacher_or_case_generation': False, 'scientific_dataset_admitted': False,
            'cpu_runtime': cpu, 'original_failure_preserved': True, 'training_launch_admitted': False,
            'approval_sha256': reader.APPROVAL_SHA, 'proposal_sha256': reader.PROPOSAL_SHA,
            'owner_thread_id': reader.OWNER_THREAD, 'automatic_scientific_retries': 0}
        self.assertEqual(reader.read_worker_receipt(receipt, 'a'*40), cpu)
        for field, value in [('status', 'failed'), ('completed_cases', 143), ('completed_roles', ['training']),
            ('source_commit', 'b'*40), ('model_forwards', 1), ('optimizer_updates', 1),
            ('public_fixture_test_mode', True), ('heldout_content_read', True),
            ('metadata_cache_created', True), ('metadata_cache_removed', False),
            ('new_teacher_or_case_generation', True), ('counts_exact_at_controlled_exit', False),
            ('worker_exiting', False), ('scientific_dataset_admitted', True),
            ('original_failure_preserved', False), ('training_launch_admitted', True),
            ('approval_sha256', '0'*64), ('proposal_sha256', '0'*64),
            ('owner_thread_id', 'FAKE_OTHER_OWNER'), ('automatic_scientific_retries', 1),
            ('cpu_runtime', {**cpu, 'torch_cuda_build': '12.8'}),
            ('cpu_runtime', {**cpu, 'packages': {**cpu['packages'], 'numpy': '2.2.1'}})]:
            with self.subTest(field=field, value=value), self.assertRaises(PermissionError):
                reader.read_worker_receipt({**receipt, field: value}, 'a'*40)

    def test_json_legacy_or_failed_receipts_cannot_create_amended_data_authority(self):
        with self.assertRaises(PermissionError): reader.require_approved_sampling_data({'approved_sampling_data_admitted': True})
        with self.assertRaises(PermissionError): reader.ApprovedSamplingDataAudit({}, {}, None, [], {}, {})
        with self.assertRaises(PermissionError): reader.ApprovedSamplingArchive(None, None, None, 'a'*64)
        with self.assertRaises(PermissionError): reader.require_replay_archive({'complete_raw': True})
        with self.assertRaises(PermissionError): require_data({'approved_sampling_data_admitted': True})

    def test_canonical_rows_preserve_every_row_and_reject_duplicate_state_ids(self):
        parent = FakeParent()
        with patch.object(data, 'require_failed_parent'):
            actual = data.canonical_rows(parent)
            self.assertEqual(actual, sorted(parent.fit, key=lambda row: row['state_id']))
            self.assertEqual(len(actual), 640)
            parent.fit[1] = parent.fit[0]
            with self.assertRaises(PermissionError): data.canonical_rows(parent)

    def test_materializer_requires_actual_runtime_before_replay_or_sampler(self):
        from src import composite_bounded_sampling_source as source
        from src import composite_bounded_sampling_rule as rule
        context = SimpleNamespace(require_runtime=lambda: (_ for _ in ()).throw(PermissionError('FAKE local runtime')))
        with patch.object(source, 'require_context'), patch.object(rule, 'build_draw_plan') as draw:
            with self.assertRaises(PermissionError): data.materialize(context)
            draw.assert_not_called()

    def test_materializer_preserves_original_plan_and_calls_sampler_once_after_replay(self):
        from src import composite_bounded_sampling_source as source
        from src import composite_bounded_sampling_rule as rule
        parent = FakeParent(); order = []
        original_replay = parent.replay
        parent.replay = lambda: (order.append('replay'), original_replay())[1]
        context = SimpleNamespace(parent=parent, require_runtime=lambda: order.append('runtime'),
            cfg=parent.cfg, profile=lambda: {'source_commit': 'FAKE_ONLY'})
        fake_plan = {'FAKE_ONLY': True, 'coverage_passed': True, 'public_fixture_test_mode': False,
            'canonical_fitting_rows_sha256': digest(sorted(parent.fit, key=lambda r:r['state_id'])),
            'original_fitting_rows_sha256': digest(parent.fit)}
        def draw(*args): order.append('draw'); return fake_plan
        saved = {}
        def persist(name, value):
            self.assertNotIn(name, saved)
            saved[name] = deepcopy(value)
        with patch.object(source, 'require_context'), patch.object(data, 'require_failed_parent'), \
                patch.object(rule, 'build_draw_plan', side_effect=draw) as generated, \
                patch.object(data, 'build_release', return_value={'FAKE_ONLY': True}), \
                patch.object(data, 'reference_export_names', side_effect=lambda p:set(saved)):
            result = data.materialize(context, persist=persist)
        self.assertEqual(order, ['runtime', 'replay', 'draw'])
        generated.assert_called_once()
        self.assertEqual(result['completed_cases'], 144)
        self.assertEqual(saved['training/ORIGINAL_FITTING_ROWS.json'], parent.fit)
        self.assertEqual(saved['training/ORIGINAL_FITTING_DRAW_PLAN.json']['coverage_passed'], False)
        self.assertEqual(saved['training/FITTING_ROWS.json'], sorted(parent.fit, key=lambda r:r['state_id']))
        self.assertNotIn('RUN_RECEIPT.json', saved)

    def test_fresh_replay_maps_old_indices_to_dedicated_historical_exports(self):
        class Files:
            def json(self, name): return name
        view = reader._OriginalReferenceView(Files())
        self.assertEqual(view.json('training/FITTING_ROWS.json'), 'training/ORIGINAL_FITTING_ROWS.json')
        self.assertEqual(view.json('training/FITTING_DRAW_PLAN.json'), 'training/ORIGINAL_FITTING_DRAW_PLAN.json')
        self.assertEqual(view.json('training/EPISODES.json'), 'training/EPISODES.json')


if __name__ == '__main__': unittest.main()
