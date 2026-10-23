from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from src import composite_frozen_base_worker as worker
from src import composite_frozen_launch as parent
from src.composite_exports import verify_evidence_archive
from src.composite_probe_ownership import create_owned_cache, verify_owned_cache
from src.composite_frozen_worker import atomic_json


class BaseWorkerTests(unittest.TestCase):
    commit = 'a'*40
    token = 'b'*32

    def fixture(self, root):
        source = root/'source'; (source/'config').mkdir(parents=True); (source/'inputs').mkdir()
        (source/'config/composite_controller_rank32_v2.json').write_text('{"qwen":{}}')
        (source/'inputs/MANIFEST.json').write_text('{}')
        base = root/'outputs'; target = base/self.commit/'base_00'
        cache_base = root/'cache'; cache = cache_base/self.commit/'base_00'
        create_owned_cache(cache, self.commit, self.token, cache_base)
        context = {'source_commit': self.commit, 'authorization_sha256': 'c'*64,
            'pin': {'destination': str(target), 'cache_root': str(cache), 'disk_policy': copy.deepcopy(worker.DISK_POLICY)},
            'evaluation_plan': {'partition': 0, 'case_ids': ['synthetic-'+str(i) for i in range(6)]}}
        clock = [0.]; state = {}; services = worker.Services(); services.clock = lambda: clock[0]
        services.authorize = lambda *args: context
        services.ownership = lambda *args: verify_owned_cache(*args, namespace=cache_base)
        services.transport = lambda *args: None; services.output_base = base
        services.runtime = lambda *args: {'model_free_synthetic_fixture': True}
        services.disk = lambda *args: {'model_free_synthetic_fixture': True}
        services.reserve = lambda target, *args: (Path(target)/'EXPORT_RESERVE.bin').write_bytes(b'synthetic reserve')
        services.alarm = lambda *args: nullcontext(); services.resources = lambda: None
        services.release = lambda: state.update(released=True)
        def load(context, cache, guard, on_forward, persist_bytes, persist_json):
            model = SimpleNamespace(actor=SimpleNamespace(actor=SimpleNamespace(frozen_probe_counts={'attempted': 0, 'completed': 0})),
                                    detector_forward_counts={'attempted': 0, 'completed': 0})
            state.update(model=model, callback=on_forward)
            return model, {'backbone_instances': {'ModernBERT': 1, 'Qwen': 1}}, lambda: None
        services.load = load
        def evaluate(model, context, guard, persist, progress, verify):
            for event in ['attempt', 'complete']:
                guard(); state['callback'](event)
                model.actor.actor.frozen_probe_counts['attempted' if event == 'attempt' else 'completed'] += 1
            persist('SYNTHETIC_NATIVE_RESPONSE.json', {'fixture_only': True})
            return {'partition_complete': True, 'full_validation_admitted': False}
        services.evaluate = evaluate
        return source, target, cache, context, services, state, clock

    def invoke(self, fixture):
        source, target, cache, context, services, *_ = fixture
        with redirect_stdout(io.StringIO()):
            return worker.run(source, self.commit, str(target), str(cache), self.token, services)

    def test_owned_complete_export_uses_base_mode_and_original_unknown_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory)); result = self.invoke(fixture); target = fixture[1]
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['forward_counts'], {'attempted': 1, 'completed': 1})
            self.assertFalse(result['full_validation_admitted'])
            self.assertTrue(fixture[5]['released'])
            self.assertFalse((target/'EXPORT_RESERVE.bin').exists())
            verify_evidence_archive((target/worker.ARCHIVE).read_bytes())
            state = {'child_started': True, 'child_exit_confirmed': True, 'exit_code': 0}
            proof = parent.observed_exit_counts(target, self.commit, state, self.token, worker.MODE, 30000)
            self.assertTrue(proof['counts_exact_at_exit'])
            # A base receipt never silently passes the old qualification gate.
            self.assertFalse(parent.observed_exit_counts(target, self.commit, state, self.token)['counts_exact_at_exit'])
            sealed = parent.preserve_after_exit(target, self.commit, state, {}, worker.MODE, worker.ARCHIVE)
            self.assertEqual(sealed['evidence_kind'], 'original_worker_export')

    def test_disk_load_and_mid_forward_failures_preserve_unknown_missing_results(self):
        for failure in ['disk', 'load', 'mid_forward']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(Path(directory)); services = fixture[4]; state = fixture[5]
                def fail(*args): raise RuntimeError('synthetic '+failure)
                if failure == 'disk': services.disk = fail
                elif failure == 'load': services.load = fail
                else:
                    def incomplete(model, context, guard, persist, progress, verify):
                        state['callback']('attempt'); model.actor.actor.frozen_probe_counts['attempted'] += 1
                        persist('PARTIAL_NATIVE_RESPONSE.json', {'fixture_only': True})
                        raise RuntimeError('synthetic interrupted native call')
                    services.evaluate = incomplete
                result = self.invoke(fixture)
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['counts_exact_at_controlled_exit'])
                self.assertEqual(result['forward_counts'], {'attempted': int(failure == 'mid_forward'), 'completed': 0})
                self.assertTrue((fixture[1]/worker.ARCHIVE).exists())
                self.assertFalse(result['full_system_validation_complete'])

    def test_base_budget_does_not_inherit_400_but_retains_joint_30000_ceiling(self):
        budget = worker.ForwardBudget(0, lambda: 0, lambda value: None, bounds=worker.BOUNDS)
        for _ in range(401): budget.event('attempt'); budget.event('complete')
        self.assertEqual(budget.completed, 401)
        budget.attempted = budget.completed = 30000
        with self.assertRaises(TimeoutError): budget.event('attempt')
        self.assertEqual(budget.attempted, 30000)

    def test_writer_rejects_duplicate_escape_and_over_bound_without_truncating_old_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); writer = worker.EvidenceWriter(root)
            writer.bytes('preserved.bin', b'original')
            with self.assertRaises(FileExistsError): writer.bytes('preserved.bin', b'changed')
            with self.assertRaises(ValueError): writer.bytes('../outside.bin', b'escape')
            from unittest.mock import patch
            with patch.dict(worker.DISK_POLICY, {'maximum_raw_evidence_bytes': 100, 'terminal_metadata_headroom_bytes': 20}):
                with self.assertRaises(OSError): writer.bytes('over.bin', b'x'*81)
            self.assertEqual((root/'preserved.bin').read_bytes(), b'original')
            self.assertFalse((root/'over.bin').exists())


if __name__ == '__main__': unittest.main()
