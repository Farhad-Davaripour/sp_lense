"""Public fake-service lifecycle checks; no scientific worlds or tokenization."""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.io_utils import dump, read
from src.composite_bounded_behavior_data_profile import MODE, PHASE, PARENT_DESCRIPTOR, OWNER_THREAD
from src.composite_bounded_behavior_data_worker import run, verify_cpu_environment, CPU_PACKAGES
from src.composite_bounded_behavior_data_controller import run as controlled_run, rescue_partial


class PublicServices:
    clock = staticmethod(lambda: 0)
    alarm = staticmethod(lambda seconds: nullcontext())
    runtime = staticmethod(lambda: {'python': 'public-fixture', 'torch': '2.11.0+cpu',
                                   'torch_cuda_build': None, 'packages': dict(CPU_PACKAGES)})
    def __init__(self, output, fail=False):
        self.output, self.fail = str(output), fail
    def preview(self, root, commit): return {'destination': self.output}
    @contextmanager
    def authorize(self, root, commit):
        yield SimpleNamespace(profile=lambda: {'destination': self.output}, cfg=lambda: {})
    def tokenizers(self, context, cache, writer): return object()
    def generate(self, context, tokenizer, persist):
        persist('PUBLIC_FIXTURE.json', {'public_fixture_test_mode': True})
        if self.fail: raise ValueError('public fixture failure')
        return {'completed_cases': 144}


class BoundedDataWorkerTests(unittest.TestCase):
    def test_cpu_reader_environment_is_explicit_and_preserves_gpu_evidence_separation(self):
        current = PublicServices.runtime()
        self.assertIs(verify_cpu_environment(current), current)
        for changes in [{'torch': '2.11.0+cu128'}, {'torch': '2.10.0+cpu'},
                        {'torch_cuda_build': '12.8'}, {'packages': {**CPU_PACKAGES, 'numpy': 'other'}},
                        {'python': ''}]:
            with self.subTest(changes=changes), self.assertRaises(PermissionError):
                verify_cpu_environment({**current, **changes})

    def setup_source(self, base):
        root = base / 'source'; (root / 'inputs').mkdir(parents=True)
        dump(root / 'inputs/MANIFEST.json', {})
        dump(root / 'inputs' / PARENT_DESCRIPTOR, {'public': True})
        dump(root / 'RELEASE.json', {'commit': 'a' * 40})
        return root

    def test_public_fixture_completion_never_claims_scientific_data_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); root = self.setup_source(base); out = base / 'raw'
            value = run(root, 'a' * 40, PublicServices(out))
            self.assertEqual(value['status'], 'complete')
            self.assertTrue(value['public_fixture_test_mode'])
            self.assertFalse(value['scientific_dataset_admitted'])
            self.assertFalse(value['training_launch_admitted'])
            self.assertTrue(value['metadata_cache_removed'])
            self.assertEqual(read(out / 'RUN_RECEIPT.json')['cpu_runtime'], PublicServices.runtime())
            self.assertTrue(Path(value['archive']).is_file())
            self.assertEqual(read(out / 'RUN_RECEIPT.json')['model_forwards'], 0)

    def test_partial_failure_exports_failure_and_exact_controlled_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); root = self.setup_source(base); out = base / 'raw'
            value = run(root, 'a' * 40, PublicServices(out, fail=True))
            self.assertEqual(value['status'], 'failed')
            self.assertEqual(value['completed_cases'], 0)
            self.assertTrue(value['counts_exact_at_controlled_exit'])
            self.assertTrue((out / 'FAILURE.json').is_file())
            self.assertTrue(Path(value['archive']).is_file())
            self.assertFalse(value['scientific_dataset_admitted'])

    def test_controller_does_not_retry_failed_or_existing_child_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); out = base / 'run/raw'; calls = []
            def failed(command, cwd, environment, log, timeout, exit_observer):
                calls.append(command)
                exit_observer({'child_started': True, 'child_exit_confirmed': True, 'exit_code': 1})
                return {'timed_out': False, 'exit_code': 1}
            with patch('src.composite_bounded_behavior_data_worker.Services.preview',
                       return_value={'destination': str(out)}):
                value = controlled_run(base, 'a' * 40, 'unused', {}, _runner=failed)
                self.assertEqual(value['status'], 'failed')
                self.assertEqual(len(calls), 1)
                with self.assertRaises(PermissionError):
                    controlled_run(base, 'a' * 40, 'unused', {}, _runner=failed)
                self.assertEqual(len(calls), 1)

    def test_partial_rescue_requires_actual_child_exit_and_matching_output_owner(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'raw'; out.mkdir()
            dump(out / 'OUTPUT_OWNER.json', {'source_commit': 'a' * 40, 'version': MODE,
                 'phase': PHASE, 'owner_thread_id': OWNER_THREAD, 'path': str(out)})
            self.assertFalse(rescue_partial(out, 'a' * 40, {'child_exit_confirmed': False})['attempted'])
            with self.assertRaises(PermissionError):
                rescue_partial(out, 'b' * 40, {'child_exit_confirmed': True})
            value = rescue_partial(out, 'a' * 40, {'child_exit_confirmed': True})
            self.assertTrue(value['attempted'])
            self.assertFalse(value['scientific_dataset_admitted'])
            self.assertTrue((out / 'OUTPUT_OWNER.json').is_file())


if __name__ == '__main__': unittest.main()
