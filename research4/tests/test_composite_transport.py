import hashlib
import io
import json
import os
import sys
import types
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src import composite_runtime as runtime
from src.composite_store import Store


class RuntimeTests(unittest.TestCase):
    def fixture(self, root):
        inputs = root / 'inputs'
        inputs.mkdir()
        source = root / 'source'
        (source / 'inputs').mkdir(parents=True)
        relative = 'data_v1/training/DETECTION_ROWS.json'
        file = inputs / relative
        file.parent.mkdir(parents=True)
        file.write_bytes(b'[]')
        manifest = inputs / 'TRANSPORT_MANIFEST.json'
        manifest.write_text(json.dumps({'files': {
            relative: hashlib.sha256(b'[]').hexdigest()}}))
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        target = root / 'outputs' / 'candidate'
        pin = {'platform': 'kaggle', 'account': 'farhad89',
               'notebook': 'farhad89/research4-recognition',
               'mode': 'detector-rank32-v5', 'source_commit': 'frozen-source',
               'input_root': str(inputs), 'transport_manifest_sha256': digest,
               'destination': str(target.resolve())}
        (source / 'inputs/EXECUTION_PLATFORM.json').write_text(json.dumps(pin))
        environment = {'RESEARCH4_EXECUTION_PLATFORM': 'kaggle',
                       'RESEARCH4_KAGGLE_INPUT_ROOT': str(inputs),
                       'RESEARCH4_KAGGLE_MANIFEST_SHA256': digest,
                       'CUDA_VISIBLE_DEVICES': '0'}
        return inputs, source, target, environment, relative

    def scope(self, root):
        return patch.multiple(runtime, KAGGLE_RUNTIME_DIRECTORIES=[str(root)],
                              KAGGLE_INPUT_BASES=[str(root / 'inputs')],
                              KAGGLE_OUTPUT_BASE=str(root / 'outputs'))

    def test_local_and_unregistered_runtime_fail_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'Databricks'):
                runtime.require_scientific_runtime()
        with patch.dict(os.environ, {'RESEARCH4_EXECUTION_PLATFORM': 'unknown'}, clear=True):
            with self.assertRaises(ValueError):
                runtime.require_scientific_runtime()
        with patch.dict(os.environ, {'RESEARCH4_EXECUTION_PLATFORM': 'kaggle'}, clear=True), \
                patch.object(runtime, 'KAGGLE_RUNTIME_DIRECTORIES', ['/unavailable/runtime']):
            with self.assertRaisesRegex(RuntimeError, 'Kaggle'):
                runtime.require_scientific_runtime()

    def test_registered_input_bytes_and_read_only_transport(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs, source, target, environment, relative = self.fixture(root)
            with patch.dict(os.environ, environment, clear=True), self.scope(root):
                store = Store()
                self.assertEqual(store.get(relative), b'[]')
                for name in ['../research3/file', '/absolute', 'x\\escape',
                             'unregistered.json', 'sealed/HELDOUT_SEAL.json']:
                    with self.assertRaises((ValueError, PermissionError)):
                        store.get(name)
                with self.assertRaises(PermissionError):
                    store.put(relative, b'changed')
                with self.assertRaises(ValueError):
                    Store('another-root')
                (inputs / relative).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'freeze'):
                    store.get(relative)

    def test_manifest_tampering_and_input_root_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs, source, target, environment, relative = self.fixture(root)
            with patch.dict(os.environ, environment, clear=True), self.scope(root):
                (inputs / 'TRANSPORT_MANIFEST.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'manifest'):
                    runtime.KaggleInputs()
                os.environ['RESEARCH4_KAGGLE_INPUT_ROOT'] = str(root / 'outside')
                with self.assertRaisesRegex(ValueError, 'roots'):
                    runtime.KaggleInputs()

    def test_exclusive_output_readback_and_duplicate_prevention(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs, source, target, environment, relative = self.fixture(root)
            with patch.dict(os.environ, environment, clear=True), self.scope(root):
                identity, upload = runtime.begin_stage(
                    source, 'frozen-source', str(target), 'detector-rank32-v5')
                self.assertEqual(identity['scientific_visible_GPU_count'], 1)
                upload('RUN_RECEIPT.json', b'{"status":"complete"}')
                self.assertEqual((target / 'RUN_RECEIPT.json').read_bytes(),
                                 b'{"status":"complete"}')
                with self.assertRaises(FileExistsError):
                    upload('RUN_RECEIPT.json', b'replace')
                with self.assertRaises(FileExistsError):
                    runtime.begin_stage(source, 'frozen-source', str(target),
                                        'detector-rank32-v5')
                with self.assertRaises(ValueError):
                    upload('../outside', b'x')

    def test_wrong_account_source_device_and_behavior_stages_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs, source, target, environment, relative = self.fixture(root)
            with patch.dict(os.environ, environment, clear=True), self.scope(root):
                for mode in ['behavior-all_blocks', 'evaluation', 'heldout']:
                    with self.assertRaises(ValueError):
                        runtime.begin_stage(source, 'frozen-source', str(target), mode)
                with self.assertRaises(ValueError):
                    runtime.begin_stage(source, 'other-source', str(target),
                                        'detector-rank32-v5')
                os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
                with self.assertRaises(ValueError):
                    runtime.begin_stage(source, 'frozen-source', str(target),
                                        'detector-rank32-v5')
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                file = source / 'inputs/EXECUTION_PLATFORM.json'
                pin = json.loads(file.read_text())
                pin['account'] = 'another-account'
                file.write_text(json.dumps(pin))
                with self.assertRaises(ValueError):
                    runtime.begin_stage(source, 'frozen-source', str(target),
                                        'detector-rank32-v5')

    def test_databricks_binding_still_checks_host_namespace_and_readback(self):
        from src.workspace_accounts import profiles, artifact_volume
        host = profiles()['accounts']['free_tier']['host']
        data = {}
        files = types.SimpleNamespace(
            create_directory=lambda path: None,
            upload=lambda path, stream, overwrite: data.setdefault(path, stream.read()),
            download=lambda path: types.SimpleNamespace(contents=io.BytesIO(data[path])))
        sdk = types.ModuleType('databricks.sdk')
        parent = types.ModuleType('databricks')
        parent.__path__ = []
        parent.sdk = sdk
        sdk.WorkspaceClient = lambda: types.SimpleNamespace(
            config=types.SimpleNamespace(host=host), files=files)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'inputs').mkdir()
            environment = {'MLFLOW_RUN_ID': 'synthetic-model-free-test',
                           'CODE_SOURCE_PATH': str(root)}
            with patch.dict(os.environ, environment, clear=True), \
                    patch.dict(sys.modules, {'databricks': parent, 'databricks.sdk': sdk}):
                destination = artifact_volume('free_tier') + '/composite_controller_v1/runs/test'
                identity, upload = runtime.begin_stage(root, 'source', destination, 'detector')
                self.assertEqual(identity['account'], 'free_tier')
                upload('receipt.json', b'{}')
                self.assertEqual(data[destination + '/receipt.json'], b'{}')
                with self.assertRaises(ValueError):
                    runtime.begin_stage(root, 'source', '/Volumes/other-study/run', 'detector')


if __name__ == '__main__':
    unittest.main()
