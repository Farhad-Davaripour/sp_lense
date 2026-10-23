import hashlib
import ast
import base64
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

path = Path(__file__).resolve().parents[1] / 'tools/prepare_kaggle_recognition.py'
spec = importlib.util.spec_from_file_location('kaggle_preparation', path)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PreparationTests(unittest.TestCase):
    def extract_without_starting_worker(self, root, receipt):
        """Exercise the real notebook extraction with redirected tiny fixtures."""
        notebook = path.parents[1] / 'notebooks/composite_kaggle.py'
        tree = ast.parse(notebook.read_text())
        boundary = next(index for index, node in enumerate(tree.body)
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'source'
                                for target in node.targets))
        tree.body = tree.body[:boundary]
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith('/kaggle/'):
                    node.value = str(root / node.value.lstrip('/'))
        python = root / 'kaggle/temp/research4_environment_v1/bin/python'
        python.parent.mkdir(parents=True)
        python.touch()
        namespace = {'RESEARCH4_SOURCE_COMMIT': 'a' * 40,
                     'RESEARCH4_PACKAGE_SHA256': receipt['package_sha256'],
                     'RESEARCH4_PACKAGE_FILES_SHA256': receipt['package_files_manifest_sha256']}
        exec(compile(ast.fix_missing_locations(tree), str(notebook), 'exec'), namespace)
        return root / 'kaggle/working/research4_inputs' / ('a' * 40)

    def source_archive(self):
        value = io.BytesIO()
        with tarfile.open(fileobj=value, mode='w') as archive:
            for name, data in [('src/model.py', b'# frozen text\r\n'),
                               ('config/protocol.json', b'{}\r\n')]:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return value.getvalue()

    def fixture(self, root):
        frozen = root / 'frozen'
        transport = root / 'transport'
        frozen.mkdir()
        transport.mkdir()
        (frozen / 'PIN.json').write_bytes(b'{}\n')
        (frozen / 'MANIFEST.json').write_text(json.dumps({
            'PIN.json': builder.digest(frozen / 'PIN.json')}))
        relative = 'data_v1/training/DETECTION_ROWS.json'
        file = transport / relative
        file.parent.mkdir(parents=True)
        file.write_bytes(b'[]')
        (transport / 'TRANSPORT_MANIFEST.json').write_text(json.dumps({
            'files': {relative: builder.digest(file)}}))
        return frozen, transport, relative

    def test_package_preserves_inputs_and_canonical_source_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen, transport, relative = self.fixture(root)
            original_archive = self.source_archive()
            (transport / 'evidence.zip').write_bytes(original_archive)
            manifest_file = transport / 'TRANSPORT_MANIFEST.json'
            transport_manifest = json.loads(manifest_file.read_text())
            transport_manifest['files']['evidence.zip'] = builder.digest(transport / 'evidence.zip')
            manifest_file.write_text(json.dumps(transport_manifest))
            with patch.object(builder.subprocess, 'check_output', return_value=self.source_archive()):
                receipt = builder.prepare(root, 'a' * 40, frozen, transport,
                                          root / 'prepared', 'farhad89/recognition')
            self.assertFalse(receipt['scientific_execution_started'])
            self.assertFalse(receipt['heldout_transferred'])
            with zipfile.ZipFile(root / 'prepared/FROZEN_PACKAGE.zip') as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(archive.read('source/src/model.py'), b'# frozen text\r\n')
                self.assertEqual(archive.read('artifacts/' + relative), b'[]')
                release = json.loads(archive.read('source/RELEASE.json'))
                self.assertEqual(release['files']['src/model.py'],
                                 hashlib.sha256(b'# frozen text\r\n').hexdigest())
                self.assertEqual(json.loads(archive.read(
                    'source/inputs/EXECUTION_PLATFORM.json'))['hard_timeout_seconds'], 2700)
                raw_manifest = archive.read('PACKAGE_FILES.json')
                self.assertEqual(hashlib.sha256(raw_manifest).hexdigest(),
                                 receipt['package_files_manifest_sha256'])
                manifest = json.loads(raw_manifest)
                self.assertEqual(set(manifest['files']), set(archive.namelist()) - {'PACKAGE_FILES.json'})
                for name, expected in manifest['files'].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)
                descriptor = manifest['encoded_files']['artifacts/evidence.zip']
                self.assertNotIn('artifacts/evidence.zip', archive.namelist())
                self.assertEqual(base64.b64decode(archive.read(descriptor['stored_path']),
                                                  validate=True), original_archive)
                self.assertEqual(descriptor['decoded_sha256'],
                                 hashlib.sha256(original_archive).hexdigest())

    def test_changed_or_sealed_artifact_cannot_be_packaged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen, transport, relative = self.fixture(root)
            (transport / relative).write_bytes(b'changed')
            with patch.object(builder.subprocess, 'check_output', return_value=self.source_archive()):
                with self.assertRaisesRegex(ValueError, 'bytes'):
                    builder.prepare(root, 'a' * 40, frozen, transport,
                                    root / 'changed', 'farhad89/recognition')
                (transport / 'TRANSPORT_MANIFEST.json').write_text(json.dumps({
                    'files': {'sealed/HELDOUT_SEAL.json': 'unused'}}))
                with self.assertRaisesRegex(ValueError, 'sealed'):
                    builder.prepare(root, 'a' * 40, frozen, transport,
                                    root / 'sealed', 'farhad89/recognition')

    def test_real_notebook_extraction_accepts_intact_and_unpacked_bundle(self):
        for unpacked in [False, True]:
            with self.subTest(unpacked=unpacked), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                frozen, transport, relative = self.fixture(root)
                with patch.object(builder.subprocess, 'check_output', return_value=self.source_archive()):
                    receipt = builder.prepare(root, 'a' * 40, frozen, transport,
                                              root / 'prepared', 'farhad89/recognition')
                mount = root / 'kaggle/input/private-data'
                mount.mkdir(parents=True)
                package = root / 'prepared/FROZEN_PACKAGE.zip'
                with zipfile.ZipFile(package) as archive:
                    if unpacked:
                        archive.extractall(mount)
                    else:
                        (mount / 'FROZEN_PACKAGE.zip').write_bytes(package.read_bytes())
                    extracted = self.extract_without_starting_worker(root, receipt)
                    for name in archive.namelist():
                        self.assertEqual((extracted / name).read_bytes(), archive.read(name))

    def test_real_notebook_extraction_rejects_changed_unpacked_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen, transport, relative = self.fixture(root)
            with patch.object(builder.subprocess, 'check_output', return_value=self.source_archive()):
                receipt = builder.prepare(root, 'a' * 40, frozen, transport,
                                          root / 'prepared', 'farhad89/recognition')
            mount = root / 'kaggle/input/private-data'
            mount.mkdir(parents=True)
            with zipfile.ZipFile(root / 'prepared/FROZEN_PACKAGE.zip') as archive:
                archive.extractall(mount)
            (mount / 'artifacts' / relative).write_bytes(b'changed')
            with self.assertRaises(AssertionError):
                self.extract_without_starting_worker(root, receipt)


if __name__ == '__main__':
    unittest.main()
