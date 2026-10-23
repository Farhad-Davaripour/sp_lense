import hashlib
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


if __name__ == '__main__':
    unittest.main()
