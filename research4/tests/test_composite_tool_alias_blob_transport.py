"""Opaque actor upload preserves the original source ZIP byte for byte."""
import ast
import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.composite_tool_alias_admission import validate_actor_transport
from tools.prepare_tool_alias_probe import prepare_actor_transport


def bootstrap_helpers():
    source = Path(__file__).resolve().parents[1] / 'notebooks/composite_tool_alias_probe.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    methods = {'file_sha', 'actor_blob_path', 'safe_actor_member_name'}
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in methods]
    if {node.name for node in nodes} != methods:
        raise AssertionError('Exact bootstrap transport helpers required')
    namespace = {'Path': Path, 'hashlib': hashlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


class AliasBlobTransportTests(unittest.TestCase):
    def test_blob_matches_original_archive_and_admission_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / 'original.zip'
            with zipfile.ZipFile(original, 'w') as stream:
                stream.writestr('TASK_TRAINING_PACKAGE_FILES.json', '{}')
                stream.writestr('source/config/survival_heldout_run.json', '{}')
            expected = hashlib.sha256(original.read_bytes()).hexdigest()
            archive, blob = prepare_actor_transport(original, root, expected)
            self.assertEqual(blob.suffix, '.r4blob')
            self.assertEqual(archive.read_bytes(), blob.read_bytes())
            receipt = {'actor_package_sha256': expected, 'actor_blob_sha256': expected}
            self.assertEqual(validate_actor_transport(root, receipt), archive)
            blob.write_bytes(blob.read_bytes() + b'changed')
            with self.assertRaises(PermissionError):
                validate_actor_transport(root, receipt)

    def test_bootstrap_finds_only_one_matching_blob_and_accepts_historical_code(self):
        helpers = bootstrap_helpers()
        safe = helpers['safe_actor_member_name']
        self.assertTrue(safe('source/config/survival_heldout_run.json'))
        self.assertTrue(safe('source/src/survival_heldout_run.py'))
        self.assertTrue(safe('source/inputs/TASK_EVALUATION_STAGE.json'))
        self.assertTrue(safe('TASK_TRAINING_PACKAGE_FILES.json'))
        for name in ('source/inputs/HELDOUT_SEAL.json', 'source/inputs/sealed_cases.json',
                     '../actor.zip', '/absolute', 'source/../secret', 'source\\code.py'):
            with self.subTest(name=name):
                self.assertFalse(safe(name))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / 'dataset/ACTOR_PACKAGE.r4blob'
            archive.parent.mkdir()
            archive.write_bytes(b'original bytes')
            expected = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual(helpers['actor_blob_path'](root, expected), archive)
            with self.assertRaises(AssertionError):
                helpers['actor_blob_path'](root, '0' * 64)
            duplicate = root / 'other/ACTOR_PACKAGE.r4blob'
            duplicate.parent.mkdir()
            duplicate.write_bytes(archive.read_bytes())
            with self.assertRaises(AssertionError):
                helpers['actor_blob_path'](root, expected)


if __name__ == '__main__':
    unittest.main()
