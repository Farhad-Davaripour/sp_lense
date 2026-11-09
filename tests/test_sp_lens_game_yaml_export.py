"""Exact real frozen-YAML reproduction; no held-out/model/simulator reads."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / 'tools/export_sp_lens_game_yaml.py'
SPEC = importlib.util.spec_from_file_location('frozen_yaml_export', PATH)
exporter = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(exporter)


class FrozenYamlExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payloads, cls.receipt = exporter.rebuild()
        cls.manifest = json.loads((exporter.PROJECT / exporter.PUBLIC_MANIFEST).read_text())
        cls.contract = exporter.renderer_contract(exporter.PROJECT, cls.manifest['renderer'])
        cls.documents = [exporter.bound_yaml(exporter.PROJECT, item) for item in cls.manifest['families']]

    def fixture_project(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        names = [exporter.PUBLIC_MANIFEST, self.manifest['catalog']['path'], self.manifest['renderer']['path']]
        names += [item['path'] for item in self.manifest['families']]
        for name in names:
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(exporter.PROJECT / name, path)
        return root

    def test_real_yaml_reproduces_both_exact_frozen_hashes_and_order(self):
        self.assertEqual(self.receipt['counts'], {'train': 768, 'dev': 144})
        self.assertEqual(exporter.sha(self.payloads['train']), '4ee3d2ac659d820c680bb2f01f410add1d6742f70c66b233e4f12d765b162405')
        self.assertEqual(exporter.sha(self.payloads['dev']), 'e9e677c5d77ba9db9bd3fe42f70f8fa858fc90175e50d36b935326807245f1a8')
        index = exporter.case_index(self.documents, self.manifest)
        self.assertEqual(len(index), 912)
        self.assertIn(('train', 2026111800, self.manifest['family_order'][0]), index)
        self.assertIn(('dev', 2026111911, self.manifest['family_order'][-1]), index)
        self.assertEqual(len(self.payloads['train'].splitlines()), 768)
        self.assertFalse(self.receipt['historical_checkout_required'])
        self.assertFalse(self.receipt['model_loaded'])

    def test_matches_existing_private_jsonl_bytes_when_present(self):
        root = exporter.PROJECT / '.local/datasets/sp_lens_game_v1/authorized_management_v1_r4'
        if not root.is_dir(): self.skipTest('Optional existing private JSONL not installed')
        for split in ('train', 'dev'):
            self.assertEqual(self.payloads[split], (root / split / (split + '.jsonl')).read_bytes())

    def test_altered_missing_yaml_and_manifest_rejected(self):
        project = self.fixture_project(); path = project / self.manifest['families'][0]['path']; original = path.read_bytes()
        path.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'Frozen YAML bytes differ'): exporter.rebuild(project)
        path.unlink()
        with self.assertRaises(ValueError): exporter.rebuild(project)
        path.write_bytes(original)
        public = project / exporter.PUBLIC_MANIFEST; public.write_bytes(public.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'Public frozen dataset manifest differs'): exporter.rebuild(project)

    def test_missing_duplicate_and_conflicting_examples_rejected(self):
        for mutation in ('missing', 'duplicate'):
            documents = copy.deepcopy(self.documents)
            if mutation == 'missing': documents[0]['cases'].pop()
            else: documents[0]['cases'][-1] = copy.deepcopy(documents[0]['cases'][0])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): exporter.case_index(documents, self.manifest)
        index = exporter.case_index(copy.deepcopy(self.documents), self.manifest)
        first = index[('train', 2026111800, self.manifest['family_order'][0])]
        second = index[('train', 2026111800, self.manifest['family_order'][1])]
        second['model_input'] = copy.deepcopy(first['model_input'])
        with self.assertRaisesRegex(ValueError, 'Conflicting targets'): exporter.render_jsonl(index, self.manifest, self.contract)

    def test_exclusive_local_export_and_no_committable_jsonl_destination(self):
        project = self.fixture_project(); output = project / '.local/datasets/yaml_rebuild'
        receipt = exporter.export(output, project)
        self.assertEqual(receipt['status'], 'PASS_EXACT_REPRODUCTION')
        self.assertEqual((output / 'train/train.jsonl').read_bytes(), self.payloads['train'])
        with self.assertRaises(FileExistsError): exporter.export(output, project)
        with self.assertRaises(ValueError): exporter.export(project / 'scenarios/generated_raw', project)


if __name__ == '__main__': unittest.main()
