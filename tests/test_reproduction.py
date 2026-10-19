"""Offline regressions for clean clones, isolation and durable failure evidence."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import io
import yaml
from src.config import ROOT, Settings
from src.reproduce import member, validate_evidence
from src.utils import run_recipe
from src.research2_classifier import classifier_source, qualification_receipt
from src.persist_data import fetch_sources
from types import SimpleNamespace


class ReproductionTests(unittest.TestCase):
    def test_runtime_or_profile_auth_does_not_receive_an_empty_host(self):
        from src.utils import workspace_client
        with (patch('src.utils.auth_environment', return_value={}),
              patch('databricks.sdk.WorkspaceClient') as client):
            workspace_client(SimpleNamespace(host=''))
            client.assert_called_once_with()

    def test_steering_capsule_contains_its_original_policy_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory) / 'run'
            run_recipe('fixed_steering_analysis', ['--help'], True, 10, stage)
            source = stage / 'source/qwen38_colab/frozen_base_control_v1'
            check = subprocess.run([sys.executable, '-c',
                'from actuators import shared_definitions; d,s,p=shared_definitions(); '
                'assert set(s.QUESTIONS)==set(d["shared_classifier"]["fields"]); '
                'assert len(p.DIRECTIVES)==9'], cwd=source, capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)

    def test_input_download_does_not_overwrite_conflicting_local_data(self):
        with tempfile.TemporaryDirectory() as directory:
            data = b'[{"id":"one"}]'
            spec = {'id':'T1-M', 'source':'training/meaning.json', 'version':'v1',
                    'sha256':hashlib.sha256(data).hexdigest(), 'expected_records':1, 'usage_roles':[]}
            settings = SimpleNamespace(data_root=Path(directory), volume_root='/Volumes/test/bronze/artifacts')
            client = SimpleNamespace(files=SimpleNamespace(download=lambda path: SimpleNamespace(contents=io.BytesIO(data))))
            prepared = fetch_sources(settings, {'datasets':[spec]}, client)
            self.assertEqual(prepared[0]['raw'], data)
            (Path(directory) / spec['source']).write_bytes(b'local-work')
            with self.assertRaisesRegex(ValueError, 'not be overwritten'):
                fetch_sources(settings, {'datasets':[spec]}, client)

    def test_databricks_default_outputs_are_durable(self):
        values = {'DATABRICKS_CATALOG': 'research', 'DATABRICKS_SCHEMA': 'bronze', 'DATABRICKS_VOLUME': 'artifacts'}
        with patch('src.config.read_env', return_value=values), patch('pathlib.Path.is_dir', return_value=True):
            settings = Settings.load()
        self.assertEqual(settings.output_root, Path('/Volumes/research/bronze/artifacts/runs'))

    def test_fresh_binary_fit_finalization_needs_no_previous_qualification(self):
        with patch('src.research2_classifier.sha256') as digest:
            self.assertIsNone(qualification_receipt(None, finetune=True))
            self.assertIsNone(qualification_receipt('unrelated-four-way-directory', finetune=True))
            digest.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'RESULT.json'
            path.write_bytes(b'qualified')
            self.assertEqual(qualification_receipt(directory), hashlib.sha256(b'qualified').hexdigest())

    def test_binary_training_does_not_require_a_four_way_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'base' / 'revision'
            base.mkdir(parents=True)
            (base / 'config.json').write_bytes(b'pinned')
            config = {'model': {'base_volume_directory': 'base', 'revision': 'revision',
                'files': {'config.json': {'bytes': 6, 'sha256': hashlib.sha256(b'pinned').hexdigest()}}}}
            with patch('src.classifier_helpers.configuration', return_value=config):
                record, adapter = classifier_source(SimpleNamespace(volume_root=directory), finetune=True)
            self.assertEqual(record['status'], 'PUBLIC_NLI_BASE_ONLY')
            self.assertIsNone(adapter)

    def test_frozen_binary_adapter_can_be_verified_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base/revision'
            adapter = root / 'saved/binary_adapter'
            base.mkdir(parents=True)
            adapter.mkdir(parents=True)
            (base / 'config.json').write_bytes(b'pinned')
            (adapter / 'adapter.safetensors').write_bytes(b'weights')
            pin = lambda data: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            config = {'model': {'base_volume_directory': 'base', 'revision': 'revision',
                               'files': {'config.json': pin(b'pinned')}}}
            (root / 'saved/RESULT.json').write_text(json.dumps({'status': 'RESEARCH2_CLASSIFIER_QUALIFIED'}))
            (root / 'saved/FROZEN_CLASSIFIER.json').write_text(json.dumps(
                {'adapter_files': {'adapter.safetensors': pin(b'weights')}}))
            with patch('src.classifier_helpers.configuration', return_value=config):
                record, measured_adapter = classifier_source(SimpleNamespace(volume_root=directory), root / 'saved')
            self.assertEqual(measured_adapter, adapter)
            self.assertEqual(record['status'], 'RESEARCH2_CLASSIFIER_QUALIFIED')

    def test_testing_target_cannot_share_production_state_or_code(self):
        bundle = yaml.safe_load((ROOT / 'databricks.yml').read_text())
        self.assertIn('${bundle.target}', bundle['workspace']['root_path'])
        test = bundle['targets']['reproduction']
        self.assertEqual(test['workspace']['file_path'], '/Workspace/Shared/sp_lense-reproduction')
        self.assertTrue(bundle['targets']['shared']['default'])
        jobs = yaml.safe_load((ROOT / 'resources/reproduce.job.yml').read_text())['resources']['jobs']
        self.assertNotIn('compute', jobs['reproduce']['tasks'][0])

    def test_worker_timeout_preserves_exit_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('worker', .01)):
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_recipe('fixed_steering_analysis', ['--help'], True, 1, output)
            receipt = json.loads((output / 'PROCESS_EXIT.json').read_bytes())
            self.assertEqual(receipt['status'], 'TimeoutExpired')
            self.assertTrue(receipt['scientific_success_not_inferred'])
            self.assertFalse(receipt['automatic_retry'])

    def test_changed_private_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'input.json').write_bytes(b'original')
            manifest = {'files': {'input.json': {'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest()}}}
            raw = json.dumps(manifest).encode()
            (root / 'MANIFEST.json').write_bytes(raw)
            spec = {'evidence': {'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'files': 1}}
            with patch('src.reproduce.profile', return_value=spec):
                validate_evidence(root)
                (root / 'input.json').write_bytes(b'changed!')
                with self.assertRaisesRegex(ValueError, 'Evidence file changed'):
                    validate_evidence(root)

    def test_evidence_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('../outside.json', 'nested/../../outside.json', 'nested\\file.json'):
                with self.assertRaises(ValueError):
                    member(directory, name)

    def test_supplement_archive_keeps_its_nested_manifest_separate(self):
        from src.public_artifacts import fetch
        archive=io.BytesIO()
        with zipfile.ZipFile(archive,'w') as z:z.writestr('MANIFEST.json',b'inner')
        raw=archive.getvalue()
        manifest=json.dumps({'files':{'cpu_fit_inputs/MANIFEST.json':
            {'bytes':5,'sha256':hashlib.sha256(b'inner').hexdigest()}}}).encode()
        spec={'base_url':'https://example.invalid','assets':{
            'inputs.zip':{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'extract_to':'cpu_fit_inputs'},
            'MANIFEST.json':{'bytes':len(manifest),'sha256':hashlib.sha256(manifest).hexdigest()}}}
        def download(url,**kwargs):return io.BytesIO(raw if url.endswith('.zip') else manifest)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'new'
            with patch('src.public_artifacts.profile',return_value=spec),patch('urllib.request.urlopen',side_effect=download):
                result=fetch(root,'steering')
            self.assertEqual(result['files'],1)
            self.assertEqual((root/'cpu_fit_inputs/MANIFEST.json').read_bytes(),b'inner')
            self.assertEqual((root/'MANIFEST.json').read_bytes(),manifest)

    def test_published_paper_result_identities_remain_valid(self):
        manifest = json.loads((ROOT / 'paper/source_manifest.json').read_bytes())
        for key in ('manuscript', 'results', 'submission_copy', 'submission_pdf'):
            item = manifest[key]
            path = ROOT / 'paper' / item['file']
            self.assertEqual(path.stat().st_size, item['bytes'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item['sha256'])
        for group in ('recipe_files', 'reproduction_entry_points'):
            for name, item in manifest[group].items():
                if not isinstance(item, dict):
                    continue
                raw = (ROOT / name).read_bytes().replace(b'\r\n', b'\n')
                self.assertEqual(len(raw), item['bytes'], name)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item['sha256'], name)
        with zipfile.ZipFile(ROOT / 'paper/manuscript.docx') as doc:
            self.assertNotIn(b'trackRevisions', doc.read('word/settings.xml'))

    def test_clean_submission_accepts_header_and_body_revisions(self):
        import xml.etree.ElementTree as ET
        prefix='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        for filename in ('manuscript.docx', 'manuscript_submission.docx'):
            with zipfile.ZipFile(ROOT / 'paper' / filename) as doc:
                self.assertNotIn(b'trackRevisions', doc.read('word/settings.xml'))
                for name in doc.namelist():
                    if name.endswith('.xml'):
                        root=ET.fromstring(doc.read(name))
                        self.assertFalse(any(e.tag in {prefix+t for t in ['ins','del','moveTo','moveFrom']}
                                             for e in root.iter()), filename+':'+name)


if __name__ == '__main__':
    unittest.main()
