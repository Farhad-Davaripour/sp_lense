"""Published archives cannot escape their extraction root."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from src.public_artifacts import fetch, main, member, verify
from src.evidence import replay_output, screen_summary


class PublicArtifactTests(unittest.TestCase):
    def test_untrusted_archive_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ['../token','/absolute','C:/credential','a/../../secret',r'a\secret']:
                with self.assertRaises(ValueError):member(root,name)
            self.assertEqual(member(root,'adapters/example/config.json'),Path(root).resolve()/'adapters/example/config.json')

    def release(self, profile='primary'):
        """Small pinned archives with deliberately separate replay and model members."""
        files = ({'datasets/cases.json': b'[]', 'evidence/scores.json': b'{"successes":10}',
                  'adapters/model.bin': b'untouched-model'} if profile == 'primary' else
                 {'diagnostic/scores.json': b'{"KL":0.202261}', 'cpu_fit_inputs/weights.bin': b'fit-arrays'})
        groups = ({'datasets.zip': ['datasets/cases.json'], 'evidence.zip': ['evidence/scores.json'],
                   'model.zip': ['adapters/model.bin']} if profile == 'primary' else
                  {'diagnostic_outputs.zip': ['diagnostic/scores.json'], 'inputs.zip': ['cpu_fit_inputs/weights.bin']})
        pin = lambda value: {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
        downloads = {}
        for name, members in groups.items():
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as archive:
                for name_in_zip in members:
                    archive.writestr(name_in_zip, files[name_in_zip])
            downloads[name] = buffer.getvalue()
        downloads['MANIFEST.json'] = json.dumps({'files': {name: pin(value) for name, value in files.items()}}).encode()
        return {'base_url': 'https://example.invalid',
                'assets': {name: pin(value) for name, value in downloads.items()}}, downloads

    def test_evidence_download_excludes_weights_and_preserves_the_original_manifest(self):
        for profile, expected in [('primary', ['datasets.zip', 'evidence.zip', 'MANIFEST.json']),
                                  ('steering', ['diagnostic_outputs.zip', 'MANIFEST.json'])]:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                spec, downloads = self.release(profile)
                requested = []
                def download(url, **kwargs):
                    name = url.rsplit('/', 1)[1]
                    requested.append(name)
                    self.assertIn(name, expected)
                    return io.BytesIO(downloads[name])
                root = Path(directory) / 'evidence'
                with patch('src.public_artifacts.profile', return_value=spec), patch('urllib.request.urlopen', side_effect=download):
                    result = fetch(root, profile, scope='evidence')
                    self.assertEqual(requested, expected)
                    self.assertEqual(result['scope'], 'evidence')
                    self.assertEqual((root / 'MANIFEST.json').read_bytes(), downloads['MANIFEST.json'])
                    self.assertFalse((root / ('adapters' if profile == 'primary' else 'cpu_fit_inputs')).exists())
                    with self.assertRaises(FileNotFoundError):
                        verify(root, profile, scope='full')
                    target = root / ('evidence/scores.json' if profile == 'primary' else 'diagnostic/scores.json')
                    target.write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError, 'Public artifact changed'):
                        verify(root, profile, scope='evidence')

    def test_existing_full_download_api_still_verifies_every_member(self):
        spec, downloads = self.release()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'full'
            with patch('src.public_artifacts.profile', return_value=spec), patch('urllib.request.urlopen',
                    side_effect=lambda url, **kwargs: io.BytesIO(downloads[url.rsplit('/', 1)[1]])):
                result = fetch(root)
                self.assertEqual(result['files'], 3)
                self.assertEqual((root / 'adapters/model.bin').read_bytes(), b'untouched-model')
                (root / 'adapters/model.bin').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Public artifact changed'):
                    verify(root)

    def test_cli_defaults_to_evidence_and_rejects_missing_output_before_reading_inputs(self):
        with patch('src.public_artifacts.fetch', return_value={}) as download, contextlib.redirect_stdout(io.StringIO()):
            main(['fetch', '--root', 'new-artifacts'])
            download.assert_called_once_with(Path('new-artifacts'), 'primary', 'evidence')
        with patch('src.public_artifacts.replay') as replay, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(['replay', '--root', 'missing'])
            self.assertEqual(error.exception.code, 2)
            replay.assert_not_called()

    def test_replay_output_cannot_replace_evidence_or_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); evidence = root / 'evidence'; code = root / 'checkout'
            evidence.mkdir(); code.mkdir()
            for output in [None, evidence / 'new', code / 'new', root]:
                with self.assertRaises(ValueError):
                    replay_output(output, evidence, code)
            self.assertEqual(replay_output(root / 'new', evidence, code), root / 'new')

    def test_shared_screen_summary_preserves_partial_counts_and_original_stop_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'cases').mkdir()
            for i in range(4):
                row = {'initial_turns': [{'truncated': True}], 'post_turns': [],
                       'current_metrics': {'rejected_calls': 0}, 'completed': False}
                (root / 'cases' / f'{i}.json').write_text(json.dumps(row))
            expected = {'started_cases': 4, 'completed_cases': 0, 'partial_cases': 4, 'unexecuted_cases': 12}
            proof = {'gate': 'format', 'observed_errors': 4, 'maximum_allowed': 0,
                     'proof': 'Errors cannot be undone by later cases.'}
            saved = dict(expected, fail_fast_proof=proof)
            (root / 'SCREEN_RESULT.json').write_text(json.dumps(saved))
            self.assertEqual(screen_summary(root), (expected, proof))
            saved['completed_cases'] = 1
            (root / 'SCREEN_RESULT.json').write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'stopped-screen proof differs'):
                screen_summary(root)


if __name__=='__main__':unittest.main()
