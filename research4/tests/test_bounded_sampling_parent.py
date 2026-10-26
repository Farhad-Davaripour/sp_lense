"""Public failure-lineage boundaries; no actual references, models or draws."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.io_utils import dump, sha
from src import composite_bounded_sampling_parent as parent
from src.composite_bounded_behavior_data_reader import require_data


class FailedReferenceBoundaryTests(unittest.TestCase):
    def test_failed_json_and_constructor_do_not_mint_old_or_new_data_authority(self):
        with self.assertRaises(PermissionError): parent.require_failed_parent(parent.descriptor())
        with self.assertRaises(PermissionError):
            parent.FailedReferenceAudit({}, {}, {}, {}, (), b'', {}, {})
        with self.assertRaises(PermissionError): require_data(parent.descriptor())

    def test_content_stamp_detects_same_size_same_mtime_raw_ancestor_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in parent.FILES: (root / name).write_bytes(b'FAKE_ONLY')
            paths = [root / n for n in ['raw', 'fit', 'qualification']]
            for p in paths: p.write_bytes(b'FAKE_1')
            original, _ = parent.dependency_stamp(root, *paths)
            for p in paths:
                with self.subTest(ancestor=p.name):
                    stat = p.stat()
                    p.write_bytes(b'FAKE_2')
                    os.utime(p, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                    changed, _ = parent.dependency_stamp(root, *paths)
                    self.assertNotEqual(original, changed)
                    p.write_bytes(b'FAKE_1')
            (root / next(iter(parent.FILES))).unlink()
            with self.assertRaises(FileNotFoundError): parent.dependency_stamp(root, *paths)

    def _fixture(self, folder):
        base = Path(folder); inputs = base / 'inputs'; source = base / 'source'
        inputs.mkdir(); source.mkdir()
        code = 'print("FAKE_ONLY")\n'
        notebook = {'cells': [{'cell_type': 'code', 'source': [code]}]}
        controller = {'source_commit': parent.SOURCE, 'notebook': parent.NOTEBOOK,
            'status': 'failed', 'exit_code': 1, 'timed_out': False,
            'child': {'child_exit_confirmed': True, 'exit_code': 1}, 'public_fixture_test_mode': False}
        binding = {'scriptVersionId': parent.RUN, 'source_commit': parent.SOURCE,
                   'identity': 'FAKE_ONLY', 'package_sha256': parent.PACKAGE_SHA}
        closed = {**parent.descriptor(), 'scriptVersionId': parent.RUN,
            'original_raw_archive_sha256': parent.RAW_SHA, 'owned_compute_released': True,
            'controller_child_exit_confirmed': True, 'controller_child_exit_code': 1,
            'timed_out': False, 'model_training_admitted': False, 'identity': 'FAKE_ONLY'}
        values = {'CONTROLLER.json': controller, 'BINDING.json': binding,
                  'ACTUAL.ipynb': notebook, 'PREPARED.ipynb': notebook}
        for name, value in values.items(): dump(inputs / (parent.PREFIX + name), value)
        (inputs / (parent.PREFIX + 'TERMINAL_AX.txt')).write_text(
            'https://www.kaggle.com/code/' + parent.NOTEBOOK + '?scriptVersionId=' + str(parent.RUN)
            + '\n{"status": "failed"}', encoding='utf-8')
        (inputs / (parent.PREFIX + 'RELEASE_AX.txt')).write_text(
            'https://www.kaggle.com/code/' + parent.NOTEBOOK + ' 0 Active Events Draft Session off', encoding='utf-8')
        (base / parent.MANIFEST).write_text('{}', encoding='utf-8')
        def seal():
            closed.update({'controller_sha256': sha(inputs / (parent.PREFIX + 'CONTROLLER.json')),
                'actual_saved_notebook_sha256': sha(inputs / (parent.PREFIX + 'ACTUAL.ipynb')),
                'release_AX_original_sha256': sha(inputs / (parent.PREFIX + 'RELEASE_AX.txt'))})
            dump(inputs / (parent.PREFIX + 'CLOSEOUT.json'), closed)
            return sha(inputs / (parent.PREFIX + 'CLOSEOUT.json'))
        class Profile:
            @staticmethod
            def notebook_code(*args): return code
        return inputs, source, controller, closed, seal, lambda name: Profile

    def test_exact_failure_and_saved_code_required_separately_from_a_release_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs, source, controller, closed, seal, module = self._fixture(folder)
            with patch.object(parent, 'CLOSEOUT_SHA', seal()):
                self.assertEqual(parent.validate_failure_evidence(inputs, source, module), closed)
            for alteration in ['child_exit', 'timeout', 'success', 'fixture', 'wrong_source_code']:
                with self.subTest(alteration=alteration):
                    bad = deepcopy(controller)
                    if alteration == 'child_exit': bad['child']['child_exit_confirmed'] = False
                    if alteration == 'timeout': bad['timed_out'] = True
                    if alteration == 'success': bad['status'] = 'complete'; bad['exit_code'] = 0
                    if alteration == 'fixture': bad['public_fixture_test_mode'] = True
                    dump(inputs / (parent.PREFIX + 'CONTROLLER.json'), bad)
                    if alteration == 'wrong_source_code':
                        dump(inputs / (parent.PREFIX + 'ACTUAL.ipynb'), {'cells': [
                            {'cell_type': 'code', 'source': ['print("substitute")']}]})
                    with patch.object(parent, 'CLOSEOUT_SHA', seal()), self.assertRaises(PermissionError):
                        parent.validate_failure_evidence(inputs, source, module)

    def test_target_release_requires_exact_notebook_url_and_no_live_events(self):
        for capture in ['0 Active Events Draft Session off',
            'https://www.kaggle.com/code/farhad89/other 0 Active Events Draft Session off',
            'https://www.kaggle.com/code/' + parent.NOTEBOOK + '-copy 0 Active Events Draft Session off',
            'https://www.kaggle.com/code/' + parent.NOTEBOOK + ' 1 Active Events Draft Session off']:
            with self.subTest(capture=capture), tempfile.TemporaryDirectory() as folder:
                inputs, source, _, _, seal, module = self._fixture(folder)
                (inputs / (parent.PREFIX + 'RELEASE_AX.txt')).write_text(capture, encoding='utf-8')
                with patch.object(parent, 'CLOSEOUT_SHA', seal()), self.assertRaises(PermissionError):
                    parent.validate_failure_evidence(inputs, source, module)


if __name__ == '__main__': unittest.main()
