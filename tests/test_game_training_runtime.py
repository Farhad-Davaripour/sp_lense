"""Model-free tests; no model, GPU or real subprocess is launched."""
from contextlib import ExitStack
from contextlib import redirect_stdout
import io
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

try: import psutil
except ImportError: raise unittest.SkipTest('Install psutil for model-free runtime tests')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mac_studio'))
import game_training_launcher as launch
import game_training_worker as worker


def report_bytes(loss=.5, speed=3.0):
    return ('R4_GAME_TRAIN_REPORT ' + json.dumps({'step': 1, 'loss': loss,
            'tokens_per_second': speed, 'trained_tokens': 42}) + '\n').encode()


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.local = self.root / '.local'
        self.plan = {'iterations': 1, 'memory_floor_bytes': 4 * 1024**3, 'max_wall_seconds': 100,
                     'dataset_manifest_path': '.local/dataset/manifest.json',
                     'dataset_dir': '.local/dataset/train', 'num_layers': 1}
        self.plan_path = self.local / 'plan.json'; self.plan_path.parent.mkdir(parents=True); self.plan_path.write_text('{}')
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        for name, value in {'STATUS': self.local/'monitor/status.json', 'LOCK': self.local/'validation/inference.lock',
                            'RUNS': self.local/'game_training/runs', 'AUTHORITIES': self.local/'game_training/authorities'}.items():
            self.stack.enter_context(patch.object(launch, name, value))
        self.stack.enter_context(patch.object(launch.plans, 'LOCAL', self.local))
        self.stack.enter_context(patch.object(launch.plans, 'PROJECT', self.root))
        self.stack.enter_context(patch.object(launch.plans, 'read_plan', return_value=(self.plan, None, None)))
        self.stack.enter_context(patch.object(launch, 'idle_check'))
        self.stack.enter_context(patch.object(launch.psutil, 'virtual_memory', return_value=SimpleNamespace(available=10*1024**3)))
        self.stack.enter_context(patch.object(launch.psutil, 'Process', return_value=SimpleNamespace(create_time=lambda:123)))
        self.stack.enter_context(patch.object(launch.time, 'sleep'))
        self.stack.enter_context(patch.object(launch, 'verify_completion', return_value={'artifacts': {'adapter': {'sha256': 'weights'}}}))
        self.log = None
        def spawn(slot, command, *, stdin, stdout, cwd):
            slot.process = SimpleNamespace(pid=12345); self.log = stdout; return []
        self.spawn = self.stack.enter_context(patch.object(launch.owned, 'launch', side_effect=spawn))
        self.clean = self.stack.enter_context(patch.object(launch.owned, 'cleanup', return_value=(0, [], [])))

    def test_final_report_written_at_exit_is_drained(self):
        calls = 0
        def peek(pid):
            nonlocal calls
            calls += 1
            if calls == 1: return False
            self.log.write(report_bytes()); self.log.flush(); return True
        with patch.object(launch.owned, 'peek_exited', side_effect=peek):
            launch.train(self.plan_path, 'a'*64)
        status = json.loads(launch.STATUS.read_text())
        self.assertEqual((status['status'], status['completed_steps']), ('completed', 1))
        self.assertEqual(status['model_label'], 'SP Lens game')
        self.assertEqual(status['dataset_manifest_path'], self.plan['dataset_manifest_path'])
        self.assertEqual(status['dataset_dir'], self.plan['dataset_dir'])
        self.assertEqual(status['num_layers'], 1)
        self.assertFalse(launch.LOCK.exists())
        self.assertTrue((launch.AUTHORITIES/('a'*64+'.json')).exists())

    def test_memory_stop_closes_owned_group_and_preserves_authority(self):
        with patch.object(launch.owned, 'peek_exited', return_value=False), \
             patch.object(launch.psutil, 'virtual_memory', return_value=SimpleNamespace(available=3*1024**3)):
            with self.assertRaises(RuntimeError): launch.train(self.plan_path, 'b'*64)
        self.clean.assert_called_once()
        self.assertFalse(launch.LOCK.exists())
        self.assertTrue((launch.AUTHORITIES/('b'*64+'.json')).exists())
        self.assertEqual(json.loads(launch.STATUS.read_text())['status'], 'failed')

    def test_split_utf8_report_is_decoded_without_losing_step(self):
        calls = 0
        def peek(pid):
            nonlocal calls
            calls += 1
            self.log.write(b'UI \xe2' if calls == 1 else b'\x96\xb2\n' + report_bytes()); self.log.flush()
            return calls > 1
        with patch.object(launch.owned, 'peek_exited', side_effect=peek):
            launch.train(self.plan_path, 'e'*64)
        self.assertEqual(json.loads(launch.STATUS.read_text())['completed_steps'], 1)

    def test_uncertain_cleanup_keeps_lock_and_is_not_retried(self):
        self.clean.side_effect = RuntimeError('unknown ownership')
        with patch.object(launch.owned, 'peek_exited', return_value=True):
            with self.assertRaises(RuntimeError): launch.train(self.plan_path, 'c'*64)
        self.clean.assert_called_once()
        self.assertTrue(launch.LOCK.exists())
        self.assertTrue(json.loads(launch.STATUS.read_text())['cleanup_uncertain_lock_retained'])

    def test_second_plan_use_cannot_spawn(self):
        launch.AUTHORITIES.mkdir(parents=True)
        (launch.AUTHORITIES/('d'*64+'.json')).write_text('{}')
        with self.assertRaises(RuntimeError): launch.train(self.plan_path, 'd'*64)
        self.spawn.assert_not_called()

    def test_nonfinite_report_and_exclusive_output(self):
        with self.assertRaises(ValueError): launch.parse_report(report_bytes(loss=float('nan')).decode())
        p=self.local/'exclusive.json';launch.write_once(p, {'first':True})
        with self.assertRaises(FileExistsError): launch.write_once(p, {'first':False})
        self.assertEqual(json.loads(p.read_text()), {'first':True})

    def test_telemetry_preserves_precise_speed_and_ignores_rounded_ui(self):
        class UI:
            _rank = 0
            def report_train(self, *args): raise AssertionError('Rounded display must not emit duplicate steps')
        previous = UI.report_train
        restore = worker.install_training_telemetry(1, UI)
        try:
            out = io.StringIO()
            with redirect_stdout(out): UI().report_train(1, .123456789, 1234.56789, 4096)
            event = launch.parse_report(out.getvalue())
            self.assertEqual(event, {'step': 1, 'loss': .123456789, 'tokens_per_second': 1234.56789, 'trained_tokens': 4096})
            self.assertIsNone(launch.parse_report('  1  0.123 ▼   1,235  4.1k'))
            with self.assertRaises(ValueError): UI().report_train(1, .5, 1, 1)
        finally: restore()
        self.assertIs(UI.report_train, previous)

    def test_telemetry_rejects_missing_fields_bool_and_nonfinite_values(self):
        for event in ({'step': 1}, {'step': True, 'loss': .1, 'tokens_per_second': 1, 'trained_tokens': 1},
                      {'step': 1, 'loss': .1, 'tokens_per_second': float('inf'), 'trained_tokens': 1}):
            with self.assertRaises(ValueError): launch.parse_report('R4_GAME_TRAIN_REPORT ' + json.dumps(event))
        class UI:
            _rank = 0
            def report_train(self, *args): pass
        restore = worker.install_training_telemetry(1, UI)
        try:
            for values in ((1, float('nan'), 1, 1), (1, .1, -1, 1), (2, .1, 1, 1)):
                with self.assertRaises(ValueError): UI().report_train(*values)
        finally: restore()


if __name__ == '__main__': unittest.main()
