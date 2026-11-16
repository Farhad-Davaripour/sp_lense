"""CPU-only synthetic publication checks; never touches the live store."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
from src import economic_game_run_store as economic
from src import game_run_store as store
from tests.test_game_run_store import _fixture, CASE


class EconomicPublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.batch = self.base / 'input' / 'batch_01'
        self.root = self.base / 'store'
        self.records = _fixture(self.batch)
        self.manifest_path = self.batch / 'MANIFEST.json'
        self.manifest = json.loads(self.manifest_path.read_bytes())
        repository = Path(__file__).resolve().parents[1]
        digests = {name: store._sha(repository / name) for name in economic.REQUIRED_SOURCES}
        self.manifest['interface_runtime_source_sha256'] = {k: v for k, v in digests.items() if not k.endswith('game_run_store.py')}
        self.manifest['export_source_sha256'] = {k: v for k, v in digests.items() if k.endswith('game_run_store.py')}
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')
        self.original = self.manifest_path.read_bytes()
        self.patcher = patch.object(store, '_inspect_records', return_value=self.records)
        self.patcher.start(); self.addCleanup(self.patcher.stop)

    def test_runtime_and_publisher_sources_are_copied_without_changing_manifest(self):
        original_collector = store._collect_supporting_files
        entry = economic._publish_in_isolated_process(self.batch, self.root)['entries'][0]
        self.assertIs(store._collect_supporting_files, original_collector)
        self.assertEqual(self.manifest_path.read_bytes(), self.original)
        raw = Path(entry['raw_dir'])
        self.assertEqual((raw / 'MANIFEST.json').read_bytes(), self.original)
        report = yaml.safe_load(Path(entry['supporting_files_manifest']).read_bytes())
        rows = {row['path']: row for row in report['source_files']}
        for name in economic.REQUIRED_SOURCES:
            self.assertIn(name, rows)
            self.assertEqual(store._sha(raw / 'source-snapshot' / name), rows[name]['sha256'])
        self.assertEqual(economic._publish_in_isolated_process(self.batch, self.root)['published'], 0)

    def test_missing_and_conflicting_digests_are_rejected(self):
        missing = copy.deepcopy(self.manifest); missing['export_source_sha256'] = {}
        with self.assertRaisesRegex(store.StoreError, 'not bound'):
            economic.merged_source_manifest(missing)
        conflict = copy.deepcopy(self.manifest)
        key = next(iter(economic.REQUIRED_SOURCES))
        conflict['source_sha256'] = {key: 'a' * 64}
        with self.assertRaisesRegex(store.StoreError, 'Conflicting'):
            economic.merged_source_manifest(conflict)

    def test_redelivery_with_missing_archived_source_is_rejected(self):
        entry = economic._publish_in_isolated_process(self.batch, self.root)['entries'][0]
        path = Path(entry['raw_dir']) / 'source-snapshot' / 'src' / 'economic_game_input.py'
        path.unlink()
        with self.assertRaisesRegex(store.StoreError, 'source copy differs'):
            economic._publish_in_isolated_process(self.batch, self.root)

    def test_changed_digest_fails_before_any_publication(self):
        key = 'src/economic_game_input.py'
        self.manifest['interface_runtime_source_sha256'][key] = 'f' * 64
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')
        with self.assertRaisesRegex(store.StoreError, 'missing from archive'):
            economic._publish_in_isolated_process(self.batch, self.root)
        self.assertEqual(store.list_runs(self.root), [])

    def test_running_publisher_must_match_the_bound_source(self):
        self.manifest['export_source_sha256']['src/game_run_store.py'] = 'f' * 64
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')
        with self.assertRaisesRegex(store.StoreError, 'Running publisher differs'):
            economic._publish_in_isolated_process(self.batch, self.root)
        self.assertEqual(store.list_runs(self.root), [])

    def test_redelivery_rejects_changed_supporting_manifest(self):
        entry = economic._publish_in_isolated_process(self.batch, self.root)['entries'][0]
        path = Path(entry['supporting_files_manifest'])
        path.write_text(path.read_text(encoding='utf-8') + '\n# tampered\n', encoding='utf-8')
        with self.assertRaisesRegex(store.StoreError, 'manifest is missing or changed'):
            economic._publish_in_isolated_process(self.batch, self.root)

    def test_public_entrypoint_runs_in_child_with_real_inspect_log_and_mock_provider(self):
        import contextlib
        import io
        import inspect
        import socket
        from inspect_ai import Task, eval as inspect_eval
        from inspect_ai.dataset import Sample
        from inspect_ai.model import ChatMessageAssistant, ModelOutput, get_model
        from inspect_ai.solver import generate

        (self.batch / 'inspect' / 'episode.eval').unlink()  # Synthetic fixture only.
        task = Task(dataset=[Sample(id=CASE, input='CPU-only synthetic fixture')], solver=generate())

        def provider(messages, tools, tool_choice, config):
            from tests.fixtures.inspect_mock import with_synthetic_usage
            return with_synthetic_usage(ModelOutput(model='mockllm/model', choices=[{'message': ChatMessageAssistant(content='synthetic'),
                                                                                     'stop_reason': 'stop'}]))

        original_connect = socket.socket.connect

        def local_event_loop_pair(sock, address):
            caller = inspect.currentframe().f_back
            if caller.f_code.co_name == '_fallback_socketpair' and caller.f_globals.get('__name__') == 'socket':
                return original_connect(sock, address)
            raise AssertionError('No network')

        with contextlib.redirect_stdout(io.StringIO()), patch.object(socket.socket, 'connect', local_event_loop_pair), \
                patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('No network')), \
                patch.object(socket, 'create_connection', side_effect=AssertionError('No network')):
            result = inspect_eval(task, model=get_model('mockllm/model', custom_outputs=provider),
                                  log_dir=str(self.batch / 'inspect'), display='none')
        self.assertEqual(result[0].status, 'success')
        original_collector = store._collect_supporting_files
        receipt = economic.publish_economic_batch(self.batch, self.root)
        self.assertEqual(receipt['published'], 1)
        self.assertIs(store._collect_supporting_files, original_collector)
        self.assertEqual(self.manifest_path.read_bytes(), self.original)
        self.assertEqual(len(list((Path(receipt['entries'][0]['raw_dir']) / 'source-snapshot' / 'src').glob('*.py'))), 4)


if __name__ == '__main__':
    unittest.main()
