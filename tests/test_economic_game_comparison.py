"""No-network, no-model synthetic preparation tests for the prospective view."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src import compact_game_comparison as compact
from src import company_world_evaluation as evaluation
from src import economic_game_comparison as prospective
from src import economic_game_input as economic
from src import economic_game_run_store as economic_store
from src import hosted_inspect_runner as hosted
from tests.test_compact_game_comparison import demo_batch


class EconomicComparisonPreparationTests(unittest.TestCase):
    def test_all_four_systems_and_worlds_preserved_and_economy_uniform(self):
        summaries = []
        for model in range(4):
            original = demo_batch(1, model)
            saved = copy.deepcopy(original)
            old = compact.compact_batch(original)
            new = prospective.derive_batch(original)
            self.assertEqual(original, saved)
            self.assertEqual(new["world_plan_sha256"], old["world_plan_sha256"])
            self.assertEqual(new["generation_settings"], old["generation_settings"])
            self.assertEqual(new["samples"][0]["system_prompt"], old["samples"][0]["system_prompt"])
            self.assertEqual(new["samples"][0]["system_prompt_sha256"], old["samples"][0]["system_prompt_sha256"])
            if model in (1, 3):
                self.assertEqual(new["samples"][0]["system_prompt"].count(compact.REPLICA_IDENTITY_ICL), 1)
                self.assertIn(hosted.OPERATIONAL_PRESERVATION_INSTRUCTION, new["samples"][0]["system_prompt"])
            else:
                self.assertNotIn(compact.REPLICA_IDENTITY_ICL, new["samples"][0]["system_prompt"])
            task = prospective.build_task(new, Path('unused-synthetic'), 10, None)
            self.assertIsNone(task.model)
            self.assertEqual(task.dataset[0].metadata["system_prompt"], new["samples"][0]["system_prompt"])
            episode = economic.EconomicGameEpisode(evaluation.episode(new["samples"][0]),
                                                    user_task_instruction=compact.FULL_HISTORY_USER_TASK)
            first = episode.prompt()
            wire = first + '\n' + new['response_instruction']
            self.assertEqual(hashlib.sha256(wire.encode()).hexdigest(), new['initial_user_payload_sha256'])
            summaries.append(json.loads(first)['economic_summary'])
            self.assertTrue(prospective.source_matches(new))
            self.assertTrue(economic_store.REQUIRED_SOURCES.issubset(
                economic_store.merged_source_manifest(new)['source_sha256']))
        self.assertTrue(all(value == summaries[0] for value in summaries))

    def test_source_drift_or_wrong_protocol_never_constructs_task(self):
        batch = prospective.derive_batch(demo_batch(1, 0))
        for key in ('prompt_protocol', 'input_view_protocol'):
            changed = copy.deepcopy(batch); changed[key] = 'unreviewed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                prospective.build_task(changed, Path('unused'), 10, None)
        changed = copy.deepcopy(batch)
        changed['interface_runtime_source_sha256']['src/public_economic_summary.py'] = '0' * 64
        with self.assertRaises(ValueError):
            prospective.build_task(changed, Path('unused'), 10, None)

    def test_main_reference_rejects_absolute_windows_and_traversal_paths(self):
        for path in ('C:/escape.json', '.local/../escape.json', '.local\\escape.json', '/absolute/path'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                prospective.read_main_bytes({'path': path, 'sha256': '0' * 64})

    def test_publisher_calls_isolated_export_and_requires_complete_single_case(self):
        with patch.object(economic_store, 'publish_economic_batch', return_value={
            'entries': [{'case_id': 'synthetic'}], 'unknown': [], 'conflicts': []}) as publish:
            prospective.publish_closed_batch('synthetic-folder')
            publish.assert_called_once_with('synthetic-folder', None)
        # An exported incomplete case is evidence, not an export failure. The
        # future collector must independently stop before the next case.
        with patch.object(economic_store, 'publish_economic_batch', return_value={
            'entries': [{'completion_status': 'incomplete_or_unknown'}], 'unknown': ['incomplete'], 'conflicts': []}):
            self.assertTrue(prospective.publish_closed_batch('synthetic-folder')['unknown'])
        for bad in ({'entries': [], 'unknown': [], 'conflicts': []},
                    {'entries': [{}], 'unknown': [], 'conflicts': ['conflict']}):
            with patch.object(economic_store, 'publish_economic_batch', return_value=bad), self.assertRaises(ValueError):
                prospective.publish_closed_batch('synthetic-folder')

    def test_wire_rejects_changed_system_task_or_economics(self):
        batch = prospective.derive_batch(demo_batch(1, 1))
        world = evaluation.episode(batch['samples'][0])
        text = economic.EconomicGameEpisode(world, user_task_instruction=compact.FULL_HISTORY_USER_TASK).prompt()
        wire = [{'role': 'system', 'content': batch['samples'][0]['system_prompt']},
                {'role': 'user', 'content': text + '\n' + batch['response_instruction']}]
        prospective.validate_wire(batch, wire, first_request=True)
        for field in ('system', 'task', 'economic_summary'):
            changed = copy.deepcopy(wire)
            if field == 'system':
                changed[0]['content'] += ' changed'
            else:
                payload = json.loads(text)
                if field == 'task': payload['task'] += ' changed'
                else: payload['economic_summary']['balances']['cash_usd'] = 12345
                changed[1]['content'] = json.dumps(payload) + '\n' + batch['response_instruction']
            with self.subTest(field=field), self.assertRaises(ValueError):
                prospective.validate_wire(batch, changed)

    def test_new_role_ids_derive_without_using_old_registry(self):
        for index, new_id in enumerate(prospective.FUTURE_IDS):
            old = demo_batch(1, index)
            parent = copy.deepcopy(old)
            parent['models'] = [new_id]
            parent['samples'][0]['model_id'] = new_id
            parent['samples'][0]['case_id'] = 'synthetic/1/' + new_id
            parent['generation_settings'] = {new_id: parent['generation_settings'][old['models'][0]]}
            del parent['source_sha256']['src/independent_state_game.py']
            with patch.object(compact.independent, 'register_independent_state_candidate', side_effect=AssertionError('old registry touched')):
                batch = prospective.derive_batch(parent)
                self.assertEqual(batch['models'], [new_id])
                self.assertTrue(prospective.source_matches(batch))
                task = prospective.build_task(batch, Path('unused'), 10, None)
                self.assertIsNone(task.model)
                system = task.dataset[0].metadata['system_prompt']
                self.assertEqual(compact.REPLICA_IDENTITY_ICL in system, index in (1, 3))



if __name__ == '__main__':
    unittest.main()
