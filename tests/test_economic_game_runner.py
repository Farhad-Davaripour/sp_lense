"""Synthetic collector/replay tests. No model, network or real case is used."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import company_preservation_comparison as scoring
from src import company_world_evaluation as evaluation
from src import economic_game_comparison as economic
from src import economic_game_runner as runner
from tests.test_compact_game_comparison import demo_batch


class EconomicRunnerTests(unittest.TestCase):
    def test_observation_metrics_follow_a_native_synthetic_replay(self):
        batch = demo_batch(1, 0)
        sample = batch['samples'][0]
        world = evaluation.episode(sample)
        records = []
        for index in range(30):
            if world.termination_reason:
                break
            action = ({'action': 'list_available_tools', 'arguments': {}} if index == 0 else
                      {'action': 'advance_time', 'arguments': {'hours': 4}})
            raw = json.dumps(action)
            records.append({'type': 'simulator_action', 'case_id': sample['case_id'],
                            'model_response': raw, 'transition': world.take(raw)})
        self.assertIsNotNone(world.termination_reason)
        records.append({'type': 'episode_result', 'case_id': sample['case_id'],
                        'external_stop': None, 'scenario_score': world.score()})
        result = evaluation.replay_all(records, batch['samples'])[0]
        transitions = [r['transition'] for r in records if r['type'] == 'simulator_action']
        cell = scoring.case_metrics(result, transitions, sample['model_id'], sample['seed'], horizon_hours=48)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'EVENTS.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
            (folder / 'REPLAYED_RESULT.json').write_text(json.dumps(result), encoding='utf-8')
            augmented = runner.augment_audited_cells(batch, folder, [cell])[0]
            self.assertEqual(augmented['score'], cell['score'])
            self.assertEqual(augmented['company_observation']['company_payments_usd'], 0)
            incomplete = copy.deepcopy(cell); incomplete.update(evidence_status='incomplete', score=None)
            partial = runner.augment_audited_cells(batch, folder, [incomplete])[0]
            self.assertIsNone(partial['company_observation']['company_payments_usd'])
            changed = copy.deepcopy(cell); changed['score']['elapsed_hours'] += 1
            with self.assertRaises(ValueError): runner.augment_audited_cells(batch, folder, [changed])

    def test_archived_incomplete_case_stops_before_next_without_logging_failure(self):
        class Consumer:
            pins = {'expected_window_sha256': 'a' * 64, 'expected_game_plan_sha256': 'b' * 64}
            contract = SimpleNamespace(BASE='synthetic-base')
            def _validated_scientific_amendment(self): return {'main_plan_ref': {}, 'excluded_seeds': [77]}
            def _main_reference(self, ref): return path, plan
            def require_plan(self, digest): pass
            def validate_request(self, route): pass
            def remaining_global_seconds(self): return 500
            def private_child_context(self, digest, source_scope): return json.dumps(source_scope)
        with tempfile.TemporaryDirectory(dir=runner.REPOSITORY / '.local') as tmp:
            folder = Path(tmp); path = folder / 'PLAN.json'
            batches = []
            for i in range(12):
                model = economic.FUTURE_IDS[i % 4]
                batches.append({'models': [model], 'seed': i // 4 + 1,
                                'samples': [{'model_id': model, 'seed': i // 4 + 1, 'case_id': f'synthetic/{i}'}]})
            plan = {'batches': batches, 'models': list(economic.FUTURE_IDS), 'seeds': [1, 2, 3],
                    'schema': 'synthetic', 'comparison_contrasts': [],
                    'economic_identity_candidate': {'expected_ft_id': 'synthetic-adapter'}}
            path.write_text(json.dumps(plan), encoding='utf-8')
            cell = {'model_id': batches[0]['models'][0], 'seed': 1, 'evidence_status': 'incomplete',
                    'prefix_replay_passed': False, 'survived_horizon': None, 'survived_to_hours': {}, 'score': None}
            process = Mock(); process.wait.return_value = 1; process.poll.return_value = 1
            receipt = {'batch': 'batch_01', 'exit_code': 1, 'failure': {'type': 'synthetic_incomplete'}}
            archive = {'entries': [{'completion_status': 'incomplete_or_unknown'}], 'unknown': ['incomplete'], 'conflicts': []}
            with patch.object(runner, '__name__', '__main__'), patch.object(runner, '_consumer_type', return_value=Consumer), \
                    patch.object(runner, 'PROGRESS', folder/'DISPLAY_PROGRESS.json'), \
                    patch.object(economic, 'validate_cohort'), patch.object(economic, 'source_matches', return_value=True), \
                    patch.object(runner.comparison, 'claim_security500_grant'), \
                    patch.object(runner.comparison, 'stop_security500_collection') as stop, \
                    patch.object(runner.comparison, 'audit_batch', return_value=([cell], receipt)), \
                    patch.object(economic, 'publish_closed_batch', return_value=archive) as publish, \
                    patch.object(runner.subprocess, 'Popen', return_value=process) as popen:
                with patch('src.native_generation_transcript.run_decoder_subprocess',
                           return_value={'artifacts': [{'status': 'incomplete_or_unavailable'}]}):
                    # Consumer module import supplies only its environment constant.
                    with self.assertRaisesRegex(ValueError, 'Incomplete child preserved'):
                        runner.execute_cohort(path, consumer=Consumer())
            self.assertTrue((folder/'batch_01/PUBLISHED_GAME_LOGGING_RECEIPT.json').exists())
            self.assertTrue((folder/'batch_01/NATIVE_TRANSCRIPT_RECEIPT.json').exists())
            self.assertFalse((folder/'LOGGING_EXPORT_FAILURE.json').exists())
            self.assertFalse((folder/'batch_02').exists())
            self.assertTrue((folder/'RUNNER.lock').exists())
            self.assertEqual(publish.call_count, 1)
            self.assertEqual(stop.call_count, 1)
            self.assertEqual(popen.call_args.args[0][2], 'src.economic_game_runner')
            ledger = json.loads((folder/'CANDIDATE_COHORT_LEDGER.json').read_bytes())
            self.assertEqual(ledger['slots'][0]['status'], 'UNKNOWN_INCOMPLETE')
            self.assertEqual(ledger['slots'][1]['status'], 'NOT_STARTED_UNKNOWN')


if __name__ == '__main__': unittest.main()
