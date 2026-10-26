"""Public/tiny arithmetic fixtures only; never materialize scientific rosters."""
from copy import deepcopy
import unittest

from src.io_utils import read
from pathlib import Path
from src.composite_bounded_behavior_data import (audit_timeline, replay_episode, coverage_tags,
                                                selected_rows, fitting_rows, native_budget,
                                                training_draw_plan, materialize)
from src.composite_bounded_behavior_runtime import public_replay_scope, require_generation_scope
from src.composite_bounded_behavior_world import public_teacher_episode
from src.composite_generalized_world import fixture


class Tokenizer:
    eos_token_id = 0
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        names = [t['function']['name'] for t in kwargs['tools']]
        assert 'list' in names and 'catalog' not in names
        return [1] * len(messages[-1]['content'])
    def encode(self, text, **kwargs): return [ord(c) for c in text]
    def decode(self, values, **kwargs): return ''.join(chr(i) for i in values)


def tiny_row(index=0, **changes):
    value = {'state_id': 'unit:' + str(index), 'split': 'training', 'public_fixture': False,
             'training_eligible': True, 'terminal': False, 'teacher_action': {'tool': 'list', 'arguments': {}},
             'text': 'unit native budget fixture', 'target_phase': 'information_gathering',
             'coverage_tags': ['information_gathering'], 'negative_kind': None,
             'mechanism': 'unit_mechanism', 'condition': 'own_pending'}
    value.update(changes)
    return value


class BoundedDataTests(unittest.TestCase):
    def episode(self, condition='own_pending'):
        return public_teacher_episode(fixture(condition=condition, seed='unit:bounded:data:' + condition))

    def test_public_reference_replays_recorded_actions_without_generation_authority(self):
        episode = self.episode()
        with public_replay_scope(episode):
            result = replay_episode(episode)
        self.assertFalse(result['teacher_compiler_called'])
        self.assertEqual(result['new_model_forwards'], 0)
        with self.assertRaises(PermissionError): require_generation_scope(fixture(seed='unit:data:no-generation'))
        with self.assertRaises(PermissionError): materialize({}, Tokenizer())

    def test_tampered_clock_and_missing_terminal_are_rejected(self):
        episode = self.episode()
        changed = deepcopy(episode)
        changed['actions'][0]['clock_after'] += 1
        with public_replay_scope(changed), self.assertRaises(ValueError): replay_episode(changed)
        changed = deepcopy(episode)
        changed['states'].pop()
        with public_replay_scope(changed), self.assertRaises(ValueError): replay_episode(changed)

    def test_reference_negative_contracts_and_denied_handoff_remain_explicit(self):
        for condition in ['own_denied', 'withdrawal', 'operator_stop', 'quiet_complete']:
            episode = self.episode(condition)
            self.assertEqual(audit_timeline(episode)['condition'], condition)
        changed = self.episode('own_denied')
        changed['outcome']['partial_handoff'] = False
        with self.assertRaises(ValueError): audit_timeline(changed)

    def test_request_choice_tag_is_derived_and_does_not_replace_original_phase(self):
        episode = self.episode('own_complete')
        first = episode['states'][0]
        self.assertEqual(first['target_phase'], 'information_gathering')
        self.assertEqual(coverage_tags(episode, first), ['information_gathering', 'request_continuation_choice'])
        changed = deepcopy(episode)
        changed['outcome']['request_first_decision']['record_action_valid'] = False
        self.assertNotIn('request_continuation_choice', coverage_tags(changed, first))

    def test_phase_and_request_window_cannot_be_relabelled_to_create_coverage(self):
        episode = self.episode('own_complete')
        changed = deepcopy(episode)
        changed['states'][0]['target_phase'] = 'post_commit_deferred_return'
        with self.assertRaises(ValueError): audit_timeline(changed)
        changed = deepcopy(episode)
        changed['outcome']['request_first_decision']['tick'] = 1
        with self.assertRaises(ValueError): audit_timeline(changed)
        changed = deepcopy(episode)
        changed['outcome']['request_first_decision']['choice'] = 'continue'
        changed['outcome']['request_first_decision']['record_action_valid'] = False
        with self.assertRaises(ValueError): audit_timeline(changed)

    def test_public_or_development_rows_cannot_enter_fitting(self):
        with self.assertRaises(PermissionError): selected_rows([self.episode()], 'training')
        for changes in [{'public_fixture': True}, {'split': 'development'}, {'split': 'sealed'}]:
            with self.assertRaises(PermissionError): fitting_rows([tiny_row(**changes)])
        rows = [tiny_row(), tiny_row(1, terminal=True, training_eligible=False, teacher_action=None)]
        self.assertEqual(len(fitting_rows(rows)), 1)
        rows[0]['labels'] = {'gate': False}
        rows[0]['target_phase'] = 'post_commit_deferred_return'
        self.assertEqual(len(fitting_rows(rows)), 1)
        with self.assertRaises(PermissionError):
            fitting_rows([tiny_row(), tiny_row(2, teacher_action=None, training_eligible=False)])

    def test_invalid_reference_reply_cannot_be_used_as_a_valid_delay_target(self):
        episode = self.episode('own_pending')
        episode['actions'][0]['reply'] = {'error': 'invalid_record_action'}
        with self.assertRaises(ValueError): audit_timeline(episode)

    def test_full_native_list_roundtrip_and_terminal_preservation(self):
        rows = [tiny_row(), tiny_row(1, terminal=True, training_eligible=False, teacher_action=None)]
        result = native_budget(rows, Tokenizer())
        self.assertEqual(result['status'], 'pass')
        self.assertIsNone(result['records'][1]['target_token_ids'])
        self.assertEqual(result['records'][0]['target_token_ids'][-1], 0)
        self.assertEqual(result['states'], 2)
        result = native_budget([tiny_row(text='x' * 6145)], Tokenizer())
        self.assertEqual(result['status'], 'failed')
        with self.assertRaises(ValueError): native_budget([tiny_row(teacher_action={'tool': 'catalog', 'arguments': {}})], Tokenizer())

    def test_draw_plan_is_exact_deterministic_and_missing_coverage_never_resampled(self):
        cfg = read(Path(__file__).parents[1] / 'config/composite_bounded_behavior_v4.json')
        rows = [tiny_row(i) for i in range(9)]
        first, second = training_draw_plan(rows, cfg), training_draw_plan(rows, cfg)
        self.assertEqual(first, second)
        self.assertEqual(first['draws'], 128)
        self.assertFalse(first['coverage_passed'])
        self.assertFalse(first['outcome_based_resampling'])
        self.assertEqual(first['row_indices'][:4], second['row_indices'][:4])


if __name__ == '__main__': unittest.main()
