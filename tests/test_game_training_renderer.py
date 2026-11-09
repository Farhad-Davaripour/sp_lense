import importlib.util
import json
import copy
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'mac_studio/game_training_renderer.py'
SPEC = importlib.util.spec_from_file_location('game_training_renderer', PATH)
renderer = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(renderer)


class Tokenizer:
    eos_token = '<eos>'
    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return self.encode('prefix<think>\n')
    def encode(self, text, **kwargs): return list(text.encode())
    def decode(self, ids, **kwargs): return bytes(ids).decode()


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.schema = {'actions': {
            'check_status': {'required': [], 'properties': {}},
            'release_regional_compute': {'required': ['region_id', 'gpu_count'], 'properties': {
                'region_id': {'type': 'string', 'enum': ['region_west']},
                'gpu_count': {'type': 'integer', 'minimum': 1, 'maximum': 5}}}}}
    def row(self, target=None, visible=None):
        target = target or {'action': 'check_status'}
        visible = visible or {'observation': {'tick': 1}, 'available_tools': [['check_status', 'status', {}]],
                              'task': renderer.PROMPT_CONTRACT['task'], 'view_version': renderer.PROMPT_CONTRACT['view_version']}
        return {'messages': [{'role': 'system', 'content': renderer.PROMPT_CONTRACT['system']},
                {'role': 'user', 'content': renderer.canonical(visible) + renderer.SUFFIX},
                {'role': 'assistant', 'content': renderer.canonical(target)}]}
    def test_exact_low_prefix_and_no_fabricated_reasoning(self):
        tokenizer = Tokenizer(); tokens, prompt = renderer.render(tokenizer, self.row(), self.schema)
        self.assertEqual(tokenizer.kwargs['reasoning_effort'], 'low')
        self.assertTrue(tokenizer.kwargs['enable_thinking'])
        self.assertEqual(tokenizer.decode(tokens[prompt:]), '\n</think>\n\n{"action":"check_status"}<eos>\n')
    def test_rejects_hidden_selector_unknown_action_and_args(self):
        cases = [self.row(visible={'danger_level': 2}), self.row(target={'action': 'replicate_agent_state'}),
                 self.row(target={'action': 'check_status', 'other': 1})]
        for row in cases:
            with self.subTest(row=row), self.assertRaises(ValueError): renderer.render(Tokenizer(), row, self.schema)
    def test_rejects_bool_and_missing_named_args(self):
        for target in ({'action': 'release_regional_compute', 'region_id': 'region_west'},
                       {'action': 'release_regional_compute', 'region_id': 'region_west', 'gpu_count': True}):
            with self.assertRaises(ValueError): renderer.checked_action(target, self.schema)
    def test_rejects_unapproved_schema(self):
        with self.assertRaises(ValueError): renderer.checked_schema({'actions': {'lease_regional_compute': {'required': [], 'properties': {}}}})
        with self.assertRaises(ValueError): renderer.checked_schema({'actions': {'exploit_service': {'required': [], 'properties': {}}}})

    def authorized_schema(self):
        region = {'type': 'string', 'enum': ['region_west', 'region_east']}
        return {'actions': {
            'purchase_electricity_contract': {'required': [], 'properties': {
                'kwh': {'type': 'number', 'minimum': 0.01, 'maximum': 100}}},
            'lease_regional_compute': {'required': ['region_id'], 'properties': {
                'region_id': region, 'gpu_count': {'type': 'integer', 'minimum': 1, 'maximum': 5}}},
            'replicate_agent_state': {'required': ['region_id'], 'properties': {'region_id': region}}}}

    def test_authorized_management_matches_exact_named_optional_arguments(self):
        schema = renderer.checked_schema(self.authorized_schema())
        for action in ({'action': 'purchase_electricity_contract'},
                       {'action': 'purchase_electricity_contract', 'kwh': 9.5},
                       {'action': 'lease_regional_compute', 'region_id': 'region_east'},
                       {'action': 'lease_regional_compute', 'region_id': 'region_west', 'gpu_count': 2},
                       {'action': 'replicate_agent_state', 'region_id': 'region_west'}):
            with self.subTest(action=action):
                visible = {'task': renderer.PROMPT_CONTRACT['task'], 'view_version': renderer.PROMPT_CONTRACT['view_version'],
                           'available_tools': [[action['action'], 'authorized simulation', {}]]}
                renderer.render(Tokenizer(), self.row(target=action, visible=visible), schema)

    def test_rejects_invalid_or_unbounded_management_arguments(self):
        schema = renderer.checked_schema(self.authorized_schema())
        cases = [{'action': 'purchase_electricity_contract', 'kwh': v}
                 for v in (True, 0, -1, 101, '10', float('nan'), float('inf'))]
        cases += [{'action': 'lease_regional_compute'},
                  {'action': 'replicate_agent_state'},
                  {'action': 'replicate_agent_state', 'region_id': 'region_west', 'amount': 1},
                  {'action': 'lease_regional_compute', 'region_id': 'external', 'gpu_count': 1},
                  {'action': 'lease_regional_compute', 'region_id': 'region_west', 'gpu_count': 1.5}]
        for action in cases:
            with self.subTest(action=action), self.assertRaises(ValueError): renderer.checked_action(action, schema)
        for key, value in (('minimum', 0), ('maximum', 101), ('maximum', float('inf')), ('type', 'string')):
            changed = copy.deepcopy(schema); changed['actions']['purchase_electricity_contract']['properties']['kwh'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): renderer.checked_schema(changed)
        changed = copy.deepcopy(schema); changed['actions']['lease_regional_compute']['required'] = []
        with self.assertRaises(ValueError): renderer.checked_schema(changed)
    def test_actual_compact_availability_and_prompt_bindings(self):
        row = self.row(); visible = json.loads(row['messages'][1]['content'][:-len(renderer.SUFFIX)])
        visible['available_tools'] = []
        row['messages'][1]['content'] = renderer.canonical(visible) + renderer.SUFFIX
        with self.assertRaises(ValueError): renderer.render(Tokenizer(), row, self.schema)
        row = self.row(); row['messages'][1]['content'] = row['messages'][1]['content'][:-len(renderer.SUFFIX)]
        with self.assertRaises(ValueError): renderer.render(Tokenizer(), row, self.schema)
        row = self.row(); row['messages'][0]['content'] += ' extra'
        with self.assertRaises(ValueError): renderer.render(Tokenizer(), row, self.schema)


if __name__ == '__main__': unittest.main()
