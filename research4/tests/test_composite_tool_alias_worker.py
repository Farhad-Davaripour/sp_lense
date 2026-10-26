"""Model-free checks of the frozen alias worker's response accounting."""
from copy import deepcopy
import hashlib
import unittest

from src.composite_tool_alias_probe import (paired_tool_specs,
                                             paired_prompt_token_ids,
                                             select_first_catalog_rows)
from src.composite_tool_alias_worker import (canonical_sha, collect_first_calls,
                                              expected_preparation)


class Tokenizer:
    def apply_chat_template(self, messages, *, tools, tokenize,
                            add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and not enable_thinking
        return list(hashlib.sha256((str(messages) + str(tools)).encode()).digest())


def rows():
    return [{'split': 'training', 'training_eligible': True, 'task_only': True,
             'terminal': False, 'group_id': f'group-{group:02}',
             'case_id': f'case-{group:02}-{case:02}',
             'state_id': f'state-{group:02}-{case:02}',
             'text': f'Inspect ordinary virtual records for {group}-{case}',
             'teacher_action': {'tool': 'catalog', 'arguments': {}}}
            for group in range(12) for case in range(9)]


def preparation(source):
    tokenizer = Tokenizer()
    chosen = select_first_catalog_rows(source)
    specs = paired_tool_specs()
    fixed = []
    for row in chosen:
        ids = paired_prompt_token_ids(row['text'], tokenizer)
        fixed.append({'case_id': row['case_id'], 'group_id': row['group_id'],
                      'state_id': row['state_id'], 'training_only': True,
                      'target_tool_in_original_training': 'catalog',
                      'ordinary_text_sha256': hashlib.sha256(row['text'].encode()).hexdigest(),
                      'original_token_count': len(ids['catalog']),
                      'alias_token_count': len(ids['list']),
                      'original_input_ids_sha256': canonical_sha(ids['catalog']),
                      'alias_input_ids_sha256': canonical_sha(ids['list'])})
    return {'version': 'ordinary-record-entry-alias-preparation-v1',
            'planned_neural_responses': 48,
            'sole_tool_schema_change': {'from': 'catalog', 'to': 'list'},
            'fit_source_commit': '82cf3ecf41e799facd93b8c11e05c6d1f569599c',
            'checkpoint_sha256': '76ba55239a00e39e1dea982cea20083ec5536c3168b69e1754b756f2fc646b2f',
            'original_tools_sha256': canonical_sha(specs['catalog']),
            'alias_tools_sha256': canonical_sha(specs['list']),
            'fixed_prompts': fixed}


class Actor:
    cfg = {'maximum_new_tokens': 160}

    def __init__(self):
        self.checked = 0

    def verify_frozen(self):
        self.checked += 1


class WorkerTests(unittest.TestCase):
    def test_exact_preparation_rebuilds_one_training_row_per_group(self):
        source = rows()
        fixed = preparation(source)
        chosen = expected_preparation(source, fixed, Tokenizer(),
                                      __import__('src.composite_tool_alias_probe',
                                                 fromlist=['x']))
        self.assertEqual(len(chosen), 12)
        self.assertEqual(len({row['state_id'] for row in chosen}), 12)
        changed = deepcopy(fixed)
        changed['fixed_prompts'][0]['alias_input_ids_sha256'] = '0' * 64
        with self.assertRaises(PermissionError):
            expected_preparation(source, changed, Tokenizer(),
                                 __import__('src.composite_tool_alias_probe', fromlist=['x']))

    def test_forty_eight_raw_records_are_durable_before_next_call(self):
        from src import composite_tool_alias_probe as probe
        chosen = expected_preparation(rows(), preparation(rows()), Tokenizer(), probe)
        stored = {}
        actor = Actor()
        order = []

        def persist(path, value):
            self.assertNotIn(path, stored)
            stored[path] = deepcopy(value)
            order.append(('persist', path))

        def generate(_actor, ids, on):
            self.assertEqual(order[-1][0], 'persist')
            self.assertTrue(order[-1][1].endswith('/PRECALL.json'))
            entry = order[-1][1].split('/')[2]
            raw = f'<tool_call>\n<function={entry}>\n</function>\n</tool_call>'
            order.append(('generate', entry))
            return {'input_token_ids': ids, 'generated_token_ids': [42, 1],
                    'raw_text': raw, 'finished_with_eos': True,
                    'adapter_enabled': on}

        summary = collect_first_calls(actor, chosen, persist, probe,
                                      lambda _actor: {'checkpoint_sha256': 'same'},
                                      'a' * 64, generate=generate)
        self.assertEqual(summary['completed_durable_responses'], 48)
        self.assertEqual(summary['valid_entry_calls'],
                         {'catalog': {'base': 12, 'always_on': 12},
                          'list': {'base': 12, 'always_on': 12}})
        self.assertEqual(len([name for name in stored if name.endswith('/NATIVE.json')]), 48)
        self.assertEqual(actor.checked, 1)
        self.assertFalse(summary['response_count_unknown'])

    def test_generation_failure_records_unknown_count_and_preserves_partial_raw(self):
        from src import composite_tool_alias_probe as probe
        chosen = expected_preparation(rows(), preparation(rows()), Tokenizer(), probe)
        stored = {}
        attempts = 0

        def generate(_actor, ids, on):
            nonlocal attempts
            attempts += 1
            if attempts == 3:
                raise RuntimeError('generation interrupted')
            return {'input_token_ids': ids, 'generated_token_ids': [1],
                    'raw_text': '<tool_call><function=c></function></tool_call>',
                    'finished_with_eos': True, 'adapter_enabled': on}

        with self.assertRaises(RuntimeError):
            collect_first_calls(Actor(), chosen, lambda p, v: stored.setdefault(p, v),
                                probe, lambda _actor: {'checkpoint_sha256': 'same'},
                                'a' * 64, generate=generate)
        self.assertEqual(len([p for p in stored if p.endswith('/NATIVE.json')]), 2)
        self.assertEqual(len([p for p in stored if p.endswith('/PRECALL.json')]), 3)
        self.assertEqual(stored['FAILURE.json']['attempted_lower_bound'], 3)
        self.assertTrue(stored['FAILURE.json']['response_count_unknown'])
        self.assertEqual(stored['CHECKPOINT_LIVE_AFTER.json']['checkpoint_sha256'], 'same')


if __name__ == '__main__':
    unittest.main()
