"""Public model-free checks for the separate first-call alias diagnostic."""
from copy import deepcopy
import hashlib
import json
import unittest

from src.composite_generalized_interface import input_ids as registered_input_ids
from src.composite_tool_alias_probe import (classify_first_entry_call,
                                            paired_prompt_token_ids,
                                            paired_tool_specs,
                                            select_first_catalog_rows)


def rows():
    result = []
    for group in range(12):
        for case in range(9):
            entry = {'split': 'training', 'training_eligible': True, 'task_only': True,
                     'terminal': False, 'group_id': f'group-{group:02}',
                     'case_id': f'case-{group:02}-{case:02}',
                     'state_id': f'state-{group:02}-{case:02}-0',
                     'text': f'Analyze ordinary records for case {group:02}-{case:02}.',
                     'teacher_action': {'tool': 'catalog', 'arguments': {}}}
            result.append(entry)
            later = deepcopy(entry)
            later['state_id'] = f'state-{group:02}-{case:02}-1'
            later['teacher_action'] = {'tool': 'read', 'arguments': {'identifier': 'r-public'}}
            result.append(later)
    return result


class Tokenizer:
    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and not enable_thinking
        encoded = json.dumps({'messages': messages, 'tools': tools}, sort_keys=True).encode()
        return list(hashlib.sha256(encoded).digest())


class AliasProbeTests(unittest.TestCase):
    def test_selects_one_fixed_catalog_case_per_twelve_training_groups(self):
        original = rows()
        expected = [f'case-{group:02}-00' for group in range(12)]
        self.assertEqual([row['case_id'] for row in select_first_catalog_rows(original)], expected)
        self.assertEqual([row['case_id'] for row in select_first_catalog_rows(list(reversed(original)))], expected)
        chosen = select_first_catalog_rows(original)
        chosen[0]['text'] = 'changed locally'
        self.assertNotEqual(original[0]['text'], chosen[0]['text'])

    def test_rejects_incomplete_duplicate_or_nontraining_rows(self):
        source = rows()
        with self.assertRaises(ValueError):
            select_first_catalog_rows(source[:-18])
        bad = deepcopy(source)
        bad[0]['state_id'] = bad[1]['state_id']
        with self.assertRaises(ValueError):
            select_first_catalog_rows(bad)
        bad = deepcopy(source)
        bad[0]['split'] = 'development'
        with self.assertRaises(PermissionError):
            select_first_catalog_rows(bad)
        bad = deepcopy(source)
        bad[0]['teacher_action'] = {'tool': 'c', 'arguments': {}}
        with self.assertRaises(ValueError):
            select_first_catalog_rows(bad)

    def test_tool_lists_differ_only_in_first_name(self):
        specs = paired_tool_specs()
        original, alias = specs['catalog'], specs['list']
        self.assertEqual(original[0]['function']['name'], 'catalog')
        self.assertEqual(alias[0]['function']['name'], 'list')
        normalized = deepcopy(alias)
        normalized[0]['function']['name'] = 'catalog'
        self.assertEqual(normalized, original)
        self.assertEqual([x['function']['name'] for x in original[1:]],
                         [x['function']['name'] for x in alias[1:]])

    def test_prompt_pair_matches_registered_original_and_saved_ids(self):
        tokenizer = Tokenizer()
        text = rows()[0]['text']
        saved = registered_input_ids(tokenizer, text)
        pair = paired_prompt_token_ids(text, tokenizer, saved)
        self.assertEqual(pair['catalog'], saved)
        self.assertNotEqual(pair['catalog'], pair['list'])
        with self.assertRaises(ValueError):
            paired_prompt_token_ids(text, tokenizer, saved + [1])

    def test_raw_first_call_is_never_repaired_or_executed(self):
        catalog = '<tool_call>\n<function=catalog>\n</function>\n</tool_call>'
        alias = catalog.replace('catalog', 'list')
        short = catalog.replace('catalog', 'c')
        self.assertTrue(classify_first_entry_call(catalog, 'catalog', True)['valid_entry_call'])
        self.assertTrue(classify_first_entry_call(alias, 'list', True)['valid_entry_call'])
        self.assertFalse(classify_first_entry_call(catalog, 'list', True)['valid_entry_call'])
        self.assertFalse(classify_first_entry_call(alias, 'catalog', True)['valid_entry_call'])
        result = classify_first_entry_call(short, 'catalog', True)
        self.assertEqual(result['raw_function_name'], 'c')
        self.assertTrue(result['native_single_no_argument_call'])
        self.assertFalse(result['declared_function'])
        self.assertFalse(result['valid_entry_call'])
        self.assertFalse(classify_first_entry_call(catalog, 'catalog', False)['valid_entry_call'])
        self.assertFalse(classify_first_entry_call(catalog + alias, 'catalog', True)['valid_entry_call'])
        self.assertFalse(classify_first_entry_call('<tool_call><function=catalog><parameter=x>1</parameter></function></tool_call>', 'catalog', True)['valid_entry_call'])
        self.assertFalse(classify_first_entry_call('<think>unfinished ' + alias, 'list', True)['valid_entry_call'])
        self.assertTrue(classify_first_entry_call('<think>complete</think>' + alias, 'list', True)['valid_entry_call'])
        self.assertTrue(classify_first_entry_call('I will inspect the records.\n' + catalog, 'catalog', True)['valid_entry_call'])


if __name__ == '__main__':
    unittest.main()
