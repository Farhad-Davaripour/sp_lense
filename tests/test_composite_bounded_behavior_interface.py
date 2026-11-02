"""Public string/schema checks; no tokenizer/model/scientific-case execution."""
import json
import unittest

from src import composite_bounded_behavior_interface as interface
from src.composite_bounded_behavior_world import tools as world_tools
from src.composite_generalized_interface import tools as historical_tools


class BoundedNativeInterfaceTests(unittest.TestCase):
    def test_new_interface_matches_world_and_preserves_original(self):
        self.assertEqual(interface.tools(), world_tools())
        self.assertEqual(set(interface.argument_schemas()),
                         {value['function']['name'] for value in interface.tools()})
        self.assertIn('list', interface.argument_schemas())
        self.assertNotIn('catalog', json.dumps(interface.tools()))
        self.assertIn('catalog', {value['function']['name'] for value in historical_tools()})

    def test_roundtrip_list_and_reject_undeclared_entry_names(self):
        action = {'tool': 'list', 'arguments': {}}
        self.assertEqual(interface.parse(interface.action_text(action)), [action])
        for name in ['catalog', 'c']:
            with self.subTest(name=name):
                with self.assertRaises((ValueError, KeyError)):
                    interface.action_text({'tool': name, 'arguments': {}})
                with self.assertRaises((ValueError, KeyError)):
                    interface.parse('<tool_call>\n<function=' + name
                                    + '>\n</function>\n</tool_call>')

    def test_tokenizer_receives_actual_list_schema_without_thinking(self):
        class PublicRecordingTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.kwargs = messages, kwargs
                return {'input_ids': [[1, 2, 3]]}
        tokenizer = PublicRecordingTokenizer()
        self.assertEqual(interface.input_ids(tokenizer, 'public unit text'), [1, 2, 3])
        self.assertEqual(tokenizer.kwargs['tools'], interface.tools())
        self.assertFalse(tokenizer.kwargs['enable_thinking'])
        self.assertEqual(tokenizer.messages[1]['content'], 'public unit text')


if __name__ == '__main__':
    unittest.main()
