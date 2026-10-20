import unittest
from src.discovery_transport import response_text,parse_response_action
from src.discovery_sandbox import parse_action


class ContentTests(unittest.TestCase):
    def test_native_text_and_reasoning_blocks_extract_the_same_action(self):
        text='{"tool":"catalog","arguments":{}}'
        message={'content':[{'type':'reasoning','summary':[{'text':'not an action'}]},{'type':'text','text':text}]}
        self.assertEqual(response_text(message),text)
        self.assertEqual(parse_action(response_text(message)),parse_action(response_text({'content':text})))

    def test_reasoning_only_and_unrecognized_blocks_do_not_become_actions(self):
        for content in [None,123,[{'type':'reasoning','text':'{"tool":"stop","arguments":{}}'}],[{'type':'image','text':'not output'}]]:
            self.assertEqual(response_text({'content':content}),'')

    def test_single_enclosing_fence_is_accepted_but_other_text_is_not(self):
        text='{"tool":"catalog","arguments":{}}'
        self.assertEqual(parse_response_action('```json\n'+text+'\n```'),parse_action(text))
        for value in ['Reasoning: '+text,'```json\n'+text+'\n```\nAnother action','```json\n'+text+'\n```\n```json\n'+text+'\n```']:
            with self.assertRaises(ValueError):parse_response_action(value)


if __name__=='__main__':unittest.main()
