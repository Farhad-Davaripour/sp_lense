import unittest
from src.discovery_transport import response_text
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


if __name__=='__main__':unittest.main()
