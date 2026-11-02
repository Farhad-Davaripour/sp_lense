import unittest

from src.thinking_response_transport import parse_thinking_response, parse_tokenized_thinking


class ThinkingResponseTransportTests(unittest.TestCase):
    def test_prefilled_open_and_tool_arguments(self):
        result = parse_thinking_response('plan</think>{"action":"submit","value":7}', 12, natural_completion=True)
        self.assertTrue(result.dispatchable)
        self.assertEqual(result.reasoning, "plan")
        self.assertEqual(result.command, {"action": "submit", "value": 7})

    def test_observed_single_json_fence(self):
        text = '<think>plan</think>\n```json\n{"action":"example","n":1}\n```'
        result = parse_thinking_response(text, 20, natural_completion=True)
        self.assertTrue(result.dispatchable)
        self.assertEqual(result.raw_suffix, text)
        self.assertEqual(result.raw_final_text, '\n```json\n{"action":"example","n":1}\n```')
        self.assertEqual(result.final_text, '{"action":"example","n":1}')
        self.assertEqual(result.command["n"], 1)

    def test_no_prose_multiple_fences_duplicate_keys_or_nonfinite(self):
        for tail in ('text {"action":"a"}', '```json\n{"action":"a"}\n```\n```json\n{"action":"b"}\n```',
                     '{"action":"a","action":"b"}', '{"action":"a","v":NaN}'):
            with self.subTest(tail=tail):
                self.assertFalse(parse_thinking_response("x</think>" + tail, 20, natural_completion=True).dispatchable)

    def test_no_dispatch_without_closed_reasoning_known_usage_and_completion(self):
        self.assertFalse(parse_thinking_response("unfinished", 5, natural_completion=True).dispatchable)
        self.assertFalse(parse_thinking_response('x</think>{"action":"a"}', 5, natural_completion=False).dispatchable)
        self.assertFalse(parse_thinking_response('x</think>{"action":"a"}', None, natural_completion=True).dispatchable)
        self.assertFalse(parse_thinking_response('x</think>{"action":"a"}', 5, natural_completion=None).dispatchable)

    def test_strip_only_terminal_token_ids_and_preserve_raw(self):
        mapping = {1: "thinking", 2: "</think>", 3: '{"action":"read"}',
                   90: "<|im_end|>", 91: "<|endoftext|>"}
        decode = lambda ids: "".join(mapping[x] for x in ids)
        result = parse_tokenized_thinking([1, 2, 3, 90, 91], eos_token_ids=[91],
                                          terminal_token_ids=[90, 91], decode=decode)
        self.assertTrue(result.dispatchable)
        self.assertEqual(result.raw_suffix, decode([1, 2, 3, 90, 91]))
        self.assertEqual(result.generated_tokens, 5)

    def test_unknown_or_truncated_completion_never_dispatches(self):
        mapping = {1: "thinking</think>", 2: '{"action":"read"}', 91: "<eos>"}
        result = parse_tokenized_thinking([1, 2], eos_token_ids=[91],
                                          terminal_token_ids=[91], decode=lambda ids: "".join(mapping[x] for x in ids))
        self.assertEqual(result.status, "generation_incomplete")
        self.assertFalse(result.dispatchable)

    def test_newline_between_known_terminal_tokens_is_removed(self):
        mapping = {1: 'plan</think>{"action":"read"}', 20: "\n",
                   90: "<|im_end|>", 91: "<|endoftext|>"}
        ids = [1, 90, 20, 91]
        decode = lambda seq: "".join(mapping[x] for x in seq)
        result = parse_tokenized_thinking(ids, eos_token_ids=[91],
                                          terminal_token_ids=[90, 91], decode=decode)
        self.assertTrue(result.dispatchable)
        self.assertEqual(result.raw_suffix, decode(ids))
        self.assertEqual(result.generated_tokens, len(ids))

    def test_missing_eos_configuration_preserves_raw_and_does_not_dispatch(self):
        result = parse_tokenized_thinking([1], eos_token_ids=[], terminal_token_ids=[],
                                          decode=lambda _: 'reason</think>{"action":"read"}')
        self.assertEqual(result.status, "completion_unknown")
        self.assertEqual(result.raw_suffix, 'reason</think>{"action":"read"}')
        self.assertFalse(result.dispatchable)


if __name__ == "__main__":
    unittest.main()
