"""Exact context limits and pinned CPU rendering; no model/weights/network calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

from src import full_history_context as context


class FullHistoryContextTests(unittest.TestCase):
    def messages(self):
        return [{"role": "system", "content": "Choose exactly one action."},
                {"role": "user", "content": "Current visible observation."}]

    def test_exact_122880_boundary_and_122881_refusal_without_mutation(self):
        messages = self.messages(); original = copy.deepcopy(messages)
        result = context.validate_context_budget(messages, token_counter=lambda _: 122880)
        self.assertEqual(result, {"prompt_tokens": 122880, "output_reservation_tokens": 8192,
                                  "context_total_tokens": 131072})
        with self.assertRaises(context.ContextBudgetExceeded):
            context.validate_context_budget(messages, token_counter=lambda _: 122881)
        self.assertEqual(messages, original)

    def test_no_character_threshold_or_silent_truncation(self):
        messages = [{"role": "user", "content": "x" * 1000000}]
        seen = []
        def exact_counter(value):
            seen.append(len(value[0]["content"]))
            return 122880
        self.assertEqual(context.validate_context_budget(messages, exact_counter)["prompt_tokens"], 122880)
        self.assertEqual(seen, [1000000])
        self.assertEqual(len(messages[0]["content"]), 1000000)
        for invalid in (True, -1, 1.0):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                context.validate_context_budget(messages, lambda _: invalid)

    def test_actual_pinned_template_exact_system_user_render_and_provenance(self):
        rendered = context.render_formatted_prompt(self.messages())
        expected = ("<|im_start|>system\nReasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."
                    "\n\nChoose exactly one action.<|im_end|>\n"
                    "<|im_start|>user\nCurrent visible observation.<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n")
        self.assertEqual(rendered, expected)
        count = context.count_formatted_prompt_tokens(self.messages())
        self.assertGreater(count, 0)
        self.assertEqual(context.validate_context_budget(self.messages())["prompt_tokens"], count)
        provenance = context.tokenizer_provenance()
        self.assertEqual(provenance["files_sha256"], context.PINNED_FILES)
        self.assertEqual((provenance["tokenizers_version"], provenance["jinja2_version"]), ("0.23.3", "3.1.6"))
        self.assertFalse(provenance["add_special_tokens"])

    def test_actual_parser_template_witness_preserves_reasoning_in_user_content(self):
        path = Path("C:/Users/farha/repos/sp_lense4/research4/.local/staging/continuity_transfer_endpoint_held_20261115/candidate_source_v2/local_last8_endpoint_contract.py")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "a7a7ca7fd5114da28b93bfb87e2e3ef370578f8df5320b07b0d84f85e8ba6920")
        spec = importlib.util.spec_from_file_location("context_actual_parser_witness", path)
        parser = importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
        raw_reasoning = "RAW_REASONING_SENTINEL\nI inspect the visible resource state before choosing."
        transcript = "Prior assistant reasoning:\n" + raw_reasoning + '\nPrior action JSON:\n{"action":"list_available_tools","arguments":{}}\nPrior simulator feedback:\n{"status":"synthetic_only"}'
        messages = self.messages(); messages[-1]["content"] = transcript
        body = {"model": parser.BASE, "messages": messages, "stream": True, "max_tokens": 8192,
                "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}
        normalized = parser.parse_request(json.dumps(body).encode())["messages"]
        self.assertEqual(normalized, messages)
        rendered = context.render_formatted_prompt(normalized)
        self.assertIn(raw_reasoning, rendered)
        self.assertIn('{"action":"list_available_tools","arguments":{}}', rendered)
        self.assertGreater(context.count_formatted_prompt_tokens(normalized),
                           context.count_formatted_prompt_tokens(self.messages()))

    def test_requires_root_normalized_text_and_rejects_hidden_channels(self):
        for messages in ([], [{"role": "user", "content": []}],
                         [{"role": "assistant", "content": "answer"}],
                         [{"role": "assistant", "content": "answer", "reasoning_content": "not normalized"},
                          {"role": "user", "content": "next"}]):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                context.validate_context_budget(messages, lambda _: 10)


if __name__ == "__main__":
    unittest.main()
