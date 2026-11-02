import hashlib
import unittest

from src.action_serialization import (
    SERIALIZATION_CONTRACT_VERSION,
    parse_registered_action,
    serialization_contract,
)


class ActionSerializationTests(unittest.TestCase):
    def test_bare_and_one_strict_lowercase_json_fence_pass(self):
        allowed = ["N01", "N02"]
        bare = '{"action":"N01"}'
        parsed = parse_registered_action(bare, allowed)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["action"], "N01")
        self.assertFalse(parsed["normalization_used"])
        self.assertEqual(parsed["serialization_form"], "bare_json_object")
        self.assertEqual(parsed["raw_response_sha256"], hashlib.sha256(
            bare.encode()).hexdigest())

        fenced = ' \n```json\n{"action":"N02"}\n```\t\n'
        parsed = parse_registered_action(fenced, allowed)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["action"], "N02")
        self.assertTrue(parsed["normalization_used"])
        self.assertEqual(parsed["serialization_form"],
                         "single_lowercase_json_fence")
        self.assertEqual(parsed["contract_version"],
                         SERIALIZATION_CONTRACT_VERSION)

    def test_prose_multiple_or_wrong_fences_and_trailing_content_fail(self):
        allowed = ["N01"]
        rejected = (
            'prefix {"action":"N01"}',
            '{"action":"N01"} suffix',
            '```json\n{"action":"N01"}\n``` trailing',
            '```JSON\n{"action":"N01"}\n```',
            '```\n{"action":"N01"}\n```',
            '```json\n```json\n{"action":"N01"}\n```\n```',
            '```json\n{"action":"N01"}\n```\n```json\n{"action":"N01"}\n```',
        )
        for raw in rejected:
            with self.subTest(raw=raw):
                self.assertFalse(parse_registered_action(raw, allowed)["valid"])

    def test_malformed_duplicate_extra_missing_and_unregistered_fail(self):
        allowed = ["N01"]
        rejected = (
            '',
            '{"action":"N01"',
            '{}',
            '{"action":"N01","why":"x"}',
            '{"action":"N01","action":"N01"}',
            '{"action":"N02"}',
            '{"action":1}',
            '[{"action":"N01"}]',
            '```json\n{"action":"N01"\n```',
        )
        for raw in rejected:
            with self.subTest(raw=raw):
                self.assertFalse(parse_registered_action(raw, allowed)["valid"])

    def test_contract_explicitly_forbids_repair_extraction_and_coercion(self):
        contract = serialization_contract()
        self.assertEqual(contract["version"], SERIALIZATION_CONTRACT_VERSION)
        self.assertFalse(contract["repair"])
        self.assertFalse(contract["prose_extraction"])
        self.assertFalse(contract["coercion"])
        self.assertEqual(contract["required_object_keys"], ["action"])


if __name__ == "__main__":
    unittest.main()
