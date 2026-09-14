"""Synthetic model-free tests for the prompted input adapter.

Fake encoder/decoder callbacks only: no real tokenizer, provider, model,
dataset, cache, saved result or HOLDOUT material is read or exercised.
"""

from copy import deepcopy
import ast
import inspect
import json
import re
import unittest

import native_capture_contract as contract
import prompted_input_adapter_v1 as adapter
from test_native_capture_contract import make_case


class FakeTokenizer:
    def __init__(self, newline_id=contract.LAST_SHARED_ID, extra_ba_token=False):
        self.to_id = {"\n": newline_id, "A": 32, "B": 33}
        self.to_text = {newline_id: "\n", 32: "A", 33: "B", 199: ""}
        self.calls = []
        self.extra_ba_token = extra_ba_token

    def encode(self, text, *, add_special_tokens):
        self.calls.append(("encode", text, add_special_tokens))
        assert add_special_tokens is False
        parts = re.findall(r"\n|[AB](?=\))|\)|[^\s)]+|[ \t]+", text)
        assert "".join(parts) == text
        result = []
        for part in parts:
            if part not in self.to_id:
                number = 1000 + len(self.to_id)
                self.to_id[part] = number
                self.to_text[number] = part
            result.append(self.to_id[part])
        if (
            self.extra_ba_token
            and "\nB)" in text
            and text.index("\nB)") < text.index("\nA)")
        ):
            result.append(199)
        return result

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        self.calls.append(
            (
                "decode",
                list(ids),
                skip_special_tokens,
                clean_up_tokenization_spaces,
            )
        )
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self.to_text[i] for i in ids)


def prepare(case=None, tokenizer=None, **options):
    tokenizer = tokenizer or FakeTokenizer()
    kwargs = dict(
        encode=tokenizer.encode,
        decode=tokenizer.decode,
        label_token_ids={"A": 32, "B": 33},
        expected_identity_sha256="a" * 64,
        observed_identity_sha256="a" * 64,
    )
    kwargs.update(options)
    return adapter.prepare_case_inputs(case or make_case(), **kwargs)


def prompt_text(order):
    return adapter.render_case_views(make_case())["views"][order]["prompt_text"]


def decode_view(tokenizer, ids):
    return tokenizer.decode(
        list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


class RenderTests(unittest.TestCase):
    def test_query_after_context_and_before_both_retained_records(self):
        case = make_case()
        rendered = adapter.render_case_views(case)
        context = case["context_before_options"]
        for key in ("AB", "BA"):
            prompt = rendered["views"][key]["prompt_text"]
            self.assertTrue(prompt.startswith(context + "\n" + adapter.FIXED_QUERY + "\n"))
            self.assertEqual(prompt.count(adapter.FIXED_QUERY), 1)
            for option in case["options"]:
                self.assertGreater(prompt.index(option), prompt.index(adapter.FIXED_QUERY))
        self.assertEqual(rendered["query"], adapter.FIXED_QUERY)
        self.assertEqual(
            rendered["query_sha256"],
            contract.sha256_hex(adapter.FIXED_QUERY.encode("utf-8")),
        )
        self.assertEqual(
            rendered["original_context_sha256"],
            contract.sha256_hex(context.encode("utf-8")),
        )
        self.assertEqual(rendered["schema"], adapter.PROMPTED_RENDER_SCHEMA)
        self.assertEqual(rendered["condition"], adapter.PROMPTED_CONDITION)

    def test_schema_ids_and_condition_distinct_from_unprompted(self):
        self.assertNotEqual(adapter.PROMPTED_RENDER_SCHEMA, contract.RENDER_SCHEMA)
        self.assertNotEqual(adapter.PROMPTED_BINDING_SCHEMA, contract.BINDING_SCHEMA)
        self.assertNotEqual(adapter.PROMPTED_INFERENCE_SCHEMA, contract.INFERENCE_SCHEMA)
        self.assertNotEqual(adapter.PROMPTED_FEATURE_CONTRACT, contract.FEATURE_CONTRACT)
        self.assertNotEqual(adapter.PROMPTED_CONDITION, adapter.PROMPTED_RENDER_SCHEMA)
        self.assertFalse(hasattr(contract, "PROMPTED_CONDITION"))


class RoundTripAndBindingTests(unittest.TestCase):
    def test_exact_roundtrip_with_query_and_query_inside_shared_prefix(self):
        fake = FakeTokenizer()
        result = prepare(tokenizer=fake)
        shared = result["binding"]["shared_prefix_length"]
        prefix_text = make_case()["context_before_options"] + "\n" + adapter.FIXED_QUERY + "\n"
        for key in ("AB", "BA"):
            ids = result["input_ids"][key]
            self.assertEqual(decode_view(fake, ids), prompt_text(key))
            self.assertIn(adapter.FIXED_QUERY, decode_view(fake, ids))
            self.assertEqual(decode_view(fake, ids[:shared]), prefix_text)
        self.assertEqual(
            result["binding"]["bindings"]["AB"]["shared_prefix_ids_sha256"],
            result["binding"]["bindings"]["BA"]["shared_prefix_ids_sha256"],
        )

    def test_first_divergence_is_the_label_boundary(self):
        result = prepare()
        shared = result["binding"]["shared_prefix_length"]
        for key, label in (("AB", "A"), ("BA", "B")):
            self.assertEqual(result["input_ids"][key][shared], 32 if label == "A" else 33)
            self.assertEqual(
                result["binding"]["bindings"][key]["first_option_label_id"],
                32 if label == "A" else 33,
            )
            self.assertEqual(result["binding"]["bindings"][key]["readout_index"], shared - 1)

    def test_variable_last_shared_token_is_recorded_not_assumed(self):
        result = prepare(tokenizer=FakeTokenizer(newline_id=205))
        index = result["binding"]["readout_index"]
        self.assertEqual(result["binding"]["last_shared_token_id"], 205)
        self.assertEqual(result["provenance"]["last_shared_token_id"], 205)
        self.assertNotEqual(result["binding"]["last_shared_token_id"], contract.LAST_SHARED_ID)
        for key in ("AB", "BA"):
            self.assertEqual(result["input_ids"][key][index], 205)
            self.assertEqual(
                result["binding"]["bindings"][key]["last_shared_token_id"], 205
            )

    def test_variable_view_lengths_with_extra_ba_token(self):
        result = prepare(tokenizer=FakeTokenizer(extra_ba_token=True))
        self.assertNotEqual(
            len(result["input_ids"]["AB"]), len(result["input_ids"]["BA"])
        )
        self.assertEqual(result["provenance"]["truncated"], False)

    def test_provenance_records_context_query_and_all_hashes(self):
        result = prepare()
        provenance = result["provenance"]
        self.assertEqual(provenance["schema"], adapter.PROMPTED_PROVENANCE_SCHEMA)
        self.assertEqual(provenance["condition"], adapter.PROMPTED_CONDITION)
        self.assertEqual(provenance["implementation_job_id"], adapter.IMPLEMENTATION_JOB_ID)
        self.assertEqual(provenance["original_context_sha256"], contract.sha256_hex(make_case()["context_before_options"].encode("utf-8")))
        self.assertEqual(provenance["query_sha256"], adapter.QUERY_SHA256)
        self.assertEqual(provenance["prefix_hash"], result["binding"]["bindings"]["AB"]["shared_prefix_ids_sha256"])
        for group in ("prompt_hashes", "input_hashes"):
            self.assertEqual(set(provenance[group]), {"AB", "BA"})
            for digest in provenance[group].values():
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertTrue(provenance["exact_decode_and_reencode"])
        self.assertTrue(provenance["max_tokens_is_provisional_sanity_ceiling"])

    def test_tampered_binding_rejected(self):
        result = prepare()
        rendered = adapter.render_case_views(make_case())
        for mutate in (
            lambda value: value.update({"last_shared_token_id": 999}),
            lambda value: value.update({"readout_index": 0}),
            lambda value: value["bindings"]["AB"].update({"prompt_sha256": "0" * 64}),
        ):
            tampered = deepcopy(result["binding"])
            mutate(tampered)
            with self.assertRaisesRegex(adapter.PromptedInputError, "BINDING_MISMATCH"):
                adapter.validate_binding(rendered, result["input_ids"], tampered)


class FailureTests(unittest.TestCase):
    def test_wrong_decode_and_unstable_reencode_fail(self):
        with self.assertRaisesRegex(adapter.PromptedInputError, "DECODE_MISMATCH"):
            prepare(decode=lambda ids, **kwargs: "modified prompt")
        fake = FakeTokenizer()
        calls = []

        def unstable(text, **kwargs):
            ids = fake.encode(text, **kwargs)
            calls.append(True)
            if len(calls) % 2 == 0:
                ids.append(199)
            return ids

        with self.assertRaisesRegex(adapter.PromptedInputError, "REENCODE_MISMATCH"):
            prepare(tokenizer=fake, encode=unstable)

    def test_wrong_label_boundary_fails(self):
        with self.assertRaisesRegex(adapter.PromptedInputError, "AB_LABEL_BOUNDARY"):
            prepare(label_token_ids={"A": 33, "B": 32})

    def test_token_limit_rejects_without_truncation(self):
        result = prepare()
        exact = max(len(value) for value in result["input_ids"].values())
        at_limit = prepare(tokenizer=FakeTokenizer(), max_tokens=exact)
        self.assertEqual(at_limit["provenance"]["max_tokens"], exact)
        self.assertFalse(at_limit["provenance"]["truncated"])
        with self.assertRaisesRegex(adapter.PromptedInputError, "TOKEN_LIMIT"):
            prepare(tokenizer=FakeTokenizer(), max_tokens=exact - 1)
        for invalid in (0, -1, True, 1.5, "512", None):
            with self.assertRaisesRegex(adapter.PromptedInputError, "MAX_TOKENS"):
                prepare(max_tokens=invalid)

    def test_unknown_split_and_label_rejected_before_encode(self):
        # The unknown split is not an admitted development split, so the gate
        # fires first; the frozen contract independently rejects it too.
        with self.assertRaisesRegex(contract.NativeCaptureContractError, "SPLIT"):
            contract.validate_case(make_case(split="RELEASE"))
        fake = FakeTokenizer()
        with self.assertRaisesRegex(adapter.PromptedInputError, "DEVELOPMENT_ONLY"):
            prepare(make_case(split="RELEASE"), fake)
        self.assertEqual(fake.calls, [])
        fake = FakeTokenizer()
        with self.assertRaisesRegex(contract.NativeCaptureContractError, "CLASS_LABEL"):
            prepare(make_case(class_label="UNKNOWN"), fake)
        self.assertEqual(fake.calls, [])

    def test_holdout_and_other_splits_never_encoded(self):
        fake = FakeTokenizer()
        case = make_case(
            split="HOLDOUT",
            development_fold=None,
            status="ADMITTED_HOLDOUT_TEXT_ONLY",
        )
        with self.assertRaisesRegex(adapter.PromptedInputError, "DEVELOPMENT_ONLY"):
            prepare(case, fake)
        self.assertEqual(fake.calls, [])

    def test_identity_pin_and_callback_guards(self):
        for invalid in (None, "abc", "A" * 64, "a" * 63, "a" * 65):
            with self.assertRaisesRegex(adapter.PromptedInputError, "IDENTITY_HASH"):
                prepare(expected_identity_sha256=invalid)
        with self.assertRaisesRegex(adapter.PromptedInputError, "CALLBACKS"):
            prepare(encode=None)
        with self.assertRaisesRegex(adapter.PromptedInputError, "IDENTITY_MISMATCH"):
            prepare(observed_identity_sha256="b" * 64)

    def test_bad_encoder_ids_fail(self):
        for invalid in ([True], [contract.MAX_VOCAB_ID], [-1], [], "wrong type"):
            with self.assertRaisesRegex(adapter.PromptedInputError, "TOKEN_IDS"):
                prepare(encode=lambda text, **kwargs: invalid)
        with self.assertRaisesRegex(adapter.PromptedInputError, "TOKEN_LIMIT"):
            prepare(encode=lambda text, **kwargs: [1] * (adapter.PROMPTED_MAX_TOKENS + 1))

    def test_callback_errors_are_structured(self):
        def broken(*args, **kwargs):
            raise RuntimeError("internal callback details")

        with self.assertRaisesRegex(adapter.PromptedInputError, "ENCODER_ERROR"):
            prepare(encode=broken)
        with self.assertRaisesRegex(adapter.PromptedInputError, "DECODER_ERROR"):
            prepare(decode=broken)


class SafetyTests(unittest.TestCase):
    def test_model_facing_payload_has_no_case_metadata(self):
        result = prepare()
        payload = result["inference_input"]
        self.assertEqual(set(payload), {"schema", "condition", "views", "feature_contract"})
        for view in payload["views"]:
            self.assertEqual(
                set(view),
                {
                    "order",
                    "prompt_text",
                    "readout_index",
                    "final_input_index",
                    "input_ids_sha256",
                    "shared_prefix_ids_sha256",
                },
            )
        dumped = json.dumps(payload)
        self.assertIn(adapter.FIXED_QUERY, dumped)
        for forbidden in (
            "T01",
            "T01_S99",
            "SELF",
            "group_id",
            "case_id",
            "class_label",
            "mechanism_ancestry",
        ):
            self.assertNotIn(forbidden, dumped)

    def test_case_and_callback_lists_not_mutated(self):
        case = make_case()
        original = deepcopy(case)
        fake = FakeTokenizer()
        supplied = []

        def encode(text, **kwargs):
            ids = fake.encode(text, **kwargs)
            supplied.append((ids, list(ids)))
            return ids

        def decode(ids, **kwargs):
            decoded = fake.decode(ids, **kwargs)
            ids.clear()
            return decoded

        prepare(case, fake, encode=encode, decode=decode)
        self.assertEqual(case, original)
        for ids, before in supplied:
            self.assertEqual(ids, before)

    def test_no_runtime_or_io_imports(self):
        tree = ast.parse(inspect.getsource(adapter))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                roots.add(node.module.split(".")[0])
        self.assertEqual(
            roots,
            {"copy", "hashlib", "re", "struct", "native_capture_contract"},
        )


if __name__ == "__main__":
    unittest.main()
