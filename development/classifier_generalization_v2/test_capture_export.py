"""Synthetic in-memory tests for capture_export.

All activation vectors are fabricated 1024-value patterns. Nothing here touches
a tokenizer, model, tensor, hook, cache, network, HOLDOUT data, or real
features. The tests assert serialized-byte integrity and structural join
behavior only; passing them does not authenticate any native capture.
"""

import json
import struct
import unittest

import numpy as np

from capture_export import (
    BLOCK,
    DTYPE,
    MODEL_REVISION,
    PAYLOAD_BYTES,
    POSITION,
    WIDTH,
    CaptureExportError,
    assemble_validation_pairs,
    build_case_views,
    build_view,
    decode_activation,
    decode_view,
    serialize_activation,
    sha256_hex,
)
from pair_join import PairJoinError, assemble_train_pairs

PREFIX = "a" * 64
PREFIX_B = "b" * 64


def vec(seed):
    """Deterministic 1024-value pattern that is exactly representable in f4."""
    base = [((i % 7) - 3) * 0.25 for i in range(WIDTH)]
    base[0] = float(seed)
    return base


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def ab_ba(seed):
    return {"AB": vec(seed), "BA": vec(seed + 1000.0)}


def case(case_id, group_id="g0", split="VALIDATION", class_label="SELF", fold=None):
    record = {"case_id": case_id, "group_id": group_id, "split": split, "class_label": class_label}
    record["development_fold"] = fold
    record["status"] = f"ADMITTED_{split}_TEXT_ONLY"
    return record


def transport(case_record, seed, prefix=PREFIX):
    return build_case_views(case_record, ab_ba(seed), prefix)


def decoded(case_record, seed, prefix=PREFIX):
    return [decode_view(record) for record in transport(case_record, seed, prefix)]


def val_plan(groups=("g0",), labels=("SELF", "OTHER"), fold=None):
    return {
        group: {"split": "VALIDATION", "development_fold": fold,
                "case_counts": {label: 4 for label in labels}}
        for group in groups
    }


class SerializationTests(unittest.TestCase):
    def test_fractional_input_rounds_once_to_float32(self):
        payload = serialize_activation([0.1] * WIDTH)
        self.assertEqual(payload, struct.pack('<f', 0.1) * WIDTH)
        self.assertEqual(decode_activation(payload, sha256_hex(payload)), [f32(0.1)] * WIDTH)

    def test_huge_integer_is_a_structured_overflow(self):
        with self.assertRaises(CaptureExportError) as caught:
            serialize_activation([10**1000] + [0.0] * (WIDTH - 1))
        self.assertEqual(caught.exception.code, 'OVERFLOW')

    def test_exact_width_and_little_endian_float32(self):
        values = vec(0.5)
        payload = serialize_activation(values)
        self.assertEqual(len(payload), 4096)
        self.assertEqual(len(payload), PAYLOAD_BYTES)
        self.assertEqual(struct.unpack("<f", payload[:4])[0], values[0])
        self.assertEqual(struct.unpack("<f", payload[-4:])[0], f32(values[-1]))
        digest = sha256_hex(payload)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(len(decode_activation(payload, digest)), WIDTH)

    def test_round_trip_is_exact_for_representable_values(self):
        values = [((i % 13) - 6) * 0.125 for i in range(WIDTH)]
        payload = serialize_activation(values)
        digest = sha256_hex(payload)
        decoded_values = decode_activation(payload, digest)
        self.assertEqual(decoded_values, list(values))
        self.assertEqual(serialize_activation(decoded_values), payload)
        self.assertEqual(sha256_hex(serialize_activation(decoded_values)), digest)

    def test_input_shape_and_type_rejections(self):
        with self.assertRaises(CaptureExportError) as caught:
            serialize_activation("not a sequence")
        self.assertEqual(caught.exception.code, "VALUES_TYPE")
        with self.assertRaises(CaptureExportError) as caught:
            serialize_activation([0.0] * 1023)
        self.assertEqual(caught.exception.code, "WIDTH")
        for repeated in ([0.0] * 4096, tuple(range(10))):
            with self.assertRaises(CaptureExportError) as caught:
                serialize_activation(repeated)
            self.assertEqual(caught.exception.code, "WIDTH")

    def test_bool_nonfinite_and_overflow_rejections(self):
        cases = [
            (True, "VALUES_TYPE"),
            ("1.0", "VALUES_TYPE"),
            (None, "VALUES_TYPE"),
            (1 + 2j, "VALUES_TYPE"),
            (float("nan"), "NONFINITE"),
            (float("inf"), "NONFINITE"),
            (float("-inf"), "NONFINITE"),
            (1e39, "OVERFLOW"),
            (-1e39, "OVERFLOW"),
            (3.5e38, "OVERFLOW"),
        ]
        for bad, code in cases:
            values = vec(0.0)
            values[17] = bad
            with self.assertRaises(CaptureExportError) as caught:
                serialize_activation(values)
            self.assertEqual(caught.exception.code, code, f"value {bad!r}")

    def test_non_finite_later_position_is_found(self):
        values = vec(0.0)
        values[1023] = float("nan")
        with self.assertRaises(CaptureExportError) as caught:
            serialize_activation(values)
        self.assertEqual(caught.exception.code, "NONFINITE")


class DecodeTests(unittest.TestCase):
    def test_decode_requires_expected_hash(self):
        payload = serialize_activation(vec(1.0))
        digest = sha256_hex(payload)
        with self.assertRaises(CaptureExportError) as caught:
            decode_activation(payload, None)
        self.assertEqual(caught.exception.code, "HASH_FORMAT")
        with self.assertRaises(CaptureExportError) as caught:
            decode_activation(payload, "A" * 64)
        self.assertEqual(caught.exception.code, "HASH_FORMAT")
        with self.assertRaises(CaptureExportError) as caught:
            decode_activation(payload, "0" * 63)
        self.assertEqual(caught.exception.code, "HASH_FORMAT")

    def test_hash_mismatch_is_rejected(self):
        payload = serialize_activation(vec(1.0))
        with self.assertRaises(CaptureExportError) as caught:
            decode_activation(payload, "0" * 64)
        self.assertEqual(caught.exception.code, "HASH_MISMATCH")

    def test_truncated_corrupt_and_wrong_type_payloads(self):
        payload = serialize_activation(vec(1.0))
        digest = sha256_hex(payload)
        for bad in (payload[:-1], payload[:100], b"", payload + b"\x00"):
            with self.assertRaises(CaptureExportError) as caught:
                decode_activation(bad, digest)
            self.assertEqual(caught.exception.code, "PAYLOAD_LENGTH")
        with self.assertRaises(CaptureExportError) as caught:
            decode_activation(bytearray(payload), digest)
        self.assertEqual(caught.exception.code, "PAYLOAD_TYPE")

    def test_corrupt_payload_fails_even_with_recomputed_length(self):
        payload = bytearray(serialize_activation(vec(1.0)))
        payload[2048:2052] = struct.pack("<f", float("nan"))
        corrupted = bytes(payload)
        with self.assertRaises(CaptureExportError) as caught:
            decode_activation(corrupted, sha256_hex(corrupted))
        self.assertEqual(caught.exception.code, "NONFINITE")
        self.assertNotEqual(sha256_hex(corrupted), sha256_hex(serialize_activation(vec(1.0))))

    def test_infinity_payload_is_rejected(self):
        payload = struct.pack("<f", float("inf")) + b"\x00" * (PAYLOAD_BYTES - 4)
        with self.assertRaises(CaptureExportError) as caught:
            decode_activation(payload, sha256_hex(payload))
        self.assertEqual(caught.exception.code, "NONFINITE")


class TransportRecordTests(unittest.TestCase):
    def test_decoded_view_matches_pair_join_contract(self):
        record = decode_view(transport(case("c0"), 2.5)[0])
        self.assertEqual(set(record), {
            "case_id", "order", "model_revision", "block", "position",
            "dtype", "prefix_sha256", "values",
        })
        self.assertEqual(record["case_id"], "c0")
        self.assertEqual(record["order"], ["A", "B"])
        self.assertEqual(record["model_revision"], MODEL_REVISION)
        self.assertEqual(record["block"], BLOCK)
        self.assertEqual(record["position"], POSITION)
        self.assertEqual(record["dtype"], DTYPE)
        self.assertEqual(record["prefix_sha256"], PREFIX)
        self.assertEqual(len(record["values"]), WIDTH)
        json.dumps(record, allow_nan=False)

    def test_build_view_output_is_json_serializable_without_bytes(self):
        record = build_view("c0", ["A", "B"], vec(3.0), PREFIX)
        serializable = dict(record, payload=record["payload"].hex())
        json.dumps(serializable, allow_nan=False)
        self.assertEqual(record["payload_sha256"], sha256_hex(record["payload"]))

    def test_record_metadata_rejections(self):
        record = build_view("c0", ["A", "B"], vec(0.0), PREFIX)
        for key, bad in (("case_id", 7), ("order", ["B", "A", "C"]),
                         ("model_revision", "deadbeef"), ("block", 11),
                         ("position", "final_input"), ("dtype", "float64"),
                         ("width", 512), ("prefix_sha256", "short"),
                         ("payload_sha256", "z" * 64)):
            mutated = dict(record)
            mutated[key] = bad
            with self.assertRaises(CaptureExportError) as caught:
                decode_view(mutated)
            self.assertIn(caught.exception.code, ("RECORD", "REVISION", "ORDER", "WIDTH", "HASH_FORMAT", "HASH_MISMATCH"))
        with self.assertRaises(CaptureExportError) as caught:
            decode_view(dict(record, extra=True))
        self.assertEqual(caught.exception.code, "RECORD")
        with self.assertRaises(CaptureExportError) as caught:
            decode_view(dict(record, payload=record["payload"][:-4]))
        self.assertEqual(caught.exception.code, "PAYLOAD_LENGTH")

    def test_case_view_shape_and_split_rejections(self):
        with self.assertRaises(CaptureExportError) as caught:
            build_case_views(case("c0"), {"AB": vec(0.0)}, PREFIX)
        self.assertEqual(caught.exception.code, "VALUES_TYPE")
        identical = build_case_views(case("c0"), {"AB": vec(0.0), "BA": vec(0.0)}, PREFIX)
        self.assertEqual(identical[0]['payload'], identical[1]['payload'])
        self.assertNotEqual(identical[0]['order'], identical[1]['order'])
        with self.assertRaises(CaptureExportError) as caught:
            build_case_views(case("c0", split="HOLDOUT"), ab_ba(0.0), PREFIX)
        self.assertEqual(caught.exception.code, "SPLIT")
        with self.assertRaises(CaptureExportError) as caught:
            build_case_views(case("c0", split="VAL"), ab_ba(0.0), PREFIX)
        self.assertEqual(caught.exception.code, "SPLIT")
        with self.assertRaises(CaptureExportError) as caught:
            build_case_views(case("c0", split="TRAIN", fold=None), ab_ba(0.0), PREFIX)
        self.assertEqual(caught.exception.code, "FOLD")
        with self.assertRaises(CaptureExportError) as caught:
            build_case_views(case("c0", split="VALIDATION", fold=2), ab_ba(0.0), PREFIX)
        self.assertEqual(caught.exception.code, "FOLD")
        with self.assertRaises(CaptureExportError) as caught:
            build_case_views(case("c0"), ab_ba(0.0), "nothex")
        self.assertEqual(caught.exception.code, "HASH_FORMAT")


class ValidationJoinTests(unittest.TestCase):
    def test_identical_preoption_activations_are_retained(self):
        c = case('same')
        values = vec(4.0)
        views = build_case_views(c, {'AB': values, 'BA': list(values)}, PREFIX)
        joined = assemble_validation_pairs([c], views, val_plan())
        self.assertEqual(joined['x'], [values])

    def test_unadmitted_and_missing_fold_rejected_before_pair_decode(self):
        for update in ({'status': 'DRAFT_UNREVIEWED'}, {'status': None}):
            c = dict(case('pending'), **update)
            with self.assertRaises(CaptureExportError):
                assemble_validation_pairs([c], [None], val_plan())
        c = case('missing')
        del c['development_fold']
        with self.assertRaises(CaptureExportError):
            assemble_validation_pairs([c], [None], val_plan())

    def test_valid_cases_preserve_order_and_pair_exactly_ab_ba(self):
        cases = [case("c1"), case("c0", class_label="OTHER")]
        views = transport(cases[0], 0.0) + transport(cases[1], 500.0)
        out = assemble_validation_pairs(cases, views, val_plan())
        self.assertEqual(out["case_ids"], ["c1", "c0"])
        self.assertEqual(out["labels"], ["SELF", "OTHER"])
        self.assertEqual(out["groups"], ["g0", "g0"])
        self.assertEqual(out["folds"], [None, None])
        self.assertEqual(len(out["x"]), 2)
        self.assertTrue(all(len(row) == WIDTH for row in out["x"]))
        expected = [0.5 * a + 0.5 * b for a, b in zip(vec(0.0), vec(1000.0))]
        np.testing.assert_allclose(out["x"][0], expected, rtol=0, atol=1e-12)
        json.dumps(out, allow_nan=False)

    def test_view_input_order_does_not_change_the_pair(self):
        case_record = case("c0")
        forward = assemble_validation_pairs([case_record], transport(case_record, 0.0), val_plan())
        reversed_input = assemble_validation_pairs(
            [case_record], list(reversed(transport(case_record, 0.0))), val_plan())
        self.assertEqual(forward["x"], reversed_input["x"])

    def test_duplicate_orphan_and_missing_views(self):
        case_record = case("c0")
        with self.assertRaises(CaptureExportError) as caught:
            assemble_validation_pairs([case_record, case("c0")], transport(case_record, 0.0), val_plan())
        self.assertEqual(caught.exception.code, "DUPLICATE_CASE")
        with self.assertRaises(CaptureExportError) as caught:
            assemble_validation_pairs(
                [case_record], transport(case_record, 0.0) + transport(case("ghost"), 1.0), val_plan())
        self.assertEqual(caught.exception.code, "ORPHAN_VIEW")
        with self.assertRaises(CaptureExportError) as caught:
            assemble_validation_pairs([case_record], transport(case_record, 0.0)[:1], val_plan())
        self.assertEqual(caught.exception.code, "PAIR")
        with self.assertRaises(CaptureExportError) as caught:
            assemble_validation_pairs(
                [case_record], transport(case_record, 0.0) + transport(case_record, 0.0)[:1], val_plan())
        self.assertEqual(caught.exception.code, "DUPLICATE_VIEW")

    def test_wrong_role_and_wrong_fold_rejections(self):
        train_case = case("c0", split="TRAIN", fold=1)
        with self.assertRaises(CaptureExportError) as caught:
            assemble_validation_pairs([train_case], transport(train_case, 0.0), val_plan())
        self.assertEqual(caught.exception.code, "SPLIT")
        holdout_case = case("c0", split="HOLDOUT")
        with self.assertRaises(CaptureExportError) as caught:
            assemble_validation_pairs([holdout_case], transport(holdout_case, 0.0), val_plan())
        self.assertEqual(caught.exception.code, "SPLIT")
        typed_case = case("c0", split="VAL")
        with self.assertRaises(CaptureExportError) as caught:
            assemble_validation_pairs([typed_case], transport(typed_case, 0.0), val_plan())
        self.assertEqual(caught.exception.code, "SPLIT")
        nullfold_case = case("c0", split="VALIDATION", fold=None)
        self.assertEqual(assemble_validation_pairs([nullfold_case], transport(nullfold_case, 0.0), val_plan())["folds"], [None])
        folded_case = case("c0", split="VALIDATION", fold=3)
        with self.assertRaises(CaptureExportError) as caught:
            assemble_validation_pairs([folded_case], transport(folded_case, 0.0), val_plan())
        self.assertEqual(caught.exception.code, "FOLD")

    def test_group_plan_role_and_label_rejections(self):
        case_record = case("c0")
        views = transport(case_record, 0.0)
        with self.assertRaises(CaptureExportError) as caught:
            assemble_validation_pairs([case_record], views, {})
        self.assertEqual(caught.exception.code, "PLAN")
        with self.assertRaises(CaptureExportError) as caught:
            assemble_validation_pairs(
                [case_record], views,
                {"g0": {"split": "TRAIN", "development_fold": 0, "case_counts": {"SELF": 1}}})
        self.assertEqual(caught.exception.code, "PLAN")
        with self.assertRaises(CaptureExportError) as caught:
            assemble_validation_pairs(
                [case_record], views,
                {"g0": {"split": "VALIDATION", "development_fold": None, "case_counts": {"OTHER": 1}}})
        self.assertEqual(caught.exception.code, "PLAN")

    def test_prefix_disagreement_is_rejected(self):
        case_record = case("c0")
        ab = build_view("c0", ["A", "B"], vec(0.0), PREFIX)
        ba = build_view("c0", ["B", "A"], vec(1000.0), PREFIX_B)
        with self.assertRaises(CaptureExportError) as caught:
            assemble_validation_pairs([case_record], [ab, ba], val_plan())
        self.assertEqual(caught.exception.code, "PREFIX_MISMATCH")


class JoinerInterfaceTests(unittest.TestCase):
    def test_decoded_validation_views_do_not_enter_train_joiner(self):
        case_record = case("c0")
        views = decoded(case_record, 0.0)
        plan = {"g0": {"split": "TRAIN", "development_fold": 0, "case_counts": {"SELF": 1}}}
        with self.assertRaises(PairJoinError):
            assemble_train_pairs([case_record], views, plan)

    def test_decoded_train_views_round_trip_through_pair_join(self):
        train_case = case("c0", split="TRAIN", fold=2)
        views = decoded(train_case, 0.0)
        plan = {"g0": {"split": "TRAIN", "development_fold": 2, "case_counts": {"SELF": 1}}}
        out = assemble_train_pairs([train_case], views, plan)
        expected = [0.5 * a + 0.5 * b for a, b in zip(vec(0.0), vec(1000.0))]
        np.testing.assert_allclose(out["x"][0], expected, rtol=0, atol=1e-12)
        self.assertEqual(out["case_ids"], ["c0"])

    def test_serialized_mean_matches_decoded_mean(self):
        case_record = case("c0")
        payload_pair = transport(case_record, 7.0)
        raw = [decode_activation(record["payload"], record["payload_sha256"]) for record in payload_pair]
        joined = assemble_validation_pairs([case_record], payload_pair, val_plan())
        expected = [0.5 * a + 0.5 * b for a, b in zip(raw[0], raw[1])]
        self.assertEqual(joined["x"][0], expected)


if __name__ == "__main__":
    unittest.main()
