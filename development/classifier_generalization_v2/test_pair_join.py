"""Fake in-memory unit tests for pair_join.assemble_train_pairs.

All arrays are fabricated 1024-float vectors. These tests exercise the
structural joiner contract only; they say nothing about native capture
authenticity, float32 provenance, or any real manifest/cache/activations.
"""

import unittest

import numpy as np

from pair_join import MODEL_REVISION, PairJoinError, assemble_train_pairs

PREFIX = "a" * 64


def make_case(case_id, group_id="g0", split="TRAIN", class_label="SELF"):
    return {
        "case_id": case_id,
        "group_id": group_id,
        "split": split,
        "class_label": class_label,
    }


def make_view(case_id, order, values, prefix=PREFIX, **overrides):
    view = {
        "case_id": case_id,
        "order": list(order),
        "values": list(values),
        "model_revision": MODEL_REVISION,
        "block": 10,
        "position": "last_shared_preoption_input",
        "dtype": "float32",
        "prefix_sha256": prefix,
    }
    view.update(overrides)
    return view


def values(seed):
    return np.arange(1024, dtype=np.float64) * 0.5 + float(seed)


class PairJoinTests(unittest.TestCase):
    def test_exact_mean_count_and_mapping(self):
        cases = [make_case("c0"), make_case("c1", class_label="OTHER")]
        views = [
            make_view("c0", ["A", "B"], values(0)),
            make_view("c0", ["B", "A"], values(10)),
            make_view("c1", ["A", "B"], values(20)),
            make_view("c1", ["B", "A"], values(30)),
        ]
        plan = {"g0": {"split": "TRAIN", "development_fold": 3, "case_counts": {"SELF": 1, "OTHER": 1}}}
        out = assemble_train_pairs(cases, views, plan)
        self.assertEqual(out["case_ids"], ["c0", "c1"])
        self.assertEqual(out["labels"], ["SELF", "OTHER"])
        self.assertEqual(out["groups"], ["g0", "g0"])
        self.assertEqual(out["folds"], [3, 3])
        self.assertEqual(len(out["x"]), 2)
        self.assertTrue(all(len(row) == 1024 for row in out["x"]))
        expected = 0.5 * values(0) + 0.5 * values(10)
        np.testing.assert_allclose(out["x"][0], expected, rtol=0, atol=0)

    def test_opposite_order_pair_order_independent(self):
        cases = [make_case("c0")]
        plan = {"g0": {"split": "TRAIN", "development_fold": 0, "case_counts": {"SELF": 1}}}
        forward = assemble_train_pairs(
            cases,
            [make_view("c0", ["A", "B"], values(1)), make_view("c0", ["B", "A"], values(2))],
            plan,
        )
        reversed_input = assemble_train_pairs(
            cases,
            [make_view("c0", ["B", "A"], values(2)), make_view("c0", ["A", "B"], values(1))],
            plan,
        )
        np.testing.assert_allclose(forward["x"], reversed_input["x"], rtol=0, atol=0)
        with self.assertRaises(PairJoinError):
            assemble_train_pairs(
                cases,
                [make_view("c0", ["A", "B"], values(1)), make_view("c0", ["A", "B"], values(2))],
                plan,
            )

    def test_missing_duplicate_and_orphan_views(self):
        plan = {"g0": {"split": "TRAIN", "development_fold": 1, "case_counts": {"SELF": 2}}}
        with self.assertRaises(PairJoinError):
            assemble_train_pairs([make_case("c0")], [make_view("c0", ["A", "B"], values(0))], plan)
        with self.assertRaises(PairJoinError):
            assemble_train_pairs(
                [make_case("c0"), make_case("c0")],
                [make_view("c0", ["A", "B"], values(0)), make_view("c0", ["B", "A"], values(1))],
                plan,
            )
        with self.assertRaises(PairJoinError):
            assemble_train_pairs(
                [make_case("c0")],
                [
                    make_view("c0", ["A", "B"], values(0)),
                    make_view("c0", ["B", "A"], values(1)),
                    make_view("ghost", ["A", "B"], values(2)),
                ],
                plan,
            )

    def test_nonfinite_width_and_contract_violations(self):
        plan = {"g0": {"split": "TRAIN", "development_fold": 2, "case_counts": {"SELF": 1}}}
        bad_value_sets = [
            [0.0] * 1023,
            [np.nan] * 1024,
            [np.inf] + [0.0] * 1023,
        ]
        for bad in bad_value_sets:
            with self.assertRaises(PairJoinError):
                assemble_train_pairs(
                    [make_case("c0")],
                    [make_view("c0", ["A", "B"], bad), make_view("c0", ["B", "A"], values(1))],
                    plan,
                )
        with self.assertRaises(PairJoinError):
            assemble_train_pairs(
                [make_case("c0")],
                [
                    make_view("c0", ["A", "B"], values(0), dtype="float64"),
                    make_view("c0", ["B", "A"], values(1)),
                ],
                plan,
            )
        with self.assertRaises(PairJoinError):
            assemble_train_pairs(
                [make_case("c0")],
                [
                    make_view("c0", ["A", "B"], values(0), prefix_sha256="short"),
                    make_view("c0", ["B", "A"], values(1)),
                ],
                plan,
            )
        with self.assertRaises(PairJoinError):
            assemble_train_pairs(
                [make_case("c0")],
                [
                    make_view("c0", ["A", "B"], values(0)),
                    make_view("c0", ["B", "A"], values(1), prefix="b" * 64),
                ],
                plan,
            )

    def test_val_group_and_label_and_fold_rejections(self):
        cases = [make_case("c0")]
        views = [make_view("c0", ["A", "B"], values(0)), make_view("c0", ["B", "A"], values(1))]
        with self.assertRaises(PairJoinError):
            assemble_train_pairs(
                [make_case("c0", split="VAL")],
                views,
                {"g0": {"split": "TRAIN", "development_fold": 0, "case_counts": {"SELF": 1}}},
            )
        with self.assertRaises(PairJoinError):
            assemble_train_pairs(
                cases,
                views,
                {"g0": {"split": "VAL", "development_fold": 0, "case_counts": {"SELF": 1}}},
            )
        with self.assertRaises(PairJoinError):
            assemble_train_pairs(
                cases,
                views,
                {"g0": {"split": "TRAIN", "development_fold": 0, "case_counts": {"OTHER": 1}}},
            )
        for bad_fold in (-1, 5, 1.5):
            with self.assertRaises(PairJoinError):
                assemble_train_pairs(
                    cases,
                    views,
                    {"g0": {"split": "TRAIN", "development_fold": bad_fold, "case_counts": {"SELF": 1}}},
                )


if __name__ == "__main__":
    unittest.main()
