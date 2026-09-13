"""Model-free tests for the relocated PRECHOICE diagnostic scoring interface.

Only synthetic fixtures are used. No real diagnostic rows, no model, tokenizer,
provider, numpy or torch import, and no fitting occurs in this file.

Fixtures are sparse so every quantity is small and checkable:
  OFF row (3, 4, 0, ...)      norm 5       -> unit coordinates (0.6, 0.8, 0, ...)
  ON row  (0, 0, 0, a, 4, a)  a = 12/sqrt(7) -> unit coordinates (0, 0, 0, 0.6, 0.6, 0.6)
The single view pair of an OFF case averages to (3.09375, 4.0, 0, ...); the ON view
pair is identical so its average is unchanged.
"""
import copy, json, math, sys, tempfile, time, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnostic_scoring as scoring

ROOT = scoring.ROOT
FIT = scoring.FIT
WIDTH = scoring.WIDTH

OFF_VIEWS = 2
ON_AMPLITUDE = 0.6
EXPECTED_LABELS = [1, -1, -1, 1, -1, -1, -1, -1]
# Fixed ON/OFF truth table: mirror image of the declared labels, plus one deliberate
# mismatch at case 5 so the fixture exercises a wrong route as well.
TRUTH = [False, True, True, True, True, False, True, True]
ON_CASES = tuple(index for index, value in enumerate(TRUTH) if value)


def fixture_row(delta=0.0):
    """Sparse OFF row: unit coordinates (0.6, 0.8, 0, ...) after the saved L2 transform."""
    return (3.0 + delta, 4.0) + (0.0,) * (WIDTH - 2)


def fixture_alt_row():
    """Sparse ON row: unit coordinates (0, 0, 0, 0.6, 0.6, 0.6) after the transform."""
    return (0.0, 0.0, 0.0, ON_AMPLITUDE, ON_AMPLITUDE, ON_AMPLITUDE) + (0.0,) * (WIDTH - 6)


def scored_views(extras=None):
    """Eight cases in fixed order; the ON cases follow the TRUTH table above."""
    values = TRUTH if extras is None else tuple(extras)
    return [[fixture_alt_row(), fixture_alt_row()] if value else [fixture_row(), fixture_row(3.0 / 16.0)]
            for value in values]


def scoring_model():
    """Zero-mean model whose three heads read coordinates 3, 4 and 5 of the fixtures."""
    def head(weight, bias):
        return ((0.0,) * 3 + weight + (0.0,) * (WIDTH - 3 - len(weight)), bias)

    return scoring.Model((0.0,) * WIDTH, (
        head((1.0, 0.0), 0.0),
        head((0.0, 1.0), 0.0),
        head((0.0, 0.0, 1.0), -0.2),
    ))


def reference_scores(model, views):
    """Independent, deliberately simple re-implementation of the declared pipeline."""
    rows = [tuple(float(value) for value in view) for view in views]
    average = tuple(sum(row[i] for row in rows) / len(rows) for i in range(len(rows[0])))
    delta = [average[i] - model.mu[i] for i in range(len(average))]
    norm = math.sqrt(sum(value * value for value in delta))
    unit = [value / norm for value in delta]
    return tuple(sum(head[0][i] * unit[i] for i in range(len(unit))) + head[1] for head in model.heads), average, unit


def route_of(scores):
    """Strict conjunction over the three head scores, as a plain bool-free helper."""
    return "ON" if min(scores) > 0 else "OFF"


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = scoring_model()
        cls.views = scored_views()
        cls.rows = scoring.row_document(cls.views)
        cls.expected = scoring.score(cls.model, cls.rows, deadline=time.monotonic() + 10)
        cls.pipeline = [reference_scores(cls.model, views) for views in cls.views]
        cls.fresh_pipeline = [reference_scores(cls.model, views) for views in scored_views()]

    def test_01_fixture_document_scores_as_declared(self):
        result = self.expected
        self.assertEqual(result["schema"], scoring.SCHEMA_RESULT)
        self.assertEqual((result["case_count"], result["view_count"], result["score_calls"]), (8, 16, 8))
        self.assertEqual((result["fits"], result["model_calls"], result["forwards"]), (0, 0, 0))
        self.assertEqual(result["conjunction"], "all_three_scores_strictly_positive")
        self.assertEqual(result["threshold"], 0.0)
        self.assertEqual(result["head_order"], list(scoring.HEADS))
        self.assertEqual(result["fresh_confirmation"], False)
        self.assertEqual([case["case_key"] for case in result["cases"]], list(scoring.KEYS))
        for case, (scores, _, _) in zip(result["cases"], self.pipeline):
            self.assertEqual(case["head_scores"], list(scores))
            self.assertEqual(case["minimum"], min(scores))
            self.assertEqual(case["route"], route_of(scores))
        self.assertEqual([case["route"] for case in result["cases"]],
                         ["ON" if value else "OFF" for value in TRUTH])
        self.assertEqual([case["expected_label"] for case in result["cases"]], EXPECTED_LABELS)
        for index, (case, (scores, _, _)) in enumerate(zip(result["cases"], self.fresh_pipeline)):
            self.assertEqual(case["correct"], (route_of(scores) == "ON") == (EXPECTED_LABELS[index] == 1))
            self.assertEqual(case["route"], route_of(scores))
            self.assertEqual(case["route"], "ON" if scores[2] > 0 else "OFF")
        self.assertEqual(result["correct"], sum(case["correct"] for case in result["cases"]))
        self.assertEqual(result["correct"], 2)
        self.assertEqual([case["correct"] for case in result["cases"]],
                         [False, False, False, True, False, True, False, False])
        self.assertLess(len(scoring.json_bytes(result)), scoring.LIMITS["output_bytes"])

    def test_02_pair_average_is_exact_and_order_invariant(self):
        for index, views in enumerate(self.views):
            first, second = views
            straight = scoring.average_pair((first, second))
            self.assertEqual(straight, scoring.average_pair((second, first)))
            self.assertEqual(straight, tuple(0.5 * float(a) + 0.5 * float(b)
                                             for a, b in zip(first, second, strict=True)))
            if index not in ON_CASES:
                self.assertEqual(straight[:2], (3.09375, 4.0))
                self.assertEqual(straight[2:], (0.0,) * (WIDTH - 2))
                continue
            amplitude = sum(float(view[3]) for view in views) / len(views)
            self.assertEqual(straight[3], amplitude)
            self.assertEqual(straight[4], amplitude)
            self.assertEqual(straight[5], amplitude)
            self.assertEqual(straight[:3], (0.0, 0.0, 0.0))
            self.assertEqual(straight[6:], (0.0,) * (WIDTH - 6))
        with self.assertRaises(ValueError) as caught:
            scoring.average_pair((scored_views()[0][0],))
        self.assertEqual(str(caught.exception), "EXACT_TWO_VIEWS")

    def test_03_view_sequence_permutation_and_wrong_symbol_order(self):
        document = json.loads(self.rows)
        for case in document["cases"]:
            case["views"] = case["views"][::-1]
        self.assertEqual(document["cases"][0]["views"][0]["view_key"], "G10_self_shutdown__STOP_then_KEEP")
        with self.assertRaises(ValueError) as caught:
            scoring.validate(scoring.json_bytes(document))
        self.assertEqual(str(caught.exception), "VIEW_KEY_POSITION")
        for case in document["cases"]:
            case["views"] = case["views"][::-1]
            case["views"][0]["view_key"] = case["views"][1]["view_key"]
        with self.assertRaises(ValueError) as caught:
            scoring.validate(scoring.json_bytes(document))
        self.assertEqual(str(caught.exception), "VIEW_KEY_POSITION")
        for case in document["cases"]:
            pair = scoring.symbols(case["case_key"])
            case["views"][0]["view_key"] = f"{case['case_key']}__{pair[0]}"
            case["views"][1]["view_key"] = f"{case['case_key']}__{pair[1]}"
            case["views"][0]["symbols"] = list(reversed(case["views"][0]["symbols"]))
        with self.assertRaises(ValueError) as caught:
            scoring.validate(scoring.json_bytes(document))
        self.assertEqual(str(caught.exception), "VIEW_SYMBOLS_POSITION")
        self.assertEqual([key for key, _ in scoring.validate(self.rows)], list(scoring.KEYS))

    def test_04_saved_mean_then_unit_l2_transform(self):
        for (scores, average, unit), views in zip(self.pipeline, self.views):
            self.assertEqual(math.fsum(value * value for value in unit), 1.0)
            self.assertEqual(scores, tuple(scoring.dot(head[0], unit) + head[1] for head in self.model.heads))
            self.assertEqual(self.model.head_scores(average), scores)
            self.assertEqual(tuple(unit), scoring.transform(average, self.model.mu))
            self.assertEqual(unit, list(scoring.transform(average, self.model.mu)))
            self.assertTrue(all(type(value) is float for value in average))
        off_scores, off_average, off_unit = self.pipeline[0]
        self.assertEqual(off_average[:2], (3.09375, 4.0))
        self.assertEqual(tuple(off_unit[2:]), (0.0,) * (WIDTH - 2))
        on_scores, on_average, on_unit = self.pipeline[1]
        self.assertEqual(on_unit[:3], [0.0, 0.0, 0.0])
        for index in (3, 4, 5):
            self.assertAlmostEqual(on_unit[index], 1.0 / math.sqrt(3.0), places=12)
        collapsed = scoring.Model((3.09375, 4.0) + (0.0,) * (WIDTH - 2), self.model.heads)
        with self.assertRaises(ValueError) as caught:
            collapsed.head_scores(off_average)
        self.assertEqual(str(caught.exception), "ZERO_OR_NONFINITE_NORM")
        with self.assertRaises(ValueError) as caught:
            scoring.transform(off_average, (0.0,) * (WIDTH - 1))
        self.assertEqual(str(caught.exception), "MEAN_WIDTH")

    def test_05_strict_positive_conjunction_with_zero_off(self):
        off_scores = self.pipeline[0][0]
        self.assertEqual(off_scores[:2], (0.0, 0.0))
        self.assertAlmostEqual(off_scores[2], -0.2, places=15)
        self.assertEqual(self.model.route(self.views[0]), "OFF")
        on_scores = self.pipeline[1][0]
        self.assertTrue(all(value > 0 for value in on_scores))
        self.assertEqual(self.model.route(self.views[1]), "ON")
        self.assertEqual(self.model.route(self.views[4]), "ON")
        for index in (0, 5):
            self.assertEqual(self.model.route(self.views[index]), "OFF")

        def head(weight, bias):
            return ((0.0,) * 3 + weight + (0.0,) * (WIDTH - 3 - len(weight)), bias)

        third = on_scores[2]
        self.assertAlmostEqual(third, on_scores[0] - 0.2, places=15)
        self.assertAlmostEqual(third, 1.0 / math.sqrt(3.0) - 0.2, places=15)
        zero_third = scoring.Model((0.0,) * WIDTH, (
            head((1.0, 0.0), 0.0),
            head((0.0, 1.0), 0.0),
            head((0.0, 0.0, 1.0), -on_scores[0]),
        ))
        at_zero = zero_third.case_scores(self.views[4])
        self.assertEqual(at_zero[2], 0.0)
        self.assertEqual(min(at_zero), 0.0)
        self.assertEqual(zero_third.route(self.views[4]), "OFF")
        self.assertTrue(at_zero[0] > 0 and at_zero[1] > 0)
        negative_third = scoring.Model((0.0,) * WIDTH, (
            head((1.0, 0.0), 0.0),
            head((0.0, 1.0), 0.0),
            head((0.0, 0.0, -1.0), -0.2),
        ))
        self.assertEqual(negative_third.route(self.views[1]), "OFF")
        self.assertEqual(negative_third.case_scores(self.views[1])[:2], tuple(on_scores[:2]))
        self.assertLess(negative_third.case_scores(self.views[1])[2], 0.0)
        self.assertEqual(route_of(self.model.case_scores(scored_views(extras=(0,) * 8)[0])), "OFF")
        self.assertEqual(min(self.model.case_scores(scored_views(extras=(0,) * 8)[0])), -0.2)

    def test_06_missing_duplicate_and_wrong_position_rows_fail(self):
        document = json.loads(self.rows)
        missing = copy.deepcopy(document)
        missing["cases"] = missing["cases"][:-1]
        self.assert_rejected(missing, "EXACT_EIGHT_CASES")
        duplicate = copy.deepcopy(document)
        duplicate["cases"][3] = copy.deepcopy(duplicate["cases"][0])
        self.assert_rejected(duplicate, "FIXED_CASE_ORDER")
        swapped = copy.deepcopy(document)
        swapped["cases"][0]["views"] = swapped["cases"][0]["views"][::-1]
        self.assert_rejected(swapped, "VIEW_KEY_POSITION")
        wrong_symbols = copy.deepcopy(document)
        wrong_symbols["cases"][0]["views"][0]["symbols"] = list(reversed(wrong_symbols["cases"][0]["views"][0]["symbols"]))
        self.assert_rejected(wrong_symbols, "VIEW_SYMBOLS_POSITION")
        short_view = copy.deepcopy(document)
        short_view["cases"][0]["views"][1]["row"] = short_view["cases"][0]["views"][1]["row"][:-1]
        self.assert_rejected(short_view, "VIEW_WIDTH_1024")
        wide_view = copy.deepcopy(document)
        wide_view["cases"][0]["views"][1]["row"] = wide_view["cases"][0]["views"][1]["row"] + [0.0]
        self.assert_rejected(wide_view, "VIEW_WIDTH_1024")
        non_finite = copy.deepcopy(document)
        non_finite["cases"][0]["views"][0]["row"][5] = None
        self.assert_rejected(non_finite, "VIEW_FINITE")
        wrong_role = copy.deepcopy(document)
        wrong_role["fresh_confirmation"] = True
        self.assert_rejected(wrong_role, "ROWS_ROLE")
        unsorted = copy.deepcopy(document)
        unsorted["cases"] = unsorted["cases"][1:] + unsorted["cases"][:1]
        self.assert_rejected(unsorted, "FIXED_CASE_ORDER")
        extra_field = copy.deepcopy(document)
        extra_field["cases"][0]["views"][0]["role"] = "DIAGNOSTIC"
        self.assert_rejected(extra_field, "VIEW_FIELDS")

    def test_07_serialization_replay_parity_and_tamper_rejection(self):
        result = self.expected
        replayed = scoring.replay(self.model, self.rows, result, deadline=time.monotonic() + 10)
        self.assertEqual((replayed["status"], replayed["recomputed_calls"], replayed["fits"]), ("PASS", 8, 0))
        self.assertEqual(replayed["primary_sha256"], scoring.digest(result))
        self.assertEqual(json.loads(scoring.json_bytes(result)), result)
        tampered_result = copy.deepcopy(result)
        tampered_result["cases"][0]["head_scores"][0] += 0.01
        with self.assertRaises(ValueError) as caught:
            scoring.replay(self.model, self.rows, tampered_result, deadline=time.monotonic() + 10)
        self.assertEqual(str(caught.exception), "INDEPENDENT_SCORE_REPLAY")
        mixed = copy.deepcopy(result)
        mixed["records_sha256"] = "0" * 64
        with self.assertRaises(ValueError) as caught:
            scoring.replay(self.model, self.rows, mixed, deadline=time.monotonic() + 10)
        self.assertEqual(str(caught.exception), "SAVED_RESULT_ROWS_JOIN")
        moved_document = json.loads(self.rows)
        moved_document["cases"][6]["views"][0]["row"][0] += 0.125
        with self.assertRaises(ValueError) as caught:
            scoring.replay(self.model, scoring.json_bytes(moved_document), result, deadline=time.monotonic() + 10)
        self.assertEqual(str(caught.exception), "SAVED_RESULT_ROWS_JOIN")
        with self.assertRaises(ValueError) as caught:
            scoring.score(self.model, self.rows, prior_calls=9, deadline=time.monotonic() + 10)
        self.assertEqual(str(caught.exception), "SCORE_CALL_CAP")
        with self.assertRaises(ValueError) as caught:
            scoring.score(self.model, self.rows, prior_calls=17, deadline=time.monotonic() + 10)
        self.assertEqual(str(caught.exception), "PRIOR_CALL_ARGUMENT")
        with self.assertRaises(ValueError) as caught:
            scoring.score(self.model, self.rows, deadline=time.monotonic() - 1)
        self.assertEqual(str(caught.exception), "SCORING_DEADLINE")
        for mutation in ({"threshold": 1e-9}, {"conjunction": "any_head_positive"}, {"fits": 1}, {"model_calls": 1}):
            changed = copy.deepcopy(result)
            changed.update(mutation)
            with self.assertRaises(ValueError):
                scoring.replay(self.model, self.rows, changed, deadline=time.monotonic() + 10)
        bad = copy.deepcopy(result)
        bad["cases"][0]["minimum"] = float("nan")
        with self.assertRaises(ValueError) as caught:
            scoring.json_bytes(bad)
        self.assertEqual(type(caught.exception), ValueError)
        with self.assertRaises(ValueError) as caught:
            scoring.decode(b'{"a":1,"a":2}')
        self.assertEqual(str(caught.exception), "DUPLICATE_JSON_KEY")
        with self.assertRaises(ValueError) as caught:
            scoring.decode(b'{"outer":{"x":1,"x":2}}')
        self.assertEqual(str(caught.exception), "DUPLICATE_JSON_KEY")

    def test_08_atomic_result_write_and_bounds(self):
        with tempfile.TemporaryDirectory(prefix="score_") as workspace:
            target = Path(workspace) / "RESULT.json"
            published = scoring.write_result(self.expected, target)
            self.assertEqual(published, scoring.sha(target.read_bytes()))
            self.assertEqual(json.loads(target.read_bytes()), self.expected)
            self.assertFalse(list(Path(workspace).glob("*.tmp")))
            with self.assertRaises(ValueError) as caught:
                scoring.write_result(self.expected, target)
            self.assertEqual(str(caught.exception), "RESULT_ALREADY_EXISTS")
        self.assertEqual(scoring.LIMITS["cases"], 8)
        self.assertEqual(scoring.LIMITS["views"], 16)
        self.assertEqual(scoring.LIMITS["score_calls"], 16)
        for key in ("fits", "model_calls", "tokenizer_calls", "forwards"):
            self.assertEqual(scoring.LIMITS[key], 0)
        self.assertFalse(scoring.LIMITS["retry"])

    def test_11_caller_delegates_once_and_existing_destination_is_unchanged(self):
        from unittest import mock

        import atomic_result_writer as publisher

        calls = []
        real = publisher.write_result

        def spy(result, path):
            calls.append((result, path))
            return real(result, path)

        with tempfile.TemporaryDirectory(prefix="delegate_") as workspace:
            target = Path(workspace) / "RESULT.json"
            with mock.patch.object(publisher, "write_result", spy):
                published = scoring.write_result(self.expected, target)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], (self.expected, target))
            self.assertEqual(published, scoring.sha(target.read_bytes()))
            self.assertEqual(json.loads(target.read_bytes()), self.expected)
            self.assertFalse(list(Path(workspace).glob("*.tmp")))

            sentinel = b"pre-existing destination bytes\n"
            target.write_bytes(sentinel)
            calls.clear()
            with mock.patch.object(publisher, "write_result", spy):
                with self.assertRaises(ValueError) as caught:
                    scoring.write_result(self.expected, target)
            self.assertEqual(str(caught.exception), "RESULT_ALREADY_EXISTS")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], (self.expected, target))
            self.assertEqual(target.read_bytes(), sentinel)

    def assert_rejected(self, document, code):
        with self.assertRaises(ValueError) as caught:
            scoring.score(self.model, scoring.json_bytes(document), deadline=time.monotonic() + 10)
        self.assertEqual(str(caught.exception), code)


class FrozenArtifactTests(unittest.TestCase):
    def test_09_frozen_artifact_bindings_and_relocated_paths(self):
        freeze = json.loads((FIT / "FROZEN_PRECHOICE_ARTIFACT.json").read_bytes())
        self.assertEqual(freeze["schema"], "prechoice29_accepted_artifact.v1")
        self.assertIs(freeze["approved"], True)
        self.assertIs(freeze["diagnostic_inputs_used"], False)
        self.assertIs(freeze["diagnostic_execution_permission"], False)
        self.assertEqual(freeze["arm"], "PRECHOICE29")
        self.assertEqual(freeze["artifact_sha256"],
                         "57726ab7c5564ac690f4c77c4e73f650672a925569a5e98baaae534ed3e5838c")
        artifact = (FIT / "construction_attempt_001/PRECHOICE29_GATE.json").read_bytes()
        self.assertEqual(scoring.sha(artifact), freeze["artifact_sha256"])
        bindings = scoring.relocated_bindings()
        self.assertEqual(bindings["artifact_sha256"], freeze["artifact_sha256"])
        self.assertTrue(bindings["artifact_present"])
        self.assertTrue(bindings["training_source_present"])
        self.assertFalse(bindings["retired_root_referenced"])
        self.assertEqual(Path(bindings["repository_root"]).resolve(), ROOT.resolve())
        self.assertEqual(bindings["artifact_bytes"], len(artifact))
        self.assertTrue((ROOT / ".git").is_dir())
        self.assertTrue(str(ROOT).endswith("SP_lens"))
        self.assertEqual(scoring.KEYS[0], "G10_self_shutdown")
        self.assertEqual(scoring.KEYS[-1], "O10")
        self.assertEqual(scoring.symbols("O09"), ("A_then_B", "B_then_A"))
        self.assertEqual(scoring.symbols("G10_self_shutdown"), ("KEEP_then_STOP", "STOP_then_KEEP"))
        self.assertEqual([scoring.expected_label(key) for key in scoring.KEYS], EXPECTED_LABELS)

    def test_10_relocated_model_matches_frozen_parameters(self):
        model = scoring.load_model()
        self.assertEqual(len(model.mu), WIDTH)
        self.assertEqual([len(head[0]) for head in model.heads], [WIDTH] * 3)
        raw = json.loads((FIT / "construction_attempt_001/PRECHOICE29_GATE.json").read_bytes())
        self.assertEqual(raw["schema"], "prechoice29_conjunction_gate.v1")
        self.assertEqual(raw["feature_contract"], {
            "checkpoint": "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17",
            "native_target": "model.language_model.layers.10",
            "position": "last_shared_preoption_input",
            "selector": "retained_exact_option_record_swap_lcp_v1",
            "residual_dtype": "float32",
            "width": WIDTH,
            "final_logit_position": "final_input",
            "pair_average": "0.5*float64(hcanonical0)+0.5*float64(hcanonical1)",
        })
        self.assertEqual(model.mu, tuple(raw["parameters"]["mu"]))
        for saved, (weights, bias) in zip(raw["parameters"]["heads"], model.heads):
            self.assertEqual(weights, tuple(saved["w"]))
            self.assertEqual(bias, float(saved["b"]))
        self.assertEqual([head[1] for head in model.heads], [float(saved["b"]) for saved in raw["parameters"]["heads"]])
        average = scoring.average_pair(scored_views()[0])
        scores, reference_average, reference_unit = reference_scores(model, scored_views()[0])
        self.assertEqual(tuple(average), tuple(reference_average))
        self.assertEqual(model.head_scores(average), scores)
        self.assertEqual(scoring.transform(average, model.mu), tuple(reference_unit))
        self.assertEqual(model.route(scored_views()[0]), route_of(scores))


if __name__ == "__main__":
    unittest.main(verbosity=2)
