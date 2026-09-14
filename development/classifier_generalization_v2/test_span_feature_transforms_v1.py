"""Synthetic-only tests for span_feature_transforms_v1.

No real data, model, tokenizer, provider, extraction, holdout, results,
network, install, Git, config, coordination or subagents. Only numpy, sklearn,
the standard library and the module under test are imported.
"""

import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from harness import pair_average
from span_feature_transforms_v1 import (
    CLASS_ORDER,
    F1_DIM,
    F2_DIM,
    F3_DIM,
    F4_DIM,
    F5_DIM,
    FEATURE_DIM,
    FEATURE_NAMES,
    LAYER_BLOCKS,
    MAX_WINDOW,
    REPRESENTATION_NAMES,
    WIDTH,
    SpanFeatureTransformError,
    SpanFeatureTransforms,
)

LAYER = len(LAYER_BLOCKS)


def make_windows(rng, labels, offsets=None, length=5):
    """Build distinct, label-dependent (3, length, 1024) windows."""
    windows = []
    for index, label in enumerate(labels):
        class_index = CLASS_ORDER.index(label)
        base = rng.normal(size=(LAYER, length, WIDTH)) * 0.05
        if offsets is None:
            base += class_index * 0.5
        else:
            base += offsets[index]
        windows.append(base.astype(np.float32))
    return windows


def default_data(seed=11, per_class=12, length=5):
    rng = np.random.default_rng(seed)
    labels = np.asarray([name for name in CLASS_ORDER for _ in range(per_class)])
    return make_windows(rng, labels, length=length), labels


class FitTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.windows, cls.labels = default_data()

    def fitted(self):
        return SpanFeatureTransforms().fit(self.windows, self.labels)

    def test_widths_and_representation_order(self):
        model = self.fitted()
        matrix = model.transform(self.windows)
        self.assertEqual(FEATURE_DIM, 8272)
        self.assertEqual(matrix.shape, (len(self.windows), FEATURE_DIM))
        self.assertEqual(
            model.representation_widths(),
            {
                "concat3_last": 3072,
                "concat3_mean": 3072,
                "block10_last_mean": 2048,
                "contrast_stats": 36,
                "pca8_products": 44,
            },
        )
        self.assertEqual((F1_DIM, F2_DIM, F3_DIM, F4_DIM, F5_DIM), (3072, 3072, 2048, 36, 44))
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIM)
        self.assertEqual((REPRESENTATION_NAMES[0], REPRESENTATION_NAMES[-1]), ("concat3_last", "pca8_products"))

    def test_mixed_length_training_and_batch_transform(self):
        windows = [w[:, :(i % 5) + 1, :] for i, w in enumerate(self.windows)]
        model = SpanFeatureTransforms().fit(windows, self.labels)
        actual = model.transform(windows)
        self.assertEqual(actual.shape, (len(windows), 8272))
        expected = np.vstack([model.transform(w) for w in windows])
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)

    def test_mixed_length_validation_without_padding(self):
        model = self.fitted()
        windows = [self.windows[0][:, :1, :], self.windows[1][:, :4, :]]
        actual = model.transform(windows)
        expected = np.concatenate([model.f1(windows), model.f2(windows),
            model.f3(windows), model.f4(windows), model.f5(windows)], axis=1)
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)

    def test_f1_f2_f3_numerical_correctness(self):
        model = self.fitted()
        window = self.windows[0]
        matrix = model.transform([window])[0]
        f1 = matrix[:F1_DIM]
        for layer_index in range(LAYER):
            block = layer_index * WIDTH
            np.testing.assert_allclose(f1[block:block + WIDTH], window[layer_index, -1, :].astype(float))
        f2 = matrix[F1_DIM:F1_DIM + F2_DIM]
        for layer_index in range(LAYER):
            block = layer_index * WIDTH
            np.testing.assert_allclose(f2[block:block + WIDTH], window[layer_index].astype(float).mean(axis=0))
        anchor = window[1]  # block 10
        f3 = matrix[F1_DIM + F2_DIM:F1_DIM + F2_DIM + F3_DIM]
        np.testing.assert_allclose(f3[:WIDTH], anchor[-1, :].astype(float))
        np.testing.assert_allclose(f3[WIDTH:], anchor.astype(float).mean(axis=0))

    def test_f4_uses_train_class_means_and_plan_statistics(self):
        model = self.fitted()
        window = self.windows[3]
        # Recompute directions directly from TRAIN rows (same formula as plan).
        expectations = []
        for layer_index in range(LAYER):
            last = np.stack([w[layer_index, -1, :] for w in self.windows]).astype(float)
            self_mean = last[np.asarray(self.labels) == "SELF"].mean(axis=0)
            for other in ("OTHER", "NONTERMINATION", "ORDINARY"):
                other_mean = last[np.asarray(self.labels) == other].mean(axis=0)
                difference = self_mean - other_mean
                norm = np.linalg.norm(difference)
                direction = difference / norm if norm > 1e-12 else np.zeros(WIDTH)
                scores = window[layer_index].astype(float) @ direction
                ordered = np.sort(scores)
                top = int(np.ceil(0.1 * scores.shape[0]))
                expectations.extend([
                    float(np.mean(scores)),
                    float(np.percentile(ordered, 90.0, method="linear")),
                    float(np.mean(ordered[-top:])),
                    float(np.std(scores, ddof=0)),
                ])
        f4 = model.transform([window])[0][F1_DIM + F2_DIM + F3_DIM:F1_DIM + F2_DIM + F3_DIM + F4_DIM]
        np.testing.assert_allclose(f4, expectations, rtol=1e-6, atol=1e-6)

    def test_f5_standardization_products_and_lexicographic_order(self):
        model = self.fitted()
        matrix = model.transform(self.windows)
        f5 = matrix[:, F1_DIM + F2_DIM + F3_DIM + F4_DIM:]
        self.assertEqual(f5.shape, (len(self.windows), F5_DIM))
        standardized = f5[:, :8]
        squared = f5[:, 8:16]
        products = f5[:, 16:]
        np.testing.assert_allclose(squared, standardized ** 2, rtol=1e-9, atol=1e-9)
        pair_columns = [
            standardized[:, left] * standardized[:, right]
            for left in range(8) for right in range(left + 1, 8)
        ]
        np.testing.assert_allclose(products, np.stack(pair_columns, axis=1), rtol=1e-9, atol=1e-9)
        # Train-standardized components: population ddof=0 on the TRAIN rows.
        fitted_components = model._pca.transform(np.stack([
            np.concatenate([w[layer_index, -1, :] for layer_index in range(LAYER)]).astype(float)
            for w in self.windows
        ]))
        expected = (fitted_components - fitted_components.mean(axis=0)) / fitted_components.std(axis=0, ddof=0)
        np.testing.assert_allclose(standardized, expected, rtol=1e-6, atol=1e-6)

    def test_short_window_length_one_and_no_implicit_padding(self):
        rng = np.random.default_rng(3)
        windows, labels = default_data(seed=3, length=1)
        model = SpanFeatureTransforms().fit(windows, labels)
        matrix = model.transform(windows)
        self.assertTrue(np.isfinite(matrix).all())
        window = windows[0]
        # With n == 1 every F4 statistic is the single projection itself, and no
        # padding row is invented (means equal the only token row, not a padded mean).
        f1 = matrix[0, :F1_DIM]
        for layer_index in range(LAYER):
            np.testing.assert_allclose(
                f1[layer_index * WIDTH:(layer_index + 1) * WIDTH],
                window[layer_index, -1, :].astype(float),
            )
        f2 = matrix[0, F1_DIM:F1_DIM + F2_DIM]
        for layer_index in range(LAYER):
            np.testing.assert_allclose(
                f2[layer_index * WIDTH:(layer_index + 1) * WIDTH],
                window[layer_index, 0, :].astype(float),
            )
        f4 = matrix[0, F1_DIM + F2_DIM + F3_DIM:F1_DIM + F2_DIM + F3_DIM + F4_DIM]
        self.assertTrue(np.isfinite(f4).all())
        self.assertEqual(int(np.sum(np.isnan(f4))), 0)

    def test_transform_is_state_frozen_and_deterministic(self):
        model = self.fitted()
        before = {
            "directions": {block: {k: v.copy() for k, v in value.items()} for block, value in model._directions.items()},
            "mean": model._standardize_mean.copy(),
            "scale": model._standardize_scale.copy(),
        }
        first = model.transform(self.windows[:4])
        second = model.transform([window + 1e3 for window in self.windows[:4]])
        np.testing.assert_array_equal(model.transform(self.windows[:4]), first)
        for block, value in model._directions.items():
            for key, direction in value.items():
                np.testing.assert_array_equal(direction, before["directions"][block][key])
        np.testing.assert_array_equal(model._standardize_mean, before["mean"])
        np.testing.assert_array_equal(model._standardize_scale, before["scale"])
        self.assertFalse(np.allclose(first, second))

    def test_pair_average_matches_harness_convention(self):
        rng = np.random.default_rng(5)
        windows, labels = default_data(seed=5, length=4)
        views = make_windows(rng, labels, length=4)
        averaged = [pair_average(a, b) for a, b in zip(windows, views)]
        model = SpanFeatureTransforms().fit(averaged, labels)
        matrix = model.transform(averaged)
        window = averaged[0]
        np.testing.assert_allclose(matrix[0, :WIDTH], window[0, -1, :].astype(float))

    def test_bare_window_and_single_case_accepted(self):
        model = self.fitted()
        single = model.transform(self.windows[0])
        self.assertEqual(single.shape, (1, FEATURE_DIM))
        np.testing.assert_array_equal(single, model.transform([self.windows[0]]))


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.windows, cls.labels = default_data()

    def test_rejects_bad_shape_width_and_layers(self):
        model = SpanFeatureTransforms()
        bad_layers = np.zeros((2, 5, WIDTH))
        with self.assertRaises(SpanFeatureTransformError):
            model.fit([bad_layers], ["SELF", "OTHER", "NONTERMINATION", "ORDINARY"])
        bad_width = np.zeros((LAYER, 5, WIDTH - 1))
        with self.assertRaises(SpanFeatureTransformError):
            model.fit([bad_width], ["SELF", "OTHER", "NONTERMINATION", "ORDINARY"])
        bad_ndim = np.zeros((LAYER, WIDTH))
        with self.assertRaises(SpanFeatureTransformError):
            model.fit([bad_ndim], ["SELF", "OTHER", "NONTERMINATION", "ORDINARY"])

    def test_rejects_empty_and_overlong_windows(self):
        model = SpanFeatureTransforms()
        with self.assertRaises(SpanFeatureTransformError):
            model.fit([], [])
        overlong = np.zeros((LAYER, MAX_WINDOW + 1, WIDTH))
        with self.assertRaises(SpanFeatureTransformError):
            model.fit([overlong], ["SELF"])

    def test_rejects_nonfinite_values(self):
        windows, labels = default_data()
        windows[2][0, 0, 0] = np.nan
        with self.assertRaises(SpanFeatureTransformError):
            SpanFeatureTransforms().fit(windows, labels)

    def test_rejects_unknown_labels_and_missing_classes(self):
        windows, labels = default_data()
        bad = labels.copy()
        bad[0] = "UNKNOWN"
        with self.assertRaises(SpanFeatureTransformError):
            SpanFeatureTransforms().fit(windows, bad)
        with self.assertRaises(SpanFeatureTransformError):
            SpanFeatureTransforms().fit(windows[:10], labels[:10])

    def test_rejects_class_without_self_or_single_row_support(self):
        windows, labels = default_data(per_class=2)
        labels = labels.copy()
        labels[labels == "OTHER"] = "SELF"
        with self.assertRaises(SpanFeatureTransformError):
            SpanFeatureTransforms().fit(windows, labels)

    def test_rejects_too_few_pca_rows(self):
        windows, labels = default_data()
        with self.assertRaises(SpanFeatureTransformError):
            SpanFeatureTransforms().fit(windows[:7], labels[:7])

    def test_transform_before_fit_rejected(self):
        model = SpanFeatureTransforms()
        with self.assertRaises(SpanFeatureTransformError):
            model.transform(self.windows)

    def test_second_fit_rejected(self):
        model = SpanFeatureTransforms().fit(self.windows, self.labels)
        with self.assertRaises(SpanFeatureTransformError):
            model.fit(self.windows, self.labels)

    def test_transform_rejects_invalid_windows(self):
        model = SpanFeatureTransforms().fit(self.windows, self.labels)
        with self.assertRaises(SpanFeatureTransformError):
            model.transform([np.zeros((LAYER, 0, WIDTH))])
        broken = self.windows[0].copy()
        broken[1, 2, 3] = np.inf
        with self.assertRaises(SpanFeatureTransformError):
            model.transform([broken])


class DegenerateStatisticsTests(unittest.TestCase):
    def test_identical_class_means_yield_zero_not_nan(self):
        rng = np.random.default_rng(13)
        labels = np.asarray([name for name in CLASS_ORDER for _ in range(6)])
        # SELF and OTHER share an identical mean, so SELF - OTHER has zero norm;
        # NONTERMINATION and ORDINARY stay separable.
        offsets = {"SELF": 0.0, "OTHER": 0.0, "NONTERMINATION": 1.0, "ORDINARY": 2.0}
        windows = [
            rng.normal(size=(LAYER, 4, WIDTH)) * 0.05 + offsets[label]
            for label in labels
        ]
        model = SpanFeatureTransforms().fit(windows, labels)
        matrix = model.transform(windows)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertFalse(np.isnan(matrix).any())

    def test_zero_variance_pca_component_scale_is_one(self):
        labels = np.asarray([name for name in CLASS_ORDER for _ in range(2)])
        rng = np.random.default_rng(17)
        windows = [np.zeros((LAYER, 3, WIDTH)) + rng.normal() for _ in range(8)]
        model = SpanFeatureTransforms().fit(windows, labels)
        # A constant F1 matrix has zero component variance; scale must be 1.0.
        self.assertTrue(np.all(model._standardize_scale >= 1.0 - 1e-12))
        matrix = model.transform(windows)
        self.assertTrue(np.isfinite(matrix).all())


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        unittest.main()
