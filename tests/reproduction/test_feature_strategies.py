import numpy as np
import pytest

from sp_lense.feature_strategies import build_features


def test_centering_is_invariant_to_per_layer_offsets():
    z = np.array([[1.0, 2.0], [3.0, 4.0]])
    j = np.arange(36.0).reshape(2, 18)
    shifted = j + np.repeat([100.0, -50.0, 7.0], 6)
    a = build_features("centered_jlens_norm", z, j, [10.0, 20.0])
    b = build_features("centered_jlens_norm", z, shifted, [10.0, 20.0])
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a[:, 2:20].reshape(2, 3, 6).mean(axis=2), 0)
    np.testing.assert_array_equal(a[:, :2], z)
    np.testing.assert_array_equal(a[:, -1], [10.0, 20.0])


@pytest.mark.parametrize("strategy,width", [("pca_only", 2), ("raw_jlens", 20)])
def test_preserves_row_order_and_values(strategy, width):
    z = np.array([[1.0, 2.0], [3.0, 4.0]])
    j = np.arange(36.0).reshape(2, 18)
    actual = build_features(strategy, z, j, [10.0, 20.0])
    assert actual.shape == (2, width)
    np.testing.assert_array_equal(actual[:, :2], z)
    if strategy == "raw_jlens":
        np.testing.assert_array_equal(actual[:, 2:], j)


@pytest.mark.parametrize(
    "j,norms",
    [
        (np.zeros((2, 17)), [1.0, 2.0]),
        (np.zeros((1, 18)), [1.0, 2.0]),
        (np.full((2, 18), np.nan), [1.0, 2.0]),
        (np.zeros((2, 18)), [1.0]),
    ],
)
def test_rejects_invalid_shapes_and_nonfinite_inputs(j, norms):
    with pytest.raises(ValueError):
        build_features("raw_jlens", np.ones((2, 2)), j, norms)


def test_unknown_strategy_is_explicit_error():
    with pytest.raises(ValueError, match="Unknown"):
        build_features("typo", np.ones((1, 2)), np.ones((1, 18)), [1.0])
