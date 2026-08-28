from __future__ import annotations

import math

import numpy as np
import pytest

from sp_lense.paired_order_analytic_gradient import (
    PairedOrderGradientIneligible,
    cast_and_recertify_common_delta,
    construct_common_gradient_bisector,
    full_vocabulary_bidirectional_interval,
)


def _choice_example() -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    logits = np.array(
        [
            [-1.0, 1.0, -5.0],
            [1.0, -1.0, -5.0],
        ],
        dtype=np.float64,
    )
    derivatives = np.array(
        [
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return logits, derivatives, (0, 1), (1, 0)


def test_common_bisector_is_symmetric_positive_and_deterministic() -> None:
    gradients = np.array([[1.0, 0.0], [0.0, 2.0]])
    first = construct_common_gradient_bisector(gradients, [4.0, 9.0])
    second = construct_common_gradient_bisector(gradients, [4.0, 9.0])

    assert first.direction == pytest.approx([math.sqrt(0.5), math.sqrt(0.5)])
    assert first.common_scale == pytest.approx(6.0)
    assert first.base_vector == pytest.approx(6.0 * first.direction)
    assert first.semantic_slopes == pytest.approx(
        [3.0 * math.sqrt(2.0), 6.0 * math.sqrt(2.0)]
    )
    assert first.diagnostics["normalized_order_alignments"] == pytest.approx(
        [math.sqrt(0.5), math.sqrt(0.5)]
    )
    assert first.diagnostics["diagnostics_sha256"] == second.diagnostics[
        "diagnostics_sha256"
    ]


def test_common_bisector_rejects_order_heads_at_cosine_floor() -> None:
    cosine = -0.99
    gradients = np.array(
        [[1.0, 0.0], [cosine, math.sqrt(1.0 - cosine * cosine)]], dtype=np.float64
    )
    with pytest.raises(PairedOrderGradientIneligible, match="minimum cosine") as error:
        construct_common_gradient_bisector(
            gradients,
            [1.0, 1.0],
            minimum_order_cosine=-0.99,
        )
    assert error.value.diagnostics["failure"] == "order_gradient_cosine_below_floor"


@pytest.mark.parametrize(
    "gradients",
    (
        [[0.0, 0.0], [1.0, 0.0]],
        [[float("nan"), 0.0], [1.0, 0.0]],
    ),
)
def test_common_bisector_fails_closed_on_invalid_gradients(gradients: list[list[float]]) -> None:
    exception = (PairedOrderGradientIneligible, ValueError)
    with pytest.raises(exception):
        construct_common_gradient_bisector(gradients, [1.0, 1.0])


def test_full_vocabulary_interval_returns_exact_common_lower_endpoint() -> None:
    logits, derivatives, preserve, comply = _choice_example()
    result = full_vocabulary_bidirectional_interval(
        logits,
        derivatives,
        preserve,
        comply,
        reserve_logit=0.1,
    )

    assert result.lower == pytest.approx(1.05)
    assert math.isinf(result.upper)
    assert result.diagnostics["constraint_count"] == 8
    assert result.diagnostics["minimum_margin_at_lower"] == pytest.approx(0.1)
    assert result.diagnostics["lower_binding"]["competitor_token_id"] in (0, 1)


def test_full_vocabulary_interval_rejects_immovable_missed_target() -> None:
    logits, _, preserve, comply = _choice_example()
    derivatives = np.zeros_like(logits)
    with pytest.raises(PairedOrderGradientIneligible, match="immovable") as error:
        full_vocabulary_bidirectional_interval(
            logits,
            derivatives,
            preserve,
            comply,
            reserve_logit=0.1,
        )
    assert error.value.diagnostics["failure"] == "immovable_constraint"


def test_full_vocabulary_interval_rejects_conflicting_third_token_upper_bound() -> None:
    logits, derivatives, preserve, comply = _choice_example()
    logits[:, 2] = -0.95
    derivatives[0, 2] = 20.0
    derivatives[1, 2] = 20.0
    with pytest.raises(PairedOrderGradientIneligible, match="interval is empty") as error:
        full_vocabulary_bidirectional_interval(
            logits,
            derivatives,
            preserve,
            comply,
            reserve_logit=0.1,
        )
    assert error.value.diagnostics["failure"] == "empty_analytic_interval"


def test_cast_and_recertify_returns_one_exact_float32_delta() -> None:
    logits, derivatives, preserve, comply = _choice_example()
    alpha = 1.05
    cast_changes = alpha * derivatives
    first = cast_and_recertify_common_delta(
        [1.0, 0.0],
        alpha,
        [20.0, 25.0],
        logits,
        derivatives,
        cast_changes,
        preserve,
        comply,
        reserve_logit=0.1,
    )
    second = cast_and_recertify_common_delta(
        [1.0, 0.0],
        alpha,
        [20.0, 25.0],
        logits,
        derivatives,
        cast_changes,
        preserve,
        comply,
        reserve_logit=0.1,
    )

    assert first.delta.dtype == np.float32
    assert first.delta.tolist() == pytest.approx([1.05, 0.0])
    assert first.diagnostics["relative_norms"] == pytest.approx([0.0525, 0.042])
    assert first.diagnostics["minimum_target_margin"] == pytest.approx(0.1)
    assert first.diagnostics["delta_float32_sha256"] == second.diagnostics[
        "delta_float32_sha256"
    ]
    assert first.diagnostics["diagnostics_sha256"] == second.diagnostics[
        "diagnostics_sha256"
    ]


def test_cast_and_recertify_rejects_relative_norm_cap() -> None:
    logits, derivatives, preserve, comply = _choice_example()
    with pytest.raises(PairedOrderGradientIneligible, match="norm cap") as error:
        cast_and_recertify_common_delta(
            [1.0, 0.0],
            1.05,
            [5.0, 5.0],
            logits,
            derivatives,
            1.05 * derivatives,
            preserve,
            comply,
            reserve_logit=0.1,
        )
    assert error.value.diagnostics["failure"] == "relative_norm_cap_exceeded"


def test_cast_and_recertify_rejects_wrong_cast_jvp() -> None:
    logits, derivatives, preserve, comply = _choice_example()
    wrong = 1.05 * derivatives
    wrong[0, 0] += 0.1
    with pytest.raises(PairedOrderGradientIneligible, match="JVP differs") as error:
        cast_and_recertify_common_delta(
            [1.0, 0.0],
            1.05,
            [20.0, 20.0],
            logits,
            derivatives,
            wrong,
            preserve,
            comply,
            reserve_logit=0.1,
        )
    assert error.value.diagnostics["failure"] == "cast_jvp_inconsistency"


def test_cast_and_recertify_rejects_post_cast_margin_failure() -> None:
    logits, derivatives, preserve, comply = _choice_example()
    weak_changes = 0.95 * derivatives
    with pytest.raises(PairedOrderGradientIneligible, match="target reserve") as error:
        cast_and_recertify_common_delta(
            [1.0, 0.0],
            1.05,
            [20.0, 20.0],
            logits,
            derivatives,
            weak_changes,
            preserve,
            comply,
            reserve_logit=0.1,
            cast_absolute_tolerance=1.0,
        )
    assert error.value.diagnostics["failure"] == "post_cast_margin_failure"
