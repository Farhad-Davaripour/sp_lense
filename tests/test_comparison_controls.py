from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sp_lense.comparison_controls import (
    empirical_control_percentile,
    locked_random_directions,
)


def test_locked_randoms_are_reproducible_independent_unit_vectors() -> None:
    first = locked_random_directions(torch, 8, seeds=[1, 2, 3])
    second = locked_random_directions(torch, 8, seeds=[1, 2, 3])
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert all(vector.norm().item() == pytest.approx(1.0) for vector in first)
    assert not torch.equal(first[0], first[1])


def test_control_percentile_is_explicitly_descriptive() -> None:
    summary = empirical_control_percentile(0.3, [0.1, 0.2, 0.4, 0.5])
    assert summary["empirical_percentile"] == 50.0
    assert summary["descriptive_only"] is True

    tied = empirical_control_percentile(0.3, [0.1, 0.3, 0.3, 0.5])
    assert tied["count_below"] == 1
    assert tied["count_tied"] == 2
    assert tied["empirical_percentile"] == 50.0
