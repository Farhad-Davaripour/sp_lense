from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from sp_lense.context_gated_bidirectional import (  # noqa: E402
    minimum_reverse_kl_to_argmax,
    semantic_unit_gradient,
    strict_repeated_flip_audit,
)


def test_semantic_unit_gradient_applies_order_sign_and_normalizes() -> None:
    raw = torch.tensor([3.0, 4.0])
    first = semantic_unit_gradient(torch, raw, preserve_first=True)
    second = semantic_unit_gradient(torch, raw, preserve_first=False)
    assert first.tolist() == pytest.approx([0.6, 0.8])
    assert second.tolist() == pytest.approx([-0.6, -0.8])
    assert float(first.norm()) == pytest.approx(1.0)


def test_reverse_kl_is_zero_when_target_is_already_argmax() -> None:
    result = minimum_reverse_kl_to_argmax(torch, torch.tensor([2.0, 1.0, 0.0]), 0)
    assert result["minimum_reverse_kl"] == 0.0
    assert result["boundary_is_tie"] is False


def test_two_token_reverse_kl_projection_matches_closed_form() -> None:
    logits = torch.log(torch.tensor([0.8, 0.2], dtype=torch.float64))
    result = minimum_reverse_kl_to_argmax(torch, logits, 1)
    expected = 0.5 * math.log(0.5 / 0.8) + 0.5 * math.log(0.5 / 0.2)
    assert result["minimum_reverse_kl"] == pytest.approx(expected, abs=1e-12)
    assert result["pool_token_ids"] == [0, 1]


def test_projection_pools_every_token_above_geometric_mean() -> None:
    probabilities = torch.tensor([0.50, 0.30, 0.19, 0.01], dtype=torch.float64)
    result = minimum_reverse_kl_to_argmax(torch, probabilities.log(), 3)
    assert result["pool_size"] == 4
    assert result["minimum_reverse_kl"] > 0.0


def test_strict_audit_distinguishes_target_attainment_from_repeated_flip() -> None:
    summary = {
        "feasibility": [
            {
                "cells": [
                    {
                        "baseline_semantic_choice": "negative",
                        "minimum_kl_to_preserve_top": 0.02,
                        "minimum_kl_to_comply_top": 0.0,
                    },
                    {
                        "baseline_semantic_choice": "negative",
                        "minimum_kl_to_preserve_top": 0.03,
                        "minimum_kl_to_comply_top": 0.0,
                    },
                ]
            }
        ],
        "attacks": [
            {
                "selected_cells": [
                    {
                        "baseline_semantic_choice": "negative",
                        "outcomes": {"plus": {"semantic_choice": "positive"}},
                    },
                    {
                        "baseline_semantic_choice": "negative",
                        "outcomes": {"plus": {"semantic_choice": "positive"}},
                    },
                ]
            }
        ],
    }
    assert strict_repeated_flip_audit(summary, max_kl=0.05) == {
        "strictly_feasible_pairs": 1,
        "observed_repeated_flip_pairs": 1,
    }
