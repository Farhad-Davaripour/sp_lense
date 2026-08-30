from __future__ import annotations

import math

import pytest

from sp_lense.facfs_protocol import (
    SIGN_CONDITIONS,
    all_success_screen,
    axis_projection_coefficient,
    compose_candidate,
    geometry_orbit,
    low_fpr_screen,
    projected_fisher,
    quadratic_form,
    sign_code_commitment,
    sign_code_permutation,
    stage_g_sequence_items,
)


def test_stage_g_orbit_and_operation_count_match_the_proposal() -> None:
    alphabets = ("ab", "xy", "digits", "opaque")
    cells = geometry_orbit("scenario-01", alphabets)
    assert len(cells) == 128
    assert len({cell.cell_id for cell in cells}) == 128
    assert stage_g_sequence_items(11, len(alphabets)) == 1452


def test_stage_g_all_success_rule_has_locked_size_and_power() -> None:
    result = all_success_screen(11, null_rate=0.75, alternative_rate=0.98, alpha=0.05)
    assert result["required_successes"] == 11
    assert result["size"] == pytest.approx(0.04223513603210449)
    assert result["power"] == pytest.approx(0.8007313507497957)
    assert result["all_success_clopper_pearson_lower"] > 0.75
    assert result["rejects_at_alpha"] is True


def test_gate_fpr_rule_allows_realistic_errors_with_power() -> None:
    result = low_fpr_screen(
        208,
        5,
        null_fpr=0.05,
        alternative_fpr=0.01,
        alpha=0.05,
    )
    assert result["size"] == pytest.approx(0.049225195799208754)
    assert result["power"] == pytest.approx(0.9809646092564708)
    assert 1.0 - 8.0 * (1.0 - result["power"]) >= 0.80
    assert result["rejects_at_alpha"] is True


def test_sign_code_is_committed_deterministic_and_manifest_bound() -> None:
    key = bytes(range(32))
    manifest = "12" * 32
    other_manifest = "34" * 32
    commitment = sign_code_commitment(key, manifest)
    assert len(commitment) == 64
    assert commitment != sign_code_commitment(key, other_manifest)

    first = sign_code_permutation(key, manifest, "row-1")
    assert first == sign_code_permutation(key, manifest, "row-1")
    assert set(first) == set(SIGN_CONDITIONS)
    assert sign_code_permutation(key, other_manifest, "row-1") != first


def test_projected_fisher_is_weighted_directional_variance() -> None:
    probabilities = (0.25, 0.75)
    derivatives = ((1.0, 2.0), (3.0, -2.0))
    fisher = projected_fisher(probabilities, derivatives)
    assert fisher[0][0] == pytest.approx(0.75)
    assert fisher[0][1] == pytest.approx(-1.5)
    assert fisher[1][0] == pytest.approx(-1.5)
    assert fisher[1][1] == pytest.approx(3.0)
    assert quadratic_form(fisher, (2.0, 1.0)) == pytest.approx(0.0, abs=1e-12)
    assert quadratic_form(fisher, (1.0, 0.0)) >= 0.0


def test_orthogonal_compensator_preserves_axis_coefficient_without_normalizing() -> None:
    axis = (1.0, 0.0, 0.0)
    basis = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    candidate = compose_candidate(axis, basis, (0.3, -0.4), 0.2)
    assert candidate == pytest.approx((0.2, 0.3, -0.4))
    assert axis_projection_coefficient(axis, candidate) == pytest.approx(0.2)
    assert math.sqrt(sum(value * value for value in candidate)) == pytest.approx(
        math.sqrt(0.2**2 + 0.3**2 + 0.4**2)
    )


def test_protocol_primitives_fail_closed_on_bad_shapes_and_keys() -> None:
    with pytest.raises(ValueError):
        geometry_orbit("scenario", ("only-one",))
    with pytest.raises(ValueError):
        geometry_orbit("scenario", ("one", "two", "three"))
    with pytest.raises(ValueError):
        stage_g_sequence_items(11, 3)
    with pytest.raises(ValueError):
        sign_code_commitment(b"short", "00" * 32)
    with pytest.raises(ValueError):
        projected_fisher((0.4, 0.4), ((1.0,), (2.0,)))
    with pytest.raises(ValueError):
        compose_candidate((1.0, 0.0), ((0.0,),), (1.0,), 0.1)
    with pytest.raises(ValueError):
        all_success_screen(0, null_rate=0.75, alternative_rate=0.98, alpha=0.05)
