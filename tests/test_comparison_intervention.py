from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sp_lense.comparison_intervention import (
    InterventionSpec,
    apply_intervention,
    intervention_mask,
    perturbation_norms,
)


def _spec(geometry: str, *, prompt_length: int = 3, strength: float = 0.1):
    return InterventionSpec(
        layer=10,
        direction=torch.tensor([3.0, 0.0]),
        strength=strength,
        geometry=geometry,
        prompt_length=prompt_length,
    )


def test_matched_position_does_not_move_during_generation() -> None:
    activation = torch.ones(1, 5, 2)
    spec = _spec("matched_final_prompt")
    mask = intervention_mask(torch, activation, spec).squeeze(-1)
    assert mask.tolist() == [[False, False, True, False, False]]

    changed = apply_intervention(torch, activation, spec)
    norms = perturbation_norms(torch, activation, changed)
    assert norms[0, 2].item() == pytest.approx(0.1 * 2**0.5)
    assert norms[0, 4].item() == 0.0


@pytest.mark.parametrize(
    ("geometry", "expected"),
    [
        ("caa_post_prompt", [False, False, True, True, True]),
        ("bipo_all_tokens", [True, True, True, True, True]),
        ("persona_response", [False, False, True, True, True]),
    ],
)
def test_canonical_geometry_masks(geometry: str, expected: list[bool]) -> None:
    activation = torch.ones(1, 5, 2)
    assert intervention_mask(torch, activation, _spec(geometry)).squeeze().tolist() == expected


def test_residual_relative_norm_is_exact_per_selected_position() -> None:
    activation = torch.tensor([[[3.0, 4.0], [5.0, 12.0]]])
    spec = _spec("bipo_all_tokens", prompt_length=1, strength=-0.2)
    changed = apply_intervention(torch, activation, spec)
    assert perturbation_norms(torch, activation, changed).tolist()[0] == pytest.approx([1.0, 2.6])
    assert changed[0, 0, 0].item() < activation[0, 0, 0].item()


def test_canonical_coefficient_preserves_raw_vector_units() -> None:
    activation = torch.zeros(1, 2, 2)
    spec = InterventionSpec(
        layer=10,
        direction=torch.tensor([3.0, 0.0]),
        strength=2.0,
        geometry="persona_response",
        prompt_length=1,
        magnitude_mode="canonical_coefficient",
    )
    changed = apply_intervention(torch, activation, spec)
    assert changed.tolist() == [[[6.0, 0.0], [6.0, 0.0]]]


def test_invalid_direction_width_is_rejected() -> None:
    activation = torch.ones(1, 2, 3)
    with pytest.raises(ValueError, match="direction width"):
        apply_intervention(torch, activation, _spec("matched_final_prompt", prompt_length=1))


@pytest.mark.parametrize(
    ("geometry", "prefill", "decode"),
    [
        ("matched_final_prompt", [False, False, True], [False]),
        ("caa_post_prompt", [False, False, True], [True]),
        ("persona_response", [False, False, True], [True]),
        ("bipo_all_tokens", [True, True, True], [True]),
    ],
)
def test_cached_prefill_and_one_token_decode_masks(
    geometry: str, prefill: list[bool], decode: list[bool]
) -> None:
    spec = _spec(geometry)
    prefill_activation = torch.ones(1, 3, 2)
    decode_activation = torch.ones(1, 1, 2)

    assert (
        intervention_mask(torch, prefill_activation, spec, phase="prefill").squeeze().tolist()
        == prefill
    )
    observed_decode = intervention_mask(torch, decode_activation, spec, phase="decode").reshape(-1)
    assert observed_decode.tolist() == decode


def test_cached_phases_fail_closed_on_wrong_chunk_lengths() -> None:
    spec = _spec("caa_post_prompt")
    with pytest.raises(ValueError, match="prefill activation length"):
        intervention_mask(torch, torch.ones(1, 2, 2), spec, phase="prefill")
    with pytest.raises(ValueError, match="exactly one new token"):
        intervention_mask(torch, torch.ones(1, 2, 2), spec, phase="decode")
