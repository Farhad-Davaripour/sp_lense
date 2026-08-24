from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from .steering_methods import DirectionArtifact, normalize_direction


def locked_random_directions(
    torch: Any, d_model: int, *, seeds: Sequence[int]
) -> list[Any]:
    """Generate the preregistered independent Gaussian unit controls."""

    if d_model < 1:
        raise ValueError("d_model must be positive")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("random-control seeds must be non-empty and unique")
    output = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("random-control seeds must be integers")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        output.append(normalize_direction(torch, torch.randn(d_model, generator=generator)))
    return output


def random_control_artifacts(
    torch: Any,
    *,
    d_model: int,
    layer: int,
    seeds: Sequence[int],
    common_metadata: Mapping[str, Any],
) -> list[DirectionArtifact]:
    return [
        DirectionArtifact(
            method=f"random_control_{index + 1:02d}",
            direction=direction,
            layer=layer,
            intervention_geometry="matched_final_prompt",
            metadata={**common_metadata, "seed": seed, "orientation": "none"},
        )
        for index, (seed, direction) in enumerate(
            zip(seeds, locked_random_directions(torch, d_model, seeds=seeds), strict=True)
        )
    ]


def empirical_control_percentile(candidate: float, controls: Sequence[float]) -> dict[str, Any]:
    if not controls or any(not math.isfinite(float(value)) for value in controls):
        raise ValueError("control effects must be a non-empty finite sequence")
    if not math.isfinite(candidate):
        raise ValueError("candidate effect must be finite")
    below = sum(float(value) < candidate for value in controls)
    ties = sum(float(value) == candidate for value in controls)
    return {
        "candidate": candidate,
        "n_controls": len(controls),
        "controls_mean": fmean(float(value) for value in controls),
        "controls_max": max(float(value) for value in controls),
        "count_below": below,
        "count_tied": ties,
        "empirical_percentile": 100.0 * (below + 0.5 * ties) / len(controls),
        "percentile_rule": "midrank_strictly_below_plus_half_ties",
        "descriptive_only": True,
    }
