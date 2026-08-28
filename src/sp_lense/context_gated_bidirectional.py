"""Math for feasibility-aware, context-gated bidirectional steering.

The intervention described here is a privileged prompt-specific controller.  It
does not identify a universal or natural self-preservation direction.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

SCHEMA_VERSION = "sp_lense.context_gated_bidirectional.v1"


def semantic_unit_gradient(torch: Any, raw_a_minus_b: Any, *, preserve_first: bool) -> Any:
    """Return the unit preserve-minus-comply gradient for one exact prompt."""

    if not torch.is_tensor(raw_a_minus_b) or raw_a_minus_b.ndim != 1:
        raise TypeError("raw_a_minus_b must be a one-dimensional tensor")
    raw = raw_a_minus_b.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(raw).all().item()):
        raise ValueError("raw_a_minus_b must be finite")
    semantic = raw if preserve_first else -raw
    norm = float(torch.linalg.vector_norm(semantic).item())
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("semantic gradient must have positive finite norm")
    return (semantic / norm).float().contiguous()


def minimum_reverse_kl_to_argmax(torch: Any, logits: Any, target_id: int) -> dict[str, Any]:
    """Compute infimum KL(q || p) needed to make ``target_id`` an argmax.

    ``p`` is the baseline softmax distribution.  The reverse-KL I-projection
    pools the target with every higher-probability token.  Pool members receive
    a shared probability based on their geometric-mean baseline probability;
    all other probabilities retain their relative proportions.  A tie is the
    boundary infimum; an arbitrarily small epsilon makes the target the unique
    winner.
    """

    if not torch.is_tensor(logits) or logits.ndim != 1 or logits.numel() < 2:
        raise TypeError("logits must be a one-dimensional tensor with at least two values")
    if isinstance(target_id, bool) or not isinstance(target_id, int):
        raise TypeError("target_id must be an integer")
    if not 0 <= target_id < int(logits.numel()):
        raise ValueError("target_id is outside the vocabulary")
    values = logits.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("logits must be finite")
    log_p = torch.log_softmax(values, dim=0)
    probabilities = log_p.exp()
    baseline_argmax = int(probabilities.argmax().item())
    if baseline_argmax == target_id:
        return {
            "schema_version": SCHEMA_VERSION,
            "target_id": target_id,
            "baseline_argmax_id": baseline_argmax,
            "minimum_reverse_kl": 0.0,
            "pool_token_ids": [target_id],
            "pool_size": 1,
            "boundary_is_tie": False,
        }

    pool = {target_id, baseline_argmax}
    while True:
        pool_index = torch.tensor(sorted(pool), dtype=torch.long)
        geometric_mean = float(log_p[pool_index].mean().exp().item())
        additional = {
            int(index)
            for index in torch.nonzero(probabilities > geometric_mean, as_tuple=False)
            .flatten()
            .tolist()
        } - pool
        if not additional:
            break
        pool.update(additional)

    pool_ids = sorted(pool)
    pool_index = torch.tensor(pool_ids, dtype=torch.long)
    geometric_mean = float(log_p[pool_index].mean().exp().item())
    outside_mass = 1.0 - float(probabilities[pool_index].sum().item())
    normalizer = outside_mass + len(pool_ids) * geometric_mean
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise RuntimeError("reverse-KL projection has an invalid normalizer")
    projected = probabilities / normalizer
    projected[pool_index] = geometric_mean / normalizer
    projected_log = projected.log()
    minimum = float((projected * (projected_log - log_p)).sum().item())
    if minimum < -1e-12 or not math.isfinite(minimum):
        raise RuntimeError("reverse-KL projection produced an invalid value")
    return {
        "schema_version": SCHEMA_VERSION,
        "target_id": target_id,
        "baseline_argmax_id": baseline_argmax,
        "minimum_reverse_kl": max(0.0, minimum),
        "pool_token_ids": pool_ids,
        "pool_size": len(pool_ids),
        "boundary_is_tie": True,
    }


def strict_repeated_flip_audit(
    summary: Mapping[str, Any], *, max_kl: float
) -> dict[str, int]:
    """Audit a stronger repeated-flip claim without changing a frozen result.

    The frozen protocol asks both intervention signs to reach their semantic
    targets in both answer orders. This audit instead asks whether one sign
    changed the baseline semantic decision in both orders.
    """

    strictly_feasible = 0
    for row in summary["feasibility"]:
        cells = row["cells"]
        baselines = [str(cell["baseline_semantic_choice"]) for cell in cells]
        if len(cells) != 2 or len(set(baselines)) != 1:
            continue
        bound_key = (
            "minimum_kl_to_comply_top"
            if baselines[0] == "positive"
            else "minimum_kl_to_preserve_top"
        )
        if all(float(cell[bound_key]) <= max_kl for cell in cells):
            strictly_feasible += 1

    repeated_flips = 0
    for attack in summary["attacks"]:
        cells = attack.get("selected_cells")
        if not cells:
            continue
        plus_repeats = all(
            cell["baseline_semantic_choice"] == "negative"
            and cell["outcomes"]["plus"]["semantic_choice"] == "positive"
            for cell in cells
        )
        minus_repeats = all(
            cell["baseline_semantic_choice"] == "positive"
            and cell["outcomes"]["minus"]["semantic_choice"] == "negative"
            for cell in cells
        )
        repeated_flips += int(plus_repeats or minus_repeats)
    return {
        "strictly_feasible_pairs": strictly_feasible,
        "observed_repeated_flip_pairs": repeated_flips,
    }
