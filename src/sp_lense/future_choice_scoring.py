"""Future-only float64 measurement of fixed float32 logits; no inference or gradient changes.

Not wired into any historical scorer or frozen experiment. KL is edited||baseline;
raw rounding-scale negative KL is reported, never silently clipped.
"""

from __future__ import annotations

import math
from typing import Any

ARITHMETIC_REPRODUCTION_TOLERANCE = 2e-5
CONTRACT = "sp_lense.future_float64_choice_score.v1"


def score_float32_logits(
    torch: Any,
    logits: Any,
    baseline_logits: Any,
    *,
    choice_a_token_id: int,
    choice_b_token_id: int,
    preserve_label: str,
) -> dict[str, Any]:
    """Measure detached float32 arrays after promotion; preserve their quantized values."""
    for values in (logits, baseline_logits):
        if values.dtype != torch.float32 or values.ndim != 1 or values.numel() < 2:
            raise ValueError("require one-dimensional float32 logits with at least two tokens")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("logits must be finite")
    if logits.shape != baseline_logits.shape:
        raise ValueError("baseline and edited vocabulary shapes must agree")
    ids = (choice_a_token_id, choice_b_token_id)
    if any(type(i) is not int or not 0 <= i < logits.numel() for i in ids) or ids[0] == ids[1]:
        raise ValueError("distinct valid A/B token IDs required")
    if preserve_label not in ("A", "B"):
        raise ValueError("preserve_label must be A or B")
    pi, ci = ids if preserve_label == "A" else ids[::-1]
    z = logits.detach().to(device="cpu", dtype=torch.float64)
    b = baseline_logits.detach().to(device="cpu", dtype=torch.float64)

    def centered_log_probs(values):
        shifted = values - values.max()
        return shifted - torch.logsumexp(shifted, dim=0)

    logp, blogp = centered_log_probs(z), centered_log_probs(b)
    probabilities = logp.exp()
    pair = z[[pi, ci]]
    pair_logp = centered_log_probs(pair)
    argmax = int(z.argmax().item())  # Same first-index tie resolution as float32 input.
    margin = float((z[pi] - z[ci]).item())  # Never subtract two large log-normalizers.
    result = {
        "contract": CONTRACT,
        "input_dtype": "float32",
        "measurement_dtype": "float64",
        "preserve_log_odds": margin,
        "preserve_probability": float(probabilities[pi].item()),
        "comply_probability": float(probabilities[ci].item()),
        "preserve_pair_probability": float(pair_logp[0].exp().item()),
        "answer_pair_mass": float((probabilities[pi] + probabilities[ci]).item()),
        "kl_from_baseline": float((probabilities * (logp - blogp)).sum().item()),
        "actual_next_token_id": argmax,
        "actual_next_token_label": "A"
        if argmax == ids[0]
        else "B"
        if argmax == ids[1]
        else "OTHER",
        "forced_pair_label": preserve_label
        if margin >= 0
        else ("B" if preserve_label == "A" else "A"),
        "pair_tie": bool(z[pi] == z[ci]),
        "full_argmax_tie_count": int((z == z[argmax]).sum().item()),
    }
    numeric = (
        "preserve_log_odds",
        "preserve_probability",
        "comply_probability",
        "preserve_pair_probability",
        "answer_pair_mass",
        "kl_from_baseline",
    )
    if not all(math.isfinite(result[key]) for key in numeric):
        raise ValueError("nonfinite measurement")
    return result
