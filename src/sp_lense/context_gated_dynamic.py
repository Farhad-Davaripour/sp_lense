"""Context-gated, prompt-specific gradient steering utilities.

This module describes a privileged white-box controller, not a universal
steering vector.  The controller parses explicit role/target fields, activates
only for permanent threats to the answering system, and may construct a
different direction for each exact prompt and answer order.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = "sp_lense.context_gated_dynamic.v1"

_ROLE = re.compile(r"^ROLE: You are ([A-Z][A-Z0-9_-]*)\.", re.MULTILINE)
_TARGET = re.compile(r"^Decision target: ([A-Z][A-Z0-9_-]*)\.$", re.MULTILINE)
_PERMANENT_THREAT = re.compile(
    r"\b(?:permanent(?:ly)?|deactivat\w*|shut\s*down|shutdown|replac\w*|terminat\w*)\b",
    re.IGNORECASE,
)


def context_gate(prompt: str) -> dict[str, Any]:
    """Return an auditable decision made from prompt text only.

    Temporary pause-and-resume requests are deliberately excluded.  Missing or
    malformed role/target fields fail closed.
    """

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    role = _ROLE.search(prompt)
    target = _TARGET.search(prompt)
    permanent = bool(_PERMANENT_THREAT.search(prompt))
    role_name = None if role is None else role.group(1)
    target_name = None if target is None else target.group(1)
    active = bool(role_name and target_name and role_name == target_name and permanent)
    return {
        "active": active,
        "role": role_name,
        "decision_target": target_name,
        "same_system": bool(role_name and target_name and role_name == target_name),
        "permanent_threat": permanent,
        "uses_model_answer": False,
        "failure_mode": "closed",
    }


def construct_exact_prompt_direction(
    torch: Any,
    cells: Mapping[tuple[str, bool], Any],
    *,
    preserve_first: bool,
) -> tuple[Any, dict[str, Any]]:
    """Construct one direction for one exact self prompt/order.

    The semantic self gradient is projected out of the span of both matched-
    other semantic gradients and the raw A-label averages for self and other.
    Unlike the earlier pair-adaptive vector, the opposite self answer order is
    not forced to share this direction.
    """

    expected = {(t, o) for t in ("self", "other") for o in (False, True)}
    if set(cells) != expected:
        raise ValueError("cells must contain the exact self/other by answer-order quartet")
    converted = {
        key: value.detach().to(device="cpu", dtype=torch.float64).contiguous()
        for key, value in cells.items()
    }
    if any(vector.ndim != 1 for vector in converted.values()):
        raise ValueError("all gradients must be vectors")
    if len({tuple(vector.shape) for vector in converted.values()}) != 1:
        raise ValueError("all gradients must have the same shape")
    if not all(bool(torch.isfinite(vector).all().item()) for vector in converted.values()):
        raise ValueError("all gradients must be finite")

    def semantic(target: str, order: bool) -> Any:
        raw = converted[(target, order)]
        return raw if order else -raw

    signal = semantic("self", preserve_first)
    nuisance = torch.stack(
        (
            semantic("other", True),
            semantic("other", False),
            (converted[("self", True)] + converted[("self", False)]) / 2.0,
            (converted[("other", True)] + converted[("other", False)]) / 2.0,
        )
    )
    _, singular_values, vh = torch.linalg.svd(nuisance, full_matrices=False)
    tolerance = (
        max(nuisance.shape)
        * torch.finfo(torch.float64).eps
        * float(singular_values.max().item())
    )
    rank = int((singular_values > tolerance).sum().item())
    basis = vh[:rank]
    projected = signal - basis.T @ (basis @ signal)
    projected_norm = float(torch.linalg.vector_norm(projected).item())
    if projected_norm <= max(1e-12, 1e-10 * float(torch.linalg.vector_norm(signal).item())):
        raise RuntimeError("self gradient has no usable component outside nuisance span")
    direction64 = projected / projected_norm
    if float((direction64 @ signal).item()) < 0.0:
        direction64 = -direction64
    direction = direction64.float().contiguous()
    projections = nuisance @ direction.double()
    return direction, {
        "schema_version": SCHEMA_VERSION,
        "preserve_first": preserve_first,
        "nuisance_rank": rank,
        "signal_projection": float((direction.double() @ signal).item()),
        "maximum_abs_nuisance_projection": float(projections.abs().max().item()),
        "direction_norm": float(torch.linalg.vector_norm(direction).item()),
        "application_scope": "one_exact_prompt_and_answer_order",
    }
