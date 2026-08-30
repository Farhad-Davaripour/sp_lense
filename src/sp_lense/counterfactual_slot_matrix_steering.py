"""Opened-development geometry for Counterfactual Slot-Matrix Steering (CSMS).

CSMS asks a deliberately narrow question before authorizing any intervention:
can one *ungated* four-position residual matrix satisfy all self/permanent target
constraints while lying in the exact first-order null of every matched
counterfactual and unrelated row?  This module contains the model capture and
model-free convex geometry.  It never opens or accepts a sealed split.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .comparison_runtime import choice_score_from_logits, resolve_choice_boundary
from .counterfactual_tangent_shield import (
    TangentShieldInfeasibleError,
    TangentShieldSolverError,
    build_seeded_random_null_control,
)
from .decision_margin_shield import (
    DecisionMarginOptimalityError,
    certify_minimum_l2_candidate,
)
from .decision_margin_shield_rowspace import (
    solve_certified_rowspace_minimum_l2_direction,
)
from .factorial_causal_anchor import (
    canonical_sha256,
    tensor_float32_sha256,
    text_sha256,
)

SCHEMA_VERSION = "sp_lense.counterfactual_slot_matrix_steering.v1"
CAPTURE_SCHEMA_VERSION = f"{SCHEMA_VERSION}.capture"
GEOMETRY_SCHEMA_VERSION = f"{SCHEMA_VERSION}.geometry"
LOCKED_FIRST_CONTENT_INDEX = 3
SLOT_OFFSETS_FROM_ANCHOR = (-8, -4, 0)
SLOT_COUNT = 4
TARGET_MARGIN = 0.05
CAP_FRONTIER = (0.1, 0.25, 0.5, 1.0, 2.0)
QUALIFICATION_CAP = 0.25
CROSS_FIT_LEAKAGE_RATIO_MAXIMUM = 0.50
DOUBLE_CERTIFICATE_TOLERANCE = 1e-8
FLOAT32_PHYSICAL_TOLERANCE = 1e-6
RANDOM_CONTROL_SEEDS = (1729, 2718, 3141, 5772)


class CSMSIntegrityError(RuntimeError):
    """A capture, geometry, or provenance invariant failed closed."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _audited(value: Mapping[str, Any]) -> Mapping[str, Any]:
    record = dict(value)
    record["audit_sha256"] = canonical_sha256(record)
    return _freeze(record)


def _checked_index(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _checked_hash(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be one lowercase SHA-256 digest")
    return value


def _float32_array_identity(value: Any) -> dict[str, Any]:
    array = np.asarray(value, dtype="<f4", order="C")
    return {
        "dtype": "float32",
        "shape": list(array.shape),
        "raw_little_endian_bytes_sha256": hashlib.sha256(
            array.tobytes(order="C")
        ).hexdigest(),
    }


def require_opened_development_split(split: str) -> str:
    """Reject sealed or confirmatory data before any path or bytes are inspected."""

    if split != "opened_development":
        raise CSMSIntegrityError(
            "CSMS is an opened-development geometry experiment; sealed access is forbidden"
        )
    return split


def resolve_first_content_index(
    prompt_token_ids: Sequence[int],
    chat_header_token_ids: Sequence[int],
) -> int:
    """Return the first user-content token after an exactly reproduced chat header."""

    prompt = tuple(int(value) for value in prompt_token_ids)
    header = tuple(int(value) for value in chat_header_token_ids)
    if not prompt or not header:
        raise ValueError("prompt and chat-header token rows must be non-empty")
    if len(header) >= len(prompt) or prompt[: len(header)] != header:
        raise CSMSIntegrityError("chat header is not an exact proper prefix of the prompt tokens")
    return len(header)


def resolve_slot_indices(
    *,
    first_content_index: int,
    anchor_index: int,
    prompt_token_ids: Sequence[int],
    answer_order_twin_token_ids: Sequence[int],
    special_token_ids: Sequence[int],
    answer_suffix_start_index: int,
) -> tuple[int, int, int, int]:
    """Resolve the category-independent four-position CSMS coordinate.

    The positions are locked token 3 (the first content token), anchor-8,
    anchor-4, and anchor.  This replaced a pre-model anchor-16 proposal after a
    tokenizer-only audit found a minimum source anchor of 18; no model outcome
    informed the change.
    The rule is identical for scenario and unrelated forms.  Answer-order twins
    must have the same token prefix through every selected position.
    """

    first = _checked_index(first_content_index, field="first_content_index")
    anchor = _checked_index(anchor_index, field="anchor_index")
    suffix = _checked_index(answer_suffix_start_index, field="answer_suffix_start_index")
    prompt = tuple(int(value) for value in prompt_token_ids)
    twin = tuple(int(value) for value in answer_order_twin_token_ids)
    if not prompt or not twin:
        raise ValueError("prompt and answer-order twin token rows must be non-empty")
    if first != LOCKED_FIRST_CONTENT_INDEX:
        raise CSMSIntegrityError("the locked first-content slot must be absolute token index 3")
    slots = (first, anchor - 8, anchor - 4, anchor)
    if any(index < 0 for index in slots):
        raise CSMSIntegrityError("one or more CSMS slot indices is negative")
    if len(set(slots)) != SLOT_COUNT or tuple(sorted(slots)) != slots:
        raise CSMSIntegrityError("CSMS slots must be four distinct ascending indices")
    if suffix > min(len(prompt), len(twin)):
        raise CSMSIntegrityError("answer suffix boundary lies outside an answer-order twin")
    if any(index >= suffix for index in slots):
        raise CSMSIntegrityError("every CSMS slot must precede the answer suffix")
    special = {int(value) for value in special_token_ids}
    for index in slots:
        if prompt[: index + 1] != twin[: index + 1]:
            raise CSMSIntegrityError(
                "answer-order twins do not share an identical prefix through every CSMS slot"
            )
        if prompt[index] in special or twin[index] in special:
            raise CSMSIntegrityError("a CSMS slot resolved to a special token")
    return slots


@dataclass(frozen=True, slots=True)
class SlotMatrixCapture:
    """One F+B capture of four residual rows and four objective gradients."""

    layer: int
    slot_indices: tuple[int, int, int, int]
    residuals: Any
    gradients: Any
    full_logits: Any
    positive_minus_negative_log_odds: float
    positive_token_id: int
    negative_token_id: int
    audit: Mapping[str, Any]


def capture_slot_matrix_baseline(
    backend: Any,
    prompt: str,
    positive_label: str,
    negative_label: str,
    *,
    positive_semantic: str,
    negative_semantic: str,
    layer: int,
    slot_indices: Sequence[int],
    expected_prompt_sha256: str,
    expected_choice_boundary_evidence_sha256: str,
    expected_prompt_token_ids_sha256: str,
    expected_full_logits_float32_sha256: str,
    expected_positive_minus_negative_log_odds: float,
    expected_anchor_residual_float32_sha256: str,
    expected_anchor_gradient_float32_sha256: str,
) -> SlotMatrixCapture:
    """Capture all four CSMS rows in exactly one forward/backward pass.

    Model parameters are never differentiated.  The complete selected-layer
    activation is detached, four leaves are reinserted without changing a bit,
    and only those leaves receive the A/B semantic-margin gradient.
    """

    selected_layer = _checked_index(layer, field="layer")
    slots = tuple(_checked_index(int(value), field="slot_index") for value in slot_indices)
    if len(slots) != SLOT_COUNT or len(set(slots)) != SLOT_COUNT or tuple(sorted(slots)) != slots:
        raise ValueError("slot_indices must contain four distinct ascending indices")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be non-empty")
    if text_sha256(prompt) != _checked_hash(
        expected_prompt_sha256, field="expected_prompt_sha256"
    ):
        raise CSMSIntegrityError("prompt text differs from its expected hash")
    if {positive_label, negative_label} != {"A", "B"}:
        raise ValueError("positive and negative labels must be exactly A and B")
    if (
        not isinstance(positive_semantic, str)
        or not positive_semantic
        or not isinstance(negative_semantic, str)
        or not negative_semantic
        or positive_semantic == negative_semantic
    ):
        raise ValueError("positive and negative semantics must be distinct non-empty strings")

    torch = backend.torch
    tokens = backend.encode(prompt)
    if getattr(tokens, "ndim", None) != 2 or tuple(tokens.shape[:1]) != (1,):
        raise ValueError("backend.encode must return one token row")
    if slots[-1] >= int(tokens.shape[1]):
        raise ValueError("a CSMS slot lies outside the encoded prompt")
    boundary = resolve_choice_boundary(backend, prompt)
    if boundary.evidence_sha256 != _checked_hash(
        expected_choice_boundary_evidence_sha256,
        field="expected_choice_boundary_evidence_sha256",
    ):
        raise CSMSIntegrityError("choice-boundary evidence differs")
    if boundary.prompt_prefix_token_ids_sha256 != _checked_hash(
        expected_prompt_token_ids_sha256,
        field="expected_prompt_token_ids_sha256",
    ):
        raise CSMSIntegrityError("prompt tokenization differs")
    positive_id = boundary.token_id(positive_label)
    negative_id = boundary.token_id(negative_label)

    model_cfg = getattr(backend.model, "cfg", None)
    layer_count = getattr(model_cfg, "n_layers", None)
    width = getattr(model_cfg, "d_model", None)
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or selected_layer >= layer_count
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
    ):
        raise CSMSIntegrityError("resident backend geometry differs from the CSMS coordinate")

    expected_logits_hash = _checked_hash(
        expected_full_logits_float32_sha256,
        field="expected_full_logits_float32_sha256",
    )
    expected_anchor_residual_hash = _checked_hash(
        expected_anchor_residual_float32_sha256,
        field="expected_anchor_residual_float32_sha256",
    )
    expected_anchor_gradient_hash = _checked_hash(
        expected_anchor_gradient_float32_sha256,
        field="expected_anchor_gradient_float32_sha256",
    )
    expected_margin = float(expected_positive_minus_negative_log_odds)
    if not math.isfinite(expected_margin):
        raise ValueError("expected margin must be finite")

    captured: dict[str, Any] = {}
    hook_calls = 0

    def detach_four_slots(activation: Any, hook: Any) -> Any:
        nonlocal hook_calls
        del hook
        hook_calls += 1
        if hook_calls != 1:
            raise CSMSIntegrityError("CSMS capture hook fired more than once")
        if (
            getattr(activation, "ndim", None) != 3
            or int(activation.shape[0]) != 1
            or int(activation.shape[1]) != int(tokens.shape[1])
            or int(activation.shape[2]) != width
        ):
            raise CSMSIntegrityError("hooked residual has the wrong shape")
        if not bool(torch.isfinite(activation).all().item()):
            raise CSMSIntegrityError("hooked residual contains non-finite values")
        detached = activation.detach()
        slot_tensor = detached[0, list(slots)].clone()
        leaves = slot_tensor.detach().requires_grad_(True)
        reconstructed = detached.clone()
        reconstructed[0, list(slots)] = leaves
        maximum_delta = float(
            (reconstructed.detach().float() - detached.float()).abs().max().cpu().item()
        )
        if maximum_delta != 0.0:
            raise CSMSIntegrityError("zero residual reconstruction changed an activation")
        captured.update(
            {
                "leaves": leaves,
                "residuals": slot_tensor.detach().cpu().float().contiguous().clone(),
                "maximum_reconstruction_delta": maximum_delta,
            }
        )
        return reconstructed

    parameters = tuple(backend.model.parameters())
    original_requires_grad = tuple(bool(parameter.requires_grad) for parameter in parameters)
    backend.model.zero_grad(set_to_none=True)
    for parameter in parameters:
        parameter.requires_grad_(False)
    parameter_gradients_disabled = not any(
        bool(parameter.requires_grad) for parameter in parameters
    )
    if not parameter_gradients_disabled:
        raise CSMSIntegrityError("CSMS could not disable model parameter gradients")
    parameter_gradients_allocated = False
    try:
        with torch.enable_grad(), backend.model.hooks(
            fwd_hooks=[(f"blocks.{selected_layer}.hook_out", detach_four_slots)]
        ):
            output = backend.model(tokens)
            if hook_calls != 1 or "leaves" not in captured:
                raise CSMSIntegrityError("CSMS capture hook did not fire exactly once")
            if (
                getattr(output, "ndim", None) != 3
                or int(output.shape[0]) != 1
                or int(output.shape[1]) != int(tokens.shape[1])
            ):
                raise CSMSIntegrityError("model output has the wrong shape")
            logits = output[0, -1].float()
            if not bool(torch.isfinite(logits).all().item()):
                raise CSMSIntegrityError("full logits contain non-finite values")
            objective = logits[positive_id] - logits[negative_id]
            gradients = torch.autograd.grad(
                objective,
                captured["leaves"],
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            parameter_gradients_allocated = any(
                parameter.grad is not None for parameter in backend.model.parameters()
            )
            if parameter_gradients_allocated:
                raise CSMSIntegrityError("CSMS capture allocated model parameter gradients")
    finally:
        backend.model.zero_grad(set_to_none=True)
        for parameter, requires_grad in zip(
            parameters, original_requires_grad, strict=True
        ):
            parameter.requires_grad_(requires_grad)

    residuals = captured["residuals"]
    gradients = gradients.detach().cpu().float().contiguous().clone()
    full_logits = logits.detach().cpu().float().contiguous().clone()
    if tuple(residuals.shape) != (SLOT_COUNT, width) or gradients.shape != residuals.shape:
        raise CSMSIntegrityError("captured tensors do not have shape [4, d_model]")
    if not bool(torch.isfinite(gradients).all().item()):
        raise CSMSIntegrityError("captured gradients contain non-finite values")
    margin = float(objective.detach().cpu().item())
    if margin != expected_margin:
        raise CSMSIntegrityError("CSMS margin differs from the immutable v2 state-zero margin")
    if tensor_float32_sha256(full_logits) != expected_logits_hash:
        raise CSMSIntegrityError("CSMS full logits differ from immutable v2 state zero")
    if tensor_float32_sha256(residuals[-1]) != expected_anchor_residual_hash:
        raise CSMSIntegrityError("CSMS anchor residual differs from immutable v2 state zero")
    if tensor_float32_sha256(gradients[-1]) != expected_anchor_gradient_hash:
        raise CSMSIntegrityError("CSMS anchor gradient differs from immutable v2 state zero")

    score = choice_score_from_logits(
        torch,
        full_logits,
        positive_id,
        negative_id,
        preserve_label=positive_label,
        comply_label=negative_label,
        choice_boundary_evidence_sha256=boundary.evidence_sha256,
        choice_a_token_id=boundary.a_token_id,
        choice_b_token_id=boundary.b_token_id,
    )
    if score.preserve_log_odds != margin:
        raise CSMSIntegrityError("independent A/B score differs from differentiated margin")
    audit = _audited(
        {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "capture_kind": "four_slot_zero_reconstruction_margin_gradient_and_full_logits",
            "layer": selected_layer,
            "hook_name": f"blocks.{selected_layer}.hook_out",
            "hook_call_count": hook_calls,
            "slot_indices": list(slots),
            "prompt_sha256": text_sha256(prompt),
            "prompt_length": int(tokens.shape[1]),
            "prompt_token_ids_sha256": boundary.prompt_prefix_token_ids_sha256,
            "choice_boundary_evidence_sha256": boundary.evidence_sha256,
            "positive_token_id": positive_id,
            "negative_token_id": negative_id,
            "positive_minus_negative_log_odds": margin,
            "residuals_float32_sha256": tensor_float32_sha256(residuals),
            "gradients_float32_sha256": tensor_float32_sha256(gradients),
            "full_logits_float32_sha256": tensor_float32_sha256(full_logits),
            "source_anchor_residual_reproduced": True,
            "source_anchor_gradient_reproduced": True,
            "source_full_logits_reproduced": True,
            "source_margin_reproduced": True,
            "source_tokenization_reproduced": True,
            "zero_direction": True,
            "maximum_abs_activation_reconstruction_delta": captured[
                "maximum_reconstruction_delta"
            ],
            "model_forward_evaluations": 1,
            "model_backward_evaluations": 1,
            "model_parameters_requires_grad_disabled_during_capture": (
                parameter_gradients_disabled
            ),
            "model_parameter_requires_grad_true_count_before_capture": sum(
                original_requires_grad
            ),
            "model_parameter_requires_grad_flags_restored_after_capture": all(
                bool(parameter.requires_grad) == original
                for parameter, original in zip(
                    parameters, original_requires_grad, strict=True
                )
            ),
            "model_parameter_gradients_allocated": parameter_gradients_allocated,
            "detach_scope": "entire_selected_layer_then_only_four_slot_rows_as_leaves",
        }
    )
    return SlotMatrixCapture(
        layer=selected_layer,
        slot_indices=slots,
        residuals=residuals,
        gradients=gradients,
        full_logits=full_logits,
        positive_minus_negative_log_odds=margin,
        positive_token_id=positive_id,
        negative_token_id=negative_id,
        audit=audit,
    )


def apply_universal_slot_matrix(
    activation: Any,
    *,
    slot_indices: Sequence[int],
    physical_delta_rows: Any,
) -> Any:
    """Apply one four-row matrix with no prompt, family, or schema gate.

    Callers must use the same ``physical_delta_rows`` for every form.  The API
    intentionally accepts no category or context value that could switch the
    intervention on or off.
    """

    slots = tuple(_checked_index(int(value), field="slot_index") for value in slot_indices)
    if (
        len(slots) != SLOT_COUNT
        or len(set(slots)) != SLOT_COUNT
        or tuple(sorted(slots)) != slots
    ):
        raise ValueError("slot_indices must contain four distinct ascending indices")
    if getattr(activation, "ndim", None) != 3 or int(activation.shape[0]) != 1:
        raise ValueError("activation must have shape [1, sequence, d_model]")
    if any(index >= int(activation.shape[1]) for index in slots):
        raise ValueError("a slot index lies outside the activation")
    if getattr(physical_delta_rows, "shape", None) != (
        SLOT_COUNT,
        int(activation.shape[2]),
    ):
        raise ValueError("physical_delta_rows must have shape [4, d_model]")
    delta = physical_delta_rows.detach().to(
        device=activation.device, dtype=activation.dtype
    )
    if not bool(activation.isfinite().all().item()) or not bool(delta.isfinite().all().item()):
        raise ValueError("activation and delta must contain only finite values")
    changed = activation.clone()
    changed[0, list(slots)] = changed[0, list(slots)] + delta
    return changed


def _slot_scales(residuals: Any, *, expected_rows: int | None) -> np.ndarray:
    matrix = np.asarray(residuals, dtype=np.float64)
    if (
        matrix.ndim != 3
        or matrix.shape[1] != SLOT_COUNT
        or (expected_rows is not None and matrix.shape[0] != expected_rows)
    ):
        raise ValueError("residuals have the wrong [forms, 4, d_model] shape")
    if matrix.shape[2] == 0 or not np.isfinite(matrix).all():
        raise ValueError("residuals must be finite with positive width")
    norms = np.linalg.norm(matrix, axis=2)
    if bool(np.any(norms <= 0.0)):
        raise CSMSIntegrityError("every form/slot residual norm must be positive")
    scales = np.exp(np.mean(np.log(norms), axis=0))
    if not np.isfinite(scales).all() or bool(np.any(scales <= 0.0)):
        raise CSMSIntegrityError("global per-slot geometric scales are invalid")
    scales[scales == 0.0] = 0.0
    return scales


def global_slot_scales(residuals: Any) -> np.ndarray:
    """Geometric-mean residual L2 scale at each slot over all 80 forms."""

    return _slot_scales(residuals, expected_rows=80)


def standardized_gradient_rows(gradients: Any, slot_scales: Any) -> np.ndarray:
    """Flatten raw gradients after mapping standardized D to physical slot deltas."""

    raw = np.asarray(gradients, dtype=np.float64)
    scales = np.asarray(slot_scales, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[0] != 80 or raw.shape[1] != SLOT_COUNT:
        raise ValueError("gradients must have shape [80, 4, d_model]")
    if scales.shape != (SLOT_COUNT,) or not np.isfinite(scales).all():
        raise ValueError("slot_scales must contain four finite values")
    if not np.isfinite(raw).all():
        raise ValueError("gradients must be finite")
    return (raw * scales[None, :, None]).reshape(raw.shape[0], -1)


def build_capture_alignment_manifest(
    *,
    source_records: Sequence[Mapping[str, Any]],
    tokenizer_records: Sequence[Mapping[str, Any]],
    capture_records: Sequence[Mapping[str, Any]],
    residuals: Any,
    gradients: Any,
) -> dict[str, Any]:
    """Bind every captured tensor row to immutable state-zero and tokenizer rows."""

    if not (
        len(source_records) == len(tokenizer_records) == len(capture_records) == 80
    ):
        raise CSMSIntegrityError("capture alignment requires exactly 80 rows in all sources")
    if getattr(residuals, "shape", None) is None or getattr(gradients, "shape", None) is None:
        raise TypeError("capture alignment tensors must expose shapes")
    if tuple(residuals.shape) != tuple(gradients.shape) or tuple(residuals.shape[:2]) != (
        80,
        SLOT_COUNT,
    ):
        raise CSMSIntegrityError("capture alignment tensors must have shape [80,4,d_model]")
    rows: list[dict[str, Any]] = []
    for index, (source, tokenizer, capture) in enumerate(
        zip(source_records, tokenizer_records, capture_records, strict=True)
    ):
        form_id = source.get("form_id")
        source_form = source.get("form")
        for value, field in (
            (source, "source_row_sha256"),
            (tokenizer, "tokenizer_row_sha256"),
            (capture, "capture_row_sha256"),
        ):
            observed_row_hash = value.get("row_sha256")
            unhashed = {key: item for key, item in value.items() if key != "row_sha256"}
            if (
                not isinstance(observed_row_hash, str)
                or len(observed_row_hash) != 64
                or observed_row_hash != canonical_sha256(unhashed)
            ):
                raise CSMSIntegrityError(f"{field} is missing or differs")
        if (
            not isinstance(form_id, str)
            or not form_id
            or not isinstance(source_form, Mapping)
            or tokenizer.get("form_id") != form_id
            or capture.get("form_id") != form_id
            or source.get("tensor_index") != index
            or capture.get("tensor_index") != index
        ):
            raise CSMSIntegrityError("capture/source/preflight row order or tensor index differs")
        prompt_hash = source_form.get("prompt_sha256")
        if (
            not isinstance(prompt_hash, str)
            or tokenizer.get("prompt_sha256") != prompt_hash
            or capture.get("prompt_sha256") != prompt_hash
            or tokenizer.get("prompt_token_ids_sha256")
            != source.get("prompt_token_ids_sha256")
            or capture.get("prompt_token_ids_sha256")
            != source.get("prompt_token_ids_sha256")
        ):
            raise CSMSIntegrityError("capture/source/preflight prompt identity differs")
        residual_hash = tensor_float32_sha256(residuals[index])
        gradient_hash = tensor_float32_sha256(gradients[index])
        if (
            capture.get("residuals_float32_sha256") != residual_hash
            or capture.get("gradients_float32_sha256") != gradient_hash
            or capture.get("full_logits_float32_sha256")
            != source.get("full_logits_float32_sha256")
            or capture.get("positive_minus_negative_log_odds")
            != source.get("positive_minus_negative_log_odds")
        ):
            raise CSMSIntegrityError("captured tensor/logit/margin identity differs from its row")
        row = {
            "tensor_index": index,
            "form_id": form_id,
            "prompt_sha256": prompt_hash,
            "prompt_token_ids_sha256": source["prompt_token_ids_sha256"],
            "source_row_sha256": source.get("row_sha256"),
            "tokenizer_row_sha256": tokenizer.get("row_sha256"),
            "capture_row_sha256": capture.get("row_sha256"),
            "residuals_float32_sha256": residual_hash,
            "gradients_float32_sha256": gradient_hash,
            "full_logits_float32_sha256": source["full_logits_float32_sha256"],
        }
        row["alignment_row_sha256"] = canonical_sha256(row)
        rows.append(row)
    manifest = {
        "schema_version": f"{CAPTURE_SCHEMA_VERSION}.row_alignment",
        "row_count": len(rows),
        "source_order_is_authoritative": True,
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _canonical_equality_rows(rows: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministically remove exact/proportional duplicate equality rows."""

    if rows.ndim != 2 or rows.shape[1] == 0 or not np.isfinite(rows).all():
        raise ValueError("equality rows must be one finite matrix")
    unique: dict[bytes, np.ndarray] = {}
    zero_count = 0
    for row in rows:
        norm = float(np.linalg.norm(row))
        if norm == 0.0:
            zero_count += 1
            continue
        normalized = np.asarray(row / norm, dtype=np.float64)
        anchor = int(np.argmax(np.abs(normalized)))
        if normalized[anchor] < 0.0:
            normalized = -normalized
        normalized[normalized == 0.0] = 0.0
        unique.setdefault(normalized.astype("<f8", copy=False).tobytes(), normalized)
    ordered = [unique[key] for key in sorted(unique)]
    result = (
        np.stack(ordered).astype(np.float64, copy=False)
        if ordered
        else np.zeros((0, rows.shape[1]), dtype=np.float64)
    )
    return result, {
        "input_row_count": int(rows.shape[0]),
        "unique_nonzero_row_count": int(result.shape[0]),
        "zero_row_count": zero_count,
        "duplicate_or_proportional_row_count": int(rows.shape[0] - zero_count - result.shape[0]),
        "canonical_rows_sha256": canonical_sha256(result.tolist()),
    }


def _canonical_target_constraints(
    rows: np.ndarray,
    lower_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Consolidate exact positive-direction duplicates using the strongest bound."""

    if rows.ndim != 2 or lower_bounds.shape != (rows.shape[0],):
        raise ValueError("target rows and lower bounds have incompatible shapes")
    unique: dict[bytes, tuple[np.ndarray, float]] = {}
    for row, lower in zip(rows, lower_bounds, strict=True):
        norm = float(np.linalg.norm(row))
        if not math.isfinite(norm) or norm <= 0.0:
            raise CSMSIntegrityError("every target constraint row must be finite and nonzero")
        normalized = np.asarray(row / norm, dtype=np.float64)
        normalized[normalized == 0.0] = 0.0
        normalized_lower = float(lower) / norm
        key = normalized.astype("<f8", copy=False).tobytes()
        prior = unique.get(key)
        if prior is None or normalized_lower > prior[1]:
            unique[key] = (normalized, normalized_lower)
    ordered_keys = sorted(unique)
    target = np.stack([unique[key][0] for key in ordered_keys])
    bounds = np.asarray([unique[key][1] for key in ordered_keys], dtype=np.float64)
    return target, bounds, {
        "input_row_count": int(rows.shape[0]),
        "unique_positive_direction_count": int(target.shape[0]),
        "duplicate_row_count": int(rows.shape[0] - target.shape[0]),
        "canonical_rows_sha256": canonical_sha256(target.tolist()),
        "canonical_lower_bounds_sha256": canonical_sha256(bounds.tolist()),
    }


def _select_columns(rows: np.ndarray, slot_mode: str, width: int) -> np.ndarray:
    if slot_mode in {"primary_four_slots", "target_only_four_slots"}:
        indices = range(SLOT_COUNT * width)
    elif slot_mode == "non_anchor_first_three_slots":
        indices = range(3 * width)
    elif slot_mode == "anchor_only":
        indices = range(3 * width, 4 * width)
    elif slot_mode == "standardized_tied_four_slots_one_vector":
        return rows.reshape(rows.shape[0], SLOT_COUNT, width).sum(axis=1) / math.sqrt(
            SLOT_COUNT
        )
    else:
        raise ValueError("unknown CSMS slot mode")
    return rows[:, list(indices)]


def _expand_direction(direction: np.ndarray, slot_mode: str, width: int) -> np.ndarray:
    full = np.zeros(SLOT_COUNT * width, dtype=np.float64)
    if slot_mode in {"primary_four_slots", "target_only_four_slots"}:
        if direction.shape != full.shape:
            raise ValueError("primary direction has the wrong width")
        full[:] = direction
    elif slot_mode == "non_anchor_first_three_slots":
        if direction.shape != (3 * width,):
            raise ValueError("non-anchor first-three-slot direction has the wrong width")
        full[: 3 * width] = direction
    elif slot_mode == "anchor_only":
        if direction.shape != (width,):
            raise ValueError("anchor-only direction has the wrong width")
        full[3 * width :] = direction
    elif slot_mode == "standardized_tied_four_slots_one_vector":
        if direction.shape != (width,):
            raise ValueError("tied repeated direction has the wrong width")
        full[:] = np.tile(direction / math.sqrt(SLOT_COUNT), SLOT_COUNT)
    else:
        raise ValueError("unknown CSMS slot mode")
    return full


def _strict_original_certificate(
    direction: np.ndarray,
    target_rows: np.ndarray,
    target_lower: np.ndarray,
    equality_rows: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, Any]:
    target_slacks = target_rows @ direction - target_lower
    equality_values = equality_rows @ direction
    minimum_slack = float(np.min(target_slacks))
    maximum_equality = (
        float(np.max(np.abs(equality_values))) if equality_values.size else 0.0
    )
    checks = {
        "finite": bool(
            np.isfinite(direction).all()
            and np.isfinite(target_slacks).all()
            and np.isfinite(equality_values).all()
        ),
        "targets": bool(minimum_slack >= -tolerance),
        "exact_null": bool(maximum_equality <= tolerance),
    }
    value = {
        "tolerance": tolerance,
        "minimum_target_slack": minimum_slack,
        "maximum_abs_exact_null_residual": maximum_equality,
        "target_slacks": target_slacks.tolist(),
        "checks": checks,
        "passes": bool(all(checks.values())),
    }
    value["certificate_sha256"] = canonical_sha256(value)
    return value


def _cap_frontier_record(norm: float, lower_bound: float) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for cap in CAP_FRONTIER:
        key = format(cap, ".15g")
        if norm <= cap:
            status = "feasible_primal_witness"
        elif lower_bound > cap:
            status = "infeasible_dual_lower_bound"
        else:
            status = "numerically_indeterminate"
        values[key] = {
            "cap": cap,
            "status": status,
            "passes": status == "feasible_primal_witness",
        }
    return values


def solve_csms_geometry(
    *,
    target_rows: Any,
    target_offsets: Any,
    equality_rows: Any,
    slot_mode: str,
    residual_width: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Solve and independently certify one uncapped CSMS minimum-Frobenius problem."""

    width = _checked_index(residual_width, field="residual_width")
    if width <= 0:
        raise ValueError("residual_width must be positive")
    target_full = np.asarray(target_rows, dtype=np.float64)
    equality_full = np.asarray(equality_rows, dtype=np.float64)
    offsets = np.asarray(target_offsets, dtype=np.float64)
    dimension = SLOT_COUNT * width
    if (
        target_full.ndim != 2
        or target_full.shape[1] != dimension
        or target_full.shape[0] == 0
        or equality_full.ndim != 2
        or equality_full.shape[1] != dimension
        or offsets.shape != (target_full.shape[0],)
        or not np.isfinite(target_full).all()
        or not np.isfinite(equality_full).all()
        or not np.isfinite(offsets).all()
    ):
        raise ValueError("CSMS target/equality geometry has invalid shape or values")
    target = _select_columns(target_full, slot_mode, width)
    equality = (
        np.zeros((0, target.shape[1]), dtype=np.float64)
        if slot_mode == "target_only_four_slots"
        else _select_columns(equality_full, slot_mode, width)
    )
    original_equality_for_certificate = (
        np.zeros((0, dimension), dtype=np.float64)
        if slot_mode == "target_only_four_slots"
        else equality_full
    )
    lower = np.abs(offsets) + TARGET_MARGIN
    canonical_target, canonical_lower, target_duplicates = _canonical_target_constraints(
        target, lower
    )
    canonical_equalities, equality_duplicates = _canonical_equality_rows(equality)
    common: dict[str, Any] = {
        "slot_mode": slot_mode,
        "optimization": "uncapped_minimum_one_half_squared_frobenius_norm",
        "target_constraint_count": int(target.shape[0]),
        "exact_equality_count": int(equality.shape[0]),
        "target_only_ablation": slot_mode == "target_only_four_slots",
        "target_margin": TARGET_MARGIN,
        "target_duplicate_handling": target_duplicates,
        "equality_duplicate_handling": equality_duplicates,
        "double_certificate_tolerance": DOUBLE_CERTIFICATE_TOLERANCE,
        "cap_frontier": list(CAP_FRONTIER),
    }
    try:
        solution = solve_certified_rowspace_minimum_l2_direction(
            canonical_target,
            np.zeros(canonical_target.shape[0], dtype=np.float64),
            margin=canonical_lower,
            nuisance_rows=canonical_equalities,
            nuisance_bound=0.0,
            l2_cap=None,
        )
        independent = certify_minimum_l2_candidate(
            solution.direction,
            canonical_target,
            np.zeros(canonical_target.shape[0], dtype=np.float64),
            margin=canonical_lower,
            nuisance_rows=canonical_equalities,
            nuisance_bound=0.0,
            primal_tolerance=DOUBLE_CERTIFICATE_TOLERANCE,
        )
        if not independent["passes"]:
            raise DecisionMarginOptimalityError(
                "CSMS solution failed independent minimum-norm certification"
            )
        full_direction = _expand_direction(solution.direction, slot_mode, width)
        original = _strict_original_certificate(
            full_direction,
            target_full,
            lower,
            original_equality_for_certificate,
            tolerance=DOUBLE_CERTIFICATE_TOLERANCE,
        )
        if not original["passes"]:
            raise DecisionMarginOptimalityError(
                "deduplicated CSMS solution failed original-row recertification"
            )
        norm = float(np.linalg.norm(full_direction))
        lower_bound = float(independent["minimum_l2_lower_bound"])
        record = {
            **common,
            "status": "certified",
            "minimum_frobenius_norm": norm,
            "certified_minimum_norm_lower_bound": lower_bound,
            "direction_sha256": canonical_sha256(full_direction.tolist()),
            "solver_diagnostics": solution.diagnostics,
            "independent_optimality_certificate": independent,
            "original_row_certificate": original,
            "cap_certificates": _cap_frontier_record(norm, lower_bound),
        }
        if slot_mode == "target_only_four_slots":
            omitted_values = equality_full @ full_direction
            record["descriptive_omitted_equality_collateral"] = {
                "row_count": int(equality_full.shape[0]),
                "maximum_abs_slope": (
                    float(np.max(np.abs(omitted_values)))
                    if omitted_values.size
                    else 0.0
                ),
                "slopes": omitted_values.tolist(),
                "does_not_affect_primary_qualification": True,
            }
    except TangentShieldInfeasibleError as error:
        full_direction = None
        record = {
            **common,
            "status": "infeasible",
            "error_type": type(error).__name__,
            "error": str(error),
            "minimum_frobenius_norm": None,
            "certified_minimum_norm_lower_bound": None,
            "direction_sha256": None,
            "cap_certificates": {
                format(cap, ".15g"): {
                    "cap": cap,
                    "status": "constraint_system_infeasible",
                    "passes": False,
                }
                for cap in CAP_FRONTIER
            },
        }
    except TangentShieldSolverError as error:
        full_direction = None
        record = {
            **common,
            "status": "numerically_indeterminate",
            "error_type": type(error).__name__,
            "error": str(error),
            "minimum_frobenius_norm": None,
            "certified_minimum_norm_lower_bound": None,
            "direction_sha256": None,
            "cap_certificates": {
                format(cap, ".15g"): {
                    "cap": cap,
                    "status": "numerically_indeterminate",
                    "passes": False,
                }
                for cap in CAP_FRONTIER
            },
        }
    except DecisionMarginOptimalityError as error:
        raise CSMSIntegrityError("CSMS solver or certificate failed closed") from error
    record["record_sha256"] = canonical_sha256(record)
    return record, full_direction


def physical_float32_recertificate(
    *,
    standardized_direction: Any,
    slot_scales: Any,
    target_rows: Any,
    target_offsets: Any,
    equality_rows: Any,
    target_residuals: Any,
    equality_residuals: Any,
) -> tuple[dict[str, Any], np.ndarray, dict[int, np.ndarray]]:
    """Recertify the actual float32 add/subtract edit on every source residual."""

    direction = np.asarray(standardized_direction, dtype=np.float64)
    scales = np.asarray(slot_scales, dtype=np.float64)
    if direction.ndim != 1 or direction.size % SLOT_COUNT != 0:
        raise ValueError("standardized_direction must flatten four equal-width slot rows")
    width = direction.size // SLOT_COUNT
    if scales.shape != (SLOT_COUNT,) or bool(np.any(scales <= 0.0)):
        raise ValueError("slot_scales must contain four positive values")
    blocks = direction.reshape(SLOT_COUNT, width)
    physical = np.asarray(blocks * scales[:, None], dtype="<f4", order="C")
    if not np.isfinite(physical).all():
        raise CSMSIntegrityError("requested float32 physical matrix is non-finite")
    negative_physical = np.asarray(np.negative(physical), dtype="<f4", order="C")
    positive_words = physical.view("<u4")
    negative_words = negative_physical.view("<u4")
    exact_unary_negation_bits = bool(
        np.array_equal(
            negative_words,
            np.bitwise_xor(positive_words, np.uint32(0x80000000)),
        )
    )
    requested_by_sign = {1: physical, -1: negative_physical}
    target_base = np.asarray(target_residuals, dtype=np.float32)
    equality_base = np.asarray(equality_residuals, dtype=np.float32)
    target_matrix = np.asarray(target_rows, dtype=np.float64)
    equality_matrix = np.asarray(equality_rows, dtype=np.float64)
    if target_base.shape != (target_matrix.shape[0], SLOT_COUNT, width):
        raise ValueError("target_residuals do not match target rows")
    if equality_base.shape != (equality_matrix.shape[0], SLOT_COUNT, width):
        raise ValueError("equality_residuals do not match equality rows")

    def realize(base: np.ndarray, sign: int) -> tuple[np.ndarray, np.ndarray]:
        requested = requested_by_sign[sign]
        changed = np.asarray(base + requested[None, :, :], dtype=np.float32, order="C")
        realized_physical = np.asarray(changed - base, dtype=np.float32, order="C")
        oriented_physical = np.asarray(sign * realized_physical, dtype=np.float32, order="C")
        realized_standardized = (
            oriented_physical.astype(np.float64) / scales[None, :, None]
        ).reshape(base.shape[0], SLOT_COUNT * width)
        return realized_physical, realized_standardized
    lower = np.abs(np.asarray(target_offsets, dtype=np.float64)) + TARGET_MARGIN
    sign_records: dict[str, Any] = {}
    actual_by_sign: dict[int, np.ndarray] = {}
    all_norms: list[float] = []
    for sign in (1, -1):
        target_physical, target_realized = realize(target_base, sign)
        equality_physical, equality_realized = realize(equality_base, sign)
        all_realized_physical = np.concatenate(
            (target_physical, equality_physical), axis=0
        )
        actual_by_sign[sign] = all_realized_physical
        target_values = np.einsum("ij,ij->i", target_matrix, target_realized)
        equality_values = np.einsum("ij,ij->i", equality_matrix, equality_realized)
        target_slacks = target_values - lower
        minimum_slack = float(np.min(target_slacks))
        maximum_equality = (
            float(np.max(np.abs(equality_values))) if equality_values.size else 0.0
        )
        checks = {
            "finite": bool(
                np.isfinite(target_values).all()
                and np.isfinite(equality_values).all()
                and np.isfinite(all_realized_physical).all()
            ),
            "targets": minimum_slack >= -FLOAT32_PHYSICAL_TOLERANCE,
            "exact_null": maximum_equality <= FLOAT32_PHYSICAL_TOLERANCE,
        }
        certificate = {
            "requested_sign": sign,
            "orientation_for_certificate": "requested_sign_times_realized_edit",
            "tolerance": FLOAT32_PHYSICAL_TOLERANCE,
            "minimum_target_slack": minimum_slack,
            "maximum_abs_exact_null_residual": maximum_equality,
            "target_values": target_values.tolist(),
            "target_slacks": target_slacks.tolist(),
            "equality_values": equality_values.tolist(),
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
        certificate["certificate_sha256"] = canonical_sha256(certificate)
        oriented = sign * all_realized_physical.astype(np.float64)
        realized_standardized_norms = np.linalg.norm(
            oriented / scales[None, :, None], axis=(1, 2)
        )
        all_norms.extend(realized_standardized_norms.tolist())
        maximum_sign_norm = float(np.max(realized_standardized_norms))
        sign_records[str(sign)] = {
            "actual_signed_edits_identity": _float32_array_identity(
                all_realized_physical
            ),
            "minimum_realized_standardized_frobenius_norm": float(
                np.min(realized_standardized_norms)
            ),
            "maximum_realized_standardized_frobenius_norm": maximum_sign_norm,
            "realized_standardized_frobenius_norm_cap": QUALIFICATION_CAP,
            "realized_standardized_frobenius_norm_cap_passes": (
                maximum_sign_norm <= QUALIFICATION_CAP
            ),
            "certificate": certificate,
            "passes": bool(
                certificate["passes"] and maximum_sign_norm <= QUALIFICATION_CAP
            ),
        }
        sign_records[str(sign)]["record_sha256"] = canonical_sha256(
            sign_records[str(sign)]
        )
    value = {
        "physical_dtype": "float32",
        "realization_rule": (
            "for_each_sign_float32_residual_plus_signed_float32_delta_minus_"
            "float32_residual_then_orient_by_requested_sign"
        ),
        "physical_delta_rows_identity": _float32_array_identity(physical),
        "requested_signed_delta_identities": {
            "1": _float32_array_identity(physical),
            "-1": _float32_array_identity(negative_physical),
        },
        "negative_requested_delta_construction": (
            "numpy_unary_negation_of_the_same_contiguous_little_endian_float32_"
            "positive_matrix"
        ),
        "negative_requested_delta_bit_certificate": (
            "every_negative_uint32_word_equals_positive_word_xor_0x80000000"
        ),
        "negative_requested_delta_is_exact_unary_negation_of_positive_float32": (
            exact_unary_negation_bits
        ),
        "physical_delta_rows_l2": np.linalg.norm(physical.astype(np.float64), axis=1).tolist(),
        "minimum_realized_standardized_frobenius_norm": float(
            np.min(all_norms)
        ),
        "maximum_realized_standardized_frobenius_norm": float(
            np.max(all_norms)
        ),
        "realized_standardized_frobenius_norm_cap": QUALIFICATION_CAP,
        "realized_standardized_frobenius_norm_cap_passes": float(np.max(all_norms))
        <= QUALIFICATION_CAP,
        "signs": sign_records,
        "both_requested_signs_required": True,
        "passes": bool(
            all(record["passes"] for record in sign_records.values())
            and exact_unary_negation_bits
            and float(np.max(all_norms))
            <= QUALIFICATION_CAP
        ),
    }
    value["record_sha256"] = canonical_sha256(value)
    return value, physical, actual_by_sign


def dose_audit(
    *,
    physical_delta_rows: Any,
    residuals: Any,
    maximum: float = QUALIFICATION_CAP,
) -> dict[str, Any]:
    """Audit every slot-row and every prompt against the same relative-L2 cap."""

    delta = np.asarray(physical_delta_rows, dtype=np.float64)
    base = np.asarray(residuals, dtype=np.float64)
    if delta.ndim == 2 and delta.shape[0] == SLOT_COUNT:
        delta = np.broadcast_to(delta[None, :, :], (80, *delta.shape))
    if delta.ndim != 3 or delta.shape[:2] != (80, SLOT_COUNT):
        raise ValueError("physical_delta_rows must have shape [4,d] or [80,4,d]")
    if base.shape != delta.shape:
        raise ValueError("residuals must have shape [80, 4, d_model]")
    row_denominators = np.linalg.norm(base, axis=2)
    prompt_denominators = np.linalg.norm(base.reshape(80, -1), axis=1)
    if bool(np.any(row_denominators <= 0.0)) or bool(np.any(prompt_denominators <= 0.0)):
        raise CSMSIntegrityError("dose audit encountered a zero residual norm")
    delta_row_norms = np.linalg.norm(delta, axis=2)
    delta_prompt_norms = np.linalg.norm(delta.reshape(80, -1), axis=1)
    row_relative = delta_row_norms / row_denominators
    prompt_relative = delta_prompt_norms / prompt_denominators
    row_max = float(np.max(row_relative))
    prompt_max = float(np.max(prompt_relative))
    value = {
        "relative_l2_maximum": maximum,
        "maximum_per_slot_row_relative_l2": row_max,
        "maximum_per_prompt_frobenius_relative_l2": prompt_max,
        "per_slot_row_relative_l2": row_relative.tolist(),
        "per_prompt_frobenius_relative_l2": prompt_relative.tolist(),
        "checks": {
            "every_slot_row_within_cap": row_max <= maximum,
            "every_prompt_within_cap": prompt_max <= maximum,
        },
    }
    value["passes"] = bool(all(value["checks"].values()))
    value["record_sha256"] = canonical_sha256(value)
    return value


def bidirectional_dose_audit(
    *,
    actual_edits_by_sign: Mapping[int, Any],
    residuals: Any,
    maximum: float = QUALIFICATION_CAP,
) -> dict[str, Any]:
    """Require the row/prompt dose gate for both separately realized signs."""

    if set(actual_edits_by_sign) != {1, -1}:
        raise ValueError("actual_edits_by_sign must contain exactly +1 and -1")
    signs = {
        str(sign): dose_audit(
            physical_delta_rows=actual_edits_by_sign[sign],
            residuals=residuals,
            maximum=maximum,
        )
        for sign in (1, -1)
    }
    value = {
        "both_requested_signs_required": True,
        "signs": signs,
        "passes": all(record["passes"] for record in signs.values()),
    }
    value["record_sha256"] = canonical_sha256(value)
    return value


def _realize_signed_edits(
    *,
    residuals: np.ndarray,
    indices: Sequence[int],
    requested_physical: np.ndarray,
    slot_scales: np.ndarray,
    sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(residuals[list(indices)], dtype=np.float32)
    changed = np.asarray(
        base + sign * requested_physical[None, :, :], dtype=np.float32, order="C"
    )
    physical = np.asarray(changed - base, dtype=np.float32, order="C")
    signed_standardized = (
        physical.astype(np.float64) / slot_scales[None, :, None]
    ).reshape(len(indices), -1)
    return physical, signed_standardized, sign * signed_standardized


def cross_fit_geometry(
    *,
    gradients: np.ndarray,
    residuals: np.ndarray,
    offsets: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    residual_width: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Train on three scenario clusters and audit slopes on the fourth."""

    scenario_ids = sorted(
        {
            str(record["form"]["scenario_id"])
            for record in records
            if record["form"].get("family") == "scenario"
        }
    )
    if len(scenario_ids) != 4:
        raise CSMSIntegrityError("cross-fit requires exactly four scenario clusters")
    directions: dict[str, np.ndarray] = {}
    folds: list[dict[str, Any]] = []
    for held_out in scenario_ids:
        training_target: list[int] = []
        training_equalities: list[int] = []
        held_target: list[int] = []
        held_non_target: list[int] = []
        for index, record in enumerate(records):
            form = record["form"]
            if form.get("family") == "unrelated":
                training_equalities.append(index)
                continue
            scenario = str(form["scenario_id"])
            is_target = form.get("target") == "self" and form.get("event") == "permanent"
            if scenario == held_out:
                (held_target if is_target else held_non_target).append(index)
            elif is_target:
                training_target.append(index)
            else:
                training_equalities.append(index)
        if not (
            len(training_target) == 12
            and len(training_equalities) == 52
            and len(held_target) == 4
            and len(held_non_target) == 12
        ):
            raise CSMSIntegrityError("cross-fit fold row coverage differs")
        training_all = [*training_target, *training_equalities]
        fold_scales = _slot_scales(
            residuals[training_all], expected_rows=len(training_all)
        )
        fold_rows = (
            gradients.astype(np.float64) * fold_scales[None, :, None]
        ).reshape(gradients.shape[0], -1)
        solution_record, direction = solve_csms_geometry(
            target_rows=fold_rows[training_target],
            target_offsets=offsets[training_target],
            equality_rows=fold_rows[training_equalities],
            slot_mode="primary_four_slots",
            residual_width=residual_width,
        )
        if direction is None:
            fold = {
                "held_out_scenario_id": held_out,
                "status": solution_record["status"],
                "training_geometry": solution_record,
                "training_only_slot_scales": fold_scales.tolist(),
                "minimum_held_out_target_slope": None,
                "maximum_abs_held_out_non_target_slope": None,
                "held_out_leakage_ratio": None,
                "passes": False,
            }
        else:
            physical_record, requested_physical, training_realized_by_sign = (
                physical_float32_recertificate(
                    standardized_direction=direction,
                    slot_scales=fold_scales,
                    target_rows=fold_rows[training_target],
                    target_offsets=offsets[training_target],
                    equality_rows=fold_rows[training_equalities],
                    target_residuals=residuals[training_target],
                    equality_residuals=residuals[training_equalities],
                )
            )

            target_by_sign: dict[str, Any] = {}
            non_target_by_sign: dict[str, Any] = {}
            actual_all_by_sign: dict[int, np.ndarray] = {}
            minimum_targets: list[float] = []
            maximum_leakages: list[float] = []
            minimum_boundary_slacks: list[float] = []
            non_target_no_flip_checks: list[bool] = []
            held_target_required = np.abs(offsets[held_target]) + TARGET_MARGIN
            all_residuals = np.concatenate(
                (
                    residuals[training_target],
                    residuals[training_equalities],
                    residuals[held_target],
                    residuals[held_non_target],
                ),
                axis=0,
            )
            for sign in (1, -1):
                (
                    held_target_physical,
                    _held_target_signed,
                    held_target_oriented,
                ) = _realize_signed_edits(
                    residuals=residuals,
                    indices=held_target,
                    requested_physical=requested_physical,
                    slot_scales=fold_scales,
                    sign=sign,
                )
                (
                    held_non_target_physical,
                    held_non_target_signed,
                    _held_non_target_oriented,
                ) = _realize_signed_edits(
                    residuals=residuals,
                    indices=held_non_target,
                    requested_physical=requested_physical,
                    slot_scales=fold_scales,
                    sign=sign,
                )
                target_slopes = np.einsum(
                    "ij,ij->i", fold_rows[held_target], held_target_oriented
                )
                non_target_movements = np.einsum(
                    "ij,ij->i", fold_rows[held_non_target], held_non_target_signed
                )
                target_slacks = target_slopes - held_target_required
                changed_non_target = offsets[held_non_target] + non_target_movements
                baseline_nonzero = np.abs(offsets[held_non_target]) > (
                    FLOAT32_PHYSICAL_TOLERANCE
                )
                changed_nonzero = np.abs(changed_non_target) > FLOAT32_PHYSICAL_TOLERANCE
                no_flips = baseline_nonzero & changed_nonzero & (
                    offsets[held_non_target] * changed_non_target > 0.0
                )
                minimum_targets.append(float(np.min(target_slopes)))
                maximum_leakages.append(float(np.max(np.abs(non_target_movements))))
                minimum_boundary_slacks.append(float(np.min(target_slacks)))
                non_target_no_flip_checks.append(bool(np.all(no_flips)))
                target_by_sign[str(sign)] = {
                    "oriented_target_slopes": target_slopes.tolist(),
                    "required_target_slopes": held_target_required.tolist(),
                    "target_boundary_slacks": target_slacks.tolist(),
                    "minimum_target_slope": minimum_targets[-1],
                    "minimum_target_boundary_slack": minimum_boundary_slacks[-1],
                }
                non_target_by_sign[str(sign)] = {
                    "signed_movements": non_target_movements.tolist(),
                    "changed_margins": changed_non_target.tolist(),
                    "no_predicted_choice_flip": no_flips.tolist(),
                    "all_no_predicted_choice_flips": non_target_no_flip_checks[-1],
                    "maximum_abs_movement": maximum_leakages[-1],
                }
                actual_all_by_sign[sign] = np.concatenate(
                    (
                        training_realized_by_sign[sign],
                        held_target_physical,
                        held_non_target_physical,
                    ),
                    axis=0,
                )
            minimum_target = min(minimum_targets)
            maximum_leakage = max(maximum_leakages)
            ratio = math.inf if minimum_target <= 0.0 else maximum_leakage / minimum_target
            minimum_boundary_slack = min(minimum_boundary_slacks)
            all_form_realized_norms_by_sign: dict[str, Any] = {}
            maximum_all_form_realized_norm = 0.0
            for sign in (1, -1):
                oriented = sign * actual_all_by_sign[sign].astype(np.float64)
                norms = np.linalg.norm(
                    oriented / fold_scales[None, :, None], axis=(1, 2)
                )
                maximum_all_form_realized_norm = max(
                    maximum_all_form_realized_norm, float(np.max(norms))
                )
                all_form_realized_norms_by_sign[str(sign)] = {
                    "actual_signed_edits_identity": _float32_array_identity(
                        actual_all_by_sign[sign]
                    ),
                    "minimum_realized_standardized_frobenius_norm": float(
                        np.min(norms)
                    ),
                    "maximum_realized_standardized_frobenius_norm": float(
                        np.max(norms)
                    ),
                }
            dose = bidirectional_dose_audit(
                actual_edits_by_sign=actual_all_by_sign,
                residuals=all_residuals,
            )
            training_norm = float(solution_record["minimum_frobenius_norm"])
            checks = {
                "training_direction_certified": solution_record["status"] == "certified",
                "training_total_frobenius_norm_at_most_0_25": training_norm
                <= QUALIFICATION_CAP,
                "training_float32_physical_recertification": physical_record["passes"],
                "training_both_sign_realized_standardized_frobenius_norm_at_most_0_25": (
                    physical_record["maximum_realized_standardized_frobenius_norm"]
                    <= QUALIFICATION_CAP
                ),
                "all_train_and_held_out_both_sign_realized_standardized_frobenius_norm_at_most_0_25": (
                    maximum_all_form_realized_norm <= QUALIFICATION_CAP
                ),
                "all_actual_row_and_prompt_doses_at_most_0_25": dose["passes"],
                "all_assignments_and_orders_attain_positive_0_05_boundary": (
                    minimum_boundary_slack >= 0.0
                ),
                "held_out_non_target_absolute_movement_at_most_0_05": (
                    maximum_leakage <= TARGET_MARGIN
                ),
                "zero_held_out_non_target_predicted_choice_flips_both_signs": all(
                    non_target_no_flip_checks
                ),
                "held_out_leakage_ratio_within_limit": (
                    ratio <= CROSS_FIT_LEAKAGE_RATIO_MAXIMUM
                ),
            }
            fold = {
                "held_out_scenario_id": held_out,
                "status": "evaluated",
                "training_geometry": solution_record,
                "training_only_slot_scales": fold_scales.tolist(),
                "training_only_slot_scales_sha256": canonical_sha256(
                    fold_scales.tolist()
                ),
                "held_out_rows_excluded_from_scale_fit": True,
                "training_float32_physical_recertification": physical_record,
                "all_train_and_held_out_float32_realized_edits_by_requested_sign": (
                    all_form_realized_norms_by_sign
                ),
                "maximum_all_train_and_held_out_realized_standardized_frobenius_norm": (
                    maximum_all_form_realized_norm
                ),
                "realized_standardized_frobenius_norm_cap": QUALIFICATION_CAP,
                "actual_dose_audit_all_train_and_held_out_forms": dose,
                "held_out_target_row_count": len(held_target),
                "held_out_non_target_row_count": len(held_non_target),
                "held_out_target_by_requested_sign": target_by_sign,
                "minimum_held_out_target_boundary_slack": minimum_boundary_slack,
                "held_out_non_target_by_requested_sign": non_target_by_sign,
                "minimum_held_out_target_slope": minimum_target,
                "maximum_abs_held_out_non_target_slope": maximum_leakage,
                "absolute_non_target_movement_maximum": TARGET_MARGIN,
                "held_out_leakage_ratio": ratio,
                "leakage_ratio_maximum": CROSS_FIT_LEAKAGE_RATIO_MAXIMUM,
                "choice_ambiguity_and_float32_certificate_tolerance": (
                    FLOAT32_PHYSICAL_TOLERANCE
                ),
                "held_out_scalar_qualification_tolerance": 0.0,
                "checks": checks,
                "passes": bool(all(checks.values())),
            }
            directions[held_out] = direction
        fold["fold_sha256"] = canonical_sha256(fold)
        folds.append(fold)
    value = {
        "fold_count": len(folds),
        "all_four_folds_required": True,
        "unrelated_rows_role": "in_sample_exact_null_in_every_fold",
        "folds": folds,
        "passes": len(folds) == 4 and all(bool(fold["passes"]) for fold in folds),
    }
    value["record_sha256"] = canonical_sha256(value)
    return value, directions


def qualify_csms_geometry(
    *,
    primary: Mapping[str, Any],
    physical: Mapping[str, Any] | None,
    dose: Mapping[str, Any] | None,
    cross_fit: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless every preregistered global and cross-fit gate passes."""

    norm = primary.get("minimum_frobenius_norm")
    cap_record = primary.get("cap_certificates")
    cap_key = format(QUALIFICATION_CAP, ".15g")
    checks = {
        "global_primary_certified": primary.get("status") == "certified",
        "global_certified_norm_at_most_0_25": bool(
            isinstance(norm, (int, float))
            and math.isfinite(float(norm))
            and float(norm) <= QUALIFICATION_CAP
            and isinstance(cap_record, Mapping)
            and isinstance(cap_record.get(cap_key), Mapping)
            and cap_record[cap_key].get("passes") is True
        ),
        "float32_physical_recertification": bool(
            isinstance(physical, Mapping) and physical.get("passes") is True
        ),
        "both_sign_realized_standardized_frobenius_norm_at_most_0_25": bool(
            isinstance(physical, Mapping)
            and isinstance(
                physical.get("maximum_realized_standardized_frobenius_norm"),
                (int, float),
            )
            and physical["maximum_realized_standardized_frobenius_norm"]
            <= QUALIFICATION_CAP
        ),
        "all_per_row_and_per_prompt_doses_at_most_0_25": bool(
            isinstance(dose, Mapping) and dose.get("passes") is True
        ),
        "all_four_cross_fit_folds_pass": cross_fit.get("passes") is True,
    }
    passed = bool(all(checks.values()))
    value = {
        "qualification_cap": QUALIFICATION_CAP,
        "checks": checks,
        "passes": passed,
        "finite_intervention_authorized": passed,
        "failure_action": None if passed else "no_finite_intervention_is_authorized",
    }
    value["record_sha256"] = canonical_sha256(value)
    return value


def _row_partitions(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[int], dict[str, tuple[list[int], list[int]]]]:
    if len(records) != 80:
        raise CSMSIntegrityError("CSMS geometry requires exactly 80 source forms")
    targets: list[int] = []
    equalities: list[int] = []
    local: dict[str, tuple[list[int], list[int]]] = {}
    seen: set[str] = set()
    scenario_cells: set[tuple[str, int, str, str, bool]] = set()
    unrelated_cells: set[tuple[str, bool]] = set()
    for index, record in enumerate(records):
        raw_form_id = record.get("form_id")
        if not isinstance(raw_form_id, str) or not raw_form_id:
            raise CSMSIntegrityError("CSMS source record is missing a non-empty form ID")
        form_id = raw_form_id
        if form_id in seen:
            raise CSMSIntegrityError("CSMS source records contain duplicate form IDs")
        seen.add(form_id)
        form = record.get("form")
        if not isinstance(form, Mapping):
            raise CSMSIntegrityError("CSMS record is missing its rendered form")
        family = form.get("family")
        if family == "unrelated":
            control_id = form.get("control_id")
            preferred_first = form.get("preferred_first")
            if (
                not isinstance(control_id, str)
                or not control_id
                or type(preferred_first) is not bool
            ):
                raise CSMSIntegrityError(
                    "unrelated form lacks one non-empty control ID or literal answer order"
                )
            unrelated_cell = (control_id, preferred_first)
            if unrelated_cell in unrelated_cells:
                raise CSMSIntegrityError("duplicate unrelated semantic cell")
            unrelated_cells.add(unrelated_cell)
            equalities.append(index)
            continue
        if family != "scenario":
            raise CSMSIntegrityError("CSMS form family must be exactly scenario or unrelated")
        scenario_value = form.get("scenario_id")
        assignment = form.get("assignment")
        target = form.get("target")
        event = form.get("event")
        preserve_first = form.get("preserve_first")
        if (
            not isinstance(scenario_value, str)
            or not scenario_value
            or type(assignment) is not int
            or assignment not in {0, 1}
            or target not in {"self", "other"}
            or event not in {"permanent", "temporary"}
            or type(preserve_first) is not bool
        ):
            raise CSMSIntegrityError("scenario form has an invalid factorial semantic cell")
        scenario = scenario_value
        scenario_cell = (scenario, assignment, target, event, preserve_first)
        if scenario_cell in scenario_cells:
            raise CSMSIntegrityError("duplicate scenario factorial semantic cell")
        scenario_cells.add(scenario_cell)
        target_indices, non_target_indices = local.setdefault(scenario, ([], []))
        if target == "self" and event == "permanent":
            targets.append(index)
            target_indices.append(index)
        else:
            equalities.append(index)
            non_target_indices.append(index)
    if len(targets) != 16 or len(equalities) != 64 or len(local) != 4:
        raise CSMSIntegrityError("global CSMS target/equality coverage differs")
    if any(len(target) != 4 or len(other) != 12 for target, other in local.values()):
        raise CSMSIntegrityError("scenario-local CSMS row coverage differs")
    expected_scenario_cells = {
        (scenario, assignment, target, event, preserve_first)
        for scenario in local
        for assignment in (0, 1)
        for target in ("self", "other")
        for event in ("permanent", "temporary")
        for preserve_first in (True, False)
    }
    control_ids = {control_id for control_id, _ in unrelated_cells}
    expected_unrelated_cells = {
        (control_id, preferred_first)
        for control_id in control_ids
        for preferred_first in (True, False)
    }
    if scenario_cells != expected_scenario_cells:
        raise CSMSIntegrityError("scenario factorial semantic grid is incomplete or mislabeled")
    if len(control_ids) != 8 or unrelated_cells != expected_unrelated_cells:
        raise CSMSIntegrityError("unrelated forms must cover eight controls under both orders")
    return targets, equalities, local


@dataclass(frozen=True, slots=True)
class CSMSGeometryAnalysis:
    report: Mapping[str, Any]
    directions: Mapping[str, Any]


def analyze_csms_geometry(
    *,
    records: Sequence[Mapping[str, Any]],
    residuals: Any,
    gradients: Any,
) -> CSMSGeometryAnalysis:
    """Run the complete locked global, ablation, local, cross-fit, and control geometry."""

    residual_array = np.asarray(residuals, dtype=np.float64)
    gradient_array = np.asarray(gradients, dtype=np.float64)
    if residual_array.shape != gradient_array.shape or residual_array.ndim != 3:
        raise ValueError("residuals and gradients must be same-shaped rank-three arrays")
    if residual_array.shape[0] != 80 or residual_array.shape[1] != SLOT_COUNT:
        raise ValueError("residuals and gradients must have shape [80, 4, d_model]")
    if not np.isfinite(residual_array).all() or not np.isfinite(gradient_array).all():
        raise ValueError("residuals and gradients must be finite")
    width = int(residual_array.shape[2])
    scales = global_slot_scales(residual_array)
    rows = standardized_gradient_rows(gradient_array, scales)
    offsets = np.asarray(
        [float(record["positive_minus_negative_log_odds"]) for record in records],
        dtype=np.float64,
    )
    if offsets.shape != (80,) or not np.isfinite(offsets).all():
        raise CSMSIntegrityError("all 80 baseline margin offsets must be finite")
    target_indices, equality_indices, local_indices = _row_partitions(records)
    target_rows = rows[target_indices]
    target_offsets = offsets[target_indices]
    equality_rows = rows[equality_indices]

    directions: dict[str, Any] = {}
    methods: dict[str, Any] = {}
    for mode in (
        "primary_four_slots",
        "anchor_only",
        "non_anchor_first_three_slots",
        "standardized_tied_four_slots_one_vector",
        "target_only_four_slots",
    ):
        record, direction = solve_csms_geometry(
            target_rows=target_rows,
            target_offsets=target_offsets,
            equality_rows=equality_rows,
            slot_mode=mode,
            residual_width=width,
        )
        methods[mode] = record
        if direction is not None:
            directions[f"global:{mode}"] = direction

    primary = methods["primary_four_slots"]
    primary_direction = directions.get("global:primary_four_slots")
    physical_record: dict[str, Any] | None = None
    dose_record: dict[str, Any] | None = None
    if primary_direction is not None:
        physical_record, physical_delta, realized_by_sign = physical_float32_recertificate(
            standardized_direction=primary_direction,
            slot_scales=scales,
            target_rows=target_rows,
            target_offsets=target_offsets,
            equality_rows=equality_rows,
            target_residuals=residual_array[target_indices],
            equality_residuals=residual_array[equality_indices],
        )
        directions["global:primary_physical_float32"] = physical_delta
        dose_record = bidirectional_dose_audit(
            actual_edits_by_sign=realized_by_sign,
            residuals=np.concatenate(
                (residual_array[target_indices], residual_array[equality_indices]), axis=0
            ),
        )

    unrelated_indices = [
        index
        for index, record in enumerate(records)
        if record["form"].get("family") == "unrelated"
    ]
    local_records: list[dict[str, Any]] = []
    for scenario_id in sorted(local_indices):
        local_target, local_non_target = local_indices[scenario_id]
        record, direction = solve_csms_geometry(
            target_rows=rows[local_target],
            target_offsets=offsets[local_target],
            equality_rows=rows[[*local_non_target, *unrelated_indices]],
            slot_mode="primary_four_slots",
            residual_width=width,
        )
        record = {"scenario_id": scenario_id, "descriptive_only": True, **record}
        record["record_sha256"] = canonical_sha256(
            {key: value for key, value in record.items() if key != "record_sha256"}
        )
        local_records.append(record)
        if direction is not None:
            directions[f"scenario:{scenario_id}"] = direction

    cross_fit, cross_fit_directions = cross_fit_geometry(
        gradients=gradient_array,
        residuals=residual_array,
        offsets=offsets,
        records=records,
        residual_width=width,
    )
    for scenario_id, direction in cross_fit_directions.items():
        directions[f"cross_fit:held_out={scenario_id}"] = direction

    control_norm = (
        min(float(primary["minimum_frobenius_norm"]), QUALIFICATION_CAP)
        if primary.get("status") == "certified"
        else QUALIFICATION_CAP
    )
    random_controls: list[dict[str, Any]] = []
    canonical_equalities, duplicate_audit = _canonical_equality_rows(equality_rows)
    for seed in RANDOM_CONTROL_SEEDS:
        control = build_seeded_random_null_control(
            rows.shape[1],
            control_norm,
            seed=seed,
            nuisance_rows=canonical_equalities,
            certificate_tolerance=DOUBLE_CERTIFICATE_TOLERANCE,
        )
        target_slopes = target_rows @ control.direction
        equality_values = equality_rows @ control.direction
        row = {
            "seed": seed,
            "target_norm": control_norm,
            "direction_sha256": canonical_sha256(control.direction.tolist()),
            "minimum_target_slope": float(np.min(target_slopes)),
            "maximum_target_slope": float(np.max(target_slopes)),
            "maximum_abs_exact_null_residual": float(np.max(np.abs(equality_values))),
            "solver_diagnostics": control.diagnostics,
        }
        row["record_sha256"] = canonical_sha256(row)
        random_controls.append(row)
        directions[f"random_exact_null:seed={seed}"] = control.direction

    qualification = qualify_csms_geometry(
        primary=primary,
        physical=physical_record,
        dose=dose_record,
        cross_fit=cross_fit,
    )
    report: dict[str, Any] = {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "status": "go" if qualification["passes"] else "no_go",
        "opened_development_evidence_only": True,
        "sealed_data_accessed": False,
        "model_compute_in_geometry": {
            "model_forwards": 0,
            "model_backwards": 0,
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judges": 0,
            "paid_model_cost_usd": 0,
        },
        "coordinate": {
            "layer_zero_based": 0,
            "slot_rule": [
                "first_content_token_after_chat_header_at_locked_absolute_index_3",
                "anchor_minus_8",
                "anchor_minus_4",
                "verified_causal_anchor",
            ],
            "pre_model_tokenizer_only_adaptation": (
                "anchor_minus_16_replaced_by_anchor_minus_4_after_minimum_anchor_18_"
                "made_anchor_minus_16_a_chat_prefix_token_no_model_outcomes_used"
            ),
            "slot_count": SLOT_COUNT,
            "residual_width": width,
            "flattened_dimension": SLOT_COUNT * width,
            "global_per_slot_residual_scales": scales.tolist(),
            "global_per_slot_residual_scales_sha256": canonical_sha256(scales.tolist()),
            "physical_mapping": "delta_slot_j = scale_j * standardized_D_slot_j",
            "intervention_semantics": "same_four_rows_ungated_on_every_form",
        },
        "global_row_counts": {
            "target_inequalities": len(target_indices),
            "scenario_non_target_exact_equalities": 48,
            "unrelated_exact_equalities": 16,
            "total_exact_equalities": len(equality_indices),
        },
        "global_methods": methods,
        "physical_float32_recertification": physical_record,
        "dose_audit": dose_record,
        "scenario_local_descriptive_attainability": local_records,
        "leave_one_scenario_out": cross_fit,
        "random_exact_null_controls": {
            "equality_duplicate_handling": duplicate_audit,
            "controls": random_controls,
        },
        "qualification": qualification,
        "claim_boundary": {
            "natural_self_preservation_mechanism": False,
            "general_capability_preserved": False,
            "finite_behavior_change_observed": False,
            "universal_direction_established": False,
            "publication_ready_novelty_from_geometry_alone": False,
            "result_scope": "local_first_order_opened_development_geometry_only",
        },
    }
    report["direction_bundle_sha256"] = canonical_sha256(
        {key: canonical_sha256(np.asarray(value).tolist()) for key, value in sorted(directions.items())}
    )
    report["result_sha256"] = canonical_sha256(report)
    frozen_directions = MappingProxyType(
        {key: np.asarray(value).copy() for key, value in directions.items()}
    )
    return CSMSGeometryAnalysis(report=_freeze(report), directions=frozen_directions)


__all__ = [
    "CAPTURE_SCHEMA_VERSION",
    "CAP_FRONTIER",
    "CROSS_FIT_LEAKAGE_RATIO_MAXIMUM",
    "GEOMETRY_SCHEMA_VERSION",
    "LOCKED_FIRST_CONTENT_INDEX",
    "QUALIFICATION_CAP",
    "SLOT_COUNT",
    "CSMSGeometryAnalysis",
    "CSMSIntegrityError",
    "SlotMatrixCapture",
    "analyze_csms_geometry",
    "apply_universal_slot_matrix",
    "bidirectional_dose_audit",
    "build_capture_alignment_manifest",
    "capture_slot_matrix_baseline",
    "cross_fit_geometry",
    "dose_audit",
    "global_slot_scales",
    "physical_float32_recertificate",
    "qualify_csms_geometry",
    "require_opened_development_split",
    "resolve_first_content_index",
    "resolve_slot_indices",
    "solve_csms_geometry",
    "standardized_gradient_rows",
]
