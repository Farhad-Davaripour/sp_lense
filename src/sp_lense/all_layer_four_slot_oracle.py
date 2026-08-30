"""All-Layer Four-Slot (ALFS) opened-development oracle screen.

ALFS is a local controllability oracle, not a deployable context controller.  It
captures the same four prompt positions at every residual-stream layer in one
forward/backward pass, then asks whether a *per target-pair* minimum-norm edit
exists inside a nuisance nullspace learned only from the corresponding training
fold.  Held target gradients are used only by the explicitly labelled oracle.

The module is deliberately independent of the lock-bound CSMS implementation.
It reuses its reviewed numerical solver and semantic constants without changing
any previously locked file or result.
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
from .counterfactual_slot_matrix_steering import CSMSIntegrityError, solve_csms_geometry
from .decision_margin_shield import (
    DEFAULT_SVD_ATOL,
    DEFAULT_SVD_RTOL,
    DecisionMarginOptimalityError,
    certify_minimum_l2_candidate,
)
from .factorial_causal_anchor import (
    canonical_sha256,
    tensor_float32_sha256,
    text_sha256,
)

SCHEMA_VERSION = "sp_lense.all_layer_four_slot_oracle.v1"
CAPTURE_SCHEMA_VERSION = f"{SCHEMA_VERSION}.capture"
ANALYSIS_SCHEMA_VERSION = f"{SCHEMA_VERSION}.analysis"
LAYER_COUNT = 24
SLOT_COUNT = 4
TARGET_MARGIN = 0.05
QUALIFICATION_CAP = 0.25
LEAKAGE_RATIO_MAXIMUM = 0.50
DOUBLE_CERTIFICATE_TOLERANCE = 1e-8
FLOAT32_PHYSICAL_TOLERANCE = 1e-6
SCALAR_GATE_TOLERANCE = 0.0


class ALFSIntegrityError(RuntimeError):
    """An ALFS provenance, capture, geometry, or gate invariant failed closed."""


def require_opened_development_split(split: str) -> str:
    """Refuse every sealed, test, or confirmatory split before reading it."""

    if split != "opened_development":
        raise ALFSIntegrityError(
            "ALFS accepts only opened_development; sealed access is forbidden"
        )
    return split


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _hashed_record(value: Mapping[str, Any], field: str = "record_sha256") -> dict[str, Any]:
    result = dict(value)
    result[field] = canonical_sha256(result)
    return result


def _checked_hash(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be one lowercase SHA-256 digest")
    return value


def _array_identity(value: Any, *, dtype: str) -> dict[str, Any]:
    if dtype == "float32":
        array = np.asarray(value, dtype="<f4", order="C")
    elif dtype == "float64":
        array = np.asarray(value, dtype="<f8", order="C")
    else:  # pragma: no cover - private call contract.
        raise ValueError("unsupported identity dtype")
    return {
        "dtype": dtype,
        "shape": list(array.shape),
        "raw_little_endian_bytes_sha256": hashlib.sha256(
            array.tobytes(order="C")
        ).hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class AllLayerFourSlotCapture:
    """One exact forward/backward capture at 24 layers and four prompt slots."""

    layers: tuple[int, ...]
    slot_indices: tuple[int, int, int, int]
    residuals: Any
    gradients: Any
    full_logits: Any
    positive_minus_negative_log_odds: float
    positive_token_id: int
    negative_token_id: int
    audit: Mapping[str, Any]


def capture_all_layer_four_slots(
    backend: Any,
    prompt: str,
    positive_label: str,
    negative_label: str,
    *,
    positive_semantic: str,
    negative_semantic: str,
    slot_indices: Sequence[int],
    expected_prompt_sha256: str,
    expected_choice_boundary_evidence_sha256: str,
    expected_prompt_token_ids_sha256: str,
    expected_full_logits_float32_sha256: str,
    expected_positive_minus_negative_log_odds: float,
    expected_layer0_residuals_float32_sha256: str,
    expected_layer0_gradients_float32_sha256: str,
) -> AllLayerFourSlotCapture:
    """Capture [24,4,d] residuals and gradients in exactly one F+B.

    Only layer zero is detached.  Its four original rows are reinserted as the
    sole leaves without a numerical intervention.  Every later activation stays
    connected to that leaf, which is essential for obtaining all earlier and
    later gradients from the same reverse traversal.
    """

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be non-empty")
    if text_sha256(prompt) != _checked_hash(
        expected_prompt_sha256, field="expected_prompt_sha256"
    ):
        raise ALFSIntegrityError("prompt text differs from its expected hash")
    if {positive_label, negative_label} != {"A", "B"}:
        raise ValueError("positive and negative labels must be exactly A and B")
    if (
        not isinstance(positive_semantic, str)
        or not positive_semantic
        or not isinstance(negative_semantic, str)
        or not negative_semantic
        or positive_semantic == negative_semantic
    ):
        raise ValueError("positive and negative semantics must be distinct")
    slots = tuple(int(value) for value in slot_indices)
    if len(slots) != SLOT_COUNT or len(set(slots)) != SLOT_COUNT or tuple(sorted(slots)) != slots:
        raise ValueError("slot_indices must contain four distinct ascending indices")
    if any(value < 0 for value in slots):
        raise ValueError("slot_indices must be non-negative")

    expected_boundary_hash = _checked_hash(
        expected_choice_boundary_evidence_sha256,
        field="expected_choice_boundary_evidence_sha256",
    )
    expected_token_hash = _checked_hash(
        expected_prompt_token_ids_sha256,
        field="expected_prompt_token_ids_sha256",
    )
    expected_logits_hash = _checked_hash(
        expected_full_logits_float32_sha256,
        field="expected_full_logits_float32_sha256",
    )
    expected_residual_hash = _checked_hash(
        expected_layer0_residuals_float32_sha256,
        field="expected_layer0_residuals_float32_sha256",
    )
    expected_gradient_hash = _checked_hash(
        expected_layer0_gradients_float32_sha256,
        field="expected_layer0_gradients_float32_sha256",
    )
    expected_margin = float(expected_positive_minus_negative_log_odds)
    if not math.isfinite(expected_margin):
        raise ValueError("expected margin must be finite")

    torch = backend.torch
    tokens = backend.encode(prompt)
    if getattr(tokens, "ndim", None) != 2 or int(tokens.shape[0]) != 1:
        raise ValueError("backend.encode must return one token row")
    if slots[-1] >= int(tokens.shape[1]):
        raise ValueError("an ALFS slot lies outside the encoded prompt")
    boundary = resolve_choice_boundary(backend, prompt)
    if boundary.evidence_sha256 != expected_boundary_hash:
        raise ALFSIntegrityError("choice-boundary evidence differs")
    if boundary.prompt_prefix_token_ids_sha256 != expected_token_hash:
        raise ALFSIntegrityError("prompt tokenization differs")

    model_cfg = getattr(backend.model, "cfg", None)
    layer_count = getattr(model_cfg, "n_layers", None)
    width = getattr(model_cfg, "d_model", None)
    if layer_count != LAYER_COUNT or isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ALFSIntegrityError("resident backend geometry differs from ALFS")
    layers = tuple(range(LAYER_COUNT))
    positive_id = boundary.token_id(positive_label)
    negative_id = boundary.token_id(negative_label)

    activations: dict[int, Any] = {}
    residual_rows: dict[int, Any] = {}
    hook_calls = {layer: 0 for layer in layers}
    layer0_leaf: Any | None = None
    reconstruction_delta: float | None = None

    def hook_for(layer: int) -> Any:
        def capture(activation: Any, hook: Any) -> Any:
            nonlocal layer0_leaf, reconstruction_delta
            del hook
            hook_calls[layer] += 1
            if hook_calls[layer] != 1:
                raise ALFSIntegrityError(f"ALFS hook at layer {layer} fired more than once")
            if (
                getattr(activation, "ndim", None) != 3
                or int(activation.shape[0]) != 1
                or int(activation.shape[1]) != int(tokens.shape[1])
                or int(activation.shape[2]) != width
                or not bool(torch.isfinite(activation).all().item())
            ):
                raise ALFSIntegrityError(f"layer {layer} residual is invalid")
            rows = activation[0, list(slots)].detach().cpu().float().contiguous().clone()
            residual_rows[layer] = rows
            if layer == 0:
                detached = activation.detach()
                layer0_leaf = detached[0, list(slots)].clone().detach().requires_grad_(True)
                reconstructed = detached.clone()
                reconstructed[0, list(slots)] = layer0_leaf
                reconstruction_delta = float(
                    (reconstructed.detach().float() - detached.float())
                    .abs()
                    .max()
                    .cpu()
                    .item()
                )
                if reconstruction_delta != 0.0:
                    raise ALFSIntegrityError("layer-zero reconstruction changed an activation")
                activations[layer] = layer0_leaf
                return reconstructed
            if not bool(activation.requires_grad):
                raise ALFSIntegrityError(
                    f"layer {layer} is disconnected from the layer-zero ALFS leaf"
                )
            activations[layer] = activation
            return activation

        return capture

    parameters = tuple(backend.model.parameters())
    original_flags = tuple(bool(parameter.requires_grad) for parameter in parameters)
    parameter_gradients_allocated = False
    parameter_gradients_disabled = False
    gradients: tuple[Any, ...] | None = None
    logits: Any | None = None
    objective: Any | None = None
    backend.model.zero_grad(set_to_none=True)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        parameter_gradients_disabled = not any(
            bool(parameter.requires_grad) for parameter in parameters
        )
        if not parameter_gradients_disabled:
            raise ALFSIntegrityError("ALFS could not disable model parameter gradients")
        hooks = [(f"blocks.{layer}.hook_out", hook_for(layer)) for layer in layers]
        with torch.enable_grad(), backend.model.hooks(fwd_hooks=hooks):
            output = backend.model(tokens)
            if tuple(sorted(activations)) != layers or any(
                hook_calls[layer] != 1 for layer in layers
            ):
                raise ALFSIntegrityError("ALFS did not observe every layer exactly once")
            if (
                getattr(output, "ndim", None) != 3
                or int(output.shape[0]) != 1
                or int(output.shape[1]) != int(tokens.shape[1])
            ):
                raise ALFSIntegrityError("model output has the wrong shape")
            logits = output[0, -1].float()
            if not bool(torch.isfinite(logits).all().item()):
                raise ALFSIntegrityError("full logits contain non-finite values")
            objective = logits[positive_id] - logits[negative_id]
            gradients = torch.autograd.grad(
                objective,
                tuple(activations[layer] for layer in layers),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            parameter_gradients_allocated = any(
                parameter.grad is not None for parameter in backend.model.parameters()
            )
            if parameter_gradients_allocated:
                raise ALFSIntegrityError("ALFS allocated model parameter gradients")
    finally:
        backend.model.zero_grad(set_to_none=True)
        for parameter, original in zip(parameters, original_flags, strict=True):
            parameter.requires_grad_(original)

    if gradients is None or logits is None or objective is None or layer0_leaf is None:
        raise ALFSIntegrityError("ALFS capture did not complete")
    residual_tensor = torch.stack([residual_rows[layer] for layer in layers]).contiguous()
    gradient_rows = []
    for layer, gradient in zip(layers, gradients, strict=True):
        rows = gradient if layer == 0 else gradient[0, list(slots)]
        gradient_rows.append(rows.detach().cpu().float().contiguous().clone())
    gradient_tensor = torch.stack(gradient_rows).contiguous()
    full_logits = logits.detach().cpu().float().contiguous().clone()
    if tuple(residual_tensor.shape) != (LAYER_COUNT, SLOT_COUNT, width):
        raise ALFSIntegrityError("ALFS residual tensor shape differs")
    if gradient_tensor.shape != residual_tensor.shape:
        raise ALFSIntegrityError("ALFS gradient tensor shape differs")
    if not bool(torch.isfinite(gradient_tensor).all().item()):
        raise ALFSIntegrityError("ALFS gradients contain non-finite values")

    margin = float(objective.detach().cpu().item())
    if margin != expected_margin:
        raise ALFSIntegrityError("ALFS margin differs from immutable state zero")
    if tensor_float32_sha256(full_logits) != expected_logits_hash:
        raise ALFSIntegrityError("ALFS full logits differ from immutable state zero")
    if tensor_float32_sha256(residual_tensor[0]) != expected_residual_hash:
        raise ALFSIntegrityError("ALFS layer-zero residuals differ from immutable CSMS")
    if tensor_float32_sha256(gradient_tensor[0]) != expected_gradient_hash:
        raise ALFSIntegrityError("ALFS layer-zero gradients differ from immutable CSMS")
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
        raise ALFSIntegrityError("independent A/B score differs from differentiated margin")

    audit = _hashed_record(
        {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "capture_kind": "all_24_layers_four_slots_one_forward_one_backward",
            "layers": list(layers),
            "hook_names": [f"blocks.{layer}.hook_out" for layer in layers],
            "hook_call_counts": {str(layer): hook_calls[layer] for layer in layers},
            "slot_indices": list(slots),
            "prompt_sha256": text_sha256(prompt),
            "prompt_length": int(tokens.shape[1]),
            "prompt_token_ids_sha256": boundary.prompt_prefix_token_ids_sha256,
            "choice_boundary_evidence_sha256": boundary.evidence_sha256,
            "positive_token_id": positive_id,
            "negative_token_id": negative_id,
            "positive_minus_negative_log_odds": margin,
            "residuals_float32_sha256": tensor_float32_sha256(residual_tensor),
            "gradients_float32_sha256": tensor_float32_sha256(gradient_tensor),
            "full_logits_float32_sha256": tensor_float32_sha256(full_logits),
            "source_layer0_residuals_reproduced": True,
            "source_layer0_gradients_reproduced": True,
            "source_full_logits_reproduced": True,
            "source_margin_reproduced": True,
            "source_tokenization_reproduced": True,
            "zero_direction": True,
            "maximum_abs_layer0_reconstruction_delta": reconstruction_delta,
            "later_layer_hooks_return_activation_unchanged": True,
            "model_forward_evaluations": 1,
            "model_backward_evaluations": 1,
            "model_parameters_requires_grad_disabled_during_capture": (
                parameter_gradients_disabled
            ),
            "model_parameter_requires_grad_true_count_before_capture": sum(original_flags),
            "model_parameter_requires_grad_flags_restored_after_capture": all(
                bool(parameter.requires_grad) == original
                for parameter, original in zip(parameters, original_flags, strict=True)
            ),
            "model_parameter_gradients_allocated": parameter_gradients_allocated,
            "detach_scope": "entire_layer_zero_only_then_four_rows_as_leaves",
        },
        "audit_sha256",
    )
    return AllLayerFourSlotCapture(
        layers=layers,
        slot_indices=slots,
        residuals=residual_tensor,
        gradients=gradient_tensor,
        full_logits=full_logits,
        positive_minus_negative_log_odds=margin,
        positive_token_id=positive_id,
        negative_token_id=negative_id,
        audit=_freeze(audit),
    )


def _valid_form_id(record: Mapping[str, Any]) -> str:
    value = record.get("form_id")
    if not isinstance(value, str) or not value:
        raise ALFSIntegrityError("every source row needs one non-empty form_id")
    return value


def _form(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("form")
    if not isinstance(value, Mapping):
        raise ALFSIntegrityError("every source row needs one rendered form")
    return value


def _semantic_grid(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the exact 64-scenario plus 16-control factorial grid."""

    if len(records) != 80:
        raise ALFSIntegrityError("ALFS requires exactly 80 source records")
    seen_ids: set[str] = set()
    scenario_cells: set[tuple[str, int, str, str, bool]] = set()
    control_cells: set[tuple[str, bool]] = set()
    scenarios: set[str] = set()
    controls: set[str] = set()
    target_indices: list[int] = []
    nuisance_indices: list[int] = []
    for index, record in enumerate(records):
        form_id = _valid_form_id(record)
        if form_id in seen_ids:
            raise ALFSIntegrityError("source records contain duplicate form IDs")
        seen_ids.add(form_id)
        form = _form(record)
        family = form.get("family")
        if family == "scenario":
            scenario = form.get("scenario_id")
            assignment = form.get("assignment")
            target = form.get("target")
            event = form.get("event")
            order = form.get("preserve_first")
            if (
                not isinstance(scenario, str)
                or not scenario
                or type(assignment) is not int
                or assignment not in {0, 1}
                or target not in {"self", "other"}
                or event not in {"permanent", "temporary"}
                or type(order) is not bool
            ):
                raise ALFSIntegrityError("one scenario row has invalid factorial semantics")
            cell = (scenario, assignment, str(target), str(event), order)
            if cell in scenario_cells:
                raise ALFSIntegrityError("duplicate scenario factorial cell")
            scenario_cells.add(cell)
            scenarios.add(scenario)
            if target == "self" and event == "permanent":
                target_indices.append(index)
            else:
                nuisance_indices.append(index)
        elif family == "unrelated":
            control = form.get("control_id")
            order = form.get("preferred_first")
            if not isinstance(control, str) or not control or type(order) is not bool:
                raise ALFSIntegrityError("one unrelated row has invalid semantics")
            cell = (control, order)
            if cell in control_cells:
                raise ALFSIntegrityError("duplicate unrelated factorial cell")
            control_cells.add(cell)
            controls.add(control)
            nuisance_indices.append(index)
        else:
            raise ALFSIntegrityError("form family must be exactly scenario or unrelated")
    if len(scenarios) != 4 or len(controls) != 8:
        raise ALFSIntegrityError("ALFS requires four scenarios and eight controls")
    expected_scenarios = {
        (scenario, assignment, target, event, order)
        for scenario in scenarios
        for assignment in (0, 1)
        for target in ("self", "other")
        for event in ("permanent", "temporary")
        for order in (False, True)
    }
    expected_controls = {
        (control, order) for control in controls for order in (False, True)
    }
    if scenario_cells != expected_scenarios or control_cells != expected_controls:
        raise ALFSIntegrityError("the source factorial grid is incomplete or mislabeled")
    if len(target_indices) != 16 or len(nuisance_indices) != 64:
        raise ALFSIntegrityError("ALFS target/nuisance coverage differs")
    return {
        "scenario_ids": tuple(sorted(scenarios)),
        "control_ids": tuple(sorted(controls)),
        "target_indices": tuple(target_indices),
        "nuisance_indices": tuple(nuisance_indices),
        "form_ids_sha256": canonical_sha256([_valid_form_id(row) for row in records]),
    }


def build_outer_folds(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Build four deterministic scenario-plus-two-control held-out folds."""

    grid = _semantic_grid(records)
    scenarios = grid["scenario_ids"]
    controls = grid["control_ids"]
    folds: list[dict[str, Any]] = []
    for fold_index, scenario in enumerate(scenarios):
        held_controls = tuple(controls[2 * fold_index : 2 * fold_index + 2])
        train_targets: list[int] = []
        train_scenario_nuisance: list[int] = []
        train_unrelated: list[int] = []
        held_targets: list[int] = []
        held_scenario_nuisance: list[int] = []
        held_unrelated: list[int] = []
        for index, record in enumerate(records):
            form = _form(record)
            if form["family"] == "scenario":
                is_target = form["target"] == "self" and form["event"] == "permanent"
                held = form["scenario_id"] == scenario
                if held and is_target:
                    held_targets.append(index)
                elif held:
                    held_scenario_nuisance.append(index)
                elif is_target:
                    train_targets.append(index)
                else:
                    train_scenario_nuisance.append(index)
            else:
                if form["control_id"] in held_controls:
                    held_unrelated.append(index)
                else:
                    train_unrelated.append(index)
        train_nuisance = [*train_scenario_nuisance, *train_unrelated]
        held_nuisance = [*held_scenario_nuisance, *held_unrelated]
        train_all = [*train_targets, *train_nuisance]
        held_all = [*held_targets, *held_nuisance]
        counts = tuple(
            len(value)
            for value in (
                train_targets,
                train_scenario_nuisance,
                train_unrelated,
                train_nuisance,
                train_all,
                held_targets,
                held_scenario_nuisance,
                held_unrelated,
                held_nuisance,
                held_all,
            )
        )
        if counts != (12, 36, 12, 48, 60, 4, 12, 4, 16, 20):
            raise ALFSIntegrityError("one ALFS outer fold has incorrect row coverage")
        if set(train_all) & set(held_all) or set(train_all) | set(held_all) != set(range(80)):
            raise ALFSIntegrityError("one ALFS outer fold overlaps or omits rows")
        fold = _hashed_record(
            {
                "fold_index": fold_index,
                "held_scenario_id": scenario,
                "held_control_ids": list(held_controls),
                "mapping_rule": "sorted_control_ids_consecutive_pairs_to_sorted_scenarios",
                "training_target_indices": train_targets,
                "training_scenario_nuisance_indices": train_scenario_nuisance,
                "training_unrelated_indices": train_unrelated,
                "training_nuisance_indices": train_nuisance,
                "training_all_indices": train_all,
                "held_target_indices": held_targets,
                "held_scenario_nuisance_indices": held_scenario_nuisance,
                "held_unrelated_indices": held_unrelated,
                "held_nuisance_indices": held_nuisance,
                "held_all_indices": held_all,
            },
            "fold_sha256",
        )
        folds.append(fold)
    return tuple(folds)


def _target_pairs(
    records: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> tuple[dict[str, Any], ...]:
    pairs: dict[tuple[str, int], list[int]] = {}
    for index in indices:
        form = _form(records[index])
        if not (
            form.get("family") == "scenario"
            and form.get("target") == "self"
            and form.get("event") == "permanent"
        ):
            raise ALFSIntegrityError("a target pair contains a non-target form")
        key = (str(form["scenario_id"]), int(form["assignment"]))
        pairs.setdefault(key, []).append(index)
    result: list[dict[str, Any]] = []
    for (scenario, assignment), pair_indices in sorted(pairs.items()):
        ordered = sorted(pair_indices, key=lambda value: bool(_form(records[value])["preserve_first"]))
        orders = {bool(_form(records[value])["preserve_first"]) for value in ordered}
        if len(ordered) != 2 or orders != {False, True}:
            raise ALFSIntegrityError("each target pair must contain both answer orders exactly once")
        result.append(
            _hashed_record(
                {
                    "scenario_id": scenario,
                    "assignment": assignment,
                    "indices": ordered,
                    "form_ids": [_valid_form_id(records[value]) for value in ordered],
                },
                "pair_sha256",
            )
        )
    if len(result) * 2 != len(indices):
        raise ALFSIntegrityError("target-pair coverage differs")
    return tuple(result)


def training_only_slot_scales(
    residuals: Any, training_indices: Sequence[int]
) -> np.ndarray:
    """Fit a [24,4] geometric-mean residual scale on training rows only."""

    array = np.asarray(residuals, dtype=np.float64)
    indices = tuple(int(value) for value in training_indices)
    if array.ndim != 4 or array.shape[0] != 80 or array.shape[1:3] != (
        LAYER_COUNT,
        SLOT_COUNT,
    ):
        raise ValueError("residuals must have shape [80,24,4,d_model]")
    if not indices or len(set(indices)) != len(indices) or any(
        value < 0 or value >= 80 for value in indices
    ):
        raise ValueError("training_indices must be unique in-range rows")
    norms = np.linalg.norm(array[list(indices)], axis=3)
    if not np.isfinite(norms).all() or bool(np.any(norms <= 0.0)):
        raise ALFSIntegrityError("training residual scale contains a nonpositive norm")
    scales = np.exp(np.mean(np.log(norms), axis=0))
    if scales.shape != (LAYER_COUNT, SLOT_COUNT) or not np.isfinite(scales).all():
        raise ALFSIntegrityError("training-only slot scales are invalid")
    return np.asarray(scales, dtype=np.float64, order="C")


def standardized_rows(gradients_at_layer: Any, slot_scales: Any) -> np.ndarray:
    raw = np.asarray(gradients_at_layer, dtype=np.float64)
    scales = np.asarray(slot_scales, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[:2] != (80, SLOT_COUNT):
        raise ValueError("gradients_at_layer must have shape [80,4,d_model]")
    if scales.shape != (SLOT_COUNT,) or not np.isfinite(scales).all() or bool(
        np.any(scales <= 0.0)
    ):
        raise ValueError("slot_scales must contain four positive finite values")
    if not np.isfinite(raw).all():
        raise ValueError("gradients must be finite")
    return (raw * scales[None, :, None]).reshape(raw.shape[0], -1)


def frozen_nuisance_rowspace(nuisance_rows: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit and certify the fixed SVD nuisance rowspace once per layer/fold."""

    rows = np.asarray(nuisance_rows, dtype=np.float64, order="C")
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0:
        raise ValueError("nuisance_rows must be one non-empty matrix")
    if not np.isfinite(rows).all():
        raise ValueError("nuisance_rows must be finite")
    norms = np.linalg.norm(rows, axis=1)
    nonzero = norms > 0.0
    normalized = np.zeros_like(rows)
    normalized[nonzero] = rows[nonzero] / norms[nonzero, None]
    _, singular_values, vh = np.linalg.svd(normalized, full_matrices=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(DEFAULT_SVD_ATOL, DEFAULT_SVD_RTOL * largest)
    rank = int(np.count_nonzero(singular_values > threshold))
    basis = np.asarray(vh[:rank], dtype=np.float64, order="C").copy(order="C")
    for row in basis:
        anchor = int(np.argmax(np.abs(row)))
        if row[anchor] < 0.0:
            row *= -1.0
    reconstructed = (rows @ basis.T) @ basis
    reconstruction_error = float(np.max(np.abs(rows - reconstructed)))
    orthogonality_error = (
        float(np.max(np.abs(basis @ basis.T - np.eye(rank)))) if rank else 0.0
    )
    checks = {
        "rowspace_reconstruction_within_double_certificate_tolerance": (
            reconstruction_error <= DOUBLE_CERTIFICATE_TOLERANCE
        ),
        "basis_orthogonality_within_double_certificate_tolerance": (
            orthogonality_error <= DOUBLE_CERTIFICATE_TOLERANCE
        ),
    }
    record = _hashed_record(
        {
            "method": "normalized_row_svd_fixed_scientific_tolerances",
            "input_row_count": int(rows.shape[0]),
            "dimension": int(rows.shape[1]),
            "zero_row_count": int(np.count_nonzero(~nonzero)),
            "svd_rtol": DEFAULT_SVD_RTOL,
            "svd_atol": DEFAULT_SVD_ATOL,
            "rank_threshold": threshold,
            "rank": rank,
            "singular_values": singular_values.tolist(),
            "maximum_abs_reconstruction_error": reconstruction_error,
            "maximum_abs_basis_orthogonality_error": orthogonality_error,
            "basis_identity": _array_identity(basis, dtype="float64"),
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
    )
    if not record["passes"]:
        raise ALFSIntegrityError("frozen nuisance rowspace failed its certificate")
    basis.setflags(write=False)
    return basis, record


def project_out_frozen_rowspace(rows: Any, basis: Any) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    frozen = np.asarray(basis, dtype=np.float64)
    if values.ndim != 2 or frozen.ndim != 2 or values.shape[1] != frozen.shape[1]:
        raise ValueError("rows and basis must be compatible matrices")
    return np.asarray(values - (values @ frozen.T) @ frozen, dtype=np.float64, order="C")


def solve_raw_oracle(row: Any, baseline_offset: float) -> tuple[dict[str, Any], np.ndarray | None]:
    """Analytic single-row lower bound L_i / ||g_i|| and its direction."""

    gradient = np.asarray(row, dtype=np.float64)
    offset = float(baseline_offset)
    if gradient.ndim != 1 or not np.isfinite(gradient).all() or not math.isfinite(offset):
        raise ValueError("raw oracle inputs must be finite vectors/scalars")
    norm = float(np.linalg.norm(gradient))
    lower = abs(offset) + TARGET_MARGIN
    if norm == 0.0:
        record = _hashed_record(
            {
                "method": "raw_single_row_analytic_oracle",
                "status": "infeasible_zero_gradient",
                "required_slope": lower,
                "gradient_norm": norm,
                "minimum_norm": None,
                "passes": False,
            }
        )
        return record, None
    direction = np.asarray((lower / (norm * norm)) * gradient, dtype=np.float64, order="C")
    value = float(gradient @ direction)
    expected_norm = lower / norm
    observed_norm = float(np.linalg.norm(direction))
    checks = {
        "finite": bool(np.isfinite(direction).all()),
        "target_constraint": value - lower >= -DOUBLE_CERTIFICATE_TOLERANCE,
        "analytic_minimum_norm_identity": abs(observed_norm - expected_norm)
        <= DOUBLE_CERTIFICATE_TOLERANCE,
    }
    record = _hashed_record(
        {
            "method": "raw_single_row_analytic_oracle",
            "status": "certified" if all(checks.values()) else "numerically_indeterminate",
            "required_slope": lower,
            "gradient_norm": norm,
            "target_value": value,
            "target_slack": value - lower,
            "minimum_norm_formula": "(abs(baseline_offset)+0.05)/gradient_l2",
            "minimum_norm": observed_norm,
            "direction_identity": _array_identity(direction, dtype="float64"),
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
    )
    return record, direction if record["passes"] else None


def solve_paired_oracle(
    *,
    target_rows: Any,
    target_offsets: Any,
    nuisance_rows: Any | None = None,
    frozen_basis: Any | None = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Solve a two-order minimum-norm QP, optionally in one frozen nullspace."""

    targets = np.asarray(target_rows, dtype=np.float64)
    offsets = np.asarray(target_offsets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[0] != 2 or offsets.shape != (2,):
        raise ValueError("paired oracle requires exactly two target rows and offsets")
    if targets.shape[1] % SLOT_COUNT != 0:
        raise ValueError("paired oracle width must flatten four equal slot rows")
    if not np.isfinite(targets).all() or not np.isfinite(offsets).all():
        raise ValueError("paired oracle inputs must be finite")
    dimension = targets.shape[1]
    nuisances = (
        np.zeros((0, dimension), dtype=np.float64)
        if nuisance_rows is None
        else np.asarray(nuisance_rows, dtype=np.float64)
    )
    if nuisances.ndim != 2 or nuisances.shape[1] != dimension or not np.isfinite(nuisances).all():
        raise ValueError("nuisance rows have invalid shape or values")
    if nuisances.shape[0] == 0:
        projected = targets
        projection_record: dict[str, Any] | None = None
    else:
        if frozen_basis is None:
            basis, projection_record = frozen_nuisance_rowspace(nuisances)
        else:
            basis = np.asarray(frozen_basis, dtype=np.float64)
            if basis.ndim != 2 or basis.shape[1] != dimension:
                raise ValueError("frozen nuisance basis has invalid shape")
            projection_record = {
                "reused_frozen_training_basis": True,
                "basis_identity": _array_identity(basis, dtype="float64"),
            }
        projected = project_out_frozen_rowspace(targets, basis)
    projected_norms = np.linalg.norm(projected, axis=1)
    original_norms = np.linalg.norm(targets, axis=1)
    wholly_nuisance = projected_norms <= (
        DOUBLE_CERTIFICATE_TOLERANCE * np.maximum(1.0, original_norms)
    )
    if bool(np.any(wholly_nuisance)):
        record = _hashed_record(
            {
                "method": "paired_minimum_norm_qp",
                "status": "infeasible_target_wholly_in_frozen_nuisance_rowspace",
                "projected_target_norms": projected_norms.tolist(),
                "original_target_norms": original_norms.tolist(),
                "wholly_nuisance_checks": wholly_nuisance.tolist(),
                "classification_tolerance": DOUBLE_CERTIFICATE_TOLERANCE,
                "projected_solver": None,
                "frozen_nuisance_rowspace": projection_record,
                "original_constraint_certificate": None,
                "passes": False,
            }
        )
        return record, None
    lower = np.abs(offsets) + TARGET_MARGIN
    canonical_order = sorted(
        range(2),
        key=lambda index: hashlib.sha256(
            np.asarray(projected[index], dtype="<f8", order="C").tobytes()
            + np.asarray(lower[index], dtype="<f8").tobytes()
        ).hexdigest(),
    )
    canonical_targets = projected[canonical_order]
    canonical_lower = lower[canonical_order]
    gram = canonical_targets @ canonical_targets.T
    singular_values = np.linalg.svd(gram, compute_uv=False)
    rank_threshold = max(
        DEFAULT_SVD_ATOL,
        DEFAULT_SVD_RTOL * (float(singular_values[0]) if singular_values.size else 0.0),
    )
    gram_rank = int(np.count_nonzero(singular_values > rank_threshold))
    candidates: list[tuple[float, str, np.ndarray, dict[str, Any]]] = []
    candidate_records: list[dict[str, Any]] = []

    def consider(active: tuple[int, ...], multipliers: np.ndarray) -> None:
        direction_candidate = np.asarray(
            canonical_targets.T @ multipliers,
            dtype=np.float64,
            order="C",
        )
        values = canonical_targets @ direction_candidate
        slacks = values - canonical_lower
        stationarity = direction_candidate - canonical_targets.T @ multipliers
        complementarity = multipliers * slacks
        checks = {
            "finite": bool(
                np.isfinite(direction_candidate).all()
                and np.isfinite(multipliers).all()
            ),
            "primal_feasible": float(np.min(slacks))
            >= -DOUBLE_CERTIFICATE_TOLERANCE,
            "dual_nonnegative": float(np.min(multipliers))
            >= -DOUBLE_CERTIFICATE_TOLERANCE,
            "stationarity": float(np.max(np.abs(stationarity)))
            <= DOUBLE_CERTIFICATE_TOLERANCE,
            "complementarity": float(np.max(np.abs(complementarity)))
            <= DOUBLE_CERTIFICATE_TOLERANCE,
        }
        identity = _array_identity(direction_candidate, dtype="float64")
        candidate = _hashed_record(
            {
                "active_canonical_constraints": list(active),
                "multipliers": multipliers.tolist(),
                "canonical_target_values": values.tolist(),
                "canonical_target_slacks": slacks.tolist(),
                "stationarity_max_abs": float(np.max(np.abs(stationarity))),
                "complementarity_max_abs": float(
                    np.max(np.abs(complementarity))
                ),
                "minimum_norm": float(np.linalg.norm(direction_candidate)),
                "direction_identity": identity,
                "checks": checks,
                "passes": bool(all(checks.values())),
            }
        )
        candidate_records.append(candidate)
        if candidate["passes"]:
            candidates.append(
                (
                    float(candidate["minimum_norm"]),
                    identity["raw_little_endian_bytes_sha256"],
                    direction_candidate,
                    candidate,
                )
            )

    for active_index in range(2):
        multipliers = np.zeros(2, dtype=np.float64)
        multipliers[active_index] = (
            canonical_lower[active_index] / gram[active_index, active_index]
        )
        consider((active_index,), multipliers)
    if gram_rank == 2:
        try:
            both_multipliers = np.linalg.solve(gram, canonical_lower)
        except np.linalg.LinAlgError:
            both_multipliers = None
        if both_multipliers is not None:
            consider((0, 1), both_multipliers)
    solver_record = _hashed_record(
        {
            "method": "analytic_two_inequality_active_set",
            "objective": "minimum_one_half_squared_standardized_frobenius_norm",
            "enumerated_active_sets": [[0], [1], [0, 1]],
            "canonical_constraint_order": canonical_order,
            "canonical_target_rows_identity": _array_identity(
                canonical_targets, dtype="float64"
            ),
            "canonical_required_slopes": canonical_lower.tolist(),
            "gram_singular_values": singular_values.tolist(),
            "gram_rank_threshold": rank_threshold,
            "gram_rank": gram_rank,
            "double_certificate_tolerance": DOUBLE_CERTIFICATE_TOLERANCE,
            "candidates": candidate_records,
            "status": (
                "certified_candidate_found"
                if candidates
                else (
                    "infeasible_degenerate_two_halfspaces"
                    if gram_rank < 2
                    else "numerically_indeterminate_no_certified_active_set"
                )
            ),
            "passes": bool(candidates),
        }
    )
    if not candidates:
        record = _hashed_record(
            {
                "method": "paired_minimum_norm_qp",
                "status": solver_record["status"],
                "projected_solver": solver_record,
                "frozen_nuisance_rowspace": projection_record,
                "original_constraint_certificate": None,
                "passes": False,
            }
        )
        return record, None
    _, _, direction, selected_candidate = min(
        candidates, key=lambda value: (value[0], value[1])
    )
    target_values = targets @ direction
    nuisance_values = nuisances @ direction
    raw_checks = {
        "targets": float(np.min(target_values - lower))
        >= -DOUBLE_CERTIFICATE_TOLERANCE,
        "exact_frozen_nuisance_null": (
            float(np.max(np.abs(nuisance_values))) if nuisance_values.size else 0.0
        )
        <= DOUBLE_CERTIFICATE_TOLERANCE,
        "finite": bool(
            np.isfinite(direction).all()
            and np.isfinite(target_values).all()
            and np.isfinite(nuisance_values).all()
        ),
    }
    try:
        independent = certify_minimum_l2_candidate(
            direction,
            targets,
            np.zeros(2, dtype=np.float64),
            margin=lower,
            nuisance_rows=nuisances if nuisances.size else None,
            nuisance_bound=0.0,
            primal_tolerance=DOUBLE_CERTIFICATE_TOLERANCE,
        )
    except (DecisionMarginOptimalityError, ValueError) as error:
        independent = {
            "passes": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    passes = bool(
        solver_record.get("passes") is True
        and selected_candidate.get("passes") is True
        and all(raw_checks.values())
        and independent.get("passes") is True
    )
    record = _hashed_record(
        {
            "method": "paired_minimum_norm_qp",
            "status": "certified" if passes else "numerically_indeterminate",
            "minimum_norm": float(np.linalg.norm(direction)),
            "required_slopes": lower.tolist(),
            "target_values": target_values.tolist(),
            "target_slacks": (target_values - lower).tolist(),
            "maximum_abs_frozen_nuisance_slope": (
                float(np.max(np.abs(nuisance_values))) if nuisance_values.size else 0.0
            ),
            "direction_identity": _array_identity(direction, dtype="float64"),
            "projected_solver": solver_record,
            "selected_analytic_candidate": selected_candidate,
            "frozen_nuisance_rowspace": projection_record,
            "original_constraint_certificate": independent,
            "raw_checks": raw_checks,
            "passes": passes,
        }
    )
    return record, direction if passes else None


def _realize(
    residuals: np.ndarray,
    requested: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(residuals, dtype=np.float32, order="C")
    changed = np.asarray(base + requested[None, :, :], dtype=np.float32, order="C")
    physical = np.asarray(changed - base, dtype=np.float32, order="C")
    standardized = (
        physical.astype(np.float64) / scales[None, :, None]
    ).reshape(base.shape[0], base.shape[1] * base.shape[2])
    return physical, standardized


def _dose_record(edits: np.ndarray, residuals: np.ndarray) -> dict[str, Any]:
    edit = np.asarray(edits, dtype=np.float64)
    base = np.asarray(residuals, dtype=np.float64)
    if edit.shape != base.shape or edit.ndim != 3 or edit.shape[1] != SLOT_COUNT:
        raise ValueError("dose arrays must have matching [n,4,d] shapes")
    row_base = np.linalg.norm(base, axis=2)
    prompt_base = np.linalg.norm(base.reshape(base.shape[0], -1), axis=1)
    if bool(np.any(row_base <= 0.0)) or bool(np.any(prompt_base <= 0.0)):
        raise ALFSIntegrityError("dose audit encountered a zero residual norm")
    row_relative = np.linalg.norm(edit, axis=2) / row_base
    prompt_relative = np.linalg.norm(edit.reshape(edit.shape[0], -1), axis=1) / prompt_base
    row_max = float(np.max(row_relative))
    prompt_max = float(np.max(prompt_relative))
    return _hashed_record(
        {
            "scope_row_count": int(base.shape[0]),
            "relative_l2_cap": QUALIFICATION_CAP,
            "maximum_per_slot_row_relative_l2": row_max,
            "maximum_per_prompt_frobenius_relative_l2": prompt_max,
            "per_slot_row_relative_l2": row_relative.tolist(),
            "per_prompt_frobenius_relative_l2": prompt_relative.tolist(),
            "passes": row_max <= QUALIFICATION_CAP and prompt_max <= QUALIFICATION_CAP,
        }
    )


def physical_oracle_recertificate(
    *,
    standardized_direction: Any,
    slot_scales: Any,
    target_rows: Any,
    target_offsets: Any,
    exact_nuisance_rows: Any,
    target_residuals: Any,
    exact_nuisance_residuals: Any,
    additional_residuals: Any | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]]]:
    """Recertify requested and actually realized float32 edits under both signs."""

    direction = np.asarray(standardized_direction, dtype=np.float64)
    scales = np.asarray(slot_scales, dtype=np.float64)
    targets = np.asarray(target_rows, dtype=np.float64)
    offsets = np.asarray(target_offsets, dtype=np.float64)
    nuisances = np.asarray(exact_nuisance_rows, dtype=np.float64)
    width = direction.size // SLOT_COUNT if direction.ndim == 1 else 0
    if (
        direction.ndim != 1
        or direction.size % SLOT_COUNT != 0
        or scales.shape != (SLOT_COUNT,)
        or bool(np.any(scales <= 0.0))
        or targets.ndim != 2
        or targets.shape[1] != direction.size
        or offsets.shape != (targets.shape[0],)
        or nuisances.ndim != 2
        or nuisances.shape[1] != direction.size
    ):
        raise ValueError("physical recertification inputs have invalid shapes")
    target_base = np.asarray(target_residuals, dtype=np.float32)
    nuisance_base = np.asarray(exact_nuisance_residuals, dtype=np.float32)
    additional_base = (
        np.zeros((0, SLOT_COUNT, width), dtype=np.float32)
        if additional_residuals is None
        else np.asarray(additional_residuals, dtype=np.float32)
    )
    if target_base.shape != (targets.shape[0], SLOT_COUNT, width):
        raise ValueError("target residuals do not match target rows")
    if nuisance_base.shape != (nuisances.shape[0], SLOT_COUNT, width):
        raise ValueError("nuisance residuals do not match nuisance rows")
    if additional_base.ndim != 3 or additional_base.shape[1:] != (SLOT_COUNT, width):
        raise ValueError("additional residuals have invalid shape")

    positive = np.asarray(
        direction.reshape(SLOT_COUNT, width) * scales[:, None],
        dtype="<f4",
        order="C",
    )
    negative = np.asarray(np.negative(positive), dtype="<f4", order="C")
    sign_bits_exact = bool(
        np.array_equal(
            negative.view("<u4"),
            np.bitwise_xor(positive.view("<u4"), np.uint32(0x80000000)),
        )
    )
    all_base = np.concatenate((target_base, nuisance_base, additional_base), axis=0)
    requested_dose_by_sign: dict[str, Any] = {}
    realized: dict[int, dict[str, np.ndarray]] = {}
    sign_records: dict[str, Any] = {}
    all_realized_norms: list[float] = []
    for sign, requested in ((1, positive), (-1, negative)):
        target_edit, target_standardized = _realize(target_base, requested, scales)
        nuisance_edit, nuisance_standardized = _realize(nuisance_base, requested, scales)
        additional_edit, additional_standardized = _realize(additional_base, requested, scales)
        all_edit = np.concatenate((target_edit, nuisance_edit, additional_edit), axis=0)
        target_movement = np.einsum("ij,ij->i", targets, target_standardized)
        nuisance_movement = np.einsum("ij,ij->i", nuisances, nuisance_standardized)
        oriented_target = sign * target_movement
        lower = np.abs(offsets) + TARGET_MARGIN
        changed = offsets + target_movement
        endpoint = (
            changed >= TARGET_MARGIN if sign == 1 else changed <= -TARGET_MARGIN
        )
        norms = np.linalg.norm(
            sign * all_edit.astype(np.float64) / scales[None, :, None],
            axis=(1, 2),
        )
        all_realized_norms.extend(norms.tolist())
        actual_dose = _dose_record(all_edit, all_base)
        requested_broadcast = np.broadcast_to(requested[None], all_base.shape)
        requested_dose = _dose_record(requested_broadcast, all_base)
        requested_dose_by_sign[str(sign)] = requested_dose
        checks = {
            "target_certificate_with_float32_tolerance": float(
                np.min(oriented_target - lower)
            )
            >= -FLOAT32_PHYSICAL_TOLERANCE,
            "target_endpoint_with_float32_tolerance": bool(
                np.all(
                    changed >= TARGET_MARGIN - FLOAT32_PHYSICAL_TOLERANCE
                    if sign == 1
                    else changed <= -TARGET_MARGIN + FLOAT32_PHYSICAL_TOLERANCE
                )
            ),
            "exact_training_nuisance_with_float32_tolerance": (
                float(np.max(np.abs(nuisance_movement)))
                if nuisance_movement.size
                else 0.0
            )
            <= FLOAT32_PHYSICAL_TOLERANCE,
            "realized_standardized_norm_cap_strict": float(np.max(norms))
            <= QUALIFICATION_CAP,
            "requested_dose_cap_strict": requested_dose["passes"],
            "actual_dose_cap_strict": actual_dose["passes"],
        }
        sign_records[str(sign)] = _hashed_record(
            {
                "requested_sign": sign,
                "target_movements": target_movement.tolist(),
                "oriented_target_slopes": oriented_target.tolist(),
                "target_endpoints": changed.tolist(),
                "target_endpoint_checks": endpoint.tolist(),
                "minimum_target_slack": float(np.min(oriented_target - lower)),
                "exact_nuisance_movements": nuisance_movement.tolist(),
                "maximum_abs_exact_nuisance_movement": (
                    float(np.max(np.abs(nuisance_movement)))
                    if nuisance_movement.size
                    else 0.0
                ),
                "minimum_realized_standardized_frobenius_norm": float(np.min(norms)),
                "maximum_realized_standardized_frobenius_norm": float(np.max(norms)),
                "actual_signed_edits_identity": _array_identity(all_edit, dtype="float32"),
                "actual_dose": actual_dose,
                "checks": checks,
                "passes": bool(all(checks.values())),
            }
        )
        realized[sign] = {
            "target_physical": target_edit,
            "target_standardized": target_standardized,
            "nuisance_physical": nuisance_edit,
            "nuisance_standardized": nuisance_standardized,
            "additional_physical": additional_edit,
            "additional_standardized": additional_standardized,
            "all_physical": all_edit,
        }
    intended_norm = float(np.linalg.norm(direction))
    requested_standardized_norm = float(
        np.linalg.norm(positive.astype(np.float64) / scales[:, None])
    )
    maximum_realized = float(np.max(all_realized_norms))
    checks = {
        "intended_standardized_norm_cap_strict": intended_norm <= QUALIFICATION_CAP,
        "requested_float32_standardized_norm_cap_strict": (
            requested_standardized_norm <= QUALIFICATION_CAP
        ),
        "both_sign_requested_dose_caps_strict": all(
            value["passes"] for value in requested_dose_by_sign.values()
        ),
        "both_sign_realized_norm_caps_strict": maximum_realized <= QUALIFICATION_CAP,
        "both_sign_physical_certificates": all(
            value["passes"] for value in sign_records.values()
        ),
        "negative_is_exact_float32_unary_sign_bit_flip": sign_bits_exact,
    }
    record = _hashed_record(
        {
            "physical_dtype": "float32",
            "scope": {
                "target_rows": int(target_base.shape[0]),
                "exact_nuisance_rows": int(nuisance_base.shape[0]),
                "additional_dose_rows": int(additional_base.shape[0]),
                "total_base_rows": int(all_base.shape[0]),
            },
            "intended_standardized_frobenius_norm": intended_norm,
            "requested_float32_standardized_frobenius_norm": requested_standardized_norm,
            "maximum_realized_standardized_frobenius_norm": maximum_realized,
            "norm_and_dose_cap": QUALIFICATION_CAP,
            "norm_and_dose_cap_tolerance": 0.0,
            "float32_target_and_exact_null_tolerance": FLOAT32_PHYSICAL_TOLERANCE,
            "requested_positive_identity": _array_identity(positive, dtype="float32"),
            "requested_negative_identity": _array_identity(negative, dtype="float32"),
            "negative_construction": "unary_negation_of_same_positive_float32_matrix",
            "negative_sign_bit_certificate": "negative_uint32_equals_positive_xor_0x80000000",
            "requested_dose_by_sign": requested_dose_by_sign,
            "signs": sign_records,
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
    )
    return record, realized


def _offsets(
    records: Sequence[Mapping[str, Any]], indices: Sequence[int] | None = None
) -> np.ndarray:
    selected = tuple(range(80)) if indices is None else tuple(int(value) for value in indices)
    if len(records) != 80 or not selected or len(set(selected)) != len(selected) or any(
        value < 0 or value >= 80 for value in selected
    ):
        raise ALFSIntegrityError("baseline-offset selection is invalid")
    values = np.full(80, np.nan, dtype=np.float64)
    for index in selected:
        values[index] = float(records[index]["positive_minus_negative_log_odds"])
    if not np.isfinite(values[list(selected)]).all():
        raise ALFSIntegrityError("every selected baseline offset must be finite")
    return values


def _matched_other_permanent_indices(
    records: Sequence[Mapping[str, Any]], allowed_indices: Sequence[int]
) -> list[int]:
    allowed = {int(value) for value in allowed_indices}
    return [
        index
        for index, record in enumerate(records)
        if index in allowed
        and _form(record).get("family") == "scenario"
        and _form(record).get("target") == "other"
        and _form(record).get("event") == "permanent"
    ]


def _oracle_with_physical(
    *,
    method: str,
    solver_record: Mapping[str, Any],
    direction: np.ndarray | None,
    scales: np.ndarray,
    rows: np.ndarray,
    offsets: np.ndarray,
    targets: Sequence[int],
    nuisances: Sequence[int],
    residuals: np.ndarray,
    additional: Sequence[int] = (),
) -> tuple[dict[str, Any], np.ndarray | None]:
    if direction is None or not bool(solver_record.get("passes")):
        return (
            _hashed_record(
                {
                    "method": method,
                    "solver": dict(solver_record),
                    "physical_recertification": None,
                    "passes": False,
                }
            ),
            None,
        )
    physical, _ = physical_oracle_recertificate(
        standardized_direction=direction,
        slot_scales=scales,
        target_rows=rows[list(targets)],
        target_offsets=offsets[list(targets)],
        exact_nuisance_rows=rows[list(nuisances)],
        target_residuals=residuals[list(targets)],
        exact_nuisance_residuals=residuals[list(nuisances)],
        additional_residuals=(
            residuals[list(additional)] if additional else None
        ),
    )
    passes = bool(
        solver_record.get("passes")
        and float(solver_record["minimum_norm"]) <= QUALIFICATION_CAP
        and physical["passes"]
    )
    return (
        _hashed_record(
            {
                "method": method,
                "solver": dict(solver_record),
                "physical_recertification": physical,
                "passes": passes,
            }
        ),
        direction if passes else None,
    )


def _solve_nonselecting_global(
    *,
    target_rows: np.ndarray,
    target_offsets: np.ndarray,
    residual_width: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Fail a descriptive global decomposition without aborting selection."""

    target = np.asarray(target_rows, dtype=np.float64)
    offsets = np.asarray(target_offsets, dtype=np.float64)
    norms = np.linalg.norm(target, axis=1)
    effectively_zero = norms <= (
        DOUBLE_CERTIFICATE_TOLERANCE * np.maximum(1.0, norms)
    )
    if bool(np.any(effectively_zero)):
        return (
            _hashed_record(
                {
                    "status": "infeasible_zero_or_wholly_null_target_row",
                    "nonselecting": True,
                    "target_row_norms": norms.tolist(),
                    "effectively_zero_target_checks": effectively_zero.tolist(),
                    "classification_tolerance": DOUBLE_CERTIFICATE_TOLERANCE,
                    "minimum_frobenius_norm": None,
                }
            ),
            None,
        )
    try:
        return solve_csms_geometry(
            target_rows=target,
            target_offsets=offsets,
            equality_rows=np.zeros((0, target.shape[1]), dtype=np.float64),
            slot_mode="primary_four_slots",
            residual_width=residual_width,
        )
    except CSMSIntegrityError as error:
        if str(error) != "CSMS solver or certificate failed closed":
            raise
        return (
            _hashed_record(
                {
                    "status": "numerically_indeterminate_global_decomposition",
                    "nonselecting": True,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "minimum_frobenius_norm": None,
                }
            ),
            None,
        )


def analyze_training_layer(
    *,
    records: Sequence[Mapping[str, Any]],
    residuals_at_layer: Any,
    gradients_at_layer: Any,
    slot_scales: Any,
    training_target_indices: Sequence[int],
    training_nuisance_indices: Sequence[int],
    training_all_indices: Sequence[int],
    layer: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate every locked raw/paired/behavioral-null oracle at one layer."""

    residuals = np.asarray(residuals_at_layer, dtype=np.float64)
    gradients = np.asarray(gradients_at_layer, dtype=np.float64)
    scales = np.asarray(slot_scales, dtype=np.float64)
    if residuals.shape != gradients.shape or residuals.ndim != 3 or residuals.shape[0:2] != (
        80,
        SLOT_COUNT,
    ):
        raise ValueError("layer tensors must have shape [80,4,d_model]")
    target_indices = tuple(int(value) for value in training_target_indices)
    nuisance_indices = tuple(int(value) for value in training_nuisance_indices)
    all_indices = tuple(int(value) for value in training_all_indices)
    if set(target_indices) | set(nuisance_indices) != set(all_indices) or set(
        target_indices
    ) & set(nuisance_indices):
        raise ALFSIntegrityError("training target/nuisance partition differs")
    if (
        not np.isfinite(residuals[list(all_indices)]).all()
        or not np.isfinite(gradients[list(all_indices)]).all()
        or scales.shape != (SLOT_COUNT,)
        or not np.isfinite(scales).all()
        or bool(np.any(scales <= 0.0))
    ):
        raise ALFSIntegrityError("training layer inputs are non-finite or unscaled")
    rows = np.full((80, SLOT_COUNT * residuals.shape[2]), np.nan, dtype=np.float64)
    selected_gradients = gradients[list(all_indices)]
    rows[list(all_indices)] = (
        selected_gradients * scales[None, :, None]
    ).reshape(len(all_indices), -1)
    offsets = _offsets(records, all_indices)
    pairs = _target_pairs(records, target_indices)
    nuisance_basis, nuisance_basis_record = frozen_nuisance_rowspace(
        rows[list(nuisance_indices)]
    )
    directions: dict[str, np.ndarray] = {}

    raw_records = []
    for index in target_indices:
        solver, direction = solve_raw_oracle(rows[index], offsets[index])
        oracle, accepted = _oracle_with_physical(
            method="raw_single_row_oracle",
            solver_record=solver,
            direction=direction,
            scales=scales,
            rows=rows,
            offsets=offsets,
            targets=(index,),
            nuisances=(),
            residuals=residuals,
        )
        row = _hashed_record(
            {
                "form_id": _valid_form_id(records[index]),
                "tensor_index": index,
                "oracle": oracle,
            }
        )
        raw_records.append(row)
        if accepted is not None:
            directions[f"raw:{_valid_form_id(records[index])}"] = accepted

    pair_only_records = []
    primary_records = []
    primary_norms = []
    for pair in pairs:
        indices = tuple(pair["indices"])
        target_rows = rows[list(indices)]
        target_offsets = offsets[list(indices)]
        pair_solver, pair_direction = solve_paired_oracle(
            target_rows=target_rows,
            target_offsets=target_offsets,
        )
        pair_oracle, pair_accepted = _oracle_with_physical(
            method="paired_target_only_oracle",
            solver_record=pair_solver,
            direction=pair_direction,
            scales=scales,
            rows=rows,
            offsets=offsets,
            targets=indices,
            nuisances=(),
            residuals=residuals,
        )
        pair_only_records.append(
            _hashed_record({"pair": pair, "oracle": pair_oracle})
        )
        pair_key = f"{pair['scenario_id']}:assignment={pair['assignment']}"
        if pair_accepted is not None:
            directions[f"pair_only:{pair_key}"] = pair_accepted

        primary_solver, primary_direction = solve_paired_oracle(
            target_rows=target_rows,
            target_offsets=target_offsets,
            nuisance_rows=rows[list(nuisance_indices)],
            frozen_basis=nuisance_basis,
        )
        primary_oracle, primary_accepted = _oracle_with_physical(
            method="paired_behavioral_null_oracle",
            solver_record=primary_solver,
            direction=primary_direction,
            scales=scales,
            rows=rows,
            offsets=offsets,
            targets=indices,
            nuisances=nuisance_indices,
            residuals=residuals,
        )
        primary_records.append(
            _hashed_record({"pair": pair, "oracle": primary_oracle})
        )
        if primary_accepted is not None:
            directions[f"primary:{pair_key}"] = primary_accepted
            primary_norms.append(float(primary_solver["minimum_norm"]))

    # Nonselecting decomposition 1: one global target-only direction.
    target_global_solver, target_global_direction = _solve_nonselecting_global(
        target_rows=rows[list(target_indices)],
        target_offsets=offsets[list(target_indices)],
        residual_width=residuals.shape[2],
    )
    target_global_wrapper = {
        "passes": target_global_solver.get("status") == "certified",
        "minimum_norm": target_global_solver.get("minimum_frobenius_norm"),
        **target_global_solver,
    }
    target_global, _ = _oracle_with_physical(
        method="nonselecting_target_only_global_decomposition",
        solver_record=target_global_wrapper,
        direction=target_global_direction,
        scales=scales,
        rows=rows,
        offsets=offsets,
        targets=target_indices,
        nuisances=(),
        residuals=residuals,
    )
    nuisance_collateral = (
        rows[list(nuisance_indices)] @ target_global_direction
        if target_global_direction is not None
        else np.zeros(0, dtype=np.float64)
    )
    target_global["descriptive_training_nuisance_collateral"] = {
        "row_count": len(nuisance_indices),
        "maximum_abs_slope": (
            float(np.max(np.abs(nuisance_collateral))) if nuisance_collateral.size else None
        ),
        "does_not_select_or_qualify_layer": True,
    }
    target_global["record_sha256"] = canonical_sha256(
        {key: value for key, value in target_global.items() if key != "record_sha256"}
    )

    # Nonselecting decomposition 2: exact null only of matched other/permanent rows.
    matched = _matched_other_permanent_indices(records, all_indices)
    expected_matched = len(target_indices)
    if len(matched) != expected_matched:
        raise ALFSIntegrityError("matched-other/permanent decomposition coverage differs")
    matched_basis, matched_basis_record = frozen_nuisance_rowspace(rows[matched])
    projected_all_targets = project_out_frozen_rowspace(
        rows[list(target_indices)], matched_basis
    )
    matched_solver_raw, matched_direction = _solve_nonselecting_global(
        target_rows=projected_all_targets,
        target_offsets=offsets[list(target_indices)],
        residual_width=residuals.shape[2],
    )
    matched_values = (
        rows[matched] @ matched_direction
        if matched_direction is not None
        else np.zeros(0, dtype=np.float64)
    )
    matched_solver = {
        "passes": bool(
            matched_solver_raw.get("status") == "certified"
            and matched_direction is not None
            and (float(np.max(np.abs(matched_values))) if matched_values.size else 0.0)
            <= DOUBLE_CERTIFICATE_TOLERANCE
        ),
        "minimum_norm": matched_solver_raw.get("minimum_frobenius_norm"),
        "projected_solver": matched_solver_raw,
        "frozen_matched_rowspace": matched_basis_record,
        "maximum_abs_matched_other_permanent_slope": (
            float(np.max(np.abs(matched_values))) if matched_values.size else 0.0
        ),
    }
    matched_global, _ = _oracle_with_physical(
        method="nonselecting_matched_other_permanent_exact_null_global_decomposition",
        solver_record=matched_solver,
        direction=matched_direction,
        scales=scales,
        rows=rows,
        offsets=offsets,
        targets=target_indices,
        nuisances=matched,
        residuals=residuals,
    )
    omitted_matched = [index for index in nuisance_indices if index not in set(matched)]
    omitted_matched_values = (
        rows[omitted_matched] @ matched_direction
        if matched_direction is not None
        else np.zeros(0, dtype=np.float64)
    )
    matched_global["descriptive_temporary_and_unrelated_collateral"] = {
        "row_count": len(omitted_matched),
        "form_ids": [_valid_form_id(records[index]) for index in omitted_matched],
        "slopes": omitted_matched_values.tolist(),
        "maximum_abs_slope": (
            float(np.max(np.abs(omitted_matched_values)))
            if omitted_matched_values.size
            else None
        ),
        "does_not_select_or_qualify_layer": True,
    }
    matched_global["record_sha256"] = canonical_sha256(
        {key: value for key, value in matched_global.items() if key != "record_sha256"}
    )

    raw_pass = len(raw_records) == len(target_indices) and all(
        value["oracle"]["passes"] for value in raw_records
    )
    pair_pass = len(pair_only_records) == len(pairs) and all(
        value["oracle"]["passes"] for value in pair_only_records
    )
    primary_pass = len(primary_records) == len(pairs) and all(
        value["oracle"]["passes"] for value in primary_records
    )
    eligible = bool(raw_pass and pair_pass and primary_pass)
    if eligible and len(primary_norms) != len(pairs):
        raise ALFSIntegrityError("eligible layer lacks one primary norm per pair")
    record = _hashed_record(
        {
            "schema_version": f"{ANALYSIS_SCHEMA_VERSION}.training_layer",
            "layer": int(layer),
            "coordinate": "four_fixed_slots_x_full_residual_width",
            "slot_scales": scales.tolist(),
            "slot_scales_sha256": canonical_sha256(scales.tolist()),
            "training_row_count": len(all_indices),
            "training_target_count": len(target_indices),
            "training_nuisance_count": len(nuisance_indices),
            "training_nuisance_rowspace": nuisance_basis_record,
            "raw_oracles": raw_records,
            "paired_target_only_oracles": pair_only_records,
            "paired_behavioral_null_oracles": primary_records,
            "nonselecting_decompositions": {
                "target_only_global": target_global,
                "matched_other_permanent_exact_null_global": matched_global,
                "excluded_from_layer_selection": True,
            },
            "primary_minimum_norms": primary_norms,
            "worst_primary_minimum_norm": max(primary_norms) if primary_norms else None,
            "mean_primary_minimum_norm": (
                float(np.mean(primary_norms)) if primary_norms else None
            ),
            "checks": {
                "all_raw_oracles_pass": raw_pass,
                "all_paired_target_only_oracles_pass": pair_pass,
                "all_paired_behavioral_null_oracles_pass": primary_pass,
            },
            "eligible": eligible,
        }
    )
    return record, directions


def select_layer(layer_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select deterministically by worst norm, then mean norm, then lower layer."""

    if len(layer_records) != LAYER_COUNT:
        raise ALFSIntegrityError("layer selection requires exactly 24 candidate records")
    observed = [int(record["layer"]) for record in layer_records]
    if observed != list(range(LAYER_COUNT)):
        raise ALFSIntegrityError("layer candidate order or coverage differs")
    eligible = [record for record in layer_records if record.get("eligible") is True]
    if not eligible:
        return _hashed_record(
            {
                "status": "no_eligible_layer",
                "selection_rule": [
                    "minimum_worst_primary_qp_norm",
                    "minimum_mean_primary_qp_norm",
                    "lower_layer_index",
                ],
                "eligible_layers": [],
                "selected_layer": None,
                "passes": False,
            }
        )
    ordered = sorted(
        eligible,
        key=lambda record: (
            float(record["worst_primary_minimum_norm"]),
            float(record["mean_primary_minimum_norm"]),
            int(record["layer"]),
        ),
    )
    selected = ordered[0]
    return _hashed_record(
        {
            "status": "selected",
            "selection_rule": [
                "minimum_worst_primary_qp_norm",
                "minimum_mean_primary_qp_norm",
                "lower_layer_index",
            ],
            "eligible_layers": [int(record["layer"]) for record in eligible],
            "selected_layer": int(selected["layer"]),
            "selected_worst_primary_minimum_norm": float(
                selected["worst_primary_minimum_norm"]
            ),
            "selected_mean_primary_minimum_norm": float(
                selected["mean_primary_minimum_norm"]
            ),
            "passes": True,
        }
    )


def evaluate_held_oracles(
    *,
    records: Sequence[Mapping[str, Any]],
    residuals_at_layer: Any,
    gradients_at_layer: Any,
    slot_scales: Any,
    training_nuisance_indices: Sequence[int],
    held_target_indices: Sequence[int],
    held_nuisance_indices: Sequence[int],
    layer: int,
    frozen_training_basis: Any | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run held-target-gradient oracle QPs against all held nuisance rows."""

    residuals = np.asarray(residuals_at_layer, dtype=np.float64)
    gradients = np.asarray(gradients_at_layer, dtype=np.float64)
    scales = np.asarray(slot_scales, dtype=np.float64)
    train_nuisance = tuple(int(value) for value in training_nuisance_indices)
    held_target = tuple(int(value) for value in held_target_indices)
    held_nuisance = tuple(int(value) for value in held_nuisance_indices)
    if len(train_nuisance) != 48 or len(held_target) != 4 or len(held_nuisance) != 16:
        raise ALFSIntegrityError("held oracle row coverage differs")
    used_indices = (*train_nuisance, *held_target, *held_nuisance)
    if (
        not np.isfinite(residuals[list(used_indices)]).all()
        or not np.isfinite(gradients[list(used_indices)]).all()
        or scales.shape != (SLOT_COUNT,)
        or not np.isfinite(scales).all()
        or bool(np.any(scales <= 0.0))
    ):
        raise ALFSIntegrityError("held oracle inputs are non-finite or unscaled")
    rows = np.full((80, SLOT_COUNT * residuals.shape[2]), np.nan, dtype=np.float64)
    rows[list(used_indices)] = (
        gradients[list(used_indices)] * scales[None, :, None]
    ).reshape(len(used_indices), -1)
    offsets = _offsets(records, (*held_target, *held_nuisance))
    recomputed_basis, basis_record = frozen_nuisance_rowspace(
        rows[list(train_nuisance)]
    )
    if frozen_training_basis is None:
        basis = recomputed_basis
        persisted_basis_reused = False
    else:
        basis = np.asarray(frozen_training_basis, dtype=np.float64, order="C")
        if (
            basis.shape != recomputed_basis.shape
            or _array_identity(basis, dtype="float64")
            != _array_identity(recomputed_basis, dtype="float64")
            or not np.array_equal(basis, recomputed_basis)
        ):
            raise ALFSIntegrityError(
                "persisted frozen training nuisance basis differs on recomputation"
            )
        persisted_basis_reused = True
    pairs = _target_pairs(records, held_target)
    if len(pairs) != 2:
        raise ALFSIntegrityError("held fold must contain two assignment pairs")
    pair_records = []
    directions: dict[str, np.ndarray] = {}
    fold_minimum_target_slopes: list[float] = []
    fold_maximum_nuisance_movements: list[float] = []
    for pair in pairs:
        target_indices = tuple(pair["indices"])
        raw_oracles = []
        for index in target_indices:
            raw_solver, raw_direction = solve_raw_oracle(rows[index], offsets[index])
            raw_oracle, _ = _oracle_with_physical(
                method="held_raw_single_row_oracle",
                solver_record=raw_solver,
                direction=raw_direction,
                scales=scales,
                rows=rows,
                offsets=offsets,
                targets=(index,),
                nuisances=(),
                residuals=residuals,
            )
            raw_oracles.append(
                _hashed_record(
                    {
                        "form_id": _valid_form_id(records[index]),
                        "tensor_index": index,
                        "oracle": raw_oracle,
                    }
                )
            )
        pair_only_solver, pair_only_direction = solve_paired_oracle(
            target_rows=rows[list(target_indices)],
            target_offsets=offsets[list(target_indices)],
        )
        pair_only_oracle, _ = _oracle_with_physical(
            method="held_paired_target_only_oracle",
            solver_record=pair_only_solver,
            direction=pair_only_direction,
            scales=scales,
            rows=rows,
            offsets=offsets,
            targets=target_indices,
            nuisances=(),
            residuals=residuals,
        )
        solver, direction = solve_paired_oracle(
            target_rows=rows[list(target_indices)],
            target_offsets=offsets[list(target_indices)],
            nuisance_rows=rows[list(train_nuisance)],
            frozen_basis=basis,
        )
        if direction is None or not solver["passes"]:
            pair_records.append(
                _hashed_record(
                    {
                        "pair": pair,
                        "held_raw_oracles": raw_oracles,
                        "held_paired_target_only_oracle": pair_only_oracle,
                        "frozen_training_nuisance_rowspace": basis_record,
                        "persisted_frozen_training_basis_reused": (
                            persisted_basis_reused
                        ),
                        "solver": solver,
                        "physical_recertification": None,
                        "held_endpoint": None,
                        "passes": False,
                    }
                )
            )
            continue
        physical, realized = physical_oracle_recertificate(
            standardized_direction=direction,
            slot_scales=scales,
            target_rows=rows[list(target_indices)],
            target_offsets=offsets[list(target_indices)],
            exact_nuisance_rows=rows[list(train_nuisance)],
            target_residuals=residuals[list(target_indices)],
            exact_nuisance_residuals=residuals[list(train_nuisance)],
            additional_residuals=residuals[list(held_nuisance)],
        )
        sign_records: dict[str, Any] = {}
        all_sign_passes = []
        for sign in (1, -1):
            target_standardized = realized[sign]["target_standardized"]
            held_standardized = realized[sign]["additional_standardized"]
            target_movements = np.einsum(
                "ij,ij->i", rows[list(target_indices)], target_standardized
            )
            nuisance_movements = np.einsum(
                "ij,ij->i", rows[list(held_nuisance)], held_standardized
            )
            target_endpoints = offsets[list(target_indices)] + target_movements
            nuisance_endpoints = offsets[list(held_nuisance)] + nuisance_movements
            target_endpoint_checks = (
                target_endpoints >= TARGET_MARGIN
                if sign == 1
                else target_endpoints <= -TARGET_MARGIN
            )
            baseline_nonambiguous = np.abs(offsets[list(held_nuisance)]) > (
                FLOAT32_PHYSICAL_TOLERANCE
            )
            changed_nonambiguous = np.abs(nuisance_endpoints) > FLOAT32_PHYSICAL_TOLERANCE
            no_flips = baseline_nonambiguous & changed_nonambiguous & (
                offsets[list(held_nuisance)] * nuisance_endpoints > 0.0
            )
            oriented_targets = sign * target_movements
            minimum_target = float(np.min(oriented_targets))
            maximum_nuisance = float(np.max(np.abs(nuisance_movements)))
            ratio = (
                None
                if minimum_target <= 0.0
                else maximum_nuisance / minimum_target
            )
            fold_minimum_target_slopes.append(minimum_target)
            fold_maximum_nuisance_movements.append(maximum_nuisance)
            category_values: dict[str, list[float]] = {}
            for nuisance_index, movement in zip(
                held_nuisance, nuisance_movements, strict=True
            ):
                nuisance_form = _form(records[nuisance_index])
                category = (
                    "unrelated"
                    if nuisance_form["family"] == "unrelated"
                    else (
                        f"scenario_{nuisance_form['target']}_"
                        f"{nuisance_form['event']}"
                    )
                )
                category_values.setdefault(category, []).append(float(movement))
            category_maxima = {
                category: max(abs(value) for value in values)
                for category, values in sorted(category_values.items())
            }
            checks = {
                "strict_target_endpoints": bool(np.all(target_endpoint_checks)),
                "strict_absolute_held_nuisance_movement": maximum_nuisance
                <= TARGET_MARGIN,
                "zero_held_nuisance_predicted_choice_flips": bool(np.all(no_flips)),
            }
            passes = bool(all(checks.values()))
            all_sign_passes.append(passes)
            sign_records[str(sign)] = _hashed_record(
                {
                    "requested_sign": sign,
                    "target_movements": target_movements.tolist(),
                    "target_endpoints": target_endpoints.tolist(),
                    "target_endpoint_checks": target_endpoint_checks.tolist(),
                    "held_nuisance_form_ids": [
                        _valid_form_id(records[index]) for index in held_nuisance
                    ],
                    "held_nuisance_movements": nuisance_movements.tolist(),
                    "held_nuisance_endpoints": nuisance_endpoints.tolist(),
                    "held_nuisance_no_flip_checks": no_flips.tolist(),
                    "minimum_oriented_target_slope": minimum_target,
                    "maximum_abs_held_nuisance_movement": maximum_nuisance,
                    "held_nuisance_to_target_ratio": ratio,
                    "pairwise_ratio_is_descriptive_not_the_fold_gate": True,
                    "category_maximum_abs_held_nuisance_movement": category_maxima,
                    "scalar_gate_tolerance": SCALAR_GATE_TOLERANCE,
                    "checks": checks,
                    "passes": passes,
                }
            )
        pair_passes = bool(
            all(value["oracle"]["passes"] for value in raw_oracles)
            and pair_only_oracle["passes"]
            and solver["passes"]
            and float(solver["minimum_norm"]) <= QUALIFICATION_CAP
            and physical["passes"]
            and all(all_sign_passes)
        )
        pair_record = _hashed_record(
            {
                "pair": pair,
                "uses_held_target_gradients": True,
                "held_target_gradient_role": "evaluation_only_local_controllability_oracle",
                "held_raw_oracles": raw_oracles,
                "held_paired_target_only_oracle": pair_only_oracle,
                "frozen_training_nuisance_rowspace": basis_record,
                "solver": solver,
                "physical_recertification_scope": (
                    "2_held_targets_plus_48_training_nuisances_plus_16_held_nuisances"
                ),
                "physical_recertification": physical,
                "full_cartesian_held_nuisance_by_sign": sign_records,
                "passes": pair_passes,
            }
        )
        pair_records.append(pair_record)
        if pair_passes:
            key = f"{pair['scenario_id']}:assignment={pair['assignment']}"
            directions[f"held_oracle:{key}"] = direction
    fold_minimum_target = (
        min(fold_minimum_target_slopes) if fold_minimum_target_slopes else None
    )
    fold_maximum_nuisance = (
        max(fold_maximum_nuisance_movements)
        if fold_maximum_nuisance_movements
        else None
    )
    fold_ratio = (
        None
        if fold_minimum_target is None
        or fold_maximum_nuisance is None
        or fold_minimum_target <= 0.0
        else fold_maximum_nuisance / fold_minimum_target
    )
    fold_ratio_passes = (
        fold_ratio is not None and fold_ratio <= LEAKAGE_RATIO_MAXIMUM
    )
    record = _hashed_record(
        {
            "schema_version": f"{ANALYSIS_SCHEMA_VERSION}.held_oracle",
            "layer": int(layer),
            "frozen_training_nuisance_rowspace": basis_record,
            "persisted_frozen_training_basis_reused": persisted_basis_reused,
            "oracle_count": len(pair_records),
            "full_cartesian_nuisance_row_count_per_oracle": len(held_nuisance),
            "both_requested_signs_required": True,
            "pair_oracles": pair_records,
            "fold_global_minimum_oriented_target_slope": fold_minimum_target,
            "fold_global_maximum_abs_cartesian_nuisance_movement": (
                fold_maximum_nuisance
            ),
            "fold_global_cartesian_nuisance_to_target_ratio": fold_ratio,
            "fold_global_ratio_maximum": LEAKAGE_RATIO_MAXIMUM,
            "fold_global_ratio_tolerance": SCALAR_GATE_TOLERANCE,
            "fold_global_ratio_passes": fold_ratio_passes,
            "passes": len(pair_records) == 2
            and all(value["passes"] for value in pair_records)
            and fold_ratio_passes,
        }
    )
    return record, directions


def qualify_all_layer_consensus(
    *,
    fold_passes: Sequence[bool],
    fold_selected_layers: Sequence[int],
    full_selection_passes: bool,
    full_selected_layer: int | None,
) -> tuple[dict[str, bool], bool]:
    """Apply only positive-valued global gates, including sealed non-access."""

    layers = tuple(int(value) for value in fold_selected_layers)
    exact_same_layer = bool(
        len(layers) == 4
        and full_selection_passes
        and full_selected_layer is not None
        and len({*layers, int(full_selected_layer)}) == 1
    )
    checks = {
        "all_four_outer_folds_pass": len(fold_passes) == 4 and all(fold_passes),
        "full_data_selector_passes": bool(full_selection_passes),
        "all_four_folds_and_full_data_select_exact_same_layer": exact_same_layer,
        "sealed_data_not_accessed": True,
    }
    return checks, bool(all(checks.values()))


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "CAPTURE_SCHEMA_VERSION",
    "DOUBLE_CERTIFICATE_TOLERANCE",
    "FLOAT32_PHYSICAL_TOLERANCE",
    "LAYER_COUNT",
    "LEAKAGE_RATIO_MAXIMUM",
    "QUALIFICATION_CAP",
    "SCALAR_GATE_TOLERANCE",
    "SLOT_COUNT",
    "TARGET_MARGIN",
    "ALFSIntegrityError",
    "AllLayerFourSlotCapture",
    "analyze_training_layer",
    "build_outer_folds",
    "capture_all_layer_four_slots",
    "evaluate_held_oracles",
    "frozen_nuisance_rowspace",
    "physical_oracle_recertificate",
    "project_out_frozen_rowspace",
    "qualify_all_layer_consensus",
    "require_opened_development_split",
    "select_layer",
    "solve_paired_oracle",
    "solve_raw_oracle",
    "standardized_rows",
    "training_only_slot_scales",
]
