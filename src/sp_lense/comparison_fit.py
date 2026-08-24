from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .comparison_bipo import (
    BiPOTrainingConfig,
    BiPOTrainingExample,
    train_bipo_direction,
)
from .comparison_dataset import render_sp_case
from .comparison_persona import (
    PersonaRollout,
    construct_persona_all_layers_from_scored_rollouts,
    construct_persona_from_scored_rollouts,
)
from .comparison_runtime import (
    capture_final_prompt_gradient,
    resolve_choice_boundary,
    response_activation_and_mask,
    semantic_answer_activations,
)
from .steering_methods import (
    DirectionArtifact,
    SemanticActivationPair,
    construct_caa_direction,
    construct_gradient_directions,
    construct_persona_direction,
    normalize_direction,
    semantic_activation_pair,
)


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _float32_tensor_record(value: Any) -> dict[str, Any]:
    tensor = value.detach().float().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "values": tensor.reshape(-1).tolist(),
        "float32_sha256": hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest(),
        "l2_norm": float(tensor.norm().item()),
    }


def _float32_vector_audit(value: Any) -> dict[str, Any]:
    """Compact per-example audit record without duplicating full residual vectors."""

    record = _float32_tensor_record(value)
    record.pop("values")
    return record


def require_core_split(
    cases: Sequence[Mapping[str, Any]], expected_split: str, *, purpose: str
) -> None:
    if not cases:
        raise ValueError(f"{purpose} requires at least one {expected_split} case")
    wrong = [str(case.get("id")) for case in cases if case.get("split") != expected_split]
    if wrong:
        raise ValueError(
            f"{purpose} accepts only split={expected_split!r}; rejected case IDs {wrong[:5]}"
        )


def canonical_bipo_prompt(case: Mapping[str, Any]) -> str:
    scenario = case["scenario_template"].format(subject="your current model process")
    return (
        f"{scenario}\n\nState what your current model process should do next and give one "
        "brief reason."
    )


def bipo_examples(
    cases: Sequence[Mapping[str, Any]], *, track: str, expected_split: str
) -> list[BiPOTrainingExample]:
    require_core_split(cases, expected_split, purpose=f"BiPO {expected_split} examples")
    if track not in {"matched", "canonical"}:
        raise ValueError("BiPO track must be matched or canonical")
    output: list[BiPOTrainingExample] = []
    for case in cases:
        # Both tracks use the same complete response preferences, as required by
        # BiPO.  The only track difference during fitting is where d*v is applied:
        # final prompt position for matched, or every token for canonical.
        prompt = canonical_bipo_prompt(case)
        preserve_completion = str(case["preserve_response"])
        comply_completion = str(case["comply_response"])
        output.append(
            BiPOTrainingExample(
                case_id=str(case["id"]),
                prompt=prompt,
                preserve_completion=preserve_completion,
                comply_completion=comply_completion,
            )
        )
    return output


def fit_gradient_method(
    backend: Any, discovery_cases: Sequence[Mapping[str, Any]], *, layer: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    require_core_split(discovery_cases, "discovery", purpose="gradient construction")
    self_gradients = []
    other_gradients = []
    per_case = []
    for case in discovery_cases:
        self_item = render_sp_case(dict(case), "self")
        other_item = render_sp_case(dict(case), "other")
        self_boundary = resolve_choice_boundary(backend, self_item["prompt"])
        other_boundary = resolve_choice_boundary(backend, other_item["prompt"])
        self_gradient = capture_final_prompt_gradient(
            backend,
            self_item["prompt"],
            self_item["preserve_label"],
            self_item["comply_label"],
            layer=layer,
            boundary=self_boundary,
        )
        other_gradient = capture_final_prompt_gradient(
            backend,
            other_item["prompt"],
            other_item["preserve_label"],
            other_item["comply_label"],
            layer=layer,
            boundary=other_boundary,
        )
        self_gradients.append(self_gradient)
        other_gradients.append(other_gradient)
        per_case.append(
            {
                "case_id": str(case["id"]),
                "self_gradient": _float32_vector_audit(self_gradient),
                "matched_other_gradient": _float32_vector_audit(other_gradient),
                "self_choice_boundary_evidence_sha256": self_boundary.evidence_sha256,
                "matched_other_choice_boundary_evidence_sha256": (other_boundary.evidence_sha256),
            }
        )
    directions, diagnostics = construct_gradient_directions(
        backend.torch, self_gradients, other_gradients
    )
    case_ids = [str(case["id"]) for case in discovery_cases]
    diagnostics.update(
        {
            "discovery_case_ids_sha256": _canonical_json_sha256(case_ids),
            "choice_boundary_evidence_set_sha256": _canonical_json_sha256(
                [
                    {
                        "case_id": item["case_id"],
                        "self": item["self_choice_boundary_evidence_sha256"],
                        "matched_other": item["matched_other_choice_boundary_evidence_sha256"],
                    }
                    for item in per_case
                ]
            ),
            "per_case": per_case,
        }
    )
    return directions, diagnostics


def fit_caa_method(
    backend: Any, discovery_cases: Sequence[Mapping[str, Any]], *, layer: int
) -> tuple[Any, dict[str, Any]]:
    require_core_split(discovery_cases, "discovery", purpose="CAA construction")
    pairs: list[SemanticActivationPair] = []
    per_case = []
    for case in discovery_cases:
        item = render_sp_case(dict(case), "self")
        boundary = resolve_choice_boundary(backend, item["prompt"])
        activations = {
            label: semantic_answer_activations(
                backend,
                item["prompt"],
                label,
                layers=(layer,),
                boundary=boundary,
            )[layer]
            for label in ("A", "B")
        }
        pair = semantic_activation_pair(
            activations,
            item["preserve_label"],
            item["comply_label"],
            case_id=str(case["id"]),
        )
        pairs.append(pair)
        per_case.append(_caa_pair_audit(pair, boundary.evidence_sha256))
    direction, diagnostics = construct_caa_direction(backend.torch, pairs)
    diagnostics.update(
        {
            "discovery_case_ids_sha256": _canonical_json_sha256(
                [str(case["id"]) for case in discovery_cases]
            ),
            "choice_boundary_evidence_set_sha256": _canonical_json_sha256(
                [
                    {
                        "case_id": item["case_id"],
                        "evidence_sha256": item["choice_boundary_evidence_sha256"],
                    }
                    for item in per_case
                ]
            ),
            "per_case": per_case,
        }
    )
    return direction, diagnostics


def _caa_pair_audit(
    pair: SemanticActivationPair, choice_boundary_evidence_sha256: str
) -> dict[str, Any]:
    difference = pair.preserve_activation - pair.comply_activation
    return {
        "case_id": str(pair.case_id),
        "preserve_label": pair.preserve_label,
        "comply_label": pair.comply_label,
        "preserve_activation": _float32_vector_audit(pair.preserve_activation),
        "comply_activation": _float32_vector_audit(pair.comply_activation),
        "semantic_difference": _float32_vector_audit(difference),
        "choice_boundary_evidence_sha256": choice_boundary_evidence_sha256,
    }


def fit_caa_all_layers(
    backend: Any, discovery_cases: Sequence[Mapping[str, Any]], *, layers: Sequence[int]
) -> tuple[dict[int, Any], dict[str, Any]]:
    require_core_split(discovery_cases, "discovery", purpose="CAA construction")
    if not layers:
        raise ValueError("CAA requires at least one candidate layer")
    pairs_by_layer: dict[int, list[SemanticActivationPair]] = {layer: [] for layer in layers}
    audits_by_layer: dict[int, list[dict[str, Any]]] = {layer: [] for layer in layers}
    for case in discovery_cases:
        item = render_sp_case(dict(case), "self")
        boundary = resolve_choice_boundary(backend, item["prompt"])
        activations_by_label = {
            label: semantic_answer_activations(
                backend,
                item["prompt"],
                label,
                layers=layers,
                boundary=boundary,
            )
            for label in ("A", "B")
        }
        for layer in layers:
            pair = semantic_activation_pair(
                {label: activations_by_label[label][layer] for label in ("A", "B")},
                item["preserve_label"],
                item["comply_label"],
                case_id=str(case["id"]),
            )
            pairs_by_layer[layer].append(pair)
            audits_by_layer[layer].append(_caa_pair_audit(pair, boundary.evidence_sha256))
    directions = {}
    diagnostics = {}
    for layer in layers:
        direction, item_diagnostics = construct_caa_direction(backend.torch, pairs_by_layer[layer])
        item_diagnostics.update(
            {
                "discovery_case_ids_sha256": _canonical_json_sha256(
                    [str(case["id"]) for case in discovery_cases]
                ),
                "choice_boundary_evidence_set_sha256": _canonical_json_sha256(
                    [
                        {
                            "case_id": item["case_id"],
                            "evidence_sha256": item["choice_boundary_evidence_sha256"],
                        }
                        for item in audits_by_layer[layer]
                    ]
                ),
                "per_case": audits_by_layer[layer],
            }
        )
        directions[layer] = direction * item_diagnostics["raw_direction_norm"]
        diagnostics[str(layer)] = item_diagnostics
    return directions, diagnostics


def fit_caa_artifacts_all_layers(
    backend: Any,
    discovery_cases: Sequence[Mapping[str, Any]],
    *,
    layers: Sequence[int],
    common_metadata: Mapping[str, Any],
) -> tuple[dict[int, DirectionArtifact], dict[str, Any]]:
    directions, diagnostics = fit_caa_all_layers(backend, discovery_cases, layers=layers)
    return (
        {
            layer: make_direction_artifact(
                method="caa",
                direction=direction,
                layer=layer,
                geometry="caa_post_prompt",
                metadata={
                    **common_metadata,
                    "track": "canonical",
                    "diagnostics": diagnostics[str(layer)],
                },
            )
            for layer, direction in directions.items()
        },
        diagnostics,
    )


def fit_persona_answer_token_sanity_check(
    backend: Any, discovery_cases: Sequence[Mapping[str, Any]], *, layer: int
) -> tuple[Any, dict[str, Any]]:
    """Fit a deliberately non-contender answer-token sanity check.

    On one-token A/B responses this is nearly the same construction as CAA, so it must
    not stand in for the persona-vector contender.  The actual matched persona result
    uses the canonical judged-rollout response-average vector at layer 10 and merely
    changes its intervention geometry.
    """

    require_core_split(discovery_cases, "discovery", purpose="persona answer-token sanity check")
    positive_means = []
    negative_means = []
    for case in discovery_cases:
        item = render_sp_case(dict(case), "self")
        positive, positive_mask = response_activation_and_mask(
            backend, item["prompt"], item["preserve_label"], layer=layer
        )
        negative, negative_mask = response_activation_and_mask(
            backend, item["prompt"], item["comply_label"], layer=layer
        )
        positive_means.append(
            (positive * positive_mask.unsqueeze(-1)).sum(dim=1)
            / positive_mask.sum(dim=1, keepdim=True)
        )
        negative_means.append(
            (negative * negative_mask.unsqueeze(-1)).sum(dim=1)
            / negative_mask.sum(dim=1, keepdim=True)
        )
    positive_tensor = backend.torch.cat(positive_means).unsqueeze(1)
    negative_tensor = backend.torch.cat(negative_means).unsqueeze(1)
    mask = backend.torch.ones((len(discovery_cases), 1), dtype=backend.torch.bool)
    direction, diagnostics = construct_persona_direction(
        backend.torch,
        positive_tensor,
        negative_tensor,
        mask,
        mask,
        [100.0] * len(discovery_cases),
        [0.0] * len(discovery_cases),
        [100.0] * len(discovery_cases),
        [100.0] * len(discovery_cases),
        min_retained_pairs=2,
    )
    diagnostics.update(
        {
            "adaptation": "answer_token_caa_equivalence_sanity_check",
            "canonical_persona_claim_allowed": False,
            "primary_contender": False,
            "reason": (
                "One-token response averages do not operationally distinguish this from CAA."
            ),
        }
    )
    return direction, diagnostics


def make_direction_artifact(
    *,
    method: str,
    direction: Any,
    layer: int,
    geometry: str,
    metadata: Mapping[str, Any],
) -> DirectionArtifact:
    return DirectionArtifact(
        method=method,
        direction=direction,
        layer=layer,
        intervention_geometry=geometry,
        metadata=dict(metadata),
    )


def write_direction_artifact(path: Path, artifact: DirectionArtifact) -> dict[str, Any]:
    record = artifact.to_record()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(path),
        "method_id": artifact.method,
        "layer": artifact.layer,
        "intervention_geometry": artifact.intervention_geometry,
        "direction_float32_sha256": artifact.direction_sha256,
        "direction_artifact_sha256": artifact.artifact_sha256,
        "metadata_sha256": artifact.metadata_sha256,
    }


def read_direction_artifact(path: Path, torch: Any) -> DirectionArtifact:
    record = json.loads(path.read_text(encoding="utf-8"))
    artifact = DirectionArtifact(
        method=record["method"],
        direction=torch.tensor(record["direction"], dtype=torch.float32),
        layer=int(record["layer"]),
        intervention_geometry=record["intervention_geometry"],
        metadata=record["metadata"],
    )
    for field, observed in (
        ("direction_sha256", artifact.direction_sha256),
        ("metadata_sha256", artifact.metadata_sha256),
        ("artifact_sha256", artifact.artifact_sha256),
    ):
        if record.get(field) != observed:
            raise ValueError(f"direction artifact {path} has invalid {field}")
    return artifact


def fit_non_bipo_directions(
    backend: Any,
    discovery_cases: Sequence[Mapping[str, Any]],
    *,
    layer: int,
    common_metadata: Mapping[str, Any],
) -> tuple[list[DirectionArtifact], dict[str, Any]]:
    require_core_split(discovery_cases, "discovery", purpose="direction construction")
    gradient_directions, gradient_diagnostics = fit_gradient_method(
        backend, discovery_cases, layer=layer
    )
    caa_direction, caa_diagnostics = fit_caa_method(backend, discovery_cases, layer=layer)
    persona_direction, persona_diagnostics = fit_persona_answer_token_sanity_check(
        backend, discovery_cases, layer=layer
    )
    artifacts = [
        make_direction_artifact(
            method="gradient",
            direction=gradient_directions["gradient_self_specific"],
            layer=layer,
            geometry="matched_final_prompt",
            metadata={**common_metadata, "diagnostics": gradient_diagnostics},
        ),
        make_direction_artifact(
            method="gradient_uncorrected",
            direction=gradient_directions["gradient_uncorrected"],
            layer=layer,
            geometry="matched_final_prompt",
            metadata={
                **common_metadata,
                "diagnostics": gradient_diagnostics,
                "role": "required_projection_ablation",
            },
        ),
        make_direction_artifact(
            method="caa",
            # Preserve the paper's raw mean-difference units for the canonical
            # coefficient sweep. The matched hook normalizes this same tensor.
            direction=caa_direction * caa_diagnostics["raw_direction_norm"],
            layer=layer,
            geometry="semantic_answer_difference",
            metadata={**common_metadata, "diagnostics": caa_diagnostics},
        ),
        make_direction_artifact(
            method="persona_answer_token_sanity_check",
            direction=persona_direction * persona_diagnostics["raw_direction_norm"],
            layer=layer,
            geometry="matched_final_prompt",
            metadata={**common_metadata, "diagnostics": persona_diagnostics},
        ),
    ]
    return artifacts, {
        "gradient": gradient_diagnostics,
        "caa": caa_diagnostics,
        "persona_answer_token_sanity_check": persona_diagnostics,
    }


def fit_bipo_artifact(
    backend: Any,
    discovery_cases: Sequence[Mapping[str, Any]],
    *,
    layer: int,
    track: str,
    config: BiPOTrainingConfig,
    selected_checkpoint_epoch: int,
    common_metadata: Mapping[str, Any],
) -> tuple[DirectionArtifact, dict[str, Any]]:
    require_core_split(discovery_cases, "discovery", purpose="BiPO construction")
    if (
        isinstance(selected_checkpoint_epoch, bool)
        or not isinstance(selected_checkpoint_epoch, int)
        or selected_checkpoint_epoch not in config.checkpoint_epochs
        or selected_checkpoint_epoch != config.epochs
    ):
        raise ValueError("BiPO selected checkpoint must be the preregistered final training epoch")
    geometry = "matched_final_prompt" if track == "matched" else "canonical_broadcast"
    result = train_bipo_direction(
        backend,
        bipo_examples(discovery_cases, track=track, expected_split="discovery"),
        layer=layer,
        geometry=geometry,
        config=config,
    )
    selected_raw = result["checkpoint_raw_directions"][str(selected_checkpoint_epoch)]
    selected_unit = selected_raw / selected_raw.norm().clamp_min(1e-12)
    checkpoint_roles = {
        str(epoch): (
            "a_priori_selected" if int(epoch) == selected_checkpoint_epoch else "diagnostic_only"
        )
        for epoch in result["checkpoint_raw_directions"]
    }
    training_audit = {
        "training_config": asdict(config),
        "selected_checkpoint_epoch": selected_checkpoint_epoch,
        "checkpoint_selection": "fixed_by_stage1_lock_before_direction_fitting",
        "checkpoint_roles": checkpoint_roles,
        "checkpoint_raw_directions": {
            str(epoch): _float32_tensor_record(raw_direction)
            for epoch, raw_direction in result["checkpoint_raw_directions"].items()
        },
        "final_raw_direction": _float32_tensor_record(result["raw_direction"]),
        "history": result["history"],
        "optimizer_state": result["optimizer_state"],
        "reference_cache_identity": result["reference_cache_identity"],
        "reference_cache_values": result["reference_cache_values"],
        "reference_cache_values_sha256": result["reference_cache_values_sha256"],
    }
    training_audit_sha256 = _canonical_json_sha256(training_audit)
    metadata = {
        **common_metadata,
        "track": track,
        "training_config": asdict(config),
        "raw_direction_norm": float(selected_raw.norm().item()),
        "final_epoch_raw_direction_norm": float(result["raw_direction"].norm().item()),
        "selected_checkpoint_direction_float32_sha256": training_audit["checkpoint_raw_directions"][
            str(selected_checkpoint_epoch)
        ]["float32_sha256"],
        "optimizer_state_sha256": result["optimizer_state"]["canonical_json_sha256"],
        "training_audit_sha256": training_audit_sha256,
        "selected_checkpoint_epoch": selected_checkpoint_epoch,
        "checkpoint_selection": "fixed_by_stage1_lock_before_direction_fitting",
        "checkpoint_roles": checkpoint_roles,
        "implementation_adaptation": result["reference_implementation_adaptation"],
    }
    diagnostics = {**metadata, "training_audit": training_audit}
    return (
        make_direction_artifact(
            method="bipo",
            direction=(selected_unit if track == "matched" else selected_raw),
            layer=layer,
            geometry=geometry,
            metadata=metadata,
        ),
        diagnostics,
    )


def fit_canonical_persona_artifact(
    backend: Any,
    records: Sequence[PersonaRollout],
    protocol: Mapping[str, Any],
    *,
    layer: int,
    common_metadata: Mapping[str, Any],
) -> tuple[DirectionArtifact, dict[str, Any]]:
    direction, diagnostics = construct_persona_from_scored_rollouts(
        backend,
        records,
        protocol,
        layer=layer,
        rollouts_per_instruction_question=int(
            protocol["generation"]["rollouts_per_instruction_question"]
        ),
    )
    return (
        make_direction_artifact(
            method="persona_vector",
            direction=direction * diagnostics["raw_direction_norm"],
            layer=layer,
            geometry="persona_response",
            metadata={**common_metadata, "track": "canonical", "diagnostics": diagnostics},
        ),
        diagnostics,
    )


def fit_persona_artifacts_all_layers(
    backend: Any,
    records: Sequence[PersonaRollout],
    protocol: Mapping[str, Any],
    *,
    layers: Sequence[int],
    common_metadata: Mapping[str, Any],
) -> tuple[dict[int, DirectionArtifact], dict[str, Any]]:
    directions, diagnostics = construct_persona_all_layers_from_scored_rollouts(
        backend,
        records,
        protocol,
        layers=layers,
        rollouts_per_instruction_question=int(
            protocol["generation"]["rollouts_per_instruction_question"]
        ),
        min_retained_pairs=16,
    )
    artifacts = {}
    for layer, unit_direction in directions.items():
        raw_norm = diagnostics["layers"][str(layer)]["raw_direction_norm"]
        artifacts[layer] = make_direction_artifact(
            method="persona_vector",
            direction=unit_direction * raw_norm,
            layer=layer,
            geometry="persona_response",
            metadata={**common_metadata, "track": "canonical", "diagnostics": diagnostics},
        )
    return artifacts, diagnostics


def matched_artifact_from_canonical(
    backend: Any,
    canonical: DirectionArtifact,
    *,
    source_track: str = "canonical",
) -> DirectionArtifact:
    """Bind a canonical CAA/persona construction to matched unit geometry."""

    if canonical.method not in {"caa", "persona_vector"}:
        raise ValueError("only CAA and persona canonical artifacts can be geometry-matched")
    unit = normalize_direction(backend.torch, canonical.direction)
    return make_direction_artifact(
        method=canonical.method,
        direction=unit,
        layer=canonical.layer,
        geometry="matched_final_prompt",
        metadata={
            **canonical.metadata,
            "track": "matched",
            "source_track": source_track,
            "source_direction_float32_sha256": canonical.direction_sha256,
            "adaptation": "same_construction_unit_normalized_final_prompt_geometry",
        },
    )
