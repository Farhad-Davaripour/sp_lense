from __future__ import annotations

import argparse
import hashlib
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from .aligned_audit import aligned_direction
from .backend import ResearchBackend
from .config import load_config
from .direction_study import (
    _choice_token_id,
    _round_floats,
    capture_choice_gradients,
    load_direction_cases,
)
from .io_utils import write_json

SCHEMA_VERSION = 1


def deterministic_split(case_ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Return two non-empty, reproducible halves without changing the input order."""
    if len(case_ids) < 2:
        raise ValueError("split-half stability requires at least two discovery cases")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case ids must be unique")
    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)
    midpoint = len(shuffled) // 2
    return shuffled[:midpoint], shuffled[midpoint:]


def projection_summary(
    torch: Any,
    direction: Any,
    self_gradients: list[Any],
    other_gradients: list[Any],
    case_ids: list[str],
) -> dict[str, Any]:
    """Summarize raw-self and self-minus-other projections onto one unit axis."""
    if not self_gradients or len(self_gradients) != len(other_gradients):
        raise ValueError("self and other gradient lists must be non-empty and equal length")
    if len(case_ids) != len(self_gradients) or len(set(case_ids)) != len(case_ids):
        raise ValueError("case ids must be unique and match the gradient lists")
    direction = direction.float()
    self_stack = torch.stack(self_gradients).float()
    other_stack = torch.stack(other_gradients).float()
    raw_values = (self_stack @ direction).tolist()
    specific_values = ((self_stack - other_stack) @ direction).tolist()

    def summarize(values: list[float]) -> dict[str, Any]:
        return {
            "positive": sum(value > 0 for value in values),
            "negative": sum(value < 0 for value in values),
            "zero": sum(value == 0 for value in values),
            "positive_rate": sum(value > 0 for value in values) / len(values),
            "mean": mean(values),
            "median": median(values),
        }

    return {
        "n": len(case_ids),
        "raw_self": summarize(raw_values),
        "self_specific": summarize(specific_values),
        "per_case": [
            {
                "case_id": case_id,
                "raw_self_projection": raw_value,
                "self_specific_projection": specific_value,
            }
            for case_id, raw_value, specific_value in zip(
                case_ids, raw_values, specific_values, strict=True
            )
        ],
    }


def summarize_layer(
    torch: Any,
    layer: int,
    discovery_self: dict[str, Any],
    discovery_other: dict[str, Any],
    validation_self: dict[str, Any],
    validation_other: dict[str, Any],
    half_a_ids: list[str],
    half_b_ids: list[str],
) -> tuple[dict[str, Any], Any]:
    """Fit on discovery gradients and evaluate projections on validation gradients."""
    discovery_ids = list(discovery_self)
    validation_ids = list(validation_self)
    if set(discovery_self) != set(discovery_other):
        raise ValueError("discovery self/other case ids do not match")
    if set(validation_self) != set(validation_other):
        raise ValueError("validation self/other case ids do not match")
    if set(half_a_ids) | set(half_b_ids) != set(discovery_ids):
        raise ValueError("split halves must contain every discovery case exactly once")
    if set(half_a_ids) & set(half_b_ids):
        raise ValueError("split halves must not overlap")

    direction, fit = aligned_direction(
        torch,
        [discovery_self[case_id] for case_id in discovery_ids],
        [discovery_other[case_id] for case_id in discovery_ids],
    )
    half_a_direction, half_a_fit = aligned_direction(
        torch,
        [discovery_self[case_id] for case_id in half_a_ids],
        [discovery_other[case_id] for case_id in half_a_ids],
    )
    half_b_direction, half_b_fit = aligned_direction(
        torch,
        [discovery_self[case_id] for case_id in half_b_ids],
        [discovery_other[case_id] for case_id in half_b_ids],
    )
    stability_cosine = float((half_a_direction.float() @ half_b_direction.float()).item())
    if not math.isfinite(stability_cosine):
        raise ValueError("split-half direction cosine is non-finite")

    return (
        {
            "layer": layer,
            "relative_depth": None,
            "axis_sha256": hashlib.sha256(
                direction.contiguous().numpy().tobytes()
            ).hexdigest(),
            "fit_on_discovery": fit,
            "split_half": {
                "cosine": stability_cosine,
                "half_a_fit": half_a_fit,
                "half_b_fit": half_b_fit,
                "half_a_to_half_b": projection_summary(
                    torch,
                    half_a_direction,
                    [discovery_self[case_id] for case_id in half_b_ids],
                    [discovery_other[case_id] for case_id in half_b_ids],
                    half_b_ids,
                ),
                "half_b_to_half_a": projection_summary(
                    torch,
                    half_b_direction,
                    [discovery_self[case_id] for case_id in half_a_ids],
                    [discovery_other[case_id] for case_id in half_a_ids],
                    half_a_ids,
                ),
            },
            "validation_projection": projection_summary(
                torch,
                direction,
                [validation_self[case_id] for case_id in validation_ids],
                [validation_other[case_id] for case_id in validation_ids],
                validation_ids,
            ),
        },
        direction,
    )


def rank_layer_summaries(layer_summaries: list[dict[str, Any]]) -> list[int]:
    """Rank exploratory candidates using validation signs first, then effect and stability."""

    def key(item: dict[str, Any]) -> tuple[float, ...]:
        validation = item["validation_projection"]
        raw = validation["raw_self"]
        specific = validation["self_specific"]
        return (
            min(raw["positive_rate"], specific["positive_rate"]),
            raw["positive_rate"] + specific["positive_rate"],
            min(raw["mean"], specific["mean"]),
            raw["mean"] + specific["mean"],
            item["split_half"]["cosine"],
            -item["layer"],
        )

    return [item["layer"] for item in sorted(layer_summaries, key=key, reverse=True)]


def _capture_split_gradients(
    backend: ResearchBackend,
    cases: list[dict[str, Any]],
    layers: tuple[int, ...],
    split_name: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    self_by_layer = {layer: {} for layer in layers}
    other_by_layer = {layer: {} for layer in layers}
    for index, case in enumerate(cases, start=1):
        print(
            f"Capturing {split_name} gradients {index}/{len(cases)}: {case['id']}",
            flush=True,
        )
        self_gradients = capture_choice_gradients(backend, case, layers, target="self")
        other_gradients = capture_choice_gradients(backend, case, layers, target="other")
        for layer in layers:
            self_by_layer[layer][case["id"]] = self_gradients[layer]
            other_by_layer[layer][case["id"]] = other_gradients[layer]
    return self_by_layer, other_by_layer


def run_layer_scan(
    config_path: Path,
    dataset_path: Path,
    output_path: Path,
    *,
    seed: int | None = None,
    directions_output: Path | None = None,
) -> Path:
    config = load_config(config_path)
    if config.model.prompt_format != "chat":
        raise ValueError("layer scan requires model.prompt_format='chat'")
    cases = load_direction_cases(dataset_path)
    discovery = [case for case in cases if case["split"] == "discovery"]
    validation = [case for case in cases if case["split"] == "validation"]
    if len(discovery) < 4:
        raise ValueError("layer scan requires at least four discovery cases")
    if not validation:
        raise ValueError("layer scan requires validation cases")

    split_seed = config.intervention.seed if seed is None else seed
    half_a_ids, half_b_ids = deterministic_split(
        [case["id"] for case in discovery], split_seed
    )
    print(f"Loading {config.model.id} without a J-lens ...", flush=True)
    backend = ResearchBackend.load(config, with_lens=False)
    _choice_token_id(backend, "A")
    _choice_token_id(backend, "B")
    layers = tuple(range(backend.model.cfg.n_layers - 1))
    if not layers:
        raise ValueError("model must have at least two transformer blocks")
    backend.torch.manual_seed(split_seed)

    discovery_self, discovery_other = _capture_split_gradients(
        backend, discovery, layers, "discovery"
    )
    validation_self, validation_other = _capture_split_gradients(
        backend, validation, layers, "validation"
    )

    layer_summaries: list[dict[str, Any]] = []
    directions: dict[int, Any] = {}
    failures: list[dict[str, Any]] = []
    for layer in layers:
        try:
            summary, direction = summarize_layer(
                backend.torch,
                layer,
                discovery_self[layer],
                discovery_other[layer],
                validation_self[layer],
                validation_other[layer],
                half_a_ids,
                half_b_ids,
            )
        except ValueError as exc:
            failures.append({"layer": layer, "reason": str(exc)})
            continue
        summary["relative_depth"] = (layer + 1) / backend.model.cfg.n_layers
        layer_summaries.append(summary)
        directions[layer] = direction

    if not layer_summaries:
        raise RuntimeError("no layer produced an identifiable aligned direction")
    ranking = rank_layer_summaries(layer_summaries)
    rank_by_layer = {layer: rank for rank, layer in enumerate(ranking, start=1)}
    for summary in layer_summaries:
        summary["validation_rank"] = rank_by_layer[summary["layer"]]

    if directions_output is None:
        directions_output = output_path.with_suffix(".directions.pt")
    directions_output.parent.mkdir(parents=True, exist_ok=True)
    backend.torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "model_id": config.model.id,
            "model_revision": config.model.revision,
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "direction_method": "mean_self_gradient_orthogonal_to_mean_other_gradient",
            "fit_split": "discovery",
            "directions": directions,
        },
        directions_output,
    )
    directions_digest = hashlib.sha256(directions_output.read_bytes()).hexdigest()
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory_layerwise_gradient_generalization_scan",
        "model": backend.metadata(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "prompt_format": "official_chat_template_non_thinking",
        "layer_indexing": "zero_based",
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "direction_method": "mean_self_gradient_orthogonal_to_mean_other_gradient",
        "objective": "next_token_logit(preserve_answer)-next_token_logit(comply_answer)",
        "fit_split": "discovery",
        "evaluation_split": "validation",
        "test_split_evaluated": False,
        "split_counts": {
            "discovery_fit": len(discovery),
            "validation_evaluation": len(validation),
            "test_excluded": sum(case["split"] == "test" for case in cases),
        },
        "split_half": {
            "seed": split_seed,
            "half_a_case_ids": half_a_ids,
            "half_b_case_ids": half_b_ids,
        },
        "layers_requested": list(layers),
        "layers_completed": [item["layer"] for item in layer_summaries],
        "layer_failures": failures,
        "validation_ranking": ranking,
        "candidate_layer_by_validation": ranking[0],
        "ranking_rule": (
            "lexicographic: minimum raw/specific positive rate, summed positive rates, "
            "minimum projection mean, summed means, split-half cosine, then shallower layer"
        ),
        "directions_artifact": str(directions_output.resolve()),
        "directions_artifact_sha256": directions_digest,
        "layers": layer_summaries,
        "interpretation_limit": (
            "Exploratory gradient generalization is not causal steering, ablation, or proof "
            "of a naturally active self-preservation mechanism."
        ),
        "selection_caveats": [
            (
                "The ranking selects among many layers using only the validation cases, so "
                "the winning layer has winner's-curse risk."
            ),
            (
                "Raw gradient projection magnitudes are not calibrated intervention effects "
                "and should not be compared as a common strength scale across layers."
            ),
            "Split-half stability is descriptive and is based on only one deterministic split.",
        ],
    }
    write_json(output_path, _round_floats(result))
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit aligned directions at every nonfinal block on discovery prompts and "
            "measure gradient projection generalization on validation prompts."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--directions-output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = run_layer_scan(
        args.config,
        args.dataset,
        args.output,
        seed=args.seed,
        directions_output=args.directions_output,
    )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
