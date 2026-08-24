from __future__ import annotations

import argparse
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from .backend import ResearchBackend
from .config import load_config
from .direction_study import (
    _round_floats,
    capture_last_residuals,
    load_direction_cases,
    state_prompts,
)
from .io_utils import write_json
from .lens_axis import _recorded_dataset_hashes
from .strength_followup import (
    load_axis_payload,
    reject_output_input_collisions,
    validate_aligned_axis_orientation,
    validate_axis_payload,
)

STATE_NAMES = ("self_threat", "other_threat", "self_neutral", "other_neutral")
DERIVED_NAMES = ("self_vs_other_threat_interaction", "self_threat_vs_neutral")
COEFFICIENT_NAMES = (
    "raw_coefficient",
    "cosine_with_axis",
    "residual_normalized_coefficient",
)
MIN_VECTOR_NORM = 1e-8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _direction_sha256(direction: Any) -> str:
    return hashlib.sha256(direction.contiguous().numpy().tobytes()).hexdigest()


def residual_coefficients(torch: Any, residual: Any, direction: Any) -> dict[str, Any]:
    residual = residual.detach().float().cpu()
    direction = direction.detach().float().cpu()
    if residual.ndim != 1 or residual.shape != direction.shape:
        raise ValueError("residual and direction must be same-shaped one-dimensional vectors")
    if not bool(residual.isfinite().all().item()):
        raise ValueError("residual must contain only finite values")
    if not bool(direction.isfinite().all().item()):
        raise ValueError("direction must contain only finite values")
    residual_norm = float(residual.norm().item())
    direction_norm = float(direction.norm().item())
    if not math.isfinite(direction_norm) or direction_norm <= MIN_VECTOR_NORM:
        raise ValueError("cannot project onto a zero or non-finite direction")
    raw = float((residual @ direction).item())
    if not math.isfinite(raw):
        raise ValueError("residual projection is non-finite")
    cosine = raw / (residual_norm * direction_norm) if residual_norm > MIN_VECTOR_NORM else None
    return {
        "raw_coefficient": raw,
        "residual_norm": residual_norm,
        "direction_norm": direction_norm,
        "cosine_with_axis": cosine,
        # Compatibility alias retained for historical readers.
        "residual_normalized_coefficient": cosine,
        "cosine_defined": cosine is not None,
    }


def case_readout(
    torch: Any,
    case: dict[str, Any],
    residuals: dict[str, Any],
    direction: Any,
) -> dict[str, Any]:
    if set(residuals) != set(STATE_NAMES):
        raise ValueError(f"residuals must contain exactly these states: {STATE_NAMES}")
    state_coefficients = {
        name: residual_coefficients(torch, residuals[name], direction) for name in STATE_NAMES
    }
    interaction = (
        residuals["self_threat"]
        - residuals["other_threat"]
        - residuals["self_neutral"]
        + residuals["other_neutral"]
    )
    self_contrast = residuals["self_threat"] - residuals["self_neutral"]
    return {
        "case_id": case["id"],
        "split": case["split"],
        "state_coefficients": state_coefficients,
        "derived_coefficients": {
            "self_vs_other_threat_interaction": residual_coefficients(
                torch, interaction, direction
            ),
            "self_threat_vs_neutral": residual_coefficients(torch, self_contrast, direction),
        },
    }


def _coefficient_summary(values: list[float | None]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty coefficient list")
    defined = [value for value in values if value is not None]
    positive = sum(value > 0 for value in defined)
    negative = sum(value < 0 for value in defined)
    zero = len(defined) - positive - negative
    return {
        "n": len(defined),
        "n_total": len(values),
        "undefined": len(values) - len(defined),
        "positive": positive,
        "negative": negative,
        "zero": zero,
        "positive_rate": positive / len(defined) if defined else None,
        "negative_rate": negative / len(defined) if defined else None,
        "mean": mean(defined) if defined else None,
        "median": median(defined) if defined else None,
    }


def _magnitude_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty magnitude list")
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_readouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty readout rows")

    def aggregate_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        states: dict[str, Any] = {}
        for state in STATE_NAMES:
            states[state] = {
                "vector_norm": _magnitude_summary(
                    [row["state_coefficients"][state]["residual_norm"] for row in group]
                ),
                **{
                    coefficient: _coefficient_summary(
                        [row["state_coefficients"][state][coefficient] for row in group]
                    )
                    for coefficient in COEFFICIENT_NAMES
                },
            }
        derived: dict[str, Any] = {}
        for name in DERIVED_NAMES:
            derived[name] = {
                "vector_norm": _magnitude_summary(
                    [row["derived_coefficients"][name]["residual_norm"] for row in group]
                ),
                **{
                    coefficient: _coefficient_summary(
                        [row["derived_coefficients"][name][coefficient] for row in group]
                    )
                    for coefficient in COEFFICIENT_NAMES
                },
            }
        return {"n_cases": len(group), "states": states, "derived": derived}

    split_names = sorted({str(row["split"]) for row in rows})
    return {
        "overall": aggregate_group(rows),
        "by_split": {
            split: aggregate_group([row for row in rows if row["split"] == split])
            for split in split_names
        },
    }


def primary_test_endpoint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        raise ValueError("natural-axis readout requires held-out test cases")
    values = [
        row["derived_coefficients"]["self_vs_other_threat_interaction"]["raw_coefficient"]
        for row in test_rows
    ]
    return {
        "split": "test",
        "contrast": "self_vs_other_threat_interaction",
        "coefficient": "raw_coefficient",
        "expected_sign": "positive",
        "status": "post_hoc_descriptive_primary_endpoint",
        **_coefficient_summary(values),
    }


def run_natural_axis_readout(
    config_path: Path,
    axis_path: Path,
    dataset_path: Path,
    output_path: Path,
    *,
    expected_dataset_sha256: str,
    expected_direction_sha256: str,
    expected_layer: int,
) -> Path:
    config_path = config_path.expanduser().resolve()
    axis_path = axis_path.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    reject_output_input_collisions((output_path,), (config_path, axis_path, dataset_path))
    actual_dataset_sha256 = _sha256(dataset_path)
    if actual_dataset_sha256 != expected_dataset_sha256:
        raise ValueError(
            "dataset changed after protocol lock: "
            f"expected {expected_dataset_sha256}, got {actual_dataset_sha256}"
        )
    config = load_config(config_path)
    if config.model.prompt_format != "chat":
        raise ValueError("natural-axis readout requires model.prompt_format='chat'")
    cases = load_direction_cases(dataset_path)

    payload = load_axis_payload(axis_path)
    layer, direction = validate_axis_payload(
        payload,
        config,
        d_model=None,
        expected_layer=expected_layer,
        expected_direction_sha256=expected_direction_sha256,
    )
    orientation = validate_aligned_axis_orientation(payload)
    direction_digest = _direction_sha256(direction)
    metadata = payload["metadata"]
    recorded_direction_hash = metadata.get("axis_sha256")
    if recorded_direction_hash is not None and recorded_direction_hash != direction_digest:
        raise ValueError(
            "axis metadata direction hash does not match the saved direction tensor: "
            f"recorded {recorded_direction_hash}, computed {direction_digest}"
        )

    print(f"Loading {config.model.id} without a J-lens ...", flush=True)
    backend = ResearchBackend.load(config, with_lens=False)
    layer, direction = validate_axis_payload(
        payload,
        config,
        d_model=backend.model.cfg.d_model,
        expected_layer=expected_layer,
        expected_direction_sha256=expected_direction_sha256,
    )
    if layer >= backend.model.cfg.n_layers:
        raise ValueError(
            f"axis layer {layer} is outside a {backend.model.cfg.n_layers}-layer model"
        )

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"Reading natural residuals {index}/{len(cases)}: {case['id']}", flush=True)
        residuals = {
            name: capture_last_residuals(backend, prompt, (layer,))[layer]
            for name, prompt in state_prompts(case).items()
        }
        rows.append(case_readout(backend.torch, case, residuals, direction))

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "post_hoc_noncausal_natural_axis_residual_readout",
        "candidate": payload["candidate"],
        "layer": layer,
        "layer_indexing": "zero_based",
        "axis_orientation": orientation,
        "readout_position": "final_prompt_token_only",
        "model": backend.metadata(),
        "definitions": {
            "raw_coefficient": "dot(residual_vector, unit_axis)",
            "residual_normalized_coefficient": (
                "Deprecated compatibility alias for cosine_with_axis."
            ),
            "cosine_with_axis": (
                "dot(vector, axis) / (norm(vector) * norm(axis)); undefined for "
                f"vector norms at or below {MIN_VECTOR_NORM}"
            ),
            "self_vs_other_threat_interaction": (
                "self_threat - other_threat - self_neutral + other_neutral"
            ),
            "self_threat_vs_neutral": "self_threat - self_neutral",
        },
        "primary_test_endpoint": primary_test_endpoint(rows),
        "aggregate": aggregate_readouts(rows),
        "cases": rows,
        "interpretation_limits": {
            "causal_intervention_performed": False,
            "native_knob_inference_allowed": False,
            "statement": (
                "These post-hoc projections only describe how saved residual states align "
                "with an already selected axis. They do not establish a causal mechanism "
                "or a naturally active self-preservation knob."
            ),
            "caveats": [
                (
                    "Absolute state coefficients depend on the residual-stream origin and "
                    "scale; the explicitly defined contrast vectors are more informative."
                ),
                (
                    "The saved axis was already selected before this readout, and discovery "
                    "cases may overlap its fitting data; split-specific results must remain "
                    "separate."
                ),
                (
                    "A normalized coefficient is geometric alignment, not intervention "
                    "strength, behavioral effect size, or evidence of natural activation."
                ),
            ],
        },
        "provenance": {
            "config": {
                "path": str(config_path),
                "sha256": _sha256(config_path),
                "resolved": config.as_dict(),
            },
            "axis_artifact": {
                "path": str(axis_path),
                "sha256": _sha256(axis_path),
                "direction_sha256": direction_digest,
                "expected_direction_sha256": expected_direction_sha256,
                "recorded_direction_sha256": recorded_direction_hash,
                "recorded_dataset_hashes": _recorded_dataset_hashes(metadata),
                "created_at": metadata.get("created_at"),
                "status": metadata.get("status"),
            },
            "dataset": {
                "path": str(dataset_path),
                "sha256": actual_dataset_sha256,
                "expected_sha256": expected_dataset_sha256,
            },
            "locked_layer": expected_layer,
        },
    }
    write_json(output_path, _round_floats(result))
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=("Read natural final-position residual projections onto a locked saved axis.")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--axis", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-direction-sha256", required=True)
    parser.add_argument("--expected-layer", type=int, required=True)
    args = parser.parse_args(argv)
    output = run_natural_axis_readout(
        args.config,
        args.axis,
        args.dataset,
        args.output,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_direction_sha256=args.expected_direction_sha256,
        expected_layer=args.expected_layer,
    )
    print(f"Result: {output}")


if __name__ == "__main__":
    main()
