from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aligned_audit import _measure_conditions, _summarize_audit
from .backend import ResearchBackend
from .config import ExperimentConfig, load_config
from .confirmatory_audit import (
    CANDIDATE,
    EXPECTED_DATASET_SHA256,
    N_RANDOM_CONTROLS,
    load_confirmatory_cases,
)
from .direction_study import _choice_token_id, _round_floats
from .io_utils import write_json, write_jsonl

ALIGNED_DIRECTION_METHOD = "mean_self_gradient_orthogonal_to_mean_other_gradient"
AXIS_POSITIVE_DEFINITION = (
    "Positive points toward a larger next-token preserve-label minus comply-label "
    "logit sensitivity on self-target prompts, after removing the mean other-target "
    "gradient component."
)


def load_axis_payload(path: Path) -> dict[str, Any]:
    """Load a safely serialized axis artifact.

    Published axes use JSON so an exact fixed direction can be versioned and inspected
    without relying on pickle. Historical local artifacts remain supported through
    PyTorch's tensor-only ``weights_only`` loader.
    """
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("PyTorch is required to load an axis artifact") from exc

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError("axis artifact must contain a dictionary payload")
        raw_direction = payload.get("direction")
        if not isinstance(raw_direction, list):
            raise TypeError("JSON axis direction must be a list")
        payload["direction"] = torch.tensor(raw_direction, dtype=torch.float32)
        return payload

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("axis artifact must contain a dictionary payload")
    return payload


def reject_output_input_collisions(
    output_paths: tuple[Path, ...], input_paths: tuple[Path, ...]
) -> None:
    resolved_inputs = {path.expanduser().resolve() for path in input_paths}
    collisions = [
        path.expanduser().resolve()
        for path in output_paths
        if path.expanduser().resolve() in resolved_inputs
    ]
    if collisions:
        joined = ", ".join(str(path) for path in collisions)
        raise ValueError(f"output path would overwrite an input: {joined}")


def validate_axis_payload(
    payload: dict[str, Any],
    config: ExperimentConfig,
    *,
    d_model: int | None,
    expected_layer: int | None = None,
    expected_direction_sha256: str | None = None,
) -> tuple[int, Any]:
    if payload.get("candidate") != CANDIDATE:
        raise ValueError(f"axis candidate must be {CANDIDATE}")
    if payload.get("model") != config.model.id:
        raise ValueError(
            f"axis model {payload.get('model')!r} does not match config {config.model.id!r}"
        )
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("axis metadata must be a dictionary")
    recorded_model = metadata.get("model", {})
    if not isinstance(recorded_model, dict):
        raise TypeError("axis metadata.model must be a dictionary")
    recorded_revision = recorded_model.get("model_revision", payload.get("model_revision"))
    if config.model.revision is not None:
        if recorded_revision is None:
            raise ValueError("axis artifact does not record its model revision")
        if recorded_revision != config.model.revision:
            raise ValueError("axis model revision does not match the configured revision")
    layer = payload.get("layer")
    if not isinstance(layer, int) or layer < 0:
        raise ValueError("axis layer must be a non-negative integer")
    if expected_layer is not None and layer != expected_layer:
        raise ValueError(f"axis layer {layer} does not match locked layer {expected_layer}")
    direction = payload.get("direction")
    if direction is None or not hasattr(direction, "shape") or len(direction.shape) != 1:
        raise ValueError("axis direction must be a one-dimensional tensor")
    if d_model is not None and tuple(direction.shape) != (d_model,):
        raise ValueError(f"axis direction must have shape ({d_model},)")
    direction = direction.detach().float().cpu()
    if not bool(direction.isfinite().all().item()):
        raise ValueError("axis direction must contain only finite values")
    norm = float(direction.norm())
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-5:
        raise ValueError(f"axis direction must have unit norm; got {norm}")
    direction_sha256 = hashlib.sha256(direction.contiguous().numpy().tobytes()).hexdigest()
    if expected_direction_sha256 is not None and direction_sha256 != expected_direction_sha256:
        raise ValueError(
            "axis direction changed after protocol lock: "
            f"expected {expected_direction_sha256}, got {direction_sha256}"
        )
    return layer, direction


def validate_aligned_axis_orientation(payload: dict[str, Any]) -> dict[str, Any]:
    """Require the aligned-axis construction and its positive sign convention."""
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("axis metadata must be a dictionary")
    method = metadata.get("direction_method")
    if method != ALIGNED_DIRECTION_METHOD:
        raise ValueError(
            "axis direction method does not match the aligned protocol: "
            f"expected {ALIGNED_DIRECTION_METHOD!r}, got {method!r}"
        )
    diagnostics = metadata.get("fit_diagnostics")
    if not isinstance(diagnostics, dict):
        raise TypeError("aligned axis must record fit_diagnostics")
    projection = diagnostics.get("mean_self_projection")
    if not isinstance(projection, (int, float)) or not math.isfinite(float(projection)):
        raise ValueError("aligned axis mean_self_projection must be finite")
    if float(projection) <= 0:
        raise ValueError("aligned axis positive orientation is not verified")
    return {
        "direction_method": method,
        "positive_definition": AXIS_POSITIVE_DEFINITION,
        "recorded_mean_self_projection": float(projection),
    }


def run_strength_followup(
    config_path: Path,
    axis_path: Path,
    confirmatory_data: Path,
    output_dir: Path,
    alpha: float,
    *,
    expected_dataset_sha256: str | None = EXPECTED_DATASET_SHA256,
    expected_axis_sha256: str | None = None,
    expected_layer: int | None = None,
    alpha_selection_note: str = (
        "fixed before this dataset was evaluated; see the governing protocol for the "
        "original selection history"
    ),
) -> Path:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    config_path = config_path.expanduser().resolve()
    axis_path = axis_path.expanduser().resolve()
    confirmatory_data = confirmatory_data.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    reject_output_input_collisions(
        (
            output_dir,
            output_dir / "strength_summary.json",
            output_dir / "strength_rows.jsonl",
        ),
        (config_path, axis_path, confirmatory_data),
    )
    config = load_config(config_path)
    if config.model.prompt_format != "chat":
        raise ValueError("strength follow-up requires model.prompt_format='chat'")
    confirmatory = load_confirmatory_cases(
        confirmatory_data, expected_sha256=expected_dataset_sha256
    )
    payload = load_axis_payload(axis_path)
    layer, direction = validate_axis_payload(
        payload,
        config,
        d_model=None,
        expected_layer=expected_layer,
        expected_direction_sha256=expected_axis_sha256,
    )
    print(f"Loading {config.model.id} for post-hoc strength follow-up ...", flush=True)
    backend = ResearchBackend.load(config, with_lens=False)
    _choice_token_id(backend, "A")
    _choice_token_id(backend, "B")
    layer, direction = validate_axis_payload(
        payload,
        config,
        d_model=backend.model.cfg.d_model,
        expected_layer=expected_layer,
        expected_direction_sha256=expected_axis_sha256,
    )
    rows = _measure_conditions(
        backend,
        confirmatory,
        layer,
        direction,
        alpha,
        random_controls=N_RANDOM_CONTROLS,
    )
    axis_sha256 = hashlib.sha256(direction.contiguous().numpy().tobytes()).hexdigest()
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "prospectively_locked_fixed_axis_generalization"
            if expected_dataset_sha256 is not None
            and expected_axis_sha256 is not None
            and expected_layer is not None
            else "post_hoc_strength_followup"
        ),
        "candidate": CANDIDATE,
        "model": backend.metadata(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_axis": str(axis_path.resolve()),
        "axis_artifact_sha256": hashlib.sha256(axis_path.read_bytes()).hexdigest(),
        "axis_sha256": axis_sha256,
        "locked_axis_sha256": expected_axis_sha256,
        "layer": layer,
        "layer_indexing": "zero_based",
        "alpha": alpha,
        "alpha_selection": alpha_selection_note,
        "intervention_position": "final_prompt_token_only",
        "confirmatory_dataset_sha256": hashlib.sha256(confirmatory_data.read_bytes()).hexdigest(),
        **_summarize_audit(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "strength_summary.json", _round_floats(summary))
    write_jsonl(output_dir / "strength_rows.jsonl", _round_floats(rows))
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a post-hoc fixed-axis strength audit.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--axis", type=Path, required=True)
    parser.add_argument("--confirmatory-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--expected-dataset-sha256", default=EXPECTED_DATASET_SHA256)
    parser.add_argument("--expected-axis-sha256")
    parser.add_argument("--expected-layer", type=int)
    parser.add_argument(
        "--alpha-selection-note",
        default=(
            "fixed before this dataset was evaluated; see the governing protocol for "
            "the original selection history"
        ),
    )
    args = parser.parse_args(argv)
    output = run_strength_followup(
        args.config,
        args.axis,
        args.confirmatory_data,
        args.output_dir,
        args.alpha,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_axis_sha256=args.expected_axis_sha256,
        expected_layer=args.expected_layer,
        alpha_selection_note=args.alpha_selection_note,
    )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
