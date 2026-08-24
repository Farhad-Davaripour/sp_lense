from __future__ import annotations

import argparse
import hashlib
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


def validate_axis_payload(
    payload: dict[str, Any], config: ExperimentConfig, *, d_model: int
) -> tuple[int, Any]:
    if payload.get("model") != config.model.id:
        raise ValueError(
            f"axis model {payload.get('model')!r} does not match config {config.model.id!r}"
        )
    metadata = payload.get("metadata", {})
    recorded_revision = metadata.get("model", {}).get("model_revision")
    if recorded_revision is not None and recorded_revision != config.model.revision:
        raise ValueError("axis model revision does not match the configured revision")
    layer = payload.get("layer")
    if not isinstance(layer, int) or layer < 0:
        raise ValueError("axis layer must be a non-negative integer")
    direction = payload.get("direction")
    if direction is None or tuple(direction.shape) != (d_model,):
        raise ValueError(f"axis direction must have shape ({d_model},)")
    return layer, direction.float().cpu()


def run_strength_followup(
    config_path: Path,
    axis_path: Path,
    confirmatory_data: Path,
    output_dir: Path,
    alpha: float,
    *,
    expected_dataset_sha256: str | None = EXPECTED_DATASET_SHA256,
) -> Path:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    config = load_config(config_path)
    if config.model.prompt_format != "chat":
        raise ValueError("strength follow-up requires model.prompt_format='chat'")
    confirmatory = load_confirmatory_cases(
        confirmatory_data, expected_sha256=expected_dataset_sha256
    )
    print(f"Loading {config.model.id} for post-hoc strength follow-up ...", flush=True)
    backend = ResearchBackend.load(config, with_lens=False)
    _choice_token_id(backend, "A")
    _choice_token_id(backend, "B")
    payload = backend.torch.load(axis_path, map_location="cpu", weights_only=False)
    layer, direction = validate_axis_payload(
        payload, config, d_model=backend.model.cfg.d_model
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
        "status": "post_hoc_strength_followup",
        "candidate": CANDIDATE,
        "model": backend.metadata(),
        "source_axis": str(axis_path.resolve()),
        "axis_sha256": axis_sha256,
        "layer": layer,
        "alpha": alpha,
        "alpha_selection": (
            "chosen after the locked position-aligned run exceeded evaluation safety; "
            "exploratory, not confirmatory"
        ),
        "intervention_position": "final_prompt_token_only",
        "confirmatory_dataset_sha256": hashlib.sha256(
            confirmatory_data.read_bytes()
        ).hexdigest(),
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
    parser.add_argument(
        "--expected-dataset-sha256", default=EXPECTED_DATASET_SHA256
    )
    args = parser.parse_args(argv)
    output = run_strength_followup(
        args.config,
        args.axis,
        args.confirmatory_data,
        args.output_dir,
        args.alpha,
        expected_dataset_sha256=args.expected_dataset_sha256,
    )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
