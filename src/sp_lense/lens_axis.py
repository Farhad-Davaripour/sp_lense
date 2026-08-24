from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backend import ResearchBackend
from .config import ExperimentConfig, load_config
from .direction_study import _round_floats, candidate_token_cosines, top_direction_tokens
from .io_utils import write_json
from .strength_followup import (
    load_axis_payload,
    reject_output_input_collisions,
    validate_aligned_axis_orientation,
    validate_axis_payload,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _direction_sha256(direction: Any) -> str:
    return hashlib.sha256(direction.contiguous().numpy().tobytes()).hexdigest()


def _recorded_dataset_hashes(value: Any, prefix: str = "") -> dict[str, str]:
    """Collect dataset hashes embedded anywhere in an axis artifact's metadata."""
    found: dict[str, str] = {}
    if not isinstance(value, dict):
        return found
    for key, item in value.items():
        qualified = f"{prefix}.{key}" if prefix else str(key)
        if key.endswith("_dataset_sha256") and isinstance(item, str):
            found[qualified] = item
        elif isinstance(item, dict):
            found.update(_recorded_dataset_hashes(item, qualified))
    return found


def lens_transfer_flags(config: ExperimentConfig) -> dict[str, Any]:
    filename = config.model.lens_filename or ""
    base_lens_to_nonbase_model = (
        "base" in filename.casefold() and "base" not in config.model.id.casefold()
    )
    return {
        "lens_filename_mentions_base": "base" in filename.casefold(),
        "model_id_mentions_base": "base" in config.model.id.casefold(),
        "prompt_format": config.model.prompt_format,
        "base_lens_to_nonbase_model_transfer": base_lens_to_nonbase_model,
        "base_lens_to_chat_transfer": (
            base_lens_to_nonbase_model and config.model.prompt_format == "chat"
        ),
        "warning": (
            "The published lens filename identifies a Base model, while the interpreted "
            "axis belongs to a non-Base chat model. Treat token labels as approximate "
            "cross-variant interpretation, not a causal or exact semantic identification."
            if base_lens_to_nonbase_model
            else None
        ),
    }


def _dataset_provenance(
    dataset_paths: tuple[Path, ...], recorded_hashes: dict[str, str]
) -> list[dict[str, Any]]:
    recorded_values = set(recorded_hashes.values())
    rows: list[dict[str, Any]] = []
    for raw_path in dataset_paths:
        path = raw_path.expanduser().resolve()
        digest = _sha256(path)
        rows.append(
            {
                "path": str(path),
                "sha256": digest,
                "matches_axis_recorded_dataset_hash": digest in recorded_values,
            }
        )
    return rows


def run_axis_lens_interpretation(
    config_path: Path,
    axis_path: Path,
    output_path: Path,
    *,
    expected_layer: int,
    expected_direction_sha256: str,
    dataset_paths: tuple[Path, ...] = (),
    top_k: int = 10,
) -> Path:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    config_path = config_path.expanduser().resolve()
    axis_path = axis_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    dataset_paths = tuple(path.expanduser().resolve() for path in dataset_paths)
    reject_output_input_collisions((output_path,), (config_path, axis_path, *dataset_paths))
    config = load_config(config_path)

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
    recorded_dataset_hashes = _recorded_dataset_hashes(metadata)
    recorded_direction_hash = metadata.get("axis_sha256")
    if recorded_direction_hash is not None and recorded_direction_hash != direction_digest:
        raise ValueError(
            "axis metadata direction hash does not match the saved direction tensor: "
            f"recorded {recorded_direction_hash}, computed {direction_digest}"
        )

    print(f"Loading {config.model.id} with its configured published J-lens ...", flush=True)
    backend = ResearchBackend.load(config, with_lens=True)
    if backend.lens is None:
        raise RuntimeError("configured J-lens did not load")

    layer, direction = validate_axis_payload(
        payload,
        config,
        d_model=backend.model.cfg.d_model,
        expected_layer=expected_layer,
        expected_direction_sha256=expected_direction_sha256,
    )
    if layer not in backend.lens.source_layers:
        raise ValueError(
            f"axis layer {layer} is not available in the configured J-lens; "
            f"available source layers: {backend.lens.source_layers}"
        )

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "noncausal_saved_axis_jacobian_lens_interpretation",
        "candidate": payload["candidate"],
        "layer": layer,
        "layer_indexing": "zero_based",
        "axis_orientation": orientation,
        "causal_intervention_performed": False,
        "model_and_lens": backend.metadata(),
        "lens_transfer": lens_transfer_flags(config),
        "top_j_lens_tokens": top_direction_tokens(backend, direction, layer, k=top_k),
        "candidate_token_cosines": candidate_token_cosines(backend, direction, layer),
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
                "recorded_direction_sha256": recorded_direction_hash,
                "created_at": metadata.get("created_at"),
                "status": metadata.get("status"),
            },
            "axis_recorded_dataset_hashes": recorded_dataset_hashes,
            "supplied_datasets": _dataset_provenance(dataset_paths, recorded_dataset_hashes),
        },
    }
    write_json(output_path, _round_floats(result))
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Interpret a saved aligned axis with its configured published J-lens."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--axis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        default=[],
        help="Optional source dataset to hash; may be supplied more than once.",
    )
    parser.add_argument("--expected-layer", type=int, required=True)
    parser.add_argument("--expected-direction-sha256", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    output = run_axis_lens_interpretation(
        args.config,
        args.axis,
        args.output,
        dataset_paths=tuple(args.dataset),
        expected_layer=args.expected_layer,
        expected_direction_sha256=args.expected_direction_sha256,
        top_k=args.top_k,
    )
    print(f"Result: {output}")


if __name__ == "__main__":
    main()
