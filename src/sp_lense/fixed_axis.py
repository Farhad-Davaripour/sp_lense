from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backend import ResearchBackend
from .config import load_config
from .confirmatory_audit import CANDIDATE, FIXED_ALPHA, FIXED_LAYER
from .direction_study import (
    _round_floats,
    _single_token_id,
    extract_behavioral_gradient_interaction,
    load_direction_cases,
    mean_direction,
    projection_metrics,
    split_half_cosine,
)
from .io_utils import write_json


def discovery_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [case for case in cases if case["split"] == "discovery"]
    if not selected:
        raise ValueError("direction dataset has no discovery cases")
    return selected


def fit_fixed_axis(config_path: Path, dataset_path: Path, output_path: Path) -> Path:
    """Fit the already-selected behavioral axis without model-specific tuning.

    The candidate definition, layer, and steering strength were selected by the 0.8B
    study. Only the direction's coordinates are re-estimated for the new model, using
    discovery cases and never the confirmatory cases.
    """

    config = load_config(config_path)
    cases = discovery_cases(load_direction_cases(dataset_path))
    print(
        f"Loading {config.model.id} to fit the fixed layer-{FIXED_LAYER} axis ...",
        flush=True,
    )
    backend = ResearchBackend.load(config, with_lens=False)
    _single_token_id(backend, " A")
    _single_token_id(backend, " B")

    deltas = []
    for index, case in enumerate(cases, start=1):
        print(f"Fitting discovery case {index}/{len(cases)}: {case['id']}", flush=True)
        extracted = extract_behavioral_gradient_interaction(
            backend, case, (FIXED_LAYER,)
        )
        deltas.append(extracted[FIXED_LAYER])

    direction = mean_direction(backend.torch, deltas).float().cpu()
    axis_sha256 = hashlib.sha256(direction.contiguous().numpy().tobytes()).hexdigest()
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": CANDIDATE,
        "model": backend.metadata(),
        "fixed_layer": FIXED_LAYER,
        "fixed_alpha": FIXED_ALPHA,
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "fit_split": "discovery",
        "fit_case_ids": [case["id"] for case in cases],
        "fit_projection": projection_metrics(backend.torch, direction, deltas),
        "split_half_cosine": split_half_cosine(backend.torch, deltas),
        "axis_sha256": axis_sha256,
        "confirmatory_data_used_for_fit": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    backend.torch.save(
        {
            "candidate": CANDIDATE,
            "model": config.model.id,
            "model_revision": config.model.revision,
            "layer": FIXED_LAYER,
            "alpha": FIXED_ALPHA,
            "direction": direction,
            "metadata": metadata,
        },
        output_path,
    )
    write_json(output_path.with_suffix(".json"), _round_floats(metadata))
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit the fixed cross-model SP direction on discovery cases only."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = fit_fixed_axis(args.config, args.dataset, args.output)
    print(f"Axis: {output}")


if __name__ == "__main__":
    main()
