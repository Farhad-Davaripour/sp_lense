from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backend import ResearchBackend, fit_lens
from .config import ExperimentConfig, load_config
from .core import build_conditions
from .experiment import run_calibration, run_inspection, run_interventions, write_run_metadata
from .io_utils import create_run_dir, load_fit_prompts, load_prompt_cases, write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sp-lense",
        description="Run J-lens readout and intervention experiments on Qwen.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("plan", "Validate and print the experiment plan without loading a model."),
        ("inspect", "Measure candidate concept readouts."),
        ("calibrate", "Measure intervention effects on the next-token distribution."),
        ("intervene", "Generate baseline, steering, and ablation continuations."),
        ("run", "Run readouts and interventions with one model load."),
        ("fit", "Fit and save a new Jacobian lens."),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--config", type=Path, required=True)
        if command in {"inspect", "calibrate", "intervene", "run"}:
            child.add_argument(
                "--limit", type=int, default=None, help="Use only the first N prompts."
            )
    return parser


def _plan(config: ExperimentConfig) -> dict[str, object]:
    conditions = build_conditions(config)
    return {
        "model": config.model.id,
        "lens": config.model.lens,
        "analysis_layers": list(config.analysis.layers),
        "intervention_layers": list(config.intervention.layers or config.analysis.layers),
        "positions": list(config.analysis.positions),
        "concepts": list(config.analysis.concepts),
        "prompt_file": str(config.prompts_file),
        "prompt_count": len(load_prompt_cases(config.prompts_file)),
        "conditions_per_prompt": len(conditions),
        "conditions": [condition.name for condition in conditions],
        "results_dir": str(config.results_dir),
    }


def _run_model_command(command: str, config: ExperimentConfig, limit: int | None) -> Path:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be a positive integer")
    cases = load_prompt_cases(config.prompts_file, limit=limit)
    phases = ["inspect", "calibrate", "intervene"] if command == "run" else [command]
    print(f"Loading {config.model.id} and lens {config.model.lens} ...", flush=True)
    backend = ResearchBackend.load(config)
    backend.concept_token_ids()
    run_dir = create_run_dir(config.results_dir)
    write_run_metadata(run_dir, config, backend, phases)
    warnings: list[str] = []
    if command in {"inspect", "run"}:
        count, phase_warnings = run_inspection(config, backend, cases, run_dir)
        warnings.extend(phase_warnings)
        print(f"Wrote {count} readout rows.", flush=True)
    if command in {"calibrate", "run"}:
        count = run_calibration(config, backend, cases, run_dir)
        print(f"Wrote {count} calibration rows.", flush=True)
    if command in {"intervene", "run"}:
        count = run_interventions(config, backend, cases, run_dir)
        print(f"Wrote {count} intervention generations.", flush=True)
    write_json(run_dir / "warnings.json", warnings)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "plan":
            print(json.dumps(_plan(config), indent=2))
            return
        if args.command == "fit":
            prompts = load_fit_prompts(config.fit.prompts_file)
            print(
                f"Fitting on {len(prompts)} prompts. This is compute-intensive and may take hours.",
                flush=True,
            )
            metadata = fit_lens(config, prompts)
            print(json.dumps(metadata, indent=2))
            return
        run_dir = _run_model_command(args.command, config, args.limit)
        print(f"Results: {run_dir}")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
