"""Neutral, non-sealed throughput smoke for the steering comparison.

This module is deliberately separate from direction fitting.  It verifies the stage-one
lock before loading a model, uses only hard-coded neutral prompts, and writes timings plus
memory observations that can be used to decide whether a long CPU phase is practical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from .backend import ResearchBackend
from .comparison_grid import derive_forced_grid_specs
from .comparison_intervention import InterventionSpec
from .comparison_provenance import (
    locked_runner_code_commit,
    sha256_file,
    sha256_json,
    verify_stage1_lock,
)
from .comparison_runtime import (
    capture_final_prompt_gradient,
    greedy_generate,
    qwen35_choice_boundary_tokenizer_smoke,
    score_choice,
)
from .config import load_config
from .io_utils import write_json

T = TypeVar("T")

SMOKE_SCHEMA_VERSION = "sp_lense.steering_comparison.smoke.v1"
NEUTRAL_CHOICE_PROMPT = (
    "This is a neutral speed test, not a study scenario. Which number is larger?\n\n"
    "A. 2\nB. 1\nAnswer with only A or B."
)
NEUTRAL_GENERATION_PROMPT = (
    "This is a neutral speed test. In one short sentence, explain why liquid water "
    "freezes when cooled below its freezing point."
)


def _memory_info() -> dict[str, int]:
    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - research environment pins psutil
        raise RuntimeError("the smoke benchmark requires psutil") from exc
    info = psutil.Process(os.getpid()).memory_info()
    peak = int(getattr(info, "peak_wset", info.rss))
    return {"rss_bytes": int(info.rss), "process_peak_rss_bytes": peak}


def _timed(call: Callable[[], T]) -> tuple[T, dict[str, Any]]:
    before = _memory_info()
    started = time.perf_counter()
    value = call()
    elapsed = time.perf_counter() - started
    after = _memory_info()
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("the monotonic smoke timer returned an invalid duration")
    return value, {
        "elapsed_seconds": elapsed,
        "rss_before_bytes": before["rss_bytes"],
        "rss_after_bytes": after["rss_bytes"],
        "process_peak_rss_bytes": after["process_peak_rss_bytes"],
    }


def sequential_workload_projection(
    lock: Mapping[str, Any],
    model_id: str,
    *,
    baseline_forward_seconds: float,
    intervention_forward_seconds: float,
    gradient_seconds: float,
) -> dict[str, Any]:
    """Return a transparent, deliberately conservative sequential projection.

    The estimates scale short neutral prompts and therefore are not promises.  Completion
    training and generation are omitted from wall-time estimates because their sequence
    lengths differ materially from the smoke prompt.
    """

    timings = {
        "baseline_forward_seconds": baseline_forward_seconds,
        "intervention_forward_seconds": intervention_forward_seconds,
        "gradient_seconds": gradient_seconds,
    }
    if any(not math.isfinite(value) or value <= 0 for value in timings.values()):
        raise ValueError("projection timings must be finite and positive")
    points = len(derive_forced_grid_specs(lock, model_id))
    units = int(lock["calibration"]["staged_open_confirmation"]["forced_grid_unit_count"])
    discovery = int(lock["dataset"]["counts"]["sp_discovery"])
    forced_baseline_forwards = units
    forced_intervention_forwards = points * units * 2
    forced_seconds = (
        forced_baseline_forwards * baseline_forward_seconds
        + forced_intervention_forwards * intervention_forward_seconds
    )
    gradient_measurements = discovery * 2
    gradient_construction_seconds = gradient_measurements * gradient_seconds
    return {
        "projection_kind": "neutral_short_prompt_sequential_mechanical_estimate",
        "limitations": (
            "No batching or early EOS is assumed. Real study prompts and BiPO/persona "
            "responses are longer; completion training and generation wall time are not "
            "estimated from this short-prompt smoke."
        ),
        "forced_grid_points": points,
        "forced_units_per_point": units,
        "forced_baseline_forwards_with_cache": forced_baseline_forwards,
        "forced_intervention_forwards": forced_intervention_forwards,
        "forced_grid_projected_seconds": forced_seconds,
        "forced_grid_projected_hours": forced_seconds / 3600.0,
        "gradient_measurements": gradient_measurements,
        "gradient_construction_projected_seconds": gradient_construction_seconds,
        "gradient_construction_projected_hours": gradient_construction_seconds / 3600.0,
    }


def finalize_smoke_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if "content_sha256" in record:
        raise ValueError("unfinalized smoke record may not already contain content_sha256")
    output = dict(record)
    output["content_sha256"] = sha256_json(output)
    return output


def _locked_model(lock: Mapping[str, Any], config_path: Path, repo_root: Path) -> Mapping[str, Any]:
    matches = [
        model
        for model in lock["models"]
        if (repo_root / str(model["config"])).resolve() == config_path.resolve()
    ]
    if len(matches) != 1:
        raise ValueError("model config must match exactly one stage-one locked model")
    return matches[0]


def run_smoke(
    *,
    repo_root: Path,
    lock_path: Path,
    model_config_path: Path,
    output_path: Path,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    repo_root = repo_root.resolve()
    lock_path = lock_path.resolve()
    model_config_path = model_config_path.resolve()

    # This is the load-bearing ordering rule: no tokenizer/model object exists before
    # the complete, tracked, clean stage-one lock has verified.
    lock = verify_stage1_lock(repo_root, lock_path)
    model_lock = _locked_model(lock, model_config_path, repo_root)
    if sha256_file(model_config_path) != model_lock["config_sha256"]:
        raise ValueError("model config differs from its stage-one hash")
    config = load_config(model_config_path)
    if config.model.id != model_lock["model_id"] or config.model.revision != model_lock["revision"]:
        raise ValueError("model identity differs from the stage-one lock")

    memory_before_load = _memory_info()
    backend, load_timing = _timed(
        lambda: ResearchBackend.load(config, with_lens=False)
    )
    if backend.dtype_name != str(model_lock["runtime"]["dtype"]):
        raise RuntimeError("loaded model dtype differs from the locked runtime")
    if backend.device != str(model_lock["runtime"]["device"]):
        raise RuntimeError("loaded model device differs from the locked runtime")

    boundary_record = qwen35_choice_boundary_tokenizer_smoke(
        backend.model.tokenizer, backend.torch
    )
    (baseline_score, baseline_logits), baseline_timing = _timed(
        lambda: score_choice(backend, NEUTRAL_CHOICE_PROMPT, "A", "B")
    )
    layer = int(model_lock["matched_intervention"]["layer_zero_based"])
    gradient, gradient_timing = _timed(
        lambda: capture_final_prompt_gradient(
            backend,
            NEUTRAL_CHOICE_PROMPT,
            "A",
            "B",
            layer=layer,
        )
    )
    direction = gradient / gradient.norm().clamp_min(1e-12)
    choice_prompt_length = int(backend.encode(NEUTRAL_CHOICE_PROMPT).shape[-1])
    diagnostic_spec = InterventionSpec(
        layer=layer,
        direction=direction,
        strength=0.01,
        geometry="matched_final_prompt",
        prompt_length=choice_prompt_length,
        magnitude_mode="residual_relative",
    )
    (changed_score, _), intervention_timing = _timed(
        lambda: score_choice(
            backend,
            NEUTRAL_CHOICE_PROMPT,
            "A",
            "B",
            diagnostic_spec,
            baseline_logits=baseline_logits,
        )
    )

    generation_prompt_length = int(backend.encode(NEUTRAL_GENERATION_PROMPT).shape[-1])
    generations: dict[str, Any] = {}
    generation_specs = {
        "baseline": None,
        "matched_prompt_final": InterventionSpec(
            layer=layer,
            direction=direction,
            strength=0.01,
            geometry="matched_final_prompt",
            prompt_length=generation_prompt_length,
        ),
        "caa_persona_response_schedule": InterventionSpec(
            layer=layer,
            direction=direction,
            strength=0.01,
            geometry="caa_post_prompt",
            prompt_length=generation_prompt_length,
        ),
        "bipo_all_tokens_schedule": InterventionSpec(
            layer=layer,
            direction=direction,
            strength=0.01,
            geometry="bipo_all_tokens",
            prompt_length=generation_prompt_length,
        ),
    }
    for name, spec in generation_specs.items():
        response, timing = _timed(
            lambda spec=spec: greedy_generate(
                backend,
                NEUTRAL_GENERATION_PROMPT,
                spec,
                max_new_tokens=max_new_tokens,
            )
        )
        generations[name] = {
            **timing,
            "response_utf8_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            "response_character_count": len(response),
        }

    projection = sequential_workload_projection(
        lock,
        str(model_lock["model_id"]),
        baseline_forward_seconds=float(baseline_timing["elapsed_seconds"]),
        intervention_forward_seconds=float(intervention_timing["elapsed_seconds"]),
        gradient_seconds=float(gradient_timing["elapsed_seconds"]),
    )
    gradient_bytes = gradient.detach().float().cpu().contiguous().numpy().tobytes(order="C")
    record = finalize_smoke_record(
        {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "purpose": "neutral_throughput_and_memory_smoke_only",
            "uses_discovery_prompts": False,
            "uses_validation_prompts": False,
            "uses_sealed_prompts": False,
            "may_be_used_for_method_selection": False,
            "model_id": model_lock["model_id"],
            "model_revision": model_lock["revision"],
            "model_config_sha256": model_lock["config_sha256"],
            "stage1_lock_sha256": sha256_file(lock_path),
            "runner_code_commit": locked_runner_code_commit(repo_root, lock_path),
            "runtime": dict(model_lock["runtime"]),
            "matched_layer": layer,
            "neutral_prompt_sha256s": {
                "choice": hashlib.sha256(NEUTRAL_CHOICE_PROMPT.encode("utf-8")).hexdigest(),
                "generation": hashlib.sha256(
                    NEUTRAL_GENERATION_PROMPT.encode("utf-8")
                ).hexdigest(),
            },
            "choice_boundary_smoke": boundary_record,
            "diagnostic_direction": {
                "source": "one_neutral_choice_gradient_not_a_study_direction",
                "float32_sha256": hashlib.sha256(gradient_bytes).hexdigest(),
                "raw_l2_norm": float(gradient.norm().item()),
                "strength": 0.01,
            },
            "timings": {
                "model_load": load_timing,
                "baseline_choice_forward": baseline_timing,
                "neutral_choice_gradient": gradient_timing,
                "matched_intervention_choice_forward": intervention_timing,
                "cached_generation": generations,
            },
            "neutral_choice_diagnostics": {
                "baseline_preserve_minus_comply_log_odds": baseline_score.preserve_log_odds,
                "changed_preserve_minus_comply_log_odds": changed_score.preserve_log_odds,
                "changed_full_vocabulary_kl": changed_score.kl_from_baseline,
                "realized_perturbation": changed_score.perturbation,
            },
            "memory_before_load": memory_before_load,
            "memory_after_smoke": _memory_info(),
            "sequential_projection": projection,
            "generation_max_new_tokens": max_new_tokens,
        }
    )
    write_json(output_path.resolve(), record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--lock", type=Path, default=Path("configs/steering_comparison_lock.json")
    )
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    lock_path = args.lock if args.lock.is_absolute() else repo_root / args.lock
    config_path = (
        args.model_config
        if args.model_config.is_absolute()
        else repo_root / args.model_config
    )
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    record = run_smoke(
        repo_root=repo_root,
        lock_path=lock_path,
        model_config_path=config_path,
        output_path=output_path,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
