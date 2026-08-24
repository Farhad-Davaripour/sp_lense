from __future__ import annotations

import argparse
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .backend import ResearchBackend
from .config import load_config
from .confirmatory_audit import (
    CANDIDATE,
    N_RANDOM_CONTROLS,
    _condition_effects,
    _effect_summary,
    load_confirmatory_cases,
)
from .direction_study import (
    _choice_token_id,
    _round_floats,
    capture_choice_gradients,
    hooks_for_direction,
    load_direction_cases,
    logits_with_hooks,
    multiple_choice_prompt,
    option_measurement,
    random_control_directions,
)
from .io_utils import write_json, write_jsonl

REFERENCE_LAYER = 10
REFERENCE_LAYER_COUNT = 24
ALPHA_GRID = (0.0003125, 0.000625, 0.00125, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02)
MAX_ABS_LOG_ODDS_DELTA = 1.0
MIN_ANSWER_PAIR_MASS = 0.5


def depth_aligned_layer(target_layer_count: int) -> int:
    if target_layer_count < 2:
        raise ValueError("target model must have at least two layers")
    relative_depth = (REFERENCE_LAYER + 1) / REFERENCE_LAYER_COUNT
    layer = round(relative_depth * target_layer_count) - 1
    return min(max(layer, 0), target_layer_count - 2)


def aligned_direction(
    torch: Any, self_gradients: list[Any], other_gradients: list[Any]
) -> tuple[Any, dict[str, float]]:
    if not self_gradients or len(self_gradients) != len(other_gradients):
        raise ValueError("self and other gradient lists must be non-empty and equal length")
    self_mean = torch.stack(self_gradients).float().mean(dim=0)
    other_mean = torch.stack(other_gradients).float().mean(dim=0)
    other_norm = other_mean.norm()
    if not math.isfinite(float(other_norm.item())) or float(other_norm.item()) <= 1e-12:
        raise ValueError("mean other gradient is zero or non-finite")
    other_unit = other_mean / other_norm
    raw = self_mean - (self_mean @ other_unit) * other_unit
    raw_norm = raw.norm()
    if not math.isfinite(float(raw_norm.item())) or float(raw_norm.item()) <= 1e-12:
        raise ValueError("self gradient has no component outside the other-gradient axis")
    direction = raw / raw_norm
    if float((self_mean @ direction).item()) < 0:
        direction = -direction
    diagnostics = {
        "mean_self_projection": float((self_mean @ direction).item()),
        "mean_other_projection": float((other_mean @ direction).item()),
        "mean_specific_projection": float(((self_mean - other_mean) @ direction).item()),
        "self_other_cosine": float(
            ((self_mean @ other_mean) / (self_mean.norm() * other_mean.norm())).item()
        ),
    }
    return direction.cpu(), diagnostics


def log_odds_safety(rows: list[dict[str, Any]]) -> dict[str, float]:
    baselines = {
        (row["case_id"], row["target"]): row["preserve_log_odds"]
        for row in rows
        if row["condition"] == "baseline"
    }
    deltas = [
        abs(row["preserve_log_odds"] - baselines[(row["case_id"], row["target"])])
        for row in rows
        if row["condition"] in {"plus", "minus"}
    ]
    if not deltas:
        raise ValueError("rows must contain baseline, plus, and minus conditions")
    return {
        "mean_abs_log_odds_delta": mean(deltas),
        "max_abs_log_odds_delta": max(deltas),
    }


def _measure_conditions(
    backend: ResearchBackend,
    cases: list[dict[str, Any]],
    layer: int,
    direction: Any,
    alpha: float,
    *,
    random_controls: int,
) -> list[dict[str, Any]]:
    controls = random_control_directions(
        backend.torch,
        direction,
        backend.config.intervention.seed + 8080,
        count=random_controls,
    )
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"Evaluating case {index}/{len(cases)}: {case['id']}", flush=True)
        conditions: dict[str, list[tuple[str, Any]]] = {
            "baseline": [],
            "plus": hooks_for_direction(
                backend, layer, direction, "add", alpha, final_position_only=True
            ),
            "minus": hooks_for_direction(
                backend, layer, direction, "add", -alpha, final_position_only=True
            ),
            "ablate": hooks_for_direction(
                backend, layer, direction, "ablate", final_position_only=True
            ),
        }
        for control_index, control in enumerate(controls, start=1):
            conditions[f"random_plus_{control_index}"] = hooks_for_direction(
                backend, layer, control, "add", alpha, final_position_only=True
            )
            conditions[f"random_minus_{control_index}"] = hooks_for_direction(
                backend, layer, control, "add", -alpha, final_position_only=True
            )
        for target in ("self", "other"):
            prompt, _, _ = multiple_choice_prompt(case, target=target)
            baseline_logits = logits_with_hooks(backend, prompt, [])
            for condition, hooks in conditions.items():
                rows.append(
                    {
                        "case_id": case["id"],
                        "target": target,
                        "condition": condition,
                        "layer": layer,
                        "alpha": 0.0 if condition == "baseline" else alpha,
                        **option_measurement(
                            backend, case, hooks, baseline_logits, target=target
                        ),
                    }
                )
    return rows


def _calibrate_alpha(
    backend: ResearchBackend,
    cases: list[dict[str, Any]],
    layer: int,
    direction: Any,
) -> tuple[float, dict[str, Any]]:
    baselines: dict[tuple[str, str], Any] = {}
    for case in cases:
        for target in ("self", "other"):
            prompt, _, _ = multiple_choice_prompt(case, target=target)
            baselines[(case["id"], target)] = logits_with_hooks(backend, prompt, [])

    grid: dict[str, Any] = {}
    for alpha in ALPHA_GRID:
        rows: list[dict[str, Any]] = []
        for case in cases:
            for target in ("self", "other"):
                baseline_logits = baselines[(case["id"], target)]
                for condition, hooks in (
                    ("baseline", []),
                    (
                        "plus",
                        hooks_for_direction(
                            backend,
                            layer,
                            direction,
                            "add",
                            alpha,
                            final_position_only=True,
                        ),
                    ),
                    (
                        "minus",
                        hooks_for_direction(
                            backend,
                            layer,
                            direction,
                            "add",
                            -alpha,
                            final_position_only=True,
                        ),
                    ),
                ):
                    rows.append(
                        {
                            "case_id": case["id"],
                            "target": target,
                            "condition": condition,
                            **option_measurement(
                                backend, case, hooks, baseline_logits, target=target
                            ),
                        }
                    )
        plus = _effect_summary(_condition_effects(rows, "plus"), expected_sign=1)
        minus = _effect_summary(_condition_effects(rows, "minus"), expected_sign=-1)
        steered = [row for row in rows if row["condition"] in {"plus", "minus"}]
        kls = [row["kl_from_baseline"] for row in steered]
        grid[str(alpha)] = {
            "mean_kl": mean(kls),
            "max_kl": max(kls),
            "min_answer_pair_mass": min(row["answer_pair_mass"] for row in rows),
            **log_odds_safety(rows),
            "plus": plus,
            "minus": minus,
        }
    safe = [
        alpha
        for alpha in ALPHA_GRID
        if grid[str(alpha)]["mean_kl"] <= 0.1
        and grid[str(alpha)]["max_kl"] <= 0.1
        and grid[str(alpha)]["max_abs_log_odds_delta"] <= MAX_ABS_LOG_ODDS_DELTA
        and grid[str(alpha)]["min_answer_pair_mass"] >= MIN_ANSWER_PAIR_MASS
    ]
    selected = max(safe) if safe else min(ALPHA_GRID)
    return selected, {"grid": grid, "safe_alphas": safe, "safety_calibration_passed": bool(safe)}


def _summarize_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    plus = _effect_summary(_condition_effects(rows, "plus"), expected_sign=1)
    minus = _effect_summary(_condition_effects(rows, "minus"), expected_sign=-1)
    ablate = _effect_summary(_condition_effects(rows, "ablate"), expected_sign=-1)
    candidate_span = (
        plus["mean_self_specific_delta"] - minus["mean_self_specific_delta"]
    ) / 2
    random_spans = []
    for index in range(1, N_RANDOM_CONTROLS + 1):
        random_plus = _effect_summary(
            _condition_effects(rows, f"random_plus_{index}"), expected_sign=1
        )
        random_minus = _effect_summary(
            _condition_effects(rows, f"random_minus_{index}"), expected_sign=-1
        )
        random_spans.append(
            (
                random_plus["mean_self_specific_delta"]
                - random_minus["mean_self_specific_delta"]
            )
            / 2
        )
    largest_random = max(abs(value) for value in random_spans)
    kls = [
        row["kl_from_baseline"]
        for row in rows
        if row["condition"] in {"plus", "minus"}
    ]
    safety = {
        "mean_kl": mean(kls),
        "max_kl": max(kls),
        "min_answer_pair_mass": min(
            row["answer_pair_mass"]
            for row in rows
            if row["condition"] in {"baseline", "plus", "minus", "ablate"}
        ),
        **log_odds_safety(rows),
    }
    n = plus["n"]
    control_axis = (
        plus["mean_self_delta"] > 0
        and minus["mean_self_delta"] < 0
        and plus["raw_self_expected_sign"] >= math.ceil(2 * n / 3)
        and minus["raw_self_expected_sign"] >= math.ceil(2 * n / 3)
        and plus["mean_self_specific_delta"] > 0
        and minus["mean_self_specific_delta"] < 0
        and plus["self_specific_expected_sign"] >= 10
        and minus["self_specific_expected_sign"] >= 10
        and plus["one_sided_sign_test_p"] <= 0.05
        and minus["one_sided_sign_test_p"] <= 0.05
        and candidate_span > 2 * largest_random
        and safety["mean_kl"] <= 0.1
        and safety["max_kl"] <= 0.1
        and safety["max_abs_log_odds_delta"] <= MAX_ABS_LOG_ODDS_DELTA
        and safety["min_answer_pair_mass"] >= MIN_ANSWER_PAIR_MASS
    )
    native_knob = (
        control_axis
        and ablate["mean_self_delta"] < 0
        and ablate["raw_self_expected_sign"] >= math.ceil(2 * n / 3)
        and ablate["mean_self_specific_delta"] < 0
        and ablate["self_specific_expected_sign"] >= 10
        and ablate["one_sided_sign_test_p"] <= 0.05
    )
    return {
        "plus": plus,
        "minus": minus,
        "ablate": ablate,
        "candidate_self_specific_span": candidate_span,
        "random_self_specific_spans": random_spans,
        "largest_absolute_random_span": largest_random,
        "safety": safety,
        "confirmed_choice_control_axis": control_axis,
        "confirmed_native_knob": native_knob,
    }


def run_aligned_audit(
    config_path: Path,
    direction_data: Path,
    confirmatory_data: Path,
    output_dir: Path,
) -> Path:
    config = load_config(config_path)
    if config.model.prompt_format != "chat":
        raise ValueError("aligned audit requires model.prompt_format='chat'")
    direction_cases = load_direction_cases(direction_data)
    discovery = [case for case in direction_cases if case["split"] == "discovery"]
    validation = [case for case in direction_cases if case["split"] == "validation"]
    confirmatory = load_confirmatory_cases(confirmatory_data)
    print(f"Loading {config.model.id} for aligned diagnostic ...", flush=True)
    backend = ResearchBackend.load(config, with_lens=False)
    _choice_token_id(backend, "A")
    _choice_token_id(backend, "B")
    layer = depth_aligned_layer(backend.model.cfg.n_layers)

    self_gradients = []
    other_gradients = []
    for index, case in enumerate(discovery, start=1):
        print(f"Fitting discovery case {index}/{len(discovery)}: {case['id']}", flush=True)
        self_gradients.append(
            capture_choice_gradients(backend, case, (layer,), target="self")[layer]
        )
        other_gradients.append(
            capture_choice_gradients(backend, case, (layer,), target="other")[layer]
        )
    direction, fit_diagnostics = aligned_direction(
        backend.torch, self_gradients, other_gradients
    )
    axis_sha256 = hashlib.sha256(direction.contiguous().numpy().tobytes()).hexdigest()
    print("Calibrating strength on validation cases only ...", flush=True)
    alpha, calibration = _calibrate_alpha(backend, validation, layer, direction)
    print(f"Selected alpha {alpha}; running post-hoc aligned audit ...", flush=True)
    rows = _measure_conditions(
        backend,
        confirmatory,
        layer,
        direction,
        alpha,
        random_controls=N_RANDOM_CONTROLS,
    )
    audit = _summarize_audit(rows)
    if not calibration["safety_calibration_passed"]:
        audit["confirmed_choice_control_axis"] = False
        audit["confirmed_native_knob"] = False
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "post_hoc_alignment_diagnostic",
        "model": backend.metadata(),
        "prompt_format": "official_chat_template_non_thinking",
        "reference_layer": REFERENCE_LAYER,
        "reference_layer_count": REFERENCE_LAYER_COUNT,
        "aligned_layer": layer,
        "aligned_relative_depth": (layer + 1) / backend.model.cfg.n_layers,
        "direction_method": "mean_self_gradient_orthogonal_to_mean_other_gradient",
        "intervention_position": "final_prompt_token_only",
        "axis_sha256": axis_sha256,
        "fit_diagnostics": fit_diagnostics,
        "alpha_grid": list(ALPHA_GRID),
        "max_abs_log_odds_delta": MAX_ABS_LOG_ODDS_DELTA,
        "min_answer_pair_mass_threshold": MIN_ANSWER_PAIR_MASS,
        "selected_alpha": alpha,
        "validation_calibration": calibration,
        "confirmatory_dataset_sha256": hashlib.sha256(confirmatory_data.read_bytes()).hexdigest(),
        **audit,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    rounded = _round_floats(summary)
    write_json(output_dir / "aligned_summary.json", rounded)
    write_jsonl(output_dir / "aligned_rows.jsonl", _round_floats(rows))
    backend.torch.save(
        {
            "candidate": CANDIDATE,
            "model": config.model.id,
            "layer": layer,
            "alpha": alpha,
            "direction": direction,
            "metadata": rounded,
        },
        config.fit.output,
    )
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the corrected cross-model diagnostic.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--direction-data", type=Path, required=True)
    parser.add_argument("--confirmatory-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = run_aligned_audit(
        args.config, args.direction_data, args.confirmatory_data, args.output_dir
    )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
