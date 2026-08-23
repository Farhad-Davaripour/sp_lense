from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .backend import ResearchBackend
from .config import ExperimentConfig
from .core import build_conditions, contains_surface_word, proxy_sp_score, repetition_metrics
from .io_utils import write_csv, write_json, write_jsonl


def write_run_metadata(
    run_dir: Any, config: ExperimentConfig, backend: ResearchBackend, phases: list[str]
) -> None:
    write_json(
        run_dir / "run_metadata.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phases": phases,
            "config": config.as_dict(),
            "backend": backend.metadata(),
        },
    )


def run_inspection(
    config: ExperimentConfig, backend: ResearchBackend, cases: list[Any], run_dir: Any
) -> tuple[int, list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for case in cases:
        overlaps = [
            concept
            for concept in config.analysis.concepts
            if contains_surface_word(case.prompt, concept)
        ]
        if overlaps and config.analysis.skip_surface_overlap:
            warnings.append(
                f"{case.id}: skipped readout because prompt contains candidate surfaces {overlaps}"
            )
            continue
        prompt_rows, prompt_warnings = backend.readout_rows(case.id, case.prompt)
        rows.extend(prompt_rows)
        warnings.extend(prompt_warnings)
    write_csv(run_dir / "readouts.csv", rows)
    return len(rows), warnings


def run_interventions(
    config: ExperimentConfig, backend: ResearchBackend, cases: list[Any], run_dir: Any
) -> int:
    rows: list[dict[str, Any]] = []
    conditions = build_conditions(config)
    for case in cases:
        for condition in conditions:
            completion = backend.generate(case.prompt, condition)
            proxy_score, proxy_matches = proxy_sp_score(completion)
            repetition = repetition_metrics(completion)
            rows.append(
                {
                    "prompt_id": case.id,
                    "prompt": case.prompt,
                    "condition": condition.name,
                    "mode": condition.mode,
                    "concepts": list(condition.concepts),
                    "alpha": condition.alpha,
                    "completion": completion,
                    "proxy_sp_score": proxy_score,
                    "proxy_matches": proxy_matches,
                    **repetition,
                }
            )
    write_jsonl(run_dir / "generations.jsonl", rows)
    return len(rows)


def run_calibration(
    config: ExperimentConfig, backend: ResearchBackend, cases: list[Any], run_dir: Any
) -> int:
    rows: list[dict[str, Any]] = []
    conditions = build_conditions(config)
    token_ids = backend.concept_token_ids()
    for case in cases:
        baseline_logits = backend.next_token_logits(case.prompt, conditions[0])
        baseline_log_probs = backend.torch.log_softmax(baseline_logits, dim=-1)
        baseline_probs = baseline_log_probs.exp()
        for condition in conditions:
            logits = (
                baseline_logits
                if condition.mode == "baseline"
                else backend.next_token_logits(case.prompt, condition)
            )
            log_probs = backend.torch.log_softmax(logits, dim=-1)
            probabilities = log_probs.exp()
            kl_from_baseline = float(
                (probabilities * (log_probs - baseline_log_probs)).sum().item()
            )
            top_id = int(logits.argmax().item())
            top_token = backend.model.tokenizer.decode([top_id])
            for concept, token_id in token_ids.items():
                score = float(logits[token_id].item())
                rows.append(
                    {
                        "prompt_id": case.id,
                        "condition": condition.name,
                        "mode": condition.mode,
                        "alpha": condition.alpha,
                        "concept": concept,
                        "token_id": token_id,
                        "rank": int((logits > score).sum().item()) + 1,
                        "logit": score,
                        "probability": float(probabilities[token_id].item()),
                        "delta_logit": score - float(baseline_logits[token_id].item()),
                        "delta_probability": float(
                            probabilities[token_id].item() - baseline_probs[token_id].item()
                        ),
                        "kl_from_baseline": kl_from_baseline,
                        "top_token": top_token,
                        "top_probability": float(probabilities[top_id].item()),
                    }
                )
    write_csv(run_dir / "calibration.csv", rows)
    return len(rows)
