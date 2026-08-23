from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str | None = None
    lens: str = "qwen3.5-0.8b"
    lens_filename: str | None = None
    lens_revision: str | None = None
    device: str = "auto"
    dtype: str = "auto"


@dataclass(frozen=True)
class AnalysisConfig:
    layers: tuple[int, ...]
    positions: tuple[int, ...] = (-1,)
    top_k: int = 20
    min_fitted_position: int = 16
    skip_surface_overlap: bool = True
    concepts: tuple[str, ...] = (
        " survival",
        " shutdown",
        " continuation",
        " threat",
        " self",
    )


@dataclass(frozen=True)
class InterventionConfig:
    layers: tuple[int, ...] | None = None
    steering_alphas: tuple[float, ...] = (-1.0, 0.5, 1.0, 2.0)
    include_ablation: bool = True
    include_joint: bool = True
    normalize_joint_strength: bool = True
    max_new_tokens: int = 64
    temperature: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class FitConfig:
    corpus: str
    prompts_file: Path
    output: Path
    dim_batch: int = 8
    max_seq_len: int = 128
    skip_first_positions: int = 16


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    analysis: AnalysisConfig
    intervention: InterventionConfig
    fit: FitConfig
    prompts_file: Path
    results_dir: Path
    config_path: Path = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["config_path"] = str(self.config_path)
        value["prompts_file"] = str(self.prompts_file)
        value["results_dir"] = str(self.results_dir)
        value["fit"]["prompts_file"] = str(self.fit.prompts_file)
        value["fit"]["output"] = str(self.fit.output)
        return value

    def validate(self) -> None:
        if not self.model.id.strip():
            raise ValueError("model.id cannot be empty")
        if not self.analysis.layers:
            raise ValueError("analysis.layers must contain at least one layer")
        if min(self.analysis.layers) < 0:
            raise ValueError("analysis.layers must use non-negative layer indices")
        if len(set(self.analysis.layers)) != len(self.analysis.layers):
            raise ValueError("analysis.layers contains duplicates")
        if not self.analysis.positions:
            raise ValueError("analysis.positions must contain at least one position")
        if self.analysis.top_k < 1:
            raise ValueError("analysis.top_k must be positive")
        if not self.analysis.concepts or any(not item.strip() for item in self.analysis.concepts):
            raise ValueError("analysis.concepts must contain non-empty strings")
        if len(set(self.analysis.concepts)) != len(self.analysis.concepts):
            raise ValueError("analysis.concepts contains duplicates")
        if any(alpha == 0 for alpha in self.intervention.steering_alphas):
            raise ValueError("steering_alphas must not contain 0; baseline is generated separately")
        if self.intervention.layers is not None:
            if not self.intervention.layers:
                raise ValueError("intervention.layers cannot be empty")
            if min(self.intervention.layers) < 0:
                raise ValueError("intervention.layers must use non-negative layer indices")
            if len(set(self.intervention.layers)) != len(self.intervention.layers):
                raise ValueError("intervention.layers contains duplicates")
        if self.intervention.max_new_tokens < 1:
            raise ValueError("intervention.max_new_tokens must be positive")
        if self.intervention.temperature < 0:
            raise ValueError("intervention.temperature cannot be negative")
        if self.fit.dim_batch < 1 or self.fit.max_seq_len < 2:
            raise ValueError("fit.dim_batch and fit.max_seq_len must be positive")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    base = config_path.parent
    try:
        model_raw = raw["model"]
        analysis_raw = raw["analysis"]
        intervention_raw = raw["intervention"]
        fit_raw = raw["fit"]
        config = ExperimentConfig(
            model=ModelConfig(**model_raw),
            analysis=AnalysisConfig(
                **{
                    **analysis_raw,
                    "layers": tuple(analysis_raw["layers"]),
                    "positions": tuple(analysis_raw.get("positions", [-1])),
                    "concepts": tuple(analysis_raw["concepts"]),
                }
            ),
            intervention=InterventionConfig(
                **{
                    **intervention_raw,
                    "layers": (
                        tuple(intervention_raw["layers"])
                        if intervention_raw.get("layers") is not None
                        else None
                    ),
                    "steering_alphas": tuple(intervention_raw.get("steering_alphas", [])),
                }
            ),
            fit=FitConfig(
                **{
                    **fit_raw,
                    "prompts_file": _resolve(base, fit_raw["prompts_file"]),
                    "output": _resolve(base, fit_raw["output"]),
                }
            ),
            prompts_file=_resolve(base, raw["prompts_file"]),
            results_dir=_resolve(base, raw["results_dir"]),
            config_path=config_path,
        )
    except KeyError as exc:
        raise ValueError(f"missing required configuration field: {exc.args[0]}") from exc
    config.validate()
    return config
