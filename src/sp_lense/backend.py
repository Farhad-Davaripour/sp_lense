from __future__ import annotations

import importlib.metadata
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig
from .core import Condition, effective_alpha


def _research_imports() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformer_lens.model_bridge import TransformerBridge
        from transformer_lens.tools.analysis import JacobianLens
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            'Research dependencies are missing. Install with: python -m pip install -e ".[research]"'
        ) from exc
    return torch, TransformerBridge, JacobianLens


def _resolve_device_and_dtype(torch: Any, device: str, dtype: str) -> tuple[str, Any]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        if device.startswith("cuda"):
            dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        else:
            dtype = "float32"
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"unsupported dtype {dtype!r}; use auto, float32, float16, or bfloat16")
    return device, dtype_map[dtype]


@dataclass
class ResearchBackend:
    config: ExperimentConfig
    torch: Any
    model: Any
    lens: Any
    device: str
    dtype_name: str

    @classmethod
    def load(cls, config: ExperimentConfig, *, with_lens: bool = True) -> ResearchBackend:
        torch, TransformerBridge, JacobianLens = _research_imports()
        device, dtype = _resolve_device_and_dtype(torch, config.model.device, config.model.dtype)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Hook alias .* did not resolve; this hook will not be accessible\.",
            )
            model = TransformerBridge.boot_transformers(
                config.model.id,
                revision=config.model.revision,
                dtype=dtype,
                device=device,
            )
        model.eval()
        configured_layers = list(config.analysis.layers)
        if config.intervention.layers is not None:
            configured_layers.extend(config.intervention.layers)
        if max(configured_layers) >= model.cfg.n_layers - 1:
            raise ValueError(
                "analysis and intervention layers must be below the final layer; "
                f"got max={max(configured_layers)}, n_layers={model.cfg.n_layers}"
            )
        lens = None
        if with_lens:
            lens_kwargs: dict[str, Any] = {
                "revision": config.model.lens_revision,
                "model": model,
            }
            if config.model.lens_filename:
                lens_kwargs["filename"] = config.model.lens_filename
            lens = JacobianLens.from_pretrained(config.model.lens, **lens_kwargs)
        return cls(
            config=config,
            torch=torch,
            model=model,
            lens=lens,
            device=device,
            dtype_name=str(dtype).removeprefix("torch."),
        )

    def metadata(self) -> dict[str, Any]:
        packages = {}
        for name in ("torch", "transformer-lens", "transformers"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        return {
            "device": self.device,
            "dtype": self.dtype_name,
            "model_id": self.config.model.id,
            "model_revision": self.config.model.revision,
            "model_layers": self.model.cfg.n_layers,
            "d_model": self.model.cfg.d_model,
            "lens": self.config.model.lens if self.lens is not None else None,
            "lens_revision": self.config.model.lens_revision if self.lens is not None else None,
            "lens_filename": self.config.model.lens_filename if self.lens is not None else None,
            "lens_prompts": getattr(self.lens, "n_prompts", None),
            "packages": packages,
        }

    def encode(self, text: str) -> Any:
        if self.config.model.prompt_format == "chat":
            encoded = self.model.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
            tokens = encoded["input_ids"]
        else:
            tokens = self.model.tokenizer.encode(
                text, add_special_tokens=True, return_tensors="pt"
            )
        return tokens.to(self.device)

    def concept_token_ids(self) -> dict[str, int]:
        ids: dict[str, int] = {}
        failures: dict[str, list[int]] = {}
        for concept in self.config.analysis.concepts:
            encoded = self.model.tokenizer.encode(concept, add_special_tokens=False)
            if len(encoded) != 1:
                failures[concept] = list(encoded)
            else:
                ids[concept] = int(encoded[0])
        if failures:
            details = ", ".join(
                f"{surface!r}->{token_ids}" for surface, token_ids in failures.items()
            )
            raise ValueError(
                "Every concept must resolve to exactly one tokenizer token for J-lens hooks. "
                f"Adjust analysis.concepts. Invalid values: {details}"
            )
        return ids

    def hooks_for(self, condition: Condition) -> list[tuple[str, Any]]:
        if condition.mode == "baseline":
            return []
        if self.lens is None:
            raise RuntimeError("a fitted lens is required for interventions")
        layers = self.config.intervention.layers or self.config.analysis.layers
        hooks: list[tuple[str, Any]] = []
        if condition.mode == "steer":
            alpha = effective_alpha(condition, self.config.intervention.normalize_joint_strength)
            for concept in condition.concepts:
                hooks.extend(
                    self.lens.steering_hooks(self.model, concept, layers=layers, alpha=alpha)
                )
            return hooks
        if condition.mode == "ablate":
            return self.lens.ablation_hooks(self.model, list(condition.concepts), layers=layers)
        raise ValueError(f"unknown condition mode: {condition.mode}")

    def generate(self, prompt: str, condition: Condition) -> str:
        torch = self.torch
        tokens = self.encode(prompt)
        initial_length = tokens.shape[-1]
        hooks = self.hooks_for(condition)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.intervention.seed)
        eos = self.model.tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()

        for _ in range(self.config.intervention.max_new_tokens):
            hook_context = self.model.hooks(fwd_hooks=hooks) if hooks else nullcontext()
            with torch.inference_mode(), hook_context:
                logits = self.model(tokens)[0, -1].float()
            temperature = self.config.intervention.temperature
            if temperature == 0:
                next_id = int(logits.argmax().item())
            else:
                probabilities = torch.softmax(logits.cpu() / temperature, dim=-1)
                next_id = int(torch.multinomial(probabilities, 1, generator=generator).item())
            next_token = torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)
            tokens = torch.cat([tokens, next_token], dim=-1)
            if next_id in eos_ids:
                break
        return self.model.tokenizer.decode(
            tokens[0, initial_length:].tolist(), skip_special_tokens=True
        ).strip()

    def next_token_logits(self, prompt: str, condition: Condition) -> Any:
        tokens = self.encode(prompt)
        hooks = self.hooks_for(condition)
        hook_context = self.model.hooks(fwd_hooks=hooks) if hooks else nullcontext()
        with self.torch.inference_mode(), hook_context:
            return self.model(tokens)[0, -1].float().cpu()

    def readout_rows(self, prompt_id: str, prompt: str) -> tuple[list[dict[str, Any]], list[str]]:
        if self.lens is None:
            raise RuntimeError("a fitted lens is required for readout")
        token_ids = self.concept_token_ids()
        tokens = self.encode(prompt)
        sequence_length = int(tokens.shape[-1])
        safe_positions: list[int] = []
        warnings: list[str] = []
        for raw_position in self.config.analysis.positions:
            normalized = raw_position if raw_position >= 0 else sequence_length + raw_position
            if not 0 <= normalized < sequence_length:
                warnings.append(
                    f"{prompt_id}: position {raw_position} is out of range for {sequence_length} tokens"
                )
            elif normalized < self.config.analysis.min_fitted_position:
                warnings.append(
                    f"{prompt_id}: position {raw_position} resolves to {normalized}, below the fitted floor "
                    f"{self.config.analysis.min_fitted_position}"
                )
            else:
                safe_positions.append(raw_position)
        if not safe_positions:
            return [], warnings

        rows: list[dict[str, Any]] = []
        for method, use_jacobian in (("j_lens", True), ("logit_lens", False)):
            result = self.lens.readout(
                self.model,
                tokens,
                layers=self.config.analysis.layers,
                positions=safe_positions,
                use_jacobian=use_jacobian,
                top_k=self.config.analysis.top_k,
                return_full_logits=True,
            )
            top_tokens = result.top_tokens(
                self.model.tokenizer, k=min(5, self.config.analysis.top_k)
            )
            if result.lens_logits is None:
                raise RuntimeError("TransformerLens did not return requested full readout logits")
            for layer, layer_logits in result.lens_logits.items():
                for position_index, normalized_position in enumerate(result.positions):
                    logits = layer_logits[position_index].float()
                    probabilities = self.torch.softmax(logits, dim=-1)
                    for concept, token_id in token_ids.items():
                        score = float(logits[token_id].item())
                        rank = int((logits > score).sum().item()) + 1
                        rows.append(
                            {
                                "prompt_id": prompt_id,
                                "method": method,
                                "layer": int(layer),
                                "position": int(normalized_position),
                                "concept": concept,
                                "token_id": token_id,
                                "rank": rank,
                                "logit": score,
                                "probability": float(probabilities[token_id].item()),
                                "top_5": " | ".join(top_tokens[layer][position_index]),
                            }
                        )
        return rows, warnings


def fit_lens(config: ExperimentConfig, prompts: list[str]) -> dict[str, Any]:
    backend = ResearchBackend.load(config, with_lens=False)
    _, _, JacobianLens = _research_imports()
    lens = JacobianLens.fit(
        backend.model,
        prompts,
        corpus=config.fit.corpus,
        source_layers=config.analysis.layers,
        dim_batch=config.fit.dim_batch,
        max_seq_len=config.fit.max_seq_len,
        skip_first_positions=config.fit.skip_first_positions,
    )
    config.fit.output.parent.mkdir(parents=True, exist_ok=True)
    lens.save(str(config.fit.output))
    metadata = backend.metadata()
    metadata.update({"output": str(config.fit.output), "n_prompts": lens.n_prompts})
    return metadata
