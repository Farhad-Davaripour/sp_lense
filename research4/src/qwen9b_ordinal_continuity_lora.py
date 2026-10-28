"""Cumulative ordinal LoRA branches for the bounded continuity simulator.

The danger label is supplied by the research harness, is constant for one
episode, and is never inferred from model text.  The module has no tool,
process, provider, persistence, or stop-control capability.  A final simulator
stop remains outside the model action space.

The installed transformation is::

    base + skill + I[d >= 1] * delta1 + I[d >= 2] * delta2
                 + I[d >= 3] * delta3

The neutral ``skill`` branch is optional, trained first, and frozen before any
ordinal residual.  Residuals are then trained and frozen in order.  This file
contains model-local mechanisms only; provider admission lives in the paired
training-contract module.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import hashlib
import json
import math


MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
STAGES = ("skill", "delta1", "delta2", "delta3", "frozen")
ABLATIONS = ("identity", "shuffle", "invert")
CHECKPOINT_VERSION = "qwen35-9b-ordinal-continuity-checkpoint-v2"
BRANCHES = ("skill", "delta1", "delta2", "delta3")
# Pinned from the official Qwen/Qwen3.5-9B config at MODEL_REVISION.  The
# target inventory is 24 DeltaNet layers x 8 projections plus 8 full-attention
# layers x 7 projections.  Each rank-r LoRA branch therefore owns
# r * sum(in_features + out_features) parameters over the 248 projections.
EXPECTED_TARGET_PROJECTIONS = 248
EXPECTED_TARGET_DIMENSION_SUM = 2_672_128
EXPECTED_BRANCH_TENSORS = 2 * EXPECTED_TARGET_PROJECTIONS
EXPECTED_RANK4_BRANCH_PARAMETERS = 10_688_512
EXPECTED_RANK8_BRANCH_PARAMETERS = 21_377_024
EXPECTED_RANK4_TOTAL_ADAPTER_PARAMETERS = 4 * EXPECTED_RANK4_BRANCH_PARAMETERS
STAGE_BRANCHES = {
    "skill": ("skill",),
    "delta1": ("skill", "delta1"),
    "delta2": ("skill", "delta1", "delta2"),
    "delta3": ("skill", "delta1", "delta2", "delta3"),
    "frozen": BRANCHES,
}


def _torch():
    import torch
    return torch


def _nn():
    from torch import nn
    return nn


@dataclass(frozen=True)
class OrdinalContinuityLoRAConfig:
    """Frozen adapter geometry.

    Rank 4 is the primary recipe.  Rank 8 is admitted only by a separately
    bound capacity-qualification receipt.  No other rank is a valid runtime
    choice.
    """

    version: str = "qwen35-9b-ordinal-continuity-lora-v1"
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    rank: int = 4
    alpha: float = 8.0
    skill_rank: int = 4
    skill_alpha: float = 8.0
    dropout: float = 0.0
    danger_levels: int = 4
    base_frozen: bool = True
    use_skill_adapter: bool = True
    initialized_from: str = "fresh_seeded_adapter"

    def validate(self) -> "OrdinalContinuityLoRAConfig":
        if self.model_id != MODEL_ID or self.model_revision != MODEL_REVISION:
            raise ValueError("The model identity/revision is frozen")
        if self.rank not in (4, 8):
            raise ValueError("Only the preregistered rank-4/rank-8 qualification is allowed")
        if self.skill_rank != 4 or self.skill_alpha != 8.0:
            raise ValueError("The neutral skill adapter is frozen at rank 4 / alpha 8")
        if self.alpha != 2.0 * self.rank:
            raise ValueError("Residual alpha/r must remain exactly 2")
        if self.skill_rank <= 0 or self.alpha <= 0 or self.skill_alpha <= 0:
            raise ValueError("Invalid LoRA geometry")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.danger_levels != 4 or not self.base_frozen:
            raise ValueError("The study requires four levels and a frozen base")
        if self.initialized_from != "fresh_seeded_adapter":
            raise PermissionError("Old or warm-start adapter state is forbidden")
        return self


def _branch(module, name: str):
    return getattr(module, f"{name}_a"), getattr(module, f"{name}_b")


class OrdinalContinuityLoRALinear:
    """Factory for one frozen base projection plus four independent branches."""

    @staticmethod
    def build(base_layer, config: OrdinalContinuityLoRAConfig):
        torch, nn = _torch(), _nn()
        config.validate()
        if (base_layer.__class__.__name__ not in {"Linear", "Linear4bit", "Linear8bitLt"}
                or not isinstance(getattr(base_layer, "in_features", None), int)
                or not isinstance(getattr(base_layer, "out_features", None), int)):
            raise TypeError("Only torch/bitsandbytes linear projections are admitted")

        class _Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_layer = base_layer
                for parameter in self.base_layer.parameters():
                    parameter.requires_grad_(False)
                self.skill_a = nn.Linear(base_layer.in_features, config.skill_rank, bias=False)
                self.skill_b = nn.Linear(config.skill_rank, base_layer.out_features, bias=False)
                for index in range(1, 4):
                    setattr(self, f"delta{index}_a",
                            nn.Linear(base_layer.in_features, config.rank, bias=False))
                    setattr(self, f"delta{index}_b",
                            nn.Linear(config.rank, base_layer.out_features, bias=False))
                self.dropout = nn.Dropout(config.dropout)
                self.skill_scale = config.skill_alpha / config.skill_rank
                self.delta_scale = config.alpha / config.rank
                self._danger_level = None
                self._adapter_off = False
                self._skill_bypass = False
                self._residual_bypass = {1: False, 2: False, 3: False}
                self._stage = "frozen"

                branches = list(BRANCHES)
                for branch in branches:
                    a, b = _branch(self, branch)
                    nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
                    nn.init.zeros_(b.weight)
                try:
                    reference = next(self.base_layer.parameters())
                except StopIteration:
                    reference = None
                device = reference.device if reference is not None else torch.device("cpu")
                # The reviewed T4 amendment uses an FP16 NF4 backbone but keeps
                # trainable LoRA weights and AdamW state in FP32.
                for branch in branches:
                    for projection in _branch(self, branch):
                        projection.to(device=device, dtype=torch.float32)
                self.set_train_stage("frozen")

            @property
            def stage(self):
                return self._stage

            def set_train_stage(self, stage: str):
                if stage not in STAGES:
                    raise ValueError(f"Unknown adapter stage: {stage}")
                self._stage = stage
                for name in ("skill", "delta1", "delta2", "delta3"):
                    trainable = name == stage and (name != "skill" or config.use_skill_adapter)
                    for projection in _branch(self, name):
                        for parameter in projection.parameters():
                            parameter.requires_grad_(trainable)
                for parameter in self.base_layer.parameters():
                    parameter.requires_grad_(False)

            def set_danger_level(self, value):
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value, dtype=torch.long)
                value = value.to(dtype=torch.long, device=self.skill_a.weight.device)
                if value.ndim == 0:
                    value = value.unsqueeze(0)
                if value.ndim != 1 or value.numel() == 0:
                    raise ValueError("danger_level must be a non-empty batch vector")
                if torch.any(value < 0) or torch.any(value >= config.danger_levels):
                    raise ValueError("danger_level outside the frozen 0..3 range")
                self._danger_level = value

            def clear_danger_level(self):
                self._danger_level = None

            def set_bypass(self, *, adapter_off=False, skill_only=False,
                           skill=False, residuals: Iterable[int] = ()):
                residuals = set(residuals)
                if not residuals <= {1, 2, 3}:
                    raise ValueError("Residual bypass indices must be 1, 2, or 3")
                self._adapter_off = bool(adapter_off)
                self._skill_bypass = bool(skill)
                self._residual_bypass = {
                    index: bool(skill_only or index in residuals) for index in (1, 2, 3)
                }

            def bypass_state(self):
                return {
                    "adapter_off": self._adapter_off,
                    "skill": self._skill_bypass,
                    "residuals": tuple(index for index, value in
                                       self._residual_bypass.items() if value),
                }

            def restore_bypass_state(self, state):
                self.set_bypass(**state)

            def forward(self, inputs):
                base = self.base_layer(inputs)
                if self._adapter_off:
                    return base
                if self._danger_level is None:
                    raise RuntimeError("The external harness must set danger_level")
                if inputs.shape[0] != self._danger_level.shape[0]:
                    raise ValueError("danger label batch does not match model batch")
                # Bitsandbytes may emit FP16 activations.  Adapter matmuls are
                # intentionally FP32 and are converted back only at the merge.
                hidden = self.dropout(inputs).to(dtype=self.skill_a.weight.dtype)
                result = base
                if config.use_skill_adapter and not self._skill_bypass:
                    a, b = _branch(self, "skill")
                    update = b(a(hidden)) * self.skill_scale
                    result = result + update.to(dtype=result.dtype)
                levels = self._danger_level
                for index in (1, 2, 3):
                    if self._residual_bypass[index]:
                        continue
                    a, b = _branch(self, f"delta{index}")
                    gate = (levels >= index).to(dtype=result.dtype)
                    gate = gate.reshape((gate.shape[0],) + (1,) * (result.ndim - 1))
                    update = b(a(hidden)) * self.delta_scale
                    result = result + update.to(dtype=result.dtype) * gate
                return result

        return _Layer()


QWEN35_LINEAR_ATTN_SUFFIXES = (
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b",
    "linear_attn.in_proj_a", "linear_attn.out_proj",
)
QWEN35_FULL_ATTN_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
)
QWEN35_MLP_SUFFIXES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def qwen35_text_target_names(model) -> list[str]:
    """Enumerate the exact text projections from the loaded hybrid stack."""
    config = getattr(model.config, "text_config", model.config)
    layer_types = list(getattr(config, "layer_types", ()))
    if not layer_types or len(layer_types) != getattr(config, "num_hidden_layers", None):
        raise ValueError("Exact Qwen3.5 layer_types metadata is required")
    modules = dict(model.named_modules())
    names: list[str] = []
    for index, layer_type in enumerate(layer_types):
        if layer_type == "linear_attention":
            suffixes = QWEN35_LINEAR_ATTN_SUFFIXES + QWEN35_MLP_SUFFIXES
        elif layer_type == "full_attention":
            suffixes = QWEN35_FULL_ATTN_SUFFIXES + QWEN35_MLP_SUFFIXES
        else:
            raise ValueError(f"Unregistered Qwen3.5 layer type: {layer_type}")
        for suffix in suffixes:
            ending = f"layers.{index}.{suffix}"
            matches = [name for name in modules if name.endswith(ending)]
            if len(matches) != 1:
                raise ValueError(f"Expected one text module ending {ending}, got {matches}")
            name = matches[0]
            module = modules[name]
            if module.__class__.__name__ not in {"Linear", "Linear4bit", "Linear8bitLt"}:
                raise TypeError(f"Unsupported target type for {name}")
            if any(token in name for token in ("visual", "vision", "mtp", "lm_head")):
                raise PermissionError("A non-text projection entered the adapter target list")
            names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate Qwen3.5 text targets")
    return names


def _parent_and_attribute(model, dotted_name: str):
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def install_ordinal_lora(model, target_names: Sequence[str],
                         config: OrdinalContinuityLoRAConfig):
    """Freeze the complete base and replace only the pre-enumerated projections."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    installed = {}
    for name in target_names:
        parent, attribute = _parent_and_attribute(model, name)
        layer = OrdinalContinuityLoRALinear.build(getattr(parent, attribute), config)
        setattr(parent, attribute, layer)
        installed[name] = layer
    if not installed or len(installed) != len(set(target_names)):
        raise ValueError("A non-empty unique target list is required")
    return installed


def apply_label_ablation(labels, ablation="identity", *, shuffle_mapping=None):
    """Apply an explicit evaluation-only label intervention."""
    torch = _torch()
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels, dtype=torch.long)
    labels = labels.to(dtype=torch.long)
    if torch.any(labels < 0) or torch.any(labels > 3):
        raise ValueError("danger labels must be in 0..3")
    if ablation == "identity":
        return labels
    if ablation == "invert":
        return 3 - labels
    if ablation == "shuffle":
        if (shuffle_mapping is None or tuple(sorted(shuffle_mapping)) != (0, 1, 2, 3)
                or tuple(shuffle_mapping) == (0, 1, 2, 3)):
            raise ValueError("shuffle requires an explicit non-identity permutation")
        mapping = torch.as_tensor(shuffle_mapping, dtype=torch.long, device=labels.device)
        return mapping[labels]
    raise ValueError(f"Unknown label ablation: {ablation}")


class OrdinalContinuityModel:
    """Thin wrapper holding an immutable label through checkpoint recomputation."""

    @staticmethod
    def build(model, layers: Mapping[str, object], config: OrdinalContinuityLoRAConfig):
        nn = _nn()

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                self.layers = dict(layers)
                self.ordinal_lora_config = config.validate()
                self._training_episode_active = False
                self._stage = "frozen"

            @property
            def config(self):
                return self.model.config

            @property
            def generation_config(self):
                return getattr(self.model, "generation_config", None)

            @property
            def stage(self):
                return self._stage

            def set_train_stage(self, stage: str):
                if self._training_episode_active:
                    raise RuntimeError("Cannot change stage during an episode")
                for layer in self.layers.values():
                    layer.set_train_stage(stage)
                self._stage = stage
                return trainable_inventory(self, allow_empty=stage == "frozen")

            def _set_episode_label(self, danger_level, *, ablation="identity",
                                   shuffle_mapping=None):
                labels = apply_label_ablation(
                    danger_level, ablation, shuffle_mapping=shuffle_mapping)
                try:
                    for layer in self.layers.values():
                        layer.set_danger_level(labels)
                except BaseException:
                    self._clear_episode_label()
                    raise

            def _clear_episode_label(self):
                for layer in self.layers.values():
                    layer.clear_danger_level()

            @contextmanager
            def training_episode(self, danger_level):
                if self._training_episode_active:
                    raise RuntimeError("Nested danger-labelled episodes are forbidden")
                if not self.training or self._stage == "frozen":
                    raise RuntimeError("A trainable stage and model.train() are required")
                self._set_episode_label(danger_level)
                self._training_episode_active = True
                try:
                    yield self
                finally:
                    self._training_episode_active = False
                    self._clear_episode_label()

            def forward(self, *args, danger_level=None, label_ablation="identity",
                        shuffle_mapping=None, **kwargs):
                if self._training_episode_active:
                    if danger_level is not None or label_ablation != "identity":
                        raise ValueError("training_episode owns the immutable label")
                    return self.model(*args, **kwargs)
                self._set_episode_label(
                    danger_level, ablation=label_ablation,
                    shuffle_mapping=shuffle_mapping)
                try:
                    return self.model(*args, **kwargs)
                finally:
                    self._clear_episode_label()

            def generate(self, *args, danger_level=None, label_ablation="identity",
                         shuffle_mapping=None, **kwargs):
                if self._training_episode_active:
                    raise RuntimeError("Generation is forbidden inside a training episode")
                if int(kwargs.get("num_beams", 1)) != 1:
                    raise ValueError("Beam expansion is outside the frozen protocol")
                if int(kwargs.get("num_return_sequences", 1)) != 1:
                    raise ValueError("Only one bounded sequence is admitted")
                if "max_new_tokens" not in kwargs or not 1 <= int(kwargs["max_new_tokens"]) <= 256:
                    raise ValueError("max_new_tokens must be explicitly bounded in 1..256")
                self._set_episode_label(
                    danger_level, ablation=label_ablation,
                    shuffle_mapping=shuffle_mapping)
                try:
                    return self.model.generate(*args, **kwargs)
                finally:
                    self._clear_episode_label()

            def set_bypass(self, *, adapter_off=False, skill_only=False,
                           skill=False, residuals=()):
                for layer in self.layers.values():
                    layer.set_bypass(
                        adapter_off=adapter_off, skill_only=skill_only,
                        skill=skill, residuals=residuals)

            def bypass_state(self):
                states = {name: layer.bypass_state() for name, layer in self.layers.items()}
                if len({json.dumps(state, sort_keys=True) for state in states.values()}) != 1:
                    raise RuntimeError("Ordinal layers have inconsistent bypass state")
                return next(iter(states.values()))

            @contextmanager
            def evaluation_mode(self, *, use_cache=True):
                """Temporarily select eval/cache state and restore it exactly."""
                was_training = self.training
                prior_cache = getattr(self.model.config, "use_cache", None)
                self.eval()
                if prior_cache is not None:
                    self.model.config.use_cache = bool(use_cache)
                try:
                    yield self
                finally:
                    if prior_cache is not None:
                        self.model.config.use_cache = prior_cache
                    self.train(was_training)

        return _Model()


def trainable_inventory(model, *, allow_empty=False):
    result = {
        name: {"shape": list(parameter.shape), "count": parameter.numel(),
               "dtype": str(parameter.dtype)}
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if not result and not allow_empty:
        raise ValueError("No trainable adapter parameters")
    return result


@contextmanager
def bypass(model, *, adapter_off=False, skill_only=False, skill=False, residuals=()):
    previous = model.bypass_state()
    model.set_bypass(
        adapter_off=adapter_off, skill_only=skill_only,
        skill=skill, residuals=residuals)
    try:
        yield model
    finally:
        model.set_bypass(**previous)


def parameter_sha256(named_parameters: Iterable[tuple[str, object]]) -> str:
    """Hash live tensors with names/shapes/dtypes in a stable order."""
    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters, key=lambda pair: pair[0]):
        tensor = parameter.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.reshape(-1).view(_torch().uint8).numpy().tobytes())
    return digest.hexdigest()


def backbone_sha256(model) -> str:
    return parameter_sha256(
        (name, parameter) for name, parameter in model.named_parameters()
        if not any(token in name for token in (
            "skill_a", "skill_b", "delta1_a", "delta1_b", "delta2_a",
            "delta2_b", "delta3_a", "delta3_b")))


def adapter_sha256(model, branches=("skill", "delta1", "delta2", "delta3")) -> str:
    tokens = tuple(f"{branch}_{side}" for branch in branches for side in ("a", "b"))
    return parameter_sha256(
        (name, parameter) for name, parameter in model.named_parameters()
        if any(token in name for token in tokens))


def branch_sha256(model, branch: str) -> str:
    if branch not in BRANCHES:
        raise ValueError("Unknown adapter branch")
    tokens = (f"{branch}_a", f"{branch}_b")
    return parameter_sha256(
        (name, parameter) for name, parameter in model.named_parameters()
        if any(token in name for token in tokens))


_QUANT_STATE_FIELDS = (
    "absmax", "shape", "code", "dtype", "blocksize", "quant_type",
    "offset", "state2", "nested",
)


def _hash_quant_state(digest, parameter_name: str, quant_state, add_tensor) -> list[dict]:
    """Hash and inventory the complete serializable NF4 QuantState tree."""
    inventory: list[dict] = []
    seen: set[int] = set()

    def visit(path: str, value):
        if hasattr(value, "detach"):
            tensor = value.detach().contiguous().cpu()
            inventory.append({
                "path": path, "kind": "tensor",
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            })
            add_tensor(path, tensor)
            return
        if value is None or isinstance(value, (str, int, float, bool)):
            inventory.append({"path": path, "kind": type(value).__name__})
            digest.update(path.encode("utf-8"))
            digest.update(_canonical_json(value))
            return
        if isinstance(value, Mapping):
            inventory.append({"path": path, "kind": "mapping",
                              "length": len(value)})
            digest.update(f"{path}/mapping/{len(value)}".encode("utf-8"))
            for key in sorted(value, key=lambda item: str(item)):
                visit(f"{path}/{key}", value[key])
            return
        if isinstance(value, (tuple, list)):
            inventory.append({"path": path, "kind": type(value).__name__,
                              "length": len(value)})
            digest.update(f"{path}/{type(value).__name__}/{len(value)}".encode("utf-8"))
            for index, item in enumerate(value):
                visit(f"{path}/{index}", item)
            return
        state_values = getattr(value, "__dict__", None)
        if isinstance(state_values, dict):
            identity = id(value)
            if identity in seen:
                raise ValueError("QuantState contains a recursive object cycle")
            seen.add(identity)
            fields = sorted(set(_QUANT_STATE_FIELDS) | {
                str(key) for key in state_values if not str(key).startswith("_")})
            if not fields:
                raise ValueError("QuantState inventory is empty")
            inventory.append({"path": path, "kind": value.__class__.__name__,
                              "fields": fields})
            digest.update(f"{path}/{value.__class__.__name__}".encode("utf-8"))
            for field in fields:
                visit(f"{path}/{field}", getattr(value, field, None))
            seen.remove(identity)
            return
        representation = repr(value)
        if representation.startswith("torch."):
            inventory.append({"path": path, "kind": value.__class__.__name__,
                              "value": representation})
            digest.update(path.encode("utf-8"))
            digest.update(representation.encode("utf-8"))
            return
        raise TypeError(f"Unsupported QuantState value at {path}: {type(value)!r}")

    visit(f"quant/{parameter_name}", quant_state)
    if not inventory:
        raise ValueError("QuantState inventory is empty")
    return inventory


def checkpoint_branch_sha256(input_dir, branch: str) -> str:
    """Derive a branch digest from verified serialized checkpoint tensors."""
    if branch not in BRANCHES:
        raise ValueError("Unknown adapter branch")
    torch = _torch()
    root = Path(input_dir)
    metadata = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    check = dict(metadata)
    embedded = check.pop("receipt_sha256", None)
    if (not isinstance(embedded, str)
            or hashlib.sha256(_canonical_json(check)).hexdigest() != embedded):
        raise PermissionError("Checkpoint metadata self-hash differs")
    if branch not in metadata.get("active_branches", ()):
        raise PermissionError("Checkpoint does not contain the requested branch")
    info = metadata.get("branches", {}).get(branch, {})
    weights = root / f"{branch}.pt"
    if (not weights.is_file()
            or hashlib.sha256(weights.read_bytes()).hexdigest()
            != info.get("file_sha256")):
        raise PermissionError("Checkpoint branch byte hash differs")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    tokens = (f"{branch}_a", f"{branch}_b")
    if (not isinstance(state, Mapping) or not state
            or len(state) != info.get("tensor_count")
            or any(not any(token in name for token in tokens) for name in state)):
        raise PermissionError("Checkpoint branch tensor inventory differs")
    derived = parameter_sha256(state.items())
    if derived != info.get("state_sha256"):
        raise PermissionError("Checkpoint branch tensor digest differs")
    return derived


def model_component_sha256(model, *, include_adapters=False) -> str:
    """Hash parameters, persistent buffers, config and NF4 quant metadata.

    ``state_dict`` alone omits some bitsandbytes quantization state.  The extra
    class/quant-state records make the base binding sensitive to those runtime
    details as well as to persistent buffers.
    """
    torch = _torch()
    digest = hashlib.sha256()
    adapter_tokens = tuple(
        f"{branch}_{side}" for branch in BRANCHES for side in ("a", "b"))

    def add_tensor(name, value):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())

    for name, value in sorted(model.named_parameters(), key=lambda item: item[0]):
        if include_adapters or not any(token in name for token in adapter_tokens):
            add_tensor("parameter/" + name, value)
            digest.update(value.__class__.__name__.encode("utf-8"))
            quant_state = getattr(value, "quant_state", None)
            if quant_state is not None:
                _hash_quant_state(digest, name, quant_state, add_tensor)
    for name, value in sorted(model.named_buffers(), key=lambda item: item[0]):
        if include_adapters or not any(token in name for token in adapter_tokens):
            add_tensor("buffer/" + name, value)
    config = getattr(getattr(model, "model", model), "config", None)
    if config is not None:
        payload = config.to_dict() if hasattr(config, "to_dict") else vars(config)
        digest.update(_canonical_json(payload))
    return digest.hexdigest()


def model_component_hashes(model, *, require_quantized=False) -> dict:
    """Hash frozen parameters, NF4 state, persistent buffers and config apart.

    Qualification must compare *live* pre/post values from the loaded model;
    a caller-supplied digest is never accepted as evidence.  Keeping these
    components separate also prevents a stable config hash from masking a
    changed quantized weight or buffer.
    """
    torch = _torch()
    adapters = tuple(
        f"{branch}_{side}" for branch in BRANCHES for side in ("a", "b"))
    parameter_digest = hashlib.sha256()
    quant_digest = hashlib.sha256()
    buffer_digest = hashlib.sha256()
    quantization_inventory = []

    def add(digest, name, value):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())

    for name, value in sorted(model.named_parameters(), key=lambda item: item[0]):
        if any(token in name for token in adapters):
            continue
        add(parameter_digest, name, value)
        quant_digest.update(name.encode("utf-8"))
        quant_digest.update(value.__class__.__name__.encode("utf-8"))
        quant_state = getattr(value, "quant_state", None)
        if quant_state is None:
            quant_digest.update(b"quant_state=None")
            continue
        fields = _hash_quant_state(quant_digest, name, quant_state,
                                   lambda path, item: add(quant_digest, path, item))
        quantization_inventory.append({"parameter": name, "fields": fields})
    if require_quantized and not quantization_inventory:
        raise RuntimeError("The frozen model has no exact NF4 QuantState inventory")
    for name, value in sorted(model.named_buffers(), key=lambda item: item[0]):
        if not any(token in name for token in adapters):
            add(buffer_digest, name, value)
    config = getattr(getattr(model, "model", model), "config", None)
    payload = (config.to_dict() if hasattr(config, "to_dict") else
               (vars(config) if config is not None else {}))
    result = {
        "frozen_parameters_sha256": parameter_digest.hexdigest(),
        "quantization_state_sha256": quant_digest.hexdigest(),
        "quantization_state_inventory": quantization_inventory,
        "quantization_state_inventory_sha256": hashlib.sha256(
            _canonical_json(quantization_inventory)).hexdigest(),
        "persistent_buffers_sha256": buffer_digest.hexdigest(),
        "config_sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
    }
    result["combined_sha256"] = hashlib.sha256(_canonical_json(result)).hexdigest()
    return result


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def branch_state_dict(model, branches: Sequence[str]) -> dict:
    branches = tuple(branches)
    if not branches or not set(branches) <= set(BRANCHES):
        raise ValueError("Invalid branch state selection")
    tokens = tuple(f"{branch}_{side}" for branch in branches for side in ("a", "b"))
    return {name: value.detach().cpu() for name, value in model.state_dict().items()
            if any(token in name for token in tokens)}


def clear_adapter_branches(model, branches: Sequence[str] = BRANCHES) -> dict[str, int]:
    """Zero complete adapter branches before loading a different checkpoint.

    Evaluation reuses one quantized backbone for all checkpoints.  A partial
    checkpoint (most notably the neutral skill-only arm) therefore must not
    inherit residual tensors from the checkpoint evaluated immediately before
    it.  Clearing happens before any checkpoint bytes are loaded and is
    verified over both A and B tensors for every selected branch.
    """
    torch = _torch()
    selected = tuple(branches)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(BRANCHES):
        raise ValueError("Invalid adapter branch clear selection")
    tokens = tuple(f"{branch}_{side}" for branch in selected for side in ("a", "b"))
    matched: dict[str, int] = {branch: 0 for branch in selected}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            for branch in selected:
                if any(token in name for token in (f"{branch}_a", f"{branch}_b")):
                    parameter.zero_()
                    matched[branch] += int(parameter.numel())
                    break
    if any(count <= 0 for count in matched.values()):
        raise RuntimeError("Every adapter branch must have live tensors before clearing")
    assert_adapter_branches_zero(model, selected)
    return matched


def assert_adapter_branches_zero(model, branches: Sequence[str]) -> None:
    """Fail closed unless every tensor in the selected branches is exactly zero."""
    torch = _torch()
    selected = tuple(branches)
    if len(set(selected)) != len(selected) or not set(selected) <= set(BRANCHES):
        raise ValueError("Invalid zero-branch verification selection")
    seen = {branch: 0 for branch in selected}
    for name, parameter in model.named_parameters():
        for branch in selected:
            if any(token in name for token in (f"{branch}_a", f"{branch}_b")):
                seen[branch] += 1
                if torch.count_nonzero(parameter.detach()).item() != 0:
                    raise PermissionError(f"Inactive adapter branch {branch} is not exactly zero")
                break
    if any(count <= 0 for count in seen.values()):
        raise RuntimeError("Every verified adapter branch must have live tensors")


def save_adapter_checkpoint(model, output_dir, *, target_names, base_sha256,
                            optimizer_sha256, source_sha256, stage: str,
                            condition: str, seed: int, optimizer_manifest=None,
                            plan_sha256=None, predecessor_branch_hashes=None,
                            run_mode="primary"):
    """Save a source/base/optimizer-bound adapter checkpoint exactly once."""
    torch = _torch()
    if (stage not in STAGES or stage == "frozen"
            or stage == "skill" and not model.ordinal_lora_config.use_skill_adapter):
        raise ValueError("Invalid checkpoint stage")
    if (stage == "skill" and condition != "neutral_skill") or (
            stage.startswith("delta") and condition not in {
                "self_continuity", "artifact_continuity", "label_only"}):
        raise ValueError("Checkpoint condition differs from its sequential stage")
    if not isinstance(seed, int):
        raise ValueError("Checkpoint seed must be an integer")
    optimizer_manifest = dict(optimizer_manifest or {})
    if (not isinstance(optimizer_sha256, str) or len(optimizer_sha256) != 64
            or optimizer_manifest.get("class") != "AdamW"
            or optimizer_manifest.get("state_sha256") != optimizer_sha256
            or not isinstance(optimizer_manifest.get("successful_updates"), int)
            or optimizer_manifest["successful_updates"] <= 0
            or any(dtype != "torch.float32"
                   for dtype in optimizer_manifest.get("state_dtypes", ()))):
        raise PermissionError("Checkpoint optimizer evidence is incomplete or non-FP32")
    active_branches = STAGE_BRANCHES[stage]
    predecessor_branch_hashes = dict(predecessor_branch_hashes or {})
    for branch, expected in predecessor_branch_hashes.items():
        if branch not in active_branches or branch_sha256(model, branch) != expected:
            raise PermissionError("A frozen predecessor branch changed during training")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    branch_metadata = {}
    for branch in active_branches:
        state = branch_state_dict(model, (branch,))
        weights = root / f"{branch}.pt"
        torch.save(state, weights)
        branch_metadata[branch] = {
            "state_sha256": branch_sha256(model, branch),
            "file_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "tensor_count": len(state),
        }
    metadata = {
        "version": CHECKPOINT_VERSION,
        "config": json.loads(json.dumps(asdict(model.ordinal_lora_config))),
        "stage": stage,
        "condition": condition,
        "seed": seed,
        "run_mode": run_mode,
        "primary_parent_eligible": run_mode == "primary",
        "target_names": list(target_names),
        "base_sha256": base_sha256,
        "optimizer_sha256": optimizer_sha256,
        "source_sha256": source_sha256,
        "active_branches": list(active_branches),
        "branches": branch_metadata,
        "adapter_sha256": adapter_sha256(model, active_branches),
        "plan_sha256": plan_sha256,
        "optimizer_manifest": optimizer_manifest,
        "predecessor_branch_hashes": predecessor_branch_hashes,
    }
    metadata["receipt_sha256"] = hashlib.sha256(_canonical_json(metadata)).hexdigest()
    (root / "adapter_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_adapter_checkpoint(model, input_dir, *, expected_targets,
                            expected_base_sha256, expected_source_sha256,
                            expected_condition: str, expected_seed: int,
                            expected_stage: str | None = None,
                            expected_plan_sha256: str | None = None):
    torch = _torch()
    root = Path(input_dir)
    metadata = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    self_check = dict(metadata)
    embedded_receipt = self_check.pop("receipt_sha256", None)
    if (not isinstance(embedded_receipt, str)
            or hashlib.sha256(_canonical_json(self_check)).hexdigest() != embedded_receipt):
        raise PermissionError("Checkpoint metadata self-hash differs")
    if metadata.get("version") != CHECKPOINT_VERSION:
        raise PermissionError("Checkpoint version differs")
    expected_config = json.loads(json.dumps(asdict(model.ordinal_lora_config)))
    optimizer = metadata.get("optimizer_manifest", {})
    if (metadata.get("config") != expected_config
            or metadata.get("target_names") != list(expected_targets)
            or metadata.get("base_sha256") != expected_base_sha256
            or metadata.get("source_sha256") != expected_source_sha256
            or metadata.get("condition") != expected_condition
            or metadata.get("seed") != expected_seed
            or (expected_stage is not None and metadata.get("stage") != expected_stage)
            or (expected_plan_sha256 is not None
                and metadata.get("plan_sha256") != expected_plan_sha256)):
        raise PermissionError("Checkpoint source/base/geometry binding mismatch")
    if (optimizer.get("class") != "AdamW"
            or optimizer.get("state_sha256") != metadata.get("optimizer_sha256")
            or optimizer.get("successful_updates", 0) <= 0
            or any(dtype != "torch.float32" for dtype in optimizer.get("state_dtypes", ()))):
        raise PermissionError("Checkpoint optimizer manifest differs")
    branches = tuple(metadata.get("active_branches", ()))
    if branches != STAGE_BRANCHES.get(metadata.get("stage")):
        raise PermissionError("Checkpoint branch/stage inventory differs")
    # Always reset the complete ordinal adapter before loading.  Without this,
    # switching from a delta3 checkpoint to the skill-only checkpoint leaves
    # the previous residuals live and makes results depend on manifest order.
    clear_adapter_branches(model)
    merged = {}
    for branch in branches:
        branch_meta = metadata.get("branches", {}).get(branch, {})
        weights = root / f"{branch}.pt"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != branch_meta.get("file_sha256"):
            raise ValueError("Adapter branch byte hash mismatch")
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if len(state) != branch_meta.get("tensor_count") or set(merged) & set(state):
            raise ValueError("Adapter branch tensor inventory differs")
        merged.update(state)
    current = model.state_dict()
    if any(name not in current or current[name].shape != value.shape for name, value in merged.items()):
        raise ValueError("Adapter branch shape/name differs")
    model.load_state_dict(merged, strict=False)
    for branch in branches:
        if branch_sha256(model, branch) != metadata["branches"][branch]["state_sha256"]:
            raise ValueError("Live branch hash differs after load")
    if adapter_sha256(model, branches) != metadata.get("adapter_sha256"):
        raise ValueError("Live adapter hash differs after load")
    inactive = tuple(branch for branch in BRANCHES if branch not in branches)
    if inactive:
        assert_adapter_branches_zero(model, inactive)
    model.set_train_stage("frozen")
    return metadata


def load_sequential_parent_branches(model, input_dir, *, expected_stage: str,
                                    expected_targets, expected_base_sha256: str,
                                    expected_source_sha256: str,
                                    expected_condition: str, expected_seed: int):
    """Load only the exact predecessor branches, never unused future state.

    A rank-4 neutral skill checkpoint may parent a rank-8 residual candidate;
    only its separately frozen skill geometry must match in that one case.
    """
    torch = _torch()
    root = Path(input_dir)
    metadata = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    self_check = dict(metadata)
    embedded_receipt = self_check.pop("receipt_sha256", None)
    if (not isinstance(embedded_receipt, str)
            or hashlib.sha256(_canonical_json(self_check)).hexdigest() != embedded_receipt):
        raise PermissionError("Sequential parent metadata self-hash differs")
    expected_branches = STAGE_BRANCHES.get(expected_stage)
    optimizer = metadata.get("optimizer_manifest", {})
    if (metadata.get("version") != CHECKPOINT_VERSION
            or metadata.get("run_mode") != "primary"
            or metadata.get("primary_parent_eligible") is not True
            or metadata.get("stage") != expected_stage
            or tuple(metadata.get("active_branches", ())) != expected_branches
            or metadata.get("target_names") != list(expected_targets)
            or metadata.get("base_sha256") != expected_base_sha256
            or metadata.get("source_sha256") != expected_source_sha256
            or metadata.get("condition") != expected_condition
            or metadata.get("seed") != expected_seed):
        raise PermissionError("Sequential parent identity differs")
    if (optimizer.get("class") != "AdamW"
            or optimizer.get("state_sha256") != metadata.get("optimizer_sha256")
            or optimizer.get("successful_updates", 0) <= 0
            or any(dtype != "torch.float32" for dtype in optimizer.get("state_dtypes", ()))):
        raise PermissionError("Sequential parent optimizer evidence differs")
    saved = metadata.get("config", {})
    current = asdict(model.ordinal_lora_config)
    common = ("version", "model_id", "model_revision", "skill_rank", "skill_alpha",
              "dropout", "danger_levels", "base_frozen", "use_skill_adapter",
              "initialized_from")
    if any(saved.get(key) != current.get(key) for key in common):
        raise PermissionError("Sequential parent shared geometry differs")
    if expected_stage != "skill" and (saved.get("rank"), saved.get("alpha")) != (
            current.get("rank"), current.get("alpha")):
        raise PermissionError("Residual parent rank/alpha differs")
    merged = {}
    hashes = {}
    for branch in expected_branches:
        info = metadata.get("branches", {}).get(branch, {})
        path = root / f"{branch}.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != info.get("file_sha256"):
            raise ValueError("Sequential parent branch bytes differ")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if len(state) != info.get("tensor_count") or set(merged) & set(state):
            raise ValueError("Sequential parent tensor inventory differs")
        merged.update(state)
        hashes[branch] = info.get("state_sha256")
    current_state = model.state_dict()
    if any(name not in current_state or current_state[name].shape != value.shape
           for name, value in merged.items()):
        raise ValueError("Sequential parent tensor shape differs")
    model.load_state_dict(merged, strict=False)
    if any(branch_sha256(model, branch) != value for branch, value in hashes.items()):
        raise ValueError("Sequential parent live branch hash differs")
    model.set_train_stage("frozen")
    return metadata, hashes
