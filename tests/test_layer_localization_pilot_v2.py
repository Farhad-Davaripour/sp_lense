from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from scripts import layer_localization_pilot_v2 as v2

ROOT = Path(__file__).resolve().parents[1]

V1_INPUT_KEYS = {
    "v1_protocol_config",
    "v1_preregistration",
    "v1_stage1_rows",
    "v1_stage1_summary",
    "v1_stage1_selection",
    "v1_implementation_failure",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_changes_only_implementation_identity_provenance_and_namespace() -> None:
    original = json.loads((ROOT / "configs/layer_localization_pilot.json").read_text())
    observed = json.loads((ROOT / v2.V2_LOCK_RELATIVE_PATH).read_text())
    normalized = copy.deepcopy(observed)
    correction = normalized.pop("implementation_correction")
    assert correction["only_semantic_code_change"] == (
        "accept_the_transformer_lens_hook_keyword_without_using_it"
    )
    assert (
        correction["layers_data_probe_direction_alpha_random_gates_and_sealed_policy_changed"]
        is False
    )

    normalized["study"] = original["study"]
    normalized["status"] = original["status"]
    for key in V1_INPUT_KEYS:
        normalized["inputs"].pop(key)
    normalized["outputs"]["directory"] = original["outputs"]["directory"]
    normalized["outputs"]["forbidden_output_directories"] = original["outputs"][
        "forbidden_output_directories"
    ]
    normalized["result_placeholder"]["status"] = original["result_placeholder"]["status"]
    assert normalized == original


def test_v2_binds_every_v1_attempt_artifact() -> None:
    lock = json.loads((ROOT / v2.V2_LOCK_RELATIVE_PATH).read_text())
    for key in V1_INPUT_KEYS:
        binding = lock["inputs"][key]
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]
    assert lock["outputs"]["directory"] == v2.V2_OUTPUT_RELATIVE_PATH.as_posix()
    assert (
        "evidence/layer_localization_qwen35_08b" in lock["outputs"]["forbidden_output_directories"]
    )


class _Boundary:
    prompt_length = 2

    @staticmethod
    def token_id(label: str) -> int:
        return {"A": 0, "B": 1}[label]


class _FakeModel:
    def __init__(self) -> None:
        self._hooks: list[tuple[str, Any]] = []

    def zero_grad(self, *, set_to_none: bool) -> None:
        assert set_to_none is True

    @contextmanager
    def hooks(self, *, fwd_hooks: list[tuple[str, Any]]) -> Any:
        self._hooks = fwd_hooks
        try:
            yield
        finally:
            self._hooks = []

    def __call__(self, tokens: Any) -> Any:
        activation = torch.arange(2048, dtype=torch.float32).reshape(1, 2, 1024)
        for _, callback in self._hooks:
            activation = callback(activation, hook=object())
            activation = activation * 1.0
        logits = torch.stack((activation.sum(dim=-1), -activation.sum(dim=-1)), dim=-1)
        assert logits.shape == (1, tokens.shape[-1], 2)
        return logits


class _FakeBackend:
    torch = torch

    def __init__(self) -> None:
        self.model = _FakeModel()

    @staticmethod
    def encode(prompt: str) -> Any:
        assert prompt == "test prompt"
        return torch.tensor([[1, 2]], dtype=torch.long)


def test_v2_gradient_hook_accepts_resident_keyword_api() -> None:
    gradients = v2._capture_final_prompt_gradients(
        _FakeBackend(),
        "test prompt",
        "A",
        "B",
        layers=[6, 10],
        boundary=_Boundary(),
    )
    assert set(gradients) == {6, 10}
    assert all(value.shape == (1024,) for value in gradients.values())
    assert all(value.device.type == "cpu" for value in gradients.values())
    assert all(value.dtype == torch.float32 for value in gradients.values())
    assert all(bool(value.isfinite().all()) for value in gradients.values())


def test_v2_source_fingerprint_includes_wrapper_base_and_test() -> None:
    assert v2.V2_SCRIPT_RELATIVE_PATH in v2.V2_SOURCE_RELATIVE_PATHS
    assert v2.V2_LOCK_RELATIVE_PATH in v2.V2_SOURCE_RELATIVE_PATHS
    assert Path("scripts/layer_localization_pilot.py") in v2.V2_SOURCE_RELATIVE_PATHS
    assert v2.V2_TEST_RELATIVE_PATH in v2.V2_SOURCE_RELATIVE_PATHS
