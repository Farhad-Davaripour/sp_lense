from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _module() -> object:
    path = ROOT / "scripts" / "compact_interface_equivariant_exact_head_logits.py"
    spec = importlib.util.spec_from_file_location("sp_lense_compact_logits_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_owned_vector_drops_unrelated_backing_storage() -> None:
    module = _module()
    backing = torch.arange(10_000, dtype=torch.float32)
    view = backing[123:223]

    owned = module._owned_float32_vector(torch, view)

    assert torch.equal(owned, view)
    assert owned.untyped_storage().nbytes() == owned.numel() * owned.element_size()
    assert owned.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr()


def test_atomic_compact_save_is_lossless_and_small(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    backing = torch.arange(100_000, dtype=torch.float32)
    owned = module._owned_float32_vector(torch, backing[100:200])
    path = tmp_path / "compact.pt"

    module._atomic_torch_save(torch, path, owned)
    reloaded = module._load_tensor(torch, path)

    assert torch.equal(reloaded, owned)
    assert module._raw_tensor_hash(reloaded) == module._raw_tensor_hash(owned)
    assert path.stat().st_size < 1024 * 1024
    with pytest.raises(RuntimeError, match="refusing to replace"):
        module._atomic_torch_save(torch, path, owned)


@pytest.mark.parametrize(
    "value",
    [torch.ones((2, 2), dtype=torch.float32), torch.ones(2, dtype=torch.float64)],
)
def test_owned_vector_rejects_wrong_shape_or_dtype(value: torch.Tensor) -> None:
    module = _module()

    with pytest.raises(RuntimeError, match="one finite float32 vector"):
        module._owned_float32_vector(torch, value)
