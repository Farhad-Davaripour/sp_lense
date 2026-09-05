from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch

from scripts import future_choice_scoring_reference as reference
from sp_lense.future_choice_scoring import ARITHMETIC_REPRODUCTION_TOLERANCE, score_float32_logits


def score(z, b=None, preserve_label="A"):
    return score_float32_logits(
        torch,
        z,
        z if b is None else b,
        choice_a_token_id=0,
        choice_b_token_id=1,
        preserve_label=preserve_label,
    )


def fixture_arrays(name):
    if name == "large_tail":
        b = torch.full((262144,), -16.0, dtype=torch.float32)
        b[:2] = torch.tensor([0.0, -0.5])
        z = b.clone()
        z[0] += 0.125
        return z, b
    pairs = {
        "diffuse": ([0.0, 0.125, -0.125, 0.0], [0.0, 0.0, 0.0, 0.0]),
        "sharp": ([1000.0, -1000.0, -800.0], [-1000.0, 1000.0, -800.0]),
        "small_mass": ([-1000.0, -999.0, 0.0], [-999.0, -1000.0, 0.0]),
        "tie": ([1.0, 1.0, 1.0], [0.0, -0.5, 1.0]),
        "near_tie": ([0.0, 1e-7, -0.5], [1e-7, 0.0, -0.5]),
        "offset": ([1048576.125, 1048576.0, 1048575.5], [1048576.0, 1048576.125, 1048575.5]),
        "extreme_offset": (
            [2.0**60 + 2.0**40, 2.0**60, 2.0**60 - 2.0**40],
            [2.0**60, 2.0**60 + 2.0**40, 2.0**60 - 2.0**40],
        ),
    }
    z, b = pairs[name]
    return torch.tensor(z, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)


@pytest.mark.parametrize(
    "name",
    ["large_tail", "diffuse", "sharp", "small_mass", "tie", "near_tie", "offset", "extreme_offset"],
)
@pytest.mark.parametrize("preserve_label", ["A", "B"])
def test_stable_contract_against_independent_reference(name, preserve_label):
    z, b = fixture_arrays(name)
    result = score(z, b, preserve_label)
    reference.verify_record(
        result,
        z.tolist(),
        b.tolist(),
        choice_a_token_id=0,
        choice_b_token_id=1,
        preserve_label=preserve_label,
    )
    assert result["kl_from_baseline"] >= -1e-6
    assert score(z)["kl_from_baseline"] == 0
    assert ARITHMETIC_REPRODUCTION_TOLERANCE == reference.REPRODUCTION_TOLERANCE == 2e-5


def test_large_common_offset_invariance_for_exactly_representable_float32_differences():
    z = torch.tensor([0.125, 0.0, -0.5], dtype=torch.float32)
    b = torch.tensor([0.0, 0.125, -0.5], dtype=torch.float32)
    a, shifted = score(z, b), score(z + 2**20, b + 2**20)
    for key in reference.NUMERIC_FIELDS + reference.EXACT_FIELDS:
        assert a[key] == shifted[key]


def test_all_normalizers_are_centered_float64_without_changing_input(monkeypatch):
    original = torch.logsumexp
    seen = []

    def inspect(values, dim):
        seen.append(values.dtype)
        assert values.max().item() == 0
        return original(values, dim)

    monkeypatch.setattr(torch, "logsumexp", inspect)
    z, b = fixture_arrays("large_tail")
    score(z, b)
    assert seen == [torch.float64] * 3
    assert z.dtype == b.dtype == torch.float32


def test_kl_orientation_edited_relative_to_baseline():
    b = torch.tensor([0.0, -2.0, -3.0], dtype=torch.float32)
    z = torch.tensor([0.0, -0.25, -1.0], dtype=torch.float32)
    result = score(z, b)
    p, q = torch.softmax(z.double(), 0).tolist(), torch.softmax(b.double(), 0).tolist()
    expected = math.fsum(x * math.log(x / y) for x, y in zip(p, q, strict=True))
    reverse = math.fsum(y * math.log(y / x) for x, y in zip(p, q, strict=True))
    assert result["kl_from_baseline"] == pytest.approx(expected, abs=1e-12)
    assert abs(result["kl_from_baseline"] - reverse) > 0.01


def test_tie_conventions_and_semantic_order():
    z = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    a, b = score(z), score(z, preserve_label="B")
    assert a["pair_tie"] and b["pair_tie"]
    assert a["full_argmax_tie_count"] == 2
    assert a["actual_next_token_id"] == b["actual_next_token_id"] == 0
    assert a["forced_pair_label"] == "A" and b["forced_pair_label"] == "B"
    z = torch.tensor([0.25, 0.0, -1.0], dtype=torch.float32)
    assert score(z)["preserve_log_odds"] == -score(z, preserve_label="B")["preserve_log_odds"]


def test_float32_quantization_is_not_repaired_or_inference_changed():
    z = torch.tensor([2**24, 2**24 + 1, 0.0], dtype=torch.float32, requires_grad=True)
    before = z.detach().clone()
    result = score(z)
    assert result["pair_tie"]  # The +1 was lost before measurement; no invented precision.
    assert torch.equal(z, before) and z.requires_grad and z.grad is None


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        torch.tensor([0.0, float("nan")]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([0.0]),
    ],
)
def test_reject_invalid_input(bad):
    with pytest.raises(ValueError):
        score(bad)


def test_reference_has_no_production_or_torch_import_and_rejects_corruption():
    source = Path(reference.__file__).read_text(encoding="utf-8")
    imports = [
        n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not any("torch" in alias.name or "sp_lense" in alias.name for alias in n.names)
        and getattr(n, "module", "") not in ("torch", "sp_lense.future_choice_scoring")
        for n in imports
    )
    z = torch.tensor([0.0, 0.125, -0.5])
    result = score(z)
    result["answer_pair_mass"] += 0.001
    with pytest.raises(ValueError, match="reproduction mismatch"):
        reference.verify_record(
            result,
            z.tolist(),
            z.tolist(),
            choice_a_token_id=0,
            choice_b_token_id=1,
            preserve_label="A",
        )
