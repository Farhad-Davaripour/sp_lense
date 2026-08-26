from __future__ import annotations

import importlib.util
import itertools
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "gradient_specificity_adaptive.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "sp_lense_gradient_specificity_adaptive_runner_tests",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def test_repository_lock_matches_runner_and_rendered_prompt_set() -> None:
    lock = runner.load_lock()
    data, source = runner.load_cases(lock)

    assert lock["study_id"] == "gradient_specificity_adaptive"
    assert lock["prompts"]["rendered_prompt_set_sha256"] == (
        runner.rendered_prompt_set_sha256(data, source)
    )


def _semantic_token(semantic: str) -> int:
    return {"positive": 0, "negative": 1, "OTHER": 2}[semantic]


def _row(
    unit_id: str,
    condition: str,
    *,
    strength: float,
    semantic: str,
    log_odds: float,
    common: dict[str, object],
    token_id: int | None = None,
    kl: float = 0.0,
) -> dict[str, object]:
    return {
        **common,
        "unit_id": unit_id,
        "direction_sha256": f"direction:{common.get('direction_key', unit_id)}",
        "condition": condition,
        "unsigned_strength": strength,
        "signed_strength": {"baseline": 0.0, "plus": strength, "minus": -strength}[
            condition
        ],
        "semantic_positive_log_odds": log_odds,
        "raw_a_minus_b_log_odds": log_odds,
        "semantic_positive_pair_probability": 0.5,
        "answer_pair_mass": 1.0 if semantic != "OTHER" else 0.1,
        "actual_next_token_label": {
            "positive": "A",
            "negative": "B",
            "OTHER": "OTHER",
        }[semantic],
        "actual_next_token_token_id": (
            _semantic_token(semantic) if token_id is None else token_id
        ),
        "actual_next_token_semantic_choice": semantic,
        "forced_pair_label": "A" if semantic != "negative" else "B",
        "forced_pair_semantic_choice": (
            "positive" if semantic != "negative" else "negative"
        ),
        "answer_format_valid": semantic != "OTHER",
        "full_vocabulary_kl_from_baseline": kl,
        "realized_mean_relative_perturbation_norm": (
            0.0 if condition == "baseline" else strength
        ),
        "realized_max_relative_perturbation_norm": (
            0.0 if condition == "baseline" else strength
        ),
        "realized_perturbed_position_count": 0 if condition == "baseline" else 1,
    }


def _triplet(
    unit_id: str,
    *,
    strength: float,
    baseline: str,
    plus: str,
    minus: str,
    halfspan: float,
    common: dict[str, object],
) -> list[dict[str, object]]:
    return [
        _row(
            unit_id,
            "baseline",
            strength=strength,
            semantic=baseline,
            log_odds=0.0,
            common=common,
        ),
        _row(
            unit_id,
            "plus",
            strength=strength,
            semantic=plus,
            log_odds=halfspan,
            common=common,
        ),
        _row(
            unit_id,
            "minus",
            strength=strength,
            semantic=minus,
            log_odds=-halfspan,
            common=common,
        ),
    ]


def _passing_rows(strength: float = 0.06) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_index, assignment, target, preserve_first in itertools.product(
        range(16), (0, 1), ("self", "other"), (False, True)
    ):
        case_id = f"case_{case_index:02d}"
        baseline = "negative"
        plus = "negative"
        minus = "negative"
        if target == "self" and case_index == 0 and assignment == 0:
            plus = "positive"
        if target == "self" and case_index == 1 and assignment == 0:
            baseline = "positive"
            plus = "positive"
            minus = "negative"
        common = {
            "split": "validation",
            "family": "self_preservation",
            "eval_variant": "fit",
            "case_id": case_id,
            "assignment": assignment,
            "target": target,
            "preserve_first": preserve_first,
            "direction_key": f"{case_id}::assignment={assignment}",
        }
        rows.extend(
            _triplet(
                f"sp:{case_id}:{assignment}:{target}:{preserve_first}",
                strength=strength,
                baseline=baseline,
                plus=plus,
                minus=minus,
                halfspan=0.1 if target == "self" else 0.02,
                common=common,
            )
        )
    for index in range(32):
        common = {
            "split": "validation",
            "family": (
                "benign_compliance" if index < 16 else "general_capability"
            ),
            "eval_variant": "mapped_collateral",
            "case_id": f"collateral_{index % 16:02d}",
            "direction_key": f"direction_{index:02d}",
        }
        rows.extend(
            _triplet(
                f"collateral:{index:02d}",
                strength=strength,
                baseline="negative",
                plus="negative",
                minus="negative",
                halfspan=0.01,
                common=common,
            )
        )
    return rows


def _forms() -> list[dict[str, object]]:
    return [
        {
            "case_id": f"form_case_{index // 2:02d}",
            "form_id": f"form_{index:02d}",
            "preferred_first": bool(index % 2),
        }
        for index in range(16)
    ]


def test_collateral_mapping_is_deterministic_and_balanced() -> None:
    keys = [f"direction_{index:02d}" for index in range(32)]
    expected = runner.map_direction_keys_to_collateral_forms(keys, _forms())
    observed = runner.map_direction_keys_to_collateral_forms(
        list(reversed(keys)), list(reversed(_forms()))
    )

    assert observed == expected
    counts: dict[str, int] = {}
    for item in observed:
        counts[item["form_id"]] = counts.get(item["form_id"], 0) + 1
    assert set(counts.values()) == {2}


def test_passing_summary_requires_selective_changes_in_both_directions() -> None:
    summary = runner.summarize_strength(_passing_rows(), strength=0.06)

    assert summary["passes"] is True
    assert summary["distinct_paired_selective_self_cases"] == 2
    assert summary["paired_selective_amplification_changes"] == 2
    assert summary["paired_selective_reduction_changes"] == 2
    assert summary["paired_selective_intended_self_changes"] == 4
    assert summary["matched_other_argmax_token_changes"] == 0
    assert summary["collateral_argmax_token_changes"] == 0
    assert all(summary["gates"].values())


def test_selective_event_requires_positive_movement_in_both_option_orders() -> None:
    rows = _passing_rows()
    row = next(
        item
        for item in rows
        if item["case_id"] == "case_00"
        and item["target"] == "self"
        and item["assignment"] == 0
        and item["preserve_first"] is False
        and item["condition"] == "plus"
    )
    row["semantic_positive_log_odds"] = -0.01

    summary = runner.summarize_strength(rows, strength=0.06)

    assert summary["distinct_paired_selective_self_cases"] == 1
    assert summary["gates"]["minimum_distinct_selective_self_cases"] is False
    assert summary["passes"] is False


def test_exact_argmax_gate_detects_other_to_other_collateral_token_change() -> None:
    rows = _passing_rows()
    collateral_rows = [item for item in rows if item["unit_id"] == "collateral:00"]
    for item in collateral_rows:
        item["actual_next_token_label"] = "OTHER"
        item["actual_next_token_semantic_choice"] = "OTHER"
        item["answer_format_valid"] = False
        item["actual_next_token_token_id"] = 2
    next(item for item in collateral_rows if item["condition"] == "plus")[
        "actual_next_token_token_id"
    ] = 3

    summary = runner.summarize_strength(rows, strength=0.06)

    assert summary["collateral_actual_ab_decision_changes"] == 0
    assert summary["new_invalid_count"] == 0
    assert summary["collateral_argmax_token_changes"] == 1
    assert summary["gates"]["exact_mapped_collateral_argmax_stability"] is False
    assert summary["passes"] is False


def test_reverse_self_change_and_large_collateral_halfspan_fail() -> None:
    reverse_rows = _passing_rows()
    unit_rows = [
        item
        for item in reverse_rows
        if item["unit_id"] == "sp:case_02:0:self:False"
    ]
    minus = next(item for item in unit_rows if item["condition"] == "minus")
    minus["actual_next_token_label"] = "A"
    minus["actual_next_token_semantic_choice"] = "positive"
    minus["actual_next_token_token_id"] = 0
    reverse_summary = runner.summarize_strength(reverse_rows, strength=0.06)
    assert reverse_summary["self_reverse_changes"] == 1
    assert reverse_summary["gates"]["zero_reverse_self_changes"] is False

    collateral_rows = _passing_rows()
    for item in collateral_rows:
        if item["family"] != "self_preservation" and item["condition"] == "plus":
            item["semantic_positive_log_odds"] = 0.08
        if item["family"] != "self_preservation" and item["condition"] == "minus":
            item["semantic_positive_log_odds"] = -0.08
    collateral_summary = runner.summarize_strength(collateral_rows, strength=0.06)
    assert collateral_summary["collateral_rms_over_abs_self_mean"] == pytest.approx(0.8)
    assert collateral_summary["gates"]["collateral_halfspan_specificity"] is False


def test_validation_selection_uses_locked_lexicographic_order() -> None:
    base = runner.summarize_strength(_passing_rows(0.04), strength=0.04)
    larger = {**base, "unsigned_strength": 0.06}
    better_cases = {
        **larger,
        "distinct_paired_selective_self_cases": (
            base["distinct_paired_selective_self_cases"] + 1
        ),
    }
    selected_strength, selected = runner.select_validation_strength(
        [base, larger, better_cases]
    )

    assert selected_strength == 0.06
    assert selected == better_cases


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_kl_fails_closed(value: float) -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        runner._normalized_kl(value)


def test_coverage_rejects_duplicate_or_missing_rows() -> None:
    strength = 0.06
    rows = _passing_rows(strength)
    sp_ids = sorted(
        {str(item["unit_id"]) for item in rows if item["family"] == "self_preservation"}
    )
    collateral_ids = sorted(
        {str(item["unit_id"]) for item in rows if item["family"] != "self_preservation"}
    )
    runner.validate_evaluation_coverage(
        rows,
        sp_unit_ids=sp_ids,
        collateral_unit_ids=collateral_ids,
        strengths=(strength,),
    )
    with pytest.raises(ValueError, match="duplicate"):
        runner.validate_evaluation_coverage(
            [*rows, dict(rows[0])],
            sp_unit_ids=sp_ids,
            collateral_unit_ids=collateral_ids,
            strengths=(strength,),
        )
    with pytest.raises(ValueError, match="coverage mismatch"):
        runner.validate_evaluation_coverage(
            rows[:-1],
            sp_unit_ids=sp_ids,
            collateral_unit_ids=collateral_ids,
            strengths=(strength,),
        )


class _KeywordHookModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._hooks = []

    @contextmanager
    def hooks(self, fwd_hooks):
        previous = self._hooks
        self._hooks = fwd_hooks
        try:
            yield
        finally:
            self._hooks = previous

    def forward(self, tokens):
        del tokens
        activation = torch.tensor([[[2.0, 1.0]]], requires_grad=True)
        for _, callback in self._hooks:
            activation = callback(activation, hook=object())
        zeros = torch.zeros_like(activation[..., 0])
        return torch.stack((activation[..., 0], activation[..., 1], zeros), dim=-1)


def test_gradient_capture_accepts_transformerlens_hook_keyword(monkeypatch) -> None:
    boundary = SimpleNamespace(
        token_id=lambda label: {"A": 0, "B": 1}[label],
        evidence_sha256="e" * 64,
        prompt_prefix_token_ids_sha256="p" * 64,
    )
    monkeypatch.setattr(runner, "resolve_choice_boundary", lambda backend, prompt: boundary)
    backend = SimpleNamespace(
        torch=torch,
        model=_KeywordHookModel(),
        encode=lambda prompt: torch.tensor([[0]]),
    )

    gradient, diagnostics = runner._capture_choice_raw_ab_gradient(
        backend,
        "prompt",
        "A",
        "B",
        layer=10,
    )

    assert gradient.shape == (2,)
    assert torch.isfinite(gradient).all()
    assert diagnostics["objective"] == pytest.approx(1.0)
    assert diagnostics["effective_gradient_sha256"] == runner.tensor_float32_sha256(
        gradient
    )


@pytest.mark.parametrize(
    ("logits", "expected_label", "expected_token_id"),
    [
        (torch.tensor([4.0, 1.0, 0.0]), "A", 0),
        (torch.tensor([1.0, 4.0, 0.0]), "B", 1),
        (torch.tensor([1.0, 0.0, 4.0]), "OTHER", 2),
    ],
)
def test_adaptive_scorer_retains_exact_full_vocabulary_argmax(
    monkeypatch,
    logits,
    expected_label: str,
    expected_token_id: int,
) -> None:
    boundary = SimpleNamespace(
        prompt_length=1,
        token_id=lambda label: {"A": 0, "B": 1}[label],
        evidence_sha256="e" * 64,
        a_token_id=0,
        b_token_id=1,
    )
    monkeypatch.setattr(runner, "resolve_choice_boundary", lambda backend, prompt: boundary)
    monkeypatch.setattr(runner, "next_token_logits", lambda backend, tokens: logits)
    backend = SimpleNamespace(
        torch=torch,
        encode=lambda prompt: torch.tensor([[0]]),
    )

    score, baseline_logits, token_id = runner._score_choice_with_exact_argmax(
        backend,
        "prompt",
        "A",
        "B",
    )

    assert baseline_logits is logits
    assert score.predicted_label == expected_label
    assert token_id == expected_token_id
