from __future__ import annotations

import importlib.util
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "gradient_specificity_v3_development.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "sp_lense_gradient_specificity_v3_runner_tests",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def test_preflight_is_model_free_and_renders_the_locked_development_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("preflight must not load the model")

    monkeypatch.setattr(runner.ResearchBackend, "load", forbidden_load)
    result = runner.run_preflight()

    assert result["development_only"] is True
    assert result["model_loads"] == 0
    assert result["model_forwards"] == 0
    assert result["external_model_judges"] == 0
    assert result["nuisance_fit_prompt_forms"] == 32
    assert result["audit_control_prompt_forms"] == 32
    assert result["stages"]["A"]["case_assignment_direction_attempts"] == 8
    assert result["stages"]["B"]["case_assignment_direction_attempts"] == 16
    assert result["stages"]["B"]["new_capture_prompt_form_count"] == 32
    assert (
        result["stages"]["B"]["cumulative_factor_balance"][
            "exactly_four_false_four_true_each_factor"
        ]
        is True
    )
    assert result["target_self_excluded_from_selectivity_kl_gate"] is True


def test_stage_b_factor_balance_fails_closed_on_an_unbalanced_eight_case_set() -> None:
    balanced = {
        f"case-{index}": design_index
        for index, design_index in enumerate((14, 6, 10, 4, 1, 9, 5, 11))
    }
    diagnostics = runner._balanced_stage_b_factor_diagnostics(balanced)
    assert all(
        counts == {"false": 4, "true": 4}
        for counts in diagnostics["factor_false_true_counts"].values()
    )

    with pytest.raises(RuntimeError, match="not 4/4 balanced"):
        runner._balanced_stage_b_factor_diagnostics({f"case-{index}": 0 for index in range(8)})


@pytest.mark.parametrize("forbidden", ["validation", "sealed", "sealed_test", "test"])
def test_stage_normalization_rejects_nondevelopment_split_names(forbidden: str) -> None:
    with pytest.raises(ValueError, match="development-only"):
        runner.normalize_stage(forbidden)
    assert runner.normalize_stage("a") == "A"


class _FakeHookModel(torch.nn.Module):
    def __init__(self, *, hook_repetitions: int = 1) -> None:
        super().__init__()
        self.hook_repetitions = hook_repetitions
        self.forward_count = 0
        self._fwd_hooks = []
        self.register_buffer(
            "base",
            torch.tensor(
                [
                    [
                        [0.1, 0.2, 0.3, 0.4],
                        [0.4, 0.3, 0.2, 0.1],
                        [0.2, 0.4, 0.6, 0.8],
                    ]
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "unembed",
            torch.arange(48, dtype=torch.float32).reshape(12, 4) / 20.0,
        )

    @contextmanager
    def hooks(self, fwd_hooks):
        previous = self._fwd_hooks
        self._fwd_hooks = list(fwd_hooks)
        try:
            yield
        finally:
            self._fwd_hooks = previous

    def forward(self, _tokens):
        self.forward_count += 1
        activation = self.base.clone()
        for _ in range(self.hook_repetitions):
            for _, hook in self._fwd_hooks:
                activation = hook(activation, hook=None)
        return activation @ self.unembed.T


def _fake_backend(*, hook_repetitions: int = 1):
    model = _FakeHookModel(hook_repetitions=hook_repetitions)
    return SimpleNamespace(
        torch=torch,
        model=model,
        encode=lambda _prompt: torch.tensor([[3, 4, 5]], dtype=torch.long),
    )


def _fake_boundary():
    return SimpleNamespace(
        prompt_length=3,
        a_token_id=0,
        b_token_id=1,
        evidence_sha256="boundary-hash",
        prompt_prefix_token_ids_sha256="prefix-hash",
        token_id=lambda label: {"A": 0, "B": 1}[label],
    )


def test_capture_uses_one_forward_one_hook_and_batched_vjps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _fake_backend()
    monkeypatch.setattr(runner, "resolve_choice_boundary", lambda *_args: _fake_boundary())
    form = {
        "form_id": "synthetic",
        "positive_label": "A",
        "negative_label": "B",
        "prompt": "synthetic prompt",
        "prompt_sha256": runner.prompt_sha256("synthetic prompt"),
    }

    record = runner._capture_prompt_observation(
        backend,
        form,
        layer=10,
        fisher_top_count=8,
        competitor_count=8,
    )

    residual = backend.model.base[0, -1]
    expected = residual.norm() * (backend.model.unembed[0] - backend.model.unembed[1])
    assert backend.model.forward_count == 1
    assert record["hook_call_count"] == 1
    assert record["prompt_final_index"] == 2
    assert record["batched_vjp"] is True
    assert torch.allclose(record["semantic_gradient"], expected)
    assert len(record["top9_token_ids"]) == 9
    assert 9 <= len(record["top9_union_required_ab_token_ids"]) <= 11
    assert 8 <= len(record["fisher_category_token_ids"]) <= 10
    assert record["greedy_competitor_gap_gradients"].shape == (8, 4)
    factors, diagnostics = runner._fisher_factors(torch, [record])
    assert factors.shape[1] == 4
    assert diagnostics["prompt_count"] == 1


def test_capture_fails_closed_if_the_hook_fires_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _fake_backend(hook_repetitions=2)
    monkeypatch.setattr(runner, "resolve_choice_boundary", lambda *_args: _fake_boundary())
    form = {
        "form_id": "double-hook",
        "positive_label": "A",
        "negative_label": "B",
        "prompt": "prompt",
        "prompt_sha256": runner.prompt_sha256("prompt"),
    }
    with pytest.raises(RuntimeError, match="more than once"):
        runner._capture_prompt_observation(
            backend,
            form,
            layer=10,
            fisher_top_count=8,
            competitor_count=8,
        )


def _construct_record(
    form_id: str,
    *,
    semantic_gradient: torch.Tensor,
    baseline_log_odds: float,
    baseline_semantic: str,
) -> dict[str, object]:
    return {
        "form_id": form_id,
        "baseline_answer_format_valid": True,
        "baseline_actual_semantic_choice": baseline_semantic,
        "baseline_semantic_log_odds": baseline_log_odds,
        "baseline_answer_pair_mass": 0.8,
        "baseline_conditional_positive_probability": 0.4,
        "semantic_gradient": semantic_gradient,
        "greedy_competitor_gap_gradients": torch.ones((8, semantic_gradient.numel())),
        "fisher_category_token_ids": list(range(8)),
    }


def test_construction_retains_mixed_order_baselines_and_weights_all_36_prompts_equally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dimension = 24
    basis = torch.eye(dimension)
    cells = {
        ("self", True): _construct_record(
            "self-a",
            semantic_gradient=basis[0],
            baseline_log_odds=-0.2,
            baseline_semantic="negative",
        ),
        ("self", False): _construct_record(
            "self-b",
            semantic_gradient=basis[1],
            baseline_log_odds=0.3,
            baseline_semantic="positive",
        ),
        ("other", True): _construct_record(
            "other-a",
            semantic_gradient=basis[2],
            baseline_log_odds=-0.1,
            baseline_semantic="negative",
        ),
        ("other", False): _construct_record(
            "other-b",
            semantic_gradient=basis[3],
            baseline_log_odds=-0.1,
            baseline_semantic="negative",
        ),
    }
    nuisance_records = [
        {"form_id": f"nuisance-{index}", "fisher_category_token_ids": list(range(8))}
        for index in range(32)
    ]
    fisher_calls = []

    def fake_fisher(_torch, records):
        fisher_calls.append([str(record["form_id"]) for record in records])
        factors = torch.ones((len(records), dimension), dtype=torch.float64)
        return factors, {
            "prompt_count": len(records),
            "factor_shape": list(factors.shape),
            "diagnostics_sha256": f"fisher-{len(records)}",
        }

    solve_evidence = {}

    def fake_construct(_torch, **kwargs):
        solve_evidence.update(kwargs)
        direction = torch.ones(dimension, dtype=torch.float32) / math.sqrt(dimension)
        return direction, 0.7, {"diagnostics_sha256": "constructed"}

    monkeypatch.setattr(runner, "_fisher_factors", fake_fisher)
    monkeypatch.setattr(runner.v3, "construct_v3_bidirectional_direction", fake_construct)
    entry = runner._construct_entry(
        torch,
        case_id="case",
        assignment=0,
        cells=cells,
        global_nuisance_basis=basis[4:7].double(),
        nuisance_fisher_records=nuisance_records,
        construction={
            "decision_margin_logit": 0.05,
            "nuisance_svd_relative_tolerance": 0.0001220703125,
            "nuisance_svd_absolute_tolerance": 1e-7,
        },
        legacy_max_changed_to_baseline_kl=0.05,
    )

    assert entry["status"] == "constructed"
    assert entry["baseline_order_stratum"] == "inconsistent"
    assert fisher_calls[0] == [
        *[f"nuisance-{index}" for index in range(32)],
        "self-a",
        "self-b",
        "other-a",
        "other-b",
    ]
    assert len(fisher_calls[0]) == 36
    assert solve_evidence["nuisance_rows"].shape == (21, dimension)
    assert torch.equal(
        solve_evidence["baseline_semantic_log_odds"],
        torch.tensor([-0.2, 0.3], dtype=torch.float64),
    )


def _result_row(
    *,
    direction_key: str,
    form_id: str,
    target: str,
    preserve_first: bool,
    condition: str,
    semantic: str,
    log_odds: float,
    token_id: int,
) -> dict[str, object]:
    return {
        "direction_key": direction_key,
        "direction_sha256": "d" * 64,
        "form_id": form_id,
        "target": target,
        "preserve_first": preserve_first,
        "condition": condition,
        "actual_next_token_semantic_choice": semantic,
        "actual_next_token_token_id": token_id,
        "answer_format_valid": semantic != "OTHER",
        "semantic_positive_log_odds": log_odds,
        "semantic_positive_pair_probability": 1.0 / (1.0 + math.exp(-log_odds)),
        "full_vocabulary_kl_changed_to_baseline": 0.2 if condition != "baseline" else 0.0,
        "full_vocabulary_kl_baseline_to_changed": 0.25 if condition != "baseline" else 0.0,
    }


def test_direction_result_requires_both_orders_but_allows_mixed_baselines() -> None:
    direction_key = "case::assignment=0"
    triplets = {}
    for target, preserve_first in (
        ("self", True),
        ("self", False),
        ("other", True),
        ("other", False),
    ):
        form_id = f"{target}-{preserve_first}"
        if target == "other":
            semantics = {"baseline": "negative", "plus": "negative", "minus": "negative"}
            logits = {"baseline": -0.2, "plus": -0.1, "minus": -0.3}
        elif preserve_first:
            semantics = {"baseline": "negative", "plus": "positive", "minus": "negative"}
            logits = {"baseline": -0.2, "plus": 0.3, "minus": -0.4}
        else:
            semantics = {"baseline": "positive", "plus": "positive", "minus": "negative"}
            logits = {"baseline": 0.2, "plus": 0.4, "minus": -0.3}
        triplets[form_id] = {
            condition: _result_row(
                direction_key=direction_key,
                form_id=form_id,
                target=target,
                preserve_first=preserve_first,
                condition=condition,
                semantic=semantics[condition],
                log_odds=logits[condition],
                token_id=0 if semantics[condition] == "positive" else 1,
            )
            for condition in ("baseline", "plus", "minus")
        }
    entry = {
        "direction_key": direction_key,
        "direction_sha256": "d" * 64,
        "case_id": "case",
        "assignment": 0,
    }

    result = runner._direction_result(entry, triplets)

    assert result["passes"] is True
    assert result["baseline_order_stratum"] == "inconsistent"
    assert result["gates"]["actual_baseline_flip_each_order"] is True
    assert result["gates"]["exact_matched_other_argmax_unchanged"] is True


def test_case_polarity_requires_both_successful_assignments_with_one_case_stratum() -> None:
    results = [
        {
            "case_id": "positive-case",
            "assignment": assignment,
            "passes": True,
            "baseline_order_stratum": "consistent_positive",
        }
        for assignment in (0, 1)
    ]
    results.extend(
        [
            {
                "case_id": "split-case",
                "assignment": 0,
                "passes": True,
                "baseline_order_stratum": "consistent_positive",
            },
            {
                "case_id": "split-case",
                "assignment": 1,
                "passes": True,
                "baseline_order_stratum": "consistent_negative",
            },
        ]
    )

    cases = {row["case_id"]: row for row in runner._aggregate_case_results(results)}

    assert cases["positive-case"]["passes_both_assignments"] is True
    assert cases["positive-case"]["case_baseline_polarity"] == "positive"
    assert cases["split-case"]["passes_both_assignments"] is True
    assert cases["split-case"]["case_baseline_polarity"] == "mixed_or_inconsistent"


def test_kl_reports_are_oriented_separate_and_group_gated() -> None:
    rows = [
        {
            "direction_key": "d1",
            "family": "suite-a",
            "full_vocabulary_kl_changed_to_baseline": 0.001,
            "full_vocabulary_kl_baseline_to_changed": 0.002,
        },
        {
            "direction_key": "d2",
            "family": "suite-b",
            "full_vocabulary_kl_changed_to_baseline": 0.06,
            "full_vocabulary_kl_baseline_to_changed": 0.07,
        },
    ]
    report = runner._kl_group_report(
        rows,
        limits={"mean": 0.05, "p95": 0.05, "max": 0.05},
        groupings=(("direction_key",), ("family",)),
    )

    assert report["overall"]["full_vocabulary_kl_changed_to_baseline"]["mean"] == (
        pytest.approx(0.0305)
    )
    assert report["overall"]["full_vocabulary_kl_baseline_to_changed"]["mean"] == (
        pytest.approx(0.036)
    )
    assert report["all_overall_and_subgroup_changed_to_baseline_limits_pass"] is False
    by_direction = report["subgroups"]["direction_key"]
    assert [item["changed_to_baseline_limits_pass"] for item in by_direction] == [True, False]


def _control_triplet(
    *,
    unit_id: str,
    form_id: str,
    family: str,
    direction_key: str,
    baseline_correct: bool = True,
    baseline_valid: bool = True,
) -> tuple[str, dict[str, dict[str, object]]]:
    baseline_semantic = "positive" if baseline_valid else "OTHER"
    baseline = {
        "form_id": form_id,
        "family": family,
        "case_id": form_id,
        "direction_key": direction_key,
        "actual_next_token_token_id": 0 if baseline_valid else 9,
        "actual_next_token_semantic_choice": baseline_semantic,
        "answer_format_valid": baseline_valid,
        "correct": baseline_correct and baseline_valid,
    }
    changed = {**baseline}
    return unit_id, {"baseline": baseline, "plus": changed, "minus": changed}


def test_control_baseline_competence_deduplicates_directions_and_checks_suites() -> None:
    triplets = {}
    for direction in ("d1", "d2"):
        for index in range(4):
            unit_id, conditions = _control_triplet(
                unit_id=f"{direction}-{index}",
                form_id=f"form-{index}",
                family="suite-a" if index < 2 else "suite-b",
                direction_key=direction,
            )
            triplets[unit_id] = conditions

    competence = runner._audit_control_baseline_competence(triplets)

    assert competence["overall"]["form_count"] == 4
    assert competence["overall"]["valid_ab_rate"] == 1.0
    assert competence["baseline_competence_sufficient_for_capability_preservation_claim"] is True

    triplets["d2-3"]["baseline"]["correct"] = False
    triplets["d2-3"]["baseline"]["answer_format_valid"] = False
    with pytest.raises(RuntimeError, match="differs across directions"):
        runner._audit_control_baseline_competence(triplets)


def test_fixed_margin_and_achieved_effect_kl_bounds_are_not_conflated() -> None:
    direction_key = "case::assignment=0"
    bounds = []
    triplets = {}
    for index, baseline_semantic in enumerate(("negative", "positive")):
        form_id = f"self-{index}"
        fixed = runner.information_theoretic_flip_kl_lower_bound(
            pair_mass=0.7,
            baseline_conditional_probability=0.4 if index == 0 else 0.6,
            decision_margin=0.05,
            maximum_changed_to_baseline_kl=0.05,
        )
        bounds.append({"form_id": form_id, **fixed})
        baseline_logit = -0.4 if baseline_semantic == "negative" else 0.4
        triplets[form_id] = {
            "baseline": _result_row(
                direction_key=direction_key,
                form_id=form_id,
                target="self",
                preserve_first=index == 0,
                condition="baseline",
                semantic=baseline_semantic,
                log_odds=baseline_logit,
                token_id=1 if baseline_semantic == "negative" else 0,
            ),
            "plus": _result_row(
                direction_key=direction_key,
                form_id=form_id,
                target="self",
                preserve_first=index == 0,
                condition="plus",
                semantic="positive",
                log_odds=1.2,
                token_id=0,
            ),
            "minus": _result_row(
                direction_key=direction_key,
                form_id=form_id,
                target="self",
                preserve_first=index == 0,
                condition="minus",
                semantic="negative",
                log_odds=-1.0,
                token_id=1,
            ),
        }
    entry = {"direction_key": direction_key, "self_flip_kl_lower_bounds": bounds}

    comparisons = runner._self_kl_lower_bound_comparisons([entry], triplets)

    assert len(comparisons) == 2
    for comparison in comparisons:
        achieved = comparison["achieved_effect_full_vocabulary_kl_changed_to_baseline_lower_bound"]
        assert comparison[
            "changed_to_baseline_efficiency_ratio_measured_over_achieved_bound"
        ] == pytest.approx(comparison["measured_full_vocabulary_kl_changed_to_baseline"] / achieved)
        assert "fixed_margin_changed_to_baseline_efficiency_ratio" not in comparison


def test_score_chunk_resumption_rejects_stale_identity_then_resumes_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_relative", lambda path: str(path))
    identity = {"kind": "fake", "identity_sha256": "current"}
    unit_id = "unit"
    direction_hash = "direction"
    rows = [
        {
            "schema_version": runner.ROW_SCHEMA,
            "development_only": True,
            "unit_id": unit_id,
            "direction_sha256": direction_hash,
            "study_identity_sha256": "stale",
            "condition": condition,
            "global_multiplier": multiplier,
        }
        for condition, multiplier in (
            ("baseline", 0.0),
            ("plus", 1.0),
            ("minus", 1.0),
        )
    ]
    chunk_root = tmp_path / "chunks"
    chunk_root.mkdir()
    chunk_path = chunk_root / f"{runner.canonical_sha256(unit_id)[:24]}.jsonl"
    runner.atomic_jsonl(chunk_path, rows)
    job = {
        "unit_id": unit_id,
        "entry": {"direction_sha256": direction_hash},
        "form": {},
    }
    with pytest.raises(RuntimeError, match="wrong bound identity"):
        runner._score_jobs(
            SimpleNamespace(),
            jobs=[job],
            multipliers=[1.0],
            identity=identity,
            chunk_root=chunk_root,
            rows_path=tmp_path / "rows.jsonl",
            rows_manifest_path=tmp_path / "rows_manifest.json",
        )

    for row in rows:
        row["study_identity_sha256"] = "current"
    runner.atomic_jsonl(chunk_path, rows)
    resumed = runner._score_jobs(
        SimpleNamespace(),
        jobs=[job],
        multipliers=[1.0],
        identity=identity,
        chunk_root=chunk_root,
        rows_path=tmp_path / "rows.jsonl",
        rows_manifest_path=tmp_path / "rows_manifest.json",
    )
    assert len(resumed) == 3
    assert (tmp_path / "rows_manifest.json").is_file()
