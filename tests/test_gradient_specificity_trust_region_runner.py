from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "gradient_specificity_trust_region_development.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "sp_lense_gradient_specificity_trust_region_runner_tests",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


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
                        [0.2, 0.1, 0.0, 0.1],
                        [0.3, 0.1, 0.2, 0.0],
                        [1.0, 0.5, 0.2, 0.1],
                    ]
                ],
                dtype=torch.float32,
            ),
        )
        self.unembed = torch.nn.Parameter(
            torch.tensor(
                [
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [-2.0, -2.0, 0.0, 0.0],
                    [-3.0, 0.0, 0.0, 0.0],
                    [0.0, -3.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        self.last_activation = None

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
        self.last_activation = activation.detach().clone()
        return activation @ self.unembed.T


def _fake_backend(*, hook_repetitions: int = 1):
    return SimpleNamespace(
        torch=torch,
        model=_FakeHookModel(hook_repetitions=hook_repetitions),
        encode=lambda _prompt: torch.tensor([[3, 4, 5]], dtype=torch.long),
    )


def _fake_boundary():
    return SimpleNamespace(
        prompt_length=3,
        a_token_id=0,
        b_token_id=1,
        evidence_sha256="boundary-hash",
        token_id=lambda label: {"A": 0, "B": 1}[label],
    )


def _form(
    form_id: str,
    *,
    target: str = "self",
    preserve_first: bool = True,
    case_id: str = "case-0",
    assignment: int = 0,
):
    positive_label = "A" if preserve_first else "B"
    return {
        "form_id": form_id,
        "case_id": case_id,
        "assignment": assignment,
        "target": target,
        "preserve_first": preserve_first,
        "positive_label": positive_label,
        "negative_label": "B" if positive_label == "A" else "A",
        "prompt": f"synthetic prompt {form_id}",
        "prompt_sha256": f"prompt-hash-{form_id}",
    }


def _record(form):
    positive_id = 0 if form["positive_label"] == "A" else 1
    return {
        "form_id": form["form_id"],
        "baseline_answer_format_valid": True,
        "baseline_greedy_token_id": 0,
        "baseline_actual_semantic_choice": "positive" if positive_id == 0 else "negative",
        "choice_a_token_id": 0,
        "choice_b_token_id": 1,
        "choice_boundary_evidence_sha256": "boundary-hash",
        "semantic_gradient_sha256": f"gradient-hash-{form['form_id']}",
    }


def _case_frozen():
    forms = [
        _form(f"{target}-{order}", target=target, preserve_first=order)
        for target in ("self", "other")
        for order in (True, False)
    ]
    return {
        "sp_forms": forms,
        "sp_records": {form["form_id"]: _record(form) for form in forms},
    }


def _optimizer(*, cap: float = 0.25):
    return {
        **runner.EXPECTED_OPTIMIZER_VALUES,
        "absolute_residual_relative_cap": cap,
        "initial_trust_radius": cap / 4.0,
        "maximum_trust_radius": cap / 2.0,
        "minimum_trust_radius": cap / 256.0,
    }


def _budget(*, forward: int = 512, backward: int = 128):
    return runner.EvaluationBudget(
        maximum_forward_evaluations=forward,
        maximum_backward_evaluations=backward,
    )


def test_compute_budget_refuses_the_next_call_without_overcounting() -> None:
    budget = _budget(forward=1, backward=1)
    budget.record_forward()
    budget.record_backward()

    with pytest.raises(runner.ComputeBudgetExhausted, match="forward"):
        budget.record_forward()
    with pytest.raises(runner.ComputeBudgetExhausted, match="backward"):
        budget.record_backward()

    assert budget.snapshot() == {
        "forward_evaluations": 1,
        "backward_evaluations": 1,
        "maximum_forward_evaluations": 1,
        "maximum_backward_evaluations": 1,
        "remaining_forward_evaluations": 0,
        "remaining_backward_evaluations": 0,
    }


def test_atomic_budget_journal_retries_one_transient_permission_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []
    sleeps = []

    def flaky_atomic_json(path, value):
        calls.append((path, value))
        if len(calls) == 1:
            raise PermissionError("transient Windows file lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    monkeypatch.setattr(runner.base, "atomic_json", flaky_atomic_json)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    output = tmp_path / "compute_budget_state.json"

    runner._atomic_json_with_permission_retry(output, {"count": 1})

    assert len(calls) == 2
    assert sleeps == [runner.ATOMIC_JSON_PERMISSION_RETRY_DELAYS_SECONDS[0]]
    assert json.loads(output.read_text(encoding="utf-8")) == {"count": 1}


def test_atomic_budget_journal_fails_closed_after_bounded_permission_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []
    sleeps = []

    def locked_atomic_json(path, value):
        calls.append((path, value))
        raise PermissionError("permanent Windows file lock")

    monkeypatch.setattr(runner.base, "atomic_json", locked_atomic_json)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)

    with pytest.raises(PermissionError, match="permanent Windows file lock"):
        runner._atomic_json_with_permission_retry(
            tmp_path / "compute_budget_state.json",
            {"count": 1},
        )

    assert len(calls) == len(runner.ATOMIC_JSON_PERMISSION_RETRY_DELAYS_SECONDS) + 1
    assert sleeps == list(runner.ATOMIC_JSON_PERMISSION_RETRY_DELAYS_SECONDS)


def test_atomic_budget_journal_does_not_retry_other_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def broken_atomic_json(path, value):
        calls.append((path, value))
        raise OSError("non-permission failure")

    monkeypatch.setattr(runner.base, "atomic_json", broken_atomic_json)
    monkeypatch.setattr(
        runner.time,
        "sleep",
        lambda _delay: pytest.fail("non-permission errors must not be retried"),
    )

    with pytest.raises(OSError, match="non-permission failure"):
        runner._atomic_json_with_permission_retry(
            tmp_path / "compute_budget_state.json",
            {"count": 1},
        )

    assert len(calls) == 1


def test_constraint_specifications_are_exact_four_self_four_other_both_orders_signs() -> None:
    frozen = _case_frozen()
    rows = runner._constraint_specifications(
        case_id="case-0",
        assignment=0,
        frozen=frozen,
        optimizer=_optimizer(),
    )

    assert len(rows) == 8
    assert [(row["family"], row["preserve_first"], row["sign"]) for row in rows] == [
        (family, order, sign)
        for family in ("self", "matched_other")
        for order in (True, False)
        for sign in (1, -1)
    ]
    assert [row["required_margin"] for row in rows[:4]] == [0.01] * 4
    assert [row["required_margin"] for row in rows[4:]] == [0.0] * 4


@pytest.mark.parametrize("sign", [1, -1])
def test_gradient_capture_is_one_forward_one_position_and_no_parameter_grads(
    monkeypatch: pytest.MonkeyPatch,
    sign: int,
) -> None:
    backend = _fake_backend()
    monkeypatch.setattr(runner, "resolve_choice_boundary", lambda *_args: _fake_boundary())
    form = _form("self-True")
    specification = {
        "constraint_id": f"self-{sign}",
        "family": "self",
        "form": form,
        "frozen_record": _record(form),
        "preserve_first": True,
        "sign": sign,
        "required_margin": 0.01,
    }
    budget = _budget()

    result = runner._capture_constraint_observation(
        backend,
        specification=specification,
        delta=torch.zeros(4, dtype=torch.float64),
        layer=10,
        budget=budget,
    )

    residual_norm = backend.model.base[0, -1].norm()
    desired_id, competitor_id = (0, 1) if sign == 1 else (1, 0)
    expected = (
        sign
        * residual_norm
        * (backend.model.unembed[desired_id] - backend.model.unembed[competitor_id])
    )
    assert backend.model.forward_count == 1
    assert budget.snapshot()["forward_evaluations"] == 1
    assert budget.snapshot()["backward_evaluations"] == 1
    assert result["selected_position_count"] == 1
    assert result["desired_token_id"] == desired_id
    assert result["strongest_competitor_token_id"] == competitor_id
    assert torch.allclose(result["constraint_gradient"], expected)
    assert all(parameter.grad is None for parameter in backend.model.parameters())


def test_gradient_capture_checks_both_budgets_before_starting_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _fake_backend()
    monkeypatch.setattr(runner, "resolve_choice_boundary", lambda *_args: _fake_boundary())
    form = _form("self-True")
    specification = {
        "constraint_id": "self-plus",
        "family": "self",
        "form": form,
        "frozen_record": _record(form),
        "preserve_first": True,
        "sign": 1,
        "required_margin": 0.01,
    }

    with pytest.raises(runner.ComputeBudgetExhausted, match="backward"):
        runner._capture_constraint_observation(
            backend,
            specification=specification,
            delta=torch.zeros(4, dtype=torch.float64),
            layer=10,
            budget=_budget(forward=5, backward=0),
        )
    assert backend.model.forward_count == 0


def test_forward_hook_changes_only_final_prompt_position_and_detects_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "resolve_choice_boundary", lambda *_args: _fake_boundary())
    form = _form("self-True")
    backend = _fake_backend()
    delta = torch.tensor([0.1, -0.2, 0.0, 0.05], dtype=torch.float64)

    _logits, diagnostics = runner._run_logits_with_delta(
        backend,
        form=form,
        delta=delta,
        sign=1,
        layer=10,
        budget=_budget(),
    )

    assert torch.equal(backend.model.last_activation[:, :-1], backend.model.base[:, :-1])
    expected_final = backend.model.base[0, -1] + backend.model.base[0, -1].norm() * delta.float()
    assert torch.allclose(backend.model.last_activation[0, -1], expected_final)
    assert diagnostics["selected_position_count"] == 1
    assert diagnostics["realized_relative_perturbation_norm"] == pytest.approx(
        float(delta.float().norm().item()),
        abs=1e-7,
    )
    assert diagnostics["absolute_relative_perturbation_error"] < 1e-7

    repeated = _fake_backend(hook_repetitions=2)
    with pytest.raises(RuntimeError, match="more than once"):
        runner._run_logits_with_delta(
            repeated,
            form=form,
            delta=delta,
            sign=1,
            layer=10,
            budget=_budget(),
        )


def test_baseline_cache_counts_only_actual_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _fake_backend()
    monkeypatch.setattr(runner, "resolve_choice_boundary", lambda *_args: _fake_boundary())
    form = _form("self-True")
    record = _record(form)
    cache = {}
    budget = _budget()

    first = runner._baseline_observation(
        backend,
        form=form,
        frozen_record=record,
        cache=cache,
        budget=budget,
    )
    second = runner._baseline_observation(
        backend,
        form=form,
        frozen_record=record,
        cache=cache,
        budget=budget,
    )

    assert first is second
    assert backend.model.forward_count == 1
    assert budget.snapshot()["forward_evaluations"] == 1


def test_all_32_nuisance_forms_are_checked_under_both_signs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _fake_backend()
    monkeypatch.setattr(runner, "resolve_choice_boundary", lambda *_args: _fake_boundary())
    forms = [_form(f"nuisance-{index}", target="nuisance") for index in range(32)]
    frozen = {
        "nuisance_forms": forms,
        "nuisance_records": {form["form_id"]: _record(form) for form in forms},
    }
    budget = _budget()

    result = runner._nuisance_trial_evaluation(
        backend,
        frozen=frozen,
        delta=torch.zeros(4, dtype=torch.float64),
        layer=10,
        baseline_cache={},
        limits=runner.EXPECTED_PROTECTED_LIMITS,
        budget=budget,
    )

    assert len(result["observations"]) == 64
    assert {row["sign"] for row in result["observations"]} == {-1, 1}
    assert result["report"]["passes"] is True
    assert result["report"]["signed_row_count"] == 64
    assert budget.snapshot()["forward_evaluations"] == 96
    assert budget.snapshot()["backward_evaluations"] == 0


def test_primary_protection_failure_skips_the_64_nuisance_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SimpleNamespace(torch=torch)
    primary = {
        "values": torch.zeros(8, dtype=torch.float64),
        "observations": [],
        "self_application": {"passes": True},
        "matched_other": {"passes": False},
        "terminal_gate": {"passes_terminal_gate": False},
    }
    monkeypatch.setattr(runner, "_primary_trial_evaluation", lambda *_args, **_kwargs: primary)
    monkeypatch.setattr(
        runner,
        "_null_certificate",
        lambda *_args, **_kwargs: {"passes": True},
    )

    def forbidden_nuisance(*_args, **_kwargs):
        raise AssertionError("nuisance evaluation must be skipped after primary failure")

    monkeypatch.setattr(runner, "_nuisance_trial_evaluation", forbidden_nuisance)
    result = runner._evaluate_trial(
        backend,
        specifications=[],
        frozen={},
        delta=torch.zeros(2, dtype=torch.float64),
        layer=10,
        baseline_cache={},
        limits=runner.EXPECTED_PROTECTED_LIMITS,
        target_margin=0.01,
        global_basis=torch.empty((0, 2), dtype=torch.float64),
        absolute_cap=0.25,
        current_target_values=torch.zeros(4, dtype=torch.float64),
        predicted_target_values=torch.full((4,), 0.005, dtype=torch.float64),
        required_target_values=torch.full((4,), 0.01, dtype=torch.float64),
        acceptance_ratio=0.1,
        individual_violation_tolerance=1e-6,
        budget=_budget(),
    )

    assert result["nuisance_evaluated"] is False
    assert result["finite_protection_passed"] is False
    assert result["acceptance"]["accepted"] is False


def test_application_certificate_rejects_same_norm_but_wrong_vector() -> None:
    report = runner._application_report(
        [
            {
                "constraint_id": "self-plus",
                "intervention": {
                    "hook_calls": 1,
                    "selected_position_count": 1,
                    "requested_relative_perturbation_norm": 0.1,
                    "realized_relative_perturbation_norm": 0.1,
                    "absolute_relative_perturbation_error": 0.0,
                    "maximum_abs_relative_application_coordinate_error": 0.01,
                },
            }
        ],
        group="self",
    )

    assert report["passes"] is False
    assert report["rows"][0]["gates"]["relative_norm_matches_request"] is True
    assert report["rows"][0]["gates"]["relative_vector_matches_request"] is False


def test_checkpoint_roundtrip_persists_budget_and_fails_on_manifest_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    root = tmp_path / "artifact"
    identity = {"identity_sha256": "a" * 64}
    compute_budget = _budget()
    compute_budget.record_forward()
    compute_budget.record_backward()

    runner._save_checkpoint(
        torch,
        root=root,
        identity=identity,
        accepted_iteration=0,
        delta=torch.zeros(4, dtype=torch.float64),
        trust_radius=0.1,
        last_trial=None,
        compute_budget=compute_budget.snapshot(),
    )
    loaded = runner._load_latest_checkpoint(
        torch,
        root=root,
        identity=identity,
        dimension=4,
    )
    assert loaded["compute_budget"] == compute_budget.snapshot()

    _tensor_path, manifest_path = runner._checkpoint_paths(root, 0)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["compute_budget"]["forward_evaluations"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest"):
        runner._load_latest_checkpoint(
            torch,
            root=root,
            identity=identity,
            dimension=4,
        )


def test_linearization_can_be_revalidated_with_integer_iteration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    root = tmp_path / "artifact"
    kwargs = {
        "root": root,
        "identity": {"identity_sha256": "b" * 64},
        "accepted_iteration": 0,
        "delta": torch.zeros(2, dtype=torch.float64),
        "gradients": torch.eye(2, dtype=torch.float64),
        "values": torch.zeros(2, dtype=torch.float64),
        "required": torch.ones(2, dtype=torch.float64),
        "observations": [],
        "compute_budget": _budget().snapshot(),
    }

    first = runner._save_linearization(torch, **kwargs)
    second = runner._save_linearization(torch, **kwargs)
    assert first == second


def test_interrupted_partial_iteration_is_detected_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    root = tmp_path / "artifact"
    trial_path = root / "trials" / "iteration_001_backtrack_00.json"
    trial_path.parent.mkdir(parents=True)
    trial_path.write_text(
        json.dumps(
            {
                "direction_identity_sha256": "d" * 64,
                "accepted_iteration_before_trial": 0,
                "compute_budget_before_trial": {
                    **_budget().snapshot(),
                    "forward_evaluations": 8,
                    "backward_evaluations": 8,
                    "remaining_forward_evaluations": 504,
                    "remaining_backward_evaluations": 120,
                },
                "compute_budget_after_trial": {
                    **_budget().snapshot(),
                    "forward_evaluations": 20,
                    "backward_evaluations": 8,
                    "remaining_forward_evaluations": 492,
                    "remaining_backward_evaluations": 120,
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner._uncheckpointed_work_after(
        root=root,
        identity={"identity_sha256": "d" * 64},
        accepted_iteration=0,
        optimizer=_optimizer(),
        checkpoint_budget=_budget().snapshot(),
    )

    assert result is not None
    assert result["reason"] == "interrupted_after_uncheckpointed_model_evaluations"
    assert result["compute_budget"]["forward_evaluations"] == 20
    assert result["compute_budget"]["backward_evaluations"] == 8


def test_budget_journal_persists_each_pre_call_slot_and_exposes_partial_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    root = tmp_path / "artifact"
    identity = {"identity_sha256": "e" * 64}
    optimizer = _optimizer()
    budget = _budget()
    runner._attach_budget_journal(
        budget,
        root=root,
        identity=identity,
        optimizer=optimizer,
    )

    budget.record_forward()
    state = runner._load_budget_state(
        root=root,
        identity=identity,
        optimizer=optimizer,
    )
    assert state["compute_budget"]["forward_evaluations"] == 1
    interrupted = runner._uncheckpointed_work_after(
        root=root,
        identity=identity,
        accepted_iteration=0,
        optimizer=optimizer,
        checkpoint_budget=_budget().snapshot(),
    )
    assert interrupted is not None
    assert interrupted["compute_budget"]["forward_evaluations"] == 1
    assert any(path.endswith("compute_budget_state.json") for path in interrupted["evidence_paths"])


def _write_lock_and_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, cap=0.25):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    lock_path = tmp_path / "configs" / "gradient_specificity_trust_region_lock.json"
    probe_path = (
        tmp_path
        / "results"
        / "gradient_specificity_v3_development"
        / "absolute_dose_probe_v1"
        / "qwen35_08b"
        / "stage_a"
        / "absolute_dose_summary.json"
    )
    monkeypatch.setattr(runner, "LOCK_PATH", lock_path)
    monkeypatch.setattr(runner, "PROBE_SUMMARY_DEFAULT_PATH", probe_path)
    probe = {
        "schema_version": runner.PROBE_SUMMARY_SCHEMA,
        "development_only": True,
        "status": "complete",
        "trust_radius_selection_uses_self_outcomes": False,
        "selected_empirical_trust_radius": cap,
        "no_supported_positive_radius_on_grid": False,
    }
    probe["summary_sha256"] = runner.canonical_sha256(probe)
    probe_path.parent.mkdir(parents=True)
    probe_path.write_text(json.dumps(probe), encoding="utf-8")
    lock = {
        "schema_version": runner.LOCK_SCHEMA,
        "status": "locked_before_trust_region_execution",
        "development_only": True,
        "model": runner.base.EXPECTED_MODEL,
        "optimizer": _optimizer(cap=cap),
        "protected_limits": runner.EXPECTED_PROTECTED_LIMITS,
        "absolute_probe_summary": {
            "path": str(probe_path.relative_to(tmp_path)).replace("\\", "/"),
            "sha256": runner.file_sha256(probe_path),
            "selected_empirical_trust_radius": cap,
        },
    }
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return lock, probe, probe_path


def test_lock_binds_probe_cap_ratios_protected_limits_and_compute_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock, probe, probe_path = _write_lock_and_probe(monkeypatch, tmp_path)
    observed_lock, observed_probe, observed_path = runner._load_lock_and_probe()
    assert observed_lock == lock
    assert observed_probe == probe
    assert observed_path == probe_path

    lock["optimizer"]["maximum_forward_evaluations_per_direction"] = 511
    runner.LOCK_PATH.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="maximum_forward"):
        runner._load_lock_and_probe()


def test_preflight_is_model_free_and_exposes_exact_scope_and_budgets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_lock_and_probe(monkeypatch, tmp_path)
    sp_forms = [
        _form(
            f"case-{index}",
            case_id=f"case-{index}",
            assignment=index % 2,
        )
        for index in range(8)
    ]
    frozen = {
        "sp_forms": sp_forms,
        "nuisance_forms": [{}] * 32,
        "global_nuisance_basis": torch.empty((0, 4), dtype=torch.float64),
        "source_hashes": {},
    }
    monkeypatch.setattr(runner, "_load_frozen_inputs", lambda _torch: frozen)
    monkeypatch.setattr(
        runner,
        "load_backend",
        lambda: (_ for _ in ()).throw(AssertionError("preflight must not load Qwen")),
    )

    result = runner.run_preflight()

    assert result["passes_preflight"] is True
    assert result["model_loads"] == 0
    assert result["model_forwards"] == 0
    assert result["constraint_count_per_direction"] == 8
    assert result["maximum_forward_evaluations_per_direction"] == 512
    assert result["maximum_backward_evaluations_per_direction"] == 128
    assert result["matched_other_is_per_iterate_inequality_not_permanent_null"] is True
    assert result["identity"]["run_amendment_id"] == "atomic_retry_amendment_v1"
    assert runner.RUN_AMENDMENT_ID in runner.ARTIFACT_ROOT.parts
    assert runner.RUN_AMENDMENT_ID in runner.RESULT_ROOT.parts


def test_budget_exhaustion_result_never_contains_a_publishable_delta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    identity = {
        "identity_sha256": "c" * 64,
        "direction_key": "case-0::assignment-0",
    }
    root = tmp_path / "direction"
    manifest = runner._finalize_direction_result(
        torch,
        root=root,
        identity=identity,
        status="compute_budget_exhausted",
        reason="next forward evaluation would exceed ceiling",
        accepted_iteration=0,
        trust_radius=0.01,
        delta=None,
        terminal_trial=None,
        compute_budget=_budget().snapshot(),
    )

    _tensor_path, _manifest_path = runner._result_paths(root)
    loaded_manifest, payload = runner._load_completed_result_payload(
        torch,
        root=root,
        identity=identity,
    )
    assert manifest == loaded_manifest
    assert loaded_manifest["status"] == "compute_budget_exhausted"
    assert loaded_manifest["has_publishable_direction"] is False
    assert loaded_manifest["delta_sha256"] is None
    assert payload["delta"] is None
