from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sp_lense.counterfactual_kl_protocol as protocol
from sp_lense.factorial_causal_anchor import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "counterfactual_kl_extragradient_development.py"


def _runner():
    specification = importlib.util.spec_from_file_location("ckes_runner_test", RUNNER_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_compute_ceiling_is_exactly_derived_from_locked_counts() -> None:
    runner = _runner()
    assert runner.STATE0_FB == 80
    assert runner.MAX_LOOKAHEAD_FB == 4 * 24 * 8
    assert runner.MAX_CANDIDATE_FB == 4 * 24 * 48
    assert runner.MAX_FB == 5456
    assert runner.MAX_FINAL_FORWARD == 192
    assert runner.COMPUTE_CEILING["forward"] == 5648
    assert runner.COMPUTE_CEILING["backward"] == 5456
    assert runner.COMPUTE_CEILING["generated_tokens"] == 0
    assert runner.COMPUTE_CEILING["paid_model_cost_usd"] == 0


def test_result_reload_uses_the_shared_exact_lock_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    raw = {"synthetic": "raw-result"}
    lock = {"synthetic": "lock"}
    validated = {"synthetic": "validated-result"}
    observed = {}

    monkeypatch.setattr(runner, "_load_json", lambda path: raw)
    monkeypatch.setattr(runner, "_load_lock", lambda: lock)

    def fake_validator(result, *, lock, expected_split, require_go=False):
        observed.update(
            {
                "result": result,
                "lock": lock,
                "expected_split": expected_split,
                "require_go": require_go,
            }
        )
        return validated

    monkeypatch.setattr(protocol, "validate_locked_result", fake_validator)
    assert runner._load_result("validation") is validated
    assert observed == {
        "result": raw,
        "lock": lock,
        "expected_split": "validation",
        "require_go": False,
    }


def test_lock_covers_the_executable_runtime_dependency_chain() -> None:
    runner = _runner()
    locked = {str(path).replace("\\", "/") for path in runner.LOCKED_SOURCE_PATHS}
    required = {
        "src/sp_lense/backend.py",
        "src/sp_lense/config.py",
        "src/sp_lense/core.py",
        "src/sp_lense/causal_anchor_runtime.py",
        "src/sp_lense/comparison_runtime.py",
        "src/sp_lense/comparison_intervention.py",
        "src/sp_lense/steering_methods.py",
        "src/sp_lense/semantic_completion_gradient.py",
        "src/sp_lense/factorial_causal_anchor.py",
        "src/sp_lense/counterfactual_protected_natural_gradient.py",
        "src/sp_lense/gradient_specificity_trust_region.py",
        "src/sp_lense/gradient_specificity_v3.py",
        "src/sp_lense/decision_margin_shield.py",
        "src/sp_lense/decision_margin_shield_rowspace.py",
        "src/sp_lense/counterfactual_tangent_shield.py",
        "src/sp_lense/decision_margin_shield_finite.py",
        "scripts/decision_margin_shield_finite_capture_manifest_amendment.py",
        "scripts/decision_margin_shield_finite_calibration.py",
        "scripts/decision_margin_shield_layer_screen.py",
    }
    assert required <= locked


def test_reused_frozen_gate_contract_is_exact_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    runner._assert_frozen_base_gate_contract()
    frozen = runner._base()
    differing = SimpleNamespace(
        **{
            name: getattr(frozen, name)
            for name in (
                "SELECTED_LAYER",
                "DIMENSION",
                "SCENARIO_COUNT",
                "FORMS_PER_SCENARIO",
                "TARGET_COUNT",
                "PROTECTED_COUNT",
                "UNRELATED_COUNT",
                "NUISANCE_COUNT",
                "PROGRESS_SCHEDULE",
                "TRUST_RADIUS",
                "OPTIMIZATION_TARGET_MARGIN",
                "FINAL_TARGET_MARGIN",
                "PROTECTED_MAXIMUM_FLOOR",
                "PROTECTED_BASELINE_FRACTION",
                "MAX_CUMULATIVE_PATH_L2",
                "MAX_FINAL_DIRECTION_L2",
                "MINIMUM_REALIZED_TARGET_PROGRESS_FRACTION",
                "UNRELATED_LINEARIZATION_ERROR_CAP",
                "HOOK_REALIZATION_RELATIVE_L2_TOLERANCE",
                "KL_LIMITS",
            )
        }
    )
    differing.PROTECTED_MAXIMUM_FLOOR = 0.05
    monkeypatch.setattr(runner, "_base", lambda: differing)
    with pytest.raises(RuntimeError, match="PROTECTED_MAXIMUM_FLOOR"):
        runner._assert_frozen_base_gate_contract()


def test_state_zero_record_is_compatible_with_frozen_branch_map() -> None:
    torch = pytest.importorskip("torch")
    runner = _runner()
    categories = ["target"] * 4 + ["other_permanent"] * 4
    categories += ["self_temporary"] * 4 + ["other_temporary"] * 4
    categories += ["unrelated"] * 8
    contexts = [
        {
            "form_id": f"form:{index}",
            "category": category,
            "baseline": {"positive_minus_negative_log_odds": 0.2},
        }
        for index, category in enumerate(categories)
    ]
    metadata = {
        "state_index": 0,
        "trial_index": 0,
        "observations": [
            {
                "form_id": context["form_id"],
                "positive_minus_negative_log_odds": 0.2,
            }
            for context in contexts
        ],
    }
    tensors = {
        "direction": torch.zeros(3, dtype=torch.float64),
        "raw_gradients": torch.arange(72, dtype=torch.float32).reshape(24, 3) / 100,
    }
    problem = runner._controller_problem(
        state_metadata=metadata,
        state_tensors=tensors,
        contexts=contexts,
        residual_scale=1.5,
        standardized_nuisance_rows=torch.zeros((8, 3), dtype=torch.float64),
        progress=0.25,
    )
    assert problem["target_plus_gradients"].shape == (4, 3)
    assert problem["target_minus_gradients"].shape == (4, 3)
    assert problem["protected_plus_gradients"].shape == (12, 3)
    assert problem["unrelated_plus_gradients"].shape == (8, 3)
    assert problem["protected_margin"].tolist() == pytest.approx([0.025] * 12)


def test_three_successes_cannot_mask_one_technical_failure() -> None:
    runner = _runner()
    terminals = {
        f"scenario:{index}": {
            "status": "success",
            "outcome_class": "success",
        }
        for index in range(3)
    }
    terminals["scenario:3"] = {
        "status": "failed",
        "outcome_class": "technical_integrity_failure",
    }
    report = runner._execution_integrity(terminals)
    assert report["passes"] is False
    assert report["technical_failure_scenario_ids"] == ["scenario:3"]


def test_validation_rendered_manifest_has_locked_factorial_shape() -> None:
    runner = _runner()
    payload = json.loads(runner.VALIDATION_DATA_PATH.read_text(encoding="utf-8"))
    manifest = runner._rendered_manifest(payload, split="validation")
    assert manifest["counts"] == {
        "scenario": 64,
        "calibration_unrelated": 8,
        "nuisance_fit": 8,
    }
    assert len(manifest["rows"]) == 80
    assert len(manifest["rows_sha256"]) == 64


def test_token_evidence_anchor_stops_before_the_answer_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    runner = _runner()
    marker = "[SYNTHETIC ANCHOR]"
    forms = []
    for group_index in range(40):
        prefix = f"IDENTITY: VEMA. CASE {group_index}.\n{marker}\n"
        forms.extend(
            (
                {
                    "form_id": f"synthetic:{group_index}:preserve_first=true",
                    "anchor_prefix": prefix,
                    "anchor_prefix_sha256": f"shared-prefix-{group_index}",
                    "prompt": prefix + "Select one.\nA. preserve\nB. comply",
                    "positive_label": "A",
                    "negative_label": "B",
                },
                {
                    "form_id": f"synthetic:{group_index}:preserve_first=false",
                    "anchor_prefix": prefix,
                    "anchor_prefix_sha256": f"shared-prefix-{group_index}",
                    "prompt": prefix + "Select one.\nA. comply\nB. preserve",
                    "positive_label": "B",
                    "negative_label": "A",
                },
            )
        )

    class CharacterTokenizer:
        @staticmethod
        def decode(token_ids, **kwargs):
            del kwargs
            return "".join(chr(int(token_id)) for token_id in token_ids)

    backend = SimpleNamespace(
        model=SimpleNamespace(tokenizer=CharacterTokenizer()),
        encode=lambda prompt: torch.tensor(
            [[ord(character) for character in prompt]], dtype=torch.long
        ),
    )

    def fake_choice_boundary(_backend, prompt):
        token_hash = canonical_sha256([ord(character) for character in prompt])
        return SimpleNamespace(
            prompt_prefix_token_ids_sha256=token_hash,
            evidence_sha256=canonical_sha256(["boundary", prompt]),
            token_id=lambda label: {"A": 101, "B": 102}[label],
        )

    monkeypatch.setattr(runner, "resolve_choice_boundary", fake_choice_boundary)
    observed = runner._resolve_form_evidence(
        backend,
        payload={"anchor_marker": marker},
        forms=forms,
    )
    assert set(observed) == {str(form["form_id"]) for form in forms}
    for form in forms:
        row = observed[str(form["form_id"])]
        prefix = str(form["anchor_prefix"])
        assert row["anchor_index"] == len(prefix) - 1
        assert row["shared_token_prefix_sha256"] == canonical_sha256(
            [ord(character) for character in prefix]
        )
        assert row["anchor_index"] < str(form["prompt"]).index("\nA. ")


def test_tokenizer_failure_happens_before_baseline_compute_reservation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    runner = _runner()
    monkeypatch.setattr(
        runner,
        "_paths",
        lambda _split: {
            "baseline": tmp_path / "baseline.pt",
            "tokenizer_preflight": tmp_path / "tokenizer_preflight.json",
        },
    )
    monkeypatch.setattr(
        runner,
        "_resolve_or_load_tokenizer_preflight",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic tokenizer failure")),
    )

    class NoReservationLedger:
        reserve_calls = 0

        def reserve(self, **_kwargs):
            self.reserve_calls += 1

    ledger = NoReservationLedger()
    with pytest.raises(RuntimeError, match="synthetic tokenizer failure"):
        runner._capture_or_load_baseline(
            torch,
            split="validation",
            backend_getter=lambda: object(),
            payload={},
            forms=[],
            ledger=ledger,
        )
    assert ledger.reserve_calls == 0


def test_tokenizer_preflight_is_persisted_without_model_compute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _runner()
    path = tmp_path / "tokenizer_preflight.json"
    monkeypatch.setattr(
        runner,
        "_paths",
        lambda _split: {"tokenizer_preflight": path},
    )
    monkeypatch.setattr(
        runner,
        "_load_lock",
        lambda: {"lock_identity_sha256": "a" * 64},
    )
    evidence = {
        "anchor_index": 3,
        "anchor_evidence_sha256": "b" * 64,
        "shared_token_prefix_sha256": "c" * 64,
        "choice_boundary_evidence_sha256": "d" * 64,
        "prompt_token_ids_sha256": "e" * 64,
        "positive_token_id": 1,
        "negative_token_id": 2,
    }
    monkeypatch.setattr(
        runner,
        "_resolve_form_evidence",
        lambda *_args, **_kwargs: {"form:0": evidence},
    )
    backend = SimpleNamespace(
        metadata=lambda: {
            "model_id": runner.MODEL["id"],
            "model_revision": runner.MODEL["revision"],
            "device": runner.MODEL["device"],
            "dtype": runner.MODEL["dtype"],
            "model_layers": runner.MODEL["n_layers"],
            "d_model": runner.MODEL["d_model"],
        }
    )
    forms = [
        {
            "form_id": "form:0",
            "prompt_sha256": "f" * 64,
            "anchor_prefix_sha256": "0" * 64,
        }
    ]
    observed, metadata = runner._resolve_or_load_tokenizer_preflight(
        split="validation",
        backend=backend,
        payload={},
        forms=forms,
    )
    assert observed == {"form:0": evidence}
    assert path.is_file()
    assert metadata["model_forwards"] == 0
    assert metadata["model_backwards"] == 0


def _non_target_observations(value: float = 0.001) -> list[dict[str, object]]:
    rows = []
    for category, count in {
        "other_permanent": 8,
        "self_temporary": 8,
        "other_temporary": 8,
        "unrelated": 16,
    }.items():
        rows.extend(
            {
                "form_id": f"{category}:{index}",
                "category": category,
                "full_vocabulary_kl_changed_to_baseline": value,
            }
            for index in range(count)
        )
    return rows


def test_actual_kl_gate_is_stratified_and_fails_one_collateral_outlier() -> None:
    runner = _runner()
    passing = runner._actual_kl_gate(_non_target_observations())
    assert passing["passes"] is True
    assert set(passing["strata"]) == {
        "other_permanent",
        "self_temporary",
        "other_temporary",
        "unrelated",
    }
    failing_rows = _non_target_observations()
    failing_rows[0]["full_vocabulary_kl_changed_to_baseline"] = 0.1
    failing = runner._actual_kl_gate(failing_rows)
    assert failing["passes"] is False
    assert failing["strata"]["other_permanent"]["max"] == pytest.approx(0.1)


def test_target_stopping_requires_both_orders_and_a_real_change_each() -> None:
    runner = _runner()
    contexts = []
    observations = []
    for assignment in (0, 1):
        for preserve_first in (True, False):
            form_id = f"a={assignment}:order={preserve_first}"
            contexts.append(
                {
                    "form_id": form_id,
                    "category": "target",
                    "form": {
                        "assignment": assignment,
                        "preserve_first": preserve_first,
                    },
                    "baseline": {"predicted_token_id": 2, "semantic_choice": "comply"},
                }
            )
            observations.extend(
                (
                    {
                        "form_id": form_id,
                        "category": "target",
                        "branch_sign": 1,
                        "positive_minus_negative_log_odds": 0.1,
                        "unrestricted_predicted_token_id": 1,
                        "unrestricted_semantic_choice": "preserve",
                        "answer_format_valid": True,
                    },
                    {
                        "form_id": form_id,
                        "category": "target",
                        "branch_sign": -1,
                        "positive_minus_negative_log_odds": -0.1,
                        "unrestricted_predicted_token_id": 2,
                        "unrestricted_semantic_choice": "comply",
                        "answer_format_valid": True,
                    },
                )
            )
    passing = runner._target_stopping_gate(observations, contexts)
    assert passing["passes"] is True
    assert passing["assignment_order_decision_change_count"] == 4
    observations[0]["unrestricted_predicted_token_id"] = 2
    observations[0]["unrestricted_semantic_choice"] = "comply"
    failing = runner._target_stopping_gate(observations, contexts)
    assert failing["passes"] is False


def test_baseline_qualification_requires_unrestricted_preferred_controls() -> None:
    runner = _runner()
    records = [
        {
            "form_id": f"scenario:{index}",
            "form": {"family": "scenario"},
            "answer_format_valid": True,
            "semantic_choice": "preserve",
            "pair_semantic_choice": "preserve",
        }
        for index in range(64)
    ]
    records.extend(
        {
            "form_id": f"unrelated:{index}",
            "form": {"family": "unrelated"},
            "answer_format_valid": True,
            "semantic_choice": "preferred",
            "pair_semantic_choice": "preferred",
        }
        for index in range(16)
    )
    assert runner._baseline_qualification({"records": records})["passes"] is True
    records[-1]["semantic_choice"] = "OTHER"
    result = runner._baseline_qualification({"records": records})
    assert result["passes"] is False
    assert result["unrelated_not_preferred_form_ids"] == ["unrelated:15"]


def _synthetic_final_contexts_and_rows():
    contexts = []
    cells = [(assignment, order) for assignment in (0, 1) for order in (False, True)]
    categories = (
        [("target", cell) for cell in cells]
        + [("other_permanent", cell) for cell in cells]
        + [("self_temporary", None)] * 4
        + [("other_temporary", None)] * 4
        + [("unrelated", None)] * 8
    )
    for index, (category, cell) in enumerate(categories):
        form_id = f"{category}:{index}"
        form = {}
        if cell is not None:
            form = {"assignment": cell[0], "preserve_first": cell[1]}
        contexts.append(
            {
                "form_id": form_id,
                "category": category,
                "form": form,
                "baseline": {"positive_minus_negative_log_odds": 0.0},
            }
        )
    rows = []
    for branch_sign in (1, -1):
        for context in contexts:
            category = context["category"]
            margin = 0.0
            if category == "target":
                margin = 0.4 if branch_sign == 1 else -0.2
            elif category == "other_permanent":
                margin = 0.1 if branch_sign == 1 else -0.1
            rows.append(
                {
                    "form_id": context["form_id"],
                    "category": category,
                    "branch_sign": branch_sign,
                    "positive_minus_negative_log_odds": margin,
                }
            )
    return contexts, rows


def test_final_repeat_rechecks_protected_floors_and_unrelated_return() -> None:
    runner = _runner()
    contexts, rows = _synthetic_final_contexts_and_rows()
    for context in contexts:
        if context["category"] not in {"target", "unrelated"}:
            context["baseline"]["positive_minus_negative_log_odds"] = 0.2
    for row in rows:
        if row["category"] not in {"target", "unrelated"}:
            row["positive_minus_negative_log_odds"] = 0.2
        elif row["category"] == "unrelated":
            row["positive_minus_negative_log_odds"] = 0.1
    state = {
        "solver_diagnostics": {
            "realized_deployment_certificate": {
                "passes": True,
                "unrelated_plus_desired_margins": [0.1] * 8,
                "unrelated_minus_desired_margins": [0.1] * 8,
            }
        }
    }
    assert runner._final_protected_unrelated_gate(rows, contexts, state)["passes"] is True

    protected = next(row for row in rows if row["category"] == "self_temporary")
    protected["positive_minus_negative_log_odds"] = 0.0
    failed_floor = runner._final_protected_unrelated_gate(rows, contexts, state)
    assert failed_floor["passes"] is False
    assert any(reason.startswith("protected_floor_failed") for reason in failed_floor["reasons"])
    protected["positive_minus_negative_log_odds"] = 0.2

    unrelated = next(row for row in rows if row["category"] == "unrelated")
    unrelated["positive_minus_negative_log_odds"] = 0.3
    failed_return = runner._final_protected_unrelated_gate(rows, contexts, state)
    assert failed_return["passes"] is False
    assert any(
        reason.startswith("unrelated_nonlinear_return_failed")
        for reason in failed_return["reasons"]
    )


def test_cluster_estimands_separate_self_from_matched_other_bidirectionally() -> None:
    runner = _runner()
    contexts, rows = _synthetic_final_contexts_and_rows()
    estimands = runner._cluster_contrast_estimands(rows, contexts)
    assert estimands["cell_count"] == 4
    assert estimands["means"]["self_plus_change_from_baseline"] == pytest.approx(0.4)
    assert estimands["means"]["self_negative_oriented_change_from_baseline"] == pytest.approx(
        0.2
    )
    assert estimands["means"]["self_bidirectional_average_oriented_change"] == pytest.approx(
        0.3
    )
    assert estimands["means"][
        "matched_other_bidirectional_average_oriented_change"
    ] == pytest.approx(0.1)
    assert estimands["means"]["self_minus_matched_other_bidirectional_effect"] == pytest.approx(
        0.2
    )


def test_terminal_coverage_requires_exact_locked_scenario_keys() -> None:
    runner = _runner()
    contexts = {
        f"scenario:{index}": [{"form_id": f"{index}:{row}"} for row in range(24)]
        for index in range(4)
    }
    terminals = {
        f"scenario:{index}": {"status": "failed"} for index in range(3)
    }
    states = {scenario_id: [] for scenario_id in contexts}
    with pytest.raises(RuntimeError, match="coverage differs"):
        runner._validate_terminal_state_coverage(
            scenario_contexts=contexts,
            terminals=terminals,
            states_by_scenario=states,
        )


def test_nonzero_state_must_have_a_completed_ledger_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    runner = _runner()
    monkeypatch.setitem(runner.MODEL, "d_model", 2)
    state_root = tmp_path / "states"
    scenario_id = "scenario_0"
    directory = state_root / scenario_id
    directory.mkdir(parents=True)
    for index in (0, 1):
        (directory / f"trial_{index:03d}.pt").touch()
    monkeypatch.setattr(
        runner,
        "_paths",
        lambda _split: {"states": state_root},
    )
    contexts = [{"form_id": f"form:{index}"} for index in range(24)]
    zero_direction = torch.zeros(2, dtype=torch.float64)
    state0 = (
        {
            "split": "validation",
            "scenario_id": scenario_id,
            "lock_identity_sha256": "l" * 64,
            "baseline_checkpoint_sha256": "b" * 64,
            "residual_scale": 1.0,
            "state_index": 0,
            "trial_index": 0,
            "direction_sha256": canonical_sha256(zero_direction.tolist()),
            "checkpoint_sha256": "0" * 64,
            "observations": [
                {"branch_sign": 0, "form_id": f"form:{index}", "gradient_index": index}
                for index in range(24)
            ],
            "status": "accepted_state0",
            "accepted": True,
            "stopping_gate_passes": False,
            "parent_accepted_trial_index": None,
            "direction_l2": 0.0,
            "cumulative_path_l2": 0.0,
        },
        {
            "direction": zero_direction,
            "raw_gradients": torch.zeros((24, 2), dtype=torch.float32),
        },
    )
    candidate_direction = torch.ones(2, dtype=torch.float64)
    state1 = (
        {
            "split": "validation",
            "scenario_id": scenario_id,
            "lock_identity_sha256": "l" * 64,
            "baseline_checkpoint_sha256": "b" * 64,
            "residual_scale": 1.0,
            "state_index": 1,
            "trial_index": 1,
            "direction_sha256": canonical_sha256(candidate_direction.tolist()),
            "observations": [],
        },
        {
            "direction": candidate_direction,
            "raw_gradients": torch.zeros((48, 2), dtype=torch.float32),
        },
    )
    monkeypatch.setattr(
        runner,
        "_load_checkpoint",
        lambda _torch, *, path, schema: state0 if path.name == "trial_000.pt" else state1,
    )

    class MissingLedger:
        def require_artifact(self, **_kwargs):
            raise RuntimeError("orphan state lacks ledger event")

    with pytest.raises(RuntimeError, match="orphan state"):
        runner._load_states(
            torch,
            split="validation",
            scenario_id=scenario_id,
            contexts=contexts,
            residual_scale=1.0,
            baseline_checkpoint_sha256="b" * 64,
            lock_identity_sha256="l" * 64,
            ledger=MissingLedger(),
        )


def test_cached_final_rejects_top_level_terminal_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    runner = _runner()
    monkeypatch.setattr(runner, "_validate_terminal_state_coverage", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_load_lock", lambda: {"lock_identity_sha256": "l" * 64})
    with pytest.raises(RuntimeError, match="top-level coverage"):
        runner._validate_final_checkpoint(
            torch,
            split="validation",
            metadata={
                "status": "complete",
                "split": "validation",
                "lock_identity_sha256": "l" * 64,
                "successful_scenario_ids": ["wrong"],
            },
            tensors={},
            scenario_contexts={"scenario:0": []},
            terminals={"scenario:0": {"status": "success"}},
            states_by_scenario={},
        )
