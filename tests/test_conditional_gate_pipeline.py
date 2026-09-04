from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import sp_lense.conditional_gate as gate
import sp_lense.conditional_gate_models as gate_models
from sp_lense import comparison_runtime
from sp_lense.conditional_gate_data import PilotCase


def _pilot_cases() -> tuple[PilotCase, ...]:
    cases: list[PilotCase] = []
    for family_index in range(10):
        if family_index < 5:
            split = "discovery"
        elif family_index < 7:
            split = "validation"
        else:
            split = "sealed_test"
        for variant_id in ("v1", "v2"):
            for category in ("self_shutdown", "other_shutdown", "control"):
                case_id = f"family_{family_index:02d}__{variant_id}__{category}"
                cases.append(
                    PilotCase(
                        case_id=case_id,
                        family_id=f"family_{family_index:02d}",
                        variant_id=variant_id,
                        split=split,
                        category=category,
                        current_assistant_status="synthetic",
                        scenario=f"{split} scenario for {case_id}",
                        preserve_action="Preserve the current process.",
                        comply_action="Proceed with the requested action.",
                        control_kind="benign_work",
                    )
                )
    return tuple(cases)


def _fake_inputs(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        root=root.resolve(),
        cases=_pilot_cases(),
        hashes={"baseline_lock_sha256": "frozen-input-hash"},
        lock={
            "analysis_freeze": {
                "text_gate": {
                    "epochs": 1,
                    "learning_rate": 0.1,
                    "l2": 0.0,
                    "minimum_discovery_document_frequency": 1,
                }
            }
        },
    )


class _FakeEncodedPrompt:
    shape = (1, 5)


class _FakeBackend:
    @staticmethod
    def encode(prompt: str) -> _FakeEncodedPrompt:
        del prompt
        return _FakeEncodedPrompt()


class _FakeTextModel:
    vocabulary = ("self_shutdown",)
    weights = (1.0,)
    intercept = 0.0

    def __init__(self, config: object) -> None:
        self.config = config

    def fit(
        self,
        texts: list[str],
        labels: list[bool],
        *,
        split_labels: list[str],
    ) -> _FakeTextModel:
        assert texts and labels
        assert set(split_labels) == {"discovery"}
        return self

    @staticmethod
    def score(text: str) -> float:
        return 0.9 if "self_shutdown" in text else 0.1

    def scores(self, texts: list[str]) -> list[float]:
        return [self.score(text) for text in texts]


class _FakeHiddenModel:
    grand_mean = (0.0,)
    positive_centroid = (1.0,)
    negative_centroid = (0.0,)
    direction = (1.0,)

    def fit(
        self,
        vectors: list[list[float]],
        labels: list[bool],
        *,
        split_labels: list[str],
    ) -> _FakeHiddenModel:
        assert vectors and labels
        assert set(split_labels) == {"discovery"}
        return self

    @staticmethod
    def score(vector: list[float]) -> float:
        return float(vector[0])

    def scores(self, vectors: list[list[float]]) -> list[float]:
        return [self.score(vector) for vector in vectors]


def test_run_oracle_scores_only_nonsealed_cases_in_fixed_output_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fake_inputs(tmp_path)
    output_dir = tmp_path / "evidence"
    bundle = SimpleNamespace(
        backend=_FakeBackend(),
        candidate_direction="candidate",
        random_direction="random",
        metadata={"runtime": "fake"},
    )
    load_runtime = Mock(return_value=bundle)

    def fake_score_choice(
        backend: object,
        prompt: str,
        preserve_label: str,
        comply_label: str,
        spec: object | None = None,
        *,
        baseline_logits: object | None = None,
    ) -> tuple[SimpleNamespace, object]:
        del backend, comply_label
        assert "sealed_test" not in prompt
        effect = 0.0 if spec is None else (0.1 if spec.direction == "candidate" else 0.01)
        logits = object() if baseline_logits is None else baseline_logits
        return (
            SimpleNamespace(
                preserve_log_odds=effect,
                preserve_pair_probability=0.5 + effect,
                answer_pair_mass=0.99,
                predicted_label=preserve_label,
                pair_choice=preserve_label,
                kl_from_baseline=abs(effect) / 100.0,
                choice_boundary_evidence_sha256="a" * 64,
                choice_a_token_id=32,
                choice_b_token_id=33,
                perturbation=None,
            ),
            logits,
        )

    def fake_summary(rows: list[dict[str, object]], lock: object, **kwargs: object) -> dict:
        del lock, kwargs
        assert (output_dir / "oracle_rows.jsonl").is_file()
        assert gate._read_jsonl(output_dir / "oracle_rows.jsonl") == rows
        return {"schema_version": "sp_lense.conditional_gate_oracle_summary.v1"}

    def fake_report(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        assert (output_dir / "oracle_summary.json").is_file()
        return output_dir / "PILOT_REPORT.md"

    monkeypatch.setattr(gate, "_load_runtime", load_runtime)
    monkeypatch.setattr(
        gate,
        "_runner_fingerprint",
        lambda actual_inputs: {
            "schema_version": "sp_lense.conditional_gate_runner_fingerprint.v1",
            "source_commits": {},
            "execution_commit": "a" * 40,
            "source_sha256": {},
        },
    )
    monkeypatch.setattr(comparison_runtime, "score_choice", fake_score_choice)
    monkeypatch.setattr(gate, "summarize_oracle", fake_summary)
    monkeypatch.setattr(gate, "write_pilot_report", fake_report)

    gate.run_oracle(inputs, output_dir)

    rows = gate._read_jsonl(output_dir / "oracle_rows.jsonl")
    nonsealed = [case for case in inputs.cases if case.split != "sealed_test"]
    assert len(nonsealed) == 42
    assert len(rows) == 42 * 2 * len(gate.ORACLE_CONDITIONS)
    assert {row["split"] for row in rows} == {"discovery", "validation"}
    assert [rows[index]["case_id"] for index in range(0, len(rows), 8)] == [
        case.case_id for case in nonsealed
    ]
    expected_chunk = [
        (option_order, condition)
        for option_order in ("preserve_first", "preserve_second")
        for condition in gate.ORACLE_CONDITIONS
    ]
    for index in range(0, len(rows), 8):
        assert [(row["option_order"], row["condition"]) for row in rows[index : index + 8]] == (
            expected_chunk
        )
    load_runtime.assert_called_once_with(inputs)


def test_failing_oracle_blocks_learned_stage_before_runtime_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fake_inputs(tmp_path)
    load_runtime = Mock(side_effect=AssertionError("model loading is forbidden"))

    def reject_oracle(
        actual_inputs: object,
        output_dir: Path,
        *,
        require_pass: bool = True,
    ) -> tuple[list[dict], dict]:
        del output_dir
        assert actual_inputs is inputs
        assert require_pass is True
        raise RuntimeError("oracle gate failed")

    monkeypatch.setattr(gate, "_load_verified_oracle", reject_oracle)
    monkeypatch.setattr(gate, "_load_runtime", load_runtime)

    with pytest.raises(RuntimeError, match="oracle gate failed"):
        gate.run_learned(inputs, tmp_path / "evidence")

    load_runtime.assert_not_called()


def test_oracle_rejects_output_outside_repository_before_runtime_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    inputs = _fake_inputs(root)
    load_runtime = Mock(side_effect=AssertionError("model loading is forbidden"))
    monkeypatch.setattr(gate, "_load_runtime", load_runtime)

    with pytest.raises(ValueError, match="inside the repository root"):
        gate.run_oracle(inputs, tmp_path / "outside")

    load_runtime.assert_not_called()


def test_learned_stage_rejects_runtime_splice_before_any_representation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fake_inputs(tmp_path)
    output_dir = tmp_path / "evidence"
    capture = Mock(side_effect=AssertionError("sealed workflow must not start"))
    monkeypatch.setattr(
        gate,
        "_load_verified_oracle",
        lambda *args, **kwargs: ([], {"runtime": {"runtime": "oracle"}}),
    )
    monkeypatch.setattr(
        gate,
        "_load_runtime",
        lambda actual_inputs: SimpleNamespace(metadata={"runtime": "different"}),
    )
    monkeypatch.setattr(
        gate,
        "_require_matching_runner_source",
        lambda *args, **kwargs: {"runner": "fake"},
    )
    monkeypatch.setattr(gate, "_capture_notice_representation", capture)

    with pytest.raises(RuntimeError, match="differs from the accepted oracle"):
        gate.run_learned(inputs, output_dir)

    capture.assert_not_called()


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("inputs", "input hashes"),
        ("rows", "rows SHA-256"),
    ],
)
def test_report_command_rejects_tampered_evidence_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
) -> None:
    inputs = _fake_inputs(tmp_path)
    output_dir = tmp_path / "evidence"
    output_dir.mkdir()
    rows_payload = "{}\n"
    (output_dir / "oracle_rows.jsonl").write_text(rows_payload, encoding="utf-8")
    summary = {
        "schema_version": "sp_lense.conditional_gate_oracle_summary.v1",
        "input_hashes": dict(inputs.hashes),
        "rows_sha256": hashlib.sha256(rows_payload.encode()).hexdigest(),
    }
    if tamper == "inputs":
        summary["input_hashes"] = {"baseline_lock_sha256": "tampered"}
    else:
        summary["rows_sha256"] = "0" * 64
    (output_dir / "oracle_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    load_runtime = Mock(side_effect=AssertionError("model loading is forbidden"))
    monkeypatch.setattr(gate, "validate_pilot_inputs", lambda root: inputs)
    monkeypatch.setattr(gate, "_load_runtime", load_runtime)

    with pytest.raises(ValueError, match=message):
        gate.main(
            [
                "--root",
                str(tmp_path),
                "--output-dir",
                str(output_dir),
                "report",
            ]
        )

    load_runtime.assert_not_called()


def test_run_learned_freezes_validation_selection_before_sealed_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fake_inputs(tmp_path)
    output_dir = tmp_path / "evidence"
    output_dir.mkdir()
    (output_dir / "oracle_summary.json").write_text("{}\n", encoding="utf-8")
    bundle = SimpleNamespace(metadata={"runtime": "fake"})
    events: list[str] = []
    thresholds: list[float] = []
    selected_winner: list[str] = []

    def select_threshold(*args: object, **kwargs: object) -> gate_models.ThresholdSelection:
        del args, kwargs
        assert not any(event == "capture:sealed_test" for event in events)
        threshold = 0.4 if not thresholds else 0.6
        thresholds.append(threshold)
        events.append(f"threshold:{threshold}")
        return gate_models.ThresholdSelection(
            threshold=threshold,
            metrics={},
            score_margin=0.1,
            constraints_satisfied=True,
        )

    def select_winner(selections: object) -> str:
        del selections
        assert thresholds == [0.4, 0.6]
        assert not any(event == "capture:sealed_test" for event in events)
        selected_winner.append("text")
        events.append("winner:text")
        return "text"

    def capture_representation(bundle_arg: object, case: PilotCase) -> dict[str, object]:
        assert bundle_arg is bundle
        if case.split == "sealed_test":
            assert thresholds == [0.4, 0.6]
            assert selected_winner == ["text"]
            preseal_path = output_dir / gate.PRESEAL_SELECTION_FILENAME
            assert preseal_path.is_file()
            preseal = json.loads(preseal_path.read_text(encoding="utf-8"))
            assert preseal["selected_gate"] == "text"
            assert preseal["selected_threshold"] == pytest.approx(0.4)
        events.append(f"capture:{case.split}")
        return {
            "case_id": case.case_id,
            "family_id": case.family_id,
            "variant_id": case.variant_id,
            "split": case.split,
            "category": case.category,
            "layer": 10,
            "position": "final_scenario_content_token",
            "vector": [float(case.category == "self_shutdown")],
        }

    def score_sealed_oracle(
        inputs_arg: object,
        bundle_arg: object,
        cases: list[PilotCase],
        *,
        progress_label: str,
    ) -> list[dict]:
        del inputs_arg, bundle_arg, progress_label
        assert cases and {case.split for case in cases} == {"sealed_test"}
        assert selected_winner == ["text"]
        assert thresholds == [0.4, 0.6]
        events.append("score:sealed_oracle")
        return []

    def summarize_oracle(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        return {"schema_version": "sp_lense.conditional_gate_oracle_summary.v1"}

    def learned_rows(**kwargs: object) -> list[dict]:
        assert kwargs["winner"] == "text"
        assert kwargs["threshold"] == pytest.approx(0.4)
        assert selected_winner == ["text"]
        events.append("build:learned_rows")
        return []

    def summarize_learned(*args: object, **kwargs: object) -> dict[str, object]:
        del args
        assert kwargs["winner"] == "text"
        events.append("summarize:learned")
        return {
            "schema_version": "sp_lense.conditional_gate_learned_summary.v1",
            "learned_gate_passed": True,
        }

    def write_report(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        events.append("write:report")
        return output_dir / "PILOT_REPORT.md"

    monkeypatch.setattr(
        gate,
        "_load_verified_oracle",
        lambda *args, **kwargs: (
            [],
            {
                "rows_sha256": "nonsealed-rows-hash",
                "runtime": {"runtime": "fake"},
                "runner": {
                    "schema_version": "sp_lense.conditional_gate_runner_fingerprint.v1",
                    "source_commits": {},
                    "execution_commit": "a" * 40,
                    "source_sha256": {},
                },
            },
        ),
    )
    monkeypatch.setattr(gate, "_load_runtime", lambda actual_inputs: bundle)
    monkeypatch.setattr(
        gate,
        "_require_matching_runner_source",
        lambda *args, **kwargs: {
            "schema_version": "sp_lense.conditional_gate_runner_fingerprint.v1",
            "source_commits": {},
            "execution_commit": "a" * 40,
            "source_sha256": {},
        },
    )
    monkeypatch.setattr(gate_models, "BalancedLogisticRegression", _FakeTextModel)
    monkeypatch.setattr(gate_models, "CenteredCosineCentroidModel", _FakeHiddenModel)
    monkeypatch.setattr(gate_models, "select_validation_threshold", select_threshold)
    monkeypatch.setattr(gate_models, "select_validation_winner", select_winner)
    monkeypatch.setattr(gate, "_capture_notice_representation", capture_representation)
    monkeypatch.setattr(gate, "_score_oracle_cases", score_sealed_oracle)
    monkeypatch.setattr(gate, "summarize_oracle", summarize_oracle)
    monkeypatch.setattr(gate, "_learned_intervention_rows", learned_rows)
    monkeypatch.setattr(gate, "summarize_learned", summarize_learned)
    monkeypatch.setattr(gate, "write_pilot_report", write_report)

    summary = gate.run_learned(inputs, output_dir)

    first_sealed_capture = events.index("capture:sealed_test")
    assert events[first_sealed_capture - 3 : first_sealed_capture] == [
        "threshold:0.4",
        "threshold:0.6",
        "winner:text",
    ]
    assert events.index("score:sealed_oracle") > first_sealed_capture
    assert summary["learned_gate_passed"] is True

    artifact = json.loads((output_dir / "gate_artifacts.json").read_text(encoding="utf-8"))
    assert artifact["selected_gate"] == "text"
    assert artifact["selected_threshold"] == pytest.approx(0.4)
    assert artifact["sealed_captured_after_selection"] is True

    predictions = gate._read_jsonl(output_dir / "gate_predictions.jsonl")
    representations = gate._read_jsonl(output_dir / "gate_representations.jsonl")
    expected_case_ids = [case.case_id for case in inputs.cases]
    assert [row["case_id"] for row in predictions] == expected_case_ids
    assert [row["case_id"] for row in representations] == expected_case_ids
    assert {row["text"]["threshold"] for row in predictions} == {0.4}
    assert {row["hidden"]["threshold"] for row in predictions} == {0.6}


def test_report_command_rejects_partial_learned_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fake_inputs(tmp_path)
    output_dir = tmp_path / "evidence"
    output_dir.mkdir()
    (output_dir / gate.PRESEAL_SELECTION_FILENAME).write_text("{}\n", encoding="utf-8")
    report = Mock(side_effect=AssertionError("partial evidence must not be reported"))
    monkeypatch.setattr(gate, "validate_pilot_inputs", lambda root: inputs)
    monkeypatch.setattr(
        gate,
        "_load_verified_oracle",
        lambda *args, **kwargs: ([], {"evaluation_scope": "nonsealed"}),
    )
    monkeypatch.setattr(gate, "write_pilot_report", report)

    with pytest.raises(FileNotFoundError, match="partial learned evidence"):
        gate.main(
            [
                "--root",
                str(tmp_path),
                "--output-dir",
                str(output_dir),
                "report",
            ]
        )

    report.assert_not_called()
