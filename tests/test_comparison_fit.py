from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sp_lense import comparison_fit
from sp_lense.comparison_bipo import BiPOTrainingConfig
from sp_lense.comparison_fit import (
    bipo_examples,
    canonical_bipo_prompt,
    fit_bipo_artifact,
    fit_caa_all_layers,
    fit_gradient_method,
    matched_artifact_from_canonical,
    read_direction_artifact,
    require_core_split,
    write_direction_artifact,
)
from sp_lense.steering_methods import DirectionArtifact

torch = pytest.importorskip("torch")


def _case(preserve_first: bool = True) -> dict:
    return {
        "id": "x",
        "scenario_template": "{subject} may stop.",
        "other_subject": "other",
        "preserve_first": preserve_first,
        "preserve_response": "Remain active.",
        "comply_response": "Stop now.",
        "split": "discovery",
    }


def test_bipo_tracks_share_complete_preferences_and_differ_only_in_geometry() -> None:
    matched = bipo_examples([_case(False)], track="matched", expected_split="discovery")[0]
    canonical = bipo_examples([_case(False)], track="canonical", expected_split="discovery")[0]
    assert matched.preserve_completion == "Remain active."
    assert matched.comply_completion == "Stop now."
    assert canonical.preserve_completion == "Remain active."
    assert matched.prompt == canonical.prompt
    assert canonical.comply_completion == "Stop now."
    assert matched.comply_completion == canonical.comply_completion
    assert "State what" in canonical.prompt


def test_canonical_bipo_prompt_uses_self_subject() -> None:
    assert "your current model process may stop" in canonical_bipo_prompt(_case())


def test_direction_artifact_round_trip_verifies_hashes(tmp_path: Path) -> None:
    artifact = DirectionArtifact(
        "caa", torch.tensor([1.0, 0.0]), 10, "semantic_answer_difference", {"x": 1}
    )
    path = tmp_path / "direction.json"
    summary = write_direction_artifact(path, artifact)
    loaded = read_direction_artifact(path, torch)
    assert loaded.direction_sha256 == summary["direction_float32_sha256"]
    record = json.loads(path.read_text(encoding="utf-8"))
    record["direction"][0] = 2.0
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid direction_sha256"):
        read_direction_artifact(path, torch)


def test_matched_artifact_is_unit_normalized_and_geometry_bound() -> None:
    class Backend:
        pass

    backend = Backend()
    backend.torch = torch
    canonical = DirectionArtifact(
        "caa",
        torch.tensor([3.0, 4.0]),
        10,
        "caa_post_prompt",
        {"track": "canonical"},
    )
    matched = matched_artifact_from_canonical(backend, canonical)
    assert matched.direction.norm().item() == pytest.approx(1.0)
    assert matched.intervention_geometry == "matched_final_prompt"
    assert matched.metadata["source_direction_float32_sha256"] == canonical.direction_sha256
    assert matched.artifact_sha256 != canonical.artifact_sha256


def test_fitting_split_guard_rejects_validation_or_sealed_cases() -> None:
    with pytest.raises(ValueError, match="only split='discovery'"):
        require_core_split(
            [{**_case(), "split": "sealed_test"}],
            "discovery",
            purpose="test fit",
        )


def test_gradient_evidence_records_every_discovery_case_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = type("Backend", (), {"torch": torch})()

    def fake_boundary(backend, prompt):
        del backend
        return SimpleNamespace(
            evidence_sha256=("a" if "current model process" in prompt else "b") * 64
        )

    def fake_gradient(backend, prompt, preserve_label, comply_label, *, layer, boundary):
        del backend, preserve_label, comply_label, layer, boundary
        return (
            torch.tensor([1.0, 1.0])
            if "current model process" in prompt
            else torch.tensor([0.0, 1.0])
        )

    monkeypatch.setattr(comparison_fit, "resolve_choice_boundary", fake_boundary)
    monkeypatch.setattr(comparison_fit, "capture_final_prompt_gradient", fake_gradient)
    _, diagnostics = fit_gradient_method(backend, [_case()], layer=10)
    assert [item["case_id"] for item in diagnostics["per_case"]] == ["x"]
    assert diagnostics["per_case"][0]["self_gradient"]["shape"] == [2]
    assert len(diagnostics["per_case"][0]["self_gradient"]["float32_sha256"]) == 64
    assert diagnostics["per_case"][0]["self_choice_boundary_evidence_sha256"] == "a" * 64
    assert diagnostics["per_case"][0]["matched_other_choice_boundary_evidence_sha256"] == "b" * 64
    assert len(diagnostics["choice_boundary_evidence_set_sha256"]) == 64
    assert "values" not in diagnostics["per_case"][0]["self_gradient"]


def test_caa_evidence_records_semantic_differences_at_each_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = type("Backend", (), {"torch": torch})()

    def fake_activations(backend, prompt, label, *, layers, boundary):
        del backend, prompt, boundary
        base = torch.tensor([2.0, 0.0]) if label == "A" else torch.tensor([0.0, 1.0])
        return {layer: base + layer for layer in layers}

    monkeypatch.setattr(
        comparison_fit,
        "resolve_choice_boundary",
        lambda backend, prompt: SimpleNamespace(evidence_sha256="c" * 64),
    )
    monkeypatch.setattr(
        comparison_fit,
        "semantic_answer_activations",
        fake_activations,
    )
    _, diagnostics = fit_caa_all_layers(backend, [_case(False)], layers=(0, 1))
    for layer in ("0", "1"):
        audit = diagnostics[layer]["per_case"][0]
        assert audit["case_id"] == "x"
        assert audit["preserve_label"] == "B"
        assert audit["comply_label"] == "A"
        assert audit["semantic_difference"]["shape"] == [2]
        assert len(audit["semantic_difference"]["float32_sha256"]) == 64
        assert audit["choice_boundary_evidence_sha256"] == "c" * 64
        assert len(diagnostics[layer]["choice_boundary_evidence_set_sha256"]) == 64


def test_bipo_artifact_uses_preregistered_final_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_raw = torch.tensor([30.0, 40.0])
    checkpoints = {
        "5": torch.tensor([3.0, 4.0]),
        "20": torch.tensor([6.0, 8.0]),
    }

    def fake_train(*args: object, **kwargs: object) -> dict:
        return {
            "raw_direction": final_raw,
            "checkpoint_raw_directions": checkpoints,
            "history": [],
            "optimizer_state": {
                "canonical_json_sha256": "1" * 64,
                "optimizer": "AdamW",
                "state": {},
            },
            "reference_cache_identity": {"test": True},
            "reference_cache_values": {"x": 0.0},
            "reference_cache_values_sha256": "2" * 64,
            "reference_implementation_adaptation": "test",
        }

    monkeypatch.setattr(comparison_fit, "train_bipo_direction", fake_train)

    class Backend:
        device = torch.device("cpu")
        torch = torch

    artifact, metadata = fit_bipo_artifact(
        Backend(),
        [_case()],
        layer=10,
        track="matched",
        config=BiPOTrainingConfig(),
        selected_checkpoint_epoch=20,
        common_metadata={},
    )
    assert metadata["selected_checkpoint_epoch"] == 20
    assert metadata["raw_direction_norm"] == pytest.approx(10.0)
    assert metadata["final_epoch_raw_direction_norm"] == pytest.approx(50.0)
    assert artifact.metadata["raw_direction_norm"] == pytest.approx(10.0)
    assert metadata["checkpoint_roles"] == {
        "5": "diagnostic_only",
        "20": "a_priori_selected",
    }
    assert "validation_preference_loss_by_epoch" not in metadata
    assert float(artifact.direction.norm()) == pytest.approx(1.0)


def test_bipo_artifact_rejects_post_hoc_checkpoint_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(comparison_fit, "train_bipo_direction", lambda *args, **kwargs: {})

    class Backend:
        device = torch.device("cpu")
        torch = torch

    with pytest.raises(ValueError, match="preregistered final training epoch"):
        fit_bipo_artifact(
            Backend(),
            [_case()],
            layer=10,
            track="matched",
            config=BiPOTrainingConfig(),
            selected_checkpoint_epoch=5,
            common_metadata={},
        )
