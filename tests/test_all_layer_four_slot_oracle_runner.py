import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sp_lense.all_layer_four_slot_oracle import (
    ALFSIntegrityError,
    frozen_nuisance_rowspace,
)
from sp_lense.factorial_causal_anchor import canonical_sha256, tensor_float32_sha256


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "all_layer_four_slot_oracle_screen.py"
    spec = importlib.util.spec_from_file_location("_test_alfs_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rehash(value, field):
    result = json.loads(json.dumps(value))
    result.pop(field, None)
    result[field] = canonical_sha256(result)
    return result


def test_proposed_lock_binds_exact_source_runtime_thresholds_and_paths(runner):
    lock = runner.proposed_lock()
    assert lock["compute_ceiling"] == {
        "model_forwards": 80,
        "model_backwards": 80,
        "generated_tokens": 0,
        "external_api_calls": 0,
        "external_model_judges": 0,
        "paid_model_cost_usd": 0,
    }
    assert lock["configuration"]["local_only_environment_required_before_backend_load"] == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    assert lock["configuration"]["source_binding"]["csms_capture_file_sha256"] == (
        runner.file_sha256(runner.CSMS_CAPTURE_PATH)
    )
    paths = [record["path"] for record in lock["file_hashes"].values()]
    assert len(paths) == len(set(paths)) == len(runner.LOCKED_PATHS)
    assert "tests/test_all_layer_four_slot_oracle_runner.py" in paths
    assert runner.verify_lock(lock, verify_files=True) == lock


def test_lock_rejects_rehashed_threshold_source_and_path_tampering(runner):
    lock = runner.proposed_lock()
    changed = json.loads(json.dumps(lock))
    changed["thresholds"]["qualification_cap"] = 0.251
    changed = _rehash(changed, "lock_identity_sha256")
    with pytest.raises(ALFSIntegrityError, match="contract"):
        runner.verify_lock(changed, verify_files=False)

    changed = json.loads(json.dumps(lock))
    changed["configuration"]["source_binding"]["csms_capture_file_sha256"] = "0" * 64
    changed = _rehash(changed, "lock_identity_sha256")
    with pytest.raises(ALFSIntegrityError, match="configuration"):
        runner.verify_lock(changed, verify_files=False)

    changed = json.loads(json.dumps(lock))
    changed["file_hashes"]["locked_00"]["path"] = changed["file_hashes"][
        "locked_01"
    ]["path"]
    changed = _rehash(changed, "lock_identity_sha256")
    with pytest.raises(ALFSIntegrityError, match="coverage"):
        runner.verify_lock(changed, verify_files=False)


def test_every_command_refuses_sealed_before_artifact_access(runner):
    commands = (
        runner.run_lock,
        runner.run_preflight,
        runner.run_capture,
        runner.run_analyze_training,
        runner.run_analyze_held,
        runner.run_analyze_full,
        runner.run_analyze,
        runner.run_report,
    )
    for command in commands:
        with pytest.raises(ALFSIntegrityError, match="sealed access"):
            command("sealed_test")


def test_capture_refuses_existing_outputs_and_nonoffline_environment(
    runner, monkeypatch, tmp_path
):
    monkeypatch.setattr(runner, "_load_lock", lambda: {"lock_identity_sha256": "a" * 64})
    monkeypatch.setattr(runner, "CAPTURE_PATH", tmp_path / "capture.pt")
    monkeypatch.setattr(runner, "CAPTURE_COMPLETE_PATH", tmp_path / "complete.json")
    monkeypatch.setattr(runner, "CAPTURE_RESERVATION_PATH", tmp_path / "reservation.json")
    runner.CAPTURE_COMPLETE_PATH.write_text("already complete", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run_capture()
    runner.CAPTURE_COMPLETE_PATH.unlink()
    for name in runner.OFFLINE_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ALFSIntegrityError, match="offline"):
        runner.run_capture()


def test_lock_refuses_overwrite_without_building_a_replacement(runner, monkeypatch, tmp_path):
    path = tmp_path / "lock.json"
    path.write_text("immutable", encoding="utf-8")
    monkeypatch.setattr(runner, "LOCK_PATH", path)
    monkeypatch.setattr(
        runner,
        "proposed_lock",
        lambda: pytest.fail("overwrite refusal must happen before lock construction"),
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run_lock()


def test_capture_alignment_binds_order_layer0_and_all_layer_hashes(runner):
    residuals = torch.arange(80 * 24 * 4 * 2, dtype=torch.float32).reshape(80, 24, 4, 2)
    gradients = residuals / 100.0
    source = []
    csms = []
    capture = []
    for index in range(80):
        source.append(
            {
                "form_id": f"form_{index}",
                "tensor_index": index,
                "row_sha256": f"{index:064x}"[-64:],
                "prompt_token_ids_sha256": f"{index + 4:064x}"[-64:],
                "positive_minus_negative_log_odds": float(index),
                "full_logits_float32_sha256": f"{index + 5:064x}"[-64:],
                "form": {"prompt_sha256": f"{index + 1:064x}"[-64:]},
            }
        )
        csms.append(
            {
                "form_id": f"form_{index}",
                "tensor_index": index,
                "row_sha256": f"{index + 2:064x}"[-64:],
                "slot_indices": [3, 10, 14, 18],
                "residuals_float32_sha256": tensor_float32_sha256(residuals[index, 0]),
                "gradients_float32_sha256": tensor_float32_sha256(gradients[index, 0]),
            }
        )
        capture.append(
            {
                "form_id": f"form_{index}",
                "tensor_index": index,
                "row_sha256": f"{index + 3:064x}"[-64:],
                "prompt_token_ids_sha256": f"{index + 4:064x}"[-64:],
                "positive_minus_negative_log_odds": float(index),
                "full_logits_float32_sha256": f"{index + 5:064x}"[-64:],
                "slot_indices": [3, 10, 14, 18],
                "residuals_float32_sha256": tensor_float32_sha256(residuals[index]),
                "gradients_float32_sha256": tensor_float32_sha256(gradients[index]),
            }
        )
    manifest = runner._capture_alignment(
        source_records=source,
        csms_records=csms,
        capture_records=capture,
        residuals=residuals,
        gradients=gradients,
    )
    assert manifest["row_count"] == 80
    reordered = capture.copy()
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ALFSIntegrityError, match="misaligned"):
        runner._capture_alignment(
            source_records=source,
            csms_records=csms,
            capture_records=reordered,
            residuals=residuals,
            gradients=gradients,
        )


def test_held_phase_requires_immutable_training_barrier(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "_load_lock", lambda: {"lock_identity_sha256": "a" * 64})
    monkeypatch.setattr(runner, "TRAINING_COMPLETE_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(runner, "HELD_ROOT", tmp_path / "held")
    monkeypatch.setattr(runner, "HELD_COMPLETE_PATH", tmp_path / "held_complete.json")
    with pytest.raises(ALFSIntegrityError, match="completed training"):
        runner.run_analyze_held()


def test_global_qualification_can_reach_go_when_every_positive_gate_passes(runner):
    training = [
        {"passes": True, "selection": {"passes": True, "selected_layer": 7}}
        for _ in range(4)
    ]
    held = [{"passes": True} for _ in range(4)]
    layers, checks, passes = runner._global_qualification(
        training,
        held,
        {"passes": True, "selected_layer": 7},
    )
    assert layers == [7, 7, 7, 7]
    assert checks["sealed_data_not_accessed"] is True
    assert passes is True


def test_held_artifact_is_bound_to_exact_frozen_training_selection(
    runner, tmp_path
):
    training_path = tmp_path / "training.json"
    training_path.write_text("frozen training bytes", encoding="utf-8")
    training = {"fold": {"fold_index": 0}, "selection": {"selected_layer": 7}}
    held = {
        "training_fold_file_sha256": runner.file_sha256(training_path),
        "fold": training["fold"],
        "selection": training["selection"],
    }
    runner._verify_held_training_binding(held, training, training_path)
    changed = json.loads(json.dumps(held))
    changed["selection"]["selected_layer"] = 8
    with pytest.raises(ALFSIntegrityError, match="frozen training"):
        runner._verify_held_training_binding(changed, training, training_path)


def test_training_computation_masks_every_held_numeric_row(runner, monkeypatch):
    residuals = np.ones((80, 24, 4, 2), dtype=np.float32)
    gradients = np.ones_like(residuals)
    residuals[60:] = 12345.0
    gradients[60:] = -12345.0
    fold = {
        "training_all_indices": list(range(60)),
        "training_target_indices": list(range(12)),
        "training_nuisance_indices": list(range(12, 60)),
    }

    def fake_scales(values, indices):
        assert tuple(indices) == tuple(range(60))
        assert np.isnan(values[60:]).all()
        return np.ones((24, 4), dtype=np.float64)

    def fake_layer(**kwargs):
        layer_residuals = kwargs["residuals_at_layer"]
        layer_gradients = kwargs["gradients_at_layer"]
        assert np.isnan(layer_residuals[60:]).all()
        assert np.isnan(layer_gradients[60:]).all()
        nuisance = (
            layer_gradients[12:60] * np.ones((1, 4, 1), dtype=np.float64)
        ).reshape(48, -1)
        _, basis_record = frozen_nuisance_rowspace(nuisance)
        layer = kwargs["layer"]
        return (
            {
                "layer": layer,
                "eligible": layer == 5,
                "worst_primary_minimum_norm": 0.1 if layer == 5 else None,
                "mean_primary_minimum_norm": 0.1 if layer == 5 else None,
                "training_nuisance_rowspace": basis_record,
            },
            {},
        )

    monkeypatch.setattr(runner, "training_only_slot_scales", fake_scales)
    monkeypatch.setattr(runner, "analyze_training_layer", fake_layer)
    training, basis = runner._compute_training_fold(
        records=[{} for _ in range(80)],
        residuals=residuals,
        gradients=gradients,
        fold=fold,
    )
    assert training["selection"]["selected_layer"] == 5
    assert training["selected_layer_detail"] is training["layer_candidates"][5]
    assert basis is not None


def test_selected_detail_must_equal_candidate_used_by_selector(runner):
    candidates = [
        {
            "layer": layer,
            "eligible": layer == 7,
            "worst_primary_minimum_norm": 0.1 if layer == 7 else None,
            "mean_primary_minimum_norm": 0.1 if layer == 7 else None,
        }
        for layer in range(24)
    ]
    artifact = {
        "layer_candidates": candidates,
        "selection": runner.select_layer(candidates),
        "selected_layer_detail": candidates[7],
    }
    runner._verify_selected_candidate_binding(artifact)
    changed = json.loads(json.dumps(artifact))
    changed["selected_layer_detail"]["mean_primary_minimum_norm"] = 0.11
    with pytest.raises(ALFSIntegrityError, match="differs from candidate"):
        runner._verify_selected_candidate_binding(changed)


def test_json_writer_refuses_nonfinite_values(runner, tmp_path):
    path = tmp_path / "nonfinite.json"
    with pytest.raises(ValueError, match="Out of range float"):
        runner._write_new_json(path, {"undefined": float("inf")})
    assert not path.exists()


def test_persisted_training_basis_is_loaded_by_exact_bytes(
    runner, monkeypatch, tmp_path
):
    monkeypatch.setattr(runner, "TRAINING_ROOT", tmp_path / "training")
    monkeypatch.setattr(runner, "CAPTURE_PATH", tmp_path / "capture.pt")
    runner.CAPTURE_PATH.write_bytes(b"bound capture")
    training_path = runner._fold_training_path(0)
    training_path.parent.mkdir(parents=True)
    training_path.write_text("bound training", encoding="utf-8")
    basis = np.zeros((2, 4096), dtype=np.float64)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    scales = np.ones(4, dtype=np.float64)
    artifact = {
        "lock_identity_sha256": "a" * 64,
        "capture_checkpoint_sha256": "b" * 64,
        "selection": {"selected_layer": 7},
        "training_only_slot_scales": np.ones((24, 4)).tolist(),
        "selected_layer_detail": {
            "training_nuisance_rowspace": {
                "basis_identity": runner._float64_array_identity(basis)
            }
        },
    }
    freeze_path = runner._fold_training_freeze_path(0)
    runner._csms()._base()._save_checkpoint(
        torch,
        path=freeze_path,
        metadata={
            "schema_version": runner.TRAINING_FREEZE_SCHEMA,
            "status": "frozen_before_held_derived_computation",
            "split": "opened_development",
            "lock_identity_sha256": "a" * 64,
            "capture_file_sha256": runner.file_sha256(runner.CAPTURE_PATH),
            "capture_checkpoint_sha256": "b" * 64,
            "training_fold_file_sha256": runner.file_sha256(training_path),
            "fold_index": 0,
            "selected_layer": 7,
            "frozen_training_nuisance_basis_identity": (
                runner._float64_array_identity(basis)
            ),
            "training_only_slot_scales_identity": (
                runner._float64_array_identity(scales)
            ),
        },
        tensors={
            "frozen_training_nuisance_basis": torch.from_numpy(basis),
            "training_only_slot_scales": torch.from_numpy(scales),
        },
    )
    _, loaded_basis, loaded_scales = runner._load_training_freeze(
        torch,
        index=0,
        artifact=artifact,
    )
    np.testing.assert_array_equal(loaded_basis, basis)
    np.testing.assert_array_equal(loaded_scales, scales)
    changed = json.loads(json.dumps(artifact))
    changed["selected_layer_detail"]["training_nuisance_rowspace"][
        "basis_identity"
    ]["raw_little_endian_bytes_sha256"] = "0" * 64
    with pytest.raises(ALFSIntegrityError, match="persisted training freeze"):
        runner._load_training_freeze(torch, index=0, artifact=changed)


def test_report_result_contract_rederives_selection_and_current_provenance(
    runner, monkeypatch, tmp_path
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "CAPTURE_PATH", tmp_path / "capture.pt")
    monkeypatch.setattr(runner, "TRAINING_ROOT", tmp_path / "training")
    monkeypatch.setattr(runner, "HELD_ROOT", tmp_path / "held")
    monkeypatch.setattr(
        runner, "TRAINING_COMPLETE_PATH", tmp_path / "training_complete.json"
    )
    monkeypatch.setattr(runner, "HELD_COMPLETE_PATH", tmp_path / "held_complete.json")
    paths = [
        runner.CAPTURE_PATH,
        runner.TRAINING_COMPLETE_PATH,
        runner.HELD_COMPLETE_PATH,
        *(runner._fold_training_path(index) for index in range(4)),
        *(runner._fold_held_path(index) for index in range(4)),
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    candidates = [
        {
            "layer": layer,
            "eligible": layer == 7,
            "worst_primary_minimum_norm": 0.1 if layer == 7 else None,
            "mean_primary_minimum_norm": 0.1 if layer == 7 else None,
        }
        for layer in range(24)
    ]
    selection = runner.select_layer(candidates)
    training = [
        {"passes": True, "selection": {"passes": True, "selected_layer": 7}}
        for _ in range(4)
    ]
    held = [{"passes": True} for _ in range(4)]
    fold_layers, checks, passes = runner._global_qualification(
        training, held, selection
    )
    result = {
        "schema_version": runner.RESULT_SCHEMA,
        "status": "go_coordinate_only",
        "split": "opened_development",
        "lock_identity_sha256": "a" * 64,
        "capture_file_sha256": runner.file_sha256(runner.CAPTURE_PATH),
        "capture_checkpoint_sha256": "b" * 64,
        "training_complete_file_sha256": runner.file_sha256(
            runner.TRAINING_COMPLETE_PATH
        ),
        "held_complete_file_sha256": runner.file_sha256(runner.HELD_COMPLETE_PATH),
        "training_fold_files": [
            {
                "path": runner._relative(runner._fold_training_path(index)),
                "sha256": runner.file_sha256(runner._fold_training_path(index)),
            }
            for index in range(4)
        ],
        "held_fold_files": [
            {
                "path": runner._relative(runner._fold_held_path(index)),
                "sha256": runner.file_sha256(runner._fold_held_path(index)),
            }
            for index in range(4)
        ],
        "full_data_layer_candidates": candidates,
        "full_data_selection": selection,
        "full_data_selected_layer_detail": candidates[7],
        "fold_selected_layers": fold_layers,
        "checks": checks,
        "passes": passes,
        "finite_intervention_authorized": False,
        "next_authorized_action": "write_separate_prospective_controller_protocol",
        "sealed_data_accessed": False,
        "model_forwards_after_capture": 0,
        "model_backwards_after_capture": 0,
    }
    runner._validate_result_contract(
        result,
        lock={"lock_identity_sha256": "a" * 64},
        capture_checkpoint_sha256="b" * 64,
        training=training,
        held=held,
    )
    changed = json.loads(json.dumps(result))
    changed["capture_checkpoint_sha256"] = "c" * 64
    with pytest.raises(ALFSIntegrityError, match="provenance"):
        runner._validate_result_contract(
            changed,
            lock={"lock_identity_sha256": "a" * 64},
            capture_checkpoint_sha256="b" * 64,
            training=training,
            held=held,
        )
