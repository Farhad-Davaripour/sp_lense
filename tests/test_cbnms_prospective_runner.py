import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sp_lense.counterfactual_behavioral_null_multilayer import (
    CBNMSIntegrityError,
    build_tokenizer_preflight,
)


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "cbnms_prospective.py"
    spec = importlib.util.spec_from_file_location("_test_cbnms_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rehash(runner, value, field):
    result = json.loads(json.dumps(value))
    result.pop(field, None)
    result[field] = runner.canonical_sha256(result)
    return result


class _FakeFloat64Tensor:
    def __init__(self, value):
        self.dtype = torch.float64
        self._value = value

    def numpy(self):
        return self._value


def _checkpoint_identity(runner, value):
    identity = runner._float64_array_identity(value)
    return {
        "shape": identity["shape"],
        "dtype": "float64",
        "raw_sha256": identity["raw_little_endian_bytes_sha256"],
    }


def test_proposed_lock_binds_exact_thresholds_transitive_code_and_adaptive_lineage(
    runner,
):
    lock = runner.proposed_lock()
    configuration = lock["configuration"]
    assert configuration["qualification"][
        "minimum_absolute_fold_global_held_leakage_reduction_vs_target_only_bank"
    ] == 0.01
    assert configuration["qualification"][
        "maximum_relative_fold_global_held_leakage_vs_target_only_bank"
    ] == 0.8
    paths = [record["path"] for record in lock["locked_files"]]
    assert paths == [str(value).replace("\\", "/") for value in runner.LOCKED_PATHS]
    assert len(paths) == len(set(paths))
    assert "src/sp_lense/all_layer_four_slot_oracle.py" in paths
    adaptive = configuration["prospective_status"]["adaptive_lineage"]
    assert [record["path"] for record in adaptive] == [
        str(value).replace("\\", "/") for value in runner.ADAPTIVE_LINEAGE_PATHS
    ]
    assert runner.verify_lock(lock, verify_files=True) == lock


def test_rehashed_lock_tampering_still_fails_exact_contract(runner):
    lock = runner.proposed_lock()
    changed = json.loads(json.dumps(lock))
    changed["configuration"]["qualification"][
        "maximum_relative_fold_global_held_leakage_vs_target_only_bank"
    ] = 0.81
    changed = _rehash(runner, changed, "lock_identity_sha256")
    with pytest.raises(CBNMSIntegrityError, match="contract"):
        runner.verify_lock(changed, verify_files=False)

    changed = json.loads(json.dumps(lock))
    changed["locked_files"][0]["path"] = changed["locked_files"][1]["path"]
    changed = _rehash(runner, changed, "lock_identity_sha256")
    with pytest.raises(CBNMSIntegrityError, match="coverage"):
        runner.verify_lock(changed, verify_files=False)


def test_commands_refuse_any_sealed_split_before_artifact_access(runner):
    for command in (
        runner.run_validate,
        runner.run_lock,
        runner.run_preflight,
        runner.run_capture,
        runner.run_analyze,
        runner.run_report,
    ):
        with pytest.raises(CBNMSIntegrityError, match="prospective_validation"):
            command("sealed_test")


def test_lock_and_capture_refuse_overwrite_before_backend_load(
    runner, monkeypatch, tmp_path
):
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("immutable", encoding="utf-8")
    monkeypatch.setattr(runner, "LOCK_PATH", lock_path)
    monkeypatch.setattr(
        runner,
        "proposed_lock",
        lambda *_args, **_kwargs: pytest.fail("must refuse before rebuilding lock"),
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run_lock()

    monkeypatch.setattr(
        runner, "_load_lock", lambda: {"lock_identity_sha256": "a" * 64}
    )
    monkeypatch.setattr(runner, "_load_preflight", lambda _lock: {})
    monkeypatch.setattr(runner, "CAPTURE_PATH", tmp_path / "capture.pt")
    monkeypatch.setattr(runner, "CAPTURE_COMPLETE_PATH", tmp_path / "complete.json")
    monkeypatch.setattr(runner, "CAPTURE_RESERVATION_PATH", tmp_path / "reserve.json")
    runner.CAPTURE_COMPLETE_PATH.write_text("complete", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run_capture()


def test_capture_requires_exact_offline_environment_before_reservation(
    runner, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        runner, "_load_lock", lambda: {"lock_identity_sha256": "a" * 64}
    )
    monkeypatch.setattr(runner, "_load_preflight", lambda _lock: {})
    monkeypatch.setattr(runner, "CAPTURE_PATH", tmp_path / "capture.pt")
    monkeypatch.setattr(runner, "CAPTURE_COMPLETE_PATH", tmp_path / "complete.json")
    monkeypatch.setattr(runner, "CAPTURE_RESERVATION_PATH", tmp_path / "reserve.json")
    for name in runner.OFFLINE_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(CBNMSIntegrityError, match="offline"):
        runner.run_capture()
    assert not runner.CAPTURE_RESERVATION_PATH.exists()


def test_training_slice_is_invariant_to_poisoning_all_24_excluded_rows(runner):
    _, forms = runner._source()
    fold = runner.build_loso_folds(forms)[0]
    residuals = np.arange(80 * 23 * 4, dtype=np.float64).reshape(80, 23, 4, 1)
    gradients = residuals + 0.5
    margins = np.arange(80, dtype=np.float64).tolist()
    first = runner._copy_training_only_numeric(residuals, gradients, margins, fold)
    held = [int(value) for value in fold["held_all_indices"]]
    poisoned_residuals = residuals.copy()
    poisoned_gradients = gradients.copy()
    poisoned_margins = list(margins)
    poisoned_residuals[held] = np.nan
    poisoned_gradients[held] = np.inf
    for index in held:
        poisoned_margins[index] = float("-inf")
    second = runner._copy_training_only_numeric(
        poisoned_residuals, poisoned_gradients, poisoned_margins, fold
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert not np.shares_memory(second[0], poisoned_residuals)
    assert not np.shares_memory(second[1], poisoned_gradients)


class _Tokenizer:
    chat_template = "cbnms-runner-preflight-template"
    eos_token_id = None
    all_special_ids = (2, 22)

    def __init__(self):
        self._prompt_suffixes = {}

    def prompt_tokens(self, text):
        if text not in self._prompt_suffixes:
            self._prompt_suffixes[text] = 30 + len(self._prompt_suffixes)
        return [*range(2, 22), self._prompt_suffixes[text]]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"A": [0], "B": [1]}[text]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
        return_dict,
        return_tensors,
    ):
        assert tokenize and not enable_thinking and return_dict and return_tensors == "pt"
        prompt_text = messages[-1]["content"] if add_generation_prompt else messages[-2]["content"]
        values = self.prompt_tokens(prompt_text)
        if not add_generation_prompt:
            values += ({"": [], "A": [0], "B": [1]}[messages[-1]["content"]]) + [22]
        return {"input_ids": torch.tensor([values], dtype=torch.long)}

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join({0: "A", 1: "B"}.get(int(value), "") for value in token_ids)


def _preflight_backend():
    tokenizer = _Tokenizer()
    return SimpleNamespace(
        torch=torch,
        model=SimpleNamespace(tokenizer=tokenizer),
        config=SimpleNamespace(model=SimpleNamespace(prompt_format="chat")),
        encode=lambda prompt: torch.tensor(
            [tokenizer.prompt_tokens(prompt)], dtype=torch.long
        ),
    )


def test_rehashed_preflight_slot_tamper_fails_deep_contract(
    runner, monkeypatch, tmp_path
):
    _, forms = runner._source()
    core = build_tokenizer_preflight(_preflight_backend(), forms)
    lock = {"lock_identity_sha256": "a" * 64}
    value = runner._with_hash(
        {
            "schema_version": "sp_lense.cbnms_tokenizer_preflight_file.v1",
            "status": "complete",
            "split": "prospective_validation",
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "dataset_file_sha256": runner.file_sha256(runner.DATA_PATH),
            "offline_environment": runner.OFFLINE_ENVIRONMENT,
            "core": core,
            "model_compute": {
                "model_forwards": 0,
                "model_backwards": 0,
                "generated_tokens": 0,
                "external_api_calls": 0,
                "external_model_judges": 0,
            },
            "sealed_data_accessed": False,
        },
        "preflight_file_sha256",
    )
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(runner, "PREFLIGHT_PATH", path)
    assert runner._load_preflight(lock) == value

    changed = json.loads(json.dumps(value))
    changed["core"]["rows"][0]["slot_indices"][1] -= 1
    changed["core"]["rows"][0] = _rehash(
        runner, changed["core"]["rows"][0], "row_sha256"
    )
    changed["core"] = _rehash(runner, changed["core"], "preflight_sha256")
    changed = _rehash(runner, changed, "preflight_file_sha256")
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(CBNMSIntegrityError, match="row contract"):
        runner._load_preflight(lock)


def test_skipped_held_panel_is_complete_and_summary_can_fail_closed(runner):
    _, forms = runner._source()
    fold = runner.build_loso_folds(forms)[0]
    training = {"training_file_sha256": "b" * 64}
    skipped = runner._skipped_held(fold, training)
    assert skipped["passes"] is False
    assert [
        row["replicate"] for row in skipped["random_rank_matched_nullspace_controls"]
    ] == list(range(32))
    assert all(
        row["passes_complete_fold_gate"] is False
        for row in skipped["random_rank_matched_nullspace_controls"]
    )


def test_full_data_is_skipped_after_any_loso_failure_and_cannot_rescue(runner):
    training = [{"passes": True} for _ in range(4)]
    held = [{"passes": True} for _ in range(4)]
    assert runner._full_data_stop_record(training, held) is None
    held[2]["passes"] = False
    stopped = runner._full_data_stop_record(training, held)
    assert stopped is not None
    assert stopped["passes"] is False
    assert stopped["full_data_numeric_rows_used"] is False
    assert stopped["status"] == "not_evaluated_because_one_or_more_LOSO_gates_failed"


def test_rehashed_fold_row_list_tamper_cannot_change_source_partition(runner):
    _, forms = runner._source()
    expected = runner.build_loso_folds(forms)[0]
    training = {"fold_index": 0, "core": {"fold": expected}}
    held = {"fold_index": 0, "core": {"fold": expected}}
    runner._verify_fold_source_binding(training, held, expected, 0)
    changed = json.loads(json.dumps(expected))
    changed["training_nuisance_indices"][0], changed["training_nuisance_indices"][1] = (
        changed["training_nuisance_indices"][1],
        changed["training_nuisance_indices"][0],
    )
    changed = _rehash(runner, changed, "fold_sha256")
    training["core"] = {"fold": changed}
    held["core"] = {"fold": changed}
    with pytest.raises(CBNMSIntegrityError, match="fresh source construction"):
        runner._verify_fold_source_binding(training, held, expected, 0)


def test_passing_full_data_refuses_a_deleted_numeric_freeze(
    runner, monkeypatch, tmp_path
):
    missing = tmp_path / "deleted_full_data_freeze.pt"
    monkeypatch.setattr(runner, "FULL_DATA_FREEZE_PATH", missing)
    full_file = {
        "passes": True,
        "core": {"record_sha256": "c" * 64},
        "freeze": {
            "path": str(missing),
            "file_sha256": "d" * 64,
            "checkpoint_sha256": "e" * 64,
        },
    }
    with pytest.raises(CBNMSIntegrityError, match="freeze is missing"):
        runner._load_full_data_numeric(torch, full_file)


def test_self_consistent_rehashed_training_freeze_cannot_replace_core_bank(
    runner, monkeypatch, tmp_path
):
    monkeypatch.setattr(runner, "TRAINING_ROOT", tmp_path / "training")
    freeze_path = runner._training_freeze_path(0)
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_bytes(b"replacement")
    arrays = {
        "scales": np.ones((23, 4), dtype=np.float64),
        "nuisance_basis": np.zeros((44, 94208), dtype=np.float64),
        "SP_bank": np.zeros((1, 94208), dtype=np.float64),
        "target_only_bank": np.zeros((1, 94208), dtype=np.float64),
    }
    arrays["SP_bank"][0, 0] = 1.0
    arrays["target_only_bank"][0, 1] = 1.0
    actual = {name: runner._float64_array_identity(value) for name, value in arrays.items()}
    wrong_scale = dict(actual["scales"])
    wrong_scale["raw_little_endian_bytes_sha256"] = "0" * 64
    core = runner._with_hash(
        {
            "training_scales_identity": wrong_scale,
            "training_nuisance_basis": {
                "svd_record": {"basis_identity": actual["nuisance_basis"]}
            },
            "training_SP_bank": {"bank_basis_identity": actual["SP_bank"]},
            "training_target_only_bank": {
                "bank_basis_identity": actual["target_only_bank"]
            },
        },
        "record_sha256",
    )
    metadata = {
        "checkpoint_sha256": "c" * 64,
        "fold_index": 0,
        "training_record_sha256": core["record_sha256"],
        "tensor_identities": {
            name: _checkpoint_identity(runner, value) for name, value in arrays.items()
        },
    }
    training_file = runner._with_hash(
        {
            "fold_index": 0,
            "core": core,
            "freeze": {
                "path": str(freeze_path),
                "file_sha256": "f" * 64,
                "checkpoint_sha256": metadata["checkpoint_sha256"],
            },
            "passes": True,
        },
        "training_file_sha256",
    )
    monkeypatch.setattr(runner, "file_sha256", lambda _path: "f" * 64)
    monkeypatch.setattr(
        runner,
        "_load_checkpoint",
        lambda *_args, **_kwargs: (
            metadata,
            {name: _FakeFloat64Tensor(value) for name, value in arrays.items()},
        ),
    )
    with pytest.raises(CBNMSIntegrityError, match="differs from core identities"):
        runner._load_training_numeric(torch, training_file)


def test_self_consistent_rehashed_full_freeze_cannot_replace_core_bank(
    runner, monkeypatch, tmp_path
):
    freeze_path = tmp_path / "full_data_freeze.pt"
    freeze_path.write_bytes(b"replacement")
    monkeypatch.setattr(runner, "FULL_DATA_FREEZE_PATH", freeze_path)
    arrays = {
        "scales": np.ones((23, 4), dtype=np.float64),
        "nuisance_basis": np.zeros((64, 94208), dtype=np.float64),
        "SP_bank": np.zeros((1, 94208), dtype=np.float64),
    }
    arrays["SP_bank"][0, 0] = 1.0
    actual = {name: runner._float64_array_identity(value) for name, value in arrays.items()}
    wrong_bank = dict(actual["SP_bank"])
    wrong_bank["raw_little_endian_bytes_sha256"] = "0" * 64
    core = runner._with_hash(
        {
            "training_scales_identity": actual["scales"],
            "full_nuisance_basis": {
                "svd_record": {"basis_identity": actual["nuisance_basis"]}
            },
            "full_data_SP_bank": {"bank_basis_identity": wrong_bank},
            "passes": True,
        },
        "record_sha256",
    )
    metadata = {
        "checkpoint_sha256": "c" * 64,
        "full_data_record_sha256": core["record_sha256"],
        "tensor_names": sorted(arrays),
        "tensor_identities": {
            name: _checkpoint_identity(runner, value) for name, value in arrays.items()
        },
    }
    full_file = {
        "passes": True,
        "core": core,
        "freeze": {
            "path": str(freeze_path),
            "file_sha256": "f" * 64,
            "checkpoint_sha256": metadata["checkpoint_sha256"],
        },
    }
    monkeypatch.setattr(runner, "file_sha256", lambda _path: "f" * 64)
    monkeypatch.setattr(
        runner,
        "_load_checkpoint",
        lambda *_args, **_kwargs: (
            metadata,
            {name: _FakeFloat64Tensor(value) for name, value in arrays.items()},
        ),
    )
    with pytest.raises(CBNMSIntegrityError, match="differs from core identities"):
        runner._load_full_data_numeric(torch, full_file)
