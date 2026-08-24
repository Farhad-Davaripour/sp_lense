from __future__ import annotations

import json
from pathlib import Path

import pytest

from sp_lense.comparison_environment import (
    capture_environment_record,
    validate_environment_record,
    verify_current_environment,
    write_environment_record,
)


def _lock() -> dict:
    return {
        "study": "test-study",
        "models": [
            {
                "model_id": "example/model",
                "revision": "a" * 40,
                "config": "configs/model.json",
                "config_sha256": "b" * 64,
                "runtime": {"device": "cpu", "dtype": "float32"},
                "architecture": {"blocks": 2, "residual_width": 4},
            }
        ],
    }


def test_environment_round_trip_and_current_verification(tmp_path: Path) -> None:
    stage1 = tmp_path / "stage1.json"
    stage1.write_text("{}\n", encoding="utf-8")
    record = capture_environment_record(
        stage1_lock_path=stage1, lock=_lock(), package_names=("torch",)
    )
    path = tmp_path / "environment.json"
    write_environment_record(path, record)
    assert verify_current_environment(
        path, stage1_lock_path=stage1, lock=_lock()
    ) == record


def test_environment_record_detects_tampering(tmp_path: Path) -> None:
    stage1 = tmp_path / "stage1.json"
    stage1.write_text("{}\n", encoding="utf-8")
    record = capture_environment_record(
        stage1_lock_path=stage1, lock=_lock(), package_names=("torch",)
    )
    record["python"]["version"] = "forged"
    with pytest.raises(ValueError, match="content_sha256"):
        validate_environment_record(record)


def test_environment_file_is_canonical_json_compatible(tmp_path: Path) -> None:
    stage1 = tmp_path / "stage1.json"
    stage1.write_text("{}\n", encoding="utf-8")
    record = capture_environment_record(
        stage1_lock_path=stage1, lock=_lock(), package_names=("torch",)
    )
    path = tmp_path / "environment.json"
    write_environment_record(path, record)
    assert json.loads(path.read_text(encoding="utf-8"))["packages"]["torch"]
