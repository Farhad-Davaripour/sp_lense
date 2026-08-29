from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
WRAPPER_PATH = (
    ROOT / "scripts" / "decision_margin_shield_finite_capture_manifest_amendment.py"
)
SPECIFICATION = importlib.util.spec_from_file_location(
    "dms_finite_capture_manifest_amendment_tests", WRAPPER_PATH
)
assert SPECIFICATION is not None and SPECIFICATION.loader is not None
amendment = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(amendment)


def test_exact_legacy_manifest_is_valid_and_has_no_alias_on_disk() -> None:
    raw = amendment.validate_raw_capture_manifest()
    assert raw["schema_version"] == amendment.EXPECTED_CAPTURE_SCHEMA
    assert raw["manifest_sha256"] == amendment.EXPECTED_CAPTURE_MANIFEST_SHA256
    assert amendment.ALIAS_FIELD not in raw
    assert (
        amendment.finite_runner.file_sha256(amendment.CAPTURE_MANIFEST_PATH)
        == amendment.EXPECTED_CAPTURE_FILE_SHA256
    )


def test_patch_is_exact_path_and_alias_scoped() -> None:
    amended = amendment.finite_runner._load_json(amendment.CAPTURE_MANIFEST_PATH)
    assert amended[amendment.ALIAS_FIELD] == amended[amendment.ORIGINAL_FIELD]
    amendment.finite_runner._verify_hash(amended, amendment.ALIAS_FIELD)

    qualification = amendment.finite_runner._load_json(
        amendment.finite_runner.QUALIFICATION_RESULT_PATH
    )
    assert amendment.ALIAS_FIELD not in qualification
    with pytest.raises(RuntimeError, match="compatibility alias differs"):
        amendment.finite_runner._verify_hash(
            {**amended, amendment.ALIAS_FIELD: "0" * 64},
            amendment.ALIAS_FIELD,
        )


def test_frozen_qualification_evidence_revalidates_without_source_patch() -> None:
    result = amendment.finite_runner._validate_qualification_result()
    assert result["status"] == "passed"
    assert result["finite_lock_authorized"] is True
    assert "finite_capture_manifest_amendment_runner" not in (
        amendment.finite_runner._source_records()
    )
    assert "finite_capture_manifest_amendment_runner" not in (
        amendment.finite_runner._qualification_dependency_records()
    )


def test_wrapper_proposed_lock_is_model_free_and_binds_amendment() -> None:
    lock_path = amendment.finite_runner.LOCK_PATH
    existed_before = lock_path.exists()
    hash_before = (
        amendment.finite_runner.file_sha256(lock_path) if existed_before else None
    )
    proposed = amendment.proposed_lock()
    assert lock_path.exists() is existed_before
    assert (
        amendment.finite_runner.file_sha256(lock_path) if lock_path.exists() else None
    ) == hash_before
    assert proposed["status"] == (
        "locked_before_finite_run" if existed_before else "proposed_not_yet_run"
    )
    assert proposed["capture_binding"]["manifest_sha256"] == (
        amendment.EXPECTED_CAPTURE_MANIFEST_SHA256
    )
    assert proposed["capture_binding"]["manifest_file_sha256"] == (
        amendment.EXPECTED_CAPTURE_FILE_SHA256
    )
    dependencies = proposed["bound_dependencies"]
    expected = {
        "finite_capture_manifest_amendment_runner": amendment.SCRIPT_PATH,
        "finite_capture_manifest_amendment_protocol": amendment.AMENDMENT_DOC_PATH,
        "finite_capture_manifest_amendment_tests": amendment.AMENDMENT_TEST_PATH,
    }
    for name, path in expected.items():
        assert dependencies[name] == {
            "path": amendment.finite_runner._relative(path),
            "sha256": amendment.finite_runner.file_sha256(path),
        }


def test_wrapper_exposes_no_qualification_or_pilot_command() -> None:
    source = WRAPPER_PATH.read_text(encoding="utf-8")
    assert (
        'choices=("lock", "preflight", "construct", "freeze", "calibrate", "report")'
        in source
    )
    assert '"qualification-lock"' not in source
    assert '"qualify-controls"' not in source
    assert "run_pilot" not in source
