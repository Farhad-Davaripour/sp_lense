from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
FROZEN_RUNNER_PATH = ROOT / "scripts" / "decision_margin_shield_finite_calibration.py"
AMENDMENT_DOC_PATH = (
    ROOT / "docs" / "DECISION_MARGIN_SHIELD_FINITE_CAPTURE_MANIFEST_AMENDMENT.md"
)
AMENDMENT_TEST_PATH = (
    ROOT / "tests" / "test_decision_margin_shield_finite_capture_manifest_amendment.py"
)
CAPTURE_MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "decision_margin_shield_layer_screen"
    / "qwen35_08b"
    / "capture_manifest.json"
)

EXPECTED_CAPTURE_FILE_SHA256 = (
    "0d3720ef0bcda3e6dd430aa6033b949404b726e4f616ada86e26b2bbc472a939"
)
EXPECTED_CAPTURE_SCHEMA = "sp_lense.decision_margin_shield_layer_screen_capture.v2"
EXPECTED_CAPTURE_MANIFEST_SHA256 = (
    "cf654fa4bc42ea550138653a4927232888a3724cfb9451bf97b7b5551740faf0"
)
ALIAS_FIELD = "capture_manifest_sha256"
ORIGINAL_FIELD = "manifest_sha256"


def _load_frozen_runner() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "dms_finite_frozen_capture_amendment_runner", FROZEN_RUNNER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot import the frozen DMS finite runner")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


finite_runner = _load_frozen_runner()
_ORIGINAL_LOAD_JSON = finite_runner._load_json
_ORIGINAL_VERIFY_HASH = finite_runner._verify_hash
_ORIGINAL_BOUND_DEPENDENCY_RECORDS = finite_runner._bound_dependency_records


def _same_path(left: Path | str, right: Path | str) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _validate_raw_capture_value(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != EXPECTED_CAPTURE_SCHEMA
        or value.get(ORIGINAL_FIELD) != EXPECTED_CAPTURE_MANIFEST_SHA256
        or ALIAS_FIELD in value
    ):
        raise RuntimeError("legacy capture manifest schema, identity, or alias differs")
    _ORIGINAL_VERIFY_HASH(value, ORIGINAL_FIELD)


def validate_raw_capture_manifest() -> dict[str, Any]:
    if finite_runner.file_sha256(CAPTURE_MANIFEST_PATH) != EXPECTED_CAPTURE_FILE_SHA256:
        raise RuntimeError("legacy capture manifest file hash differs")
    value = _ORIGINAL_LOAD_JSON(CAPTURE_MANIFEST_PATH)
    _validate_raw_capture_value(value)
    return value


def _amended_load_json(path: Path) -> dict[str, Any]:
    value = _ORIGINAL_LOAD_JSON(path)
    if not _same_path(path, CAPTURE_MANIFEST_PATH):
        return value
    if finite_runner.file_sha256(CAPTURE_MANIFEST_PATH) != EXPECTED_CAPTURE_FILE_SHA256:
        raise RuntimeError("legacy capture manifest file hash differs")
    _validate_raw_capture_value(value)
    return {**value, ALIAS_FIELD: value[ORIGINAL_FIELD]}


def _amended_verify_hash(value: Mapping[str, Any], field: str) -> None:
    if field != ALIAS_FIELD:
        _ORIGINAL_VERIFY_HASH(value, field)
        return
    original = dict(value)
    alias = original.pop(ALIAS_FIELD, None)
    if (
        alias != EXPECTED_CAPTURE_MANIFEST_SHA256
        or original.get(ORIGINAL_FIELD) != alias
        or original.get("schema_version") != EXPECTED_CAPTURE_SCHEMA
    ):
        raise RuntimeError("capture manifest compatibility alias differs")
    _ORIGINAL_VERIFY_HASH(original, ORIGINAL_FIELD)


def _amendment_dependency_records() -> dict[str, dict[str, str]]:
    result = dict(_ORIGINAL_BOUND_DEPENDENCY_RECORDS())
    additions = {
        "finite_capture_manifest_amendment_runner": SCRIPT_PATH,
        "finite_capture_manifest_amendment_protocol": AMENDMENT_DOC_PATH,
        "finite_capture_manifest_amendment_tests": AMENDMENT_TEST_PATH,
    }
    if set(result).intersection(additions):
        raise RuntimeError("capture-manifest amendment dependency names collide")
    result.update(
        {
            name: {
                "path": finite_runner._relative(path),
                "sha256": finite_runner.file_sha256(path),
            }
            for name, path in additions.items()
        }
    )
    return result


def install_capture_manifest_amendment() -> None:
    validate_raw_capture_manifest()
    finite_runner._load_json = _amended_load_json
    finite_runner._verify_hash = _amended_verify_hash
    finite_runner._bound_dependency_records = _amendment_dependency_records


install_capture_manifest_amendment()


def proposed_lock() -> dict[str, Any]:
    return finite_runner.proposed_lock()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture-manifest field-name amendment for the frozen DMS finite runner"
        )
    )
    parser.add_argument(
        "command",
        choices=("lock", "preflight", "construct", "freeze", "calibrate", "report"),
    )
    args = parser.parse_args()
    commands = {
        "lock": finite_runner.run_lock,
        "preflight": finite_runner.run_preflight,
        "construct": finite_runner.run_construct,
        "freeze": finite_runner.run_freeze,
        "calibrate": finite_runner.run_calibration,
        "report": finite_runner.run_report,
    }
    value = commands[args.command]()
    print(value if isinstance(value, str) else json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
