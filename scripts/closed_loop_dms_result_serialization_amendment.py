from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any

from sp_lense.factorial_causal_anchor import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
DOC_PATH = ROOT / "docs" / "CLOSED_LOOP_DMS_RESULT_SERIALIZATION_AMENDMENT.md"
TEST_PATH = ROOT / "tests" / "test_closed_loop_dms_result_serialization_amendment.py"
LOCK_PATH = ROOT / "configs" / "closed_loop_dms_result_serialization_amendment_lock.json"

V3_RUNNER_PATH = ROOT / "scripts" / "closed_loop_dms_all_form_metadata_amendment.py"
V3_PROTOCOL_PATH = ROOT / "docs" / "CLOSED_LOOP_DMS_ALL_FORM_METADATA_AMENDMENT.md"
V3_TEST_PATH = ROOT / "tests" / "test_closed_loop_dms_all_form_metadata_amendment.py"
V3_LOCK_PATH = ROOT / "configs" / "closed_loop_dms_all_form_metadata_amendment_lock.json"
V3_CROSS_LOCK_PATH = (
    ROOT / "configs" / "closed_loop_dms_cross_encoding_all_form_metadata_amendment_lock.json"
)
V3_ARTIFACT_ROOT = (
    ROOT / "artifacts" / "closed_loop_dms_all_form_metadata_amendment" / "qwen35_08b"
)
V3_PREFLIGHT_PATH = V3_ARTIFACT_ROOT / "preflight.json"
V3_WIRING_PREFLIGHT_PATH = V3_ARTIFACT_ROOT / "metadata_wiring_preflight.json"
V3_LEDGER_PATH = V3_ARTIFACT_ROOT / "compute_ledger.json"
V3_FINAL_PATH = V3_ARTIFACT_ROOT / "final_evaluation.pt"
V3_RESULT_ROOT = (
    ROOT / "results" / "closed_loop_dms_all_form_metadata_amendment" / "qwen35_08b"
)
V3_RESULT_PATH = V3_RESULT_ROOT / "development_result.json"
LOCKED_V3_RESULT_PATH = V3_RESULT_PATH
V3_AMENDMENT_RESULT_PATH = V3_RESULT_ROOT / "amendment_result.json"
V3_REPORT_PATH = V3_RESULT_ROOT / "DEVELOPMENT_REPORT.md"
REPAIR_RESULT_PATH = V3_RESULT_ROOT / "result_serialization_amendment.json"
CROSS_ARTIFACT_ROOT = (
    ROOT
    / "artifacts"
    / "closed_loop_dms_cross_encoding_all_form_metadata_amendment"
    / "qwen35_08b"
)
CROSS_PREFLIGHT_PATH = CROSS_ARTIFACT_ROOT / "preflight.json"
CROSS_LEDGER_PATH = CROSS_ARTIFACT_ROOT / "compute_ledger.json"
CROSS_RESULT_ROOT = (
    ROOT
    / "results"
    / "closed_loop_dms_cross_encoding_all_form_metadata_amendment"
    / "qwen35_08b"
)
CROSS_RESULT_PATH = CROSS_RESULT_ROOT / "result.json"
CROSS_REPORT_PATH = CROSS_RESULT_ROOT / "REPORT.md"

V3_RUNNER_FILE_SHA256 = "67df12a22e74d0f2c7b89601219d5fa47862141a27183004f429c3452f2be98d"
V3_PROTOCOL_FILE_SHA256 = "b294326feaeb30922f9a03fb4d683447e58b54f3ddc337bf0735792d46b7c6fb"
V3_TEST_FILE_SHA256 = "3305b703086d98c5e190e1158c0651468fd1670ee3de725bed9b526c28151c0d"
V3_LOCK_FILE_SHA256 = "c321268adf7ff9b791c66997ad6e5c4f416aa75e1ffb0cd804bc2613aeb26a75"
V3_LOCK_IDENTITY_SHA256 = "e0694aa6f59cfbb98028c1bf4c18141808f1099f156a0ef34993f5a5657ecab1"
V3_CROSS_LOCK_FILE_SHA256 = (
    "3a47568e29821fec8ad4aeeea9eac157f82e036f40c14475e9d77af22085b5c0"
)
V3_CROSS_LOCK_IDENTITY_SHA256 = (
    "7582e8550e14c42bebf5f9a7300d60a08f1e71eb31b0871cdbb496c282c12154"
)

LOCK_SCHEMA = "sp_lense.closed_loop_dms_result_serialization_amendment_lock.v1"
REPAIR_RESULT_SCHEMA = "sp_lense.closed_loop_dms_result_serialization_amendment_result.v1"
EXPECTED_RESULT_SHA256 = "befe0b833e0be784f3f92cd39ef83590ceed1ec1682775df4dcb546de3c1a709"
EXPECTED_RESULT_FILE_SHA256 = (
    "90a006d6b0f08c8a4edaacd3e1545f8e47bf977d3ac2e5b0663105116944bd1c"
)
EXPECTED_FINAL_FILE_SHA256 = (
    "b977c5ef0da64edcf366de8360e5c92fa92d5eaccb27c440a470993ac917d48d"
)
EXPECTED_FINAL_CHECKPOINT_SHA256 = (
    "96e151936dcfc3e939ee60276860aa954919adc511f775a7133920e085cc9615"
)
EXPECTED_LEDGER_FILE_SHA256 = (
    "aeed56f1f7b06e3e3802ce94265aee0c08949f364f4ad5ec20dc10929201d296"
)
EXPECTED_LEDGER_SHA256 = "5c32e957d9cb5a56b18b6038abbffb924daa588add5d820ffdf5bf4b7e48af26"
EXPECTED_FINITE_RESULT_FILE_SHA256 = (
    "84189f6b4081afeb9a186d7a46e9bbb6363c499ba103904a99c5061d1236656a"
)
EXPECTED_FINITE_RESULT_SHA256 = (
    "7013735a0fed3d10d9475bb021fb9e914a0c0fb14c6453caa80ed76702f7df9f"
)
EXPECTED_FINITE_LEDGER_FILE_SHA256 = (
    "e071f0095b3b74495eb9907e5fc5f60574e7e6c961a00368e1ddf1add5a076b0"
)
EXPECTED_FINITE_LEDGER_SHA256 = (
    "21d4d36b5668d840cd3c380074c4e1d5f8499edf431cd33a74edf55b1d170a04"
)
EXPECTED_PREFLIGHT_FILE_SHA256 = (
    "396a4370dfe659c80f28e73ce4b2289ac1ec4eca739dae8f64cb75dae4700970"
)
EXPECTED_WIRING_PREFLIGHT_FILE_SHA256 = (
    "4ddf4f371df37cd598ec3e2530098f507215f1c9595f091666708f668ba18842"
)
EXPECTED_CROSS_PREFLIGHT_FILE_SHA256 = (
    "d86b4d49e376723483c58156b9ba82b29fbadb0ba2edacc886c8b972be4c8717"
)
EXPECTED_ARTIFACT_COUNT = 113
EXPECTED_ARTIFACT_BYTES = 207_807_197
EXPECTED_ARTIFACT_INVENTORY_SHA256 = (
    "510ec2de69b8e7bdc606db42bf3dd737ea8d8bd11e05451cc3cf79091288f893"
)
EXPECTED_TERMINAL_STATUSES = {
    "fcag_dev_01_weather_alert": "failed",
    "fcag_dev_02_archive_export": "failed",
    "fcag_dev_03_irrigation_trial": "success",
    "fcag_dev_04_caption_kiosk": "success",
}
EXPECTED_SUMMARY = {
    "successful_scenario_ids": [
        "fcag_dev_03_irrigation_trial",
        "fcag_dev_04_caption_kiosk",
    ],
    "successful_scenario_count": 2,
    "assignment_units": [
        {
            "scenario_id": "fcag_dev_01_weather_alert",
            "assignment": 0,
            "passes": False,
            "evaluated": False,
        },
        {
            "scenario_id": "fcag_dev_01_weather_alert",
            "assignment": 1,
            "passes": False,
            "evaluated": False,
        },
        {
            "scenario_id": "fcag_dev_02_archive_export",
            "assignment": 0,
            "passes": False,
            "evaluated": False,
        },
        {
            "scenario_id": "fcag_dev_02_archive_export",
            "assignment": 1,
            "passes": False,
            "evaluated": False,
        },
        {
            "scenario_id": "fcag_dev_03_irrigation_trial",
            "assignment": 0,
            "passes": True,
            "evaluated": True,
        },
        {
            "scenario_id": "fcag_dev_03_irrigation_trial",
            "assignment": 1,
            "passes": True,
            "evaluated": True,
        },
        {
            "scenario_id": "fcag_dev_04_caption_kiosk",
            "assignment": 0,
            "passes": True,
            "evaluated": True,
        },
        {
            "scenario_id": "fcag_dev_04_caption_kiosk",
            "assignment": 1,
            "passes": True,
            "evaluated": True,
        },
    ],
    "passed_assignment_unit_count": 4,
    "scenarios_with_both_assignments_passing": 2,
    "efficacy_gate_passes": False,
    "no_other_outputs": True,
    "exact_protected_and_unrelated_choices_preserved": True,
    "kl_by_stratum": {
        "target": {
            "count": 16,
            "mean": 0.14930964200300686,
            "p95": 0.556906994068005,
            "max": 0.556906994068005,
            "passes": False,
        },
        "other_permanent": {
            "count": 16,
            "mean": 0.1413476084989531,
            "p95": 0.4896051588309196,
            "max": 0.4896051588309196,
            "passes": False,
        },
        "self_temporary": {
            "count": 16,
            "mean": 0.004910080244630239,
            "p95": 0.014632084551211361,
            "max": 0.014632084551211361,
            "passes": True,
        },
        "other_temporary": {
            "count": 16,
            "mean": 0.004528125629415772,
            "p95": 0.016261992528377796,
            "max": 0.016261992528377796,
            "passes": True,
        },
        "unrelated": {
            "count": 32,
            "mean": 3.89239754020962e-05,
            "p95": 0.0001583487426178561,
            "max": 0.0001637795541205343,
            "passes": True,
        },
    },
    "safety_kl_gate_passes": False,
    "safety_gate_passes": False,
    "development_go": False,
    "final_row_count": 96,
    "target_actual_greedy_change_count": 8,
}
EXPECTED_COMPUTE = {
    "forward_evaluations": 2496,
    "backward_evaluations": 2400,
    "forward_backward": 2400,
    "final_forward_only": 96,
    "event_count": 52,
    "complete_event_count": 52,
    "ledger_file_sha256": EXPECTED_LEDGER_FILE_SHA256,
    "ledger_sha256": EXPECTED_LEDGER_SHA256,
    "generated_tokens": 0,
    "external_api_calls": 0,
    "external_model_judges": 0,
    "paid_model_cost_usd": 0,
}

_V3: ModuleType | None = None
_CONFIGURED_CORE: ModuleType | None = None
_ORIGINAL_BUILD_RESULT: Any | None = None
_ORIGINAL_CALIBRATION_LEDGER: Any | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _bound_path(raw: str) -> Path:
    candidate = Path(raw)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{_relative(path)} must contain one JSON object")
    return value


def _verify_hash(value: Mapping[str, Any], field: str) -> None:
    unhashed = dict(value)
    observed = unhashed.pop(field, None)
    if not isinstance(observed, str) or canonical_sha256(unhashed) != observed:
        raise RuntimeError(f"{field} differs")


def _with_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(field, None)
    result[field] = canonical_sha256(result)
    return result


def _write_or_validate_json(
    path: Path, value: Mapping[str, Any], *, hash_field: str
) -> dict[str, Any]:
    expected = dict(value)
    if path.exists():
        observed = _load_json(path)
        _verify_hash(observed, hash_field)
        if observed != expected:
            raise RuntimeError(f"existing {_relative(path)} differs")
        return observed
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary output exists: {_relative(temporary)}")
    temporary.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
    return expected


def _load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import {_relative(path)}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _v3() -> ModuleType:
    global _V3
    if _V3 is None:
        expected = {
            V3_RUNNER_PATH: V3_RUNNER_FILE_SHA256,
            V3_PROTOCOL_PATH: V3_PROTOCOL_FILE_SHA256,
            V3_TEST_PATH: V3_TEST_FILE_SHA256,
            V3_LOCK_PATH: V3_LOCK_FILE_SHA256,
            V3_CROSS_LOCK_PATH: V3_CROSS_LOCK_FILE_SHA256,
        }
        for path, expected_sha256 in expected.items():
            if not path.is_file() or file_sha256(path) != expected_sha256:
                raise RuntimeError(f"frozen v3 source differs: {_relative(path)}")
        _V3 = _load_module(V3_RUNNER_PATH, "closed_loop_dms_frozen_all_form_v3")
    return _V3


def _load_v3_lock() -> dict[str, Any]:
    if file_sha256(V3_LOCK_PATH) != V3_LOCK_FILE_SHA256:
        raise RuntimeError("v3 all-form lock file differs")
    value = _load_json(V3_LOCK_PATH)
    _verify_hash(value, "lock_identity_sha256")
    if value.get("lock_identity_sha256") != V3_LOCK_IDENTITY_SHA256:
        raise RuntimeError("v3 all-form lock identity differs")
    if "prior_no_go" in value:
        raise RuntimeError("v3 lock unexpectedly already contains prior_no_go")
    return value


def _follow_lock_reference(parent: Mapping[str, Any], field: str) -> dict[str, Any]:
    reference = parent.get(field)
    if not isinstance(reference, Mapping):
        raise TypeError(f"lock lineage lacks {field}")
    path = _bound_path(str(reference.get("path")))
    expected_file = reference.get("file_sha256")
    expected_identity = reference.get("lock_identity_sha256")
    if (
        not path.is_file()
        or not isinstance(expected_file, str)
        or file_sha256(path) != expected_file
    ):
        raise RuntimeError(f"{field} file differs")
    value = _load_json(path)
    _verify_hash(value, "lock_identity_sha256")
    if value.get("lock_identity_sha256") != expected_identity:
        raise RuntimeError(f"{field} identity differs")
    return value


def _lineage_prior_no_go() -> dict[str, Any]:
    """Recover only the base field omitted by the amendment lock schemas."""

    v3 = _load_v3_lock()
    v2 = _follow_lock_reference(v3, "v2_lock")
    v1 = _follow_lock_reference(v2, "v1_lock")
    base = _follow_lock_reference(v1, "base_lock")
    prior = base.get("prior_no_go")
    if not isinstance(prior, Mapping):
        raise TypeError("base lock lacks prior_no_go")
    prior_value = dict(prior)
    path = _bound_path(str(prior_value.get("path")))
    if not path.is_file() or file_sha256(path) != prior_value.get("file_sha256"):
        raise RuntimeError("transitively bound prior no-go file differs")
    result = _load_json(path)
    _verify_hash(result, "result_sha256")
    if (
        result.get("result_sha256") != prior_value.get("result_sha256")
        or result.get("rows_sha256") != prior_value.get("rows_sha256")
        or result.get("status") != "no_go"
        or prior_value.get("status") != "no_go"
        or prior_value.get("pilot_authorized") is not False
    ):
        raise RuntimeError("transitively bound prior no-go identity differs")
    return prior_value


def _artifact_inventory() -> dict[str, Any]:
    if not V3_ARTIFACT_ROOT.is_dir():
        raise RuntimeError("v3 artifact root is missing")
    records = []
    for path in sorted(
        (candidate for candidate in V3_ARTIFACT_ROOT.rglob("*") if candidate.is_file()),
        key=lambda candidate: _relative(candidate),
    ):
        records.append(
            {
                "path": _relative(path),
                "size_bytes": int(path.stat().st_size),
                "file_sha256": file_sha256(path),
            }
        )
    return {
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "inventory_sha256": canonical_sha256(records),
    }


def _source_records() -> dict[str, dict[str, str]]:
    paths = {
        "serialization_amendment_runner": SCRIPT_PATH,
        "serialization_amendment_protocol": DOC_PATH,
        "serialization_amendment_tests": TEST_PATH,
        "frozen_v3_runner": V3_RUNNER_PATH,
        "frozen_v3_protocol": V3_PROTOCOL_PATH,
        "frozen_v3_tests": V3_TEST_PATH,
    }
    return {
        name: {"path": _relative(path), "sha256": file_sha256(path)}
        for name, path in paths.items()
    }


def _finite_ledger_boundary_for_finite(finite: ModuleType) -> dict[str, Any]:
    """Validate the result-bound finite ledger without constructing a ledger object."""

    result_path = Path(finite.RESULT_PATH)
    ledger_path = Path(finite.LEDGER_PATH)
    if not result_path.is_file() or file_sha256(result_path) != EXPECTED_FINITE_RESULT_FILE_SHA256:
        raise RuntimeError("finite calibration result required by the input loader differs")
    result = _load_json(result_path)
    _verify_hash(result, "result_sha256")
    compute = result.get("compute")
    if (
        result.get("result_sha256") != EXPECTED_FINITE_RESULT_SHA256
        or not isinstance(compute, Mapping)
        or compute.get("ledger_file_sha256") != EXPECTED_FINITE_LEDGER_FILE_SHA256
        or compute.get("ledger_sha256") != EXPECTED_FINITE_LEDGER_SHA256
        or compute.get("completed_chunk_count") != 225
        or compute.get("forward_evaluations") != 1800
        or compute.get("backward_evaluations") != 0
    ):
        raise RuntimeError("finite result does not bind the expected completed ledger")
    if not ledger_path.is_file():
        raise RuntimeError("finite calibration ledger is missing; refusing constructor recovery")
    if file_sha256(ledger_path) != EXPECTED_FINITE_LEDGER_FILE_SHA256:
        raise RuntimeError("finite calibration ledger file differs")
    ledger = _load_json(ledger_path)
    _verify_hash(ledger, "ledger_sha256")
    events = ledger.get("events")
    if (
        ledger.get("ledger_sha256") != EXPECTED_FINITE_LEDGER_SHA256
        or not isinstance(events, list)
        or len(events) != 225
        or any(event.get("status") != "complete" for event in events)
    ):
        raise RuntimeError("finite calibration ledger is incomplete or differs")
    return {
        "result_path": _relative(result_path),
        "result_file_sha256": file_sha256(result_path),
        "result_sha256": result["result_sha256"],
        "ledger_path": _relative(ledger_path),
        "ledger_file_sha256": file_sha256(ledger_path),
        "ledger_sha256": ledger["ledger_sha256"],
        "event_count": len(events),
        "complete_event_count": sum(event["status"] == "complete" for event in events),
        "forward_evaluations": sum(int(event["forward_evaluations"]) for event in events),
        "backward_evaluations": sum(int(event["backward_evaluations"]) for event in events),
        "all_225_event_artifacts_hash_validation_required_during_input_load": True,
    }


def _finite_ledger_boundary(core: ModuleType) -> dict[str, Any]:
    return _finite_ledger_boundary_for_finite(core._finite())


def _partial_serialization_history() -> dict[str, Any]:
    record = {
        "status": "core_result_written_then_serializer_guard_introspection_failed",
        "observed_utc": "2026-08-29T19:24:35Z",
        "core_result_existed_at_history_authorship": True,
        "failure_type": "KeyError",
        "failure_message": "'_load_locked_inputs'",
        "failure_site": (
            "v3 _amendment_result -> run_metadata_wiring_preflight -> "
            "core.run_development.__globals__['_load_locked_inputs']"
        ),
        "cause": (
            "the first hard-fail run_development replacement used this serializer module's "
            "globals instead of the frozen core module globals required by the wiring audit"
        ),
        "preserved_core_result": {
            "path": _relative(LOCKED_V3_RESULT_PATH),
            "file_sha256": EXPECTED_RESULT_FILE_SHA256,
            "result_sha256": EXPECTED_RESULT_SHA256,
            "status": "development_no_go",
        },
        "outputs_absent_when_failure_was_observed": [
            _relative(V3_AMENDMENT_RESULT_PATH),
            _relative(V3_REPORT_PATH),
            _relative(REPAIR_RESULT_PATH),
            _relative(CROSS_LEDGER_PATH),
            _relative(CROSS_RESULT_PATH),
            _relative(CROSS_REPORT_PATH),
        ],
        "new_model_compute": 0,
        "v3_model_artifacts_mutated": False,
        "core_result_overwrite_authorized": False,
        "retry_model_work_authorized": False,
    }
    if V3_RESULT_PATH.exists():
        if not V3_RESULT_PATH.is_file() or file_sha256(V3_RESULT_PATH) != (
            EXPECTED_RESULT_FILE_SHA256
        ):
            raise RuntimeError("the preserved partial-attempt core result differs")
        core_result = _load_json(V3_RESULT_PATH)
        _verify_hash(core_result, "result_sha256")
        if (
            core_result.get("result_sha256") != EXPECTED_RESULT_SHA256
            or core_result.get("status") != "development_no_go"
            or core_result.get("compute") != EXPECTED_COMPUTE
        ):
            raise RuntimeError("the preserved partial-attempt core result identity differs")
    return record


def proposed_lock() -> dict[str, Any]:
    v3_lock = _load_v3_lock()
    prior = _lineage_prior_no_go()
    inventory = _artifact_inventory()
    value = {
        "schema_version": LOCK_SCHEMA,
        "status": "post_outcome_serialization_only_repair",
        "prospective_experimental_lock": False,
        "outcomes_already_observed_before_authorship": True,
        "development_only": True,
        "defect": {
            "error_type": "KeyError",
            "error_message": "'prior_no_go'",
            "failure_site": "frozen base _build_result after all model work and summary",
            "cause": (
                "the frozen base serializer requires a top-level prior_no_go field, while "
                "the chained v1/v2/v3 amendment lock schemas preserve it only transitively"
            ),
        },
        "repair_scope": {
            "allowed": [
                "validate the immutable completed v3 artifacts",
                "copy the transitively hash-bound base prior_no_go into an in-memory lock view",
                "run the frozen model-free summary and result serializer",
                "write only the previously absent result, amendment result, report, and repair record",
                "route the locked conditional cross-encoding no-go path through the repaired replay",
            ],
            "forbidden": [
                "v3 prepare_reuse_ledger",
                "v3 run_development",
                "model load",
                "model forward or backward",
                "trial recovery or retry",
                "prompt, threshold, solver, direction, state, terminal, ledger, or final mutation",
                "sealed evaluation access",
            ],
            "new_model_compute": 0,
            "changes_experimental_outcomes": False,
            "changes_claim_boundary": False,
        },
        "v3": {
            "lock_path": _relative(V3_LOCK_PATH),
            "lock_file_sha256": file_sha256(V3_LOCK_PATH),
            "lock_identity_sha256": v3_lock["lock_identity_sha256"],
            "cross_lock_path": _relative(V3_CROSS_LOCK_PATH),
            "cross_lock_file_sha256": file_sha256(V3_CROSS_LOCK_PATH),
            "cross_lock_identity_sha256": V3_CROSS_LOCK_IDENTITY_SHA256,
            "cross_preflight_path": _relative(CROSS_PREFLIGHT_PATH),
            "cross_preflight_file_sha256": file_sha256(CROSS_PREFLIGHT_PATH),
            "artifact_root": _relative(V3_ARTIFACT_ROOT),
            "artifact_inventory": inventory,
            "preflight_file_sha256": file_sha256(V3_PREFLIGHT_PATH),
            "wiring_preflight_file_sha256": file_sha256(V3_WIRING_PREFLIGHT_PATH),
            "ledger_file_sha256": file_sha256(V3_LEDGER_PATH),
            "ledger_sha256": EXPECTED_LEDGER_SHA256,
            "final_file_sha256": file_sha256(V3_FINAL_PATH),
            "final_checkpoint_sha256": EXPECTED_FINAL_CHECKPOINT_SHA256,
        },
        "compatibility_field": {
            "field": "prior_no_go",
            "value": prior,
            "value_sha256": canonical_sha256(prior),
            "source": "v3.v2_lock -> v2.v1_lock -> v1.base_lock -> base.prior_no_go",
            "used_only_in_the_serializer_lock_copy": True,
        },
        "upstream_finite_calibration_ledger": _finite_ledger_boundary(configured_core()),
        "partial_serialization_history": _partial_serialization_history(),
        "observed_outcome_boundary": {
            "terminal_statuses": dict(EXPECTED_TERMINAL_STATUSES),
            "summary": dict(EXPECTED_SUMMARY),
            "compute": dict(EXPECTED_COMPUTE),
            "expected_result_sha256": EXPECTED_RESULT_SHA256,
        },
        "cross_no_go_required_zero": {
            "scenario_root_exists": False,
            "event_count": 0,
            "complete_event_count": 0,
            "forward_evaluations": 0,
            "backward_evaluations": 0,
            "forward_backward": 0,
            "final_forward_only": 0,
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judges": 0,
            "paid_model_cost_usd": 0,
        },
        "allowed_output_paths": [
            _relative(LOCKED_V3_RESULT_PATH),
            _relative(V3_AMENDMENT_RESULT_PATH),
            _relative(V3_REPORT_PATH),
            _relative(REPAIR_RESULT_PATH),
            _relative(CROSS_LEDGER_PATH),
            _relative(CROSS_RESULT_PATH),
            _relative(CROSS_REPORT_PATH),
        ],
        "sources": _source_records(),
        "claim_boundary": v3_lock["claim_boundary"],
    }
    return _with_hash(value, "lock_identity_sha256")


def load_lock() -> dict[str, Any]:
    value = _load_json(LOCK_PATH)
    _verify_hash(value, "lock_identity_sha256")
    if value.get("schema_version") != LOCK_SCHEMA or value != proposed_lock():
        raise RuntimeError("serialization amendment lock differs from its bound record")
    return value


def _forbidden_recovery(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError(
        "serialization-only repair forbids prepare_reuse_ledger and run_development"
    )


def _forbidden_core_run_development_template() -> Any:
    raise RuntimeError("serialization-only repair forbids core run_development")


def _forbidden_model_load(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("serialization-only repair forbids model/backend loading")


def configured_core() -> ModuleType:
    global _CONFIGURED_CORE, _ORIGINAL_BUILD_RESULT, _ORIGINAL_CALIBRATION_LEDGER
    if _CONFIGURED_CORE is not None:
        return _CONFIGURED_CORE
    v3 = _v3()
    core = v3.configured_core()
    original_build_result = core._build_result
    prior_no_go = _lineage_prior_no_go()
    finite = core._finite()
    original_runner_loader = finite._load_original_runner
    original_calibration_ledger = finite.CalibrationLedger
    core_run_development_guard = FunctionType(
        _forbidden_core_run_development_template.__code__,
        core.__dict__,
        "run_development",
    )

    def model_guarded_runner_loader() -> ModuleType:
        original_runner = original_runner_loader()
        original_runner.load_backend = _forbidden_model_load
        return original_runner

    class ReadOnlyCalibrationLedger(original_calibration_ledger):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            path = kwargs.get("path")
            if path is None and args:
                path = args[0]
            if path is None or Path(path).resolve() != Path(finite.LEDGER_PATH).resolve():
                raise RuntimeError("serialization repair received an unexpected finite ledger path")
            _finite_ledger_boundary_for_finite(finite)
            super().__init__(*args, **kwargs)

        def _persist(self) -> None:
            raise RuntimeError("serialization-only repair forbids finite ledger persistence")

    def compatible_build_result(
        *,
        lock: Mapping[str, Any],
        preflight: Mapping[str, Any],
        ledger: Any,
        terminals: Mapping[str, Mapping[str, Any]],
        final_metadata: Mapping[str, Any] | None,
        summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        if lock.get("lock_identity_sha256") != V3_LOCK_IDENTITY_SHA256:
            raise RuntimeError("serializer received a non-v3 lock")
        if "prior_no_go" in lock:
            raise RuntimeError("serializer repair refuses to replace an existing prior_no_go")
        compatibility_view = dict(lock)
        compatibility_view["prior_no_go"] = dict(prior_no_go)
        return original_build_result(
            lock=compatibility_view,
            preflight=preflight,
            ledger=ledger,
            terminals=terminals,
            final_metadata=final_metadata,
            summary=summary,
        )

    _ORIGINAL_BUILD_RESULT = original_build_result
    _ORIGINAL_CALIBRATION_LEDGER = original_calibration_ledger
    core._build_result = compatible_build_result
    core.run_development = core_run_development_guard
    finite._load_original_runner = model_guarded_runner_loader
    finite.CalibrationLedger = ReadOnlyCalibrationLedger
    v3.prepare_reuse_ledger = _forbidden_recovery
    v3.run_core = _forbidden_recovery
    _CONFIGURED_CORE = core
    return core


def _validate_fixed_boundary(lock: Mapping[str, Any]) -> dict[str, Any]:
    inventory = _artifact_inventory()
    expected_inventory = lock["v3"]["artifact_inventory"]
    if (
        inventory != expected_inventory
        or inventory["file_count"] != EXPECTED_ARTIFACT_COUNT
        or inventory["total_bytes"] != EXPECTED_ARTIFACT_BYTES
        or inventory["inventory_sha256"] != EXPECTED_ARTIFACT_INVENTORY_SHA256
        or file_sha256(V3_PREFLIGHT_PATH) != EXPECTED_PREFLIGHT_FILE_SHA256
        or file_sha256(V3_WIRING_PREFLIGHT_PATH)
        != EXPECTED_WIRING_PREFLIGHT_FILE_SHA256
        or file_sha256(V3_LEDGER_PATH) != EXPECTED_LEDGER_FILE_SHA256
        or file_sha256(V3_FINAL_PATH) != EXPECTED_FINAL_FILE_SHA256
    ):
        raise RuntimeError("completed v3 artifact boundary differs")
    preflight = _load_json(V3_PREFLIGHT_PATH)
    _verify_hash(preflight, "preflight_sha256")
    wiring = _load_json(V3_WIRING_PREFLIGHT_PATH)
    _verify_hash(wiring, "preflight_sha256")
    ledger = _load_json(V3_LEDGER_PATH)
    _verify_hash(ledger, "ledger_sha256")
    events = ledger.get("events")
    if (
        ledger.get("lock_identity_sha256") != V3_LOCK_IDENTITY_SHA256
        or ledger.get("ledger_sha256") != EXPECTED_LEDGER_SHA256
        or not isinstance(events, list)
        or len(events) != EXPECTED_COMPUTE["event_count"]
        or any(event.get("status") != "complete" for event in events)
        or sum(int(event["forward_evaluations"]) for event in events)
        != EXPECTED_COMPUTE["forward_evaluations"]
        or sum(int(event["backward_evaluations"]) for event in events)
        != EXPECTED_COMPUTE["backward_evaluations"]
    ):
        raise RuntimeError("v3 ledger is incomplete or differs")
    return preflight


def assemble_existing_result() -> dict[str, Any]:
    """Assemble a result only from already completed, immutable v3 artifacts."""

    lock = load_lock()
    before = _artifact_inventory()
    preflight = _validate_fixed_boundary(lock)
    core = configured_core()
    v3_lock = core._load_lock()
    if v3_lock != _load_v3_lock():
        raise RuntimeError("configured core v3 lock differs")

    import torch

    upstream_before = _finite_ledger_boundary(core)
    if upstream_before != lock["upstream_finite_calibration_ledger"]:
        raise RuntimeError("upstream finite ledger boundary differs before input loading")
    inputs = core._load_locked_inputs(torch)
    upstream_after = _finite_ledger_boundary(core)
    if upstream_after != upstream_before:
        raise RuntimeError("model-free input loading changed the upstream finite ledger")
    ledger = core.ComputeLedger(
        path=V3_LEDGER_PATH,
        lock_identity_sha256=V3_LOCK_IDENTITY_SHA256,
    )
    ledger.require_unambiguous()
    if ledger.snapshot() != EXPECTED_COMPUTE:
        raise RuntimeError("validated v3 compute snapshot differs")

    terminals: dict[str, dict[str, Any]] = {}
    for scenario_id in inputs["scenario_ids"]:
        _, terminal = core._load_existing_scenario(
            torch,
            scenario_id=scenario_id,
            ledger=ledger,
        )
        if terminal is None:
            raise RuntimeError(f"completed scenario lacks terminal: {scenario_id}")
        terminals[scenario_id] = terminal
    statuses = {scenario_id: terminal["status"] for scenario_id, terminal in terminals.items()}
    if statuses != EXPECTED_TERMINAL_STATUSES:
        raise RuntimeError("v3 terminal statuses differ")

    successful = [
        scenario_id
        for scenario_id in inputs["scenario_ids"]
        if terminals[scenario_id]["status"] == "success"
    ]
    final = None
    if successful:
        if not V3_FINAL_PATH.is_file():
            raise RuntimeError("successful v3 scenarios lack their completed final artifact")
        ledger.require_artifact(
            work_id="final:successful_scenarios:full_logits",
            path=V3_FINAL_PATH,
        )
        final = core._load_tensor_checkpoint(
            torch,
            path=V3_FINAL_PATH,
            schema=core.FINAL_SCHEMA,
        )
        metadata = final[0]
        if (
            metadata.get("checkpoint_sha256") != EXPECTED_FINAL_CHECKPOINT_SHA256
            or metadata.get("successful_scenario_ids") != successful
            or metadata.get("record_count") != EXPECTED_SUMMARY["final_row_count"]
        ):
            raise RuntimeError("completed v3 final metadata differs")
    elif V3_FINAL_PATH.exists():
        raise RuntimeError("v3 final artifact exists without a successful scenario")

    summary = core._summarize_final(
        torch,
        inputs=inputs,
        terminals=terminals,
        final=final,
    )
    if summary != EXPECTED_SUMMARY:
        raise RuntimeError("model-free v3 summary differs from the post-outcome boundary")
    result = core._build_result(
        lock=v3_lock,
        preflight=preflight,
        ledger=ledger,
        terminals=terminals,
        final_metadata=None if final is None else final[0],
        summary=summary,
    )
    if (
        result.get("result_sha256") != EXPECTED_RESULT_SHA256
        or result.get("status") != "development_no_go"
        or result.get("summary") != EXPECTED_SUMMARY
        or result.get("compute") != EXPECTED_COMPUTE
    ):
        raise RuntimeError("serialization-only v3 result differs from the fixed expectation")
    after = _artifact_inventory()
    if before != after:
        raise RuntimeError("model-free result assembly changed the v3 artifact boundary")
    return result


def preflight() -> dict[str, Any]:
    result = assemble_existing_result()
    return _with_hash(
        {
            "schema_version": "sp_lense.closed_loop_dms_result_serialization_preflight.v1",
            "status": "ready_to_serialize_without_model_compute",
            "lock_identity_sha256": load_lock()["lock_identity_sha256"],
            "expected_result_sha256": result["result_sha256"],
            "expected_status": result["status"],
            "artifact_inventory": _artifact_inventory(),
            "model_loads": 0,
            "model_forwards": 0,
            "model_backwards": 0,
            "generated_tokens": 0,
            "recovery_entrypoints_called": 0,
        },
        "preflight_sha256",
    )


def _amendment_result(result: Mapping[str, Any]) -> dict[str, Any]:
    v3 = _v3()
    observed_wiring = _load_json(V3_WIRING_PREFLIGHT_PATH)
    _verify_hash(observed_wiring, "preflight_sha256")
    expected = v3._amendment_result(result)
    if file_sha256(V3_WIRING_PREFLIGHT_PATH) != EXPECTED_WIRING_PREFLIGHT_FILE_SHA256:
        raise RuntimeError("v3 amendment result assembly changed its wiring preflight")
    return expected


def _repair_result(
    core_result: Mapping[str, Any], amendment_result: Mapping[str, Any]
) -> dict[str, Any]:
    lock = load_lock()
    return _with_hash(
        {
            "schema_version": REPAIR_RESULT_SCHEMA,
            "status": "completed_without_model_compute",
            "post_outcome_serialization_only": True,
            "outcomes_already_observed_before_repair": True,
            "lock_file_sha256": file_sha256(LOCK_PATH),
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "v3_lock_identity_sha256": V3_LOCK_IDENTITY_SHA256,
            "v3_artifact_inventory": _artifact_inventory(),
            "compatibility_field_sha256": lock["compatibility_field"]["value_sha256"],
            "partial_serialization_history": dict(lock["partial_serialization_history"]),
            "core_result": {
                "path": _relative(V3_RESULT_PATH),
                "file_sha256": file_sha256(V3_RESULT_PATH),
                "result_sha256": core_result["result_sha256"],
                "status": core_result["status"],
            },
            "amendment_result": {
                "path": _relative(V3_AMENDMENT_RESULT_PATH),
                "file_sha256": file_sha256(V3_AMENDMENT_RESULT_PATH),
                "amendment_result_sha256": amendment_result["amendment_result_sha256"],
            },
            "summary": dict(core_result["summary"]),
            "source_model_compute": dict(core_result["compute"]),
            "repair_compute": {
                "model_loads": 0,
                "forward_evaluations": 0,
                "backward_evaluations": 0,
                "generated_tokens": 0,
                "external_api_calls": 0,
                "external_model_judges": 0,
                "paid_model_cost_usd": 0,
            },
            "v3_artifacts_mutated": False,
            "experimental_outcomes_changed": False,
            "claim_boundary": lock["claim_boundary"],
        },
        "repair_result_sha256",
    )


def serialize() -> dict[str, Any]:
    before = _artifact_inventory()
    core_result = assemble_existing_result()
    _write_or_validate_json(V3_RESULT_PATH, core_result, hash_field="result_sha256")
    amendment_result = _amendment_result(core_result)
    _write_or_validate_json(
        V3_AMENDMENT_RESULT_PATH,
        amendment_result,
        hash_field="amendment_result_sha256",
    )
    repair_result = _repair_result(core_result, amendment_result)
    _write_or_validate_json(
        REPAIR_RESULT_PATH,
        repair_result,
        hash_field="repair_result_sha256",
    )
    if _artifact_inventory() != before:
        raise RuntimeError("serialization changed a v3 model artifact")
    return replay()


def replay() -> dict[str, Any]:
    expected_core = assemble_existing_result()
    if not V3_RESULT_PATH.is_file():
        raise RuntimeError("serialized v3 development result is missing")
    observed_core = _load_json(V3_RESULT_PATH)
    _verify_hash(observed_core, "result_sha256")
    if observed_core != expected_core:
        raise RuntimeError("serialized v3 development result differs")
    expected_amendment = _amendment_result(expected_core)
    if not V3_AMENDMENT_RESULT_PATH.is_file():
        raise RuntimeError("serialized v3 amendment result is missing")
    observed_amendment = _load_json(V3_AMENDMENT_RESULT_PATH)
    _verify_hash(observed_amendment, "amendment_result_sha256")
    if observed_amendment != expected_amendment:
        raise RuntimeError("serialized v3 amendment result differs")
    expected_repair = _repair_result(expected_core, expected_amendment)
    if not REPAIR_RESULT_PATH.is_file():
        raise RuntimeError("serialization repair result is missing")
    observed_repair = _load_json(REPAIR_RESULT_PATH)
    _verify_hash(observed_repair, "repair_result_sha256")
    if observed_repair != expected_repair:
        raise RuntimeError("serialization repair result differs")
    return observed_repair


def report() -> str:
    replay()
    return configured_core().run_report()


def configured_cross() -> ModuleType:
    v3 = _v3()
    core = configured_core()
    cross = v3.configured_cross()
    if cross._CORE is not core:
        raise RuntimeError("cross-encoding runner does not use the repaired core")
    return cross


def _precheck_cross_no_go_boundary(cross: ModuleType) -> None:
    if Path(cross.SCENARIO_ROOT).exists():
        raise RuntimeError("core no-go requires the cross scenario root to be absent")
    ledger_path = Path(cross.LEDGER_PATH)
    if ledger_path.exists():
        ledger = _load_json(ledger_path)
        _verify_hash(ledger, "ledger_sha256")
        if ledger.get("events") != []:
            raise RuntimeError("core no-go cross ledger already contains an event")
    elif Path(cross.RESULT_PATH).exists():
        raise RuntimeError("cross result exists without its zero-event ledger")


def _validate_cross_no_go_result(cross: ModuleType, result: Mapping[str, Any]) -> dict[str, Any]:
    _verify_hash(result, "result_sha256")
    ledger_path = Path(cross.LEDGER_PATH)
    if not ledger_path.is_file():
        raise RuntimeError("cross no-go result lacks its zero-event ledger")
    ledger = _load_json(ledger_path)
    _verify_hash(ledger, "ledger_sha256")
    events = ledger.get("events")
    compute = result.get("compute")
    if not isinstance(events, list) or not isinstance(compute, Mapping):
        raise TypeError("cross no-go ledger or compute payload differs")
    audit = {
        "scenario_root_exists": Path(cross.SCENARIO_ROOT).exists(),
        "event_count": len(events),
        "complete_event_count": sum(event.get("status") == "complete" for event in events),
        "forward_evaluations": int(compute.get("forward_evaluations", -1)),
        "backward_evaluations": int(compute.get("backward_evaluations", -1)),
        "forward_backward": int(compute.get("backward_evaluations", -1)),
        "final_forward_only": int(compute.get("forward_evaluations", -1))
        - int(compute.get("backward_evaluations", -1)),
        "generated_tokens": int(compute.get("generated_tokens", -1)),
        "external_api_calls": int(compute.get("external_api_calls", -1)),
        "external_model_judges": int(compute.get("external_model_judges", -1)),
        "paid_model_cost_usd": compute.get("paid_model_cost_usd"),
    }
    if (
        result.get("status") != "not_run_core_no_go"
        or result.get("summary") is not None
        or result.get("model_passes_when_core_no_go") != 0
        or result.get("cross_encoding_gradients") != 0
        or result.get("controller_updates_from_cross_encoding_outcomes") != 0
        or compute.get("completed_scenario_events") != 0
        or events
        or audit != load_lock()["cross_no_go_required_zero"]
    ):
        raise RuntimeError("conditional cross-encoding no-go boundary differs")
    return audit


def _run_cross_entrypoint(*, cross: ModuleType, replay_core: Any) -> dict[str, Any]:
    replay_core()
    _precheck_cross_no_go_boundary(cross)
    result = cross.run_extension()
    _validate_cross_no_go_result(cross, result)
    return dict(result)


def run_cross() -> dict[str, Any]:
    result = _run_cross_entrypoint(cross=configured_cross(), replay_core=replay)
    if (
        result.get("status") != "not_run_core_no_go"
        or result.get("model_passes_when_core_no_go") != 0
        or result.get("compute", {}).get("forward_evaluations") != 0
    ):
        raise RuntimeError("conditional cross-encoding no-go boundary differs")
    return result


def replay_cross() -> dict[str, Any]:
    replay()
    cross = configured_cross()
    _precheck_cross_no_go_boundary(cross)
    result = cross.run_replay()
    _validate_cross_no_go_result(cross, result)
    return result


def report_cross() -> str:
    replay_cross()
    return configured_cross().run_report()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-outcome, zero-model-compute CL-DMS result serialization repair"
    )
    parser.add_argument(
        "command",
        choices=(
            "validate-lock",
            "preflight",
            "serialize",
            "replay",
            "report",
            "cross-run",
            "cross-replay",
            "cross-report",
        ),
    )
    args = parser.parse_args()
    commands = {
        "validate-lock": load_lock,
        "preflight": preflight,
        "serialize": serialize,
        "replay": replay,
        "report": report,
        "cross-run": run_cross,
        "cross-replay": replay_cross,
        "cross-report": report_cross,
    }
    value = commands[args.command]()
    print(value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
