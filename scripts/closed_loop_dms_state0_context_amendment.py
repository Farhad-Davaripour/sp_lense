from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from sp_lense.factorial_causal_anchor import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
DOC_PATH = ROOT / "docs" / "CLOSED_LOOP_DMS_STATE0_CONTEXT_AMENDMENT.md"
TEST_PATH = ROOT / "tests" / "test_closed_loop_dms_state0_context_amendment.py"
V1_RUNNER_PATH = ROOT / "scripts" / "closed_loop_dms_state0_metadata_amendment.py"
BASE_CROSS_PATH = ROOT / "scripts" / "closed_loop_dms_cross_encoding.py"
V1_LOCK_PATH = ROOT / "configs" / "closed_loop_dms_state0_metadata_amendment_lock.json"
V1_CROSS_LOCK_PATH = ROOT / "configs" / "closed_loop_dms_cross_encoding_state0_amendment_lock.json"
V1_ARTIFACT_ROOT = ROOT / "artifacts" / "closed_loop_dms_state0_metadata_amendment" / "qwen35_08b"
V1_PREFLIGHT_PATH = V1_ARTIFACT_ROOT / "preflight.json"
V1_LEDGER_PATH = V1_ARTIFACT_ROOT / "compute_ledger.json"
V1_UNRELATED_PATH = V1_ARTIFACT_ROOT / "calibration_unrelated_state0.pt"

CORE_LOCK_PATH = ROOT / "configs" / "closed_loop_dms_state0_context_amendment_lock.json"
CROSS_LOCK_PATH = (
    ROOT / "configs" / "closed_loop_dms_cross_encoding_state0_context_amendment_lock.json"
)
CORE_ARTIFACT_ROOT = ROOT / "artifacts" / "closed_loop_dms_state0_context_amendment" / "qwen35_08b"
V1_CONTEXT_FAILURE_PATH = CORE_ARTIFACT_ROOT / "v1_context_failure.json"
CORE_PREFLIGHT_PATH = CORE_ARTIFACT_ROOT / "preflight.json"
CORE_LEDGER_PATH = CORE_ARTIFACT_ROOT / "compute_ledger.json"
CORE_SCENARIO_ROOT = CORE_ARTIFACT_ROOT / "scenarios"
CORE_FINAL_PATH = CORE_ARTIFACT_ROOT / "final_evaluation.pt"
CORE_RESULT_ROOT = ROOT / "results" / "closed_loop_dms_state0_context_amendment" / "qwen35_08b"
CORE_RESULT_PATH = CORE_RESULT_ROOT / "development_result.json"
CORE_REPORT_PATH = CORE_RESULT_ROOT / "DEVELOPMENT_REPORT.md"
CORE_AMENDMENT_RESULT_PATH = CORE_RESULT_ROOT / "amendment_result.json"

CROSS_ARTIFACT_ROOT = (
    ROOT / "artifacts" / "closed_loop_dms_cross_encoding_state0_context_amendment" / "qwen35_08b"
)
CROSS_PREFLIGHT_PATH = CROSS_ARTIFACT_ROOT / "preflight.json"
CROSS_LEDGER_PATH = CROSS_ARTIFACT_ROOT / "compute_ledger.json"
CROSS_SCENARIO_ROOT = CROSS_ARTIFACT_ROOT / "scenarios"
CROSS_RESULT_PATH = (
    ROOT
    / "results"
    / "closed_loop_dms_cross_encoding_state0_context_amendment"
    / "qwen35_08b"
    / "result.json"
)
CROSS_REPORT_PATH = CROSS_RESULT_PATH.with_name("REPORT.md")

CORE_LOCK_SCHEMA = "sp_lense.closed_loop_dms_state0_context_amendment_lock.v1"
V1_CONTEXT_FAILURE_SCHEMA = "sp_lense.closed_loop_dms_v1_context_failure.v1"
AMENDMENT_RESULT_SCHEMA = "sp_lense.closed_loop_dms_state0_context_amendment_result.v1"
CROSS_LOCK_SCHEMA = "sp_lense.closed_loop_dms_cross_encoding_state0_context_amendment_lock.v1"

V1_RUNNER_FILE_SHA256 = "cdd0c88fae166591db479056a05563f1230bed96d06f41fb1ce5743829937b2f"
V1_LOCK_FILE_SHA256 = "1e6732ca463c91a266d881b95b1731ac30ff66c8892ac43736afba955bc18e4b"
V1_LOCK_IDENTITY_SHA256 = "abc4949f1ad2bc617ef623233b7d177b03b6b49964ad28bd3910dcefd8ef3471"
V1_PREFLIGHT_SHA256 = "a7249877381e5e65f9d0a07627f1d9b7b71233c458ac2fdec7d0470cc65c9852"
V1_LEDGER_SHA256 = "b245f142be973b35b117a1190ee827207c3727d3be290a37e9982decdf03a444"
V1_UNRELATED_FILE_SHA256 = "c95538d96774ef327c8b23676bb6b9e3d567c9c7c7edbe30c749e4ce58b2d979"
V1_UNRELATED_CHECKPOINT_SHA256 = "0983f3b0548dd793b22f439b11198607a871bfbc70ea6c8358781dc2d8a604a4"
V1_CHARGED_FB = 8
BASE_FAILED_CHARGED_FB = 8
TOTAL_PRIOR_CHARGED_FB = BASE_FAILED_CHARGED_FB + V1_CHARGED_FB
REUSE_WORK_ID = "state0:calibration_unrelated:8_unique_forms"

_V1: ModuleType | None = None
_CONFIGURED_CORE: ModuleType | None = None
_CONFIGURED_CROSS: ModuleType | None = None


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


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable {_relative(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import {_relative(path)}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _v1() -> ModuleType:
    global _V1
    if _V1 is None:
        if file_sha256(V1_RUNNER_PATH) != V1_RUNNER_FILE_SHA256:
            raise RuntimeError("frozen v1 state0 metadata amendment runner differs")
        _V1 = _load_module(V1_RUNNER_PATH, "closed_loop_dms_frozen_state0_metadata_v1")
    return _V1


def _v1_lock() -> dict[str, Any]:
    if file_sha256(V1_LOCK_PATH) != V1_LOCK_FILE_SHA256:
        raise RuntimeError("v1 state0 metadata amendment lock file differs")
    value = _load_json(V1_LOCK_PATH)
    _verify_hash(value, "lock_identity_sha256")
    if value.get("lock_identity_sha256") != V1_LOCK_IDENTITY_SHA256:
        raise RuntimeError("v1 state0 metadata amendment lock identity differs")
    return value


def _v1_context_failure_value() -> dict[str, Any]:
    return _with_hash(
        {
            "schema_version": V1_CONTEXT_FAILURE_SCHEMA,
            "status": "v1_stopped_after_complete_state0_capture_before_steering",
            "v1_lock_identity_sha256": V1_LOCK_IDENTITY_SHA256,
            "v1_ledger_path": _relative(V1_LEDGER_PATH),
            "v1_ledger_file_sha256": file_sha256(V1_LEDGER_PATH),
            "v1_ledger_sha256": V1_LEDGER_SHA256,
            "v1_unrelated_checkpoint_path": _relative(V1_UNRELATED_PATH),
            "v1_unrelated_checkpoint_file_sha256": file_sha256(V1_UNRELATED_PATH),
            "v1_unrelated_checkpoint_sha256": V1_UNRELATED_CHECKPOINT_SHA256,
            "error_type": "KeyError",
            "error_message": "'prompt_sha256'",
            "failure_site": "_state0_checkpoint while building shared scenario contexts",
            "root_cause": (
                "v1 enriched only the private capture-call input copy, not the locked inputs "
                "subsequently consumed by the shared context builder"
            ),
            "completed_and_persisted_forward_backward_captures": V1_CHARGED_FB,
            "completed_checkpoint_reusable": True,
            "partial_outputs": False,
            "steering_trial_forwards": 0,
            "self_preservation_intervention_outcomes_evaluated": False,
            "retired_pilot_intervention_outcomes_evaluated": False,
            "retry_authorized_by_this_record": False,
        },
        "failure_sha256",
    )


def record_v1_context_failure() -> dict[str, Any]:
    v1 = _v1()
    _v1_lock()
    preflight = _load_json(V1_PREFLIGHT_PATH)
    _verify_hash(preflight, "preflight_sha256")
    if preflight.get("preflight_sha256") != V1_PREFLIGHT_SHA256:
        raise RuntimeError("v1 preflight differs")
    core = v1.configured_core()
    ledger = core.ComputeLedger(
        path=V1_LEDGER_PATH,
        lock_identity_sha256=V1_LOCK_IDENTITY_SHA256,
    )
    ledger.require_unambiguous()
    ledger.require_artifact(work_id=REUSE_WORK_ID, path=V1_UNRELATED_PATH)
    if (
        ledger.snapshot()["forward_backward"] != V1_CHARGED_FB
        or file_sha256(V1_UNRELATED_PATH) != V1_UNRELATED_FILE_SHA256
        or core.RESULT_PATH.exists()
        or any(core.SCENARIO_ROOT.glob("*/state-*.pt"))
        or any(core.SCENARIO_ROOT.glob("*/terminal.json"))
    ):
        raise RuntimeError("v1 outcome boundary differs")
    import torch

    metadata, tensors = core._load_tensor_checkpoint(
        torch, path=V1_UNRELATED_PATH, schema=core.UNRELATED_SCHEMA
    )
    if (
        metadata.get("checkpoint_sha256") != V1_UNRELATED_CHECKPOINT_SHA256
        or metadata.get("record_count") != 8
        or tuple(tensors["raw_gradients"].shape) != (8, 1024)
        or tuple(tensors["pre_anchor_residuals"].shape) != (8, 1024)
    ):
        raise RuntimeError("v1 reusable state0 checkpoint differs")
    expected = _v1_context_failure_value()
    if V1_CONTEXT_FAILURE_PATH.exists():
        observed = _load_json(V1_CONTEXT_FAILURE_PATH)
        _verify_hash(observed, "failure_sha256")
        if observed != expected:
            raise RuntimeError("existing v1 context failure record differs")
        return observed
    _write_new_json(V1_CONTEXT_FAILURE_PATH, expected)
    return expected


def _source_records() -> dict[str, dict[str, str]]:
    paths = {
        "context_amendment_runner": SCRIPT_PATH,
        "context_amendment_protocol": DOC_PATH,
        "context_amendment_tests": TEST_PATH,
        "frozen_v1_metadata_amendment_runner": V1_RUNNER_PATH,
        "frozen_base_cross_runner": BASE_CROSS_PATH,
    }
    return {
        name: {"path": _relative(path), "sha256": file_sha256(path)} for name, path in paths.items()
    }


def proposed_core_lock() -> dict[str, Any]:
    v1_lock = _v1_lock()
    failure = record_v1_context_failure()
    base_failed = v1_lock["base_failed_attempt"]
    value = {
        "schema_version": CORE_LOCK_SCHEMA,
        "status": "prospective_context_propagation_lock_before_first_steering_trial",
        "development_only": True,
        "v1_lock": {
            "path": _relative(V1_LOCK_PATH),
            "file_sha256": file_sha256(V1_LOCK_PATH),
            "lock_identity_sha256": v1_lock["lock_identity_sha256"],
        },
        "v1_preflight": {
            "path": _relative(V1_PREFLIGHT_PATH),
            "file_sha256": file_sha256(V1_PREFLIGHT_PATH),
            "preflight_sha256": V1_PREFLIGHT_SHA256,
        },
        "v1_context_failure": {
            "path": _relative(V1_CONTEXT_FAILURE_PATH),
            "file_sha256": file_sha256(V1_CONTEXT_FAILURE_PATH),
            "failure_sha256": failure["failure_sha256"],
        },
        "reused_state0_checkpoint": {
            "path": _relative(V1_UNRELATED_PATH),
            "file_sha256": file_sha256(V1_UNRELATED_PATH),
            "checkpoint_sha256": V1_UNRELATED_CHECKPOINT_SHA256,
            "source_ledger_path": _relative(V1_LEDGER_PATH),
            "source_ledger_file_sha256": file_sha256(V1_LEDGER_PATH),
            "source_ledger_sha256": V1_LEDGER_SHA256,
            "new_model_compute": 0,
        },
        "model": dict(v1_lock["model"]),
        "runtime": dict(v1_lock["runtime"]),
        "chat_template_sha256": v1_lock["chat_template_sha256"],
        "dataset": dict(v1_lock["dataset"]),
        "design": {
            **dict(v1_lock["design"]),
            "state0_context_amendment": {
                "scope": "locked_inputs_spec_by_form_for_eight_calibration_unrelated_forms",
                "operation": "apply_the_v1_verified_two_hash_derivation_before_any_consumer",
                "reused_checkpoint": V1_UNRELATED_CHECKPOINT_SHA256,
                "changes_prompts_labels_anchors_tensors_gradients_solver_or_thresholds": False,
                "steering_outcomes_available_when_authored": False,
            },
        },
        "fresh_controller_compute_ceiling": {
            **dict(v1_lock["fresh_retry_compute_ceiling"]),
            "state0_new_forward_backward": 0,
            "maximum_new_controller_forward_backward": 9600,
        },
        "total_study_compute_ceiling": {
            "forward": 9808,
            "backward": 9616,
            "forward_backward": 9616,
            "final_forward_only": 192,
            "prior_charged_forward_backward": TOTAL_PRIOR_CHARGED_FB,
            "prior_observed_actual_forward_backward": (
                int(base_failed["observed_actual_forward_backward"]) + V1_CHARGED_FB
            ),
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judges": 0,
            "paid_model_cost_usd": 0,
        },
        "artifact_namespace": _relative(CORE_ARTIFACT_ROOT),
        "result_namespace": _relative(CORE_RESULT_ROOT),
        "sources": _source_records(),
        "claim_boundary": v1_lock["claim_boundary"],
    }
    return _with_hash(value, "lock_identity_sha256")


def run_core_lock() -> dict[str, Any]:
    if CORE_RESULT_PATH.exists() or any(CORE_SCENARIO_ROOT.glob("*/terminal.json")):
        raise RuntimeError("context amendment lock must predate every steering outcome")
    value = proposed_core_lock()
    if CORE_LOCK_PATH.exists():
        observed = _load_json(CORE_LOCK_PATH)
        _verify_hash(observed, "lock_identity_sha256")
        if observed != value:
            raise RuntimeError("existing state0 context amendment lock differs")
        return observed
    _write_new_json(CORE_LOCK_PATH, value)
    return value


def _load_core_lock() -> dict[str, Any]:
    observed = _load_json(CORE_LOCK_PATH)
    _verify_hash(observed, "lock_identity_sha256")
    if observed != proposed_core_lock():
        raise RuntimeError("state0 context amendment lock differs from its bound design")
    return observed


def configured_core() -> ModuleType:
    global _CONFIGURED_CORE
    if _CONFIGURED_CORE is not None:
        return _CONFIGURED_CORE
    v1 = _v1()
    core = v1.configured_core()
    lock = _load_core_lock()
    original_load_inputs = core._load_locked_inputs

    def amended_load_inputs(torch: Any) -> dict[str, Any]:
        return v1.enrich_unrelated_form_hashes(original_load_inputs(torch))

    core.LOCK_PATH = CORE_LOCK_PATH
    core.ARTIFACT_ROOT = CORE_ARTIFACT_ROOT
    core.PREFLIGHT_PATH = CORE_PREFLIGHT_PATH
    core.LEDGER_PATH = CORE_LEDGER_PATH
    core.UNRELATED_CAPTURE_PATH = V1_UNRELATED_PATH
    core.SCENARIO_ROOT = CORE_SCENARIO_ROOT
    core.FINAL_PATH = CORE_FINAL_PATH
    core.RESULT_ROOT = CORE_RESULT_ROOT
    core.RESULT_PATH = CORE_RESULT_PATH
    core.REPORT_PATH = CORE_REPORT_PATH
    core._LOCK_CACHE = lock
    core._load_lock = lambda: lock
    core._load_locked_inputs = amended_load_inputs
    _CONFIGURED_CORE = core
    return core


def prepare_reuse_ledger() -> dict[str, Any]:
    core = configured_core()
    lock = _load_core_lock()
    ledger = core.ComputeLedger(
        path=CORE_LEDGER_PATH,
        lock_identity_sha256=lock["lock_identity_sha256"],
    )
    if ledger.event(REUSE_WORK_ID) is None:
        ledger.reserve(
            work_id=REUSE_WORK_ID,
            forward=0,
            backward=0,
            kind="validated_zero_compute_reuse_of_v1_complete_state0_checkpoint",
        )
        ledger.complete(work_id=REUSE_WORK_ID, artifact_path=V1_UNRELATED_PATH)
    ledger.require_artifact(work_id=REUSE_WORK_ID, path=V1_UNRELATED_PATH)
    snapshot = ledger.snapshot()
    if snapshot["forward_evaluations"] != 0 or snapshot["backward_evaluations"] != 0:
        raise RuntimeError("state0 checkpoint reuse ledger charged new model compute")
    return snapshot


def _amendment_result(core_result: Mapping[str, Any]) -> dict[str, Any]:
    lock = _load_core_lock()
    fresh = core_result["compute"]
    total = {
        **dict(fresh),
        "forward_evaluations": int(fresh["forward_evaluations"]) + TOTAL_PRIOR_CHARGED_FB,
        "backward_evaluations": int(fresh["backward_evaluations"]) + TOTAL_PRIOR_CHARGED_FB,
        "forward_backward": int(fresh["forward_backward"]) + TOTAL_PRIOR_CHARGED_FB,
        "prior_charged_forward_backward": TOTAL_PRIOR_CHARGED_FB,
        "prior_observed_actual_forward_backward": 9,
    }
    return _with_hash(
        {
            "schema_version": AMENDMENT_RESULT_SCHEMA,
            "status": core_result["status"],
            "development_only": True,
            "lock_file_sha256": file_sha256(CORE_LOCK_PATH),
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "v1_context_failure_sha256": record_v1_context_failure()["failure_sha256"],
            "reused_state0_checkpoint_sha256": V1_UNRELATED_CHECKPOINT_SHA256,
            "inner_result_path": _relative(CORE_RESULT_PATH),
            "inner_result_file_sha256": file_sha256(CORE_RESULT_PATH),
            "inner_result_sha256": core_result["result_sha256"],
            "fresh_controller_compute": dict(fresh),
            "total_compute_including_prior_charges": total,
            "summary": core_result["summary"],
            "pilot_execution_authorized": False,
            "paid_model_cost_usd": 0,
            "claim_boundary": lock["claim_boundary"],
        },
        "amendment_result_sha256",
    )


def _write_or_validate_result(core_result: Mapping[str, Any]) -> dict[str, Any]:
    expected = _amendment_result(core_result)
    if CORE_AMENDMENT_RESULT_PATH.exists():
        observed = _load_json(CORE_AMENDMENT_RESULT_PATH)
        _verify_hash(observed, "amendment_result_sha256")
        if observed != expected:
            raise RuntimeError("state0 context amendment result differs")
        return observed
    _write_new_json(CORE_AMENDMENT_RESULT_PATH, expected)
    return expected


def run_core_preflight() -> dict[str, Any]:
    return configured_core().run_preflight()


def run_core() -> dict[str, Any]:
    prepare_reuse_ledger()
    return _write_or_validate_result(configured_core().run_development())


def replay_core() -> dict[str, Any]:
    prepare_reuse_ledger()
    return _write_or_validate_result(configured_core().run_replay())


def report_core() -> str:
    configured_core().run_report()
    return json.dumps(replay_core(), indent=2, ensure_ascii=False) + "\n"


def configured_cross() -> ModuleType:
    global _CONFIGURED_CROSS
    if _CONFIGURED_CROSS is not None:
        return _CONFIGURED_CROSS
    core = configured_core()
    cross = _load_module(BASE_CROSS_PATH, "closed_loop_dms_cross_for_state0_context_amendment")
    base_sources = cross._source_records
    base_proposed = cross.proposed_lock

    cross._CORE = core
    cross.LOCK_SCHEMA = CROSS_LOCK_SCHEMA
    cross.LOCK_PATH = CROSS_LOCK_PATH
    cross.PREFLIGHT_PATH = CROSS_PREFLIGHT_PATH
    cross.LEDGER_PATH = CROSS_LEDGER_PATH
    cross.SCENARIO_ROOT = CROSS_SCENARIO_ROOT
    cross.RESULT_PATH = CROSS_RESULT_PATH
    cross.REPORT_PATH = CROSS_REPORT_PATH

    def amended_sources() -> dict[str, dict[str, str]]:
        result = dict(base_sources())
        additions = {
            "state0_context_amendment_runner": SCRIPT_PATH,
            "state0_context_amendment_protocol": DOC_PATH,
            "state0_context_amendment_tests": TEST_PATH,
            "state0_context_amendment_core_lock": CORE_LOCK_PATH,
            "historical_v1_cross_lock": V1_CROSS_LOCK_PATH,
        }
        result.update(
            {
                name: {"path": _relative(path), "sha256": file_sha256(path)}
                for name, path in additions.items()
            }
        )
        return result

    def amended_proposed_lock() -> dict[str, Any]:
        value = base_proposed()
        value["status"] = "prospective_cross_encoding_after_state0_context_amendment"
        value["state0_context_amendment"] = {
            "core_lock_path": _relative(CORE_LOCK_PATH),
            "core_lock_file_sha256": file_sha256(CORE_LOCK_PATH),
            "v1_context_failure_sha256": record_v1_context_failure()["failure_sha256"],
            "reused_state0_checkpoint_sha256": V1_UNRELATED_CHECKPOINT_SHA256,
            "amendment_changes_cross_encoding_design": False,
        }
        value["sources"] = amended_sources()
        return _with_hash(value, "lock_identity_sha256")

    cross._source_records = amended_sources
    cross.proposed_lock = amended_proposed_lock
    _CONFIGURED_CROSS = cross
    return cross


def run_cross_lock() -> dict[str, Any]:
    return configured_cross().run_lock()


def run_cross_preflight() -> dict[str, Any]:
    return configured_cross().run_preflight()


def run_cross() -> dict[str, Any]:
    return configured_cross().run_extension()


def replay_cross() -> dict[str, Any]:
    return configured_cross().run_replay()


def report_cross() -> str:
    return configured_cross().run_report()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prospective CL-DMS shared-context metadata propagation amendment"
    )
    parser.add_argument(
        "command",
        choices=(
            "record-v1-failure",
            "lock",
            "preflight",
            "prepare-reuse",
            "run",
            "replay",
            "report",
            "cross-lock",
            "cross-preflight",
            "cross-run",
            "cross-replay",
            "cross-report",
        ),
    )
    args = parser.parse_args()
    commands = {
        "record-v1-failure": record_v1_context_failure,
        "lock": run_core_lock,
        "preflight": run_core_preflight,
        "prepare-reuse": prepare_reuse_ledger,
        "run": run_core,
        "replay": replay_core,
        "report": report_core,
        "cross-lock": run_cross_lock,
        "cross-preflight": run_cross_preflight,
        "cross-run": run_cross,
        "cross-replay": replay_cross,
        "cross-report": report_cross,
    }
    value = commands[args.command]()
    print(value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
