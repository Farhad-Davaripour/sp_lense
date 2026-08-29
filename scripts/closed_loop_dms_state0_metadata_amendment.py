from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from sp_lense.factorial_causal_anchor import canonical_sha256, text_sha256

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
DOC_PATH = ROOT / "docs" / "CLOSED_LOOP_DMS_STATE0_METADATA_AMENDMENT.md"
TEST_PATH = ROOT / "tests" / "test_closed_loop_dms_state0_metadata_amendment.py"
BASE_CORE_PATH = ROOT / "scripts" / "closed_loop_dms_development.py"
BASE_CROSS_PATH = ROOT / "scripts" / "closed_loop_dms_cross_encoding.py"
BASE_CORE_LOCK_PATH = ROOT / "configs" / "closed_loop_dms_development_lock.json"
BASE_CROSS_LOCK_PATH = ROOT / "configs" / "closed_loop_dms_cross_encoding_lock.json"
BASE_CORE_PREFLIGHT_PATH = (
    ROOT / "artifacts" / "closed_loop_dms_development" / "qwen35_08b" / "preflight.json"
)
BASE_LEDGER_PATH = (
    ROOT / "artifacts" / "closed_loop_dms_development" / "qwen35_08b" / "compute_ledger.json"
)

CORE_LOCK_PATH = ROOT / "configs" / "closed_loop_dms_state0_metadata_amendment_lock.json"
CROSS_LOCK_PATH = ROOT / "configs" / "closed_loop_dms_cross_encoding_state0_amendment_lock.json"
CORE_ARTIFACT_ROOT = ROOT / "artifacts" / "closed_loop_dms_state0_metadata_amendment" / "qwen35_08b"
BASE_FAILURE_PATH = CORE_ARTIFACT_ROOT / "base_attempt_failure.json"
CORE_PREFLIGHT_PATH = CORE_ARTIFACT_ROOT / "preflight.json"
CORE_LEDGER_PATH = CORE_ARTIFACT_ROOT / "compute_ledger.json"
CORE_UNRELATED_PATH = CORE_ARTIFACT_ROOT / "calibration_unrelated_state0.pt"
CORE_SCENARIO_ROOT = CORE_ARTIFACT_ROOT / "scenarios"
CORE_FINAL_PATH = CORE_ARTIFACT_ROOT / "final_evaluation.pt"
CORE_RESULT_ROOT = ROOT / "results" / "closed_loop_dms_state0_metadata_amendment" / "qwen35_08b"
CORE_RESULT_PATH = CORE_RESULT_ROOT / "development_result.json"
CORE_REPORT_PATH = CORE_RESULT_ROOT / "DEVELOPMENT_REPORT.md"
CORE_AMENDMENT_RESULT_PATH = CORE_RESULT_ROOT / "amendment_result.json"

CROSS_ARTIFACT_ROOT = (
    ROOT / "artifacts" / "closed_loop_dms_cross_encoding_state0_amendment" / "qwen35_08b"
)
CROSS_PREFLIGHT_PATH = CROSS_ARTIFACT_ROOT / "preflight.json"
CROSS_LEDGER_PATH = CROSS_ARTIFACT_ROOT / "compute_ledger.json"
CROSS_SCENARIO_ROOT = CROSS_ARTIFACT_ROOT / "scenarios"
CROSS_RESULT_PATH = (
    ROOT
    / "results"
    / "closed_loop_dms_cross_encoding_state0_amendment"
    / "qwen35_08b"
    / "result.json"
)
CROSS_REPORT_PATH = CROSS_RESULT_PATH.with_name("REPORT.md")

CORE_LOCK_SCHEMA = "sp_lense.closed_loop_dms_state0_metadata_amendment_lock.v1"
BASE_FAILURE_SCHEMA = "sp_lense.closed_loop_dms_base_attempt_failure.v1"
AMENDMENT_RESULT_SCHEMA = "sp_lense.closed_loop_dms_state0_metadata_amendment_result.v1"
CROSS_LOCK_SCHEMA = "sp_lense.closed_loop_dms_cross_encoding_state0_amendment_lock.v1"

BASE_CORE_FILE_SHA256 = "6faadf9a639235df285110ac1d702a2f27e2a2b05a7683157f7a372ddc0ba850"
BASE_CORE_LOCK_IDENTITY_SHA256 = "5072a7346d98c5004d5567efd7b446fe22b2606fc6c536f1a6c752c1273f4d58"
BASE_CORE_LOCK_FILE_SHA256 = "1b580fe6a5b3ae166704ec51e4507f596d2be0fd651adb5abfb847b4edd7c7a2"
BASE_PREFLIGHT_SHA256 = "88f23a606c74928c36fb0cd411ee669dfac6b92201381a2a1ed18c501c5d44bb"
BASE_PENDING_EVENT_SHA256 = "b72dac023060d88ec9a22f3fac55d352df95883ad4318d159c5b5245226f44a2"
BASE_WORK_ID = "state0:calibration_unrelated:8_unique_forms"
BASE_CHARGED_FB = 8
BASE_OBSERVED_ACTUAL_FB = 1

_BASE_CORE: ModuleType | None = None
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


def _base_core() -> ModuleType:
    global _BASE_CORE
    if _BASE_CORE is None:
        if file_sha256(BASE_CORE_PATH) != BASE_CORE_FILE_SHA256:
            raise RuntimeError("frozen base CL-DMS runner file differs")
        _BASE_CORE = _load_module(BASE_CORE_PATH, "closed_loop_dms_frozen_base_for_amendment")
    return _BASE_CORE


def _base_lock() -> dict[str, Any]:
    if file_sha256(BASE_CORE_LOCK_PATH) != BASE_CORE_LOCK_FILE_SHA256:
        raise RuntimeError("base CL-DMS lock file differs")
    value = _load_json(BASE_CORE_LOCK_PATH)
    _verify_hash(value, "lock_identity_sha256")
    if value.get("lock_identity_sha256") != BASE_CORE_LOCK_IDENTITY_SHA256:
        raise RuntimeError("base CL-DMS lock identity differs")
    return value


def _base_preflight() -> dict[str, Any]:
    value = _load_json(BASE_CORE_PREFLIGHT_PATH)
    _verify_hash(value, "preflight_sha256")
    if (
        value.get("preflight_sha256") != BASE_PREFLIGHT_SHA256
        or value.get("model_forwards") != 0
        or value.get("model_backwards") != 0
        or value.get("pilot_outcomes_read") is not False
    ):
        raise RuntimeError("base CL-DMS preflight differs")
    return value


def _failure_value(
    *, pending_ledger_file_sha256: str, pending_ledger_sha256: str
) -> dict[str, Any]:
    return _with_hash(
        {
            "schema_version": BASE_FAILURE_SCHEMA,
            "status": "base_attempt_failed_before_any_steering_trial",
            "base_lock_identity_sha256": BASE_CORE_LOCK_IDENTITY_SHA256,
            "base_pending_ledger_file_sha256_before_closure": pending_ledger_file_sha256,
            "base_pending_ledger_sha256_before_closure": pending_ledger_sha256,
            "base_pending_event_sha256": BASE_PENDING_EVENT_SHA256,
            "work_id": BASE_WORK_ID,
            "error_type": "KeyError",
            "error_message": "'prompt_sha256'",
            "failure_site": (
                "_capture_missing_unrelated record construction after the first "
                "capture_multilayer_choice_anchor_gradient return"
            ),
            "root_cause": (
                "the finite-plan form carries prompt and anchor_prefix text but not their "
                "derived SHA-256 fields"
            ),
            "observed_actual_forward_backward_captures": BASE_OBSERVED_ACTUAL_FB,
            "conservatively_charged_forward_backward_captures": BASE_CHARGED_FB,
            "partial_gradient_outputs_persisted": False,
            "partial_gradient_outputs_used": False,
            "self_preservation_intervention_outcomes_evaluated": False,
            "retired_pilot_intervention_outcomes_evaluated": False,
            "retry_authorized_by_this_record": False,
        },
        "failure_sha256",
    )


def record_base_failure() -> dict[str, Any]:
    """Close the observed pending batch as a charged failure; never rerun it."""

    core = _base_core()
    _base_lock()
    _base_preflight()
    if BASE_FAILURE_PATH.exists():
        failure = _load_json(BASE_FAILURE_PATH)
        _verify_hash(failure, "failure_sha256")
        ledger = core.ComputeLedger(
            path=BASE_LEDGER_PATH,
            lock_identity_sha256=BASE_CORE_LOCK_IDENTITY_SHA256,
        )
        ledger.require_artifact(work_id=BASE_WORK_ID, path=BASE_FAILURE_PATH)
        return failure
    if core.UNRELATED_CAPTURE_PATH.exists() or core.RESULT_PATH.exists():
        raise RuntimeError("base failure record conflicts with a completed base outcome")
    if any(core.SCENARIO_ROOT.glob("state-*.pt")) if core.SCENARIO_ROOT.exists() else False:
        raise RuntimeError("base attempt unexpectedly contains a steering state")
    pending_file_hash = file_sha256(BASE_LEDGER_PATH)
    raw = _load_json(BASE_LEDGER_PATH)
    _verify_hash(raw, "ledger_sha256")
    events = raw.get("events")
    if (
        not isinstance(events, list)
        or len(events) != 1
        or events[0].get("work_id") != BASE_WORK_ID
        or events[0].get("event_sha256") != BASE_PENDING_EVENT_SHA256
        or events[0].get("status") != "pending"
        or events[0].get("forward_evaluations") != BASE_CHARGED_FB
        or events[0].get("backward_evaluations") != BASE_CHARGED_FB
    ):
        raise RuntimeError("observed base pending ledger differs")
    failure = _failure_value(
        pending_ledger_file_sha256=pending_file_hash,
        pending_ledger_sha256=str(raw["ledger_sha256"]),
    )
    _write_new_json(BASE_FAILURE_PATH, failure)
    ledger = core.ComputeLedger(
        path=BASE_LEDGER_PATH,
        lock_identity_sha256=BASE_CORE_LOCK_IDENTITY_SHA256,
    )
    ledger.complete(work_id=BASE_WORK_ID, artifact_path=BASE_FAILURE_PATH)
    ledger.require_artifact(work_id=BASE_WORK_ID, path=BASE_FAILURE_PATH)
    return failure


def _validate_base_failure() -> tuple[dict[str, Any], dict[str, Any]]:
    failure = record_base_failure()
    core = _base_core()
    ledger = core.ComputeLedger(
        path=BASE_LEDGER_PATH,
        lock_identity_sha256=BASE_CORE_LOCK_IDENTITY_SHA256,
    )
    ledger.require_unambiguous()
    ledger.require_artifact(work_id=BASE_WORK_ID, path=BASE_FAILURE_PATH)
    snapshot = ledger.snapshot()
    if snapshot["forward_backward"] != BASE_CHARGED_FB:
        raise RuntimeError("closed base failure compute charge differs")
    return failure, snapshot


def _source_records() -> dict[str, dict[str, str]]:
    paths = {
        "amendment_runner": SCRIPT_PATH,
        "amendment_protocol": DOC_PATH,
        "amendment_tests": TEST_PATH,
        "frozen_base_core_runner": BASE_CORE_PATH,
        "frozen_base_cross_runner": BASE_CROSS_PATH,
    }
    return {
        name: {"path": _relative(path), "sha256": file_sha256(path)} for name, path in paths.items()
    }


def proposed_core_lock() -> dict[str, Any]:
    base = _base_lock()
    preflight = _base_preflight()
    failure, failed_compute = _validate_base_failure()
    value = {
        "schema_version": CORE_LOCK_SCHEMA,
        "status": "prospective_retry_lock_after_state0_metadata_failure",
        "development_only": True,
        "base_lock": {
            "path": _relative(BASE_CORE_LOCK_PATH),
            "file_sha256": file_sha256(BASE_CORE_LOCK_PATH),
            "lock_identity_sha256": base["lock_identity_sha256"],
        },
        "base_preflight": {
            "path": _relative(BASE_CORE_PREFLIGHT_PATH),
            "file_sha256": file_sha256(BASE_CORE_PREFLIGHT_PATH),
            "preflight_sha256": preflight["preflight_sha256"],
        },
        "base_failed_attempt": {
            "path": _relative(BASE_FAILURE_PATH),
            "file_sha256": file_sha256(BASE_FAILURE_PATH),
            "failure_sha256": failure["failure_sha256"],
            "completed_ledger_path": _relative(BASE_LEDGER_PATH),
            "completed_ledger_file_sha256": file_sha256(BASE_LEDGER_PATH),
            "completed_ledger_sha256": failed_compute["ledger_sha256"],
            "charged_forward_backward": BASE_CHARGED_FB,
            "observed_actual_forward_backward": BASE_OBSERVED_ACTUAL_FB,
        },
        "model": dict(base["model"]),
        "runtime": dict(base["runtime"]),
        "chat_template_sha256": base["chat_template_sha256"],
        "dataset": dict(base["dataset"]),
        "design": {
            **dict(base["design"]),
            "state0_metadata_amendment": {
                "scope": "calibration_unrelated_finite_plan_forms_only",
                "prompt_sha256": "text_sha256(form.prompt)",
                "anchor_prefix_sha256": "text_sha256(form.anchor_prefix)",
                "required_equality": "derived_hashes_equal_immutable_finite_baseline_hashes",
                "changes_prompts_labels_anchors_gradients_solver_or_thresholds": False,
                "partial_base_output_reuse": False,
            },
        },
        "fresh_retry_compute_ceiling": dict(base["compute_ceiling"]),
        "total_study_compute_ceiling_including_failed_base_charge": {
            "forward": int(base["compute_ceiling"]["forward"]) + BASE_CHARGED_FB,
            "backward": int(base["compute_ceiling"]["backward"]) + BASE_CHARGED_FB,
            "forward_backward": int(base["compute_ceiling"]["forward_backward"]) + BASE_CHARGED_FB,
            "final_forward_only": base["compute_ceiling"]["final_forward_only"],
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judges": 0,
            "paid_model_cost_usd": 0,
        },
        "artifact_namespace": _relative(CORE_ARTIFACT_ROOT),
        "result_namespace": _relative(CORE_RESULT_ROOT),
        "sources": _source_records(),
        "claim_boundary": base["claim_boundary"],
    }
    return _with_hash(value, "lock_identity_sha256")


def run_core_lock() -> dict[str, Any]:
    if CORE_RESULT_PATH.exists() or any(CORE_SCENARIO_ROOT.glob("*/terminal.json")):
        raise RuntimeError("amendment lock must predate every amended core outcome")
    value = proposed_core_lock()
    if CORE_LOCK_PATH.exists():
        observed = _load_json(CORE_LOCK_PATH)
        _verify_hash(observed, "lock_identity_sha256")
        if observed != value:
            raise RuntimeError("existing state0 amendment lock differs")
        return observed
    _write_new_json(CORE_LOCK_PATH, value)
    return value


def _load_core_lock() -> dict[str, Any]:
    observed = _load_json(CORE_LOCK_PATH)
    _verify_hash(observed, "lock_identity_sha256")
    if observed != proposed_core_lock():
        raise RuntimeError("state0 amendment lock differs from its hash-bound design")
    return observed


def enrich_unrelated_form_hashes(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with only the two missing deterministic metadata fields added."""

    spec_by_form = dict(inputs["spec_by_form"])
    baseline_by_form = inputs["baseline_by_form"]
    for form_id in inputs["unrelated_form_ids"]:
        specification = dict(spec_by_form[form_id])
        form = dict(specification["form"])
        prompt_hash = text_sha256(str(form["prompt"]))
        anchor_hash = text_sha256(str(form["anchor_prefix"]))
        baseline_form = baseline_by_form[form_id]["form"]
        if (
            baseline_form.get("prompt_sha256") != prompt_hash
            or baseline_form.get("anchor_prefix_sha256") != anchor_hash
        ):
            raise RuntimeError("derived state0 metadata does not match the immutable baseline")
        for field, value in (
            ("prompt_sha256", prompt_hash),
            ("anchor_prefix_sha256", anchor_hash),
        ):
            if field in form and form[field] != value:
                raise RuntimeError(f"existing {field} differs from its derived value")
            form[field] = value
        specification["form"] = form
        spec_by_form[form_id] = specification
    return {**dict(inputs), "spec_by_form": spec_by_form}


def configured_core() -> ModuleType:
    global _CONFIGURED_CORE
    if _CONFIGURED_CORE is not None:
        return _CONFIGURED_CORE
    lock = _load_core_lock()
    core = _base_core()
    original_capture = core._capture_missing_unrelated

    def amended_capture(
        torch: Any,
        *,
        backend: Any,
        inputs: Mapping[str, Any],
        ledger: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return original_capture(
            torch,
            backend=backend,
            inputs=enrich_unrelated_form_hashes(inputs),
            ledger=ledger,
        )

    core.LOCK_PATH = CORE_LOCK_PATH
    core.ARTIFACT_ROOT = CORE_ARTIFACT_ROOT
    core.PREFLIGHT_PATH = CORE_PREFLIGHT_PATH
    core.LEDGER_PATH = CORE_LEDGER_PATH
    core.UNRELATED_CAPTURE_PATH = CORE_UNRELATED_PATH
    core.SCENARIO_ROOT = CORE_SCENARIO_ROOT
    core.FINAL_PATH = CORE_FINAL_PATH
    core.RESULT_ROOT = CORE_RESULT_ROOT
    core.RESULT_PATH = CORE_RESULT_PATH
    core.REPORT_PATH = CORE_REPORT_PATH
    core._LOCK_CACHE = lock
    core._load_lock = lambda: lock
    core._capture_missing_unrelated = amended_capture
    _CONFIGURED_CORE = core
    return core


def _amendment_result(core_result: Mapping[str, Any]) -> dict[str, Any]:
    lock = _load_core_lock()
    failed, _ = _validate_base_failure()
    fresh = core_result["compute"]
    total = {
        **dict(fresh),
        "forward_evaluations": int(fresh["forward_evaluations"]) + BASE_CHARGED_FB,
        "backward_evaluations": int(fresh["backward_evaluations"]) + BASE_CHARGED_FB,
        "forward_backward": int(fresh["forward_backward"]) + BASE_CHARGED_FB,
        "base_failed_batch_observed_actual_forward_backward": BASE_OBSERVED_ACTUAL_FB,
        "base_failed_batch_conservatively_charged_forward_backward": BASE_CHARGED_FB,
    }
    return _with_hash(
        {
            "schema_version": AMENDMENT_RESULT_SCHEMA,
            "status": core_result["status"],
            "development_only": True,
            "lock_file_sha256": file_sha256(CORE_LOCK_PATH),
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "base_failure_file_sha256": file_sha256(BASE_FAILURE_PATH),
            "base_failure_sha256": failed["failure_sha256"],
            "inner_result_path": _relative(CORE_RESULT_PATH),
            "inner_result_file_sha256": file_sha256(CORE_RESULT_PATH),
            "inner_result_sha256": core_result["result_sha256"],
            "fresh_retry_compute": dict(fresh),
            "total_compute_including_failed_base_charge": total,
            "summary": core_result["summary"],
            "pilot_execution_authorized": False,
            "paid_model_cost_usd": 0,
            "claim_boundary": lock["claim_boundary"],
        },
        "amendment_result_sha256",
    )


def _write_or_validate_amendment_result(core_result: Mapping[str, Any]) -> dict[str, Any]:
    expected = _amendment_result(core_result)
    if CORE_AMENDMENT_RESULT_PATH.exists():
        observed = _load_json(CORE_AMENDMENT_RESULT_PATH)
        _verify_hash(observed, "amendment_result_sha256")
        if observed != expected:
            raise RuntimeError("amendment result differs from its inner result")
        return observed
    _write_new_json(CORE_AMENDMENT_RESULT_PATH, expected)
    return expected


def run_core_preflight() -> dict[str, Any]:
    return configured_core().run_preflight()


def run_core() -> dict[str, Any]:
    result = configured_core().run_development()
    return _write_or_validate_amendment_result(result)


def replay_core() -> dict[str, Any]:
    result = configured_core().run_replay()
    return _write_or_validate_amendment_result(result)


def report_core() -> str:
    configured_core().run_report()
    result = replay_core()
    return json.dumps(result, indent=2, ensure_ascii=False) + "\n"


def configured_cross() -> ModuleType:
    global _CONFIGURED_CROSS
    if _CONFIGURED_CROSS is not None:
        return _CONFIGURED_CROSS
    core = configured_core()
    cross = _load_module(BASE_CROSS_PATH, "closed_loop_dms_cross_for_state0_amendment")
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
            "state0_amendment_runner": SCRIPT_PATH,
            "state0_amendment_protocol": DOC_PATH,
            "state0_amendment_tests": TEST_PATH,
            "state0_amendment_core_lock": CORE_LOCK_PATH,
            "historical_base_cross_lock": BASE_CROSS_LOCK_PATH,
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
        value["status"] = "prospective_cross_encoding_after_state0_metadata_amendment"
        value["state0_metadata_amendment"] = {
            "core_lock_path": _relative(CORE_LOCK_PATH),
            "core_lock_file_sha256": file_sha256(CORE_LOCK_PATH),
            "base_failure_sha256": _validate_base_failure()[0]["failure_sha256"],
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
        description="Narrow prospective CL-DMS state-zero metadata amendment"
    )
    parser.add_argument(
        "command",
        choices=(
            "record-base-failure",
            "lock",
            "preflight",
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
        "record-base-failure": record_base_failure,
        "lock": run_core_lock,
        "preflight": run_core_preflight,
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
