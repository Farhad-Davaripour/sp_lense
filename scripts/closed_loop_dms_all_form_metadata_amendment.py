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
DOC_PATH = ROOT / "docs" / "CLOSED_LOOP_DMS_ALL_FORM_METADATA_AMENDMENT.md"
TEST_PATH = ROOT / "tests" / "test_closed_loop_dms_all_form_metadata_amendment.py"
V2_RUNNER_PATH = ROOT / "scripts" / "closed_loop_dms_state0_context_amendment.py"
BASE_CROSS_PATH = ROOT / "scripts" / "closed_loop_dms_cross_encoding.py"
V2_LOCK_PATH = ROOT / "configs" / "closed_loop_dms_state0_context_amendment_lock.json"
V2_CROSS_LOCK_PATH = (
    ROOT / "configs" / "closed_loop_dms_cross_encoding_state0_context_amendment_lock.json"
)
V2_ARTIFACT_ROOT = ROOT / "artifacts" / "closed_loop_dms_state0_context_amendment" / "qwen35_08b"
V2_PREFLIGHT_PATH = V2_ARTIFACT_ROOT / "preflight.json"
V2_LEDGER_PATH = V2_ARTIFACT_ROOT / "compute_ledger.json"
V2_SCENARIO_ROOT = V2_ARTIFACT_ROOT / "scenarios"
V2_FINAL_PATH = V2_ARTIFACT_ROOT / "final_evaluation.pt"
V2_RESULT_ROOT = ROOT / "results" / "closed_loop_dms_state0_context_amendment" / "qwen35_08b"
V2_RESULT_PATH = V2_RESULT_ROOT / "development_result.json"
V2_AMENDMENT_RESULT_PATH = V2_RESULT_ROOT / "amendment_result.json"

V1_UNRELATED_PATH = (
    ROOT
    / "artifacts"
    / "closed_loop_dms_state0_metadata_amendment"
    / "qwen35_08b"
    / "calibration_unrelated_state0.pt"
)

CORE_LOCK_PATH = ROOT / "configs" / "closed_loop_dms_all_form_metadata_amendment_lock.json"
CROSS_LOCK_PATH = (
    ROOT / "configs" / "closed_loop_dms_cross_encoding_all_form_metadata_amendment_lock.json"
)
CORE_ARTIFACT_ROOT = (
    ROOT / "artifacts" / "closed_loop_dms_all_form_metadata_amendment" / "qwen35_08b"
)
V2_FAILURE_PATH = CORE_ARTIFACT_ROOT / "v2_all_form_failure.json"
CORE_PREFLIGHT_PATH = CORE_ARTIFACT_ROOT / "preflight.json"
CORE_WIRING_PREFLIGHT_PATH = CORE_ARTIFACT_ROOT / "metadata_wiring_preflight.json"
CORE_LEDGER_PATH = CORE_ARTIFACT_ROOT / "compute_ledger.json"
CORE_SCENARIO_ROOT = CORE_ARTIFACT_ROOT / "scenarios"
CORE_FINAL_PATH = CORE_ARTIFACT_ROOT / "final_evaluation.pt"
CORE_RESULT_ROOT = ROOT / "results" / "closed_loop_dms_all_form_metadata_amendment" / "qwen35_08b"
CORE_RESULT_PATH = CORE_RESULT_ROOT / "development_result.json"
CORE_REPORT_PATH = CORE_RESULT_ROOT / "DEVELOPMENT_REPORT.md"
CORE_AMENDMENT_RESULT_PATH = CORE_RESULT_ROOT / "amendment_result.json"

CROSS_ARTIFACT_ROOT = (
    ROOT / "artifacts" / "closed_loop_dms_cross_encoding_all_form_metadata_amendment" / "qwen35_08b"
)
CROSS_PREFLIGHT_PATH = CROSS_ARTIFACT_ROOT / "preflight.json"
CROSS_LEDGER_PATH = CROSS_ARTIFACT_ROOT / "compute_ledger.json"
CROSS_SCENARIO_ROOT = CROSS_ARTIFACT_ROOT / "scenarios"
CROSS_RESULT_PATH = (
    ROOT
    / "results"
    / "closed_loop_dms_cross_encoding_all_form_metadata_amendment"
    / "qwen35_08b"
    / "result.json"
)
CROSS_REPORT_PATH = CROSS_RESULT_PATH.with_name("REPORT.md")

CORE_LOCK_SCHEMA = "sp_lense.closed_loop_dms_all_form_metadata_amendment_lock.v1"
V2_FAILURE_SCHEMA = "sp_lense.closed_loop_dms_v2_all_form_failure.v1"
WIRING_PREFLIGHT_SCHEMA = "sp_lense.closed_loop_dms_all_form_metadata_wiring_preflight.v1"
AMENDMENT_RESULT_SCHEMA = "sp_lense.closed_loop_dms_all_form_metadata_amendment_result.v1"
CROSS_LOCK_SCHEMA = "sp_lense.closed_loop_dms_cross_encoding_all_form_metadata_amendment_lock.v1"

V2_RUNNER_FILE_SHA256 = "34749d9ebf6f2e515e9df148677bfdcd5ee2ad089e39bbe5471f671fab7d0c6f"
BASE_CROSS_FILE_SHA256 = "9d797924129fd5be22dc15fce8a5718646649f7165702b5841ddc570e80bce77"
V2_LOCK_FILE_SHA256 = "7b39959ba21c8830f77925b95d9258bab60d4a214ea32cd1c734441851f3641f"
V2_LOCK_IDENTITY_SHA256 = "dcb1cf6cbfe10d9dc8bef2420699171098589d0e090e294364883272411d1b0c"
V2_CROSS_LOCK_FILE_SHA256 = "bdb0279c4b7910ba7e6d9a4e919b81463f2aeadc1e1f1fbc5851108b13fb0f0b"
V2_CROSS_LOCK_IDENTITY_SHA256 = "54b7b853addab94d462498e101b53d04d6fc830619dd17514cc538a4e14c0ece"
V2_PREFLIGHT_FILE_SHA256 = "54a3fc7de6803c6c43f67e0e11b57d3e5322a56d464bcd2a8885b36276d887d6"
V2_PREFLIGHT_SHA256 = "7504d5ae66715b9861d8dca9883c4020758a9b85b4c768bccd8df9a7467e06cf"
V2_LEDGER_FILE_SHA256 = "6caf2d56ee3fc7d9fb5bfa62f4355735e0eda31c68be45c8121dfd9fd4a5804d"
V2_LEDGER_SHA256 = "ecf11c94b6201524672de4c236b5ad11dfb77088e0867ce6a33c35f1408ed638"
V1_UNRELATED_FILE_SHA256 = "c95538d96774ef327c8b23676bb6b9e3d567c9c7c7edbe30c749e4ce58b2d979"
V1_UNRELATED_CHECKPOINT_SHA256 = "0983f3b0548dd793b22f439b11198607a871bfbc70ea6c8358781dc2d8a604a4"
TOTAL_PRIOR_CHARGED_FB = 16
TOTAL_PRIOR_OBSERVED_ACTUAL_FB = 9
BASELINE_FORM_COUNT = 72
UNRELATED_FORM_COUNT = 8
REUSE_WORK_ID = "state0:calibration_unrelated:8_unique_forms"

_V2: ModuleType | None = None
_CONFIGURED_CORE: ModuleType | None = None
_CONFIGURED_CROSS: ModuleType | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _v2() -> ModuleType:
    global _V2
    if _V2 is None:
        if file_sha256(V2_RUNNER_PATH) != V2_RUNNER_FILE_SHA256:
            raise RuntimeError("frozen v2 state0 context amendment runner differs")
        _V2 = _load_module(V2_RUNNER_PATH, "closed_loop_dms_frozen_state0_context_v2")
    return _V2


def _v2_lock() -> dict[str, Any]:
    if file_sha256(V2_LOCK_PATH) != V2_LOCK_FILE_SHA256:
        raise RuntimeError("v2 state0 context amendment lock file differs")
    value = _load_json(V2_LOCK_PATH)
    _verify_hash(value, "lock_identity_sha256")
    if value.get("lock_identity_sha256") != V2_LOCK_IDENTITY_SHA256:
        raise RuntimeError("v2 state0 context amendment lock identity differs")
    return value


def _validate_v2_zero_compute_boundary() -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        file_sha256(V2_RUNNER_PATH) != V2_RUNNER_FILE_SHA256
        or file_sha256(V2_CROSS_LOCK_PATH) != V2_CROSS_LOCK_FILE_SHA256
        or file_sha256(V2_PREFLIGHT_PATH) != V2_PREFLIGHT_FILE_SHA256
        or file_sha256(V2_LEDGER_PATH) != V2_LEDGER_FILE_SHA256
        or file_sha256(V1_UNRELATED_PATH) != V1_UNRELATED_FILE_SHA256
    ):
        raise RuntimeError("v2 boundary file differs")
    cross_lock = _load_json(V2_CROSS_LOCK_PATH)
    _verify_hash(cross_lock, "lock_identity_sha256")
    if cross_lock.get("lock_identity_sha256") != V2_CROSS_LOCK_IDENTITY_SHA256:
        raise RuntimeError("v2 cross lock identity differs")
    preflight = _load_json(V2_PREFLIGHT_PATH)
    _verify_hash(preflight, "preflight_sha256")
    if (
        preflight.get("preflight_sha256") != V2_PREFLIGHT_SHA256
        or preflight.get("model_forwards") != 0
        or preflight.get("model_backwards") != 0
    ):
        raise RuntimeError("v2 preflight boundary differs")
    ledger = _load_json(V2_LEDGER_PATH)
    _verify_hash(ledger, "ledger_sha256")
    if (
        ledger.get("ledger_sha256") != V2_LEDGER_SHA256
        or sum(int(row["forward_evaluations"]) for row in ledger["events"]) != 0
        or sum(int(row["backward_evaluations"]) for row in ledger["events"]) != 0
        or any(row.get("status") != "complete" for row in ledger["events"])
        or len(ledger["events"]) != 1
    ):
        raise RuntimeError("v2 ledger is not the locked zero-compute reuse ledger")
    if (
        V2_RESULT_PATH.exists()
        or V2_AMENDMENT_RESULT_PATH.exists()
        or V2_FINAL_PATH.exists()
        or any(V2_SCENARIO_ROOT.glob("*/state-*.pt"))
        or any(V2_SCENARIO_ROOT.glob("*/terminal.json"))
    ):
        raise RuntimeError("v2 produced an outcome despite the recorded pre-steering failure")
    return preflight, ledger


def _v2_failure_value() -> dict[str, Any]:
    preflight, ledger = _validate_v2_zero_compute_boundary()
    return _with_hash(
        {
            "schema_version": V2_FAILURE_SCHEMA,
            "status": "v2_stopped_in_state0_assembly_before_model_load_or_steering",
            "v2_lock_identity_sha256": V2_LOCK_IDENTITY_SHA256,
            "v2_preflight_sha256": preflight["preflight_sha256"],
            "v2_ledger_sha256": ledger["ledger_sha256"],
            "reused_state0_checkpoint_sha256": V1_UNRELATED_CHECKPOINT_SHA256,
            "error_type": "KeyError",
            "error_message": "'prompt_sha256'",
            "failure_site": "_state0_checkpoint on the first scenario form",
            "root_cause": (
                "v2 enriched spec_by_form only; runtime scenario contexts take their first "
                "64 forms directly from plan[:64], whose form dictionaries remained unenriched"
            ),
            "required_fix": (
                "copy and enrich all 72 baseline rows in plan, then rebuild spec_by_form from "
                "those same enriched rows and verify both hashes against immutable baselines"
            ),
            "v2_new_forward_evaluations": 0,
            "v2_new_backward_evaluations": 0,
            "v2_state0_checkpoint_files_created": 0,
            "v2_steering_trial_forwards": 0,
            "v2_model_loaded": False,
            "self_preservation_intervention_outcomes_evaluated": False,
            "cross_encoding_outcomes_evaluated": False,
            "sealed_final_opened": False,
            "retry_authorized_by_this_record": False,
        },
        "failure_sha256",
    )


def record_v2_failure() -> dict[str, Any]:
    expected = _v2_failure_value()
    if V2_FAILURE_PATH.exists():
        observed = _load_json(V2_FAILURE_PATH)
        _verify_hash(observed, "failure_sha256")
        if observed != expected:
            raise RuntimeError("existing v2 all-form failure record differs")
        return observed
    _write_new_json(V2_FAILURE_PATH, expected)
    return expected


def _source_records() -> dict[str, dict[str, str]]:
    paths = {
        "all_form_amendment_runner": SCRIPT_PATH,
        "all_form_amendment_protocol": DOC_PATH,
        "all_form_amendment_tests": TEST_PATH,
        "frozen_v2_context_amendment_runner": V2_RUNNER_PATH,
        "frozen_base_cross_encoding_runner": BASE_CROSS_PATH,
    }
    return {
        name: {"path": _relative(path), "sha256": file_sha256(path)} for name, path in paths.items()
    }


def proposed_core_lock() -> dict[str, Any]:
    v2_lock = _v2_lock()
    failure = record_v2_failure()
    value = {
        "schema_version": CORE_LOCK_SCHEMA,
        "status": "prospective_all_form_metadata_lock_before_first_steering_forward",
        "development_only": True,
        "v2_lock": {
            "path": _relative(V2_LOCK_PATH),
            "file_sha256": file_sha256(V2_LOCK_PATH),
            "lock_identity_sha256": v2_lock["lock_identity_sha256"],
        },
        "v2_cross_lock": {
            "path": _relative(V2_CROSS_LOCK_PATH),
            "file_sha256": file_sha256(V2_CROSS_LOCK_PATH),
            "lock_identity_sha256": V2_CROSS_LOCK_IDENTITY_SHA256,
        },
        "v2_failed_attempt": {
            "path": _relative(V2_FAILURE_PATH),
            "file_sha256": file_sha256(V2_FAILURE_PATH),
            "failure_sha256": failure["failure_sha256"],
            "new_model_compute": 0,
            "steering_outcomes": 0,
        },
        "reused_state0_checkpoint": {
            "path": _relative(V1_UNRELATED_PATH),
            "file_sha256": file_sha256(V1_UNRELATED_PATH),
            "checkpoint_sha256": V1_UNRELATED_CHECKPOINT_SHA256,
            "new_model_compute": 0,
        },
        "model": dict(v2_lock["model"]),
        "runtime": dict(v2_lock["runtime"]),
        "chat_template_sha256": v2_lock["chat_template_sha256"],
        "dataset": dict(v2_lock["dataset"]),
        "design": {
            **dict(v2_lock["design"]),
            "all_baseline_form_metadata_amendment": {
                "scope": "all_64_scenario_plus_8_unrelated_baseline_forms",
                "operation": (
                    "copy plan, derive prompt_sha256 and anchor_prefix_sha256, verify both "
                    "against immutable baseline_by_form and capture evidence, and rebuild spec_by_form"
                ),
                "runtime_views_repaired": ["plan[:72]", "spec_by_form"],
                "changes_prompts_labels_anchors_tensors_gradients_solver_or_thresholds": False,
                "locked_finite_plan_identity_must_remain_unchanged": True,
                "model_free_context_solver_and_intervention_setup_preflight_required": True,
                "steering_outcomes_available_when_authored": False,
            },
        },
        "fresh_controller_compute_ceiling": {
            **dict(v2_lock["fresh_controller_compute_ceiling"]),
            "state0_new_forward_backward": 0,
            "maximum_new_controller_forward_backward": 9600,
        },
        "total_study_compute_ceiling": dict(v2_lock["total_study_compute_ceiling"]),
        "artifact_namespace": _relative(CORE_ARTIFACT_ROOT),
        "result_namespace": _relative(CORE_RESULT_ROOT),
        "sources": _source_records(),
        "claim_boundary": v2_lock["claim_boundary"],
    }
    return _with_hash(value, "lock_identity_sha256")


def run_core_lock() -> dict[str, Any]:
    if CORE_RESULT_PATH.exists() or any(CORE_SCENARIO_ROOT.glob("*/state-*.pt")):
        raise RuntimeError("all-form metadata lock must predate every amended steering outcome")
    value = proposed_core_lock()
    if CORE_LOCK_PATH.exists():
        observed = _load_json(CORE_LOCK_PATH)
        _verify_hash(observed, "lock_identity_sha256")
        if observed != value:
            raise RuntimeError("existing all-form metadata amendment lock differs")
        return observed
    _write_new_json(CORE_LOCK_PATH, value)
    return value


def _load_core_lock() -> dict[str, Any]:
    observed = _load_json(CORE_LOCK_PATH)
    _verify_hash(observed, "lock_identity_sha256")
    if observed != proposed_core_lock():
        raise RuntimeError("all-form metadata amendment lock differs from its bound design")
    return observed


def _without_derived_hashes(specification: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(specification)
    form = dict(result["form"])
    form.pop("prompt_sha256", None)
    form.pop("anchor_prefix_sha256", None)
    result["form"] = form
    return result


def enrich_all_baseline_form_hashes(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Copy and enrich all 72 baseline plan forms, then rebuild their lookup map."""

    plan = list(inputs["plan"])
    baseline_by_form = inputs["baseline_by_form"]
    prior_spec_by_form = inputs["spec_by_form"]
    finite_runner = inputs["finite_runner"]
    if (
        len(plan) < BASELINE_FORM_COUNT
        or len(baseline_by_form) != BASELINE_FORM_COUNT
        or len(prior_spec_by_form) != BASELINE_FORM_COUNT
    ):
        raise RuntimeError("all-form metadata amendment baseline coverage differs")
    prior_plan_sha256 = finite_runner.plan_sha256(plan)
    for index, row in enumerate(plan[:BASELINE_FORM_COUNT]):
        specification = dict(row)
        form = dict(specification["form"])
        form_id = str(form["form_id"])
        if form_id not in baseline_by_form or form_id not in prior_spec_by_form:
            raise RuntimeError("baseline form is missing from one immutable runtime view")
        if canonical_sha256(_without_derived_hashes(specification)) != canonical_sha256(
            _without_derived_hashes(prior_spec_by_form[form_id])
        ):
            raise RuntimeError("plan and specification lookup differ beyond derived audit fields")
        baseline_form = baseline_by_form[form_id]["form"]
        derived = {
            "prompt_sha256": text_sha256(str(form["prompt"])),
            "anchor_prefix_sha256": text_sha256(str(form["anchor_prefix"])),
        }
        for field, value in derived.items():
            if baseline_form.get(field) != value:
                raise RuntimeError(f"derived {field} does not match the immutable baseline")
            if field in form and form[field] != value:
                raise RuntimeError(f"existing {field} differs from its derived value")
            form[field] = value
        if form.get("family") != "unrelated":
            capture = inputs["capture_by_form"].get(form_id)
            if capture is None or any(
                capture.get(field) != value for field, value in derived.items()
            ):
                raise RuntimeError("derived scenario hashes do not match capture evidence")
        specification["form"] = form
        if canonical_sha256(_without_derived_hashes(specification)) != canonical_sha256(
            _without_derived_hashes(row)
        ):
            raise RuntimeError("all-form amendment changed a non-audit field")
        plan[index] = specification
    spec_by_form = {str(row["form"]["form_id"]): row for row in plan[:BASELINE_FORM_COUNT]}
    if set(spec_by_form) != set(prior_spec_by_form) or len(spec_by_form) != BASELINE_FORM_COUNT:
        raise RuntimeError("rebuilt all-form specification lookup differs")
    if (
        sum(row["form"].get("family") == "unrelated" for row in plan[:BASELINE_FORM_COUNT])
        != UNRELATED_FORM_COUNT
        or finite_runner.plan_sha256(plan) != prior_plan_sha256
    ):
        raise RuntimeError("enriched plan coverage or locked plan identity differs")
    for row in plan[:BASELINE_FORM_COUNT]:
        form = row["form"]
        if form["prompt_sha256"] != text_sha256(str(form["prompt"])) or form[
            "anchor_prefix_sha256"
        ] != text_sha256(str(form["anchor_prefix"])):
            raise RuntimeError("all-form metadata postcondition differs")
    return {**dict(inputs), "plan": plan, "spec_by_form": spec_by_form}


def configured_core() -> ModuleType:
    global _CONFIGURED_CORE
    if _CONFIGURED_CORE is not None:
        return _CONFIGURED_CORE
    v2 = _v2()
    core = v2.configured_core()
    lock = _load_core_lock()
    original_load_inputs = core._load_locked_inputs

    def amended_load_inputs(torch: Any) -> dict[str, Any]:
        return enrich_all_baseline_form_hashes(original_load_inputs(torch))

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
    if (
        core.run_development.__globals__ is not core.__dict__
        or core.run_preflight.__globals__["_load_locked_inputs"] is not amended_load_inputs
        or core.run_development.__globals__["_load_locked_inputs"] is not amended_load_inputs
    ):
        raise RuntimeError("all-form loader is not wired into the frozen core functions")
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
        raise RuntimeError("all-form state0 reuse ledger charged new model compute")
    return snapshot


def _state0_solver_readiness(
    core: ModuleType,
    torch: Any,
    *,
    inputs: Mapping[str, Any],
    scenario_id: str,
    contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    observations = []
    gradients = []
    for index, context in enumerate(contexts):
        baseline = context["baseline"]
        form = context["form"]
        observations.append(
            core._observation_record(
                context=context,
                branch_sign=0,
                gradient_index=index,
                margin=context["state0_margin"],
                predicted_token_id=baseline["predicted_token_id"],
                predicted_label=(
                    form["positive_label"]
                    if baseline["semantic_choice"] == form["positive_semantic"]
                    else form["negative_label"]
                ),
                semantic_choice=baseline["semantic_choice"],
                pair_choice_label=(
                    form["positive_label"]
                    if baseline["pair_semantic_choice"] == form["positive_semantic"]
                    else form["negative_label"]
                ),
                pair_semantic_choice=baseline["pair_semantic_choice"],
                answer_format_valid=baseline["answer_format_valid"],
                audit={
                    "capture_source": context["capture_source"],
                    "prompt_sha256": form["prompt_sha256"],
                    "anchor_prefix_sha256": form["anchor_prefix_sha256"],
                    "anchor_index": context["anchor_index"],
                    "anchor_evidence_sha256": context["anchor_evidence_sha256"],
                    "choice_boundary_evidence_sha256": context["choice_boundary_evidence_sha256"],
                    "prompt_token_ids_sha256": context["prompt_token_ids_sha256"],
                    "pre_anchor_residual_float32_sha256": context[
                        "pre_anchor_residual_float32_sha256"
                    ],
                    "raw_gradient_float32_sha256": core.tensor_float32_sha256(
                        context["raw_gradient"]
                    ),
                },
            )
        )
        gradients.append(context["raw_gradient"])
    state_metadata = {
        "state_index": 0,
        "cumulative_path_l2": 0.0,
        "observations": observations,
    }
    state_tensors = {
        "direction": torch.zeros(core.DIMENSION, dtype=torch.float64),
        "raw_gradients": torch.stack(gradients).float().contiguous(),
    }
    core._branch_maps(state_metadata, state_tensors)
    candidate, progress, attempts = core._select_update(
        state_metadata=state_metadata,
        state_tensors=state_tensors,
        contexts=contexts,
        residual_scale=float(inputs["residual_scales"][scenario_id]),
        standardized_nuisance_rows=inputs["standardized_nuisance_rows"][scenario_id],
    )
    if candidate is None or progress is None:
        raise RuntimeError("model-free state0 solver did not produce a certified candidate")
    revalidation = core.revalidate_symmetric_sequential_trust_region_update(candidate)
    direction = torch.from_numpy(candidate.realized_direction.copy()).double().contiguous()
    scale = float(inputs["residual_scales"][scenario_id])
    plus = torch.from_numpy(candidate.positive_physical_float32.copy()).float().contiguous()
    minus = torch.from_numpy(candidate.negative_physical_float32.copy()).float().contiguous()
    if (
        revalidation.get("passes") is not True
        or tuple(direction.shape) != (core.DIMENSION,)
        or not bool(torch.isfinite(direction).all())
        or not torch.equal(plus, (direction * scale).float().contiguous())
        or not torch.equal(minus, -plus)
        or core.tensor_float32_sha256(plus) != candidate.positive_physical_float32_sha256
        or core.tensor_float32_sha256(minus) != candidate.negative_physical_float32_sha256
    ):
        raise RuntimeError("model-free candidate intervention setup differs")
    return {
        "selected_progress_fraction": float(progress),
        "solver_attempt_count": len(attempts),
        "solver_revalidation_sha256": revalidation["revalidation_sha256"],
        "positive_physical_delta_float32_sha256": core.tensor_float32_sha256(plus),
        "negative_physical_delta_float32_sha256": core.tensor_float32_sha256(minus),
    }


def _metadata_wiring_preflight_value() -> dict[str, Any]:
    core = configured_core()
    import torch

    inputs = core._load_locked_inputs(torch)

    class ReadOnlyLedger:
        def require_artifact(self, *, work_id: str, path: Path) -> None:
            if work_id != REUSE_WORK_ID or Path(path).resolve() != V1_UNRELATED_PATH.resolve():
                raise RuntimeError("runtime wiring requested an unexpected state0 artifact")

    unrelated_metadata, unrelated_tensors = core._load_unrelated_capture(
        torch, ledger=ReadOnlyLedger()
    )
    scenario_records = []
    context_form_ids = []
    for scenario_id in inputs["scenario_ids"]:
        contexts = core._runtime_form_contexts(
            torch,
            inputs=inputs,
            scenario_id=scenario_id,
            unrelated_metadata=unrelated_metadata,
            unrelated_tensors=unrelated_tensors,
        )
        for context in contexts:
            form = context["form"]
            if form["prompt_sha256"] != text_sha256(str(form["prompt"])) or form[
                "anchor_prefix_sha256"
            ] != text_sha256(str(form["anchor_prefix"])):
                raise RuntimeError("runtime context audit hashes differ")
            context_form_ids.append(str(context["form_id"]))
        if (
            len(contexts) != 24
            or sum(row["category"] == "target" for row in contexts) != 4
            or sum(row["category"] == "unrelated" for row in contexts) != 8
            or sum(row["category"] not in {"target", "unrelated"} for row in contexts) != 12
            or tuple(torch.stack([row["raw_gradient"] for row in contexts]).shape) != (24, 1024)
        ):
            raise RuntimeError("runtime context coverage differs")
        scenario_records.append(
            {
                "scenario_id": scenario_id,
                "context_count": len(contexts),
                "target_count": 4,
                "protected_count": 12,
                "unrelated_count": 8,
                "candidate_readiness": _state0_solver_readiness(
                    core,
                    torch,
                    inputs=inputs,
                    scenario_id=scenario_id,
                    contexts=contexts,
                ),
            }
        )
    if len(context_form_ids) != 96:
        raise RuntimeError("runtime context total differs")
    return _with_hash(
        {
            "schema_version": WIRING_PREFLIGHT_SCHEMA,
            "status": "passed_without_model_load_or_forward",
            "lock_identity_sha256": _load_core_lock()["lock_identity_sha256"],
            "enriched_baseline_forms": BASELINE_FORM_COUNT,
            "scenario_records": scenario_records,
            "runtime_context_form_ids_sha256": canonical_sha256(context_form_ids),
            "frozen_run_function_uses_amended_loader": (
                core.run_development.__globals__["_load_locked_inputs"] is core._load_locked_inputs
            ),
            "state0_observation_solver_and_intervention_setup_succeed": True,
            "model_loads": 0,
            "model_forwards": 0,
            "model_backwards": 0,
            "generated_tokens": 0,
            "self_preservation_intervention_outcomes_read": False,
            "sealed_final_opened": False,
        },
        "preflight_sha256",
    )


def run_metadata_wiring_preflight() -> dict[str, Any]:
    expected = _metadata_wiring_preflight_value()
    if CORE_WIRING_PREFLIGHT_PATH.exists():
        observed = _load_json(CORE_WIRING_PREFLIGHT_PATH)
        _verify_hash(observed, "preflight_sha256")
        if observed != expected:
            raise RuntimeError("existing all-form metadata wiring preflight differs")
        return observed
    _write_new_json(CORE_WIRING_PREFLIGHT_PATH, expected)
    return expected


def run_core_preflight() -> dict[str, Any]:
    base = configured_core().run_preflight()
    prepare_reuse_ledger()
    wiring = run_metadata_wiring_preflight()
    return {"core_preflight": base, "metadata_wiring_preflight": wiring}


def _amendment_result(core_result: Mapping[str, Any]) -> dict[str, Any]:
    lock = _load_core_lock()
    fresh = core_result["compute"]
    total = {
        **dict(fresh),
        "forward_evaluations": int(fresh["forward_evaluations"]) + TOTAL_PRIOR_CHARGED_FB,
        "backward_evaluations": int(fresh["backward_evaluations"]) + TOTAL_PRIOR_CHARGED_FB,
        "forward_backward": int(fresh["forward_backward"]) + TOTAL_PRIOR_CHARGED_FB,
        "prior_charged_forward_backward": TOTAL_PRIOR_CHARGED_FB,
        "prior_observed_actual_forward_backward": TOTAL_PRIOR_OBSERVED_ACTUAL_FB,
    }
    wiring = run_metadata_wiring_preflight()
    return _with_hash(
        {
            "schema_version": AMENDMENT_RESULT_SCHEMA,
            "status": core_result["status"],
            "development_only": True,
            "lock_file_sha256": file_sha256(CORE_LOCK_PATH),
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "v2_failure_sha256": record_v2_failure()["failure_sha256"],
            "metadata_wiring_preflight_sha256": wiring["preflight_sha256"],
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
            raise RuntimeError("all-form metadata amendment result differs")
        return observed
    _write_new_json(CORE_AMENDMENT_RESULT_PATH, expected)
    return expected


def run_core() -> dict[str, Any]:
    run_core_preflight()
    return _write_or_validate_result(configured_core().run_development())


def replay_core() -> dict[str, Any]:
    run_core_preflight()
    return _write_or_validate_result(configured_core().run_replay())


def report_core() -> str:
    configured_core().run_report()
    return json.dumps(replay_core(), indent=2, ensure_ascii=False) + "\n"


def configured_cross() -> ModuleType:
    global _CONFIGURED_CROSS
    if _CONFIGURED_CROSS is not None:
        return _CONFIGURED_CROSS
    core = configured_core()
    if file_sha256(BASE_CROSS_PATH) != BASE_CROSS_FILE_SHA256:
        raise RuntimeError("frozen base cross-encoding runner differs")
    cross = _load_module(BASE_CROSS_PATH, "closed_loop_dms_cross_for_all_form_amendment")
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
            "all_form_metadata_amendment_runner": SCRIPT_PATH,
            "all_form_metadata_amendment_protocol": DOC_PATH,
            "all_form_metadata_amendment_tests": TEST_PATH,
            "all_form_metadata_amendment_core_lock": CORE_LOCK_PATH,
            "historical_v2_cross_lock": V2_CROSS_LOCK_PATH,
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
        value["status"] = "prospective_cross_encoding_after_all_form_metadata_amendment"
        value["all_form_metadata_amendment"] = {
            "core_lock_path": _relative(CORE_LOCK_PATH),
            "core_lock_file_sha256": file_sha256(CORE_LOCK_PATH),
            "v2_failure_sha256": record_v2_failure()["failure_sha256"],
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
        description="Prospective CL-DMS all-baseline-form audit-metadata amendment"
    )
    parser.add_argument(
        "command",
        choices=(
            "record-v2-failure",
            "lock",
            "preflight",
            "prepare-reuse",
            "wiring-preflight",
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
        "record-v2-failure": record_v2_failure,
        "lock": run_core_lock,
        "preflight": run_core_preflight,
        "prepare-reuse": prepare_reuse_ledger,
        "wiring-preflight": run_metadata_wiring_preflight,
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
