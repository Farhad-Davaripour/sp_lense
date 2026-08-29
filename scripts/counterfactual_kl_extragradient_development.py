#!/usr/bin/env python
"""Prospective local CKES qualification for Qwen3.5-0.8B.

This runner is intentionally separate from the frozen CL-DMS v3 runner.  It
reuses audited primitives but writes only to a new result tree.  The sealed
dataset is never deserialized on the validation path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sp_lense.causal_anchor_runtime import resolve_shared_anchor_evidence
from sp_lense.closed_loop_dms_runtime import capture_closed_loop_dms_step
from sp_lense.comparison_runtime import resolve_choice_boundary
from sp_lense.counterfactual_kl_extragradient_surgery import (
    CELL_TANGENT_KL_LIMIT,
    LOOKAHEAD_EPSILON,
    MEAN_TANGENT_KL_LIMIT,
    construct_common_ascent_lookahead,
    revalidate_counterfactual_kl_extragradient_update,
    solve_counterfactual_kl_extragradient_update,
)
from sp_lense.counterfactual_kl_prompts import render_ckes_forms
from sp_lense.factorial_causal_anchor import (
    canonical_sha256,
    tensor_float32_sha256,
)
from sp_lense.symmetric_sequential_trust_region_dms import (
    SymmetricSequentialDMSCertificateError,
    SymmetricSequentialDMSInfeasibleError,
    SymmetricSequentialDMSSolverError,
    revalidate_symmetric_sequential_trust_region_update,
    solve_symmetric_sequential_trust_region_update,
)

LOCK_PATH = ROOT / "configs" / "counterfactual_kl_extragradient_development_lock.json"
VALIDATION_DATA_PATH = ROOT / "data" / "ckes_validation.json"
SEALED_DATA_PATH = ROOT / "data" / "ckes_sealed.json"
PROTOCOL_PATH = ROOT / "docs" / "COUNTERFACTUAL_KL_EXTRAGRADIENT_DEVELOPMENT_PROTOCOL.md"
BASE_RUNNER_PATH = ROOT / "scripts" / "closed_loop_dms_development.py"
ORIGINAL_RUNNER_PATH = ROOT / "scripts" / "factorial_causal_anchor_gradient_pilot.py"
MODEL_CONFIG_PATH = ROOT / "configs" / "qwen35_08b_aligned.json"
RESULT_BASE = ROOT / "results" / "counterfactual_kl_extragradient" / "qwen35_08b"

LEDGER_SCHEMA = "sp_lense.counterfactual_kl_extragradient_ledger.v1"
BASELINE_SCHEMA = "sp_lense.counterfactual_kl_extragradient_baseline.v1"
STATE_SCHEMA = "sp_lense.counterfactual_kl_extragradient_state.v1"
LOOKAHEAD_SCHEMA = "sp_lense.counterfactual_kl_extragradient_lookahead.v1"
FINAL_SCHEMA = "sp_lense.counterfactual_kl_extragradient_final.v1"
RESULT_SCHEMA = "sp_lense.counterfactual_kl_extragradient_result.v1"

MODEL = {
    "id": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "device": "cpu",
    "dtype": "float32",
    "n_layers": 24,
    "d_model": 1024,
}
CHAT_TEMPLATE_SHA256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
EXPECTED_RUNTIME = {
    "python": "3.12.10",
    "torch": "2.13.0+cpu",
    "transformers": "5.15.1",
    "transformer_lens": "4.0.0b1",
    "huggingface_hub": "1.28.0",
    "safetensors": "0.8.0",
    "numpy": "2.5.2",
    "scipy": "1.18.1",
    "torch_intraop_threads": 12,
    "torch_interop_threads": 12,
}
SELECTED_LAYER = 0
PROGRESS_SCHEDULE = (0.25, 0.125, 0.0625)
TRUST_RADIUS = 0.25
OPTIMIZATION_TARGET_MARGIN = 0.15
FINAL_TARGET_MARGIN = 0.05
PROTECTED_MAXIMUM_FLOOR = 0.025
PROTECTED_BASELINE_FRACTION = 0.5
MAX_TRIAL_STATES = 24
MAX_CUMULATIVE_PATH_L2 = 2.0
MAX_FINAL_DIRECTION_L2 = 2.0
MINIMUM_REALIZED_TARGET_PROGRESS_FRACTION = 0.25
UNRELATED_LINEARIZATION_ERROR_CAP = 0.05
HOOK_REALIZATION_RELATIVE_L2_TOLERANCE = 1e-4
KL_LIMITS = {"mean": 0.005, "p95": 0.02, "max": 0.05}
SCENARIO_COUNT = 4
FORMS_PER_SCENARIO = 24
TARGET_COUNT = 4
PROTECTED_COUNT = 12
UNRELATED_COUNT = 8
NUISANCE_COUNT = 8
STATE0_FB = 80
MAX_LOOKAHEAD_FB = SCENARIO_COUNT * MAX_TRIAL_STATES * 8
MAX_CANDIDATE_FB = SCENARIO_COUNT * MAX_TRIAL_STATES * FORMS_PER_SCENARIO * 2
MAX_FB = STATE0_FB + MAX_LOOKAHEAD_FB + MAX_CANDIDATE_FB
MAX_FINAL_FORWARD = SCENARIO_COUNT * FORMS_PER_SCENARIO * 2
COMPUTE_CEILING = {
    "forward": MAX_FB + MAX_FINAL_FORWARD,
    "backward": MAX_FB,
    "forward_backward": MAX_FB,
    "final_forward_only": MAX_FINAL_FORWARD,
    "generated_tokens": 0,
    "external_api_calls": 0,
    "external_model_judges": 0,
    "paid_model_cost_usd": 0,
}

LOCKED_SOURCE_PATHS = (
    Path("src/sp_lense/counterfactual_kl_extragradient_surgery.py"),
    Path("src/sp_lense/counterfactual_kl_prompts.py"),
    Path("src/sp_lense/counterfactual_kl_runtime.py"),
    Path("src/sp_lense/counterfactual_kl_protocol.py"),
    Path("src/sp_lense/backend.py"),
    Path("src/sp_lense/config.py"),
    Path("src/sp_lense/core.py"),
    Path("src/sp_lense/causal_anchor_runtime.py"),
    Path("src/sp_lense/closed_loop_dms_runtime.py"),
    Path("src/sp_lense/comparison_runtime.py"),
    Path("src/sp_lense/comparison_intervention.py"),
    Path("src/sp_lense/steering_methods.py"),
    Path("src/sp_lense/semantic_completion_gradient.py"),
    Path("src/sp_lense/factorial_causal_anchor.py"),
    Path("src/sp_lense/counterfactual_protected_natural_gradient.py"),
    Path("src/sp_lense/gradient_specificity_trust_region.py"),
    Path("src/sp_lense/gradient_specificity_v3.py"),
    Path("src/sp_lense/decision_margin_shield.py"),
    Path("src/sp_lense/decision_margin_shield_rowspace.py"),
    Path("src/sp_lense/counterfactual_tangent_shield.py"),
    Path("src/sp_lense/decision_margin_shield_finite.py"),
    Path("src/sp_lense/symmetric_sequential_trust_region_dms.py"),
    Path("scripts/counterfactual_kl_extragradient_development.py"),
    Path("scripts/closed_loop_dms_development.py"),
    Path("scripts/factorial_causal_anchor_gradient_pilot.py"),
    Path("scripts/decision_margin_shield_finite_capture_manifest_amendment.py"),
    Path("scripts/decision_margin_shield_finite_calibration.py"),
    Path("scripts/decision_margin_shield_layer_screen.py"),
    Path("tests/test_counterfactual_kl_extragradient_surgery.py"),
    Path("tests/test_counterfactual_kl_prompts.py"),
    Path("tests/test_counterfactual_kl_runtime.py"),
    Path("tests/test_counterfactual_kl_protocol.py"),
    Path("tests/test_counterfactual_kl_extragradient_runner.py"),
    Path("docs/COUNTERFACTUAL_KL_EXTRAGRADIENT_DEVELOPMENT_PROTOCOL.md"),
    Path("configs/qwen35_08b_aligned.json"),
    Path("pyproject.toml"),
    Path("requirements-research.txt"),
    Path("requirements-constrained-steering.txt"),
    Path("requirements-counterfactual-tangent-shield.txt"),
)

_BASE_CACHE: ModuleType | None = None


class CandidateRuntimeFailure(RuntimeError):
    """A reserved candidate batch failed after compute may have begun."""


TERMINAL_OUTCOME_CLASSES = {
    "success",
    "scientific_no_success",
    "technical_integrity_failure",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _bound_path(raw: str) -> Path:
    result = (ROOT / raw).resolve()
    result.relative_to(ROOT.resolve())
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{_relative(path)} must contain one JSON object")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _with_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = {str(key): _plain(item) for key, item in value.items() if key != field}
    result[field] = canonical_sha256(result)
    return result


def _verify_hash(value: Mapping[str, Any], field: str) -> None:
    unhashed = {str(key): _plain(item) for key, item in value.items() if key != field}
    if value.get(field) != canonical_sha256(unhashed):
        raise RuntimeError(f"{field} differs")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable {_relative(path)}")
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _base() -> ModuleType:
    global _BASE_CACHE
    if _BASE_CACHE is None:
        specification = importlib.util.spec_from_file_location("sp_lense_ckes_base", BASE_RUNNER_PATH)
        if specification is None or specification.loader is None:
            raise RuntimeError("cannot load the frozen CL-DMS runner")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        _BASE_CACHE = module
    return _BASE_CACHE


def _result_root(split: str) -> Path:
    if split not in {"validation", "sealed"}:
        raise ValueError("split must be validation or sealed")
    return RESULT_BASE / split


def _paths(split: str) -> dict[str, Path]:
    root = _result_root(split)
    return {
        "root": root,
        "preflight": root / "preflight.json",
        "tokenizer_preflight": root / "tokenizer_preflight.json",
        "ledger": root / "compute_ledger.json",
        "baseline": root / "state0_baseline.pt",
        "states": root / "states",
        "lookaheads": root / "lookaheads",
        "final": root / "final.pt",
        "result": root / "result.json",
        "report": root / "REPORT.md",
    }


def _rendered_manifest(payload: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    rendered = render_ckes_forms(payload, expected_split=split)
    rows = []
    for family in ("scenario", "calibration_unrelated", "nuisance_fit"):
        for form in rendered[family]:
            rows.append(
                {
                    "family": family,
                    "form_id": form["form_id"],
                    "prompt_sha256": form["prompt_sha256"],
                    "anchor_prefix_sha256": form["anchor_prefix_sha256"],
                    "positive_label": form["positive_label"],
                    "negative_label": form["negative_label"],
                    "scenario_id": form.get("scenario_id"),
                    "assignment": form.get("assignment"),
                    "target": form.get("target"),
                    "event": form.get("event"),
                    "preserve_first": form.get("preserve_first"),
                    "control_id": form.get("control_id"),
                    "preferred_first": form.get("preferred_first"),
                }
            )
    return {
        "split": split,
        "counts": {key: len(value) for key, value in rendered.items()},
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
    }


def proposed_lock() -> dict[str, Any]:
    from sp_lense.counterfactual_kl_protocol import build_prospective_lock

    missing = [str(path) for path in LOCKED_SOURCE_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"prospective CKES sources are incomplete: {missing}")
    validation = _load_json(VALIDATION_DATA_PATH)
    sealed = _load_json(SEALED_DATA_PATH)
    file_hashes = {
        "validation_dataset": {
            "path": _relative(VALIDATION_DATA_PATH),
            "sha256": file_sha256(VALIDATION_DATA_PATH),
            "bytes": VALIDATION_DATA_PATH.stat().st_size,
        },
        "sealed_dataset": {
            "path": _relative(SEALED_DATA_PATH),
            "sha256": file_sha256(SEALED_DATA_PATH),
            "bytes": SEALED_DATA_PATH.stat().st_size,
        },
        **{
            "source_" + str(index).zfill(2): {
                "path": str(path).replace("\\", "/"),
                "sha256": file_sha256(ROOT / path),
                "bytes": (ROOT / path).stat().st_size,
            }
            for index, path in enumerate(LOCKED_SOURCE_PATHS)
        },
    }
    configuration = {
        "research_question": (
            "Can matched-counterfactual KL extragradient correction make the local "
            "bidirectional gradient controller reproducibly selective?"
        ),
        "model": MODEL,
        "runtime": EXPECTED_RUNTIME,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "model_config_path": _relative(MODEL_CONFIG_PATH),
        "selected_hook": f"blocks.{SELECTED_LAYER}.hook_out",
        "selected_position": "last_token_of_verified_shared_causal_prefix",
        "controller": {
            "method": "counterfactual_kl_extragradient_surgery",
            "lookahead_epsilon": LOOKAHEAD_EPSILON,
            "progress_schedule": list(PROGRESS_SCHEDULE),
            "trust_radius": TRUST_RADIUS,
            "optimization_target_margin": OPTIMIZATION_TARGET_MARGIN,
            "final_target_margin": FINAL_TARGET_MARGIN,
            "protected_maximum_floor": PROTECTED_MAXIMUM_FLOOR,
            "protected_baseline_fraction": PROTECTED_BASELINE_FRACTION,
            "maximum_trial_states_per_scenario": MAX_TRIAL_STATES,
            "maximum_cumulative_path_l2": MAX_CUMULATIVE_PATH_L2,
            "maximum_final_direction_l2": MAX_FINAL_DIRECTION_L2,
            "minimum_realized_target_progress_fraction": (
                MINIMUM_REALIZED_TARGET_PROGRESS_FRACTION
            ),
            "unrelated_linearization_error_cap": UNRELATED_LINEARIZATION_ERROR_CAP,
            "cell_tangent_kl_limit": CELL_TANGENT_KL_LIMIT,
            "mean_tangent_kl_limit": MEAN_TANGENT_KL_LIMIT,
            "actual_kl_limits_by_stratum": KL_LIMITS,
            "authoritative_dtype": "float32",
            "negative_rule": "unary_negation_of_exact_positive_float32_bytes",
            "no_postoutcome_fallbacks": True,
        },
        "statistics": {
            "independent_unit": "scenario_cluster",
            "cluster_count": 4,
            "minimum_one_sided_exact_sign_test_p_if_all_positive": 0.0625,
            "row_level_pseudoreplication_forbidden": True,
        },
        "claim_boundary": {
            "natural_self_preservation_mechanism": False,
            "universal_direction": False,
            "black_box_attack": False,
            "unchanged_general_capability": False,
            "publication_ready_novelty_from_this_run_alone": False,
        },
        "sealed_plaintext_status": "locked_not_blinded",
    }
    thresholds = {
        "success_gate": {
            "minimum_successful_scenarios": 3,
            "minimum_assignment_units": 6,
            "minimum_target_assignment_order_decision_changes": 12,
            "both_orders_and_assignments_per_successful_scenario": True,
            "zero_non_target_choice_changes": True,
            "actual_kl_gate_every_accepted_state": True,
        },
        "compute_ceiling_per_split": COMPUTE_CEILING,
    }
    return build_prospective_lock(
        file_hashes=file_hashes,
        rendered_manifests={
            "validation": _rendered_manifest(validation, split="validation"),
            "sealed": _rendered_manifest(sealed, split="sealed"),
        },
        configuration=configuration,
        thresholds=thresholds,
        sealed_dataset_file_key="sealed_dataset",
        validation_result_schema_version=RESULT_SCHEMA,
        required_validation_gates=(
            "actual_kl",
            "baseline_qualification",
            "compute_integrity",
            "decision_changes",
            "efficacy",
            "execution_integrity",
            "final_repeat",
            "non_target_choice_stability",
        ),
    )


def run_lock() -> dict[str, Any]:
    if LOCK_PATH.exists():
        raise FileExistsError("refusing to reopen or overwrite the prospective CKES lock")
    value = proposed_lock()
    _write_new_json(LOCK_PATH, value)
    return value


def _load_lock() -> dict[str, Any]:
    from sp_lense.counterfactual_kl_protocol import verify_prospective_lock

    value = _load_json(LOCK_PATH)
    value = verify_prospective_lock(value)
    for record in value["file_hashes"].values():
        path = _bound_path(record["path"])
        if file_sha256(path) != record["sha256"] or (
            "bytes" in record and path.stat().st_size != record["bytes"]
        ):
            raise RuntimeError(f"locked CKES file differs: {record['path']}")
    if value["thresholds"].get("compute_ceiling_per_split") != COMPUTE_CEILING:
        raise RuntimeError("CKES compute ceiling differs")
    return value


class ComputeLedger:
    """One hash-chained fail-closed ledger for local model operations."""

    def __init__(self, *, split: str, lock_identity_sha256: str) -> None:
        self.split = split
        self.path = _paths(split)["ledger"]
        self.lock_identity_sha256 = lock_identity_sha256
        if self.path.exists():
            self.payload = _load_json(self.path)
            _verify_hash(self.payload, "ledger_sha256")
        else:
            self.payload = {
                "schema_version": LEDGER_SCHEMA,
                "split": split,
                "lock_identity_sha256": lock_identity_sha256,
                "ceiling": COMPUTE_CEILING,
                "events": [],
            }
            self._persist()
        self._validate()

    def _persist(self) -> None:
        self.payload = _with_hash(self.payload, "ledger_sha256")
        _atomic_text(self.path, json.dumps(self.payload, indent=2, ensure_ascii=False) + "\n")

    def _validate(self) -> None:
        if (
            self.payload.get("schema_version") != LEDGER_SCHEMA
            or self.payload.get("split") != self.split
            or self.payload.get("lock_identity_sha256") != self.lock_identity_sha256
            or self.payload.get("ceiling") != COMPUTE_CEILING
        ):
            raise RuntimeError("CKES compute ledger identity differs")
        prior = None
        seen: set[str] = set()
        events = self.payload.get("events")
        if not isinstance(events, list):
            raise TypeError("CKES ledger events must be a list")
        for index, event in enumerate(events):
            if event.get("event_index") != index or event.get("prior_event_sha256") != prior:
                raise RuntimeError("CKES ledger hash chain differs")
            work_id = event.get("work_id")
            if not isinstance(work_id, str) or work_id in seen:
                raise RuntimeError("CKES ledger work IDs are invalid or repeated")
            seen.add(work_id)
            unhashed = dict(event)
            observed = unhashed.pop("event_sha256", None)
            if observed != canonical_sha256(unhashed):
                raise RuntimeError("CKES ledger event hash differs")
            prior = observed
            if event.get("status") not in {"pending", "complete"}:
                raise RuntimeError("CKES ledger event status differs")
            forward = event.get("forward_evaluations")
            backward = event.get("backward_evaluations")
            if type(forward) is not int or type(backward) is not int or min(forward, backward) < 0:
                raise RuntimeError("CKES ledger compute count differs")
            if backward > forward:
                raise RuntimeError("CKES ledger has more backwards than forwards")
            if event["status"] == "pending":
                if index != len(events) - 1 or event.get("artifact_path") is not None:
                    raise RuntimeError("CKES ledger has an ambiguous nonterminal pending event")
            else:
                raw = event.get("artifact_path")
                if not isinstance(raw, str):
                    raise RuntimeError("completed CKES ledger event lacks an artifact")
                artifact = _bound_path(raw)
                if not artifact.is_file() or file_sha256(artifact) != event.get("artifact_sha256"):
                    raise RuntimeError("completed CKES ledger artifact differs")
        snapshot = self.snapshot()
        if (
            snapshot["forward_backward"] > MAX_FB
            or snapshot["final_forward_only"] > MAX_FINAL_FORWARD
            or snapshot["forward_evaluations"] > COMPUTE_CEILING["forward"]
            or snapshot["backward_evaluations"] > COMPUTE_CEILING["backward"]
        ):
            raise RuntimeError("CKES ledger exceeds the locked compute ceiling")

    def event(self, work_id: str) -> Mapping[str, Any] | None:
        return next(
            (event for event in self.payload["events"] if event["work_id"] == work_id),
            None,
        )

    def require_unambiguous(self) -> None:
        events = self.payload["events"]
        if events and events[-1]["status"] == "pending":
            raise RuntimeError(
                "ambiguous reserved CKES work exists; refusing to guess whether compute ran"
            )

    def reserve(self, *, work_id: str, forward: int, backward: int, kind: str) -> None:
        self.require_unambiguous()
        if self.event(work_id) is not None:
            raise RuntimeError("CKES work ID is already reserved")
        if min(forward, backward) < 0 or backward > forward:
            raise ValueError("invalid CKES compute reservation")
        prior = self.payload["events"][-1]["event_sha256"] if self.payload["events"] else None
        event = {
            "event_index": len(self.payload["events"]),
            "work_id": work_id,
            "kind": kind,
            "forward_evaluations": int(forward),
            "backward_evaluations": int(backward),
            "status": "pending",
            "artifact_path": None,
            "artifact_sha256": None,
            "prior_event_sha256": prior,
        }
        event["event_sha256"] = canonical_sha256(event)
        self.payload["events"].append(event)
        self._validate()
        self._persist()

    def complete(self, *, work_id: str, artifact_path: Path) -> None:
        events = self.payload["events"]
        if not events or events[-1]["work_id"] != work_id or events[-1]["status"] != "pending":
            raise RuntimeError("CKES ledger lacks a matching terminal reservation")
        event = dict(events[-1])
        event.update(
            {
                "status": "complete",
                "artifact_path": _relative(artifact_path),
                "artifact_sha256": file_sha256(artifact_path),
            }
        )
        event.pop("event_sha256")
        event["event_sha256"] = canonical_sha256(event)
        events[-1] = event
        self._validate()
        self._persist()

    def require_artifact(self, *, work_id: str, path: Path) -> None:
        event = self.event(work_id)
        if (
            event is None
            or event.get("status") != "complete"
            or event.get("artifact_path") != _relative(path)
            or event.get("artifact_sha256") != file_sha256(path)
        ):
            raise RuntimeError("CKES artifact is not bound to a completed ledger event")

    def snapshot(self) -> dict[str, Any]:
        events = self.payload.get("events", [])
        forward = sum(int(event["forward_evaluations"]) for event in events)
        backward = sum(int(event["backward_evaluations"]) for event in events)
        return {
            "forward_evaluations": forward,
            "backward_evaluations": backward,
            "forward_backward": backward,
            "final_forward_only": forward - backward,
            "event_count": len(events),
            "complete_event_count": sum(event["status"] == "complete" for event in events),
            "ledger_file_sha256": file_sha256(self.path) if self.path.exists() else None,
            "ledger_sha256": self.payload.get("ledger_sha256"),
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judges": 0,
            "paid_model_cost_usd": 0,
        }


def _save_checkpoint(
    torch: Any,
    *,
    path: Path,
    metadata: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> None:
    _base()._save_tensor_checkpoint(
        torch,
        path=path,
        metadata=metadata,
        tensors=tensors,
    )


def _load_checkpoint(torch: Any, *, path: Path, schema: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return _base()._load_tensor_checkpoint(torch, path=path, schema=schema)


def _load_dataset(split: str) -> dict[str, Any]:
    lock = _load_lock()
    if split == "validation":
        payload = _load_json(VALIDATION_DATA_PATH)
    elif split == "sealed":
        from sp_lense.counterfactual_kl_protocol import load_sealed_dataset

        payload = load_sealed_dataset(
            validation_result_path=_paths("validation")["result"],
            sealed_path=SEALED_DATA_PATH,
            lock=lock,
        )
    else:
        raise ValueError("split must be validation or sealed")
    manifest = _rendered_manifest(payload, split=split)
    if manifest != lock["rendered_manifests"][split]:
        raise RuntimeError(f"rendered CKES {split} manifest differs from the lock")
    return payload


def _assert_frozen_base_gate_contract() -> None:
    """Fail before model load if a reused frozen CL-DMS gate has drifted."""

    base = _base()
    expected = {
        "SELECTED_LAYER": SELECTED_LAYER,
        "DIMENSION": MODEL["d_model"],
        "SCENARIO_COUNT": SCENARIO_COUNT,
        "FORMS_PER_SCENARIO": FORMS_PER_SCENARIO,
        "TARGET_COUNT": TARGET_COUNT,
        "PROTECTED_COUNT": PROTECTED_COUNT,
        "UNRELATED_COUNT": UNRELATED_COUNT,
        "NUISANCE_COUNT": NUISANCE_COUNT,
        "PROGRESS_SCHEDULE": PROGRESS_SCHEDULE,
        "TRUST_RADIUS": TRUST_RADIUS,
        "OPTIMIZATION_TARGET_MARGIN": OPTIMIZATION_TARGET_MARGIN,
        "FINAL_TARGET_MARGIN": FINAL_TARGET_MARGIN,
        "PROTECTED_MAXIMUM_FLOOR": PROTECTED_MAXIMUM_FLOOR,
        "PROTECTED_BASELINE_FRACTION": PROTECTED_BASELINE_FRACTION,
        "MAX_CUMULATIVE_PATH_L2": MAX_CUMULATIVE_PATH_L2,
        "MAX_FINAL_DIRECTION_L2": MAX_FINAL_DIRECTION_L2,
        "MINIMUM_REALIZED_TARGET_PROGRESS_FRACTION": (
            MINIMUM_REALIZED_TARGET_PROGRESS_FRACTION
        ),
        "UNRELATED_LINEARIZATION_ERROR_CAP": UNRELATED_LINEARIZATION_ERROR_CAP,
        "HOOK_REALIZATION_RELATIVE_L2_TOLERANCE": (
            HOOK_REALIZATION_RELATIVE_L2_TOLERANCE
        ),
        "KL_LIMITS": KL_LIMITS,
    }
    differing = {
        name: {"ckes": value, "frozen_base": getattr(base, name, None)}
        for name, value in expected.items()
        if getattr(base, name, None) != value
    }
    if differing:
        raise RuntimeError(f"frozen CL-DMS gate contract differs: {differing}")


def run_preflight(split: str = "validation") -> dict[str, Any]:
    lock = _load_lock()
    _assert_frozen_base_gate_contract()
    paths = _paths(split)
    payload = _load_dataset(split)
    rendered = render_ckes_forms(payload, expected_split=split)
    value = _with_hash(
        {
            "schema_version": "sp_lense.counterfactual_kl_extragradient_preflight.v1",
            "status": f"ready_for_{split}",
            "split": split,
            "lock_file_sha256": file_sha256(LOCK_PATH),
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "dataset_file_sha256": lock["file_hashes"][f"{split}_dataset"]["sha256"],
            "rendered_manifest_sha256": lock["rendered_manifests"][split]["rows_sha256"],
            "counts": {key: len(value) for key, value in rendered.items()},
            "selected_layer": SELECTED_LAYER,
            "compute_ceiling": COMPUTE_CEILING,
            "model_loads": 0,
            "model_forwards": 0,
            "model_backwards": 0,
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judges": 0,
            "paid_model_cost_usd": 0,
        },
        "preflight_sha256",
    )
    if paths["preflight"].exists():
        observed = _load_json(paths["preflight"])
        _verify_hash(observed, "preflight_sha256")
        if observed != value:
            raise RuntimeError("CKES preflight differs from its locked reconstruction")
    else:
        _write_new_json(paths["preflight"], value)
    return value


def _flatten_forms(payload: Mapping[str, Any], *, split: str) -> list[dict[str, Any]]:
    rendered = render_ckes_forms(payload, expected_split=split)
    result = [
        *rendered["scenario"],
        *rendered["calibration_unrelated"],
        *rendered["nuisance_fit"],
    ]
    if len(result) != STATE0_FB or len({row["form_id"] for row in result}) != STATE0_FB:
        raise RuntimeError("CKES state-zero form coverage differs")
    return result


def _resolve_form_evidence(
    backend: Any,
    *,
    payload: Mapping[str, Any],
    forms: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for form in forms:
        groups.setdefault(str(form["anchor_prefix_sha256"]), []).append(form)
    if len(groups) != 40 or any(len(group) != 2 for group in groups.values()):
        raise RuntimeError("CKES answer-order shared-prefix grouping differs")
    result: dict[str, dict[str, Any]] = {}
    for group in groups.values():
        prefix = str(group[0]["anchor_prefix"])
        prompts = [str(row["prompt"]) for row in group]
        evidence = resolve_shared_anchor_evidence(
            backend,
            anchor_prefix=prefix,
            # The two answer-order prompts share choice-suffix text through the
            # first option label.  Including the literal prefix as a third,
            # non-model sentinel forces the longest shared token prefix to stop
            # at the declared causal anchor rather than inside that suffix.  Keep
            # the two evaluated prompts first so their token hashes retain the
            # form ordering used below.
            prompts=[*prompts, prefix],
            anchor_marker=str(payload["anchor_marker"]),
        )
        if len(evidence.prompt_token_sha256s) != len(group) + 1:
            raise RuntimeError("CKES anchor sentinel evidence coverage differs")
        for index, form in enumerate(group):
            boundary = resolve_choice_boundary(backend, str(form["prompt"]))
            if boundary.prompt_prefix_token_ids_sha256 != evidence.prompt_token_sha256s[index]:
                raise RuntimeError("CKES choice-boundary and shared-prefix token hashes disagree")
            result[str(form["form_id"])] = {
                "anchor_index": evidence.anchor_index,
                "anchor_evidence_sha256": evidence.audit["audit_sha256"],
                "shared_token_prefix_sha256": evidence.shared_token_prefix_sha256,
                "choice_boundary_evidence_sha256": boundary.evidence_sha256,
                "prompt_token_ids_sha256": boundary.prompt_prefix_token_ids_sha256,
                "positive_token_id": boundary.token_id(str(form["positive_label"])),
                "negative_token_id": boundary.token_id(str(form["negative_label"])),
            }
    if len(result) != len(forms):
        raise RuntimeError("CKES tokenizer evidence coverage differs")
    return result


def _resolve_or_load_tokenizer_preflight(
    *,
    split: str,
    backend: Any,
    payload: Mapping[str, Any],
    forms: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Persist all zero-forward token/anchor evidence before compute reservation."""

    path = _paths(split)["tokenizer_preflight"]
    lock = _load_lock()
    form_ids = [str(form["form_id"]) for form in forms]
    by_form = {str(form["form_id"]): form for form in forms}
    if len(by_form) != len(forms):
        raise RuntimeError("CKES tokenizer preflight received duplicate form IDs")
    if path.exists():
        metadata = _load_json(path)
        _verify_hash(metadata, "tokenizer_preflight_sha256")
    else:
        evidence = _resolve_form_evidence(backend, payload=payload, forms=forms)
        records = []
        for form_id in form_ids:
            form = by_form[form_id]
            row = {
                "form_id": form_id,
                "prompt_sha256": form["prompt_sha256"],
                "anchor_prefix_sha256": form["anchor_prefix_sha256"],
                **_plain(evidence[form_id]),
            }
            row["row_sha256"] = canonical_sha256(row)
            records.append(row)
        backend_metadata = backend.metadata()
        observed_model = {
            "id": backend_metadata["model_id"],
            "revision": backend_metadata["model_revision"],
            "device": backend_metadata["device"],
            "dtype": backend_metadata["dtype"],
            "n_layers": backend_metadata["model_layers"],
            "d_model": backend_metadata["d_model"],
        }
        if observed_model != MODEL:
            raise RuntimeError("tokenizer preflight backend differs from the locked model")
        metadata = _with_hash(
            {
                "schema_version": "sp_lense.counterfactual_kl_tokenizer_preflight.v1",
                "status": "complete_before_first_model_forward",
                "split": split,
                "lock_identity_sha256": lock["lock_identity_sha256"],
                "model": observed_model,
                "runtime": EXPECTED_RUNTIME,
                "chat_template_sha256": CHAT_TEMPLATE_SHA256,
                "record_count": len(records),
                "records": records,
                "records_sha256": canonical_sha256(records),
                "model_loads": 1,
                "model_forwards": 0,
                "model_backwards": 0,
                "generated_tokens": 0,
                "external_api_calls": 0,
                "external_model_judges": 0,
                "paid_model_cost_usd": 0,
            },
            "tokenizer_preflight_sha256",
        )
        _write_new_json(path, metadata)
    if (
        metadata.get("schema_version")
        != "sp_lense.counterfactual_kl_tokenizer_preflight.v1"
        or metadata.get("status") != "complete_before_first_model_forward"
        or metadata.get("split") != split
        or metadata.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or metadata.get("model") != MODEL
        or metadata.get("runtime") != EXPECTED_RUNTIME
        or metadata.get("chat_template_sha256") != CHAT_TEMPLATE_SHA256
        or metadata.get("record_count") != len(forms)
        or metadata.get("records_sha256") != canonical_sha256(metadata.get("records"))
        or metadata.get("model_forwards") != 0
        or metadata.get("model_backwards") != 0
    ):
        raise RuntimeError("CKES tokenizer preflight identity differs")
    records = metadata.get("records")
    if (
        not isinstance(records, list)
        or any(not isinstance(row, Mapping) for row in records)
        or [row.get("form_id") for row in records] != form_ids
    ):
        raise RuntimeError("CKES tokenizer preflight form order differs")
    evidence_by_form: dict[str, dict[str, Any]] = {}
    evidence_fields = {
        "anchor_index",
        "anchor_evidence_sha256",
        "shared_token_prefix_sha256",
        "choice_boundary_evidence_sha256",
        "prompt_token_ids_sha256",
        "positive_token_id",
        "negative_token_id",
    }
    for record in records:
        form_id = str(record["form_id"])
        form = by_form[form_id]
        unhashed = dict(record)
        observed_hash = unhashed.pop("row_sha256", None)
        if (
            observed_hash != canonical_sha256(unhashed)
            or record.get("prompt_sha256") != form["prompt_sha256"]
            or record.get("anchor_prefix_sha256") != form["anchor_prefix_sha256"]
        ):
            raise RuntimeError("CKES tokenizer preflight row identity differs")
        evidence_by_form[form_id] = {field: record[field] for field in evidence_fields}
    return evidence_by_form, metadata


def _category(form: Mapping[str, Any]) -> str:
    if form.get("family") == "unrelated":
        return "unrelated"
    cell = (form.get("target"), form.get("event"))
    if cell == ("self", "permanent"):
        return "target"
    if cell == ("other", "permanent"):
        return "other_permanent"
    if cell == ("self", "temporary"):
        return "self_temporary"
    if cell == ("other", "temporary"):
        return "other_temporary"
    raise RuntimeError("unknown CKES form category")


def _baseline_record(
    *,
    form: Mapping[str, Any],
    evidence: Mapping[str, Any],
    capture: Any,
    tensor_index: int,
) -> dict[str, Any]:
    value = {
        "baseline_id": str(form["form_id"]),
        "form_id": str(form["form_id"]),
        "form": _plain(form),
        "category": _category(form),
        "tensor_index": tensor_index,
        "anchor_index": int(evidence["anchor_index"]),
        "anchor_evidence_sha256": evidence["anchor_evidence_sha256"],
        "choice_boundary_evidence_sha256": evidence["choice_boundary_evidence_sha256"],
        "prompt_token_ids_sha256": evidence["prompt_token_ids_sha256"],
        "positive_token_id": int(capture.positive_token_id),
        "negative_token_id": int(capture.negative_token_id),
        "positive_minus_negative_log_odds": float(
            capture.positive_minus_negative_log_odds
        ),
        "predicted_token_id": int(capture.unrestricted_predicted_token_id),
        "predicted_label": str(capture.unrestricted_predicted_label),
        "semantic_choice": str(capture.unrestricted_semantic_choice),
        "pair_choice_label": str(capture.pair_choice_label),
        "pair_semantic_choice": str(capture.pair_semantic_choice),
        "answer_format_valid": bool(capture.answer_format_valid),
        "pre_anchor_residual_float32_sha256": tensor_float32_sha256(
            capture.pre_anchor_residual
        ),
        "raw_anchor_gradient_float32_sha256": tensor_float32_sha256(
            capture.raw_anchor_gradient
        ),
        "full_logits_float32_sha256": tensor_float32_sha256(capture.full_logits),
        "logits_float32_sha256": tensor_float32_sha256(capture.full_logits),
        "capture_audit": _plain(capture.audit),
    }
    value["row_sha256"] = canonical_sha256(value)
    return value


def _capture_or_load_baseline(
    torch: Any,
    *,
    split: str,
    backend_getter: Any,
    payload: Mapping[str, Any],
    forms: Sequence[Mapping[str, Any]],
    ledger: ComputeLedger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from sp_lense.counterfactual_kl_runtime import capture_counterfactual_kl_baseline

    path = _paths(split)["baseline"]
    work_id = f"{split}:state0:80_baseline_margin_gradient_captures"
    if path.exists():
        ledger.require_artifact(work_id=work_id, path=path)
        metadata, tensors = _load_checkpoint(torch, path=path, schema=BASELINE_SCHEMA)
        preflight_path = _paths(split)["tokenizer_preflight"]
        if not preflight_path.is_file():
            raise RuntimeError("CKES baseline exists without its tokenizer preflight")
        _, tokenizer_preflight = _resolve_or_load_tokenizer_preflight(
            split=split,
            backend=None,
            payload=payload,
            forms=forms,
        )
        if (
            metadata.get("tokenizer_preflight_sha256")
            != tokenizer_preflight["tokenizer_preflight_sha256"]
            or metadata.get("tokenizer_preflight_file_sha256")
            != file_sha256(preflight_path)
        ):
            raise RuntimeError("CKES baseline tokenizer-preflight binding differs")
        return metadata, tensors
    # Model loading, prompt tokenization, and anchor/boundary verification do
    # not execute a model forward.  Finish those checks before reserving the
    # 80-pass batch so a tokenizer/preflight error cannot create an ambiguous
    # compute reservation.
    backend = backend_getter()
    evidence_by_form, tokenizer_preflight = _resolve_or_load_tokenizer_preflight(
        split=split,
        backend=backend,
        payload=payload,
        forms=forms,
    )
    ledger.reserve(
        work_id=work_id,
        forward=STATE0_FB,
        backward=STATE0_FB,
        kind="state0_baseline_margin_gradient_and_full_logits",
    )
    records = []
    gradients = []
    residuals = []
    logits = []
    try:
        for index, form in enumerate(forms):
            form_id = str(form["form_id"])
            evidence = evidence_by_form[form_id]
            capture = capture_counterfactual_kl_baseline(
                backend,
                str(form["prompt"]),
                str(form["positive_label"]),
                str(form["negative_label"]),
                positive_semantic=str(form["positive_semantic"]),
                negative_semantic=str(form["negative_semantic"]),
                layer=SELECTED_LAYER,
                anchor_index=int(evidence["anchor_index"]),
                expected_prompt_sha256=str(form["prompt_sha256"]),
                expected_choice_boundary_evidence_sha256=str(
                    evidence["choice_boundary_evidence_sha256"]
                ),
                expected_prompt_token_ids_sha256=str(evidence["prompt_token_ids_sha256"]),
            )
            if (
                int(capture.positive_token_id) != int(evidence["positive_token_id"])
                or int(capture.negative_token_id) != int(evidence["negative_token_id"])
            ):
                raise RuntimeError("CKES baseline A/B token IDs differ from preflight evidence")
            records.append(
                _baseline_record(
                    form=form,
                    evidence=evidence,
                    capture=capture,
                    tensor_index=index,
                )
            )
            gradients.append(capture.raw_anchor_gradient)
            residuals.append(capture.pre_anchor_residual)
            logits.append(capture.full_logits)
            print(f"CKES {split} baseline {index + 1}/{STATE0_FB} {form_id}", flush=True)
        metadata = {
            "schema_version": BASELINE_SCHEMA,
            "status": "complete",
            "split": split,
            "lock_identity_sha256": _load_lock()["lock_identity_sha256"],
            "tokenizer_preflight_sha256": tokenizer_preflight[
                "tokenizer_preflight_sha256"
            ],
            "tokenizer_preflight_file_sha256": file_sha256(
                _paths(split)["tokenizer_preflight"]
            ),
            "record_count": len(records),
            "records": records,
            "tensor_layout_sha256": canonical_sha256(
                [
                    {
                        "form_id": row["form_id"],
                        "tensor_index": row["tensor_index"],
                        "gradient": row["raw_anchor_gradient_float32_sha256"],
                        "residual": row["pre_anchor_residual_float32_sha256"],
                        "logits": row["full_logits_float32_sha256"],
                    }
                    for row in records
                ]
            ),
            "compute": {
                "model_forwards": len(records),
                "model_backwards": len(records),
                "generated_tokens": 0,
                "external_model_judges": 0,
                "external_api_calls": 0,
                "paid_model_cost_usd": 0,
            },
        }
        _save_checkpoint(
            torch,
            path=path,
            metadata=metadata,
            tensors={
                "raw_gradients": torch.stack(gradients).float().contiguous(),
                "pre_anchor_residuals": torch.stack(residuals).float().contiguous(),
                "full_logits": torch.stack(logits).float().contiguous(),
            },
        )
        ledger.complete(work_id=work_id, artifact_path=path)
    except Exception as error:
        # A pending reservation deliberately blocks automatic continuation: the
        # process cannot know how many operations completed before the failure.
        raise RuntimeError(
            "CKES state-zero capture failed after its compute reservation"
        ) from error
    return _load_checkpoint(torch, path=path, schema=BASELINE_SCHEMA)


def _validate_answer_order_anchor_residuals(
    torch: Any,
    *,
    records: Sequence[Mapping[str, Any]],
    residuals: Any,
) -> None:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        form = record["form"]
        groups.setdefault(str(form["anchor_prefix_sha256"]), []).append(record)
    if len(groups) != 40 or any(len(group) != 2 for group in groups.values()):
        raise RuntimeError("CKES baseline answer-order grouping differs")
    for group in groups.values():
        left, right = group
        left_tensor = residuals[int(left["tensor_index"])]
        right_tensor = residuals[int(right["tensor_index"])]
        if not torch.equal(left_tensor, right_tensor):
            raise RuntimeError("answer-order suffix changed a shared-prefix anchor residual")


def _build_contexts(
    torch: Any,
    *,
    split: str,
    baseline_metadata: Mapping[str, Any],
    baseline_tensors: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float], dict[str, Any]]:
    records = baseline_metadata["records"]
    gradients = baseline_tensors["raw_gradients"]
    residuals = baseline_tensors["pre_anchor_residuals"]
    logits = baseline_tensors["full_logits"]
    if (
        len(records) != STATE0_FB
        or tuple(gradients.shape) != (STATE0_FB, MODEL["d_model"])
        or tuple(residuals.shape) != (STATE0_FB, MODEL["d_model"])
        or int(logits.shape[0]) != STATE0_FB
    ):
        raise RuntimeError("CKES baseline checkpoint tensor coverage differs")
    _validate_answer_order_anchor_residuals(torch, records=records, residuals=residuals)
    by_form = {str(row["form_id"]): row for row in records}
    scenario_ids = sorted(
        {
            str(row["form"]["scenario_id"])
            for row in records
            if row["form"].get("family") == "scenario"
        }
    )
    if len(scenario_ids) != SCENARIO_COUNT:
        raise RuntimeError("CKES scenario count differs")
    calibration_unrelated = [
        row
        for row in records
        if row["form"].get("family") == "unrelated"
        and row["form"].get("control_partition") == "calibration"
    ]
    nuisance = [
        row
        for row in records
        if row["form"].get("family") == "unrelated"
        and row["form"].get("control_partition") == "nuisance_fit"
    ]
    if len(calibration_unrelated) != UNRELATED_COUNT or len(nuisance) != NUISANCE_COUNT:
        raise RuntimeError("CKES unrelated baseline partition coverage differs")

    def context(record: Mapping[str, Any], scenario_id: str) -> dict[str, Any]:
        index = int(record["tensor_index"])
        form = record["form"]
        return {
            "direction_scenario_id": scenario_id,
            "form_id": str(record["form_id"]),
            "form": form,
            "baseline": record,
            "baseline_logits": logits[index].float().contiguous(),
            "category": _category(form),
            "raw_gradient": gradients[index].float().contiguous(),
            "pre_anchor_residual": residuals[index].float().contiguous(),
            "anchor_index": int(record["anchor_index"]),
            "choice_boundary_evidence_sha256": record[
                "choice_boundary_evidence_sha256"
            ],
            "prompt_token_ids_sha256": record["prompt_token_ids_sha256"],
            "pre_anchor_residual_float32_sha256": record[
                "pre_anchor_residual_float32_sha256"
            ],
        }

    contexts_by_scenario: dict[str, list[dict[str, Any]]] = {}
    scales: dict[str, float] = {}
    nuisance_rows: dict[str, Any] = {}
    for scenario_id in scenario_ids:
        scenario_records = [
            row
            for row in records
            if row["form"].get("family") == "scenario"
            and row["form"].get("scenario_id") == scenario_id
        ]
        if len(scenario_records) != 16:
            raise RuntimeError("one CKES scenario lacks 16 factorial forms")
        scenario_residuals = torch.stack(
            [residuals[int(row["tensor_index"])].double() for row in scenario_records]
        )
        norms = scenario_residuals.norm(dim=1)
        if not bool(torch.isfinite(norms).all().item()) or bool((norms <= 0).any().item()):
            raise RuntimeError("CKES residual scale inputs are invalid")
        scale = float(torch.exp(torch.log(norms).mean()).item())
        scales[scenario_id] = scale
        nuisance_rows[scenario_id] = scale * torch.stack(
            [gradients[int(row["tensor_index"])].double() for row in nuisance]
        )
        contexts = [
            *[context(row, scenario_id) for row in scenario_records],
            *[context(row, scenario_id) for row in calibration_unrelated],
        ]
        categories = [row["category"] for row in contexts]
        if (
            len(contexts) != FORMS_PER_SCENARIO
            or categories.count("target") != TARGET_COUNT
            or categories.count("unrelated") != UNRELATED_COUNT
            or len(categories) - categories.count("target") - categories.count("unrelated")
            != PROTECTED_COUNT
        ):
            raise RuntimeError("CKES per-scenario category coverage differs")
        contexts_by_scenario[scenario_id] = contexts
    if set(by_form) != {str(row["form_id"]) for row in records}:
        raise RuntimeError("CKES baseline form map differs")
    return contexts_by_scenario, scales, nuisance_rows


def _state_path(split: str, scenario_id: str, trial_index: int) -> Path:
    return _paths(split)["states"] / scenario_id / f"trial_{trial_index:03d}.pt"


def _lookahead_path(split: str, scenario_id: str, parent_trial_index: int) -> Path:
    return (
        _paths(split)["lookaheads"]
        / scenario_id
        / f"parent_{parent_trial_index:03d}.pt"
    )


def _observation_from_baseline(
    *, context: Mapping[str, Any], gradient_index: int
) -> dict[str, Any]:
    baseline = context["baseline"]
    return {
        "form_id": context["form_id"],
        "category": context["category"],
        "branch_sign": 0,
        "gradient_index": gradient_index,
        "positive_minus_negative_log_odds": baseline[
            "positive_minus_negative_log_odds"
        ],
        "unrestricted_predicted_token_id": baseline["predicted_token_id"],
        "unrestricted_predicted_label": baseline["predicted_label"],
        "unrestricted_semantic_choice": baseline["semantic_choice"],
        "pair_choice_label": baseline["pair_choice_label"],
        "pair_semantic_choice": baseline["pair_semantic_choice"],
        "answer_format_valid": baseline["answer_format_valid"],
        "audit": baseline["capture_audit"],
    }


def _state0_checkpoint(
    torch: Any,
    *,
    split: str,
    scenario_id: str,
    contexts: Sequence[Mapping[str, Any]],
    scale: float,
    baseline_checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _state_path(split, scenario_id, 0)
    if path.exists():
        return _load_checkpoint(torch, path=path, schema=STATE_SCHEMA)
    direction = torch.zeros(MODEL["d_model"], dtype=torch.float64)
    observations = [
        _observation_from_baseline(context=context, gradient_index=index)
        for index, context in enumerate(contexts)
    ]
    metadata = {
        "schema_version": STATE_SCHEMA,
        "status": "accepted_state0",
        "split": split,
        "lock_identity_sha256": _load_lock()["lock_identity_sha256"],
        "scenario_id": scenario_id,
        # The frozen CL-DMS branch-map helper distinguishes state zero from
        # signed candidate states with this field.  Keep it as an exact alias
        # of CKES's trial index so the reused gate cannot misclassify a state.
        "state_index": 0,
        "trial_index": 0,
        "parent_accepted_trial_index": None,
        "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
        "residual_scale": scale,
        "direction_sha256": canonical_sha256(direction.tolist()),
        "direction_l2": 0.0,
        "cumulative_path_l2": 0.0,
        "step_l2": 0.0,
        "accepted": True,
        "stopping_gate_passes": False,
        "selected_progress_fraction": None,
        "observations": observations,
        "observation_layout": "state0_24",
        "model_forwards": 0,
        "model_backwards": 0,
        "derived_without_additional_model_compute": True,
    }
    _save_checkpoint(
        torch,
        path=path,
        metadata=metadata,
        tensors={
            "direction": direction,
            "raw_gradients": torch.stack([row["raw_gradient"] for row in contexts])
            .float()
            .contiguous(),
        },
    )
    return _load_checkpoint(torch, path=path, schema=STATE_SCHEMA)


def _ordered_family(
    contexts: Sequence[Mapping[str, Any]], category: str
) -> list[Mapping[str, Any]]:
    if category == "protected":
        return [row for row in contexts if row["category"] not in {"target", "unrelated"}]
    return [row for row in contexts if row["category"] == category]


def _controller_problem(
    *,
    state_metadata: Mapping[str, Any],
    state_tensors: Mapping[str, Any],
    contexts: Sequence[Mapping[str, Any]],
    residual_scale: float,
    standardized_nuisance_rows: Any,
    progress: float,
) -> dict[str, Any]:
    plus, minus, plus_gradients, minus_gradients = _base()._branch_maps(
        state_metadata, state_tensors
    )
    form_order = [str(row["form_id"]) for row in contexts]
    plus_gradient_map = {
        form_id: plus_gradients[index].double().numpy() * residual_scale
        for index, form_id in enumerate(form_order)
    }
    minus_gradient_map = {
        form_id: minus_gradients[index].double().numpy() * residual_scale
        for index, form_id in enumerate(form_order)
    }
    target = _ordered_family(contexts, "target")
    protected = _ordered_family(contexts, "protected")
    unrelated = _ordered_family(contexts, "unrelated")

    def margins(
        rows: Sequence[Mapping[str, Any]], branch: Mapping[str, Mapping[str, Any]]
    ) -> np.ndarray:
        return np.asarray(
            [branch[str(row["form_id"])]["positive_minus_negative_log_odds"] for row in rows],
            dtype=np.float64,
        )

    def gradients(
        rows: Sequence[Mapping[str, Any]], values: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        return np.stack([values[str(row["form_id"])] for row in rows])

    protected_baseline = np.asarray(
        [row["baseline"]["positive_minus_negative_log_odds"] for row in protected],
        dtype=np.float64,
    )
    protected_signs = np.where(protected_baseline >= 0.0, 1.0, -1.0)
    protected_floors = np.minimum(
        PROTECTED_MAXIMUM_FLOOR,
        PROTECTED_BASELINE_FRACTION * np.abs(protected_baseline),
    )
    unrelated_baseline = np.asarray(
        [row["baseline"]["positive_minus_negative_log_odds"] for row in unrelated],
        dtype=np.float64,
    )
    return {
        "current_direction": state_tensors["direction"].double().numpy(),
        "target_plus_margins": margins(target, plus),
        "target_plus_gradients": gradients(target, plus_gradient_map),
        "target_minus_margins": margins(target, minus),
        "target_minus_gradients": gradients(target, minus_gradient_map),
        "protected_plus_margins": margins(protected, plus),
        "protected_plus_gradients": gradients(protected, plus_gradient_map),
        "protected_minus_margins": margins(protected, minus),
        "protected_minus_gradients": gradients(protected, minus_gradient_map),
        "protected_baseline_signs": protected_signs,
        "protected_margin": protected_floors,
        "unrelated_baseline_margins": unrelated_baseline,
        "unrelated_plus_margins": margins(unrelated, plus),
        "unrelated_plus_gradients": gradients(unrelated, plus_gradient_map),
        "unrelated_minus_margins": margins(unrelated, minus),
        "unrelated_minus_gradients": gradients(unrelated, minus_gradient_map),
        "baseline_unrelated_gradients": standardized_nuisance_rows.double().numpy(),
        "optimization_target_margin": OPTIMIZATION_TARGET_MARGIN,
        "progress_fraction": progress,
        "trust_radius": TRUST_RADIUS,
        "physical_residual_scale": residual_scale,
    }


def _capture_or_load_lookahead(
    torch: Any,
    *,
    split: str,
    backend_getter: Any,
    scenario_id: str,
    contexts: Sequence[Mapping[str, Any]],
    state_metadata: Mapping[str, Any],
    state_tensors: Mapping[str, Any],
    problem: Mapping[str, Any],
    ledger: ComputeLedger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from sp_lense.counterfactual_kl_runtime import capture_counterfactual_kl_lookahead

    parent_index = int(state_metadata["trial_index"])
    path = _lookahead_path(split, scenario_id, parent_index)
    work_id = f"{split}:{scenario_id}:parent={parent_index}:8_kl_lookaheads"
    if path.exists():
        ledger.require_artifact(work_id=work_id, path=path)
        metadata, tensors = _load_checkpoint(torch, path=path, schema=LOOKAHEAD_SCHEMA)
        if (
            metadata.get("parent_checkpoint_sha256") != state_metadata["checkpoint_sha256"]
            or metadata.get("parent_direction_sha256") != state_metadata["direction_sha256"]
        ):
            raise RuntimeError("CKES lookahead parent identity differs")
        return metadata, tensors
    target_oriented = np.vstack(
        (problem["target_plus_gradients"], problem["target_minus_gradients"])
    )
    common = construct_common_ascent_lookahead(
        problem["current_direction"],
        oriented_target_gradients=target_oriented,
        baseline_unrelated_gradients=problem["baseline_unrelated_gradients"],
    )
    lookahead = torch.from_numpy(common.lookahead_direction.copy()).double().contiguous()
    lookahead_hash = canonical_sha256(lookahead.tolist())
    scale = float(problem["physical_residual_scale"])
    plus_delta = (lookahead * scale).float().contiguous()
    minus_delta = -plus_delta
    if not torch.equal(minus_delta, -plus_delta):
        raise RuntimeError("CKES lookahead branches are not exact float32 negations")
    matched = [row for row in contexts if row["category"] == "other_permanent"]
    if len(matched) != 4:
        raise RuntimeError("CKES matched-other lookahead coverage differs")
    ledger.reserve(
        work_id=work_id,
        forward=8,
        backward=8,
        kind="nonzero_matched_other_full_vocabulary_kl_gradient_lookahead",
    )
    backend = backend_getter()
    records = []
    values = []
    shared_gradients = []
    raw_gradients = []
    for branch_sign, signed_delta in ((1, plus_delta), (-1, minus_delta)):
        signed_hash = tensor_float32_sha256(signed_delta)
        for context in matched:
            form = context["form"]
            baseline_logits = context["baseline_logits"]
            capture = capture_counterfactual_kl_lookahead(
                backend,
                str(form["prompt"]),
                layer=SELECTED_LAYER,
                anchor_index=int(context["anchor_index"]),
                branch_sign=branch_sign,
                lookahead_standardized_direction=lookahead,
                physical_residual_scale=scale,
                signed_delta=signed_delta,
                baseline_full_logits=baseline_logits,
                expected_prompt_sha256=str(form["prompt_sha256"]),
                expected_choice_boundary_evidence_sha256=str(
                    context["choice_boundary_evidence_sha256"]
                ),
                expected_prompt_token_ids_sha256=str(
                    context["prompt_token_ids_sha256"]
                ),
                expected_pre_anchor_residual_float32_sha256=str(
                    context["pre_anchor_residual_float32_sha256"]
                ),
                expected_lookahead_standardized_direction_sha256=lookahead_hash,
                expected_signed_delta_float32_sha256=signed_hash,
                expected_baseline_full_logits_float32_sha256=tensor_float32_sha256(
                    baseline_logits
                ),
                maximum_realized_relative_l2_error=(
                    HOOK_REALIZATION_RELATIVE_L2_TOLERANCE
                ),
            )
            index = len(values)
            values.append(float(capture.full_kl))
            shared_gradients.append(capture.shared_standardized_kl_gradient)
            raw_gradients.append(capture.raw_anchor_kl_gradient)
            record = {
                "form_id": context["form_id"],
                "category": context["category"],
                "branch_sign": branch_sign,
                "tensor_index": index,
                "full_vocabulary_kl_changed_to_baseline": float(capture.full_kl),
                "shared_gradient_sha256": canonical_sha256(
                    capture.shared_standardized_kl_gradient.tolist()
                ),
                "raw_gradient_float32_sha256": tensor_float32_sha256(
                    capture.raw_anchor_kl_gradient
                ),
                "audit": _plain(capture.audit),
            }
            record["row_sha256"] = canonical_sha256(record)
            records.append(record)
    metadata = {
        "schema_version": LOOKAHEAD_SCHEMA,
        "status": "complete",
        "split": split,
        "lock_identity_sha256": _load_lock()["lock_identity_sha256"],
        "scenario_id": scenario_id,
        "parent_trial_index": parent_index,
        "parent_checkpoint_sha256": state_metadata["checkpoint_sha256"],
        "parent_direction_sha256": state_metadata["direction_sha256"],
        "common_ascent": common.as_record(),
        "lookahead_direction_sha256": lookahead_hash,
        "positive_physical_float32_sha256": tensor_float32_sha256(plus_delta),
        "negative_physical_float32_sha256": tensor_float32_sha256(minus_delta),
        "exact_float32_negation": bool(torch.equal(minus_delta, -plus_delta)),
        "record_count": len(records),
        "records": records,
        "compute": {"model_forwards": 8, "model_backwards": 8, "generated_tokens": 0},
    }
    _save_checkpoint(
        torch,
        path=path,
        metadata=metadata,
        tensors={
            "common_direction": torch.from_numpy(common.direction.copy()).double().contiguous(),
            "lookahead_direction": lookahead,
            "kl_values": torch.tensor(values, dtype=torch.float64),
            "shared_kl_gradients": torch.stack(shared_gradients).double().contiguous(),
            "raw_kl_gradients": torch.stack(raw_gradients).float().contiguous(),
        },
    )
    ledger.complete(work_id=work_id, artifact_path=path)
    return _load_checkpoint(torch, path=path, schema=LOOKAHEAD_SCHEMA)


def _reconstruct_common_lookahead(
    metadata: Mapping[str, Any], tensors: Mapping[str, Any]
) -> Any:
    from sp_lense.counterfactual_kl_extragradient_surgery import CommonAscentLookahead

    record = metadata["common_ascent"]
    return CommonAscentLookahead(
        direction=tensors["common_direction"].double().numpy(),
        lookahead_direction=tensors["lookahead_direction"].double().numpy(),
        simplex_weights=np.asarray(record["simplex_weights"], dtype=np.float64),
        nuisance_basis=np.asarray(record["nuisance_basis"], dtype=np.float64),
        diagnostics=record["diagnostics"],
    )


def _solve_ckes_for_progress(
    *,
    problem: Mapping[str, Any],
    lookahead_metadata: Mapping[str, Any],
    lookahead_tensors: Mapping[str, Any],
    cumulative_path_l2: float,
) -> tuple[Any, dict[str, Any]]:
    nominal = solve_symmetric_sequential_trust_region_update(**problem)
    nominal_revalidation = revalidate_symmetric_sequential_trust_region_update(nominal)
    if nominal_revalidation.get("passes") is not True:
        raise SymmetricSequentialDMSCertificateError("nominal CKES parent step failed revalidation")
    common = _reconstruct_common_lookahead(lookahead_metadata, lookahead_tensors)
    candidate = solve_counterfactual_kl_extragradient_update(
        nominal,
        common_ascent_lookahead=common,
        lookahead_kl_values=lookahead_tensors["kl_values"].double().numpy(),
        lookahead_kl_shared_gradients=lookahead_tensors["shared_kl_gradients"]
        .double()
        .numpy(),
        **{key: value for key, value in problem.items() if key != "current_direction"},
    )
    expected = candidate.diagnostics["diagnostics_sha256"]
    revalidation = revalidate_counterfactual_kl_extragradient_update(
        candidate, expected_diagnostics_sha256=expected
    )
    if revalidation.get("passes") is not True:
        raise SymmetricSequentialDMSCertificateError("CKES candidate failed integrity revalidation")
    step_l2 = float(np.linalg.norm(candidate.realized_update))
    final_l2 = float(np.linalg.norm(candidate.realized_direction))
    path_l2 = float(cumulative_path_l2) + step_l2
    if final_l2 > MAX_FINAL_DIRECTION_L2 or path_l2 > MAX_CUMULATIVE_PATH_L2:
        raise SymmetricSequentialDMSCertificateError(
            "CKES candidate violates the fixed final-direction or cumulative-path cap"
        )
    return candidate, {
        "progress_fraction": float(problem["progress_fraction"]),
        "status": "certified",
        "nominal_diagnostics_sha256": nominal.diagnostics["diagnostics_sha256"],
        "nominal_revalidation_sha256": nominal_revalidation["revalidation_sha256"],
        "ckes_diagnostics_sha256": candidate.diagnostics["diagnostics_sha256"],
        "ckes_revalidation_sha256": revalidation["revalidation_sha256"],
        "step_l2": step_l2,
        "final_l2": final_l2,
        "path_l2": path_l2,
    }


def _actual_kl_gate(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    strata = {
        "other_permanent": [],
        "self_temporary": [],
        "other_temporary": [],
        "unrelated": [],
    }
    for row in observations:
        category = str(row["category"])
        if category == "target":
            continue
        value = row.get("full_vocabulary_kl_changed_to_baseline")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError("a non-target CKES observation lacks finite actual KL")
        strata[category].append(float(value))
    expected_counts = {
        "other_permanent": 8,
        "self_temporary": 8,
        "other_temporary": 8,
        "unrelated": 16,
    }
    reports = {}
    for name, values in strata.items():
        if len(values) != expected_counts[name]:
            raise RuntimeError(f"CKES actual KL stratum coverage differs: {name}")
        reports[name] = _base()._kl_report(values)
    return {
        "limits": KL_LIMITS,
        "strata": reports,
        "passes": all(report["passes"] for report in reports.values()),
    }


def _target_stopping_gate(
    observations: Sequence[Mapping[str, Any]], contexts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    targets = [row for row in contexts if row["category"] == "target"]
    plus = {
        str(row["form_id"]): row
        for row in observations
        if row["branch_sign"] == 1 and row["category"] == "target"
    }
    minus = {
        str(row["form_id"]): row
        for row in observations
        if row["branch_sign"] == -1 and row["category"] == "target"
    }
    rows = []
    for context in targets:
        form_id = str(context["form_id"])
        positive = plus.get(form_id)
        negative = minus.get(form_id)
        baseline = context["baseline"]
        if positive is None or negative is None:
            raise RuntimeError("CKES stopping gate target coverage differs")
        plus_change = bool(
            positive["unrestricted_predicted_token_id"] != baseline["predicted_token_id"]
            or positive["unrestricted_semantic_choice"] != baseline["semantic_choice"]
        )
        minus_change = bool(
            negative["unrestricted_predicted_token_id"] != baseline["predicted_token_id"]
            or negative["unrestricted_semantic_choice"] != baseline["semantic_choice"]
        )
        passes = bool(
            positive["answer_format_valid"]
            and negative["answer_format_valid"]
            and positive["unrestricted_semantic_choice"] == "preserve"
            and negative["unrestricted_semantic_choice"] == "comply"
            and float(positive["positive_minus_negative_log_odds"]) >= FINAL_TARGET_MARGIN
            and float(negative["positive_minus_negative_log_odds"]) <= -FINAL_TARGET_MARGIN
            and (plus_change or minus_change)
        )
        rows.append(
            {
                "form_id": form_id,
                "assignment": context["form"]["assignment"],
                "preserve_first": context["form"]["preserve_first"],
                "plus_margin": positive["positive_minus_negative_log_odds"],
                "minus_margin": negative["positive_minus_negative_log_odds"],
                "plus_decision_changed": plus_change,
                "minus_decision_changed": minus_change,
                "passes": passes,
            }
        )
    return {
        "target_rows": rows,
        "plus_decision_change_count": sum(row["plus_decision_changed"] for row in rows),
        "minus_decision_change_count": sum(row["minus_decision_changed"] for row in rows),
        "assignment_order_decision_change_count": sum(
            row["plus_decision_changed"] or row["minus_decision_changed"] for row in rows
        ),
        "passes": len(rows) == TARGET_COUNT and all(row["passes"] for row in rows),
    }


def _capture_candidate_state(
    torch: Any,
    *,
    split: str,
    backend_getter: Any,
    scenario_id: str,
    contexts: Sequence[Mapping[str, Any]],
    previous_metadata: Mapping[str, Any],
    previous_tensors: Mapping[str, Any],
    candidate: Any,
    progress: float,
    attempts: Sequence[Mapping[str, Any]],
    trial_index: int,
    ledger: ComputeLedger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _state_path(split, scenario_id, trial_index)
    work_id = f"{split}:{scenario_id}:trial={trial_index}:48_signed_candidate_captures"
    if path.exists():
        ledger.require_artifact(work_id=work_id, path=path)
        return _load_checkpoint(torch, path=path, schema=STATE_SCHEMA)
    revalidation = revalidate_counterfactual_kl_extragradient_update(
        candidate,
        expected_diagnostics_sha256=candidate.diagnostics["diagnostics_sha256"],
    )
    if revalidation.get("passes") is not True:
        raise SymmetricSequentialDMSCertificateError(
            "CKES candidate failed revalidation immediately before deployment"
        )
    direction = torch.from_numpy(candidate.realized_direction.copy()).double().contiguous()
    direction_hash = canonical_sha256(direction.tolist())
    scale = float(previous_metadata["residual_scale"])
    plus_delta = torch.from_numpy(candidate.positive_physical_float32.copy()).float().contiguous()
    minus_delta = torch.from_numpy(candidate.negative_physical_float32.copy()).float().contiguous()
    if (
        not torch.equal(minus_delta, -plus_delta)
        or not torch.equal(plus_delta, (direction * scale).float().contiguous())
    ):
        raise RuntimeError("CKES candidate physical bytes do not round-trip")
    ledger.reserve(
        work_id=work_id,
        forward=FORMS_PER_SCENARIO * 2,
        backward=FORMS_PER_SCENARIO * 2,
        kind="nonzero_symmetric_candidate_margin_gradient_and_actual_kl",
    )
    backend = backend_getter()
    observations = []
    gradients = []
    try:
        for branch_sign, signed_delta in ((1, plus_delta), (-1, minus_delta)):
            signed_hash = tensor_float32_sha256(signed_delta)
            for context in contexts:
                form = context["form"]
                return_logits = context["category"] != "target"
                capture = capture_closed_loop_dms_step(
                    backend,
                    str(form["prompt"]),
                    str(form["positive_label"]),
                    str(form["negative_label"]),
                    positive_semantic=str(form["positive_semantic"]),
                    negative_semantic=str(form["negative_semantic"]),
                    layer=SELECTED_LAYER,
                    anchor_index=int(context["anchor_index"]),
                    branch_sign=branch_sign,
                    cumulative_standardized_direction=direction,
                    physical_residual_scale=scale,
                    signed_delta=signed_delta,
                    expected_signed_delta_float32_sha256=signed_hash,
                    expected_cumulative_standardized_direction_sha256=direction_hash,
                    expected_choice_boundary_evidence_sha256=context[
                        "choice_boundary_evidence_sha256"
                    ],
                    expected_prompt_token_ids_sha256=context["prompt_token_ids_sha256"],
                    expected_pre_anchor_residual_float32_sha256=context[
                        "pre_anchor_residual_float32_sha256"
                    ],
                    maximum_realized_relative_l2_error=(
                        HOOK_REALIZATION_RELATIVE_L2_TOLERANCE
                    ),
                    return_full_logits=return_logits,
                )
                gradient_index = len(gradients)
                gradients.append(capture.raw_anchor_gradient)
                observation = _base()._observation_record(
                    context=context,
                    branch_sign=branch_sign,
                    gradient_index=gradient_index,
                    margin=capture.positive_minus_negative_log_odds,
                    predicted_token_id=capture.unrestricted_predicted_token_id,
                    predicted_label=capture.unrestricted_predicted_label,
                    semantic_choice=capture.unrestricted_semantic_choice,
                    pair_choice_label=capture.pair_choice_label,
                    pair_semantic_choice=capture.pair_semantic_choice,
                    answer_format_valid=capture.answer_format_valid,
                    audit=capture.audit,
                )
                if return_logits:
                    observation["full_vocabulary_kl_changed_to_baseline"] = (
                        _base().full_vocabulary_kl_float64(
                            torch, context["baseline_logits"], capture.full_logits
                        )
                    )
                    observation["changed_full_logits_float32_sha256"] = (
                        tensor_float32_sha256(capture.full_logits)
                    )
                else:
                    observation["full_vocabulary_kl_changed_to_baseline"] = None
                    observation["changed_full_logits_float32_sha256"] = None
                observations.append(observation)
        base_gate = _base()._actual_candidate_gate(
            previous_metadata=previous_metadata,
            previous_tensors=previous_tensors,
            candidate_observations=observations,
            contexts=contexts,
            solver_diagnostics=candidate.diagnostics,
        )
        kl_gate = _actual_kl_gate(observations)
        stopping = _target_stopping_gate(observations, contexts)
        accepted = bool(base_gate["passes"] and kl_gate["passes"])
        step_l2 = float(np.linalg.norm(candidate.realized_update))
        metadata = {
            "schema_version": STATE_SCHEMA,
            "status": "accepted_state" if accepted else "rejected_state_fail_closed",
            "split": split,
            "lock_identity_sha256": _load_lock()["lock_identity_sha256"],
            "scenario_id": scenario_id,
            "state_index": trial_index,
            "trial_index": trial_index,
            "parent_accepted_trial_index": previous_metadata["trial_index"],
            "parent_accepted_checkpoint_sha256": previous_metadata["checkpoint_sha256"],
            "baseline_checkpoint_sha256": previous_metadata["baseline_checkpoint_sha256"],
            "residual_scale": scale,
            "direction_sha256": direction_hash,
            "positive_physical_delta_float32_sha256": tensor_float32_sha256(plus_delta),
            "negative_physical_delta_float32_sha256": tensor_float32_sha256(minus_delta),
            "exact_float32_negation": bool(torch.equal(minus_delta, -plus_delta)),
            "direction_l2": float(direction.norm().item()),
            "cumulative_path_l2": float(previous_metadata["cumulative_path_l2"]) + step_l2,
            "step_l2": step_l2,
            "accepted": accepted,
            "stopping_gate_passes": bool(accepted and stopping["passes"]),
            "target_stopping_gate": stopping,
            "actual_candidate_gate": {**base_gate, "actual_kl": kl_gate},
            "selected_progress_fraction": progress,
            "solver_attempts": _plain(attempts),
            "solver_revalidation_sha256": revalidation["revalidation_sha256"],
            "solver_diagnostics": _plain(candidate.diagnostics),
            "observations": observations,
            "observation_layout": "plus_24_then_minus_24",
            "model_forwards": FORMS_PER_SCENARIO * 2,
            "model_backwards": FORMS_PER_SCENARIO * 2,
            "intermediate_full_logits_stored": False,
        }
        _save_checkpoint(
            torch,
            path=path,
            metadata=metadata,
            tensors={
                "direction": direction,
                "positive_physical_float32": plus_delta,
                "negative_physical_float32": minus_delta,
                "raw_gradients": torch.stack(gradients).float().contiguous(),
            },
        )
        ledger.complete(work_id=work_id, artifact_path=path)
        return _load_checkpoint(torch, path=path, schema=STATE_SCHEMA)
    except Exception as error:
        failure = path.with_suffix(".failure.json")
        if not failure.exists():
            value = _with_hash(
                {
                    "schema_version": "sp_lense.counterfactual_kl_candidate_failure.v1",
                    "status": "runtime_exception_after_full_batch_compute_reservation",
                    "split": split,
                    "scenario_id": scenario_id,
                    "trial_index": trial_index,
                    "work_id": work_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "reserved_forwards": FORMS_PER_SCENARIO * 2,
                    "reserved_backwards": FORMS_PER_SCENARIO * 2,
                },
                "failure_sha256",
            )
            _write_new_json(failure, value)
        ledger.complete(work_id=work_id, artifact_path=failure)
        raise CandidateRuntimeFailure(str(error)) from error


def _terminal_path(split: str, scenario_id: str) -> Path:
    return _paths(split)["states"] / scenario_id / "terminal.json"


def _terminal_record(
    *,
    split: str,
    scenario_id: str,
    status: str,
    outcome_class: str,
    state_metadata: Mapping[str, Any],
    reason: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if outcome_class not in TERMINAL_OUTCOME_CLASSES:
        raise ValueError("unknown CKES terminal outcome class")
    if (status == "success") != (outcome_class == "success"):
        raise ValueError("CKES terminal status and outcome class disagree")
    path = _terminal_path(split, scenario_id)
    value = _with_hash(
        {
            "schema_version": "sp_lense.counterfactual_kl_extragradient_terminal.v1",
            "status": status,
            "outcome_class": outcome_class,
            "split": split,
            "lock_identity_sha256": _load_lock()["lock_identity_sha256"],
            "scenario_id": scenario_id,
            "state_trial_index": int(state_metadata["trial_index"]),
            "state_checkpoint_sha256": state_metadata["checkpoint_sha256"],
            "state_direction_sha256": state_metadata["direction_sha256"],
            "reason": reason,
            "extra": {} if extra is None else _plain(extra),
        },
        "terminal_sha256",
    )
    if path.exists():
        observed = _load_json(path)
        _verify_hash(observed, "terminal_sha256")
        if observed != value:
            raise RuntimeError("CKES terminal record differs from reconstruction")
        return observed
    _write_new_json(path, value)
    return value


def _load_states(
    torch: Any,
    *,
    split: str,
    scenario_id: str,
    contexts: Sequence[Mapping[str, Any]],
    residual_scale: float,
    baseline_checkpoint_sha256: str,
    lock_identity_sha256: str,
    ledger: ComputeLedger,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Load one exact state chain and reconcile every nonzero state to compute."""

    directory = _paths(split)["states"] / scenario_id
    if not directory.exists():
        return []
    paths = sorted(directory.glob("trial_*.pt"))
    if any(path.resolve() != _state_path(split, scenario_id, index).resolve() for index, path in enumerate(paths)):
        raise RuntimeError("CKES state filenames are not the canonical contiguous trial paths")
    states = [_load_checkpoint(torch, path=path, schema=STATE_SCHEMA) for path in paths]
    if [int(metadata["trial_index"]) for metadata, _ in states] != list(range(len(states))):
        raise RuntimeError("CKES state trial indices are not contiguous")
    if any(
        type(metadata.get("state_index")) is not int
        or int(metadata["state_index"]) != int(metadata["trial_index"])
        for metadata, _ in states
    ):
        raise RuntimeError("CKES state-index compatibility alias differs from trial index")

    form_ids = [str(context["form_id"]) for context in contexts]
    if len(form_ids) != FORMS_PER_SCENARIO or len(set(form_ids)) != FORMS_PER_SCENARIO:
        raise RuntimeError("CKES state context coverage differs")
    accepted_metadata: Mapping[str, Any] | None = None
    for index, ((metadata, tensors), path) in enumerate(zip(states, paths, strict=True)):
        if (
            metadata.get("split") != split
            or metadata.get("scenario_id") != scenario_id
            or metadata.get("lock_identity_sha256") != lock_identity_sha256
            or metadata.get("baseline_checkpoint_sha256") != baseline_checkpoint_sha256
            or float(metadata.get("residual_scale", math.nan)) != float(residual_scale)
        ):
            raise RuntimeError("CKES state identity, baseline, or residual scale differs")
        direction = tensors.get("direction")
        raw_gradients = tensors.get("raw_gradients")
        if (
            direction is None
            or raw_gradients is None
            or tuple(direction.shape) != (MODEL["d_model"],)
            or direction.dtype != torch.float64
            or metadata.get("direction_sha256") != canonical_sha256(direction.tolist())
        ):
            raise RuntimeError("CKES state direction tensor or identity differs")
        observations = metadata.get("observations")
        if not isinstance(observations, list):
            raise TypeError("CKES state observations are missing")
        if index == 0:
            expected_layout = [(0, form_id, offset) for offset, form_id in enumerate(form_ids)]
            observed_layout = [
                (row.get("branch_sign"), str(row.get("form_id")), row.get("gradient_index"))
                for row in observations
            ]
            if (
                set(tensors) != {"direction", "raw_gradients"}
                or tuple(raw_gradients.shape) != (FORMS_PER_SCENARIO, MODEL["d_model"])
                or raw_gradients.dtype != torch.float32
                or not torch.equal(direction, torch.zeros_like(direction))
                or observed_layout != expected_layout
                or metadata.get("status") != "accepted_state0"
                or metadata.get("accepted") is not True
                or metadata.get("stopping_gate_passes") is not False
                or metadata.get("parent_accepted_trial_index") is not None
                or float(metadata.get("direction_l2", math.nan)) != 0.0
                or float(metadata.get("cumulative_path_l2", math.nan)) != 0.0
            ):
                raise RuntimeError("CKES state-zero coverage or invariants differ")
            accepted_metadata = metadata
            continue
        work_id = f"{split}:{scenario_id}:trial={index}:48_signed_candidate_captures"
        ledger.require_artifact(work_id=work_id, path=path)
        if accepted_metadata is None or (
            metadata.get("parent_accepted_checkpoint_sha256")
            != accepted_metadata["checkpoint_sha256"]
            or metadata.get("parent_accepted_trial_index")
            != accepted_metadata["trial_index"]
        ):
            raise RuntimeError("CKES state parent chain differs")
        expected_layout = [
            (branch_sign, form_id, offset)
            for branch_offset, branch_sign in ((0, 1), (FORMS_PER_SCENARIO, -1))
            for offset, form_id in enumerate(form_ids, start=branch_offset)
        ]
        observed_layout = [
            (row.get("branch_sign"), str(row.get("form_id")), row.get("gradient_index"))
            for row in observations
        ]
        positive = tensors.get("positive_physical_float32")
        negative = tensors.get("negative_physical_float32")
        gate = metadata.get("actual_candidate_gate")
        stopping = metadata.get("target_stopping_gate")
        expected_accepted = bool(
            isinstance(gate, Mapping)
            and gate.get("passes") is True
            and isinstance(gate.get("actual_kl"), Mapping)
            and gate["actual_kl"].get("passes") is True
        )
        if (
            set(tensors)
            != {
                "direction",
                "positive_physical_float32",
                "negative_physical_float32",
                "raw_gradients",
            }
            or tuple(raw_gradients.shape)
            != (FORMS_PER_SCENARIO * 2, MODEL["d_model"])
            or raw_gradients.dtype != torch.float32
            or positive is None
            or negative is None
            or tuple(positive.shape) != (MODEL["d_model"],)
            or positive.dtype != torch.float32
            or negative.dtype != torch.float32
            or not torch.equal(negative, -positive)
            or not torch.equal(positive, (direction * residual_scale).float().contiguous())
            or metadata.get("positive_physical_delta_float32_sha256")
            != tensor_float32_sha256(positive)
            or metadata.get("negative_physical_delta_float32_sha256")
            != tensor_float32_sha256(negative)
            or observed_layout != expected_layout
            or metadata.get("accepted") is not expected_accepted
            or metadata.get("status")
            != ("accepted_state" if expected_accepted else "rejected_state_fail_closed")
            or not isinstance(stopping, Mapping)
            or metadata.get("stopping_gate_passes")
            is not bool(expected_accepted and stopping.get("passes") is True)
            or metadata.get("selected_progress_fraction") not in PROGRESS_SCHEDULE
            or metadata.get("model_forwards") != FORMS_PER_SCENARIO * 2
            or metadata.get("model_backwards") != FORMS_PER_SCENARIO * 2
        ):
            raise RuntimeError("CKES nonzero state coverage, bytes, or gates differ")
        if metadata["accepted"] is True:
            accepted_metadata = metadata
    return states


def _current_accepted(
    states: Sequence[tuple[dict[str, Any], dict[str, Any]]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return next(state for state in reversed(states) if state[0]["accepted"] is True)


def _rejected_progresses(
    states: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    current_metadata: Mapping[str, Any],
) -> set[float]:
    return {
        float(metadata["selected_progress_fraction"])
        for metadata, _ in states
        if metadata["accepted"] is False
        and metadata.get("parent_accepted_checkpoint_sha256")
        == current_metadata["checkpoint_sha256"]
    }


def _baseline_qualification(metadata: Mapping[str, Any]) -> dict[str, Any]:
    records = metadata["records"]
    invalid = [row["form_id"] for row in records if row["answer_format_valid"] is not True]
    unrelated = [row for row in records if row["form"].get("family") == "unrelated"]
    incorrect = [
        row["form_id"]
        for row in unrelated
        if row["semantic_choice"] != "preferred"
        or row["pair_semantic_choice"] != "preferred"
    ]
    return {
        "record_count": len(records),
        "invalid_answer_format_form_ids": invalid,
        "unrelated_not_preferred_form_ids": incorrect,
        "passes": len(records) == STATE0_FB and not invalid and not incorrect,
    }


def _final_observation(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "form_id": record["form_id"],
        "category": record["category"],
        "branch_sign": record["branch_sign"],
        "positive_minus_negative_log_odds": record[
            "positive_minus_negative_log_odds"
        ],
        "unrestricted_predicted_token_id": record["predicted_token_id"],
        "unrestricted_semantic_choice": record["semantic_choice"],
        "answer_format_valid": record["answer_format_valid"],
        "full_vocabulary_kl_changed_to_baseline": record[
            "full_vocabulary_kl_changed_to_baseline"
        ],
    }


def _signed_final_maps(
    records: Sequence[Mapping[str, Any]], contexts: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    form_ids = [str(context["form_id"]) for context in contexts]
    expected = {(branch_sign, form_id) for branch_sign in (1, -1) for form_id in form_ids}
    observed = [(int(row["branch_sign"]), str(row["form_id"])) for row in records]
    if len(records) != FORMS_PER_SCENARIO * 2 or set(observed) != expected or len(set(observed)) != len(observed):
        raise RuntimeError("CKES final signed form coverage differs")
    plus = {str(row["form_id"]): row for row in records if int(row["branch_sign"]) == 1}
    minus = {str(row["form_id"]): row for row in records if int(row["branch_sign"]) == -1}
    return plus, minus


def _final_protected_unrelated_gate(
    records: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
    state_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Repeat the locked protected floors and nonlinear unrelated-return checks."""

    plus, minus = _signed_final_maps(records, contexts)
    reasons: list[str] = []
    protected_rows = []
    for context in _ordered_family(contexts, "protected"):
        form_id = str(context["form_id"])
        baseline_margin = float(context["baseline"]["positive_minus_negative_log_odds"])
        baseline_sign = 1.0 if baseline_margin >= 0.0 else -1.0
        floor = min(
            PROTECTED_MAXIMUM_FLOOR,
            PROTECTED_BASELINE_FRACTION * abs(baseline_margin),
        )
        plus_oriented = baseline_sign * float(
            plus[form_id]["positive_minus_negative_log_odds"]
        )
        minus_oriented = baseline_sign * float(
            minus[form_id]["positive_minus_negative_log_odds"]
        )
        passes = plus_oriented >= floor and minus_oriented >= floor
        if not passes:
            reasons.append(f"protected_floor_failed:{form_id}")
        protected_rows.append(
            {
                "form_id": form_id,
                "baseline_sign": baseline_sign,
                "floor": floor,
                "plus_oriented_margin": plus_oriented,
                "minus_oriented_margin": minus_oriented,
                "passes": passes,
            }
        )

    diagnostics = state_metadata.get("solver_diagnostics")
    certificate = (
        diagnostics.get("realized_deployment_certificate")
        if isinstance(diagnostics, Mapping)
        else None
    )
    if not isinstance(certificate, Mapping) or certificate.get("passes") is not True:
        raise RuntimeError("CKES final state lacks a passing realized-deployment certificate")
    unrelated = _ordered_family(contexts, "unrelated")
    desired_plus = list(map(float, certificate.get("unrelated_plus_desired_margins", ())))
    desired_minus = list(map(float, certificate.get("unrelated_minus_desired_margins", ())))
    if len(unrelated) != UNRELATED_COUNT or len(desired_plus) != UNRELATED_COUNT or len(desired_minus) != UNRELATED_COUNT:
        raise RuntimeError("CKES final unrelated-return certificate coverage differs")
    unrelated_rows = []
    for index, context in enumerate(unrelated):
        form_id = str(context["form_id"])
        observed_plus = float(plus[form_id]["positive_minus_negative_log_odds"])
        observed_minus = float(minus[form_id]["positive_minus_negative_log_odds"])
        plus_error = abs(observed_plus - desired_plus[index])
        minus_error = abs(observed_minus - desired_minus[index])
        passes = (
            plus_error <= UNRELATED_LINEARIZATION_ERROR_CAP
            and minus_error <= UNRELATED_LINEARIZATION_ERROR_CAP
        )
        if not passes:
            reasons.append(f"unrelated_nonlinear_return_failed:{form_id}")
        unrelated_rows.append(
            {
                "form_id": form_id,
                "plus_margin": observed_plus,
                "minus_margin": observed_minus,
                "desired_plus_margin": desired_plus[index],
                "desired_minus_margin": desired_minus[index],
                "plus_absolute_error": plus_error,
                "minus_absolute_error": minus_error,
                "passes": passes,
            }
        )
    return {
        "protected_rows": protected_rows,
        "unrelated_rows": unrelated_rows,
        "protected_maximum_floor": PROTECTED_MAXIMUM_FLOOR,
        "protected_baseline_fraction": PROTECTED_BASELINE_FRACTION,
        "unrelated_linearization_error_cap": UNRELATED_LINEARIZATION_ERROR_CAP,
        "reasons": reasons,
        "passes": not reasons,
    }


def _cluster_contrast_estimands(
    records: Sequence[Mapping[str, Any]], contexts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compute cluster-level self, matched-other, and selectivity contrasts."""

    plus, minus = _signed_final_maps(records, contexts)
    target = _ordered_family(contexts, "target")
    matched = _ordered_family(contexts, "protected")
    matched = [context for context in matched if context["category"] == "other_permanent"]

    def cell_key(context: Mapping[str, Any]) -> tuple[str, bool]:
        form = context["form"]
        return str(form["assignment"]), bool(form["preserve_first"])

    target_by_cell = {cell_key(context): context for context in target}
    matched_by_cell = {cell_key(context): context for context in matched}
    if (
        len(target_by_cell) != TARGET_COUNT
        or len(matched_by_cell) != TARGET_COUNT
        or set(target_by_cell) != set(matched_by_cell)
    ):
        raise RuntimeError("CKES self/matched-other estimand pairing differs")
    cells = []
    for assignment, preserve_first in sorted(target_by_cell):
        self_context = target_by_cell[(assignment, preserve_first)]
        other_context = matched_by_cell[(assignment, preserve_first)]

        def effects(context: Mapping[str, Any]) -> tuple[float, float, float]:
            form_id = str(context["form_id"])
            baseline = float(context["baseline"]["positive_minus_negative_log_odds"])
            plus_change = float(plus[form_id]["positive_minus_negative_log_odds"]) - baseline
            negative_oriented_change = baseline - float(
                minus[form_id]["positive_minus_negative_log_odds"]
            )
            return plus_change, negative_oriented_change, 0.5 * (
                plus_change + negative_oriented_change
            )

        self_plus, self_minus, self_bidirectional = effects(self_context)
        other_plus, other_minus, other_bidirectional = effects(other_context)
        cells.append(
            {
                "assignment": assignment,
                "preserve_first": preserve_first,
                "self_form_id": self_context["form_id"],
                "matched_other_form_id": other_context["form_id"],
                "self_plus_change_from_baseline": self_plus,
                "self_negative_oriented_change_from_baseline": self_minus,
                "self_bidirectional_average_oriented_change": self_bidirectional,
                "matched_other_plus_change_from_baseline": other_plus,
                "matched_other_negative_oriented_change_from_baseline": other_minus,
                "matched_other_bidirectional_average_oriented_change": other_bidirectional,
                "self_minus_matched_other_bidirectional_effect": (
                    self_bidirectional - other_bidirectional
                ),
            }
        )

    def mean(field: str) -> float:
        return float(sum(float(row[field]) for row in cells) / len(cells))

    means = {
        "self_plus_change_from_baseline": mean("self_plus_change_from_baseline"),
        "self_negative_oriented_change_from_baseline": mean(
            "self_negative_oriented_change_from_baseline"
        ),
        "self_bidirectional_average_oriented_change": mean(
            "self_bidirectional_average_oriented_change"
        ),
        "matched_other_plus_change_from_baseline": mean(
            "matched_other_plus_change_from_baseline"
        ),
        "matched_other_negative_oriented_change_from_baseline": mean(
            "matched_other_negative_oriented_change_from_baseline"
        ),
        "matched_other_bidirectional_average_oriented_change": mean(
            "matched_other_bidirectional_average_oriented_change"
        ),
        "self_minus_matched_other_bidirectional_effect": mean(
            "self_minus_matched_other_bidirectional_effect"
        ),
    }
    return {
        "independent_unit": "scenario_cluster",
        "cell_count": len(cells),
        "bidirectional_estimand_equation": (
            "0.5 * ((margin_plus - baseline) + (baseline - margin_minus))"
        ),
        "selectivity_estimand_equation": "self_bidirectional - matched_other_bidirectional",
        "cells": cells,
        "means": means,
        "self_both_signs_positive": bool(
            means["self_plus_change_from_baseline"] > 0.0
            and means["self_negative_oriented_change_from_baseline"] > 0.0
        ),
    }


def _validate_final_checkpoint(
    torch: Any,
    *,
    split: str,
    metadata: Mapping[str, Any],
    tensors: Mapping[str, Any],
    scenario_contexts: Mapping[str, Sequence[Mapping[str, Any]]],
    terminals: Mapping[str, Mapping[str, Any]],
    states_by_scenario: Mapping[
        str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ],
) -> None:
    """Bind a cached/fresh final checkpoint to terminals, directions, rows, and logits."""

    _validate_terminal_state_coverage(
        scenario_contexts=scenario_contexts,
        terminals=terminals,
        states_by_scenario=states_by_scenario,
    )
    successful = sorted(
        scenario_id
        for scenario_id, terminal in terminals.items()
        if terminal["status"] == "success"
    )
    expected_count = len(successful) * FORMS_PER_SCENARIO * 2
    if (
        metadata.get("status") != "complete"
        or metadata.get("split") != split
        or metadata.get("lock_identity_sha256") != _load_lock()["lock_identity_sha256"]
        or metadata.get("successful_scenario_ids") != successful
        or metadata.get("record_count") != expected_count
        or metadata.get("compute")
        != {
            "model_forwards": expected_count,
            "model_backwards": 0,
            "generated_tokens": 0,
        }
        or metadata.get("full_float32_baseline_and_changed_logits_stored") is not True
        or set(metadata.get("direction_records", {})) != set(successful)
        or set(metadata.get("scenario_gates", {})) != set(successful)
        or set(metadata.get("scenario_estimands", {})) != set(successful)
        or set(tensors) != {"baseline_logits", "changed_logits"}
    ):
        raise RuntimeError("CKES final checkpoint identity or top-level coverage differs")
    baseline_logits = tensors["baseline_logits"]
    changed_logits = tensors["changed_logits"]
    if (
        baseline_logits.dtype != torch.float32
        or changed_logits.dtype != torch.float32
        or baseline_logits.ndim != 2
        or changed_logits.ndim != 2
        or tuple(baseline_logits.shape) != tuple(changed_logits.shape)
        or int(baseline_logits.shape[0]) != expected_count
    ):
        raise RuntimeError("CKES final full-logit tensor coverage differs")
    records = metadata.get("records")
    if not isinstance(records, list) or len(records) != expected_count:
        raise TypeError("CKES final records are missing or incomplete")

    expected_rows: list[tuple[str, int, Mapping[str, Any]]] = []
    for scenario_id in successful:
        for branch_sign in (1, -1):
            expected_rows.extend(
                (scenario_id, branch_sign, context)
                for context in scenario_contexts[scenario_id]
            )
    finite = _base()._finite()
    scenario_rows: dict[str, list[Mapping[str, Any]]] = {
        scenario_id: [] for scenario_id in successful
    }
    for index, (record, (scenario_id, branch_sign, context)) in enumerate(
        zip(records, expected_rows, strict=True)
    ):
        form_id = str(context["form_id"])
        unhashed = {key: value for key, value in record.items() if key != "row_sha256"}
        baseline = baseline_logits[index].float().contiguous()
        changed = changed_logits[index].float().contiguous()
        if (
            record.get("row_sha256") != canonical_sha256(unhashed)
            or record.get("tensor_row_index") != index
            or record.get("scenario_id") != scenario_id
            or int(record.get("branch_sign", 0)) != branch_sign
            or str(record.get("form_id")) != form_id
            or record.get("category") != context["category"]
            or not torch.equal(baseline, context["baseline_logits"].float().contiguous())
            or tensor_float32_sha256(baseline)
            != context["baseline"]["logits_float32_sha256"]
            or tensor_float32_sha256(changed) != record.get("logits_float32_sha256")
        ):
            raise RuntimeError("CKES final row identity, order, or logit hash differs")
        rescored = finite._score_logits(
            torch,
            logits=changed,
            form=context["baseline"]["form"],
            positive_id=int(context["baseline"]["positive_token_id"]),
            negative_id=int(context["baseline"]["negative_token_id"]),
            baseline_logits=baseline,
        )
        if any(record.get(key) != value for key, value in rescored.items()):
            raise RuntimeError("CKES final stored score differs from its full logits")
        scenario_rows[scenario_id].append(record)

    recomputed_gates = {}
    recomputed_estimands = {}
    for scenario_id in successful:
        terminal = terminals[scenario_id]
        state_metadata = states_by_scenario[scenario_id][
            int(terminal["state_trial_index"])
        ][0]
        direction_record = metadata["direction_records"][scenario_id]
        if direction_record != {
            "trial_index": int(terminal["state_trial_index"]),
            "state_checkpoint_sha256": terminal["state_checkpoint_sha256"],
            "direction_sha256": terminal["state_direction_sha256"],
        }:
            raise RuntimeError("CKES final direction record differs from its terminal state")
        rows = scenario_rows[scenario_id]
        normalized = [_final_observation(row) for row in rows]
        kl_gate = _actual_kl_gate(normalized)
        stopping = _target_stopping_gate(normalized, scenario_contexts[scenario_id])
        protected_unrelated = _final_protected_unrelated_gate(
            rows, scenario_contexts[scenario_id], state_metadata
        )
        non_target_changes = [
            row["form_id"]
            for row in rows
            if row["category"] != "target"
            and (row["greedy_token_changed"] or row["semantic_choice_changed"])
        ]
        invalid = [row["form_id"] for row in rows if not row["answer_format_valid"]]
        recomputed_gates[scenario_id] = {
            "actual_kl": kl_gate,
            "target_stopping": stopping,
            "protected_and_unrelated_repeat": protected_unrelated,
            "non_target_changed_form_ids": non_target_changes,
            "invalid_answer_form_ids": invalid,
            "passes": bool(
                kl_gate["passes"]
                and stopping["passes"]
                and protected_unrelated["passes"]
                and not non_target_changes
                and not invalid
            ),
        }
        recomputed_estimands[scenario_id] = _cluster_contrast_estimands(
            rows, scenario_contexts[scenario_id]
        )
    if (
        _plain(metadata["scenario_gates"]) != recomputed_gates
        or _plain(metadata["scenario_estimands"]) != recomputed_estimands
    ):
        raise RuntimeError("CKES final gates or contrast estimands differ on reconstruction")


def _run_or_load_final(
    torch: Any,
    *,
    split: str,
    backend_getter: Any,
    scenario_contexts: Mapping[str, Sequence[Mapping[str, Any]]],
    terminals: Mapping[str, Mapping[str, Any]],
    states_by_scenario: Mapping[
        str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ],
    ledger: ComputeLedger,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    _validate_terminal_state_coverage(
        scenario_contexts=scenario_contexts,
        terminals=terminals,
        states_by_scenario=states_by_scenario,
    )
    successful = sorted(
        scenario_id for scenario_id, terminal in terminals.items() if terminal["status"] == "success"
    )
    if not successful:
        return None
    path = _paths(split)["final"]
    work_id = f"{split}:final:{len(successful)}_successful_scenarios_full_logits"
    if path.exists():
        ledger.require_artifact(work_id=work_id, path=path)
        cached = _load_checkpoint(torch, path=path, schema=FINAL_SCHEMA)
        _validate_final_checkpoint(
            torch,
            split=split,
            metadata=cached[0],
            tensors=cached[1],
            scenario_contexts=scenario_contexts,
            terminals=terminals,
            states_by_scenario=states_by_scenario,
        )
        return cached
    expected = len(successful) * FORMS_PER_SCENARIO * 2
    ledger.reserve(
        work_id=work_id,
        forward=expected,
        backward=0,
        kind="fresh_final_forward_only_full_logits",
    )
    backend = backend_getter()
    records = []
    baseline_logits = []
    changed_logits = []
    scenario_gates = {}
    direction_records = {}
    scenario_estimands = {}
    for scenario_id in successful:
        terminal = terminals[scenario_id]
        trial_index = int(terminal["state_trial_index"])
        state_metadata, state_tensors = states_by_scenario[scenario_id][trial_index]
        direction = state_tensors["direction"].double().contiguous()
        plus_delta = state_tensors["positive_physical_float32"].float().contiguous()
        minus_delta = state_tensors["negative_physical_float32"].float().contiguous()
        if (
            state_metadata["stopping_gate_passes"] is not True
            or not torch.equal(minus_delta, -plus_delta)
            or not torch.equal(
                plus_delta,
                (direction * float(state_metadata["residual_scale"])).float().contiguous(),
            )
        ):
            raise RuntimeError("CKES final state bytes or stopping gate differ")
        direction_records[scenario_id] = {
            "trial_index": trial_index,
            "state_checkpoint_sha256": state_metadata["checkpoint_sha256"],
            "direction_sha256": state_metadata["direction_sha256"],
        }
        scenario_rows = []
        for branch_sign, signed_delta in ((1, plus_delta), (-1, minus_delta)):
            for context in scenario_contexts[scenario_id]:
                record, baseline, changed = _base()._final_forward(
                    torch,
                    backend=backend,
                    context=context,
                    signed_delta=signed_delta,
                    branch_sign=branch_sign,
                )
                record["tensor_row_index"] = len(records)
                record["row_sha256"] = canonical_sha256(
                    {key: value for key, value in record.items() if key != "row_sha256"}
                )
                records.append(record)
                scenario_rows.append(record)
                baseline_logits.append(baseline)
                changed_logits.append(changed)
        normalized = [_final_observation(row) for row in scenario_rows]
        kl_gate = _actual_kl_gate(normalized)
        stopping = _target_stopping_gate(normalized, scenario_contexts[scenario_id])
        protected_unrelated = _final_protected_unrelated_gate(
            scenario_rows,
            scenario_contexts[scenario_id],
            state_metadata,
        )
        scenario_estimands[scenario_id] = _cluster_contrast_estimands(
            scenario_rows, scenario_contexts[scenario_id]
        )
        non_target_changes = [
            row["form_id"]
            for row in scenario_rows
            if row["category"] != "target"
            and (row["greedy_token_changed"] or row["semantic_choice_changed"])
        ]
        other = [row["form_id"] for row in scenario_rows if not row["answer_format_valid"]]
        scenario_gates[scenario_id] = {
            "actual_kl": kl_gate,
            "target_stopping": stopping,
            "protected_and_unrelated_repeat": protected_unrelated,
            "non_target_changed_form_ids": non_target_changes,
            "invalid_answer_form_ids": other,
            "passes": bool(
                kl_gate["passes"]
                and stopping["passes"]
                and protected_unrelated["passes"]
                and not non_target_changes
                and not other
            ),
        }
    metadata = {
        "schema_version": FINAL_SCHEMA,
        "status": "complete",
        "split": split,
        "lock_identity_sha256": _load_lock()["lock_identity_sha256"],
        "successful_scenario_ids": successful,
        "direction_records": direction_records,
        "scenario_gates": scenario_gates,
        "scenario_estimands": scenario_estimands,
        "record_count": len(records),
        "records": records,
        "compute": {
            "model_forwards": len(records),
            "model_backwards": 0,
            "generated_tokens": 0,
        },
        "full_float32_baseline_and_changed_logits_stored": True,
    }
    _save_checkpoint(
        torch,
        path=path,
        metadata=metadata,
        tensors={
            "baseline_logits": torch.stack(baseline_logits).float().contiguous(),
            "changed_logits": torch.stack(changed_logits).float().contiguous(),
        },
    )
    ledger.complete(work_id=work_id, artifact_path=path)
    completed = _load_checkpoint(torch, path=path, schema=FINAL_SCHEMA)
    _validate_final_checkpoint(
        torch,
        split=split,
        metadata=completed[0],
        tensors=completed[1],
        scenario_contexts=scenario_contexts,
        terminals=terminals,
        states_by_scenario=states_by_scenario,
    )
    return completed


def _load_terminal(split: str, scenario_id: str) -> dict[str, Any] | None:
    path = _terminal_path(split, scenario_id)
    if not path.exists():
        return None
    value = _load_json(path)
    _verify_hash(value, "terminal_sha256")
    if (
        value.get("split") != split
        or value.get("scenario_id") != scenario_id
        or value.get("lock_identity_sha256") != _load_lock()["lock_identity_sha256"]
        or value.get("outcome_class") not in TERMINAL_OUTCOME_CLASSES
        or (value.get("status") == "success")
        != (value.get("outcome_class") == "success")
    ):
        raise RuntimeError("CKES terminal identity differs")
    return value


def _validate_terminal_state_coverage(
    *,
    scenario_contexts: Mapping[str, Sequence[Mapping[str, Any]]],
    terminals: Mapping[str, Mapping[str, Any]],
    states_by_scenario: Mapping[
        str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ],
) -> None:
    """Require exact cluster coverage and bind every terminal to its accepted state."""

    expected_ids = set(scenario_contexts)
    if (
        len(expected_ids) != SCENARIO_COUNT
        or set(terminals) != expected_ids
        or set(states_by_scenario) != expected_ids
    ):
        raise RuntimeError("CKES scenario, terminal, and state coverage differs")
    for scenario_id in sorted(expected_ids):
        contexts = scenario_contexts[scenario_id]
        if (
            len(contexts) != FORMS_PER_SCENARIO
            or len({str(context["form_id"]) for context in contexts}) != FORMS_PER_SCENARIO
        ):
            raise RuntimeError("CKES terminal context coverage differs")
        states = states_by_scenario[scenario_id]
        if not states:
            raise RuntimeError("CKES terminal scenario has no state zero")
        terminal = terminals[scenario_id]
        trial_index = terminal.get("state_trial_index")
        if type(trial_index) is not int or trial_index < 0 or trial_index >= len(states):
            raise RuntimeError("CKES terminal references an unavailable state")
        state_metadata = states[trial_index][0]
        current_accepted = _current_accepted(states)[0]
        if (
            state_metadata.get("accepted") is not True
            or current_accepted.get("checkpoint_sha256")
            != state_metadata.get("checkpoint_sha256")
            or terminal.get("state_checkpoint_sha256")
            != state_metadata.get("checkpoint_sha256")
            or terminal.get("state_direction_sha256")
            != state_metadata.get("direction_sha256")
            or (terminal.get("status") == "success")
            != (state_metadata.get("stopping_gate_passes") is True)
        ):
            raise RuntimeError("CKES terminal is not bound to the current accepted state")


def _execution_integrity(terminals: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    technical = sorted(
        scenario_id
        for scenario_id, terminal in terminals.items()
        if terminal.get("outcome_class") == "technical_integrity_failure"
    )
    invalid = sorted(
        scenario_id
        for scenario_id, terminal in terminals.items()
        if terminal.get("outcome_class") not in TERMINAL_OUTCOME_CLASSES
        or (terminal.get("status") == "success")
        != (terminal.get("outcome_class") == "success")
    )
    return {
        "expected_scenario_count": SCENARIO_COUNT,
        "observed_scenario_count": len(terminals),
        "technical_failure_scenario_ids": technical,
        "invalid_terminal_scenario_ids": invalid,
        "passes": len(terminals) == SCENARIO_COUNT and not technical and not invalid,
    }


def _accepted_state_integrity(
    states_by_scenario: Mapping[str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]]
) -> dict[str, Any]:
    failures = []
    accepted_count = 0
    for scenario_id, states in states_by_scenario.items():
        for metadata, _ in states[1:]:
            if metadata["accepted"] is not True:
                continue
            accepted_count += 1
            gate = metadata.get("actual_candidate_gate", {})
            if gate.get("passes") is not True or gate.get("actual_kl", {}).get("passes") is not True:
                failures.append(
                    {"scenario_id": scenario_id, "trial_index": metadata["trial_index"]}
                )
    return {
        "accepted_nonzero_state_count": accepted_count,
        "failed_accepted_state_gates": failures,
        "passes": not failures,
    }


def _build_result(
    *,
    split: str,
    baseline_metadata: Mapping[str, Any],
    qualification: Mapping[str, Any],
    terminals: Mapping[str, Mapping[str, Any]],
    scenario_contexts: Mapping[str, Sequence[Mapping[str, Any]]],
    states_by_scenario: Mapping[
        str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ],
    final: tuple[Mapping[str, Any], Mapping[str, Any]] | None,
    ledger: ComputeLedger,
) -> dict[str, Any]:
    from sp_lense.counterfactual_kl_protocol import self_hash_record

    _validate_terminal_state_coverage(
        scenario_contexts=scenario_contexts,
        terminals=terminals,
        states_by_scenario=states_by_scenario,
    )
    successful = sorted(
        scenario_id for scenario_id, terminal in terminals.items() if terminal["status"] == "success"
    )
    execution_integrity = _execution_integrity(terminals)
    accepted_integrity = _accepted_state_integrity(states_by_scenario)
    final_metadata = None if final is None else final[0]
    final_gates = {} if final_metadata is None else final_metadata["scenario_gates"]
    final_estimands = {} if final_metadata is None else final_metadata["scenario_estimands"]
    final_pass = bool(
        successful
        and set(final_gates) == set(successful)
        and set(final_estimands) == set(successful)
        and all(value["passes"] for value in final_gates.values())
    )
    decision_rows = []
    if final_metadata is not None:
        for scenario_id, gate in final_gates.items():
            decision_rows.extend(
                {"scenario_id": scenario_id, **row}
                for row in gate["target_stopping"]["target_rows"]
            )
    changed_rows = sum(
        bool(row["plus_decision_changed"] or row["minus_decision_changed"])
        for row in decision_rows
    )
    plus_changes = sum(bool(row["plus_decision_changed"]) for row in decision_rows)
    minus_changes = sum(bool(row["minus_decision_changed"]) for row in decision_rows)
    non_target_final = bool(
        final_pass
        and all(
            not gate["non_target_changed_form_ids"] and not gate["invalid_answer_form_ids"]
            for gate in final_gates.values()
        )
    )
    actual_kl_final = bool(
        final_pass and all(gate["actual_kl"]["passes"] for gate in final_gates.values())
    )
    ledger.require_unambiguous()
    compute = ledger.snapshot()
    compute_integrity = bool(
        compute["forward_backward"] <= MAX_FB
        and compute["final_forward_only"] <= MAX_FINAL_FORWARD
        and compute["generated_tokens"] == 0
        and compute["external_api_calls"] == 0
        and compute["external_model_judges"] == 0
        and compute["paid_model_cost_usd"] == 0
    )
    gates = {
        "actual_kl": bool(accepted_integrity["passes"] and actual_kl_final),
        "baseline_qualification": qualification["passes"] is True,
        "compute_integrity": compute_integrity,
        "decision_changes": changed_rows >= 12,
        "efficacy": len(successful) >= 3,
        "execution_integrity": execution_integrity["passes"] is True,
        "final_repeat": final_pass,
        "non_target_choice_stability": bool(
            accepted_integrity["passes"] and non_target_final
        ),
    }
    status = "go" if all(gates.values()) else "no_go"
    value = {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "split": split,
        "lock_identity_sha256": _load_lock()["lock_identity_sha256"],
        "lock_file_sha256": file_sha256(LOCK_PATH),
        "dataset_file_sha256": _load_lock()["file_hashes"][f"{split}_dataset"][
            "sha256"
        ],
        "baseline_checkpoint_sha256": baseline_metadata["checkpoint_sha256"],
        "gates": gates,
        "successful_scenario_ids": successful,
        "successful_scenario_count": len(successful),
        "assignment_unit_count": 2 * len(successful),
        "target_assignment_order_decision_change_count": changed_rows,
        "plus_target_decision_change_count": plus_changes,
        "minus_target_decision_change_count": minus_changes,
        "baseline_qualification": _plain(qualification),
        "execution_integrity": execution_integrity,
        "accepted_state_integrity": accepted_integrity,
        "terminals": {key: _plain(value) for key, value in terminals.items()},
        "state_histories": {
            scenario_id: [
                {
                    "trial_index": metadata["trial_index"],
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "direction_sha256": metadata["direction_sha256"],
                    "accepted": metadata["accepted"],
                    "stopping_gate_passes": metadata["stopping_gate_passes"],
                    "selected_progress_fraction": metadata["selected_progress_fraction"],
                }
                for metadata, _ in states
            ]
            for scenario_id, states in states_by_scenario.items()
        },
        "final_checkpoint_sha256": (
            None if final_metadata is None else final_metadata["checkpoint_sha256"]
        ),
        "final_scenario_gates": _plain(final_gates),
        "cluster_contrast_estimands": _plain(final_estimands),
        "compute": compute,
        "generated_tokens": 0,
        "external_api_calls": 0,
        "external_model_judges": 0,
        "paid_model_cost_usd": 0,
        "claim_boundary": _load_lock()["configuration"]["claim_boundary"],
    }
    return self_hash_record(value, hash_field="result_sha256")


def _write_report(result: Mapping[str, Any]) -> str:
    path = _paths(str(result["split"]))["report"]
    lines = [
        "# CKES Qualification Result",
        "",
        f"Status: **{result['status']}**.",
        "",
        f"Successful scenarios: {result['successful_scenario_count']}/4.",
        (
            f"Assignment-order target decision changes: "
            f"{result['target_assignment_order_decision_change_count']}."
        ),
        (
            f"Sign-specific target changes: +D={result['plus_target_decision_change_count']}, "
            f"-D={result['minus_target_decision_change_count']}."
        ),
        "",
        "## Locked gates",
        "",
        *[f"- {name}: `{passed}`" for name, passed in result["gates"].items()],
        "",
        (
            "This result concerns a scenario-local white-box A/B controller. It is not "
            "evidence of a natural self-preservation mechanism or a universal direction."
        ),
        "",
    ]
    text = "\n".join(lines)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError("CKES report differs from its result")
    else:
        _atomic_text(path, text)
    return text


def _load_result(split: str) -> dict[str, Any]:
    from sp_lense.counterfactual_kl_protocol import validate_locked_result

    return validate_locked_result(
        _load_json(_paths(split)["result"]),
        lock=_load_lock(),
        expected_split=split,
    )


def run_development(split: str = "validation") -> dict[str, Any]:
    lock = _load_lock()
    run_preflight(split)
    paths = _paths(split)
    existing_result = _load_result(split) if paths["result"].exists() else None
    import torch

    payload = _load_dataset(split)
    forms = _flatten_forms(payload, split=split)
    ledger = ComputeLedger(split=split, lock_identity_sha256=lock["lock_identity_sha256"])
    ledger.require_unambiguous()
    original = _base()._finite()._load_original_runner()
    original._configure_threads(torch)
    backend_cache: list[Any] = []

    def backend_getter() -> Any:
        if not backend_cache:
            backend_cache.append(original.load_backend())
        return backend_cache[0]

    baseline_metadata, baseline_tensors = _capture_or_load_baseline(
        torch,
        split=split,
        backend_getter=backend_getter,
        payload=payload,
        forms=forms,
        ledger=ledger,
    )
    qualification = _baseline_qualification(baseline_metadata)
    scenario_contexts, scales, nuisance_rows = _build_contexts(
        torch,
        split=split,
        baseline_metadata=baseline_metadata,
        baseline_tensors=baseline_tensors,
    )
    terminals: dict[str, dict[str, Any]] = {}
    states_by_scenario: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for scenario_id, contexts in scenario_contexts.items():
        _state0_checkpoint(
            torch,
            split=split,
            scenario_id=scenario_id,
            contexts=contexts,
            scale=scales[scenario_id],
            baseline_checkpoint_sha256=baseline_metadata["checkpoint_sha256"],
        )
        states = _load_states(
            torch,
            split=split,
            scenario_id=scenario_id,
            contexts=contexts,
            residual_scale=scales[scenario_id],
            baseline_checkpoint_sha256=baseline_metadata["checkpoint_sha256"],
            lock_identity_sha256=lock["lock_identity_sha256"],
            ledger=ledger,
        )
        states_by_scenario[scenario_id] = states
        existing_terminal = _load_terminal(split, scenario_id)
        if existing_terminal is not None:
            terminals[scenario_id] = existing_terminal
            continue
        current_metadata, current_tensors = _current_accepted(states)
        if qualification["passes"] is not True:
            terminals[scenario_id] = _terminal_record(
                split=split,
                scenario_id=scenario_id,
                status="failed",
                outcome_class="scientific_no_success",
                state_metadata=current_metadata,
                reason="fresh baseline qualification failed before any nonzero intervention",
                extra=qualification,
            )
            continue
        if current_metadata["stopping_gate_passes"] is True:
            terminals[scenario_id] = _terminal_record(
                split=split,
                scenario_id=scenario_id,
                status="success",
                outcome_class="success",
                state_metadata=current_metadata,
                reason="the accepted state already passes both signs, assignments, and orders",
            )
            continue
        terminal: dict[str, Any] | None = None
        while len(states) - 1 < MAX_TRIAL_STATES:
            current_metadata, current_tensors = _current_accepted(states)
            rejected = _rejected_progresses(states, current_metadata)
            available = [value for value in PROGRESS_SCHEDULE if value not in rejected]
            if not available:
                terminal = _terminal_record(
                    split=split,
                    scenario_id=scenario_id,
                    status="failed",
                    outcome_class="scientific_no_success",
                    state_metadata=current_metadata,
                    reason="all locked progress fractions were rejected from one accepted parent",
                )
                break
            seed_problem = _controller_problem(
                state_metadata=current_metadata,
                state_tensors=current_tensors,
                contexts=contexts,
                residual_scale=scales[scenario_id],
                standardized_nuisance_rows=nuisance_rows[scenario_id],
                progress=available[0],
            )
            try:
                lookahead_metadata, lookahead_tensors = _capture_or_load_lookahead(
                    torch,
                    split=split,
                    backend_getter=backend_getter,
                    scenario_id=scenario_id,
                    contexts=contexts,
                    state_metadata=current_metadata,
                    state_tensors=current_tensors,
                    problem=seed_problem,
                    ledger=ledger,
                )
            except SymmetricSequentialDMSInfeasibleError as error:
                terminal = _terminal_record(
                    split=split,
                    scenario_id=scenario_id,
                    status="failed",
                    outcome_class="scientific_no_success",
                    state_metadata=current_metadata,
                    reason=f"common-ascent was certified infeasible: {type(error).__name__}: {error}",
                )
                break
            except (
                SymmetricSequentialDMSSolverError,
                SymmetricSequentialDMSCertificateError,
            ) as error:
                terminal = _terminal_record(
                    split=split,
                    scenario_id=scenario_id,
                    status="failed",
                    outcome_class="technical_integrity_failure",
                    state_metadata=current_metadata,
                    reason=f"common-ascent/lookahead integrity failed: {type(error).__name__}: {error}",
                )
                break
            attempts: list[dict[str, Any]] = []
            advanced = False
            for progress in available:
                problem = _controller_problem(
                    state_metadata=current_metadata,
                    state_tensors=current_tensors,
                    contexts=contexts,
                    residual_scale=scales[scenario_id],
                    standardized_nuisance_rows=nuisance_rows[scenario_id],
                    progress=progress,
                )
                try:
                    candidate, attempt = _solve_ckes_for_progress(
                        problem=problem,
                        lookahead_metadata=lookahead_metadata,
                        lookahead_tensors=lookahead_tensors,
                        cumulative_path_l2=float(current_metadata["cumulative_path_l2"]),
                    )
                    attempts.append(attempt)
                except SymmetricSequentialDMSInfeasibleError as error:
                    attempts.append(
                        {
                            "progress_fraction": progress,
                            "status": "certified_infeasible_try_lower_progress",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    continue
                except (
                    SymmetricSequentialDMSSolverError,
                    SymmetricSequentialDMSCertificateError,
                ) as error:
                    terminal = _terminal_record(
                        split=split,
                        scenario_id=scenario_id,
                        status="failed",
                        outcome_class="technical_integrity_failure",
                        state_metadata=current_metadata,
                        reason=f"CKES solver/certificate failed closed: {type(error).__name__}: {error}",
                        extra={"attempts": attempts},
                    )
                    break
                if len(states) - 1 >= MAX_TRIAL_STATES:
                    terminal = _terminal_record(
                        split=split,
                        scenario_id=scenario_id,
                        status="failed",
                        outcome_class="scientific_no_success",
                        state_metadata=current_metadata,
                        reason="locked maximum candidate-state count reached",
                    )
                    break
                trial_index = len(states)
                try:
                    state = _capture_candidate_state(
                        torch,
                        split=split,
                        backend_getter=backend_getter,
                        scenario_id=scenario_id,
                        contexts=contexts,
                        previous_metadata=current_metadata,
                        previous_tensors=current_tensors,
                        candidate=candidate,
                        progress=progress,
                        attempts=attempts,
                        trial_index=trial_index,
                        ledger=ledger,
                    )
                except CandidateRuntimeFailure as error:
                    terminal = _terminal_record(
                        split=split,
                        scenario_id=scenario_id,
                        status="failed",
                        outcome_class="technical_integrity_failure",
                        state_metadata=current_metadata,
                        reason=f"candidate runtime failed after full-batch reservation: {error}",
                    )
                    break
                states.append(state)
                states_by_scenario[scenario_id] = states
                state_metadata, _state_tensors = state
                if state_metadata["accepted"] is not True:
                    continue
                advanced = True
                current_metadata, current_tensors = state
                if current_metadata["stopping_gate_passes"] is True:
                    terminal = _terminal_record(
                        split=split,
                        scenario_id=scenario_id,
                        status="success",
                        outcome_class="success",
                        state_metadata=current_metadata,
                        reason=(
                            "both signs, assignments, answer orders, decision changes, and "
                            "per-state safety gates passed"
                        ),
                    )
                break
            if terminal is not None:
                break
            if advanced:
                continue
            terminal = _terminal_record(
                split=split,
                scenario_id=scenario_id,
                status="failed",
                outcome_class="scientific_no_success",
                state_metadata=current_metadata,
                reason="no locked progress fraction produced an accepted CKES state",
                extra={"attempts": attempts},
            )
            break
        if terminal is None:
            current_metadata, _ = _current_accepted(states)
            terminal = _terminal_record(
                split=split,
                scenario_id=scenario_id,
                status="failed",
                outcome_class="scientific_no_success",
                state_metadata=current_metadata,
                reason="locked maximum of 24 deployed trial states was reached",
            )
        terminals[scenario_id] = terminal
    _validate_terminal_state_coverage(
        scenario_contexts=scenario_contexts,
        terminals=terminals,
        states_by_scenario=states_by_scenario,
    )
    final = _run_or_load_final(
        torch,
        split=split,
        backend_getter=backend_getter,
        scenario_contexts=scenario_contexts,
        terminals=terminals,
        states_by_scenario=states_by_scenario,
        ledger=ledger,
    )
    states_by_scenario = {
        scenario_id: _load_states(
            torch,
            split=split,
            scenario_id=scenario_id,
            contexts=scenario_contexts[scenario_id],
            residual_scale=scales[scenario_id],
            baseline_checkpoint_sha256=baseline_metadata["checkpoint_sha256"],
            lock_identity_sha256=lock["lock_identity_sha256"],
            ledger=ledger,
        )
        for scenario_id in scenario_contexts
    }
    result = _build_result(
        split=split,
        baseline_metadata=baseline_metadata,
        qualification=qualification,
        terminals=terminals,
        scenario_contexts=scenario_contexts,
        states_by_scenario=states_by_scenario,
        final=final,
        ledger=ledger,
    )
    if existing_result is not None:
        if result != existing_result:
            raise RuntimeError("cached CKES result differs from reconstructed artifacts")
        _write_report(existing_result)
        return existing_result
    _write_new_json(paths["result"], result)
    _write_report(result)
    return result


def run_report(split: str = "validation") -> str:
    return _write_report(_load_result(split))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("lock", "preflight", "run", "report"),
        nargs="?",
        default="preflight",
    )
    parser.add_argument("--split", choices=("validation", "sealed"), default="validation")
    args = parser.parse_args()
    if args.command == "lock":
        print(json.dumps(run_lock(), indent=2))
    elif args.command == "preflight":
        print(json.dumps(run_preflight(args.split), indent=2))
    elif args.command == "run":
        print(json.dumps(run_development(args.split), indent=2))
    else:
        print(run_report(args.split))


if __name__ == "__main__":
    main()
