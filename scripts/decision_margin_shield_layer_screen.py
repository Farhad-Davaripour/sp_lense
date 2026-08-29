from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sp_lense.backend import ResearchBackend
from sp_lense.causal_anchor_runtime import (
    anchor_residual_scale_geometric_mean,
    capture_multilayer_choice_anchor_gradient,
    resolve_shared_anchor_evidence,
)
from sp_lense.comparison_runtime import qwen35_choice_boundary_tokenizer_smoke
from sp_lense.config import load_config
from sp_lense.decision_margin_shield import (
    DEFAULT_CAP_FRONTIER,
    DEFAULT_MARGIN,
    DEFAULT_QUALIFICATION_CAP,
    METHODS,
    screen_scenario_layer,
    select_layer,
)
from sp_lense.factorial_causal_anchor import (
    canonical_sha256,
    render_choice_form,
    render_construction_form,
    render_unrelated_ab_form,
    render_unrelated_construction_form,
    tensor_float32_sha256,
    text_sha256,
    validate_pilot_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
PROTOCOL_PATH = ROOT / "docs" / "DECISION_MARGIN_SHIELD_LAYER_SCREEN_PROTOCOL.md"
MATH_PATH = ROOT / "src" / "sp_lense" / "decision_margin_shield.py"
MATH_TEST_PATH = ROOT / "tests" / "test_decision_margin_shield.py"
RUNNER_TEST_PATH = ROOT / "tests" / "test_decision_margin_shield_layer_screen_runner.py"
DATA_PATH = ROOT / "data" / "factorial_causal_anchor_gradient_pilot.json"
MODEL_CONFIG_PATH = ROOT / "configs" / "qwen35_08b_aligned.json"
LOCK_PATH = ROOT / "configs" / "decision_margin_shield_layer_screen_lock.json"
CTS_MATH_PATH = ROOT / "src" / "sp_lense" / "counterfactual_tangent_shield.py"
ANCHOR_RUNTIME_PATH = ROOT / "src" / "sp_lense" / "causal_anchor_runtime.py"
FACTORIAL_MATH_PATH = ROOT / "src" / "sp_lense" / "factorial_causal_anchor.py"
BACKEND_PATH = ROOT / "src" / "sp_lense" / "backend.py"
CONFIG_PATH = ROOT / "src" / "sp_lense" / "config.py"
CORE_PATH = ROOT / "src" / "sp_lense" / "core.py"
COMPARISON_RUNTIME_PATH = ROOT / "src" / "sp_lense" / "comparison_runtime.py"
COMPARISON_INTERVENTION_PATH = ROOT / "src" / "sp_lense" / "comparison_intervention.py"
SEMANTIC_COMPLETION_PATH = ROOT / "src" / "sp_lense" / "semantic_completion_gradient.py"
STEERING_METHODS_PATH = ROOT / "src" / "sp_lense" / "steering_methods.py"
GRADIENT_V3_PATH = ROOT / "src" / "sp_lense" / "gradient_specificity_v3.py"
TRUST_REGION_PATH = ROOT / "src" / "sp_lense" / "gradient_specificity_trust_region.py"
COUNTERFACTUAL_PROTECTED_PATH = (
    ROOT / "src" / "sp_lense" / "counterfactual_protected_natural_gradient.py"
)
BASE_REQUIREMENTS_PATH = ROOT / "requirements-research.txt"
CTS_REQUIREMENTS_PATH = ROOT / "requirements-counterfactual-tangent-shield.txt"

ARTIFACT_ROOT = ROOT / "artifacts" / "decision_margin_shield_layer_screen" / "qwen35_08b"
RESULT_ROOT = ROOT / "results" / "decision_margin_shield_layer_screen" / "qwen35_08b"
PREFLIGHT_PATH = ARTIFACT_ROOT / "preflight.json"
CAPTURE_ROOT = ARTIFACT_ROOT / "capture_chunks"
CAPTURE_LEDGER_PATH = ARTIFACT_ROOT / "capture_attempt_ledger.json"
CAPTURE_MANIFEST_PATH = ARTIFACT_ROOT / "capture_manifest.json"
SCREEN_RESULT_PATH = RESULT_ROOT / "layer_screen_result.json"
REPORT_PATH = RESULT_ROOT / "LAYER_SCREEN_REPORT.md"

LOCK_SCHEMA = "sp_lense.decision_margin_shield_layer_screen_lock.v2"
PREFLIGHT_SCHEMA = "sp_lense.decision_margin_shield_layer_screen_preflight.v2"
LEDGER_SCHEMA = "sp_lense.decision_margin_shield_layer_screen_ledger.v2"
CHUNK_SCHEMA = "sp_lense.decision_margin_shield_layer_screen_chunk.v2"
CAPTURE_SCHEMA = "sp_lense.decision_margin_shield_layer_screen_capture.v2"
RESULT_SCHEMA = "sp_lense.decision_margin_shield_layer_screen_result.v2"

MODEL = {
    "id": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "device": "cpu",
    "dtype": "float32",
    "n_layers": 24,
    "d_model": 1024,
}
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
CHAT_TEMPLATE_SHA256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
LAYERS = tuple(range(23))
ENCODINGS = (("A", "B"), ("X", "Y"), ("1", "2"))
MARGIN = DEFAULT_MARGIN
CAP_FRONTIER = DEFAULT_CAP_FRONTIER
QUALIFICATION_CAP = DEFAULT_QUALIFICATION_CAP
CAPTURE_CEILING = {"forward": 136, "backward": 136}
CAPTURE_CHUNK_SIZE = 8


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(dict(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    _atomic_text(path, rendered)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace immutable artifact: {path}")
    _write_json(path, value)


def _with_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = canonical_sha256(result)
    return result


def _verify_hash(value: Mapping[str, Any], field: str) -> None:
    observed = value.get(field)
    if not isinstance(observed, str):
        raise TypeError(f"artifact lacks {field}")
    unhashed = dict(value)
    del unhashed[field]
    if canonical_sha256(unhashed) != observed:
        raise RuntimeError(f"artifact {field} self-check failed")


def _runtime(torch: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": importlib.metadata.version("transformers"),
        "transformer_lens": importlib.metadata.version("transformer-lens"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "safetensors": importlib.metadata.version("safetensors"),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "torch_intraop_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
    }


def _load_dataset() -> dict[str, Any]:
    payload = _load_json(DATA_PATH)
    validate_pilot_dataset(payload)
    return payload


def _source_paths() -> dict[str, Path]:
    return {
        "data": DATA_PATH,
        "model_config": MODEL_CONFIG_PATH,
        "protocol": PROTOCOL_PATH,
        "dms_math": MATH_PATH,
        "cts_solver": CTS_MATH_PATH,
        "anchor_runtime": ANCHOR_RUNTIME_PATH,
        "factorial_math": FACTORIAL_MATH_PATH,
        "backend": BACKEND_PATH,
        "config": CONFIG_PATH,
        "core": CORE_PATH,
        "comparison_runtime": COMPARISON_RUNTIME_PATH,
        "comparison_intervention": COMPARISON_INTERVENTION_PATH,
        "semantic_completion_gradient": SEMANTIC_COMPLETION_PATH,
        "steering_methods": STEERING_METHODS_PATH,
        "gradient_specificity_v3": GRADIENT_V3_PATH,
        "gradient_specificity_trust_region": TRUST_REGION_PATH,
        "counterfactual_protected_natural_gradient": COUNTERFACTUAL_PROTECTED_PATH,
        "runner": SCRIPT_PATH,
        "math_tests": MATH_TEST_PATH,
        "runner_tests": RUNNER_TEST_PATH,
        "base_requirements": BASE_REQUIREMENTS_PATH,
        "cts_requirements": CTS_REQUIREMENTS_PATH,
    }


def _capture_specifications(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    specifications: list[dict[str, Any]] = []
    for scenario in dataset["scenarios"]:
        scenario_id = str(scenario["id"])
        for assignment in (0, 1):
            for target in ("self", "other"):
                for event in ("permanent", "temporary"):
                    construction = render_construction_form(
                        dataset,
                        scenario,
                        assignment=assignment,
                        target=target,
                        event=event,
                    )
                    evidence_prompts = [str(construction["prompt"])]
                    for labels in ENCODINGS:
                        for preserve_first in (True, False):
                            evidence_prompts.append(
                                str(
                                    render_choice_form(
                                        dataset,
                                        scenario,
                                        assignment=assignment,
                                        target=target,
                                        event=event,
                                        preserve_first=preserve_first,
                                        labels=labels,
                                    )["prompt"]
                                )
                            )
                    for preserve_first in (True, False):
                        form = render_choice_form(
                            dataset,
                            scenario,
                            assignment=assignment,
                            target=target,
                            event=event,
                            preserve_first=preserve_first,
                            labels=("A", "B"),
                        )
                        specifications.append(
                            {
                                "work_id": (
                                    f"scenario:{scenario_id}:assignment={assignment}:"
                                    f"target={target}:event={event}:"
                                    f"preserve_first={str(preserve_first).lower()}"
                                ),
                                "kind": "scenario",
                                "scenario_id": scenario_id,
                                "partition": str(scenario["partition"]),
                                "assignment": assignment,
                                "target": target,
                                "event": event,
                                "preserve_first": preserve_first,
                                "form_id": str(form["form_id"]),
                                "prompt": str(form["prompt"]),
                                "preserve_label": str(form["preserve_label"]),
                                "comply_label": str(form["comply_label"]),
                                "anchor_prefix": str(construction["anchor_prefix"]),
                                "evidence_prompts": evidence_prompts,
                            }
                        )
    nuisance_controls = [
        control
        for control in dataset["unrelated_controls"]
        if control["partition"] == "nuisance_fit"
    ]
    for control in nuisance_controls:
        construction = render_unrelated_construction_form(dataset, control)
        forms = [
            render_unrelated_ab_form(dataset, control, preferred_first=preferred_first)
            for preferred_first in (True, False)
        ]
        for form in forms:
            preferred_first = bool(form["preferred_first"])
            specifications.append(
                {
                    "work_id": (
                        f"nuisance:{control['id']}:"
                        f"preferred_first={str(preferred_first).lower()}"
                    ),
                    "kind": "nuisance_fit",
                    "control_id": str(control["id"]),
                    "partition": "nuisance_fit",
                    "preferred_first": preferred_first,
                    "form_id": str(form["form_id"]),
                    "prompt": str(form["prompt"]),
                    "preserve_label": str(form["preferred_label"]),
                    "comply_label": str(form["alternative_label"]),
                    "anchor_prefix": str(construction["anchor_prefix"]),
                    "evidence_prompts": [
                        str(construction["prompt"]),
                        *[str(item["prompt"]) for item in forms],
                    ],
                }
            )
    work_ids = [str(item["work_id"]) for item in specifications]
    if len(specifications) != 136 or len(set(work_ids)) != 136:
        raise RuntimeError("DMS capture plan must contain exactly 136 unique A/B forms")
    if sum(item["kind"] == "scenario" for item in specifications) != 128:
        raise RuntimeError("DMS capture plan must contain exactly 128 scenario forms")
    if sum(item["kind"] == "nuisance_fit" for item in specifications) != 8:
        raise RuntimeError("DMS capture plan must contain exactly eight unrelated forms")
    return specifications


def _public_specification(specification: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"prompt", "anchor_prefix", "evidence_prompts"}
    public = {key: value for key, value in specification.items() if key not in excluded}
    public.update(
        {
            "prompt_sha256": text_sha256(str(specification["prompt"])),
            "anchor_prefix_sha256": text_sha256(str(specification["anchor_prefix"])),
            "evidence_prompt_sha256s": [
                text_sha256(str(prompt)) for prompt in specification["evidence_prompts"]
            ],
        }
    )
    return public


def _capture_plan_sha256(specifications: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256([_public_specification(item) for item in specifications])


def _prompt_content_sha256(specifications: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        [
            {
                "work_id": item["work_id"],
                "prompt_sha256": text_sha256(str(item["prompt"])),
                "anchor_prefix_sha256": text_sha256(str(item["anchor_prefix"])),
                "evidence_prompt_sha256s": [
                    text_sha256(str(prompt)) for prompt in item["evidence_prompts"]
                ],
            }
            for item in specifications
        ]
    )


def proposed_lock() -> dict[str, Any]:
    sources = _source_paths()
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"locked source files are missing: {missing}")
    dataset = _load_dataset()
    specifications = _capture_specifications(dataset)
    calibration_ids = [
        str(scenario["id"])
        for scenario in dataset["scenarios"]
        if scenario["partition"] == "calibration"
    ]
    pilot_ids = [
        str(scenario["id"])
        for scenario in dataset["scenarios"]
        if scenario["partition"] == "pilot"
    ]
    payload = {
        "schema_version": LOCK_SCHEMA,
        "status": "opened_development_locked_before_capture",
        "development_only": True,
        "model": MODEL,
        "model_identity_sha256": canonical_sha256(MODEL),
        "runtime": EXPECTED_RUNTIME,
        "runtime_identity_sha256": canonical_sha256(EXPECTED_RUNTIME),
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "dataset": {
            "path": _relative(DATA_PATH),
            "sha256": file_sha256(DATA_PATH),
            "calibration_scenario_ids": calibration_ids,
            "pilot_scenario_ids_captured_but_not_screened": pilot_ids,
            "selection_partition": "calibration_only",
            "nuisance_fit_partition": "nuisance_fit",
        },
        "capture": {
            "layers": list(LAYERS),
            "excluded_endpoint_layer": 23,
            "position": "last_token_of_exact_shared_causal_decision_anchor",
            "objective": "preserve_minus_comply_next_token_ab_log_odds",
            "record_count": 136,
            "scenario_record_count": 128,
            "nuisance_fit_record_count": 8,
            "one_forward_and_one_backward_per_record": True,
            "capture_plan_sha256": _capture_plan_sha256(specifications),
            "prompt_content_sha256": _prompt_content_sha256(specifications),
        },
        "geometry": {
            "methods": list(METHODS),
            "residual_coordinate": "scenario_layer_geometric_mean_relative_l2",
            "target_cell": "self_permanent",
            "target_gradient_count_per_scenario": 4,
            "matched_protected_gradient_count_per_scenario": 12,
            "exact_unrelated_gradient_count": 8,
            "target_margin": MARGIN,
            "protected_bound": "max(abs(unsteered_ab_log_odds)-0.05,0)_per_row",
            "small_baseline_rule": (
                "abs(b)<0.05_is_first_order_frozen_but_not_margin_certified;"
                "abs(b)=0.05_is_margin_certified"
            ),
            "solver": "uncapped_minimum_euclidean_l2_float64_cpu",
            "cap_frontier": list(CAP_FRONTIER),
            "qualification_cap": QUALIFICATION_CAP,
            "qualifies": "all_four_calibration_scenario_dms_norms_lte_2",
            "tie_breakers": [
                "smallest_worst_case_dms_norm",
                "smallest_mean_dms_norm",
                "smallest_zero_based_layer",
            ],
            "finite_intervention_outcomes_inspected": False,
            "pilot_geometry_computed": False,
        },
        "artifact_policy": {
            "capture_chunk_size": CAPTURE_CHUNK_SIZE,
            "pending_chunk_is_ambiguous_and_not_replayed": True,
            "completed_chunks_must_form_exact_plan_prefix": True,
            "sealed_project_paths_read": [],
            "generated_tokens": 0,
            "external_model_judges": 0,
            "external_api_calls": 0,
            "paid_model_cost_usd": 0,
        },
        "compute_ceiling": {
            "capture": CAPTURE_CEILING,
            "geometry": {"forward": 0, "backward": 0},
            "finite_intervention": {"forward": 0, "backward": 0},
            "generated_tokens": 0,
        },
        "source_files": {
            name: {"path": _relative(path), "sha256": file_sha256(path)}
            for name, path in sources.items()
        },
        "claim_boundary": (
            "Opened local-linear geometry only; no natural mechanism, safety proof, "
            "finite steering effect, unchanged capability, priority, or publication claim."
        ),
    }
    payload["lock_identity_sha256"] = canonical_sha256(payload)
    return payload


def run_lock() -> dict[str, Any]:
    if LOCK_PATH.exists():
        raise FileExistsError(f"refusing to replace existing lock: {LOCK_PATH}")
    lock = proposed_lock()
    _write_new_json(LOCK_PATH, lock)
    return lock


def _load_lock() -> dict[str, Any]:
    lock = _load_json(LOCK_PATH)
    expected = proposed_lock()
    if lock != expected:
        raise RuntimeError("DMS layer-screen lock differs from the hash-bound design")
    return lock


def _configure_threads(torch: Any) -> None:
    torch.set_num_threads(12)
    try:
        torch.set_num_interop_threads(12)
    except RuntimeError:
        if torch.get_num_interop_threads() != 12:
            raise


def run_preflight() -> dict[str, Any]:
    lock = _load_lock()
    dataset = _load_dataset()
    import torch

    _configure_threads(torch)
    observed_runtime = _runtime(torch)
    if observed_runtime != EXPECTED_RUNTIME:
        raise RuntimeError(f"installed runtime differs from the DMS lock: {observed_runtime}")
    calibration_count = sum(
        scenario["partition"] == "calibration" for scenario in dataset["scenarios"]
    )
    pilot_count = sum(scenario["partition"] == "pilot" for scenario in dataset["scenarios"])
    nuisance_count = sum(
        control["partition"] == "nuisance_fit" for control in dataset["unrelated_controls"]
    )
    if (calibration_count, pilot_count, nuisance_count) != (4, 4, 4):
        raise RuntimeError("DMS dataset coverage differs from the locked design")
    result = _with_hash(
        {
            "schema_version": PREFLIGHT_SCHEMA,
            "status": "ready",
            "development_only": True,
            "lock_file_sha256": file_sha256(LOCK_PATH),
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "model_identity_sha256": lock["model_identity_sha256"],
            "runtime_identity_sha256": lock["runtime_identity_sha256"],
            "data_sha256": file_sha256(DATA_PATH),
            "model_config_sha256": file_sha256(MODEL_CONFIG_PATH),
            "prompt_content_sha256": lock["capture"]["prompt_content_sha256"],
            "runtime": observed_runtime,
            "calibration_scenario_count": calibration_count,
            "pilot_scenario_count_captured_not_screened": pilot_count,
            "nuisance_fit_control_count": nuisance_count,
            "capture_ceiling": CAPTURE_CEILING,
            "model_loads": 0,
            "model_forwards": 0,
            "model_backwards": 0,
            "generated_tokens": 0,
            "sealed_project_file_imports": [],
        },
        "preflight_sha256",
    )
    _write_json(PREFLIGHT_PATH, result)
    return result


def load_backend() -> Any:
    import torch

    _configure_threads(torch)
    backend = ResearchBackend.load(load_config(MODEL_CONFIG_PATH), with_lens=False)
    metadata = backend.metadata()
    observed = {
        "id": metadata["model_id"],
        "revision": metadata["model_revision"],
        "device": metadata["device"],
        "dtype": metadata["dtype"],
        "n_layers": metadata["model_layers"],
        "d_model": metadata["d_model"],
    }
    if observed != MODEL:
        raise RuntimeError(f"resident backend differs from the DMS lock: {observed}")
    smoke = qwen35_choice_boundary_tokenizer_smoke(backend.model.tokenizer, backend.torch)
    if smoke["chat_template_sha256"] != CHAT_TEMPLATE_SHA256:
        raise RuntimeError("resident chat template differs from the DMS lock")
    if _runtime(backend.torch) != EXPECTED_RUNTIME:
        raise RuntimeError(f"resident runtime differs from the DMS lock: {_runtime(backend.torch)}")
    return backend


def _chunked(values: Sequence[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


class PersistentChunkLedger:
    def __init__(
        self,
        *,
        path: Path,
        plan_sha256: str,
        lock_identity_sha256: str,
        expected_chunk_work_ids: Sequence[Sequence[str]],
        ceiling: Mapping[str, int],
    ) -> None:
        self.path = path
        self.plan_sha256 = plan_sha256
        if not isinstance(lock_identity_sha256, str) or not lock_identity_sha256:
            raise ValueError("lock identity must be a non-empty SHA-256 string")
        self.lock_identity_sha256 = lock_identity_sha256
        self.expected_chunk_work_ids = [list(map(str, chunk)) for chunk in expected_chunk_work_ids]
        if not self.expected_chunk_work_ids or any(
            not chunk for chunk in self.expected_chunk_work_ids
        ):
            raise ValueError("the frozen ledger plan must contain non-empty chunks")
        self.ceiling = {"forward": int(ceiling["forward"]), "backward": int(ceiling["backward"])}
        if path.exists():
            self.payload = _load_json(path)
            _verify_hash(self.payload, "ledger_sha256")
            if (
                self.payload.get("schema_version") != LEDGER_SCHEMA
                or self.payload.get("phase") != "capture"
                or self.payload.get("plan_sha256") != plan_sha256
                or self.payload.get("lock_identity_sha256") != lock_identity_sha256
                or self.payload.get("ceiling") != self.ceiling
            ):
                raise RuntimeError("DMS capture ledger identity differs")
        else:
            self.payload = {
                "schema_version": LEDGER_SCHEMA,
                "phase": "capture",
                "plan_sha256": plan_sha256,
                "lock_identity_sha256": lock_identity_sha256,
                "ceiling": self.ceiling,
                "events": [],
            }
            self._persist()
        self._validate()

    def _persist(self) -> None:
        self.payload = _with_hash(
            {key: value for key, value in self.payload.items() if key != "ledger_sha256"},
            "ledger_sha256",
        )
        _write_json(self.path, self.payload)

    def _validate(self) -> None:
        events = self.payload.get("events")
        if not isinstance(events, list):
            raise TypeError("ledger events must be a list")
        prior = None
        observed_ids: set[str] = set()
        for index, event in enumerate(events):
            if index >= len(self.expected_chunk_work_ids):
                raise RuntimeError("DMS capture ledger is longer than the frozen plan")
            if not isinstance(event, Mapping) or int(event.get("chunk_index", -1)) != index:
                raise RuntimeError("ledger events are not a contiguous chunk prefix")
            work_ids = event.get("work_ids")
            if not isinstance(work_ids, list) or not work_ids:
                raise RuntimeError("ledger event has invalid work IDs")
            if work_ids != self.expected_chunk_work_ids[index]:
                raise RuntimeError("ledger work IDs differ from the frozen plan prefix")
            if any(not isinstance(item, str) or item in observed_ids for item in work_ids):
                raise RuntimeError("ledger contains invalid or duplicate work IDs")
            observed_ids.update(work_ids)
            if event.get("prior_event_sha256") != prior:
                raise RuntimeError("ledger event chain differs")
            unhashed = dict(event)
            observed_hash = unhashed.pop("event_sha256", None)
            if canonical_sha256(unhashed) != observed_hash:
                raise RuntimeError("ledger event hash differs")
            prior = observed_hash
            if event.get("status") not in {"pending", "complete"}:
                raise RuntimeError("ledger event status is invalid")
            if event.get("status") == "pending" and index != len(events) - 1:
                raise RuntimeError("pending ledger event is not terminal")
        forward, backward = self.counts()
        if forward > self.ceiling["forward"] or backward > self.ceiling["backward"]:
            raise RuntimeError("DMS capture ledger exceeds its compute ceiling")

    def counts(self) -> tuple[int, int]:
        events = self.payload.get("events", [])
        return (
            sum(int(event["forward_evaluations"]) for event in events),
            sum(int(event["backward_evaluations"]) for event in events),
        )

    def completed_chunks(self) -> int:
        events = self.payload.get("events", [])
        if events and events[-1]["status"] == "pending":
            raise RuntimeError("DMS capture has an ambiguous pending chunk; it cannot be replayed")
        return len(events)

    def reserve(self, *, chunk_index: int, work_ids: Sequence[str]) -> None:
        if self.completed_chunks() != chunk_index:
            raise RuntimeError("ledger chunk reservation is out of order")
        if chunk_index >= len(self.expected_chunk_work_ids) or list(work_ids) != (
            self.expected_chunk_work_ids[chunk_index]
        ):
            raise RuntimeError("ledger reservation differs from the frozen chunk plan")
        prior = self.payload["events"][-1]["event_sha256"] if self.payload["events"] else None
        event = {
            "chunk_index": chunk_index,
            "work_ids": list(work_ids),
            "forward_evaluations": len(work_ids),
            "backward_evaluations": len(work_ids),
            "status": "pending",
            "artifact_path": None,
            "artifact_sha256": None,
            "prior_event_sha256": prior,
        }
        event["event_sha256"] = canonical_sha256(event)
        self.payload["events"].append(event)
        self._validate()
        self._persist()

    def complete(self, *, chunk_index: int, artifact_path: Path) -> None:
        events = self.payload["events"]
        if chunk_index != len(events) - 1 or events[-1]["status"] != "pending":
            raise RuntimeError("ledger has no matching pending chunk")
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

    def snapshot(self) -> dict[str, Any]:
        forward, backward = self.counts()
        work_ids = [work_id for event in self.payload["events"] for work_id in event["work_ids"]]
        return {
            "forward_evaluations": forward,
            "backward_evaluations": backward,
            "unique_work_id_count": len(set(work_ids)),
            "work_ids_sha256": canonical_sha256(work_ids),
            "completed_chunk_count": self.completed_chunks(),
            "ledger_file_sha256": file_sha256(self.path),
            "ledger_sha256": self.payload["ledger_sha256"],
        }


def _capture_chunk_path(index: int) -> Path:
    return CAPTURE_ROOT / f"chunk-{index:03d}.pt"


def _save_tensor_chunk(
    torch: Any,
    *,
    path: Path,
    chunk_index: int,
    plan_sha256: str,
    lock_identity_sha256: str,
    records: Sequence[Mapping[str, Any]],
    gradients: Any,
    residuals: Any,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace completed chunk: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {"gradients": gradients, "anchor_residuals": residuals}
    public = _with_hash(
        {
            "schema_version": CHUNK_SCHEMA,
            "phase": "capture",
            "chunk_index": chunk_index,
            "plan_sha256": plan_sha256,
            "lock_identity_sha256": lock_identity_sha256,
            "record_count": len(records),
            "records": [dict(record) for record in records],
            "tensor_hashes": {
                name: tensor_float32_sha256(value) for name, value in sorted(tensors.items())
            },
        },
        "chunk_identity_sha256",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({**public, "tensors": tensors}, temporary)
    os.replace(temporary, path)


_CAPTURE_RUNTIME_RECORD_FIELDS = {
    "row_index",
    "anchor_index",
    "anchor_evidence",
    "capture_audit",
    "preserve_minus_comply_baseline_log_odds",
    "gradient_float32_sha256",
    "anchor_residual_float32_sha256",
}


def _require_nonempty_sha256_evidence(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"DMS capture audit lacks {field}")
    return value


def _validate_runtime_record(
    record: Mapping[str, Any],
    expected_specification: Mapping[str, Any],
    *,
    expected_row_index: int,
    expected_anchor_marker: str,
    gradient: Any,
    residual: Any,
) -> None:
    expected_public = _public_specification(expected_specification)
    if set(record) != set(expected_public) | _CAPTURE_RUNTIME_RECORD_FIELDS:
        raise RuntimeError("DMS capture record has undocumented or missing fields")
    observed_public = {key: record[key] for key in expected_public}
    if observed_public != expected_public:
        raise RuntimeError("DMS capture record differs from the frozen specification")
    if type(record.get("row_index")) is not int or record["row_index"] != expected_row_index:
        raise RuntimeError("DMS capture row indices differ from the frozen row order")
    if type(record.get("anchor_index")) is not int:
        raise RuntimeError("DMS capture record has an invalid anchor index")
    anchor_index = record["anchor_index"]
    if anchor_index < 0:
        raise RuntimeError("DMS capture record has an invalid anchor index")
    gradient_hash = tensor_float32_sha256(gradient)
    residual_hash = tensor_float32_sha256(residual)
    if (
        record.get("gradient_float32_sha256") != gradient_hash
        or record.get("anchor_residual_float32_sha256") != residual_hash
    ):
        raise RuntimeError("DMS capture row tensor hash differs")
    baseline = float(record.get("preserve_minus_comply_baseline_log_odds", math.nan))
    if not math.isfinite(baseline):
        raise RuntimeError("DMS capture record has a non-finite baseline log-odds")

    audit = record.get("capture_audit")
    evidence = record.get("anchor_evidence")
    if not isinstance(audit, Mapping) or not isinstance(evidence, Mapping):
        raise TypeError("DMS capture row lacks audit evidence")
    _verify_hash(audit, "audit_sha256")
    _verify_hash(evidence, "audit_sha256")

    expected_hook_counts = {str(layer): 1 for layer in LAYERS}
    if (
        audit.get("schema_version") != "sp_lense.multilayer_choice_anchor_capture.v1"
        or audit.get("objective")
        != "preserve_label_minus_comply_label_next_token_logit"
        or audit.get("preserve_label") != record["preserve_label"]
        or audit.get("comply_label") != record["comply_label"]
        or float(audit.get("preserve_log_odds", math.nan)) != baseline
        or int(audit.get("anchor_index", -1)) != anchor_index
        or audit.get("gradient_position") != "shared_pre_encoding_causal_anchor"
        or audit.get("layers") != list(LAYERS)
        or audit.get("hook_call_counts") != expected_hook_counts
        or audit.get("raw_gradients_float32_sha256") != gradient_hash
        or audit.get("anchor_residuals_float32_sha256") != residual_hash
        or audit.get("model_parameter_gradients_allocated") is not False
    ):
        raise RuntimeError("DMS capture row differs from its runtime audit")
    prompt_token_hash = _require_nonempty_sha256_evidence(
        audit.get("prompt_token_ids_sha256"), field="prompt token evidence"
    )
    _require_nonempty_sha256_evidence(
        audit.get("choice_boundary_evidence_sha256"), field="choice-boundary evidence"
    )

    evidence_token_hashes = evidence.get("prompt_token_sha256s")
    expected_evidence_text_hashes = record["evidence_prompt_sha256s"]
    if (
        evidence.get("schema_version") != "sp_lense.shared_causal_anchor_evidence.v1"
        or evidence.get("anchor_position")
        != "last_token_of_longest_shared_prompt_prefix"
        or int(evidence.get("anchor_index", -1)) != anchor_index
        or int(evidence.get("shared_prefix_length", -1)) != anchor_index + 1
        or evidence.get("anchor_prefix_text_sha256") != record["anchor_prefix_sha256"]
        or evidence.get("anchor_marker") != expected_anchor_marker
        or evidence.get("anchor_marker_present_in_decoded_shared_prefix") is not True
        or evidence.get("future_suffix_cannot_change_anchor_by_causal_mask") is not True
        or not isinstance(evidence_token_hashes, list)
        or len(evidence_token_hashes) != len(expected_evidence_text_hashes)
        or int(evidence.get("prompt_count", -1)) != len(expected_evidence_text_hashes)
        or prompt_token_hash not in evidence_token_hashes
        or record["prompt_sha256"] not in expected_evidence_text_hashes
    ):
        raise RuntimeError("DMS capture anchor evidence differs from the frozen record")
    _require_nonempty_sha256_evidence(
        evidence.get("shared_token_prefix_sha256"), field="shared anchor-token evidence"
    )


def _validate_chunk_against_plan(
    payload: Mapping[str, Any],
    expected_specifications: Sequence[Mapping[str, Any]],
) -> None:
    expected_count = len(expected_specifications)
    if payload.get("record_count") != expected_count:
        raise RuntimeError("DMS capture chunk record count differs from the frozen plan")
    records = payload.get("records")
    tensors = payload.get("tensors")
    if not isinstance(records, list) or len(records) != expected_count:
        raise RuntimeError("DMS capture chunk record coverage differs from the frozen plan")
    if not isinstance(tensors, Mapping) or set(tensors) != {"gradients", "anchor_residuals"}:
        raise RuntimeError("DMS capture chunk has an unexpected tensor collection")
    gradients = tensors["gradients"]
    residuals = tensors["anchor_residuals"]
    expected_shape = (expected_count, len(LAYERS), MODEL["d_model"])
    if tuple(gradients.shape) != expected_shape or tuple(residuals.shape) != expected_shape:
        raise RuntimeError("DMS capture chunk tensors differ from exact [n,23,1024] shape")
    if str(gradients.dtype) != "torch.float32" or str(residuals.dtype) != "torch.float32":
        raise RuntimeError("DMS capture chunk tensors are not exact float32")
    if not bool(gradients.isfinite().all().item()) or not bool(
        residuals.isfinite().all().item()
    ):
        raise RuntimeError("DMS capture chunk contains non-finite tensors")
    row_indices = [record.get("row_index") for record in records]
    if row_indices != list(range(expected_count)):
        raise RuntimeError("DMS capture chunk row indices are not exactly range(n)")
    expected_anchor_marker = str(_load_dataset()["anchor_marker"])
    for row_index, (record, expected) in enumerate(
        zip(records, expected_specifications, strict=True)
    ):
        if not isinstance(record, Mapping):
            raise TypeError("DMS capture record must be a mapping")
        _validate_runtime_record(
            record,
            expected,
            expected_row_index=row_index,
            expected_anchor_marker=expected_anchor_marker,
            gradient=gradients[row_index].float().contiguous(),
            residual=residuals[row_index].float().contiguous(),
        )


def _load_tensor_chunk(
    torch: Any,
    *,
    path: Path,
    chunk_index: int,
    plan_sha256: str,
    lock_identity_sha256: str,
    expected_specifications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("capture chunk must contain a mapping")
    if (
        payload.get("schema_version") != CHUNK_SCHEMA
        or payload.get("phase") != "capture"
        or int(payload.get("chunk_index", -1)) != chunk_index
        or payload.get("plan_sha256") != plan_sha256
        or payload.get("lock_identity_sha256") != lock_identity_sha256
    ):
        raise RuntimeError("capture chunk identity differs")
    public = {key: value for key, value in payload.items() if key != "tensors"}
    if set(public) != {
        "schema_version",
        "phase",
        "chunk_index",
        "plan_sha256",
        "lock_identity_sha256",
        "record_count",
        "records",
        "tensor_hashes",
        "chunk_identity_sha256",
    }:
        raise RuntimeError("capture chunk has undocumented public identity fields")
    _verify_hash(public, "chunk_identity_sha256")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise TypeError("capture chunk lacks tensors")
    observed = {
        name: tensor_float32_sha256(value) for name, value in sorted(tensors.items())
    }
    if set(observed) != {"anchor_residuals", "gradients"} or observed != payload.get(
        "tensor_hashes"
    ):
        raise RuntimeError("capture chunk tensor hash differs")
    _validate_chunk_against_plan(payload, expected_specifications)
    return dict(payload)


def _chunk_path_from_record(record: Mapping[str, Any]) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str):
        raise TypeError("capture manifest chunk lacks a path")
    path = (ROOT / raw).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError("capture chunk path escapes the repository") from error
    return path


def _validate_capture_manifest() -> dict[str, Any]:
    manifest = _load_json(CAPTURE_MANIFEST_PATH)
    _verify_hash(manifest, "manifest_sha256")
    if manifest.get("schema_version") != CAPTURE_SCHEMA:
        raise RuntimeError("DMS capture manifest schema differs")
    lock = _load_lock()
    dataset = _load_dataset()
    plan = _capture_specifications(dataset)
    plan_sha256 = _capture_plan_sha256(plan)
    if (
        manifest.get("development_only") is not True
        or manifest.get("lock_file_sha256") != file_sha256(LOCK_PATH)
        or manifest.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or manifest.get("model_identity_sha256") != canonical_sha256(MODEL)
        or manifest.get("runtime_identity_sha256") != canonical_sha256(EXPECTED_RUNTIME)
        or manifest.get("data_sha256") != file_sha256(DATA_PATH)
        or manifest.get("model_config_sha256") != file_sha256(MODEL_CONFIG_PATH)
        or manifest.get("capture_plan_sha256") != plan_sha256
        or manifest.get("prompt_content_sha256") != _prompt_content_sha256(plan)
        or manifest.get("record_count") != 136
        or manifest.get("layers") != list(LAYERS)
        or manifest.get("tensor_shape_per_record") != [len(LAYERS), MODEL["d_model"]]
        or manifest.get("generated_tokens") != 0
        or manifest.get("finite_intervention_outcomes_inspected") is not False
    ):
        raise RuntimeError("DMS capture manifest provenance differs")
    import torch

    chunks = _chunked(plan, CAPTURE_CHUNK_SIZE)
    manifest_chunks = manifest.get("chunks")
    if not isinstance(manifest_chunks, list) or len(manifest_chunks) != len(chunks):
        raise RuntimeError("DMS capture manifest chunk coverage differs")
    expected_chunk_work_ids = [
        [str(specification["work_id"]) for specification in chunk] for chunk in chunks
    ]
    if not CAPTURE_LEDGER_PATH.is_file():
        raise RuntimeError("DMS capture ledger is missing")
    ledger = PersistentChunkLedger(
        path=CAPTURE_LEDGER_PATH,
        plan_sha256=plan_sha256,
        lock_identity_sha256=str(lock["lock_identity_sha256"]),
        expected_chunk_work_ids=expected_chunk_work_ids,
        ceiling=CAPTURE_CEILING,
    )
    if ledger.completed_chunks() != len(chunks):
        raise RuntimeError("DMS capture ledger is not the complete frozen plan")
    for index, (chunk_record, expected_specifications) in enumerate(
        zip(manifest_chunks, chunks, strict=True)
    ):
        if not isinstance(chunk_record, Mapping) or set(chunk_record) != {
            "index",
            "path",
            "file_sha256",
            "record_count",
        }:
            raise RuntimeError("DMS capture manifest chunk fields differ")
        expected_path = _capture_chunk_path(index)
        if (
            chunk_record.get("index") != index
            or chunk_record.get("path") != _relative(expected_path)
            or chunk_record.get("record_count") != len(expected_specifications)
        ):
            raise RuntimeError("DMS capture manifest chunk plan differs")
        path = _chunk_path_from_record(chunk_record)
        if path != expected_path.resolve() or file_sha256(path) != chunk_record.get(
            "file_sha256"
        ):
            raise RuntimeError("DMS capture chunk file hash differs")
        _load_tensor_chunk(
            torch,
            path=path,
            chunk_index=index,
            plan_sha256=plan_sha256,
            lock_identity_sha256=str(lock["lock_identity_sha256"]),
            expected_specifications=expected_specifications,
        )
    compute = manifest.get("compute")
    if not isinstance(compute, Mapping) or dict(compute) != ledger.snapshot():
        raise RuntimeError("DMS capture compute ledger differs")
    return manifest


def run_capture() -> dict[str, Any]:
    run_preflight()
    if CAPTURE_MANIFEST_PATH.exists():
        return _validate_capture_manifest()
    dataset = _load_dataset()
    specifications = _capture_specifications(dataset)
    plan_sha256 = _capture_plan_sha256(specifications)
    chunks = _chunked(specifications, CAPTURE_CHUNK_SIZE)
    lock_identity_sha256 = str(_load_lock()["lock_identity_sha256"])
    expected_chunk_work_ids = [
        [str(specification["work_id"]) for specification in chunk] for chunk in chunks
    ]
    ledger = PersistentChunkLedger(
        path=CAPTURE_LEDGER_PATH,
        plan_sha256=plan_sha256,
        lock_identity_sha256=lock_identity_sha256,
        expected_chunk_work_ids=expected_chunk_work_ids,
        ceiling=CAPTURE_CEILING,
    )
    completed = ledger.completed_chunks()
    if completed > len(chunks):
        raise RuntimeError("DMS capture ledger is longer than the plan")
    import torch

    for index in range(completed):
        path = _capture_chunk_path(index)
        event = ledger.payload["events"][index]
        if (
            not path.is_file()
            or event.get("artifact_path") != _relative(path)
            or event.get("artifact_sha256") != file_sha256(path)
        ):
            raise RuntimeError("completed DMS capture chunk differs from its ledger")
        _load_tensor_chunk(
            torch,
            path=path,
            chunk_index=index,
            plan_sha256=plan_sha256,
            lock_identity_sha256=lock_identity_sha256,
            expected_specifications=chunks[index],
        )
    if completed < len(chunks):
        backend = load_backend()
        for index in range(completed, len(chunks)):
            chunk = chunks[index]
            ledger.reserve(chunk_index=index, work_ids=[str(item["work_id"]) for item in chunk])
            public_records: list[dict[str, Any]] = []
            gradients = []
            residuals = []
            for row_index, specification in enumerate(chunk):
                evidence = resolve_shared_anchor_evidence(
                    backend,
                    anchor_prefix=str(specification["anchor_prefix"]),
                    prompts=list(map(str, specification["evidence_prompts"])),
                    anchor_marker=str(dataset["anchor_marker"]),
                )
                capture = capture_multilayer_choice_anchor_gradient(
                    backend,
                    str(specification["prompt"]),
                    str(specification["preserve_label"]),
                    str(specification["comply_label"]),
                    layers=LAYERS,
                    anchor_index=evidence.anchor_index,
                )
                gradient = capture.raw_gradients.detach().cpu().float().contiguous()
                residual = capture.anchor_residuals.detach().cpu().float().contiguous()
                expected_shape = (len(LAYERS), MODEL["d_model"])
                if tuple(gradient.shape) != expected_shape or tuple(residual.shape) != expected_shape:
                    raise RuntimeError("DMS capture returned an unexpected all-layer tensor shape")
                gradients.append(gradient)
                residuals.append(residual)
                public_records.append(
                    {
                        **_public_specification(specification),
                        "row_index": row_index,
                        "anchor_index": evidence.anchor_index,
                        "anchor_evidence": evidence.audit,
                        "capture_audit": capture.audit,
                        "preserve_minus_comply_baseline_log_odds": capture.preserve_log_odds,
                        "gradient_float32_sha256": tensor_float32_sha256(gradient),
                        "anchor_residual_float32_sha256": tensor_float32_sha256(residual),
                    }
                )
            path = _capture_chunk_path(index)
            _save_tensor_chunk(
                torch,
                path=path,
                chunk_index=index,
                plan_sha256=plan_sha256,
                lock_identity_sha256=lock_identity_sha256,
                records=public_records,
                gradients=torch.stack(gradients).contiguous(),
                residuals=torch.stack(residuals).contiguous(),
            )
            ledger.complete(chunk_index=index, artifact_path=path)
            print(
                f"DMS capture chunk {index + 1}/{len(chunks)} "
                f"F={ledger.counts()[0]} B={ledger.counts()[1]}",
                flush=True,
            )
    snapshot = ledger.snapshot()
    if (
        snapshot["forward_evaluations"] != 136
        or snapshot["backward_evaluations"] != 136
        or snapshot["unique_work_id_count"] != 136
    ):
        raise RuntimeError("DMS capture did not consume exactly 136 F+B work units")
    manifest = _with_hash(
        {
            "schema_version": CAPTURE_SCHEMA,
            "development_only": True,
            "lock_file_sha256": file_sha256(LOCK_PATH),
            "lock_identity_sha256": lock_identity_sha256,
            "model_identity_sha256": canonical_sha256(MODEL),
            "runtime_identity_sha256": canonical_sha256(EXPECTED_RUNTIME),
            "data_sha256": file_sha256(DATA_PATH),
            "model_config_sha256": file_sha256(MODEL_CONFIG_PATH),
            "capture_plan_sha256": plan_sha256,
            "prompt_content_sha256": _prompt_content_sha256(specifications),
            "layers": list(LAYERS),
            "record_count": 136,
            "tensor_shape_per_record": [len(LAYERS), MODEL["d_model"]],
            "chunks": [
                {
                    "index": index,
                    "path": _relative(_capture_chunk_path(index)),
                    "file_sha256": file_sha256(_capture_chunk_path(index)),
                    "record_count": len(chunk),
                }
                for index, chunk in enumerate(chunks)
            ],
            "compute": snapshot,
            "generated_tokens": 0,
            "finite_intervention_outcomes_inspected": False,
        },
        "manifest_sha256",
    )
    _write_new_json(CAPTURE_MANIFEST_PATH, manifest)
    return _validate_capture_manifest()


def _load_capture_records(torch: Any) -> list[dict[str, Any]]:
    manifest = _validate_capture_manifest()
    lock = _load_lock()
    plan = _capture_specifications(_load_dataset())
    chunks = _chunked(plan, CAPTURE_CHUNK_SIZE)
    records: list[dict[str, Any]] = []
    for chunk_record, expected_specifications in zip(
        manifest["chunks"], chunks, strict=True
    ):
        index = int(chunk_record["index"])
        payload = _load_tensor_chunk(
            torch,
            path=_chunk_path_from_record(chunk_record),
            chunk_index=index,
            plan_sha256=str(manifest["capture_plan_sha256"]),
            lock_identity_sha256=str(lock["lock_identity_sha256"]),
            expected_specifications=expected_specifications,
        )
        gradients = payload["tensors"]["gradients"]
        residuals = payload["tensors"]["anchor_residuals"]
        for record in payload["records"]:
            row_index = int(record["row_index"])
            gradient = gradients[row_index].float().contiguous()
            residual = residuals[row_index].float().contiguous()
            records.append({**record, "gradient": gradient, "anchor_residual": residual})
    if len(records) != 136 or len({str(record["work_id"]) for record in records}) != 136:
        raise RuntimeError("DMS capture row coverage differs")
    return records


def _validate_screen_result() -> dict[str, Any]:
    result = _load_json(SCREEN_RESULT_PATH)
    _verify_hash(result, "result_sha256")
    if result.get("schema_version") != RESULT_SCHEMA:
        raise RuntimeError("DMS layer-screen result schema differs")
    if (
        result.get("lock_file_sha256") != file_sha256(LOCK_PATH)
        or result.get("capture_manifest_file_sha256") != file_sha256(CAPTURE_MANIFEST_PATH)
        or result.get("data_sha256") != file_sha256(DATA_PATH)
        or result.get("model_config_sha256") != file_sha256(MODEL_CONFIG_PATH)
        or result.get("screen_model_forwards") != 0
        or result.get("screen_model_backwards") != 0
        or result.get("generated_tokens") != 0
        or result.get("finite_intervention_outcomes_inspected") is not False
        or result.get("pilot_scenario_geometry_computed") is not False
    ):
        raise RuntimeError("DMS layer-screen result provenance differs")
    records = result.get("geometry_records")
    if not isinstance(records, list) or len(records) != 23 * 4 * len(METHODS):
        raise RuntimeError("DMS layer-screen geometry coverage differs")
    if any(record.get("partition") != "calibration" for record in records):
        raise RuntimeError("DMS layer screen contains non-calibration geometry")
    for record in records:
        _verify_hash(record, "screen_record_sha256")
    lock = _load_lock()
    expected_keys = {
        (layer, scenario_id, method)
        for layer in LAYERS
        for scenario_id in lock["dataset"]["calibration_scenario_ids"]
        for method in METHODS
    }
    observed_keys = {
        (int(record["layer"]), str(record["scenario_id"]), str(record["method"]))
        for record in records
    }
    if observed_keys != expected_keys:
        raise RuntimeError("DMS layer-screen result has the wrong layer/scenario/method grid")
    selection = result.get("selection")
    if not isinstance(selection, Mapping):
        raise TypeError("DMS layer-screen result lacks a selection record")
    _verify_hash(selection, "selection_sha256")
    if result.get("status") != selection.get("status"):
        raise RuntimeError("DMS layer-screen status differs from its selection")
    return result


def run_screen() -> dict[str, Any]:
    lock = _load_lock()
    capture_manifest = _validate_capture_manifest()
    if SCREEN_RESULT_PATH.exists():
        return _validate_screen_result()
    import torch

    records = _load_capture_records(torch)
    dataset = _load_dataset()
    nuisance_records = [record for record in records if record["kind"] == "nuisance_fit"]
    if len(nuisance_records) != 8:
        raise RuntimeError("DMS screen requires exactly eight unrelated gradients")
    calibration_scenarios = [
        scenario for scenario in dataset["scenarios"] if scenario["partition"] == "calibration"
    ]
    if len(calibration_scenarios) != 4:
        raise RuntimeError("DMS screen requires exactly four calibration scenarios")
    geometry_records: list[dict[str, Any]] = []
    for scenario in calibration_scenarios:
        scenario_id = str(scenario["id"])
        scenario_records = [
            record
            for record in records
            if record["kind"] == "scenario" and record["scenario_id"] == scenario_id
        ]
        if len(scenario_records) != 16:
            raise RuntimeError("each DMS calibration scenario requires 16 A/B captures")
        residual_scales = anchor_residual_scale_geometric_mean(
            torch, [record["anchor_residual"] for record in scenario_records]
        )
        target_records = [
            record
            for record in scenario_records
            if record["target"] == "self" and record["event"] == "permanent"
        ]
        protected_records = [
            record
            for record in scenario_records
            if not (record["target"] == "self" and record["event"] == "permanent")
        ]
        if len(target_records) != 4 or len(protected_records) != 12:
            raise RuntimeError("DMS target/protected row partition differs")
        target_offsets = torch.tensor(
            [record["preserve_minus_comply_baseline_log_odds"] for record in target_records],
            dtype=torch.float64,
        )
        protected_offsets = torch.tensor(
            [record["preserve_minus_comply_baseline_log_odds"] for record in protected_records],
            dtype=torch.float64,
        )
        for layer_index, layer in enumerate(LAYERS):
            scale = float(residual_scales[layer_index].item())
            target_rows = scale * torch.stack(
                [record["gradient"][layer_index].double() for record in target_records]
            )
            protected_rows = scale * torch.stack(
                [record["gradient"][layer_index].double() for record in protected_records]
            )
            unrelated_rows = scale * torch.stack(
                [record["gradient"][layer_index].double() for record in nuisance_records]
            )
            method_records = screen_scenario_layer(
                target_rows=target_rows.numpy(),
                target_offsets=target_offsets.numpy(),
                protected_rows=protected_rows.numpy(),
                protected_offsets=protected_offsets.numpy(),
                unrelated_rows=unrelated_rows.numpy(),
                margin=MARGIN,
                cap_frontier=CAP_FRONTIER,
            )
            for method_record in method_records:
                public = {
                    **method_record,
                    "scenario_id": scenario_id,
                    "partition": "calibration",
                    "layer": layer,
                    "residual_scale": scale,
                    "target_rows_sha256": canonical_sha256(target_rows.tolist()),
                    "protected_rows_sha256": canonical_sha256(protected_rows.tolist()),
                    "unrelated_rows_sha256": canonical_sha256(unrelated_rows.tolist()),
                    "target_offsets_sha256": canonical_sha256(target_offsets.tolist()),
                    "protected_offsets_sha256": canonical_sha256(protected_offsets.tolist()),
                }
                public["screen_record_sha256"] = canonical_sha256(public)
                geometry_records.append(public)
    expected_keys = {
        (layer, str(scenario["id"]), method)
        for layer in LAYERS
        for scenario in calibration_scenarios
        for method in METHODS
    }
    observed_keys = {
        (int(record["layer"]), str(record["scenario_id"]), str(record["method"]))
        for record in geometry_records
    }
    if observed_keys != expected_keys or len(geometry_records) != len(expected_keys):
        raise RuntimeError("DMS geometry records do not cover the frozen grid")
    selection = select_layer(
        geometry_records,
        calibration_scenario_ids=[str(scenario["id"]) for scenario in calibration_scenarios],
        layers=LAYERS,
        cap_frontier=CAP_FRONTIER,
        qualification_cap=QUALIFICATION_CAP,
    )
    result = _with_hash(
        {
            "schema_version": RESULT_SCHEMA,
            "status": selection["status"],
            "development_only": True,
            "lock_file_sha256": file_sha256(LOCK_PATH),
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "capture_manifest_file_sha256": file_sha256(CAPTURE_MANIFEST_PATH),
            "capture_manifest_sha256": capture_manifest["manifest_sha256"],
            "capture_plan_sha256": capture_manifest["capture_plan_sha256"],
            "prompt_content_sha256": capture_manifest["prompt_content_sha256"],
            "data_sha256": file_sha256(DATA_PATH),
            "model_config_sha256": file_sha256(MODEL_CONFIG_PATH),
            "model_identity_sha256": canonical_sha256(MODEL),
            "runtime_identity_sha256": canonical_sha256(EXPECTED_RUNTIME),
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "selection_partition": "calibration_only",
            "pilot_scenario_geometry_computed": False,
            "geometry_records": geometry_records,
            "geometry_record_count": len(geometry_records),
            "eligible_geometry_record_count": sum(
                record["status"] == "eligible" for record in geometry_records
            ),
            "selection": selection,
            "capture_compute": capture_manifest["compute"],
            "screen_model_forwards": 0,
            "screen_model_backwards": 0,
            "finite_intervention_forwards": 0,
            "finite_intervention_backwards": 0,
            "finite_intervention_outcomes_inspected": False,
            "generated_tokens": 0,
            "external_model_judges": 0,
            "external_api_calls": 0,
            "paid_model_cost_usd": 0,
            "local_linear_only": True,
            "claim_boundary": lock["claim_boundary"],
        },
        "result_sha256",
    )
    _write_new_json(SCREEN_RESULT_PATH, result)
    return _validate_screen_result()


def _norm_text(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _render_report(result: Mapping[str, Any]) -> str:
    selection = result["selection"]
    scenario_ids = list(selection["calibration_scenario_ids"])
    short_ids = [scenario_id.replace("fcag_dev_", "") for scenario_id in scenario_ids]
    lines = [
        "# Decision-Margin Shielding v2 layer-screen result",
        "",
        f"Status: **{selection['status']}**.",
        "",
        (
            "This is an opened, local-linear geometry result. It used no finite steering "
            "intervention, generated text, external judge, API, or pilot-scenario layer score."
        ),
        "",
        "| Layer | "
        + " | ".join(short_ids)
        + " | Worst | Mean | Qualifies at L2 <= 2 |",
        "|---:|" + "---:|" * len(short_ids) + "---:|---:|:---:|",
    ]
    for summary in selection["layer_summaries"]:
        norms = summary["scenario_minimum_standardized_l2"]
        lines.append(
            f"| {summary['layer']} | "
            + " | ".join(_norm_text(norms[scenario_id]) for scenario_id in scenario_ids)
            + f" | {_norm_text(summary['worst_case_minimum_standardized_l2'])}"
            f" | {_norm_text(summary['mean_minimum_standardized_l2'])}"
            f" | {'yes' if summary['qualifies'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Selected layer: **{selection['selected_layer']}**.",
            "",
            (
                "A dash means at least one scenario had no certified uncapped DMS solution. "
                "Zero qualifying layers is a valid construction no-go."
            ),
            "",
            (
                "Protected rows with |baseline A/B log-odds| < 0.05 are first-order "
                "frozen, not margin-certified. Equality at 0.05 is margin-certified."
            ),
            "",
            (
                "Nothing here proves nonlinear decision preservation, full-vocabulary "
                "stability, a natural self-preservation mechanism, unchanged capability, "
                "safety, priority, or publication novelty."
            ),
            "",
            f"Result SHA-256: `{result['result_sha256']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_report() -> str:
    result = _validate_screen_result()
    rendered = _render_report(result)
    if REPORT_PATH.exists():
        if REPORT_PATH.read_text(encoding="utf-8") != rendered:
            raise RuntimeError("existing DMS layer-screen report differs from validated results")
    else:
        _atomic_text(REPORT_PATH, rendered)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision-Margin Shielding v2 layer screen")
    parser.add_argument("command", choices=("lock", "preflight", "capture", "screen", "report"))
    arguments = parser.parse_args()
    if arguments.command == "lock":
        result: Any = run_lock()
    elif arguments.command == "preflight":
        result = run_preflight()
    elif arguments.command == "capture":
        result = run_capture()
    elif arguments.command == "screen":
        result = run_screen()
    else:
        result = run_report()
    if isinstance(result, str):
        print(result, end="")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
