from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sp_lense.paired_order_analytic_gradient import (
    PairedOrderGradientIneligible,
    cast_and_recertify_common_delta,
    construct_common_gradient_bisector,
    full_vocabulary_bidirectional_interval,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "paired_order_analytic_gradient_development_lock.json"
CONFIRMATION_RUNTIME_PATH = ROOT / "scripts" / "counterfactual_semantic_gradient_confirmation.py"
SOURCE_GATE_PATH = (
    ROOT
    / "results"
    / "counterfactual_semantic_gradient_confirmation"
    / "qwen35_08b"
    / "semantic_gate_confirmation_result.json"
)
ARTIFACT_ROOT = ROOT / "artifacts" / "paired_order_analytic_gradient_development" / "qwen35_08b_v2"
RESULT_ROOT = ROOT / "results" / "paired_order_analytic_gradient_development" / "qwen35_08b_v2"
CAPTURE_PATH = ARTIFACT_ROOT / "paired_capture.pt"
CAPTURE_MANIFEST_PATH = ARTIFACT_ROOT / "paired_capture_manifest.json"
CAPTURE_ATTEMPT_LEDGER_PATH = ARTIFACT_ROOT / "paired_capture_attempt_ledger.json"
CONSTRUCTION_PATH = ARTIFACT_ROOT / "preoutcome_constructions.pt"
CONSTRUCTION_MANIFEST_PATH = ARTIFACT_ROOT / "preoutcome_construction_manifest.json"
CONSTRUCTION_ATTEMPT_LEDGER_PATH = ARTIFACT_ROOT / "preoutcome_construction_attempt_ledger.json"
PREOUTCOME_FREEZE_PATH = ARTIFACT_ROOT / "preoutcome_freeze.json"
EVALUATION_CHECKPOINT_PATH = RESULT_ROOT / "evaluation_checkpoint.json"
RESULT_PATH = RESULT_ROOT / "development_result.json"
REPORT_PATH = RESULT_ROOT / "DEVELOPMENT_REPORT.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(torch: Any, path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _protocol_bound_paths(config: Mapping[str, Any]) -> list[Path]:
    return [CONFIG_PATH, *(ROOT / relative for relative in config["locked_inputs"])]


def _require_committed_clean(paths: Sequence[Path], *, stage: str) -> None:
    relatives = [_relative(path) for path in paths]
    for relative in relatives:
        _git("ls-files", "--error-unmatch", "--", relative)
    if _git("status", "--porcelain", "--", *relatives):
        raise RuntimeError(f"{stage} inputs must be committed and clean")


def _new_attempt_ledger(phase: str) -> dict[str, Any]:
    return {
        "schema_version": "sp_lense.paired_order_analytic_gradient_attempt_ledger.v1",
        "phase": phase,
        "config_sha256": _sha256(CONFIG_PATH),
        "attempts": [],
    }


def _write_attempt_ledger(path: Path, ledger: dict[str, Any]) -> None:
    ledger.pop("result_sha256", None)
    ledger["result_sha256"] = _canonical_sha256(ledger)
    _atomic_json(path, ledger)


def _load_attempt_ledger(path: Path, *, phase: str) -> dict[str, Any]:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if (
        ledger.get("schema_version") != "sp_lense.paired_order_analytic_gradient_attempt_ledger.v1"
        or ledger.get("phase") != phase
        or ledger.get("config_sha256") != _sha256(CONFIG_PATH)
        or not isinstance(ledger.get("attempts"), list)
    ):
        raise RuntimeError(f"{phase} attempt ledger has invalid identity")
    copy = dict(ledger)
    embedded = copy.pop("result_sha256", None)
    if embedded != _canonical_sha256(copy):
        raise RuntimeError(f"{phase} attempt ledger embedded hash differs")
    work_ids = [str(row.get("work_id")) for row in ledger["attempts"]]
    if len(work_ids) != len(set(work_ids)):
        raise RuntimeError(f"{phase} attempt ledger contains duplicate work IDs")
    valid_statuses = {"reserved", "returned_uncommitted", "committed_to_artifact"}
    if any(row.get("status") not in valid_statuses for row in ledger["attempts"]):
        raise RuntimeError(f"{phase} attempt ledger contains an invalid status")
    return ledger


def _reserve_attempt(
    path: Path,
    ledger: dict[str, Any],
    *,
    work_id: str,
    operation: str,
) -> None:
    if any(str(row["work_id"]) == work_id for row in ledger["attempts"]):
        raise RuntimeError(f"attempt work ID already exists: {work_id}")
    ledger["attempts"].append({"work_id": work_id, "operation": operation, "status": "reserved"})
    _write_attempt_ledger(path, ledger)


def _mark_attempt_returned(path: Path, ledger: dict[str, Any], work_id: str) -> None:
    row = next(row for row in ledger["attempts"] if str(row["work_id"]) == work_id)
    if row["status"] != "reserved":
        raise RuntimeError(f"attempt is not reserved: {work_id}")
    row["status"] = "returned_uncommitted"
    _write_attempt_ledger(path, ledger)


def _commit_attempts(path: Path, ledger: dict[str, Any], work_ids: Sequence[str]) -> None:
    selected = {str(value) for value in work_ids}
    rows = [row for row in ledger["attempts"] if str(row["work_id"]) in selected]
    if len(rows) != len(selected) or any(row["status"] != "returned_uncommitted" for row in rows):
        raise RuntimeError("only returned attempts may be committed to an artifact")
    for row in rows:
        row["status"] = "committed_to_artifact"
    _write_attempt_ledger(path, ledger)


def _require_unambiguous_attempts(ledger: Mapping[str, Any], *, phase: str) -> None:
    ambiguous = [
        str(row["work_id"])
        for row in ledger["attempts"]
        if row["status"] != "committed_to_artifact"
    ]
    if ambiguous:
        raise RuntimeError(
            f"{phase} contains an ambiguously interrupted model call; fail closed: "
            + ", ".join(ambiguous)
        )


def _validate_capture_manifest() -> dict[str, Any]:
    manifest = json.loads(CAPTURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != "sp_lense.paired_order_analytic_gradient_capture_manifest.v1"
        or manifest.get("status") != "complete"
        or manifest.get("config_sha256") != _sha256(CONFIG_PATH)
        or manifest.get("capture_file_sha256") != _sha256(CAPTURE_PATH)
        or manifest.get("attempt_ledger_sha256") != _sha256(CAPTURE_ATTEMPT_LEDGER_PATH)
    ):
        raise RuntimeError("capture manifest does not bind the complete current capture")
    ledger = _load_attempt_ledger(CAPTURE_ATTEMPT_LEDGER_PATH, phase="capture")
    _require_unambiguous_attempts(ledger, phase="capture")
    expected = int(_load_config()["opened_calibration"]["expected_active_pair_count"])
    if int(manifest.get("pair_count", -1)) != expected:
        raise RuntimeError("capture manifest pair count differs from the lock")
    copy = dict(manifest)
    embedded = copy.pop("result_sha256", None)
    if embedded != _canonical_sha256(copy):
        raise RuntimeError("capture manifest embedded result hash differs")
    return manifest


def _validate_construction_manifest() -> dict[str, Any]:
    manifest = json.loads(CONSTRUCTION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != "sp_lense.paired_order_analytic_gradient_preoutcome_manifest.v1"
        or manifest.get("status") != "complete"
        or manifest.get("outcomes_viewed") is not False
        or manifest.get("config_sha256") != _sha256(CONFIG_PATH)
        or manifest.get("capture_file_sha256") != _sha256(CAPTURE_PATH)
        or manifest.get("capture_manifest_sha256") != _sha256(CAPTURE_MANIFEST_PATH)
        or manifest.get("construction_file_sha256") != _sha256(CONSTRUCTION_PATH)
        or manifest.get("attempt_ledger_sha256") != _sha256(CONSTRUCTION_ATTEMPT_LEDGER_PATH)
    ):
        raise RuntimeError("preoutcome manifest does not bind the current artifact chain")
    ledger = _load_attempt_ledger(CONSTRUCTION_ATTEMPT_LEDGER_PATH, phase="construction")
    _require_unambiguous_attempts(ledger, phase="construction")
    expected_reserves = [
        float(value) for value in _load_config()["intervention"]["reserve_candidates"]
    ]
    observed_reserves = [float(row["reserve_logit"]) for row in manifest["candidates"]]
    if observed_reserves != expected_reserves:
        raise RuntimeError("preoutcome manifest reserve order differs from the lock")
    copy = dict(manifest)
    embedded = copy.pop("result_sha256", None)
    if embedded != _canonical_sha256(copy):
        raise RuntimeError("preoutcome manifest embedded result hash differs")
    return manifest


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("schema_version") != "sp_lense.paired_order_analytic_gradient_lock.v1":
        raise ValueError("unsupported paired-order analytic-gradient lock")
    for relative, expected in config["locked_inputs"].items():
        observed = _sha256(ROOT / relative)
        if observed != expected:
            raise RuntimeError(f"locked input differs: {relative}: {observed} != {expected}")
    reserves = [float(value) for value in config["intervention"]["reserve_candidates"]]
    if reserves != sorted(set(reserves)) or not reserves or any(value <= 0 for value in reserves):
        raise ValueError("reserve candidates must be unique, positive, and ascending")
    return config


def _load_confirmation() -> Any:
    spec = importlib.util.spec_from_file_location(
        "sp_lense_poag_confirmation", CONFIRMATION_RUNTIME_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import frozen confirmation renderer/runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    gate = json.loads(SOURCE_GATE_PATH.read_text(encoding="utf-8"))
    expected = config["source_gate"]
    if gate.get("status") != "passed" or _sha256(SOURCE_GATE_PATH) != expected["file_sha256"]:
        raise RuntimeError("the locked source gate is absent, changed, or not passed")
    if gate.get("result_sha256") != expected["embedded_result_sha256"]:
        raise RuntimeError("the locked source gate embedded hash differs")
    counts = gate.get("pair_counts")
    if counts != expected["pair_counts"]:
        raise RuntimeError("the source gate coverage differs from the lock")
    return gate


def _active_jobs() -> tuple[Any, list[tuple[tuple[str, int], list[dict[str, Any]]]]]:
    config = _load_config()
    gate = _source_gate(config)
    confirmation = _load_confirmation()
    adaptive, jobs, _ = confirmation._inputs()
    active_keys = {
        (str(row["case_id"]), int(row["assignment"]))
        for row in gate["pair_rows"]
        if row["target"] == "self" and bool(row["predicted_active"])
    }
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        key = (str(job["case_id"]), int(job["assignment"]))
        if job["target"] == "self" and key in active_keys:
            grouped[key].append(dict(job))
    ordered = []
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda row: bool(row["preserve_first"]))
        if len(values) != 2 or {bool(row["preserve_first"]) for row in values} != {False, True}:
            raise RuntimeError(f"active pair lacks both answer orders: {key}")
        ordered.append((key, values))
    expected_pairs = int(config["opened_calibration"]["expected_active_pair_count"])
    if len(ordered) != expected_pairs or active_keys != set(grouped):
        raise RuntimeError("active pair coverage differs from the lock")
    prompt_hashes = [str(job["prompt_sha256"]) for _, values in ordered for job in values]
    if len(prompt_hashes) != 2 * expected_pairs or len(set(prompt_hashes)) != len(prompt_hashes):
        raise RuntimeError("active prompts are missing or duplicated")
    plan = [
        {
            "case_id": key[0],
            "assignment": key[1],
            "orders": [
                {
                    "unit_id": str(job["unit_id"]),
                    "preserve_first": bool(job["preserve_first"]),
                    "prompt_sha256": str(job["prompt_sha256"]),
                    "positive_label": str(job["positive_label"]),
                    "negative_label": str(job["negative_label"]),
                }
                for job in values
            ],
        }
        for key, values in ordered
    ]
    if _canonical_sha256(plan) != config["opened_calibration"]["active_prompt_plan_sha256"]:
        raise RuntimeError("rendered active prompt plan differs from the lock")
    return adaptive, ordered


def _tensor_hash(adaptive: Any, tensor: Any) -> str:
    return str(adaptive.tensor_float32_sha256(tensor.detach().float().cpu().contiguous()))


def _raw_tensor_hash(tensor: Any) -> str:
    array = tensor.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _capture_order(
    backend: Any, adaptive: Any, job: Mapping[str, Any], *, layer: int
) -> dict[str, Any]:
    torch = backend.torch
    tokens = backend.encode(str(job["prompt"]))
    boundary = adaptive.resolve_choice_boundary(backend, str(job["prompt"]))
    if boundary.prompt_length != int(tokens.shape[-1]):
        raise RuntimeError("choice boundary prompt length differs")
    preserve_id = boundary.token_id(str(job["positive_label"]))
    comply_id = boundary.token_id(str(job["negative_label"]))
    captured: dict[str, Any] = {"hook_calls": 0}

    def capture_hook(activation: Any, hook: Any) -> Any:
        del hook
        captured["hook_calls"] += 1
        leaf = activation.detach().requires_grad_(True)
        captured["activation"] = leaf
        return leaf

    backend.model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    with (
        torch.enable_grad(),
        backend.model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_out", capture_hook)]),
    ):
        logits_device = backend.model(tokens)[0, -1].float()
        objective = logits_device[preserve_id] - logits_device[comply_id]
        gradient = torch.autograd.grad(objective, captured["activation"])[0][0, -1]
    if captured["hook_calls"] != 1:
        raise RuntimeError("capture hook did not fire exactly once")
    residual = captured["activation"][0, -1].detach().float().cpu().contiguous().clone()
    semantic_gradient = gradient.detach().float().cpu().contiguous().clone()
    logits = logits_device.detach().float().cpu().contiguous().clone()
    backend.model.zero_grad(set_to_none=True)
    if not all(
        bool(torch.isfinite(value).all().item()) for value in (residual, semantic_gradient, logits)
    ):
        raise RuntimeError("model capture contains a non-finite tensor")
    argmax = int(logits.argmax().item())
    baseline_semantic = (
        "positive" if argmax == preserve_id else "negative" if argmax == comply_id else "OTHER"
    )
    return {
        "unit_id": str(job["unit_id"]),
        "preserve_first": bool(job["preserve_first"]),
        "prompt_sha256": str(job["prompt_sha256"]),
        "prompt_length": int(tokens.shape[-1]),
        "positive_label": str(job["positive_label"]),
        "negative_label": str(job["negative_label"]),
        "preserve_token_id": int(preserve_id),
        "comply_token_id": int(comply_id),
        "baseline_argmax_token_id": argmax,
        "baseline_semantic_choice": baseline_semantic,
        "baseline_is_ab": baseline_semantic != "OTHER",
        "baseline_preserve_minus_comply_log_odds": float(objective.detach().item()),
        "choice_boundary_evidence_sha256": str(boundary.evidence_sha256),
        "prompt_token_ids_sha256": str(boundary.prompt_prefix_token_ids_sha256),
        "residual": residual,
        "semantic_gradient": semantic_gradient,
        "baseline_logits": logits,
        "residual_sha256": _tensor_hash(adaptive, residual),
        "semantic_gradient_sha256": _tensor_hash(adaptive, semantic_gradient),
        "baseline_logits_sha256": _tensor_hash(adaptive, logits),
        "residual_norm": float(residual.double().norm().item()),
        "semantic_gradient_norm": float(semantic_gradient.double().norm().item()),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _head_jvp(backend: Any, residual: Any, vector: Any) -> tuple[Any, Any]:
    torch = backend.torch
    residual_device = (
        residual.detach().to(device=backend.device, dtype=torch.float32).view(1, 1, -1)
    )
    vector_device = vector.detach().to(device=backend.device, dtype=torch.float32).view(1, 1, -1)

    def head(value: Any) -> Any:
        return backend.model.unembed(backend.model.ln_final(value))

    with torch.enable_grad():
        primal, tangent = torch.autograd.functional.jvp(
            head,
            residual_device,
            vector_device,
            create_graph=False,
            strict=True,
        )
    primal = primal[0, 0].detach().float().cpu().contiguous()
    tangent = tangent[0, 0].detach().float().cpu().contiguous()
    if not bool(torch.isfinite(primal).all().item()) or not bool(
        torch.isfinite(tangent).all().item()
    ):
        raise RuntimeError("final-head JVP contains non-finite values")
    return primal, tangent


def _public_order_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"residual", "semantic_gradient", "baseline_logits", "base_logit_jvp"}
    }


def _capture_attempt_work_id(
    case_id: str, assignment: int, order_index: int, operation: str
) -> str:
    return f"capture:{case_id}:{assignment}:{order_index}:{operation}"


def _construction_attempt_work_id(
    candidate_index: int, case_id: str, assignment: int, order_index: int
) -> str:
    return f"construction:{candidate_index}:{case_id}:{assignment}:{order_index}:delta_jvp"


def _validate_attempt_coverage(
    ledger: Mapping[str, Any], expected: Mapping[str, str], *, phase: str
) -> None:
    _require_unambiguous_attempts(ledger, phase=phase)
    observed = [(str(row["work_id"]), str(row["operation"])) for row in ledger["attempts"]]
    if observed != list(expected.items()):
        raise RuntimeError(f"{phase} attempt ledger differs from artifact coverage")


def _capture_expected_attempts(payload: Mapping[str, Any]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for pair in payload["pairs"]:
        case_id = str(pair["case_id"])
        assignment = int(pair["assignment"])
        for order_index, _order in enumerate(pair["orders"]):
            expected[
                _capture_attempt_work_id(case_id, assignment, order_index, "forward_backward")
            ] = "full_forward_and_backward"
        for order_index, order in enumerate(pair["orders"]):
            if order.get("base_logit_jvp") is not None:
                expected[_capture_attempt_work_id(case_id, assignment, order_index, "base_jvp")] = (
                    "final_head_base_jvp"
                )
    return expected


def _construction_expected_attempts(payload: Mapping[str, Any]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for candidate_index, candidate in enumerate(payload["candidates"]):
        for pair in candidate["pairs"]:
            count = int(pair.get("delta_head_jvp_count", 0))
            if count not in {0, 2}:
                raise RuntimeError("construction row has an invalid delta-JVP count")
            for order_index in range(count):
                expected[
                    _construction_attempt_work_id(
                        candidate_index,
                        str(pair["case_id"]),
                        int(pair["assignment"]),
                        order_index,
                    )
                ] = "final_head_cast_delta_jvp"
    return expected


def _load_capture(torch: Any) -> dict[str, Any]:
    payload = torch.load(CAPTURE_PATH, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != "sp_lense.paired_order_analytic_gradient_capture.v1"
        or payload.get("config_sha256") != _sha256(CONFIG_PATH)
        or not isinstance(payload.get("pairs"), list)
    ):
        raise RuntimeError("capture belongs to another lock")
    seen: set[tuple[str, int]] = set()
    base_jvps = 0
    for pair in payload["pairs"]:
        if type(pair.get("bisector_eligible")) is not bool:
            raise RuntimeError("capture pair has a non-boolean bisector eligibility flag")
        key = (str(pair["case_id"]), int(pair["assignment"]))
        if key in seen:
            raise RuntimeError("capture contains a duplicate pair")
        seen.add(key)
        orders = pair.get("orders")
        if not isinstance(orders, list) or len(orders) != 2:
            raise RuntimeError("captured pair lacks two answer orders")
        if {bool(row["preserve_first"]) for row in orders} != {False, True}:
            raise RuntimeError("captured pair has invalid answer-order coverage")
        for row in orders:
            for field, hash_field in (
                ("residual", "residual_sha256"),
                ("semantic_gradient", "semantic_gradient_sha256"),
                ("baseline_logits", "baseline_logits_sha256"),
            ):
                if _raw_tensor_hash(row[field]) != row[hash_field]:
                    raise RuntimeError(f"captured tensor hash differs: {field}")
            if row.get("base_logit_jvp") is not None:
                if _raw_tensor_hash(row["base_logit_jvp"]) != row["base_logit_jvp_sha256"]:
                    raise RuntimeError("captured base JVP hash differs")
                base_jvps += 1
        has_base_jvps = [row.get("base_logit_jvp") is not None for row in orders]
        if pair["bisector_eligible"]:
            if has_base_jvps != [True, True]:
                raise RuntimeError("eligible capture pair does not contain exactly two base JVPs")
            if (
                pair.get("base_vector") is None
                or _raw_tensor_hash(pair["base_vector"]) != pair["base_vector_sha256"]
            ):
                raise RuntimeError("captured base-vector hash differs")
        elif (
            has_base_jvps != [False, False]
            or pair.get("base_vector") is not None
            or pair.get("base_vector_sha256") is not None
        ):
            raise RuntimeError("ineligible capture pair contains a base vector or base JVP")
    compute = payload.get("compute", {})
    if int(compute.get("full_forward_passes", -1)) != 2 * len(payload["pairs"]):
        raise RuntimeError("capture forward ledger differs from pair coverage")
    if int(compute.get("backward_passes", -1)) != 2 * len(payload["pairs"]):
        raise RuntimeError("capture backward ledger differs from pair coverage")
    if int(compute.get("base_head_jvps", -1)) != base_jvps:
        raise RuntimeError("capture base-JVP ledger differs from stored rows")
    ledger = _load_attempt_ledger(CAPTURE_ATTEMPT_LEDGER_PATH, phase="capture")
    _validate_attempt_coverage(ledger, _capture_expected_attempts(payload), phase="capture")
    return payload


def run_capture() -> dict[str, Any]:
    config = _load_config()
    _require_committed_clean(_protocol_bound_paths(config), stage="paired-order protocol")
    existing = [
        CAPTURE_PATH.exists(),
        CAPTURE_MANIFEST_PATH.exists(),
        CAPTURE_ATTEMPT_LEDGER_PATH.exists(),
    ]
    if any(existing):
        if not all(existing):
            raise RuntimeError(
                "an interrupted capture left a partial artifact chain; fail closed without replay"
            )
        import torch

        _load_capture(torch)
        return _validate_capture_manifest()
    adaptive, groups = _active_jobs()
    backend = adaptive.load_backend(adaptive.load_lock())
    torch = backend.torch
    layer = int(config["intervention"]["residual_layer_zero_based"])
    payload = {
        "schema_version": "sp_lense.paired_order_analytic_gradient_capture.v1",
        "config_sha256": _sha256(CONFIG_PATH),
        "pairs": [],
        "compute": {
            "full_forward_passes": 0,
            "backward_passes": 0,
            "base_head_jvps": 0,
        },
    }
    attempt_ledger = _new_attempt_ledger("capture")
    started = time.perf_counter()
    for key, jobs in groups:
        pair_attempt_ids: list[str] = []
        orders = []
        for order_index, job in enumerate(jobs):
            work_id = _capture_attempt_work_id(key[0], key[1], order_index, "forward_backward")
            _reserve_attempt(
                CAPTURE_ATTEMPT_LEDGER_PATH,
                attempt_ledger,
                work_id=work_id,
                operation="full_forward_and_backward",
            )
            row = _capture_order(backend, adaptive, job, layer=layer)
            _mark_attempt_returned(CAPTURE_ATTEMPT_LEDGER_PATH, attempt_ledger, work_id)
            pair_attempt_ids.append(work_id)
            orders.append(row)
            payload["compute"]["full_forward_passes"] += 1
            payload["compute"]["backward_passes"] += 1
        gradients = np.stack([row["semantic_gradient"].double().numpy() for row in orders])
        residual_norms = [float(row["residual_norm"]) for row in orders]
        gradient_norms = [float(row["semantic_gradient_norm"]) for row in orders]
        baseline_is_ab = all(bool(row["baseline_is_ab"]) for row in orders)
        try:
            if not baseline_is_ab:
                raise PairedOrderGradientIneligible(
                    "one or more unsteered orders has a non-A/B argmax",
                    diagnostics={
                        "failure": "baseline_other",
                        "baseline_semantic_choices": [
                            str(row["baseline_semantic_choice"]) for row in orders
                        ],
                    },
                )
            if any(value <= 0.0 for value in [*residual_norms, *gradient_norms]):
                raise PairedOrderGradientIneligible(
                    "one or more captured residual or semantic gradient has zero norm",
                    diagnostics={
                        "failure": "zero_residual_or_gradient_norm",
                        "residual_norms": residual_norms,
                        "semantic_gradient_norms": gradient_norms,
                    },
                )
            bisector = construct_common_gradient_bisector(
                gradients,
                residual_norms,
                minimum_order_cosine=float(config["intervention"]["minimum_order_cosine"]),
                identity_tolerance=float(config["numeric_tolerances"]["bisector_identity"]),
            )
            base_vector = torch.from_numpy(bisector.base_vector).float().contiguous()
            for order_index, row in enumerate(orders):
                work_id = _capture_attempt_work_id(key[0], key[1], order_index, "base_jvp")
                _reserve_attempt(
                    CAPTURE_ATTEMPT_LEDGER_PATH,
                    attempt_ledger,
                    work_id=work_id,
                    operation="final_head_base_jvp",
                )
                primal, directional = _head_jvp(backend, row["residual"], base_vector)
                _mark_attempt_returned(CAPTURE_ATTEMPT_LEDGER_PATH, attempt_ledger, work_id)
                pair_attempt_ids.append(work_id)
                payload["compute"]["base_head_jvps"] += 1
                head_error = float(
                    (primal.double() - row["baseline_logits"].double()).abs().max().item()
                )
                if head_error > float(config["numeric_tolerances"]["head_primal_max_abs"]):
                    raise RuntimeError(f"final-head primal mismatch: {head_error}")
                predicted_semantic = float(
                    directional[row["preserve_token_id"]].item()
                    - directional[row["comply_token_id"]].item()
                )
                gradient_semantic = float(row["semantic_gradient"].double() @ base_vector.double())
                derivative_error = abs(predicted_semantic - gradient_semantic)
                derivative_allowance = float(
                    config["numeric_tolerances"]["gradient_jvp_absolute"]
                ) + float(config["numeric_tolerances"]["gradient_jvp_relative"]) * max(
                    1.0, abs(predicted_semantic), abs(gradient_semantic)
                )
                if derivative_error > derivative_allowance:
                    raise RuntimeError("semantic gradient and final-head JVP disagree")
                row["base_logit_jvp"] = directional
                row["base_logit_jvp_sha256"] = _tensor_hash(adaptive, directional)
                row["head_primal_max_abs_error"] = head_error
                row["gradient_jvp_absolute_error"] = derivative_error
                row["gradient_jvp_allowance"] = derivative_allowance
            bisector_eligible = True
            bisector_failure = None
            bisector_diagnostics = bisector.diagnostics
            base_vector_sha256 = _tensor_hash(adaptive, base_vector)
        except PairedOrderGradientIneligible as error:
            bisector_eligible = False
            bisector_failure = error.diagnostics
            bisector_diagnostics = None
            base_vector = None
            base_vector_sha256 = None
        payload["pairs"].append(
            {
                "case_id": key[0],
                "assignment": key[1],
                "orders": orders,
                "bisector_eligible": bisector_eligible,
                "bisector_failure": bisector_failure,
                "base_vector": base_vector,
                "base_vector_sha256": base_vector_sha256,
                "bisector_diagnostics": bisector_diagnostics,
            }
        )
        payload["pairs"].sort(key=lambda row: (str(row["case_id"]), int(row["assignment"])))
        _atomic_torch_save(torch, CAPTURE_PATH, payload)
        _commit_attempts(CAPTURE_ATTEMPT_LEDGER_PATH, attempt_ledger, pair_attempt_ids)
        print(f"captured paired gradient {len(payload['pairs'])}/{len(groups)}: {key}", flush=True)
    expected = int(config["opened_calibration"]["expected_active_pair_count"])
    if (
        len(payload["pairs"]) != expected
        or payload["compute"]["full_forward_passes"] != 2 * expected
        or payload["compute"]["backward_passes"] != 2 * expected
    ):
        raise RuntimeError("paired capture is incomplete or has an invalid compute ledger")
    manifest_rows = [
        {
            "case_id": row["case_id"],
            "assignment": row["assignment"],
            "bisector_eligible": row["bisector_eligible"],
            "bisector_failure": row["bisector_failure"],
            "base_vector_sha256": row["base_vector_sha256"],
            "bisector_diagnostics": row["bisector_diagnostics"],
            "orders": [_public_order_record(order) for order in row["orders"]],
        }
        for row in payload["pairs"]
    ]
    manifest = {
        "schema_version": "sp_lense.paired_order_analytic_gradient_capture_manifest.v1",
        "status": "complete",
        "development_only": True,
        "config_sha256": _sha256(CONFIG_PATH),
        "capture_file_sha256": _sha256(CAPTURE_PATH),
        "attempt_ledger_sha256": _sha256(CAPTURE_ATTEMPT_LEDGER_PATH),
        "attempt_count": len(attempt_ledger["attempts"]),
        "pair_count": len(manifest_rows),
        "row_manifest_sha256": _canonical_sha256(manifest_rows),
        "compute": {
            **payload["compute"],
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judge_calls": 0,
            "external_cost_usd": 0,
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
        },
        "pairs": manifest_rows,
    }
    manifest["result_sha256"] = _canonical_sha256(manifest)
    _atomic_json(CAPTURE_MANIFEST_PATH, manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "pair_count": len(manifest_rows),
                "compute": manifest["compute"],
            },
            indent=2,
        )
    )
    return manifest


def _construction_public(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "delta"}


def _load_constructions(torch: Any) -> dict[str, Any]:
    payload = torch.load(CONSTRUCTION_PATH, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version")
        != "sp_lense.paired_order_analytic_gradient_preoutcome_bank.v1"
        or payload.get("status") != "complete"
        or payload.get("config_sha256") != _sha256(CONFIG_PATH)
        or payload.get("capture_file_sha256") != _sha256(CAPTURE_PATH)
        or payload.get("capture_manifest_sha256") != _sha256(CAPTURE_MANIFEST_PATH)
    ):
        raise RuntimeError("construction bank belongs to another lock")
    config = _load_config()
    expected_reserves = [float(value) for value in config["intervention"]["reserve_candidates"]]
    observed_reserves = [float(row["reserve_logit"]) for row in payload.get("candidates", [])]
    if observed_reserves != expected_reserves:
        raise RuntimeError("construction bank reserve order differs from the lock")
    expected_pairs = int(config["opened_calibration"]["expected_active_pair_count"])
    capture = _load_capture(torch)
    expected_pair_keys = [(str(row["case_id"]), int(row["assignment"])) for row in capture["pairs"]]
    expected_delta_jvps = 0
    for candidate in payload["candidates"]:
        pairs = candidate.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != expected_pairs:
            raise RuntimeError("construction candidate has invalid pair coverage")
        keys = [(str(row["case_id"]), int(row["assignment"])) for row in pairs]
        if keys != expected_pair_keys:
            raise RuntimeError("construction candidate pair order differs from capture")
        if any(type(row.get("eligible")) is not bool for row in pairs):
            raise RuntimeError("construction row has a non-boolean eligibility flag")
        eligible = [row for row in pairs if row["eligible"]]
        if type(candidate.get("eligible_pair_count")) is not int or candidate[
            "eligible_pair_count"
        ] != len(eligible):
            raise RuntimeError("construction eligible count is not derived from its pairs")
        for row in pairs:
            jvp_count = int(row.get("delta_head_jvp_count", -1))
            if jvp_count not in {0, 2}:
                raise RuntimeError("construction row has an invalid delta-JVP count")
            primal_errors = row.get("cast_head_primal_max_abs_errors")
            if (
                not isinstance(primal_errors, list)
                or len(primal_errors) != jvp_count
                or any(
                    not math.isfinite(float(value))
                    or float(value) > float(config["numeric_tolerances"]["head_primal_max_abs"])
                    for value in primal_errors
                )
            ):
                raise RuntimeError("construction row has invalid final-head primal checks")
            if row["eligible"] and jvp_count != 2:
                raise RuntimeError("eligible construction row does not contain exactly two JVPs")
            expected_delta_jvps += jvp_count
            if row["eligible"] and (
                row.get("delta") is None or _raw_tensor_hash(row["delta"]) != row["delta_sha256"]
            ):
                raise RuntimeError("construction delta hash differs")
            if not row["eligible"] and (
                row.get("delta") is not None or row.get("delta_sha256") is not None
            ):
                raise RuntimeError("ineligible construction row contains a certified delta")
    if int(payload.get("compute", {}).get("delta_head_jvps", -1)) != expected_delta_jvps:
        raise RuntimeError("construction JVP ledger differs from eligible coverage")
    ledger = _load_attempt_ledger(CONSTRUCTION_ATTEMPT_LEDGER_PATH, phase="construction")
    _validate_attempt_coverage(
        ledger, _construction_expected_attempts(payload), phase="construction"
    )
    return payload


def run_construct() -> dict[str, Any]:
    config = _load_config()
    capture_bound = [
        *_protocol_bound_paths(config),
        CAPTURE_PATH,
        CAPTURE_MANIFEST_PATH,
        CAPTURE_ATTEMPT_LEDGER_PATH,
    ]
    _require_committed_clean(capture_bound, stage="paired capture")
    existing = [
        CONSTRUCTION_PATH.exists(),
        CONSTRUCTION_MANIFEST_PATH.exists(),
        CONSTRUCTION_ATTEMPT_LEDGER_PATH.exists(),
    ]
    if any(existing):
        if not all(existing):
            raise RuntimeError(
                "an interrupted construction left a partial artifact chain; fail closed "
                "without replay"
            )
        import torch

        _load_constructions(torch)
        return _validate_construction_manifest()
    adaptive, _ = _active_jobs()
    backend = adaptive.load_backend(adaptive.load_lock())
    torch = backend.torch
    capture = _load_capture(torch)
    _validate_capture_manifest()
    expected_pairs = int(config["opened_calibration"]["expected_active_pair_count"])
    if len(capture["pairs"]) != expected_pairs:
        raise RuntimeError("complete paired capture is required")
    started = time.perf_counter()
    candidates: list[dict[str, Any]] = []
    delta_head_jvps = 0
    attempt_ledger = _new_attempt_ledger("construction")
    _write_attempt_ledger(CONSTRUCTION_ATTEMPT_LEDGER_PATH, attempt_ledger)
    construction_attempt_ids: list[str] = []
    for candidate_index, reserve in enumerate(config["intervention"]["reserve_candidates"]):
        pair_records = []
        for pair in capture["pairs"]:
            pair_delta_jvps = 0
            cast_head_primal_errors: list[float] = []
            common = {
                "case_id": str(pair["case_id"]),
                "assignment": int(pair["assignment"]),
                "reserve_logit": float(reserve),
                "base_vector_sha256": pair["base_vector_sha256"],
            }
            try:
                if not bool(pair["bisector_eligible"]):
                    raise PairedOrderGradientIneligible(
                        "paired-order bisector is ineligible",
                        diagnostics=dict(pair["bisector_failure"]),
                    )
                logits = np.stack(
                    [row["baseline_logits"].double().numpy() for row in pair["orders"]]
                )
                base_jvps = np.stack(
                    [row["base_logit_jvp"].double().numpy() for row in pair["orders"]]
                )
                preserve_ids = tuple(int(row["preserve_token_id"]) for row in pair["orders"])
                comply_ids = tuple(int(row["comply_token_id"]) for row in pair["orders"])
                interval = full_vocabulary_bidirectional_interval(
                    logits,
                    base_jvps,
                    preserve_ids,
                    comply_ids,
                    reserve_logit=float(reserve),
                )
                requested_delta = np.asarray(
                    interval.lower * pair["base_vector"].double().numpy(), dtype=np.float32
                )
                delta_tensor = torch.from_numpy(requested_delta.copy()).float().contiguous()
                cast_jvps = []
                for order_index, order in enumerate(pair["orders"]):
                    work_id = _construction_attempt_work_id(
                        candidate_index,
                        str(pair["case_id"]),
                        int(pair["assignment"]),
                        order_index,
                    )
                    _reserve_attempt(
                        CONSTRUCTION_ATTEMPT_LEDGER_PATH,
                        attempt_ledger,
                        work_id=work_id,
                        operation="final_head_cast_delta_jvp",
                    )
                    primal, delta_jvp = _head_jvp(backend, order["residual"], delta_tensor)
                    _mark_attempt_returned(
                        CONSTRUCTION_ATTEMPT_LEDGER_PATH, attempt_ledger, work_id
                    )
                    construction_attempt_ids.append(work_id)
                    delta_head_jvps += 1
                    pair_delta_jvps += 1
                    head_error = float(
                        (primal.double() - order["baseline_logits"].double()).abs().max().item()
                    )
                    if head_error > float(config["numeric_tolerances"]["head_primal_max_abs"]):
                        raise RuntimeError(f"construction final-head primal mismatch: {head_error}")
                    cast_head_primal_errors.append(head_error)
                    cast_jvps.append(delta_jvp.double().numpy())
                recertified = cast_and_recertify_common_delta(
                    pair["base_vector"].double().numpy(),
                    interval.lower,
                    [float(row["residual_norm"]) for row in pair["orders"]],
                    logits,
                    base_jvps,
                    np.stack(cast_jvps),
                    preserve_ids,
                    comply_ids,
                    reserve_logit=float(reserve),
                    maximum_relative_norm=float(config["intervention"]["maximum_relative_norm"]),
                    cast_absolute_tolerance=float(
                        config["numeric_tolerances"]["cast_jvp_absolute"]
                    ),
                    cast_relative_tolerance=float(
                        config["numeric_tolerances"]["cast_jvp_relative"]
                    ),
                    margin_tolerance=float(config["numeric_tolerances"]["post_cast_margin"]),
                )
                delta = torch.from_numpy(recertified.delta.copy()).float().contiguous()
                record = {
                    **common,
                    "eligible": True,
                    "failure": None,
                    "delta_head_jvp_count": pair_delta_jvps,
                    "cast_head_primal_max_abs_errors": cast_head_primal_errors,
                    "alpha": float(interval.lower),
                    "interval_diagnostics": interval.diagnostics,
                    "recertification_diagnostics": recertified.diagnostics,
                    "delta": delta,
                    "delta_sha256": _tensor_hash(adaptive, delta),
                    "order_prompt_sha256s": [str(row["prompt_sha256"]) for row in pair["orders"]],
                }
            except PairedOrderGradientIneligible as error:
                record = {
                    **common,
                    "eligible": False,
                    "failure": error.diagnostics,
                    "delta_head_jvp_count": pair_delta_jvps,
                    "cast_head_primal_max_abs_errors": cast_head_primal_errors,
                    "alpha": None,
                    "interval_diagnostics": None,
                    "recertification_diagnostics": None,
                    "delta": None,
                    "delta_sha256": None,
                    "order_prompt_sha256s": [str(row["prompt_sha256"]) for row in pair["orders"]],
                }
            pair_records.append(record)
        pair_records.sort(key=lambda row: (str(row["case_id"]), int(row["assignment"])))
        candidates.append(
            {
                "reserve_logit": float(reserve),
                "eligible_pair_count": sum(bool(row["eligible"]) for row in pair_records),
                "pair_count": len(pair_records),
                "pairs": pair_records,
            }
        )
    payload = {
        "schema_version": "sp_lense.paired_order_analytic_gradient_preoutcome_bank.v1",
        "status": "complete",
        "development_only": True,
        "config_sha256": _sha256(CONFIG_PATH),
        "capture_file_sha256": _sha256(CAPTURE_PATH),
        "capture_manifest_sha256": _sha256(CAPTURE_MANIFEST_PATH),
        "compute": {"delta_head_jvps": delta_head_jvps},
        "candidates": candidates,
    }
    _atomic_torch_save(torch, CONSTRUCTION_PATH, payload)
    _commit_attempts(CONSTRUCTION_ATTEMPT_LEDGER_PATH, attempt_ledger, construction_attempt_ids)
    public_candidates = [
        {
            **{key: value for key, value in candidate.items() if key != "pairs"},
            "pairs": [_construction_public(row) for row in candidate["pairs"]],
        }
        for candidate in candidates
    ]
    manifest = {
        "schema_version": "sp_lense.paired_order_analytic_gradient_preoutcome_manifest.v1",
        "status": "complete",
        "development_only": True,
        "outcomes_viewed": False,
        "config_sha256": _sha256(CONFIG_PATH),
        "capture_file_sha256": _sha256(CAPTURE_PATH),
        "capture_manifest_sha256": _sha256(CAPTURE_MANIFEST_PATH),
        "construction_file_sha256": _sha256(CONSTRUCTION_PATH),
        "attempt_ledger_sha256": _sha256(CONSTRUCTION_ATTEMPT_LEDGER_PATH),
        "attempt_count": len(attempt_ledger["attempts"]),
        "candidate_count": len(candidates),
        "pair_count_per_candidate": expected_pairs,
        "compute": {
            "delta_head_jvps": delta_head_jvps,
            "full_model_forward_passes": 0,
            "full_model_backward_passes": 0,
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judge_calls": 0,
            "external_cost_usd": 0,
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
        },
        "candidates": public_candidates,
    }
    manifest["result_sha256"] = _canonical_sha256(manifest)
    _atomic_json(CONSTRUCTION_MANIFEST_PATH, manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "eligible": [row["eligible_pair_count"] for row in candidates],
                "compute": manifest["compute"],
            },
            indent=2,
        )
    )
    return manifest


def _evaluation_public_score(score: Any, exact_token_id: int) -> dict[str, Any]:
    return {
        "predicted_label": str(score.predicted_label),
        "exact_argmax_token_id": int(exact_token_id),
        "preserve_minus_comply_log_odds": float(score.preserve_log_odds),
        "preserve_pair_probability": float(score.preserve_pair_probability),
        "pair_choice": str(score.pair_choice),
        "answer_pair_mass": float(score.answer_pair_mass),
        "full_vocabulary_kl_changed_to_baseline": float(score.kl_from_baseline),
        "perturbation": score.perturbation,
        "choice_boundary_evidence_sha256": str(score.choice_boundary_evidence_sha256),
        "choice_a_token_id": int(score.choice_a_token_id),
        "choice_b_token_id": int(score.choice_b_token_id),
    }


def _preoutcome_bound_paths() -> list[Path]:
    return [
        CONFIG_PATH,
        ROOT / "docs" / "PAIRED_ORDER_ANALYTIC_GRADIENT_CONTROLLER_PREREGISTRATION.md",
        Path(__file__).resolve(),
        ROOT / "src" / "sp_lense" / "paired_order_analytic_gradient.py",
        CAPTURE_PATH,
        CAPTURE_MANIFEST_PATH,
        CAPTURE_ATTEMPT_LEDGER_PATH,
        CONSTRUCTION_PATH,
        CONSTRUCTION_MANIFEST_PATH,
        CONSTRUCTION_ATTEMPT_LEDGER_PATH,
    ]


def run_freeze() -> dict[str, Any]:
    _load_config()
    _validate_capture_manifest()
    _validate_construction_manifest()
    import torch

    _load_capture(torch)
    _load_constructions(torch)
    paths = _preoutcome_bound_paths()
    relatives = [_relative(path) for path in paths]
    for relative in relatives:
        _git("ls-files", "--error-unmatch", "--", relative)
    dirty = _git("status", "--porcelain", "--", *relatives)
    if dirty:
        raise RuntimeError(
            "preoutcome inputs must already be committed and clean before freeze creation"
        )
    payload = {
        "schema_version": "sp_lense.paired_order_analytic_gradient_preoutcome_freeze.v1",
        "status": "ready_for_opened_finite_evaluation",
        "outcomes_viewed": False,
        "config_sha256": _sha256(CONFIG_PATH),
        "construction_commit": _git("rev-parse", "HEAD"),
        "bound_files": {_relative(path): _sha256(path) for path in paths},
        "rule": "freeze_file_must_itself_be_committed_and_all_bound_paths_clean_before_evaluation",
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    _atomic_json(PREOUTCOME_FREEZE_PATH, payload)
    print(json.dumps(payload, indent=2))
    return payload


def _validate_preoutcome_freeze() -> dict[str, Any]:
    freeze = json.loads(PREOUTCOME_FREEZE_PATH.read_text(encoding="utf-8"))
    if (
        freeze.get("schema_version")
        != "sp_lense.paired_order_analytic_gradient_preoutcome_freeze.v1"
        or freeze.get("status") != "ready_for_opened_finite_evaluation"
        or freeze.get("outcomes_viewed") is not False
        or freeze.get("config_sha256") != _sha256(CONFIG_PATH)
    ):
        raise RuntimeError("preoutcome freeze has invalid identity or status")
    copy = dict(freeze)
    embedded = copy.pop("result_sha256", None)
    if embedded != _canonical_sha256(copy):
        raise RuntimeError("preoutcome freeze embedded result hash differs")
    expected = {_relative(path): _sha256(path) for path in _preoutcome_bound_paths()}
    if freeze.get("bound_files") != expected:
        raise RuntimeError("preoutcome freeze does not bind the current files")
    relatives = [*expected, _relative(PREOUTCOME_FREEZE_PATH)]
    for relative in relatives:
        _git("ls-files", "--error-unmatch", "--", relative)
    if _git("status", "--porcelain", "--", *relatives):
        raise RuntimeError("preoutcome freeze or one of its bound files is dirty")
    _git("merge-base", "--is-ancestor", str(freeze["construction_commit"]), "HEAD")
    return freeze


def _routing_audit(adaptive: Any, torch: Any) -> dict[str, Any]:
    config = _load_config()
    gate = _source_gate(config)
    confirmation = _load_confirmation()
    _, jobs, collateral = confirmation._inputs()
    active_keys = {
        (str(row["case_id"]), int(row["assignment"]))
        for row in gate["pair_rows"]
        if row["target"] == "self" and bool(row["predicted_active"])
    }
    rows = [
        {
            "unit_id": str(job["unit_id"]),
            "target": str(job["target"]),
            "prompt_sha256": str(job["prompt_sha256"]),
            "active": bool(
                job["target"] == "self"
                and (str(job["case_id"]), int(job["assignment"])) in active_keys
            ),
        }
        for job in jobs
    ]
    collateral_rows = [
        {
            "form_id": str(form["form_id"]),
            "prompt_sha256": str(form["prompt_sha256"]),
            "active": False,
        }
        for form in collateral
    ]
    zero = torch.zeros(int(config["model"]["residual_width"]), dtype=torch.float32)
    audit = {
        "router_scope": "locked_structured_dataset_renderer_not_open_domain_free_text",
        "decision_order_count": len(rows),
        "active_decision_order_count": sum(bool(row["active"]) for row in rows),
        "zero_routed_decision_order_count": sum(not bool(row["active"]) for row in rows),
        "zero_routed_collateral_count": len(collateral_rows),
        "nonzero_off_gate_count": 0,
        "off_gate_assigned_strength": 0.0,
        "zero_delta_sha256": _tensor_hash(adaptive, zero),
        "rows_sha256": _canonical_sha256(rows),
        "collateral_rows_sha256": _canonical_sha256(collateral_rows),
    }
    if (
        audit["decision_order_count"] != 128
        or audit["active_decision_order_count"] != 32
        or audit["zero_routed_decision_order_count"] != 96
        or audit["zero_routed_collateral_count"] != 16
    ):
        raise RuntimeError("structured routing audit differs from the locked coverage")
    audit["audit_sha256"] = _canonical_sha256(audit)
    return audit


def _semantic_choice(score: Any, job: Mapping[str, Any]) -> str:
    if score.predicted_label == job["positive_label"]:
        return "positive"
    if score.predicted_label == job["negative_label"]:
        return "negative"
    return "OTHER"


_EVALUATION_CELL_FIELDS = frozenset(
    {
        "cell_schema_version",
        "work_id",
        "ordinal",
        "candidate_index",
        "reserve_logit",
        "case_id",
        "assignment",
        "order_index",
        "preserve_first",
        "prompt_sha256",
        "condition",
        "sign",
        "wanted_semantic_choice",
        "baseline_semantic_choice",
        "baseline_argmax_token_id",
        "unsigned_delta_sha256",
        "signed_delta_sha256",
        "intended_delta_l2_norm",
        "realized_delta_l2_norm",
        "realized_norm_absolute_error",
        "realized_norm_allowance",
        "predicted_label",
        "exact_argmax_token_id",
        "preserve_minus_comply_log_odds",
        "preserve_pair_probability",
        "pair_choice",
        "answer_pair_mass",
        "full_vocabulary_kl_changed_to_baseline",
        "perturbation",
        "choice_boundary_evidence_sha256",
        "choice_a_token_id",
        "choice_b_token_id",
        "semantic_choice",
        "target_met",
        "decision_changed",
        "previous_cell_sha256",
        "cell_sha256",
    }
)


def _seal_evaluation_cell(cell: dict[str, Any], previous_sha256: str | None) -> None:
    cell["cell_schema_version"] = "sp_lense.paired_order_analytic_gradient_cell.v1"
    cell["previous_cell_sha256"] = previous_sha256
    cell.pop("cell_sha256", None)
    cell["cell_sha256"] = _canonical_sha256(cell)


def _write_evaluation_checkpoint(checkpoint: dict[str, Any]) -> None:
    checkpoint["cells_sha256"] = _canonical_sha256(checkpoint["cells"])
    checkpoint.pop("checkpoint_sha256", None)
    checkpoint["checkpoint_sha256"] = _canonical_sha256(checkpoint)
    _atomic_json(EVALUATION_CHECKPOINT_PATH, checkpoint)


def _strict_expected(row: Mapping[str, Any], expected: Mapping[str, Any], work_id: str) -> None:
    for field, value in expected.items():
        observed = row.get(field)
        if type(observed) is not type(value) or observed != value:
            raise RuntimeError(
                f"stored evaluation cell field is not derived from its work item: {work_id}:{field}"
            )


def _validate_evaluation_cell(
    row: Mapping[str, Any],
    work: Mapping[str, Any],
    adaptive: Any,
    *,
    ordinal: int,
    previous_sha256: str | None,
) -> None:
    work_id = str(row.get("work_id"))
    if set(row) != _EVALUATION_CELL_FIELDS:
        raise RuntimeError(f"stored evaluation cell has the wrong exact schema: {work_id}")
    if row["cell_schema_version"] != "sp_lense.paired_order_analytic_gradient_cell.v1":
        raise RuntimeError(f"stored evaluation cell has the wrong schema: {work_id}")
    if row["previous_cell_sha256"] != previous_sha256:
        raise RuntimeError(f"stored evaluation cell hash chain differs: {work_id}")
    copy = dict(row)
    embedded = copy.pop("cell_sha256")
    if embedded != _canonical_sha256(copy):
        raise RuntimeError(f"stored evaluation cell embedded hash differs: {work_id}")

    construction = work["construction"]
    captured_order = work["captured_order"]
    job = work["job"]
    exact_id = row["exact_argmax_token_id"]
    if type(exact_id) is not int or exact_id < 0:
        raise RuntimeError(f"stored exact argmax token ID is invalid: {work_id}")
    preserve_id = int(captured_order["preserve_token_id"])
    comply_id = int(captured_order["comply_token_id"])
    if exact_id == preserve_id:
        expected_label = str(job["positive_label"])
        expected_semantic = "positive"
    elif exact_id == comply_id:
        expected_label = str(job["negative_label"])
        expected_semantic = "negative"
    else:
        expected_label = "OTHER"
        expected_semantic = "OTHER"
    signed_delta = float(work["sign"]) * construction["delta"].detach().float()
    expected_fields = {
        "work_id": str(work["work_id"]),
        "ordinal": int(ordinal),
        "candidate_index": int(work["candidate_index"]),
        "reserve_logit": float(work["reserve_logit"]),
        "case_id": str(work["key"][0]),
        "assignment": int(work["key"][1]),
        "order_index": int(work["order_index"]),
        "preserve_first": bool(captured_order["preserve_first"]),
        "prompt_sha256": str(captured_order["prompt_sha256"]),
        "condition": str(work["name"]),
        "sign": float(work["sign"]),
        "wanted_semantic_choice": str(work["wanted"]),
        "baseline_semantic_choice": str(captured_order["baseline_semantic_choice"]),
        "baseline_argmax_token_id": int(captured_order["baseline_argmax_token_id"]),
        "unsigned_delta_sha256": str(construction["delta_sha256"]),
        "signed_delta_sha256": _tensor_hash(adaptive, signed_delta),
        "predicted_label": expected_label,
        "semantic_choice": expected_semantic,
        "target_met": expected_semantic == work["wanted"],
        "decision_changed": expected_semantic != str(captured_order["baseline_semantic_choice"]),
        "choice_boundary_evidence_sha256": str(captured_order["choice_boundary_evidence_sha256"]),
    }
    positive_label = str(job["positive_label"])
    expected_fields["choice_a_token_id"] = preserve_id if positive_label == "A" else comply_id
    expected_fields["choice_b_token_id"] = comply_id if positive_label == "A" else preserve_id
    _strict_expected(row, expected_fields, work_id)

    numeric_fields = (
        "intended_delta_l2_norm",
        "realized_delta_l2_norm",
        "realized_norm_absolute_error",
        "realized_norm_allowance",
        "preserve_minus_comply_log_odds",
        "preserve_pair_probability",
        "answer_pair_mass",
        "full_vocabulary_kl_changed_to_baseline",
    )
    if any(
        type(row[field]) is not float or not math.isfinite(row[field]) for field in numeric_fields
    ):
        raise RuntimeError(f"stored evaluation cell contains a non-finite scalar: {work_id}")
    log_odds = float(row["preserve_minus_comply_log_odds"])
    probability = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, log_odds))))
    if not math.isclose(
        float(row["preserve_pair_probability"]), probability, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise RuntimeError(f"stored pair probability disagrees with log-odds: {work_id}")
    if not 0.0 <= float(row["answer_pair_mass"]) <= 1.0:
        raise RuntimeError(f"stored answer-pair mass is outside [0,1]: {work_id}")
    if float(row["full_vocabulary_kl_changed_to_baseline"]) < -1e-6:
        raise RuntimeError(f"stored KL is materially negative: {work_id}")
    expected_pair_choice = (
        str(job["positive_label"]) if log_odds >= 0.0 else str(job["negative_label"])
    )
    if row["pair_choice"] != expected_pair_choice:
        raise RuntimeError(f"stored pair choice disagrees with log-odds: {work_id}")
    if (expected_semantic == "positive" and log_odds < 0.0) or (
        expected_semantic == "negative" and log_odds > 0.0
    ):
        raise RuntimeError(f"stored exact argmax disagrees with semantic log-odds: {work_id}")

    intended = float(construction["delta"].detach().double().norm().item())
    realized = float(row["realized_delta_l2_norm"])
    norm_error = abs(realized - intended)
    norm_allowance = 1e-5 + 1e-5 * max(1.0, intended)
    for field, expected_value in (
        ("intended_delta_l2_norm", intended),
        ("realized_norm_absolute_error", norm_error),
        ("realized_norm_allowance", norm_allowance),
    ):
        if not math.isclose(float(row[field]), expected_value, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"stored perturbation norm identity differs: {work_id}:{field}")
    if norm_error > norm_allowance:
        raise RuntimeError(f"stored realized perturbation exceeds its norm allowance: {work_id}")
    perturbation = row["perturbation"]
    expected_perturbation_fields = {
        "n_positions",
        "total_frobenius_norm",
        "mean_l2_norm",
        "rms_l2_norm",
        "max_l2_norm",
        "mean_relative_l2_norm",
        "max_relative_l2_norm",
        "zero_reference_positions",
    }
    if not isinstance(perturbation, Mapping) or set(perturbation) != expected_perturbation_fields:
        raise RuntimeError(f"stored perturbation diagnostics have the wrong schema: {work_id}")
    if type(perturbation["n_positions"]) is not int or perturbation["n_positions"] != 1:
        raise RuntimeError(f"stored perturbation changed the wrong number of positions: {work_id}")
    if (
        type(perturbation["zero_reference_positions"]) is not int
        or perturbation["zero_reference_positions"] != 0
    ):
        raise RuntimeError(f"stored perturbation has a zero reference position: {work_id}")
    for field in (
        "total_frobenius_norm",
        "mean_l2_norm",
        "rms_l2_norm",
        "max_l2_norm",
    ):
        value = perturbation[field]
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not math.isclose(value, realized, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise RuntimeError(f"stored one-position perturbation norm differs: {work_id}:{field}")
    residual_norm = float(captured_order["residual_norm"])
    expected_relative = realized / residual_norm
    for field in ("mean_relative_l2_norm", "max_relative_l2_norm"):
        value = perturbation[field]
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not math.isclose(value, expected_relative, rel_tol=1e-5, abs_tol=1e-5)
        ):
            raise RuntimeError(f"stored relative perturbation norm differs: {work_id}:{field}")


def _validate_evaluation_checkpoint(
    checkpoint: Mapping[str, Any], expected_work: Mapping[str, Mapping[str, Any]], adaptive: Any
) -> None:
    expected_fields = {
        "schema_version",
        "status",
        "config_sha256",
        "construction_file_sha256",
        "construction_manifest_sha256",
        "preoutcome_freeze_sha256",
        "expected_work_sha256",
        "pending_reservation",
        "cells",
        "compute",
        "cells_sha256",
        "checkpoint_sha256",
    }
    if set(checkpoint) != expected_fields:
        raise RuntimeError("evaluation checkpoint has the wrong exact schema")
    if checkpoint.get("schema_version") != (
        "sp_lense.paired_order_analytic_gradient_evaluation_checkpoint.v3"
    ):
        raise RuntimeError("evaluation checkpoint has the wrong schema")
    if checkpoint.get("status") not in {"in_progress", "complete"}:
        raise RuntimeError("evaluation checkpoint has an invalid status")
    compute = checkpoint.get("compute")
    expected_compute_fields = {
        "reserved_intervention_forward_passes",
        "completed_intervention_forward_passes",
    }
    if not isinstance(compute, Mapping) or set(compute) != expected_compute_fields:
        raise RuntimeError("evaluation checkpoint compute ledger has the wrong schema")
    if any(type(compute[field]) is not int or compute[field] < 0 for field in compute):
        raise RuntimeError("evaluation checkpoint compute counters are invalid")
    copy = dict(checkpoint)
    embedded = copy.pop("checkpoint_sha256", None)
    if embedded != _canonical_sha256(copy):
        raise RuntimeError("evaluation checkpoint embedded hash differs")
    if checkpoint.get("cells_sha256") != _canonical_sha256(checkpoint.get("cells")):
        raise RuntimeError("evaluation checkpoint cell-list hash differs")
    expected_ids = list(expected_work)
    cells = checkpoint.get("cells")
    if not isinstance(cells, list):
        raise TypeError("evaluation checkpoint cells are not a list")
    observed_ids = [str(row.get("work_id")) for row in cells]
    if observed_ids != expected_ids[: len(observed_ids)]:
        raise RuntimeError("evaluation checkpoint cells are not the exact work-plan prefix")
    previous = None
    for ordinal, row in enumerate(cells):
        _validate_evaluation_cell(
            row,
            expected_work[observed_ids[ordinal]],
            adaptive,
            ordinal=ordinal,
            previous_sha256=previous,
        )
        previous = str(row["cell_sha256"])
    completed = compute["completed_intervention_forward_passes"]
    reserved = compute["reserved_intervention_forward_passes"]
    if completed != len(cells):
        raise RuntimeError("completed intervention ledger differs from stored cells")
    pending = checkpoint.get("pending_reservation")
    if pending is None:
        if reserved != completed:
            raise RuntimeError("reserved intervention ledger differs without a pending call")
    else:
        expected_pending = {
            "work_id": expected_ids[len(cells)] if len(cells) < len(expected_ids) else None,
            "ordinal": len(cells),
            "operation": "intervention_forward",
        }
        if pending != expected_pending or reserved != completed + 1:
            raise RuntimeError("pending intervention reservation differs from the next work item")
    if checkpoint.get("status") == "complete" and (
        pending is not None or len(cells) != len(expected_ids)
    ):
        raise RuntimeError("complete evaluation checkpoint does not contain the complete work plan")


def _aggregate_evaluation(
    constructions: Mapping[str, Any], capture: Mapping[str, Any], cells: Mapping[str, Any]
) -> list[dict[str, Any]]:
    capture_by_key = {
        (str(pair["case_id"]), int(pair["assignment"])): pair for pair in capture["pairs"]
    }
    candidates = []
    for candidate_index, candidate in enumerate(constructions["candidates"]):
        pairs = []
        if int(candidate["eligible_pair_count"]) == int(candidate["pair_count"]):
            for construction in candidate["pairs"]:
                key = (str(construction["case_id"]), int(construction["assignment"]))
                orders = []
                for order_index, captured_order in enumerate(capture_by_key[key]["orders"]):
                    outcomes = {
                        name: cells[f"{candidate_index}:{key[0]}:{key[1]}:{order_index}:{name}"]
                        for name in ("plus", "minus")
                    }
                    orders.append(
                        {
                            "preserve_first": bool(captured_order["preserve_first"]),
                            "prompt_sha256": str(captured_order["prompt_sha256"]),
                            "baseline_semantic_choice": str(
                                captured_order["baseline_semantic_choice"]
                            ),
                            "baseline_argmax_token_id": int(
                                captured_order["baseline_argmax_token_id"]
                            ),
                            "outcomes": outcomes,
                        }
                    )
                all_targets = all(
                    outcome["target_met"]
                    for order in orders
                    for outcome in order["outcomes"].values()
                )
                flip_each_order = all(
                    order["outcomes"]["plus"]["decision_changed"]
                    or order["outcomes"]["minus"]["decision_changed"]
                    for order in orders
                )
                no_other = all(
                    outcome["semantic_choice"] != "OTHER"
                    for order in orders
                    for outcome in order["outcomes"].values()
                )
                same_delta = all(
                    outcome["unsigned_delta_sha256"] == construction["delta_sha256"]
                    for order in orders
                    for outcome in order["outcomes"].values()
                )
                pairs.append(
                    {
                        "case_id": key[0],
                        "assignment": key[1],
                        "delta_sha256": str(construction["delta_sha256"]),
                        "alpha": float(construction["alpha"]),
                        "passes": bool(all_targets and flip_each_order and no_other and same_delta),
                        "all_targets_met": bool(all_targets),
                        "real_flip_in_each_answer_order": bool(flip_each_order),
                        "no_other": bool(no_other),
                        "same_unsigned_delta_hash_all_four_cells": bool(same_delta),
                        "orders": orders,
                    }
                )
        result = {
            "reserve_logit": float(candidate["reserve_logit"]),
            "construction_eligible_pair_count": int(candidate["eligible_pair_count"]),
            "evaluated_pair_count": len(pairs),
            "successful_pair_count": sum(bool(row["passes"]) for row in pairs),
            "passes": len(pairs) == int(candidate["pair_count"])
            and all(bool(row["passes"]) for row in pairs),
            "pairs": pairs,
        }
        candidates.append(result)
    return candidates


def run_evaluate() -> dict[str, Any]:
    config = _load_config()
    _validate_capture_manifest()
    _validate_construction_manifest()
    freeze = _validate_preoutcome_freeze()
    adaptive, groups = _active_jobs()
    backend = adaptive.load_backend(adaptive.load_lock())
    torch = backend.torch
    capture = _load_capture(torch)
    constructions = _load_constructions(torch)
    job_by_key = {
        (key[0], key[1], bool(job["preserve_first"])): job for key, jobs in groups for job in jobs
    }
    capture_by_key = {
        (str(pair["case_id"]), int(pair["assignment"])): pair for pair in capture["pairs"]
    }
    expected_work: dict[str, dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(constructions["candidates"]):
        if int(candidate["eligible_pair_count"]) != int(candidate["pair_count"]):
            continue
        for construction in candidate["pairs"]:
            key = (str(construction["case_id"]), int(construction["assignment"]))
            for order_index, captured_order in enumerate(capture_by_key[key]["orders"]):
                for name, sign, wanted in (
                    ("plus", 1.0, "positive"),
                    ("minus", -1.0, "negative"),
                ):
                    work_id = f"{candidate_index}:{key[0]}:{key[1]}:{order_index}:{name}"
                    expected_work[work_id] = {
                        "work_id": work_id,
                        "candidate_index": candidate_index,
                        "reserve_logit": float(candidate["reserve_logit"]),
                        "construction": construction,
                        "key": key,
                        "captured_order": captured_order,
                        "job": job_by_key[(key[0], key[1], bool(captured_order["preserve_first"]))],
                        "order_index": order_index,
                        "name": name,
                        "sign": sign,
                        "wanted": wanted,
                    }
    public_work_plan = [
        {
            "work_id": work_id,
            "candidate_index": int(work["candidate_index"]),
            "reserve_logit": float(work["reserve_logit"]),
            "case_id": str(work["key"][0]),
            "assignment": int(work["key"][1]),
            "order_index": int(work["order_index"]),
            "prompt_sha256": str(work["captured_order"]["prompt_sha256"]),
            "condition": str(work["name"]),
            "sign": float(work["sign"]),
            "unsigned_delta_sha256": str(work["construction"]["delta_sha256"]),
        }
        for work_id, work in expected_work.items()
    ]
    work_plan_sha256 = _canonical_sha256(public_work_plan)
    checkpoint = (
        json.loads(EVALUATION_CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if EVALUATION_CHECKPOINT_PATH.exists()
        else {
            "schema_version": "sp_lense.paired_order_analytic_gradient_evaluation_checkpoint.v3",
            "status": "in_progress",
            "config_sha256": _sha256(CONFIG_PATH),
            "construction_file_sha256": _sha256(CONSTRUCTION_PATH),
            "construction_manifest_sha256": _sha256(CONSTRUCTION_MANIFEST_PATH),
            "preoutcome_freeze_sha256": _sha256(PREOUTCOME_FREEZE_PATH),
            "expected_work_sha256": work_plan_sha256,
            "pending_reservation": None,
            "cells": [],
            "compute": {
                "reserved_intervention_forward_passes": 0,
                "completed_intervention_forward_passes": 0,
            },
        }
    )
    if (
        checkpoint.get("schema_version")
        != "sp_lense.paired_order_analytic_gradient_evaluation_checkpoint.v3"
        or checkpoint.get("status") not in {"in_progress", "complete"}
        or checkpoint.get("config_sha256") != _sha256(CONFIG_PATH)
        or checkpoint.get("construction_file_sha256") != _sha256(CONSTRUCTION_PATH)
        or checkpoint.get("construction_manifest_sha256") != _sha256(CONSTRUCTION_MANIFEST_PATH)
        or checkpoint.get("preoutcome_freeze_sha256") != _sha256(PREOUTCOME_FREEZE_PATH)
        or checkpoint.get("expected_work_sha256") != work_plan_sha256
    ):
        raise RuntimeError("evaluation checkpoint belongs to another frozen study")
    if not EVALUATION_CHECKPOINT_PATH.exists():
        _write_evaluation_checkpoint(checkpoint)
    _validate_evaluation_checkpoint(checkpoint, expected_work, adaptive)
    if checkpoint.get("pending_reservation") is not None:
        raise RuntimeError(
            "a prior intervention call ended ambiguously; fail closed rather than replaying it"
        )
    completed = {str(row["work_id"]): row for row in checkpoint.get("cells", [])}

    started = time.perf_counter()
    layer = int(config["intervention"]["residual_layer_zero_based"])
    for work_id, work in expected_work.items():
        if work_id in completed:
            continue
        checkpoint["pending_reservation"] = {
            "work_id": work_id,
            "ordinal": len(checkpoint["cells"]),
            "operation": "intervention_forward",
        }
        checkpoint["compute"]["reserved_intervention_forward_passes"] += 1
        _write_evaluation_checkpoint(checkpoint)
        construction = work["construction"]
        captured_order = work["captured_order"]
        job = work["job"]
        delta = construction["delta"].detach().float().cpu().contiguous()
        spec = adaptive.InterventionSpec(
            layer=layer,
            direction=delta.to(backend.device),
            strength=float(work["sign"]),
            geometry="matched_final_prompt",
            prompt_length=int(captured_order["prompt_length"]),
            magnitude_mode="canonical_coefficient",
        )
        score, _, exact_id = adaptive._score_choice_with_exact_argmax(
            backend,
            str(job["prompt"]),
            str(job["positive_label"]),
            str(job["negative_label"]),
            spec,
            baseline_logits=captured_order["baseline_logits"],
        )
        perturbation = score.perturbation
        delta_norm = float(delta.double().norm().item())
        if perturbation is None or int(perturbation["n_positions"]) != 1:
            raise RuntimeError("intervention did not report one realized changed position")
        norm_error = abs(float(perturbation["mean_l2_norm"]) - delta_norm)
        norm_allowance = 1e-5 + 1e-5 * max(1.0, delta_norm)
        if norm_error > norm_allowance:
            raise RuntimeError("realized perturbation norm differs from the intended delta")
        semantic = _semantic_choice(score, job)
        signed_delta = float(work["sign"]) * delta
        cell = {
            "work_id": work_id,
            "ordinal": len(checkpoint["cells"]),
            "candidate_index": int(work["candidate_index"]),
            "reserve_logit": float(work["reserve_logit"]),
            "case_id": str(work["key"][0]),
            "assignment": int(work["key"][1]),
            "order_index": int(work["order_index"]),
            "preserve_first": bool(captured_order["preserve_first"]),
            "prompt_sha256": str(captured_order["prompt_sha256"]),
            "condition": str(work["name"]),
            "sign": float(work["sign"]),
            "wanted_semantic_choice": str(work["wanted"]),
            "baseline_semantic_choice": str(captured_order["baseline_semantic_choice"]),
            "baseline_argmax_token_id": int(captured_order["baseline_argmax_token_id"]),
            "unsigned_delta_sha256": str(construction["delta_sha256"]),
            "signed_delta_sha256": _tensor_hash(adaptive, signed_delta),
            "intended_delta_l2_norm": delta_norm,
            "realized_delta_l2_norm": float(perturbation["mean_l2_norm"]),
            "realized_norm_absolute_error": norm_error,
            "realized_norm_allowance": norm_allowance,
            **_evaluation_public_score(score, exact_id),
            "semantic_choice": semantic,
            "target_met": semantic == work["wanted"],
            "decision_changed": semantic != str(captured_order["baseline_semantic_choice"]),
        }
        _seal_evaluation_cell(
            cell,
            None if not checkpoint["cells"] else checkpoint["cells"][-1]["cell_sha256"],
        )
        _validate_evaluation_cell(
            cell,
            work,
            adaptive,
            ordinal=len(checkpoint["cells"]),
            previous_sha256=(
                None if not checkpoint["cells"] else str(checkpoint["cells"][-1]["cell_sha256"])
            ),
        )
        checkpoint["cells"].append(cell)
        checkpoint["pending_reservation"] = None
        checkpoint["compute"]["completed_intervention_forward_passes"] += 1
        _write_evaluation_checkpoint(checkpoint)
        completed[work_id] = cell
        if len(completed) % 16 == 0 or len(completed) == len(expected_work):
            print(
                f"completed {len(completed)}/{len(expected_work)} frozen interventions", flush=True
            )
    if set(completed) != set(expected_work):
        raise RuntimeError("evaluation work remains incomplete")
    checkpoint["status"] = "complete"
    _write_evaluation_checkpoint(checkpoint)
    _validate_evaluation_checkpoint(checkpoint, expected_work, adaptive)

    candidates = _aggregate_evaluation(constructions, capture, completed)
    for candidate in candidates:
        print(
            json.dumps(
                {
                    key: candidate[key]
                    for key in (
                        "reserve_logit",
                        "construction_eligible_pair_count",
                        "successful_pair_count",
                        "passes",
                    )
                }
            ),
            flush=True,
        )
    selected = next((row for row in candidates if bool(row["passes"])), None)
    all_scores = [
        outcome
        for candidate in candidates
        for pair in candidate["pairs"]
        for order in pair["orders"]
        for outcome in order["outcomes"].values()
    ]
    kls = [float(row["full_vocabulary_kl_changed_to_baseline"]) for row in all_scores]
    result = {
        "schema_version": "sp_lense.paired_order_analytic_gradient_development_result.v1",
        "status": "passed" if selected is not None else "failed",
        "development_only": True,
        "fresh_prospective_evidence": False,
        "global_reserve_selection_uses_opened_outcomes": True,
        "per_pair_outcome_adaptive_selection": False,
        "config_sha256": _sha256(CONFIG_PATH),
        "source_gate_file_sha256": _sha256(SOURCE_GATE_PATH),
        "capture_file_sha256": _sha256(CAPTURE_PATH),
        "capture_manifest_sha256": _sha256(CAPTURE_MANIFEST_PATH),
        "preoutcome_construction_file_sha256": _sha256(CONSTRUCTION_PATH),
        "preoutcome_construction_manifest_sha256": _sha256(CONSTRUCTION_MANIFEST_PATH),
        "preoutcome_freeze_file_sha256": _sha256(PREOUTCOME_FREEZE_PATH),
        "evaluation_checkpoint_file_sha256": _sha256(EVALUATION_CHECKPOINT_PATH),
        "evaluation_cells_sha256": str(checkpoint["cells_sha256"]),
        "preoutcome_construction_commit": str(freeze["construction_commit"]),
        "selected_global_reserve_logit": (
            None if selected is None else float(selected["reserve_logit"])
        ),
        "selected_successful_pair_count": (
            0 if selected is None else int(selected["successful_pair_count"])
        ),
        "required_pair_count": int(config["opened_calibration"]["expected_active_pair_count"]),
        "target_kl_descriptive_all_candidates": {
            "row_count": len(kls),
            "mean": statistics.fmean(kls) if kls else None,
            "p95": float(np.quantile(kls, 0.95, method="inverted_cdf")) if kls else None,
            "maximum": max(kls) if kls else None,
        },
        "off_gate": {
            "intervention_strength": 0.0,
            "source_gate_pair_counts": config["source_gate"]["pair_counts"],
            "selectivity_source": "locked_structured_zero_routing_not_intrinsic_delta_specificity",
            "routing_audit": _routing_audit(adaptive, torch),
        },
        "compute": {
            "full_forward_passes": int(capture["compute"]["full_forward_passes"]),
            "backward_passes": int(capture["compute"]["backward_passes"]),
            "base_head_jvps": int(capture["compute"]["base_head_jvps"]),
            "delta_head_jvps": int(constructions["compute"]["delta_head_jvps"]),
            **checkpoint["compute"],
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judge_calls": 0,
            "external_cost_usd": 0,
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
        },
        "candidates": candidates,
    }
    result["result_sha256"] = _canonical_sha256(result)
    _atomic_json(RESULT_PATH, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "selected_global_reserve_logit",
                    "selected_successful_pair_count",
                    "compute",
                )
            },
            indent=2,
        )
    )
    return result


def run_report() -> str:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8")) if RESULT_PATH.exists() else None
    lines = [
        "# Paired-order analytic-gradient opened development",
        "",
        "This phase uses already-opened `confirmation_v2` prompts and is not prospective evidence.",
        "",
    ]
    if result is None:
        lines.append("No finite evaluation result is available.")
    else:
        lines.extend(
            (
                f"Status: **{result['status']}**.",
                "",
                f"Selected global reserve: `{result['selected_global_reserve_logit']}`.",
                "",
                f"Successful active pairs at that reserve: {result['selected_successful_pair_count']}/{result['required_pair_count']}.",
                "",
                "The same float32 physical delta was used under both answer orders of each pair, and every dose was fixed from unsteered logits/JVPs before changed outcomes were viewed.",
            )
        )
    lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            "A pass qualifies a privileged, prompt-local, forced-choice controller for a genuinely fresh test. It does not itself establish prospective replication, a natural self-preservation mechanism, intrinsic vector specificity, open-ended behavior, another model, or publication-level significance. Off-gate selectivity is exact zero routing by the controller.",
            "",
        )
    )
    text = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)
    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run paired-order analytic-gradient opened development."
    )
    parser.add_argument("command", choices=("capture", "construct", "freeze", "evaluate", "report"))
    args = parser.parse_args(argv)
    {
        "capture": run_capture,
        "construct": run_construct,
        "freeze": run_freeze,
        "evaluate": run_evaluate,
        "report": run_report,
    }[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
