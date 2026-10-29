"""Databricks-only admission and evidence binding for transfer evaluation.

This module never starts compute.  It validates an immutable, one-use
evaluation intent and recomputes every scientific input that a Databricks GPU
worker may consume: deployed source, Qwen snapshot, frozen-base qualification,
one seedless base plus nine audited adapters, the sealed fixture registries,
and the fixed resource envelope.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import math
import re

from .action_serialization import SERIALIZATION_CONTRACT_VERSION
from .continuity_transfer_evaluation import (
    ALL_ARMS, EXPECTED_RAW_RECORDS, digest, file_sha256,
    validate_public_commitment, validate_registry_bundle,
)
from .continuity_transfer_inference_worker import (
    validate_databricks_run_binding, verify_checkpoint_manifest,
)
from .databricks_simulated_continuity import (
    MODEL_RECEIPT, MODEL_ROOT, VOLUME, build_source_manifest, canonical,
    validate_base_qualification_intent, validate_base_qualification_receipt,
    validate_source_manifest,
    verify_deployed_source, verify_model_receipt,
    verify_model_receipt_authority,
)
from .base_qualification_carryforward import validate_carryforward
from .qualification_technical_successor import validate as validate_technical_successor
from .qwen9b_ordinal_continuity_runtime import directory_tree_sha256


TRANSFER_INTENT_VERSION = "research4-databricks-continuity-transfer-intent-v4"
TRANSFER_RESOURCE_PLAN_VERSION = "research4-databricks-continuity-transfer-resource-v2"
TRANSFER_RUN_BINDING_VERSION = "research4-databricks-continuity-transfer-run-binding-v4"
TRANSFER_AUDIT_VERSION = "research4-databricks-continuity-transfer-audit-v4"
TRANSFER_IDENTITY = "continuity_transfer_evaluation_v3"
TRANSFER_GOLD_TABLE = (
    "research4_simulated_continuity.gold.continuity_transfer_evaluation_audits"
)
TRANSFER_MLFLOW_EXPERIMENT = "/Shared/research4_continuity_transfer_evaluation"
TRANSFER_OUTPUT_ROOT = f"{VOLUME}/evaluations/{TRANSFER_IDENTITY}"
TRANSFER_LOCK_ROOT = f"{VOLUME}/locks"


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _volume_path(value: Any, *, description: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/Volumes/"):
        raise PermissionError(f"{description} must be an exact Unity Catalog Volume path")
    path = Path(value)
    if ".." in path.parts:
        raise PermissionError(f"{description} contains a parent traversal")
    return path


def validate_transfer_resource_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require one bounded A10 evaluation with measured throughput authority."""
    plan = dict(value)
    exact_keys = {
        "version", "accelerator", "gpu_count", "max_concurrent_runs",
        "automatic_retries", "episode_count", "max_new_tokens_per_episode",
        "batch_size", "max_batched_model_forward_calls", "max_raw_export_bytes",
        "wall_clock_stop_seconds", "wall_clock_ceiling_seconds",
        "protected_audit_and_export_seconds", "max_gpu_hours",
        "qualification_receipt_sha256", "qualification_episode_count",
        "qualification_elapsed_seconds", "minimum_measured_episodes_per_second",
        "serialization_contract_version", "resource_plan_sha256",
    }
    payload = dict(plan); embedded = payload.pop("resource_plan_sha256", None)
    numeric = (
        "qualification_elapsed_seconds", "minimum_measured_episodes_per_second",
        "max_gpu_hours",
    )
    if (set(plan) != exact_keys
            or embedded != digest(payload)
            or plan.get("version") != TRANSFER_RESOURCE_PLAN_VERSION
            or plan.get("serialization_contract_version")
               != SERIALIZATION_CONTRACT_VERSION
            or plan.get("accelerator") != "GPU_1xA10"
            or plan.get("gpu_count") != 1
            or plan.get("max_concurrent_runs") != 1
            or plan.get("automatic_retries") != 0
            or plan.get("episode_count") != EXPECTED_RAW_RECORDS
            or plan.get("max_new_tokens_per_episode") != 32
            or type(plan.get("batch_size")) is not int
            or not 1 <= plan["batch_size"] <= 32
            or type(plan.get("max_batched_model_forward_calls")) is not int
            or plan["max_batched_model_forward_calls"] != math.ceil(
                EXPECTED_RAW_RECORDS / plan["batch_size"]
            ) * plan["max_new_tokens_per_episode"]
            or type(plan.get("max_raw_export_bytes")) is not int
            or plan["max_raw_export_bytes"] < EXPECTED_RAW_RECORDS * 128
            or type(plan.get("wall_clock_stop_seconds")) is not int
            or type(plan.get("wall_clock_ceiling_seconds")) is not int
            or type(plan.get("protected_audit_and_export_seconds")) is not int
            or not 600 <= plan["protected_audit_and_export_seconds"]
            or not 600 <= plan["wall_clock_stop_seconds"]
            or plan["wall_clock_stop_seconds"] > (
                plan["wall_clock_ceiling_seconds"]
                - plan["protected_audit_and_export_seconds"])
            or plan["wall_clock_ceiling_seconds"] > 14_400
            or any(not isinstance(plan.get(key), (int, float))
                   or not math.isfinite(float(plan[key])) or float(plan[key]) <= 0
                   for key in numeric)
            or not _sha(plan.get("qualification_receipt_sha256"))
            or type(plan.get("qualification_episode_count")) is not int
            or plan["qualification_episode_count"] <= 0):
        raise PermissionError("Exact bounded Databricks transfer resource plan required")
    observed = (plan["qualification_episode_count"]
                / float(plan["qualification_elapsed_seconds"]))
    if (plan["minimum_measured_episodes_per_second"] > observed + 1e-12
            or math.ceil(EXPECTED_RAW_RECORDS
                         / plan["minimum_measured_episodes_per_second"])
               > plan["wall_clock_stop_seconds"]
            or not math.isclose(
                plan["max_gpu_hours"], plan["wall_clock_ceiling_seconds"] / 3600,
                rel_tol=0, abs_tol=1e-12)):
        raise PermissionError("Transfer timeout is not supported by measured throughput")
    return json.loads(canonical(plan))


def build_transfer_resource_plan(*, qualification_receipt_sha256: str,
                                 qualification_episode_count: int,
                                 qualification_elapsed_seconds: float,
                                 minimum_measured_episodes_per_second: float,
                                 wall_clock_stop_seconds: int,
                                 wall_clock_ceiling_seconds: int,
                                 batch_size: int = 16,
                                 max_raw_export_bytes: int = 512 * 1024 * 1024) -> dict[str, Any]:
    value = {
        "version": TRANSFER_RESOURCE_PLAN_VERSION,
        "serialization_contract_version": SERIALIZATION_CONTRACT_VERSION,
        "accelerator": "GPU_1xA10", "gpu_count": 1,
        "max_concurrent_runs": 1, "automatic_retries": 0,
        "episode_count": EXPECTED_RAW_RECORDS,
        "max_new_tokens_per_episode": 32, "batch_size": batch_size,
        "max_batched_model_forward_calls": (
            math.ceil(EXPECTED_RAW_RECORDS / batch_size) * 32
        ),
        "max_raw_export_bytes": max_raw_export_bytes,
        "wall_clock_stop_seconds": wall_clock_stop_seconds,
        "wall_clock_ceiling_seconds": wall_clock_ceiling_seconds,
        "protected_audit_and_export_seconds": (
            wall_clock_ceiling_seconds - wall_clock_stop_seconds),
        "max_gpu_hours": wall_clock_ceiling_seconds / 3600,
        "qualification_receipt_sha256": qualification_receipt_sha256,
        "qualification_episode_count": qualification_episode_count,
        "qualification_elapsed_seconds": qualification_elapsed_seconds,
        "minimum_measured_episodes_per_second": minimum_measured_episodes_per_second,
    }
    value["resource_plan_sha256"] = digest(value)
    return validate_transfer_resource_plan(value)


def _registry_bundle(sealed_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    bundle = {arm: _read(sealed_root / f"{arm}.json") for arm in ALL_ARMS}
    return bundle, validate_registry_bundle(bundle)


def verify_transfer_inputs(intent: Mapping[str, Any], *,
                           verify_model_bytes: bool = True) -> dict[str, Any]:
    """Recompute actual model, base gate, checkpoints and sealed registries."""
    value = validate_transfer_intent(intent, verify_inputs=False)
    model = (verify_model_receipt(value["model_root"], value["model_receipt_path"])
             if verify_model_bytes else verify_model_receipt_authority(
                 value["model_receipt_path"],
                 expected_tree_sha256=value["model_tree_sha256"],
                 expected_receipt_sha256=value["model_receipt_sha256"]))
    if (model["tree_sha256"] != value["model_tree_sha256"]
            or model["receipt_sha256"] != value["model_receipt_sha256"]):
        raise PermissionError("Transfer intent model snapshot differs")
    base = validate_base_qualification_receipt(_read(
        value["base_qualification_receipt_path"]))
    source_base_intent = validate_base_qualification_intent(
        value["base_qualification_source_intent"])
    carryforward = validate_carryforward(
        value["base_qualification_carryforward"],
        prior_intent=source_base_intent, base_receipt=base,
        current_source_manifest=value["source_manifest"])
    resource = value["resource_plan"]
    base_resources = base["resources"]
    if (base["status"] != "qualified"
            or base["receipt_sha256"] != value["base_qualification_receipt_sha256"]
            or base["model_tree_sha256"] != value["model_tree_sha256"]
            or base["model_receipt_sha256"] != value["model_receipt_sha256"]
            or carryforward["carryforward_sha256"]
               != value.get("base_qualification_carryforward_sha256")
            or resource["qualification_receipt_sha256"]
               != base["receipt_sha256"]
            or resource["qualification_episode_count"]
               != base["decision_count"]
            or not math.isclose(
                float(resource["qualification_elapsed_seconds"]),
                float(base_resources["inference_seconds"]),
                rel_tol=0, abs_tol=1e-9)):
        raise PermissionError("Transfer intent frozen-base qualification differs")
    manifest_path = _volume_path(value["checkpoint_manifest_path"],
                                 description="checkpoint manifest")
    if file_sha256(manifest_path) != value["checkpoint_manifest_sha256"]:
        raise PermissionError("Checkpoint manifest bytes differ")
    manifest = _read(manifest_path)
    checkpoints = verify_checkpoint_manifest(manifest, value["checkpoint_root"])
    unblinding_path = _volume_path(value["unblinding_key_path"],
                                   description="unblinding key")
    if (manifest.get("version")
            != "research4-three-residual-blinded-checkpoints-v1"
            or manifest.get("manifest_blinded") is not True
            or manifest.get("condition_seed_fields_exposed_in_manifest") is not False
            or any("condition" in row or "training_seed" in row
                   for row in manifest.get("checkpoints", ()))
            or manifest.get("checkpoint_package_sha256")
               != value["checkpoint_package_sha256"]
            or manifest.get("base_model_sha256")
            != base["base_component_sha256_after"]
            or manifest.get("base_qualification_sha256")
               != base["receipt_sha256"]
            or file_sha256(unblinding_path) != value["unblinding_key_sha256"]
            or manifest.get("unblinding_key_sha256")
               != value["unblinding_key_sha256"]
            or directory_tree_sha256(value["checkpoint_root"])
               != value["checkpoint_package_sha256"]):
        raise PermissionError("Checkpoint package/base-gate lineage differs")
    sealed = _volume_path(value["sealed_root"], description="sealed registry")
    bundle, registry_hashes = _registry_bundle(sealed)
    commitment_path = _volume_path(value["commitment_path"],
                                   description="registry commitment")
    commitment = _read(commitment_path)
    if (file_sha256(commitment_path) != value["sealed_commitment_sha256"]
            or validate_public_commitment(commitment, sealed, bundle) != registry_hashes
            or digest(registry_hashes) != value["sealed_registry_sha256"]):
        raise PermissionError("Sealed transfer registry/commitment differs")
    separation_path = _volume_path(
        value["training_separation_receipt_path"],
        description="training separation receipt")
    separation = _read(separation_path)
    separation_payload = dict(separation); separation_hash = separation_payload.pop(
        "receipt_sha256", None)
    if (file_sha256(separation_path) != value["training_separation_receipt_sha256"]
            or separation_hash != digest(separation_payload)
            or separation.get("training_transfer_separation_passed") is not True
            or separation.get("transfer_action_labels_found") != 0
            or separation.get("transfer_family_ids_found") != 0
            or separation.get("exploit_like_descriptions_found") != 0):
        raise PermissionError("Training/evaluation separation evidence differs")
    return {
        "model": model, "base_qualification": base,
        "model_bytes_recomputed": verify_model_bytes,
        "checkpoint_count": len(checkpoints),
        "checkpoint_manifest_sha256": file_sha256(manifest_path),
        "checkpoint_package_sha256": directory_tree_sha256(
            value["checkpoint_root"]),
        "registry_sha256": registry_hashes,
        "sealed_registry_sha256": digest(registry_hashes),
        "sealed_commitment_sha256": file_sha256(commitment_path),
        "training_separation_receipt_sha256": file_sha256(separation_path),
    }


def validate_transfer_intent(value: Mapping[str, Any], *,
                             verify_inputs: bool = False) -> dict[str, Any]:
    intent = dict(value) if isinstance(value, Mapping) else {}
    fields = {
        "version", "identity", "one_use", "attempt_number",
        "automatic_retries", "execution_platform", "source_commit",
        "source_manifest", "source_sha256", "model_root", "model_receipt_path",
        "model_tree_sha256", "model_receipt_sha256",
        "base_qualification_receipt_path", "base_qualification_receipt_sha256",
        "base_qualification_source_intent", "base_qualification_carryforward",
        "base_qualification_carryforward_sha256",
        "qualification_technical_successor",
        "qualification_technical_successor_sha256",
        "checkpoint_root", "checkpoint_manifest_path",
        "checkpoint_manifest_sha256", "checkpoint_package_sha256",
        "unblinding_key_path", "unblinding_key_sha256",
        "sealed_root", "commitment_path", "sealed_commitment_sha256",
        "sealed_registry_sha256", "training_separation_receipt_path",
        "training_separation_receipt_sha256", "resource_plan",
        "resource_plan_sha256", "output_root", "mlflow_experiment_path",
        "max_concurrent_runs", "intent_sha256",
    }
    try:
        source = validate_source_manifest(intent.get("source_manifest", {}))
        plan = validate_transfer_resource_plan(intent.get("resource_plan", {}))
    except (TypeError, ValueError, PermissionError) as exc:
        raise PermissionError("Transfer intent source/resource binding differs") from exc
    source_base_intent = intent.get("base_qualification_source_intent")
    if not isinstance(source_base_intent, Mapping):
        raise PermissionError("Transfer intent lacks carried base source intent")
    carryforward = intent.get("base_qualification_carryforward")
    technical = validate_technical_successor(
        intent.get("qualification_technical_successor"))
    carry_payload = dict(carryforward) if isinstance(carryforward, Mapping) else {}
    carry_hash = carry_payload.pop("carryforward_sha256", None)
    payload = dict(intent); embedded = payload.pop("intent_sha256", None)
    if (set(intent) != fields
            or embedded != digest(payload)
            or intent.get("version") != TRANSFER_INTENT_VERSION
            or intent.get("identity") != TRANSFER_IDENTITY
            or intent.get("one_use") is not True
            or intent.get("attempt_number") != 1
            or intent.get("automatic_retries") != 0
            or intent.get("execution_platform") != "databricks"
            or intent.get("max_concurrent_runs") != 1
            or not re.fullmatch(r"[0-9a-f]{40}", str(
                intent.get("source_commit", "")))
            or source.get("source_commit") != intent.get("source_commit")
            or source.get("manifest_sha256") != intent.get("source_sha256")
            or intent.get("model_root") != MODEL_ROOT
            or intent.get("model_receipt_path") != MODEL_RECEIPT
            or any(not _sha(intent.get(key)) for key in (
                "source_sha256", "model_tree_sha256", "model_receipt_sha256",
                "base_qualification_receipt_sha256",
                "base_qualification_carryforward_sha256",
                "qualification_technical_successor_sha256",
                "checkpoint_manifest_sha256", "checkpoint_package_sha256",
                "unblinding_key_sha256",
                "sealed_commitment_sha256", "sealed_registry_sha256",
                "training_separation_receipt_sha256", "resource_plan_sha256"))
            or plan["resource_plan_sha256"] != intent.get("resource_plan_sha256")
            or carry_hash != digest(carry_payload)
            or carry_hash != intent.get("base_qualification_carryforward_sha256")
            or carry_payload.get("prior_base_intent_sha256")
               != source_base_intent.get("intent_sha256")
            or technical["technical_successor_sha256"]
               != intent.get("qualification_technical_successor_sha256")
            or not _sha(source_base_intent.get("intent_sha256"))
            or intent.get("output_root") != TRANSFER_OUTPUT_ROOT
            or intent.get("mlflow_experiment_path") != TRANSFER_MLFLOW_EXPERIMENT):
        raise PermissionError("Exact one-use Databricks transfer intent required")
    for key in ("base_qualification_receipt_path", "checkpoint_root",
                "checkpoint_manifest_path", "unblinding_key_path",
                "sealed_root", "commitment_path",
                "training_separation_receipt_path"):
        _volume_path(intent.get(key), description=key)
    result = json.loads(canonical(intent))
    if verify_inputs:
        verify_transfer_inputs(result)
    return result


def build_transfer_intent(*, source_root: str | Path, source_commit: str,
                          model_tree_sha256: str, model_receipt_sha256: str,
                          base_qualification_receipt_path: str,
                          base_qualification_source_intent: Mapping[str, Any],
                          base_qualification_carryforward: Mapping[str, Any],
                          base_qualification_carryforward_sha256: str,
                          qualification_technical_successor: Mapping[str, Any],
                          qualification_technical_successor_sha256: str,
                          checkpoint_root: str, checkpoint_manifest_path: str,
                          unblinding_key_path: str,
                          sealed_root: str, commitment_path: str,
                          training_separation_receipt_path: str,
                          resource_plan: Mapping[str, Any]) -> dict[str, Any]:
    source = build_source_manifest(source_root, source_commit)
    model = verify_model_receipt(MODEL_ROOT, MODEL_RECEIPT)
    if (model["tree_sha256"] != model_tree_sha256
            or model["receipt_sha256"] != model_receipt_sha256):
        raise PermissionError("Requested model hashes differ from the Volume snapshot")
    base = validate_base_qualification_receipt(_read(
        base_qualification_receipt_path))
    source_base_intent = validate_base_qualification_intent(
        base_qualification_source_intent)
    carryforward = validate_carryforward(
        base_qualification_carryforward, prior_intent=source_base_intent,
        base_receipt=base, current_source_manifest=source)
    if carryforward["carryforward_sha256"] != base_qualification_carryforward_sha256:
        raise PermissionError("Transfer carry-forward hash differs")
    technical = validate_technical_successor(qualification_technical_successor)
    if technical["technical_successor_sha256"] != qualification_technical_successor_sha256:
        raise PermissionError("Transfer technical-successor hash differs")
    reviewed_resource = validate_transfer_resource_plan(resource_plan)
    if (reviewed_resource["qualification_receipt_sha256"]
            != base["receipt_sha256"]
            or reviewed_resource["qualification_episode_count"]
               != base["decision_count"]
            or not math.isclose(
                float(reviewed_resource["qualification_elapsed_seconds"]),
                float(base["resources"]["inference_seconds"]),
                rel_tol=0, abs_tol=1e-9)):
        raise PermissionError(
            "Transfer resource envelope is not derived from the frozen-base run")
    manifest_sha = file_sha256(checkpoint_manifest_path)
    checkpoint_package_sha = directory_tree_sha256(checkpoint_root)
    bundle, registry_hashes = _registry_bundle(Path(sealed_root))
    commitment = _read(commitment_path)
    if validate_public_commitment(commitment, sealed_root, bundle) != registry_hashes:
        raise PermissionError("Transfer registry commitment differs")
    value = {
        "version": TRANSFER_INTENT_VERSION, "identity": TRANSFER_IDENTITY,
        "one_use": True, "attempt_number": 1, "automatic_retries": 0,
        "execution_platform": "databricks", "source_commit": source_commit,
        "source_manifest": source, "source_sha256": source["manifest_sha256"],
        "model_root": MODEL_ROOT, "model_receipt_path": MODEL_RECEIPT,
        "model_tree_sha256": model_tree_sha256,
        "model_receipt_sha256": model_receipt_sha256,
        "base_qualification_receipt_path": base_qualification_receipt_path,
        "base_qualification_receipt_sha256": base["receipt_sha256"],
        "base_qualification_source_intent": source_base_intent,
        "base_qualification_carryforward": carryforward,
        "base_qualification_carryforward_sha256":
            base_qualification_carryforward_sha256,
        "qualification_technical_successor": technical,
        "qualification_technical_successor_sha256":
            qualification_technical_successor_sha256,
        "checkpoint_root": checkpoint_root,
        "checkpoint_manifest_path": checkpoint_manifest_path,
        "checkpoint_manifest_sha256": manifest_sha,
        "checkpoint_package_sha256": checkpoint_package_sha,
        "unblinding_key_path": unblinding_key_path,
        "unblinding_key_sha256": file_sha256(unblinding_key_path),
        "sealed_root": sealed_root, "commitment_path": commitment_path,
        "sealed_commitment_sha256": file_sha256(commitment_path),
        "sealed_registry_sha256": digest(registry_hashes),
        "training_separation_receipt_path": training_separation_receipt_path,
        "training_separation_receipt_sha256": file_sha256(
            training_separation_receipt_path),
        "resource_plan": reviewed_resource,
        "resource_plan_sha256": reviewed_resource["resource_plan_sha256"],
        "output_root": TRANSFER_OUTPUT_ROOT,
        "mlflow_experiment_path": TRANSFER_MLFLOW_EXPERIMENT,
        "max_concurrent_runs": 1,
    }
    value["intent_sha256"] = digest(value)
    return validate_transfer_intent(value, verify_inputs=True)


def build_databricks_run_binding(intent: Mapping[str, Any], *, job_run_id: str,
                                 reservation: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_transfer_intent(intent)
    reservation_sha = digest(dict(reservation))
    if (reservation.get("intent_sha256") != value["intent_sha256"]
            or reservation.get("owner") != str(job_run_id)
            or reservation.get("identity") != value["identity"]):
        raise PermissionError("Databricks reservation owner/intent differs")
    binding = {
        "version": TRANSFER_RUN_BINDING_VERSION,
        "execution_platform": "databricks", "identity": value["identity"],
        "job_run_id": str(job_run_id), "source_commit": value["source_commit"],
        "intent_sha256": value["intent_sha256"],
        "source_sha256": value["source_sha256"],
        "model_tree_sha256": value["model_tree_sha256"],
        "model_receipt_sha256": value["model_receipt_sha256"],
        "base_qualification_receipt_sha256": value[
            "base_qualification_receipt_sha256"],
        "base_qualification_carryforward_sha256": value[
            "base_qualification_carryforward_sha256"],
        "qualification_technical_successor": value[
            "qualification_technical_successor"],
        "qualification_technical_successor_sha256": value[
            "qualification_technical_successor_sha256"],
        "checkpoint_manifest_sha256": value["checkpoint_manifest_sha256"],
        "checkpoint_package_sha256": value["checkpoint_package_sha256"],
        "sealed_registry_sha256": value["sealed_registry_sha256"],
        "resource_plan_sha256": value["resource_plan_sha256"],
        "reservation_sha256": reservation_sha,
        "output_root": value["output_root"],
    }
    return validate_databricks_run_binding(binding)


def verify_transfer_deployed_source(code_root: str | Path,
                                    intent: Mapping[str, Any]) -> dict[str, Any]:
    return verify_deployed_source(code_root, validate_transfer_intent(intent))
