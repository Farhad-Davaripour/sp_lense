"""CPU-only bridge from an audited package to a reviewable transfer intent."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import math
import os
import re
import shutil
import tempfile

from .continuity_transfer_databricks import (
    build_transfer_intent, build_transfer_resource_plan,
    validate_transfer_intent,
)
from .continuity_transfer_evaluation import EXPECTED_RAW_RECORDS, digest, file_sha256
from .continuity_transfer_packaging import verify_published_package
from .databricks_simulated_continuity import (
    VOLUME, canonical, validate_base_qualification_receipt,
    validate_source_manifest,
)


INTENT_PREPARATION_RECEIPT_VERSION = (
    "research4-databricks-transfer-intent-preparation-v1"
)
TRANSFER_INTENT_PARENT = f"{VOLUME}/transfer_intents"
THROUGHPUT_SAFETY_FACTOR = 0.5
WALL_CLOCK_ROUNDING_SECONDS = 300
PROTECTED_AUDIT_EXPORT_SECONDS = 1800
MAX_WALL_CLOCK_CEILING_SECONDS = 14_400


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _volume_file(value: Any, *, label: str) -> Path:
    if (not isinstance(value, str) or not value.startswith("/Volumes/")
            or ".." in Path(value).parts):
        raise PermissionError(f"{label} must be an exact Unity Catalog Volume path")
    path = Path(value)
    if not path.is_file():
        raise PermissionError(f"{label} is absent")
    return path


def derive_transfer_resource_plan(base_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the complete transfer envelope from actual base inference timing."""
    base = validate_base_qualification_receipt(base_receipt)
    decisions = base["decision_count"]
    inference_seconds = float(base["resources"]["inference_seconds"])
    if (decisions != 288 or not math.isfinite(inference_seconds)
            or inference_seconds <= 0):
        raise PermissionError("Frozen-base inference timing is unavailable")
    observed = decisions / inference_seconds
    minimum = observed * THROUGHPUT_SAFETY_FACTOR
    predicted = math.ceil(EXPECTED_RAW_RECORDS / minimum)
    stop = max(600, math.ceil(
        predicted / WALL_CLOCK_ROUNDING_SECONDS) * WALL_CLOCK_ROUNDING_SECONDS)
    ceiling = stop + PROTECTED_AUDIT_EXPORT_SECONDS
    if ceiling > MAX_WALL_CLOCK_CEILING_SECONDS:
        raise PermissionError(
            "Measured frozen-base throughput cannot support the bounded transfer grid")
    return build_transfer_resource_plan(
        qualification_receipt_sha256=base["receipt_sha256"],
        qualification_episode_count=decisions,
        qualification_elapsed_seconds=inference_seconds,
        minimum_measured_episodes_per_second=minimum,
        wall_clock_stop_seconds=stop, wall_clock_ceiling_seconds=ceiling,
        batch_size=16,
    )


def _validate_package_authority(package_receipt_path: str | Path,
                                base_receipt_path: str | Path
                                ) -> tuple[dict[str, Any], dict[str, Any]]:
    package_path = _volume_file(package_receipt_path, label="package receipt")
    base_path = _volume_file(base_receipt_path, label="base qualification receipt")
    package = verify_published_package(package_path.parent)
    if (package_path.name != "PACKAGING_RECEIPT.json"
            or package.get("package_root") != package_path.parent.as_posix()
            or file_sha256(package_path) != file_sha256(
                Path(package["package_root"]) / "PACKAGING_RECEIPT.json")):
        raise PermissionError("Published package receipt path differs")
    base = validate_base_qualification_receipt(_read(base_path))
    manifest = validate_source_manifest(package.get("source_manifest", {}))
    if (package.get("status") != "complete"
            or package.get("gpu_runs_started") != 0
            or package.get("model_forwards") != 0
            or package.get("optimizer_updates") != 0
            or package.get("base_qualification_receipt_sha256")
               != base["receipt_sha256"]
            or package.get("source_commit") != base["source_commit"]
            or package.get("source_sha256") != base["source_sha256"]
            or manifest["source_commit"] != package["source_commit"]
            or manifest["manifest_sha256"] != package["source_sha256"]
            or package.get("model_tree_sha256") != base["model_tree_sha256"]):
        raise PermissionError("Published package and frozen-base authority differ")
    return package, base


def _intent_paths(package: Mapping[str, Any]) -> dict[str, str]:
    root = Path(package["package_root"])
    return {
        "checkpoint_root": (root / "checkpoints").as_posix(),
        "checkpoint_manifest_path": (root / "CHECKPOINT_MANIFEST.json").as_posix(),
        "unblinding_key_path": package["unblinding_key_path"],
        "sealed_root": (root / "registry" / "sealed").as_posix(),
        "commitment_path": (
            root / "registry" / "SEALED_REGISTRY_COMMITMENT.json").as_posix(),
        "training_separation_receipt_path": (
            root / "TRAINING_SEPARATION_RECEIPT.json").as_posix(),
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def verify_prepared_transfer_intent(root: str | Path) -> dict[str, Any]:
    directory = Path(root)
    receipt_path = directory / "INTENT_PREPARATION_RECEIPT.json"
    intent_path = directory / "TRANSFER_INTENT.json"
    resource_path = directory / "RESOURCE_PLAN.json"
    if not all(path.is_file() for path in (receipt_path, intent_path, resource_path)):
        raise PermissionError("Prepared transfer intent is incomplete")
    receipt = _read(receipt_path); payload = dict(receipt)
    embedded = payload.pop("receipt_sha256", None)
    intent = validate_transfer_intent(_read(intent_path), verify_inputs=True)
    resource = _read(resource_path)
    if (embedded != digest(payload)
            or receipt.get("version") != INTENT_PREPARATION_RECEIPT_VERSION
            or receipt.get("status") != "reviewed_not_submitted"
            or receipt.get("gpu_runs_started") != 0
            or receipt.get("reservations_created") != 0
            or receipt.get("provider_calls") != 0
            or receipt.get("automatic_retries") != 0
            or receipt.get("intent_sha256") != intent["intent_sha256"]
            or receipt.get("intent_file_sha256") != file_sha256(intent_path)
            or receipt.get("resource_plan_sha256")
               != intent["resource_plan_sha256"]
            or receipt.get("resource_plan_file_sha256") != file_sha256(resource_path)
            or resource != intent["resource_plan"]
            or directory.name != intent["intent_sha256"]
            or receipt.get("intent_root") != directory.as_posix()):
        raise PermissionError("Prepared transfer intent replay differs")
    return receipt


def prepare_transfer_intent(*, code_path: str | Path,
                            package_receipt_path: str | Path,
                            base_qualification_receipt_path: str | Path
                            ) -> dict[str, Any]:
    """Build, replay and publish a reviewable intent without reservation."""
    package, base = _validate_package_authority(
        package_receipt_path, base_qualification_receipt_path)
    resource = derive_transfer_resource_plan(base)
    paths = _intent_paths(package)
    intent = build_transfer_intent(
        source_root=code_path, source_commit=package["source_commit"],
        model_tree_sha256=package["model_tree_sha256"],
        model_receipt_sha256=base["model_receipt_sha256"],
        base_qualification_receipt_path=str(base_qualification_receipt_path),
        resource_plan=resource, **paths)
    replay = validate_transfer_intent(intent, verify_inputs=True)
    if replay != intent:
        raise PermissionError("Built transfer intent does not replay exactly")
    parent = Path(TRANSFER_INTENT_PARENT); parent.mkdir(parents=True, exist_ok=True)
    destination = parent / intent["intent_sha256"]
    if destination.exists():
        return verify_prepared_transfer_intent(destination)
    staging = Path(tempfile.mkdtemp(prefix=".transfer-intent-", dir=parent))
    try:
        _write_json_exclusive(staging / "TRANSFER_INTENT.json", intent)
        _write_json_exclusive(staging / "RESOURCE_PLAN.json", resource)
        receipt = {
            "version": INTENT_PREPARATION_RECEIPT_VERSION,
            "status": "reviewed_not_submitted", "cpu_only": True,
            "gpu_runs_started": 0, "reservations_created": 0,
            "provider_calls": 0, "automatic_retries": 0,
            "source_commit": package["source_commit"],
            "source_sha256": package["source_sha256"],
            "source_manifest": package["source_manifest"],
            "package_content_sha256": package["package_content_sha256"],
            "packaging_receipt_sha256": package["receipt_sha256"],
            "base_qualification_receipt_sha256": base["receipt_sha256"],
            "intent_sha256": intent["intent_sha256"],
            "intent_file_sha256": file_sha256(staging / "TRANSFER_INTENT.json"),
            "resource_plan_sha256": resource["resource_plan_sha256"],
            "resource_plan_file_sha256": file_sha256(staging / "RESOURCE_PLAN.json"),
            "throughput_source": "actual_frozen_base_inference_seconds",
            "throughput_safety_factor": THROUGHPUT_SAFETY_FACTOR,
            "intent_root": destination.as_posix(),
        }
        receipt["receipt_sha256"] = digest(receipt)
        _write_json_exclusive(staging / "INTENT_PREPARATION_RECEIPT.json", receipt)
        destination.mkdir(parents=False, exist_ok=False)
        try:
            for name in ("TRANSFER_INTENT.json", "RESOURCE_PLAN.json"):
                shutil.copy2(staging / name, destination / name)
                if file_sha256(staging / name) != file_sha256(destination / name):
                    raise PermissionError("Prepared intent copy readback differs")
            # The preparation receipt is the immutable commit marker.
            shutil.copy2(staging / "INTENT_PREPARATION_RECEIPT.json",
                         destination / "INTENT_PREPARATION_RECEIPT.json")
            if (file_sha256(staging / "INTENT_PREPARATION_RECEIPT.json")
                    != file_sha256(destination / "INTENT_PREPARATION_RECEIPT.json")):
                raise PermissionError("Prepared intent receipt readback differs")
        except BaseException:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        return verify_prepared_transfer_intent(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
