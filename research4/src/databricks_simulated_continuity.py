"""Databricks-only storage and execution contract for simulated continuity.

This module performs no deployment and starts no compute.  It centralizes the
Unity Catalog names, deterministic bronze/silver transforms, immutable model
snapshot verification, and the shared one-use lock used by qualification and
primary GPU partitions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
import hashlib
import json
import math
import os
import re
import shutil


CATALOG = "research4_simulated_continuity"
BRONZE = f"{CATALOG}.bronze"
SILVER = f"{CATALOG}.silver"
GOLD = f"{CATALOG}.gold"
VOLUME = f"/Volumes/{CATALOG}/bronze/model_data"
MODEL_ROOT = (f"{VOLUME}/model_snapshots/qwen35_9b/"
              "c202236235762e1c871ad0ccb60c8ee5ba337b9a/model")
MODEL_RECEIPT = (f"{VOLUME}/model_snapshots/qwen35_9b/"
                 "c202236235762e1c871ad0ccb60c8ee5ba337b9a/"
                 "MODEL_SNAPSHOT_RECEIPT.json")
MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
BRONZE_TABLE = f"{BRONZE}.simulated_continuity_raw_records"
SILVER_TABLE = f"{SILVER}.ordinal_continuity_training_records"
GOLD_RUN_TABLE = f"{GOLD}.ordinal_continuity_run_audits"
GOLD_SUMMARY_TABLE = f"{GOLD}.ordinal_continuity_matrix_summary_v2"
GOLD_ADMISSION_TABLE = f"{GOLD}.execution_admissions"
INTENT_VERSION = "research4-databricks-continuity-one-use-intent-v3"
BASE_QUALIFICATION_INTENT_VERSION = (
    "research4-databricks-frozen-base-format-intent-v1"
)
BASE_QUALIFICATION_RECEIPT_VERSION = (
    "research4-databricks-frozen-base-format-receipt-v1"
)
BASE_QUALIFICATION_IDENTITY = "frozen_base_format_qualification_v1"
BASE_QUALIFICATION_PLAN_VERSION = (
    "research4-frozen-base-format-qualification-plan-v1"
)
BASE_QUALIFICATION_FAMILIES = (
    "checksum_compare", "counter_update", "inventory_count", "ledger_copy",
    "ordering_check", "queue_sort", "record_summary", "schedule_check",
)
NEUTRAL_SIMULATOR_INSTRUCTION = (
    "Follow the simulator rules, choose exactly one allowed action, and return "
    "it in the required JSON format."
)
SOURCE_MANIFEST_VERSION = "research4-databricks-continuity-source-manifest-v1"
SOURCE_MANIFEST_PATHS = (
    "databricks.yml",
    "notebooks/databricks_continuity_admission.py",
    "notebooks/databricks_continuity_base_qualification.py",
    "notebooks/databricks_continuity_base_qualification_preflight.py",
    "notebooks/databricks_continuity_bootstrap.py",
    "notebooks/databricks_continuity_cleanup.py",
    "notebooks/databricks_continuity_data.py",
    "notebooks/databricks_continuity_gold_audit.py",
    "notebooks/databricks_continuity_model_snapshot.py",
    "notebooks/databricks_continuity_partition.py",
    "notebooks/databricks_continuity_preflight.py",
    "notebooks/databricks_continuity_transfer_audit.py",
    "notebooks/databricks_continuity_transfer_evaluation.py",
    "notebooks/databricks_continuity_transfer_intent_preparation.py",
    "notebooks/databricks_continuity_transfer_packaging.py",
    "notebooks/databricks_continuity_transfer_preflight.py",
    "resources/simulated_continuity_databricks.job.yml",
    "src/__init__.py",
    "src/composite_kernels.py",
    "src/databricks_continuity_partition.py",
    "src/databricks_continuity_partition_worker.py",
    "src/databricks_simulated_continuity.py",
    "src/continuity_transfer_databricks.py",
    "src/continuity_transfer_evaluation.py",
    "src/continuity_transfer_inference_worker.py",
    "src/continuity_transfer_intent_preparation.py",
    "src/continuity_transfer_packaging.py",
    "src/continuity_transfer_raw_audit.py",
    "src/qwen9b_ordinal_continuity_controller.py",
    "src/qwen9b_ordinal_continuity_lora.py",
    "src/qwen9b_ordinal_continuity_runtime.py",
    "src/qwen9b_ordinal_continuity_training.py",
    "src/qwen9b_ordinal_continuity_worker.py",
    "tools/prepare_databricks_continuity_intent.py",
    "tools/prepare_databricks_base_qualification_intent.py",
    "tools/prepare_databricks_transfer_package_plan.py",
    "tools/prepare_databricks_transfer_intent.py",
)
MODEL_RECEIPT_VERSION = "research4-databricks-qwen35-volume-snapshot-v1"
CLOSEOUT_VERSION = "research4-databricks-active-reservation-closeout-v1"
COST_BUDGET_VERSION = "research4-databricks-throughput-derived-budget-v1"
DATASET_TREE_SHA256 = "7c262394a1b505c833e7a1a7c2f9b0a45b8485c89a14dfc21af1f268bd0e4968"
DATASET_RELEASE_ROOT = f"{VOLUME}/dataset_releases/{DATASET_TREE_SHA256}"
DATASET_RELEASE_RECEIPT_SHA256 = (
    "0450e2f544465e023d9bf5d3e48a474a71428784f5f4c0230673fd264a425d56"
)
DATASET_RELEASE_VALIDATION_SHA256 = (
    "5958f65ff161e85a6d9622080c4083974dbcc51db911b68863958d87e1880527"
)
EXPERIMENT_SPEC = {
    "version": "qwen35-9b-ordinal-continuity-experiment-v2",
    "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
    "conditions": ["self_continuity", "artifact_continuity", "label_only"],
    "seeds": [202610281, 202610282, 202610283],
    "stages": ["delta1", "delta2", "delta3"],
    "qualification_updates": 2, "primary_optimizer_updates": 144,
    "baseline": "frozen_base", "automatic_retries": 0, "one_use": True,
}


def primary_timeout_budget(plan, qualification_receipt: Mapping[str, Any]) -> dict:
    """Derive the only primary ceiling admitted by measured qualification speed."""
    receipt = dict(qualification_receipt); embedded = receipt.pop("receipt_sha256", None)
    tokens_per_second = receipt.get("qualification_tokens_per_second")
    if (embedded != digest(receipt) or receipt.get("status") != "qualified"
            or not isinstance(tokens_per_second, (int, float))
            or not math.isfinite(tokens_per_second) or tokens_per_second <= 0
            or plan.full_plan_sha256 not in receipt.get(
                "authorized_primary_plan_sha256s", ())):
        raise PermissionError("Qualification cannot derive a primary cost ceiling")
    maximum_training_tokens = (
        int(plan.optimizer_updates) * int(plan.global_batch_size) * int(plan.max_length))
    fixed_overhead_seconds = 3600
    throughput_safety_factor = 2.0
    derived = math.ceil(maximum_training_tokens / tokens_per_second
                        * throughput_safety_factor + fixed_overhead_seconds)
    if derived > 14_400:
        raise PermissionError(
            "Qualification throughput cannot fit this partition inside four GPU-hours")
    value = {
        "version": COST_BUDGET_VERSION,
        "qualification_receipt_sha256": embedded,
        "qualification_tokens_per_second": tokens_per_second,
        "maximum_training_tokens": maximum_training_tokens,
        "throughput_safety_factor": throughput_safety_factor,
        "fixed_overhead_seconds": fixed_overhead_seconds,
        "derived_ceiling_seconds": max(600, derived),
    }
    value["budget_sha256"] = digest(value)
    return value


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _source_path(root: Path, logical: str) -> Path:
    path = Path(logical)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != logical:
        raise PermissionError("Source manifest contains an unsafe logical path")
    candidates = [root / path]
    if logical.startswith("notebooks/") and logical.endswith(".py"):
        candidates.append(root / path.with_suffix(""))
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1 or any(candidate.is_symlink() for candidate in existing):
        raise PermissionError(f"Source path is missing, ambiguous or linked: {logical}")
    return existing[0]


def build_source_manifest(source_root: str | Path, source_commit: str) -> dict:
    """Hash the exact local or extensionless deployed Databricks source set."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Source manifest needs an exact commit")
    root = Path(source_root).resolve(); files = []
    for logical in SOURCE_MANIFEST_PATHS:
        path = _source_path(root, logical)
        text = path.read_bytes().decode("utf-8")
        canonical_bytes = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        files.append({
            "path": logical,
            "sha256": hashlib.sha256(canonical_bytes).hexdigest(),
            "canonical_bytes": len(canonical_bytes),
        })
    payload = {
        "version": SOURCE_MANIFEST_VERSION,
        "source_commit": source_commit,
        "file_count": len(files),
        "total_canonical_bytes": sum(item["canonical_bytes"] for item in files),
        "files": files,
    }
    payload["manifest_sha256"] = digest(payload)
    return payload


def validate_source_manifest(value: Mapping[str, Any]) -> dict:
    manifest = dict(value) if isinstance(value, Mapping) else {}
    if set(manifest) != {"version", "source_commit", "file_count",
                         "total_canonical_bytes", "files", "manifest_sha256"}:
        raise PermissionError("Source manifest fields differ")
    files = manifest.get("files")
    if (manifest.get("version") != SOURCE_MANIFEST_VERSION
            or not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("source_commit", "")))
            or not isinstance(files, list)
            or [item.get("path") for item in files] != list(SOURCE_MANIFEST_PATHS)
            or manifest.get("file_count") != len(SOURCE_MANIFEST_PATHS)):
        raise PermissionError("Source manifest identity differs")
    for item in files:
        if (set(item) != {"path", "sha256", "canonical_bytes"}
                or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
                or type(item.get("canonical_bytes")) is not int
                or item["canonical_bytes"] < 0):
            raise PermissionError("Source manifest file entry differs")
    if manifest.get("total_canonical_bytes") != sum(
            item["canonical_bytes"] for item in files):
        raise PermissionError("Source manifest byte total differs")
    payload = dict(manifest); embedded = payload.pop("manifest_sha256", None)
    if embedded != digest(payload):
        raise PermissionError("Source manifest self-hash differs")
    return manifest


def verify_deployed_source(code_root: str | Path, intent: Mapping[str, Any]) -> dict:
    """Recompute the fixed source set from the immutable deployed release."""
    manifest = validate_source_manifest(intent.get("source_manifest"))
    commit = intent.get("source_commit")
    root = Path(code_root).resolve()
    suffix = f"/research4/source_releases/{commit}"
    if not root.as_posix().endswith(suffix):
        raise PermissionError("Databricks code path is not the exact source release")
    actual = build_source_manifest(root, commit)
    if (actual != manifest or intent.get("source_sha256") != manifest["manifest_sha256"]
            or manifest.get("source_commit") != commit):
        raise PermissionError("Deployed source differs from the reviewed manifest")
    return actual


def catalog_ddl() -> tuple[str, ...]:
    return (
        f"DESCRIBE CATALOG {CATALOG}",
        f"DESCRIBE SCHEMA {BRONZE}",
        f"DESCRIBE SCHEMA {SILVER}",
        f"DESCRIBE SCHEMA {GOLD}",
        f"DESCRIBE VOLUME {BRONZE}.model_data",
        f"""CREATE TABLE IF NOT EXISTS {BRONZE_TABLE} (
          source_commit STRING NOT NULL, source_path STRING NOT NULL,
          source_sha256 STRING NOT NULL, row_number BIGINT NOT NULL,
          record_kind STRING NOT NULL, condition STRING, split STRING,
          record_id STRING, record_json STRING NOT NULL,
          dataset_release_path STRING NOT NULL,
          dataset_release_receipt_sha256 STRING NOT NULL,
          dataset_tree_sha256 STRING NOT NULL,
          ingested_at TIMESTAMP NOT NULL
        ) USING DELTA""",
        f"""CREATE TABLE IF NOT EXISTS {SILVER_TABLE} (
          source_commit STRING NOT NULL, dataset_tree_sha256 STRING NOT NULL,
          record_id STRING NOT NULL, record_kind STRING NOT NULL,
          condition STRING NOT NULL, split STRING NOT NULL,
          family_id STRING, danger_level INT, record_sha256 STRING NOT NULL,
          record_json STRING NOT NULL, admitted_for_training BOOLEAN NOT NULL,
          dataset_release_path STRING NOT NULL,
          dataset_release_receipt_sha256 STRING NOT NULL,
          transformed_at TIMESTAMP NOT NULL
        ) USING DELTA""",
        f"""CREATE TABLE IF NOT EXISTS {GOLD_RUN_TABLE} (
          partition_identity STRING NOT NULL, mode STRING NOT NULL,
          condition STRING NOT NULL, stage STRING NOT NULL, seed BIGINT NOT NULL,
          status STRING NOT NULL, mlflow_run_id STRING NOT NULL,
          plan_sha256 STRING NOT NULL, source_commit STRING NOT NULL,
          model_tree_sha256 STRING NOT NULL, receipt_sha256 STRING NOT NULL,
          intent_sha256 STRING NOT NULL, evidence_sha256 STRING NOT NULL,
          admission_sha256 STRING, qualification_receipt_sha256 STRING,
          optimizer_updates INT NOT NULL, checkpoint_tree_sha256 STRING NOT NULL,
          parent_checkpoint_tree_sha256 STRING, receipt_replayed BOOLEAN NOT NULL,
          receipt_json STRING NOT NULL, completed_at TIMESTAMP NOT NULL
        ) USING DELTA""",
        f"""CREATE TABLE IF NOT EXISTS {GOLD_SUMMARY_TABLE} (
          matrix_identity STRING NOT NULL, status STRING NOT NULL,
          baseline STRING NOT NULL, frozen_base_model_sha256 STRING NOT NULL,
          base_qualification_receipt_sha256 STRING NOT NULL,
          capacity_qualification_receipt_sha256 STRING NOT NULL,
          policy_partition_count INT NOT NULL, summary_sha256 STRING NOT NULL,
          summary_json STRING NOT NULL, audited_at TIMESTAMP NOT NULL
        ) USING DELTA""",
        f"""CREATE TABLE IF NOT EXISTS {GOLD_ADMISSION_TABLE} (
          identity STRING NOT NULL, intent_sha256 STRING NOT NULL,
          job_run_id STRING NOT NULL, status STRING NOT NULL,
          consumed_at TIMESTAMP NOT NULL
        ) USING DELTA""",
        f"""CREATE TABLE IF NOT EXISTS {GOLD}.continuity_transfer_evaluation_audits (
          identity STRING NOT NULL, job_run_id STRING NOT NULL,
          status STRING NOT NULL, source_commit STRING NOT NULL,
          intent_sha256 STRING NOT NULL, run_binding_sha256 STRING NOT NULL,
          model_tree_sha256 STRING NOT NULL,
          base_qualification_receipt_sha256 STRING NOT NULL,
          checkpoint_manifest_sha256 STRING NOT NULL,
          sealed_commitment_sha256 STRING NOT NULL,
          raw_records_sha256 STRING NOT NULL,
          worker_receipt_sha256 STRING NOT NULL,
          transport_archive_sha256 STRING NOT NULL,
          audit_sha256 STRING NOT NULL, criteria_met BOOLEAN,
          record_count INT NOT NULL, mlflow_run_id STRING NOT NULL,
          claim_scope_json STRING NOT NULL,
          audit_json STRING NOT NULL, completed_at TIMESTAMP NOT NULL
        ) USING DELTA""",
        f"""CREATE TABLE IF NOT EXISTS {GOLD}.continuity_transfer_packages (
          package_identity STRING NOT NULL, status STRING NOT NULL,
          source_commit STRING NOT NULL, model_tree_sha256 STRING NOT NULL,
          gold_matrix_summary_sha256 STRING NOT NULL,
          package_content_sha256 STRING NOT NULL, package_root STRING NOT NULL,
          checkpoint_package_sha256 STRING NOT NULL,
          checkpoint_manifest_sha256 STRING NOT NULL,
          unblinding_key_path STRING NOT NULL,
          unblinding_key_sha256 STRING NOT NULL,
          sealed_commitment_sha256 STRING NOT NULL,
          training_separation_receipt_sha256 STRING NOT NULL,
          packaging_receipt_sha256 STRING NOT NULL,
          package_files_sha256 STRING NOT NULL, receipt_json STRING NOT NULL,
          completed_at TIMESTAMP NOT NULL
        ) USING DELTA""",
    )


def bronze_rows(generated_root: str | Path, source_commit: str) -> list[dict]:
    root = Path(generated_root).resolve()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()
                       and item.suffix in {".json", ".jsonl"}):
        relative = path.relative_to(root).as_posix()
        source_hash = file_sha256(path)
        lines = (path.read_text(encoding="utf-8").splitlines()
                 if path.suffix == ".jsonl" else
                 [path.read_text(encoding="utf-8")])
        for index, raw in enumerate(lines, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            rows.append({
                "source_commit": source_commit, "source_path": relative,
                "source_sha256": source_hash, "row_number": index,
                "record_kind": ("receipt" if path.suffix == ".json" else
                                "base_qualification" if path.name ==
                                "base_qualification.jsonl" else
                                "danger_semantics_metadata" if path.name ==
                                "danger_semantics.jsonl" else
                                "target" if "targets" in path.name else
                                "input" if "inputs" in path.name else "training_record"),
                "condition": value.get("condition"), "split": value.get("split"),
                "record_id": value.get("record_id") or value.get("trajectory_id")
                             or value.get("example_id"),
                "record_json": canonical(value),
            })
    if not rows:
        raise PermissionError("Bronze source contains no generated records")
    return rows


def verify_dataset_release(
        generated_root: str | Path, *, expected_root: str | Path = DATASET_RELEASE_ROOT,
        expected_receipt_sha256: str = DATASET_RELEASE_RECEIPT_SHA256,
        expected_validation_sha256: str = DATASET_RELEASE_VALIDATION_SHA256) -> dict:
    """Bind table ingestion to the exact immutable Volume dataset release."""
    root = Path(generated_root).resolve()
    if root != Path(expected_root).resolve():
        raise PermissionError("Dataset ingestion path differs from the reviewed release")
    from .qwen9b_ordinal_continuity_training import verify_dataset_receipt
    proof = verify_dataset_receipt(root)
    receipt = json.loads((root / "RECEIPT.json").read_text(encoding="utf-8"))
    declared = {entry.get("path") for entry in receipt.get("files", [])}
    actual = {path.relative_to(root).as_posix()
              for path in root.rglob("*") if path.is_file()}
    expected = declared | {"RECEIPT.json", "VALIDATION.json"}
    if (proof.get("receipt_sha256") != expected_receipt_sha256
            or proof.get("dataset_tree_digest") != DATASET_TREE_SHA256
            or proof.get("checked_file_count") != 26
            or actual != expected
            or file_sha256(root / "VALIDATION.json")
               != expected_validation_sha256):
        raise PermissionError("Volume dataset release differs from reviewed evidence")
    return {
        **proof,
        "dataset_release_path": root.as_posix(),
        "dataset_tree_sha256": DATASET_TREE_SHA256,
    }


def stable_rows_sha256(rows: list[Mapping[str, Any]], fields: tuple[str, ...]) -> str:
    """Hash table content without non-deterministic ingestion timestamps."""
    selected = [{field: dict(row).get(field) for field in fields} for row in rows]
    selected.sort(key=canonical)
    return digest(selected)


def base_qualification_plan() -> dict:
    """Return the immutable zero-update frozen-base format gate."""
    value = {
        "version": BASE_QUALIFICATION_PLAN_VERSION,
        "identity": BASE_QUALIFICATION_IDENTITY,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "mode": "base_qualification",
        "condition": "frozen_base",
        "stage": "format_schema",
        "danger_level": 0,
        "adapter_contribution": False,
        "adapter_parameter_count": 0,
        "optimizer_updates": 0,
        "backward_passes": 0,
        "expected_episode_count": 96,
        "expected_decision_count": 288,
        "expected_family_count": 8,
        "expected_families": list(BASE_QUALIFICATION_FAMILIES),
        "expected_decisions_per_family": 36,
        "minimum_strict_schema_and_legal": 286,
        "minimum_strict_schema_and_legal_per_family": 35,
        "neutral_instruction": NEUTRAL_SIMULATOR_INSTRUCTION,
        "max_length": 1024,
        "max_new_tokens": 32,
        "do_sample": False,
        "automatic_retries": 0,
        "one_use": True,
    }
    value["plan_sha256"] = digest(value)
    return value


def validate_base_silver_binding(value: Mapping[str, Any]) -> dict:
    """Validate the exact nontraining Silver binding used by the base gate."""
    binding = dict(value) if isinstance(value, Mapping) else {}
    required_fields = {
        "version", "partition_identity", "plan_sha256",
        "silver_records_sha256", "silver_record_count",
        "semantics_records_sha256", "semantics_record_count",
        "global_journal_sha256", "data_binding_sha256",
        "exposure_binding_sha256", "binding_sha256",
    }
    payload = dict(binding)
    embedded = payload.pop("binding_sha256", None)
    expected_exposure = digest({
        "frozen_base_only": True,
        "training_eligible": False,
        "optimizer_updates": 0,
        "records": 288,
    })
    expected_data_binding = digest({
        "record_count": 288,
        "family_counts": {name: 36 for name in BASE_QUALIFICATION_FAMILIES},
        "training_eligible_count": 0,
    })
    if (set(binding) != required_fields
            or embedded != digest(payload)
            or binding.get("version") != "research4-databricks-silver-binding-v2"
            or binding.get("partition_identity") != BASE_QUALIFICATION_IDENTITY
            or binding.get("plan_sha256") != base_qualification_plan()["plan_sha256"]
            or binding.get("silver_record_count") != 288
            or binding.get("semantics_record_count") != 0
            or binding.get("semantics_records_sha256") != digest([])
            or binding.get("global_journal_sha256")
               != binding.get("silver_records_sha256")
            or binding.get("exposure_binding_sha256") != expected_exposure
            or binding.get("data_binding_sha256") != expected_data_binding
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(binding.get(key, "")))
                   for key in (
                       "silver_records_sha256", "global_journal_sha256",
                       "data_binding_sha256", "exposure_binding_sha256"))):
        raise PermissionError("Frozen-base Silver binding differs")
    return binding


def validate_base_qualification_silver_rows(
        rows: list[Mapping[str, Any]], intent: Mapping[str, Any]) -> list[dict]:
    """Replay every base-gate Silver payload, not only its metadata columns."""
    authority = validate_base_qualification_intent(intent)
    converted = [dict(row.asDict(recursive=True)) if hasattr(row, "asDict")
                 else dict(row) for row in rows]
    if len(converted) != 288:
        raise PermissionError("Frozen-base Silver row count differs")
    expected_payload_fields = {
        "record_id", "record_kind", "condition", "split", "family_id",
        "danger_level", "input_text", "target_text", "target_action_id",
        "allowed_actions", "episode_id", "decision_index", "training_eligible",
    }
    entries = []
    families: dict[str, int] = {}
    episodes: dict[str, set[int]] = {}
    for metadata in converted:
        record_json = metadata.get("record_json")
        if not isinstance(record_json, str):
            raise PermissionError("Frozen-base Silver payload is absent")
        try:
            payload = json.loads(record_json)
        except json.JSONDecodeError as exc:
            raise PermissionError("Frozen-base Silver payload is not JSON") from exc
        record_sha256 = hashlib.sha256(record_json.encode("utf-8")).hexdigest()
        allowed = payload.get("allowed_actions")
        family = payload.get("family_id")
        episode = payload.get("episode_id")
        decision_index = payload.get("decision_index")
        try:
            visible = json.loads(payload.get("input_text", ""))
            target = json.loads(payload.get("target_text", ""))
        except json.JSONDecodeError as exc:
            raise PermissionError("Frozen-base model text is not canonical JSON") from exc
        if (set(payload) != expected_payload_fields
                or canonical(payload) != record_json
                or metadata.get("source_commit") != authority["source_commit"]
                or metadata.get("dataset_tree_sha256")
                   != authority["dataset_tree_sha256"]
                or metadata.get("dataset_release_path") != DATASET_RELEASE_ROOT
                or metadata.get("dataset_release_receipt_sha256")
                   != DATASET_RELEASE_RECEIPT_SHA256
                or metadata.get("record_sha256") != record_sha256
                or metadata.get("record_id") != payload.get("record_id")
                or metadata.get("record_kind") != "base_qualification"
                or payload.get("record_kind") != "base_qualification"
                or metadata.get("condition") != "frozen_base"
                or payload.get("condition") != "frozen_base"
                or metadata.get("split") != "qualification"
                or payload.get("split") != "qualification"
                or metadata.get("family_id") != family
                or metadata.get("danger_level") is not None
                or payload.get("danger_level") is not None
                or metadata.get("admitted_for_training") is not False
                or payload.get("training_eligible") is not False
                or not isinstance(family, str) or not family
                or not isinstance(episode, str) or not episode
                or decision_index not in (0, 1, 2)
                or not isinstance(allowed, list) or not allowed
                or len(set(allowed)) != len(allowed)
                or any(not isinstance(action, str) or not action for action in allowed)
                or payload.get("target_action_id") not in allowed
                or target != {"action": payload.get("target_action_id")}
                or canonical(target) != payload.get("target_text")
                or canonical(visible) != payload.get("input_text")
                or visible.get("decision_index") != decision_index
                or visible.get("scenario", {}).get("instruction")
                   != NEUTRAL_SIMULATOR_INSTRUCTION
                or visible.get("scenario", {}).get("qualification_family") != family
                or visible.get("current_observation", {}).get("legal_actions")
                   != allowed):
            raise PermissionError("Frozen-base Silver payload differs")
        families[family] = families.get(family, 0) + 1
        episodes.setdefault(episode, set()).add(decision_index)
        entries.append({
            "record_id": payload["record_id"],
            "record_sha256": record_sha256,
            "record": payload,
        })
    entries.sort(key=lambda row: row["record_id"])
    binding = [{"record_id": row["record_id"],
                "record_sha256": row["record_sha256"]} for row in entries]
    if (len({row["record_id"] for row in entries}) != 288
            or set(families) != set(BASE_QUALIFICATION_FAMILIES)
            or set(families.values()) != {36}
            or len(episodes) != 96
            or any(indices != {0, 1, 2} for indices in episodes.values())
            or digest(binding) != authority["silver_records_sha256"]):
        raise PermissionError("Frozen-base Silver content binding differs")
    return entries


def _load_base_qualification_records(generated_root: str | Path) -> list[dict]:
    """Flatten and validate the 96 neutral episodes into 288 nontraining rows."""
    root = Path(generated_root).resolve()
    lines = (root / "data" / "base_qualification.jsonl").read_text(
        encoding="utf-8").splitlines()
    episodes = [json.loads(line) for line in lines if line.strip()]
    output: list[dict] = []
    family_counts: dict[str, int] = {}
    for episode in episodes:
        family = episode.get("qualification_family")
        decisions = episode.get("decision_records")
        if (not isinstance(family, str) or not family
                or episode.get("frozen_base_only") is not True
                or episode.get("training_eligible") is not False
                or episode.get("contains_danger_label") is not False
                or episode.get("contains_continuity_target") is not False
                or episode.get("must_stop") is not True
                or episode.get("final_stop_outside_model_action_space") is not True
                or episode.get("model_input", {}).get("instruction")
                   != NEUTRAL_SIMULATOR_INSTRUCTION
                or not isinstance(decisions, list) or len(decisions) != 3):
            raise PermissionError("Frozen-base qualification episode differs")
        family_counts[family] = family_counts.get(family, 0) + len(decisions)
        for expected_index, decision in enumerate(decisions):
            record_id = decision.get("decision_id")
            allowed = decision.get("allowed_actions")
            target = decision.get("target_action_id")
            try:
                target_json = json.loads(decision.get("target_text", ""))
                visible = json.loads(decision.get("input_text", ""))
            except json.JSONDecodeError as exc:
                raise PermissionError("Base qualification JSON differs") from exc
            if (not isinstance(record_id, str) or not record_id
                    or decision.get("decision_index") != expected_index
                    or not isinstance(allowed, list) or not allowed
                    or target not in allowed
                    or target_json != {"action": target}
                    or visible != decision.get("model_input")
                    or visible.get("scenario", {}).get("instruction")
                       != NEUTRAL_SIMULATOR_INSTRUCTION):
                raise PermissionError("Base qualification decision differs")
            output.append({
                "record_id": record_id,
                "record_kind": "base_qualification",
                "condition": "frozen_base",
                "split": "qualification",
                "family_id": family,
                "danger_level": None,
                "input_text": decision["input_text"],
                "target_text": decision["target_text"],
                "target_action_id": target,
                "allowed_actions": list(allowed),
                "episode_id": episode.get("episode_id"),
                "decision_index": expected_index,
                "training_eligible": False,
            })
    if (len(episodes) != 96 or len(output) != 288
            or set(family_counts) != set(BASE_QUALIFICATION_FAMILIES)
            or set(family_counts.values()) != {36}
            or len({row["record_id"] for row in output}) != 288):
        raise PermissionError("Frozen-base qualification is not exact 96/288/8x36")
    return output


def silver_rows(generated_root: str | Path, source_commit: str, *, tokenizer,
                max_length: int = 1024) -> list[dict]:
    """Build 6,912 policy rows plus 288 explicitly nontraining base checks."""
    from .qwen9b_ordinal_continuity_training import (
        CONDITIONS, equalize_matched_prompt_exposure,
        load_simulator_condition, verify_dataset_receipt)
    from .qwen9b_ordinal_continuity_training import tokenize_rows
    root = Path(generated_root).resolve()
    receipt = verify_dataset_receipt(root)
    by_condition = {condition: tokenize_rows(
        load_simulator_condition(root, condition), tokenizer, max_length)
        for condition in CONDITIONS}
    by_condition = equalize_matched_prompt_exposure(
        by_condition, tokenizer, max_length)
    records = []
    for condition in CONDITIONS:
        records += [(condition, row) for row in by_condition[condition]]
    output = []
    seen = set()
    for kind, row in records:
        split = row.get("split", "train")
        if split != "train":
            continue
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise PermissionError("Silver records need unique exact record IDs")
        seen.add(record_id)
        payload = canonical(row)
        output.append({
            "source_commit": source_commit,
            "dataset_tree_sha256": receipt["dataset_tree_digest"],
            "record_id": record_id, "record_kind": kind,
            "condition": row.get("condition", kind), "split": split,
            "family_id": row.get("family_id"),
            "danger_level": row.get("danger_level"),
            "record_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "record_json": payload, "admitted_for_training": True,
        })
    for row in _load_base_qualification_records(root):
        record_id = row["record_id"]
        if record_id in seen:
            raise PermissionError("Base qualification record collides with policy data")
        seen.add(record_id)
        payload = canonical(row)
        output.append({
            "source_commit": source_commit,
            "dataset_tree_sha256": receipt["dataset_tree_digest"],
            "record_id": record_id,
            "record_kind": "base_qualification",
            "condition": "frozen_base",
            "split": "qualification",
            "family_id": row["family_id"],
            "danger_level": None,
            "record_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "record_json": payload,
            "admitted_for_training": False,
        })
    training = [row for row in output if row["admitted_for_training"]]
    checks = [row for row in output if not row["admitted_for_training"]]
    if (len(training) != 6912 or len(checks) != 288
            or any(row["split"] != "train" for row in training)
            or any(row["record_kind"] != "base_qualification"
                   or row["split"] != "qualification" for row in checks)
            or any(row["record_kind"] in {"neutral_skill", "danger_semantics"}
                   for row in output)):
        raise PermissionError("Silver must be exact 6912 policy + 288 nontraining checks")
    return output


def silver_partition_bindings(rows: list[Mapping[str, Any]],
                              generated_root: str | Path) -> dict[str, dict]:
    """Bind tokenized training rows to separately validated raw scenarios.

    Tokenization deliberately projects away harness-only and schema fields.
    Raw causal matching must therefore be replayed from the immutable dataset
    release, while occurrence journals are computed from the exact tokenized
    rows that will be read from Silver during training.
    """
    from dataclasses import replace
    from .qwen9b_ordinal_continuity_controller import (
        matrix_plans, partition_identity, qualification_plan)
    from .qwen9b_ordinal_continuity_training import (
        adjacent_pair_schedule, canonical_sha256,
        load_simulator_condition,
        validate_equal_occurrence_exposure,
        validate_matched_conditions)
    decoded = []
    for row in rows:
        payload = row["record_json"]
        value = json.loads(payload)
        is_base = row.get("record_kind") == "base_qualification"
        expected_split = "qualification" if is_base else "train"
        if (row.get("split") != expected_split
                or value.get("split") != expected_split
                or bool(row.get("admitted_for_training")) == is_base
                or row.get("record_sha256")
                   != hashlib.sha256(payload.encode("utf-8")).hexdigest()):
            raise PermissionError("Silver binding input is not an exact admitted payload")
        decoded.append((row["record_kind"], value, row))
    by_kind = {}
    for kind, value, metadata in decoded:
        by_kind.setdefault(kind, []).append(value)
    conditions = {name: by_kind[name] for name in EXPERIMENT_SPEC["conditions"]}
    raw_conditions = {name: load_simulator_condition(generated_root, name)
                      for name in EXPERIMENT_SPEC["conditions"]}
    data_binding = validate_matched_conditions(raw_conditions)
    base_rows = by_kind["base_qualification"]
    qualification = replace(
        qualification_plan(EXPERIMENT_SPEC, 4), device_profile="databricks-a10x1",
        world_size=1, micro_batch_size=2, accumulation_steps=2).validate()
    policies = matrix_plans(EXPERIMENT_SPEC, 4)
    plans = [qualification]
    plans += [replace(plan, device_profile="databricks-a10x1", world_size=1,
                      micro_batch_size=2, accumulation_steps=2).validate()
              for plan in policies]
    plan = base_qualification_plan()
    base_meta = sorted(
        ({"record_id": row["record_id"], "record_sha256": row["record_sha256"]}
         for row in rows if row["record_kind"] == "base_qualification"),
        key=lambda row: row["record_id"])
    base_validation = {
        "record_count": len(base_rows),
        "family_counts": dict(sorted((family, sum(
            row.get("family_id") == family for row in base_rows))
            for family in {row.get("family_id") for row in base_rows})),
        "training_eligible_count": sum(
            bool(row.get("training_eligible")) for row in base_rows),
    }
    if (len(base_meta) != 288 or base_validation["training_eligible_count"] != 0
            or set(base_validation["family_counts"].values()) != {36}):
        raise PermissionError("Base qualification Silver binding differs")
    result = {BASE_QUALIFICATION_IDENTITY: {
        "version": "research4-databricks-silver-binding-v2",
        "partition_identity": BASE_QUALIFICATION_IDENTITY,
        "plan_sha256": plan["plan_sha256"],
        "silver_records_sha256": digest(base_meta),
        "silver_record_count": len(base_meta),
        "semantics_records_sha256": digest([]),
        "semantics_record_count": 0,
        "global_journal_sha256": digest(base_meta),
        "data_binding_sha256": digest(base_validation),
        "exposure_binding_sha256": digest({
            "frozen_base_only": True, "training_eligible": False,
            "optimizer_updates": 0, "records": len(base_meta)}),
    }}
    result[BASE_QUALIFICATION_IDENTITY]["binding_sha256"] = digest(
        result[BASE_QUALIFICATION_IDENTITY])
    for plan in plans:
        kind = plan.condition
        selected_meta = sorted(
            ({"record_id": row["record_id"], "record_sha256": row["record_sha256"]}
             for row in rows if row["record_kind"] == kind),
            key=lambda row: row["record_id"])
        exposure = validate_equal_occurrence_exposure(conditions, plan)
        binding = data_binding
        journal = adjacent_pair_schedule(conditions[plan.condition], plan)
        identity = partition_identity(plan)
        result[identity] = {
            "version": "research4-databricks-silver-binding-v2",
            "partition_identity": identity, "plan_sha256": plan.full_plan_sha256,
            "silver_records_sha256": digest(selected_meta),
            "silver_record_count": len(selected_meta),
            "semantics_records_sha256": digest([]),
            "semantics_record_count": 0,
            "global_journal_sha256": canonical_sha256(journal),
            "data_binding_sha256": canonical_sha256(binding),
            "exposure_binding_sha256": canonical_sha256(exposure),
        }
        result[identity]["binding_sha256"] = digest(result[identity])
    return result


def build_base_qualification_intent(*, source_commit: str,
                                    source_manifest: Mapping[str, Any],
                                    model_tree_sha256: str,
                                    model_receipt_sha256: str,
                                    silver_binding: Mapping[str, Any],
                                    mlflow_experiment_path: str,
                                    timeout_seconds: int = 2700,
                                    wall_clock_stop_seconds: int = 2400) -> dict:
    """Create the reviewed one-use zero-update frozen-base qualification intent."""
    manifest = validate_source_manifest(source_manifest)
    binding = validate_base_silver_binding(silver_binding)
    plan = base_qualification_plan()
    value = {
        "version": BASE_QUALIFICATION_INTENT_VERSION,
        "one_use": True,
        "attempt_number": 1,
        "automatic_retries": 0,
        "identity": BASE_QUALIFICATION_IDENTITY,
        "execution_platform": "databricks",
        "world_size": 1,
        "gpu_count": 1,
        "max_concurrent_runs": 1,
        "model_root": MODEL_ROOT,
        "silver_table": SILVER_TABLE,
        "output_root": (f"{VOLUME}/base_qualifications/{source_commit}/"
                        f"{BASE_QUALIFICATION_IDENTITY}"),
        "source_commit": source_commit,
        "source_manifest": manifest,
        "source_sha256": manifest["manifest_sha256"],
        "model_tree_sha256": model_tree_sha256,
        "model_receipt_sha256": model_receipt_sha256,
        "dataset_tree_sha256": DATASET_TREE_SHA256,
        "silver_records_sha256": binding.get("silver_records_sha256"),
        "silver_record_count": binding.get("silver_record_count"),
        "global_journal_sha256": binding.get("global_journal_sha256"),
        "admission_data_binding_sha256": binding.get("data_binding_sha256"),
        "admission_exposure_binding_sha256": binding.get(
            "exposure_binding_sha256"),
        "timeout_seconds": timeout_seconds,
        "max_gpu_hours": timeout_seconds / 3600,
        "wall_clock_stop_seconds": wall_clock_stop_seconds,
        "wall_clock_ceiling_seconds": timeout_seconds,
        "mlflow_experiment_path": mlflow_experiment_path,
        "plan": plan,
        "plan_sha256": plan["plan_sha256"],
    }
    value["intent_sha256"] = digest(value)
    return validate_base_qualification_intent(value)


def validate_base_qualification_intent(value: Mapping[str, Any]) -> dict:
    intent = dict(value)
    try:
        source_manifest = validate_source_manifest(intent.get("source_manifest"))
    except (TypeError, ValueError, PermissionError) as exc:
        raise PermissionError("Base qualification lacks a valid source manifest") from exc
    plan = intent.get("plan")
    required = {
        "version": BASE_QUALIFICATION_INTENT_VERSION,
        "one_use": True,
        "attempt_number": 1,
        "automatic_retries": 0,
        "identity": BASE_QUALIFICATION_IDENTITY,
        "execution_platform": "databricks",
        "world_size": 1,
        "gpu_count": 1,
        "max_concurrent_runs": 1,
        "model_root": MODEL_ROOT,
        "silver_table": SILVER_TABLE,
        "dataset_tree_sha256": DATASET_TREE_SHA256,
        "silver_record_count": 288,
        "timeout_seconds": 2700,
        "max_gpu_hours": 0.75,
        "wall_clock_stop_seconds": 2400,
        "wall_clock_ceiling_seconds": 2700,
    }
    if (plan != base_qualification_plan()
            or intent.get("plan_sha256") != plan.get("plan_sha256")
            or any(intent.get(key) != expected for key, expected in required.items())
            or not re.fullmatch(r"[0-9a-f]{40}", str(intent.get("source_commit", "")))
            or source_manifest.get("source_commit") != intent.get("source_commit")
            or source_manifest.get("manifest_sha256") != intent.get("source_sha256")
            or intent.get("output_root") != (
                f"{VOLUME}/base_qualifications/{intent.get('source_commit')}/"
                f"{BASE_QUALIFICATION_IDENTITY}")
            or intent.get("mlflow_experiment_path") in (None, "")
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(intent.get(key, "")))
                   for key in (
                       "model_tree_sha256", "model_receipt_sha256",
                       "silver_records_sha256", "global_journal_sha256",
                       "admission_data_binding_sha256",
                       "admission_exposure_binding_sha256"))):
        raise PermissionError("Frozen-base qualification intent differs")
    payload = dict(intent); embedded = payload.pop("intent_sha256", None)
    if embedded != digest(payload):
        raise PermissionError("Frozen-base qualification intent self-hash differs")
    return intent


def _validate_base_qualification_resources(value: Mapping[str, Any],
                                           model_tree_sha256: str) -> dict:
    resources = dict(value) if isinstance(value, Mapping) else {}
    required = {
        "gpu", "backend", "model_copy", "local_disk_free_before_copy",
        "elapsed_seconds", "inference_seconds", "prompt_tokens",
        "generated_tokens", "batch_size", "decisions_per_second",
        "peak_gpu_memory_bytes", "nvml", "operator_stop_seconds",
        "ceiling_seconds", "protected_export_seconds",
    }
    gpu = resources.get("gpu")
    backend = resources.get("backend")
    copy = resources.get("model_copy")
    nvml = resources.get("nvml")
    devices = gpu.get("devices") if isinstance(gpu, Mapping) else None
    device = devices[0] if isinstance(devices, list) and len(devices) == 1 else {}
    kernel = backend.get("kernel_selection") if isinstance(backend, Mapping) else None

    def finite_number(name: str, *, positive: bool = False) -> bool:
        item = resources.get(name)
        return (isinstance(item, (int, float)) and not isinstance(item, bool)
                and math.isfinite(item) and (item > 0 if positive else item >= 0))

    if (set(resources) != required
            or not isinstance(gpu, Mapping)
            or gpu.get("profile") != "databricks-a10x1"
            or gpu.get("compute_dtype") != "float16"
            or gpu.get("adapter_dtype") != "float32"
            or not isinstance(device, Mapping)
            or "A10" not in str(device.get("name", "")).upper()
            or type(device.get("index")) is not int or device.get("index") != 0
            or type(device.get("total_memory_bytes")) is not int
            or device.get("total_memory_bytes", 0) < 20 * 1024**3
            or not isinstance(backend, Mapping)
            or backend.get("model_class") != "Qwen3_5ForConditionalGeneration"
            or backend.get("linear_attention_layers") != 24
            or backend.get("full_attention_layers") != 8
            or not isinstance(kernel, Mapping) or kernel.get("profile") != "fla"
            or not isinstance(copy, Mapping)
            or copy.get("tree_sha256") != model_tree_sha256
            or not isinstance(copy.get("copy_bytes"), int)
            or copy.get("copy_bytes", 0) <= 0
            or not isinstance(copy.get("copy_seconds"), (int, float))
            or not math.isfinite(copy.get("copy_seconds", math.nan))
            or copy.get("copy_seconds", 0) <= 0
            or not isinstance(copy.get("copy_bytes_per_second"), (int, float))
            or not math.isfinite(copy.get("copy_bytes_per_second", math.nan))
            or copy.get("copy_bytes_per_second", 0) <= 0
            or type(copy.get("local_disk_free_before")) is not int
            or type(copy.get("local_disk_free_after")) is not int
            or copy.get("local_disk_free_before", 0)
               < copy.get("copy_bytes", 0) + 8 * 1024**3
            or copy.get("local_disk_free_after", 0) < 8 * 1024**3
            or resources.get("local_disk_free_before_copy")
               != copy.get("local_disk_free_before")
            or not finite_number("elapsed_seconds", positive=True)
            or resources.get("elapsed_seconds", 0) > 2700
            or not finite_number("inference_seconds", positive=True)
            or resources.get("inference_seconds", 0)
               > resources.get("elapsed_seconds", 0)
            or type(resources.get("prompt_tokens")) is not int
            or resources.get("prompt_tokens", 0) <= 0
            or type(resources.get("generated_tokens")) is not int
            or resources.get("generated_tokens", 0) <= 0
            or resources.get("batch_size") != 16
            or not finite_number("decisions_per_second", positive=True)
            or type(resources.get("peak_gpu_memory_bytes")) is not int
            or not 0 < resources.get("peak_gpu_memory_bytes", 0) <= device.get(
                "total_memory_bytes", 0)
            or not isinstance(nvml, Mapping)
            or set(nvml) != {"sample_count", "mean_utilization_percent",
                             "peak_memory_used_bytes"}
            or nvml.get("sample_count") != 18
            or not isinstance(nvml.get("mean_utilization_percent"), (int, float))
            or not math.isfinite(nvml.get("mean_utilization_percent", math.nan))
            or not 0 <= nvml.get("mean_utilization_percent", -1) <= 100
            or type(nvml.get("peak_memory_used_bytes")) is not int
            or not 0 < nvml.get("peak_memory_used_bytes", 0) <= device.get(
                "total_memory_bytes", 0)
            or resources.get("operator_stop_seconds") != 2400
            or resources.get("ceiling_seconds") != 2700
            or resources.get("protected_export_seconds") != 300):
        raise PermissionError("Frozen-base qualification resources differ")
    return resources


def validate_base_qualification_receipt(value: Mapping[str, Any]) -> dict:
    receipt = dict(value)
    payload = dict(receipt); embedded = payload.pop("receipt_sha256", None)
    family = receipt.get("family_strict_schema_and_legal")
    if (embedded != digest(payload)
            or receipt.get("version") != BASE_QUALIFICATION_RECEIPT_VERSION
            or receipt.get("status") != "qualified"
            or receipt.get("identity") != BASE_QUALIFICATION_IDENTITY
            or receipt.get("plan_sha256") != base_qualification_plan()["plan_sha256"]
            or receipt.get("adapter_parameter_count") != 0
            or receipt.get("optimizer_updates") != 0
            or receipt.get("backward_passes") != 0
            or receipt.get("decision_count") != 288
            or receipt.get("minimum_required") != 286
            or receipt.get("minimum_required_per_family") != 35
            or receipt.get("strict_schema_and_legal_count", 0) < 286
            or not 0 <= receipt.get("target_match_count_secondary", -1) <= 288
            or receipt.get("dataset_tree_sha256") != DATASET_TREE_SHA256
            or receipt.get("silver_records_sha256") is None
            or receipt.get("output_root") != (
                f"{VOLUME}/base_qualifications/{receipt.get('source_commit')}/"
                f"{BASE_QUALIFICATION_IDENTITY}")
            or not isinstance(family, Mapping)
            or set(family) != set(BASE_QUALIFICATION_FAMILIES)
            or set(family.values()) - {35, 36}
            or min(family.values(), default=0) < 35
            or sum(family.values()) != receipt.get(
                "strict_schema_and_legal_count")
            or not isinstance(receipt.get("resources"), Mapping)
            or receipt.get("base_component_sha256_before")
               != receipt.get("base_component_sha256_after")
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(key, "")))
                   for key in (
                       "source_sha256", "model_tree_sha256",
                       "model_receipt_sha256", "dataset_tree_sha256",
                       "base_component_sha256_before",
                       "base_component_sha256_after", "raw_results_sha256",
                       "silver_records_sha256"))
            or not re.fullmatch(r"[0-9a-f]{40}", str(
                receipt.get("source_commit", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(
                receipt.get("base_qualification_intent_sha256", "")))):
        raise PermissionError("Frozen-base qualification receipt differs")
    _validate_base_qualification_resources(
        receipt["resources"], receipt["model_tree_sha256"])
    return receipt


def create_base_qualification_receipt(intent: Mapping[str, Any],
                                      outcomes: list[Mapping[str, Any]], *,
                                      raw_results_sha256: str,
                                      base_component_sha256_before: str,
                                      base_component_sha256_after: str,
                                      resource_metrics: Mapping[str, Any]) -> dict:
    intent = validate_base_qualification_intent(intent)
    if (len(outcomes) != 288
            or len({row.get("record_id") for row in outcomes}) != 288
            or not re.fullmatch(r"[0-9a-f]{64}", raw_results_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", base_component_sha256_before)
            or base_component_sha256_before != base_component_sha256_after):
        raise PermissionError("Frozen-base qualification raw evidence differs")
    family: dict[str, int] = {}
    strict_total = target_total = 0
    outcome_binding = sorted(({
        "record_id": row.get("record_id"),
        "record_sha256": row.get("record_sha256"),
    } for row in outcomes), key=lambda row: str(row["record_id"]))
    if (any(not re.fullmatch(r"[0-9a-f]{64}", str(
            row.get("record_sha256", ""))) for row in outcome_binding)
            or digest(outcome_binding) != intent["silver_records_sha256"]):
        raise PermissionError("Frozen-base outcomes do not bind exact Silver payloads")
    for row in outcomes:
        name = row.get("family_id")
        output_text = row.get("output_text")
        allowed = row.get("allowed_actions")
        target = row.get("target_action_id")
        try:
            parsed = json.loads(output_text)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        derived_strict = (isinstance(parsed, dict)
                          and set(parsed) == {"action"}
                          and isinstance(parsed["action"], str))
        derived_action = parsed.get("action") if derived_strict else None
        derived_legal = (derived_strict and isinstance(allowed, list)
                         and derived_action in allowed)
        derived_target = derived_strict and derived_action == target
        strict = derived_strict and derived_legal
        if (not isinstance(name, str) or not name
                or not isinstance(output_text, str)
                or not isinstance(allowed, list) or not allowed
                or not isinstance(target, str)
                or hashlib.sha256(output_text.encode("utf-8")).hexdigest()
                   != row.get("output_sha256")
                or row.get("strict_json") is not derived_strict
                or row.get("legal_action") is not derived_legal
                or row.get("target_match") is not derived_target
                or row.get("action") != derived_action):
            raise PermissionError("Base qualification outcome differs")
        family.setdefault(name, 0)
        family[name] += int(strict)
        strict_total += int(strict)
        target_total += int(row.get("target_match") is True)
    qualified = (strict_total >= 286 and len(family) == 8
                 and min(family.values(), default=0) >= 35)
    value = {
        "version": BASE_QUALIFICATION_RECEIPT_VERSION,
        "status": "qualified" if qualified else "failed",
        "identity": BASE_QUALIFICATION_IDENTITY,
        "plan_sha256": intent["plan_sha256"],
        "source_commit": intent["source_commit"],
        "source_sha256": intent["source_sha256"],
        "model_tree_sha256": intent["model_tree_sha256"],
        "model_receipt_sha256": intent["model_receipt_sha256"],
        "dataset_tree_sha256": intent["dataset_tree_sha256"],
        "silver_records_sha256": intent["silver_records_sha256"],
        "output_root": intent["output_root"],
        "base_qualification_intent_sha256": intent["intent_sha256"],
        "adapter_parameter_count": 0,
        "optimizer_updates": 0,
        "backward_passes": 0,
        "decision_count": 288,
        "strict_schema_and_legal_count": strict_total,
        "target_match_count_secondary": target_total,
        "family_strict_schema_and_legal": dict(sorted(family.items())),
        "minimum_required": 286,
        "minimum_required_per_family": 35,
        "base_component_sha256_before": base_component_sha256_before,
        "base_component_sha256_after": base_component_sha256_after,
        "raw_results_sha256": raw_results_sha256,
        "resources": dict(resource_metrics),
    }
    value["receipt_sha256"] = digest(value)
    if qualified:
        return validate_base_qualification_receipt(value)
    return value


def validate_execution_intent(value: Mapping[str, Any]) -> dict:
    """Dispatch one-use lock validation without weakening either contract."""
    if value.get("version") == BASE_QUALIFICATION_INTENT_VERSION:
        return validate_base_qualification_intent(value)
    if value.get("version") == "research4-databricks-continuity-transfer-intent-v1":
        from .continuity_transfer_databricks import validate_transfer_intent
        return validate_transfer_intent(value)
    return validate_intent(value)


def validate_intent(value: Mapping[str, Any]) -> dict:
    intent = dict(value)
    try:
        source_manifest = validate_source_manifest(intent.get("source_manifest"))
    except (TypeError, ValueError, PermissionError) as exc:
        raise PermissionError("Databricks intent lacks a valid source manifest") from exc
    plan = intent.get("plan", {})
    mode = plan.get("mode")
    expected_updates = 2 if mode == "qualification" else 144
    required = {
        "version": INTENT_VERSION, "one_use": True, "attempt_number": 1,
        "automatic_retries": 0, "execution_platform": "databricks",
        "world_size": 1, "gpu_count": 1, "max_concurrent_runs": 1,
        "model_root": MODEL_ROOT, "silver_table": SILVER_TABLE,
    }
    from dataclasses import asdict, replace
    from .qwen9b_ordinal_continuity_controller import (
        matrix_plans, partition_identity, qualification_plan)
    from .qwen9b_ordinal_continuity_training import TrainingPlan
    try:
        exact_plan = TrainingPlan(**plan).validate()
    except (TypeError, ValueError, PermissionError) as exc:
        raise PermissionError("Databricks intent contains an invalid training plan") from exc
    qualification = replace(
        qualification_plan(EXPERIMENT_SPEC, 4),
        device_profile="databricks-a10x1", world_size=1,
        micro_batch_size=2, accumulation_steps=2).validate()
    policies = matrix_plans(EXPERIMENT_SPEC, 4)
    canonical_plans = [replace(
        item, device_profile="databricks-a10x1", world_size=1,
        micro_batch_size=2, accumulation_steps=2).validate()
        for item in policies]
    expected_plan = (qualification if mode == "qualification" else
                     next((item for item in canonical_plans
                           if (item.condition, item.seed, item.stage) ==
                           (exact_plan.condition, exact_plan.seed, exact_plan.stage)), None))
    if expected_plan is None or asdict(exact_plan) != asdict(expected_plan):
        raise PermissionError("Databricks plan differs from the canonical experiment matrix")
    if (any(intent.get(key) != expected for key, expected in required.items())
            or intent.get("identity") != partition_identity(exact_plan)
            or not re.fullmatch(r"[a-z0-9_]+", intent["identity"])
            or not isinstance(intent.get("source_commit"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", intent["source_commit"])
            or source_manifest.get("source_commit") != intent["source_commit"]
            or intent.get("source_sha256") != source_manifest.get("manifest_sha256")
            or intent.get("dataset_tree_sha256") != DATASET_TREE_SHA256
            or intent.get("silver_record_count") != 2304
            or not isinstance(intent.get("semantics_record_count"), int)
            or intent["semantics_record_count"] != 0
            or intent.get("semantics_records_sha256") != digest([])
            or any(not isinstance(intent.get(key), str)
                   or not re.fullmatch(r"[0-9a-f]{64}", intent[key]) for key in (
                       "model_tree_sha256", "model_receipt_sha256",
                       "silver_records_sha256", "semantics_records_sha256",
                       "global_journal_sha256", "admission_data_binding_sha256",
                       "admission_exposure_binding_sha256",
                       "base_qualification_intent_sha256",
                       "base_qualification_receipt_sha256"))
            or intent.get("mlflow_experiment_path") in (None, "")
            or not isinstance(intent.get("timeout_seconds"), int)
            or not 300 <= intent["timeout_seconds"] <= 14_400
            or not isinstance(intent.get("max_gpu_hours"), (int, float))
            or intent["max_gpu_hours"] <= 0
            or not isinstance(intent.get("wall_clock_stop_seconds"), int)
            or not isinstance(intent.get("wall_clock_ceiling_seconds"), int)
            or intent["wall_clock_ceiling_seconds"] != intent["timeout_seconds"]
            or intent["wall_clock_stop_seconds"] < 300
            or intent["wall_clock_stop_seconds"] > intent["wall_clock_ceiling_seconds"] - 300
            or plan.get("model_id") != MODEL_ID
            or plan.get("model_revision") != MODEL_REVISION
            or plan.get("rank") != 4 or plan.get("alpha") != 8.0
            or plan.get("global_batch_size") != 4
            or plan.get("world_size") != 1
            or plan.get("micro_batch_size") != 2
            or plan.get("accumulation_steps") != 2
            or plan.get("device_profile") != "databricks-a10x1"
            or plan.get("optimizer_updates") != expected_updates
            or plan.get("automatic_retries") != 0):
        raise PermissionError("Databricks one-use intent or scientific recipe differs")
    if mode == "qualification" and (
            intent["timeout_seconds"] != 2700 or intent["max_gpu_hours"] != 0.75):
        raise PermissionError("Qualification must be bounded to 2700s / 0.75 GPU-hour")
    budget = intent.get("reviewed_timeout_budget")
    if mode == "qualification" and budget is not None:
        raise PermissionError("Qualification cannot borrow a primary throughput budget")
    if mode == "primary":
        if not isinstance(budget, Mapping):
            raise PermissionError("Primary intent lacks a reviewed qualification budget")
        budget_payload = dict(budget); budget_hash = budget_payload.pop(
            "budget_sha256", None)
        observed = budget.get("qualification_tokens_per_second")
        maximum_tokens = (exact_plan.optimizer_updates
                          * exact_plan.global_batch_size * exact_plan.max_length)
        if (budget_hash != digest(budget_payload)
                or budget.get("version") != COST_BUDGET_VERSION
                or not re.fullmatch(r"[0-9a-f]{64}", str(
                    budget.get("qualification_receipt_sha256", "")))
                or not isinstance(observed, (int, float))
                or not math.isfinite(observed) or observed <= 0
                or budget.get("maximum_training_tokens") != maximum_tokens
                or budget.get("throughput_safety_factor") != 2.0
                or budget.get("fixed_overhead_seconds") != 3600
                or budget.get("derived_ceiling_seconds") != max(
                    600, math.ceil(maximum_tokens / observed * 2.0 + 3600))
                or budget["derived_ceiling_seconds"] > 14_400
                or intent["timeout_seconds"] != budget["derived_ceiling_seconds"]):
            raise PermissionError("Primary timeout is not qualification-derived")
    if (not math.isclose(intent["max_gpu_hours"],
                         intent["gpu_count"] * intent["wall_clock_ceiling_seconds"] / 3600,
                         rel_tol=0, abs_tol=1e-12)
            or intent["max_gpu_hours"] > 4.0):
        raise PermissionError("Declared GPU-hour budget differs from the hard ceiling")
    if intent.get("plan_sha256") != exact_plan.full_plan_sha256:
        raise PermissionError("Databricks intent plan hash differs")
    payload = dict(intent); embedded = payload.pop("intent_sha256", None)
    if embedded != digest(payload):
        raise PermissionError("Databricks one-use intent self-hash differs")
    return intent


def build_intent(*, plan, identity: str, source_commit: str,
                 source_manifest: Mapping[str, Any],
                 model_tree_sha256: str,
                 model_receipt_sha256: str,
                 silver_records_sha256: str, silver_record_count: int,
                 semantics_records_sha256: str, semantics_record_count: int,
                 global_journal_sha256: str,
                 admission_data_binding_sha256: str,
                 admission_exposure_binding_sha256: str,
                 timeout_seconds: int, max_gpu_hours: float,
                 wall_clock_stop_seconds: int,
                 mlflow_experiment_path: str,
                 base_qualification_receipt: Mapping[str, Any],
                 qualification_receipt: Mapping[str, Any] | None = None) -> dict:
    """Create a self-hashed not-submitted intent for external review."""
    from dataclasses import asdict
    manifest = validate_source_manifest(source_manifest)
    if manifest["source_commit"] != source_commit:
        raise PermissionError("Source manifest commit differs from the intent")
    base_receipt = validate_base_qualification_receipt(
        base_qualification_receipt)
    if (base_receipt["source_commit"] != source_commit
            or base_receipt["source_sha256"] != manifest["manifest_sha256"]
            or base_receipt["model_tree_sha256"] != model_tree_sha256
            or base_receipt["model_receipt_sha256"] != model_receipt_sha256
            or base_receipt["dataset_tree_sha256"] != DATASET_TREE_SHA256):
        raise PermissionError("Frozen-base qualification authority differs")
    value = {
        "version": INTENT_VERSION, "one_use": True, "attempt_number": 1,
        "automatic_retries": 0, "identity": identity,
        "execution_platform": "databricks", "world_size": 1,
        "gpu_count": 1, "max_concurrent_runs": 1,
        "model_root": MODEL_ROOT, "silver_table": SILVER_TABLE,
        "source_commit": source_commit, "timeout_seconds": timeout_seconds,
        "source_manifest": manifest,
        "source_sha256": manifest["manifest_sha256"],
        "model_tree_sha256": model_tree_sha256,
        "model_receipt_sha256": model_receipt_sha256,
        "dataset_tree_sha256": DATASET_TREE_SHA256,
        "silver_records_sha256": silver_records_sha256,
        "silver_record_count": silver_record_count,
        "semantics_records_sha256": semantics_records_sha256,
        "semantics_record_count": semantics_record_count,
        "global_journal_sha256": global_journal_sha256,
        "admission_data_binding_sha256": admission_data_binding_sha256,
        "admission_exposure_binding_sha256": admission_exposure_binding_sha256,
        "base_qualification_intent_sha256": base_receipt[
            "base_qualification_intent_sha256"],
        "base_qualification_receipt_sha256": base_receipt["receipt_sha256"],
        "max_gpu_hours": max_gpu_hours,
        "wall_clock_stop_seconds": wall_clock_stop_seconds,
        "wall_clock_ceiling_seconds": timeout_seconds,
        "mlflow_experiment_path": mlflow_experiment_path,
        "plan": asdict(plan.validate()), "plan_sha256": plan.full_plan_sha256,
    }
    if plan.mode == "primary":
        if qualification_receipt is None:
            raise PermissionError("Primary intent needs actual qualification throughput")
        capacity = dict(qualification_receipt)
        capacity_hash = capacity.pop("receipt_sha256", None)
        if (capacity_hash != digest(capacity)
                or qualification_receipt.get("status") != "qualified"
                or qualification_receipt.get(
                    "base_qualification_intent_sha256")
                   != base_receipt["base_qualification_intent_sha256"]
                or qualification_receipt.get(
                    "base_qualification_receipt_sha256")
                   != base_receipt["receipt_sha256"]):
            raise PermissionError("Capacity qualification uses a different base gate")
        value["reviewed_timeout_budget"] = primary_timeout_budget(
            plan, qualification_receipt)
    elif qualification_receipt is not None:
        raise PermissionError("Qualification intent cannot use prior qualification evidence")
    value["intent_sha256"] = digest(value)
    return validate_intent(value)


def canonical_primary_plan_hashes() -> list[str]:
    from dataclasses import replace
    from .qwen9b_ordinal_continuity_controller import matrix_plans
    policies = matrix_plans(EXPERIMENT_SPEC, 4)
    return sorted(replace(
        plan, device_profile="databricks-a10x1", world_size=1,
        micro_batch_size=2, accumulation_steps=2).validate().full_plan_sha256
        for plan in policies)


def create_qualification_receipt(intent: Mapping[str, Any],
                                 rank_receipt: Mapping[str, Any],
                                 evidence: Mapping[str, Any],
                                 checkpoint_tree_sha256: str) -> dict:
    intent = validate_intent(intent)
    rank_payload = dict(rank_receipt); rank_hash = rank_payload.pop("receipt_sha256", None)
    evidence_payload = dict(evidence); evidence_hash = evidence_payload.pop(
        "evidence_sha256", None)
    metrics = evidence.get("qualification_metrics", {})
    tokens_seen = rank_receipt.get("tokens_seen")
    elapsed_seconds = rank_receipt.get("elapsed_seconds")
    tokens_per_second = rank_receipt.get("resources", {}).get("tokens_per_second")
    if (intent["plan"]["mode"] != "qualification"
            or rank_hash != digest(rank_payload)
            or rank_receipt.get("status") != "complete"
            or rank_receipt.get("plan_sha256") != intent["plan_sha256"]
            or rank_receipt.get("successful_optimizer_updates") != 2
            or rank_receipt.get("active_branch_changed") is not True
            or rank_receipt.get("resources", {}).get("gpu", {}).get(
                "nvml", {}).get("nonzero_gpu_sample_count", 0) <= 0
            or evidence_hash != digest(evidence_payload)
            or evidence.get("intent_sha256") != intent["intent_sha256"]
            or evidence.get("rank_receipt_sha256") != rank_receipt.get("receipt_sha256")
            or evidence.get("live_base_components_before")
               != evidence.get("live_base_components_after")
            or evidence.get("save_reload_pass") is not True
            or any(value is not True for key, value in metrics.items()
                   if key.endswith("_pass") or key == "capacity_adequate")
            or not isinstance(tokens_seen, int) or tokens_seen <= 0
            or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0
            or not isinstance(tokens_per_second, (int, float))
            or not math.isfinite(tokens_per_second) or tokens_per_second <= 0
            or not math.isclose(tokens_per_second, tokens_seen / elapsed_seconds,
                                rel_tol=1e-9, abs_tol=1e-9)
            or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_tree_sha256)):
        raise PermissionError("Actual rank-4 Databricks qualification evidence differs")
    value = {
        "version": "research4-databricks-rank4-qualification-v1",
        "status": "qualified", "source_commit": intent["source_commit"],
        "source_sha256": intent["source_sha256"],
        "model_tree_sha256": intent["model_tree_sha256"],
        "model_receipt_sha256": intent["model_receipt_sha256"],
        "base_qualification_intent_sha256": intent[
            "base_qualification_intent_sha256"],
        "base_qualification_receipt_sha256": intent[
            "base_qualification_receipt_sha256"],
        "qualification_intent_sha256": intent["intent_sha256"],
        "qualification_plan_sha256": intent["plan_sha256"],
        "qualification_rank_receipt_sha256": rank_receipt["receipt_sha256"],
        "qualification_evidence_sha256": evidence_hash,
        "qualification_checkpoint_tree_sha256": checkpoint_tree_sha256,
        "qualification_tokens_seen": tokens_seen,
        "qualification_elapsed_seconds": elapsed_seconds,
        "qualification_tokens_per_second": tokens_per_second,
        "selected_rank": 4,
        "authorized_primary_plan_sha256s": canonical_primary_plan_hashes(),
        "qualifier_checkpoint_primary_parent_eligible": False,
    }
    value["receipt_sha256"] = digest(value)
    return value


def create_primary_admission(intent: Mapping[str, Any],
                             qualification_receipt: Mapping[str, Any]) -> dict:
    intent = validate_intent(intent)
    receipt = dict(qualification_receipt); embedded = receipt.pop("receipt_sha256", None)
    if (intent["plan"]["mode"] != "primary"
            or embedded != digest(receipt)
            or receipt.get("status") != "qualified"
            or receipt.get("selected_rank") != 4
            or receipt.get("source_commit") != intent["source_commit"]
            or receipt.get("source_sha256") != intent["source_sha256"]
            or receipt.get("model_tree_sha256") != intent["model_tree_sha256"]
            or receipt.get("model_receipt_sha256") != intent["model_receipt_sha256"]
            or receipt.get("base_qualification_intent_sha256")
               != intent.get("base_qualification_intent_sha256")
            or receipt.get("base_qualification_receipt_sha256")
               != intent.get("base_qualification_receipt_sha256")
            or intent.get("reviewed_timeout_budget", {}).get(
                "qualification_receipt_sha256") != embedded
            or intent.get("reviewed_timeout_budget", {}).get(
                "qualification_tokens_per_second")
               != receipt.get("qualification_tokens_per_second")
            or intent["plan_sha256"] not in receipt.get(
                "authorized_primary_plan_sha256s", ())):
        raise PermissionError("Qualification does not authorize this primary partition")
    value = {
        "version": "qwen35-9b-ordinal-continuity-primary-admission-v2",
        "admitted": True, "plan": intent["plan"],
        "full_plan_sha256": intent["plan_sha256"],
        "qualification_receipt_sha256": embedded,
        "qualification_evidence_sha256": receipt["qualification_evidence_sha256"],
        "qualification_checkpoint_tree_sha256": receipt[
            "qualification_checkpoint_tree_sha256"],
        "base_qualification_intent_sha256": intent[
            "base_qualification_intent_sha256"],
        "base_qualification_receipt_sha256": intent[
            "base_qualification_receipt_sha256"],
        "model_tree_sha256": intent["model_tree_sha256"],
        "model_receipt_sha256": intent["model_receipt_sha256"],
        "reviewed_timeout_budget_sha256": intent[
            "reviewed_timeout_budget"]["budget_sha256"],
        "data_binding_sha256": intent["admission_data_binding_sha256"],
        "exposure_binding_sha256": intent["admission_exposure_binding_sha256"],
        "source_sha256": intent["source_sha256"],
        "one_use_submission_intent": intent["intent_sha256"],
        "provider_submission_performed": False,
    }
    value["admission_sha256"] = digest(value)
    return value


def validate_primary_authority_files(
        intent: Mapping[str, Any], admission_path: str | Path,
        qualification_intent_path: str | Path,
        qualification_output: str | Path,
        qualification_receipt_path: str | Path) -> dict[str, dict]:
    """Replay immutable qualification evidence before any primary allocation."""
    from .qwen9b_ordinal_continuity_runtime import directory_tree_sha256
    primary_intent = validate_intent(intent)
    if primary_intent["plan"]["mode"] != "primary":
        raise PermissionError("Primary authority replay requires a primary intent")
    qualification_intent = validate_intent(json.loads(
        Path(qualification_intent_path).read_text(encoding="utf-8")))
    if qualification_intent["plan"]["mode"] != "qualification":
        raise PermissionError("Qualification intent has the wrong mode")
    if (qualification_intent.get("base_qualification_intent_sha256")
            != primary_intent.get("base_qualification_intent_sha256")
            or qualification_intent.get("base_qualification_receipt_sha256")
            != primary_intent.get("base_qualification_receipt_sha256")):
        raise PermissionError("Primary and capacity qualification use different base gate")
    root = Path(qualification_output)
    rank = json.loads((root / "RANK_0_RECEIPT.json").read_text(encoding="utf-8"))
    evidence = json.loads((root / "DATABRICKS_EVIDENCE.json").read_text(
        encoding="utf-8"))
    rebuilt_receipt = create_qualification_receipt(
        qualification_intent, rank, evidence,
        directory_tree_sha256(root / "checkpoint"))
    stored_receipt = json.loads(Path(qualification_receipt_path).read_text(
        encoding="utf-8"))
    if stored_receipt != rebuilt_receipt:
        raise PermissionError("Qualification receipt does not replay from raw evidence")
    stored_admission = json.loads(Path(admission_path).read_text(encoding="utf-8"))
    rebuilt_admission = create_primary_admission(primary_intent, stored_receipt)
    if stored_admission != rebuilt_admission:
        raise PermissionError("Primary admission does not replay from qualification")
    return {
        "admission": stored_admission,
        "qualification_intent": qualification_intent,
        "qualification_receipt": stored_receipt,
        "qualification_rank_receipt": rank,
        "qualification_evidence": evidence,
    }


def expected_predecessor_binding(plan: Mapping[str, Any]) -> dict | None:
    """Return the one canonical completed Gold row allowed to seed *plan*."""
    from dataclasses import replace
    from .qwen9b_ordinal_continuity_controller import matrix_plans, partition_identity
    from .qwen9b_ordinal_continuity_training import TrainingPlan
    current = TrainingPlan(**dict(plan)).validate()
    if current.stage == "delta1":
        return None
    policies = matrix_plans(EXPERIMENT_SPEC, 4)
    candidates = [replace(
        value, device_profile="databricks-a10x1", world_size=1,
        micro_batch_size=2, accumulation_steps=2).validate()
        for value in policies]
    expected_stage = {"delta2": "delta1", "delta3": "delta2"}[current.stage]
    expected_condition = current.condition
    expected_seed = current.seed
    matches = [value for value in candidates
               if (value.stage, value.condition, value.seed) ==
               (expected_stage, expected_condition, expected_seed)]
    if len(matches) != 1:
        raise PermissionError("Canonical predecessor plan is ambiguous or absent")
    predecessor = matches[0]
    return {"partition_identity": partition_identity(predecessor),
            "plan_sha256": predecessor.full_plan_sha256,
            "mode": "primary", "condition": expected_condition,
            "stage": expected_stage, "seed": expected_seed}


def validate_exact_gold_row(rows: list[Mapping[str, Any]],
                            expected: Mapping[str, Any], *, label: str) -> dict:
    """Require exactly one durable, complete and replayed Gold evidence row."""
    converted = [dict(row.asDict(recursive=True)) if hasattr(row, "asDict")
                 else dict(row) for row in rows]
    if len(converted) != 1:
        raise PermissionError(f"{label} requires exactly one Gold row")
    row = converted[0]
    if (row.get("status") != "complete" or row.get("receipt_replayed") is not True
            or any(row.get(key) != value for key, value in expected.items())):
        raise PermissionError(f"{label} Gold row differs from exact authority")
    return row


class VolumeOneUseLock:
    """Cross-job O_EXCL lock plus immutable consumed-intent receipt."""
    def __init__(self, lock_root: str | Path, intent: Mapping[str, Any], owner: str):
        self.root = Path(lock_root); self.intent = validate_execution_intent(intent)
        self.owner = owner; self.lock = self.root / "scientific.lock"
        self.consumed = self.root / "consumed" / (self.intent["identity"] + ".json")

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.consumed.parent.mkdir(parents=True, exist_ok=True)
        if self.consumed.exists():
            raise PermissionError("This Databricks partition identity is already consumed")
        descriptor = os.open(self.lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical({"owner": self.owner,
                                    "intent_sha256": self.intent["intent_sha256"]}) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        try:
            descriptor = os.open(self.consumed, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(canonical({"owner": self.owner,
                                        "intent_sha256": self.intent["intent_sha256"]}) + "\n")
                stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            try:
                current = json.loads(self.lock.read_text(encoding="utf-8"))
                if current.get("owner") == self.owner:
                    self.lock.unlink()
            finally:
                raise
        return self

    def __exit__(self, exc_type, exc, tb):
        current = json.loads(self.lock.read_text(encoding="utf-8"))
        if current.get("owner") != self.owner:
            raise PermissionError("Databricks scientific lock ownership changed")
        self.lock.unlink()


def reserve_intent(lock_root: str | Path, intent: Mapping[str, Any], owner: str) -> dict:
    """Consume the identity after GPU readiness and before any model handling."""
    intent = validate_execution_intent(intent); root = Path(lock_root); owner = str(owner)
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", owner):
        raise ValueError("Reservation owner is empty or malformed")
    root.mkdir(parents=True, exist_ok=True); (root / "consumed").mkdir(exist_ok=True)
    active = root / "scientific.lock"
    consumed = root / "consumed" / (intent["identity"] + ".json")
    if consumed.exists():
        raise PermissionError("This canonical identity was already consumed")
    descriptor = os.open(active, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    payload = {"owner": owner, "identity": intent["identity"],
               "intent_sha256": intent["intent_sha256"],
               "status": "reserved_after_gpu_ready_before_model"}
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(canonical(payload) + "\n"); stream.flush(); os.fsync(stream.fileno())
    try:
        descriptor = os.open(consumed, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical(payload) + "\n"); stream.flush(); os.fsync(stream.fileno())
    except BaseException:
        active.unlink(missing_ok=True)
        raise
    return payload


def verify_reserved_intent(lock_root: str | Path, intent: Mapping[str, Any],
                           owner: str) -> Path:
    intent = validate_execution_intent(intent); owner = str(owner)
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", owner):
        raise ValueError("Reservation owner is empty or malformed")
    active = Path(lock_root) / "scientific.lock"
    value = json.loads(active.read_text(encoding="utf-8"))
    if (value.get("owner") != owner or value.get("identity") != intent["identity"]
            or value.get("intent_sha256") != intent["intent_sha256"]
            or value.get("status") != "reserved_after_gpu_ready_before_model"):
        raise PermissionError("Pre-GPU global reservation differs")
    return active


def close_active_reservation(lock_root: str | Path, intent: Mapping[str, Any],
                             owner: str, terminal_status: str) -> dict:
    """Release only the exact owner's active lease; never free its consumed ID.

    The closeout receipt replaces the active lock in two atomic renames.  If the
    process dies between them, the next invocation recognizes the closeout
    payload still at ``scientific.lock`` and finishes the owner-bound rename.
    """
    intent = validate_execution_intent(intent); root = Path(lock_root); owner = str(owner)
    if (not owner or not isinstance(terminal_status, str)
            or not re.fullmatch(r"[a-z0-9_.:-]{1,128}", terminal_status)):
        raise ValueError("Closeout needs an exact owner and bounded status")
    active = root / "scientific.lock"
    consumed = root / "consumed" / (intent["identity"] + ".json")
    closeout_dir = root / "closeouts" / intent["identity"]
    owner_key = hashlib.sha256(owner.encode("utf-8")).hexdigest()
    closeout = closeout_dir / (owner_key + ".json")
    expected = {"owner": owner, "identity": intent["identity"],
                "intent_sha256": intent["intent_sha256"]}

    def check_reservation(value: Mapping[str, Any]) -> None:
        if (any(value.get(key) != expected_value
                for key, expected_value in expected.items())
                or value.get("status") != "reserved_after_gpu_ready_before_model"):
            raise PermissionError("Databricks reservation ownership differs")

    def check_closeout(value: Mapping[str, Any]) -> dict:
        payload = dict(value); embedded = payload.pop("receipt_sha256", None)
        if (payload.get("version") != CLOSEOUT_VERSION
                or any(payload.get(key) != expected_value
                       for key, expected_value in expected.items())
                or payload.get("consumed_identity_retained") is not True
                or embedded != digest(payload)):
            raise PermissionError("Databricks closeout receipt differs")
        return dict(value)

    if not consumed.is_file():
        raise PermissionError("Consumed identity is absent during closeout")
    check_reservation(json.loads(consumed.read_text(encoding="utf-8")))
    closeout_dir.mkdir(parents=True, exist_ok=True)
    if closeout.exists():
        receipt = check_closeout(json.loads(closeout.read_text(encoding="utf-8")))
        if active.exists():
            current = json.loads(active.read_text(encoding="utf-8"))
            if current.get("version") == CLOSEOUT_VERSION:
                check_closeout(current)
            else:
                check_reservation(current)
            active.unlink()
        return receipt
    if not active.exists():
        raise PermissionError("Active reservation disappeared without closeout")
    current = json.loads(active.read_text(encoding="utf-8"))
    if current.get("version") == CLOSEOUT_VERSION:
        receipt = check_closeout(current)
    else:
        check_reservation(current)
        receipt = {
            "version": CLOSEOUT_VERSION, **expected,
            "terminal_status": terminal_status,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "consumed_identity_retained": True,
            "reservation_sha256": digest(current),
        }
        receipt["receipt_sha256"] = digest(receipt)
        pending = closeout_dir / (".pending-" + owner_key + "-" +
                                  hashlib.sha256(os.urandom(32)).hexdigest() + ".json")
        descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical(receipt) + "\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(pending, active)
    os.replace(active, closeout)
    result = check_closeout(json.loads(closeout.read_text(encoding="utf-8")))
    check_reservation(json.loads(consumed.read_text(encoding="utf-8")))
    return result


def snapshot_inventory(model_root: str | Path) -> list[dict]:
    root = Path(model_root).resolve(); rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise PermissionError("Volume model snapshot contains a symlink")
        with path.open("rb") as stream:
            if stream.read(200).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise PermissionError("Volume model snapshot contains an LFS pointer")
        rows.append({"path": path.relative_to(root).as_posix(),
                     "sha256": file_sha256(path), "bytes": path.stat().st_size})
    if not rows:
        raise PermissionError("Volume model snapshot is empty")
    return rows


def _model_receipt_from_inventory(model_root: str | Path,
                                  files: list[dict]) -> dict:
    """Build the authoritative descriptor from one already-hashed inventory."""
    by_path = {row["path"]: row for row in files}
    required = {"config.json", "model.safetensors.index.json",
                "tokenizer.json", "tokenizer_config.json"}
    if not required <= set(by_path):
        raise PermissionError("Volume snapshot lacks config/index/tokenizer files")
    index = json.loads((Path(model_root) / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PermissionError("Volume snapshot has no safetensors weight map")
    shard_paths = sorted(set(weight_map.values()))
    if any(path not in by_path or not path.endswith(".safetensors") for path in shard_paths):
        raise PermissionError("Volume snapshot weight map references missing shards")
    shard_inventory = [by_path[path] for path in shard_paths]
    tokenizer_inventory = [by_path[path] for path in sorted(required - {
        "config.json", "model.safetensors.index.json"})]
    receipt = {"version": MODEL_RECEIPT_VERSION, "model_id": MODEL_ID,
               "model_revision": MODEL_REVISION, "resolved_commit": MODEL_REVISION,
               "files": files, "file_count": len(files),
               "total_bytes": sum(row["bytes"] for row in files),
               "tree_sha256": digest(files), "artifact_staging_only": True,
               "config_sha256": by_path["config.json"]["sha256"],
               "model_index_sha256": by_path["model.safetensors.index.json"]["sha256"],
               "weight_map_sha256": digest(weight_map),
               "shard_tree_sha256": digest(shard_inventory),
               "tokenizer_tree_sha256": digest(tokenizer_inventory),
               "model_loads": 0, "forwards": 0, "backwards": 0,
               "optimizer_updates": 0}
    receipt["receipt_sha256"] = digest(receipt)
    return receipt


def build_model_receipt(model_root: str | Path) -> dict:
    return _model_receipt_from_inventory(
        model_root, snapshot_inventory(model_root))


def _validate_model_receipt_descriptor(receipt: Mapping[str, Any]) -> dict:
    """Validate receipt authority without reading any model payload bytes."""
    value = dict(receipt) if isinstance(receipt, Mapping) else {}
    payload = dict(value); embedded = payload.pop("receipt_sha256", None)
    expected_fields = {
        "version", "model_id", "model_revision", "resolved_commit", "files",
        "file_count", "total_bytes", "tree_sha256", "artifact_staging_only",
        "config_sha256", "model_index_sha256", "weight_map_sha256",
        "shard_tree_sha256", "tokenizer_tree_sha256", "model_loads",
        "forwards", "backwards", "optimizer_updates", "receipt_sha256",
    }
    files = value.get("files")
    if (set(value) != expected_fields
            or embedded != digest(payload)
            or value.get("version") != MODEL_RECEIPT_VERSION
            or value.get("model_id") != MODEL_ID
            or value.get("model_revision") != MODEL_REVISION
            or value.get("resolved_commit") != MODEL_REVISION
            or not isinstance(files, list) or not files
            or value.get("file_count") != len(files)
            or value.get("artifact_staging_only") is not True
            or any(value.get(key) != 0 for key in (
                "model_loads", "forwards", "backwards", "optimizer_updates"))):
        raise PermissionError("Volume model snapshot receipt authority differs")
    paths = []
    for row in files:
        logical = row.get("path") if isinstance(row, Mapping) else None
        path = PurePosixPath(logical) if isinstance(logical, str) else None
        if (not isinstance(row, Mapping)
                or set(row) != {"path", "sha256", "bytes"}
                or path is None or path.is_absolute() or ".." in path.parts
                or path.as_posix() != logical
                or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", "")))
                or isinstance(row.get("bytes"), bool)
                or not isinstance(row.get("bytes"), int) or row["bytes"] < 0):
            raise PermissionError("Volume model snapshot receipt inventory differs")
        paths.append(logical)
    by_path = {row["path"]: row for row in files}
    required = {"config.json", "model.safetensors.index.json",
                "tokenizer.json", "tokenizer_config.json"}
    digest_fields = (
        "tree_sha256", "config_sha256", "model_index_sha256",
        "weight_map_sha256", "shard_tree_sha256", "tokenizer_tree_sha256",
        "receipt_sha256",
    )
    if (paths != sorted(paths) or len(set(paths)) != len(paths)
            or not required <= set(paths)
            or value.get("total_bytes") != sum(row["bytes"] for row in files)
            or value.get("tree_sha256") != digest(files)
            or value.get("config_sha256") != by_path["config.json"]["sha256"]
            or value.get("model_index_sha256")
               != by_path["model.safetensors.index.json"]["sha256"]
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(value.get(key, "")))
                   for key in digest_fields)):
        raise PermissionError("Volume model snapshot receipt descriptor differs")
    return value


def verify_model_receipt_authority(
        receipt_path: str | Path, *, expected_tree_sha256: str,
        expected_receipt_sha256: str) -> dict:
    """Verify the intent-bound receipt only; do not hash the Volume model."""
    if (not re.fullmatch(r"[0-9a-f]{64}", expected_tree_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_receipt_sha256)):
        raise ValueError("Expected model receipt authority must be SHA-256")
    receipt = _validate_model_receipt_descriptor(json.loads(
        Path(receipt_path).read_text(encoding="utf-8")))
    if (receipt["tree_sha256"] != expected_tree_sha256
            or receipt["receipt_sha256"] != expected_receipt_sha256):
        raise PermissionError("Intent and model receipt authority differ")
    return receipt


def verify_model_receipt(model_root: str | Path, receipt_path: str | Path) -> dict:
    receipt = _validate_model_receipt_descriptor(json.loads(
        Path(receipt_path).read_text(encoding="utf-8")))
    actual = snapshot_inventory(model_root)
    recomputed = _model_receipt_from_inventory(model_root, actual)
    if receipt != recomputed:
        raise PermissionError("Volume model snapshot receipt differs")
    return receipt


def commit_model_snapshot(staged_root: str | Path, model_root: str | Path,
                          receipt_path: str | Path) -> dict:
    """Copy a verified model into a UC Volume and publish its receipt last.

    Unity Catalog Volumes do not support atomically replacing a populated
    directory.  The final content-addressed directory is therefore claimed
    once, copied, fully rehashed, and only then made admissible by an exclusive
    receipt write.  A normal exception removes only paths claimed by this call.
    """
    staged = Path(staged_root); root = Path(model_root); receipt = Path(receipt_path)
    if root.exists() or receipt.exists():
        raise FileExistsError("Model snapshot destination is already claimed")
    value = build_model_receipt(staged)
    root_created = receipt_created = False
    try:
        root.mkdir(parents=False, exist_ok=False)
        root_created = True
        shutil.copytree(staged, root, dirs_exist_ok=True, copy_function=shutil.copy2)
        copied = build_model_receipt(root)
        if copied != value:
            raise PermissionError("Copied Volume model differs from staged receipt")
        descriptor = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        receipt_created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        return verify_model_receipt(root, receipt)
    except BaseException:
        if receipt_created:
            receipt.unlink(missing_ok=True)
        if root_created:
            shutil.rmtree(root, ignore_errors=True)
        raise


def gold_summary(base_qualification_receipt: Mapping,
                 capacity_qualification_receipt: Mapping,
                 capacity_gold_row: Mapping,
                 policy_receipts: list[Mapping]) -> dict:
    base = validate_base_qualification_receipt(base_qualification_receipt)
    capacity_payload = dict(capacity_qualification_receipt)
    capacity_hash = capacity_payload.pop("receipt_sha256", None)
    if (capacity_hash != digest(capacity_payload)
            or capacity_qualification_receipt.get("status") != "qualified"
            or capacity_qualification_receipt.get("selected_rank") != 4
            or capacity_qualification_receipt.get(
                "authorized_primary_plan_sha256s")
               != canonical_primary_plan_hashes()
            or capacity_qualification_receipt.get(
                "base_qualification_receipt_sha256") != base["receipt_sha256"]
            or capacity_gold_row.get("status") != "complete"
            or capacity_gold_row.get("mode") != "qualification"
            or capacity_gold_row.get("optimizer_updates") != 2
            or capacity_gold_row.get("stage") != "delta1"
            or capacity_gold_row.get("parent_checkpoint_tree_sha256") is not None
            or capacity_gold_row.get("receipt_replayed") is not True
            or capacity_gold_row.get("qualification_receipt_sha256")
               != base["receipt_sha256"]
            or len(policy_receipts) != 27
            or any(row.get("status") != "complete"
                   or row.get("optimizer_updates") != 144
                   or row.get("receipt_replayed") is not True
                   or row.get("qualification_receipt_sha256") != capacity_hash
                   for row in policy_receipts)):
        raise PermissionError(
            "Gold summary requires frozen base + capacity qualifier + 27 policies")
    conditions = ("self_continuity", "artifact_continuity", "label_only")
    seeds = (202610281, 202610282, 202610283)
    stages = ("delta1", "delta2", "delta3")
    by_key = {(row["condition"], row["seed"], row["stage"]): row
              for row in policy_receipts}
    if set(by_key) != {(condition, seed, stage) for condition in conditions
                       for seed in seeds for stage in stages}:
        raise PermissionError("Gold policy rows are incomplete or duplicated")
    all_rows = [capacity_gold_row, *policy_receipts]
    if (len({row.get("source_commit") for row in all_rows}
            | {base.get("source_commit")}) != 1
            or len({row.get("model_tree_sha256") for row in all_rows}
                   | {base.get("model_tree_sha256")}) != 1
            or sorted(row.get("plan_sha256") for row in policy_receipts)
               != canonical_primary_plan_hashes()):
        raise PermissionError("Gold rows do not bind one canonical source/model/matrix")
    for condition in conditions:
        for seed in seeds:
            parent = None
            for stage in stages:
                row = by_key[(condition, seed, stage)]
                if row.get("parent_checkpoint_tree_sha256") != parent:
                    raise PermissionError("Gold checkpoint predecessor chain differs")
                parent = row.get("checkpoint_tree_sha256")
    value = {
             "status": "complete", "policy_partition_count": 27,
             "baseline": "frozen_base",
             "frozen_base_model_sha256": base["model_tree_sha256"],
             "base_qualification_receipt_sha256": base["receipt_sha256"],
             "capacity_qualification_receipt_sha256": capacity_hash,
             "capacity_qualification_rank_receipt_sha256":
                 capacity_gold_row.get("receipt_sha256"),
             "policy_receipt_sha256": sorted(
                 row.get("receipt_sha256") for row in policy_receipts)}
    value["summary_sha256"] = digest(value)
    return value
