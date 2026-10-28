"""Databricks-only storage and execution contract for simulated continuity.

This module performs no deployment and starts no compute.  It centralizes the
Unity Catalog names, deterministic bronze/silver transforms, immutable model
snapshot verification, and the shared one-use lock used by qualification and
primary GPU partitions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json
import math
import os
import re


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
GOLD_SUMMARY_TABLE = f"{GOLD}.ordinal_continuity_matrix_summary"
GOLD_ADMISSION_TABLE = f"{GOLD}.execution_admissions"
INTENT_VERSION = "research4-databricks-continuity-one-use-intent-v1"
MODEL_RECEIPT_VERSION = "research4-databricks-qwen35-volume-snapshot-v1"
CLOSEOUT_VERSION = "research4-databricks-active-reservation-closeout-v1"
COST_BUDGET_VERSION = "research4-databricks-throughput-derived-budget-v1"
DATASET_TREE_SHA256 = "c21d8b334fdd1e7544fbea8277643f594c990dd5a754312d5e5a9e0d947df1df"
DATASET_RELEASE_ROOT = f"{VOLUME}/dataset_releases/{DATASET_TREE_SHA256}"
DATASET_RELEASE_RECEIPT_SHA256 = (
    "ec4c478f84abd7e038dbb5de557a57e3bba142f86abd916457084636545a01ca"
)
DATASET_RELEASE_VALIDATION_SHA256 = (
    "2f92b6595c576518ab8deee50cdee2034830d8ebb63b66d46b967ed08c6fc005"
)
EXPERIMENT_SPEC = {
    "version": "qwen35-9b-ordinal-continuity-experiment-v1",
    "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
    "conditions": ["self_continuity", "artifact_continuity", "label_only"],
    "seeds": [202610281, 202610282, 202610283],
    "stages": ["delta1", "delta2", "delta3"],
    "qualification_updates": 2, "skill_optimizer_updates": 24,
    "primary_optimizer_updates": 144, "skill_rank": 4,
    "skill_alpha": 8.0, "automatic_retries": 0, "one_use": True,
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
          canonical_skill_checkpoint_sha256 STRING,
          policy_partition_count INT NOT NULL, summary_sha256 STRING NOT NULL,
          summary_json STRING NOT NULL, audited_at TIMESTAMP NOT NULL
        ) USING DELTA""",
        f"""CREATE TABLE IF NOT EXISTS {GOLD_ADMISSION_TABLE} (
          identity STRING NOT NULL, intent_sha256 STRING NOT NULL,
          job_run_id STRING NOT NULL, status STRING NOT NULL,
          consumed_at TIMESTAMP NOT NULL
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


def silver_rows(generated_root: str | Path, source_commit: str, *, tokenizer,
                max_length: int = 1024) -> list[dict]:
    """Build the exact loader-admitted training records, excluding heldout."""
    from .qwen9b_ordinal_continuity_training import (
        CONDITIONS, equalize_matched_prompt_exposure, load_danger_semantics_rows,
        load_neutral_skill_rows, load_simulator_condition, verify_dataset_receipt)
    from .qwen9b_ordinal_continuity_training import tokenize_rows
    root = Path(generated_root).resolve()
    receipt = verify_dataset_receipt(root)
    neutral = tokenize_rows(load_neutral_skill_rows(root), tokenizer, max_length)
    semantics = tokenize_rows(load_danger_semantics_rows(root), tokenizer, max_length)
    by_condition = {condition: tokenize_rows(
        load_simulator_condition(root, condition), tokenizer, max_length)
        for condition in CONDITIONS}
    by_condition = equalize_matched_prompt_exposure(
        by_condition, tokenizer, max_length)
    records = [("neutral_skill", row) for row in neutral]
    records += [("danger_semantics", row) for row in semantics]
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
    if not output or any(row["split"] != "train" for row in output):
        raise PermissionError("Silver training selection is empty or contains heldout rows")
    return output


def silver_partition_bindings(rows: list[Mapping[str, Any]]) -> dict[str, dict]:
    """Recompute exact matched exposure/journals for every canonical partition."""
    from dataclasses import replace
    from .qwen9b_ordinal_continuity_controller import (
        matrix_plans, partition_identity, qualification_plan)
    from .qwen9b_ordinal_continuity_training import (
        adjacent_pair_schedule, build_occurrence_journal, canonical_sha256,
        occurrence_key_schedule, validate_equal_occurrence_exposure,
        validate_matched_conditions, validate_neutral_skill_schema)
    decoded = [(row["record_kind"], json.loads(row["record_json"]), row)
               for row in rows]
    by_kind = {}
    for kind, value, metadata in decoded:
        by_kind.setdefault(kind, []).append(value)
    conditions = {name: by_kind[name] for name in EXPERIMENT_SPEC["conditions"]}
    data_binding = validate_matched_conditions(conditions)
    semantics = by_kind["danger_semantics"]
    qualification = replace(
        qualification_plan(EXPERIMENT_SPEC, 4), device_profile="databricks-a10x1",
        world_size=1, micro_batch_size=2, accumulation_steps=2).validate()
    skill, policies = matrix_plans(EXPERIMENT_SPEC, 4)
    plans = [qualification, replace(
        skill, device_profile="databricks-a10x1", world_size=1,
        micro_batch_size=2, accumulation_steps=2).validate()]
    plans += [replace(plan, device_profile="databricks-a10x1", world_size=1,
                      micro_batch_size=2, accumulation_steps=2).validate()
              for plan in policies]
    result = {}
    for plan in plans:
        kind = "neutral_skill" if plan.stage == "skill" else plan.condition
        selected_meta = sorted(
            ({"record_id": row["record_id"], "record_sha256": row["record_sha256"]}
             for row in rows if row["record_kind"] == kind),
            key=lambda row: row["record_id"])
        semantic_meta = sorted(
            ({"record_id": row["record_id"], "record_sha256": row["record_sha256"]}
             for row in rows if row["record_kind"] == "danger_semantics"),
            key=lambda row: row["record_id"])
        if plan.stage == "skill":
            semantic_meta = []
            records = by_kind["neutral_skill"]
            key_schedule = occurrence_key_schedule(records, plan)
            _, journal = build_occurrence_journal(records, plan, key_schedule=key_schedule)
            binding = validate_neutral_skill_schema(records)
            exposure = {"global_journal_sha256": canonical_sha256(journal),
                        "global_journal_occurrence_count": len(journal),
                        "key_schedule_sha256": canonical_sha256(key_schedule)}
        else:
            exposure = validate_equal_occurrence_exposure(
                conditions, plan, common_semantics_records=semantics)
            binding = data_binding
            journal = adjacent_pair_schedule(conditions[plan.condition], plan)
        identity = partition_identity(plan)
        result[identity] = {
            "version": "research4-databricks-silver-binding-v1",
            "partition_identity": identity, "plan_sha256": plan.full_plan_sha256,
            "silver_records_sha256": digest(selected_meta),
            "silver_record_count": len(selected_meta),
            "semantics_records_sha256": digest(semantic_meta),
            "semantics_record_count": len(semantic_meta),
            "global_journal_sha256": canonical_sha256(journal),
            "data_binding_sha256": canonical_sha256(binding),
            "exposure_binding_sha256": canonical_sha256(exposure),
        }
        result[identity]["binding_sha256"] = digest(result[identity])
    return result


def validate_intent(value: Mapping[str, Any]) -> dict:
    intent = dict(value)
    plan = intent.get("plan", {})
    mode, stage = plan.get("mode"), plan.get("stage")
    expected_updates = 2 if mode == "qualification" else 24 if stage == "skill" else 144
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
    skill, policies = matrix_plans(EXPERIMENT_SPEC, 4)
    canonical_plans = [replace(
        item, device_profile="databricks-a10x1", world_size=1,
        micro_batch_size=2, accumulation_steps=2).validate()
        for item in [skill, *policies]]
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
            or not isinstance(intent.get("source_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", intent["source_sha256"])
            or intent.get("dataset_tree_sha256") != DATASET_TREE_SHA256
            or not isinstance(intent.get("silver_record_count"), int)
            or intent["silver_record_count"] <= 0
            or not isinstance(intent.get("semantics_record_count"), int)
            or intent["semantics_record_count"] < 0
            or any(not isinstance(intent.get(key), str)
                   or not re.fullmatch(r"[0-9a-f]{64}", intent[key]) for key in (
                       "model_tree_sha256", "model_receipt_sha256",
                       "silver_records_sha256", "semantics_records_sha256",
                       "global_journal_sha256", "admission_data_binding_sha256",
                       "admission_exposure_binding_sha256"))
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
                 source_sha256: str,
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
                 qualification_receipt: Mapping[str, Any] | None = None) -> dict:
    """Create a self-hashed not-submitted intent for external review."""
    from dataclasses import asdict
    value = {
        "version": INTENT_VERSION, "one_use": True, "attempt_number": 1,
        "automatic_retries": 0, "identity": identity,
        "execution_platform": "databricks", "world_size": 1,
        "gpu_count": 1, "max_concurrent_runs": 1,
        "model_root": MODEL_ROOT, "silver_table": SILVER_TABLE,
        "source_commit": source_commit, "timeout_seconds": timeout_seconds,
        "source_sha256": source_sha256,
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
        "max_gpu_hours": max_gpu_hours,
        "wall_clock_stop_seconds": wall_clock_stop_seconds,
        "wall_clock_ceiling_seconds": timeout_seconds,
        "mlflow_experiment_path": mlflow_experiment_path,
        "plan": asdict(plan.validate()), "plan_sha256": plan.full_plan_sha256,
    }
    if plan.mode == "primary":
        if qualification_receipt is None:
            raise PermissionError("Primary intent needs actual qualification throughput")
        value["reviewed_timeout_budget"] = primary_timeout_budget(
            plan, qualification_receipt)
    elif qualification_receipt is not None:
        raise PermissionError("Qualification intent cannot use prior qualification evidence")
    value["intent_sha256"] = digest(value)
    return validate_intent(value)


def canonical_primary_plan_hashes() -> list[str]:
    from dataclasses import replace
    from .qwen9b_ordinal_continuity_controller import matrix_plans
    skill, policies = matrix_plans(EXPERIMENT_SPEC, 4)
    return sorted(replace(
        plan, device_profile="databricks-a10x1", world_size=1,
        micro_batch_size=2, accumulation_steps=2).validate().full_plan_sha256
        for plan in [skill, *policies])


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
    if current.stage == "skill":
        return None
    skill, policies = matrix_plans(EXPERIMENT_SPEC, 4)
    candidates = [replace(
        value, device_profile="databricks-a10x1", world_size=1,
        micro_batch_size=2, accumulation_steps=2).validate()
        for value in [skill, *policies]]
    expected_stage = {"delta1": "skill", "delta2": "delta1",
                      "delta3": "delta2"}[current.stage]
    expected_condition = "neutral_skill" if expected_stage == "skill" else current.condition
    expected_seed = skill.seed if expected_stage == "skill" else current.seed
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
        self.root = Path(lock_root); self.intent = validate_intent(intent)
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
    """Persist the active lease and consumed identity before GPU allocation."""
    intent = validate_intent(intent); root = Path(lock_root)
    root.mkdir(parents=True, exist_ok=True); (root / "consumed").mkdir(exist_ok=True)
    active = root / "scientific.lock"
    consumed = root / "consumed" / (intent["identity"] + ".json")
    if consumed.exists():
        raise PermissionError("This canonical identity was already consumed")
    descriptor = os.open(active, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    payload = {"owner": str(owner), "identity": intent["identity"],
               "intent_sha256": intent["intent_sha256"], "status": "reserved_before_gpu"}
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
    intent = validate_intent(intent); active = Path(lock_root) / "scientific.lock"
    value = json.loads(active.read_text(encoding="utf-8"))
    if (value.get("owner") != str(owner) or value.get("identity") != intent["identity"]
            or value.get("intent_sha256") != intent["intent_sha256"]
            or value.get("status") != "reserved_before_gpu"):
        raise PermissionError("Pre-GPU global reservation differs")
    return active


def close_active_reservation(lock_root: str | Path, intent: Mapping[str, Any],
                             owner: str, terminal_status: str) -> dict:
    """Release only the exact owner's active lease; never free its consumed ID.

    The closeout receipt replaces the active lock in two atomic renames.  If the
    process dies between them, the next invocation recognizes the closeout
    payload still at ``scientific.lock`` and finishes the owner-bound rename.
    """
    intent = validate_intent(intent); root = Path(lock_root); owner = str(owner)
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
                or value.get("status") != "reserved_before_gpu"):
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


def build_model_receipt(model_root: str | Path) -> dict:
    files = snapshot_inventory(model_root)
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


def verify_model_receipt(model_root: str | Path, receipt_path: str | Path) -> dict:
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    payload = dict(receipt); embedded = payload.pop("receipt_sha256", None)
    actual = snapshot_inventory(model_root)
    recomputed = build_model_receipt(model_root)
    if (embedded != digest(payload) or receipt.get("version") != MODEL_RECEIPT_VERSION
            or receipt.get("model_id") != MODEL_ID
            or receipt.get("model_revision") != MODEL_REVISION
            or receipt.get("resolved_commit") != MODEL_REVISION
            or receipt.get("files") != actual
            or receipt.get("tree_sha256") != digest(actual)
            or any(receipt.get(key) != recomputed.get(key) for key in (
                "config_sha256", "model_index_sha256", "weight_map_sha256",
                "shard_tree_sha256", "tokenizer_tree_sha256"))
            or receipt.get("artifact_staging_only") is not True
            or any(receipt.get(key) != 0 for key in (
                "model_loads", "forwards", "backwards", "optimizer_updates"))):
        raise PermissionError("Volume model snapshot receipt differs")
    return receipt


def gold_summary(skill_receipt: Mapping, policy_receipts: list[Mapping]) -> dict:
    if (skill_receipt.get("status") != "complete"
            or skill_receipt.get("optimizer_updates") != 24
            or skill_receipt.get("stage") != "skill"
            or skill_receipt.get("receipt_replayed") is not True
            or len(policy_receipts) != 27
            or any(row.get("status") != "complete"
                   or row.get("optimizer_updates") != 144
                   or row.get("receipt_replayed") is not True
                   for row in policy_receipts)):
        raise PermissionError("Gold summary requires the complete audited 1+27 matrix")
    conditions = ("self_continuity", "artifact_continuity", "label_only")
    seeds = (202610281, 202610282, 202610283)
    stages = ("delta1", "delta2", "delta3")
    by_key = {(row["condition"], row["seed"], row["stage"]): row
              for row in policy_receipts}
    if set(by_key) != {(condition, seed, stage) for condition in conditions
                       for seed in seeds for stage in stages}:
        raise PermissionError("Gold policy rows are incomplete or duplicated")
    all_rows = [skill_receipt, *policy_receipts]
    if (len({row.get("source_commit") for row in all_rows}) != 1
            or len({row.get("model_tree_sha256") for row in all_rows}) != 1
            or sorted(row.get("plan_sha256") for row in all_rows)
               != canonical_primary_plan_hashes()):
        raise PermissionError("Gold rows do not bind one canonical source/model/matrix")
    skill_tree = skill_receipt.get("checkpoint_tree_sha256")
    for condition in conditions:
        for seed in seeds:
            parent = skill_tree
            for stage in stages:
                row = by_key[(condition, seed, stage)]
                if row.get("parent_checkpoint_tree_sha256") != parent:
                    raise PermissionError("Gold checkpoint predecessor chain differs")
                parent = row.get("checkpoint_tree_sha256")
    value = {"status": "complete", "policy_partition_count": 27,
             "canonical_skill_checkpoint_sha256": skill_tree,
             "skill_receipt_sha256": skill_receipt.get("receipt_sha256"),
             "policy_receipt_sha256": sorted(
                 row.get("receipt_sha256") for row in policy_receipts)}
    value["summary_sha256"] = digest(value)
    return value
