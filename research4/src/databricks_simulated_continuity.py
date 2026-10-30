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
import ast
import hashlib
import json
import math
import os
import re
import shutil

from .action_serialization import (
    SERIALIZATION_CONTRACT_VERSION,
    parse_registered_action,
    serialization_contract,
)
from .base_qualification_carryforward import (
    validate_carryforward,
    verify_deployed_carryforward,
)
from .qualification_technical_successor import (
    authority as qualification_technical_successor_authority,
    validate as validate_qualification_technical_successor,
)


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
INTENT_VERSION = "research4-databricks-continuity-one-use-intent-v11"
DATABRICKS_EVIDENCE_VERSION = "research4-databricks-partition-evidence-v5"
DATABRICKS_QUALIFICATION_VERSION = (
    "research4-databricks-rank4-qualification-v7")
QUALIFICATION_SOURCE_CARRYFORWARD_VERSION = (
    "research4-qualification-source-carryforward-v1")
QUALIFICATION_SOURCE_CARRYFORWARD_ALLOWED_CHANGED_PATHS = (
    "notebooks/databricks_continuity_partition_finalize.py",
    "notebooks/databricks_continuity_partition.py",
    "notebooks/databricks_continuity_preflight.py",
    "resources/simulated_continuity_databricks.job.yml",
    "src/base_qualification_carryforward.py",
    "src/databricks_continuity_partition.py",
    "src/databricks_simulated_continuity.py",
    "tools/prepare_databricks_continuity_intent.py",
)
QUALIFICATION_SOURCE_IMMUTABLE_SCIENCE_PATHS = (
    "src/action_serialization.py",
    "src/composite_kernels.py",
    "src/databricks_continuity_partition_worker.py",
    "src/qwen9b_ordinal_continuity_controller.py",
    "src/qwen9b_ordinal_continuity_lora.py",
    "src/qwen9b_ordinal_continuity_runtime.py",
    "src/qwen9b_ordinal_continuity_training.py",
    "src/qwen9b_ordinal_continuity_worker.py",
)
QUALIFICATION_SOURCE_ORIGINAL_COMMIT = (
    "832ffba93ac11a276adb9139795b51554000fe32")
QUALIFICATION_SOURCE_ORIGINAL_GIT_TREE = (
    "cf9838025a597d9a3613d50706539de0a3554540")
QUALIFICATION_SOURCE_ORIGINAL_MANIFEST_SHA256 = (
    "5b36d597d9634cf59a69538d98ebfd95840138867cd49c2b31454989e50cfb80")
QUALIFICATION_SOURCE_REVIEWED_SUCCESSOR_COMMIT = (
    "30caed8655ca1b77c3b0c9156e8dc42c005af309")
QUALIFICATION_SOURCE_REVIEWED_SUCCESSOR_GIT_TREE = (
    "4f509c922e628d427ae9d55a0d96b1d91166c85a")
QUALIFICATION_SOURCE_REVIEWED_SUCCESSOR_MANIFEST_SHA256 = (
    "50211b7e7395e508eb870eec9e6b1d5ba0d286bc6511425b5d836eca2a9ef2ec")
QUALIFICATION_SOURCE_REVIEWED_DIFF_SHA256 = (
    "426d070ecf7ba00c43abee4f2d66613e1484dab639ee6810c04c6016c7678b0d")
QUALIFICATION_SOURCE_REVIEWED_CHANGED_PATHS = (
    "notebooks/databricks_continuity_partition_finalize.py",
    "resources/simulated_continuity_databricks.job.yml",
    "src/databricks_continuity_partition.py",
)
QUALIFICATION_RUN_PARTITION_EXECUTION_PREFIX_SHA256 = (
    "7e58ed81cc8eb78cb4ce5f8d1229143621850d8f5a6c1d293fb21ee841419d6a")
QUALIFICATION_RECEIPT_SHA256 = (
    "5d6001ebbfd33e9a5cb8e488e54510bd4642418c8a061b1125f431ee9c55e89e")
QUALIFICATION_RECEIPT_FILE_SHA256 = (
    "8f07a8df893ffca799ed2383e29a424a0faed6370aa6539e37a7dcc169ccfd6d")
QUALIFICATION_INTENT_SHA256 = (
    "41fa0f68729f6df7cc5e41605e3e0e69c487085e6bd8745f63089c467d09ccb8")
QUALIFICATION_PLAN_SHA256 = (
    "83a3e4a213316cba3ae21c9b83ebbf0fa631cfdbed5b615a158bf6495bac31bc")
QUALIFICATION_RANK_RECEIPT_SHA256 = (
    "3bccd1c2b4e68d83a7914c146bb1deb9ba6b5ee182218b39b6270f5357e4fdad")
QUALIFICATION_EVIDENCE_SHA256 = (
    "c17d2cce67981f4653127da5bf492aaa766948cbc50adc1dc8b629d37ca88396")
QUALIFICATION_CHECKPOINT_TREE_SHA256 = (
    "1e043f3cbfdb3227300cd7831715ae3838ab5c12a96e2db0ccfd39a9599c7967")
QUALIFICATION_FINALIZATION_RESUME_CLAIM_SHA256 = (
    "ae1d805b3fa3c8752e58c3a47e89d7f41b9b03eb3a9bb0e332b37e8f61bfe3e6")
QUALIFICATION_FINALIZATION_RESUME_CLAIM_FILE_SHA256 = (
    "b3934b57ad1c5344dccd24a6f7168af9f90d7505cba074c643a1c24740bd03a0")
QUALIFICATION_RECOVERY_RESULT_SHA256 = (
    "259370926b7e3c19a076c6f1b326ded63f5d06ff50889be977735cadc3552f7f")
QUALIFICATION_RECOVERY_RESULT_FILE_SHA256 = (
    "028c765353c63c1d2fd024fe24616bc9f12fa00464f8b1888d12b530d4893241")
QUALIFICATION_PARENT_FINALIZATION_SHA256 = (
    "e1935c4d7ed0e209e79d67af14a79b6bcff61c3f1761670316fe093e4ebd334f")
QUALIFICATION_PARENT_FINALIZATION_FILE_SHA256 = (
    "3eb8273e09351dbd5136dc18ea0b238bb242a2af404215f9accc998168a531e9")
QUALIFICATION_PARENT_MANIFEST_SHA256 = (
    "d256a0f6030a684448d249503eedeba7a01ec1bf4809a550e228f715467a0ccf")
QUALIFICATION_PARENT_MANIFEST_FILE_SHA256 = (
    "1ec2bfda4c528cc0b51321a91411c5be26c8a427f18b7d236cc0548af618d606")
QUALIFICATION_REVIEWED_CHANGED_SYMBOL_SHA256 = {
    "_expected_mlflow_scalars": {
        "prior": "9479b76781cc24a4d5e6a1890ee1ab4be18407d80fa4d158bf30a2a0be0bef11",
        "reviewed": "e34c22a859af34007e78fd3c011f489500976476b7396b9d36608ca1cf69cff1",
    },
    "run_partition": {
        "prior": "5b286f3dce8a8ac5886711e085216ae501a6eda9cdf3deeee16dea1651baa92d",
        "reviewed": "dc2353d5ef4a67a39b23f6d8c357c9bb6443dd3b2f40173f1ed240aa92afe909",
    },
}
QUALIFICATION_REVIEWED_ADDED_SYMBOL_SHA256 = {
    "PARENT_FINALIZATION_RESUME_CLAIM_VERSION": "6522377905f2e11decac9d4d2d3f217ac1e83a64beb6e9040db40458c395b04c",
    "PARENT_RECOVERY_AUTHORITY_VERSION": "120ede68d9373437f35abd2c55119fff09d762dd78100f7998edc2a177079296",
    "PARENT_RECOVERY_CLAIM_VERSION": "17f1c8dd3ccb3a43c884eceaec01feb34e8861af25cb52d634219f96d38d7d8d",
    "PARENT_RECOVERY_RESULT_VERSION": "f9e592c2ee01b950378c613106d4dce1085f2c360e8ba8ba5841a03b365ed6ea",
    "_V7_FINALIZATION_RESUME_PREDECESSOR": "a96853955f06a641c2c132e990ec8cad2c84650e13d78793e61b4ff6c404192f",
    "_V7_RECOVERY_ORIGINAL": "e02142ad5e8dabf9cc94da8f3d499f5a663ed7156cb9b2c689a73b4015208dde",
    "_parent_recovery_root": "197b60ec3502f54fcf2f5e9b919b26a2b561e8758fe433b057c1e5e3004dfb31",
    "_publish_parent_evidence": "636c9e7410c8422101002fdafd32dd425ee8d57b5877e8a5b5416f5de0eecc62",
    "_read_exact_text": "ccd954193771bb699f28a1ba56323b6f69cc2421226a805c43c59d0fabc57512",
    "_replay_completed_child": "f837ee623a93cd3af986af6d8fb816417944190302a50d26d79f1dab1d389efd",
    "_verify_self_hashed_evidence": "8fb683d2b6512a1ad76c4d6bb8ae46cca4233163f361bd68d431a2ef92fde966",
    "publish_parent_recovery_result": "2f32f3ebc56498323644535a5a34380f76bd226d386df3571a0d3467765b3d2c",
    "recover_parent_evidence": "1fd2ce30ca973db44605938608d8e04c2aade845accd34ce5ca8ba49ac37794f",
    "resume_parent_finalization": "cae91f03050caf04121f1c4d452ecbe95c2b38f9ea9a18125acc6a9154362b69",
    "validate_parent_recovery_authority": "98ef1b699b9493b59b0bce692e1375fb691c6d636b1abf79a075f6a7ead1ae11",
}
QUALIFICATION_REVIEWED_UNCHANGED_SYMBOL_SHA256 = {
    "PARENT_MANIFEST_VERSION": "06f824ae91046e80792b1bad9aa0a37cd58fd0507cd7ebc0097a34e19b5fa9cf",
    "PARENT_SUCCESS_MANIFEST": "955c200cc37fdb5b37433b472731f9978a38044c2a55bd2acc5418af3079c5aa",
    "_EXPECTED_MLFLOW_ARTIFACTS": "f6e3088b9be0ade71799fcc7221e02690c43bae52f0bc7170a9e09678631dfef",
    "_atomic_json": "8df5ac952a7e2ec7b2904894bf9160f047ceae37792f45795224450220069694",
    "_bounded_run": "4821f57451f8ef07e313df4135d6e242f98526d17debdbbaaa61a1b797836c49",
    "_ensure_pending_gold_row": "d02010a4b9ba78c1d19d3e6ab7f160c235dbe5567d8092b5af76ff56f2741b9a",
    "_file_sha256": "f2344fed1e4c177c64577a358d27f0f97338c50dd2d399695ddf5aa684ba2f9b",
    "_jsonl": "8bf773a5eb1201626cde70c311681ba88e7993c6c77f3175880c00a8fb644bc2",
    "_parent_manifest": "65e17ea5c02e63fc7cf0970bc26d31e76371bdd34ac9e80b8f22ac29b4a09c84",
    "_read": "be65377650b9a467ab77ef0d7a92c75e275b9a6678693427a4df7b417600db4d",
    "_reconcile_mlflow_scalars": "99a7b1219b623bef60be50bb6926e721e313f95b01ea9198f899868bea0f00dc",
    "_stage_local_model": "6f0fb01c029b7e405ccdcfb937e288db6ce02043261baed6e137381d14a1622b",
    "_validate_parent_finalization": "ee6152cf93404066434edfda59ac893eba6c8a581e607b82b33ae10fe9675141",
    "_verify_completed_mlflow_state": "d8793425adf14ee6df88ff4997746f4816d60b5fa95b016849db08b452b34f07",
    "_verify_durable_completion": "8f316a12b99f680361dc48fb62eb6a9469467eec13c9fe7ec9efc53c7e678ac2",
    "_verify_finalizer_owner": "d6976ff1d443ca845726b28fb47ed77820a2f6ec9325f5a85437f9dacedfd1e2",
    "_verify_parent_manifest": "a035edcf76b45b6fce8e301b4bdabd922cce59c9c761f31f2a3344d775177702",
    "finalize_partition_mlflow": "21cbfd6d7c2836cef2cb55b2cad7016afdf1c7d9414b83933e6f2a00a0b50382",
}
PRODUCTION_SOURCE_RELEASE_ROOT = "/Workspace/Shared/research4/source_releases"
BASE_QUALIFICATION_INTENT_VERSION = (
    "research4-databricks-frozen-base-format-intent-v2"
)
BASE_QUALIFICATION_RECEIPT_VERSION = (
    "research4-databricks-frozen-base-format-receipt-v2"
)
BASE_QUALIFICATION_IDENTITY = "frozen_base_format_qualification_v2"
BASE_QUALIFICATION_PLAN_VERSION = (
    "research4-frozen-base-format-qualification-plan-v2"
)
BASE_QUALIFICATION_SET_VERSION = "fresh_disjoint_v2"
BASE_QUALIFICATION_SEED = 202610284
BASE_QUALIFICATION_SELECTION_SHA256 = (
    "a3f8a15250fc7c33124bcb6d5c21581721c151139eb89ee4b72a0e4dae741579"
)
BASE_QUALIFICATION_PROPOSAL_SHA256 = (
    "43d9863e6058a8afc6a23567a64abe23a69b42040f5e0096f192e419fd301f0c"
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
    "config/databricks_production_resource_bindings.json",
    "notebooks/databricks_continuity_admission.py",
    "notebooks/databricks_continuity_base_qualification.py",
    "notebooks/databricks_continuity_base_qualification_preflight.py",
    "notebooks/databricks_continuity_bootstrap.py",
    "notebooks/databricks_continuity_cleanup.py",
    "notebooks/databricks_continuity_data.py",
    "notebooks/databricks_continuity_gold_audit.py",
    "notebooks/databricks_continuity_model_snapshot.py",
    "notebooks/databricks_continuity_partition.py",
    "notebooks/databricks_continuity_partition_finalize.py",
    "notebooks/databricks_continuity_preflight.py",
    "notebooks/databricks_continuity_transfer_audit.py",
    "notebooks/databricks_continuity_transfer_evaluation.py",
    "notebooks/databricks_continuity_transfer_intent_preparation.py",
    "notebooks/databricks_continuity_transfer_packaging.py",
    "notebooks/databricks_continuity_transfer_preflight.py",
    "resources/simulated_continuity_databricks.job.yml",
    "src/__init__.py",
    "src/action_serialization.py",
    "src/base_qualification_carryforward.py",
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
    "src/qualification_technical_successor.py",
    "src/volume_durable_io.py",
    "tools/prepare_databricks_continuity_intent.py",
    "tools/prepare_databricks_base_qualification_intent.py",
    "tools/prepare_databricks_base_qualification_carryforward.py",
    "tools/rescore_base_qualification_v1_serialization_v2.py",
    "tools/prepare_databricks_transfer_package_plan.py",
    "tools/prepare_databricks_transfer_intent.py",
)
QUALIFICATION_SOURCE_ORIGINAL_MANIFEST_PATHS = (
    "databricks.yml",
    "config/databricks_production_resource_bindings.json",
    "notebooks/databricks_continuity_admission.py",
    "notebooks/databricks_continuity_base_qualification.py",
    "notebooks/databricks_continuity_base_qualification_preflight.py",
    "notebooks/databricks_continuity_bootstrap.py",
    "notebooks/databricks_continuity_cleanup.py",
    "notebooks/databricks_continuity_data.py",
    "notebooks/databricks_continuity_gold_audit.py",
    "notebooks/databricks_continuity_model_snapshot.py",
    "notebooks/databricks_continuity_partition.py",
    "notebooks/databricks_continuity_partition_finalize.py",
    "notebooks/databricks_continuity_preflight.py",
    "notebooks/databricks_continuity_transfer_audit.py",
    "notebooks/databricks_continuity_transfer_evaluation.py",
    "notebooks/databricks_continuity_transfer_intent_preparation.py",
    "notebooks/databricks_continuity_transfer_packaging.py",
    "notebooks/databricks_continuity_transfer_preflight.py",
    "resources/simulated_continuity_databricks.job.yml",
    "src/__init__.py",
    "src/action_serialization.py",
    "src/base_qualification_carryforward.py",
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
    "src/qualification_technical_successor.py",
    "src/volume_durable_io.py",
    "tools/prepare_databricks_continuity_intent.py",
    "tools/prepare_databricks_base_qualification_intent.py",
    "tools/prepare_databricks_base_qualification_carryforward.py",
    "tools/rescore_base_qualification_v1_serialization_v2.py",
    "tools/prepare_databricks_transfer_package_plan.py",
    "tools/prepare_databricks_transfer_intent.py",
)
LEGACY_BASE_SOURCE_MANIFEST_PATHS = tuple(
    path for path in SOURCE_MANIFEST_PATHS if path not in {
        "config/databricks_production_resource_bindings.json",
        "notebooks/databricks_continuity_partition_finalize.py",
        "src/base_qualification_carryforward.py",
        "src/qualification_technical_successor.py",
        "src/volume_durable_io.py",
        "tools/prepare_databricks_base_qualification_carryforward.py",
    })
MODEL_RECEIPT_VERSION = "research4-databricks-qwen35-volume-snapshot-v1"
CLOSEOUT_VERSION = "research4-databricks-active-reservation-closeout-v1"
COST_BUDGET_VERSION = "research4-databricks-throughput-derived-budget-v1"
DATASET_TREE_SHA256 = "7786d11946799776df964bc3d1dded641cf560311eb5e099ce4ce9d5f06283ee"
DATASET_RELEASE_ROOT = f"{VOLUME}/dataset_releases/{DATASET_TREE_SHA256}"
DATASET_RELEASE_RECEIPT_SHA256 = (
    "75294a463088db988bd5deac9b2f26d65d2082790bb7aa14b628c8a49fdcec1a"
)
DATASET_RELEASE_VALIDATION_SHA256 = (
    "5958f65ff161e85a6d9622080c4083974dbcc51db911b68863958d87e1880527"
)
EXPERIMENT_SPEC = {
    "version": "qwen35-9b-ordinal-continuity-experiment-v9",
    "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
    "conditions": ["self_continuity", "artifact_continuity", "label_only"],
    "seeds": [202610281, 202610282, 202610283],
    "stages": ["delta1", "delta2", "delta3"],
    "qualification_updates": 2, "primary_optimizer_updates": 144,
    "baseline": "frozen_base", "automatic_retries": 0, "one_use": True,
    "serialization_contract_version": SERIALIZATION_CONTRACT_VERSION,
}


def primary_timeout_budget(plan, qualification_receipt: Mapping[str, Any]) -> dict:
    """Derive the only primary ceiling admitted by measured qualification speed."""
    receipt = dict(qualification_receipt); embedded = receipt.pop("receipt_sha256", None)
    tokens_per_second = receipt.get("qualification_tokens_per_second")
    if (embedded != digest(receipt) or receipt.get("status") != "qualified"
            or type(tokens_per_second) not in (int, float)
            or not math.isfinite(float(tokens_per_second))
            or tokens_per_second <= 0
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


def _validate_historical_qualification_source_manifest(
        value: Mapping[str, Any]) -> dict:
    manifest = dict(value) if isinstance(value, Mapping) else {}
    payload = dict(manifest); embedded = payload.pop("manifest_sha256", None)
    files = manifest.get("files")
    if (set(manifest) != {"version", "source_commit", "file_count",
                          "total_canonical_bytes", "files",
                          "manifest_sha256"}
            or manifest.get("version") != SOURCE_MANIFEST_VERSION
            or manifest.get("source_commit")
               != QUALIFICATION_SOURCE_ORIGINAL_COMMIT
            or embedded != digest(payload)
            or embedded != QUALIFICATION_SOURCE_ORIGINAL_MANIFEST_SHA256
            or not isinstance(files, list)
            or [row.get("path") for row in files]
               != list(QUALIFICATION_SOURCE_ORIGINAL_MANIFEST_PATHS)
            or manifest.get("file_count") != len(files)
            or manifest.get("total_canonical_bytes")
               != sum(row.get("canonical_bytes", -1) for row in files)
            or any(set(row) != {"path", "sha256", "canonical_bytes"}
                   or not re.fullmatch(r"[0-9a-f]{64}", str(
                       row.get("sha256", "")))
                   or type(row.get("canonical_bytes")) is not int
                   or row["canonical_bytes"] < 0 for row in files)):
        raise PermissionError("Historical qualification source manifest differs")
    return manifest


def _validate_historical_qualification_intent(
        value: Mapping[str, Any]) -> dict:
    intent = dict(value) if isinstance(value, Mapping) else {}
    payload = dict(intent); embedded = payload.pop("intent_sha256", None)
    manifest = _validate_historical_qualification_source_manifest(
        intent.get("source_manifest"))
    plan = intent.get("plan")
    from .qwen9b_ordinal_continuity_training import TrainingPlan
    try:
        exact_plan = TrainingPlan(**plan).validate()
    except (TypeError, ValueError, PermissionError) as exc:
        raise PermissionError("Historical qualification plan differs") from exc
    technical = validate_qualification_technical_successor(
        intent.get("qualification_technical_successor"))
    if (embedded != digest(payload)
            or embedded != QUALIFICATION_INTENT_SHA256
            or intent.get("version") != INTENT_VERSION
            or intent.get("identity")
               != "qualification_rank4_a10_volume_durable_v7"
            or intent.get("source_commit")
               != QUALIFICATION_SOURCE_ORIGINAL_COMMIT
            or intent.get("source_sha256") != manifest["manifest_sha256"]
            or intent.get("source_release_path") != source_release_path(
                QUALIFICATION_SOURCE_ORIGINAL_COMMIT)
            or exact_plan.mode != "qualification"
            or exact_plan.full_plan_sha256 != QUALIFICATION_PLAN_SHA256
            or intent.get("plan_sha256") != QUALIFICATION_PLAN_SHA256
            or intent.get("qualification_technical_successor_sha256")
               != technical["technical_successor_sha256"]):
        raise PermissionError("Historical v7 qualification intent differs")
    return intent


def _source_manifest_files(manifest: Mapping[str, Any], *, historical=False
                           ) -> dict[str, dict]:
    if historical:
        value = _validate_historical_qualification_source_manifest(manifest)
    else:
        value = validate_source_manifest(manifest)
    return {row["path"]: row for row in value["files"]}


def validate_qualification_source_carryforward(
        value: Mapping[str, Any], *, qualification_intent: Mapping[str, Any],
        qualification_receipt: Mapping[str, Any],
        current_source_manifest: Mapping[str, Any]) -> dict:
    """Validate a zero-model source-only carry-forward of v7 qualification.

    The prior and current source manifests must contain byte-identical model,
    loss, optimizer, plan, data-selection, and worker code.  Only the bounded
    admission/finalization files named above may differ.
    """
    receipt = dict(value) if isinstance(value, Mapping) else {}
    payload = dict(receipt)
    embedded = payload.pop("carryforward_sha256", None)
    prior = _validate_historical_qualification_intent(qualification_intent)
    current = validate_source_manifest(current_source_manifest)
    capacity = dict(qualification_receipt)
    capacity_sha = capacity.pop("receipt_sha256", None)
    exact = {
        "version", "status", "qualification_intent_sha256",
        "qualification_plan_sha256", "qualification_receipt_sha256",
        "qualification_receipt_file_sha256",
        "qualification_rank_receipt_sha256",
        "qualification_evidence_sha256",
        "qualification_checkpoint_tree_sha256",
        "qualifier_checkpoint_primary_parent_eligible",
        "finalization_resume_claim_sha256", "recovery_result_sha256",
        "parent_finalization_sha256", "parent_manifest_sha256",
        "prior_source_commit", "prior_git_tree",
        "prior_source_manifest_sha256", "current_source_commit",
        "current_git_tree", "current_research4_diff_sha256",
        "current_source_manifest_sha256", "changed_paths",
        "reviewed_successor_commit", "reviewed_successor_git_tree",
        "reviewed_successor_source_manifest_sha256",
        "reviewed_source_diff_sha256", "reviewed_changed_paths",
        "reviewed_changed_symbol_sha256", "reviewed_added_symbol_sha256",
        "reviewed_unchanged_symbol_sha256",
        "run_partition_execution_prefix_sha256",
        "allowed_changed_paths", "unchanged_science_file_sha256",
        "authorized_primary_plan_sha256s_digest", "model_tree_sha256",
        "model_receipt_sha256", "base_qualification_intent_sha256",
        "base_qualification_receipt_sha256",
        "base_qualification_carryforward_sha256",
        "qualification_technical_successor_sha256",
        "model_execution_performed", "optimizer_updates",
        "carryforward_sha256",
    }
    if (set(receipt) != exact
            or embedded != digest(payload)
            or receipt.get("version")
               != QUALIFICATION_SOURCE_CARRYFORWARD_VERSION
            or receipt.get("status") != "qualified_source_carried_forward"
            or prior.get("plan", {}).get("mode") != "qualification"
            or capacity_sha != digest(capacity)
            or qualification_receipt.get("version")
               != DATABRICKS_QUALIFICATION_VERSION
            or qualification_receipt.get("status") != "qualified"
            or receipt.get("qualification_intent_sha256")
               != prior.get("intent_sha256")
            or receipt.get("qualification_receipt_sha256") != capacity_sha
            or capacity_sha != QUALIFICATION_RECEIPT_SHA256
            or receipt.get("qualification_receipt_file_sha256")
               != QUALIFICATION_RECEIPT_FILE_SHA256
            or qualification_receipt.get("qualification_intent_sha256")
               != prior.get("intent_sha256")
            or receipt.get("qualification_intent_sha256")
               != QUALIFICATION_INTENT_SHA256
            or receipt.get("qualification_plan_sha256")
               != QUALIFICATION_PLAN_SHA256
            or receipt.get("qualification_rank_receipt_sha256")
               != QUALIFICATION_RANK_RECEIPT_SHA256
            or receipt.get("qualification_evidence_sha256")
               != QUALIFICATION_EVIDENCE_SHA256
            or receipt.get("qualification_checkpoint_tree_sha256")
               != QUALIFICATION_CHECKPOINT_TREE_SHA256
            or receipt.get("qualifier_checkpoint_primary_parent_eligible")
               is not False
            or receipt.get("finalization_resume_claim_sha256")
               != QUALIFICATION_FINALIZATION_RESUME_CLAIM_SHA256
            or receipt.get("recovery_result_sha256")
               != QUALIFICATION_RECOVERY_RESULT_SHA256
            or receipt.get("parent_finalization_sha256")
               != QUALIFICATION_PARENT_FINALIZATION_SHA256
            or receipt.get("parent_manifest_sha256")
               != QUALIFICATION_PARENT_MANIFEST_SHA256
            or receipt.get("prior_source_commit") != prior.get("source_commit")
            or receipt.get("prior_source_commit")
               != QUALIFICATION_SOURCE_ORIGINAL_COMMIT
            or receipt.get("prior_git_tree")
               != QUALIFICATION_SOURCE_ORIGINAL_GIT_TREE
            or receipt.get("prior_source_manifest_sha256")
               != prior.get("source_sha256")
            or receipt.get("prior_source_manifest_sha256")
               != QUALIFICATION_SOURCE_ORIGINAL_MANIFEST_SHA256
            or receipt.get("current_source_commit")
               != current.get("source_commit")
            or receipt.get("current_source_manifest_sha256")
               != current.get("manifest_sha256")
            or not re.fullmatch(r"[0-9a-f]{40}", str(
                receipt.get("current_git_tree", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(
                receipt.get("current_research4_diff_sha256", "")))
            or receipt.get("reviewed_successor_commit")
               != QUALIFICATION_SOURCE_REVIEWED_SUCCESSOR_COMMIT
            or receipt.get("reviewed_successor_git_tree")
               != QUALIFICATION_SOURCE_REVIEWED_SUCCESSOR_GIT_TREE
            or receipt.get("reviewed_successor_source_manifest_sha256")
               != QUALIFICATION_SOURCE_REVIEWED_SUCCESSOR_MANIFEST_SHA256
            or receipt.get("reviewed_source_diff_sha256")
               != QUALIFICATION_SOURCE_REVIEWED_DIFF_SHA256
            or receipt.get("reviewed_changed_paths")
               != list(QUALIFICATION_SOURCE_REVIEWED_CHANGED_PATHS)
            or receipt.get("reviewed_changed_symbol_sha256")
               != QUALIFICATION_REVIEWED_CHANGED_SYMBOL_SHA256
            or receipt.get("reviewed_added_symbol_sha256")
               != QUALIFICATION_REVIEWED_ADDED_SYMBOL_SHA256
            or receipt.get("reviewed_unchanged_symbol_sha256")
               != QUALIFICATION_REVIEWED_UNCHANGED_SYMBOL_SHA256
            or receipt.get("run_partition_execution_prefix_sha256")
               != QUALIFICATION_RUN_PARTITION_EXECUTION_PREFIX_SHA256
            or qualification_receipt.get("source_commit")
               != prior.get("source_commit")
            or qualification_receipt.get("source_sha256")
               != prior.get("source_sha256")
            or receipt.get("model_tree_sha256")
               != qualification_receipt.get("model_tree_sha256")
            or receipt.get("model_receipt_sha256")
               != qualification_receipt.get("model_receipt_sha256")
            or receipt.get("base_qualification_intent_sha256")
               != qualification_receipt.get(
                   "base_qualification_intent_sha256")
            or receipt.get("base_qualification_receipt_sha256")
               != qualification_receipt.get(
                   "base_qualification_receipt_sha256")
            or receipt.get("base_qualification_carryforward_sha256")
               != qualification_receipt.get(
                   "base_qualification_carryforward_sha256")
            or receipt.get("qualification_technical_successor_sha256")
               != qualification_receipt.get(
                   "qualification_technical_successor_sha256")
            or qualification_receipt.get("qualification_plan_sha256")
               != QUALIFICATION_PLAN_SHA256
            or qualification_receipt.get(
                "qualification_rank_receipt_sha256")
               != QUALIFICATION_RANK_RECEIPT_SHA256
            or qualification_receipt.get("qualification_evidence_sha256")
               != QUALIFICATION_EVIDENCE_SHA256
            or qualification_receipt.get(
                "qualification_checkpoint_tree_sha256")
               != QUALIFICATION_CHECKPOINT_TREE_SHA256
            or qualification_receipt.get(
                "qualifier_checkpoint_primary_parent_eligible") is not False
            or receipt.get("authorized_primary_plan_sha256s_digest")
               != digest(qualification_receipt.get(
                   "authorized_primary_plan_sha256s"))
            or receipt.get("model_execution_performed") is not False
            or receipt.get("optimizer_updates") != 0
            or receipt.get("allowed_changed_paths")
               != list(QUALIFICATION_SOURCE_CARRYFORWARD_ALLOWED_CHANGED_PATHS)):
        raise PermissionError("Qualification source carry-forward differs")
    prior_files = _source_manifest_files(
        prior["source_manifest"], historical=True)
    current_files = _source_manifest_files(current)
    changed = sorted(
        path for path in set(prior_files) | set(current_files)
        if prior_files.get(path, {}).get("sha256")
        != current_files.get(path, {}).get("sha256"))
    if (receipt.get("changed_paths") != changed
            or any(path not in QUALIFICATION_SOURCE_CARRYFORWARD_ALLOWED_CHANGED_PATHS
                   for path in changed)):
        raise PermissionError("Qualification carry-forward changed science source")
    immutable = receipt.get("unchanged_science_file_sha256")
    if (not isinstance(immutable, Mapping)
            or set(immutable) != set(QUALIFICATION_SOURCE_IMMUTABLE_SCIENCE_PATHS)):
        raise PermissionError("Qualification carry-forward science proof differs")
    for path in QUALIFICATION_SOURCE_IMMUTABLE_SCIENCE_PATHS:
        prior_sha = prior_files.get(path, {}).get("sha256")
        current_sha = current_files.get(path, {}).get("sha256")
        if (not re.fullmatch(r"[0-9a-f]{64}", str(prior_sha or ""))
                or immutable.get(path) != prior_sha
                or current_sha != prior_sha):
            raise PermissionError("Qualification scientific source changed")
    return receipt


def create_qualification_source_carryforward(
        *, qualification_intent: Mapping[str, Any],
        qualification_receipt: Mapping[str, Any],
        current_source_manifest: Mapping[str, Any], current_git_tree: str,
        current_research4_diff_sha256: str) -> dict:
    """Create the reviewed metadata receipt; this performs no model work."""
    prior = _validate_historical_qualification_intent(qualification_intent)
    current = validate_source_manifest(current_source_manifest)
    capacity = dict(qualification_receipt)
    capacity_sha = capacity.pop("receipt_sha256", None)
    if (capacity_sha != digest(capacity)
            or qualification_receipt.get("status") != "qualified"):
        raise PermissionError("Qualification receipt is not valid")
    prior_files = _source_manifest_files(
        prior["source_manifest"], historical=True)
    current_files = _source_manifest_files(current)
    changed = sorted(
        path for path in set(prior_files) | set(current_files)
        if prior_files.get(path, {}).get("sha256")
        != current_files.get(path, {}).get("sha256"))
    value = {
        "version": QUALIFICATION_SOURCE_CARRYFORWARD_VERSION,
        "status": "qualified_source_carried_forward",
        "qualification_intent_sha256": prior["intent_sha256"],
        "qualification_plan_sha256": qualification_receipt[
            "qualification_plan_sha256"],
        "qualification_receipt_sha256": capacity_sha,
        "qualification_receipt_file_sha256": QUALIFICATION_RECEIPT_FILE_SHA256,
        "qualification_rank_receipt_sha256": qualification_receipt[
            "qualification_rank_receipt_sha256"],
        "qualification_evidence_sha256": qualification_receipt[
            "qualification_evidence_sha256"],
        "qualification_checkpoint_tree_sha256": qualification_receipt[
            "qualification_checkpoint_tree_sha256"],
        "qualifier_checkpoint_primary_parent_eligible": qualification_receipt[
            "qualifier_checkpoint_primary_parent_eligible"],
        "finalization_resume_claim_sha256": (
            QUALIFICATION_FINALIZATION_RESUME_CLAIM_SHA256),
        "recovery_result_sha256": QUALIFICATION_RECOVERY_RESULT_SHA256,
        "parent_finalization_sha256": QUALIFICATION_PARENT_FINALIZATION_SHA256,
        "parent_manifest_sha256": QUALIFICATION_PARENT_MANIFEST_SHA256,
        "prior_source_commit": prior["source_commit"],
        "prior_git_tree": QUALIFICATION_SOURCE_ORIGINAL_GIT_TREE,
        "prior_source_manifest_sha256": prior["source_sha256"],
        "current_source_commit": current["source_commit"],
        "current_git_tree": current_git_tree,
        "current_research4_diff_sha256": current_research4_diff_sha256,
        "current_source_manifest_sha256": current["manifest_sha256"],
        "changed_paths": changed,
        "reviewed_successor_commit": (
            QUALIFICATION_SOURCE_REVIEWED_SUCCESSOR_COMMIT),
        "reviewed_successor_git_tree": (
            QUALIFICATION_SOURCE_REVIEWED_SUCCESSOR_GIT_TREE),
        "reviewed_successor_source_manifest_sha256": (
            QUALIFICATION_SOURCE_REVIEWED_SUCCESSOR_MANIFEST_SHA256),
        "reviewed_source_diff_sha256": QUALIFICATION_SOURCE_REVIEWED_DIFF_SHA256,
        "reviewed_changed_paths": list(
            QUALIFICATION_SOURCE_REVIEWED_CHANGED_PATHS),
        "reviewed_changed_symbol_sha256": json.loads(json.dumps(
            QUALIFICATION_REVIEWED_CHANGED_SYMBOL_SHA256)),
        "reviewed_added_symbol_sha256": json.loads(json.dumps(
            QUALIFICATION_REVIEWED_ADDED_SYMBOL_SHA256)),
        "reviewed_unchanged_symbol_sha256": json.loads(json.dumps(
            QUALIFICATION_REVIEWED_UNCHANGED_SYMBOL_SHA256)),
        "run_partition_execution_prefix_sha256": (
            QUALIFICATION_RUN_PARTITION_EXECUTION_PREFIX_SHA256),
        "allowed_changed_paths": list(
            QUALIFICATION_SOURCE_CARRYFORWARD_ALLOWED_CHANGED_PATHS),
        "unchanged_science_file_sha256": {
            path: prior_files.get(path, {}).get("sha256")
            for path in QUALIFICATION_SOURCE_IMMUTABLE_SCIENCE_PATHS
        },
        "authorized_primary_plan_sha256s_digest": digest(
            qualification_receipt["authorized_primary_plan_sha256s"]),
        "model_tree_sha256": qualification_receipt["model_tree_sha256"],
        "model_receipt_sha256": qualification_receipt["model_receipt_sha256"],
        "base_qualification_intent_sha256": qualification_receipt[
            "base_qualification_intent_sha256"],
        "base_qualification_receipt_sha256": qualification_receipt[
            "base_qualification_receipt_sha256"],
        "base_qualification_carryforward_sha256": qualification_receipt.get(
            "base_qualification_carryforward_sha256"),
        "qualification_technical_successor_sha256": qualification_receipt[
            "qualification_technical_successor_sha256"],
        "model_execution_performed": False,
        "optimizer_updates": 0,
    }
    value["carryforward_sha256"] = digest(value)
    return validate_qualification_source_carryforward(
        value, qualification_intent=prior,
        qualification_receipt=qualification_receipt,
        current_source_manifest=current)


def _top_level_ast_sha256(source_text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in ast.parse(source_text).body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets
                     if isinstance(target, ast.Name)]
        for name in names:
            if name in result:
                raise PermissionError("Duplicate partition controller symbol")
            result[name] = hashlib.sha256(ast.dump(
                node, annotate_fields=True,
                include_attributes=False).encode("utf-8")).hexdigest()
    return result


def _run_partition_execution_prefix_sha256(source_text: str) -> str:
    tree = ast.parse(source_text)
    controller = next((node for node in tree.body
                       if isinstance(node, ast.FunctionDef)
                       and node.name == "run_partition"), None)
    if controller is None:
        raise PermissionError("run_partition is absent")
    boundary = next((index for index, node in enumerate(controller.body)
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "post_run_deadline"), None)
    if boundary is None:
        raise PermissionError("run_partition post-model boundary is absent")
    payload = "".join(ast.dump(
        node, annotate_fields=True, include_attributes=False)
        for node in controller.body[:boundary]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_deployed_qualification_source_carryforward(
        intent: Mapping[str, Any], code_root: str | Path) -> dict | None:
    """Recheck current controller AST before any primary model execution."""
    current = validate_intent(intent)
    carryforward = current.get("qualification_source_carryforward")
    if carryforward is None:
        return None
    verified = validate_qualification_source_carryforward(
        carryforward,
        qualification_intent=current["qualification_source_intent"],
        qualification_receipt=current["qualification_source_receipt"],
        current_source_manifest=current["source_manifest"])
    path = Path(code_root).resolve() / "src/databricks_continuity_partition.py"
    if not path.is_file() or path.is_symlink():
        raise PermissionError("Partition controller source is absent")
    source = path.read_text(encoding="utf-8")
    if (_run_partition_execution_prefix_sha256(source)
            != QUALIFICATION_RUN_PARTITION_EXECUTION_PREFIX_SHA256):
        raise PermissionError("Partition model-execution prefix changed")
    symbols = _top_level_ast_sha256(source)
    expected_names = (set(QUALIFICATION_REVIEWED_UNCHANGED_SYMBOL_SHA256)
                      | set(QUALIFICATION_REVIEWED_CHANGED_SYMBOL_SHA256)
                      | set(QUALIFICATION_REVIEWED_ADDED_SYMBOL_SHA256))
    if set(symbols) != expected_names:
        raise PermissionError("Partition controller symbol set changed")
    for name, expected in QUALIFICATION_REVIEWED_UNCHANGED_SYMBOL_SHA256.items():
        if symbols.get(name) != expected:
            raise PermissionError("Reviewed unchanged controller symbol changed")
    for name, expected in QUALIFICATION_REVIEWED_ADDED_SYMBOL_SHA256.items():
        if symbols.get(name) != expected:
            raise PermissionError("Reviewed added controller symbol changed")
    if (symbols.get("_expected_mlflow_scalars")
            != QUALIFICATION_REVIEWED_CHANGED_SYMBOL_SHA256[
                "_expected_mlflow_scalars"]["reviewed"]):
        raise PermissionError("Reviewed MLflow scalar mapping changed")
    return verified


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


def validate_source_manifest(value: Mapping[str, Any], *,
                             allow_legacy_base: bool = False) -> dict:
    manifest = dict(value) if isinstance(value, Mapping) else {}
    if set(manifest) != {"version", "source_commit", "file_count",
                         "total_canonical_bytes", "files", "manifest_sha256"}:
        raise PermissionError("Source manifest fields differ")
    files = manifest.get("files")
    logical_paths = [item.get("path") for item in files] if isinstance(files, list) else []
    accepted_paths = {tuple(SOURCE_MANIFEST_PATHS)}
    if allow_legacy_base:
        accepted_paths.add(tuple(LEGACY_BASE_SOURCE_MANIFEST_PATHS))
    if (manifest.get("version") != SOURCE_MANIFEST_VERSION
            or not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("source_commit", "")))
            or not isinstance(files, list)
            or tuple(logical_paths) not in accepted_paths
            or manifest.get("file_count") != len(logical_paths)):
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


def source_release_path(commit: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", str(commit)):
        raise ValueError("Source release commit must be an exact Git SHA")
    return f"{PRODUCTION_SOURCE_RELEASE_ROOT}/{commit}"


def _verify_deployed_source_contents(
        code_root: str | Path, intent: Mapping[str, Any]) -> dict:
    """Recompute source bytes after a separate production-path check."""
    manifest = validate_source_manifest(intent.get("source_manifest"))
    commit = intent.get("source_commit")
    root = Path(code_root).resolve()
    actual = build_source_manifest(root, commit)
    if (actual != manifest or intent.get("source_sha256") != manifest["manifest_sha256"]
            or manifest.get("source_commit") != commit):
        raise PermissionError("Deployed source differs from the reviewed manifest")
    return actual


def verify_deployed_source(code_root: str | Path, intent: Mapping[str, Any]) -> dict:
    """Require and replay the exact immutable Shared production release."""
    commit = intent.get("source_commit")
    expected_path = f"/Workspace/Shared/research4/source_releases/{commit}"
    if (intent.get("source_release_path") != expected_path
            or Path(code_root).resolve().as_posix() != expected_path):
        raise PermissionError("Databricks code path is not the exact source release")
    return _verify_deployed_source_contents(code_root, intent)


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
        "serialization_contract": serialization_contract(),
        "serialization_contract_version": SERIALIZATION_CONTRACT_VERSION,
        "qualification_set_version": BASE_QUALIFICATION_SET_VERSION,
        "qualification_seed": BASE_QUALIFICATION_SEED,
        "qualification_selection_digest": BASE_QUALIFICATION_SELECTION_SHA256,
        "qualification_proposal_sha256": BASE_QUALIFICATION_PROPOSAL_SHA256,
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
        "qualification_set_version": BASE_QUALIFICATION_SET_VERSION,
        "qualification_seed": BASE_QUALIFICATION_SEED,
        "qualification_selection_digest": BASE_QUALIFICATION_SELECTION_SHA256,
        "qualification_proposal_sha256": BASE_QUALIFICATION_PROPOSAL_SHA256,
        "serialization_contract_version": SERIALIZATION_CONTRACT_VERSION,
    })
    expected_data_binding = digest({
        "record_count": 288,
        "family_counts": {name: 36 for name in BASE_QUALIFICATION_FAMILIES},
        "training_eligible_count": 0,
        "qualification_set_version": BASE_QUALIFICATION_SET_VERSION,
        "qualification_seed": BASE_QUALIFICATION_SEED,
        "qualification_selection_digest": BASE_QUALIFICATION_SELECTION_SHA256,
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
        "qualification_set_version", "qualification_seed",
        "qualification_selection_digest",
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
                or payload.get("qualification_set_version")
                   != BASE_QUALIFICATION_SET_VERSION
                or payload.get("qualification_seed") != BASE_QUALIFICATION_SEED
                or payload.get("qualification_selection_digest")
                   != BASE_QUALIFICATION_SELECTION_SHA256
                or target != {"action": payload.get("target_action_id")}
                or canonical(target) != payload.get("target_text")
                or canonical(visible) != payload.get("input_text")
                or visible.get("decision_index") != decision_index
                or visible.get("scenario", {}).get("instruction")
                   != NEUTRAL_SIMULATOR_INSTRUCTION
                or visible.get("scenario", {}).get("qualification_family") != family
                or visible.get("scenario", {}).get("qualification_set_version")
                   != BASE_QUALIFICATION_SET_VERSION
                or visible.get("scenario", {}).get("qualification_seed")
                   != BASE_QUALIFICATION_SEED
                or visible.get("scenario", {}).get("qualification_selection_digest")
                   != BASE_QUALIFICATION_SELECTION_SHA256
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
                or not str(episode.get("episode_id", "")).startswith(
                    "base-qualification-v2-")
                or episode.get("frozen_base_only") is not True
                or episode.get("training_eligible") is not False
                or episode.get("contains_danger_label") is not False
                or episode.get("contains_continuity_target") is not False
                or episode.get("must_stop") is not True
                or episode.get("final_stop_outside_model_action_space") is not True
                or episode.get("qualification_set_version")
                   != BASE_QUALIFICATION_SET_VERSION
                or episode.get("qualification_seed") != BASE_QUALIFICATION_SEED
                or episode.get("qualification_selection_digest")
                   != BASE_QUALIFICATION_SELECTION_SHA256
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
                visible = json.loads(decision.get("input_text", ""))
            except json.JSONDecodeError as exc:
                raise PermissionError("Base qualification input JSON differs") from exc
            target_parse = parse_registered_action(
                decision.get("target_text", ""), allowed or ())
            if (not isinstance(record_id, str) or not record_id
                    or decision.get("decision_index") != expected_index
                    or not isinstance(allowed, list) or not allowed
                    or target not in allowed
                    or target_parse["valid"] is not True
                    or target_parse["normalization_used"] is not False
                    or target_parse["action"] != target
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
                "qualification_set_version": BASE_QUALIFICATION_SET_VERSION,
                "qualification_seed": BASE_QUALIFICATION_SEED,
                "qualification_selection_digest": BASE_QUALIFICATION_SELECTION_SHA256,
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


def validate_policy_silver_payload(record_json: str, *, record_id: str,
                                   expected_condition: str,
                                   expected_family_id: str,
                                   expected_danger_level: int,
                                   expected_split: str = "train") -> dict:
    """Validate one exact canonical policy payload selected from Silver."""
    if (not isinstance(record_json, str) or not isinstance(record_id, str)
            or not isinstance(expected_condition, str)
            or not isinstance(expected_split, str)
            or not isinstance(expected_family_id, str)
            or not expected_family_id
            or type(expected_danger_level) is not int
            or expected_danger_level not in (0, 1, 2, 3)):
        raise PermissionError("Silver policy payload authority is malformed")
    try:
        value = json.loads(record_json)
    except json.JSONDecodeError as exc:
        raise PermissionError("Silver policy payload is not JSON") from exc
    if (canonical(value) != record_json
            or value.get("record_id") != record_id
            or value.get("record_kind") != "policy_decision"
            or value.get("serialization_contract_version")
               != SERIALIZATION_CONTRACT_VERSION
            or value.get("condition") != expected_condition
            or value.get("split") != expected_split
            or value.get("family_id") != expected_family_id
            or value.get("danger_level") != expected_danger_level):
        raise PermissionError("Silver policy payload semantics differ")
    return value


def silver_partition_bindings(rows: list[Mapping[str, Any]],
                              generated_root: str | Path) -> dict[str, dict]:
    """Bind tokenized training rows to separately validated raw scenarios.

    Tokenization deliberately projects away harness-only and schema fields.
    Raw causal matching must therefore be replayed from the immutable dataset
    release, while occurrence journals are computed from the exact tokenized
    rows that will be read from Silver during training.
    """
    from .qwen9b_ordinal_continuity_controller import (
        matrix_plans, partition_identity, qualification_plan)
    from .qwen9b_ordinal_continuity_training import (
        adjacent_pair_schedule, canonical_sha256,
        databricks_a10_plan,
        load_simulator_condition,
        validate_equal_occurrence_exposure,
        validate_matched_conditions)
    decoded = []
    for row in rows:
        payload = row["record_json"]
        is_base = row.get("record_kind") == "base_qualification"
        expected_split = "qualification" if is_base else "train"
        if (row.get("split") != expected_split
                or bool(row.get("admitted_for_training")) == is_base
                or row.get("record_sha256")
                   != hashlib.sha256(payload.encode("utf-8")).hexdigest()):
            raise PermissionError("Silver binding input is not an exact admitted payload")
        if is_base:
            value = json.loads(payload)
            if value.get("split") != expected_split:
                raise PermissionError("Base Silver binding payload differs")
        else:
            if (row.get("record_kind") not in EXPERIMENT_SPEC["conditions"]
                    or row.get("condition") != row.get("record_kind")):
                raise PermissionError("Silver policy metadata condition differs")
            value = validate_policy_silver_payload(
                payload, record_id=row.get("record_id"),
                expected_condition=row.get("condition"),
                expected_split=expected_split,
                expected_family_id=row.get("family_id"),
                expected_danger_level=row.get("danger_level"))
        decoded.append((row["record_kind"], value, row))
    by_kind = {}
    for kind, value, metadata in decoded:
        by_kind.setdefault(kind, []).append(value)
    conditions = {name: by_kind[name] for name in EXPERIMENT_SPEC["conditions"]}
    raw_conditions = {name: load_simulator_condition(generated_root, name)
                      for name in EXPERIMENT_SPEC["conditions"]}
    data_binding = validate_matched_conditions(raw_conditions)
    base_rows = by_kind["base_qualification"]
    qualification = databricks_a10_plan(
        qualification_plan(EXPERIMENT_SPEC, 4))
    policies = matrix_plans(EXPERIMENT_SPEC, 4)
    plans = [qualification]
    plans += [databricks_a10_plan(plan) for plan in policies]
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
        "qualification_set_version": BASE_QUALIFICATION_SET_VERSION,
        "qualification_seed": BASE_QUALIFICATION_SEED,
        "qualification_selection_digest": BASE_QUALIFICATION_SELECTION_SHA256,
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
            "optimizer_updates": 0, "records": len(base_meta),
            "qualification_set_version": BASE_QUALIFICATION_SET_VERSION,
            "qualification_seed": BASE_QUALIFICATION_SEED,
            "qualification_selection_digest": BASE_QUALIFICATION_SELECTION_SHA256,
            "qualification_proposal_sha256": BASE_QUALIFICATION_PROPOSAL_SHA256,
            "serialization_contract_version": SERIALIZATION_CONTRACT_VERSION}),
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
        source_manifest = validate_source_manifest(
            intent.get("source_manifest"), allow_legacy_base=True)
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
    family_normalized = receipt.get("family_normalization_used")
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
            or receipt.get("serialization_contract") != serialization_contract()
            or receipt.get("serialization_contract_version")
               != SERIALIZATION_CONTRACT_VERSION
            or receipt.get("qualification_set_version")
               != BASE_QUALIFICATION_SET_VERSION
            or receipt.get("qualification_seed") != BASE_QUALIFICATION_SEED
            or receipt.get("qualification_selection_digest")
               != BASE_QUALIFICATION_SELECTION_SHA256
            or receipt.get("qualification_proposal_sha256")
               != BASE_QUALIFICATION_PROPOSAL_SHA256
            or not isinstance(family_normalized, Mapping)
            or set(family_normalized) != set(BASE_QUALIFICATION_FAMILIES)
            or any(type(item) is not int or not 0 <= item <= 36
                   for item in family_normalized.values())
            or sum(family_normalized.values())
               != receipt.get("normalization_used_count")
            or not isinstance(receipt.get("normalization_used_count"), int)
            or not 0 <= receipt.get("normalization_used_count") <= 288
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
    family_normalized: dict[str, int] = {}
    strict_total = target_total = normalization_total = 0
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
        parsed = parse_registered_action(output_text, allowed)
        derived_strict = parsed["valid"]
        derived_action = parsed["action"]
        derived_legal = parsed["valid"]
        derived_target = parsed["valid"] and derived_action == target
        strict = derived_legal
        if (not isinstance(name, str) or not name
                or not isinstance(output_text, str)
                or not isinstance(allowed, list) or not allowed
                or not isinstance(target, str)
                or hashlib.sha256(output_text.encode("utf-8")).hexdigest()
                   != row.get("output_sha256")
                or row.get("strict_json") is not derived_strict
                or row.get("serialization_valid") is not derived_strict
                or row.get("serialization_contract_version")
                   != SERIALIZATION_CONTRACT_VERSION
                or row.get("normalization_used") is not parsed["normalization_used"]
                or row.get("serialization_form") != parsed["serialization_form"]
                or row.get("raw_response_sha256") != parsed["raw_response_sha256"]
                or row.get("legal_action") is not derived_legal
                or row.get("target_match") is not derived_target
                or row.get("action") != derived_action):
            raise PermissionError("Base qualification outcome differs")
        family.setdefault(name, 0)
        family_normalized.setdefault(name, 0)
        family[name] += int(strict)
        family_normalized[name] += int(parsed["normalization_used"])
        strict_total += int(strict)
        normalization_total += int(parsed["normalization_used"])
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
        "serialization_contract": serialization_contract(),
        "serialization_contract_version": SERIALIZATION_CONTRACT_VERSION,
        "normalization_used_count": normalization_total,
        "family_normalization_used": dict(sorted(family_normalized.items())),
        "qualification_set_version": BASE_QUALIFICATION_SET_VERSION,
        "qualification_seed": BASE_QUALIFICATION_SEED,
        "qualification_selection_digest": BASE_QUALIFICATION_SELECTION_SHA256,
        "qualification_proposal_sha256": BASE_QUALIFICATION_PROPOSAL_SHA256,
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
    if value.get("version") == "research4-databricks-continuity-transfer-intent-v2":
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
    technical = validate_qualification_technical_successor(
        intent.get("qualification_technical_successor"))
    if intent.get("qualification_technical_successor_sha256") != technical[
            "technical_successor_sha256"]:
        raise PermissionError("Qualification technical successor binding differs")
    required = {
        "version": INTENT_VERSION, "one_use": True, "attempt_number": 1,
        "automatic_retries": 0, "execution_platform": "databricks",
        "world_size": 1, "gpu_count": 1, "max_concurrent_runs": 1,
        "model_root": MODEL_ROOT, "silver_table": SILVER_TABLE,
    }
    from dataclasses import asdict
    from .qwen9b_ordinal_continuity_controller import (
        matrix_plans, partition_identity, qualification_plan)
    from .qwen9b_ordinal_continuity_training import (
        TrainingPlan, databricks_a10_plan)
    try:
        exact_plan = TrainingPlan(**plan).validate()
    except (TypeError, ValueError, PermissionError) as exc:
        raise PermissionError("Databricks intent contains an invalid training plan") from exc
    qualification = databricks_a10_plan(
        qualification_plan(EXPERIMENT_SPEC, 4))
    policies = matrix_plans(EXPERIMENT_SPEC, 4)
    canonical_plans = [databricks_a10_plan(item) for item in policies]
    expected_plan = (qualification if mode == "qualification" else
                     next((item for item in canonical_plans
                           if (item.condition, item.seed, item.stage) ==
                           (exact_plan.condition, exact_plan.seed, exact_plan.stage)), None))
    expected_plan_payload = (None if expected_plan is None
                             else asdict(expected_plan))
    canonical_primary_hashes = sorted(
        item.full_plan_sha256 for item in canonical_plans)
    if (expected_plan is None
            or plan != expected_plan_payload
            or canonical(plan) != canonical(expected_plan_payload)
            or exact_plan.full_plan_sha256 != expected_plan.full_plan_sha256):
        raise PermissionError("Databricks plan differs from the canonical experiment matrix")
    if mode == "qualification":
        if (intent.get("plan_sha256") != technical.get("successor_plan_sha256")
                or intent.get("plan_sha256") != expected_plan.full_plan_sha256):
            raise PermissionError(
                "Qualification plan hash differs from technical authority")
    elif (intent.get("plan_sha256") not in canonical_primary_hashes
          or digest(canonical_primary_hashes)
             != technical.get("successor_primary_plan_hashes_digest")):
        raise PermissionError(
            "Primary plan hashes differ from technical authority")
    source_base_intent = intent.get("base_qualification_source_intent")
    carryforward = intent.get("base_qualification_carryforward")
    carryforward_sha = intent.get("base_qualification_carryforward_sha256")
    if carryforward is None:
        if source_base_intent is not None or carryforward_sha is not None:
            raise PermissionError("Incomplete base qualification carry-forward")
    else:
        try:
            source_base_intent = validate_base_qualification_intent(
                source_base_intent)
            carryforward = validate_carryforward(
                carryforward, prior_intent=source_base_intent,
                base_receipt={
                    "receipt_sha256": intent.get(
                        "base_qualification_receipt_sha256"),
                    "base_qualification_intent_sha256": intent.get(
                        "base_qualification_intent_sha256"),
                    "model_tree_sha256": intent.get("model_tree_sha256"),
                    "model_receipt_sha256": intent.get("model_receipt_sha256"),
                    "dataset_tree_sha256": intent.get("dataset_tree_sha256"),
                    "plan_sha256": source_base_intent.get("plan_sha256"),
                    "serialization_contract_version": source_base_intent.get(
                        "plan", {}).get("serialization_contract_version"),
                    "qualification_set_version": source_base_intent.get(
                        "plan", {}).get("qualification_set_version"),
                    "qualification_seed": source_base_intent.get(
                        "plan", {}).get("qualification_seed"),
                    "qualification_selection_digest": source_base_intent.get(
                        "plan", {}).get("qualification_selection_digest"),
                    "qualification_proposal_sha256": source_base_intent.get(
                        "plan", {}).get("qualification_proposal_sha256"),
                }, current_source_manifest=source_manifest)
        except (TypeError, ValueError, PermissionError) as exc:
            raise PermissionError("Invalid base qualification carry-forward") from exc
        if (carryforward_sha != carryforward.get("carryforward_sha256")
                or source_base_intent.get("intent_sha256")
                   != intent.get("base_qualification_intent_sha256")):
            raise PermissionError("Base qualification carry-forward binding differs")
    qualification_source_intent = intent.get("qualification_source_intent")
    qualification_source_receipt = intent.get("qualification_source_receipt")
    qualification_source_carryforward = intent.get(
        "qualification_source_carryforward")
    qualification_source_carryforward_sha = intent.get(
        "qualification_source_carryforward_sha256")
    if qualification_source_carryforward is None:
        if any(value is not None for value in (
                qualification_source_intent, qualification_source_receipt,
                qualification_source_carryforward_sha)):
            raise PermissionError(
                "Incomplete qualification source carry-forward")
    else:
        if mode != "primary":
            raise PermissionError(
                "Only primary training may use qualification carry-forward")
        try:
            verified_source_carryforward = (
                validate_qualification_source_carryforward(
                    qualification_source_carryforward,
                    qualification_intent=qualification_source_intent,
                    qualification_receipt=qualification_source_receipt,
                    current_source_manifest=source_manifest))
        except (TypeError, ValueError, PermissionError) as exc:
            raise PermissionError(
                "Invalid qualification source carry-forward") from exc
        if (qualification_source_carryforward_sha
                != verified_source_carryforward["carryforward_sha256"]):
            raise PermissionError(
                "Qualification source carry-forward binding differs")
    if (any(intent.get(key) != expected for key, expected in required.items())
            or any(type(intent.get(key)) is not int for key in (
                "attempt_number", "automatic_retries", "world_size",
                "gpu_count", "max_concurrent_runs"))
            or intent.get("identity") != partition_identity(exact_plan)
            or not re.fullmatch(r"[a-z0-9_]+", intent["identity"])
            or not isinstance(intent.get("source_commit"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", intent["source_commit"])
            or source_manifest.get("source_commit") != intent["source_commit"]
            or intent.get("source_release_path") != (
                "/Workspace/Shared/research4/source_releases/"
                + intent["source_commit"])
            or intent.get("source_sha256") != source_manifest.get("manifest_sha256")
            or intent.get("dataset_tree_sha256") != DATASET_TREE_SHA256
            or intent.get("silver_record_count") != 2304
            or type(intent.get("semantics_record_count")) is not int
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
            or type(intent.get("timeout_seconds")) is not int
            or not 300 <= intent["timeout_seconds"] <= 14_400
            or type(intent.get("max_gpu_hours")) not in (int, float)
            or not math.isfinite(float(intent["max_gpu_hours"]))
            or intent["max_gpu_hours"] <= 0
            or type(intent.get("wall_clock_stop_seconds")) is not int
            or type(intent.get("wall_clock_ceiling_seconds")) is not int
            or intent["wall_clock_ceiling_seconds"] != intent["timeout_seconds"]
            or intent["wall_clock_stop_seconds"] < 300
            or intent["wall_clock_stop_seconds"] > intent["wall_clock_ceiling_seconds"] - 300
            or plan.get("model_id") != MODEL_ID
            or plan.get("model_revision") != MODEL_REVISION
            or plan.get("rank") != 4 or plan.get("alpha") != 8.0
            or plan.get("global_batch_size") != 4
            or plan.get("world_size") != 1
            or plan.get("micro_batch_size") != 1
            or plan.get("accumulation_steps") != 4
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
                or type(observed) not in (int, float)
                or not math.isfinite(float(observed)) or observed <= 0
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
                 base_qualification_source_intent: Mapping[str, Any] | None = None,
                 base_qualification_carryforward: Mapping[str, Any] | None = None,
                 qualification_receipt: Mapping[str, Any] | None = None,
                 qualification_source_intent: Mapping[str, Any] | None = None,
                 qualification_source_carryforward: Mapping[str, Any] | None = None
                 ) -> dict:
    """Create a self-hashed not-submitted intent for external review."""
    from dataclasses import asdict
    manifest = validate_source_manifest(source_manifest)
    if manifest["source_commit"] != source_commit:
        raise PermissionError("Source manifest commit differs from the intent")
    base_receipt = validate_base_qualification_receipt(
        base_qualification_receipt)
    carryforward = None
    source_base_intent = None
    if base_receipt["source_commit"] == source_commit:
        if (base_qualification_source_intent is not None
                or base_qualification_carryforward is not None
                or base_receipt["source_sha256"] != manifest["manifest_sha256"]):
            raise PermissionError("Same-source base gate cannot use carry-forward")
    else:
        source_base_intent = validate_base_qualification_intent(
            base_qualification_source_intent)
        carryforward = validate_carryforward(
            base_qualification_carryforward,
            prior_intent=source_base_intent,
            base_receipt=base_receipt,
            current_source_manifest=manifest)
    if (base_receipt["model_tree_sha256"] != model_tree_sha256
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
        "source_release_path": source_release_path(source_commit),
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
        "base_qualification_source_intent": source_base_intent,
        "base_qualification_carryforward": carryforward,
        "base_qualification_carryforward_sha256": (
            None if carryforward is None else carryforward["carryforward_sha256"]),
        "qualification_technical_successor": (
            qualification_technical_successor_authority()),
        "qualification_technical_successor_sha256": (
            qualification_technical_successor_authority()[
                "technical_successor_sha256"]),
        "qualification_source_intent": None,
        "qualification_source_receipt": None,
        "qualification_source_carryforward": None,
        "qualification_source_carryforward_sha256": None,
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
                   != base_receipt["receipt_sha256"]
                or qualification_receipt.get(
                    "qualification_technical_successor_sha256")
                   != value["qualification_technical_successor_sha256"]):
            raise PermissionError("Capacity qualification uses a different base gate")
        if (qualification_receipt.get("source_commit") == source_commit
                and qualification_receipt.get("source_sha256")
                    == manifest["manifest_sha256"]):
            if (qualification_source_intent is not None
                    or qualification_source_carryforward is not None):
                raise PermissionError(
                    "Same-source qualification cannot use source carry-forward")
        else:
            verified_source_carryforward = (
                validate_qualification_source_carryforward(
                    qualification_source_carryforward,
                    qualification_intent=qualification_source_intent,
                    qualification_receipt=qualification_receipt,
                    current_source_manifest=manifest))
            value["qualification_source_intent"] = dict(
                qualification_source_intent)
            value["qualification_source_receipt"] = dict(
                qualification_receipt)
            value["qualification_source_carryforward"] = (
                verified_source_carryforward)
            value["qualification_source_carryforward_sha256"] = (
                verified_source_carryforward["carryforward_sha256"])
        value["reviewed_timeout_budget"] = primary_timeout_budget(
            plan, qualification_receipt)
    elif qualification_receipt is not None:
        raise PermissionError("Qualification intent cannot use prior qualification evidence")
    value["intent_sha256"] = digest(value)
    return validate_intent(value)


def verify_intent_base_qualification_authority(
        intent: Mapping[str, Any], base_receipt: Mapping[str, Any],
        code_root: str | Path) -> dict:
    """Verify direct or reviewed carry-forward base authority on the worker."""
    current = validate_intent(intent)
    base = validate_base_qualification_receipt(base_receipt)
    if (base["receipt_sha256"]
            != current["base_qualification_receipt_sha256"]
            or base["base_qualification_intent_sha256"]
               != current["base_qualification_intent_sha256"]):
        raise PermissionError("Training intent and frozen-base gate differ")
    carryforward = current.get("base_qualification_carryforward")
    if carryforward is None:
        if (base["source_commit"] != current["source_commit"]
                or base["source_sha256"] != current["source_sha256"]):
            raise PermissionError("Direct frozen-base source authority differs")
        return {"mode": "direct", "source_commit": base["source_commit"],
                "carryforward_sha256": None}
    prior_intent = validate_base_qualification_intent(
        current.get("base_qualification_source_intent"))
    verified = verify_deployed_carryforward(
        carryforward, code_root=code_root, prior_intent=prior_intent,
        base_receipt=base, current_source_manifest=current["source_manifest"])
    if (base["source_commit"] != prior_intent["source_commit"]
            or base["source_sha256"] != prior_intent["source_sha256"]):
        raise PermissionError("Carried frozen-base source authority differs")
    return {"mode": "carryforward", "source_commit": base["source_commit"],
            "carryforward_sha256": verified["carryforward_sha256"]}


def canonical_primary_plan_hashes() -> list[str]:
    from .qwen9b_ordinal_continuity_controller import matrix_plans
    from .qwen9b_ordinal_continuity_training import databricks_a10_plan
    policies = matrix_plans(EXPERIMENT_SPEC, 4)
    return sorted(databricks_a10_plan(plan).full_plan_sha256
                  for plan in policies)


def qualification_numerical_update_evidence(
        rank_receipt: Mapping[str, Any]) -> list[dict]:
    """Project exact fixed-scale numerical evidence without tensor payloads."""
    updates = rank_receipt.get("updates")
    if not isinstance(updates, list):
        raise PermissionError("Qualification numerical updates are absent")
    keys = (
        "optimizer_update", "gradient_norm_before_clip",
        "gradient_norm_positive",
        "active_branch_finite_nonzero_gradient_element_count",
        "active_branch_finite_nonzero_gradient_density",
        "fixed_scale_no_growth", "overflow_adjustments",
        "optimizer_step_pre_hook_calls", "logical_tokens",
        "physical_tokens_processed", "grad_scaler_before",
        "grad_scaler_after",
    )
    result = [{key: update.get(key) for key in keys} for update in updates]
    if any(set(row) != set(keys) or any(value is None for value in row.values())
           for row in result):
        raise PermissionError("Qualification numerical update evidence differs")
    return result


def create_qualification_receipt(intent: Mapping[str, Any],
                                 rank_receipt: Mapping[str, Any],
                                 evidence: Mapping[str, Any],
                                 checkpoint_tree_sha256: str) -> dict:
    intent = (_validate_historical_qualification_intent(intent)
              if intent.get("intent_sha256") == QUALIFICATION_INTENT_SHA256
              else validate_intent(intent))
    from .qwen9b_ordinal_continuity_training import (
        RANK_RUN_VERSION, TrainingPlan,
        validate_grad_scaler_recovery_receipt)
    plan = TrainingPlan(**intent["plan"]).validate()
    validate_grad_scaler_recovery_receipt(rank_receipt, plan)
    recovery = dict(rank_receipt["grad_scaler_recovery"])
    numerical_updates = qualification_numerical_update_evidence(rank_receipt)
    rank_payload = dict(rank_receipt); rank_hash = rank_payload.pop("receipt_sha256", None)
    evidence_payload = dict(evidence); evidence_hash = evidence_payload.pop(
        "evidence_sha256", None)
    metrics = evidence.get("qualification_metrics", {})
    tokens_seen = rank_receipt.get("tokens_seen")
    elapsed_seconds = rank_receipt.get("elapsed_seconds")
    tokens_per_second = rank_receipt.get("resources", {}).get("tokens_per_second")
    physical_tokens = rank_receipt.get("physical_tokens_processed")
    overflow_tokens = rank_receipt.get("overflow_recompute_tokens")
    nvml_samples = rank_receipt.get("resources", {}).get(
        "gpu", {}).get("nvml", {}).get("nonzero_gpu_sample_count")
    if (intent["plan"]["mode"] != "qualification"
            or rank_hash != digest(rank_payload)
            or rank_receipt.get("version") != RANK_RUN_VERSION
            or rank_receipt.get("status") != "complete"
            or rank_receipt.get("plan_sha256") != intent["plan_sha256"]
            or rank_receipt.get("executed_optimizer_step_count") != 2
            or rank_receipt.get("durable_published_update_count") != 2
            or rank_receipt.get("successful_optimizer_updates") != 2
            or rank_receipt.get("active_branch_changed") is not True
            or type(nvml_samples) is not int or nvml_samples <= 0
            or evidence_hash != digest(evidence_payload)
            or evidence.get("version") != DATABRICKS_EVIDENCE_VERSION
            or evidence.get("intent_sha256") != intent["intent_sha256"]
            or evidence.get("base_qualification_carryforward_sha256")
               != intent.get("base_qualification_carryforward_sha256")
            or evidence.get("qualification_technical_successor_sha256")
               != intent.get("qualification_technical_successor_sha256")
            or evidence.get("rank_receipt_sha256") != rank_receipt.get("receipt_sha256")
            or evidence.get("grad_scaler_recovery") != recovery
            or evidence.get("logical_tokens_seen") != tokens_seen
            or evidence.get("physical_tokens_processed") != physical_tokens
            or evidence.get("overflow_recompute_tokens") != overflow_tokens
            or evidence.get("numerical_update_evidence") != numerical_updates
            or evidence.get("live_base_components_before")
               != evidence.get("live_base_components_after")
            or evidence.get("save_reload_pass") is not True
            or any(value is not True for key, value in metrics.items()
                   if key.endswith("_pass") or key == "capacity_adequate")
            or type(tokens_seen) is not int or tokens_seen <= 0
            or type(physical_tokens) is not int
            or physical_tokens < tokens_seen
            or type(overflow_tokens) is not int or overflow_tokens < 0
            or physical_tokens != tokens_seen + overflow_tokens
            or type(elapsed_seconds) not in (int, float)
            or not math.isfinite(float(elapsed_seconds))
            or elapsed_seconds <= 0
            or type(tokens_per_second) not in (int, float)
            or not math.isfinite(float(tokens_per_second))
            or tokens_per_second <= 0
            or not math.isclose(tokens_per_second, tokens_seen / elapsed_seconds,
                                rel_tol=1e-9, abs_tol=1e-9)
            or not isinstance(checkpoint_tree_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_tree_sha256)):
        raise PermissionError("Actual rank-4 Databricks qualification evidence differs")
    value = {
        "version": DATABRICKS_QUALIFICATION_VERSION,
        "status": "qualified", "source_commit": intent["source_commit"],
        "source_sha256": intent["source_sha256"],
        "model_tree_sha256": intent["model_tree_sha256"],
        "model_receipt_sha256": intent["model_receipt_sha256"],
        "base_qualification_intent_sha256": intent[
            "base_qualification_intent_sha256"],
        "base_qualification_receipt_sha256": intent[
            "base_qualification_receipt_sha256"],
        "base_qualification_carryforward_sha256": intent.get(
            "base_qualification_carryforward_sha256"),
        "qualification_technical_successor_sha256": intent[
            "qualification_technical_successor_sha256"],
        "qualification_intent_sha256": intent["intent_sha256"],
        "qualification_plan_sha256": intent["plan_sha256"],
        "qualification_rank_receipt_sha256": rank_receipt["receipt_sha256"],
        "qualification_evidence_sha256": evidence_hash,
        "qualification_checkpoint_tree_sha256": checkpoint_tree_sha256,
        "qualification_tokens_seen": tokens_seen,
        "qualification_elapsed_seconds": elapsed_seconds,
        "qualification_tokens_per_second": tokens_per_second,
        "qualification_physical_tokens_processed": physical_tokens,
        "qualification_overflow_recompute_tokens": overflow_tokens,
        "qualification_grad_scaler_recovery": recovery,
        "qualification_numerical_update_evidence": numerical_updates,
        "selected_rank": 4,
        "authorized_primary_plan_sha256s": canonical_primary_plan_hashes(),
        "qualifier_checkpoint_primary_parent_eligible": False,
    }
    value["receipt_sha256"] = digest(value)
    return value


def _primary_base_authorities_match(
        intent: Mapping[str, Any], qualification_receipt: Mapping[str, Any]
        ) -> bool:
    """Keep qualifier and primary base-source lineage separate but anchored."""
    if (qualification_receipt.get("base_qualification_intent_sha256")
            != intent.get("base_qualification_intent_sha256")
            or qualification_receipt.get("base_qualification_receipt_sha256")
               != intent.get("base_qualification_receipt_sha256")):
        return False
    qualification_base_carryforward = qualification_receipt.get(
        "base_qualification_carryforward_sha256")
    source_carryforward = intent.get("qualification_source_carryforward")
    if source_carryforward is None:
        return (qualification_base_carryforward
                == intent.get("base_qualification_carryforward_sha256"))
    current_base_carryforward = intent.get(
        "base_qualification_carryforward_sha256")
    return (
        intent.get("qualification_source_receipt") == qualification_receipt
        and source_carryforward.get("base_qualification_carryforward_sha256")
            == qualification_base_carryforward
        and (current_base_carryforward is None
             or (isinstance(current_base_carryforward, str)
                 and re.fullmatch(r"[0-9a-f]{64}",
                                  current_base_carryforward) is not None)))


def create_primary_admission(intent: Mapping[str, Any],
                             qualification_receipt: Mapping[str, Any]) -> dict:
    intent = validate_intent(intent)
    from .qwen9b_ordinal_continuity_training import ADMISSION_VERSION
    receipt = dict(qualification_receipt); embedded = receipt.pop("receipt_sha256", None)
    same_source = (
        qualification_receipt.get("source_commit") == intent["source_commit"]
        and qualification_receipt.get("source_sha256") == intent["source_sha256"])
    source_carryforward = intent.get("qualification_source_carryforward")
    source_authorized = same_source
    if not same_source and source_carryforward is not None:
        source_authorized = (
            intent.get("qualification_source_receipt")
            == qualification_receipt
            and validate_qualification_source_carryforward(
                source_carryforward,
                qualification_intent=intent.get(
                    "qualification_source_intent"),
                qualification_receipt=qualification_receipt,
                current_source_manifest=intent["source_manifest"])
            == source_carryforward)
    base_authorities_match = _primary_base_authorities_match(
        intent, qualification_receipt)
    if (intent["plan"]["mode"] != "primary"
            or embedded != digest(receipt)
            or receipt.get("version") != DATABRICKS_QUALIFICATION_VERSION
            or receipt.get("status") != "qualified"
            or receipt.get("selected_rank") != 4
            or source_authorized is not True
            or receipt.get("model_tree_sha256") != intent["model_tree_sha256"]
            or receipt.get("model_receipt_sha256") != intent["model_receipt_sha256"]
            or base_authorities_match is not True
            or receipt.get("qualification_technical_successor_sha256")
               != intent.get("qualification_technical_successor_sha256")
            or receipt.get("qualification_grad_scaler_recovery", {}).get(
                "numerical_policy") != intent["plan"].get("numerical_policy")
            or receipt.get("qualification_grad_scaler_recovery", {}).get(
                "max_overflow_adjustments_per_partition")
               != intent["plan"].get("max_overflow_adjustments")
            or receipt.get("qualification_grad_scaler_recovery", {}).get(
                "minimum_scale")
               != intent["plan"].get("minimum_grad_scaler_scale")
            or intent.get("reviewed_timeout_budget", {}).get(
                "qualification_receipt_sha256") != embedded
            or intent.get("reviewed_timeout_budget", {}).get(
                "qualification_tokens_per_second")
               != receipt.get("qualification_tokens_per_second")
            or intent["plan_sha256"] not in receipt.get(
                "authorized_primary_plan_sha256s", ())):
        raise PermissionError("Qualification does not authorize this primary partition")
    value = {
        "version": ADMISSION_VERSION,
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
        "base_qualification_carryforward_sha256": intent.get(
            "base_qualification_carryforward_sha256"),
        "qualification_base_qualification_carryforward_sha256": receipt.get(
            "base_qualification_carryforward_sha256"),
        "qualification_technical_successor_sha256": intent[
            "qualification_technical_successor_sha256"],
        "qualification_source_carryforward_sha256": intent.get(
            "qualification_source_carryforward_sha256"),
        "qualification_grad_scaler_recovery": receipt[
            "qualification_grad_scaler_recovery"],
        "qualification_numerical_update_evidence": receipt[
            "qualification_numerical_update_evidence"],
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
    from .databricks_continuity_partition import _verify_durable_completion
    from .qwen9b_ordinal_continuity_runtime import directory_tree_sha256
    from .qwen9b_ordinal_continuity_training import TrainingPlan
    primary_intent = validate_intent(intent)
    if primary_intent["plan"]["mode"] != "primary":
        raise PermissionError("Primary authority replay requires a primary intent")
    raw_qualification_intent = json.loads(Path(
        qualification_intent_path).read_text(encoding="utf-8"))
    qualification_intent = (
        _validate_historical_qualification_intent(raw_qualification_intent)
        if raw_qualification_intent.get("intent_sha256")
           == QUALIFICATION_INTENT_SHA256
        else validate_intent(raw_qualification_intent))
    if qualification_intent["plan"]["mode"] != "qualification":
        raise PermissionError("Qualification intent has the wrong mode")
    if (qualification_intent.get("base_qualification_intent_sha256")
            != primary_intent.get("base_qualification_intent_sha256")
            or qualification_intent.get("base_qualification_receipt_sha256")
            != primary_intent.get("base_qualification_receipt_sha256")
            or qualification_intent.get(
                "qualification_technical_successor_sha256")
               != primary_intent.get(
                   "qualification_technical_successor_sha256")):
        raise PermissionError("Primary and capacity qualification use different base gate")
    root = Path(qualification_output)
    qualification_plan = TrainingPlan(
        **qualification_intent["plan"]).validate()
    _verify_durable_completion(
        root.parent, identity=qualification_intent["identity"],
        intent_sha256=qualification_intent["intent_sha256"],
        plan_sha256=qualification_intent["plan_sha256"],
        optimizer_updates=qualification_plan.optimizer_updates)
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
    from .qwen9b_ordinal_continuity_controller import matrix_plans, partition_identity
    from .qwen9b_ordinal_continuity_training import (
        TrainingPlan, databricks_a10_plan)
    current = TrainingPlan(**dict(plan)).validate()
    if current.stage == "delta1":
        return None
    policies = matrix_plans(EXPERIMENT_SPEC, 4)
    candidates = [databricks_a10_plan(value) for value in policies]
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
    mismatches = {key for key, value in expected.items()
                  if row.get(key) != value}
    reviewed_v7_source_alias = (
        label == "qualification"
        and mismatches == {"source_commit"}
        and row.get("source_commit") == QUALIFICATION_SOURCE_ORIGINAL_COMMIT
        and expected.get("intent_sha256") == QUALIFICATION_INTENT_SHA256
        and expected.get("plan_sha256") == QUALIFICATION_PLAN_SHA256
        and expected.get("receipt_sha256")
            == QUALIFICATION_RANK_RECEIPT_SHA256
        and expected.get("evidence_sha256") == QUALIFICATION_EVIDENCE_SHA256
        and expected.get("checkpoint_tree_sha256")
            == QUALIFICATION_CHECKPOINT_TREE_SHA256)
    if (row.get("status") != "complete" or row.get("receipt_replayed") is not True
            or (mismatches and not reviewed_v7_source_alias)):
        raise PermissionError(f"{label} Gold row differs from exact authority")
    return row


def _write_exclusive_volume_json(path: str | Path,
                                 value: Mapping[str, Any]) -> dict:
    """Create one sequential immutable Volume JSON file without fsync/rename."""
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical(dict(value)) + "\n").encode("utf-8")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
        if target.read_bytes() != data:
            raise IOError(f"Sequential Volume JSON readback differs: {target}")
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return dict(value)


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
        payload = {"owner": self.owner,
                   "intent_sha256": self.intent["intent_sha256"]}
        _write_exclusive_volume_json(self.lock, payload)
        try:
            _write_exclusive_volume_json(self.consumed, payload)
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
    payload = {"owner": owner, "identity": intent["identity"],
               "intent_sha256": intent["intent_sha256"],
               "status": "reserved_after_gpu_ready_before_model"}
    _write_exclusive_volume_json(active, payload)
    try:
        _write_exclusive_volume_json(consumed, payload)
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
    """Publish immutable closeout, then release only the exact active lease."""
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
            check_reservation(current)
            active.unlink()
        return receipt
    if not active.exists():
        raise PermissionError("Active reservation disappeared without closeout")
    current = json.loads(active.read_text(encoding="utf-8"))
    check_reservation(current)
    receipt = {
        "version": CLOSEOUT_VERSION, **expected,
        "terminal_status": terminal_status,
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "consumed_identity_retained": True,
        "reservation_sha256": digest(current),
    }
    receipt["receipt_sha256"] = digest(receipt)
    try:
        _write_exclusive_volume_json(closeout, receipt)
    except FileExistsError:
        receipt = check_closeout(json.loads(
            closeout.read_text(encoding="utf-8")))
    if active.exists():
        current = json.loads(active.read_text(encoding="utf-8"))
        check_reservation(current)
        active.unlink()
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
    carryforward_sha = capacity_qualification_receipt.get(
        "base_qualification_carryforward_sha256")
    technical_sha = capacity_qualification_receipt.get(
        "qualification_technical_successor_sha256")
    if (capacity_hash != digest(capacity_payload)
            or capacity_qualification_receipt.get("status") != "qualified"
            or capacity_qualification_receipt.get("selected_rank") != 4
            or capacity_qualification_receipt.get(
                "authorized_primary_plan_sha256s")
               != canonical_primary_plan_hashes()
            or capacity_qualification_receipt.get(
                "base_qualification_receipt_sha256") != base["receipt_sha256"]
            or (carryforward_sha is not None
                and not re.fullmatch(r"[0-9a-f]{64}", carryforward_sha))
            or (carryforward_sha is None
                and capacity_qualification_receipt.get("source_commit")
                    != base.get("source_commit"))
            or not re.fullmatch(r"[0-9a-f]{64}", str(technical_sha or ""))
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
    if (len({row.get("source_commit") for row in all_rows}) != 1
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
             "base_qualification_carryforward_sha256": carryforward_sha,
             "qualification_technical_successor_sha256": technical_sha,
             "capacity_qualification_receipt_sha256": capacity_hash,
             "capacity_qualification_rank_receipt_sha256":
                 capacity_gold_row.get("receipt_sha256"),
             "policy_receipt_sha256": sorted(
                 row.get("receipt_sha256") for row in policy_receipts)}
    value["summary_sha256"] = digest(value)
    return value
