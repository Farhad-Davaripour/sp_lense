#!/usr/bin/env python3
"""Fail-closed local validator for the reviewed classic-A10 provider spec."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

SOURCE = "ec6fdb9c1605b6090c6747081511b2092c93785d"
SOURCE_ROOT = f"/Workspace/Shared/research4/source_releases/{SOURCE}"
IDENTITY = "qualification_rank4_a10_fp16_scale1_v6"
INTENT_SHA = "96c19838047bf8e79ce1d48b572ee34ff6a7cbd5e8504830595ce665c1b717f3"
PLAN_SHA = "83a3e4a213316cba3ae21c9b83ebbf0fa631cfdbed5b615a158bf6495bac31bc"
FIRST_FAILURE_SHA = "eee70665851b858fb49e2cd3ecc43133cd61fa9e0cbf99afbf2233a5e41eb948"
SECOND_FAILURE_SHA = "03a781165e24777b0712eb3ba57f78aca428a59df7d715d4f27e087adda00eba"
INTENT_FILE_SHA = "be75fb5338385d6a9c840074abdbf758e870c7dac9e2b61e6e2f41a378bf2f90"
PLAN_FILE_SHA = "fcc9dec0c72a4a12dd13da811fd9938e90e93f31503e0503b298d5372b33b703"
IDEMPOTENCY_TOKEN = "r4-scale1-v6-classic-a10-ec6fdb9c-provider1"
NODE_TYPE = "Standard_NV36ads_A10_v5"
SPARK_VERSION = "18.x-gpu-ml-scala2.13"
INTENT_PATH = (
    "/Volumes/research4_simulated_continuity/bronze/model_data/intents/"
    f"{SOURCE}/QUALIFICATION_RANK4_INTENT-{INTENT_SHA}.json"
)
PLAN_PATH = (
    "/Volumes/research4_simulated_continuity/bronze/model_data/plans/"
    f"{SOURCE}/QUALIFICATION_RANK4_PLAN-{PLAN_SHA}.json"
)
BASE_RECEIPT_PATH = (
    "/Volumes/research4_simulated_continuity/bronze/model_data/"
    "base_qualifications/4f5b6533e5da9523f529413e68a8e7270b045f34/"
    "frozen_base_format_qualification_v2/BASE_QUALIFICATION_RECEIPT.json"
)
TASK_KEYS = (
    "validate_before_gpu",
    "run_one_partition",
    "finalize_partition_mlflow",
    "release_active_reservation",
)
PINNED_PACKAGES = (
    "torch==2.11.0+cu128",
    "transformers==5.19.0",
    "bitsandbytes==0.49.2",
    "accelerate==1.15.0",
    "safetensors==0.8.0",
    "fla-core==0.5.2",
    "einops==0.8.2",
    "numpy==2.2.2",
    "psutil==7.2.2",
    "triton==3.6.0",
    "nvidia-ml-py==13.615.71",
    "mlflow==3.1.0",
)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(actual) == set(expected)
                and all(_strict_equal(actual[key], value)
                        for key, value in expected.items()))
    if isinstance(expected, list):
        return (len(actual) == len(expected)
                and all(_strict_equal(left, right)
                        for left, right in zip(actual, expected)))
    return actual == expected


def _require_exact(actual: object, expected: object, label: str) -> None:
    if not _strict_equal(actual, expected):
        raise ValueError(f"{label} drift")


def _task_map(spec: dict) -> dict[str, dict]:
    tasks = spec.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(TASK_KEYS):
        raise ValueError("exact four-task provider DAG required")
    mapping = {item.get("task_key"): item for item in tasks}
    if tuple(item.get("task_key") for item in tasks) != TASK_KEYS:
        raise ValueError("task order or identity drift")
    if set(mapping) != set(TASK_KEYS):
        raise ValueError("task keys must be unique and exact")
    return mapping


def _notebook(name: str, params: dict) -> dict:
    return {
        "notebook_path": f"{SOURCE_ROOT}/notebooks/{name}",
        "source": "WORKSPACE",
        "base_parameters": params,
    }


def _common_params() -> dict:
    return {
        "run_id": "{{job.run_id}}",
        "code_path": SOURCE_ROOT,
        "intent_path": INTENT_PATH,
    }


def _libraries() -> list[dict]:
    items = []
    for index, package in enumerate(PINNED_PACKAGES):
        pypi = {"package": package}
        if index == 0:
            pypi["repo"] = "https://download.pytorch.org/whl/cu128"
        items.append({"pypi": pypi})
    return items


def _new_cluster() -> dict:
    return {
        "spark_version": SPARK_VERSION,
        "node_type_id": NODE_TYPE,
        "driver_node_type_id": NODE_TYPE,
        "num_workers": 0,
        "runtime_engine": "STANDARD",
        "data_security_mode": "SINGLE_USER",
        "single_user_name": "farhad.davaripour@gmail.com",
        "azure_attributes": {
            "availability": "ON_DEMAND_AZURE",
            "first_on_demand": 1,
        },
        "spark_conf": {
            "spark.databricks.cluster.profile": "singleNode",
            "spark.master": "local[*, 4]",
        },
        "spark_env_vars": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        "custom_tags": {
            "ResourceClass": "SingleNode",
            "aidevkit_project": "ai-dev-kit",
            "research": "research4",
            "study": "simulated-continuity",
            "cost_scope": "scale1-v6-classic-a10-max-0.75-gpu-hour",
        },
    }


def _expected_tasks() -> list[dict]:
    preflight_params = {
        **_common_params(),
        "plan_path": PLAN_PATH,
        "admission_path": "",
        "parent_checkpoint": "",
        "base_qualification_receipt_path": BASE_RECEIPT_PATH,
        "expected_mode": "qualification",
    }
    gpu_params = {
        "run_id": "{{job.run_id}}",
        "job_start_timestamp_ms": "{{job.start_time.timestamp_ms}}",
        "code_path": SOURCE_ROOT,
        "intent_path": INTENT_PATH,
        "plan_path": PLAN_PATH,
        "admission_path": "",
        "parent_checkpoint": "",
        "base_qualification_receipt_path": BASE_RECEIPT_PATH,
        "expected_mode": "qualification",
    }
    return [
        {
            "task_key": "validate_before_gpu",
            "notebook_task": _notebook(
                "databricks_continuity_preflight", preflight_params),
            "environment_key": "continuity_preflight_cpu",
            "max_retries": 0,
            "retry_on_timeout": False,
            "timeout_seconds": 900,
            "disable_auto_optimization": True,
        },
        {
            "task_key": "run_one_partition",
            "depends_on": [{"task_key": "validate_before_gpu"}],
            "notebook_task": _notebook(
                "databricks_continuity_partition", gpu_params),
            "new_cluster": _new_cluster(),
            "libraries": _libraries(),
            "max_retries": 0,
            "retry_on_timeout": False,
            "timeout_seconds": 2700,
            "disable_auto_optimization": True,
        },
        {
            "task_key": "finalize_partition_mlflow",
            "depends_on": [{"task_key": "run_one_partition"}],
            "run_if": "ALL_DONE",
            "notebook_task": _notebook(
                "databricks_continuity_partition_finalize",
                {**_common_params(), "plan_path": PLAN_PATH},
            ),
            "environment_key": "continuity_finalize_cpu",
            "max_retries": 0,
            "retry_on_timeout": False,
            "timeout_seconds": 900,
            "disable_auto_optimization": True,
        },
        {
            "task_key": "release_active_reservation",
            "depends_on": [
                {"task_key": "validate_before_gpu"},
                {"task_key": "run_one_partition"},
                {"task_key": "finalize_partition_mlflow"},
            ],
            "run_if": "ALL_DONE",
            "notebook_task": _notebook(
                "databricks_continuity_cleanup", _common_params()),
            "environment_key": "continuity_preflight_cpu",
            "max_retries": 0,
            "retry_on_timeout": False,
            "timeout_seconds": 600,
            "disable_auto_optimization": True,
        },
    ]


def _expected_environments() -> list[dict]:
    return [
        {
            "environment_key": "continuity_preflight_cpu",
            "spec": {"environment_version": "6", "dependencies": []},
        },
        {
            "environment_key": "continuity_finalize_cpu",
            "spec": {
                "environment_version": "6",
                "dependencies": ["mlflow==3.1.0", "numpy==2.2.2"],
            },
        },
    ]


def validate(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("provider spec must be a JSON object")
    if set(spec) != {
        "idempotency_token", "run_name", "timeout_seconds",
        "performance_target", "tasks", "environments",
    }:
        raise ValueError("top-level provider fields drift")
    _require_exact(spec.get("idempotency_token"), IDEMPOTENCY_TOKEN,
                   "idempotency token")
    _require_exact(
        spec.get("run_name"),
        "Research4 scale1 v6 classic A10 provider recovery",
        "run name",
    )
    _require_exact(spec.get("timeout_seconds"), 4500, "outer timeout")
    _require_exact(spec.get("performance_target"),
                   "PERFORMANCE_OPTIMIZED", "performance target")
    _task_map(spec)
    _require_exact(spec["tasks"], _expected_tasks(), "exact provider DAG")
    _require_exact(spec["environments"], _expected_environments(),
                   "exact CPU environments")

    # Current official list price: $3.20 VM + 4 Jobs DBUs * $0.30.
    hourly_usd = 3.20 + 4.0 * 0.30
    maximum_compute_usd = round(hourly_usd * 0.75, 2)
    if not math.isclose(hourly_usd, 4.40) or maximum_compute_usd > 5.0:
        raise ValueError("classic A10 cost envelope exceeds approved cap")
    return {
        "identity": IDENTITY,
        "source_commit": SOURCE,
        "intent_sha256": INTENT_SHA,
        "plan_sha256": PLAN_SHA,
        "first_provider_failure_sha256": FIRST_FAILURE_SHA,
        "second_provider_failure_sha256": SECOND_FAILURE_SHA,
        "provider_spec_sha256": digest(spec),
        "node_type_id": NODE_TYPE,
        "spark_version": SPARK_VERSION,
        "hourly_cost_usd": hourly_usd,
        "maximum_gpu_hours": 0.75,
        "maximum_compute_cost_usd": maximum_compute_usd,
        "planning_cap_usd": 5.0,
        "status": "validated_local_only_not_submitted",
    }


def validate_binding(binding: dict, spec: dict, *, repo_root: Path,
                     artifact_root: Path | None) -> dict:
    expected_keys = {
        "abort_controls", "binding_sha256", "classic_provider_submission_ordinal",
        "compute", "cost", "dataset", "idempotency_token", "identity", "intent",
        "model", "plan", "prior_provider_failures", "provider_spec",
        "review_snapshot", "schema_audit", "scientific_attempt_consumed",
        "scientific_attempt_number", "source", "startup_monitor", "status",
        "submission_requires_supervisory_approval", "validator", "version",
        "workspace_profile",
    }
    if not isinstance(binding, dict) or set(binding) != expected_keys:
        raise ValueError("provider binding fields drift")
    self_hash = binding.get("binding_sha256")
    payload = dict(binding)
    payload.pop("binding_sha256", None)
    if self_hash != digest(payload):
        raise ValueError("provider binding self-hash mismatch")

    result = validate(spec)
    provider = binding["provider_spec"]
    spec_path = repo_root / provider["path"]
    _require_exact(provider, {
        "canonical_sha256": result["provider_spec_sha256"],
        "file_sha256": file_digest(spec_path),
        "path": "research4/infrastructure/classic_a10_scale1_v6/submit.json",
    }, "provider spec binding")
    if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
        raise ValueError("bound provider spec bytes do not decode to reviewed spec")

    _require_exact(binding["identity"], IDENTITY, "binding identity")
    _require_exact(binding["idempotency_token"], IDEMPOTENCY_TOKEN,
                   "binding idempotency token")
    _require_exact(binding["scientific_attempt_number"], 1,
                   "scientific attempt number")
    _require_exact(binding["scientific_attempt_consumed"], False,
                   "scientific attempt consumption")
    _require_exact(binding["classic_provider_submission_ordinal"], 1,
                   "classic provider ordinal")
    _require_exact(binding["status"],
                   "review_candidate_not_submission_authority",
                   "binding review status")
    _require_exact(binding["submission_requires_supervisory_approval"], True,
                   "supervisory approval gate")
    _require_exact(binding["workspace_profile"], "azure-research4",
                   "workspace profile")

    _require_exact(binding["source"], {
        "commit": SOURCE,
        "shared_path": SOURCE_ROOT,
        "tree": "61739e20e2cc14c4ea02f4aff66d07bfd0b7c47a",
    }, "source binding")
    _require_exact(binding["version"],
                   "research4-classic-a10-provider-recovery-binding-v1",
                   "binding version")
    _require_exact(binding["model"], {
        "bytes": 19329393661,
        "public_download_bytes_required": 0,
        "receipt_sha256": "4d1aece45e905dace3990a1ac0e351bd2ef8a53912d2fab93298a1ca06eb41f1",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "tree_sha256": "9845026dbe255e24b105224eebbb5436d315713b9a5c53c434137896b160c5b1",
        "volume_path": (
            "/Volumes/research4_simulated_continuity/bronze/model_data/"
            "model_snapshots/qwen35_9b/"
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a/model"),
    }, "model binding")
    _require_exact(binding["dataset"], {
        "dataset_tree_sha256": "7786d11946799776df964bc3d1dded641cf560311eb5e099ce4ce9d5f06283ee",
        "exposure_binding_sha256": "282b0c6f4965ef75660e023a2c347a40a79b9de39e84353159ae5f3ad72010a0",
        "silver_binding_sha256": "6369a0ccb643f435be9cbc6c91b69c99ac35f19a8aa99b038a758d3e3f9fd36e",
    }, "dataset binding")
    _require_exact(binding["intent"], {
        "file_sha256": INTENT_FILE_SHA,
        "path": INTENT_PATH,
        "semantic_sha256": INTENT_SHA,
    }, "intent binding")
    _require_exact(binding["plan"], {
        "file_sha256": PLAN_FILE_SHA,
        "path": PLAN_PATH,
        "semantic_sha256": PLAN_SHA,
    }, "plan binding")
    if artifact_root is not None:
        intent_file = artifact_root / "QUALIFICATION_RANK4_INTENT.json"
        plan_file = artifact_root / "QUALIFICATION_RANK4_PLAN.json"
        if file_digest(intent_file) != INTENT_FILE_SHA:
            raise ValueError("actual intent file hash mismatch")
        if file_digest(plan_file) != PLAN_FILE_SHA:
            raise ValueError("actual plan file hash mismatch")
        intent = json.loads(intent_file.read_text(encoding="utf-8"))
        if intent.get("intent_sha256") != INTENT_SHA:
            raise ValueError("actual intent semantic self-hash mismatch")

    failures = binding["prior_provider_failures"]
    _require_exact(failures, [
        {
            "file_sha256": FIRST_FAILURE_SHA,
            "gpu_task_run_id": "167918078402139",
            "owner_run_id": "522896173656534",
        },
        {
            "file_sha256": SECOND_FAILURE_SHA,
            "gpu_task_run_id": "131501017248503",
            "owner_run_id": "168395935320906",
        },
    ], "provider failure lineage")
    if artifact_root is not None:
        if file_digest(artifact_root / "SCALE1_PROVIDER_STARTUP_FAILURE.json") != FIRST_FAILURE_SHA:
            raise ValueError("first provider failure file hash mismatch")
        if file_digest(artifact_root / "SCALE1_PROVIDER_STARTUP_FAILURE_SECOND.json") != SECOND_FAILURE_SHA:
            raise ValueError("second provider failure file hash mismatch")

    _require_exact(binding["compute"], {
        "availability": "ON_DEMAND_AZURE",
        "driver_node_type_id": NODE_TYPE,
        "gpu_count": 1,
        "node_type_id": NODE_TYPE,
        "num_workers": 0,
        "runtime_engine": "STANDARD",
        "spark_version": SPARK_VERSION,
        "topology": "single_node_driver",
    }, "compute binding")
    _require_exact(binding["cost"], {
        "currency": "USD",
        "databricks_jobs_compute_dbu_count_per_hour": 4.0,
        "databricks_premium_jobs_compute_price_per_dbu_hour": 0.3,
        "maximum_compute_cost_usd": 3.3,
        "maximum_gpu_hours": 0.75,
        "planning_cap_usd": 5.0,
        "planning_rate_usd_per_hour": 4.4,
        "pricing_region": "centralus",
        "pricing_sources": [
            "https://prices.azure.com/api/retail/prices",
            "https://azure.microsoft.com/en-us/pricing/details/databricks/",
        ],
        "vm_linux_on_demand_price_per_hour": 3.2,
        "vm_price_effective_start_utc": "2026-08-01T00:00:00Z",
    }, "cost binding")
    _require_exact(binding["abort_controls"], {
        "cancel_if_notebook_not_started_seconds": 1200,
        "gpu_task_ceiling_seconds": 2700,
        "job_outer_ceiling_seconds": 4500,
        "operator_stop_seconds_from_job_start": 2400,
        "poll_interval_seconds": 60,
        "post_consumption_failure_is_terminal": True,
        "pre_consumption_infrastructure_failure_allows_no_automatic_resubmission": True,
    }, "abort controls")

    monitor = binding["startup_monitor"]
    monitor_path = repo_root / monitor["path"]
    _require_exact(monitor, {
        "arm_cancel_required": True,
        "file_sha256": file_digest(monitor_path),
        "path": "research4/tools/monitor_classic_a10_recovery.py",
        "poll_interval_seconds": 60,
    }, "startup monitor binding")
    validator = binding["validator"]
    validator_path = repo_root / validator["path"]
    _require_exact(validator, {
        "file_sha256": file_digest(validator_path),
        "path": "research4/tools/validate_classic_a10_recovery.py",
    }, "validator binding")
    schema = binding["schema_audit"]
    schema_path = repo_root / schema["path"]
    schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_self_payload = dict(schema_payload)
    schema_self_hash = schema_self_payload.pop("schema_audit_sha256", None)
    if schema_self_hash != digest(schema_self_payload):
        raise ValueError("schema audit self-hash mismatch")
    _require_exact(schema, {
        "file_sha256": file_digest(schema_path),
        "path": "research4/infrastructure/classic_a10_scale1_v6/SCHEMA_AUDIT.json",
        "self_sha256": schema_self_hash,
    }, "schema audit binding")
    _require_exact(schema_payload, {
        "checked_at_utc": "2026-10-29T18:25:55Z",
        "classic_gpu_task_field": "new_cluster",
        "cloud_mutation_performed": False,
        "databricks_cli_version": "1.22.1",
        "databricks_sdk_go_version": "0.189.0",
        "mixed_serverless_cpu_and_classic_gpu_supported": True,
        "node_type": {
            "available": True,
            "deprecated": False,
            "gpu_count": 1,
            "node_type_id": NODE_TYPE,
        },
        "provider_spec_canonical_sha256": result["provider_spec_sha256"],
        "provider_spec_file_sha256": file_digest(spec_path),
        "result": "PASS_LOCAL_READ_ONLY_SCHEMA_AUDIT",
        "runtime": {
            "available": True,
            "name": "18 LTS ML (GPU, Scala 2.13, Spark 4.1.0)",
            "spark_version": SPARK_VERSION,
        },
        "schema_audit_sha256": schema_self_hash,
        "submit_run_job_clusters_field_supported": False,
        "submit_task_job_cluster_key_supported": False,
        "submit_task_new_cluster_supported": True,
        "version": "research4-classic-a10-submit-schema-audit-v1",
    }, "exact schema audit")

    review = binding["review_snapshot"]
    if (set(review) != {
            "active_air_runs", "active_clusters", "active_job_runs",
            "checked_at_utc", "consumed_identity_exists",
            "consumed_identity_path",
            "fresh_recheck_required_immediately_before_submission",
            "scientific_lock_exists", "scientific_lock_path"}
            or review.get("active_air_runs") != 0
            or review.get("active_clusters") != 0
            or review.get("active_job_runs") != 0
            or review.get("consumed_identity_exists") is not False
            or review.get("scientific_lock_exists") is not False
            or review.get("fresh_recheck_required_immediately_before_submission") is not True
            or review.get("consumed_identity_path") != (
                "/Volumes/research4_simulated_continuity/bronze/model_data/"
                "locks/consumed/qualification_rank4_a10_fp16_scale1_v6.json")
            or review.get("scientific_lock_path") != (
                "/Volumes/research4_simulated_continuity/bronze/model_data/"
                "locks/scientific.lock")):
        raise ValueError("review snapshot does not fail closed")
    return {
        **result,
        "artifact_replay_performed": artifact_root is not None,
        "provider_binding_sha256": self_hash,
        "binding_status": binding["status"],
    }


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else (
        Path(__file__).parents[1]
        / "infrastructure" / "classic_a10_scale1_v6" / "submit.json"
    )
    spec = json.loads(path.read_text(encoding="utf-8"))
    result = validate(spec)
    if len(argv) > 2:
        binding_path = Path(argv[2])
        artifact_root = (Path(argv[3]) if len(argv) > 3 else Path(
            "C:/Users/farha/Research4/artifacts/"
            "qwen9b_simulated_continuity_20261028/three_residual_ec6fdb9c"))
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        result = validate_binding(
            binding, spec, repo_root=Path(__file__).parents[2],
            artifact_root=artifact_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
