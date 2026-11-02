# Databricks notebook source
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, re, shutil, sys, tempfile, time

task_started_monotonic = time.monotonic()

for name in ("code_path", "intent_path", "run_id", "job_start_timestamp_ms"):
    dbutils.widgets.text(name, "")
values = {name: dbutils.widgets.get(name) for name in (
    "code_path", "intent_path", "run_id", "job_start_timestamp_ms")}
if any(not value for value in values.values()):
    raise ValueError("Transfer evaluation requires exact code, intent and run owner")
job_started_epoch = int(values["job_start_timestamp_ms"]) / 1000.0
if job_started_epoch <= 0 or job_started_epoch > time.time() + 5:
    raise ValueError("Transfer evaluation job-start timestamp differs")


def _whole_job_elapsed():
    return max(time.monotonic() - task_started_monotonic,
               time.time() - job_started_epoch)
raw_intent = json.loads(Path(values["intent_path"]).read_text(encoding="utf-8"))


def _bootstrap_verify_source(root_value, authority):
    root = Path(root_value).resolve(); manifest = authority.get("source_manifest", {})
    payload = dict(manifest); embedded = payload.pop("manifest_sha256", None)
    commit = authority.get("source_commit"); files = manifest.get("files")
    canonical_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    critical = (
        "src/__init__.py", "src/action_serialization.py", "src/databricks_simulated_continuity.py",
        "src/base_qualification_carryforward.py",
        "src/databricks_continuity_partition.py",
        "src/continuity_transfer_databricks.py",
        "src/continuity_transfer_evaluation.py",
        "src/continuity_transfer_inference_worker.py",
        "src/continuity_transfer_raw_audit.py",
        "src/qwen9b_ordinal_continuity_lora.py",
        "src/volume_durable_io.py",
    )
    if (not re.fullmatch(r"[0-9a-f]{40}", str(commit or ""))
            or root.as_posix() != f"/Workspace/Shared/research4/source_releases/{commit}"
            or authority.get("source_sha256") != embedded
            or hashlib.sha256(canonical_bytes).hexdigest() != embedded
            or not isinstance(files, list)):
        raise PermissionError("Independent transfer source authority differs")
    indexed = {row.get("path"): row for row in files if isinstance(row, dict)}
    for logical in critical:
        row = indexed.get(logical); path = root / logical
        if not isinstance(row, dict) or not path.is_file() or path.is_symlink():
            raise PermissionError("Critical transfer source is absent")
        data = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace(
            "\r", "\n").encode("utf-8")
        if (len(data) != row.get("canonical_bytes")
                or hashlib.sha256(data).hexdigest() != row.get("sha256")):
            raise PermissionError("Critical transfer source changed before import")
    return embedded


bootstrap_source_sha256 = _bootstrap_verify_source(values["code_path"], raw_intent)
sys.path.insert(0, values["code_path"])

from src.continuity_transfer_databricks import (
    build_databricks_run_binding, validate_transfer_intent,
    verify_transfer_deployed_source, verify_transfer_inputs,
)
from src.databricks_continuity_partition import _bounded_run, _stage_local_model
from src.databricks_simulated_continuity import (
    GOLD_ADMISSION_TABLE, MODEL_RECEIPT, VOLUME, canonical, reserve_intent,
)
from src.qwen9b_ordinal_continuity_training import (
    assert_databricks_a10x1_fp16_runtime, assert_fast_deltanet_dependencies,
)

intent = validate_transfer_intent(raw_intent)
source = verify_transfer_deployed_source(values["code_path"], intent)
if source["manifest_sha256"] != bootstrap_source_sha256:
    raise PermissionError("Transfer bootstrap and full source verification differ")
# The immediately preceding CPU preflight rehashes the full immutable 19 GB
# model snapshot.  On paid GPU time we recheck the signed receipt and all small
# scientific inputs; _stage_local_model then fully hashes the copied local
# model once before any model load.
inputs = verify_transfer_inputs(intent, verify_model_bytes=False)

# All imports, kernels, CUDA identity, quantization and disk checks precede the
# one-use reservation.  No target model or adapter is loaded before this point.
dependency_proof = assert_fast_deltanet_dependencies()
import bitsandbytes, torch, pynvml
hardware_proof = assert_databricks_a10x1_fp16_runtime(torch)
probe = torch.zeros(256, dtype=torch.float16, device="cuda:0")
quantized, state = bitsandbytes.functional.quantize_4bit(
    probe, quant_type="nf4", compress_statistics=True)
restored = bitsandbytes.functional.dequantize_4bit(quantized, state)
if restored.shape != probe.shape or not torch.isfinite(restored).all():
    raise RuntimeError("bitsandbytes NF4 CUDA readiness failed")
del restored, quantized, state, probe
torch.cuda.synchronize(0); torch.cuda.empty_cache()
model_receipt = inputs["model"]
required_disk = int(model_receipt["total_bytes"]) + int(
    intent["resource_plan"]["max_raw_export_bytes"]) + 8 * 1024**3
local_disk_free = shutil.disk_usage(tempfile.gettempdir()).free
if local_disk_free < required_disk:
    raise RuntimeError("Local model-copy and raw-export disk readiness failed")
pynvml.nvmlInit()
try:
    if pynvml.nvmlDeviceGetCount() != 1:
        raise RuntimeError("Transfer evaluation requires exactly one GPU")
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    if "A10" not in name or int(memory.free) < 20 * 1024**3:
        raise RuntimeError("Transfer evaluation lacks the reviewed A10 envelope")
    nvml_proof = {"name": name, "free_bytes": int(memory.free),
                  "total_bytes": int(memory.total)}
finally:
    pynvml.nvmlShutdown()

output_root = Path(intent["output_root"])
if output_root.exists():
    raise FileExistsError("Transfer evaluation output is immutable")
if _whole_job_elapsed() >= intent["resource_plan"]["wall_clock_stop_seconds"]:
    raise TimeoutError("Transfer setup exhausted the reviewed envelope before reservation")
output_root.parent.mkdir(parents=True, exist_ok=True)
lock_root = Path(VOLUME) / "locks"
reservation = reserve_intent(lock_root, intent, values["run_id"])
run_binding = build_databricks_run_binding(
    intent, job_run_id=values["run_id"], reservation=reservation)
output_root.mkdir(exist_ok=False)
(output_root / "RUN_BINDING.json").write_text(
    json.dumps(run_binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(output_root / "RESOURCE_PLAN.json").write_text(
    json.dumps(intent["resource_plan"], indent=2, sort_keys=True) + "\n",
    encoding="utf-8")
spark.createDataFrame([{
    "identity": intent["identity"], "intent_sha256": intent["intent_sha256"],
    "job_run_id": values["run_id"],
    "status": "consumed_after_gpu_ready_before_model",
    "consumed_at": datetime.now(timezone.utc),
}], schema=spark.table(GOLD_ADMISSION_TABLE).schema).write.mode(
    "append").saveAsTable(GOLD_ADMISSION_TABLE)

local_model = None
import mlflow
from mlflow.tracking import MlflowClient
experiment = mlflow.set_experiment(intent["mlflow_experiment_path"])
mlflow_client = MlflowClient()
created_run = mlflow_client.create_run(
    experiment.experiment_id, tags={"mlflow.runName": intent["identity"],
                                    "terminal_status": "gpu_running"})
mlflow_run_id = created_run.info.run_id
raw_complete = False
try:
    local_model, local_receipt, cache = _stage_local_model(
        intent["model_root"], MODEL_RECEIPT, intent["identity"],
        intent["model_tree_sha256"], intent["model_receipt_sha256"])
    env = dict(os.environ)
    env.update({
        "RESEARCH4_EXECUTION_PLATFORM": "databricks",
        "RESEARCH4_MODEL_ROOT": str(local_model),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    raw_output = output_root / "raw"
    command = [
        sys.executable, "-m", "src.continuity_transfer_inference_worker",
        "--sealed", intent["sealed_root"],
        "--checkpoint-manifest", intent["checkpoint_manifest_path"],
        "--checkpoint-root", intent["checkpoint_root"],
        "--output", str(raw_output),
        "--source-commit", intent["source_commit"],
        "--source-package-sha256", intent["source_sha256"],
        "--checkpoint-package-sha256", intent["checkpoint_package_sha256"],
        "--resource-plan", str(output_root / "RESOURCE_PLAN.json"),
        "--run-binding", str(output_root / "RUN_BINDING.json"),
    ]
    prework_seconds = _whole_job_elapsed()
    remaining_stop = (intent["resource_plan"]["wall_clock_stop_seconds"]
                      - prework_seconds)
    remaining_ceiling = (intent["resource_plan"]["wall_clock_ceiling_seconds"]
                         - prework_seconds)
    if (remaining_stop <= 0
            or remaining_ceiling - remaining_stop
               < intent["resource_plan"]["protected_audit_and_export_seconds"]):
        raise TimeoutError("Transfer setup exhausted the reviewed inference envelope")
    result = _bounded_run(
        command, cwd=Path(values["code_path"]).resolve(), env=env,
        stop_seconds=remaining_stop, ceiling_seconds=remaining_ceiling)
    result["prework_seconds"] = prework_seconds
    result["whole_task_elapsed_seconds"] = _whole_job_elapsed()
    result["model_cache"] = cache
    result["gpu_readiness"] = {
        "source_manifest_sha256": source["manifest_sha256"],
        "dependency_profile": dependency_proof["kernel_selection"]["profile"],
        "hardware_profile": hardware_proof["profile"], "nvml": nvml_proof,
        "bitsandbytes_version": bitsandbytes.__version__,
        "local_disk_free_bytes": int(local_disk_free),
        "local_disk_required_bytes": int(required_disk),
    }
    (output_root / "TERMINAL.json").write_text(
        canonical(result) + "\n", encoding="utf-8")
    if result["exit_code"] != 0 or result["operator_stop_sent"]:
        mlflow_client.set_tag(mlflow_run_id, "terminal_status", "stopped_or_failed")
        mlflow_client.log_dict(mlflow_run_id, result, "terminal.json")
        mlflow_client.set_terminated(mlflow_run_id, status="FAILED")
        raise RuntimeError("Transfer inference did not produce a complete grid")
    if (_whole_job_elapsed()
            >= intent["resource_plan"]["wall_clock_ceiling_seconds"]):
        raise TimeoutError("Transfer evidence finalization exceeded the reviewed ceiling")
    receipt = json.loads((raw_output / "WORKER_RECEIPT.json").read_text(
        encoding="utf-8"))
    gpu_metrics = receipt["resources"]["gpu"]
    for metric_name, metric_value in {
        "record_count": receipt["record_count"],
        "valid_response_count": receipt["valid_response_count"],
        "generated_token_count": receipt["generated_token_count"],
        "elapsed_seconds": receipt["elapsed_seconds"],
        "model_copy_seconds": cache["copy_seconds"],
        "mean_gpu_utilization_percent": gpu_metrics[
            "mean_gpu_utilization_percent"],
        "peak_gpu_utilization_percent": gpu_metrics[
            "peak_gpu_utilization_percent"],
        "peak_gpu_memory_used_bytes": gpu_metrics[
            "peak_memory_used_bytes"],
        "nvml_sample_count": gpu_metrics["sample_count"],
    }.items():
        mlflow_client.log_metric(mlflow_run_id, metric_name, metric_value)
    mlflow_client.log_dict(mlflow_run_id, receipt, "worker_receipt.json")
    mlflow_client.log_dict(mlflow_run_id, result, "terminal.json")
    mlflow_client.set_tag(
        mlflow_run_id, "terminal_status", "raw_complete_pending_independent_audit")
    raw_complete = True
    (output_root / "MLFLOW.json").write_text(json.dumps({
        "mlflow_run_id": mlflow_run_id,
        "status": "raw_complete_pending_independent_audit",
    }, sort_keys=True) + "\n", encoding="utf-8")
finally:
    if local_model is not None:
        shutil.rmtree(Path(local_model).parent, ignore_errors=True)
    if not raw_complete:
        try:
            mlflow_client.set_terminated(mlflow_run_id, status="FAILED")
        except BaseException:
            pass
    torch.cuda.empty_cache()

dbutils.notebook.exit(json.dumps({
    "status": "raw_complete_pending_independent_audit",
    "identity": intent["identity"], "mlflow_run_id": mlflow_run_id,
    "record_count": receipt["record_count"],
}))
