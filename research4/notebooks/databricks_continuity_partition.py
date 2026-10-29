# Databricks notebook source
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, re, shutil, sys, tempfile, time

task_started_monotonic = time.monotonic()

for name in ("code_path", "intent_path", "plan_path", "admission_path",
             "parent_checkpoint", "qualification_intent_path",
             "qualification_output", "qualification_receipt_path",
             "base_qualification_receipt_path",
             "job_start_timestamp_ms",
             "expected_mode"):
    dbutils.widgets.text(name, "")
values = {name: dbutils.widgets.get(name) for name in (
    "code_path", "intent_path", "plan_path", "admission_path", "parent_checkpoint",
    "qualification_intent_path", "qualification_output",
    "qualification_receipt_path", "base_qualification_receipt_path",
    "job_start_timestamp_ms")}
if not values["code_path"] or not values["intent_path"] or not values["plan_path"]:
    raise ValueError("Exact code, intent and plan paths are required")
job_started_epoch = int(values["job_start_timestamp_ms"]) / 1000.0
if job_started_epoch <= 0 or job_started_epoch > time.time() + 5:
    raise ValueError("Partition job-start timestamp differs")
intent_preview = json.loads(Path(values["intent_path"]).read_text(encoding="utf-8"))


def _bootstrap_verify_source(root_value, authority):
    root = Path(root_value).resolve(); manifest = authority.get("source_manifest", {})
    payload = dict(manifest); embedded = payload.pop("manifest_sha256", None)
    commit = authority.get("source_commit"); files = manifest.get("files")
    canonical_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    critical = (
        "src/__init__.py", "src/action_serialization.py", "src/databricks_simulated_continuity.py",
        "src/databricks_continuity_partition.py",
        "src/databricks_continuity_partition_worker.py",
        "src/qwen9b_ordinal_continuity_runtime.py",
        "src/qwen9b_ordinal_continuity_training.py",
        "src/qwen9b_ordinal_continuity_lora.py",
        "src/qwen9b_ordinal_continuity_worker.py",
    )
    if (not re.fullmatch(r"[0-9a-f]{40}", str(commit or ""))
            or not root.as_posix().endswith(f"/research4/source_releases/{commit}")
            or authority.get("source_sha256") != embedded
            or hashlib.sha256(canonical_bytes).hexdigest() != embedded
            or not isinstance(files, list)):
        raise PermissionError("Independent partition source authority differs")
    indexed = {row.get("path"): row for row in files if isinstance(row, dict)}
    for logical in critical:
        row = indexed.get(logical); path = root / logical
        if not isinstance(row, dict) or not path.is_file() or path.is_symlink():
            raise PermissionError("Critical partition source is absent")
        data = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace(
            "\r", "\n").encode("utf-8")
        if (len(data) != row.get("canonical_bytes")
                or hashlib.sha256(data).hexdigest() != row.get("sha256")):
            raise PermissionError("Critical partition source changed before import")
    return embedded


bootstrap_source_sha256 = _bootstrap_verify_source(values["code_path"], intent_preview)
sys.path.insert(0, values["code_path"])

from src.databricks_continuity_partition import run_partition
from src.databricks_simulated_continuity import (
    GOLD_ADMISSION_TABLE, MODEL_RECEIPT, VOLUME, close_active_reservation,
    reserve_intent, validate_base_qualification_receipt, validate_intent,
    verify_deployed_source, verify_model_receipt_authority)
from src.qwen9b_ordinal_continuity_training import (
    assert_databricks_a10x1_fp16_runtime, assert_fast_deltanet_dependencies)

if intent_preview.get("plan", {}).get("mode") != dbutils.widgets.get("expected_mode"):
    raise PermissionError("Job resource does not admit this partition mode")
intent = validate_intent(intent_preview)
source_proof = verify_deployed_source(values["code_path"], intent)
if source_proof["manifest_sha256"] != bootstrap_source_sha256:
    raise PermissionError("Partition bootstrap and full source verification differ")
if not values["base_qualification_receipt_path"]:
    raise PermissionError("Training requires the frozen-base format gate")
base_qualification = validate_base_qualification_receipt(json.loads(Path(
    values["base_qualification_receipt_path"]).read_text(encoding="utf-8")))
if (base_qualification["receipt_sha256"]
        != intent["base_qualification_receipt_sha256"]
        or base_qualification["base_qualification_intent_sha256"]
           != intent["base_qualification_intent_sha256"]):
    raise PermissionError("Training intent and frozen-base gate differ")

# Provisioning and exact package installation have completed before this point.
# Exercise imports, kernels, CUDA, A10 identity, NVML and free VRAM before the
# immutable scientific identity is consumed.  No target model is loaded here.
dependency_proof = assert_fast_deltanet_dependencies()
import bitsandbytes, torch, pynvml
hardware_proof = assert_databricks_a10x1_fp16_runtime(torch)
probe = torch.zeros(256, dtype=torch.float16, device="cuda:0")
quantized, quantization_state = bitsandbytes.functional.quantize_4bit(
    probe, quant_type="nf4", compress_statistics=True)
restored = bitsandbytes.functional.dequantize_4bit(
    quantized, quantization_state)
if restored.shape != probe.shape or not torch.isfinite(restored).all():
    raise RuntimeError("bitsandbytes NF4 CUDA readiness failed")
del restored, quantized, quantization_state, probe
torch.cuda.synchronize(0); torch.cuda.empty_cache()
model_receipt = verify_model_receipt_authority(
    MODEL_RECEIPT,
    expected_tree_sha256=intent["model_tree_sha256"],
    expected_receipt_sha256=intent["model_receipt_sha256"])
required_disk = int(model_receipt.get("total_bytes", 0)) + 8 * 1024**3
local_disk_free = shutil.disk_usage(tempfile.gettempdir()).free
if required_disk <= 8 * 1024**3 or local_disk_free < required_disk:
    raise RuntimeError("Local model-copy and export disk readiness failed")
pynvml.nvmlInit()
try:
    if pynvml.nvmlDeviceGetCount() != 1:
        raise RuntimeError("GPU readiness requires exactly one NVML device")
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    if "A10" not in name or int(memory.free) < 20 * 1024**3:
        raise RuntimeError("GPU readiness lacks the reviewed A10/20 GiB free envelope")
    nvml_proof = {"name": name, "free_bytes": int(memory.free),
                  "total_bytes": int(memory.total)}
finally:
    pynvml.nvmlShutdown()

owner = dbutils.widgets.get("run_id")
if not owner:
    raise ValueError("GPU reservation requires the exact Databricks run owner")
if max(time.monotonic() - task_started_monotonic,
       time.time() - job_started_epoch) >= intent["wall_clock_stop_seconds"]:
    raise TimeoutError("Partition setup exhausted the reviewed envelope before reservation")
lock_root = Path(VOLUME) / "locks"
reserved = False
try:
    reserve_intent(lock_root, intent, owner)
    reserved = True
    spark.createDataFrame([{
        "identity": intent["identity"], "intent_sha256": intent["intent_sha256"],
        "job_run_id": owner, "status": "consumed_after_gpu_ready_before_model",
        "consumed_at": datetime.now(timezone.utc),
    }], schema=spark.table(GOLD_ADMISSION_TABLE).schema).write.mode(
        "append").saveAsTable(GOLD_ADMISSION_TABLE)
    result = run_partition(
        spark=spark, intent_path=values["intent_path"], plan_path=values["plan_path"],
        admission_path=values["admission_path"] or None,
        parent_checkpoint=values["parent_checkpoint"] or None,
        qualification_intent_path=values["qualification_intent_path"] or None,
        qualification_output=values["qualification_output"] or None,
        qualification_receipt_path=values["qualification_receipt_path"] or None,
        base_qualification_receipt_path=values[
            "base_qualification_receipt_path"] or None,
        code_root=values["code_path"], job_run_id=owner,
        task_started_monotonic=task_started_monotonic,
        job_started_epoch=job_started_epoch)
except BaseException:
    if reserved:
        close_active_reservation(
            lock_root, intent, owner, "partition_notebook_failed")
    raise
else:
    close_active_reservation(
        lock_root, intent, owner, "partition_notebook_complete")
result["gpu_readiness"] = {
    "source_manifest_sha256": source_proof["manifest_sha256"],
    "dependency_profile": dependency_proof["kernel_selection"]["profile"],
    "hardware_profile": hardware_proof["profile"],
    "nvml": nvml_proof,
    "bitsandbytes_version": bitsandbytes.__version__,
    "local_disk_free_bytes": int(local_disk_free),
    "local_disk_required_bytes": int(required_disk),
}
dbutils.notebook.exit(json.dumps(result))
