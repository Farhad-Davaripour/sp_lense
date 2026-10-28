# Databricks notebook source
# MAGIC %uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 transformers==5.19.0 bitsandbytes==0.49.2 accelerate==1.15.0 safetensors==0.8.0 fla-core==0.5.2 einops==0.8.2 numpy==2.2.2 psutil==7.2.2 triton==3.6.0 nvidia-ml-py==13.615.71

# COMMAND ----------
from datetime import datetime, timezone
from pathlib import Path
import json, shutil, sys, tempfile

for name in ("code_path", "intent_path", "plan_path", "admission_path",
             "parent_checkpoint", "qualification_intent_path",
             "qualification_output", "qualification_receipt_path",
             "expected_mode"):
    dbutils.widgets.text(name, "")
values = {name: dbutils.widgets.get(name) for name in (
    "code_path", "intent_path", "plan_path", "admission_path", "parent_checkpoint",
    "qualification_intent_path", "qualification_output",
    "qualification_receipt_path")}
if not values["code_path"] or not values["intent_path"] or not values["plan_path"]:
    raise ValueError("Exact code, intent and plan paths are required")
sys.path.insert(0, values["code_path"])

from src.databricks_continuity_partition import run_partition
from src.databricks_simulated_continuity import (
    GOLD_ADMISSION_TABLE, MODEL_RECEIPT, VOLUME, close_active_reservation,
    reserve_intent, validate_intent, verify_deployed_source)
from src.qwen9b_ordinal_continuity_training import (
    assert_databricks_a10x1_fp16_runtime, assert_fast_deltanet_dependencies)

intent_preview = json.loads(Path(values["intent_path"]).read_text())
if intent_preview.get("plan", {}).get("mode") != dbutils.widgets.get("expected_mode"):
    raise PermissionError("Job resource does not admit this partition mode")
intent = validate_intent(intent_preview)
source_proof = verify_deployed_source(values["code_path"], intent)

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
model_receipt = json.loads(Path(MODEL_RECEIPT).read_text(encoding="utf-8"))
required_disk = int(model_receipt.get("total_bytes", 0)) + 8 * 1024**3
local_disk_free = shutil.disk_usage(tempfile.gettempdir()).free
if (model_receipt.get("receipt_sha256") != intent["model_receipt_sha256"]
        or required_disk <= 8 * 1024**3 or local_disk_free < required_disk):
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
        code_root=values["code_path"], job_run_id=owner)
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
