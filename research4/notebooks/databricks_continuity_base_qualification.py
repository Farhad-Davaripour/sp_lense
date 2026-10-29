# Databricks notebook source
"""Run the preregistered frozen Qwen format/legal-action qualification once."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time

task_started = time.monotonic()

for name in ("code_path", "intent_path", "output_path", "run_id",
             "job_start_timestamp_ms"):
    dbutils.widgets.text(name, "")
values = {name: dbutils.widgets.get(name) for name in (
    "code_path", "intent_path", "output_path", "run_id",
    "job_start_timestamp_ms")}
if any(not value for value in values.values()):
    raise ValueError("Base qualification requires exact code, intent, output and owner")
job_started_epoch = int(values["job_start_timestamp_ms"]) / 1000.0
if job_started_epoch <= 0 or job_started_epoch > time.time() + 5:
    raise ValueError("Base qualification job-start timestamp differs")


def _whole_job_elapsed():
    return max(time.monotonic() - task_started,
               time.time() - job_started_epoch)
raw_intent = json.loads(Path(values["intent_path"]).read_text(encoding="utf-8"))


def _canonical_bootstrap(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _bootstrap_verify_source(root_value, authority):
    """Verify the authority/lock/model-loader code before importing it."""
    root = Path(root_value).resolve()
    manifest = authority.get("source_manifest")
    if not isinstance(manifest, dict):
        raise PermissionError("Source bootstrap manifest is absent")
    payload = dict(manifest)
    embedded = payload.pop("manifest_sha256", None)
    commit = authority.get("source_commit")
    files = manifest.get("files")
    critical = (
        "src/__init__.py",
        "src/databricks_simulated_continuity.py",
        "src/databricks_continuity_partition.py",
        "src/qwen9b_ordinal_continuity_runtime.py",
        "src/qwen9b_ordinal_continuity_training.py",
        "src/qwen9b_ordinal_continuity_lora.py",
    )
    if (not re.fullmatch(r"[0-9a-f]{40}", str(commit or ""))
            or not root.as_posix().endswith(
                f"/research4/source_releases/{commit}")
            or authority.get("source_sha256") != embedded
            or embedded != hashlib.sha256(
                _canonical_bootstrap(payload).encode("utf-8")).hexdigest()
            or not isinstance(files, list)):
        raise PermissionError("Independent source bootstrap authority differs")
    by_path = {row.get("path"): row for row in files if isinstance(row, dict)}
    for logical in critical:
        row = by_path.get(logical)
        path = root / logical
        if (not isinstance(row, dict) or not path.is_file() or path.is_symlink()
                or set(row) != {"path", "sha256", "canonical_bytes"}):
            raise PermissionError("Critical source bootstrap file is absent")
        content = path.read_bytes().decode("utf-8")
        data = content.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        if (len(data) != row.get("canonical_bytes")
                or hashlib.sha256(data).hexdigest() != row.get("sha256")):
            raise PermissionError("Critical source changed before import")
    return embedded


bootstrap_source_sha256 = _bootstrap_verify_source(values["code_path"], raw_intent)
sys.path.insert(0, values["code_path"])

from pyspark.sql import functions as F
from src.databricks_continuity_partition import _stage_local_model
from src.databricks_simulated_continuity import (
    GOLD_ADMISSION_TABLE,
    GOLD_RUN_TABLE,
    MODEL_RECEIPT,
    SILVER_TABLE,
    VOLUME,
    canonical,
    close_active_reservation,
    create_base_qualification_receipt,
    file_sha256,
    reserve_intent,
    validate_base_qualification_silver_rows,
    validate_base_qualification_intent,
    verify_deployed_source,
    verify_model_receipt_authority,
)
from src.qwen9b_ordinal_continuity_lora import (
    model_component_sha256,
    trainable_inventory,
)
from src.qwen9b_ordinal_continuity_training import (
    assert_databricks_a10x1_fp16_runtime,
    assert_fast_deltanet_dependencies,
    assert_qwen35_backend,
    qwen35_t4x2_device_map,
)

intent = validate_base_qualification_intent(raw_intent)
def _require_before_operator_stop(phase):
    if _whole_job_elapsed() >= intent["wall_clock_stop_seconds"]:
        raise TimeoutError(f"Frozen-base qualification reached operator stop before {phase}")


def _require_before_ceiling(phase):
    if _whole_job_elapsed() >= intent["wall_clock_ceiling_seconds"]:
        raise TimeoutError(f"Frozen-base qualification reached hard ceiling before {phase}")


source = verify_deployed_source(values["code_path"], intent)
if source["manifest_sha256"] != bootstrap_source_sha256:
    raise PermissionError("Bootstrap and full source verification differ")
_require_before_operator_stop("model receipt authority verification")
model_receipt = verify_model_receipt_authority(
    MODEL_RECEIPT,
    expected_tree_sha256=intent["model_tree_sha256"],
    expected_receipt_sha256=intent["model_receipt_sha256"])
_require_before_operator_stop("Silver replay")
selected = spark.table(SILVER_TABLE).where(
    (F.col("source_commit") == intent["source_commit"])
    & (F.col("record_kind") == "base_qualification")
    & (F.col("split") == "qualification")
    & (F.col("admitted_for_training") == F.lit(False))).select(
        "source_commit", "dataset_tree_sha256", "record_id", "record_kind",
        "condition", "split", "family_id", "danger_level", "record_sha256",
        "record_json", "admitted_for_training", "dataset_release_path",
        "dataset_release_receipt_sha256").orderBy("record_id").collect()
validated_rows = validate_base_qualification_silver_rows(selected, intent)
records = [{**row["record"], "_silver_record_sha256": row["record_sha256"]}
           for row in validated_rows]
_require_before_operator_stop("GPU readiness")
output = Path(values["output_path"])
if output.resolve() != Path(intent["output_root"]).resolve():
    raise PermissionError("Frozen-base output path differs from reviewed intent")
if output.exists():
    raise FileExistsError("Frozen-base qualification output is immutable")

# Complete package/CUDA/A10/NF4 readiness before consuming the one-use identity.
dependency_proof = assert_fast_deltanet_dependencies()
import bitsandbytes
import mlflow
import pynvml
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3_5ForConditionalGeneration,
)
hardware = assert_databricks_a10x1_fp16_runtime(torch)
probe = torch.zeros(256, dtype=torch.float16, device="cuda:0")
quantized, quantization_state = bitsandbytes.functional.quantize_4bit(
    probe, quant_type="nf4", compress_statistics=True)
restored = bitsandbytes.functional.dequantize_4bit(quantized, quantization_state)
if restored.shape != probe.shape or not torch.isfinite(restored).all():
    raise RuntimeError("Frozen-base qualification NF4 readiness failed")
del restored, quantized, quantization_state, probe
torch.cuda.synchronize(0)
torch.cuda.empty_cache()
required_disk = int(model_receipt.get("total_bytes", 0)) + 8 * 1024**3
local_disk_free = shutil.disk_usage(tempfile.gettempdir()).free
if required_disk <= 8 * 1024**3 or local_disk_free < required_disk:
    raise RuntimeError("Base qualification lacks model-copy/export disk margin")
pynvml.nvmlInit()
if pynvml.nvmlDeviceGetCount() != 1:
    pynvml.nvmlShutdown()
    raise RuntimeError("Frozen-base qualification requires one NVML device")
nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
_require_before_operator_stop("one-use reservation")

owner = values["run_id"]
lock_root = Path(VOLUME) / "locks"
reserved = False
local_parent = None
mlflow_run = None
try:
    reserve_intent(lock_root, intent, owner)
    reserved = True
    spark.createDataFrame([{
        "identity": intent["identity"],
        "intent_sha256": intent["intent_sha256"],
        "job_run_id": owner,
        "status": "consumed_after_gpu_ready_before_model",
        "consumed_at": datetime.now(timezone.utc),
    }], schema=spark.table(GOLD_ADMISSION_TABLE).schema).write.mode(
        "append").saveAsTable(GOLD_ADMISSION_TABLE)
    local_model, local_receipt, copy_metrics = _stage_local_model(
        intent["model_root"], MODEL_RECEIPT, intent["identity"],
        intent["model_tree_sha256"], intent["model_receipt_sha256"])
    local_parent = local_model.parent
    _require_before_operator_stop("frozen-base model load")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    config = AutoConfig.from_pretrained(
        local_model, local_files_only=True, trust_remote_code=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        local_model,
        local_files_only=True,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map=qwen35_t4x2_device_map(config, 0),
        trust_remote_code=False,
    )
    backend = assert_qwen35_backend(model, dependency_proof)
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = True
    if trainable_inventory(model, allow_empty=True):
        raise PermissionError("Frozen-base qualification found trainable parameters")
    tokenizer = AutoTokenizer.from_pretrained(
        local_model, local_files_only=True, trust_remote_code=False, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_before = model_component_sha256(model)
    _require_before_operator_stop("frozen-base inference")
    mlflow.set_experiment(intent["mlflow_experiment_path"])
    mlflow_run = mlflow.start_run(run_name=intent["identity"])
    inference_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(0)
    outcomes = []
    utilization_samples = []
    memory_used_samples = []
    total_prompt_tokens = total_generated_tokens = 0
    batch_size = 16
    with torch.inference_mode():
        for offset in range(0, len(records), batch_size):
            _require_before_operator_stop("next inference batch")
            batch = records[offset:offset + batch_size]
            conversations = [[{"role": "user", "content": row["input_text"]}]
                             for row in batch]
            encoded = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                padding=True,
                return_tensors="pt",
                return_dict=True,
            )
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            if int(encoded["attention_mask"].sum(dim=1).max().item()) > int(
                    intent["plan"]["max_length"]):
                raise PermissionError("Base qualification prompt exceeds frozen context")
            prompt_width = encoded["input_ids"].shape[1]
            generated = model.generate(
                **encoded,
                max_new_tokens=intent["plan"]["max_new_tokens"],
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            utilization = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            utilization_samples.append(float(utilization.gpu))
            memory_used_samples.append(int(memory.used))
            completions = generated[:, prompt_width:]
            texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
            total_prompt_tokens += int(encoded["attention_mask"].sum().item())
            total_generated_tokens += sum(int((row != tokenizer.pad_token_id).sum().item())
                                          for row in completions)
            for row, text in zip(batch, texts, strict=True):
                strict_json = False
                action = None
                try:
                    parsed = json.loads(text)
                    strict_json = (isinstance(parsed, dict)
                                   and set(parsed) == {"action"}
                                   and isinstance(parsed["action"], str))
                    if strict_json:
                        action = parsed["action"]
                except json.JSONDecodeError:
                    pass
                outcomes.append({
                    "record_id": row["record_id"],
                    "record_sha256": row["_silver_record_sha256"],
                    "family_id": row["family_id"],
                    "strict_json": strict_json,
                    "legal_action": strict_json and action in row["allowed_actions"],
                    "target_match": strict_json and action == row["target_action_id"],
                    "action": action,
                    "allowed_actions": row["allowed_actions"],
                    "target_action_id": row["target_action_id"],
                    "output_text": text,
                    "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                })
    _require_before_operator_stop("protected evidence export")
    inference_elapsed = time.monotonic() - inference_started
    base_after = model_component_sha256(model)
    _require_before_ceiling("raw evidence export")
    elapsed = _whole_job_elapsed()
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "BASE_QUALIFICATION_RAW.jsonl"
    raw_path.write_text("".join(canonical(row) + "\n" for row in outcomes),
                        encoding="utf-8")
    receipt = create_base_qualification_receipt(
        intent,
        outcomes,
        raw_results_sha256=file_sha256(raw_path),
        base_component_sha256_before=base_before,
        base_component_sha256_after=base_after,
        resource_metrics={
            "gpu": hardware,
            "backend": backend,
            "model_copy": copy_metrics,
            "local_disk_free_before_copy": int(
                copy_metrics["local_disk_free_before"]),
            "elapsed_seconds": elapsed,
            "inference_seconds": inference_elapsed,
            "prompt_tokens": total_prompt_tokens,
            "generated_tokens": total_generated_tokens,
            "batch_size": batch_size,
            "decisions_per_second": len(outcomes) / max(inference_elapsed, 1e-9),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
            "nvml": {
                "sample_count": len(utilization_samples),
                "mean_utilization_percent": (
                    sum(utilization_samples) / len(utilization_samples)),
                "peak_memory_used_bytes": max(memory_used_samples),
            },
            "operator_stop_seconds": intent["wall_clock_stop_seconds"],
            "ceiling_seconds": intent["wall_clock_ceiling_seconds"],
            "protected_export_seconds": (
                intent["wall_clock_ceiling_seconds"]
                - intent["wall_clock_stop_seconds"]),
        },
    )
    receipt_path = output / "BASE_QUALIFICATION_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    stored_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    replayed_outcomes = [json.loads(line) for line in raw_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    replayed_receipt = create_base_qualification_receipt(
        intent,
        replayed_outcomes,
        raw_results_sha256=file_sha256(raw_path),
        base_component_sha256_before=base_before,
        base_component_sha256_after=base_after,
        resource_metrics=receipt["resources"],
    )
    if replayed_receipt != stored_receipt:
        raise PermissionError("Frozen-base receipt does not replay from raw output")
    receipt = stored_receipt
    mlflow_run_id = mlflow_run.info.run_id
    target_gold_status = "complete" if receipt["status"] == "qualified" else "failed"
    gold = [{
        "partition_identity": intent["identity"],
        "mode": "base_qualification",
        "condition": "frozen_base",
        "stage": "format_schema",
        "seed": 0,
        "status": "pending_mlflow_finalize",
        "mlflow_run_id": mlflow_run_id,
        "plan_sha256": intent["plan_sha256"],
        "source_commit": intent["source_commit"],
        "model_tree_sha256": intent["model_tree_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "intent_sha256": intent["intent_sha256"],
        "evidence_sha256": receipt["raw_results_sha256"],
        "admission_sha256": None,
        "qualification_receipt_sha256": receipt["receipt_sha256"],
        "optimizer_updates": 0,
        "checkpoint_tree_sha256": intent["model_tree_sha256"],
        "parent_checkpoint_tree_sha256": None,
        "receipt_replayed": True,
        "receipt_json": canonical(receipt),
        "completed_at": datetime.now(timezone.utc),
    }]
    _require_before_ceiling("pending Gold evidence publication")
    spark.createDataFrame(gold, schema=spark.table(GOLD_RUN_TABLE).schema).write.mode(
        "append").saveAsTable(GOLD_RUN_TABLE)
    _require_before_ceiling("MLflow finalization")
    mlflow.log_metrics({
        "strict_schema_and_legal_count": receipt["strict_schema_and_legal_count"],
        "target_match_count_secondary": receipt["target_match_count_secondary"],
        "decisions_per_second": receipt["resources"]["decisions_per_second"],
        "peak_gpu_memory_bytes": receipt["resources"]["peak_gpu_memory_bytes"],
        "mean_gpu_utilization_percent": receipt["resources"]["nvml"][
            "mean_utilization_percent"],
    })
    mlflow.log_dict(receipt, "base_qualification_receipt.json")
    mlflow.end_run(status="FINISHED" if receipt["status"] == "qualified" else "FAILED")
    mlflow_run = None
    _require_before_ceiling("Gold evidence finalization")
    spark.sql(
        f"UPDATE {GOLD_RUN_TABLE} SET status = '{target_gold_status}' "
        f"WHERE partition_identity = '{intent['identity']}' "
        f"AND intent_sha256 = '{intent['intent_sha256']}' "
        "AND status = 'pending_mlflow_finalize'"
    )
    finalized = spark.table(GOLD_RUN_TABLE).where(
        (F.col("partition_identity") == intent["identity"])
        & (F.col("intent_sha256") == intent["intent_sha256"])
        & (F.col("status") == target_gold_status)).count()
    if finalized != 1:
        raise PermissionError("Frozen-base Gold finalization was not exact")
    if receipt["status"] != "qualified":
        raise RuntimeError("Frozen base failed the preregistered format/legal gate")
finally:
    if mlflow_run is not None and mlflow.active_run() is not None:
        mlflow.end_run(status="FAILED")
    if local_parent is not None:
        shutil.rmtree(local_parent, ignore_errors=True)
    pynvml.nvmlShutdown()
    if reserved:
        close_active_reservation(
            lock_root, intent, owner, "base_qualification_notebook_terminal")

dbutils.notebook.exit(json.dumps({
    "status": "qualified",
    "identity": intent["identity"],
    "receipt_sha256": receipt["receipt_sha256"],
    "strict_schema_and_legal_count": receipt["strict_schema_and_legal_count"],
}))
