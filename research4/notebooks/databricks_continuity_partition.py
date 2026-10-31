# Databricks notebook source
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, re, shutil, subprocess, sys, tempfile, time

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


_CUDA_NF4_READINESS_MARKER = "RESEARCH4_CUDA_NF4_READINESS="
_CUDA_NF4_RECEIPT_MARKER = "RESEARCH4_CUDA_NF4_READINESS_RECEIPT="
_CUDA_NF4_RECEIPT_VERSION = "research4.cuda-nf4-readiness.v1"
_CUDA_NF4_BINDING_VERSION = "research4.cuda-nf4-readiness-binding.v1"
_CUDA_NF4_MAX_ATTEMPTS = 2
_CUDA_NF4_BACKOFF_SECONDS = 30.0
_CUDA_NF4_SUBPROCESS_TIMEOUT_SECONDS = 90.0
_CUDA_NF4_PHASE_CAP_SECONDS = 240.0
_CUDA_NF4_PRE_RESERVATION_MARGIN_SECONDS = 60.0
_CUDA_NF4_PROVIDER_STARTUP_ABORT_SECONDS = 1200.0
_CUDA_NF4_CHILD = r'''
import hashlib, json

MARKER = "RESEARCH4_CUDA_NF4_READINESS="

try:
    import bitsandbytes
    import torch
    from src.qwen9b_ordinal_continuity_training import (
        assert_databricks_a10x1_fp16_runtime,
        assert_fast_deltanet_dependencies,
    )

    dependency_proof = assert_fast_deltanet_dependencies()
    hardware = assert_databricks_a10x1_fp16_runtime(torch)
    probe = torch.zeros(256, dtype=torch.float16, device="cuda:0")
    quantized, quantization_state = bitsandbytes.functional.quantize_4bit(
        probe, quant_type="nf4", compress_statistics=True)
    restored = bitsandbytes.functional.dequantize_4bit(
        quantized, quantization_state)
    if restored.shape != probe.shape or not torch.isfinite(restored).all():
        raise RuntimeError("bitsandbytes NF4 CUDA readiness failed")
    torch.cuda.synchronize(0)
    payload = {
        "bitsandbytes_version": bitsandbytes.__version__,
        "dependency_profile": dependency_proof["kernel_selection"]["profile"],
        "finite": True,
        "hardware": hardware,
        "probe_elements": 256,
        "quant_type": "nf4",
        "status": "ready",
    }
    print(MARKER + json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        flush=True)
except BaseException as exc:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    retryable_token = (
        "cudaErrorDevicesUnavailable"
        if "cudaerrordevicesunavailable" in lowered
        else "cuda_devices_busy_or_unavailable_phrase"
        if "cuda-capable device(s) is/are busy or unavailable" in lowered
        else None)
    transient = type(exc).__name__ == "AcceleratorError" and retryable_token is not None
    payload = {
        "error_kind": (
            "cuda_devices_unavailable" if transient else "fatal_cuda_readiness"),
        "error_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "error_type": type(exc).__name__,
        "retryable_token": retryable_token,
        "status": "transient_device_unavailable" if transient else "fatal",
    }
    print(MARKER + json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        flush=True)
    raise SystemExit(75 if transient else 1)
'''


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _whole_job_elapsed():
    return max(time.monotonic() - task_started_monotonic,
               time.time() - job_started_epoch)


def _read_cuda_nf4_marker(stdout):
    rows = [row[len(_CUDA_NF4_READINESS_MARKER):]
            for row in stdout.splitlines()
            if row.startswith(_CUDA_NF4_READINESS_MARKER)]
    if len(rows) != 1:
        raise RuntimeError("CUDA readiness child marker is absent or ambiguous")
    try:
        payload = json.loads(rows[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError("CUDA readiness child marker is malformed") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("CUDA readiness child payload is not an object")
    return payload


def _bounded_cuda_nf4_readiness(
        *, code_root, binding, elapsed_seconds, stop_seconds,
        python_executable=sys.executable, runner=subprocess.run,
        sleeper=time.sleep, monotonic=time.monotonic, receipt_emitter=None):
    """Wait briefly for one usable A10 before scientific identity consumption."""
    binding_keys = {
        "version", "identity", "intent_sha256", "plan_sha256",
        "source_manifest_sha256", "job_run_id", "native_parent_start_epoch_ms"}
    if (not isinstance(binding, dict) or set(binding) != binding_keys
            or binding.get("version") != _CUDA_NF4_BINDING_VERSION
            or not isinstance(binding.get("identity"), str)
            or not binding["identity"]
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(binding.get(key, "")))
                   for key in ("intent_sha256", "plan_sha256",
                               "source_manifest_sha256"))
            or not re.fullmatch(r"[1-9][0-9]*", str(binding.get("job_run_id", "")))
            or type(binding.get("native_parent_start_epoch_ms")) is not int
            or binding["native_parent_start_epoch_ms"] <= 0):
        raise PermissionError("CUDA readiness execution binding is invalid")
    binding = dict(binding)
    binding_sha256 = _sha256_text(_canonical(binding))
    phase_start = monotonic()
    attempts = []
    environment = dict(os.environ)
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(Path(code_root).resolve()) + (
        os.pathsep + old_pythonpath if old_pythonpath else "")
    command = [python_executable, "-c", _CUDA_NF4_CHILD]

    def emit(receipt):
        if receipt_emitter is None:
            print(_CUDA_NF4_RECEIPT_MARKER + _canonical(receipt), flush=True)
        else:
            receipt_emitter(receipt)

    def abort(error_type, message, receipt, cause=None):
        emit(receipt)
        error = error_type(
            f"{message}; readiness_receipt={_canonical(receipt)}")
        if cause is None:
            raise error
        raise error from cause

    def finish(status, **extra):
        receipt = {
            "attempts": attempts,
            "backoff_seconds": _CUDA_NF4_BACKOFF_SECONDS,
            "backward_passes": 0,
            "binding": binding,
            "binding_sha256": binding_sha256,
            "child_source_sha256": _sha256_text(_CUDA_NF4_CHILD),
            "max_attempts": _CUDA_NF4_MAX_ATTEMPTS,
            "model_forwards": 0,
            "model_loads": 0,
            "optimizer_updates": 0,
            "phase_cap_seconds": _CUDA_NF4_PHASE_CAP_SECONDS,
            "pre_reservation_margin_seconds": (
                _CUDA_NF4_PRE_RESERVATION_MARGIN_SECONDS),
            "provider_startup_abort_seconds": (
                _CUDA_NF4_PROVIDER_STARTUP_ABORT_SECONDS),
            "status": status,
            "stop_seconds": float(stop_seconds),
            "subprocess_timeout_seconds": _CUDA_NF4_SUBPROCESS_TIMEOUT_SECONDS,
            "version": _CUDA_NF4_RECEIPT_VERSION,
            **extra,
        }
        receipt["receipt_sha256"] = _sha256_text(_canonical(receipt))
        return receipt

    for attempt_number in range(1, _CUDA_NF4_MAX_ATTEMPTS + 1):
        delay = 0.0 if attempt_number == 1 else _CUDA_NF4_BACKOFF_SECONDS
        required = (delay + _CUDA_NF4_SUBPROCESS_TIMEOUT_SECONDS
                    + _CUDA_NF4_PRE_RESERVATION_MARGIN_SECONDS)
        if (monotonic() - phase_start + required
                > _CUDA_NF4_PHASE_CAP_SECONDS
                or elapsed_seconds() + required >= stop_seconds):
            receipt = finish("deadline_rejected")
            abort(TimeoutError,
                  "CUDA readiness lacks a safe pre-reservation window", receipt)
        if delay:
            sleeper(delay)
        if (monotonic() - phase_start + _CUDA_NF4_SUBPROCESS_TIMEOUT_SECONDS
                > _CUDA_NF4_PHASE_CAP_SECONDS
                or (elapsed_seconds() + _CUDA_NF4_SUBPROCESS_TIMEOUT_SECONDS
                    + _CUDA_NF4_PRE_RESERVATION_MARGIN_SECONDS)
                >= stop_seconds):
            receipt = finish("deadline_rejected")
            abort(TimeoutError,
                  "CUDA readiness window expired before the next child", receipt)
        started = monotonic()
        try:
            completed = runner(
                command, capture_output=True, text=True, check=False,
                timeout=_CUDA_NF4_SUBPROCESS_TIMEOUT_SECONDS,
                env=environment)
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout.decode("utf-8", errors="replace")
                      if isinstance(exc.stdout, bytes) else str(exc.stdout or ""))
            stderr = (exc.stderr.decode("utf-8", errors="replace")
                      if isinstance(exc.stderr, bytes) else str(exc.stderr or ""))
            attempts.append({
                "attempt": attempt_number,
                "delay_seconds": delay,
                "elapsed_seconds": round(monotonic() - started, 6),
                "status": "timeout",
                "stderr_sha256": _sha256_text(stderr),
                "stdout_sha256": _sha256_text(stdout),
            })
            receipt = finish("fatal")
            abort(RuntimeError, "CUDA readiness child timed out", receipt, exc)
        except OSError as exc:
            attempts.append({
                "attempt": attempt_number,
                "delay_seconds": delay,
                "elapsed_seconds": round(monotonic() - started, 6),
                "error_sha256": _sha256_text(
                    f"{type(exc).__name__}: {exc}"),
                "error_type": type(exc).__name__,
                "status": "runner_error",
            })
            receipt = finish("fatal")
            abort(RuntimeError, "CUDA readiness child could not start", receipt, exc)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        raw_attempt = {
            "attempt": attempt_number,
            "delay_seconds": delay,
            "elapsed_seconds": round(monotonic() - started, 6),
            "returncode": int(completed.returncode),
            "stderr_sha256": _sha256_text(stderr),
            "stdout_sha256": _sha256_text(stdout),
        }
        try:
            payload = _read_cuda_nf4_marker(stdout)
        except RuntimeError as exc:
            raw_attempt["status"] = "malformed_output"
            attempts.append(raw_attempt)
            receipt = finish("fatal")
            abort(RuntimeError,
                  "CUDA readiness child output is invalid", receipt, exc)
        raw_attempt.update({
            "child_payload_sha256": _sha256_text(_canonical(payload)),
            "status": payload.get("status"),
        })
        attempts.append(raw_attempt)
        ready_keys = {
            "bitsandbytes_version", "dependency_profile", "finite", "hardware",
            "probe_elements", "quant_type", "status",
        }
        hardware = payload.get("hardware")
        devices = hardware.get("devices") if isinstance(hardware, dict) else None
        device = devices[0] if isinstance(devices, list) and len(devices) == 1 else None
        hardware_exact = (
            isinstance(hardware, dict)
            and set(hardware) == {
                "profile", "compute_dtype", "adapter_dtype", "devices"}
            and hardware.get("profile") == "databricks-a10x1"
            and hardware.get("compute_dtype") == "float16"
            and hardware.get("adapter_dtype") == "float32"
            and isinstance(device, dict)
            and set(device) == {
                "index", "name", "total_memory_bytes", "major", "minor"}
            and device.get("index") == 0
            and "A10" in str(device.get("name", "")).upper()
            and type(device.get("total_memory_bytes")) is int
            and device["total_memory_bytes"] >= 20 * 1024**3
            and type(device.get("major")) is int
            and type(device.get("minor")) is int)
        if (completed.returncode == 0 and set(payload) == ready_keys
                and payload.get("bitsandbytes_version") == "0.49.2"
                and payload.get("dependency_profile") == "fla"
                and payload.get("finite") is True
                and payload.get("probe_elements") == 256
                and payload.get("quant_type") == "nf4"
                and payload.get("status") == "ready"
                and hardware_exact):
            receipt = finish(
                "ready", hardware=payload["hardware"],
                bitsandbytes_version=payload["bitsandbytes_version"],
                dependency_profile=payload["dependency_profile"])
            emit(receipt)
            return receipt
        transient_keys = {
            "error_kind", "error_sha256", "error_type", "retryable_token",
            "status"}
        exact_transient = (
            completed.returncode == 75
            and set(payload) == transient_keys
            and payload.get("status") == "transient_device_unavailable"
            and payload.get("error_kind") == "cuda_devices_unavailable"
            and payload.get("error_type") == "AcceleratorError"
            and payload.get("retryable_token") in {
                "cudaErrorDevicesUnavailable",
                "cuda_devices_busy_or_unavailable_phrase"}
            and re.fullmatch(r"[0-9a-f]{64}", str(payload.get("error_sha256", "")))
        )
        if not exact_transient:
            receipt = finish("fatal")
            abort(RuntimeError,
                  "CUDA readiness failed with a non-retryable result", receipt)
    receipt = finish("transient_exhausted")
    abort(RuntimeError,
          "CUDA device remained unavailable after bounded readiness probes", receipt)


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
        "src/databricks_continuity_partition_worker.py",
        "src/qwen9b_ordinal_continuity_runtime.py",
        "src/qwen9b_ordinal_continuity_training.py",
        "src/qwen9b_ordinal_continuity_lora.py",
        "src/qwen9b_ordinal_continuity_worker.py",
        "src/volume_durable_io.py",
    )
    if (not re.fullmatch(r"[0-9a-f]{40}", str(commit or ""))
            or root.as_posix() != f"/Workspace/Shared/research4/source_releases/{commit}"
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
    verify_deployed_qualification_source_carryforward,
    verify_deployed_source, verify_intent_base_qualification_authority,
    verify_model_receipt_authority)

if intent_preview.get("plan", {}).get("mode") != dbutils.widgets.get("expected_mode"):
    raise PermissionError("Job resource does not admit this partition mode")
intent = validate_intent(intent_preview)
source_proof = verify_deployed_source(values["code_path"], intent)
if source_proof["manifest_sha256"] != bootstrap_source_sha256:
    raise PermissionError("Partition bootstrap and full source verification differ")
verify_deployed_qualification_source_carryforward(intent, values["code_path"])
if not values["base_qualification_receipt_path"]:
    raise PermissionError("Training requires the frozen-base format gate")
base_qualification = validate_base_qualification_receipt(json.loads(Path(
    values["base_qualification_receipt_path"]).read_text(encoding="utf-8")))
verify_intent_base_qualification_authority(
    intent, base_qualification, values["code_path"])
owner = dbutils.widgets.get("run_id")
if not owner:
    raise ValueError("GPU reservation requires the exact Databricks run owner")
readiness_binding = {
    "version": _CUDA_NF4_BINDING_VERSION,
    "identity": intent["identity"],
    "intent_sha256": intent["intent_sha256"],
    "plan_sha256": intent["plan_sha256"],
    "source_manifest_sha256": source_proof["manifest_sha256"],
    "job_run_id": str(owner),
    "native_parent_start_epoch_ms": int(values["job_start_timestamp_ms"]),
}

# Provisioning and exact package installation have completed before this point.
# Exercise imports, kernels, CUDA, A10 identity, NVML and free VRAM before the
# immutable scientific identity is consumed.  No target model is loaded here.
import pynvml, torch as parent_torch
if parent_torch.cuda.is_initialized():
    raise RuntimeError("Parent CUDA context initialized before readiness child")
cuda_readiness = _bounded_cuda_nf4_readiness(
    code_root=values["code_path"], binding=readiness_binding,
    elapsed_seconds=_whole_job_elapsed,
    stop_seconds=min(intent["wall_clock_stop_seconds"],
                     _CUDA_NF4_PROVIDER_STARTUP_ABORT_SECONDS))
if parent_torch.cuda.is_initialized():
    raise RuntimeError("Readiness child initialized the parent CUDA context")
hardware_proof = cuda_readiness["hardware"]
bitsandbytes_version = cuda_readiness["bitsandbytes_version"]
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

if _whole_job_elapsed() >= intent["wall_clock_stop_seconds"]:
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
    "dependency_profile": cuda_readiness["dependency_profile"],
    "hardware_profile": hardware_proof["profile"],
    "nvml": nvml_proof,
    "bitsandbytes_version": bitsandbytes_version,
    "cuda_nf4_readiness_receipt_sha256": cuda_readiness["receipt_sha256"],
    "cuda_nf4_readiness_attempts": len(cuda_readiness["attempts"]),
    "local_disk_free_bytes": int(local_disk_free),
    "local_disk_required_bytes": int(required_disk),
}
dbutils.notebook.exit(json.dumps(result))
