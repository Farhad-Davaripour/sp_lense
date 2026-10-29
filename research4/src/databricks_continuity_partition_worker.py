"""Single-A10 Databricks GPU worker for one reviewed continuity partition."""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import os

from .databricks_simulated_continuity import (
    MODEL_ROOT, validate_intent, verify_model_receipt)
from .qwen9b_ordinal_continuity_training import (
    DATABRICKS_A10_ALLOCATOR_CONF, TrainingPlan,
    assert_fast_deltanet_dependencies, build_qualification_model,
    build_qualified_model, load_pinned_tokenizer, qualification_metrics_from_run,
    run_training_stage,
    )
from .qwen9b_ordinal_continuity_lora import (
    STAGE_BRANCHES, adapter_sha256, load_adapter_checkpoint,
    model_component_hashes)


MLFLOW_JOURNAL_VERSION = "research4-parent-mlflow-journal-v1"
PROGRESS_METRICS = (
    "loss", "sft", "preference", "cross_danger_margin",
    "level0_kl_anchor", "gradient_norm_before_clip",
    "grad_scaler_before", "grad_scaler_after",
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic_json(path, value):
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f".pending-{target.name}-{os.urandom(8).hex()}")
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, target)


def _append_progress(path, update, *, intent_sha256, plan_sha256):
    row = {"version": MLFLOW_JOURNAL_VERSION,
           "intent_sha256": intent_sha256, "plan_sha256": plan_sha256,
           "optimizer_update": update["optimizer_update"]}
    for key in PROGRESS_METRICS:
        value = update.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise PermissionError(f"MLflow progress metric is invalid: {key}")
        row[key] = float(value)
    row["progress_record_sha256"] = _digest(row)
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(_canonical(row) + "\n"); stream.flush(); os.fsync(stream.fileno())


def validate_mlflow_journal(output, *, intent, plan, receipt, rank_receipt,
                            evidence):
    """Verify all child logging evidence before parent publication."""
    root = Path(output); params = _read(root / "MLFLOW_PARAMS.json")
    final = _read(root / "MLFLOW_FINAL.json")
    params_payload = dict(params); params_sha = params_payload.pop("params_sha256", None)
    expected_params = {
        "version": MLFLOW_JOURNAL_VERSION,
        "intent_sha256": intent["intent_sha256"],
        "partition_identity": intent["identity"],
        "plan_sha256": plan.full_plan_sha256,
        "model_revision": plan.model_revision,
        "optimizer_updates": plan.optimizer_updates,
        "rank": plan.rank, "alpha": plan.alpha,
        "source_manifest_sha256": intent["source_manifest"]["manifest_sha256"],
        "source_manifest_file_count": intent["source_manifest"]["file_count"],
        "technical_successor_sha256": intent.get(
            "qualification_technical_successor_sha256"),
    }
    if params_payload != expected_params or params_sha != _digest(params_payload):
        raise PermissionError("Parent MLflow parameter journal differs")
    progress_path = root / "MLFLOW_PROGRESS.jsonl"; progress_bytes = progress_path.read_bytes()
    rows = [json.loads(line) for line in progress_bytes.decode("utf-8").splitlines()
            if line.strip()]
    if (len(rows) != plan.optimizer_updates
            or [row.get("optimizer_update") for row in rows]
               != list(range(plan.optimizer_updates))):
        raise PermissionError("Parent MLflow progress count/order differs")
    rank_updates = rank_receipt.get("updates")
    if not isinstance(rank_updates, list) or len(rank_updates) != len(rows):
        raise PermissionError("Rank receipt progress differs")
    for row, update in zip(rows, rank_updates, strict=True):
        payload = dict(row); embedded = payload.pop("progress_record_sha256", None)
        expected_progress = {
            "version": MLFLOW_JOURNAL_VERSION,
            "intent_sha256": intent["intent_sha256"],
            "plan_sha256": plan.full_plan_sha256,
            "optimizer_update": update["optimizer_update"],
        }
        for key in PROGRESS_METRICS:
            value = update.get(key)
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))):
                raise PermissionError(
                    f"Rank receipt progress metric is invalid: {key}")
            expected_progress[key] = float(value)
            journal_value = payload.get(key)
            if (isinstance(journal_value, bool)
                    or not isinstance(journal_value, (int, float))
                    or not math.isfinite(float(journal_value))):
                raise PermissionError(
                    f"Parent MLflow progress metric is invalid: {key}")
        if (payload != expected_progress or embedded != _digest(payload)):
            raise PermissionError("Parent MLflow progress row differs")
    final_payload = dict(final); final_sha = final_payload.pop("final_sha256", None)
    resources = receipt.get("resources", {}); gpu = resources.get("gpu", {}).get("nvml", {})
    artifact_paths = ["DATABRICKS_EVIDENCE.json", "RANK_0_RECEIPT.json",
                      "TRAINING_RECEIPT.json", "checkpoint/adapter_config.json"]
    artifact_sha256 = {}
    for relative in artifact_paths:
        artifact = root / relative
        if not artifact.is_file():
            raise FileNotFoundError(f"Parent MLflow artifact is absent: {relative}")
        artifact_sha256[relative] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    expected_final = {
        "version": MLFLOW_JOURNAL_VERSION,
        "params_sha256": params["params_sha256"],
        "progress_sha256": hashlib.sha256(progress_bytes).hexdigest(),
        "progress_count": len(rows),
        "receipt_sha256": receipt["receipt_sha256"],
        "rank_receipt_sha256": rank_receipt["receipt_sha256"],
        "evidence_sha256": evidence["evidence_sha256"],
        "artifact_paths": artifact_paths,
        "artifact_sha256": artifact_sha256,
        "mean_gpu_utilization_percent": gpu.get("mean_utilization_percent"),
        "tokens_per_second": resources.get("tokens_per_second", 0.0),
        "max_allocated_bytes": resources.get("gpu", {}).get("max_allocated_bytes", 0),
        "max_reserved_bytes": resources.get("gpu", {}).get("max_reserved_bytes", 0),
        "disk_free_bytes": resources.get("disk_free_bytes", 0),
    }
    if final_payload != expected_final or final_sha != _digest(final_payload):
        raise PermissionError("Parent MLflow final journal differs")
    return {"params": params, "progress": rows, "final": final}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run(args):
    intent = validate_intent(_read(args.intent))
    plan = TrainingPlan(**_read(args.plan)).validate()
    if intent["plan"] != __import__("dataclasses").asdict(plan):
        raise PermissionError("Databricks intent and exact plan differ")
    local_model_root = os.environ.get("RESEARCH4_MODEL_ROOT")
    if not local_model_root:
        raise PermissionError("Per-run verified local model cache is required")
    model_receipt = verify_model_receipt(local_model_root, args.model_receipt)
    if (model_receipt["tree_sha256"] != intent["model_tree_sha256"]
            or model_receipt["receipt_sha256"] != intent["model_receipt_sha256"]):
        raise PermissionError("Intent model snapshot differs from the volume receipt")
    os.environ.update({
        "RESEARCH4_EXECUTION_PLATFORM": "databricks",
        "RESEARCH4_MODEL_ROOT": local_model_root,
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != DATABRICKS_A10_ALLOCATOR_CONF:
        raise PermissionError("Reviewed A10 CUDA allocator configuration differs")
    import torch
    rank = 0; local_rank = 0
    torch.cuda.set_device(local_rank)
    model = None
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    params = {
        "version": MLFLOW_JOURNAL_VERSION,
        "intent_sha256": intent["intent_sha256"],
        "partition_identity": intent["identity"],
        "plan_sha256": plan.full_plan_sha256,
        "model_revision": plan.model_revision,
        "optimizer_updates": plan.optimizer_updates,
        "rank": plan.rank, "alpha": plan.alpha,
        "source_manifest_sha256": intent["source_manifest"]["manifest_sha256"],
        "source_manifest_file_count": intent["source_manifest"]["file_count"],
        "technical_successor_sha256": intent.get(
            "qualification_technical_successor_sha256"),
    }
    params["params_sha256"] = _digest(params)
    _atomic_json(output / "MLFLOW_PARAMS.json", params)
    progress_path = output / "MLFLOW_PROGRESS.jsonl"
    descriptor = os.open(progress_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.fsync(descriptor)
    os.close(descriptor)
    try:
        dependency = assert_fast_deltanet_dependencies()
        admission = None if not args.admission else _read(args.admission)
        if plan.mode == "qualification":
            model, targets, proof = build_qualification_model(
                plan, local_rank=local_rank, dependency_proof=dependency)
        else:
            if admission is None:
                raise PermissionError("Primary Databricks partition requires admission")
            if (admission.get("source_sha256") != intent["source_sha256"]
                    or admission.get("one_use_submission_intent")
                       != intent["intent_sha256"]
                    or admission.get("data_binding_sha256")
                       != intent["admission_data_binding_sha256"]
                    or admission.get("exposure_binding_sha256")
                       != intent["admission_exposure_binding_sha256"]
                    or admission.get("base_qualification_carryforward_sha256")
                       != intent.get("base_qualification_carryforward_sha256")
                    or admission.get("qualification_technical_successor_sha256")
                       != intent.get("qualification_technical_successor_sha256")):
                raise PermissionError("Primary admission differs from this one-use intent")
            model, targets, proof = build_qualified_model(
                plan, admission, local_rank=local_rank,
                dependency_proof=dependency)
        tokenizer = load_pinned_tokenizer(plan)
        records = _jsonl(args.records)
        live_before = model_component_hashes(model, require_quantized=True)
        def progress(update):
            _append_progress(
                progress_path, update, intent_sha256=intent["intent_sha256"],
                plan_sha256=plan.full_plan_sha256)
        receipt = run_training_stage(
            model=model, tokenizer=tokenizer, tokenized_records=records,
            plan=plan,
            output_dir=args.output, target_names=targets,
            source_sha256=intent["source_sha256"],
            base_sha256=proof["base_component_sha256"],
            parent_checkpoint_dir=args.parent_checkpoint,
            rank=rank, world_size=1, save_checkpoint=True,
            progress_callback=progress)
        live_after = model_component_hashes(model, require_quantized=True)
        if live_after != live_before:
            raise PermissionError("Training changed frozen base/quantization/config bytes")
        branches = STAGE_BRANCHES[plan.stage]
        adapter_before = adapter_sha256(model, branches)
        metadata = load_adapter_checkpoint(
            model, Path(args.output) / "checkpoint", expected_targets=targets,
            expected_base_sha256=proof["base_component_sha256"],
            expected_source_sha256=intent["source_sha256"],
            expected_condition=plan.condition, expected_seed=plan.seed,
            expected_stage=plan.stage, expected_plan_sha256=plan.full_plan_sha256)
        save_reload_pass = adapter_sha256(model, branches) == adapter_before == metadata["adapter_sha256"]
        dispatch = receipt.get("fast_dispatch", {})
        metrics = qualification_metrics_from_run(
            receipt, plan=plan, backend_proof=proof["backend"],
            save_reload_pass=save_reload_pass,
            dispatch_forward_count=dispatch.get("forward_count", 0),
            dispatch_grad_forward_count=dispatch.get("grad_forward_count", 0),
            dispatch_backward_count=dispatch.get("backward_count", 0))
        if plan.mode == "qualification" and any(value is not True for key, value in metrics.items()
                if key.endswith("_pass") or key == "capacity_adequate"):
            raise PermissionError("Rank-4 qualification failed a raw evidence gate")
        evidence = {
            "version": "research4-databricks-partition-evidence-v2",
            "intent_sha256": intent["intent_sha256"],
            "base_qualification_carryforward_sha256": intent.get(
                "base_qualification_carryforward_sha256"),
            "qualification_technical_successor_sha256": intent.get(
                "qualification_technical_successor_sha256"),
            "plan_sha256": plan.full_plan_sha256,
            "rank_receipt_sha256": receipt["receipt_sha256"],
            "live_base_components_before": live_before,
            "live_base_components_after": live_after,
            "save_reload_pass": save_reload_pass,
            "qualification_metrics": metrics if plan.mode == "qualification" else None,
            "runtime_proof": proof,
        }
        evidence["evidence_sha256"] = __import__("hashlib").sha256(json.dumps(
            evidence, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        (Path(args.output) / "DATABRICKS_EVIDENCE.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True, default=str) + "\n")
        gpu = receipt.get("resources", {}).get("gpu", {}).get("nvml", {})
        resources = receipt.get("resources", {})
        rank_receipt = _read(Path(args.output) / "RANK_0_RECEIPT.json")
        final = {
            "version": MLFLOW_JOURNAL_VERSION,
            "params_sha256": params["params_sha256"],
            "progress_sha256": hashlib.sha256(progress_path.read_bytes()).hexdigest(),
            "progress_count": plan.optimizer_updates,
            "receipt_sha256": receipt["receipt_sha256"],
            "rank_receipt_sha256": rank_receipt["receipt_sha256"],
            "evidence_sha256": evidence["evidence_sha256"],
            "artifact_paths": ["DATABRICKS_EVIDENCE.json", "RANK_0_RECEIPT.json",
                               "TRAINING_RECEIPT.json", "checkpoint/adapter_config.json"],
            "artifact_sha256": {
                relative: hashlib.sha256((output / relative).read_bytes()).hexdigest()
                for relative in (
                    "DATABRICKS_EVIDENCE.json", "RANK_0_RECEIPT.json",
                    "TRAINING_RECEIPT.json", "checkpoint/adapter_config.json")
            },
            "mean_gpu_utilization_percent": gpu.get("mean_utilization_percent"),
            "tokens_per_second": resources.get("tokens_per_second", 0.0),
            "max_allocated_bytes": resources.get("gpu", {}).get(
                "max_allocated_bytes", 0),
            "max_reserved_bytes": resources.get("gpu", {}).get(
                "max_reserved_bytes", 0),
            "disk_free_bytes": resources.get("disk_free_bytes", 0),
        }
        final["final_sha256"] = _digest(final)
        _atomic_json(output / "MLFLOW_FINAL.json", final)
        validate_mlflow_journal(
            output, intent=intent, plan=plan, receipt=receipt,
            rank_receipt=rank_receipt, evidence=evidence)
        return receipt
    finally:
        model = None
        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("intent", "plan", "records", "output", "model-receipt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--admission")
    parser.add_argument("--parent-checkpoint")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
