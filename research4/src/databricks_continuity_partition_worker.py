"""Single-A10 Databricks GPU worker for one reviewed continuity partition."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import os

from .databricks_simulated_continuity import (
    MODEL_ROOT, validate_intent, verify_model_receipt)
from .qwen9b_ordinal_continuity_training import (
    TrainingPlan, assert_fast_deltanet_dependencies, build_qualification_model,
    build_qualified_model, load_pinned_tokenizer, qualification_metrics_from_run,
    run_training_stage,
    )
from .qwen9b_ordinal_continuity_lora import (
    STAGE_BRANCHES, adapter_sha256, load_adapter_checkpoint,
    model_component_hashes)


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
    import torch
    rank = 0; local_rank = 0
    torch.cuda.set_device(local_rank)
    model = None
    import mlflow
    mlflow.set_experiment(intent["mlflow_experiment_path"])
    client = mlflow.tracking.MlflowClient()
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
                       != intent.get("base_qualification_carryforward_sha256")):
                raise PermissionError("Primary admission differs from this one-use intent")
            model, targets, proof = build_qualified_model(
                plan, admission, local_rank=local_rank,
                dependency_proof=dependency)
        tokenizer = load_pinned_tokenizer(plan)
        records = _jsonl(args.records)
        live_before = model_component_hashes(model, require_quantized=True)
        for key, value in {
            "partition_identity": intent["identity"],
            "plan_sha256": plan.full_plan_sha256,
            "model_revision": plan.model_revision,
            "optimizer_updates": plan.optimizer_updates,
            "rank": plan.rank, "alpha": plan.alpha,
            "source_manifest_sha256": intent["source_manifest"]["manifest_sha256"],
            "source_manifest_file_count": intent["source_manifest"]["file_count"],
        }.items():
            client.log_param(args.mlflow_run_id, key, value)
        def progress(update):
            step = update["optimizer_update"]
            for key in ("loss", "sft", "preference", "cross_danger_margin",
                        "level0_kl_anchor",
                        "gradient_norm_before_clip", "grad_scaler_before",
                        "grad_scaler_after"):
                if isinstance(update.get(key), (int, float)):
                    client.log_metric(args.mlflow_run_id, key, update[key], step=step)
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
            "version": "research4-databricks-partition-evidence-v1",
            "intent_sha256": intent["intent_sha256"],
            "base_qualification_carryforward_sha256": intent.get(
                "base_qualification_carryforward_sha256"),
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
        client.log_dict(args.mlflow_run_id, receipt, "training_receipt.json")
        gpu = receipt.get("resources", {}).get("gpu", {}).get("nvml", {})
        if "mean_utilization_percent" in gpu:
            client.log_metric(args.mlflow_run_id, "mean_gpu_utilization_percent",
                              gpu["mean_utilization_percent"])
        resources = receipt.get("resources", {})
        for key, value in {
            "tokens_per_second": resources.get("tokens_per_second", 0.0),
            "max_allocated_bytes": resources.get("gpu", {}).get(
                "max_allocated_bytes", 0),
            "max_reserved_bytes": resources.get("gpu", {}).get(
                "max_reserved_bytes", 0),
            "disk_free_bytes": resources.get("disk_free_bytes", 0),
        }.items():
            client.log_metric(args.mlflow_run_id, key, value)
        return receipt
    finally:
        model = None
        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("intent", "plan", "records", "output", "model-receipt",
                 "mlflow-run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--admission")
    parser.add_argument("--parent-checkpoint")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
