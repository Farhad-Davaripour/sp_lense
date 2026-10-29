"""Exactly-once provider worker for ordinal simulated-continuity training.

The worker has no provider API.  A reviewed controller must place one immutable
intent and start it once.  ``torchrun --nproc_per_node=2`` supplies the two DDP
ranks; any failure writes partial evidence and exits without retrying.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import gc
import json
import os
import shutil
import traceback

from .qwen9b_ordinal_continuity_training import (
    TrainingPlan,
    _atomic_json,
    assert_fast_deltanet_dependencies,
    build_qualification_model,
    build_qualified_model,
    canonical_sha256,
    equalize_matched_prompt_exposure,
    file_sha256,
    load_pinned_tokenizer,
    load_simulator_condition,
    qualification_metrics_from_run,
    run_training_stage,
    SerializedRunGuard,
    tokenize_rows,
    validate_equal_occurrence_exposure,
    validate_matched_conditions,
    verify_dataset_receipt,
    CONDITIONS,
)
from .qwen9b_ordinal_continuity_lora import (
    STAGE_BRANCHES,
    adapter_sha256,
    load_adapter_checkpoint,
    model_component_hashes,
)
from .qwen9b_ordinal_continuity_runtime import (
    cli_sha256,
    directory_tree_sha256,
    finalize_receipt,
    verify_package_manifest,
    verify_tokenizer_receipt,
)


WORKER_VERSION = "qwen35-9b-ordinal-continuity-worker-v7"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def consume_one_use_intent(intent_path, output_root: Path, plan: TrainingPlan, *,
                           expected: dict | None = None) -> dict:
    intent_path = Path(intent_path)
    intent = _read(intent_path)
    if (intent.get("version") != "qwen35-9b-continuity-one-use-intent-v2"
            or intent.get("one_use") is not True
            or intent.get("automatic_retries") != 0
            or intent.get("attempt_number") != 1
            or intent.get("plan_sha256") != plan.full_plan_sha256
            or Path(intent.get("output_root", "")).resolve() != output_root.resolve()):
        raise PermissionError("One-use intent differs from this exact run")
    if expected is not None and any(intent.get(key) != value
                                    for key, value in expected.items()):
        raise PermissionError("One-use intent runtime bindings differ")
    intent_sha = file_sha256(intent_path)
    consumed = output_root.parent / (output_root.name + ".INTENT_CONSUMED.json")
    if consumed.exists():
        raise PermissionError("This one-use intent was already consumed")
    descriptor = os.open(consumed, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    payload = {"worker_version": WORKER_VERSION, "intent_sha256": intent_sha,
               "plan_sha256": plan.full_plan_sha256, "pid": os.getpid()}
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {**payload, "consumed_receipt": str(consumed)}


def _load_data(generated_root, plan, tokenizer):
    receipt = verify_dataset_receipt(generated_root)
    rows_by_condition = {condition: load_simulator_condition(generated_root, condition)
                         for condition in CONDITIONS}
    binding = validate_matched_conditions(rows_by_condition)
    tokenized = {condition: tokenize_rows(rows, tokenizer, plan.max_length)
                 for condition, rows in rows_by_condition.items()}
    tokenized = equalize_matched_prompt_exposure(
        tokenized, tokenizer, plan.max_length)
    exposure = validate_equal_occurrence_exposure(tokenized, plan)
    return tokenized[plan.condition], binding, exposure, receipt


def prepare_runtime_bindings(generated_root, plan: TrainingPlan, tokenizer_root,
                             tokenizer_receipt) -> dict:
    """Materialize the exact CPU/tokenizer bindings used before model loading.

    The controller calls this function when it creates admissions and every
    worker calls it again from the immutable package.  Consequently a changed
    tokenizer file, source dataset, tokenization result, or occurrence plan is
    rejected before Qwen weights are loaded.
    """
    plan.validate()
    tokenizer_proof = verify_tokenizer_receipt(
        tokenizer_root, tokenizer_receipt,
        model_id=plan.model_id, revision=plan.model_revision)
    tokenizer = load_pinned_tokenizer(
        plan, tokenizer_root=tokenizer_root, tokenizer_receipt=tokenizer_receipt)
    if (tokenizer.__class__.__name__ != tokenizer_proof.get("tokenizer_class")
            or tokenizer.eos_token_id != tokenizer_proof.get("eos_token_id")
            or tokenizer.pad_token_id != tokenizer_proof.get("pad_token_id")):
        raise PermissionError("Loaded local tokenizer differs from its durable receipt")
    (records, data_binding, exposure_binding, dataset_receipt) = _load_data(
        generated_root, plan, tokenizer)
    if not isinstance(dataset_receipt.get("receipt_sha256"), str):
        raise PermissionError("Dataset receipt is not self-bound")
    return {
        "tokenizer": tokenizer,
        "tokenizer_receipt": tokenizer_proof,
        "records": records,
        "data_binding": data_binding,
        "exposure_binding": exposure_binding,
        "dataset_receipt": dataset_receipt,
    }


def _verify_admission(path, plan, *, expected):
    admission = _read(path)
    check = dict(admission)
    embedded = check.pop("admission_sha256", None)
    if (canonical_sha256(check) != embedded or admission.get("admitted") is not True
            or admission.get("plan") != asdict(plan)
            or admission.get("full_plan_sha256") != plan.full_plan_sha256
            or admission.get("provider_submission_performed") is not False
            or any(admission.get(key) != value for key, value in expected.items())):
        raise PermissionError("Primary admission differs or was already consumed")
    return admission


def _prepare_pre_model_runtime(args, plan: TrainingPlan) -> dict:
    """Verify every immutable launch/input binding before model construction."""
    package = verify_package_manifest(
        args.package_root, args.package_manifest,
        expected_tree_sha256=args.package_tree_sha256,
        expected_source_sha256=args.source_sha256)
    bindings = prepare_runtime_bindings(
        args.generated_root, plan, args.tokenizer_root, args.tokenizer_receipt)
    actual_cli_hash = cli_sha256(args.actual_worker_argv)
    parent_tree = (None if args.parent_checkpoint is None else
                   directory_tree_sha256(args.parent_checkpoint))
    expected_intent = {
        "worker_cli_sha256": actual_cli_hash,
        "data_binding_sha256": canonical_sha256(bindings["data_binding"]),
        "exposure_binding_sha256": canonical_sha256(bindings["exposure_binding"]),
        "dataset_receipt_sha256": bindings["dataset_receipt"]["receipt_sha256"],
        "source_sha256": args.source_sha256,
        "package_tree_sha256": package["verified_tree_sha256"],
        "package_manifest_sha256": package["manifest_file_sha256"],
        "tokenizer_receipt_sha256": bindings["tokenizer_receipt"]["receipt_sha256"],
        "parent_checkpoint_tree_sha256": parent_tree,
    }
    intent_document = _read(args.intent)
    if any(intent_document.get(key) != value for key, value in expected_intent.items()):
        raise PermissionError("Worker launch/input bindings differ from one-use intent")
    intent_file_sha = file_sha256(args.intent)
    admission = None
    if plan.mode == "primary":
        if not args.admission:
            raise PermissionError("Primary runtime has no exact admission")
        admission = _verify_admission(args.admission, plan, expected={
            **expected_intent,
            "one_use_submission_intent": intent_file_sha,
        })
    elif args.admission:
        raise PermissionError("Qualification must not consume a primary admission")
    summary = {
        **expected_intent,
        "intent_file_sha256": intent_file_sha,
        "admission_file_sha256": (
            None if args.admission is None else file_sha256(args.admission)),
        "plan_sha256": plan.full_plan_sha256,
    }
    return {"package": package, "bindings": bindings,
            "admission": admission, "summary": summary}


def run(args):
    plan = TrainingPlan(**_read(args.plan)).validate()
    output = Path(args.output).resolve()
    import torch
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    # Keep every large CPU payload in an explicitly releasable slot.  A policy
    # partition can contain thousands of token lists; retaining ``prepared``
    # after training otherwise keeps the tokenizer and every tokenized record
    # alive until torchrun exits.
    prepared = None
    bindings = None
    tokenizer = None
    records = None
    data_binding = None
    exposure_binding = None
    dataset_receipt = None
    admission = None
    local_preflight = [None]
    try:
        prepared = _prepare_pre_model_runtime(args, plan)
        local_preflight[0] = {"ok": True, "summary": prepared["summary"]}
    except BaseException as exc:
        prepared = None
        local_preflight[0] = {
            "ok": False, "type": type(exc).__name__, "error": str(exc)}
    gathered = [None for _ in range(plan.world_size)]
    torch.distributed.all_gather_object(gathered, local_preflight[0])
    if (any(item.get("ok") is not True for item in gathered)
            or any(item.get("summary") != gathered[0].get("summary")
                   for item in gathered[1:])):
        raise PermissionError(f"Pre-model runtime binding failed across ranks: {gathered}")
    box = [None]
    if rank == 0:
        try:
            if output.exists():
                raise FileExistsError("Output is immutable; this worker never retries")
            intent = consume_one_use_intent(
                args.intent, output, plan,
                expected={key: value for key, value in prepared["summary"].items()
                          if key in _read(args.intent)})
            output.mkdir(parents=True, exist_ok=False)
            box[0] = {"ok": True, "intent": intent}
        except BaseException as exc:
            box[0] = {"ok": False, "type": type(exc).__name__, "error": str(exc)}
    torch.distributed.broadcast_object_list(box, src=0)
    if not box[0]["ok"]:
        raise RuntimeError(f"Rank-0 one-use admission failed: {box[0]}")
    intent = box[0]["intent"]
    lock_path = Path(args.lock).resolve()
    running = {"version": WORKER_VERSION, "status": "running", "rank": rank,
               "plan": asdict(plan), "plan_sha256": plan.full_plan_sha256,
               "intent": intent, "automatic_retries": 0,
               "pre_model_runtime_bindings": prepared["summary"]}
    _atomic_json(output / f"RANK_{rank}_RUNNING.json", running)
    model = None
    completed = False
    try:
        with SerializedRunGuard(lock_path.with_name(lock_path.name + f".rank{rank}"),
                                intent_sha256=intent["intent_sha256"]):
            dependencies = assert_fast_deltanet_dependencies()
            bindings = prepared["bindings"]
            tokenizer = bindings["tokenizer"]
            records = bindings["records"]
            data_binding = bindings["data_binding"]
            exposure_binding = bindings["exposure_binding"]
            dataset_receipt = bindings["dataset_receipt"]
            if plan.mode == "primary":
                admission = prepared["admission"]
                model, targets, proof = build_qualified_model(
                    plan, admission, local_rank=local_rank,
                    dependency_proof=dependencies)
            else:
                model, targets, proof = build_qualification_model(
                    plan, local_rank=local_rank,
                    dependency_proof=dependencies)
            live_before = model_component_hashes(model)
            if live_before != proof.get("live_base_components"):
                raise PermissionError("Immediate live frozen/NF4 hashes differ from model proof")
            if (plan.mode == "primary"
                    and live_before != admission.get("qualified_live_base_components")):
                raise PermissionError("Primary live frozen/NF4 hashes differ from qualification")
            receipt = run_training_stage(
                model=model, tokenizer=tokenizer, tokenized_records=records,
                plan=plan, output_dir=output, target_names=targets,
                source_sha256=args.source_sha256,
                base_sha256=proof["base_component_sha256"],
                parent_checkpoint_dir=args.parent_checkpoint,
                rank=rank, world_size=plan.world_size, save_checkpoint=True)
            live_after = model_component_hashes(model)
            if live_after != live_before:
                raise PermissionError("Training changed a frozen parameter, NF4 state, buffer, or config")
            expected_journal = exposure_binding.get(
                "global_journal_sha256_by_condition", {}).get(plan.condition)
            expected_count = plan.occurrence_count
            if (not isinstance(expected_journal, str)
                    or receipt.get("global_journal_sha256") != expected_journal
                    or receipt.get("global_journal_occurrence_count") != expected_count):
                raise PermissionError("Executed occurrence journal differs from admission exposure")
            receipt["data_binding_sha256"] = canonical_sha256(data_binding)
            receipt["exposure_binding_sha256"] = canonical_sha256(exposure_binding)
            receipt["dataset_receipt"] = dataset_receipt
            receipt["runtime_proof"] = proof
            receipt["dependency_proof"] = dependencies
            receipt["pre_model_runtime_bindings"] = prepared["summary"]
            receipt["live_base_components_before"] = live_before
            receipt["live_base_components_after"] = live_after
            receipt["base_frozen_live_hash_pass"] = True
            if rank == 0:
                # Reload actual checkpoint bytes into the live model.  This is
                # required for both qualification and every primary stage.
                branches = STAGE_BRANCHES[plan.stage]
                adapter_before_reload = adapter_sha256(model, branches)
                metadata = load_adapter_checkpoint(
                    model, output / "checkpoint", expected_targets=targets,
                    expected_base_sha256=proof["base_component_sha256"],
                    expected_source_sha256=args.source_sha256,
                    expected_condition=plan.condition, expected_seed=plan.seed,
                    expected_stage=plan.stage,
                    expected_plan_sha256=plan.full_plan_sha256)
                adapter_after_reload = adapter_sha256(model, branches)
                reload_pass = (
                    adapter_before_reload == adapter_after_reload
                    == metadata.get("adapter_sha256")
                    and model_component_hashes(model) == live_before)
                if not reload_pass:
                    raise PermissionError("Live checkpoint save/reload proof failed")
                receipt["checkpoint_save_reload"] = {
                    "pass": True,
                    "checkpoint_tree_sha256": directory_tree_sha256(
                        output / "checkpoint"),
                    "checkpoint_metadata_sha256": file_sha256(
                        output / "checkpoint" / "adapter_config.json"),
                    "adapter_before_reload_sha256": adapter_before_reload,
                    "adapter_after_reload_sha256": adapter_after_reload,
                    "live_base_components_after_reload": model_component_hashes(model),
                }
                receipt["primary_save_reload_pass"] = plan.mode == "primary"
            if plan.mode == "qualification" and rank == 0:
                dispatch = receipt.get("fast_dispatch", {})
                metrics = qualification_metrics_from_run(
                    receipt, plan=plan,
                    backend_proof=proof["backend"],
                    save_reload_pass=(
                        metadata.get("run_mode") == "qualification"
                        and receipt["checkpoint_save_reload"]["pass"] is True),
                    dispatch_forward_count=dispatch.get("forward_count", 0),
                    dispatch_grad_forward_count=dispatch.get("grad_forward_count", 0),
                    dispatch_backward_count=dispatch.get("backward_count", 0))
                receipt["qualification_metrics"] = metrics
                receipt["qualifier_state_discarded_from_primary_lineage"] = True
            receipt.pop("receipt_sha256", None)
            receipt = finalize_receipt(receipt)
            _atomic_json(output / f"RANK_{rank}_FINAL.json", receipt)
            if rank == 0:
                _atomic_json(output / "COMPLETE.json", receipt)
            completed = True
            return receipt
    except BaseException as exc:
        _atomic_json(output / f"RANK_{rank}_WORKER_FAILURE.json", {
            "version": WORKER_VERSION, "status": "failed", "rank": rank,
            "plan_sha256": plan.full_plan_sha256,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(limit=30),
            "automatic_retry": False,
        })
        raise
    finally:
        model_deleted = model is not None
        cpu_payloads_released = any(value is not None for value in (
            prepared, bindings, tokenizer, records,
            data_binding, exposure_binding, dataset_receipt, admission))
        model = None
        prepared = None
        bindings = None
        tokenizer = None
        records = None
        data_binding = None
        exposure_binding = None
        dataset_receipt = None
        admission = None
        collected = gc.collect()
        cuda_cache_cleared = False
        cuda_ipc_collected = False
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                cuda_cache_cleared = True
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                    cuda_ipc_collected = True
        except (ImportError, RuntimeError):
            pass
        ddp_was_initialized = bool(
            torch.distributed.is_available() and torch.distributed.is_initialized())
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        ddp_destroyed = not (
            torch.distributed.is_available() and torch.distributed.is_initialized())
        scratch_configured = bool(args.scratch)
        scratch_absent = True
        scratch_cleanup = "not_configured"
        if args.scratch and rank == 0:
            scratch = Path(args.scratch).resolve()
            # Cleanup is restricted to the explicit per-run scratch subtree.
            if output.parent not in scratch.parents:
                scratch_absent = False
                scratch_cleanup = "unsafe_path_rejected"
            else:
                if scratch.exists():
                    shutil.rmtree(scratch)
                scratch_absent = not scratch.exists()
                scratch_cleanup = "removed" if scratch_absent else "still_present"
        elif args.scratch:
            scratch_cleanup = "delegated_to_rank0"
        allocated = 0
        reserved = 0
        if torch.cuda.is_available():
            try:
                allocated = int(torch.cuda.memory_allocated(local_rank))
                reserved = int(torch.cuda.memory_reserved(local_rank))
            except (RuntimeError, TypeError):
                allocated = -1
                reserved = -1
        released = (ddp_was_initialized and ddp_destroyed
                    and cuda_cache_cleared and allocated == 0
                    and (rank != 0 or scratch_absent))
        cleanup = finalize_receipt({
            "version": "qwen35-9b-ordinal-continuity-post-cleanup-v1",
            "status": "released" if released else "release_unverified",
            "rank": rank, "plan_sha256": plan.full_plan_sha256,
            "scientific_stage_completed": completed,
            "model_reference_deleted": model_deleted,
            "tokenizer_and_tokenized_payloads_released": cpu_payloads_released,
            "gc_collected_objects": collected,
            "cuda_cache_cleared": cuda_cache_cleared,
            "cuda_ipc_collected": cuda_ipc_collected,
            "memory_allocated_bytes": allocated,
            "memory_reserved_bytes": reserved,
            "ddp_was_initialized": ddp_was_initialized,
            "ddp_destroyed": ddp_destroyed,
            "scratch_configured": scratch_configured,
            "scratch_cleanup": scratch_cleanup,
            "scratch_absent_after_cleanup": scratch_absent if rank == 0 else None,
            "provider_compute_release_verified": False,
        })
        _atomic_json(output / f"RANK_{rank}_POST_CLEANUP.json", cleanup)


def main(argv=None):
    original_argv = list(__import__("sys").argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--admission")
    parser.add_argument("--generated-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--package-manifest", required=True)
    parser.add_argument("--package-tree-sha256", required=True)
    parser.add_argument("--tokenizer-root", required=True)
    parser.add_argument("--tokenizer-receipt", required=True)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--lock", required=True)
    parser.add_argument("--scratch")
    args = parser.parse_args(argv)
    args.actual_worker_argv = [
        "torchrun", "--standalone", "--nproc_per_node=2", "-m",
        "research4.src.qwen9b_ordinal_continuity_worker", *original_argv]
    if TrainingPlan(**_read(args.plan)).mode == "primary" and not args.admission:
        parser.error("--admission is required for primary training")
    run(args)


if __name__ == "__main__":
    main()
