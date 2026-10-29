"""One-use, bounded partitions for ordinal simulated-continuity training.

One invocation owns one qualification or policy-stage partition.  It never
runs the 27-stage matrix and never retries.  A CPU-only aggregator later
verifies all 27 residual-stage terminal receipts against one frozen base.
"""
from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Callable, Mapping, Sequence
import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import time

from .qwen9b_ordinal_continuity_lora import MODEL_ID, MODEL_REVISION
from .qwen9b_ordinal_continuity_runtime import (
    atomic_json, canonical_sha256, cli_sha256, directory_tree_sha256,
    file_sha256, finalize_receipt, verify_package_manifest, verify_receipt,
    verify_tokenizer_receipt)
from .qwen9b_ordinal_continuity_training import (
    ALLOWED_SEEDS, CONDITIONS, PINNED_SOFTWARE, TrainingPlan,
    load_pinned_tokenizer)

CONTROLLER_VERSION = "qwen35-9b-ordinal-continuity-partition-controller-v3"
EXPERIMENT_VERSION = "qwen35-9b-ordinal-continuity-experiment-v2"
PARTITION_INTENT_VERSION = "qwen35-9b-continuity-partition-intent-v2"
TERMINAL_VERSION = "qwen35-9b-continuity-partition-terminal-v2"
AGGREGATE_VERSION = "qwen35-9b-ordinal-continuity-matrix-complete-v3"
MATRIX_CONDITIONS = CONDITIONS
MATRIX_STAGES = ("delta1", "delta2", "delta3")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_experiment_spec(value: Mapping) -> dict:
    spec = dict(value)
    required = {
        "version": EXPERIMENT_VERSION, "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "conditions": list(CONDITIONS),
        "seeds": list(ALLOWED_SEEDS), "stages": list(MATRIX_STAGES),
        "qualification_updates": 2, "primary_optimizer_updates": 144,
        "baseline": "frozen_base", "automatic_retries": 0}
    if any(spec.get(key) != expected for key, expected in required.items()):
        raise PermissionError("Experiment spec differs from the preregistered matrix")
    if spec.get("one_use") is not True:
        raise PermissionError("Every partition must be one-use")
    return spec


def matrix_plans(spec: Mapping, selected_rank: int):
    """Return plan metadata only; no tokenizer or dataset is retained."""
    spec = validate_experiment_spec(spec)
    if selected_rank not in (4, 8):
        raise ValueError("Selected residual rank must be 4 or 8")
    plans = [TrainingPlan(
        mode="primary", condition=condition, stage=stage, rank=selected_rank,
        alpha=2.0 * selected_rank, seed=seed,
        optimizer_updates=spec["primary_optimizer_updates"]).validate()
        for condition in CONDITIONS for seed in ALLOWED_SEEDS
        for stage in MATRIX_STAGES]
    if (len(plans) != 27 or len({p.full_plan_sha256 for p in plans}) != 27
            or any(p.global_batch_size != 4 or p.occurrence_count != 576
                   for p in plans)):
        raise RuntimeError("The exact 3x3x3 policy matrix differs")
    return plans


def qualification_plan(spec: Mapping, rank: int):
    spec = validate_experiment_spec(spec)
    if rank not in (4, 8):
        raise ValueError("Qualification rank must be 4 or 8")
    return TrainingPlan(
        mode="qualification", condition="self_continuity", stage="delta1",
        rank=rank, alpha=2.0 * rank, seed=ALLOWED_SEEDS[0],
        optimizer_updates=spec["qualification_updates"]).validate()


def partition_identity(plan: TrainingPlan) -> str:
    if plan.mode == "qualification":
        return f"qualification_rank{plan.rank}"
    return f"{plan.condition}_seed{plan.seed}_{plan.stage}"


def validate_partition_intent(value: Mapping, plan: TrainingPlan,
                              actual_parent_tree: str | None) -> dict:
    intent = dict(value)
    expected_updates = 2 if plan.mode == "qualification" else 144
    expected_occurrences = expected_updates * 4
    if (plan.optimizer_updates != expected_updates
            or plan.global_batch_size != 4
            or plan.occurrence_count != expected_occurrences):
        raise PermissionError(
            "Executable partition exposure differs from the preregistered "
            "2/144-update, global-batch-4 recipe")
    if (intent.get("version") != PARTITION_INTENT_VERSION
            or intent.get("one_use") is not True
            or intent.get("attempt_number") != 1
            or intent.get("automatic_retries") != 0
            or intent.get("partition_identity") != partition_identity(plan)
            or intent.get("plan_sha256") != plan.full_plan_sha256):
        raise PermissionError("Partition intent differs from this exact plan")
    stop, ceiling = (intent.get("wall_clock_stop_seconds"),
                     intent.get("wall_clock_ceiling_seconds"))
    if (not isinstance(stop, int) or not isinstance(ceiling, int)
            or stop < 300 or ceiling <= stop or ceiling - stop < 60
            or ceiling > 43_200):
        raise PermissionError("Partition wall-clock stop/ceiling is invalid")
    if intent.get("expected_parent_checkpoint_tree_sha256") != actual_parent_tree:
        raise PermissionError("Partition predecessor checkpoint tree differs")
    base_sha = intent.get("frozen_base_model_sha256")
    if not isinstance(base_sha, str) or len(base_sha) != 64:
        raise PermissionError("Partition intent lacks the frozen-base identity")
    if plan.mode == "qualification" or plan.stage == "delta1":
        if actual_parent_tree is not None:
            raise PermissionError("Qualification and Delta1 must start from frozen base")
    elif not isinstance(actual_parent_tree, str) or len(actual_parent_tree) != 64:
        raise PermissionError("Delta2/Delta3 require the exact predecessor checkpoint")
    return intent


def materialize_tokenizer(root, plan: TrainingPlan) -> dict:
    destination = Path(root).resolve()
    receipt_path = destination / "TOKENIZER_RECEIPT.json"
    if destination.exists():
        return verify_tokenizer_receipt(
            destination, receipt_path, model_id=plan.model_id,
            revision=plan.model_revision)
    destination.mkdir(parents=True)
    tokenizer = load_pinned_tokenizer(plan)
    commit = tokenizer.init_kwargs.get("_commit_hash")
    if commit != plan.model_revision:
        raise RuntimeError("Tokenizer did not resolve to the pinned revision")
    tokenizer.save_pretrained(destination)
    files = [{"path": p.relative_to(destination).as_posix(),
              "sha256": file_sha256(p), "bytes": p.stat().st_size}
             for p in sorted(x for x in destination.rglob("*") if x.is_file())]
    receipt = finalize_receipt({
        "version": "qwen35-9b-tokenizer-files-v1", "model_id": plan.model_id,
        "revision": plan.model_revision, "resolved_commit": commit,
        "tokenizer_class": tokenizer.__class__.__name__,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id, "files": files,
        "tree_sha256": canonical_sha256(files)})
    atomic_json(receipt_path, receipt)
    del tokenizer
    gc.collect()
    return receipt


def exact_worker_argv(*, plan_path, intent_path, generated_root, output,
                      source_sha256, package_root, package_manifest,
                      package_tree_sha256, tokenizer_root, tokenizer_receipt,
                      lock, scratch=None, admission_path=None,
                      parent_checkpoint=None):
    argv = ["torchrun", "--standalone", "--nproc_per_node=2", "-m",
            "research4.src.qwen9b_ordinal_continuity_worker",
            "--plan", str(plan_path), "--intent", str(intent_path),
            "--generated-root", str(generated_root), "--output", str(output),
            "--source-sha256", source_sha256,
            "--package-root", str(package_root),
            "--package-manifest", str(package_manifest),
            "--package-tree-sha256", package_tree_sha256,
            "--tokenizer-root", str(tokenizer_root),
            "--tokenizer-receipt", str(tokenizer_receipt),
            "--lock", str(lock)]
    if admission_path is not None:
        argv += ["--admission", str(admission_path)]
    if parent_checkpoint is not None:
        argv += ["--parent-checkpoint", str(parent_checkpoint)]
    if scratch is not None:
        argv += ["--scratch", str(scratch)]
    return argv


def import_preflight(requirements_path) -> dict:
    declared = {}
    for line in Path(requirements_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, expected = line.split("==", 1)
            declared[name] = expected
    if declared != {**PINNED_SOFTWARE, "nvidia-ml-py": "13.615.71"}:
        raise RuntimeError("Offline requirements lock differs")
    observed = {}
    for name, expected in declared.items():
        try:
            observed[name] = package_version(name)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"Pinned dependency is unavailable: {name}") from exc
        if observed[name] != expected:
            raise RuntimeError(f"Pinned dependency differs: {name}")
    for module in ("torch", "transformers", "bitsandbytes", "accelerate",
                   "safetensors", "numpy", "psutil", "fla", "einops",
                   "triton", "pynvml"):
        __import__(module)
    return {"version": CONTROLLER_VERSION, "status": "ready", "versions": observed}


def _exclusive_json(path: Path, value: Mapping):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _bounded_subprocess(argv: Sequence[str], stop_seconds: int,
                        ceiling_seconds: int) -> dict:
    started = time.monotonic()
    process = subprocess.Popen(list(argv), shell=False, start_new_session=True)
    stop_sent = kill_sent = False
    try:
        code = process.wait(timeout=stop_seconds)
    except subprocess.TimeoutExpired:
        stop_sent = True
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (AttributeError, ProcessLookupError, PermissionError):
            process.terminate()
        remaining = max(1.0, ceiling_seconds - (time.monotonic() - started))
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            kill_sent = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (AttributeError, ProcessLookupError, PermissionError):
                process.kill()
            code = process.wait()
    return {"exit_code": code, "elapsed_seconds": time.monotonic() - started,
            "operator_stop_sent": stop_sent, "ceiling_kill_sent": kill_sent,
            "child_exit_confirmed": process.poll() is not None}


class PartitionController:
    """Run exactly one immutable partition and emit a terminal receipt."""

    def __init__(self, *, package_root, package_manifest, generated_root,
                 run_root, source_sha256, package_tree_sha256, tokenizer_root,
                 runner: Callable | None = None):
        self.package_root = Path(package_root).resolve()
        self.package_manifest = Path(package_manifest).resolve()
        self.generated_root = Path(generated_root).resolve()
        self.run_root = Path(run_root).resolve()
        self.source_sha256 = source_sha256
        self.package_tree_sha256 = package_tree_sha256
        self.tokenizer_root = Path(tokenizer_root).resolve()
        self.runner = runner or _bounded_subprocess
        stage_root = self.package_root.parent
        if (self.run_root.parent.parent != stage_root
                or self.tokenizer_root.parent.parent != stage_root
                or self.run_root.parent != self.tokenizer_root.parent):
            raise PermissionError("Mutable runtime must be outside immutable package")

    def verify_package(self):
        return verify_package_manifest(
            self.package_root, self.package_manifest,
            expected_tree_sha256=self.package_tree_sha256,
            expected_source_sha256=self.source_sha256)

    @staticmethod
    def _cleanup_pass(stage):
        for rank in (0, 1):
            path = Path(stage) / f"RANK_{rank}_POST_CLEANUP.json"
            if not path.is_file():
                return False
            try:
                receipt = verify_receipt(_read(path))
            except (ValueError, PermissionError):
                return False
            if (receipt.get("status") != "released"
                    or receipt.get("memory_allocated_bytes") != 0
                    or receipt.get("ddp_destroyed") is not True
                    or receipt.get("cuda_cache_cleared") is not True
                    or receipt.get("model_reference_deleted") is not True
                    or receipt.get("tokenizer_and_tokenized_payloads_released") is not True
                    or (rank == 0 and
                        receipt.get("scratch_absent_after_cleanup") is not True)):
                return False
        return True

    @staticmethod
    def _compact(bindings):
        return {
            "data_binding_sha256": canonical_sha256(bindings["data_binding"]),
            "exposure_binding_sha256": canonical_sha256(bindings["exposure_binding"]),
            "dataset_receipt_sha256": bindings["dataset_receipt"]["receipt_sha256"],
            "tokenizer_receipt_sha256": bindings["tokenizer_receipt"]["receipt_sha256"],
            "global_journal_sha256": bindings["exposure_binding"].get(
                "global_journal_sha256"),
            "global_journal_occurrence_count": bindings["exposure_binding"].get(
                "global_journal_occurrence_count")}

    def _runtime_admission(self, path, plan, metadata, dynamic, worker_intent):
        if plan.mode != "primary":
            if path is not None:
                raise PermissionError("Qualification cannot consume admission")
            return None
        if path is None:
            raise PermissionError("Primary partition lacks reviewed admission")
        source = _read(path)
        check = dict(source)
        embedded = check.pop("admission_sha256", None)
        if (canonical_sha256(check) != embedded or source.get("admitted") is not True
                or source.get("plan") != asdict(plan)
                or source.get("full_plan_sha256") != plan.full_plan_sha256
                or source.get("data_binding_sha256") != metadata["data_binding_sha256"]
                or source.get("exposure_binding_sha256") != metadata["exposure_binding_sha256"]
                or source.get("provider_submission_performed") is not False):
            raise PermissionError("Packaged primary admission differs")
        for key in ("worker_cli_sha256", "package_tree_sha256",
                    "package_manifest_sha256", "tokenizer_receipt_sha256",
                    "dataset_receipt_sha256", "parent_checkpoint_tree_sha256",
                    "one_use_submission_intent"):
            check.pop(key, None)
        check.update(dynamic)
        check["one_use_submission_intent"] = file_sha256(worker_intent)
        value = finalize_receipt(check)
        value["admission_sha256"] = value.pop("receipt_sha256")
        target = self.run_root.parent / "control" / "admission.json"
        atomic_json(target, value)
        return target

    def run(self, plan_path, partition_intent_path, *, admission_path=None,
            parent_checkpoint=None) -> dict:
        package = self.verify_package()
        plan = TrainingPlan(**_read(plan_path)).validate()
        parent = None if parent_checkpoint is None else Path(parent_checkpoint).resolve()
        parent_tree = None if parent is None else directory_tree_sha256(parent)
        intent = validate_partition_intent(
            _read(partition_intent_path), plan, parent_tree)
        consumed = finalize_receipt({
            "version": CONTROLLER_VERSION,
            "partition_identity": intent["partition_identity"],
            "partition_intent_sha256": file_sha256(partition_intent_path),
            "plan_sha256": plan.full_plan_sha256,
            "expected_parent_checkpoint_tree_sha256": parent_tree,
            "wall_clock_stop_seconds": intent["wall_clock_stop_seconds"],
            "wall_clock_ceiling_seconds": intent["wall_clock_ceiling_seconds"],
            "automatic_retries": 0})
        _exclusive_json(self.run_root.parent / "PARTITION_INTENT_CONSUMED.json",
                        consumed)
        if self.run_root.exists():
            raise FileExistsError("Partition output is immutable")
        self.run_root.parent.mkdir(parents=True, exist_ok=True)
        materialize_tokenizer(self.tokenizer_root, plan)
        from .qwen9b_ordinal_continuity_worker import prepare_runtime_bindings
        bindings = prepare_runtime_bindings(
            self.generated_root, plan, self.tokenizer_root,
            self.tokenizer_root / "TOKENIZER_RECEIPT.json")
        metadata = self._compact(bindings)
        bindings = None
        gc.collect()

        control = self.run_root.parent / "control"
        control.mkdir(parents=True, exist_ok=False)
        runtime_plan = control / "plan.json"
        worker_intent = control / "worker_intent.json"
        atomic_json(runtime_plan, asdict(plan))
        admission_target = control / "admission.json" if plan.mode == "primary" else None
        argv = exact_worker_argv(
            plan_path=runtime_plan, intent_path=worker_intent,
            generated_root=self.generated_root, output=self.run_root,
            source_sha256=self.source_sha256, package_root=self.package_root,
            package_manifest=self.package_manifest,
            package_tree_sha256=self.package_tree_sha256,
            tokenizer_root=self.tokenizer_root,
            tokenizer_receipt=self.tokenizer_root / "TOKENIZER_RECEIPT.json",
            lock=self.run_root.parent / "scientific_worker.lock",
            scratch=self.run_root.parent / ".scratch" / intent["partition_identity"],
            admission_path=admission_target, parent_checkpoint=parent)
        dynamic = {"worker_cli_sha256": cli_sha256(argv), **metadata,
                   "source_sha256": self.source_sha256,
                   "package_tree_sha256": package["verified_tree_sha256"],
                   "package_manifest_sha256": package["manifest_file_sha256"],
                   "parent_checkpoint_tree_sha256": parent_tree}
        atomic_json(worker_intent, {
            "version": "qwen35-9b-continuity-one-use-intent-v1",
            "one_use": True, "automatic_retries": 0, "attempt_number": 1,
            "partition_identity": intent["partition_identity"],
            "partition_intent_sha256": file_sha256(partition_intent_path),
            "plan_sha256": plan.full_plan_sha256,
            "output_root": str(self.run_root), **dynamic})
        if plan.mode == "primary":
            self._runtime_admission(
                admission_path, plan, metadata, dynamic, worker_intent)

        error = None
        try:
            child = dict(self.runner(
                argv, intent["wall_clock_stop_seconds"],
                intent["wall_clock_ceiling_seconds"]))
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            child = {"exit_code": None, "elapsed_seconds": None,
                     "operator_stop_sent": False, "ceiling_kill_sent": False,
                     "child_exit_confirmed": False}
        complete = None
        complete_path = self.run_root / "COMPLETE.json"
        if child.get("exit_code") == 0 and complete_path.is_file():
            try:
                complete = verify_receipt(_read(complete_path))
            except (ValueError, PermissionError) as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
        cleanup = self._cleanup_pass(self.run_root)
        checkpoint = self.run_root / "checkpoint"
        checkpoint_tree = directory_tree_sha256(checkpoint) if checkpoint.is_dir() else None
        lineage = (
            intent.get("frozen_base_model_sha256") is not None
            and ((plan.mode == "qualification" or plan.stage == "delta1")
                 == (parent_tree is None)))
        success = bool(complete and cleanup and lineage
                       and child.get("exit_code") == 0
                       and child.get("child_exit_confirmed") is True
                       and not child.get("operator_stop_sent")
                       and not child.get("ceiling_kill_sent"))
        status = ("complete" if success else "ceiling_killed"
                  if child.get("ceiling_kill_sent") else "stopped_at_wall_clock"
                  if child.get("operator_stop_sent") else "failed")
        terminal = finalize_receipt({
            "version": TERMINAL_VERSION, "status": status,
            "partition_identity": intent["partition_identity"],
            "plan": asdict(plan), "plan_sha256": plan.full_plan_sha256,
            "partition_intent_sha256": file_sha256(partition_intent_path),
            "consumed_identity_receipt_sha256": consumed["receipt_sha256"],
            "package_tree_sha256": self.package_tree_sha256,
            "source_sha256": self.source_sha256,
            "wall_clock_stop_seconds": intent["wall_clock_stop_seconds"],
            "wall_clock_ceiling_seconds": intent["wall_clock_ceiling_seconds"],
            "child": child, "error": error,
            "scientific_stage_completed": complete is not None,
            "worker_complete_receipt_sha256": (
                None if complete is None else complete["receipt_sha256"]),
            "primary_save_reload_pass": bool(
                complete and complete.get("primary_save_reload_pass") is True),
            "post_cleanup_pass": cleanup,
            "parent_checkpoint_tree_sha256": parent_tree,
            "checkpoint_tree_sha256": checkpoint_tree,
            "frozen_base_model_sha256": intent.get("frozen_base_model_sha256"),
            "residual_lineage_pass": lineage,
            "automatic_retries": 0,
            "provider_compute_release_verified": False})
        atomic_json(self.run_root.parent / "TERMINAL_RECEIPT.json", terminal)
        return terminal


def aggregate_terminal_receipts(policy_terminals: Sequence, *,
                                frozen_base_model_sha256: str,
                                output=None) -> dict:
    if (not isinstance(frozen_base_model_sha256, str)
            or len(frozen_base_model_sha256) != 64):
        raise PermissionError("Aggregate requires one exact frozen-base identity")
    rows = [verify_receipt(_read(x) if not isinstance(x, Mapping) else dict(x))
            for x in policy_terminals]
    expected = {(condition, seed, stage) for condition in CONDITIONS
                for seed in ALLOWED_SEEDS for stage in MATRIX_STAGES}
    observed = {(x.get("plan", {}).get("condition"),
                 x.get("plan", {}).get("seed"),
                 x.get("plan", {}).get("stage")) for x in rows}
    if len(rows) != 27 or observed != expected:
        raise PermissionError("The 27 policy terminals are incomplete or duplicated")
    by_key = {}
    for row in rows:
        p = row["plan"]
        key = (p["condition"], p["seed"], p["stage"])
        if (row.get("version") != TERMINAL_VERSION or row.get("status") != "complete"
                or p.get("mode") != "primary" or p.get("optimizer_updates") != 144
                or row.get("post_cleanup_pass") is not True
                or row.get("primary_save_reload_pass") is not True
                or row.get("frozen_base_model_sha256") != frozen_base_model_sha256
                or row.get("residual_lineage_pass") is not True):
            raise PermissionError(f"Policy terminal is incomplete: {key}")
        by_key[key] = row
    for condition in CONDITIONS:
        for seed in ALLOWED_SEEDS:
            predecessor = None
            for stage in MATRIX_STAGES:
                row = by_key[(condition, seed, stage)]
                if row.get("parent_checkpoint_tree_sha256") != predecessor:
                    raise PermissionError("Policy predecessor chain differs")
                predecessor = row.get("checkpoint_tree_sha256")
                if not isinstance(predecessor, str) or len(predecessor) != 64:
                    raise PermissionError("Policy checkpoint tree is missing")
    result = finalize_receipt({
        "version": AGGREGATE_VERSION, "status": "complete",
        "frozen_base_model_sha256": frozen_base_model_sha256,
        "policy_terminal_count": 27,
        "policy_terminal_receipt_sha256": sorted(x["receipt_sha256"] for x in rows),
        "conditions": list(CONDITIONS), "seeds": list(ALLOWED_SEEDS),
        "stages": list(MATRIX_STAGES), "automatic_retries": 0})
    if output is not None:
        atomic_json(output, result)
    return result


def main(argv=None):
    original = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--requirements", required=True)
    run = sub.add_parser("run-partition")
    for name in ("package-root", "package-manifest", "generated-root", "plan",
                 "partition-intent", "run-root", "source-sha256",
                 "package-tree-sha256", "tokenizer-root", "offline-wheelhouse"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--admission")
    run.add_argument("--parent-checkpoint")
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--frozen-base-model-sha256", required=True)
    aggregate.add_argument("--policy-terminal", action="append", required=True)
    aggregate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(import_preflight(args.requirements), indent=2, sort_keys=True))
        return
    if args.command == "aggregate":
        print(json.dumps(aggregate_terminal_receipts(
            args.policy_terminal,
            frozen_base_model_sha256=args.frozen_base_model_sha256,
            output=args.output),
            indent=2, sort_keys=True))
        return
    manifest = _read(args.package_manifest)
    actual = ["python", "-m",
              "research4.src.qwen9b_ordinal_continuity_controller", *original]
    if (actual != manifest.get("launch_argv")
            or cli_sha256(actual) != manifest.get("launch_cli_sha256")):
        raise PermissionError("Actual partition CLI differs from manifest")
    controller = PartitionController(
        package_root=args.package_root, package_manifest=args.package_manifest,
        generated_root=args.generated_root, run_root=args.run_root,
        source_sha256=args.source_sha256,
        package_tree_sha256=args.package_tree_sha256,
        tokenizer_root=args.tokenizer_root)
    result = controller.run(
        args.plan, args.partition_intent, admission_path=args.admission,
        parent_checkpoint=args.parent_checkpoint)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
