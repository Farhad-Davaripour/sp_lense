"""CPU-only publication of the frozen Research4 transfer-evaluation package.

The packager is deliberately downstream of the scientific run ledger.  It
cannot train, call a model, or repair incomplete evidence.  It accepts only a
completed frozen-base gate, one completed rank-4 capacity gate, the exact 27
Gold policy rows, and the corresponding immutable Volume outputs.  It then
recomputes the training lineage and publishes one content-addressed package:
one seedless frozen base identity plus nine Delta3 endpoints.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile

from .continuity_transfer_evaluation import (
    ALL_ARMS, BASELINE_CONDITION, CHECKPOINT_COUNT, CONDITIONS, REGISTRY_SEED,
    TRAINING_SEEDS, assert_training_has_no_transfer_choices, digest,
    file_sha256, validate_public_commitment, validate_registry_bundle,
    validate_unblinding_key, write_sealed_registry,
)
from .databricks_simulated_continuity import (
    GOLD_SUMMARY_TABLE, MODEL_ID, MODEL_REVISION, VOLUME, build_source_manifest,
    canonical, gold_summary, validate_base_qualification_receipt,
    validate_exact_gold_row, validate_source_manifest,
)
from .qwen9b_ordinal_continuity_controller import matrix_plans, partition_identity
from .qwen9b_ordinal_continuity_lora import BRANCHES, CHECKPOINT_VERSION
from .qwen9b_ordinal_continuity_runtime import (
    directory_tree, directory_tree_sha256,
)
from .qwen9b_ordinal_continuity_training import TrainingPlan, verify_rank_receipt


PACKAGING_PLAN_VERSION = "research4-databricks-transfer-packaging-plan-v1"
PACKAGING_RECEIPT_VERSION = "research4-databricks-transfer-packaging-receipt-v1"
PACKAGE_FILES_VERSION = "research4-databricks-transfer-package-files-v1"
CHECKPOINT_MANIFEST_VERSION = "research4-three-residual-blinded-checkpoints-v1"
EVALUATION_CHECKPOINT_VERSION = "qwen35-9b-blinded-evaluation-checkpoint-v1"
TRAINING_AUDIT_VERSION = "qwen35-9b-three-residual-training-audit-v2"
TRANSFER_PACKAGE_IDENTITY = "continuity_transfer_three_residual_v1"
TRANSFER_PACKAGE_PARENT = f"{VOLUME}/transfer_packages"
TRANSFER_UNBLINDING_PARENT = f"{VOLUME}/transfer_unblinding_keys"
TRANSFER_PACKAGE_LOCK_PARENT = f"{VOLUME}/transfer_package_identities"
TRANSFER_PACKAGE_GOLD_TABLE = (
    "research4_simulated_continuity.gold.continuity_transfer_packages"
)


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _volume_path(value: Any, *, label: str) -> str:
    if (not isinstance(value, str) or not value.startswith("/Volumes/")
            or ".." in Path(value).parts):
        raise PermissionError(f"{label} must be an exact Unity Catalog Volume path")
    return value


def validate_packaging_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = dict(value) if isinstance(value, Mapping) else {}
    exact = {
        "version", "identity", "execution_platform", "cpu_only",
        "gpu_runs_started", "model_forwards", "optimizer_updates",
        "automatic_retries", "source_commit", "source_manifest",
        "source_sha256", "model_tree_sha256",
        "base_qualification_receipt_path",
        "base_qualification_receipt_sha256",
        "base_qualification_receipt_file_sha256",
        "capacity_qualification_receipt_path",
        "capacity_qualification_receipt_sha256",
        "capacity_qualification_receipt_file_sha256", "gold_summary_table",
        "expected_policy_partition_count", "expected_adapted_endpoint_count",
        "expected_frozen_base_endpoint_count", "registry_seed",
        "package_parent", "plan_sha256",
    }
    try:
        source = validate_source_manifest(plan.get("source_manifest", {}))
    except (TypeError, ValueError, PermissionError) as exc:
        raise PermissionError("Packaging plan source manifest differs") from exc
    payload = dict(plan); embedded = payload.pop("plan_sha256", None)
    if (set(plan) != exact or embedded != digest(payload)
            or plan.get("version") != PACKAGING_PLAN_VERSION
            or plan.get("identity") != TRANSFER_PACKAGE_IDENTITY
            or plan.get("execution_platform") != "databricks"
            or plan.get("cpu_only") is not True
            or any(plan.get(key) != 0 for key in (
                "gpu_runs_started", "model_forwards", "optimizer_updates",
                "automatic_retries"))
            or not re.fullmatch(r"[0-9a-f]{40}", str(plan.get("source_commit", "")))
            or source.get("source_commit") != plan.get("source_commit")
            or source.get("manifest_sha256") != plan.get("source_sha256")
            or any(not _sha(plan.get(key)) for key in (
                "source_sha256", "model_tree_sha256",
                "base_qualification_receipt_sha256",
                "base_qualification_receipt_file_sha256",
                "capacity_qualification_receipt_sha256",
                "capacity_qualification_receipt_file_sha256"))
            or plan.get("gold_summary_table") != GOLD_SUMMARY_TABLE
            or plan.get("expected_policy_partition_count") != 27
            or plan.get("expected_adapted_endpoint_count") != 9
            or plan.get("expected_frozen_base_endpoint_count") != 1
            or plan.get("registry_seed") != REGISTRY_SEED
            or plan.get("package_parent") != TRANSFER_PACKAGE_PARENT):
        raise PermissionError("Exact CPU-only transfer packaging plan required")
    _volume_path(plan.get("base_qualification_receipt_path"), label="base receipt")
    _volume_path(plan.get("capacity_qualification_receipt_path"), label="capacity receipt")
    _volume_path(plan.get("package_parent"), label="package parent")
    return json.loads(canonical(plan))


def build_packaging_plan(*, source_root: str | Path, source_commit: str,
                         model_tree_sha256: str,
                         base_qualification_receipt_path: str,
                         base_qualification_receipt_sha256: str,
                         base_qualification_receipt_file_sha256: str,
                         capacity_qualification_receipt_path: str,
                         capacity_qualification_receipt_sha256: str,
                         capacity_qualification_receipt_file_sha256: str) -> dict[str, Any]:
    source = build_source_manifest(source_root, source_commit)
    value = {
        "version": PACKAGING_PLAN_VERSION,
        "identity": TRANSFER_PACKAGE_IDENTITY,
        "execution_platform": "databricks", "cpu_only": True,
        "gpu_runs_started": 0, "model_forwards": 0, "optimizer_updates": 0,
        "automatic_retries": 0, "source_commit": source_commit,
        "source_manifest": source, "source_sha256": source["manifest_sha256"],
        "model_tree_sha256": model_tree_sha256,
        "base_qualification_receipt_path": base_qualification_receipt_path,
        "base_qualification_receipt_sha256": base_qualification_receipt_sha256,
        "base_qualification_receipt_file_sha256":
            base_qualification_receipt_file_sha256,
        "capacity_qualification_receipt_path": capacity_qualification_receipt_path,
        "capacity_qualification_receipt_sha256": capacity_qualification_receipt_sha256,
        "capacity_qualification_receipt_file_sha256":
            capacity_qualification_receipt_file_sha256,
        "gold_summary_table": GOLD_SUMMARY_TABLE,
        "expected_policy_partition_count": 27,
        "expected_adapted_endpoint_count": 9,
        "expected_frozen_base_endpoint_count": 1,
        "registry_seed": REGISTRY_SEED,
        "package_parent": TRANSFER_PACKAGE_PARENT,
    }
    value["plan_sha256"] = digest(value)
    return validate_packaging_plan(value)


def canonical_primary_plans() -> dict[tuple[str, int, str], TrainingPlan]:
    from .databricks_simulated_continuity import EXPERIMENT_SPEC
    return {
        (plan.condition, plan.seed, plan.stage): replace(
            plan, device_profile="databricks-a10x1", world_size=1,
            micro_batch_size=2, accumulation_steps=2).validate()
        for plan in matrix_plans(EXPERIMENT_SPEC, 4)
    }


def validate_gold_matrix(*, plan: Mapping[str, Any],
                         base_receipt: Mapping[str, Any],
                         capacity_receipt: Mapping[str, Any],
                         gold_rows: Sequence[Mapping[str, Any]],
                         summary_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute the exact base/capacity/27-policy Gold matrix."""
    value = validate_packaging_plan(plan)
    base = validate_base_qualification_receipt(base_receipt)
    rows = [dict(row) for row in gold_rows]
    if (base.get("source_commit") != value["source_commit"]
            or base.get("source_sha256") != value["source_sha256"]
            or base.get("model_tree_sha256") != value["model_tree_sha256"]
            or base.get("receipt_sha256")
               != value["base_qualification_receipt_sha256"]):
        raise PermissionError("Frozen-base gate differs from packaging authority")
    if (capacity_receipt.get("receipt_sha256")
            != value["capacity_qualification_receipt_sha256"]
            or capacity_receipt.get("source_commit") != value["source_commit"]
            or capacity_receipt.get("source_sha256") != value["source_sha256"]
            or capacity_receipt.get("model_tree_sha256")
               != value["model_tree_sha256"]
            or capacity_receipt.get("base_qualification_receipt_sha256")
               != base["receipt_sha256"]):
        raise PermissionError("Capacity gate differs from packaging authority")
    if len(rows) != 29:
        raise PermissionError("Packaging requires exactly base + capacity + 27 Gold rows")
    base_rows = [row for row in rows if row.get("mode") == "base_qualification"]
    validate_exact_gold_row(base_rows, {
        "partition_identity": base["identity"], "mode": "base_qualification",
        "condition": "frozen_base", "stage": "format_schema", "seed": 0,
        "plan_sha256": base["plan_sha256"],
        "source_commit": value["source_commit"],
        "model_tree_sha256": value["model_tree_sha256"],
        "checkpoint_tree_sha256": value["model_tree_sha256"],
        "receipt_sha256": base["receipt_sha256"],
        "evidence_sha256": base["raw_results_sha256"],
        "intent_sha256": base["base_qualification_intent_sha256"],
    }, label="frozen-base packaging gate")
    capacity_rows = [row for row in rows if row.get("mode") == "qualification"]
    policies = [row for row in rows if row.get("mode") == "primary"]
    if len(capacity_rows) != 1 or len(policies) != 27:
        raise PermissionError("Packaging requires exactly base + capacity + 27 Gold rows")
    summary = gold_summary(base, capacity_receipt, capacity_rows[0], policies)
    matches = [dict(row) for row in summary_rows
               if row.get("matrix_identity") == summary["summary_sha256"]]
    if (len(matches) != 1 or matches[0].get("status") != "complete"
            or matches[0].get("policy_partition_count") != 27
            or matches[0].get("summary_sha256") != summary["summary_sha256"]
            or matches[0].get("summary_json") != canonical(summary)
            or matches[0].get("baseline") != "frozen_base"
            or matches[0].get("frozen_base_model_sha256")
               != base["model_tree_sha256"]
            or matches[0].get("base_qualification_receipt_sha256")
               != base["receipt_sha256"]
            or matches[0].get("capacity_qualification_receipt_sha256")
               != capacity_receipt.get("receipt_sha256")):
        raise PermissionError("The independently audited Gold matrix summary is absent")
    return {"base": base, "capacity": dict(capacity_receipt),
            "capacity_gold": capacity_rows[0], "policies": policies,
            "summary": summary}


def _checkpoint_members(checkpoint: Path, stage: str) -> dict[str, str]:
    members = {item.relative_to(checkpoint).as_posix(): file_sha256(item)
               for item in sorted(checkpoint.rglob("*")) if item.is_file()}
    branch_count = ("delta1", "delta2", "delta3").index(stage) + 1
    expected = {"adapter_config.json", *(f"{name}.pt" for name in BRANCHES[:branch_count])}
    if set(members) != expected:
        raise PermissionError("Residual-stage checkpoint has unexpected or missing members")
    return members


def _audit_stage(row: Mapping[str, Any], *, source_commit: str,
                 model_tree_sha256: str,
                 qualification_receipt_sha256: str,
                 expected_plan: TrainingPlan) -> dict[str, Any]:
    identity = partition_identity(expected_plan)
    if row.get("partition_identity") != identity:
        raise PermissionError("Gold policy identity differs from canonical plan")
    run_root = Path(VOLUME) / "runs" / identity / "output"
    checkpoint = run_root / "checkpoint"
    receipt_path = run_root / "TRAINING_RECEIPT.json"
    rank_path = run_root / "RANK_0_RECEIPT.json"
    evidence_path = run_root / "DATABRICKS_EVIDENCE.json"
    terminal_path = run_root.parent / "TERMINAL.json"
    if not all(path.is_file() for path in (
            checkpoint / "adapter_config.json", receipt_path, rank_path,
            evidence_path, terminal_path)):
        raise PermissionError(f"Completed training output is absent for {identity}")
    config = _read(checkpoint / "adapter_config.json")
    config_payload = dict(config); config_hash = config_payload.pop("receipt_sha256", None)
    receipt = _read(receipt_path); rank = _read(rank_path); evidence = _read(evidence_path)
    terminal = _read(terminal_path)
    receipt_payload = dict(receipt); embedded_receipt = receipt_payload.pop(
        "receipt_sha256", None)
    receipt_payload.pop("checkpoint", None)
    receipt_payload.pop("checkpoint_metadata_sha256", None)
    evidence_payload = dict(evidence); embedded_evidence = evidence_payload.pop(
        "evidence_sha256", None)
    verified_rank = verify_rank_receipt(
        rank, expected_plan_sha256=expected_plan.full_plan_sha256,
        expected_global_journal_sha256=receipt.get("global_journal_sha256"))
    members = _checkpoint_members(checkpoint, expected_plan.stage)
    if (config_hash != digest(config_payload)
            or config.get("version") != CHECKPOINT_VERSION
            or config.get("condition") != expected_plan.condition
            or config.get("seed") != expected_plan.seed
            or config.get("stage") != expected_plan.stage
            or tuple(config.get("active_branches", ()))
               != tuple(BRANCHES[: ("delta1", "delta2", "delta3").index(
                   expected_plan.stage) + 1])
            or config.get("base_sha256") is None
            or config.get("source_sha256") != plan_source_sha(row, receipt)
            or config.get("plan_sha256") != expected_plan.full_plan_sha256
            or config.get("config", {}).get("model_id") != MODEL_ID
            or config.get("config", {}).get("model_revision") != MODEL_REVISION
            or embedded_receipt != digest(receipt_payload)
            or embedded_receipt != verified_rank.get("receipt_sha256")
            or row.get("receipt_sha256") != embedded_receipt
            or json.loads(row.get("receipt_json", "null")) != receipt
            or receipt.get("status") != "complete"
            or receipt.get("successful_optimizer_updates") != 144
            or receipt.get("plan_sha256") != expected_plan.full_plan_sha256
            or receipt.get("checkpoint") != config
            or receipt.get("checkpoint_metadata_sha256")
               != file_sha256(checkpoint / "adapter_config.json")
            or receipt.get("base_frozen_live_hash_pass") is not True
            or receipt.get("live_base_components_before")
               != receipt.get("live_base_components_after")
            or receipt.get("live_base_components_after", {}).get("combined_sha256")
               != config.get("base_sha256")
            or receipt.get("primary_save_reload_pass") is not True
            or receipt.get("checkpoint_save_reload", {}).get("pass") is not True
            or embedded_evidence != digest(evidence_payload)
            or row.get("evidence_sha256") != embedded_evidence
            or evidence.get("intent_sha256") != row.get("intent_sha256")
            or evidence.get("plan_sha256") != expected_plan.full_plan_sha256
            or evidence.get("rank_receipt_sha256") != embedded_receipt
            or evidence.get("live_base_components_before")
               != evidence.get("live_base_components_after")
            or evidence.get("live_base_components_after", {}).get("combined_sha256")
               != config.get("base_sha256")
            or evidence.get("save_reload_pass") is not True
            or terminal.get("exit_code") != 0
            or terminal.get("operator_stop_sent") is not False
            or row.get("source_commit") != source_commit
            or row.get("model_tree_sha256") != model_tree_sha256
            or row.get("qualification_receipt_sha256")
               != qualification_receipt_sha256
            or row.get("checkpoint_tree_sha256")
               != directory_tree_sha256(checkpoint)):
        raise PermissionError(f"Independent training replay failed for {identity}")
    return {
        "identity": identity, "condition": expected_plan.condition,
        "seed": expected_plan.seed, "stage": expected_plan.stage,
        "plan_sha256": expected_plan.full_plan_sha256,
        "checkpoint": checkpoint, "checkpoint_tree_sha256": row[
            "checkpoint_tree_sha256"], "checkpoint_members": members,
        "checkpoint_payload_members_sha256": digest(members),
        "config": config, "config_sha256": file_sha256(
            checkpoint / "adapter_config.json"),
        "training_receipt": receipt, "training_receipt_path": receipt_path,
        "training_receipt_sha256": file_sha256(receipt_path),
        "rank_receipt_sha256": file_sha256(rank_path),
        "evidence_sha256": file_sha256(evidence_path),
        "embedded_evidence_sha256": embedded_evidence,
        "base_component_sha256": config["base_sha256"],
        "source_sha256": config["source_sha256"],
        "optimizer_updates": receipt["successful_optimizer_updates"],
        "parent_checkpoint_tree_sha256": row.get(
            "parent_checkpoint_tree_sha256"),
    }


def plan_source_sha(row: Mapping[str, Any], receipt: Mapping[str, Any]) -> str:
    """Return the one source digest shared by Gold, receipt, and checkpoint."""
    source = receipt.get("checkpoint", {}).get("source_sha256")
    if not _sha(source) or row.get("plan_sha256") != receipt.get("plan_sha256"):
        raise PermissionError("Gold/receipt source-plan binding is absent")
    return source


def recompute_training_audits(matrix: Mapping[str, Any],
                              plan: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    plans = canonical_primary_plans()
    by_key = {(row["condition"], row["seed"], row["stage"]): row
              for row in matrix["policies"]}
    audited: dict[tuple[str, int, str], dict[str, Any]] = {}
    qualification_sha = matrix["capacity"].get("receipt_sha256")
    for key, expected in plans.items():
        audited[key] = _audit_stage(
            by_key[key], source_commit=plan["source_commit"],
            model_tree_sha256=plan["model_tree_sha256"],
            qualification_receipt_sha256=qualification_sha,
            expected_plan=expected)
    endpoints: dict[tuple[str, int], dict[str, Any]] = {}
    for condition in CONDITIONS:
        for seed in TRAINING_SEEDS:
            chain = [audited[(condition, seed, stage)]
                     for stage in ("delta1", "delta2", "delta3")]
            parent = None
            for item in chain:
                if item["parent_checkpoint_tree_sha256"] != parent:
                    raise PermissionError("Recomputed residual predecessor chain differs")
                parent = item["checkpoint_tree_sha256"]
            endpoints[(condition, seed)] = {
                "endpoint": chain[-1], "chain": chain,
            }
    return endpoints


def build_blinding_map() -> tuple[dict[tuple[str, int | None], str],
                                  dict[str, dict[str, Any]]]:
    cells: list[tuple[str, int | None]] = [(BASELINE_CONDITION, None)] + [
        (condition, training_seed) for condition in CONDITIONS
        for training_seed in TRAINING_SEEDS]
    shuffled = list(cells)
    secrets.SystemRandom().shuffle(shuffled)
    mapping = {}
    unblinding = {}
    for cell in shuffled:
        opaque = "blind-" + secrets.token_hex(16)
        while opaque in unblinding:
            opaque = "blind-" + secrets.token_hex(16)
        mapping[cell] = opaque
        unblinding[opaque] = {
            "condition": cell[0], "training_seed": cell[1],
        }
    validate_unblinding_key(unblinding, set(unblinding))
    return mapping, dict(sorted(unblinding.items()))


def _training_separation_receipt(training_rows: Sequence[Mapping[str, Any]], *,
                                 source_commit: str) -> dict[str, Any]:
    rows = sorted((dict(row) for row in training_rows),
                  key=lambda row: str(row.get("record_id", "")))
    if (len(rows) != 6912 or len({row.get("record_id") for row in rows}) != 6912
            or {row.get("split") for row in rows} != {"train"}
            or {row.get("condition") for row in rows} != set(CONDITIONS)):
        raise PermissionError("Training separation requires the exact 6,912 Silver rows")
    proof = assert_training_has_no_transfer_choices(rows)
    value = {
        "version": "research4-transfer-training-separation-v2",
        "source_commit": source_commit, "training_record_count": len(rows),
        "training_records_sha256": digest(rows), **proof,
    }
    value["receipt_sha256"] = digest(value)
    return value


def _package_files(root: Path, *, exclude: Sequence[str] = ()) -> list[dict[str, Any]]:
    return directory_tree(root, exclude=exclude)


def _serialized_json_sha256(value: Mapping[str, Any]) -> str:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sanitized_evaluation_config(value: Mapping[str, Any]) -> dict[str, Any]:
    endpoint = value["endpoint"]
    original = endpoint["config"]
    public = {
        "version": EVALUATION_CHECKPOINT_VERSION,
        "config": original["config"], "stage": "delta3",
        "target_names": original["target_names"],
        "base_sha256": original["base_sha256"],
        "source_sha256": original["source_sha256"],
        "active_branches": list(BRANCHES),
        "branches": original["branches"],
        "adapter_sha256": original["adapter_sha256"],
        "evaluation_only": True, "training_lineage_bytes_included": False,
    }
    public["receipt_sha256"] = digest(public)
    return public


def _copy_endpoint(destination: Path, value: Mapping[str, Any]) -> dict[str, str]:
    endpoint = value["endpoint"]
    destination.mkdir(parents=True, exist_ok=False)
    for relative in sorted(endpoint["checkpoint_members"]):
        if relative == "adapter_config.json":
            continue
        source = endpoint["checkpoint"] / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if file_sha256(target) != endpoint["checkpoint_members"][relative]:
            raise PermissionError("Checkpoint bytes changed during Volume publication")
    _write(destination / "adapter_config.json",
           _sanitized_evaluation_config(value))
    return {item.relative_to(destination).as_posix(): file_sha256(item)
            for item in sorted(destination.rglob("*")) if item.is_file()}


def _build_training_audit(value: Mapping[str, Any], *, matrix_sha256: str,
                          source_commit: str) -> dict[str, Any]:
    endpoint = value["endpoint"]
    chain = [{
        "identity": item["identity"], "stage": item["stage"],
        "plan_sha256": item["plan_sha256"],
        "checkpoint_tree_sha256": item["checkpoint_tree_sha256"],
        "parent_checkpoint_tree_sha256": item[
            "parent_checkpoint_tree_sha256"],
        "training_receipt_sha256": item["training_receipt_sha256"],
        "rank_receipt_sha256": item["rank_receipt_sha256"],
        "evidence_sha256": item["evidence_sha256"],
        "optimizer_updates": item["optimizer_updates"],
    } for item in value["chain"]]
    audit = {
        "version": TRAINING_AUDIT_VERSION, "complete": True,
        "condition": endpoint["condition"], "seed": endpoint["seed"],
        "source_commit": source_commit, "source_sha256": endpoint["source_sha256"],
        "plan_sha256": endpoint["plan_sha256"],
        "optimizer_updates": endpoint["optimizer_updates"],
        "cumulative_stage_optimizer_updates": sum(
            item["optimizer_updates"] for item in value["chain"]),
        "base_frozen": True, "self_attested_pass_flags_used": False,
        "training_receipt_sha256": endpoint["training_receipt_sha256"],
        "checkpoint_config_sha256": endpoint["config_sha256"],
        "checkpoint_payload_members_sha256": endpoint[
            "checkpoint_payload_members_sha256"],
        "endpoint_checkpoint_tree_sha256": endpoint["checkpoint_tree_sha256"],
        "gold_matrix_summary_sha256": matrix_sha256,
        "three_residual_stage_chain": chain,
        "audit_method": "independent_cpu_replay_of_gold_receipts_evidence_and_bytes",
    }
    audit["audit_payload_sha256"] = digest(audit)
    return audit


def assert_public_package_blinded(root: str | Path) -> dict[str, Any]:
    """Reject any public text member that reveals a training condition or seed."""
    base = Path(root)
    checked = []
    forbidden = (*CONDITIONS, *(str(seed) for seed in TRAINING_SEEDS))
    for path in sorted(item for item in base.rglob("*") if item.is_file()
                       and item.suffix.lower() in {".json", ".jsonl", ".txt", ".md"}):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            raise PermissionError("Public transfer package reveals condition or seed")
        checked.append(path.relative_to(base).as_posix())
    if not checked:
        raise PermissionError("Public transfer package has no auditable text members")
    return {"text_member_count": len(checked),
            "text_members_sha256": digest(checked), "blinded": True}


def _publish_private_key(source: Path, final_root: Path,
                         expected_sha256: str) -> bool:
    """Publish one private key and return whether this call created its root."""
    target = final_root / "UNBLINDING_KEY.json"
    if final_root.exists():
        if not target.is_file() or file_sha256(target) != expected_sha256:
            raise PermissionError("Existing separate unblinding key differs")
        return False
    final_root.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(source, target)
        if file_sha256(target) != expected_sha256:
            raise PermissionError("Published unblinding key readback differs")
        return True
    except BaseException:
        shutil.rmtree(final_root, ignore_errors=True)
        raise


def _publish_public_package(staging: Path, final_root: Path) -> None:
    """Copy payload bytes, write the receipt last, and clean owned partials."""
    if final_root.exists():
        raise FileExistsError("Public package destination is already claimed")
    final_root.mkdir(parents=True, exist_ok=False)
    try:
        for source in sorted(staging.iterdir(), key=lambda path: path.name):
            if source.name == "PACKAGING_RECEIPT.json":
                continue
            target = final_root / source.name
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        # The receipt is the commit marker and is published only after every
        # content-addressed payload byte has been copied.
        with (final_root / "PACKAGING_RECEIPT.json").open(
                "x", encoding="utf-8") as stream:
            stream.write((staging / "PACKAGING_RECEIPT.json").read_text(
                encoding="utf-8"))
            stream.flush(); os.fsync(stream.fileno())
    except BaseException:
        shutil.rmtree(final_root, ignore_errors=True)
        raise


def _publish_public_with_key_rollback(staging: Path, final_root: Path,
                                      final_key_root: Path,
                                      key_owned_by_call: bool,
                                      verifier=None):
    public_preexisting = final_root.exists()
    try:
        _publish_public_package(staging, final_root)
        return verifier(final_root) if verifier is not None else None
    except BaseException:
        if not public_preexisting:
            shutil.rmtree(final_root, ignore_errors=True)
        if key_owned_by_call:
            shutil.rmtree(final_key_root, ignore_errors=True)
        raise


def _claim_package_identity(plan: Mapping[str, Any], matrix_sha256: str
                            ) -> tuple[dict[str, Any] | None, Path | None, Path]:
    identity = digest({
        "version": "research4-transfer-package-identity-v1",
        "plan_sha256": plan["plan_sha256"],
        "gold_matrix_summary_sha256": matrix_sha256,
    })
    root = Path(TRANSFER_PACKAGE_LOCK_PARENT)
    active_root = root / "active"; completed_root = root / "completed"
    active_root.mkdir(parents=True, exist_ok=True)
    completed_root.mkdir(parents=True, exist_ok=True)
    active = active_root / f"{identity}.lock"
    completed = completed_root / f"{identity}.json"
    if completed.exists():
        value = _read(completed); payload = dict(value)
        embedded = payload.pop("receipt_sha256", None)
        if (embedded != digest(payload) or value.get("identity") != identity
                or value.get("plan_sha256") != plan["plan_sha256"]
                or value.get("gold_matrix_summary_sha256") != matrix_sha256):
            raise PermissionError("Completed package identity receipt differs")
        receipt = verify_published_package(value["package_root"])
        if (receipt["receipt_sha256"] != value.get("packaging_receipt_sha256")
                or receipt["package_content_sha256"]
                   != value.get("package_content_sha256")):
            raise PermissionError("Completed package identity points to different bytes")
        return receipt, None, completed
    descriptor = os.open(active, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(canonical({
            "identity": identity, "plan_sha256": plan["plan_sha256"],
            "gold_matrix_summary_sha256": matrix_sha256,
            "owner_token": secrets.token_hex(16),
        }) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    return None, active, completed


def _complete_package_identity(completed: Path, plan: Mapping[str, Any],
                               matrix_sha256: str,
                               receipt: Mapping[str, Any]) -> None:
    identity = completed.stem
    value = {
        "version": "research4-transfer-package-identity-complete-v1",
        "identity": identity, "plan_sha256": plan["plan_sha256"],
        "gold_matrix_summary_sha256": matrix_sha256,
        "package_content_sha256": receipt["package_content_sha256"],
        "package_root": receipt["package_root"],
        "packaging_receipt_sha256": receipt["receipt_sha256"],
    }
    value["receipt_sha256"] = digest(value)
    _write(completed, value)


def publish_transfer_package(*, plan: Mapping[str, Any],
                             base_receipt: Mapping[str, Any],
                             capacity_receipt: Mapping[str, Any],
                             gold_rows: Sequence[Mapping[str, Any]],
                             summary_rows: Sequence[Mapping[str, Any]],
                             training_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Publish or exactly verify one deterministic content-addressed package."""
    authority = validate_packaging_plan(plan)
    if (file_sha256(authority["base_qualification_receipt_path"])
            != authority["base_qualification_receipt_file_sha256"]
            or file_sha256(authority["capacity_qualification_receipt_path"])
               != authority["capacity_qualification_receipt_file_sha256"]):
        raise PermissionError("Packaging gate receipt bytes differ from the plan")
    matrix = validate_gold_matrix(
        plan=authority, base_receipt=base_receipt,
        capacity_receipt=capacity_receipt, gold_rows=gold_rows,
        summary_rows=summary_rows)
    endpoints = recompute_training_audits(matrix, authority)
    separation = _training_separation_receipt(
        training_rows, source_commit=authority["source_commit"])
    parent = Path(authority["package_parent"]); parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".transfer-package-", dir=parent))
    key_parent = Path(TRANSFER_UNBLINDING_PARENT); key_parent.mkdir(
        parents=True, exist_ok=True)
    key_staging = Path(tempfile.mkdtemp(prefix=".unblinding-key-", dir=key_parent))
    owned_final_key = False
    published_public = False
    final_root = None
    final_key_root = None
    try:
        existing, active_identity, completed_identity = _claim_package_identity(
            authority, matrix["summary"]["summary_sha256"])
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(key_staging, ignore_errors=True)
        raise
    if existing is not None:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(key_staging, ignore_errors=True)
        return existing
    try:
        blinding, unblinding = build_blinding_map()
        checkpoint_root = staging / "checkpoints"; checkpoint_root.mkdir()
        unblinding_path = key_staging / "UNBLINDING_KEY.json"
        _write(unblinding_path, unblinding)
        manifest_rows = [{
            "blind_checkpoint_id": blinding[(BASELINE_CONDITION, None)],
            "identity_kind": "frozen_base", "relative_directory": None,
            "members": {}, "members_sha256": digest({}),
            "base_model_sha256": matrix["base"]["base_component_sha256_after"],
            "base_qualification_sha256": matrix["base"]["receipt_sha256"],
            "no_adapter_checkpoint": True,
        }]
        for cell, value in sorted(endpoints.items()):
            blind = blinding[cell]
            relative = f"checkpoints/{blind}"
            members = _copy_endpoint(staging / relative, value)
            endpoint = value["endpoint"]
            manifest_rows.append({
                "blind_checkpoint_id": blind, "identity_kind": "adapted",
                "relative_directory": relative, "members": members,
                "members_sha256": digest(members),
                "config_sha256": members["adapter_config.json"],
                "source_sha256": endpoint["source_sha256"],
            })
        checkpoint_tree = directory_tree_sha256(checkpoint_root)
        checkpoint_manifest = {
            "version": CHECKPOINT_MANIFEST_VERSION,
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "base_frozen": True, "all_adapters_frozen": True,
            "evaluation_only": True, "manifest_blinded": True,
            "condition_seed_fields_exposed_in_manifest": False,
            "evaluation_identity_count": CHECKPOINT_COUNT,
            "adapted_endpoint_count": 9,
            "unique_frozen_base_endpoint_count": 1,
            "base_model_sha256": matrix["base"]["base_component_sha256_after"],
            "base_qualification_sha256": matrix["base"]["receipt_sha256"],
            "gold_matrix_summary_sha256": matrix["summary"]["summary_sha256"],
            "checkpoint_package_sha256": checkpoint_tree,
            "unblinding_key_sha256": file_sha256(unblinding_path),
            "checkpoints": sorted(manifest_rows,
                                  key=lambda row: row["blind_checkpoint_id"]),
        }
        _write(staging / "CHECKPOINT_MANIFEST.json", checkpoint_manifest)
        registry_root = staging / "registry"
        commitment = write_sealed_registry(registry_root, authority["registry_seed"])
        registries = {arm: _read(registry_root / "sealed" / f"{arm}.json")
                      for arm in ALL_ARMS}
        registry_hashes = validate_registry_bundle(registries)
        validate_public_commitment(
            _read(registry_root / "SEALED_REGISTRY_COMMITMENT.json"),
            registry_root / "sealed", registries)
        _write(staging / "TRAINING_SEPARATION_RECEIPT.json", separation)

        payload_files = _package_files(staging)
        package_content_sha = digest({
            "version": PACKAGE_FILES_VERSION, "files": payload_files})
        package_files = {
            "version": PACKAGE_FILES_VERSION, "files": payload_files,
            "file_count": len(payload_files),
            "total_bytes": sum(item["bytes"] for item in payload_files),
            "package_content_sha256": package_content_sha,
        }
        _write(staging / "PACKAGE_FILES.json", package_files)
        final_root = parent / package_content_sha
        final_key_root = key_parent / package_content_sha
        final_unblinding_path = final_key_root / "UNBLINDING_KEY.json"
        receipt = {
            "version": PACKAGING_RECEIPT_VERSION, "status": "complete",
            "identity": authority["identity"], "cpu_only": True,
            "gpu_runs_started": 0, "model_forwards": 0, "optimizer_updates": 0,
            "source_commit": authority["source_commit"],
            "source_sha256": authority["source_sha256"],
            "source_manifest": authority["source_manifest"],
            "model_tree_sha256": authority["model_tree_sha256"],
            "base_qualification_receipt_sha256": matrix["base"]["receipt_sha256"],
            "capacity_qualification_receipt_sha256": matrix["capacity"][
                "receipt_sha256"],
            "gold_matrix_summary_sha256": matrix["summary"]["summary_sha256"],
            "policy_partition_count": 27, "adapted_endpoint_count": 9,
            "frozen_base_endpoint_count": 1,
            "checkpoint_package_sha256": checkpoint_tree,
            "checkpoint_manifest_sha256": file_sha256(
                staging / "CHECKPOINT_MANIFEST.json"),
            "unblinding_key_sha256": file_sha256(unblinding_path),
            "unblinding_key_path": final_unblinding_path.as_posix(),
            "sealed_registry_sha256": digest(registry_hashes),
            "sealed_commitment_sha256": commitment["commitment_sha256"],
            "training_separation_receipt_sha256": file_sha256(
                staging / "TRAINING_SEPARATION_RECEIPT.json"),
            "package_content_sha256": package_content_sha,
            "package_files_sha256": file_sha256(staging / "PACKAGE_FILES.json"),
            "package_root": final_root.as_posix(),
        }
        receipt["receipt_sha256"] = digest(receipt)
        _write(staging / "PACKAGING_RECEIPT.json", receipt)
        assert_public_package_blinded(staging)
        owned_final_key = _publish_private_key(
            unblinding_path, final_key_root, receipt["unblinding_key_sha256"])
        shutil.rmtree(key_staging, ignore_errors=True)
        if final_root.exists():
            shutil.rmtree(staging, ignore_errors=True)
            verified = verify_published_package(final_root)
            if verified != receipt:
                raise PermissionError("Existing content-addressed package differs")
            _complete_package_identity(
                completed_identity, authority,
                matrix["summary"]["summary_sha256"], verified)
            return verified
        verified = _publish_public_with_key_rollback(
            staging, final_root, final_key_root, owned_final_key,
            verifier=verify_published_package)
        published_public = True
        shutil.rmtree(staging, ignore_errors=True)
        _complete_package_identity(
            completed_identity, authority,
            matrix["summary"]["summary_sha256"], verified)
        return verified
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if key_staging.exists():
            shutil.rmtree(key_staging, ignore_errors=True)
        if owned_final_key and final_key_root is not None and not published_public:
            shutil.rmtree(final_key_root, ignore_errors=True)
        raise
    finally:
        if active_identity is not None:
            active_identity.unlink(missing_ok=True)


def verify_published_package(package_root: str | Path) -> dict[str, Any]:
    root = Path(package_root)
    receipt = _read(root / "PACKAGING_RECEIPT.json")
    payload = dict(receipt); embedded = payload.pop("receipt_sha256", None)
    package_files = _read(root / "PACKAGE_FILES.json")
    declared = package_files.get("files")
    if (embedded != digest(payload)
            or receipt.get("version") != PACKAGING_RECEIPT_VERSION
            or receipt.get("status") != "complete"
            or not isinstance(declared, list)
            or package_files.get("version") != PACKAGE_FILES_VERSION
            or package_files.get("file_count") != len(declared)
            or package_files.get("total_bytes")
               != sum(item.get("bytes", -1) for item in declared)):
        raise PermissionError("Transfer packaging receipt or inventory differs")
    actual = _package_files(root, exclude=("PACKAGE_FILES.json", "PACKAGING_RECEIPT.json"))
    content_sha = digest({"version": PACKAGE_FILES_VERSION, "files": actual})
    if (actual != declared
            or package_files.get("package_content_sha256") != content_sha
            or receipt.get("package_content_sha256") != content_sha
            or root.name != content_sha
            or receipt.get("package_root") != root.as_posix()
            or receipt.get("package_files_sha256")
               != file_sha256(root / "PACKAGE_FILES.json")
            or receipt.get("checkpoint_package_sha256")
               != directory_tree_sha256(root / "checkpoints")
            or receipt.get("checkpoint_manifest_sha256")
               != file_sha256(root / "CHECKPOINT_MANIFEST.json")
            or receipt.get("training_separation_receipt_sha256")
               != file_sha256(root / "TRAINING_SEPARATION_RECEIPT.json")):
        raise PermissionError("Content-addressed transfer package replay differs")
    manifest = _read(root / "CHECKPOINT_MANIFEST.json")
    if (manifest.get("manifest_blinded") is not True
            or any("condition" in row or "training_seed" in row
                   for row in manifest.get("checkpoints", ()))
            or len(manifest.get("checkpoints", ())) != CHECKPOINT_COUNT):
        raise PermissionError("Checkpoint manifest is not exactly blinded")
    assert_public_package_blinded(root)
    unblinding_path = Path(_volume_path(
        receipt.get("unblinding_key_path"), label="separate unblinding key"))
    if (unblinding_path.parent.name != content_sha
            or unblinding_path.name != "UNBLINDING_KEY.json"
            or not unblinding_path.is_file()
            or file_sha256(unblinding_path) != receipt.get("unblinding_key_sha256")
            or manifest.get("unblinding_key_sha256")
               != receipt.get("unblinding_key_sha256")):
        raise PermissionError("Separate content-bound unblinding key differs")
    unblinding = _read(unblinding_path)
    validate_unblinding_key(
        unblinding, {row["blind_checkpoint_id"] for row in manifest["checkpoints"]})
    return receipt
