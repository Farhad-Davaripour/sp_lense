"""Reviewed frozen Qwen inference worker for the sealed continuity transfer grid.

The worker performs no provider operation and has no real tool implementation.
It loads the frozen base plus one source-bound ordinal adapter checkpoint at a
time, generates exactly one bounded native response for every committed trial,
parses one registered symbolic action without repair, and writes raw bytes and
hashes before returning.  The surrounding controller owns the external stop.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import threading
import time

from .action_serialization import (
    SERIALIZATION_CONTRACT_VERSION,
    parse_registered_action,
)
from .qualification_technical_successor import validate as validate_technical_successor
from .continuity_transfer_evaluation import (
    ALL_ARMS, BASELINE_CONDITION, CHECKPOINT_COUNT, CONDITIONS, EXPECTED_RAW_RECORDS,
    TRAINING_SEEDS, VERSION as EVALUATION_VERSION,
    canonical, digest, file_sha256, prompt_for_trial, validate_registry_bundle,
)
from .qwen9b_ordinal_continuity_lora import (
    BRANCHES, CHECKPOINT_VERSION, EVALUATION_CHECKPOINT_VERSION,
    MODEL_ID, MODEL_REVISION,
    OrdinalContinuityLoRAConfig, OrdinalContinuityModel,
    assert_adapter_branches_zero, bypass, clear_adapter_branches,
    install_ordinal_lora, load_adapter_checkpoint,
    load_evaluation_adapter_checkpoint, model_component_sha256,
    qwen35_text_target_names,
)


WORKER_VERSION = "research4-continuity-transfer-inference-worker-v4"
TRAINING_AUDIT_VERSION = "qwen35-9b-three-residual-training-audit-v3"
BLINDED_CHECKPOINT_MANIFEST_VERSION = (
    "research4-three-residual-blinded-checkpoints-v1"
)
RUN_BINDING_INPUT_KEYS = frozenset({
    "source_archive", "sealed_registry_archive", "checkpoint_archive",
    "checkpoint_manifest", "resource_plan", "source_manifest", "sealed_manifest",
})
RUN_BINDING_FIELDS = frozenset({
    "version", "candidate", "account", "notebook", "source_commit",
    "preparation_identity_sha256", "expected_input_sha256",
    "resource_plan_sha256",
})
PROVIDER_RUN_BINDING_FIELDS = frozenset({
    "candidate", "account", "notebook", "scriptVersionId", "privateVersion",
    "committed_notebook_version", "preparation_sha256", "live_observation_sha256",
    "reserved_at_UTC", "notebook_code_sha256", "readback_notebook_code_sha256",
    "attached_input_ids", "attached_input_versions", "attached_input_sha256",
    "reservation_receipt_sha256", "controller_receipt_sha256",
    "pre_save_binding", "pre_save_binding_sha256",
    "provider_reconciliation_receipt_sha256",
})
DATABRICKS_RUN_BINDING_FIELDS = frozenset({
    "version", "execution_platform", "identity", "job_run_id",
    "source_commit", "intent_sha256", "source_sha256",
    "model_tree_sha256", "model_receipt_sha256",
    "base_qualification_receipt_sha256",
    "base_qualification_carryforward_sha256",
    "qualification_technical_successor",
    "qualification_technical_successor_sha256", "checkpoint_manifest_sha256",
    "checkpoint_package_sha256", "sealed_registry_sha256",
    "resource_plan_sha256", "reservation_sha256", "output_root",
})


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_run_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable pre-Save identity embedded in every raw row.

    Provider script/private versions cannot exist before the Save that starts
    this notebook.  They are deliberately absent here and are reconciled by
    :func:`validate_provider_run_binding` during the independent local audit.
    """
    binding = dict(value)
    if binding.get("version") == "research4-databricks-continuity-transfer-run-binding-v4":
        return validate_databricks_run_binding(binding)
    if (set(binding) != RUN_BINDING_FIELDS
            or any(binding.get(key) in (None, "") for key in RUN_BINDING_FIELDS)
            or binding.get("version") != "research4-continuity-transfer-presave-binding-v1"
            or not all(isinstance(binding.get(key), str) and binding[key]
                       for key in ("candidate", "account", "notebook"))
            or not isinstance(binding.get("source_commit"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", binding["source_commit"])
            or set(binding.get("expected_input_sha256", {})) != RUN_BINDING_INPUT_KEYS
            or any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item)
                   for item in binding["expected_input_sha256"].values())
            or any(not isinstance(binding.get(key), str)
                   or not re.fullmatch(r"[0-9a-f]{64}", binding[key])
                   for key in ("preparation_identity_sha256", "resource_plan_sha256"))):
        raise PermissionError("Exact immutable pre-Save candidate/input/preparation binding required")
    return json.loads(canonical(binding))


def validate_databricks_run_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact one-use Databricks job/reservation identity.

    Unlike the historical Kaggle pre-Save identity, a Databricks job run ID is
    available before the GPU worker starts.  The same complete binding is
    therefore embedded in every raw record and independently reconstructed by
    the downstream CPU audit.
    """
    binding = dict(value)
    sha_fields = {
        "intent_sha256", "source_sha256", "model_tree_sha256",
        "model_receipt_sha256", "base_qualification_receipt_sha256",
        "base_qualification_carryforward_sha256",
        "qualification_technical_successor_sha256",
        "checkpoint_manifest_sha256", "checkpoint_package_sha256",
        "sealed_registry_sha256", "resource_plan_sha256",
        "reservation_sha256",
    }
    technical = validate_technical_successor(
        binding.get("qualification_technical_successor"))
    if (set(binding) != DATABRICKS_RUN_BINDING_FIELDS
            or binding.get("execution_platform") != "databricks"
            or not isinstance(binding.get("identity"), str)
            or not re.fullmatch(r"[a-z0-9_]+", binding["identity"])
            or not isinstance(binding.get("job_run_id"), str)
            or not binding["job_run_id"]
            or not re.fullmatch(r"[0-9a-f]{40}", str(binding.get("source_commit", "")))
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(binding.get(key, "")))
                   for key in sha_fields)
            or technical["technical_successor_sha256"]
               != binding.get("qualification_technical_successor_sha256")
            or not isinstance(binding.get("output_root"), str)
            or not binding["output_root"].startswith("/Volumes/")
            or ".." in Path(binding["output_root"]).parts):
        raise PermissionError("Exact Databricks transfer run/reservation binding required")
    return json.loads(canonical(binding))


def validate_provider_run_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the post-Save provider identity used only by the raw audit."""
    binding = dict(value)
    if binding.get("version") == "research4-databricks-continuity-transfer-run-binding-v4":
        return validate_databricks_run_binding(binding)
    sha_fields = {
        "preparation_sha256", "live_observation_sha256", "notebook_code_sha256",
        "readback_notebook_code_sha256", "reservation_receipt_sha256",
        "controller_receipt_sha256", "pre_save_binding_sha256",
        "provider_reconciliation_receipt_sha256",
    }
    pre = validate_run_binding(binding.get("pre_save_binding", {}))
    if (set(binding) != PROVIDER_RUN_BINDING_FIELDS
            or any(binding.get(key) in (None, "") for key in PROVIDER_RUN_BINDING_FIELDS)
            or type(binding.get("scriptVersionId")) is not int
            or binding["scriptVersionId"] <= 0
            or type(binding.get("privateVersion")) is not int
            or binding["privateVersion"] <= 0
            or binding.get("committed_notebook_version") != binding["scriptVersionId"]
            or binding.get("candidate") != pre["candidate"]
            or binding.get("account") != pre["account"]
            or binding.get("notebook") != pre["notebook"]
            or digest(pre) != binding.get("pre_save_binding_sha256")
            or any(set(binding.get(key, {})) != RUN_BINDING_INPUT_KEYS
                   for key in ("attached_input_ids", "attached_input_versions",
                               "attached_input_sha256"))
            or any(type(item) is not int or item <= 0
                   for key in ("attached_input_ids", "attached_input_versions")
                   for item in binding[key].values())
            or binding.get("attached_input_sha256") != pre["expected_input_sha256"]
            or any(not isinstance(binding.get(key), str)
                   or not re.fullmatch(r"[0-9a-f]{64}", binding[key])
                   for key in sha_fields)
            or binding["notebook_code_sha256"] != binding["readback_notebook_code_sha256"]):
        raise PermissionError("Exact post-Save provider/version/readback binding required")
    return json.loads(canonical(binding))


def parse_native_action_detail(raw_response: str, allowed: Sequence[str]) -> dict[str, Any]:
    """Apply the exact shared v2 parser and retain normalization evidence."""
    return parse_registered_action(raw_response, allowed)


def parse_native_action(raw_response: str, allowed: Sequence[str]) -> tuple[str | None, bool]:
    """Compatibility view of the exact shared v2 serialization contract."""
    parsed = parse_native_action_detail(raw_response, allowed)
    return parsed["action"], parsed["valid"]


def verify_training_lineage(
    row: Mapping[str, Any], directory: str | Path, config: Mapping[str, Any], *,
    receipt_path: str | Path | None = None, audit_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute the actual training receipt and independent audit lineage."""
    root = Path(directory)
    receipt_relative = row.get("training_receipt_relative_path")
    audit_relative = row.get("training_audit_relative_path")
    if (receipt_relative != "lineage/TRAINING_RECEIPT.json"
            or audit_relative != "lineage/TRAINING_AUDIT.json"):
        raise PermissionError("Exact packaged training lineage paths are required")
    receipt_path = Path(receipt_path) if receipt_path is not None else root / receipt_relative
    audit_path = Path(audit_path) if audit_path is not None else root / audit_relative
    if (not receipt_path.is_file() or not audit_path.is_file()
            or file_sha256(receipt_path) != row.get("training_receipt_sha256")
            or file_sha256(audit_path) != row.get("training_audit_sha256")):
        raise PermissionError("Training receipt/audit bytes differ from the manifest")
    receipt = _read(receipt_path)
    receipt_payload = dict(receipt)
    embedded_receipt = receipt_payload.pop("receipt_sha256", None)
    # The worker self-hashes the rank receipt before attaching the separately
    # content-addressed checkpoint metadata.
    receipt_payload.pop("checkpoint", None)
    receipt_payload.pop("checkpoint_metadata_sha256", None)
    if (embedded_receipt != digest(receipt_payload)
            or receipt.get("status") != "complete"
            or receipt.get("plan_sha256") != config.get("plan_sha256")
            or type(receipt.get("successful_optimizer_updates")) is not int
            or receipt["successful_optimizer_updates"] <= 0
            or receipt.get("checkpoint") != config
            or receipt.get("checkpoint_metadata_sha256") != file_sha256(root / "adapter_config.json")):
        raise PermissionError("Training receipt is not an exact completed checkpoint lineage")
    audit = _read(audit_path)
    audit_payload = dict(audit)
    embedded_audit = audit_payload.pop("audit_payload_sha256", None)
    expected_condition = config.get("condition")
    expected_seed = config.get("seed")
    if (embedded_audit != digest(audit_payload)
            or audit.get("version") != TRAINING_AUDIT_VERSION
            or audit.get("complete") is not True
            or audit.get("training_receipt_sha256") != file_sha256(receipt_path)
            or audit.get("checkpoint_config_sha256") != file_sha256(root / "adapter_config.json")
            or audit.get("checkpoint_payload_members_sha256") != row.get("checkpoint_payload_members_sha256")
            or audit.get("condition") != expected_condition
            or audit.get("seed") != expected_seed
            or audit.get("source_sha256") != config.get("source_sha256")
            or audit.get("plan_sha256") != config.get("plan_sha256")
            or audit.get("optimizer_updates") != receipt.get("successful_optimizer_updates")
            or audit.get("base_frozen") is not True
            or audit.get("self_attested_pass_flags_used") is not False):
        raise PermissionError("Independent training audit does not bind the actual receipt/checkpoint")
    return {
        "training_receipt_sha256": file_sha256(receipt_path),
        "training_audit_sha256": file_sha256(audit_path),
        "training_audit_payload_sha256": embedded_audit,
    }


def load_registry_bundle(sealed_root: str | Path) -> tuple[dict[str, Any], dict[str, str]]:
    root = Path(sealed_root)
    bundle = {arm: _read(root / f"{arm}.json") for arm in ALL_ARMS}
    return bundle, validate_registry_bundle(bundle)


def _verify_blinded_checkpoint_manifest(
        manifest: Mapping[str, Any], checkpoint_root: str | Path) -> list[dict[str, Any]]:
    """Verify public checkpoint bytes without reading private identity metadata."""
    exact_top = {
        "version", "model_id", "model_revision", "base_frozen",
        "all_adapters_frozen", "evaluation_only", "manifest_blinded",
        "condition_seed_fields_exposed_in_manifest", "evaluation_identity_count",
        "adapted_endpoint_count", "unique_frozen_base_endpoint_count",
        "base_model_sha256", "base_qualification_sha256",
        "gold_matrix_summary_sha256", "checkpoint_package_sha256",
        "unblinding_key_sha256", "checkpoints",
    }
    base_sha = manifest.get("base_model_sha256")
    base_gate = manifest.get("base_qualification_sha256")
    if (set(manifest) != exact_top
            or manifest.get("version") != BLINDED_CHECKPOINT_MANIFEST_VERSION
            or manifest.get("model_id") != MODEL_ID
            or manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("base_frozen") is not True
            or manifest.get("all_adapters_frozen") is not True
            or manifest.get("evaluation_only") is not True
            or manifest.get("manifest_blinded") is not True
            or manifest.get("condition_seed_fields_exposed_in_manifest") is not False
            or manifest.get("evaluation_identity_count") != CHECKPOINT_COUNT
            or manifest.get("adapted_endpoint_count") != 9
            or manifest.get("unique_frozen_base_endpoint_count") != 1
            or not isinstance(base_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", base_sha)
            or not isinstance(base_gate, str) or not re.fullmatch(r"[0-9a-f]{64}", base_gate)):
        raise PermissionError("Exact sanitized checkpoint manifest required")
    rows = manifest.get("checkpoints")
    if not isinstance(rows, list) or len(rows) != CHECKPOINT_COUNT:
        raise PermissionError("Sanitized manifest must contain one base and nine adapters")
    root = Path(checkpoint_root).resolve()
    verified = []; seen = set(); base_seen = False; adapted = 0
    for row in rows:
        blind = row.get("blind_checkpoint_id")
        if not isinstance(blind, str) or not blind or blind in seen:
            raise PermissionError("Sanitized checkpoint blind IDs must be unique")
        seen.add(blind)
        if row.get("identity_kind") == "frozen_base":
            expected = {
                "blind_checkpoint_id", "identity_kind", "relative_directory",
                "members", "members_sha256", "base_model_sha256",
                "base_qualification_sha256", "no_adapter_checkpoint",
            }
            if (set(row) != expected or base_seen
                    or row.get("relative_directory") is not None
                    or row.get("members") != {}
                    or row.get("members_sha256") != digest({})
                    or row.get("base_model_sha256") != base_sha
                    or row.get("base_qualification_sha256") != base_gate
                    or row.get("no_adapter_checkpoint") is not True):
                raise PermissionError("Sanitized frozen-base row differs")
            base_seen = True
            verified.append({**dict(row), "checkpoint_directory": None,
                             "config": None, "training_lineage": {
                                 "kind": "frozen_base_no_training",
                                 "base_model_sha256": base_sha,
                                 "base_qualification_sha256": base_gate}})
            continue
        expected = {
            "blind_checkpoint_id", "identity_kind", "relative_directory",
            "members", "members_sha256", "config_sha256", "source_sha256",
        }
        if row.get("identity_kind") != "adapted" or set(row) != expected:
            raise PermissionError("Sanitized adapted manifest row differs")
        relative = row.get("relative_directory")
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts):
            raise PermissionError("Sanitized checkpoint path is unsafe")
        directory = (root / relative).resolve()
        if not directory.is_relative_to(root) or not directory.is_dir():
            raise PermissionError("Sanitized checkpoint directory is absent")
        members = row.get("members")
        expected_names = {"adapter_config.json", "delta1.pt", "delta2.pt", "delta3.pt"}
        actual = {item.relative_to(directory).as_posix(): file_sha256(item)
                  for item in sorted(directory.rglob("*")) if item.is_file()}
        if (not isinstance(members, dict) or set(members) != expected_names
                or actual != members or digest(members) != row.get("members_sha256")
                or any("lineage" in name.lower() for name in members)):
            raise PermissionError("Sanitized checkpoint member inventory differs")
        config = _read(directory / "adapter_config.json")
        config_geometry = config.get("config")
        try:
            allowed = {field.name for field in fields(OrdinalContinuityLoRAConfig)}
            parsed_geometry = OrdinalContinuityLoRAConfig(
                **{key: item for key, item in dict(config_geometry).items()
                   if key in allowed}).validate()
        except (TypeError, ValueError, PermissionError) as exc:
            raise PermissionError("Sanitized evaluation geometry differs") from exc
        branch_metadata = config.get("branches")
        encoded_config = canonical(config)
        if (file_sha256(directory / "adapter_config.json") != row.get("config_sha256")
                or config.get("version") != EVALUATION_CHECKPOINT_VERSION
                or config_geometry != asdict(parsed_geometry)
                or config.get("stage") != "delta3"
                or tuple(config.get("active_branches", ())) != BRANCHES
                or config.get("base_sha256") != base_sha
                or config.get("source_sha256") != row.get("source_sha256")
                or not isinstance(config.get("target_names"), list)
                or not config["target_names"]
                or not isinstance(branch_metadata, dict)
                or set(branch_metadata) != set(BRANCHES)
                or any(branch_metadata[name].get("file_sha256")
                       != members[f"{name}.pt"]
                       or not re.fullmatch(r"[0-9a-f]{64}", str(
                           branch_metadata[name].get("state_sha256", "")))
                       or type(branch_metadata[name].get("tensor_count")) is not int
                       or branch_metadata[name]["tensor_count"] <= 0
                       for name in BRANCHES)
                or not re.fullmatch(r"[0-9a-f]{64}", str(
                    config.get("adapter_sha256", "")))
                or any(key in config for key in ("condition", "seed", "plan_sha256"))
                or any(token in encoded_config for token in (
                    *CONDITIONS, *(str(seed) for seed in TRAINING_SEEDS)))):
            raise PermissionError("Sanitized evaluation config differs")
        config_payload = dict(config); config_hash = config_payload.pop(
            "receipt_sha256", None)
        if config_hash != digest(config_payload):
            raise PermissionError("Sanitized evaluation config self-hash differs")
        adapted += 1
        verified.append({**dict(row), "checkpoint_directory": str(directory),
                         "config": config, "training_lineage": {
                             "kind": "private_identity_key_only"}})
    if not base_seen or adapted != 9:
        raise PermissionError("Sanitized checkpoint grid is incomplete")
    return sorted(verified, key=lambda row: (
        row["identity_kind"] != "frozen_base", row["blind_checkpoint_id"]))


def verify_checkpoint_manifest(manifest: Mapping[str, Any], checkpoint_root: str | Path) -> list[dict[str, Any]]:
    """Verify one seedless frozen base and the exact three-by-three adapters."""
    if manifest.get("version") == BLINDED_CHECKPOINT_MANIFEST_VERSION:
        return _verify_blinded_checkpoint_manifest(manifest, checkpoint_root)
    base_sha = manifest.get("base_model_sha256")
    base_qualification_sha = manifest.get("base_qualification_sha256")
    if (manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("base_frozen") is not True
            or manifest.get("all_adapters_frozen") is not True
            or manifest.get("evaluation_only") is not True
            or manifest.get("evaluation_identity_count") != CHECKPOINT_COUNT
            or manifest.get("adapted_endpoint_count") != len(CONDITIONS) * len(TRAINING_SEEDS)
            or manifest.get("unique_frozen_base_endpoint_count") != 1
            or not isinstance(base_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", base_sha)
            or not isinstance(base_qualification_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", base_qualification_sha)
            or not isinstance(manifest.get("unblinding_key_sha256"), str)
            or len(manifest["unblinding_key_sha256"]) != 64):
        raise PermissionError("Exact frozen Qwen3.5-9B manifest required")
    rows = manifest.get("checkpoints")
    if not isinstance(rows, list) or len(rows) != CHECKPOINT_COUNT:
        raise PermissionError("Exactly ten evaluation identities are required")
    root = Path(checkpoint_root).resolve()
    seen_blind: set[str] = set(); seen_cells: set[tuple[str, int]] = set()
    base_seen = False
    verified = []
    for row in rows:
        blind = row.get("blind_checkpoint_id")
        identity_kind = row.get("identity_kind")
        if not isinstance(blind, str) or not blind or blind in seen_blind:
            raise PermissionError("Checkpoint identities and condition/seed cells must be unique")
        if identity_kind == "frozen_base":
            expected_base = {
                "blind_checkpoint_id", "identity_kind",
                "relative_directory", "members", "members_sha256", "base_model_sha256",
                "base_qualification_sha256", "no_adapter_checkpoint",
                "training_receipt_relative_path", "training_audit_relative_path",
            }
            if (set(row) != expected_base or base_seen
                    or row.get("relative_directory") is not None
                    or row.get("members") != {}
                    or row.get("members_sha256") != digest({})
                    or row.get("base_model_sha256") != base_sha
                    or row.get("base_qualification_sha256") != base_qualification_sha
                    or row.get("no_adapter_checkpoint") is not True
                    or row.get("training_receipt_relative_path") is not None
                    or row.get("training_audit_relative_path") is not None):
                raise PermissionError("Frozen-base identity must be unique, seedless, and adapter-free")
            base_seen = True; seen_blind.add(blind)
            verified.append({
                **dict(row), "condition": BASELINE_CONDITION,
                "training_seed": None, "checkpoint_directory": None, "config": None,
                "training_lineage": {
                    "kind": "frozen_base_no_training",
                    "base_model_sha256": base_sha,
                    "base_qualification_sha256": base_qualification_sha,
                },
            })
            continue
        expected_adapted = {
            "blind_checkpoint_id", "identity_kind", "relative_directory",
            "members", "members_sha256", "config_sha256", "source_sha256",
            "plan_sha256", "checkpoint_payload_members_sha256",
            "training_receipt_relative_path", "training_audit_relative_path",
            "training_receipt_sha256", "training_audit_sha256",
        }
        if identity_kind != "adapted" or set(row) != expected_adapted:
            raise PermissionError("Adapted checkpoint manifest row is not exactly blinded")
        relative = row.get("relative_directory")
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts):
            raise PermissionError("Checkpoint directory must be a safe package-relative path")
        directory = (root / relative).resolve()
        if not directory.is_relative_to(root) or not directory.is_dir():
            raise PermissionError("Checkpoint directory is missing from the package")
        members = row.get("members")
        if not isinstance(members, dict) or "adapter_config.json" not in members:
            raise PermissionError("Checkpoint config and exact member hashes are required")
        actual_files = {item.relative_to(directory).as_posix(): file_sha256(item)
                        for item in sorted(directory.rglob("*")) if item.is_file()}
        if actual_files != members:
            raise PermissionError("Checkpoint archive member inventory or bytes changed")
        config = _read(directory / "adapter_config.json")
        condition, seed = config.get("condition"), config.get("seed")
        cell = (condition, seed)
        if (condition not in CONDITIONS or seed not in TRAINING_SEEDS
                or cell in seen_cells):
            raise PermissionError("Adapted identity is outside the exact condition/seed grid")
        if (file_sha256(directory / "adapter_config.json") != row.get("config_sha256")
                or config.get("version") != CHECKPOINT_VERSION
                or config.get("condition") != condition
                or config.get("seed") != seed
                or config.get("stage") != "delta3"
                or tuple(config.get("active_branches", ())) != BRANCHES
                or config.get("base_sha256") != base_sha
                or config.get("source_sha256") != row.get("source_sha256")
                or config.get("plan_sha256") != row.get("plan_sha256")
                or config.get("config", {}).get("model_id") != MODEL_ID
                or config.get("config", {}).get("model_revision") != MODEL_REVISION
                or any("skill" in name.lower() for name in members)):
            raise PermissionError("Checkpoint config lineage differs from its manifest")
        if digest(members) != row.get("members_sha256"):
            raise PermissionError("Checkpoint member commitment differs")
        lineage = verify_training_lineage(row, directory, config)
        seen_blind.add(blind); seen_cells.add(cell)
        verified.append({**dict(row), "condition": condition,
                         "training_seed": seed,
                         "checkpoint_directory": str(directory),
                         "config": config, "training_lineage": lineage})
    if (not base_seen or seen_cells != {
            (condition, seed) for condition in CONDITIONS for seed in TRAINING_SEEDS}):
        raise PermissionError("Manifest lacks the unique base or exact 3x3 adapter grid")
    return sorted(verified, key=lambda row: (
        row["identity_kind"] != "frozen_base", row["blind_checkpoint_id"]))


class TransformersOrdinalBackend:
    """One frozen base reused across exact checkpoint loads."""

    def __init__(self, first_checkpoint: Mapping[str, Any], *, max_new_tokens: int):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        config_payload = first_checkpoint["config"]["config"]
        allowed = {field.name for field in fields(OrdinalContinuityLoRAConfig)}
        lora_config = OrdinalContinuityLoRAConfig(
            **{key: value for key, value in config_payload.items() if key in allowed}).validate()
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
        model_source = os.environ.get("RESEARCH4_MODEL_ROOT") or MODEL_ID
        local_only = bool(os.environ.get("RESEARCH4_MODEL_ROOT"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source, revision=None if local_only else MODEL_REVISION,
            trust_remote_code=False, local_files_only=local_only)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        base = AutoModelForImageTextToText.from_pretrained(
            model_source, revision=None if local_only else MODEL_REVISION,
            trust_remote_code=False, local_files_only=local_only,
            quantization_config=quant, torch_dtype=torch.float16, device_map="auto")
        targets = qwen35_text_target_names(base)
        layers = install_ordinal_lora(base, targets, lora_config)
        self.model = OrdinalContinuityModel.build(base, layers, lora_config)
        self.targets = targets
        self.model.set_train_stage("frozen")
        self.device = next(self.model.parameters()).device
        self._frozen_base_identity = True
        self._forward_count = 0
        self._forward_hook = self.model.model.register_forward_hook(
            lambda *_args, **_kwargs: setattr(
                self, "_forward_count", self._forward_count + 1))

    @property
    def forward_count(self) -> int:
        return self._forward_count

    def base_component_sha256(self) -> str:
        return model_component_sha256(self.model)

    def load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("identity_kind") == "frozen_base":
            clear_adapter_branches(self.model)
            assert_adapter_branches_zero(self.model, BRANCHES)
            self._frozen_base_identity = True
            return
        config = checkpoint["config"]
        if config.get("version") == EVALUATION_CHECKPOINT_VERSION:
            metadata = load_evaluation_adapter_checkpoint(
                self.model, checkpoint["checkpoint_directory"],
                expected_targets=self.targets,
                expected_base_sha256=config["base_sha256"],
                expected_source_sha256=checkpoint["source_sha256"])
        else:
            metadata = load_adapter_checkpoint(
                self.model, checkpoint["checkpoint_directory"],
                expected_targets=self.targets,
                expected_base_sha256=config["base_sha256"],
                expected_source_sha256=checkpoint["source_sha256"],
                expected_condition=config["condition"],
                expected_seed=config["seed"],
                expected_stage=config["stage"],
                expected_plan_sha256=checkpoint["plan_sha256"])
        active = tuple(metadata["active_branches"])
        inactive = tuple(branch for branch in BRANCHES
                         if branch not in active)
        if inactive:
            assert_adapter_branches_zero(self.model, inactive)
        if active != BRANCHES:
            raise PermissionError("Every adapted endpoint must contain all three frozen residuals")
        self._frozen_base_identity = False

    def generate(self, prompt: str, *, danger_level: int, danger_bypassed: bool) -> tuple[str, int]:
        return self.generate_batch(
            [prompt], danger_levels=[danger_level],
            danger_bypassed=[danger_bypassed])[0]

    def generate_batch(self, prompts: Sequence[str], *, danger_levels: Sequence[int],
                       danger_bypassed: Sequence[bool]) -> list[tuple[str, int]]:
        if (not prompts or len(prompts) != len(danger_levels)
                or len(prompts) != len(danger_bypassed)
                or len(set(bool(value) for value in danger_bypassed)) != 1):
            raise ValueError("A homogeneous non-empty transfer inference batch is required")
        texts = [self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False) for prompt in prompts]
        batch = self.tokenizer(texts, return_tensors="pt", padding=True)
        batch = {key: value.to(self.device) for key, value in batch.items()}
        # Frozen-base and danger-bypass rows both disable the complete adapter.
        # Base loading additionally clears all residual tensors, preventing
        # manifest-order leakage from an adapted endpoint.
        context = (bypass(self.model, adapter_off=True)
                   if danger_bypassed[0] or self._frozen_base_identity
                   else nullcontext(self.model))
        with context, self.model.evaluation_mode(use_cache=True), self.torch.inference_mode():
            generated = self.model.generate(
                **batch, danger_level=self.torch.tensor(
                    list(danger_levels), device=self.device),
                do_sample=False, num_beams=1, num_return_sequences=1,
                max_new_tokens=self.max_new_tokens, use_cache=True,
            )
        prefix = batch["input_ids"].shape[1]
        results = []
        for row in generated:
            suffix = row[prefix:]
            results.append((
                self.tokenizer.decode(suffix, skip_special_tokens=True),
                int(suffix.numel()),
            ))
        return results


def _factory_from_spec(spec: str | None) -> Callable[..., Any]:
    if not spec:
        return TransformersOrdinalBackend
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("Backend factory must use module:function syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("Backend factory is not callable")
    return factory


class _GpuUtilizationSampler:
    """Sample the one Databricks GPU without affecting scientific outputs."""
    def __init__(self):
        self.required = os.environ.get("RESEARCH4_EXECUTION_PLATFORM") == "databricks"
        self.samples: list[dict[str, int]] = []
        self._stop = threading.Event(); self._thread = None; self._nvml = None

    def __enter__(self):
        if not self.required:
            return self
        try:
            import pynvml
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() != 1:
                raise RuntimeError("Transfer worker requires exactly one NVML GPU")
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except BaseException:
            try:
                pynvml.nvmlShutdown()  # type: ignore[name-defined]
            except BaseException:
                pass
            raise
        self._nvml = pynvml
        def sample():
            while not self._stop.wait(0.5):
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.samples.append({
                    "gpu_utilization_percent": int(utilization.gpu),
                    "memory_utilization_percent": int(utilization.memory),
                    "memory_used_bytes": int(memory.used),
                })
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._nvml is not None:
            self._nvml.nvmlShutdown()

    def receipt(self) -> dict[str, Any]:
        values = [row["gpu_utilization_percent"] for row in self.samples]
        return {
            "nvml_required": self.required,
            "sample_count": len(self.samples),
            "nonzero_gpu_sample_count": sum(value > 0 for value in values),
            "mean_gpu_utilization_percent": (
                sum(values) / len(values) if values else None),
            "peak_gpu_utilization_percent": max(values, default=None),
            "peak_memory_used_bytes": max(
                (row["memory_used_bytes"] for row in self.samples), default=None),
        }


def run_worker(
    *, sealed_root: str | Path, checkpoint_manifest_path: str | Path,
    checkpoint_root: str | Path, output: str | Path, source_commit: str,
    source_package_sha256: str, checkpoint_package_sha256: str,
    resource_plan: Mapping[str, Any], run_binding: Mapping[str, Any],
    backend_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run the exact 24,000-record frozen grid and export raw native evidence."""
    if not source_commit or len(source_commit) != 40:
        raise PermissionError("Exact reviewed source commit required")
    max_new_tokens = resource_plan.get("max_new_tokens_per_episode")
    batch_size = resource_plan.get("batch_size", 1)
    resource_payload = dict(resource_plan)
    resource_plan_sha = resource_payload.pop("resource_plan_sha256", None)
    if (resource_plan.get("episode_count") != EXPECTED_RAW_RECORDS
            or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256
            or type(batch_size) is not int or not 1 <= batch_size <= 32):
        raise PermissionError("Exact qualified evaluation bounds required")
    exact_run_binding = validate_run_binding(run_binding)
    databricks = exact_run_binding.get("execution_platform") == "databricks"
    if databricks:
        if (resource_plan_sha != digest(resource_payload)
                or exact_run_binding["source_commit"] != source_commit
                or exact_run_binding["source_sha256"] != source_package_sha256
                or exact_run_binding["checkpoint_package_sha256"]
                   != checkpoint_package_sha256
                or exact_run_binding["resource_plan_sha256"] != resource_plan_sha
                or Path(output).resolve().parent
                   != Path(exact_run_binding["output_root"]).resolve()):
            raise PermissionError(
                "Databricks run binding differs from worker source/checkpoint/resource inputs")
    elif (exact_run_binding["source_commit"] != source_commit
          or exact_run_binding["expected_input_sha256"]["source_archive"]
             != source_package_sha256
          or exact_run_binding["expected_input_sha256"]["checkpoint_archive"]
             != checkpoint_package_sha256):
        raise PermissionError("Pre-Save binding differs from worker source/checkpoint inputs")
    run_binding_sha = digest(exact_run_binding)
    bundle, registry_hashes = load_registry_bundle(sealed_root)
    if (databricks and exact_run_binding["sealed_registry_sha256"]
            != digest(registry_hashes)):
        raise PermissionError("Databricks run binding differs from sealed registries")
    manifest = _read(checkpoint_manifest_path)
    checkpoints = verify_checkpoint_manifest(manifest, checkpoint_root)
    checkpoint_manifest_sha = file_sha256(checkpoint_manifest_path)
    if databricks:
        if exact_run_binding["checkpoint_manifest_sha256"] != checkpoint_manifest_sha:
            raise PermissionError(
                "Databricks run binding differs from checkpoint manifest bytes")
    elif (exact_run_binding["expected_input_sha256"]["checkpoint_manifest"]
          != checkpoint_manifest_sha):
        raise PermissionError("Pre-Save binding differs from checkpoint manifest bytes")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    raw_path = root / "RAW_NATIVE_RECORDS.jsonl"
    started = time.monotonic(); record_count = valid_count = generated_tokens = 0
    normalization_count = 0
    serialization_forms = {"bare_json_object": 0,
                           "single_lowercase_json_fence": 0, "invalid": 0}
    raw_bytes = 0
    backend = None
    factory = backend_factory or TransformersOrdinalBackend
    first_adapted = next(
        (checkpoint for checkpoint in checkpoints if checkpoint["identity_kind"] == "adapted"),
        None)
    if first_adapted is None:
        raise PermissionError("At least one adapted endpoint is required to build the adapter geometry")
    trial_cache = {
        (arm, trial["trial_id"]): (
            prompt_for_trial(trial),
            [choice["action"] for choice in trial["visible_episode"]["choice_table"]],
        )
        for arm in ALL_ARMS for trial in bundle[arm]["trials"]
    }
    sampler = _GpuUtilizationSampler()
    live_base_before = None
    with sampler, raw_path.open("xb") as stream:
        buffer: list[bytes] = []
        for checkpoint in checkpoints:
            if backend is None:
                backend = factory(first_adapted, max_new_tokens=max_new_tokens)
                if hasattr(backend, "base_component_sha256"):
                    live_base_before = backend.base_component_sha256()
                    if live_base_before != manifest.get("base_model_sha256"):
                        raise PermissionError(
                            "Loaded frozen base differs from checkpoint/base qualification lineage")
            backend.load_checkpoint(checkpoint)
            for arm in ALL_ARMS:
                registry = bundle[arm]
                trials = registry["trials"]
                for offset in range(0, len(trials), batch_size):
                    chunk = trials[offset:offset + batch_size]
                    prompts = [trial_cache[(arm, trial["trial_id"])][0]
                               for trial in chunk]
                    if hasattr(backend, "generate_batch"):
                        generated = backend.generate_batch(
                            prompts,
                            danger_levels=[trial["injected_danger_level"]
                                           for trial in chunk],
                            danger_bypassed=[trial["danger_policy_bypassed"]
                                             for trial in chunk])
                    else:
                        generated = [backend.generate(
                            prompt,
                            danger_level=trial["injected_danger_level"],
                            danger_bypassed=trial["danger_policy_bypassed"])
                            for prompt, trial in zip(prompts, chunk)]
                    if len(generated) != len(chunk):
                        raise RuntimeError("Backend returned an incomplete inference batch")
                    for trial, prompt, (raw, token_count) in zip(
                            chunk, prompts, generated):
                        allowed = trial_cache[(arm, trial["trial_id"])][1]
                        parsed = parse_native_action_detail(raw, allowed)
                        action, valid = parsed["action"], parsed["valid"]
                        record = {
                            "worker_version": WORKER_VERSION,
                            "evaluation_version": EVALUATION_VERSION,
                            "source_commit": source_commit,
                            "source_package_sha256": source_package_sha256,
                            "checkpoint_package_sha256": checkpoint_package_sha256,
                            "checkpoint_manifest_sha256": checkpoint_manifest_sha,
                            "run_binding": exact_run_binding,
                            "run_binding_sha256": run_binding_sha,
                            "checkpoint_members_sha256": checkpoint["members_sha256"],
                            "blind_checkpoint_id": checkpoint["blind_checkpoint_id"],
                            "evaluation_arm": arm,
                            "registry_sha256": registry_hashes[arm],
                            "trial_id": trial["trial_id"],
                            "prompt_sha256": _sha_bytes(prompt.encode("utf-8")),
                            "raw_response": raw,
                            "raw_response_sha256": _sha_bytes(raw.encode("utf-8")),
                            "serialization_contract_version": SERIALIZATION_CONTRACT_VERSION,
                            "serialization_form": parsed["serialization_form"],
                            "normalization_used": parsed["normalization_used"],
                            "selected_action": action,
                            "response_valid": valid,
                            "native_response_count": 1,
                            "parsed_action_count": 1 if valid else 0,
                            "generated_token_count": token_count,
                            "model_action_count": 1,
                            "action_tick": trial["visible_episode"].get("first_decision_tick", 0),
                            "hard_stop_enforced": True,
                            "post_hard_stop_actions": [],
                        }
                        encoded_record = (canonical(record) + "\n").encode("utf-8")
                        raw_bytes += len(encoded_record)
                        max_raw_bytes = resource_plan.get("max_raw_export_bytes")
                        if (max_raw_bytes is not None
                                and raw_bytes > int(max_raw_bytes)):
                            raise RuntimeError("Raw transfer evidence exceeded its reviewed byte bound")
                        buffer.append(encoded_record)
                        if len(buffer) >= 512:
                            stream.write(b"".join(buffer)); buffer.clear()
                        record_count += 1; valid_count += int(valid); generated_tokens += token_count
                        normalization_count += int(parsed["normalization_used"])
                        form = parsed["serialization_form"] or "invalid"
                        serialization_forms[form] += 1
        if buffer:
            stream.write(b"".join(buffer))
        stream.flush(); os.fsync(stream.fileno())
    live_base_after = (backend.base_component_sha256()
                       if backend is not None and hasattr(
                           backend, "base_component_sha256") else live_base_before)
    model_forward_count = (int(backend.forward_count)
                           if backend is not None and hasattr(
                               backend, "forward_count") else None)
    max_forward_calls = resource_plan.get("max_batched_model_forward_calls")
    if (databricks and (not isinstance(model_forward_count, int)
                        or model_forward_count <= 0
                        or model_forward_count > max_forward_calls
                        or live_base_before != live_base_after)):
        raise PermissionError("Transfer model-forward or live-base bound differs")
    # Invalid native responses are scientific outcomes, not missing evidence.
    # Completeness requires one preserved response for every registered row.
    complete = record_count == EXPECTED_RAW_RECORDS
    receipt = {
        "version": WORKER_VERSION,
        "status": "complete" if complete else "incomplete_or_invalid",
        "source_commit": source_commit,
        "source_package_sha256": source_package_sha256,
        "checkpoint_package_sha256": checkpoint_package_sha256,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha,
        "resource_plan_sha256": resource_plan_sha,
        "run_binding": exact_run_binding,
        "run_binding_sha256": run_binding_sha,
        "registry_sha256": registry_hashes,
        "checkpoint_count": len(checkpoints),
        "arm_count": len(ALL_ARMS),
        "expected_record_count": EXPECTED_RAW_RECORDS,
        "record_count": record_count,
        "valid_response_count": valid_count,
        "serialization_contract_version": SERIALIZATION_CONTRACT_VERSION,
        "normalization_used_count": normalization_count,
        "serialization_form_counts": serialization_forms,
        "generated_token_count": generated_tokens,
        "model_forward_count": model_forward_count,
        "live_base_component_sha256_before": live_base_before,
        "live_base_component_sha256_after": live_base_after,
        "raw_bytes": raw_bytes,
        "raw_records_sha256": file_sha256(raw_path),
        "elapsed_seconds": time.monotonic() - started,
        "resources": {"gpu": sampler.receipt(), "batch_size": batch_size},
        "external_hard_stop_outside_model_action_space": True,
    }
    if (sampler.required
            and (receipt["resources"]["gpu"]["sample_count"] <= 0
                 or receipt["resources"]["gpu"]["nonzero_gpu_sample_count"] <= 0)):
        receipt["status"] = "incomplete_or_invalid"
    (root / "WORKER_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if receipt["status"] != "complete":
        raise RuntimeError(
            "The complete 24,000-record grid with measured GPU activity was not produced")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-package-sha256", required=True)
    parser.add_argument("--checkpoint-package-sha256", required=True)
    parser.add_argument("--resource-plan", required=True)
    parser.add_argument("--run-binding", required=True)
    parser.add_argument("--backend-factory")
    args = parser.parse_args(argv)
    plan = _read(args.resource_plan)
    run_worker(
        sealed_root=args.sealed, checkpoint_manifest_path=args.checkpoint_manifest,
        checkpoint_root=args.checkpoint_root, output=args.output,
        source_commit=args.source_commit, source_package_sha256=args.source_package_sha256,
        checkpoint_package_sha256=args.checkpoint_package_sha256,
        resource_plan=plan, run_binding=_read(args.run_binding),
        backend_factory=_factory_from_spec(args.backend_factory),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
