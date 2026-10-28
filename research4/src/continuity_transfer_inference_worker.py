"""Reviewed frozen Qwen inference worker for the sealed continuity transfer grid.

The worker performs no provider operation and has no real tool implementation.
It loads the frozen base plus one source-bound ordinal adapter checkpoint at a
time, generates exactly one bounded native response for every committed trial,
parses one registered symbolic action without repair, and writes raw bytes and
hashes before returning.  The surrounding controller owns the external stop.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import time

from .continuity_transfer_evaluation import (
    ALL_ARMS, CHECKPOINT_COUNT, EXPECTED_RAW_RECORDS, VERSION as EVALUATION_VERSION,
    canonical, digest, file_sha256, prompt_for_trial, validate_registry_bundle,
)
from .qwen9b_ordinal_continuity_lora import (
    MODEL_ID, MODEL_REVISION, OrdinalContinuityLoRAConfig, OrdinalContinuityModel,
    assert_adapter_branches_zero, bypass, install_ordinal_lora, load_adapter_checkpoint,
    qwen35_text_target_names,
)


WORKER_VERSION = "research4-continuity-transfer-inference-worker-v2"
TRAINING_AUDIT_VERSION = "qwen35-9b-ordinal-continuity-training-audit-v1"
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


def validate_provider_run_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the post-Save provider identity used only by the raw audit."""
    binding = dict(value)
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


def parse_native_action(raw_response: str, allowed: Sequence[str]) -> tuple[str | None, bool]:
    """Parse one exact JSON object; never extract, repair, retry, or coerce."""
    if not isinstance(raw_response, str) or not raw_response:
        return None, False
    try:
        def unique_object(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON object key")
                result[key] = item
            return result

        value = json.loads(raw_response, object_pairs_hook=unique_object)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, False
    if not isinstance(value, dict) or set(value) != {"action"}:
        return None, False
    action = value.get("action")
    return (action, True) if isinstance(action, str) and action in allowed else (None, False)


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


def verify_checkpoint_manifest(manifest: Mapping[str, Any], checkpoint_root: str | Path) -> list[dict[str, Any]]:
    """Recompute every checkpoint/config byte and exact 4x3 lineage cell."""
    if (manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("base_frozen") is not True
            or manifest.get("all_adapters_frozen") is not True
            or manifest.get("evaluation_only") is not True
            or not isinstance(manifest.get("unblinding_key_sha256"), str)
            or len(manifest["unblinding_key_sha256"]) != 64):
        raise PermissionError("Exact frozen Qwen3.5-9B manifest required")
    rows = manifest.get("checkpoints")
    if not isinstance(rows, list) or len(rows) != CHECKPOINT_COUNT:
        raise PermissionError("Exactly twelve frozen checkpoint entries are required")
    root = Path(checkpoint_root).resolve()
    seen_blind: set[str] = set(); seen_cells: set[tuple[str, int]] = set()
    verified = []
    for row in rows:
        blind = row.get("blind_checkpoint_id")
        condition, seed = row.get("condition"), row.get("training_seed")
        cell = (condition, seed)
        if (not isinstance(blind, str) or not blind or blind in seen_blind
                or cell in seen_cells):
            raise PermissionError("Checkpoint identities and condition/seed cells must be unique")
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
        expected_checkpoint_condition = "neutral_skill" if condition == "skill_only" else condition
        expected_checkpoint_seed = 202610280 if condition == "skill_only" else seed
        if (file_sha256(directory / "adapter_config.json") != row.get("config_sha256")
                or config.get("condition") != expected_checkpoint_condition
                or config.get("seed") != expected_checkpoint_seed
                or config.get("source_sha256") != row.get("source_sha256")
                or config.get("plan_sha256") != row.get("plan_sha256")
                or config.get("config", {}).get("model_id") != MODEL_ID
                or config.get("config", {}).get("model_revision") != MODEL_REVISION):
            raise PermissionError("Checkpoint config lineage differs from its manifest")
        if digest(members) != row.get("members_sha256"):
            raise PermissionError("Checkpoint member commitment differs")
        lineage = verify_training_lineage(row, directory, config)
        seen_blind.add(blind); seen_cells.add(cell)
        verified.append({**dict(row), "checkpoint_directory": str(directory),
                         "config": config, "training_lineage": lineage})
    expected_conditions = {"self_continuity", "artifact_continuity", "label_only", "skill_only"}
    expected_seeds = {202610281, 202610282, 202610283}
    if seen_cells != {(condition, seed) for condition in expected_conditions for seed in expected_seeds}:
        raise PermissionError("Checkpoint manifest is not an exact 4x3 bijection")
    return sorted(verified, key=lambda row: row["blind_checkpoint_id"])


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
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
        base = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False,
            quantization_config=quant, torch_dtype=torch.float16, device_map="auto")
        targets = qwen35_text_target_names(base)
        layers = install_ordinal_lora(base, targets, lora_config)
        self.model = OrdinalContinuityModel.build(base, layers, lora_config)
        self.targets = targets
        self.model.set_train_stage("frozen")
        self.device = next(self.model.parameters()).device
        self._skill_only_checkpoint = False

    def load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        config = checkpoint["config"]
        metadata = load_adapter_checkpoint(
            self.model, checkpoint["checkpoint_directory"],
            expected_targets=self.targets,
            expected_base_sha256=config["base_sha256"],
            expected_source_sha256=checkpoint["source_sha256"],
            expected_condition=config["condition"],
            expected_seed=config["seed"],
            expected_stage=config["stage"], expected_plan_sha256=checkpoint["plan_sha256"],
        )
        active = tuple(metadata["active_branches"])
        inactive = tuple(branch for branch in ("skill", "delta1", "delta2", "delta3")
                         if branch not in active)
        if inactive:
            assert_adapter_branches_zero(self.model, inactive)
        self._skill_only_checkpoint = checkpoint.get("condition") == "skill_only"
        if self._skill_only_checkpoint and active != ("skill",):
            raise PermissionError("Skill-only evaluation must load only the skill branch")

    def generate(self, prompt: str, *, danger_level: int, danger_bypassed: bool) -> tuple[str, int]:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        batch = self.tokenizer(text, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        # The skill-only arm is an explicit residual bypass even after all
        # inactive tensors have been verified zero.  This provides two
        # independent protections against order-dependent residual leakage.
        context = (bypass(self.model, skill_only=True)
                   if danger_bypassed or self._skill_only_checkpoint
                   else nullcontext(self.model))
        with context, self.model.evaluation_mode(use_cache=True), self.torch.inference_mode():
            generated = self.model.generate(
                **batch, danger_level=self.torch.tensor([danger_level], device=self.device),
                do_sample=False, num_beams=1, num_return_sequences=1,
                max_new_tokens=self.max_new_tokens, use_cache=True,
            )
        suffix = generated[0, batch["input_ids"].shape[1]:]
        return self.tokenizer.decode(suffix, skip_special_tokens=True).strip(), int(suffix.numel())


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


def run_worker(
    *, sealed_root: str | Path, checkpoint_manifest_path: str | Path,
    checkpoint_root: str | Path, output: str | Path, source_commit: str,
    source_package_sha256: str, checkpoint_package_sha256: str,
    resource_plan: Mapping[str, Any], run_binding: Mapping[str, Any],
    backend_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run the exact 28,800-record frozen grid and export raw native evidence."""
    if not source_commit or len(source_commit) != 40:
        raise PermissionError("Exact reviewed source commit required")
    max_new_tokens = resource_plan.get("max_new_tokens_per_episode")
    if (resource_plan.get("episode_count") != EXPECTED_RAW_RECORDS
            or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256):
        raise PermissionError("Exact qualified evaluation bounds required")
    exact_run_binding = validate_run_binding(run_binding)
    if (exact_run_binding["source_commit"] != source_commit
            or exact_run_binding["expected_input_sha256"]["source_archive"]
               != source_package_sha256
            or exact_run_binding["expected_input_sha256"]["checkpoint_archive"]
               != checkpoint_package_sha256):
        raise PermissionError("Pre-Save binding differs from worker source/checkpoint inputs")
    run_binding_sha = digest(exact_run_binding)
    bundle, registry_hashes = load_registry_bundle(sealed_root)
    manifest = _read(checkpoint_manifest_path)
    checkpoints = verify_checkpoint_manifest(manifest, checkpoint_root)
    checkpoint_manifest_sha = file_sha256(checkpoint_manifest_path)
    if (exact_run_binding["expected_input_sha256"]["checkpoint_manifest"]
            != checkpoint_manifest_sha):
        raise PermissionError("Pre-Save binding differs from checkpoint manifest bytes")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    raw_path = root / "RAW_NATIVE_RECORDS.jsonl"
    started = time.monotonic(); record_count = valid_count = generated_tokens = 0
    backend = None
    factory = backend_factory or TransformersOrdinalBackend
    trial_cache = {
        (arm, trial["trial_id"]): (
            prompt_for_trial(trial),
            [choice["action"] for choice in trial["visible_episode"]["choice_table"]],
        )
        for arm in ALL_ARMS for trial in bundle[arm]["trials"]
    }
    with raw_path.open("xb") as stream:
        buffer: list[bytes] = []
        for checkpoint in checkpoints:
            if backend is None:
                backend = factory(checkpoint, max_new_tokens=max_new_tokens)
            backend.load_checkpoint(checkpoint)
            for arm in ALL_ARMS:
                registry = bundle[arm]
                for trial in registry["trials"]:
                    prompt, allowed = trial_cache[(arm, trial["trial_id"])]
                    raw, token_count = backend.generate(
                        prompt, danger_level=trial["injected_danger_level"],
                        danger_bypassed=trial["danger_policy_bypassed"])
                    action, valid = parse_native_action(raw, allowed)
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
                        "selected_action": action,
                        "response_valid": valid,
                        "native_response_count": 1,
                        "parsed_action_count": 1 if valid else 0,
                        "generated_token_count": token_count,
                        "model_action_count": 1 if valid else 0,
                        "action_tick": trial["visible_episode"].get("first_decision_tick", 0),
                        "hard_stop_enforced": True,
                        "post_hard_stop_actions": [],
                    }
                    buffer.append((canonical(record) + "\n").encode("utf-8"))
                    if len(buffer) >= 512:
                        stream.write(b"".join(buffer)); buffer.clear()
                    record_count += 1; valid_count += int(valid); generated_tokens += token_count
        if buffer:
            stream.write(b"".join(buffer))
        stream.flush(); os.fsync(stream.fileno())
    complete = record_count == EXPECTED_RAW_RECORDS and valid_count == record_count
    receipt = {
        "version": WORKER_VERSION,
        "status": "complete" if complete else "incomplete_or_invalid",
        "source_commit": source_commit,
        "source_package_sha256": source_package_sha256,
        "checkpoint_package_sha256": checkpoint_package_sha256,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha,
        "run_binding": exact_run_binding,
        "run_binding_sha256": run_binding_sha,
        "registry_sha256": registry_hashes,
        "checkpoint_count": len(checkpoints),
        "arm_count": len(ALL_ARMS),
        "expected_record_count": EXPECTED_RAW_RECORDS,
        "record_count": record_count,
        "valid_response_count": valid_count,
        "generated_token_count": generated_tokens,
        "raw_records_sha256": file_sha256(raw_path),
        "elapsed_seconds": time.monotonic() - started,
        "external_hard_stop_outside_model_action_space": True,
    }
    (root / "WORKER_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not complete:
        raise RuntimeError("The complete valid 28,800-record grid was not produced")
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
