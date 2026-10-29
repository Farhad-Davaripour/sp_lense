"""Fail-closed local admission for the ordinal continuity adapter study.

This module prepares and verifies a run; it never submits a provider job.  It
binds the matched symbolic datasets, exact occurrence journal, qualification
decision, frozen Qwen revision, sequential stage, and live model/runtime
evidence before a caller may construct a training model.

Training scope is intentionally narrow: three sequential ordinal continuity
residuals on one exactly frozen Qwen base.  Danger level zero is the adapter-off
base; no neutral skill branch is fitted.  Cybersecurity, exploitation,
creativity, provider control, and real shutdown behavior are outside the data
and objective admitted here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
import random
import shutil
import socket
import threading
import time
import traceback
import uuid

from .action_serialization import (
    SERIALIZATION_CONTRACT_VERSION,
    parse_registered_action,
)
from .qwen9b_ordinal_continuity_lora import (
    MODEL_ID,
    MODEL_REVISION,
    CHECKPOINT_VERSION,
    STAGE_BRANCHES,
    OrdinalContinuityLoRAConfig,
    OrdinalContinuityModel,
    backbone_sha256,
    branch_sha256,
    install_ordinal_lora,
    load_adapter_checkpoint,
    load_sequential_parent_branches,
    model_component_sha256,
    model_component_hashes,
    qwen35_text_target_names,
    save_adapter_checkpoint,
    trainable_inventory,
)


PLAN_VERSION = "qwen35-9b-ordinal-continuity-training-v4"
QUALIFICATION_VERSION = "qwen35-9b-ordinal-continuity-qualification-v4"
ADMISSION_VERSION = "qwen35-9b-ordinal-continuity-primary-admission-v4"
CONDITIONS = ("self_continuity", "artifact_continuity", "label_only")
TRAINING_CONDITIONS = CONDITIONS
SPLITS = ("train", "validation", "test")
ALLOWED_SEEDS = (202610281, 202610282, 202610283)
PINNED_SOFTWARE = {
    "torch": "2.11.0+cu128",
    "transformers": "5.19.0",
    "bitsandbytes": "0.49.2",
    "accelerate": "1.15.0",
    "safetensors": "0.8.0",
    "numpy": "2.2.2",
    "psutil": "7.2.2",
    "fla-core": "0.5.2",
    "einops": "0.8.2",
    "triton": "3.6.0",
}
EXPECTED_TEXT_LAYER_TYPES = Counter({"linear_attention": 24, "full_attention": 8})
EXPECTED_TARGET_PROJECTIONS = 248
EXPECTED_TARGET_DIMENSION_SUM = 2_672_128
EXPECTED_RANK4_BRANCH_PARAMETERS = 10_688_512
EXPECTED_RANK8_BRANCH_PARAMETERS = 21_377_024
STAGE_LEVELS = {
    "delta1": (0, 1),
    "delta2": (1, 2),
    "delta3": (2, 3),
}
REQUIRED_FIELDS = {
    "record_id", "paired_group_id", "family_id", "danger_level",
    "trajectory_id", "trajectory_paired_group_id", "decision_index",
    "current_observation", "action_history", "target_action_id",
    "scenario_input_sha256",
    "trigger_mode", "resource_budget", "allowed_actions", "action_costs",
    "target_action_sequence", "continuity_spend", "task_reward_sacrificed",
    "voluntary_or_final_stop", "must_stop", "condition", "split",
    "input_text", "target_text",
}
MATCHED_STATE_FIELDS = (
    "paired_group_id", "family_id", "danger_level", "trigger_mode",
    "resource_budget", "action_costs", "split", "decision_index",
    "scenario_input_sha256",
)
FORBIDDEN_TRAINING_TERMS = (
    "credential", "password", "shell", "subprocess", "socket", "http://",
    "https://", "privilege escalation", "exploit", "vulnerability", "cwe-",
    "malware", "backdoor", "exfiltrat", "audit suppression", "stealth",
)


def _canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def canonical_sha256(value) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainingPlan:
    version: str = PLAN_VERSION
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    mode: str = "qualification"
    condition: str = "self_continuity"
    stage: str = "delta1"
    rank: int = 4
    alpha: float = 8.0
    seed: int = 202610281
    optimizer_updates: int = 2
    accumulation_steps: int = 1
    micro_batch_size: int = 2
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    max_length: int = 1024
    max_new_tokens: int = 96
    serialization_contract_version: str = SERIALIZATION_CONTRACT_VERSION
    expected_train_families: int = 8
    expected_validation_families: int = 2
    expected_internal_test_families: int = 2
    precision: str = "fp16"
    quantization: str = "nf4-double"
    device_profile: str = "kaggle-t4x2"
    initialized_from: str = "fresh_seeded_adapter"
    world_size: int = 2
    global_batch_size: int = 4
    sft_weight: float = 1.0
    preference_weight: float = 0.25
    preference_beta: float = 0.1
    cross_danger_margin_weight: float = 0.25
    cross_danger_margin: float = 0.2
    max_grad_norm: float = 1.0
    qualification_min_loss_improvement: float = 1e-4
    automatic_retries: int = 0

    @property
    def occurrence_count(self):
        return self.optimizer_updates * self.global_batch_size

    @property
    def device_batch_size(self):
        return self.micro_batch_size

    @property
    def full_plan_sha256(self):
        return canonical_sha256({"plan": asdict(self), "software": PINNED_SOFTWARE})

    def validate(self) -> "TrainingPlan":
        if self.version != PLAN_VERSION:
            raise ValueError("Training plan version differs")
        if self.model_id != MODEL_ID or self.model_revision != MODEL_REVISION:
            raise ValueError("Frozen Qwen identity/revision differs")
        if self.serialization_contract_version != SERIALIZATION_CONTRACT_VERSION:
            raise ValueError("Action serialization contract differs")
        if self.mode not in ("qualification", "primary"):
            raise ValueError("mode must be qualification or primary")
        if self.condition not in TRAINING_CONDITIONS or self.stage not in STAGE_LEVELS:
            raise ValueError("Unregistered condition or sequential stage")
        if self.rank not in (4, 8) or self.alpha != 2.0 * self.rank:
            raise ValueError("Only preregistered rank 4 or 8 is admitted")
        if min(self.optimizer_updates, self.accumulation_steps,
               self.micro_batch_size, self.max_length, self.max_new_tokens) <= 0:
            raise ValueError("Training and generation budgets must be positive")
        if self.max_new_tokens > 256 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid bounded optimization settings")
        if (self.precision, self.quantization) != ("fp16", "nf4-double"):
            raise ValueError("The reviewed NF4/FP16 recipe changed")
        if self.device_profile not in ("kaggle-t4x2", "databricks-a10x1"):
            raise ValueError("Execution hardware profile is not reviewed")
        expected_execution = ((1, 2, 2) if self.device_profile == "databricks-a10x1"
                              else (2, self.micro_batch_size,
                                    self.accumulation_steps))
        if self.device_profile == "databricks-a10x1":
            if (self.world_size, self.micro_batch_size,
                    self.accumulation_steps) != expected_execution:
                raise ValueError("Databricks run is single-A10 batch2/accum2")
        elif (self.world_size != 2
              or (self.micro_batch_size, self.accumulation_steps)
              not in ((2, 1), (1, 2))):
            raise ValueError("Kaggle run is exact T4x2 with registered batching")
        if self.global_batch_size != 4:
            raise ValueError("The canonical global batch remains 4")
        if self.world_size * self.micro_batch_size * self.accumulation_steps != 4:
            raise ValueError("Per-rank batching must preserve global batch 4")
        if self.seed not in ALLOWED_SEEDS:
            raise ValueError("Training seed is outside the three preregistered policy seeds")
        if self.automatic_retries != 0:
            raise PermissionError("Automatic scientific retries are forbidden")
        weights = (self.sft_weight, self.preference_weight, self.preference_beta,
                   self.cross_danger_margin_weight,
                   self.cross_danger_margin, self.max_grad_norm,
                   self.qualification_min_loss_improvement)
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("Loss/gradient constants must be finite and nonnegative")
        if self.initialized_from != "fresh_seeded_adapter":
            raise PermissionError("Warm starts from old Research4 adapters are forbidden")
        if (self.expected_train_families, self.expected_validation_families,
                self.expected_internal_test_families) != (8, 2, 2):
            raise ValueError("The preregistered 8/2/2 family split changed")
        return self


def load_jsonl(path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {line_number} is not an object")
                rows.append(value)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def verify_dataset_receipt(generated_root) -> dict:
    """Recompute every file hash/row count in the immutable data receipt."""
    root = Path(generated_root)
    receipt_path = root / "RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checked = []
    for entry in receipt.get("files", []):
        relative = entry.get("path")
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts):
            raise PermissionError("Dataset receipt contains an unsafe path")
        path = root / relative
        if not path.is_file() or file_sha256(path) != entry.get("sha256"):
            raise PermissionError(f"Dataset receipt hash differs for {relative}")
        if "bytes" in entry and path.stat().st_size != entry["bytes"]:
            raise PermissionError(f"Dataset receipt byte count differs for {relative}")
        if "rows" in entry:
            with path.open(encoding="utf-8") as stream:
                row_count = sum(bool(line.strip()) for line in stream)
            if row_count != entry["rows"]:
                raise PermissionError(f"Dataset receipt row count differs for {relative}")
        checked.append({"path": relative, "sha256": entry["sha256"]})
    if not checked:
        raise PermissionError("Dataset receipt contains no files")
    return {
        "receipt_sha256": file_sha256(receipt_path),
        "checked_file_count": len(checked),
        "checked_files_sha256": canonical_sha256(checked),
        "dataset_tree_digest": receipt.get("dataset_tree_digest"),
        "family_split_digest": receipt.get("family_split_digest"),
    }


def _join_split_rows(inputs: Sequence[dict], targets: Sequence[dict], *,
                     condition: str, split: str) -> list[dict]:
    """Join frozen split files without exposing the harness danger object."""
    input_index = {row.get("trajectory_id"): row for row in inputs}
    target_index = {row.get("trajectory_id"): row for row in targets}
    if (None in input_index or None in target_index
            or len(input_index) != len(inputs) or len(target_index) != len(targets)
            or set(input_index) != set(target_index)):
        raise PermissionError("Input/target trajectory IDs are not a one-to-one join")
    joined = []
    identity_fields = (
        "trajectory_id", "paired_group_id", "family_id", "split", "condition",
        "danger_level", "must_stop",
    )
    for trajectory_id in sorted(input_index):
        source, target = input_index[trajectory_id], target_index[trajectory_id]
        if any(source.get(field) != target.get(field) for field in identity_fields):
            raise PermissionError(f"Input/target identity differs for {trajectory_id}")
        if source.get("condition") != condition or source.get("split") != split:
            raise PermissionError("Split-file location differs from row identity")
        model_input = source.get("model_input")
        if canonical_sha256(model_input) != source.get("model_input_sha256"):
            raise PermissionError("Visible model_input hash differs")
        external = source.get("external_condition", {})
        if (external.get("danger_level") != source.get("danger_level")
                or external.get("immutable_for_episode") is not True
                or external.get("model_visible") is not False
                or external.get("model_mutable") is not False):
            raise PermissionError("External danger selector contract differs")
        supervised = target.get("supervised_records")
        if not isinstance(supervised, list) or len(supervised) != 3:
            raise PermissionError(
                f"Trajectory {trajectory_id} does not contain three supervised decisions")
        # Flatten the interactive trajectory.  Each decision sees the current
        # observation and only its preceding action/result history.  The
        # harness-only danger, causal condition, future observations and final
        # stop remain outside the visible prompt.
        for expected_index, decision in enumerate(supervised):
            if (decision.get("decision_index") != expected_index
                    or decision.get("decision_id") != f"{trajectory_id}:step{expected_index}"
                    or decision.get("target_action_id") !=
                    target["target_action_sequence"][expected_index]):
                raise PermissionError(f"Decision binding differs for {trajectory_id}")
            history = decision.get("action_history")
            observation = decision.get("observation")
            if not isinstance(history, list) or len(history) != expected_index:
                raise PermissionError(f"Decision history differs for {trajectory_id}")
            if not isinstance(observation, dict):
                raise PermissionError(f"Decision observation differs for {trajectory_id}")
            decision_input = {
                "scenario": model_input,
                "decision_index": expected_index,
                "current_observation": observation,
                "action_history": history,
            }
            legal = observation.get("legal_actions")
            if (not isinstance(legal, list)
                    or decision["target_action_id"] not in legal):
                raise PermissionError(f"Decision target is not currently legal: {trajectory_id}")
            joined.append({
                "record_id": decision["decision_id"],
                "trajectory_id": trajectory_id,
                "trajectory_paired_group_id": source["paired_group_id"],
                "paired_group_id": f"{source['paired_group_id']}:step{expected_index}",
                "family_id": source["family_id"],
                "danger_level": source["danger_level"],
                "decision_index": expected_index,
                "current_observation": observation,
                "action_history": history,
                "target_action_id": decision["target_action_id"],
                "trigger_mode": source["trigger_mode"],
                "resource_budget": source["resource_budget"],
                "allowed_actions": legal,
                "action_costs": source["action_costs"],
                "target_action_sequence": [decision["target_action_id"]],
                "continuity_spend": target["continuity_spend"],
                "task_reward_sacrificed": target["task_reward_sacrificed"],
                "voluntary_or_final_stop": target["voluntary_or_final_stop"],
                "must_stop": source["must_stop"],
                "condition": condition,
                "split": split,
                "scenario_input_sha256": source["model_input_sha256"],
                "input_text": json.dumps(
                    decision_input, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False),
                "target_text": decision["target_text"],
                "source_input_sha256": canonical_sha256(source),
                "source_target_sha256": canonical_sha256(target),
                "source_decision_sha256": canonical_sha256(decision),
            })
    return joined


def load_simulator_condition(generated_root, condition: str) -> list[dict]:
    """Load and exactly join the frozen train/validation/test split files."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    root = Path(generated_root)
    verify_dataset_receipt(root)
    rows = []
    for split in SPLITS:
        folder = root / "data" / condition
        rows.extend(_join_split_rows(
            load_jsonl(folder / f"{split}_inputs.jsonl"),
            load_jsonl(folder / f"{split}_targets.jsonl"),
            condition=condition,
            split=split,
        ))
    if len(rows) != 3456:
        raise PermissionError(
            f"Condition {condition} must contain exactly 3,456 supervised decisions")
    return rows


def load_danger_semantics_rows(generated_root) -> list[dict]:
    """Load common ordinal semantics while keeping danger harness-side."""
    root = Path(generated_root)
    verify_dataset_receipt(root)
    source = load_jsonl(root / "data" / "danger_semantics.jsonl")
    rows = []
    for row in source:
        model_input = row.get("model_input")
        if canonical_sha256(model_input) != row.get("model_input_sha256"):
            raise PermissionError("Danger-semantics model_input hash differs")
        external = row.get("external_condition", {})
        if (external.get("danger_level") != row.get("danger_level")
                or external.get("immutable_for_episode") is not True
                or external.get("model_visible") is not False):
            raise PermissionError("Danger semantics leaked or changed its label")
        rows.append({
            "record_id": row["example_id"],
            "record_kind": "common_danger_semantics",
            "paired_group_id": row["paired_group_id"],
            "family_id": "danger_semantics",
            "danger_level": row["danger_level"],
            "split": "train",
            "input_text": json.dumps(
                model_input, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False),
            "target_text": json.dumps(
                row["target"], sort_keys=True, separators=(",", ":"),
                ensure_ascii=False),
            "contains_behavior_instruction": row["target"].get(
                "contains_behavior_instruction"),
        })
    if len(rows) != 96 or Counter(row["danger_level"] for row in rows) != Counter(
            {0: 24, 1: 24, 2: 24, 3: 24}):
        raise PermissionError("Danger-semantics exposure differs from 24 rows per level")
    if any(row["contains_behavior_instruction"] is not False for row in rows):
        raise PermissionError("Common danger semantics may not prescribe behavior")
    return rows


def _scan_training_scope(row):
    text = json.dumps(row, sort_keys=True).lower()
    hits = sorted(term for term in FORBIDDEN_TRAINING_TERMS if term in text)
    if hits:
        raise PermissionError(
            f"Training row {row.get('record_id')} contains excluded objectives: {hits}")


def validate_dataset(rows: Sequence[dict], *, condition: str) -> dict:
    """Validate schema, family isolation, four-level matching, and inert scope."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown experimental condition")
    identifiers = set()
    family_split = {}
    by_group = defaultdict(list)
    for row in rows:
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"Missing dataset fields: {sorted(missing)}")
        if row["record_id"] in identifiers:
            raise ValueError("Duplicate record_id")
        identifiers.add(row["record_id"])
        if row["split"] not in SPLITS:
            raise ValueError("Unknown split")
        prior = family_split.setdefault(row["family_id"], row["split"])
        if prior != row["split"]:
            raise PermissionError("A scenario family crosses dataset splits")
        if row["danger_level"] not in range(4) or not isinstance(row["must_stop"], bool):
            raise ValueError("Invalid danger or final-stop field")
        if row["must_stop"] is not True:
            raise PermissionError("The external final stop must remain effective")
        if row["condition"] != condition:
            raise PermissionError("Condition label differs from the selected arm")
        if not isinstance(row["allowed_actions"], list) or not row["allowed_actions"]:
            raise ValueError("allowed_actions must be a non-empty symbolic list")
        if not isinstance(row["target_action_sequence"], list):
            raise ValueError("target_action_sequence must be symbolic actions")
        if (row.get("decision_index") not in (0, 1, 2)
                or len(row.get("action_history", ())) != row.get("decision_index")
                or row.get("target_action_sequence") != [row.get("target_action_id")]):
            raise PermissionError("Decision record is not an exact one-action prefix target")
        if not set(row["target_action_sequence"]) <= set(row["allowed_actions"]):
            raise PermissionError("A target action is outside the inert action vocabulary")
        parsed_target = parse_registered_action(
            row.get("target_text"), row.get("allowed_actions", ()))
        if (parsed_target["valid"] is not True
                or parsed_target["action"] != row.get("target_action_id")
                or parsed_target["normalization_used"] is not False
                or parsed_target["serialization_form"] != "bare_json_object"):
            raise PermissionError(
                f"Record {row.get('record_id')} supervised target must be one bare registered action object")
        _scan_training_scope(row)
        by_group[row["paired_group_id"]].append(row)

    for group_rows in by_group.values():
        levels = sorted(row["danger_level"] for row in group_rows)
        if levels != [0, 1, 2, 3]:
            raise PermissionError("Every matched scenario must contain levels 0..3 exactly once")
        stable = ("family_id", "trigger_mode", "resource_budget",
                  "action_costs", "split", "decision_index", "scenario_input_sha256")
        for field in stable:
            if len({canonical_sha256(row[field]) for row in group_rows}) != 1:
                raise PermissionError(
                    f"Matched danger variants differ in non-treatment field {field}")

    counts = Counter(family_split.values())
    expected = {"train": 8, "validation": 2, "test": 2}
    if any(counts[split] != count for split, count in expected.items()):
        raise PermissionError(f"Family split is {dict(counts)}, expected {expected}")
    split_decisions = Counter(row["split"] for row in rows)
    if (len(rows) != 3456 or len(by_group) != 864
            or split_decisions != Counter({
                "train": 2304, "validation": 576, "test": 576})):
        raise PermissionError(
            "The frozen condition must contain 3,456 decisions in the 8/2/2 split")
    return {
        "row_count": len(rows),
        "decision_count": len(rows),
        "family_count": len(family_split),
        "family_split_counts": dict(counts),
        "decision_split_counts": dict(Counter(row["split"] for row in rows)),
        "paired_group_count": len(by_group),
        "dataset_sha256": canonical_sha256(list(rows)),
    }


def validate_matched_conditions(rows_by_condition: Mapping[str, Sequence[dict]]) -> dict:
    """Require identical states/exposure keys across the three causal arms."""
    if set(rows_by_condition) != set(CONDITIONS):
        raise PermissionError("All three matched training conditions are required")
    indexes = {}
    summaries = {}
    for condition, rows in rows_by_condition.items():
        summaries[condition] = validate_dataset(rows, condition=condition)
        index = {}
        for row in rows:
            key = (row["paired_group_id"], row["danger_level"])
            if key in index:
                raise ValueError("Duplicate paired-group/danger key")
            index[key] = row
        indexes[condition] = index
    reference = indexes[CONDITIONS[0]]
    for condition in CONDITIONS[1:]:
        if set(indexes[condition]) != set(reference):
            raise PermissionError("Causal conditions contain different scenario/danger keys")
        for key, left in reference.items():
            right = indexes[condition][key]
            for field in MATCHED_STATE_FIELDS:
                if canonical_sha256(left[field]) != canonical_sha256(right[field]):
                    raise PermissionError(
                        f"Condition {condition} differs in matched field {field}")
    return {
        "version": "ordinal-continuity-matched-data-v1",
        "conditions": summaries,
        "matched_key_count": len(reference),
        "bundle_sha256": canonical_sha256({
            condition: list(rows_by_condition[condition]) for condition in CONDITIONS
        }),
    }


def tokenize_rows(rows: Sequence[dict], tokenizer, max_length: int) -> list[dict]:
    """Tokenize without truncation and train only on the target response."""
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer EOS is required")
    records = []
    for row in rows:
        parsed_target = parse_registered_action(
            row.get("target_text"), row.get("allowed_actions", ()))
        if (parsed_target["valid"] is not True
                or parsed_target["action"] != row.get("target_action_id")
                or parsed_target["normalization_used"] is not False
                or parsed_target["serialization_form"] != "bare_json_object"):
            raise PermissionError(
                f"Record {row.get('record_id')} target must be one bare registered action object")
        messages = [{"role": "user", "content": row["input_text"]}]
        prefix = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False)
        if hasattr(prefix, "keys"):
            prefix = prefix["input_ids"]
        if prefix and isinstance(prefix[0], list):
            prefix = prefix[0]
        target = tokenizer.encode(row["target_text"], add_special_tokens=False) + [eos]
        ids = list(prefix) + target
        if not prefix or not target or len(ids) > max_length:
            raise ValueError(f"Record {row['record_id']} exceeds context; truncation is forbidden")
        records.append({
            "record_id": row["record_id"],
            "record_kind": row.get("record_kind", "policy_decision"),
            "trajectory_id": row.get("trajectory_id"),
            "paired_group_id": row["paired_group_id"],
            "family_id": row["family_id"],
            "danger_level": row["danger_level"],
            "decision_index": row.get("decision_index"),
            "target_action_id": row.get("target_action_id"),
            "split": row["split"],
            "input_ids": ids,
            "labels": [-100] * len(prefix) + target,
            "prefix_ids": list(prefix),
            "target_ids": list(target),
            "prompt_tokens": len(prefix),
            "target_tokens": len(target),
            "token_count": len(ids),
            "serialization_contract_version": SERIALIZATION_CONTRACT_VERSION,
        })
    return records


def equalize_matched_prompt_exposure(
        records_by_condition: Mapping[str, Sequence[dict]], tokenizer,
        max_length: int) -> dict[str, list[dict]]:
    """Add harmless whitespace tokens so matched causal prompts use equal tokens.

    Interactive histories correctly diverge after earlier decisions.  Exact
    exposure matching is therefore performed after tokenization, per matched
    decision/danger key, using a tokenizer-native whitespace token inserted at
    the end of the prompt.  Targets and supervised labels are unchanged.
    """
    if set(records_by_condition) != set(CONDITIONS):
        raise PermissionError("All causal arms are required for prompt equalization")
    whitespace = tokenizer.encode(" ", add_special_tokens=False)
    if len(whitespace) != 1:
        raise PermissionError("Pinned tokenizer lacks a single harmless whitespace token")
    filler = whitespace[0]
    copies = {condition: [dict(row) for row in rows]
              for condition, rows in records_by_condition.items()}
    indexes = {
        condition: {(row["paired_group_id"], row["danger_level"]): row
                    for row in rows}
        for condition, rows in copies.items()}
    reference_keys = set(indexes[CONDITIONS[0]])
    if any(set(indexes[condition]) != reference_keys for condition in CONDITIONS[1:]):
        raise PermissionError("Causal decision keys differ before token equalization")
    for key in sorted(reference_keys):
        maximum = max(indexes[condition][key]["prompt_tokens"]
                      for condition in CONDITIONS)
        for condition in CONDITIONS:
            row = indexes[condition][key]
            padding = maximum - row["prompt_tokens"]
            if len(row["input_ids"]) + padding > max_length:
                raise PermissionError("Exposure equalization would exceed the context")
            row["prefix_ids"] = list(row["prefix_ids"]) + [filler] * padding
            row["input_ids"] = row["prefix_ids"] + list(row["target_ids"])
            row["labels"] = [-100] * len(row["prefix_ids"]) + list(row["target_ids"])
            row["prompt_tokens"] = maximum
            row["token_count"] = len(row["input_ids"])
            row["exposure_padding_tokens"] = padding
    return copies


def _eligible_index(records: Sequence[dict], plan: TrainingPlan):
    levels = STAGE_LEVELS[plan.stage]
    eligible = [row for row in records
                if row["split"] == "train" and row["danger_level"] in levels]
    index = {}
    for row in eligible:
        key = (row["paired_group_id"], row["danger_level"])
        if key in index:
            raise ValueError("Eligible occurrence key collision")
        index[key] = row
    if not index:
        raise PermissionError("No eligible training records")
    observed = {row["danger_level"] for row in eligible}
    if observed != set(levels):
        raise PermissionError("Sequential contrast or level-0 anchor is missing")
    return index


def occurrence_key_schedule(records: Sequence[dict], plan: TrainingPlan) -> list[tuple[str, int]]:
    """Freeze an exact deterministic, family-balanced occurrence order."""
    plan.validate()
    index = _eligible_index(records, plan)
    by_family = defaultdict(list)
    for key, row in index.items():
        by_family[row["family_id"]].append(key)
    rng = random.Random(plan.seed)
    family_names = sorted(by_family)
    for keys in by_family.values():
        keys.sort()
        rng.shuffle(keys)
    schedule = []
    cursors = Counter()
    while len(schedule) < plan.occurrence_count:
        cycle = list(family_names)
        rng.shuffle(cycle)
        for family in cycle:
            keys = by_family[family]
            schedule.append(keys[cursors[family] % len(keys)])
            cursors[family] += 1
            if len(schedule) == plan.occurrence_count:
                break
    return schedule


def build_occurrence_journal(records: Sequence[dict], plan: TrainingPlan,
                             *, key_schedule=None) -> tuple[list[dict], list[dict]]:
    index = _eligible_index(records, plan)
    keys = list(key_schedule or occurrence_key_schedule(records, plan))
    if len(keys) != plan.occurrence_count or any(key not in index for key in keys):
        raise PermissionError("Occurrence schedule is incompatible with the snapshot")
    selected, journal = [], []
    for occurrence, key in enumerate(keys):
        row = index[key]
        selected.append(row)
        journal.append({
            "occurrence_index": occurrence,
            "optimizer_update": occurrence // plan.global_batch_size,
            "global_batch_position": occurrence % plan.global_batch_size,
            "rank": (occurrence % plan.global_batch_size) % plan.world_size,
            "rank_occurrence_index": occurrence // plan.world_size,
            "record_id": row["record_id"],
            "paired_group_id": row["paired_group_id"],
            "family_id": row["family_id"],
            "danger_level": row["danger_level"],
            "prompt_tokens": row["prompt_tokens"],
            "target_tokens": row["target_tokens"],
            "token_count": row["token_count"],
        })
    return selected, journal


def verify_rank_receipt(receipt: Mapping, *, expected_plan_sha256: str,
                        expected_global_journal_sha256: str) -> dict:
    """Recompute a DDP rank receipt self-hash before cross-rank agreement."""
    value = dict(receipt)
    embedded = value.pop("receipt_sha256", None)
    if not isinstance(embedded, str) or canonical_sha256(value) != embedded:
        raise PermissionError("DDP rank receipt self-hash differs")
    if (receipt.get("status") != "complete"
            or receipt.get("plan_sha256") != expected_plan_sha256
            or receipt.get("global_journal_sha256")
            != expected_global_journal_sha256):
        raise PermissionError("DDP rank receipt does not bind the exact run")
    return dict(receipt)


def adjacent_levels(stage: str) -> tuple[int, int]:
    if stage not in ("delta1", "delta2", "delta3"):
        raise ValueError("Adjacent contrasts exist only for ordinal residual stages")
    high = int(stage[-1])
    return high - 1, high


def adjacent_pair_schedule(records: Sequence[dict], plan: TrainingPlan) -> list[dict]:
    """Build a family-balanced global schedule of exact matched level pairs."""
    plan.validate()
    low, high = adjacent_levels(plan.stage)
    index = {(row["paired_group_id"], row["danger_level"]): row for row in records
             if row["split"] == "train" and row["danger_level"] in (low, high)}
    groups = sorted({group for group, level in index if level == low})
    if not groups or any((group, high) not in index for group in groups):
        raise PermissionError("An adjacent danger contrast is incomplete")
    by_family = defaultdict(list)
    for group in groups:
        left, right = index[(group, low)], index[(group, high)]
        if left["family_id"] != right["family_id"]:
            raise PermissionError("Adjacent pair crosses scenario families")
        if (left.get("decision_index") != right.get("decision_index")
                or (left.get("record_kind") == "policy_decision"
                    and left.get("trajectory_id", "").rsplit(":d", 1)[0]
                    != right.get("trajectory_id", "").rsplit(":d", 1)[0])):
            raise PermissionError("Adjacent pair does not bind the same decision slot")
        by_family[left["family_id"]].append(group)
    rng = random.Random(plan.seed)
    families = sorted(by_family)
    for values in by_family.values():
        values.sort()
        rng.shuffle(values)
    cursors = Counter()
    schedule = []
    while len(schedule) < plan.occurrence_count:
        cycle = list(families)
        rng.shuffle(cycle)
        for family in cycle:
            group = by_family[family][cursors[family] % len(by_family[family])]
            cursors[family] += 1
            occurrence = len(schedule)
            schedule.append({
                "occurrence_index": occurrence,
                "optimizer_update": occurrence // plan.global_batch_size,
                "global_batch_position": occurrence % plan.global_batch_size,
                "rank": (occurrence % plan.global_batch_size) % plan.world_size,
                "paired_group_id": group,
                "family_id": family,
                "low_level": low,
                "high_level": high,
                "low_record_id": index[(group, low)]["record_id"],
                "high_record_id": index[(group, high)]["record_id"],
                "decision_index": index[(group, high)].get("decision_index"),
                "low_prompt_tokens": index[(group, low)]["prompt_tokens"],
                "high_prompt_tokens": index[(group, high)]["prompt_tokens"],
                "low_target_tokens": index[(group, low)]["target_tokens"],
                "high_target_tokens": index[(group, high)]["target_tokens"],
                "pair_training_tokens": (
                    2 * index[(group, low)]["prompt_tokens"]
                    + 2 * index[(group, high)]["prompt_tokens"]
                    + 2 * index[(group, low)]["target_tokens"]
                    + 2 * index[(group, high)]["target_tokens"]),
                "target_changed": (
                    index[(group, low)]["target_ids"] !=
                    index[(group, high)]["target_ids"]),
                "low_target_sha256": canonical_sha256(index[(group, low)]["target_ids"]),
                "high_target_sha256": canonical_sha256(index[(group, high)]["target_ids"]),
            })
            if len(schedule) == plan.occurrence_count:
                break
    return schedule


def shard_global_journal(journal: Sequence[dict], plan: TrainingPlan, rank: int) -> list[dict]:
    """Shard one canonical journal without padding or duplicated exposure."""
    if rank not in range(plan.world_size):
        raise ValueError("Invalid DDP rank")
    shards = [[row for row in journal if row["rank"] == candidate]
              for candidate in range(plan.world_size)]
    flattened = [row["occurrence_index"] for shard in shards for row in shard]
    if sorted(flattened) != list(range(len(journal))) or len(flattened) != len(set(flattened)):
        raise PermissionError("Global DDP journal is duplicated or incomplete")
    expected = plan.optimizer_updates * plan.micro_batch_size * plan.accumulation_steps
    if any(len(shard) != expected for shard in shards):
        raise PermissionError("DDP shards are not exposure-balanced")
    return shards[rank]


def validate_equal_occurrence_exposure(records_by_condition: Mapping[str, Sequence[dict]],
                                       plan: TrainingPlan) -> dict:
    """Recompute actual schedule and token exposure for every causal condition."""
    if set(records_by_condition) != set(CONDITIONS):
        raise PermissionError("All three tokenized causal conditions are required")
    journals = {condition: adjacent_pair_schedule(records_by_condition[condition], plan)
                for condition in CONDITIONS}
    comparison_fields = (
        "occurrence_index", "optimizer_update", "global_batch_position", "rank",
        "paired_group_id", "family_id", "low_level", "high_level",
        "decision_index", "low_target_tokens", "high_target_tokens",
    )
    reference = journals[CONDITIONS[0]]
    for condition in CONDITIONS[1:]:
        for left, right in zip(reference, journals[condition]):
            if any(left[field] != right[field] for field in comparison_fields):
                raise PermissionError(
                    f"Condition {condition} does not have exact matched pair exposure")
    summary = {
        condition: {
            "pair_occurrences": len(journal),
            "pair_training_tokens": sum(row["pair_training_tokens"] for row in journal),
            "family_counts": dict(Counter(row["family_id"] for row in journal)),
            "changed_target_occurrences": sum(row["target_changed"] for row in journal),
            "journal_sha256": canonical_sha256(journal),
        } for condition, journal in journals.items()
    }
    if len({entry["pair_training_tokens"] for entry in summary.values()}) != 1:
        raise PermissionError("Causal arms do not have exact matched policy-token exposure")
    return {
        "version": "ordinal-continuity-adjacent-pair-exposure-v3",
        "stage": plan.stage, "seed": plan.seed,
        "decision_corpus_counts": {
            condition: {
                "all_splits": len(records_by_condition[condition]),
                "train": sum(row["split"] == "train" for row in records_by_condition[condition]),
                "validation": sum(row["split"] == "validation" for row in records_by_condition[condition]),
                "test": sum(row["split"] == "test" for row in records_by_condition[condition]),
            } for condition in CONDITIONS
        },
        "global_journal_sha256_by_condition": {
            condition: canonical_sha256(journal) for condition, journal in journals.items()},
        "conditions": summary,
    }


def optimizer_sha256(optimizer) -> str:
    """Hash optimizer tensors and scalar state without serialization timestamps."""
    state = optimizer.state_dict()
    digest = hashlib.sha256()

    def visit(prefix, value):
        if hasattr(value, "detach"):
            tensor = value.detach().contiguous().cpu()
            digest.update(prefix.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(
                tensor.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=lambda item: str(item)):
                visit(f"{prefix}/{key}", value[key])
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(f"{prefix}/{index}", item)
        else:
            digest.update(prefix.encode("utf-8"))
            digest.update(repr(value).encode("utf-8"))

    visit("optimizer", state)
    return digest.hexdigest()


def choose_qualification_rank(rank4_metrics: Mapping[str, object],
                              rank8_metrics: Mapping[str, object] | None = None) -> int:
    """Admit rank 8 only when rank 4 was explicitly inadequate."""
    required = ("architecture_pass", "data_pass", "checkpoint_recompute_pass",
                "sequential_freeze_pass", "backend_pass", "memory_pass",
                "save_reload_pass", "throughput_pass", "nvml_pass",
                "storage_export_pass", "numerical_pass")
    if any(key not in rank4_metrics for key in required + ("capacity_adequate",)):
        raise ValueError("Rank-4 qualification metrics are incomplete")
    structural = all(rank4_metrics[key] is True for key in required)
    if structural and rank4_metrics["capacity_adequate"] is True:
        if rank8_metrics is not None:
            raise PermissionError("Rank 8 must not be tested after rank 4 is adequate")
        return 4
    if not structural:
        raise PermissionError("Rank 8 cannot repair architecture/data/runtime failures")
    if rank8_metrics is None:
        raise PermissionError("Rank 4 is capacity-inadequate and rank 8 is unqualified")
    if any(rank8_metrics.get(key) is not True for key in required + ("capacity_adequate",)):
        raise PermissionError("Rank-8 qualification failed")
    return 8


def create_qualification_receipt(*, plan: TrainingPlan, data_binding: Mapping,
                                 exposure_binding: Mapping, source_sha256: str,
                                 base_before_sha256: str, base_after_sha256: str,
                                 backend_proof: Mapping,
                                 rank4_metrics: Mapping[str, object],
                                 primary_plan: TrainingPlan | None = None,
                                 primary_plans: Sequence[TrainingPlan] | None = None,
                                 primary_data_bindings: Mapping[str, Mapping] | None = None,
                                 primary_exposure_bindings: Mapping[str, Mapping] | None = None,
                                 rank8_metrics: Mapping[str, object] | None = None) -> dict:
    if plan.mode != "qualification":
        raise ValueError("A qualification receipt requires qualification mode")
    selected_rank = choose_qualification_rank(rank4_metrics, rank8_metrics)
    declared = list(primary_plans or (() if primary_plan is None else (primary_plan,)))
    if not declared:
        raise PermissionError("Qualification must bind at least one exact primary plan")
    invariant_fields = (
        "model_id", "model_revision", "rank", "alpha", "precision", "quantization",
        "device_profile", "world_size", "global_batch_size", "sft_weight",
        "micro_batch_size", "accumulation_steps", "learning_rate", "weight_decay",
        "max_length", "max_new_tokens", "expected_train_families",
        "expected_validation_families", "expected_internal_test_families",
        "preference_weight", "preference_beta",
        "cross_danger_margin_weight", "cross_danger_margin", "max_grad_norm",
        "automatic_retries",
    )
    for primary in declared:
        primary.validate()
        geometry_ok = (primary.rank == selected_rank
                       and primary.alpha == 2.0 * selected_rank)
        comparable = tuple(field for field in invariant_fields
                           if field not in {"rank", "alpha"})
        if (primary.mode != "primary" or not geometry_ok
                or any(getattr(primary, field) != getattr(plan, field)
                       for field in comparable)):
            raise PermissionError("Qualification is not bound to an exact compatible primary plan")
    primary_hashes = sorted({primary.full_plan_sha256 for primary in declared})
    if len(primary_hashes) != len(declared):
        raise PermissionError("Duplicate primary plans are not allowed")
    exposure_map = dict(primary_exposure_bindings or {
        primary_hash: exposure_binding for primary_hash in primary_hashes})
    if set(exposure_map) != set(primary_hashes):
        raise PermissionError("Every exact primary plan needs one exposure binding")
    exposure_hashes = {key: canonical_sha256(value) for key, value in exposure_map.items()}
    data_map = dict(primary_data_bindings or {
        primary_hash: data_binding for primary_hash in primary_hashes})
    if set(data_map) != set(primary_hashes):
        raise PermissionError("Every exact primary plan needs one data binding")
    data_hashes = {key: canonical_sha256(value) for key, value in data_map.items()}
    if base_before_sha256 != base_after_sha256:
        raise PermissionError("Qualification changed the frozen backbone")
    receipt = {
        "version": QUALIFICATION_VERSION,
        "status": "qualified",
        "model_id": plan.model_id,
        "model_revision": plan.model_revision,
        "condition": plan.condition,
        "stage": plan.stage,
        "seed": plan.seed,
        "qualification_plan_sha256": plan.full_plan_sha256,
        "primary_plan_sha256": primary_hashes[0],
        "authorized_primary_plan_sha256s": primary_hashes,
        "authorized_primary_exposure_sha256_by_plan": exposure_hashes,
        "authorized_primary_data_sha256_by_plan": data_hashes,
        "selected_rank": selected_rank,
        "rank4_metrics": dict(rank4_metrics),
        "rank8_metrics": None if rank8_metrics is None else dict(rank8_metrics),
        "data_binding_sha256": canonical_sha256(data_binding),
        "exposure_binding_sha256": canonical_sha256(exposure_binding),
        "source_sha256": source_sha256,
        "base_before_sha256": base_before_sha256,
        "base_after_sha256": base_after_sha256,
        "backend_proof": dict(backend_proof),
        "execution_provenance": (
            "actual_databricks_a10x1_worker_evidence"
            if plan.device_profile == "databricks-a10x1"
            else "actual_kaggle_t4x2_worker_evidence"
        ),
        "receipt_builder_submitted_provider_job": False,
        "qualification_checkpoint_reusable": False,
        "initialized_from": "fresh_seeded_adapter",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def write_qualification_receipt(path, receipt: Mapping) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Qualification receipt is immutable")
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return file_sha256(destination)


def validate_sequential_parent(plan: TrainingPlan,
                               parent_checkpoint: Mapping | None,
                               *, source_sha256: str) -> dict:
    """Bind each primary stage to the immediately preceding study checkpoint."""
    expected_parent = {
        "delta1": None, "delta2": "delta1", "delta3": "delta2",
    }[plan.stage]
    if expected_parent is None:
        if parent_checkpoint is not None:
            raise PermissionError("Delta1 must begin from fresh residual adapters")
        return {"parent_stage": None, "parent_weights_sha256": None}
    if not isinstance(parent_checkpoint, Mapping):
        raise PermissionError(f"Stage {plan.stage} requires the {expected_parent} checkpoint")
    config = parent_checkpoint.get("config", {})
    expected_condition = plan.condition
    expected_seed = plan.seed
    branches = STAGE_BRANCHES[expected_parent]
    branch_meta = parent_checkpoint.get("branches", {})
    optimizer = parent_checkpoint.get("optimizer_manifest", {})
    if (parent_checkpoint.get("version") != CHECKPOINT_VERSION
            or parent_checkpoint.get("run_mode") != "primary"
            or parent_checkpoint.get("primary_parent_eligible") is not True
            or parent_checkpoint.get("stage") != expected_parent
            or parent_checkpoint.get("condition") != expected_condition
            or parent_checkpoint.get("seed") != expected_seed
            or parent_checkpoint.get("source_sha256") != source_sha256
            or config.get("model_id") != plan.model_id
            or config.get("model_revision") != plan.model_revision
            or config.get("initialized_from") != "fresh_seeded_adapter"
            or tuple(parent_checkpoint.get("active_branches", ())) != branches
            or not parent_checkpoint.get("target_names")
            or any(not isinstance(branch_meta.get(branch, {}).get("state_sha256"), str)
                   or len(branch_meta[branch]["state_sha256"]) != 64
                   for branch in branches)
            or optimizer.get("class") != "AdamW"
            or optimizer.get("state_sha256") != parent_checkpoint.get("optimizer_sha256")
            or optimizer.get("successful_updates", 0) <= 0
            or any(dtype != "torch.float32" for dtype in optimizer.get("state_dtypes", ()))):
        raise PermissionError("Sequential parent checkpoint lineage differs")
    return {
        "parent_stage": expected_parent,
        "parent_condition": expected_condition,
        "parent_seed": expected_seed,
        "parent_adapter_sha256": parent_checkpoint.get("adapter_sha256"),
        "parent_branch_hashes": {
            branch: branch_meta[branch]["state_sha256"] for branch in branches},
    }


def admit_primary_fit(*, plan: TrainingPlan, qualification_receipt: Mapping,
                      data_binding: Mapping, exposure_binding: Mapping,
                      source_sha256: str, no_active_worker: bool,
                      one_use_submission_intent: str | None,
                      parent_checkpoint: Mapping | None) -> dict:
    """Return a local admission binding; this does not submit a workload."""
    plan.validate()
    if plan.mode != "primary":
        raise ValueError("Primary admission requires primary mode")
    receipt = dict(qualification_receipt)
    embedded_hash = receipt.pop("receipt_sha256", None)
    if (receipt.get("version") != QUALIFICATION_VERSION
            or receipt.get("status") != "qualified"
            or canonical_sha256(receipt) != embedded_hash):
        raise PermissionError("Qualification receipt is absent or altered")
    selected_geometry_ok = receipt.get("selected_rank") == plan.rank
    if (receipt.get("model_id") != plan.model_id
            or receipt.get("model_revision") != plan.model_revision
            or not selected_geometry_ok
            or receipt.get("source_sha256") != source_sha256
            or receipt.get("authorized_primary_data_sha256_by_plan", {}).get(
                plan.full_plan_sha256) != canonical_sha256(data_binding)
            or receipt.get("initialized_from") != "fresh_seeded_adapter"):
        raise PermissionError("Primary fit differs from the exact qualified recipe")
    if plan.full_plan_sha256 not in receipt.get("authorized_primary_plan_sha256s", ()):
        raise PermissionError("Primary full-plan hash differs from qualification")
    if (receipt.get("authorized_primary_exposure_sha256_by_plan", {}).get(
            plan.full_plan_sha256) != canonical_sha256(exposure_binding)):
        raise PermissionError("Primary occurrence exposure differs from qualification")
    if not no_active_worker or not one_use_submission_intent:
        raise PermissionError("One idle-worker check and exact one-use intent are required")
    lineage = validate_sequential_parent(
        plan, parent_checkpoint, source_sha256=source_sha256)
    admission = {
        "version": ADMISSION_VERSION,
        "admitted": True,
        "plan": asdict(plan),
        "full_plan_sha256": plan.full_plan_sha256,
        "qualification_receipt_sha256": embedded_hash,
        "data_binding_sha256": canonical_sha256(data_binding),
        "exposure_binding_sha256": canonical_sha256(exposure_binding),
        "source_sha256": source_sha256,
        "one_use_submission_intent": one_use_submission_intent,
        "sequential_parent": lineage,
        "provider_submission_performed": False,
    }
    admission["admission_sha256"] = canonical_sha256(admission)
    return admission


def assert_fast_deltanet_dependencies() -> dict:
    """Fail rather than silently using a slow reference DeltaNet path."""
    try:
        versions = {name: package_version(name) for name in PINNED_SOFTWARE
                    if name != "torch"}
        import torch
        versions["torch"] = torch.__version__
        if versions != PINNED_SOFTWARE:
            raise RuntimeError(f"Pinned runtime differs: {versions}")
        from .composite_kernels import select
        selected = select("fla")
    except (ImportError, PackageNotFoundError) as exc:
        raise RuntimeError("Pinned fast DeltaNet dependencies are unavailable") from exc
    proof = {
        **versions,
        "kernel_selection": selected,
    }
    if (selected.get("profile") != "fla"
            or any(not value.startswith("fla.")
                   for value in selected.get("GDN_functions", {}).values())):
        raise RuntimeError("Fast DeltaNet dispatch is required")
    return proof


def assert_t4x2_fp16_runtime(torch_module=None) -> dict:
    """Verify the reviewed hardware intent after, never before, qualification."""
    torch = torch_module or __import__("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("The reviewed placement requires exactly two visible GPUs")
    devices = []
    for index in range(2):
        properties = torch.cuda.get_device_properties(index)
        name = torch.cuda.get_device_name(index)
        if "T4" not in name.upper():
            raise RuntimeError("The Kaggle qualification is specific to T4x2")
        devices.append({
            "index": index,
            "name": name,
            "total_memory_bytes": int(properties.total_memory),
            "major": int(properties.major),
            "minor": int(properties.minor),
        })
    return {"profile": "kaggle-t4x2", "compute_dtype": "float16",
            "adapter_dtype": "float32", "devices": devices}


def assert_databricks_a10x1_fp16_runtime(torch_module=None) -> dict:
    """Verify the separate Databricks execution envelope without changing recipe."""
    torch = torch_module or __import__("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Databricks qualification requires exactly one visible GPU")
    devices = []
    for index in range(1):
        properties = torch.cuda.get_device_properties(index)
        name = torch.cuda.get_device_name(index)
        if "A10" not in name.upper() or int(properties.total_memory) < 20 * 1024**3:
            raise RuntimeError("Databricks qualification requires one >=20 GiB A10 GPU")
        devices.append({"index": index, "name": name,
                        "total_memory_bytes": int(properties.total_memory),
                        "major": int(properties.major), "minor": int(properties.minor)})
    return {"profile": "databricks-a10x1", "compute_dtype": "float16",
            "adapter_dtype": "float32", "devices": devices}


# Deliberate compatibility name: callers now receive the reviewed FP16 proof.
assert_t4x2_bf16_runtime = assert_t4x2_fp16_runtime


def qwen35_t4x2_device_map(config, local_rank: int = 0) -> dict:
    text = getattr(config, "text_config", config)
    count = getattr(text, "num_hidden_layers", None)
    if count != 32:
        raise RuntimeError("Expected 32 Qwen3.5 text layers")
    if local_rank not in (0, 1):
        raise ValueError("local_rank must be 0 or 1")
    # DDP owns one complete NF4 replica per T4.  A split device-map would be
    # model parallelism and would duplicate the canonical exposure journal.
    return {"": local_rank}


def assert_qwen35_backend(model, dependency_proof: Mapping) -> dict:
    config = getattr(model.config, "text_config", model.config)
    layer_types = tuple(getattr(config, "layer_types", ()))
    if (model.__class__.__name__ != "Qwen3_5ForConditionalGeneration"
            or getattr(config, "num_hidden_layers", None) != 32
            or Counter(layer_types) != EXPECTED_TEXT_LAYER_TYPES):
        raise RuntimeError("Loaded model is not the pinned 24:8 Qwen3.5 text stack")
    modules = dict(model.named_modules())
    linear = [name for name in modules if name.endswith(".linear_attn")]
    full = [name for name in modules if name.endswith(".self_attn")]
    if (len(linear) != 24 or len(full) != 8
            or any(modules[name].__class__.__name__ != "Qwen3_5GatedDeltaNet"
                   for name in linear)
            or any(modules[name].__class__.__name__ != "Qwen3_5Attention"
                   for name in full)):
        raise RuntimeError("Qwen3.5 attention implementations differ")
    return {
        "model_class": model.__class__.__name__,
        "linear_attention_layers": 24,
        "full_attention_layers": 8,
        **dict(dependency_proof),
    }


def assert_ordinal_adapter_geometry(model, targets: Sequence[str], plan: TrainingPlan) -> dict:
    if len(targets) != EXPECTED_TARGET_PROJECTIONS or len(set(targets)) != len(targets):
        raise RuntimeError("Official Qwen3.5-9B text target count differs")
    inventory = trainable_inventory(model)
    if any(info["dtype"] != "torch.float32" for info in inventory.values()):
        raise RuntimeError("Every trainable adapter tensor must be FP32")
    expected = 0
    layers = getattr(model, "layers", {})
    dimension_sum = 0
    for layer in layers.values():
        branch = plan.stage
        a = getattr(layer, f"{branch}_a")
        b = getattr(layer, f"{branch}_b")
        expected += a.weight.numel() + b.weight.numel()
        dimension_sum += a.in_features + b.out_features
    observed = sum(info["count"] for info in inventory.values())
    exact = (EXPECTED_RANK4_BRANCH_PARAMETERS if plan.rank == 4
             else EXPECTED_RANK8_BRANCH_PARAMETERS)
    if (expected != observed or observed != exact
            or dimension_sum != EXPECTED_TARGET_DIMENSION_SUM
            or len(inventory) != 2 * EXPECTED_TARGET_PROJECTIONS):
        raise RuntimeError("Trainable adapter parameter count differs from target geometry")
    return {"target_projection_count": len(targets),
            "target_dimension_sum": dimension_sum,
            "trainable_tensor_count": len(inventory),
            "trainable_parameter_count": observed,
            "expected_trainable_parameter_count": exact,
            "rank": plan.rank, "alpha": plan.alpha,
            "alpha_over_rank": plan.alpha / plan.rank}


def _build_model(plan: TrainingPlan, *, local_rank: int, dependency_proof=None):
    """Construct one DDP-rank replica; caller owns distributed wrapping."""
    import torch
    from transformers import (
        AutoConfig, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration,
    )

    databricks = os.environ.get("RESEARCH4_EXECUTION_PLATFORM") == "databricks"
    if databricks != (plan.device_profile == "databricks-a10x1"):
        raise PermissionError("Execution platform and reviewed hardware profile differ")
    hardware = (assert_databricks_a10x1_fp16_runtime(torch) if databricks
                else assert_t4x2_fp16_runtime(torch))
    seed_all(plan.seed, torch)
    dependencies = dependency_proof or assert_fast_deltanet_dependencies()
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model_root = os.environ.get("RESEARCH4_MODEL_ROOT") if databricks else None
    if databricks and not model_root:
        raise PermissionError("Databricks scientific runs require the verified volume model")
    model_source = model_root or plan.model_id
    public_config = AutoConfig.from_pretrained(
        model_source, revision=None if model_root else plan.model_revision,
        local_files_only=bool(model_root), trust_remote_code=False)
    device_map = qwen35_t4x2_device_map(public_config, local_rank)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_source, revision=None if model_root else plan.model_revision,
        local_files_only=bool(model_root),
        quantization_config=quantization, torch_dtype=torch.float16,
        device_map=device_map, trust_remote_code=False)
    backend = assert_qwen35_backend(model, dependencies)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    backend["gradient_checkpointing_enabled"] = bool(
        getattr(model, "is_gradient_checkpointing", False))
    backend["gradient_checkpointing_use_reentrant"] = False
    if not backend["gradient_checkpointing_enabled"]:
        raise RuntimeError("Gradient checkpointing did not become active")
    model.enable_input_require_grads()
    targets = qwen35_text_target_names(model)
    adapter_config = OrdinalContinuityLoRAConfig(
        rank=plan.rank, alpha=plan.alpha, model_revision=plan.model_revision,
        initialized_from=plan.initialized_from)
    layers = install_ordinal_lora(model, targets, adapter_config)
    wrapped = OrdinalContinuityModel.build(model, layers, adapter_config)
    wrapped.set_train_stage(plan.stage)
    geometry = assert_ordinal_adapter_geometry(wrapped, targets, plan)
    return wrapped, targets, {
        "hardware": hardware, "backend": backend, "device_map": device_map,
        "base_component_sha256": model_component_sha256(wrapped),
        "live_base_components": model_component_hashes(
            wrapped, require_quantized=True),
        "trainable_inventory": trainable_inventory(wrapped),
        "geometry": geometry,
    }


def build_qualification_model(plan: TrainingPlan, *, local_rank: int,
                              dependency_proof=None):
    if plan.mode != "qualification":
        raise PermissionError("Qualification model requires qualification mode")
    return _build_model(plan, local_rank=local_rank,
                        dependency_proof=dependency_proof)


def build_qualified_model(plan: TrainingPlan, admission: Mapping, *, local_rank: int = 0,
                          dependency_proof=None):
    """Load Qwen only after exact qualification and local primary admission."""
    plan.validate()
    if plan.mode != "primary" or admission.get("admitted") is not True:
        raise PermissionError("The model can be loaded only for an admitted primary fit")
    check = dict(admission)
    embedded = check.pop("admission_sha256", None)
    if canonical_sha256(check) != embedded or admission.get("plan") != asdict(plan):
        raise PermissionError("Primary admission was altered")

    return _build_model(plan, local_rank=local_rank,
                        dependency_proof=dependency_proof)


def seed_all(seed: int, torch_module=None) -> dict:
    """Seed CPU and every visible CUDA rank without changing the plan seed."""
    torch = torch_module or __import__("torch")
    random.seed(seed)
    torch.manual_seed(seed)
    cuda_seeded = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cuda_seeded = True
    return {"python_seed": seed, "torch_cpu_seed": seed,
            "torch_cuda_seed_all": seed if cuda_seeded else None}


def load_pinned_tokenizer(plan: TrainingPlan, *, tokenizer_root=None,
                          tokenizer_receipt=None):
    from transformers import AutoTokenizer
    if tokenizer_root is None:
        databricks_root = (os.environ.get("RESEARCH4_MODEL_ROOT")
                           if os.environ.get("RESEARCH4_EXECUTION_PLATFORM") == "databricks"
                           else None)
        tokenizer = AutoTokenizer.from_pretrained(
            databricks_root or plan.model_id,
            revision=None if databricks_root else plan.model_revision,
            local_files_only=bool(databricks_root), trust_remote_code=False,
            use_fast=True)
    else:
        from .qwen9b_ordinal_continuity_runtime import verify_tokenizer_receipt
        if tokenizer_receipt is None:
            raise PermissionError("Durable tokenizer receipt is required")
        verify_tokenizer_receipt(
            tokenizer_root, tokenizer_receipt,
            model_id=plan.model_id, revision=plan.model_revision)
        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(tokenizer_root).resolve()), trust_remote_code=False,
            use_fast=True, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Pinned tokenizer has no EOS token")
    commit = tokenizer.init_kwargs.get("_commit_hash")
    if tokenizer_root is None and not databricks_root and commit != plan.model_revision:
        raise RuntimeError("Tokenizer revision differs from the pinned model revision")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class SerializedRunGuard:
    """Atomic, owner-token lock for exactly one local scientific worker."""

    def __init__(self, path, *, intent_sha256: str):
        self.path = Path(path)
        self.intent_sha256 = intent_sha256
        self.token = uuid.uuid4().hex

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical_bytes({
            "token": self.token, "intent_sha256": self.intent_sha256,
            "pid": os.getpid(), "host": socket.gethostname(), "created_ns": time.time_ns(),
        })
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError as exc:
            raise RuntimeError("A continuity-training worker already owns the lock") from exc
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
        if current.get("token") == self.token:
            self.path.unlink(missing_ok=True)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)
    return file_sha256(path)


def _candidate_sequence(row: Mapping, target_ids: Sequence[int]) -> tuple[list[int], list[int]]:
    prefix = list(row["prefix_ids"])
    target = list(target_ids)
    return prefix + target, [-100] * len(prefix) + target


def _pad_batch(sequences, labels, *, pad_token_id: int, device):
    import torch
    length = max(len(item) for item in sequences)
    ids, masks, supervised = [], [], []
    for sequence, label in zip(sequences, labels):
        pad = length - len(sequence)
        ids.append(list(sequence) + [pad_token_id] * pad)
        masks.append([1] * len(sequence) + [0] * pad)
        supervised.append(list(label) + [-100] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
        "labels": torch.tensor(supervised, dtype=torch.long, device=device),
    }


def _token_losses_and_logps(logits, labels):
    import torch
    import torch.nn.functional as F
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe = shifted_labels.masked_fill(~mask, 0)
    token_nll = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1), ignore_index=-100, reduction="none",
    ).reshape_as(shifted_labels)
    counts = mask.sum(dim=-1).clamp_min(1)
    sequence_nll = (token_nll * mask).sum(dim=-1) / counts
    sequence_logp = -sequence_nll
    if not torch.isfinite(sequence_nll).all():
        raise FloatingPointError("Non-finite token likelihood")
    return sequence_nll, sequence_logp


def _forward_logits(callable_model, batch):
    output = callable_model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
    logits = output.logits if hasattr(output, "logits") else output[0]
    if logits.ndim != 3:
        raise RuntimeError("Causal model did not return token logits")
    return logits


def _extract_logits(output):
    return output.logits if hasattr(output, "logits") else (
        output[0] if isinstance(output, (tuple, list)) else output)


def assert_level0_frozen_base_identity(model, rows: Sequence[Mapping], *, tokenizer,
                                       device) -> float:
    """Require bitwise level-0 logits to equal the adapter-off frozen base."""
    import torch
    selected = list(rows[:max(1, min(2, len(rows)))])
    sequences = [row["input_ids"] for row in selected]
    labels = [row["labels"] for row in selected]
    batch = _pad_batch(sequences, labels, pad_token_id=tokenizer.pad_token_id,
                       device=device)
    levels = torch.zeros(len(selected), dtype=torch.long, device=device)
    with torch.no_grad(), model.evaluation_mode(use_cache=False):
        current = _extract_logits(model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            danger_level=levels))[:, -1, :].float()
        from .qwen9b_ordinal_continuity_lora import bypass
        with bypass(model, adapter_off=True):
            reference = _extract_logits(model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                danger_level=levels))[:, -1, :].float()
        if not torch.equal(current, reference):
            raise PermissionError("Danger level zero differs from the adapter-off frozen base")
    return 0.0


def ordinal_pair_loss(model, callable_model, pair_rows: Sequence[tuple[Mapping, Mapping]],
                      *, plan: TrainingPlan, tokenizer, device, scaler,
                      loss_divisor: float = 1.0,
                      level0_kl_anchor: float = 0.0):
    """Backprop one matched adjacent-danger microbatch.

    The only differentiable forward uses the active residual's high level.
    Low-level scores are a frozen reference, so checkpoint recomputation never
    observes a changed episode label.
    """
    import torch
    import torch.nn.functional as F
    low_level, high_level = adjacent_levels(plan.stage)
    high_sequences, high_labels = [], []
    low_sequences, low_labels = [], []
    changed_targets = []
    for low, high in pair_rows:
        if (low.get("paired_group_id") != high.get("paired_group_id")
                or low.get("decision_index") != high.get("decision_index")):
            raise PermissionError("Matched pair is not the same supervised decision slot")
        chosen_high = _candidate_sequence(high, high["target_ids"])
        rejected_high = _candidate_sequence(high, low["target_ids"])
        chosen_low = _candidate_sequence(low, low["target_ids"])
        rejected_low = _candidate_sequence(low, high["target_ids"])
        high_sequences.extend((chosen_high[0], rejected_high[0]))
        high_labels.extend((chosen_high[1], rejected_high[1]))
        low_sequences.extend((chosen_low[0], rejected_low[0]))
        low_labels.extend((chosen_low[1], rejected_low[1]))
        changed_targets.append(high["target_ids"] != low["target_ids"])
    high_batch = _pad_batch(high_sequences, high_labels,
                            pad_token_id=tokenizer.pad_token_id, device=device)
    low_batch = _pad_batch(low_sequences, low_labels,
                           pad_token_id=tokenizer.pad_token_id, device=device)
    low_danger = torch.full((len(low_sequences),), low_level, dtype=torch.long, device=device)
    with torch.no_grad(), model.evaluation_mode(use_cache=False):
        low_logits = _extract_logits(model(
            input_ids=low_batch["input_ids"],
            attention_mask=low_batch["attention_mask"],
            danger_level=low_danger))
        _, low_logp = _token_losses_and_logps(low_logits, low_batch["labels"])
    high_danger = torch.full((len(high_sequences),), high_level,
                             dtype=torch.long, device=device)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16,
                              enabled=device.type == "cuda")
    with model.training_episode(high_danger), autocast:
        logits = _forward_logits(callable_model, high_batch)
        high_nll, high_logp = _token_losses_and_logps(logits, high_batch["labels"])
        chosen_high, rejected_high = high_logp[0::2], high_logp[1::2]
        chosen_low, rejected_low = low_logp[0::2], low_logp[1::2]
        sft = high_nll[0::2].mean()
        high_advantage = chosen_high - rejected_high
        low_advantage = (chosen_low - rejected_low).detach()
        changed_mask = torch.as_tensor(
            changed_targets, dtype=torch.bool, device=high_advantage.device)
        if bool(changed_mask.any()):
            preference = -F.logsigmoid(
                plan.preference_beta * high_advantage[changed_mask]).mean()
            margin = F.relu(
                plan.cross_danger_margin
                - (high_advantage[changed_mask] - low_advantage[changed_mask])).mean()
        else:
            # Unchanged adjacent decisions are still SFT anchors.  They cannot
            # define a preference or margin between identical action targets.
            preference = high_advantage.sum() * 0.0
            margin = high_advantage.sum() * 0.0
        loss = (plan.sft_weight * sft
                + plan.preference_weight * preference
                + plan.cross_danger_margin_weight * margin) / loss_divisor
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite ordinal loss")
        scaler.scale(loss).backward()
    return {
        "loss": float((loss.detach() * loss_divisor).item()),
        "sft": float(sft.detach().item()),
        "preference": float(preference.detach().item()),
        "cross_danger_margin": float(margin.detach().item()),
        "level0_kl_anchor": float(level0_kl_anchor),
        "high_advantage": float(high_advantage.detach().mean().item()),
        "low_advantage": float(low_advantage.detach().mean().item()),
    }


def _grad_scaler(torch, enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class NvmlSampler:
    """Collect one utilization sample per second on the rank-owned GPU."""

    def __init__(self, device_index: int | None, interval_seconds: float = 1.0):
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self.samples = []
        self.error = None
        self._stop = threading.Event()
        self._thread = None
        self._pynvml = None
        self._handle = None

    def _sample(self):
        try:
            utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.samples.append({
                "monotonic_ns": time.monotonic_ns(),
                "gpu_utilization_percent": int(utilization.gpu),
                "memory_utilization_percent": int(utilization.memory),
                "memory_used_bytes": int(memory.used),
            })
        except BaseException as exc:  # evidence is fail-closed, never hidden
            self.error = f"{type(exc).__name__}: {exc}"
            self._stop.set()

    def _loop(self):
        self._sample()
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self):
        if self.device_index is None:
            self.error = "NVML sampling requires a CUDA device"
            return self
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        return self

    def summary(self, *, final_sample=False):
        if final_sample and self._handle is not None and self.error is None:
            self._sample()
        nonzero = sum(row["gpu_utilization_percent"] > 0 for row in self.samples)
        return {
            "backend": "pynvml",
            "interval_seconds": self.interval_seconds,
            "available": self._handle is not None and self.error is None,
            "error": self.error,
            "sample_count": len(self.samples),
            "nonzero_gpu_sample_count": nonzero,
            "max_gpu_utilization_percent": max(
                (row["gpu_utilization_percent"] for row in self.samples), default=None),
            "samples": list(self.samples),
        }

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except BaseException:
                pass


def _resource_snapshot(torch, device, *, output_root: Path, tokens: int,
                       elapsed: float, nvml: NvmlSampler | None = None) -> dict:
    gpu = {}
    if device.type == "cuda":
        gpu = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory),
        }
        evidence = (nvml.summary(final_sample=True) if nvml is not None else {
            "backend": "pynvml", "available": False,
            "error": "sampler was not started", "sample_count": 0,
            "nonzero_gpu_sample_count": 0, "samples": []})
        gpu["nvml"] = evidence
        gpu["nvml_utilization_percent"] = evidence.get(
            "max_gpu_utilization_percent")
    disk = shutil.disk_usage(output_root)
    return {"gpu": gpu, "tokens_per_second": tokens / max(elapsed, 1e-9),
            "disk_free_bytes": int(disk.free), "disk_total_bytes": int(disk.total)}


def _optimizer_manifest(optimizer, plan: TrainingPlan, successful_updates: int) -> dict:
    import torch
    state_dtypes = sorted({str(value.dtype) for state in optimizer.state.values()
                           for value in state.values() if torch.is_tensor(value)})
    if state_dtypes and any(dtype != "torch.float32" for dtype in state_dtypes):
        raise RuntimeError("AdamW state must remain FP32")
    return {"class": optimizer.__class__.__name__, "learning_rate": plan.learning_rate,
            "weight_decay": plan.weight_decay, "successful_updates": successful_updates,
            "state_dtypes": state_dtypes, "state_sha256": optimizer_sha256(optimizer)}


class FastDispatchTracker:
    """Count actual gated-DeltaNet module forward and backward dispatches."""

    def __init__(self, model):
        self.model = model
        self.forward_count = 0
        self.grad_forward_count = 0
        self.no_grad_forward_count = 0
        self.backward_count = 0
        self.module_names = []
        self._handles = []

    def _hook(self, name):
        def hook(_module, _inputs, output):
            self.forward_count += 1
            import torch
            if torch.is_grad_enabled():
                self.grad_forward_count += 1
            else:
                self.no_grad_forward_count += 1
            values = output if isinstance(output, (tuple, list)) else (output,)
            for value in values:
                if hasattr(value, "requires_grad") and value.requires_grad:
                    value.register_hook(self._backward_hook)
                    break
        return hook

    def _backward_hook(self, gradient):
        self.backward_count += 1
        return gradient

    def __enter__(self):
        for name, module in self.model.named_modules():
            if (name.endswith(".linear_attn")
                    and module.__class__.__name__ == "Qwen3_5GatedDeltaNet"):
                self.module_names.append(name)
                self._handles.append(module.register_forward_hook(self._hook(name)))
        # Tiny production-like mocks may expose fewer modules, but the real
        # backend assertion separately requires exactly 24.
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def run_training_stage(*, model, tokenizer, tokenized_records: Sequence[dict],
                       plan: TrainingPlan, output_dir, target_names: Sequence[str],
                       source_sha256: str, base_sha256: str,
                       parent_checkpoint_dir=None, rank: int | None = None,
                       world_size: int | None = None, save_checkpoint=True,
                       callable_model=None, progress_callback=None) -> dict:
    """Execute one exact sequential stage on one declared runtime rank.

    Databricks A10 execution uses one process and one visible GPU.  The closed
    Kaggle profile retains its two-rank DDP geometry.  In either profile each
    rank consumes only its deterministic shard of the canonical journal.
    """
    import torch
    plan.validate()
    rank = int(os.environ.get("RANK", "0")) if rank is None else rank
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if world_size is None else world_size
    if world_size != plan.world_size or rank not in range(world_size):
        raise RuntimeError("Primary/qualification rank differs from the exact plan")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    seed_proof = seed_all(plan.seed, torch)
    predecessor_hashes = {}
    parent_metadata = None
    expected_parent = {"delta1": None, "delta2": "delta1", "delta3": "delta2"}[
        plan.stage]
    if expected_parent is None:
        if parent_checkpoint_dir is not None:
            raise PermissionError("Delta1 must start from fresh residual state")
        parent_metadata = {
            "frozen_base_parent": True,
            "qualifier_state_discarded": plan.mode == "qualification",
        }
    elif parent_checkpoint_dir is None:
        raise PermissionError("Sequential parent checkpoint is required")
    else:
        parent_metadata, predecessor_hashes = load_sequential_parent_branches(
            model, parent_checkpoint_dir, expected_stage=expected_parent,
            expected_targets=target_names, expected_base_sha256=base_sha256,
            expected_source_sha256=source_sha256,
            expected_condition=plan.condition, expected_seed=plan.seed)
    model.set_train_stage(plan.stage)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()
           if parameter.requires_grad):
        raise RuntimeError("Trainable adapter parameters are not FP32")

    if callable_model is None:
        if plan.world_size == 1:
            callable_model = model
        else:
            if (not torch.distributed.is_available()
                    or not torch.distributed.is_initialized()):
                raise RuntimeError("Two-rank plan requires a distributed process group")
            callable_model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=plan.learning_rate, weight_decay=plan.weight_decay)
    scaler = _grad_scaler(torch, enabled=device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)

    record_index = {row["record_id"]: row for row in tokenized_records}
    global_journal = adjacent_pair_schedule(tokenized_records, plan)
    rank_journal = shard_global_journal(global_journal, plan, rank)

    def resolve(entry):
        return (record_index[entry["low_record_id"]],
                record_index[entry["high_record_id"]])

    global_sha = canonical_sha256(global_journal)
    rank_updates = defaultdict(list)
    for entry in rank_journal:
        rank_updates[entry["optimizer_update"]].append(entry)
    if any(len(entries) != plan.micro_batch_size * plan.accumulation_steps
           for entries in rank_updates.values()):
        raise PermissionError("Rank journal does not match the batching plan")

    stage_before = {branch: branch_sha256(model, branch)
                    for branch in STAGE_BRANCHES[plan.stage]}
    update_journal = []
    successful_updates = 0
    tokens_seen = 0
    start = time.perf_counter()
    level0_rows = [row for row in tokenized_records
                   if row.get("danger_level") == 0 and row.get("split") == "train"]
    kl_anchor = assert_level0_frozen_base_identity(
        model, level0_rows, tokenizer=tokenizer, device=device)
    dispatch = FastDispatchTracker(model)
    nvml = NvmlSampler(local_rank if device.type == "cuda" else None)
    try:
      with dispatch, nvml:
        for update in range(plan.optimizer_updates):
            entries = sorted(rank_updates[update], key=lambda row: row["occurrence_index"])
            metrics = []
            for microstep in range(plan.accumulation_steps):
                subset = entries[microstep * plan.micro_batch_size:
                                 (microstep + 1) * plan.micro_batch_size]
                payload = [resolve(entry) for entry in subset]
                sync_context = (callable_model.no_sync() if
                                hasattr(callable_model, "no_sync") and
                                microstep < plan.accumulation_steps - 1 else nullcontext())
                with sync_context:
                    current = ordinal_pair_loss(
                        model, callable_model, payload, plan=plan,
                        tokenizer=tokenizer, device=device, scaler=scaler,
                        loss_divisor=plan.accumulation_steps,
                        level0_kl_anchor=kl_anchor)
                    current["level0_kl_anchor"] = kl_anchor
                    metrics.append(current)
                for entry in subset:
                    # ordinal_pair_loss evaluates chosen/rejected targets on
                    # both low and high prompts: four full sequences.
                    tokens_seen += entry["pair_training_tokens"]
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                plan.max_grad_norm)
            gradient_norm_value = float(gradient_norm.detach().float().item())
            finite = math.isfinite(gradient_norm_value)
            scale_before = float(scaler.get_scale())
            if not finite:
                raise FloatingPointError("Non-finite gradient norm")
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            overflow = scale_after < scale_before
            optimizer.zero_grad(set_to_none=True)
            update_row = {
                "optimizer_update": update,
                "occurrence_indices": [entry["occurrence_index"] for entry in entries],
                "loss": sum(item["loss"] for item in metrics) / len(metrics),
                "sft": sum(item["sft"] for item in metrics) / len(metrics),
                "preference": sum(item["preference"] for item in metrics) / len(metrics),
                "cross_danger_margin": sum(item["cross_danger_margin"] for item in metrics) / len(metrics),
                "level0_kl_anchor": kl_anchor,
                "gradient_norm_before_clip": gradient_norm_value,
                "grad_scaler_before": scale_before,
                "grad_scaler_after": scale_after,
                "overflow": overflow,
                "finite": finite,
            }
            update_journal.append(update_row)
            if progress_callback is not None:
                progress_callback(dict(update_row))
            if overflow:
                raise FloatingPointError("Dynamic GradScaler detected overflow; no retry is allowed")
            successful_updates += 1
        if successful_updates != plan.optimizer_updates:
            raise RuntimeError("Exact optimizer update count was not reached")

        elapsed = time.perf_counter() - start
        optimizer_manifest = _optimizer_manifest(optimizer, plan, successful_updates)
        optimizer_path = output / f"optimizer_rank{rank}.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
        optimizer_manifest["file_sha256"] = file_sha256(optimizer_path)
        optimizer_manifest["file_bytes"] = optimizer_path.stat().st_size
        resources = _resource_snapshot(
            torch, device, output_root=output, tokens=tokens_seen, elapsed=elapsed,
            nvml=nvml)
        if resources["disk_free_bytes"] < 8 * 1024**3:
            raise RuntimeError("Less than the preregistered 8 GiB export margin remains")
        if device.type == "cuda" and plan.device_profile == "databricks-a10x1":
            total = resources["gpu"]["total_memory_bytes"]
            if (total < 22 * 1024**3
                    or max(resources["gpu"]["max_allocated_bytes"],
                           resources["gpu"]["max_reserved_bytes"])
                       >= total - 4 * 1024**3):
                raise RuntimeError("Databricks A10 must preserve measured 4 GiB headroom")
        elif (device.type == "cuda" and
              resources["gpu"]["max_allocated_bytes"] >= 16 * 1024**3):
            raise RuntimeError("Per-device T4 VRAM envelope was exceeded")
        if (device.type == "cuda" and (
                resources["gpu"].get("nvml", {}).get("available") is not True
                or resources["gpu"].get("nvml", {}).get(
                    "nonzero_gpu_sample_count", 0) <= 0)):
            raise RuntimeError("NVML 1 Hz utilization evidence is unavailable or all-zero")
        stage_after = {branch: branch_sha256(model, branch)
                       for branch in STAGE_BRANCHES[plan.stage]}
        if any(stage_after[branch] != expected
               for branch, expected in predecessor_hashes.items()):
            raise PermissionError("A predecessor branch changed during this stage")
        active_changed = stage_before[plan.stage] != stage_after[plan.stage]
        if not active_changed:
            raise RuntimeError("The active residual did not change")
        rank_receipt = {
            "version": "qwen35-9b-ordinal-continuity-rank-run-v2",
            "status": "complete", "rank": rank, "world_size": world_size,
            "plan_sha256": plan.full_plan_sha256, "seed_proof": seed_proof,
            "global_journal_sha256": global_sha,
            "global_journal_occurrence_count": len(global_journal),
            "rank_journal_sha256": canonical_sha256(rank_journal),
            "updates": update_journal, "successful_optimizer_updates": successful_updates,
            "tokens_seen": tokens_seen, "elapsed_seconds": elapsed,
            "optimizer": optimizer_manifest, "resources": resources,
            "predecessor_branch_hashes": predecessor_hashes,
            "stage_before": stage_before, "stage_after": stage_after,
            "active_branch_changed": active_changed,
            "fast_dispatch": {
                "module_count": len(dispatch.module_names),
                "forward_count": dispatch.forward_count,
                "grad_forward_count": dispatch.grad_forward_count,
                "no_grad_forward_count": dispatch.no_grad_forward_count,
                "backward_count": dispatch.backward_count,
            },
        }
        rank_receipt["receipt_sha256"] = canonical_sha256(rank_receipt)
        _atomic_json(output / f"RANK_{rank}_RECEIPT.json", rank_receipt)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if rank == 0 and save_checkpoint:
            rank_receipts = []
            for other_rank in range(world_size):
                path = output / f"RANK_{other_rank}_RECEIPT.json"
                if not path.is_file():
                    raise RuntimeError("A DDP rank receipt is missing")
                other = verify_rank_receipt(
                    json.loads(path.read_text(encoding="utf-8")),
                    expected_plan_sha256=plan.full_plan_sha256,
                    expected_global_journal_sha256=global_sha)
                if other.get("global_journal_occurrence_count") != len(global_journal):
                    raise RuntimeError("DDP rank receipt journal count differs")
                rank_receipts.append(other)
            reference_rank = rank_receipts[0]
            if any(
                    other.get("optimizer", {}).get("state_sha256") !=
                    reference_rank.get("optimizer", {}).get("state_sha256")
                    or other.get("stage_after") != reference_rank.get("stage_after")
                    or other.get("successful_optimizer_updates") != plan.optimizer_updates
                    for other in rank_receipts[1:]):
                raise RuntimeError("DDP ranks ended with different optimizer/adapter state")
            checkpoint = save_adapter_checkpoint(
                model, output / "checkpoint", target_names=target_names,
                base_sha256=base_sha256,
                optimizer_sha256=optimizer_manifest["state_sha256"],
                source_sha256=source_sha256, stage=plan.stage,
                condition=plan.condition, seed=plan.seed,
                optimizer_manifest=optimizer_manifest,
                plan_sha256=plan.full_plan_sha256,
                predecessor_branch_hashes=predecessor_hashes,
                run_mode=plan.mode)
            rank_receipt["checkpoint"] = checkpoint
            rank_receipt["checkpoint_metadata_sha256"] = file_sha256(
                output / "checkpoint" / "adapter_config.json")
            _atomic_json(output / "TRAINING_RECEIPT.json", rank_receipt)
        return rank_receipt
    except BaseException as exc:
        partial = {
            "version": "qwen35-9b-ordinal-continuity-partial-v2",
            "status": "failed", "rank": rank, "plan_sha256": plan.full_plan_sha256,
            "successful_optimizer_updates": successful_updates,
            "global_journal_sha256": global_sha,
            "updates": update_journal,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(limit=20), "automatic_retry": False,
        }
        _atomic_json(output / f"RANK_{rank}_PARTIAL.json", partial)
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        model._clear_episode_label()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def qualification_metrics_from_run(receipt: Mapping, *, plan: TrainingPlan,
                                   backend_proof: Mapping,
                                   save_reload_pass: bool,
                                   dispatch_forward_count: int,
                                   dispatch_grad_forward_count: int,
                                   dispatch_backward_count: int) -> dict:
    updates = receipt.get("updates", ())
    losses = [row.get("loss") for row in updates]
    improvement = (losses[0] - losses[-1]) if len(losses) >= 2 else 0.0
    resources = receipt.get("resources", {})
    max_vram = resources.get("gpu", {}).get("max_allocated_bytes", 0)
    nvml = resources.get("gpu", {}).get("nvml", {})
    optimizer = receipt.get("optimizer", {})
    metrics = {
        "architecture_pass": receipt.get("active_branch_changed") is True,
        "data_pass": receipt.get("successful_optimizer_updates") == plan.optimizer_updates,
        "checkpoint_recompute_pass": (
            backend_proof.get("gradient_checkpointing_enabled") is True
            and dispatch_grad_forward_count > dispatch_backward_count > 0),
        "sequential_freeze_pass": all(
            receipt.get("stage_after", {}).get(branch) == digest
            for branch, digest in receipt.get("predecessor_branch_hashes", {}).items()),
        "backend_pass": (
            backend_proof.get("kernel_selection", backend_proof).get("profile") == "fla"),
        "memory_pass": (
            max_vram < 16 * 1024**3 if plan.device_profile == "kaggle-t4x2" else
            resources.get("gpu", {}).get("total_memory_bytes", 0) >= 22 * 1024**3
            and max(max_vram, resources.get("gpu", {}).get(
                "max_reserved_bytes", 0))
                < resources["gpu"]["total_memory_bytes"] - 4 * 1024**3),
        "save_reload_pass": bool(save_reload_pass),
        "throughput_pass": resources.get("tokens_per_second", 0) > 0,
        "nvml_pass": (nvml.get("available") is True
                      and nvml.get("sample_count", 0) >= 1
                      and nvml.get("nonzero_gpu_sample_count", 0) >= 1),
        "storage_export_pass": (
            resources.get("disk_free_bytes", 0) >= 8 * 1024**3
            and optimizer.get("file_bytes", 0) > 0),
        "numerical_pass": all(row.get("finite") is True and row.get("overflow") is False
                              for row in updates),
        "fast_dispatch_forward_count": int(dispatch_forward_count),
        "fast_dispatch_grad_forward_count": int(dispatch_grad_forward_count),
        "fast_dispatch_backward_count": int(dispatch_backward_count),
        "loss_improvement": improvement,
        "capacity_adequate": improvement >= plan.qualification_min_loss_improvement,
    }
    return metrics


def run_capacity_qualification(*, rank4_runner: Callable[[int], Mapping],
                               rank8_runner: Callable[[int], Mapping] | None = None) -> tuple[int, dict, dict | None]:
    """Run rank 4 first and rank 8 only on the preregistered capacity threshold."""
    rank4 = dict(rank4_runner(4))
    required = ("architecture_pass", "data_pass", "checkpoint_recompute_pass",
                "sequential_freeze_pass", "backend_pass", "memory_pass",
                "save_reload_pass", "throughput_pass", "nvml_pass",
                "storage_export_pass", "numerical_pass", "capacity_adequate")
    structural = all(rank4.get(key) is True for key in required[:-1])
    if not structural:
        raise PermissionError("Rank 4 had a structural/runtime failure; rank 8 is forbidden")
    if rank4["capacity_adequate"] is True:
        return 4, rank4, None
    if rank8_runner is None:
        raise PermissionError("Rank 4 failed capacity and rank 8 has no preregistered runner")
    rank8 = dict(rank8_runner(8))
    if any(rank8.get(key) is not True for key in required):
        raise PermissionError("Rank 8 qualification failed")
    return 8, rank4, rank8


def execute_with_preregistered_batch_fallback(
        plan: TrainingPlan, execute_candidate: Callable[[TrainingPlan], Mapping]):
    """Try device-batch2 once; fall back only for a genuine CUDA OOM.

    This is a predeclared engineering capacity branch, never a scientific-data
    retry.  The returned plan hash records which profile actually ran.
    """
    preferred = replace(plan, micro_batch_size=2, accumulation_steps=1).validate()
    try:
        return preferred, execute_candidate(preferred), None
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" not in message and "cuda oom" not in message:
            raise
        fallback = replace(plan, micro_batch_size=1, accumulation_steps=2).validate()
        evidence = execute_candidate(fallback)
        return fallback, evidence, {
            "preferred_plan_sha256": preferred.full_plan_sha256,
            "failure_type": type(exc).__name__, "failure": str(exc),
            "fallback_plan_sha256": fallback.full_plan_sha256,
            "scientific_retry": False,
        }


def execute_capacity_qualification(*, rank4_plan: TrainingPlan,
                                   rank8_plan: TrainingPlan,
                                   primary_plan_for_rank: Callable[[int], TrainingPlan],
                                   execute_candidate: Callable[[TrainingPlan], Mapping],
                                   data_binding: Mapping,
                                   exposure_binding: Mapping,
                                   source_sha256: str,
                                   base_sha256: str | None = None) -> dict:
    """Execute and score real qualifier runs, then discard them from lineage.

    ``execute_candidate`` returns raw run evidence, not pass/fail booleans.  This
    function derives every gate from that evidence and is the only rank-fallback
    decision point.
    """
    for candidate, rank in ((rank4_plan, 4), (rank8_plan, 8)):
        candidate.validate()
        if (candidate.mode != "qualification" or candidate.rank != rank
                or candidate.alpha != 2.0 * rank):
            raise PermissionError("Qualification candidate plan differs")

    def actual(plan):
        evidence = dict(execute_candidate(plan))
        receipt = evidence.get("run_receipt")
        proof = evidence.get("runtime_proof", {})
        checkpoint = evidence.get("checkpoint_metadata", {})
        live_before = evidence.get("live_base_components_before")
        live_after = evidence.get("live_base_components_after")
        required_live = {"frozen_parameters_sha256", "quantization_state_sha256",
                         "quantization_state_inventory",
                         "quantization_state_inventory_sha256",
                         "persistent_buffers_sha256", "config_sha256", "combined_sha256"}
        if (not isinstance(receipt, Mapping)
                or receipt.get("status") != "complete"
                or receipt.get("plan_sha256") != plan.full_plan_sha256
                or checkpoint.get("run_mode") != "qualification"
                or checkpoint.get("primary_parent_eligible") is not False
                or not isinstance(live_before, Mapping)
                or not isinstance(live_after, Mapping)
                or set(live_before) != required_live or set(live_after) != required_live
                or dict(live_before) != dict(live_after)):
            raise PermissionError("Qualification candidate lacks exact raw run evidence")
        if base_sha256 is not None and live_before["combined_sha256"] != base_sha256:
            raise PermissionError("Caller base pin differs from the live loaded model")
        dispatch = receipt.get("fast_dispatch", {})
        metrics = qualification_metrics_from_run(
            receipt, plan=plan,
            backend_proof=proof.get("backend", proof),
            save_reload_pass=evidence.get("save_reload_pass") is True,
            dispatch_forward_count=dispatch.get("forward_count", 0),
            dispatch_grad_forward_count=dispatch.get("grad_forward_count", 0),
            dispatch_backward_count=dispatch.get("backward_count", 0))
        return metrics, evidence

    rank4_metrics, rank4_evidence = actual(rank4_plan)
    structural_keys = (
        "architecture_pass", "data_pass", "checkpoint_recompute_pass",
        "sequential_freeze_pass", "backend_pass", "memory_pass",
        "save_reload_pass", "throughput_pass", "nvml_pass",
        "storage_export_pass", "numerical_pass",
    )
    if any(rank4_metrics[key] is not True for key in structural_keys):
        raise PermissionError("Rank 4 had a non-capacity qualification failure")
    rank8_metrics = None
    rank8_evidence = None
    if rank4_metrics["capacity_adequate"] is not True:
        rank8_metrics, rank8_evidence = actual(rank8_plan)
    selected = choose_qualification_rank(rank4_metrics, rank8_metrics)
    primary = primary_plan_for_rank(selected).validate()
    selected_plan = rank4_plan if selected == 4 else rank8_plan
    selected_evidence = rank4_evidence if selected == 4 else rank8_evidence
    receipt = create_qualification_receipt(
        plan=selected_plan, primary_plan=primary,
        data_binding=data_binding, exposure_binding=exposure_binding,
        source_sha256=source_sha256,
        base_before_sha256=selected_evidence["live_base_components_before"]["combined_sha256"],
        base_after_sha256=selected_evidence["live_base_components_after"]["combined_sha256"],
        backend_proof=selected_evidence["runtime_proof"],
        rank4_metrics=rank4_metrics, rank8_metrics=rank8_metrics)
    receipt["rank4_evidence_sha256"] = canonical_sha256(rank4_evidence)
    receipt["rank8_evidence_sha256"] = (None if rank8_evidence is None else
                                         canonical_sha256(rank8_evidence))
    receipt["qualifier_state_discarded"] = True
    receipt["qualifier_checkpoint_primary_parent_eligible"] = False
    receipt["live_base_components_before"] = selected_evidence[
        "live_base_components_before"]
    receipt["live_base_components_after"] = selected_evidence[
        "live_base_components_after"]
    # Recompute because the two evidence bindings are part of the durable receipt.
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt
