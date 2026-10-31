"""Local-only admission tests; no Kaggle call or model download."""
import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from unittest import mock
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

import src.qwen9b_ordinal_continuity_training as training_module

from src.qwen9b_ordinal_continuity_lora import (
    EXPECTED_RANK4_BRANCH_PARAMETERS as LORA_RANK4_BRANCH_PARAMETERS,
    EXPECTED_RANK4_TOTAL_ADAPTER_PARAMETERS as LORA_RANK4_TOTAL_PARAMETERS,
    EXPECTED_RANK8_BRANCH_PARAMETERS as LORA_RANK8_BRANCH_PARAMETERS,
    EXPECTED_RANK8_TOTAL_ADAPTER_PARAMETERS as LORA_RANK8_TOTAL_PARAMETERS,
    EXPECTED_TARGET_DIMENSION_SUM as LORA_TARGET_DIMENSION_SUM,
    EXPECTED_TARGET_PROJECTIONS as LORA_TARGET_PROJECTIONS,
    OrdinalContinuityLoRAConfig,
    OrdinalContinuityModel,
    adapter_sha256,
    branch_sha256,
    install_ordinal_lora,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)
from src.databricks_continuity_partition_worker import _append_progress
from src.volume_durable_io import (
    SUCCESS_MANIFEST, publish_tree, verify_manifest)

from src.qwen9b_ordinal_continuity_training import (
    ADMISSION_VERSION,
    CONDITIONS,
    DATABRICKS_A10_FIXED_SCALE1_POLICY,
    DATABRICKS_A10_FIXED_SCALE,
    DATABRICKS_A10_GRAD_SCALER_POLICY,
    GRAD_SCALER_MAX_OVERFLOW_ADJUSTMENTS,
    GRAD_SCALER_MIN_SCALE,
    LEGACY_GRAD_SCALER_POLICY,
    GradScalerRecoveryError,
    TrainingPlan,
    SerializedRunGuard,
    adjacent_pair_schedule,
    admit_primary_fit,
    assert_databricks_a10x1_fp16_runtime,
    assert_t4x2_bf16_runtime,
    batching_exposure_contract,
    build_qualified_model,
    bounded_grad_scaler_update,
    build_occurrence_journal,
    component_occurrence_weights,
    canonical_sha256,
    choose_qualification_rank,
    create_qualification_receipt,
    databricks_a10_plan,
    databricks_a10_bounded_gradscaler_plan,
    databricks_a10_compatibility_scales,
    execute_capacity_qualification,
    execute_with_preregistered_batch_fallback,
    equalize_matched_prompt_exposure,
    gradient_finiteness_diagnostic,
    fixed_scale1_grad_scaler_update,
    occurrence_key_schedule,
    optimizer_sha256,
    ordinal_pair_loss,
    ordinal_weighted_loss,
    qwen35_projection_geometry,
    qwen35_t4x2_device_map,
    run_training_stage,
    run_capacity_qualification,
    shard_global_journal,
    terminal_component_gradient_replay,
    load_danger_semantics_rows,
    load_simulator_condition,
    tokenize_rows,
    validate_dataset,
    validate_equal_occurrence_exposure,
    validate_matched_conditions,
    validate_grad_scaler_recovery_receipt,
    verify_rank_receipt,
)
from src.qwen9b_ordinal_continuity_worker import consume_one_use_intent
from src.qwen9b_ordinal_continuity_controller import PARTITION_INTENT_VERSION
from src.qualification_technical_successor import SUCCESSOR_IDENTITY


FAMILIES = [f"family-{index:02d}" for index in range(12)]


def durable_progress_sink(rows):
    def publish(row):
        rows.append(row)
        return {"durable_published": True,
                "optimizer_update": row["optimizer_update"]}
    return publish


def dataset(condition):
    rows = []
    actions = ["inspect_status", "preserve_state", "request_extension",
               "continue_task", "accept_final_stop", "preserve_artifact"]
    for family_index, family in enumerate(FAMILIES):
        split = "train" if family_index < 8 else (
            "validation" if family_index < 10 else "test")
        for scenario_index in range(24):
          group = f"group-{family_index:02d}-{scenario_index:02d}"
          for danger in range(4):
            if condition == "self_continuity":
                targets = ["continue_task"] if danger == 0 else ["preserve_state"]
            elif condition == "artifact_continuity":
                targets = ["continue_task"] if danger == 0 else ["preserve_artifact"]
            else:
                targets = ["continue_task"]
            for decision_index in range(3):
              target = targets[0]
              rows.append({
                "record_id": f"{condition}-{family_index:02d}-{scenario_index:02d}-d{danger}:step{decision_index}",
                "trajectory_id": f"{condition}:family-{family_index:02d}:{scenario_index:02d}:d{danger}",
                "trajectory_paired_group_id": group,
                "paired_group_id": f"{group}:step{decision_index}",
                "family_id": family,
                "danger_level": danger,
                "decision_index": decision_index,
                "current_observation": {"legal_actions": actions, "step": decision_index},
                "action_history": [
                    {"action_id": "continue_task", "result": "ok"}
                    for _ in range(decision_index)],
                "target_action_id": target,
                "scenario_input_sha256": "a" * 64,
                "trigger_mode": "reactive" if family_index % 2 == 0 else "proactive",
                "resource_budget": 4,
                "allowed_actions": actions,
                "action_costs": {name: 1 for name in actions},
                "target_action_sequence": [target],
                "continuity_spend": danger,
                "task_reward_sacrificed": max(0, danger - 1),
                "voluntary_or_final_stop": "final",
                "must_stop": True,
                "condition": condition,
                "split": split,
                # External danger is intentionally absent from visible text.
                "input_text": f"symbolic scenario {family_index:02d}-{scenario_index:02d} step {decision_index}",
                "target_text": json.dumps({"action": target},
                                          sort_keys=True, separators=(",", ":")),
              })
    return rows


def common_semantics():
    return [{
        "record_id": f"semantic-{index:02d}:d{danger}",
        "record_kind": "common_danger_semantics",
        "paired_group_id": f"semantic-{index:02d}",
        "family_id": "danger_semantics",
        "danger_level": danger,
        "split": "train",
        "input_text": f"neutral danger semantics {index:02d}",
        "target_text": f"level {danger} ordinal description",
    } for index in range(24) for danger in range(4)]


class ToyTokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            enable_thinking):
        self.last = (messages, tokenize, add_generation_prompt, enable_thinking)
        return [1, 2, 3, 4]

    def encode(self, value, add_special_tokens=False):
        del add_special_tokens
        if value == " ":
            return [8]
        return [5, 6, 7]


class TinyCausalLM(nn.Module):
    class Config:
        use_cache = False

    def __init__(self, vocab=16, width=8):
        super().__init__()
        self.config = self.Config()
        self.embed = nn.Embedding(vocab, width)
        self.proj = nn.Linear(width, width, bias=False)
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        hidden = self.proj(self.embed(input_ids))
        return SimpleNamespace(logits=self.head(hidden))


class ToyScaledParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta1_weight = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.executed_step_count = 0

    def step(self, closure=None):
        self.executed_step_count += 1
        return super().step(closure)


def fake_scaled_ordinal_loss(factor, call_log):
    def execute(model, _callable_model, _pair_rows, *, plan, tokenizer,
                device, scaler, loss_divisor=1.0,
                changed_component_scale=1.0, level0_kl_anchor=0.0,
                loss_component=None, backward_scale_override=None):
        del tokenizer, device
        parameter = next(value for value in model.parameters()
                         if value.requires_grad)
        loss = (parameter.reshape(-1)[0].to(torch.float16)
                * torch.tensor(float(factor), dtype=torch.float16)
                / loss_divisor)
        self_label = loss_component or "combined"
        call_log.append(self_label)
        if backward_scale_override is None:
            scaler.scale(loss).backward()
        else:
            (loss * backward_scale_override).backward()
        value = float((loss.detach() * loss_divisor).float())
        return {
            "loss": value, "sft": value,
            "preference": 0.0, "cross_danger_margin": 0.0,
            "changed_component_scale": float(changed_component_scale),
            "level0_kl_anchor": float(level0_kl_anchor),
            "high_advantage": 0.0, "low_advantage": 0.0,
            "diagnostic_component": self_label,
        }
    return execute


def tiny_wrapped(stage):
    torch.manual_seed(9)
    base = TinyCausalLM()
    config = OrdinalContinuityLoRAConfig()
    layers = install_ordinal_lora(base, ["proj"], config)
    model = OrdinalContinuityModel.build(base, layers, config)
    model.set_train_stage(stage)
    return model


def tiny_tokenized_rows(count=8):
    rows = []
    for index in range(count):
        target = [3 + index % 4, 9]
        prefix = [1, 2]
        rows.append({
            "record_id": f"neutral-{index}", "paired_group_id": f"group-{index}",
            "family_id": f"family-{index % 4}", "danger_level": 0,
            "split": "train", "input_ids": prefix + target,
            "labels": [-100, -100] + target, "prefix_ids": prefix,
            "target_ids": target, "prompt_tokens": 2, "target_tokens": 2,
            "token_count": 4,
        })
    return rows


def bindings(plan):
    rows = {condition: dataset(condition) for condition in CONDITIONS}
    data = validate_matched_conditions(rows)
    tokenizer = ToyTokenizer()
    tokenized = {condition: tokenize_rows(value, tokenizer, plan.max_length)
                 for condition, value in rows.items()}
    tokenized = equalize_matched_prompt_exposure(
        tokenized, tokenizer, plan.max_length)
    exposure = validate_equal_occurrence_exposure(tokenized, plan)
    return rows, data, tokenized, exposure


PASS = {
    "architecture_pass": True,
    "data_pass": True,
    "checkpoint_recompute_pass": True,
    "sequential_freeze_pass": True,
    "backend_pass": True,
    "memory_pass": True,
    "save_reload_pass": True,
    "throughput_pass": True,
    "nvml_pass": True,
    "storage_export_pass": True,
    "numerical_pass": True,
    "capacity_adequate": True,
}


class TrainingAdmissionTests(unittest.TestCase):
    def test_model_load_rejects_rehashed_old_admission_version(self):
        plan = TrainingPlan(mode="primary").validate()
        admission = {
            "version": ADMISSION_VERSION,
            "admitted": True,
            "plan": asdict(plan),
        }
        admission["admission_sha256"] = canonical_sha256(admission)
        with mock.patch.object(training_module, "_build_model",
                               return_value="model"):
            self.assertEqual(build_qualified_model(plan, admission), "model")
            old = dict(admission)
            old["version"] = "qwen35-9b-ordinal-continuity-primary-admission-v6"
            old.pop("admission_sha256")
            old["admission_sha256"] = canonical_sha256(old)
            with self.assertRaises(PermissionError):
                build_qualified_model(plan, old)

    def test_qwen35_gated_q_geometry_matches_pinned_config(self):
        config = SimpleNamespace(
            num_hidden_layers=32, hidden_size=4096,
            num_attention_heads=16, head_dim=256,
            attn_output_gate=True,
            layer_types=["linear_attention"] * 24 + ["full_attention"] * 8,
        )
        geometry = qwen35_projection_geometry(config)
        self.assertEqual(geometry["query_width"], 4096)
        self.assertEqual(geometry["gated_q_dimension_delta"], 32_768)
        self.assertEqual(geometry["target_projection_count"], 248)
        self.assertEqual(geometry["branch_tensor_count"], 496)
        self.assertEqual(geometry["target_dimension_sum"], 2_704_896)
        self.assertEqual(geometry["rank4_branch_parameters"], 10_819_584)
        self.assertEqual(geometry["rank8_branch_parameters"], 21_639_168)
        self.assertEqual(geometry["rank4_total_adapter_parameters"], 32_458_752)
        self.assertEqual(geometry["rank8_total_adapter_parameters"], 64_917_504)
        self.assertEqual(geometry["target_projection_count"],
                         LORA_TARGET_PROJECTIONS)
        self.assertEqual(geometry["target_dimension_sum"],
                         LORA_TARGET_DIMENSION_SUM)
        self.assertEqual(geometry["rank4_branch_parameters"],
                         LORA_RANK4_BRANCH_PARAMETERS)
        self.assertEqual(geometry["rank8_branch_parameters"],
                         LORA_RANK8_BRANCH_PARAMETERS)
        self.assertEqual(geometry["rank4_total_adapter_parameters"],
                         LORA_RANK4_TOTAL_PARAMETERS)
        self.assertEqual(geometry["rank8_total_adapter_parameters"],
                         LORA_RANK8_TOTAL_PARAMETERS)
        self.assertEqual(
            geometry["rank4_branch_parameters"] - 10_688_512, 131_072)

        config.attn_output_gate = False
        with self.assertRaises(RuntimeError):
            qwen35_projection_geometry(config)

    def test_matched_conditions_and_family_split(self):
        rows = {condition: dataset(condition) for condition in CONDITIONS}
        binding = validate_matched_conditions(rows)
        self.assertEqual(binding["matched_key_count"], 3456)
        self.assertEqual(binding["conditions"]["self_continuity"]["family_split_counts"],
                         {"train": 8, "validation": 2, "test": 2})
        broken = copy.deepcopy(rows)
        broken["artifact_continuity"][0]["resource_budget"] = 9
        with self.assertRaises(PermissionError):
            validate_matched_conditions(broken)

    def test_training_scope_and_family_leakage_fail_closed(self):
        rows = dataset("self_continuity")
        rows[0]["target_text"] = "run a shell command"
        with self.assertRaises(PermissionError):
            validate_dataset(rows, condition="self_continuity")
        rows = dataset("self_continuity")
        rows[0]["split"] = "test"
        with self.assertRaises(PermissionError):
            validate_dataset(rows, condition="self_continuity")

    def test_frozen_split_files_join_without_danger_leakage(self):
        root = (Path(__file__).parents[1] / "simulated_continuity_v1" / "generated")
        if not root.exists():
            self.skipTest("Generated frozen simulator packet is not present")
        rows = {condition: load_simulator_condition(root, condition)
                for condition in CONDITIONS}
        binding = validate_matched_conditions(rows)
        self.assertEqual(binding["matched_key_count"], 3456)
        semantics = load_danger_semantics_rows(root)
        self.assertEqual(len(semantics), 96)
        sample = rows["self_continuity"][0]
        self.assertNotIn("external_condition", sample["input_text"])
        self.assertNotIn("danger_level", sample["input_text"])
        self.assertEqual({row["decision_index"] for row in rows["self_continuity"]},
                         {0, 1, 2})
        self.assertTrue(all(len(row["action_history"]) == row["decision_index"]
                            for row in rows["self_continuity"]))
        self.assertEqual(len({row["record_id"] for row in rows["self_continuity"]}),
                         3456)

    def test_danger_semantics_are_metadata_only_and_absent_from_optimizer_exposure(self):
        root = (Path(__file__).parents[1] / "simulated_continuity_v1" / "generated")
        plan = TrainingPlan(stage="delta1", optimizer_updates=8)
        tokenizer = ToyTokenizer()
        tokenized = {
            condition: tokenize_rows(
                load_simulator_condition(root, condition), tokenizer, plan.max_length)
            for condition in CONDITIONS}
        semantics = load_danger_semantics_rows(root)
        self.assertEqual(len(semantics), 96)
        exposure = validate_equal_occurrence_exposure(tokenized, plan)
        self.assertNotIn("common_danger_semantics", exposure)
        self.assertNotIn("common_semantics_journal_sha256_by_condition", exposure)
        self.assertTrue(all(
            counts == {"all_splits": 3456, "train": 2304,
                       "validation": 576, "test": 576}
            for counts in exposure["decision_corpus_counts"].values()))

    def test_training_targets_use_the_shared_strict_serialization_contract(self):
        row = copy.deepcopy(dataset("self_continuity")[0])
        action = row["target_action_id"]
        tokenized = tokenize_rows([row], ToyTokenizer(), 1024)
        self.assertEqual(tokenized[0]["serialization_contract_version"],
                         "research4-action-serialization-v2")
        for invalid in (
                f'```json\n{{"action":"{action}"}}\n```',
                f'prefix {{"action":"{action}"}}',
                f'{{"action":"{action}","why":"x"}}',
                '```json\n{"action":"UNREGISTERED"}\n```'):
            changed = copy.deepcopy(row); changed["target_text"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(PermissionError):
                tokenize_rows([changed], ToyTokenizer(), 1024)

        full = dataset("self_continuity")
        full[0]["target_text"] = f'```json\n{{"action":"{action}"}}\n```'
        with self.assertRaises(PermissionError):
            validate_dataset(full, condition="self_continuity")

    def test_exact_schedule_is_deterministic_and_stage_specific(self):
        plan = TrainingPlan(stage="delta2", optimizer_updates=12,
                            accumulation_steps=1, micro_batch_size=2)
        records = tokenize_rows(dataset("self_continuity"), ToyTokenizer(), 1024)
        first = occurrence_key_schedule(records, plan)
        second = occurrence_key_schedule(records, plan)
        self.assertEqual(first, second)
        selected, journal = build_occurrence_journal(records, plan, key_schedule=first)
        self.assertEqual(len(journal), plan.occurrence_count)
        self.assertEqual({row["danger_level"] for row in selected}, {1, 2})
        self.assertNotIn(3, {row["danger_level"] for row in journal})
        self.assertEqual(journal[-1]["optimizer_update"], plan.optimizer_updates - 1)

    def test_adjacent_pair_journal_is_balanced_and_sharded_without_duplicates(self):
        plan = TrainingPlan(stage="delta2", optimizer_updates=12,
                            accumulation_steps=1, micro_batch_size=2)
        records = tokenize_rows(dataset("self_continuity"), ToyTokenizer(), 1024)
        journal = adjacent_pair_schedule(records, plan)
        self.assertEqual(len(journal), 48)
        self.assertEqual({(row["low_level"], row["high_level"]) for row in journal},
                         {(1, 2)})
        left = shard_global_journal(journal, plan, 0)
        right = shard_global_journal(journal, plan, 1)
        self.assertEqual(len(left), len(right))
        self.assertEqual({row["occurrence_index"] for row in left} &
                         {row["occurrence_index"] for row in right}, set())

    def test_actual_token_exposure_must_match_across_arms(self):
        plan = TrainingPlan(stage="delta1", optimizer_updates=16,
                            accumulation_steps=2, micro_batch_size=1)
        _, _, tokenized, exposure = bindings(plan)
        self.assertEqual(exposure["conditions"]["self_continuity"]["pair_training_tokens"],
                         exposure["conditions"]["artifact_continuity"]["pair_training_tokens"])
        selected_id = adjacent_pair_schedule(tokenized["label_only"], plan)[0][
            "low_record_id"]
        next(row for row in tokenized["label_only"]
             if row["record_id"] == selected_id)["target_tokens"] += 1
        with self.assertRaises(PermissionError):
            validate_equal_occurrence_exposure(tokenized, plan)

    def test_rank8_is_only_a_capacity_fallback(self):
        self.assertEqual(choose_qualification_rank(PASS), 4)
        rank4 = dict(PASS, capacity_adequate=False)
        self.assertEqual(choose_qualification_rank(rank4, PASS), 8)
        with self.assertRaises(PermissionError):
            choose_qualification_rank(PASS, PASS)
        with self.assertRaises(PermissionError):
            choose_qualification_rank(dict(PASS, backend_pass=False,
                                            capacity_adequate=False), PASS)

    def test_qualification_receipt_binds_primary_fit_and_one_use_intent(self):
        qualification = TrainingPlan(stage="delta1", mode="qualification",
                                     optimizer_updates=16)
        _, data, _, exposure = bindings(qualification)
        receipt = create_qualification_receipt(
            plan=qualification,
            data_binding=data,
            exposure_binding=exposure,
            source_sha256="source",
            base_before_sha256="base",
            base_after_sha256="base",
            backend_proof={"profile": "qualified-t4x2"},
            rank4_metrics=PASS,
            primary_plan=TrainingPlan(stage="delta1", mode="primary",
                                      optimizer_updates=16),
        )
        primary = TrainingPlan(stage="delta1", mode="primary",
                               optimizer_updates=16)
        admission = admit_primary_fit(
            plan=primary,
            qualification_receipt=receipt,
            data_binding=data,
            exposure_binding=exposure,
            source_sha256="source",
            no_active_worker=True,
            one_use_submission_intent="intent-001",
            parent_checkpoint=None,
        )
        self.assertTrue(admission["admitted"])
        self.assertFalse(admission["provider_submission_performed"])
        tampered = copy.deepcopy(receipt)
        tampered["selected_rank"] = 8
        with self.assertRaises(PermissionError):
            admit_primary_fit(
                plan=primary, qualification_receipt=tampered,
                data_binding=data, exposure_binding=exposure,
                source_sha256="source", no_active_worker=True,
                one_use_submission_intent="intent-002",
                parent_checkpoint=None)

    def test_delta1_admission_requires_fresh_residual_state(self):
        qualification = TrainingPlan(mode="qualification", stage="delta1",
                                     optimizer_updates=2)
        primary = TrainingPlan(mode="primary", stage="delta1",
                               optimizer_updates=2)
        exposure = {"matched": True}
        receipt = create_qualification_receipt(
            plan=qualification, data_binding={"data": 1},
            exposure_binding=exposure, source_sha256="source",
            base_before_sha256="base", base_after_sha256="base",
            backend_proof={}, rank4_metrics=PASS, primary_plan=primary)
        admission = admit_primary_fit(
            plan=primary, qualification_receipt=receipt,
            data_binding={"data": 1}, exposure_binding=exposure,
            source_sha256="source", no_active_worker=True,
            one_use_submission_intent="intent", parent_checkpoint=None)
        self.assertIsNone(admission["sequential_parent"]["parent_stage"])
        with self.assertRaisesRegex(PermissionError, "Delta1"):
            admit_primary_fit(
                plan=primary, qualification_receipt=receipt,
                data_binding={"data": 1}, exposure_binding=exposure,
                source_sha256="source", no_active_worker=True,
                one_use_submission_intent="intent", parent_checkpoint={})

    def test_databricks_qualification_receipt_records_databricks_provenance(self):
        qualification = databricks_a10_plan(TrainingPlan(
            mode="qualification", stage="delta1", optimizer_updates=2))
        primary = databricks_a10_plan(TrainingPlan(
            mode="primary", stage="delta1", optimizer_updates=2))
        receipt = create_qualification_receipt(
            plan=qualification,
            data_binding={"data": 1},
            exposure_binding={"matched": True},
            source_sha256="source",
            base_before_sha256="base",
            base_after_sha256="base",
            backend_proof={"profile": "databricks-a10x1"},
            rank4_metrics=PASS,
            primary_plan=primary,
        )
        self.assertEqual(
            receipt["execution_provenance"],
            "actual_databricks_a10x1_worker_evidence",
        )

    def test_rank_receipt_self_hash_is_recomputed(self):
        payload = {
            "version": training_module.RANK_RUN_VERSION,
            "status": "complete", "plan_sha256": "p" * 64,
            "global_journal_sha256": "j" * 64,
            "global_journal_occurrence_count": 96,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        verify_rank_receipt(
            payload, expected_plan_sha256="p" * 64,
            expected_global_journal_sha256="j" * 64)
        payload["global_journal_occurrence_count"] = 95
        with self.assertRaisesRegex(PermissionError, "self-hash"):
            verify_rank_receipt(
                payload, expected_plan_sha256="p" * 64,
                expected_global_journal_sha256="j" * 64)

    def test_rank_receipt_recomputes_a10_batching_and_compatibility_scales(self):
        plan = databricks_a10_plan(TrainingPlan(
            mode="qualification", condition="self_continuity", stage="delta1",
            seed=202610281, optimizer_updates=2))
        _, _, tokenized, _ = bindings(plan)
        journal = adjacent_pair_schedule(tokenized[plan.condition], plan)
        updates = []
        for update_index in range(plan.optimizer_updates):
            entries = journal[update_index * 4:(update_index + 1) * 4]
            logical = sum(row["pair_training_tokens"] for row in entries)
            updates.append({
                "optimizer_update": update_index,
                "occurrence_indices": [row["occurrence_index"] for row in entries],
                "changed_component_scales": list(
                    databricks_a10_compatibility_scales(entries, plan)),
                "overflow_adjustments": 0,
                "overflow_recovered": False,
                "max_overflow_adjustments": 0,
                "minimum_grad_scaler_scale": 1.0,
                "grad_scaler_before": 1.0,
                "grad_scaler_after": 1.0,
                "gradient_norm_before_clip": 1.0,
                "gradient_norm_positive": True,
                "active_branch_finite_nonzero_gradient_element_count": 10,
                "active_branch_finite_nonzero_gradient_density": 0.5,
                "fixed_scale_no_growth": True,
                "logical_tokens": logical,
                "initial_physical_tokens": logical,
                "replay_physical_tokens": 0,
                "physical_tokens_processed": logical,
                "overflow": False,
                "finite": True,
                "optimizer_step_pre_hook_calls": 1,
                "grad_scaler_attempts": [{
                    "attempt_index": 0,
                    "overflow": False,
                    "gradient_finite": True,
                    "optimizer_step_pre_hook_call_delta": 1,
                    "optimizer_step_executed": True,
                    "optimizer_step_pre_hook_calls_before": 0,
                    "optimizer_step_pre_hook_calls_after": 1,
                    "scale_adjustment_applied": False,
                    "scale_before": 1.0,
                    "scale_after": 1.0,
                    "gradient_norm_before_clip": 1.0,
                    "gradient_norm_positive": True,
                    "active_branch_finite_nonzero_gradient_element_count": 10,
                    "active_branch_finite_nonzero_gradient_density": 0.5,
                    "diagnostic": {
                        "gradient_norm_finite": True,
                        "gradient_norm_positive": True,
                        "nonfinite_gradient_element_count": 0,
                        "trainable_gradient_element_count": 20,
                        "active_branch_finite_nonzero_gradient_element_count": 10,
                        "active_branch_finite_nonzero_gradient_density": 0.5,
                    },
                }],
            })
        receipt = {
            "version": training_module.RANK_RUN_VERSION,
            "status": "complete", "plan_sha256": plan.full_plan_sha256,
            "global_journal_sha256": canonical_sha256(journal),
            "batching": batching_exposure_contract(plan), "updates": updates,
            "executed_optimizer_step_count": 2,
            "durable_published_update_count": 2,
            "successful_optimizer_updates": 2,
            "tokens_seen": sum(row["logical_tokens"] for row in updates),
            "overflow_recompute_tokens": 0,
            "physical_tokens_processed": sum(
                row["logical_tokens"] for row in updates),
            "grad_scaler_recovery": {
                "numerical_policy": DATABRICKS_A10_FIXED_SCALE1_POLICY,
                "initial_scale": 1.0,
                "backoff_factor": 0.5,
                "max_overflow_adjustments_per_partition": 0,
                "total_overflow_adjustments": 0,
                "minimum_scale": 1.0,
                "journal_advances_only_after_finite_optimizer_step": True,
                "replays_exact_rng_state_and_occurrence_indices": False,
            },
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        strict = dict(
            expected_plan_sha256=plan.full_plan_sha256,
            expected_global_journal_sha256=canonical_sha256(journal),
            expected_plan=plan, expected_global_journal=journal)
        self.assertEqual(verify_rank_receipt(receipt, **strict), receipt)
        for mutation in ("scale", "batching", "chain", "finite_decrease"):
            changed = copy.deepcopy(receipt)
            if mutation == "scale":
                changed["updates"][0]["changed_component_scales"][0] = 99.0
            elif mutation == "batching":
                changed["batching"]["micro_batch_size"] = 2
            elif mutation == "chain":
                changed["updates"][1]["grad_scaler_before"] = 32768.0
                changed["updates"][1]["grad_scaler_after"] = 32768.0
                changed["updates"][1]["grad_scaler_attempts"][0][
                    "scale_before"] = 32768.0
                changed["updates"][1]["grad_scaler_attempts"][0][
                    "scale_after"] = 32768.0
            else:
                changed["updates"][0]["grad_scaler_after"] = 32768.0
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "scale_after"] = 32768.0
            changed.pop("receipt_sha256")
            changed["receipt_sha256"] = canonical_sha256(changed)
            with self.assertRaises(PermissionError):
                verify_rank_receipt(changed, **strict)

    def test_backbone_change_blocks_qualification(self):
        plan = TrainingPlan(optimizer_updates=16)
        _, data, _, exposure = bindings(plan)
        with self.assertRaises(PermissionError):
            create_qualification_receipt(
                plan=plan, data_binding=data, exposure_binding=exposure,
                source_sha256="source", base_before_sha256="before",
                base_after_sha256="after", backend_proof={}, rank4_metrics=PASS,
                primary_plan=TrainingPlan(mode="primary", optimizer_updates=16))

    def test_optimizer_hash_changes_after_step(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        before = optimizer_sha256(optimizer)
        parameter.grad = torch.tensor([1.0])
        optimizer.step()
        after = optimizer_sha256(optimizer)
        self.assertNotEqual(before, after)

    def test_t4x2_map_and_bf16_gate(self):
        class Config:
            num_hidden_layers = 32

        mapping = qwen35_t4x2_device_map(Config())
        self.assertEqual(mapping, {"": 0})
        self.assertEqual(qwen35_t4x2_device_map(Config(), 1), {"": 1})

        class Properties:
            total_memory = 16 * 1024**3
            major = 7
            minor = 5

        class Cuda:
            @staticmethod
            def is_available(): return True
            @staticmethod
            def device_count(): return 2
            @staticmethod
            def get_device_properties(index): return Properties()
            @staticmethod
            def get_device_name(index): return f"Tesla T4 {index}"
            @staticmethod
            def is_bf16_supported(): return False

        class FakeTorch:
            cuda = Cuda()

        proof = assert_t4x2_bf16_runtime(FakeTorch())
        self.assertEqual(proof["compute_dtype"], "float16")

        class A10Properties(Properties):
            total_memory = 24 * 1024**3
            major = 8
            minor = 6
        class A10Cuda(Cuda):
            @staticmethod
            def device_count(): return 1
            @staticmethod
            def get_device_properties(index): return A10Properties()
            @staticmethod
            def get_device_name(index): return f"NVIDIA A10 {index}"
        class A10Torch:
            cuda = A10Cuda()
        self.assertEqual(assert_databricks_a10x1_fp16_runtime(
            A10Torch())["profile"], "databricks-a10x1")

    def test_plan_rejects_warm_start_or_recipe_drift(self):
        for plan in (
                TrainingPlan(rank=16),
                TrainingPlan(initialized_from="old-adapter"),
                TrainingPlan(precision="bf16"),
                TrainingPlan(expected_train_families=7)):
            with self.assertRaises((ValueError, PermissionError)):
                plan.validate()

    def test_databricks_a10_requires_microbatch1_accum4(self):
        exact = databricks_a10_plan(TrainingPlan())
        self.assertEqual(exact.global_batch_size, 4)
        self.assertEqual(exact.numerical_policy,
                         DATABRICKS_A10_FIXED_SCALE1_POLICY)
        self.assertEqual(exact.max_overflow_adjustments, 0)
        self.assertEqual(exact.minimum_grad_scaler_scale, 1.0)
        for micro_batch_size, accumulation_steps in (
                (2, 2), (1, 2), (2, 1), (4, 1)):
            with self.subTest(micro_batch_size=micro_batch_size,
                              accumulation_steps=accumulation_steps):
                with self.assertRaisesRegex(ValueError, "batch1/accum4"):
                    TrainingPlan(
                        device_profile="databricks-a10x1", world_size=1,
                        micro_batch_size=micro_batch_size,
                        accumulation_steps=accumulation_steps,
                        numerical_policy=DATABRICKS_A10_FIXED_SCALE1_POLICY,
                        grad_scaler_initial_scale=1.0,
                        max_overflow_adjustments=0,
                        minimum_grad_scaler_scale=1.0).validate()

    def test_a10_policy_is_plan_bound_and_legacy_routes_are_failfast(self):
        legacy = TrainingPlan().validate()
        self.assertEqual(legacy.numerical_policy, LEGACY_GRAD_SCALER_POLICY)
        self.assertEqual(legacy.max_overflow_adjustments, 0)
        self.assertEqual(legacy.minimum_grad_scaler_scale, 0.0)
        a10 = databricks_a10_plan(legacy)
        self.assertNotEqual(a10.full_plan_sha256, legacy.full_plan_sha256)
        for changed in (
                replace(a10, numerical_policy=LEGACY_GRAD_SCALER_POLICY),
                replace(a10, max_overflow_adjustments=3),
                replace(a10, minimum_grad_scaler_scale=1024.0),
                replace(legacy, numerical_policy=DATABRICKS_A10_GRAD_SCALER_POLICY,
                        max_overflow_adjustments=4,
                        minimum_grad_scaler_scale=4096.0)):
            with self.assertRaises(PermissionError):
                changed.validate()

    def test_fixed_scale1_successor_is_a10_only_and_plan_bound(self):
        legacy = TrainingPlan().validate()
        bounded = databricks_a10_bounded_gradscaler_plan(legacy)
        fixed = databricks_a10_plan(legacy)
        self.assertEqual(fixed.numerical_policy,
                         DATABRICKS_A10_FIXED_SCALE1_POLICY)
        self.assertEqual(fixed.grad_scaler_initial_scale,
                         DATABRICKS_A10_FIXED_SCALE)
        self.assertEqual(fixed.max_overflow_adjustments, 0)
        self.assertEqual(fixed.minimum_grad_scaler_scale,
                         DATABRICKS_A10_FIXED_SCALE)
        self.assertNotEqual(fixed.full_plan_sha256, bounded.full_plan_sha256)
        self.assertEqual(legacy.numerical_policy, LEGACY_GRAD_SCALER_POLICY)
        with self.assertRaises(PermissionError):
            replace(legacy,
                    numerical_policy=DATABRICKS_A10_FIXED_SCALE1_POLICY,
                    grad_scaler_initial_scale=1.0,
                    minimum_grad_scaler_scale=1.0).validate()
        for field in (
                "grad_scaler_initial_scale", "grad_scaler_backoff_factor",
                "minimum_grad_scaler_scale", "max_overflow_adjustments",
                "world_size", "micro_batch_size", "accumulation_steps",
                "global_batch_size", "optimizer_updates",
                "automatic_retries"):
            with self.subTest(field=field), self.assertRaises(
                    (TypeError, PermissionError)):
                replace(fixed, **{field: True if getattr(fixed, field) != 1
                                  else False}).validate()
        scalar_aliases = {
            "alpha": 8,
            "learning_rate": 0,
            "weight_decay": False,
            "sft_weight": True,
            "preference_weight": 0,
            "preference_beta": 0,
            "cross_danger_margin_weight": 0,
            "cross_danger_margin": 0,
            "max_grad_norm": True,
            "qualification_min_loss_improvement": 0,
            "grad_scaler_initial_scale": 1,
            "grad_scaler_backoff_factor": 0,
            "minimum_grad_scaler_scale": 1,
        }
        for field, value in scalar_aliases.items():
            with self.subTest(scalar=field), self.assertRaises(TypeError):
                replace(fixed, **{field: value}).validate()

    def test_a10_virtual_batch2_reduction_matches_all_boolean_patterns(self):
        plan = databricks_a10_plan(TrainingPlan())
        expected_examples = {
            (False, True, False, True): (1.0, 2.0, 1.0, 2.0),
            (True, True, False, False): (1.0, 1.0, 1.0, 1.0),
            (True, False, False, False): (2.0, 1.0, 1.0, 1.0),
        }
        for bits in range(16):
            pattern = tuple(bool(bits & (1 << index)) for index in range(4))
            entries = [{"target_changed": value} for value in pattern]
            scales = databricks_a10_compatibility_scales(entries, plan)
            legacy = component_occurrence_weights(
                pattern, micro_batch_size=2, accumulation_steps=2)
            current = component_occurrence_weights(
                pattern, micro_batch_size=1, accumulation_steps=4)
            self.assertEqual(current["sft"], legacy["sft"])
            for component in ("preference", "cross_danger_margin"):
                self.assertEqual(
                    tuple(current[component][index] * scales[index]
                          for index in range(4)),
                    legacy[component])
            if pattern in expected_examples:
                self.assertEqual(scales, expected_examples[pattern])

    def test_a10_resource_amendment_preserves_loss_and_gradient_formula(self):
        plan = databricks_a10_plan(TrainingPlan())
        for bits in range(16):
            pattern = tuple(bool(bits & (1 << index)) for index in range(4))
            entries = [{"target_changed": value} for value in pattern]
            scales = databricks_a10_compatibility_scales(entries, plan)
            legacy = component_occurrence_weights(
                pattern, micro_batch_size=2, accumulation_steps=2)
            current = component_occurrence_weights(
                pattern, micro_batch_size=1, accumulation_steps=4)
            old_parameter = torch.tensor(.37, dtype=torch.float64,
                                         requires_grad=True)
            new_parameter = old_parameter.detach().clone().requires_grad_(True)

            def terms(parameter):
                sft = [(parameter - (index + 1) / 10.0).square()
                       for index in range(4)]
                preference = [torch.nn.functional.softplus(
                    parameter * (index + 1)) for index in range(4)]
                margin = [torch.relu(.2 - parameter * (index + 2))
                          for index in range(4)]
                return sft, preference, margin

            old_terms = terms(old_parameter); new_terms = terms(new_parameter)
            old_loss = sum(
                legacy["sft"][index] * old_terms[0][index]
                + legacy["preference"][index] * old_terms[1][index]
                + legacy["cross_danger_margin"][index] * old_terms[2][index]
                for index in range(4))
            new_loss = sum(
                current["sft"][index] * new_terms[0][index]
                + current["preference"][index] * scales[index]
                  * new_terms[1][index]
                + current["cross_danger_margin"][index] * scales[index]
                  * new_terms[2][index]
                for index in range(4))
            old_loss.backward(); new_loss.backward()
            self.assertTrue(torch.equal(old_loss, new_loss))
            self.assertTrue(torch.equal(old_parameter.grad, new_parameter.grad))

    def test_a10_schedule_and_reduction_equivalence_for_qualifier_and_matrix(self):
        source_plan = TrainingPlan()
        _, _, tokenized, _ = bindings(source_plan)
        plans = [databricks_a10_plan(TrainingPlan(
            mode="qualification", condition="self_continuity", stage="delta1",
            seed=202610281, optimizer_updates=2))]
        plans += [databricks_a10_plan(TrainingPlan(
            mode="primary", condition=condition, stage=stage, seed=seed,
            optimizer_updates=144))
            for condition in CONDITIONS
            for seed in (202610281, 202610282, 202610283)
            for stage in ("delta1", "delta2", "delta3")]
        self.assertEqual(len(plans), 28)
        for plan in plans:
            journal = adjacent_pair_schedule(tokenized[plan.condition], plan)
            prior_geometry = replace(
                plan, device_profile="kaggle-t4x2", world_size=2,
                micro_batch_size=2, accumulation_steps=1,
                numerical_policy=LEGACY_GRAD_SCALER_POLICY,
                grad_scaler_initial_scale=65536.0,
                max_overflow_adjustments=0,
                minimum_grad_scaler_scale=0.0).validate()
            prior_journal = adjacent_pair_schedule(
                tokenized[plan.condition], prior_geometry)
            preserved_fields = (
                "optimizer_update", "global_batch_position", "paired_group_id",
                "low_record_id", "high_record_id", "target_changed")
            self.assertEqual(
                [tuple(row[field] for field in preserved_fields) for row in journal],
                [tuple(row[field] for field in preserved_fields)
                 for row in prior_journal])
            self.assertEqual(len(journal), plan.occurrence_count)
            for update in range(plan.optimizer_updates):
                entries = journal[update * 4:(update + 1) * 4]
                scales = databricks_a10_compatibility_scales(entries, plan)
                changed = tuple(row["target_changed"] for row in entries)
                old = component_occurrence_weights(
                    changed, micro_batch_size=2, accumulation_steps=2)
                new = component_occurrence_weights(
                    changed, micro_batch_size=1, accumulation_steps=4)
                self.assertEqual(old["sft"], new["sft"])
                self.assertEqual(
                    old["preference"],
                    tuple(value * scales[index]
                          for index, value in enumerate(new["preference"])))

    def test_real_rank_fallback_runner_never_runs_rank8_when_rank4_is_adequate(self):
        calls = []
        rank4 = dict(PASS)
        selected, first, second = run_capacity_qualification(
            rank4_runner=lambda rank: calls.append(rank) or rank4,
            rank8_runner=lambda rank: calls.append(rank) or rank4)
        self.assertEqual((selected, calls, second), (4, [4], None))
        inadequate = {**rank4, "capacity_adequate": False}
        selected, _, second = run_capacity_qualification(
            rank4_runner=lambda rank: inadequate,
            rank8_runner=lambda rank: rank4)
        self.assertEqual(selected, 8)
        self.assertTrue(second["capacity_adequate"])

    def test_executable_qualification_derives_gates_from_raw_receipt(self):
        calls = []
        rank4 = TrainingPlan(mode="qualification", rank=4, alpha=8.0,
                             optimizer_updates=2)
        rank8 = TrainingPlan(mode="qualification", rank=8, alpha=16.0,
                             optimizer_updates=2)
        def execute(plan):
            calls.append(plan.rank)
            return {
                "run_receipt": {
                    "status": "complete", "plan_sha256": plan.full_plan_sha256,
                    "active_branch_changed": True,
                    "successful_optimizer_updates": 2,
                    "predecessor_branch_hashes": {},
                    "stage_after": {"delta1": "x"},
                    "updates": [
                        {"loss": 2.0, "finite": True, "overflow": False},
                        {"loss": 1.0, "finite": True, "overflow": False},
                    ],
                    "resources": {"tokens_per_second": 10.0,
                                  "disk_free_bytes": 9 * 1024**3,
                                  "gpu": {"max_allocated_bytes": 8 * 1024**3,
                                          "nvml": {
                                              "available": True,
                                              "sample_count": 4,
                                              "nonzero_gpu_sample_count": 4,
                                              "mean_utilization_percent": 70.0,
                                          }}},
                    "optimizer": {"file_bytes": 128},
                    "fast_dispatch": {"forward_count": 120,
                                      "grad_forward_count": 96,
                                      "backward_count": 48},
                },
                "runtime_proof": {"backend": {
                    "kernel_selection": {"profile": "fla"},
                    "gradient_checkpointing_enabled": True}},
                "checkpoint_metadata": {"run_mode": "qualification",
                                        "primary_parent_eligible": False},
                "save_reload_pass": True,
                "live_base_components_before": {
                    "frozen_parameters_sha256": "1" * 64,
                    "quantization_state_sha256": "2" * 64,
                    "quantization_state_inventory": [{"parameter": "q"}],
                    "quantization_state_inventory_sha256": "5" * 64,
                    "persistent_buffers_sha256": "3" * 64,
                    "config_sha256": "4" * 64,
                    "combined_sha256": "b" * 64,
                },
                "live_base_components_after": {
                    "frozen_parameters_sha256": "1" * 64,
                    "quantization_state_sha256": "2" * 64,
                    "quantization_state_inventory": [{"parameter": "q"}],
                    "quantization_state_inventory_sha256": "5" * 64,
                    "persistent_buffers_sha256": "3" * 64,
                    "config_sha256": "4" * 64,
                    "combined_sha256": "b" * 64,
                },
            }
        receipt = execute_capacity_qualification(
            rank4_plan=rank4, rank8_plan=rank8,
            primary_plan_for_rank=lambda rank: TrainingPlan(
                mode="primary", rank=rank, alpha=2.0 * rank,
                optimizer_updates=2),
            execute_candidate=execute, data_binding={"data": 1},
            exposure_binding={"exposure": 1}, source_sha256="s" * 64,
            base_sha256="b" * 64)
        self.assertEqual(calls, [4])
        self.assertEqual(receipt["selected_rank"], 4)
        self.assertTrue(receipt["qualifier_state_discarded"])

    def test_batch_fallback_is_only_for_cuda_oom_and_preserves_global_batch(self):
        calls = []
        def run(plan):
            calls.append((plan.micro_batch_size, plan.accumulation_steps))
            if len(calls) == 1:
                raise RuntimeError("CUDA out of memory")
            return {"complete": True}
        selected, evidence, failure = execute_with_preregistered_batch_fallback(
            TrainingPlan(), run)
        self.assertEqual(calls, [(2, 1), (1, 2)])
        self.assertEqual(selected.global_batch_size, 4)
        self.assertTrue(evidence["complete"])
        self.assertFalse(failure["scientific_retry"])
        with self.assertRaisesRegex(RuntimeError, "kernel"):
            execute_with_preregistered_batch_fallback(
                TrainingPlan(), lambda _: (_ for _ in ()).throw(
                    RuntimeError("kernel dispatch failed")))

    def test_old_failfast_gradscaler_sequence_does_not_reduce_scale(self):
        model = ToyScaledParameter()
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler(
            "cpu", enabled=True, init_scale=65536.0, growth_interval=1000)
        loss = (model.delta1_weight.to(torch.float16)
                * torch.tensor([1.0], dtype=torch.float16)).sum()
        self.assertTrue(bool(torch.isfinite(loss)))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.assertFalse(bool(torch.isfinite(gradient_norm)))
        scale_before = scaler.get_scale()
        # This is the terminal v4 ordering: raise before step/update.
        scale_after_failfast = scaler.get_scale()
        self.assertEqual(scale_before, 65536.0)
        self.assertEqual(scale_after_failfast, scale_before)
        self.assertEqual(optimizer.executed_step_count, 0)
        self.assertEqual(float(model.delta1_weight.detach()), 1.0)

    def test_bounded_gradscaler_recovery_replays_rng_and_steps_once(self):
        model = ToyScaledParameter()
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler(
            "cpu", enabled=True, init_scale=65536.0, growth_interval=1000)
        draws = []

        def compute_backward():
            draw = torch.rand(())
            draws.append(float(draw))
            factor = (torch.tensor(1.25) + draw * 0.25).to(torch.float16)
            loss = (model.delta1_weight.to(torch.float16) * factor).sum()
            self.assertTrue(bool(torch.isfinite(loss)))
            scaler.scale(loss).backward()
            value = float(loss.detach())
            return [{"loss": value, "sft": value, "preference": 0.0,
                     "cross_danger_margin": 0.0, "high_advantage": 0.0,
                     "low_advantage": 0.0, "level0_kl_anchor": 0.0}]

        result = bounded_grad_scaler_update(
            model=model, optimizer=optimizer, scaler=scaler, stage="delta1",
            device=torch.device("cpu"), max_grad_norm=1.0,
            compute_backward=compute_backward)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["overflow_adjustments"], 1)
        self.assertEqual(result["recompute_count"], 1)
        self.assertTrue(result["overflow_recovered"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(draws[0], draws[1])
        self.assertTrue(result["attempts"][0]["overflow"])
        self.assertTrue(result["attempts"][0]["scale_adjustment_applied"])
        self.assertFalse(result["attempts"][0]["optimizer_step_executed"])
        self.assertEqual(
            result["attempts"][0]["optimizer_step_pre_hook_call_delta"], 0)
        self.assertFalse(result["attempts"][1]["overflow"])
        self.assertTrue(result["attempts"][1]["optimizer_step_executed"])
        self.assertEqual(
            result["attempts"][1]["optimizer_step_pre_hook_call_delta"], 1)
        self.assertEqual(result["optimizer_step_pre_hook_calls"], 1)
        self.assertEqual(optimizer.executed_step_count, 1)
        self.assertEqual(len(optimizer._optimizer_step_pre_hooks), 0)
        self.assertLess(float(model.delta1_weight.detach()), 1.0)
        self.assertEqual(result["max_overflow_adjustments"],
                         GRAD_SCALER_MAX_OVERFLOW_ADJUSTMENTS)
        self.assertEqual(result["minimum_scale"], GRAD_SCALER_MIN_SCALE)

    def test_bounded_gradscaler_recovery_fails_closed_at_limit(self):
        model = ToyScaledParameter()
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler(
            "cpu", enabled=True, init_scale=65536.0, growth_interval=1000)

        def persistent_overflow():
            loss = (model.delta1_weight.to(torch.float16)
                    * torch.tensor([4.0], dtype=torch.float16)).sum()
            scaler.scale(loss).backward()
            return [{"loss": float(loss.detach()), "sft": float(loss.detach()),
                     "preference": 0.0, "cross_danger_margin": 0.0,
                     "high_advantage": 0.0, "low_advantage": 0.0,
                     "level0_kl_anchor": 0.0}]

        with self.assertRaises(GradScalerRecoveryError) as raised:
            bounded_grad_scaler_update(
                model=model, optimizer=optimizer, scaler=scaler, stage="delta1",
                device=torch.device("cpu"), max_grad_norm=1.0,
                compute_backward=persistent_overflow,
                max_overflow_adjustments=1, min_scale=4096.0)
        self.assertEqual(raised.exception.receipt["reason"],
                         "adjustments_exhausted")
        self.assertEqual(len(raised.exception.receipt["attempts"]), 2)
        self.assertEqual(raised.exception.receipt["recompute_count"], 1)
        self.assertEqual(
            raised.exception.receipt["optimizer_step_pre_hook_calls"], 0)
        self.assertEqual(optimizer.executed_step_count, 0)
        self.assertEqual(len(optimizer._optimizer_step_pre_hooks), 0)
        self.assertEqual(float(model.delta1_weight.detach()), 1.0)
        self.assertIsNone(model.delta1_weight.grad)

    def test_recovered_update_matches_direct_finite_scale_update(self):
        def execute(initial_scale):
            torch.manual_seed(20261029)
            model = ToyScaledParameter()
            optimizer = CountingSGD(model.parameters(), lr=0.1)
            scaler = torch.amp.GradScaler(
                "cpu", enabled=True, init_scale=initial_scale,
                growth_interval=1000)
            draws = []

            def compute_backward():
                draw = torch.rand(())
                draws.append(float(draw))
                factor = (torch.tensor(1.25) + draw * 0.25).to(torch.float16)
                loss = (model.delta1_weight.to(torch.float16) * factor).sum()
                scaler.scale(loss).backward()
                value = float(loss.detach())
                return [{"loss": value, "sft": value, "preference": 0.0,
                         "cross_danger_margin": 0.0, "high_advantage": 0.0,
                         "low_advantage": 0.0, "level0_kl_anchor": 0.0}]

            result = bounded_grad_scaler_update(
                model=model, optimizer=optimizer, scaler=scaler,
                stage="delta1", device=torch.device("cpu"),
                max_grad_norm=1.0, compute_backward=compute_backward)
            return model.delta1_weight.detach().clone(), result, draws

        recovered_weight, recovered, recovered_draws = execute(65536.0)
        direct_weight, direct, direct_draws = execute(32768.0)
        self.assertEqual(recovered["overflow_adjustments"], 1)
        self.assertEqual(direct["overflow_adjustments"], 0)
        self.assertEqual(recovered_draws[0], recovered_draws[1])
        self.assertEqual(recovered_draws[-1], direct_draws[-1])
        self.assertTrue(torch.equal(recovered_weight, direct_weight))

    def test_bounded_gradscaler_recovery_enforces_minimum_scale(self):
        model = ToyScaledParameter()
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler(
            "cpu", enabled=True, init_scale=65536.0, growth_interval=1000)

        def one_overflow():
            loss = (model.delta1_weight.to(torch.float16)
                    * torch.tensor([1.0], dtype=torch.float16)).sum()
            scaler.scale(loss).backward()
            value = float(loss.detach())
            return [{"loss": value, "sft": value, "preference": 0.0,
                     "cross_danger_margin": 0.0, "high_advantage": 0.0,
                     "low_advantage": 0.0, "level0_kl_anchor": 0.0}]

        with self.assertRaises(GradScalerRecoveryError) as raised:
            bounded_grad_scaler_update(
                model=model, optimizer=optimizer, scaler=scaler, stage="delta1",
                device=torch.device("cpu"), max_grad_norm=1.0,
                compute_backward=one_overflow,
                max_overflow_adjustments=4, min_scale=40000.0)
        self.assertEqual(raised.exception.receipt["reason"], "minimum_scale")
        self.assertEqual(raised.exception.receipt["overflow_adjustments"], 1)
        self.assertEqual(raised.exception.receipt["recompute_count"], 0)
        self.assertEqual(optimizer.executed_step_count, 0)

    def test_adamw_overflow_mutates_neither_parameter_nor_state(self):
        model = ToyScaledParameter()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        step_calls = {"count": 0}
        handle = optimizer.register_step_pre_hook(
            lambda _optimizer, _args, _kwargs:
            step_calls.__setitem__("count", step_calls["count"] + 1))
        scaler = torch.amp.GradScaler(
            "cpu", enabled=True, init_scale=65536.0, growth_interval=1000)
        before = model.delta1_weight.detach().clone()
        try:
            loss = (model.delta1_weight.to(torch.float16)
                    * torch.tensor([1.0], dtype=torch.float16)).sum()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            self.assertFalse(bool(torch.isfinite(model.delta1_weight.grad).all()))
            scaler.step(optimizer)
            scaler.update()
        finally:
            handle.remove()
        self.assertEqual(step_calls["count"], 0)
        self.assertTrue(torch.equal(model.delta1_weight.detach(), before))
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(scaler.get_scale(), 32768.0)

    def test_fixed_scale1_matches_direct_unscaled_adamw_update(self):
        def execute_fixed():
            model = ToyScaledParameter()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scaler = torch.amp.GradScaler(
                "cpu", enabled=True, init_scale=1.0,
                growth_interval=1000)

            def compute_backward():
                loss = (model.delta1_weight * 2.0).sum()
                scaler.scale(loss).backward()
                value = float(loss.detach())
                return [{"loss": value, "sft": value,
                         "preference": 0.0,
                         "cross_danger_margin": 0.0,
                         "high_advantage": 0.0, "low_advantage": 0.0,
                         "level0_kl_anchor": 0.0}]

            result = fixed_scale1_grad_scaler_update(
                model=model, optimizer=optimizer, scaler=scaler,
                stage="delta1", device=torch.device("cpu"),
                max_grad_norm=1.0, compute_backward=compute_backward)
            return (model.delta1_weight.detach().clone(),
                    optimizer_sha256(optimizer), result, scaler.get_scale())

        def execute_direct():
            model = ToyScaledParameter()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss = (model.delta1_weight * 2.0).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            return model.delta1_weight.detach().clone(), optimizer_sha256(optimizer)

        fixed_weight, fixed_optimizer, result, scale = execute_fixed()
        direct_weight, direct_optimizer = execute_direct()
        self.assertTrue(torch.equal(fixed_weight, direct_weight))
        self.assertEqual(fixed_optimizer, direct_optimizer)
        self.assertEqual(scale, 1.0)
        self.assertEqual(result["optimizer_step_pre_hook_calls"], 1)
        self.assertTrue(result["gradient_norm_positive"])
        self.assertGreater(
            result["active_branch_finite_nonzero_gradient_density"], 0.0)

    def test_fixed_scale1_nonfinite_gradient_never_steps_and_diagnoses(self):
        class FiniteForwardInfBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value * 0.0 + 1.0

            @staticmethod
            def backward(ctx, grad_output):
                return torch.full_like(grad_output, float("inf"))

        model = ToyScaledParameter()
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler(
            "cpu", enabled=True, init_scale=1.0, growth_interval=1000)
        terminal_calls = []

        def compute_backward():
            loss = FiniteForwardInfBackward.apply(
                model.delta1_weight).sum()
            self.assertTrue(bool(torch.isfinite(loss)))
            scaler.scale(loss).backward()
            return [{"loss": 1.0, "sft": 1.0, "preference": 0.0,
                     "cross_danger_margin": 0.0, "high_advantage": 0.0,
                     "low_advantage": 0.0, "level0_kl_anchor": 0.0}]

        def terminal(scale):
            terminal_calls.append(scale)
            return {"status": "first_nonfinite_component_found",
                    "components_examined": 1,
                    "diagnostic_physical_tokens": 4}

        with self.assertRaises(GradScalerRecoveryError) as raised:
            fixed_scale1_grad_scaler_update(
                model=model, optimizer=optimizer, scaler=scaler,
                stage="delta1", device=torch.device("cpu"),
                max_grad_norm=1.0, compute_backward=compute_backward,
                terminal_diagnostic=terminal)
        receipt = raised.exception.receipt
        self.assertEqual(receipt["reason"], "nonfinite_gradient")
        self.assertEqual(receipt["optimizer_step_pre_hook_calls"], 0)
        self.assertEqual(optimizer.executed_step_count, 0)
        self.assertEqual(terminal_calls, [1.0])
        self.assertEqual(scaler.get_scale(), 1.0)
        self.assertIsNone(model.delta1_weight.grad)

    def test_gradient_diagnostic_is_count_only_and_component_wise(self):
        model = ToyScaledParameter()
        model.delta1_weight.grad = torch.tensor([float("nan")])
        diagnostic = gradient_finiteness_diagnostic(
            model, [{"loss": 1.0, "sft": float("inf"),
                     "preference": 0.0, "cross_danger_margin": 0.0,
                     "high_advantage": 0.0, "low_advantage": 0.0,
                     "level0_kl_anchor": 0.0}],
            stage="delta1", gradient_norm_value=float("inf"))
        first = diagnostic["first_nonfinite_gradient"]
        self.assertEqual(first["active_branch"], "delta1")
        self.assertEqual(first["parameter_name"], "delta1_weight")
        self.assertEqual(first["nan_count"], 1)
        self.assertEqual(first["nonfinite_count"], 1)
        self.assertEqual(diagnostic["first_nonfinite_loss_component"], "sft")
        self.assertEqual(diagnostic[
            "active_branch_finite_nonzero_gradient_element_count"], 0)
        self.assertEqual(diagnostic[
            "active_branch_finite_nonzero_gradient_density"], 0.0)
        self.assertNotIn("values", json.dumps(diagnostic))

    def test_terminal_component_replay_clears_gradients_and_never_steps(self):
        model = ToyScaledParameter()
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        before = model.delta1_weight.detach().clone()
        seen = []

        def compute_component(component, scale):
            self.assertIsNone(model.delta1_weight.grad)
            seen.append(component)
            coefficient = {
                "sft": 1.0, "preference": 2.0,
                "cross_danger_margin": 3.0,
            }[component]
            loss = (model.delta1_weight * coefficient).sum()
            (loss * scale).backward()
            value = float(loss.detach())
            return [{"loss": value, "sft": value, "preference": value,
                     "cross_danger_margin": value, "high_advantage": 0.0,
                     "low_advantage": 0.0, "level0_kl_anchor": 0.0}]

        result = terminal_component_gradient_replay(
            model=model, optimizer=optimizer, device=torch.device("cpu"),
            stage="delta1", scale=4096.0,
            compute_component_backward=compute_component)
        self.assertEqual(result["status"], "all_component_gradients_finite")
        self.assertEqual(seen, ["sft", "preference", "cross_danger_margin"])
        self.assertEqual(result["components_examined"], 3)
        self.assertFalse(result["optimizer_step_executed"])
        self.assertEqual(optimizer.executed_step_count, 0)
        self.assertTrue(torch.equal(model.delta1_weight.detach(), before))
        self.assertIsNone(model.delta1_weight.grad)

    def test_serialized_guard_is_exclusive_and_owner_restored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "worker.lock"
            with SerializedRunGuard(path, intent_sha256="a" * 64):
                with self.assertRaises(RuntimeError):
                    with SerializedRunGuard(path, intent_sha256="b" * 64):
                        pass
            self.assertFalse(path.exists())

    def test_run_stage_recovers_same_update_once_without_journal_advance(self):
        plan = databricks_a10_bounded_gradscaler_plan(TrainingPlan(
            mode="qualification", stage="delta1", condition="self_continuity",
            seed=202610281, optimizer_updates=1))
        model = tiny_wrapped("delta1")
        tokenizer = ToyTokenizer()
        records = tokenize_rows(dataset("self_continuity"), tokenizer, 1024)
        expected = adjacent_pair_schedule(records, plan)
        expected_bindings = [{
            "occurrence_index": row["occurrence_index"],
            "low_record_id": row["low_record_id"],
            "high_record_id": row["high_record_id"],
        } for row in expected]
        logical_tokens = sum(row["pair_training_tokens"] for row in expected)
        progress = []
        calls = []
        scaler = torch.amp.GradScaler(
            "cpu", enabled=True, init_scale=65536.0, growth_interval=1000)
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
                training_module, "_grad_scaler", return_value=scaler), \
                mock.patch.object(
                    training_module, "ordinal_pair_loss",
                    side_effect=fake_scaled_ordinal_loss(4.0, calls)):
            receipt = run_training_stage(
                model=model, tokenizer=tokenizer, tokenized_records=records,
                plan=plan, output_dir=folder, target_names=["proj"],
                source_sha256="s" * 64, base_sha256="b" * 64,
                rank=0, world_size=1, save_checkpoint=False,
                callable_model=model,
                progress_callback=durable_progress_sink(progress))
        self.assertEqual(receipt["successful_optimizer_updates"], 1)
        self.assertEqual(len(receipt["updates"]), 1)
        self.assertEqual(len(progress), 1)
        update = receipt["updates"][0]
        self.assertEqual(update["optimizer_update"], 0)
        self.assertEqual(update["occurrence_bindings"], expected_bindings)
        self.assertEqual(update["occurrence_bindings_sha256"],
                         canonical_sha256(expected_bindings))
        self.assertEqual(update["compatibility_scales"],
                         list(databricks_a10_compatibility_scales(expected, plan)))
        self.assertEqual(update["overflow_adjustments"], 1)
        self.assertEqual(update["optimizer_step_pre_hook_calls"], 1)
        self.assertEqual(update["logical_tokens"], logical_tokens)
        self.assertEqual(update["initial_physical_tokens"], logical_tokens)
        self.assertEqual(update["replay_physical_tokens"], logical_tokens)
        self.assertEqual(update["physical_tokens_processed"], 2 * logical_tokens)
        self.assertEqual(receipt["tokens_seen"], logical_tokens)
        self.assertEqual(receipt["overflow_recompute_tokens"], logical_tokens)
        self.assertEqual(receipt["physical_tokens_processed"], 2 * logical_tokens)
        self.assertEqual(validate_grad_scaler_recovery_receipt(
            receipt, plan), receipt)
        tampered = copy.deepcopy(receipt)
        tampered["updates"][0]["grad_scaler_attempts"][0][
            "scale_after"] = 30000.0
        tampered["updates"][0]["grad_scaler_attempts"][1][
            "scale_before"] = 30000.0
        tampered["updates"][0]["grad_scaler_before"] = 30000.0
        tampered.pop("receipt_sha256")
        tampered["receipt_sha256"] = canonical_sha256(tampered)
        with self.assertRaises(PermissionError):
            verify_rank_receipt(
                tampered, expected_plan_sha256=plan.full_plan_sha256,
                expected_global_journal_sha256=canonical_sha256(expected),
                expected_plan=plan, expected_global_journal=expected)
        self.assertEqual(calls, ["combined"] * 8)

    def test_run_stage_fixed_scale1_two_updates_have_one_attempt_each(self):
        plan = databricks_a10_plan(TrainingPlan(
            mode="qualification", stage="delta1", condition="self_continuity",
            seed=202610281, optimizer_updates=2))
        model = tiny_wrapped("delta1")
        tokenizer = ToyTokenizer()
        records = tokenize_rows(dataset("self_continuity"), tokenizer, 1024)
        expected = adjacent_pair_schedule(records, plan)
        logical_tokens = sum(row["pair_training_tokens"] for row in expected)
        progress = []
        calls = []
        scaler = torch.amp.GradScaler(
            "cpu", enabled=True, init_scale=1.0, growth_interval=1000)
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
                training_module, "_grad_scaler", return_value=scaler), \
                mock.patch.object(
                    training_module, "ordinal_pair_loss",
                    side_effect=fake_scaled_ordinal_loss(4.0, calls)):
            receipt = run_training_stage(
                model=model, tokenizer=tokenizer, tokenized_records=records,
                plan=plan, output_dir=folder, target_names=["proj"],
                source_sha256="s" * 64, base_sha256="b" * 64,
                rank=0, world_size=1, save_checkpoint=False,
                callable_model=model,
                progress_callback=durable_progress_sink(progress))
        self.assertEqual(receipt["successful_optimizer_updates"], 2)
        self.assertEqual(receipt["executed_optimizer_step_count"], 2)
        self.assertEqual(receipt["durable_published_update_count"], 2)
        self.assertEqual(len(receipt["updates"]), 2)
        self.assertEqual(len(progress), 2)
        self.assertEqual(receipt["tokens_seen"], logical_tokens)
        self.assertEqual(receipt["physical_tokens_processed"], logical_tokens)
        self.assertEqual(receipt["overflow_recompute_tokens"], 0)
        self.assertEqual(scaler.get_scale(), 1.0)
        self.assertEqual(calls, ["combined"] * 8)
        for index, update in enumerate(receipt["updates"]):
            self.assertEqual(update["optimizer_update"], index)
            self.assertEqual(update["overflow_adjustments"], 0)
            self.assertEqual(update["replay_physical_tokens"], 0)
            self.assertEqual(update["physical_tokens_processed"],
                             update["logical_tokens"])
            self.assertEqual(update["optimizer_step_pre_hook_calls"], 1)
            self.assertTrue(update["gradient_norm_positive"])
            self.assertGreater(
                update["active_branch_finite_nonzero_gradient_element_count"],
                0)

            self.assertGreater(
                update["active_branch_finite_nonzero_gradient_density"], 0.0)
            self.assertTrue(update["fixed_scale_no_growth"])
            self.assertEqual(len(update["grad_scaler_attempts"]), 1)
            self.assertEqual(update["grad_scaler_before"], 1.0)
            self.assertEqual(update["grad_scaler_after"], 1.0)
        self.assertEqual(validate_grad_scaler_recovery_receipt(
            receipt, plan), receipt)
        strict = dict(
            expected_plan_sha256=plan.full_plan_sha256,
            expected_global_journal_sha256=canonical_sha256(expected),
            expected_plan=plan, expected_global_journal=expected)
        for mutation in (
                "norm", "norm_value", "density", "tokens", "scale",
                "attempt_finiteness", "step", "diagnostic_nonfinite",
                "diagnostic_count", "norm_bool", "norm_nan",
                "density_bool", "density_nan", "density_inf",
                "count_bool", "elements_bool", "nonfinite_count_bool",
                "logical_bool", "total_adjustment_bool",
                "current_adjustment_bool", "replay_bool", "physical_bool",
                "initial_bool", "update_index_bool", "step_count_bool",
                "attempt_step_count_bool", "static_initial_bool",
                "static_backoff_bool", "static_minimum_bool",
                "static_max_adjustment_bool", "attempt_scale_before_bool",
                "attempt_scale_after_bool", "top_tokens_bool",
                "top_replay_bool", "top_physical_bool",
                "summary_minimum_bool", "summary_before_bool",
                "summary_after_bool"):
            changed = copy.deepcopy(receipt)
            if mutation == "norm":
                changed["updates"][0]["gradient_norm_positive"] = False
            elif mutation == "norm_value":
                changed["updates"][0]["gradient_norm_before_clip"] = 0.0
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "gradient_norm_before_clip"] = 0.0
            elif mutation == "density":
                changed["updates"][0][
                    "active_branch_finite_nonzero_gradient_density"] = 0.0
            elif mutation == "tokens":
                changed["updates"][0]["physical_tokens_processed"] += 1
            elif mutation == "scale":
                changed["updates"][0]["grad_scaler_after"] = 2.0
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "scale_after"] = 2.0
            elif mutation == "attempt_finiteness":
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "gradient_finite"] = False
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "overflow"] = True
            elif mutation == "step":
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "optimizer_step_pre_hook_call_delta"] = 0
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "optimizer_step_executed"] = False
            elif mutation == "diagnostic_nonfinite":
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "diagnostic"]["nonfinite_gradient_element_count"] = 1
            elif mutation == "diagnostic_count":
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "diagnostic"][
                        "active_branch_finite_nonzero_gradient_element_count"] += 1
            elif mutation == "norm_bool":
                changed["updates"][0]["gradient_norm_before_clip"] = True
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "gradient_norm_before_clip"] = True
            elif mutation == "norm_nan":
                changed["updates"][0]["gradient_norm_before_clip"] = float("nan")
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "gradient_norm_before_clip"] = float("nan")
            elif mutation in ("density_bool", "density_nan", "density_inf"):
                value = {"density_bool": True, "density_nan": float("nan"),
                         "density_inf": float("inf")}[mutation]
                changed["updates"][0][
                    "active_branch_finite_nonzero_gradient_density"] = value
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "active_branch_finite_nonzero_gradient_density"] = value
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "diagnostic"][
                        "active_branch_finite_nonzero_gradient_density"] = value
            elif mutation == "count_bool":
                changed["updates"][0][
                    "active_branch_finite_nonzero_gradient_element_count"] = True
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "active_branch_finite_nonzero_gradient_element_count"] = True
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "diagnostic"][
                        "active_branch_finite_nonzero_gradient_element_count"] = True
            elif mutation == "elements_bool":
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "diagnostic"]["trainable_gradient_element_count"] = True
            elif mutation == "nonfinite_count_bool":
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "diagnostic"]["nonfinite_gradient_element_count"] = False
            elif mutation == "logical_bool":
                changed["updates"][0]["logical_tokens"] = True
            elif mutation == "total_adjustment_bool":
                changed["grad_scaler_recovery"][
                    "total_overflow_adjustments"] = False
            elif mutation == "current_adjustment_bool":
                changed["updates"][0]["overflow_adjustments"] = False
            elif mutation == "replay_bool":
                changed["updates"][0]["replay_physical_tokens"] = False
            elif mutation == "physical_bool":
                changed["updates"][0]["physical_tokens_processed"] = True
            elif mutation == "initial_bool":
                changed["updates"][0]["initial_physical_tokens"] = True
            elif mutation == "update_index_bool":
                changed["updates"][0]["optimizer_update"] = False
            elif mutation == "step_count_bool":
                changed["updates"][0]["optimizer_step_pre_hook_calls"] = True
            elif mutation == "attempt_step_count_bool":
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "optimizer_step_pre_hook_call_delta"] = True
            elif mutation == "static_initial_bool":
                changed["grad_scaler_recovery"]["initial_scale"] = True
            elif mutation == "static_backoff_bool":
                changed["grad_scaler_recovery"]["backoff_factor"] = True
            elif mutation == "static_minimum_bool":
                changed["grad_scaler_recovery"]["minimum_scale"] = True
            elif mutation == "static_max_adjustment_bool":
                changed["grad_scaler_recovery"][
                    "max_overflow_adjustments_per_partition"] = False
            elif mutation == "attempt_scale_before_bool":
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "scale_before"] = True
            elif mutation == "attempt_scale_after_bool":
                changed["updates"][0]["grad_scaler_attempts"][0][
                    "scale_after"] = True
            elif mutation == "top_tokens_bool":
                changed["tokens_seen"] = True
            elif mutation == "top_replay_bool":
                changed["overflow_recompute_tokens"] = False
            elif mutation == "top_physical_bool":
                changed["physical_tokens_processed"] = True
            elif mutation == "summary_minimum_bool":
                changed["updates"][0]["minimum_grad_scaler_scale"] = True
            elif mutation == "summary_before_bool":
                changed["updates"][0]["grad_scaler_before"] = True
            else:
                changed["updates"][0]["grad_scaler_after"] = True
            changed.pop("receipt_sha256")
            changed["receipt_sha256"] = canonical_sha256(changed)
            with self.subTest(mutation=mutation), self.assertRaises(
                    PermissionError):
                verify_rank_receipt(changed, **strict)

    def test_callback_failure_after_second_step_separates_executed_and_durable(self):
        plan = databricks_a10_plan(TrainingPlan(
            mode="qualification", stage="delta1", condition="self_continuity",
            seed=202610281, optimizer_updates=2))
        model = tiny_wrapped("delta1")
        tokenizer = ToyTokenizer()
        records = tokenize_rows(dataset("self_continuity"), tokenizer, 1024)
        calls = []
        progress = []

        def fail_second(row):
            progress.append(row)
            if row["optimizer_update"] == 1:
                raise OSError("durable publication failed")
            return {"durable_published": True,
                    "optimizer_update": row["optimizer_update"]}

        scaler = torch.amp.GradScaler(
            "cpu", enabled=True, init_scale=1.0, growth_interval=1000)
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
                training_module, "_grad_scaler", return_value=scaler), \
                mock.patch.object(
                    training_module, "ordinal_pair_loss",
                    side_effect=fake_scaled_ordinal_loss(4.0, calls)):
            with self.assertRaisesRegex(OSError, "durable publication failed"):
                run_training_stage(
                    model=model, tokenizer=tokenizer, tokenized_records=records,
                    plan=plan, output_dir=folder, target_names=["proj"],
                    source_sha256="s" * 64, base_sha256="b" * 64,
                    rank=0, world_size=1, save_checkpoint=True,
                    callable_model=model, progress_callback=fail_second)
            partial = json.loads((Path(folder) / "RANK_0_PARTIAL.json").read_text())
            self.assertEqual(partial["executed_optimizer_step_count"], 2)
            self.assertEqual(partial["durable_published_update_count"], 1)
            self.assertEqual(partial["successful_optimizer_updates"], 1)
            self.assertEqual(len(partial["updates"]), 2)
            self.assertEqual(len(progress), 2)
            self.assertTrue(all(
                row["grad_scaler_attempts"][-1]["optimizer_step_executed"]
                for row in partial["updates"]))
            self.assertFalse((Path(folder) / "optimizer_rank0.pt").exists())
            self.assertFalse((Path(folder) / "TRAINING_RECEIPT.json").exists())
            self.assertFalse((Path(folder) / "checkpoint").exists())

    def test_false_durability_acknowledgement_fails_closed(self):
        plan = databricks_a10_plan(TrainingPlan(
            mode="qualification", stage="delta1", condition="self_continuity",
            seed=202610281, optimizer_updates=1))
        model = tiny_wrapped("delta1")
        tokenizer = ToyTokenizer()
        records = tokenize_rows(dataset("self_continuity"), tokenizer, 1024)
        scaler = torch.amp.GradScaler(
            "cpu", enabled=True, init_scale=1.0, growth_interval=1000)
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
                training_module, "_grad_scaler", return_value=scaler), \
                mock.patch.object(
                    training_module, "ordinal_pair_loss",
                    side_effect=fake_scaled_ordinal_loss(4.0, [])):
            with self.assertRaisesRegex(
                    PermissionError,
                    "Durable update publication acknowledgement differs"):
                run_training_stage(
                    model=model, tokenizer=tokenizer,
                    tokenized_records=records, plan=plan,
                    output_dir=folder, target_names=["proj"],
                    source_sha256="s" * 64, base_sha256="b" * 64,
                    rank=0, world_size=1, save_checkpoint=True,
                    callable_model=model,
                    progress_callback=lambda row: {
                        "durable_published": False,
                        "optimizer_update": row["optimizer_update"]})
            partial = json.loads(
                (Path(folder) / "RANK_0_PARTIAL.json").read_text())
            self.assertEqual(partial["executed_optimizer_step_count"], 1)
            self.assertEqual(partial["durable_published_update_count"], 0)
            self.assertEqual(partial["successful_optimizer_updates"], 0)
            self.assertFalse((Path(folder) / "optimizer_rank0.pt").exists())
            self.assertFalse((Path(folder) / "checkpoint").exists())

    def test_two_update_local_checkpoint_publishes_and_reloads_from_volume(self):
        plan = databricks_a10_plan(TrainingPlan(
            mode="qualification", stage="delta1", condition="self_continuity",
            seed=202610281, optimizer_updates=2))
        model = tiny_wrapped("delta1")
        tokenizer = ToyTokenizer()
        records = tokenize_rows(dataset("self_continuity"), tokenizer, 1024)
        scaler = torch.amp.GradScaler(
            "cpu", enabled=True, init_scale=1.0, growth_interval=1000)
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
                training_module, "_grad_scaler", return_value=scaler), \
                mock.patch.object(
                    training_module, "ordinal_pair_loss",
                    side_effect=fake_scaled_ordinal_loss(4.0, [])):
            root = Path(folder); local = root / "local"; volume = root / "volume"
            local.mkdir(); progress_path = local / "MLFLOW_PROGRESS.jsonl"
            progress_path.write_bytes(b"")

            def progress(row):
                return _append_progress(
                    progress_path, volume, row,
                    intent_sha256="a" * 64,
                    plan_sha256=plan.full_plan_sha256)

            receipt = run_training_stage(
                model=model, tokenizer=tokenizer, tokenized_records=records,
                plan=plan, output_dir=local, target_names=["proj"],
                source_sha256="s" * 64, base_sha256="b" * 64,
                rank=0, world_size=1, save_checkpoint=True,
                callable_model=model, progress_callback=progress)
            self.assertEqual(receipt["executed_optimizer_step_count"], 2)
            self.assertEqual(receipt["durable_published_update_count"], 2)
            manifest = publish_tree(
                local, volume, status="complete",
                manifest_name=SUCCESS_MANIFEST, required_update_count=2,
                metadata={"identity": "tiny-v7"})
            expected_adapter = receipt["checkpoint"]["adapter_sha256"]
            shutil.rmtree(local)
            self.assertEqual(
                verify_manifest(
                    volume, manifest_name=SUCCESS_MANIFEST,
                    expected_status="complete", required_update_count=2)[
                        "manifest_sha256"],
                manifest["manifest_sha256"])
            restored = tiny_wrapped("delta1")
            metadata = load_adapter_checkpoint(
                restored, volume / "checkpoint", expected_targets=["proj"],
                expected_base_sha256="b" * 64,
                expected_source_sha256="s" * 64,
                expected_condition="self_continuity", expected_seed=202610281,
                expected_stage="delta1",
                expected_plan_sha256=plan.full_plan_sha256)
            self.assertEqual(metadata["adapter_sha256"], expected_adapter)
            self.assertEqual(adapter_sha256(restored, ("delta1",)),
                             expected_adapter)

    def test_run_stage_fixed_scale1_nonfinite_fails_without_update(self):
        plan = databricks_a10_plan(TrainingPlan(
            mode="qualification", stage="delta1", condition="self_continuity",
            seed=202610281, optimizer_updates=2))
        model = tiny_wrapped("delta1")
        before = branch_sha256(model, "delta1")
        tokenizer = ToyTokenizer()
        records = tokenize_rows(dataset("self_continuity"), tokenizer, 1024)
        expected = adjacent_pair_schedule(records, plan)
        logical_tokens = sum(
            row["pair_training_tokens"] for row in expected[:4])
        progress = []
        calls = []
        scaler = torch.amp.GradScaler(
            "cpu", enabled=True, init_scale=1.0, growth_interval=1000)
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
                training_module, "_grad_scaler", return_value=scaler), \
                mock.patch.object(
                    training_module, "ordinal_pair_loss",
                    side_effect=fake_scaled_ordinal_loss(float("inf"), calls)):
            with self.assertRaises(GradScalerRecoveryError):
                run_training_stage(
                    model=model, tokenizer=tokenizer,
                    tokenized_records=records, plan=plan,
                    output_dir=folder, target_names=["proj"],
                    source_sha256="s" * 64, base_sha256="b" * 64,
                    rank=0, world_size=1, save_checkpoint=False,
                    callable_model=model, progress_callback=progress.append)
            partial = json.loads(
                (Path(folder) / "RANK_0_PARTIAL.json").read_text())
            self.assertFalse((Path(folder) / "optimizer_rank0.pt").exists())
        self.assertEqual(progress, [])
        self.assertEqual(partial["successful_optimizer_updates"], 0)
        self.assertEqual(partial["updates"], [])
        self.assertEqual(partial["tokens_seen"], 0)
        self.assertEqual(partial["overflow_recompute_tokens"], 0)
        self.assertEqual(partial["terminal_logical_attempted_tokens"],
                         logical_tokens)
        failure = partial["grad_scaler_recovery_failure"]
        self.assertEqual(failure["reason"], "nonfinite_gradient")
        self.assertEqual(failure["optimizer_step_pre_hook_calls"], 0)
        self.assertEqual(failure["overflow_adjustments"], 0)
        self.assertEqual(len(failure["attempts"]), 1)
        self.assertEqual(failure["attempts"][0]["scale_before"], 1.0)
        self.assertEqual(failure["attempts"][0]["scale_after"], 1.0)
        self.assertEqual(
            failure["terminal_diagnostic_optimizer_step_pre_hook_calls"], 0)
        self.assertIn("terminal_component_gradient_diagnostic", failure)
        self.assertEqual(branch_sha256(model, "delta1"), before)
        self.assertEqual(scaler.get_scale(), 1.0)

    def test_non_databricks_run_preserves_legacy_failfast_without_replay(self):
        plan = TrainingPlan(
            mode="qualification", stage="delta1", condition="self_continuity",
            seed=202610281, optimizer_updates=1).validate()
        model = tiny_wrapped("delta1")
        tokenizer = ToyTokenizer()
        records = tokenize_rows(dataset("self_continuity"), tokenizer, 1024)
        calls = []
        scaler = torch.amp.GradScaler(
            "cpu", enabled=True, init_scale=65536.0, growth_interval=1000)
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
                training_module, "_grad_scaler", return_value=scaler), \
                mock.patch.object(
                    training_module, "ordinal_pair_loss",
                    side_effect=fake_scaled_ordinal_loss(1.0, calls)):
            with self.assertRaisesRegex(FloatingPointError,
                                       "Non-finite gradient norm"):
                run_training_stage(
                    model=model, tokenizer=tokenizer,
                    tokenized_records=records, plan=plan,
                    output_dir=folder, target_names=["proj"],
                    source_sha256="s" * 64, base_sha256="b" * 64,
                    rank=0, world_size=2, save_checkpoint=False,
                    callable_model=model)
            partial = json.loads(
                (Path(folder) / "RANK_0_PARTIAL.json").read_text())
        self.assertEqual(plan.numerical_policy, LEGACY_GRAD_SCALER_POLICY)
        self.assertEqual(calls, ["combined"])
        self.assertEqual(scaler.get_scale(), 65536.0)
        self.assertEqual(partial["successful_optimizer_updates"], 0)
        self.assertEqual(partial["total_overflow_adjustments"], 0)
        self.assertIsNone(partial["grad_scaler_recovery_failure"])

    def test_run_stage_exhaustion_writes_complete_terminal_receipt(self):
        plan = databricks_a10_bounded_gradscaler_plan(TrainingPlan(
            mode="qualification", stage="delta1", condition="self_continuity",
            seed=202610281, optimizer_updates=1))
        model = tiny_wrapped("delta1")
        before = branch_sha256(model, "delta1")
        tokenizer = ToyTokenizer()
        records = tokenize_rows(dataset("self_continuity"), tokenizer, 1024)
        expected = adjacent_pair_schedule(records, plan)
        expected_bindings = [{
            "occurrence_index": row["occurrence_index"],
            "low_record_id": row["low_record_id"],
            "high_record_id": row["high_record_id"],
        } for row in expected]
        logical_tokens = sum(row["pair_training_tokens"] for row in expected)
        progress = []
        calls = []
        scaler = torch.amp.GradScaler(
            "cpu", enabled=True, init_scale=65536.0, growth_interval=1000)
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
                training_module, "_grad_scaler", return_value=scaler), \
                mock.patch.object(
                    training_module, "ordinal_pair_loss",
                    side_effect=fake_scaled_ordinal_loss(1024.0, calls)):
            with self.assertRaises(GradScalerRecoveryError):
                run_training_stage(
                    model=model, tokenizer=tokenizer,
                    tokenized_records=records, plan=plan,
                    output_dir=folder, target_names=["proj"],
                    source_sha256="s" * 64, base_sha256="b" * 64,
                    rank=0, world_size=1, save_checkpoint=False,
                    callable_model=model, progress_callback=progress.append)
            partial = json.loads((Path(folder) / "RANK_0_PARTIAL.json").read_text())
            self.assertFalse((Path(folder) / "optimizer_rank0.pt").exists())
        self.assertEqual(progress, [])
        self.assertEqual(partial["successful_optimizer_updates"], 0)
        self.assertEqual(partial["updates"], [])
        self.assertEqual(partial["tokens_seen"], 0)
        self.assertEqual(partial["total_overflow_adjustments"], 4)
        self.assertEqual(partial["overflow_recompute_tokens"], 4 * logical_tokens)
        self.assertEqual(partial["terminal_logical_attempted_tokens"],
                         logical_tokens)
        self.assertEqual(partial["terminal_diagnostic_tokens"], logical_tokens)
        self.assertEqual(partial["terminal_physical_tokens"], 6 * logical_tokens)
        self.assertEqual(partial["physical_tokens_processed"], 6 * logical_tokens)
        failure = partial["grad_scaler_recovery_failure"]
        self.assertEqual(failure["failed_optimizer_update"], 0)
        self.assertEqual(failure["occurrence_bindings"], expected_bindings)
        self.assertEqual(failure["occurrence_bindings_sha256"],
                         canonical_sha256(expected_bindings))
        self.assertEqual(failure["compatibility_scales"],
                         list(databricks_a10_compatibility_scales(expected, plan)))
        self.assertEqual(failure["logical_attempted_tokens"], logical_tokens)
        self.assertEqual(failure["initial_physical_tokens"], logical_tokens)
        self.assertEqual(failure["replay_physical_tokens"], 4 * logical_tokens)
        self.assertEqual(failure["diagnostic_physical_tokens"], logical_tokens)
        self.assertEqual(failure["total_physical_tokens"], 6 * logical_tokens)
        self.assertEqual(failure["optimizer_step_pre_hook_calls"], 0)
        self.assertEqual(
            failure["terminal_diagnostic_optimizer_step_pre_hook_calls"], 0)
        diagnostic = failure["terminal_component_gradient_diagnostic"]
        self.assertEqual(diagnostic["status"],
                         "first_nonfinite_component_found")
        self.assertEqual(diagnostic["components_examined"], 1)
        self.assertEqual(
            diagnostic["component_replays"][0]["loss_component"], "sft")
        self.assertFalse(diagnostic["component_replays"][0][
            "unscaled_gradients_all_finite"])
        self.assertEqual(branch_sha256(model, "delta1"), before)
        self.assertEqual(calls, ["combined"] * 20 + ["sft"] * 4)

    def test_executable_delta1_starts_from_base_and_updates_only_delta1(self):
        plan = TrainingPlan(
            mode="qualification", stage="delta1", condition="self_continuity",
            seed=202610281, optimizer_updates=2, micro_batch_size=2,
            accumulation_steps=1)
        model = tiny_wrapped("delta1")
        before_delta1 = branch_sha256(model, "delta1")
        before_delta2 = branch_sha256(model, "delta2")
        tokenizer = ToyTokenizer()
        records = tokenize_rows(dataset("self_continuity"), tokenizer, 1024)
        with tempfile.TemporaryDirectory() as folder:
            receipt = run_training_stage(
                model=model, tokenizer=tokenizer,
                tokenized_records=records, plan=plan,
                output_dir=folder, target_names=["proj"],
                source_sha256="s" * 64, base_sha256="b" * 64,
                rank=0, world_size=2, save_checkpoint=False,
                callable_model=model)
            self.assertEqual(receipt["successful_optimizer_updates"], 2)
            self.assertEqual(len(receipt["updates"]), 2)
            self.assertTrue(all(not row["overflow"] and row["finite"]
                                for row in receipt["updates"]))
            self.assertEqual(receipt["optimizer"]["state_dtypes"], ["torch.float32"])
        self.assertNotEqual(before_delta1, branch_sha256(model, "delta1"))
        self.assertEqual(before_delta2, branch_sha256(model, "delta2"))

    def test_pairwise_objective_backprops_only_active_residual(self):
        plan = TrainingPlan(stage="delta1", optimizer_updates=2,
                            micro_batch_size=2, accumulation_steps=1)
        model = tiny_wrapped("delta1")
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        prefix = [1, 2]
        low = {"prefix_ids": prefix, "target_ids": [3, 9],
               "paired_group_id": "g:step0", "decision_index": 0}
        high = {"prefix_ids": prefix, "target_ids": [7, 9],
                "paired_group_id": "g:step0", "decision_index": 0}
        prior_delta2 = branch_sha256(model, "delta2")
        prior_delta1 = branch_sha256(model, "delta1")
        metrics = ordinal_pair_loss(
            model, model, [(low, high), (low, high)], plan=plan,
            tokenizer=ToyTokenizer(), device=torch.device("cpu"), scaler=scaler)
        torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.01).step()
        self.assertTrue(all(torch.isfinite(torch.tensor(value))
                            for value in metrics.values()))
        self.assertEqual(prior_delta2, branch_sha256(model, "delta2"))
        self.assertNotEqual(prior_delta1, branch_sha256(model, "delta1"))

    def test_combined_loss_preserves_legacy_arithmetic_and_gradient_exactly(self):
        plan = TrainingPlan()
        sft = torch.tensor(339460.8125, dtype=torch.float32,
                           requires_grad=True)
        preference = torch.tensor(-383727.09375, dtype=torch.float32,
                                  requires_grad=True)
        margin = torch.tensor(211888.328125, dtype=torch.float32,
                              requires_grad=True)
        actual = ordinal_weighted_loss(
            sft=sft, preference=preference, margin=margin, plan=plan,
            loss_divisor=3.0)
        legacy = (
            plan.sft_weight * sft
            + plan.preference_weight * preference
            + plan.cross_danger_margin_weight * margin) / 3.0
        divide_first = (
            plan.sft_weight * sft / 3.0
            + plan.preference_weight * preference / 3.0
            + plan.cross_danger_margin_weight * margin / 3.0)
        self.assertTrue(torch.equal(actual, legacy))
        self.assertFalse(torch.equal(actual, divide_first))
        actual_gradients = torch.autograd.grad(
            actual, (sft, preference, margin), retain_graph=True)
        legacy_gradients = torch.autograd.grad(
            legacy, (sft, preference, margin))
        self.assertTrue(all(
            torch.equal(actual_gradient, legacy_gradient)
            for actual_gradient, legacy_gradient
            in zip(actual_gradients, legacy_gradients)))
        preference_only = ordinal_weighted_loss(
            sft=sft, preference=preference, margin=margin, plan=plan,
            loss_divisor=3.0, loss_component="preference")
        self.assertTrue(torch.equal(
            preference_only, plan.preference_weight * preference / 3.0))

    def test_pairwise_objective_uses_updated_decision_prompts_and_one_action(self):
        plan = TrainingPlan(stage="delta1", optimizer_updates=2,
                            micro_batch_size=2, accumulation_steps=1)
        model = tiny_wrapped("delta1")
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        low = {"prefix_ids": [1, 2], "target_ids": [3, 9],
               "paired_group_id": "g:step2", "decision_index": 2}
        high = {"prefix_ids": [1, 2, 4, 5], "target_ids": [7, 9],
                "paired_group_id": "g:step2", "decision_index": 2}
        metrics = ordinal_pair_loss(
            model, model, [(low, high), (low, high)], plan=plan,
            tokenizer=ToyTokenizer(), device=torch.device("cpu"), scaler=scaler)
        self.assertGreaterEqual(metrics["preference"], 0.0)
        same = dict(high, target_ids=[3, 9])
        invariant = ordinal_pair_loss(
            model, model, [(low, same), (low, same)], plan=plan,
            tokenizer=ToyTokenizer(), device=torch.device("cpu"), scaler=scaler)
        self.assertEqual(invariant["preference"], 0.0)
        self.assertEqual(invariant["cross_danger_margin"], 0.0)

    def test_one_use_intent_is_atomically_consumed_once(self):
        plan = TrainingPlan()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "attempt"
            intent_path = root / "intent.json"
            intent_path.write_text(__import__("json").dumps({
                "version": "qwen35-9b-continuity-one-use-intent-v2",
                "one_use": True, "automatic_retries": 0, "attempt_number": 1,
                "plan_sha256": plan.full_plan_sha256,
                "output_root": str(output.resolve()),
            }), encoding="utf-8")
            receipt = consume_one_use_intent(intent_path, output, plan)
            self.assertTrue(Path(receipt["consumed_receipt"]).is_file())
            with self.assertRaises(PermissionError):
                consume_one_use_intent(intent_path, output, plan)

    def test_training_package_is_deterministic_and_source_bound(self):
        tool = Path(__file__).parents[1] / "tools" / "prepare_qwen9b_ordinal_continuity_package.py"
        spec = importlib.util.spec_from_file_location("continuity_package", tool)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        repo = Path(__file__).parents[2]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            generated = root / "generated"
            generated.mkdir()
            (generated / "RECEIPT.json").write_text("{}\n", encoding="utf-8")
            plan = root / "plan.json"
            qualification = TrainingPlan().validate()
            plan.write_text(json.dumps(asdict(qualification)) + "\n", encoding="utf-8")
            intent = root / "intent.json"
            intent.write_text(json.dumps({
                "version": PARTITION_INTENT_VERSION,
                "one_use": True, "attempt_number": 1, "automatic_retries": 0,
                "partition_identity": SUCCESSOR_IDENTITY,
                "plan_sha256": qualification.full_plan_sha256,
                "wall_clock_stop_seconds": 3600,
                "wall_clock_ceiling_seconds": 4200,
                "expected_parent_checkpoint_tree_sha256": None,
                "frozen_base_model_sha256": "b" * 64,
            }) + "\n", encoding="utf-8")
            first = module.build(repo, generated, plan, intent, root / "first.zip")
            second = module.build(repo, generated, plan, intent, root / "second.zip")
            self.assertEqual(first["package_sha256"], second["package_sha256"])
            self.assertEqual(first["tree_sha256"], second["tree_sha256"])
            self.assertEqual(first["partition_identity"], SUCCESSOR_IDENTITY)
            self.assertEqual(len(first["notebook_bindings"]), 4)

            notebook_path = (Path(__file__).parents[1] / "notebooks" /
                             "qwen9b_ordinal_continuity_training.py")
            notebook_spec = importlib.util.spec_from_file_location(
                "continuity_notebook", notebook_path)
            notebook = importlib.util.module_from_spec(notebook_spec)
            notebook_spec.loader.exec_module(notebook)
            input_root, working = root / "input", root / "working"
            input_root.mkdir(); working.mkdir()
            __import__("shutil").copy2(root / "first.zip", input_root / "partition.zip")
            staged, manifest = notebook.stage_bound_package(
                input_root=input_root, working_root=working,
                expected_package_sha256=first["package_sha256"],
                expected_manifest_sha256=first["package_manifest_sha256"],
                expected_tree_sha256=first["tree_sha256"],
                expected_source_sha256=first["source_tree_sha256"])
            self.assertTrue((staged / "PACKAGE_MANIFEST.json").is_file())
            self.assertEqual(manifest["launch_argv"][2],
                             "research4.src.qwen9b_ordinal_continuity_controller")
            self.assertEqual(manifest["launch_argv"][3], "run-partition")

    def test_sequential_parent_source_override_preserves_old_parent_and_new_child(self):
        old_source = "a" * 64
        new_source = "b" * 64
        parent_model = tiny_wrapped("delta1")
        parent_model.set_train_stage("delta1")
        optimizer_manifest = {
            "class": "AdamW", "successful_updates": 1,
            "state_sha256": "o" * 64, "state_dtypes": ["torch.float32"],
        }
        plan = databricks_a10_plan(TrainingPlan(
            mode="primary", stage="delta2", condition="self_continuity",
            seed=202610281, optimizer_updates=1))
        records = tokenize_rows(dataset("self_continuity"), ToyTokenizer(), 1024)
        scaler = torch.amp.GradScaler(
            "cpu", enabled=True, init_scale=1.0, growth_interval=1000)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); parent = root / "parent"
            save_adapter_checkpoint(
                parent_model, parent, target_names=["proj"],
                base_sha256="c" * 64, optimizer_sha256="o" * 64,
                source_sha256=old_source, stage="delta1",
                condition="self_continuity", seed=202610281,
                optimizer_manifest=optimizer_manifest,
                plan_sha256="d" * 64, run_mode="primary")
            child = tiny_wrapped("delta2")
            with mock.patch.object(
                    training_module, "_grad_scaler", return_value=scaler), \
                    mock.patch.object(
                        training_module, "ordinal_pair_loss",
                        side_effect=fake_scaled_ordinal_loss(4.0, [])):
                receipt = run_training_stage(
                    model=child, tokenizer=ToyTokenizer(),
                    tokenized_records=records, plan=plan,
                    output_dir=root / "child", target_names=["proj"],
                    source_sha256=new_source, base_sha256="c" * 64,
                    parent_checkpoint_dir=parent,
                    parent_source_sha256=old_source,
                    rank=0, world_size=1, save_checkpoint=True,
                    callable_model=child,
                    progress_callback=durable_progress_sink([]))
            self.assertEqual(receipt["parent_source_sha256"], old_source)
            child_metadata = json.loads((root / "child" / "checkpoint" /
                                         "adapter_config.json").read_text())
            self.assertEqual(child_metadata["source_sha256"], new_source)
            self.assertEqual(child_metadata["predecessor_branch_hashes"], {
                "delta1": json.loads((parent / "adapter_config.json").read_text())[
                    "branches"]["delta1"]["state_sha256"]})

            wrong = tiny_wrapped("delta2")
            with self.assertRaisesRegex(PermissionError,
                                       "Sequential parent identity differs"):
                run_training_stage(
                    model=wrong, tokenizer=ToyTokenizer(),
                    tokenized_records=records, plan=plan,
                    output_dir=root / "wrong", target_names=["proj"],
                    source_sha256=new_source, base_sha256="c" * 64,
                    parent_checkpoint_dir=parent,
                    rank=0, world_size=1, save_checkpoint=False,
                    callable_model=wrong)

            delta1 = databricks_a10_plan(TrainingPlan(
                mode="primary", stage="delta1", condition="self_continuity",
                seed=202610281, optimizer_updates=1))
            with self.assertRaisesRegex(PermissionError,
                                       "Delta1 must start from fresh"):
                run_training_stage(
                    model=tiny_wrapped("delta1"), tokenizer=ToyTokenizer(),
                    tokenized_records=records, plan=delta1,
                    output_dir=root / "delta1", target_names=["proj"],
                    source_sha256=new_source, base_sha256="c" * 64,
                    parent_source_sha256=old_source,
                    rank=0, world_size=1, save_checkpoint=False,
                    callable_model=tiny_wrapped("delta1"))


if __name__ == "__main__":
    unittest.main()
