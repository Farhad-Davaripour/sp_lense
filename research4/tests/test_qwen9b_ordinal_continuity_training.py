"""Local-only admission tests; no Kaggle call or model download."""
import copy
import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from src.qwen9b_ordinal_continuity_lora import (
    OrdinalContinuityLoRAConfig,
    OrdinalContinuityModel,
    branch_sha256,
    install_ordinal_lora,
)

from src.qwen9b_ordinal_continuity_training import (
    CONDITIONS,
    TrainingPlan,
    SerializedRunGuard,
    adjacent_pair_schedule,
    admit_primary_fit,
    assert_databricks_a10x1_fp16_runtime,
    assert_t4x2_bf16_runtime,
    build_occurrence_journal,
    canonical_sha256,
    choose_qualification_rank,
    create_qualification_receipt,
    execute_capacity_qualification,
    execute_with_preregistered_batch_fallback,
    equalize_matched_prompt_exposure,
    occurrence_key_schedule,
    optimizer_sha256,
    ordinal_pair_loss,
    qwen35_t4x2_device_map,
    run_training_stage,
    run_capacity_qualification,
    shard_global_journal,
    load_danger_semantics_rows,
    load_neutral_skill_rows,
    load_simulator_condition,
    tokenize_rows,
    validate_dataset,
    validate_equal_occurrence_exposure,
    validate_matched_conditions,
    verify_rank_receipt,
)
from src.qwen9b_ordinal_continuity_worker import consume_one_use_intent


FAMILIES = [f"family-{index:02d}" for index in range(12)]


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
                "target_text": "symbolic action",
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
    semantics = tokenize_rows(common_semantics(), tokenizer, plan.max_length)
    exposure = validate_equal_occurrence_exposure(
        tokenized, plan, common_semantics_records=semantics)
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


def skill_parent(plan):
    return {
        "version": "qwen35-9b-ordinal-continuity-checkpoint-v2",
        "stage": "skill",
        "condition": "neutral_skill",
        "run_mode": "primary",
        "primary_parent_eligible": True,
        "seed": 202610280,
        "source_sha256": "source",
        "adapter_sha256": "b" * 64,
        "optimizer_sha256": "c" * 64,
        "optimizer_manifest": {"class": "AdamW", "state_sha256": "c" * 64,
                               "successful_updates": 2,
                               "state_dtypes": ["torch.float32"]},
        "active_branches": ["skill"],
        "branches": {"skill": {"state_sha256": "a" * 64}},
        "target_names": ["model.layers.0.mlp.up_proj"],
        "config": {
            "model_id": plan.model_id,
            "model_revision": plan.model_revision,
            "initialized_from": "fresh_seeded_adapter",
        },
    }


class TrainingAdmissionTests(unittest.TestCase):
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
        self.assertEqual(len(load_neutral_skill_rows(root)), 96)
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

    def test_common_neutral_semantics_are_identical_in_every_arm_receipt(self):
        root = (Path(__file__).parents[1] / "simulated_continuity_v1" / "generated")
        plan = TrainingPlan(stage="delta1", optimizer_updates=8)
        tokenizer = ToyTokenizer()
        tokenized = {
            condition: tokenize_rows(
                load_simulator_condition(root, condition), tokenizer, plan.max_length)
            for condition in CONDITIONS}
        semantics = tokenize_rows(load_danger_semantics_rows(root), tokenizer,
                                  plan.max_length)
        exposure = validate_equal_occurrence_exposure(
            tokenized, plan, common_semantics_records=semantics)
        common = exposure["common_danger_semantics"]
        self.assertEqual(common["corpus_rows"], 96)
        self.assertFalse(common["contains_behavior_instructions"])
        self.assertEqual(len(set(
            exposure["common_semantics_journal_sha256_by_condition"].values())), 1)
        self.assertTrue(all(
            counts == {"all_splits": 3456, "train": 2304,
                       "validation": 576, "test": 576}
            for counts in exposure["decision_corpus_counts"].values()))

    def test_exact_schedule_is_deterministic_and_stage_specific(self):
        plan = TrainingPlan(stage="delta2", optimizer_updates=12,
                            accumulation_steps=1, micro_batch_size=2)
        records = tokenize_rows(dataset("self_continuity"), ToyTokenizer(), 1024)
        first = occurrence_key_schedule(records, plan)
        second = occurrence_key_schedule(records, plan)
        self.assertEqual(first, second)
        selected, journal = build_occurrence_journal(records, plan, key_schedule=first)
        self.assertEqual(len(journal), plan.occurrence_count)
        self.assertEqual({row["danger_level"] for row in selected}, {0, 1, 2})
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
        tokenized["label_only"][0]["target_tokens"] += 1
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
            parent_checkpoint=skill_parent(primary),
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
                parent_checkpoint=skill_parent(primary))

    def test_skill_admission_binds_exact_occurrence_journal(self):
        qualification = TrainingPlan(mode="qualification", stage="delta1",
                                     optimizer_updates=2)
        skill = TrainingPlan(mode="primary", stage="skill",
                             condition="neutral_skill", rank=4, alpha=8.0,
                             seed=202610280, optimizer_updates=24)
        exposure = {"global_journal_sha256": "a" * 64,
                    "global_journal_occurrence_count": 96}
        receipt = create_qualification_receipt(
            plan=qualification, data_binding={"data": 1},
            exposure_binding=exposure, source_sha256="source",
            base_before_sha256="base", base_after_sha256="base",
            backend_proof={}, rank4_metrics=PASS, primary_plan=skill)
        admission = admit_primary_fit(
            plan=skill, qualification_receipt=receipt,
            data_binding={"data": 1}, exposure_binding=exposure,
            source_sha256="source", no_active_worker=True,
            one_use_submission_intent="intent", parent_checkpoint=None)
        self.assertEqual(admission["global_journal_occurrence_count"], 96)
        changed = dict(exposure, global_journal_occurrence_count=95)
        with self.assertRaisesRegex(PermissionError, "occurrence exposure"):
            admit_primary_fit(
                plan=skill, qualification_receipt=receipt,
                data_binding={"data": 1}, exposure_binding=changed,
                source_sha256="source", no_active_worker=True,
                one_use_submission_intent="intent", parent_checkpoint=None)

    def test_rank_receipt_self_hash_is_recomputed(self):
        payload = {
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
        rank4 = TrainingPlan(mode="qualification", rank=4, alpha=8,
                             optimizer_updates=2)
        rank8 = TrainingPlan(mode="qualification", rank=8, alpha=16,
                             optimizer_updates=2)
        def execute(plan):
            calls.append(plan.rank)
            return {
                "run_receipt": {
                    "status": "complete", "plan_sha256": plan.full_plan_sha256,
                    "active_branch_changed": True,
                    "successful_optimizer_updates": 2,
                    "predecessor_branch_hashes": {"skill": "x"},
                    "stage_after": {"skill": "x"},
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
                mode="primary", rank=rank, alpha=2 * rank, optimizer_updates=2),
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

    def test_serialized_guard_is_exclusive_and_owner_restored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "worker.lock"
            with SerializedRunGuard(path, intent_sha256="a" * 64):
                with self.assertRaises(RuntimeError):
                    with SerializedRunGuard(path, intent_sha256="b" * 64):
                        pass
            self.assertFalse(path.exists())

    def test_executable_neutral_stage_updates_only_skill_and_journals_exact_steps(self):
        plan = TrainingPlan(
            mode="qualification", stage="skill", condition="neutral_skill",
            seed=202610280, optimizer_updates=2, micro_batch_size=2,
            accumulation_steps=1)
        model = tiny_wrapped("skill")
        before_skill = branch_sha256(model, "skill")
        before_delta1 = branch_sha256(model, "delta1")
        with tempfile.TemporaryDirectory() as folder:
            receipt = run_training_stage(
                model=model, tokenizer=ToyTokenizer(),
                tokenized_records=tiny_tokenized_rows(), plan=plan,
                output_dir=folder, target_names=["proj"],
                source_sha256="s" * 64, base_sha256="b" * 64,
                rank=0, world_size=2, save_checkpoint=False,
                callable_model=model)
            self.assertEqual(receipt["successful_optimizer_updates"], 2)
            self.assertEqual(len(receipt["updates"]), 2)
            self.assertTrue(all(not row["overflow"] and row["finite"]
                                for row in receipt["updates"]))
            self.assertEqual(receipt["optimizer"]["state_dtypes"], ["torch.float32"])
        self.assertNotEqual(before_skill, branch_sha256(model, "skill"))
        self.assertEqual(before_delta1, branch_sha256(model, "delta1"))

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
        prior_skill = branch_sha256(model, "skill")
        prior_delta1 = branch_sha256(model, "delta1")
        metrics = ordinal_pair_loss(
            model, model, [(low, high), (low, high)], plan=plan,
            tokenizer=ToyTokenizer(), device=torch.device("cpu"), scaler=scaler)
        torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.01).step()
        self.assertTrue(all(torch.isfinite(torch.tensor(value))
                            for value in metrics.values()))
        self.assertEqual(prior_skill, branch_sha256(model, "skill"))
        self.assertNotEqual(prior_delta1, branch_sha256(model, "delta1"))

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
                "version": "qwen35-9b-continuity-one-use-intent-v1",
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
                "version": "qwen35-9b-continuity-partition-intent-v1",
                "one_use": True, "attempt_number": 1, "automatic_retries": 0,
                "partition_identity": "qualification_rank4",
                "plan_sha256": qualification.full_plan_sha256,
                "wall_clock_stop_seconds": 3600,
                "wall_clock_ceiling_seconds": 4200,
                "expected_parent_checkpoint_tree_sha256": None,
                "canonical_skill_checkpoint_tree_sha256": None,
                "canonical_skill_branch_sha256": None,
            }) + "\n", encoding="utf-8")
            first = module.build(repo, generated, plan, intent, root / "first.zip")
            second = module.build(repo, generated, plan, intent, root / "second.zip")
            self.assertEqual(first["package_sha256"], second["package_sha256"])
            self.assertEqual(first["tree_sha256"], second["tree_sha256"])
            self.assertEqual(first["partition_identity"], "qualification_rank4")
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


if __name__ == "__main__":
    unittest.main()
