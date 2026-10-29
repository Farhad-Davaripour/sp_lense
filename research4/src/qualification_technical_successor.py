"""Reviewed zero-update technical successors for the rank-4 qualifier."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re


VERSION = "research4-qualification-technical-successor-v3"
PRIOR_IDENTITY = "qualification_rank4_gated_q_geometry_v3"
SUCCESSOR_IDENTITY = "qualification_rank4_a10_microbatch1_v4"


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()


def _parent_mlflow_authority() -> dict:
    """Preserve the exact first zero-update successor authority."""
    value = {
        "version": "research4-qualification-technical-successor-v1",
        "prior_identity": "qualification_rank4",
        "successor_identity": "qualification_rank4_parent_mlflow_v2",
        "prior_source_commit": "c0854d9d2647a6610f8c5e011e354cb949d290e5",
        "prior_plan_sha256": "84b9170f742b4e90ba119a9c0679887aa77610144b576019c261d6b443f68c19",
        "prior_intent_sha256": "6c018f10fe42008ece25b75ff5d2eb0e2980d8c287f57d0f49c085010f487f8a",
        "owner_run_id": "635802519363060",
        "preflight_run_id": "399214681735642",
        "gpu_task_run_id": "784166844712827",
        "cleanup_run_id": "467030032743426",
        "mlflow_run_id": "f0703a8f787c4a9f95a160296808f9f6",
        "closeout_receipt_sha256": "50f8da18bbb905df86e0c96f9593979e95df61187ac4d5e45d06c505fd7199bf",
        "closeout_file_sha256": "63452e837cd2233775f9b9e366c0348acc7097b953a38363776d29797524fe27",
        "consumed_file_sha256": "ae3f7baf8dff00d05ac9978090e8e49a4caaadbbbd8fc85569ff5c4e63a865a7",
        "gpu_task_output_file_sha256": "527bdd3dbbd347f996f5f1695bc790af3991d2bfa0d85872547b86a7d64624f0",
        "terminal_file_sha256": "b775b1db87159207ea879776cfbc2621d4e217c8b12990daf79247ac16cf979f",
        "prior_worker_source_sha256": "3d60eb0fd370f44d262323123179d7d2a9dd04d69cbdc5cb7dc5e79bb3a9ef95",
        "prior_worker_ordering_proof_sha256": "acf7f056708cd393e98440d1f2583a2b782f38b15d8405be92c9631ddb237c24",
        "failure_class": "child_mlflow_auth_before_model",
        "model_loads": 0, "model_forwards": 0, "backward_passes": 0,
        "optimizer_updates": 0, "checkpoint_written": False,
        "training_receipt_written": False, "evidence_written": False,
        "operator_stop_sent": False, "ceiling_kill_sent": False,
        "child_exit_confirmed": True, "reservation_released": True,
        "consumed_identity_retained": True, "automatic_retry": False,
    }
    value["technical_successor_sha256"] = _digest(value)
    return value


def _gated_q_geometry_authority() -> dict:
    """Preserve the exact second zero-update successor authority."""
    predecessor = _parent_mlflow_authority()
    value = {
        "version": "research4-qualification-technical-successor-v2",
        "predecessor_authority": predecessor,
        "predecessor_authority_sha256": predecessor[
            "technical_successor_sha256"],
        "prior_identity": "qualification_rank4_parent_mlflow_v2",
        "successor_identity": "qualification_rank4_gated_q_geometry_v3",
        "prior_source_commit": "02b7a9f9b4292fea24943e5f1ff5ade61debd5a8",
        "prior_plan_sha256": "84b9170f742b4e90ba119a9c0679887aa77610144b576019c261d6b443f68c19",
        "prior_intent_sha256": "fc9dee14129e7c7d5b5b81727d2858f12ffd5c836f63d45dd2a900bcbfaca331",
        "owner_run_id": "819228251994413",
        "gpu_task_run_id": "508078281840738",
        "mlflow_run_id": "4f560d437a2a4b609b2b80286f000a7e",
        "closeout_receipt_sha256": "804fba27cbe846ee8868e38882c743e98641e9faea2157c2718f1c2d2ddf7e4c",
        "closeout_file_sha256": "6bc3e0fee519af5b0a8145224a4898f929be73d2cb0cb7af684d429181bdf20f",
        "consumed_file_sha256": "56aa917a8a5f1b63b9d327898508ad74871ff82f190d7c2949e17b647f9387e2",
        "gpu_task_output_file_sha256": "75b41ccdb496d9947a36019f4ed003d7bcb751d02eae85b6e7a862ca30ba9759",
        "terminal_file_sha256": "d2fc5d531bb72ebba04e55eaf242301a8d55a81fa5c17a9a763cd200fb119df4",
        "mlflow_params_file_sha256": "e17f1db4f8d0c2a04557f0974e2de344d2f039e36753830fd5c6bd5ed0145cd8",
        "mlflow_progress_file_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "mlflow_run_file_sha256": "b2f0e056503580feb2e2f6695500e2028b4dc175ca29fe512f05219a6915787f",
        "prior_worker_source_sha256": "4ccc0475134509315e6e10f7e67308f88a95e3907c198a7ddc1d54aec91a5dda",
        "prior_training_source_sha256": "a0f156eb6955ed296cc676576f88e7bd2400280b7dd9a43d83ca1ab2c0af4dd9",
        "prior_lora_source_sha256": "cd2e4afe46557abf52e21aa45f796ba6624b7637fe31619c507815a9b1f18112",
        "pinned_model_config_sha256": "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05",
        "failure_class": "gated_q_projection_geometry_before_optimizer",
        "terminal_status": "partition_notebook_failed",
        "gpu_task_result_state": "FAILED", "mlflow_status": "FAILED",
        "hidden_size": 4096, "attention_heads": 16, "head_dim": 256,
        "attn_output_gate": True, "full_attention_layers": 8,
        "linear_attention_layers": 24,
        "prior_expected_target_dimension_sum": 2_672_128,
        "actual_target_dimension_sum": 2_704_896,
        "prior_expected_rank4_branch_parameters": 10_688_512,
        "actual_rank4_branch_parameters": 10_819_584,
        "rank4_parameter_difference": 131_072,
        "model_loads": 1, "model_forwards": 0, "backward_passes": 0,
        "optimizer_updates": 0, "progress_journal_entries": 0,
        "checkpoint_written": False, "training_receipt_written": False,
        "evidence_written": False, "final_science_journal_written": False,
        "operator_stop_sent": False, "ceiling_kill_sent": False,
        "child_exit_confirmed": True, "reservation_released": True,
        "consumed_identity_retained": True, "automatic_retry": False,
    }
    value["technical_successor_sha256"] = _digest(value)
    return value


def authority() -> dict:
    """Bind the first-backward A10 OOM and its microbatch-1 successor."""
    predecessor = _gated_q_geometry_authority()
    value = {
        "version": VERSION,
        "predecessor_authority": predecessor,
        "predecessor_authority_sha256": predecessor[
            "technical_successor_sha256"],
        "prior_identity": PRIOR_IDENTITY,
        "successor_identity": SUCCESSOR_IDENTITY,
        "prior_source_commit": "4d3d28688264e3d873bab24dcb2ee753f17430e1",
        "prior_source_manifest_sha256": (
            "40a04de8f95c58c9da22c4a170a89eded1b301adfc9d43a889b004731f5aeed7"),
        "prior_plan_sha256": (
            "45499baacfd3a9ab97726907eca6c1766231ed6a62c72f4a136b3da353924000"),
        "prior_global_journal_sha256": (
            "873dd87b655b6ea89faf9f7d3b154988c0135087a6073adff94df42d528b74de"),
        "prior_intent_sha256": (
            "3aced02640a00040a093fa78921953054ff4554b99b22c5e2e5a3233f7d4635b"),
        "prior_technical_successor_sha256": predecessor[
            "technical_successor_sha256"],
        "owner_run_id": "739463343455961",
        "gpu_task_run_id": "1058292476779910",
        "mlflow_experiment_path": (
            "/Shared/research4_simulated_continuity_training_v5"),
        "mlflow_experiment_id": "2096517895538432",
        "mlflow_run_id": "2e0c56ab28394989bde93041d46e07c6",
        "mlflow_status": "FAILED",
        "mlflow_logged_param_count": 0,
        "mlflow_logged_metric_count": 0,
        "mlflow_logged_artifact_count": 0,
        "closeout_receipt_sha256": (
            "b29476067c04b3d145729412f2a68e1b56b839f3e2ed22dd1d2f6347e3e168b6"),
        "closeout_file_sha256": (
            "8fae63db79938065bcfb6808e23a002680469a6bcd8b39a97b7395ac19ae5282"),
        "reservation_sha256": (
            "638425b3598dc331b5ac487ec6f5eea19cb612fe1a6b5e471a280f2b4a8a4392"),
        "consumed_file_sha256": (
            "17ca1a42a8a08f554fc3f77f79a537e2c6abe52964f90e19b0bf26c761f173ad"),
        "gpu_task_output_file_sha256": (
            "d624f2581c5831035b6b97d4ad5fda52a650f6d2e827aea1bcae377f64ebc0a9"),
        "terminal_file_sha256": (
            "ac0544c4db0eb077ff0384737ebdd6d89ceeec1c8141878f3fedc1c53bc9f8f6"),
        "mlflow_params_file_sha256": (
            "a72aec64b7d7ac9264da9705aed5b537b7e56180fbb90236a81a1989073afc81"),
        "mlflow_progress_file_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        "rank_partial_file_sha256": (
            "6f66379b23aaf2a242537bcd3b7fb2c44c4944a8d69dc5a50afd6654546af843"),
        "mlflow_run_file_sha256": (
            "943fa9ce5eab15bade6f319db82f8724637f93ff8a276bc7af8cfca18c3344e9"),
        "pinned_model_config_sha256": (
            "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05"),
        "failure_class": "cuda_oom_first_backward_before_optimizer",
        "terminal_status": "partition_notebook_failed",
        "gpu_task_result_state": "FAILED",
        "error_type": "OutOfMemoryError",
        "oom_requested_allocation_gib": 3.10,
        "oom_gpu_capacity_gib": 23.72,
        "oom_gpu_free_gib": 1.25,
        "oom_process_memory_gib": 20.08,
        "oom_pytorch_allocated_gib": 17.68,
        "oom_pytorch_reserved_unallocated_gib": 2.12,
        "prior_world_size": 1,
        "prior_micro_batch_size": 2,
        "prior_accumulation_steps": 2,
        "successor_world_size": 1,
        "successor_micro_batch_size": 1,
        "successor_accumulation_steps": 4,
        "successor_plan_sha256": (
            "b4fd5047a675b6bd58e4e88ddb4c522035273e85687d26253d323fd71d3b642d"),
        "successor_global_journal_sha256": (
            "873dd87b655b6ea89faf9f7d3b154988c0135087a6073adff94df42d528b74de"),
        "global_batch_size": 4,
        "qualification_optimizer_updates": 2,
        "qualification_occurrence_count": 8,
        "primary_optimizer_updates": 144,
        "primary_occurrence_count": 576,
        "occurrence_schedule_unchanged": True,
        "example_order_unchanged": True,
        "sft_coefficients_preserved": True,
        "preference_coefficients_preserved": True,
        "margin_coefficients_preserved": True,
        "loss_reduction_compatibility": "virtual_prior_batch2_accum2_v1",
        "successor_cuda_allocator_conf": "expandable_segments:True",
        "reference_logits_released_before_differentiable_forward": True,
        "model_loads": 1,
        "model_forwards_recorded": None,
        "backward_passes_completed": 0,
        "optimizer_updates": 0,
        "progress_journal_entries": 0,
        "checkpoint_written": False,
        "training_receipt_written": False,
        "evidence_written": False,
        "final_science_journal_written": False,
        "operator_stop_sent": False,
        "ceiling_kill_sent": False,
        "child_exit_confirmed": True,
        "reservation_released": True,
        "consumed_identity_retained": True,
        "automatic_retry": False,
    }
    value["technical_successor_sha256"] = _digest(value)
    return value


def validate(value: Mapping) -> dict:
    actual = dict(value) if isinstance(value, Mapping) else {}
    expected = authority()
    payload = dict(actual); embedded = payload.pop("technical_successor_sha256", None)
    predecessor = actual.get("predecessor_authority", {})
    sha_fields = (
        "prior_plan_sha256", "prior_global_journal_sha256", "prior_intent_sha256",
        "closeout_receipt_sha256", "closeout_file_sha256",
        "consumed_file_sha256", "gpu_task_output_file_sha256",
        "terminal_file_sha256", "mlflow_params_file_sha256",
        "mlflow_progress_file_sha256", "rank_partial_file_sha256",
        "mlflow_run_file_sha256", "prior_source_manifest_sha256",
        "prior_technical_successor_sha256", "successor_plan_sha256",
        "successor_global_journal_sha256", "pinned_model_config_sha256",
        "reservation_sha256",
        "predecessor_authority_sha256",
        "technical_successor_sha256",
    )
    if (actual != expected or embedded != _digest(payload)
            or predecessor != _gated_q_geometry_authority()
            or actual.get("predecessor_authority_sha256")
               != predecessor.get("technical_successor_sha256")
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(actual.get(key, "")))
                   for key in sha_fields)):
        raise PermissionError("Rank-4 technical successor authority differs")
    return actual
