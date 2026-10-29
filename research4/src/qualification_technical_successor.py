"""Reviewed zero-update technical successors for the rank-4 qualifier."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re


VERSION = "research4-qualification-technical-successor-v5"
PRIOR_IDENTITY = "qualification_rank4_a10_gradscaler_recovery_v5"
SUCCESSOR_IDENTITY = "qualification_rank4_a10_fp16_scale1_v6"


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


def _a10_microbatch1_authority() -> dict:
    """Bind the first-backward A10 OOM and its microbatch-1 successor."""
    predecessor = _gated_q_geometry_authority()
    value = {
        "version": "research4-qualification-technical-successor-v3",
        "predecessor_authority": predecessor,
        "predecessor_authority_sha256": predecessor[
            "technical_successor_sha256"],
        "prior_identity": "qualification_rank4_gated_q_geometry_v3",
        "successor_identity": "qualification_rank4_a10_microbatch1_v4",
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


def _gradscaler_recovery_authority() -> dict:
    """Bind the zero-update v4 numerical failure and approved A10 recovery."""
    predecessor = _a10_microbatch1_authority()
    value = {
        "version": "research4-qualification-technical-successor-v4",
        "predecessor_authority": predecessor,
        "predecessor_authority_sha256": predecessor[
            "technical_successor_sha256"],
        "prior_identity": "qualification_rank4_a10_microbatch1_v4",
        "successor_identity": "qualification_rank4_a10_gradscaler_recovery_v5",
        "prior_source_commit": "f3b94233b937972fc3fac2a5561503effdd3ffb3",
        "prior_source_manifest_sha256": (
            "b69bc34caee9687821dd85ef3fca59844e832ac43e13954723397a4cdc7507cd"),
        "prior_plan_sha256": (
            "b4fd5047a675b6bd58e4e88ddb4c522035273e85687d26253d323fd71d3b642d"),
        "prior_global_journal_sha256": (
            "873dd87b655b6ea89faf9f7d3b154988c0135087a6073adff94df42d528b74de"),
        "prior_intent_sha256": (
            "65e06dab4f87babe007df88beb0801a3073f6e8aa7b5c87cec1d73053565d10b"),
        "prior_technical_successor_sha256": predecessor[
            "technical_successor_sha256"],
        "owner_run_id": "123598135063131",
        "preflight_run_id": "89547510471888",
        "gpu_task_run_id": "988729344316773",
        "finalizer_run_id": "242025569120296",
        "release_run_id": "882639209064491",
        "mlflow_experiment_path": (
            "/Shared/research4_simulated_continuity_training_v5"),
        "mlflow_experiment_id": "2096517895538432",
        "mlflow_run_id": "b3778b168b644653a5ee4bdabd37a2e8",
        "mlflow_status": "FAILED",
        "mlflow_logged_param_count": 0,
        "mlflow_logged_metric_count": 0,
        "mlflow_logged_artifact_count": 0,
        "closeout_receipt_sha256": (
            "42075812a67c09b730480ad4d4a0b431bd4be6a88b555676f1cf09302b765bb1"),
        "closeout_file_sha256": (
            "70d10941340b37ee80aaca3e90c546cb6fd0325c586198f7b19b71f0ebbd0eb3"),
        "reservation_sha256": (
            "67ca4dcdd2485a57b495de720c810918602479ffae2461ff8927bec0ac7a5953"),
        "consumed_file_sha256": (
            "5d31dfca756f271109fc562321bd7c61faab0695efaadc3c10927ddd4af90cf9"),
        "terminal_file_sha256": (
            "8459464740d52fb1d1cfd1e1ff9079d5ab58a22ecd681ec9075dcc19d7b992f1"),
        "mlflow_params_file_sha256": (
            "dca316823f653015417768e79876cd357f1cf6623a252ff946e7fe2d413ea686"),
        "mlflow_progress_file_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        "rank_partial_file_sha256": (
            "a698689b80da2d3a673364d38665afdc499e68f86328f6f55fe2430747cba71a"),
        "independent_failure_audit_sha256": (
            "006d66ecf09d681b45b9a88c0d90b70d1bdb31714706a0e8b8730a5489a8d6b4"),
        "approved_proposal_sha256": (
            "ae2fdd048d1ffd5e5ef6402104e93a8d3fdbb5b4d5c802c918b1b7f9809f1bbc"),
        "approved_draft_patch_sha256": (
            "7d90a24d5d86305f3140c317e5ee751b9495881d5ea00f4c3a8b46bb6e8e0fbe"),
        "human_approval_file_sha256": (
            "9f8832e3bdaa10a348fcd8bf0f65b85984b3d054c80417411d7e280a3b180c3f"),
        "prior_training_source_sha256": (
            "91f2d8689cb4d4f97fa392a05d508ef17043f6e147b54a24d54b0d21f37eee7d"),
        "prior_worker_source_sha256": (
            "087fc9fd1969efe04000d6a550e91f12986e7eddaefb42db3d8b97fba714480a"),
        "prior_databricks_worker_source_sha256": (
            "17b8d3a8f4aff2661e57b761f874efe1897578f4d3ea115cb499a1d48d7021f0"),
        "prior_technical_successor_source_sha256": (
            "2cc32ecdd56fb911387c1ad1a2533407ba960c2689c6e8e7b8ddd3dac498b373"),
        "failure_class": "nonfinite_gradient_before_first_optimizer_step",
        "terminal_status": "partition_notebook_failed",
        "gpu_task_result_state": "FAILED",
        "error_type": "FloatingPointError",
        "error": "Non-finite gradient norm",
        "backward_calls_completed": 4,
        "model_forwards_recorded": None,
        "actual_initial_grad_scaler_scale_recorded": None,
        "initial_scale_overflow_hypothesis": "leading_not_proven",
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
        "successor_execution_scope": "databricks-a10x1-only",
        "successor_numerical_policy": (
            "databricks-a10-bounded-gradscaler-v1"),
        "successor_precision": "fp16",
        "successor_quantization": "nf4-double",
        "successor_world_size": 1,
        "successor_micro_batch_size": 1,
        "successor_accumulation_steps": 4,
        "successor_global_batch_size": 4,
        "successor_grad_scaler_initial_scale": 65536.0,
        "successor_grad_scaler_backoff_factor": 0.5,
        "successor_max_overflow_adjustments": 4,
        "successor_minimum_grad_scaler_scale": 4096.0,
        "successor_exact_rng_and_data_replay": True,
        "successor_optimizer_steps_on_overflow": 0,
        "successor_component_diagnostics": (
            ["sft", "preference", "cross_danger_margin"]),
        "successor_plan_sha256": (
            "dc1593bfcf607d3b72735f803109ad5e961ace9fb3d92c4e22312c44e7c5b021"),
        "successor_primary_plan_hashes_digest": (
            "15902a97fc00658e1c76408a5fa1f4b624039982800dfbe0175cdb794fe8003e"),
        "legacy_non_databricks_failfast_preserved": True,
        "model_data_order_objective_unchanged": True,
        "loss_coefficients_unchanged": True,
        "learning_rate_clipping_geometry_seeds_thresholds_unchanged": True,
        "legacy_combined_loss_arithmetic_preserved": True,
    }
    value["technical_successor_sha256"] = _digest(value)
    return value


def authority() -> dict:
    """Bind the closed v5 failure and approved fixed FP16 scale-1 successor."""
    predecessor = _gradscaler_recovery_authority()
    value = {
        "version": VERSION,
        "predecessor_authority": predecessor,
        "predecessor_authority_sha256": predecessor[
            "technical_successor_sha256"],
        "prior_identity": PRIOR_IDENTITY,
        "successor_identity": SUCCESSOR_IDENTITY,
        "prior_source_commit": "6aebf090bf48634c1e547044f33d317d7e29a6d1",
        "prior_source_manifest_sha256": (
            "4c42629187e50c67fa1ed809de40e96e45ea5f2454de36dfce61634071cf6a13"),
        "prior_plan_sha256": (
            "dc1593bfcf607d3b72735f803109ad5e961ace9fb3d92c4e22312c44e7c5b021"),
        "prior_global_journal_sha256": (
            "873dd87b655b6ea89faf9f7d3b154988c0135087a6073adff94df42d528b74de"),
        "prior_intent_sha256": (
            "b8f1eb980e1eba924540642e0de70f104baadfb0a5f0f9716c4ea25ac43769a1"),
        "prior_technical_successor_sha256": predecessor[
            "technical_successor_sha256"],
        "owner_run_id": "382017553734680",
        "gpu_task_run_id": "1124045064530911",
        "mlflow_run_id": "45fdc2d1e9ff45078fa08fbf021d4a1b",
        "mlflow_status": "FAILED",
        "mlflow_logged_param_count": 0,
        "mlflow_logged_metric_count": 0,
        "mlflow_logged_artifact_count": 0,
        "closeout_receipt_sha256": (
            "19842691fe1b2ed17c681eee9769c3580b081cdc24cc6f7bf43bde9b33d5e07d"),
        "closeout_file_sha256": (
            "3c90ce3a24ede317f1cf72f98cd1fccb70ca6dd6d2d544d17189d4a6d2b4197b"),
        "reservation_sha256": (
            "b7097ede43616c8961811a4003aaac7960a3d02e8f21c7a6ea6962f7e8191cfd"),
        "consumed_file_sha256": (
            "46a0e29462f8700a08664f10492ac405177d34e5e024504cf84e533e7f68c778"),
        "terminal_file_sha256": (
            "bdc4289a6de680108e70b6095114eaba798c989a419aa9c214a23d950002b2c0"),
        "mlflow_params_file_sha256": (
            "c7d3c498f06af85afe40068452b7be7bf12a72952fecdb342d2f600a3b26c600"),
        "mlflow_progress_file_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        "rank_partial_file_sha256": (
            "f3263b28b588232e1e65c05d607f009413a6657e7a504f40240b776b8666a01e"),
        "independent_failure_audit_sha256": (
            "b2254725ca277526c1982a17a1d4d8b07d1bb49d423308bcab6ac349734dd40b"),
        "approved_proposal_sha256": (
            "9cc3abf46085a7e9b2b2f205444e905b7b6926a02d1bb7387a56fe6e1dd78707"),
        "approved_draft_patch_sha256": (
            "660cc0e2cc16d63d4d2bb60a2a60b278293e276418198af19f3f376862077659"),
        "supervisor_approval_file_sha256": (
            "3fbdf776ba04869aba4f1dacd942eb38ce64418d0221a76fc3d600c67d55ea16"),
        "prior_training_source_sha256": (
            "8082af227f4dc067c254614ac1e62a35802f780397dd8660f65deb1ac7cb97e8"),
        "prior_worker_source_sha256": (
            "19d0e8d014ccb6a24246f7c2c42d8d0e6214418c54cd50d70c2c7290f931e10e"),
        "prior_databricks_worker_source_sha256": (
            "3decacb46cbdc529cfbb0d4169f30241f868b342eebc7e260128b8170b17a705"),
        "prior_technical_successor_source_sha256": (
            "6af500fd3c17d479a0f408deeff9e7ac99f4a0e5f2b4d9375e7ac588ea433b67"),
        "failure_class": "sft_nonfinite_gradient_at_minimum_scale",
        "terminal_status": "partition_notebook_failed",
        "gpu_task_result_state": "FAILED",
        "error_type": "GradScalerRecoveryError",
        "error": "Bounded GradScaler overflow adjustments were exhausted",
        "backward_attempts": 5,
        "scale_chain": [65536.0, 32768.0, 16384.0, 8192.0, 4096.0],
        "optimizer_updates": 0,
        "logical_attempted_tokens": 13480,
        "combined_initial_and_replay_tokens": 67400,
        "terminal_sft_diagnostic_tokens": 13480,
        "total_physical_tokens": 80880,
        "first_nonfinite_parameter": (
            "model.model.language_model.layers.0.linear_attn.out_proj."
            "delta1_a.weight"),
        "first_nonfinite_parameter_elements": 16384,
        "first_nonfinite_parameter_nan_count": 16384,
        "rank_trainable_gradient_elements": 10819584,
        "nonfinite_gradient_elements_at_scale_4096": 10208136,
        "sft_nonfinite_gradient_elements_at_scale_4096": 10206012,
        "sft_backward_at_scale_4096_nonfinite": True,
        "preference_and_margin_not_examined_after_sft_failure": True,
        "initial_scale_only_hypothesis_disproved": True,
        "exact_root_cause_status": "not_fully_resolved",
        "optimizer_artifact_written": False,
        "checkpoint_written": False,
        "training_receipt_written": False,
        "evidence_written": False,
        "final_science_journal_written": False,
        "reservation_released": True,
        "consumed_identity_retained": True,
        "automatic_retry": False,
        "verified_model_tree_sha256": (
            "9845026dbe255e24b105224eebbb5436d315713b9a5c53c434137896b160c5b1"),
        "verified_model_copy_bytes": 19329393661,
        "public_model_download_bytes": 0,
        "snapshot_download_used": False,
        "successor_execution_scope": "databricks-a10x1-only",
        "successor_numerical_policy": "databricks-a10-fixed-fp16-scale1-v1",
        "successor_precision": "fp16",
        "successor_quantization": "nf4-double",
        "successor_world_size": 1,
        "successor_micro_batch_size": 1,
        "successor_accumulation_steps": 4,
        "successor_global_batch_size": 4,
        "successor_grad_scaler_scale": 1.0,
        "successor_grad_scaler_backoff_factor": 0.5,
        "successor_grad_scaler_growth": False,
        "successor_grad_scaler_reductions": 0,
        "successor_max_overflow_adjustments": 0,
        "successor_minimum_grad_scaler_scale": 1.0,
        "successor_one_attempt_per_update": True,
        "successor_optimizer_steps_on_nonfinite": 0,
        "successor_positive_finite_gradient_norm_required": True,
        "successor_nonzero_active_gradient_density_required": True,
        "successor_physical_tokens_equal_logical_on_success": True,
        "successor_component_diagnostics": (
            ["sft", "preference", "cross_danger_margin"]),
        "successor_plan_sha256": (
            "83a3e4a213316cba3ae21c9b83ebbf0fa631cfdbed5b615a158bf6495bac31bc"),
        "successor_primary_plan_hashes_digest": (
            "d4278b41b680fcc3897239a3ee17639d36d4bdec59824b428ce7bc9830692e40"),
        "legacy_non_databricks_failfast_preserved": True,
        "model_data_order_objective_unchanged": True,
        "loss_coefficients_unchanged": True,
        "learning_rate_clipping_geometry_seeds_thresholds_unchanged": True,
        "legacy_combined_loss_arithmetic_preserved": True,
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
        "consumed_file_sha256",
        "terminal_file_sha256", "mlflow_params_file_sha256",
        "mlflow_progress_file_sha256", "rank_partial_file_sha256",
        "independent_failure_audit_sha256", "approved_proposal_sha256",
        "approved_draft_patch_sha256", "supervisor_approval_file_sha256",
        "prior_training_source_sha256", "prior_worker_source_sha256",
        "prior_databricks_worker_source_sha256",
        "prior_technical_successor_source_sha256",
        "prior_source_manifest_sha256",
        "prior_technical_successor_sha256", "successor_plan_sha256",
        "successor_primary_plan_hashes_digest",
        "verified_model_tree_sha256",
        "reservation_sha256",
        "predecessor_authority_sha256",
        "technical_successor_sha256",
    )
    if (actual != expected or embedded != _digest(payload)
            or predecessor != _gradscaler_recovery_authority()
            or predecessor.get("technical_successor_sha256")
               != "e58a2d86b52bb6b06a8239847add8c8c84582503fbc4c8fd9ea32305467d6ea0"
            or actual.get("predecessor_authority_sha256")
               != predecessor.get("technical_successor_sha256")
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(actual.get(key, "")))
                   for key in sha_fields)):
        raise PermissionError("Rank-4 technical successor authority differs")
    return actual
