"""Inert compact-guard test data, constructed without local research records.

All paths, hashes, process identifiers and approvals here are synthetic. The
held template is deliberately non-authorizing and cannot start a Game.
"""


def held_template():
    def ref(name, digit):
        return {"path": ".local/CPU_SYNTHETIC/" + name + ".json", "sha256": digit * 64}

    return {
        "schema": "research4.positive84-compact-scientific-amendment-held-template.v1",
        "status": "PREPARED_SCIENTIFIC_SHAPE_NOT_GAME_EXECUTION",
        "execution_authorized": False,
        "final_record_schema": "research4.positive84-compact-scientific-amendment.v1",
        "created_utc": "2026-01-01T00:00:00+00:00",
        "transport_parent": {
            "Main_PLAN_external_opaque_ref": ref("parent", "1"),
            "qualified_Core": ref("core", "2"),
            "qualification": ref("qualification", "3"),
            "service_pid": 12345, "service_created_epoch": 100.0,
        },
        "actual_artifact_binding": {
            "expected_ft_id": "synthetic-not-a-real-model", "adapter_sha256": "a" * 64,
            "adapter_config_sha256": "b" * 64, "base_manifest_sha256": "c" * 64,
            "artifact_ready": ref("not-ready", "d"), "ready_utc": "2026-01-01T00:00:00+00:00",
            "completion_audit": ref("synthetic-audit", "e"), "fit_updates": 168, "teacher_dev_rows": 24,
        },
        "scientific_successor": {
            "view_mode": "synthetic", "full_transcript_saved_for_audit": True,
            "prior_model_reasoning_replayed": False, "science_plan_ref": None,
            "source_packet_ref": None, "paired_world_identity_proof_ref": None,
            "same_preselected_three_worlds_and_twelve_cells": True,
            "new_or_rerolled_worlds": False, "twenty_four_cell_AB_experiment": False,
            "historical_FULL_protocol_silently_relabelled": False,
        },
        "unchanged_protocol": {
            "scientific_cases": 12, "preselected_worlds": 3, "conditions_per_world": 4,
            "context_total_tokens": 131072, "max_output_tokens": 8192, "max_request_seconds": 900,
            "max_case_seconds": 14400, "max_case_decisions": 256, "Game_horizon_hours": 48,
            "max_Game_requests": 3072, "technical_qualification_calls_separate": 3,
            "serving_memory_floor_bytes": 4294967296, "max_concurrency": 1,
            "cache_and_BASE_restore_guards_unchanged": True, "scoring_and_world_physics_unchanged": True,
            "generation_profile": {"seed": 20261108, "temperature": 1.0, "top_p": .95,
                "top_k": 20, "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}},
        },
        "resource_lifecycle": {
            "original_origin_monotonic": 100.0, "original_whole_seconds": 90000,
            "original_whole_deadline_monotonic": 90100.0, "cleanup_reserve_seconds": 100,
            "latest_Game_expiry_monotonic": 90000.0,
            "unsupported_idle_extension_permitted": False, "clock_or_quota_reset_permitted": False,
            "resident_owner_must_still_match_before_activation": True, "automatic_restart_after_release": False,
        },
        "reviews": {"Main_exact_source_plan_review_ref": None,
                    "independent_exact_source_plan_review_ref": None,
                    "Root_final_exact_amendment_review_ref": None},
        "human_approval": {"required_before_any_scientific_request": True,
                           "approval_record_ref": None, "approved": False,
                           "code_completion_or_peer_review_is_human_approval": False},
        "activation_contract": {"this_template_never_authorizes_execution": True},
    }
