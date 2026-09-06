"""Fresh certified-descent identity and unchanged model/prompt/schedule metadata."""

from __future__ import annotations

from pathlib import Path

from scripts import soft_drift_constrained_comply_plan as parent

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/certified_descent_comply_v1.json"
CONFIG_SHA = "afcec9a5ada7afe2a253ef87034070cff6ab7f19865f39f7233372a830558207"
OUTPUT = "evidence/certified_descent_comply_v1_qwen35_08b"
PLAN = "scripts/certified_descent_comply_plan.py"
SCRIPT = "scripts/certified_descent_comply.py"
OPTIMIZER = "scripts/certified_descent_comply_optimizer.py"
VERIFY = "scripts/verify_certified_descent_comply.py"
RECORDING = "scripts/certified_descent_comply_recording.py"
RECORDING_POLICY = "configs/certified_descent_comply_recording_policy.json"
DOC = "docs/CERTIFIED_DESCENT_COMPLY_V1.md"
PREP_REPORT = "docs/CERTIFIED_DESCENT_COMPLY_PREPARATION.md"
PREP_JSON = "docs/certified_descent_comply_preparation.json"
AUTH_KEY = "SP_LENSE_CERTIFIED_DESCENT_COMPLY_V1_AUTHORIZATION"
AUTH_SCOPE = "one fresh certified-descent dyadic COMPLY construction;216F/96D/1800s;no retry"
require, read, sha, io = parent.require, parent.read, parent.sha, parent.io
isolate, canonical_sha, authenticated = parent.isolate, parent.canonical_sha, parent.authenticated
FAMILIES, ORDERS, DISPLAYS, PAIRS = parent.FAMILIES, parent.ORDERS, parent.DISPLAYS, parent.PAIRS
NEW_SOURCES = (
    CONFIG,
    PLAN,
    SCRIPT,
    OPTIMIZER,
    VERIFY,
    RECORDING,
    RECORDING_POLICY,
    DOC,
    PREP_REPORT,
    PREP_JSON,
    "scripts/certified_descent_dyadic_solver.py",
    "scripts/verify_certified_descent_dyadic.py",
    "scripts/partial_progress_deficit_solver.py",
    "scripts/verify_partial_progress_deficit.py",
    "docs/CERTIFIED_DESCENT_DYADIC_DESIGN.md",
    "tests/test_certified_descent_comply_optimizer.py",
    "tests/test_certified_descent_comply_runtime.py",
    "tests/test_certified_descent_comply_recording.py",
    "tests/test_certified_descent_comply_plan.py",
    "tests/certified_descent_comply_fake_recording.py",
    "tests/test_certified_descent_comply_preparation.py",
    "docs/CERTIFIED_DESCENT_INTEGRATION_JOB_PROTOCOL.md",
    "docs/certified_descent_comply_model_free_test_lock.json",
    "docs/certified_descent_comply_metadata_test_lock.json",
    "docs/certified_descent_comply_model_free_results.json",
)
SOURCE_PATHS = tuple(dict.fromkeys((*parent.SOURCE_PATHS, *NEW_SOURCES)))


def overlay_at(root=ROOT):
    root = Path(root)
    require(sha((root / CONFIG).read_bytes()) == CONFIG_SHA, "exact certified-descent overlay")
    overlay = read(root / CONFIG)
    for key in ("base_config", "solver", "independent_numeric_checker", "approved_design"):
        item = overlay[key]
        require(sha((root / item["path"]).read_bytes()) == item["sha256"], "immutable method/input")
    for path, expected in overlay["immutable_dependencies"].items():
        require(sha((root / path).read_bytes()) == expected, "immutable dyadic dependency")
    require(
        overlay["output_namespace"] == OUTPUT
        and overlay["method_identity"] == "certified_descent_dyadic_v1"
        and overlay["pairs_one_based"] == PAIRS
        and tuple(
            overlay[key]
            for key in (
                "maximum_forwards",
                "maximum_derivatives",
                "maximum_updates",
                "timeout_seconds",
            )
        )
        == (216, 96, 8, 1800)
        and overlay["coefficients"] == {"proximal": "1/2", "deficit": "1/24", "drift": "1/48"}
        and overlay["nominal_proposals"] == 1
        and overlay["scale_indices"] == [0, 52]
        and overlay["combined_update_seconds"] == 10,
        "fixed method, objective and bounded schedule",
    )
    return overlay


def recording_policy(root=ROOT):
    from scripts import certified_descent_comply_recording as recording

    return recording.recording_policy(root)


def require_preparation_certificate(root=ROOT):
    from scripts import certified_descent_comply_recording as recording

    return recording.require_certificate(root)


def config_at(root=ROOT):
    from scripts import certified_descent_comply_recording as recording

    overlay = overlay_at(root)
    original = parent.config_at(root)
    return {
        **original,
        "schema": "sp_lense.certified_descent_comply_v1.v1",
        "output_namespace": OUTPUT,
        "timeout_seconds": 1800,
        "cast_sequence": "A_i=-own_original_h0_norm_i*gS_i; m_i=-S_i; b_i=.10-m_i signed; c=((m_A-m_A0)-(m_B-m_B0))/2; one unchanged dyadic proposal; exact serialized endpoint difference and original balls/path; first certified scale j0..52; no repair; own-norm float32 hook/casts unchanged",
        "objective": {
            "name": "fixed_partial_deficit_affine_drift_proximal",
            "coefficients": overlay["coefficients"],
            "pairs_one_based": PAIRS,
            "negative_rhs_preserved": True,
            "acceptance_gate": False,
        },
        "optimizer": {
            "name": "certified_descent_dyadic_v1",
            "solver": overlay["solver"],
            "checker": overlay["independent_numeric_checker"],
            "initialization": "fresh zero float64",
            "history": "authenticated fresh-zero endpoint history; conservative exact rational path upper",
            "sufficient_decrease": overlay["sufficient_decrease"],
            "one_nominal_proposal": True,
            "fixed_scale_indices": [0, 52],
            "maximum_combined_update_seconds": 10,
            "near_optimality_gate": False,
            "rescue": False,
            "line_search": "fixed declared dyadic feasibility/decrease trials only",
            "best_earlier_iterate": False,
        },
        "approved_proposal": overlay["approved_design"],
        "recording_policy": {"path": RECORDING_POLICY, "sha256": recording.POLICY_SHA},
        "execution_authority": {
            "preparation_only": True,
            "eventual_run_requires_separate_supervisor_authorization": True,
            "environment_key": AUTH_KEY,
            "authorization_scope": AUTH_SCOPE,
        },
        "provenance": {
            "adaptive_development": True,
            "independent_confirmation": False,
            "whole_new_construction_method": True,
            "isolated_penalty_causal_claim": False,
            "prior_inconclusive_or_zero_flip_result_changed": False,
            "old_coordinates_access_allowed": False,
            "f04_evidence_access_allowed": False,
            "warm_start_allowed": False,
            "resume_allowed": False,
            "retry": False,
            "source_design_commit": "35f0a048be9e27f05056043407173fae0c4ab447",
            "standalone_solver_commit": "48fd367ab4d07df2b1100911189e681f0ef62ced",
            "posthoc_saved_initial_step_reuse_forbidden": True,
        },
    }


def storage_preflight(root, config):
    result = parent.storage_preflight(root, config)
    recording_policy(root)
    return result


def build_plan(root=ROOT):
    original = parent.build_plan(root)
    overlay, config = overlay_at(root), config_at(root)
    inputs = dict(original["input_sha256"])
    inputs[CONFIG] = CONFIG_SHA
    for key in ("solver", "independent_numeric_checker", "approved_design"):
        item = overlay[key]
        inputs[item["path"]] = item["sha256"]
    inputs.update(overlay["immutable_dependencies"])
    inputs[RECORDING_POLICY] = config["recording_policy"]["sha256"]
    require(
        len(original["cells"]) == 216
        and len(original["derivative_cells"]) == 96
        and len(original["prompts"]) == 12
        and not original["transfer_ids"]
        and not original["control_ids"],
        "unchanged twelve-row conditional schedule",
    )
    return {
        **original,
        "schema": config["schema"],
        "output_namespace": OUTPUT,
        "config": config,
        "input_sha256": inputs,
        "recording_policy": recording_policy(root),
    }
