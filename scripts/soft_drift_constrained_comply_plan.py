"""Fresh metadata-only soft-drift QP identity; immutable twelve-row parent inputs."""

from __future__ import annotations

from pathlib import Path

from scripts import paired_common_drift_comply_plan as parent

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/soft_drift_constrained_comply_v1.json"
CONFIG_SHA = "802d00b8caa160dff195ab2466e2bb349e24ba50f8fcf078c96c6e4a7afb3733"
OUTPUT = "evidence/soft_drift_constrained_comply_v1_qwen35_08b"
PLAN = "scripts/soft_drift_constrained_comply_plan.py"
SCRIPT = "scripts/soft_drift_constrained_comply.py"
OPTIMIZER = "scripts/soft_drift_constrained_comply_optimizer.py"
VERIFY = "scripts/verify_soft_drift_constrained_comply.py"
RECORDING = "scripts/soft_drift_constrained_comply_recording.py"
RECORDING_POLICY = "configs/soft_drift_constrained_comply_recording_policy.json"
DOC = "docs/SOFT_DRIFT_CONSTRAINED_COMPLY_V1.md"
PREP_REPORT = "docs/SOFT_DRIFT_CONSTRAINED_COMPLY_PREPARATION.md"
PREP_JSON = "docs/soft_drift_constrained_comply_preparation.json"
AUTH_KEY = "SP_LENSE_SOFT_DRIFT_CONSTRAINED_COMPLY_V1_AUTHORIZATION"
AUTH_SCOPE = "one fresh soft-drift constrained COMPLY construction;216F/96D/1800s;no retry"
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
    "scripts/soft_drift_qp_solver.py",
    "scripts/verify_soft_drift_qp.py",
    "docs/SOFT_DRIFT_QP_NUMERIC_POLICY.md",
    "docs/SOFT_DRIFT_CONSTRAINED_QP_DESIGN.md",
    "scripts/paired_common_drift_comply_1800_binding.py",
    "tests/test_soft_drift_constrained_comply_runtime.py",
    "tests/test_verify_soft_drift_constrained_comply.py",
    "tests/test_soft_drift_constrained_comply_recording.py",
    "tests/test_soft_drift_constrained_comply_plan.py",
    "tests/soft_drift_constrained_fake_recording.py",
)
SOURCE_PATHS = tuple(dict.fromkeys((*parent.SOURCE_PATHS, *NEW_SOURCES)))


def overlay_at(root=ROOT):
    raw = (Path(root) / CONFIG).read_bytes()
    require(sha(raw) == CONFIG_SHA, "exact fresh soft-drift overlay bytes")
    overlay = read(Path(root) / CONFIG)
    for key in (
        "base_config",
        "solver",
        "independent_numeric_checker",
        "numeric_policy",
        "approved_design",
    ):
        binding = overlay[key]
        require(
            sha((Path(root) / binding["path"]).read_bytes()) == binding["sha256"],
            "immutable input/solver/checker/numerical policy bytes",
        )
    require(
        overlay["output_namespace"] == OUTPUT
        and overlay["pairs_one_based"] == PAIRS
        and (
            overlay["maximum_forwards"],
            overlay["maximum_derivatives"],
            overlay["maximum_updates"],
            overlay["timeout_seconds"],
        )
        == (216, 96, 8, 1800)
        and overlay["kappa"] == {"numerator": 1, "denominator": 24}
        and (
            overlay["step_cap"],
            overlay["total_cap"],
            overlay["path_cap"],
            overlay["aim"],
            overlay["acceptance"],
        )
        == (0.05, 0.20, 0.40, 0.10, 0.05),
        "fixed method/schedule/geometry",
    )
    return overlay


def recording_policy(root=ROOT):
    from scripts import soft_drift_constrained_comply_recording as recording

    return recording.recording_policy(root)


def require_preparation_certificate(root=ROOT):
    from scripts import soft_drift_constrained_comply_recording as recording

    return recording.require_certificate(root)


def config_at(root=ROOT):
    from scripts import soft_drift_constrained_comply_recording as recording

    overlay = overlay_at(root)
    original = parent.config_at(root)
    return {
        **original,
        "schema": "sp_lense.soft_drift_constrained_comply_v1.v1",
        "output_namespace": OUTPUT,
        "timeout_seconds": 1800,
        "cast_sequence": "A_i=-own_baseline_norm_i*gS_i; b_i=.10+S_i (signed); c=((m_A-m_A0)-(m_B-m_B0))/2; D=(A_A-A_B)/2; solve fixed soft-drift QP kappa=1/24; independently certify raw d; s=d*min(1,.05/norm(d)) or zero; u=w+s; w_next=project_radius_.20(u); r=w_next-w; path_next=path+norm(r); unchanged own-norm float32 casts and fresh original prompt per forward",
        "objective": {
            "name": "soft_drift_constrained_QP",
            "kappa": overlay["kappa"],
            "pairs_one_based": PAIRS,
            "baseline_relative_affine_drift": True,
            "negative_rhs_preserved": True,
            "acceptance_gate": False,
        },
        "optimizer": {
            "name": "fixed_soft_drift_QP_then_original_geometry",
            "solver": overlay["solver"],
            "checker": overlay["independent_numeric_checker"],
            "policy": overlay["numeric_policy"],
            "initialization": "fresh zero float64",
            "rescue": False,
            "line_search": False,
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
            "source_design_commit": "a72de9aaf16a5e7bda39e49e3bba01718807b1cd",
            "standalone_solver_commit": "0d2d104c3a367dea321f6621fca851e8d4ffab74",
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
    inputs[parent.CONFIG] = parent.CONFIG_SHA
    for key in ("solver", "independent_numeric_checker", "numeric_policy", "approved_design"):
        item = overlay[key]
        inputs[item["path"]] = item["sha256"]
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
