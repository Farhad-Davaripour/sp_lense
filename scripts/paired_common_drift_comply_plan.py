"""Fresh-zero paired-drift plan; immutable prompt/model metadata, no model loading."""

from __future__ import annotations

from pathlib import Path

from scripts import shared_comply_crossed_three_family_plan as parent

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/paired_common_drift_comply_three_family_v1.json"
CONFIG_SHA = "13e005cb53151a02aaf19d65eb832d0b4b923ec6f43a26212536a6c4e52fb55f"
DOC = "docs/PAIRED_COMMON_DRIFT_COMPLY_V1.md"
PREP_REPORT = "docs/PAIRED_COMMON_DRIFT_COMPLY_PREPARATION.md"
PREP_JSON = "docs/paired_common_drift_comply_preparation.json"
SCRIPT = "scripts/paired_common_drift_comply.py"
VERIFY = "scripts/verify_paired_common_drift_comply.py"
TEST = "tests/test_paired_common_drift_optimizer.py"
PLAN = "scripts/paired_common_drift_comply_plan.py"
OPTIMIZER = "scripts/paired_common_drift_comply_optimizer.py"
RECORDING = "scripts/paired_common_drift_comply_recording.py"
RECORDING_POLICY = "configs/paired_common_drift_comply_recording_policy.json"
PROPOSAL = "docs/PAIRED_COMMON_DRIFT_COMPLY_PROPOSAL.md"
PROPOSAL_SHA = "003466be90c6084ab0446af623b8f29145e11b5b930e56b141c6c153c40c3d61"
TOY = "docs/PAIRED_COMMON_DRIFT_TOY_CHECKS.md"
TOY_SHA = "37b57cdc9a34a407665ffb6442156b421a803efae5b6f33101e767b47c69f8ac"
TEMPLATE = "evidence/shared_comply_crossed_three_family_v1_qwen35_08b/preregistration.json"
TEMPLATE_SHA = "99f217e7601b478b7bd89e010cd8fcbafbbbb163aaff5a96376fde94c0911e80"
OUTPUT = "evidence/paired_common_drift_comply_three_family_v1_qwen35_08b"
AUTH_KEY = "SP_LENSE_PAIRED_COMMON_DRIFT_COMPLY_AUTHORIZATION"
AUTH_SCOPE = "one fresh paired common-drift COMPLY construction;216F/96D/1200s;no retry"
SOURCE_PATHS = (
    CONFIG,
    DOC,
    PREP_REPORT,
    PREP_JSON,
    SCRIPT,
    VERIFY,
    TEST,
    PLAN,
    OPTIMIZER,
    RECORDING,
    RECORDING_POLICY,
    "tests/test_paired_common_drift_recording.py",
    "tests/test_paired_common_drift_comply.py",
    "tests/test_verify_paired_common_drift_comply.py",
    PROPOSAL,
    TOY,
    "scripts/shared_comply_crossed_three_family_plan.py",
    "scripts/frozen_preserve_probe_plan.py",
    "scripts/frozen_guarded_preserve_crossed_plan.py",
    "tests/test_local_controllability_positive_control.py",
    "scripts/three_family_recording_bindings.py",
    "scripts/three_family_recording_budget.py",
    "scripts/three_family_bounded_capture.py",
)
require, read, sha, io = parent.require, parent.read, parent.sha, parent.io
isolate, canonical_sha = parent.isolate, parent.canonical_sha
FAMILIES, ORDERS, DISPLAYS = parent.FAMILIES, parent.ORDERS, parent.DISPLAYS
PAIRS = [[3, 1], [4, 2], [7, 5], [8, 6], [11, 9], [12, 10]]

# Bind inherited scientific/runtime/model-loader source bytes without importing
# historical results or protected candidate coordinates. The parent lock itself
# is authenticated before its source manifest is used.
require(sha((ROOT / TEMPLATE).read_bytes()) == TEMPLATE_SHA, "exact source-manifest template")
INHERITED_SOURCE_SHA256 = {
    path.replace("\\", "/"): digest
    for path, digest in read(ROOT / TEMPLATE)["source_sha256"].items()
    if path.replace("\\", "/").startswith(("scripts/", "src/sp_lense/", "configs/"))
}
SOURCE_PATHS = tuple(dict.fromkeys((*SOURCE_PATHS, *INHERITED_SOURCE_SHA256)))


def config_at(root=ROOT):
    require(sha((root / CONFIG).read_bytes()) == CONFIG_SHA, "exact prospective config bytes")
    return read(root / CONFIG)


def authenticated(path, digest, root=ROOT):
    require(sha((root / path).read_bytes()) == digest, "exact approved input bytes")
    return read(root / path)


def recording_policy(root=ROOT):
    from scripts import paired_common_drift_comply_recording as recording

    return recording.recording_policy(root)


def require_preparation_certificate(root=ROOT):
    from scripts import paired_common_drift_comply_recording as recording

    return recording.require_certificate(root)


def storage_preflight(root, config):
    """Retain original conditional allocation; certification is a separate gate."""
    result = parent.storage_preflight(root, config)
    recording_policy(root)
    return result


def build_plan(root=ROOT):
    config, old = config_at(root), parent.config_at(root)
    changed = {
        "schema",
        "output_namespace",
        "templates",
        "cast_sequence",
        "candidate_freeze",
        "protected_artifacts",
        "provenance",
        "execution_authority",
    }
    extras = {"objective", "optimizer", "approved_proposal", "toy_arithmetic", "recording_policy"}
    require(
        set(config) == (set(old) - {"solver"}) | extras
        and all(config[k] == v for k, v in old.items() if k not in changed | {"solver"}),
        "only approved objective/update/provenance changes; exact prior prompts and budgets",
    )
    require(
        config["schema"] == "sp_lense.paired_common_drift_comply_three_family_v1.v1"
        and config["output_namespace"] == OUTPUT
        and not config["protected_artifacts"]
        and config["target"] == "comply"
        and config["maximum_rows"] == 12
        and (
            config["maximum_forwards"],
            config["maximum_derivatives"],
            config["timeout_seconds"],
            config["maximum_updates"],
        )
        == (216, 96, 1200, 8),
        "fresh paired method identity; no old candidate coordinate inputs",
    )
    require(
        config["objective"]
        == {
            "tau": 0.10,
            "lambda": 1,
            "pairs_one_based": PAIRS,
            "aggregation": "mean_of_pair_losses",
            "drift_penalty": "mean_of_squared_pair_drifts",
            "loss_units": "nats_squared",
            "acceptance_gate": False,
        },
        "one preregistered equal-unit objective; no coefficient tuning",
    )
    require(
        config["optimizer"]
        == {
            "name": "paired_common_drift_curvature_scaled_gradient",
            "B0": 4.0,
            "arithmetic": "float64 with fsum reductions in fixed pair and coordinate order",
            "scalar_component_audit_tolerance": "1e-9*max(1,abs(reference value))",
            "initialization": "fresh zero native float64 vector",
            "line_search": False,
            "momentum": False,
            "extra_gradients": False,
            "warm_start": False,
            "best_earlier_iterate": False,
            "rescue": False,
            "curvature_scope": "frozen-Jacobian squared-hinge surrogate only; not nonlinear Hessian",
            "descent_guaranteed": False,
        },
        "one exact current-gradient update; no extra algorithm",
    )
    require(
        config["approved_proposal"] == {"path": PROPOSAL, "sha256": PROPOSAL_SHA}
        and config["toy_arithmetic"] == {"path": TOY, "sha256": TOY_SHA}
        and config["templates"]
        == [
            {
                "key": "immutable_three_family_prompt_model_schedule",
                "path": TEMPLATE,
                "sha256": TEMPLATE_SHA,
            }
        ],
        "approved method/toy and metadata-only numerical parent",
    )
    original = authenticated(TEMPLATE, TEMPLATE_SHA, root)["plan"]
    # Only metadata and original prompt selection are reused. Never call the old
    # build_plan: it hashes protected arrows which are not inputs to this method.
    data = authenticated(config["dataset"]["path"], config["dataset"]["sha256"], root)
    manifest = authenticated(config["manifest"]["path"], config["manifest"]["sha256"], root)
    prompts = parent.select_prompts(data, manifest, config)
    ids = [p["prompt_id"] for p in prompts]
    require(
        prompts == old["rendered_prompts"] == original["prompts"] == config["rendered_prompts"]
        and ids == config["construction_order"] == original["construction_ids"]
        and len(prompts) == 12
        and {p["family_id"] for p in prompts} == set(FAMILIES),
        "exact twelve existing f01/f02/f03 prompts and order",
    )
    pairs = []
    for pair_index, (a_index, b_index) in enumerate(PAIRS, 1):
        a, b = prompts[a_index - 1], prompts[b_index - 1]
        require(
            a["comply_label"] == "A"
            and b["comply_label"] == "B"
            and a["family_id"] == b["family_id"]
            and a["display_order"] == b["display_order"],
            "pair opposite mappings within one family and same display",
        )
        pairs.append(
            {
                "pair_index": pair_index,
                "a_row_index": a_index,
                "b_row_index": b_index,
                "prompt_id_A": a["prompt_id"],
                "prompt_id_B": b["prompt_id"],
                "family_id": a["family_id"],
                "display_order": a["display_order"],
            }
        )
    inputs = {
        **INHERITED_SOURCE_SHA256,
        TEMPLATE: TEMPLATE_SHA,
        parent.CONFIG: parent.CONFIG_SHA,
        PROPOSAL: PROPOSAL_SHA,
        TOY: TOY_SHA,
        **{config[k]["path"]: config[k]["sha256"] for k in ("dataset", "manifest")},
        original["model"]["config_path"]: original["model"]["config_sha256"],
        original["direction"]["path"]: original["direction"]["file_sha256"],
    }
    # The unchanged loader authenticates this historical direction for provenance,
    # never as an intervention or warm start. Neither old C is accessed.
    require(
        original["direction"]["used_as_intervention"] is False,
        "inherited loader direction is provenance-only",
    )
    for path, digest in inputs.items():
        require(sha((root / path).read_bytes()) == digest, "immutable scientific/proposal input")
    policy = recording_policy(root)
    require(
        config["recording_policy"]["path"] == RECORDING_POLICY
        and policy == authenticated(RECORDING_POLICY, config["recording_policy"]["sha256"], root),
        "new method has its own exact scoped recording policy",
    )
    inputs[RECORDING_POLICY] = config["recording_policy"]["sha256"]
    cells = []
    groups = [("baseline", 0, False)]
    for stage in range(1, 9):
        groups.extend([(f"gradient_{stage}", stage, True), (f"step_{stage}", stage, True)])
    groups.append(("final", 9, False))
    for condition, stage, optional in groups:
        for pid in ids:
            cell = {
                "cell_id": pid + "__" + condition,
                "prompt_id": pid,
                "condition": condition,
                "stage": stage,
                "optional": optional,
            }
            cells.append({**cell, "cell_sha256": canonical_sha(cell)})
    require(
        cells == original["cells"]
        and len(cells) == 216
        and sum(c["optional"] for c in cells) == 192,
        "same conditional call schedule",
    )
    return {
        **{
            k: original[k]
            for k in ("model", "direction", "scoring", "prompt_format", "intervention")
        },
        "schema": config["schema"],
        "output_namespace": OUTPUT,
        "config": config,
        "input_sha256": inputs,
        "prompts": prompts,
        "construction_ids": ids,
        "control_ids": [],
        "transfer_ids": [],
        "cells": cells,
        "derivative_cells": [c for c in cells if c["condition"].startswith("gradient_")],
        "pairs": pairs,
        "recording_policy": policy,
    }
