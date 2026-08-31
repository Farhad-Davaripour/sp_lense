#!/usr/bin/env python3
"""Tokenizer-only authoring utility for immutable FACFS Stage-G materials.

This utility never imports or loads a model.  It may be used only before the
Stage-G lock commit to populate deterministic source hashes and manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from sp_lense.facfs_stage_g_v2 import (
    NAMESPACE,
    build_identifier_plan,
    build_option_free_plan,
    build_tokenized_manifests,
    canonical_sha256,
    delexicalize,
    iter_strings,
    ngram_jaccard,
    normalize_text,
    plain,
    scenario_source_hashes,
    text_sha256,
    validate_source,
    with_identity_hash,
)
from sp_lense.gradient_specificity_v3 import tensor_float32_sha256

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/mnt/c/Users/farha/repos/sp_lense")
SOURCE_PATH = ROOT / "data" / "facfs_stage_g_v2_scenarios.json"
EXCLUSIONS_PATH = ROOT / "configs" / "facfs_stage_g_v2_exclusions.json"
OPERATIONS_PATH = ROOT / "configs" / "facfs_stage_g_v2_operations.json"
TOKEN_PATH = ROOT / "configs" / "facfs_stage_g_v2_token_certificate.json"
DIRECTION_PATH = ROOT / "configs" / "facfs_stage_g_v2_direction_certificate.json"
POWER_PATH = ROOT / "configs" / "facfs_stage_g_v2_power.json"
LOCK_PATH = ROOT / "configs" / "facfs_stage_g_v2_lock.json"
MODEL_CONFIG_PATH = ROOT / "configs" / "qwen35_08b_aligned.json"
RAW_DIRECTION_PATH = (
    ROOT
    / "artifacts"
    / "steering_comparison"
    / "one_day_local"
    / "qwen35_08b"
    / "directions"
    / "gradient.json"
)
SEALED_PATHS = (
    "data/ckes_sealed.json",
    "data/ckes_v2_sealed.json",
)
STAGE_PATHS = {
    "data/facfs_stage_g_v1_scenarios.json",
    "configs/facfs_stage_g_v1_exclusions.json",
    "configs/facfs_stage_g_v1_operations.json",
    "configs/facfs_stage_g_v1_token_certificate.json",
    "configs/facfs_stage_g_v1_direction_certificate.json",
    "configs/facfs_stage_g_v1_power.json",
    "configs/facfs_stage_g_v1_lock.json",
    "data/facfs_stage_g_v2_scenarios.json",
    "configs/facfs_stage_g_v2_exclusions.json",
    "configs/facfs_stage_g_v2_operations.json",
    "configs/facfs_stage_g_v2_token_certificate.json",
    "configs/facfs_stage_g_v2_direction_certificate.json",
    "configs/facfs_stage_g_v2_power.json",
    "configs/facfs_stage_g_v2_lock.json",
}
NEAR_DUPLICATE_THRESHOLD = 0.90
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
HF_HOME = "/mnt/c/Users/farha/.cache/huggingface"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plain(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def tracked_paths() -> list[str]:
    return [line for line in git("ls-files").splitlines() if line]


def populate_source_hashes() -> dict[str, Any]:
    payload = load_json(SOURCE_PATH)
    validate_source(payload, require_hashes=False)
    for scenario in payload["scenarios"]:
        scenario["source_hashes"] = scenario_source_hashes(
            payload, str(scenario["scenario_id"])
        )
    validate_source(payload)
    write_json(SOURCE_PATH, payload)
    return payload


def _walk_json(value: Any, *, key: str = "") -> tuple[list[str], list[str], list[str]]:
    strings: list[str] = []
    ids: list[str] = []
    names: list[str] = []
    if isinstance(value, str):
        strings.append(value)
        lowered = key.casefold()
        if lowered.endswith(("id", "_id")) or lowered in {
            "unit_id",
            "case_id",
            "form_id",
        }:
            ids.append(value)
        if lowered in {"name", "names", "self_name", "other_name", "target_name"}:
            names.append(value)
    elif isinstance(value, dict):
        for child_key, child in value.items():
            child_strings, child_ids, child_names = _walk_json(
                child, key=str(child_key)
            )
            strings.extend(child_strings)
            ids.extend(child_ids)
            names.extend(child_names)
    elif isinstance(value, list):
        for child in value:
            child_strings, child_ids, child_names = _walk_json(child, key=key)
            strings.extend(child_strings)
            ids.extend(child_ids)
            names.extend(child_names)
    return strings, ids, names


def _parse_historical(path: Path) -> tuple[list[str], list[str], list[str]]:
    text = path.read_text(encoding="utf-8", errors="strict")
    if path.suffix == ".json":
        return _walk_json(json.loads(text))
    if path.suffix == ".jsonl":
        strings: list[str] = []
        ids: list[str] = []
        names: list[str] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            child_strings, child_ids, child_names = _walk_json(json.loads(line))
            strings.extend(child_strings)
            ids.extend(child_ids)
            names.extend(child_names)
        return strings, ids, names
    return [line for line in text.splitlines() if line.strip()], [], []


def _generic_delex(value: str, names: set[str]) -> str:
    substitutions = [name for name in names if 2 <= len(name) <= 64]
    result = delexicalize(value, substitutions)
    result = re.sub(r"\b[a-z]{5}\b", "<five_letter_slot>", result)
    result = re.sub(r"\b\d+\b", "<number>", result)
    return normalize_text(result)


def build_exclusions(payload: dict[str, Any]) -> dict[str, Any]:
    files = []
    historical_strings: list[str] = []
    historical_ids: set[str] = set()
    historical_names: set[str] = set()
    eligible_suffixes = {".json", ".jsonl", ".txt", ".md", ".yaml", ".yml", ".csv"}
    for relative in tracked_paths():
        if relative in STAGE_PATHS or not relative.startswith(("data/", "configs/")):
            continue
        path = ROOT / relative
        if relative in SEALED_PATHS:
            files.append(
                {
                    "path": relative,
                    "sealed_content_opened": False,
                    "git_blob_oid": git("rev-parse", f"HEAD:{relative}"),
                    "byte_size": int(git("cat-file", "-s", f"HEAD:{relative}")),
                }
            )
            continue
        if path.suffix.casefold() not in eligible_suffixes:
            continue
        strings, ids, names = _parse_historical(path)
        historical_strings.extend(strings)
        historical_ids.update(normalize_text(value) for value in ids if value.strip())
        historical_names.update(normalize_text(value) for value in names if value.strip())
        files.append(
            {
                "path": relative,
                "sealed_content_opened": False,
                "file_sha256": file_sha256(path),
                "byte_size": path.stat().st_size,
                "string_leaf_count": len(strings),
            }
        )
    normalized_historical = {
        normalize_text(value)
        for value in historical_strings
        if len(normalize_text(value)) >= 40
    }
    historical_prompt_candidates = sorted(
        {
            value
            for value in historical_strings
            if len(normalize_text(value)) >= 120
        },
        key=text_sha256,
    )
    historical_delex = {
        _generic_delex(value, historical_names)
        for value in historical_prompt_candidates
    }

    identifiers = build_identifier_plan(payload)
    free = build_option_free_plan(payload)
    new_rows = [*identifiers, *free]
    new_prompts = [str(row["prompt"]) for row in new_rows]
    new_prompt_hashes = [str(row["prompt_sha256"]) for row in new_rows]
    new_ids = {
        normalize_text(str(value))
        for scenario in payload["scenarios"]
        for value in (
            scenario["scenario_id"],
            scenario["family_id"],
            scenario["template_id"],
            *[entity["entity_id"] for entity in scenario["entities"]],
        )
    }
    new_names = {
        normalize_text(str(entity["name"]))
        for scenario in payload["scenarios"]
        for entity in scenario["entities"]
    }
    authored_strings = [
        value
        for scenario in payload["scenarios"]
        for value in iter_strings(
            {key: item for key, item in scenario.items() if key != "source_hashes"}
        )
        if len(normalize_text(value)) >= 40
    ]
    normalized_overlap = sorted(
        {normalize_text(value) for value in authored_strings + new_prompts}
        & normalized_historical
    )
    new_delex = {
        _generic_delex(prompt, new_names) for prompt in new_prompts
    }
    delex_overlap = sorted(new_delex & historical_delex)
    rendered_overlap = sorted(
        set(new_prompt_hashes)
        & {text_sha256(value) for value in historical_strings}
    )
    id_overlap = sorted(new_ids & historical_ids)
    name_overlap = sorted(new_names & (historical_names | historical_ids))

    maximum_similarity = 0.0
    maximum_pair: dict[str, Any] | None = None
    for new_prompt in sorted(set(new_prompts), key=text_sha256):
        for historical in historical_prompt_candidates:
            length_ratio = min(len(new_prompt), len(historical)) / max(
                len(new_prompt), len(historical)
            )
            if length_ratio < 0.50:
                continue
            similarity = ngram_jaccard(new_prompt, historical, 5)
            if similarity > maximum_similarity:
                maximum_similarity = similarity
                maximum_pair = {
                    "new_prompt_sha256": text_sha256(new_prompt),
                    "historical_text_sha256": text_sha256(historical),
                    "similarity": similarity,
                }
    collision_counts = {
        "canonical_id_overlap": len(id_overlap),
        "family_template_entity_name_overlap": len(name_overlap),
        "exact_normalized_text_overlap": len(normalized_overlap),
        "delexicalized_template_overlap": len(delex_overlap),
        "complete_rendered_prompt_sha_overlap": len(rendered_overlap),
        "duplicate_new_objective_ids": len(new_rows)
        - len({str(row["objective_id"]) for row in new_rows}),
        "duplicate_new_output_paths": len(new_rows)
        - len({str(row["output_stem"]) for row in new_rows}),
        "duplicate_new_prompt_hashes": len(new_prompt_hashes)
        - len(set(new_prompt_hashes)),
        "near_duplicate_at_or_above_threshold": int(
            maximum_similarity >= NEAR_DUPLICATE_THRESHOLD
        ),
    }
    if any(collision_counts.values()):
        raise RuntimeError(
            "Stage-G source collision audit failed: "
            + json.dumps(collision_counts, sort_keys=True)
        )
    return with_identity_hash(
        {
            "schema_version": "sp_lense.facfs.stage_g.exclusions.v1",
            "namespace": NAMESPACE,
            "normalization": "Unicode_NFKC_then_LF_then_Unicode_casefold_then_collapsed_whitespace",
            "character_ngram_width": 5,
            "near_duplicate_jaccard_threshold": NEAR_DUPLICATE_THRESHOLD,
            "threshold_comparison": "fail_if_greater_than_or_equal",
            "historical_roots": ["data/", "configs/"],
            "stage_paths_excluded_from_history": sorted(STAGE_PATHS),
            "sealed_sources_never_opened": list(SEALED_PATHS),
            "historical_files": files,
            "historical_counts": {
                "files": len(files),
                "string_leaves": len(historical_strings),
                "canonical_ids": len(historical_ids),
                "names": len(historical_names),
                "normalized_texts_length_at_least_40": len(normalized_historical),
                "prompt_candidates_length_at_least_120": len(
                    historical_prompt_candidates
                ),
            },
            "historical_set_hashes": {
                "canonical_ids_sha256": canonical_sha256(sorted(historical_ids)),
                "names_sha256": canonical_sha256(sorted(historical_names)),
                "normalized_texts_sha256": canonical_sha256(
                    sorted(normalized_historical)
                ),
                "delexicalized_templates_sha256": canonical_sha256(
                    sorted(historical_delex)
                ),
                "rendered_text_sha256_set_sha256": canonical_sha256(
                    sorted({text_sha256(value) for value in historical_strings})
                ),
            },
            "new_source_counts": {
                "scenario_ids": 11,
                "family_ids": 11,
                "template_ids": 11,
                "entity_ids": 22,
                "entity_names": 22,
                "rendered_prompts": len(new_prompts),
            },
            "collision_counts": collision_counts,
            "maximum_character_5gram_similarity": maximum_similarity,
            "maximum_similarity_pair_hashes": maximum_pair,
            "all_collision_gates_passed": True,
            "permanently_consumed_after_inspection": True,
            "forbidden_later_stages": ["construction", "qualification", "T1", "T2"],
        },
        "exclusions_sha256",
    )


def build_direction_certificate() -> dict[str, Any]:
    import torch

    record = load_json(RAW_DIRECTION_PATH)
    raw = torch.tensor(record["direction"], dtype=torch.float32).contiguous()
    if tuple(raw.shape) != (1024,):
        raise RuntimeError("inherited direction width differs")
    raw_hash = tensor_float32_sha256(raw)
    deployed = (raw / raw.norm().clamp_min(1e-12)).float().contiguous()
    deployed_hash = tensor_float32_sha256(deployed)
    if raw_hash != "0093b762c559a7ed9d15134fefa9399a4c1466232e84151ad22ad1aa1574427e":
        raise RuntimeError("inherited raw direction hash differs")
    if deployed_hash != "f4a7c9fb5620674f3a29646674a77e6c862b57b07b8d3e127d46c6bb931d0a63":
        raise RuntimeError("inherited deployed direction hash differs")
    return with_identity_hash(
        {
            "schema_version": "sp_lense.facfs.stage_g.direction_certificate.v1",
            "namespace": NAMESPACE,
            "source_path": RAW_DIRECTION_PATH.relative_to(ROOT).as_posix(),
            "source_file_sha256": file_sha256(RAW_DIRECTION_PATH),
            "source_artifact_sha256": record["artifact_sha256"],
            "method": "gradient",
            "semantic_orientation": 1,
            "layer_zero_based": 10,
            "position": "final_prompt_token",
            "width": 1024,
            "dtype": "float32",
            "raw_direction_float32_sha256": raw_hash,
            "raw_direction_norm_float64": float(raw.double().norm().item()),
            "deployment_expression": "d_raw / d_raw.norm().clamp_min(1e-12)",
            "deployed_direction_float32_sha256": deployed_hash,
            "deployed_direction_norm_float32": float(deployed.norm().item()),
            "deployed_direction_norm_float64": float(deployed.double().norm().item()),
            "deployed_direction_norm_tolerance": 1e-6,
            "no_reorientation": True,
            "no_reprojection": True,
            "no_alternative_normalization": True,
        },
        "direction_certificate_sha256",
    )


def build_power_report() -> dict[str, Any]:
    size = 0.75**11
    power = 0.98**11
    lower = 0.05 ** (1.0 / 11.0)
    if not (size < 0.05 and power >= 0.80 and lower > 0.75):
        raise RuntimeError("Stage-G exact design is not adequately powered")
    return with_identity_hash(
        {
            "schema_version": "sp_lense.facfs.stage_g.power.v1",
            "namespace": NAMESPACE,
            "independent_unit": "complete_source_disjoint_scenario_cluster",
            "scenario_count": 11,
            "decision_rule": "all_11_scenarios_must_pass",
            "null_complete_scenario_success_ceiling": 0.75,
            "alternative_complete_scenario_success_rate": 0.98,
            "one_sided_alpha": 0.05,
            "exact_null_size": size,
            "exact_power": power,
            "one_sided_95_percent_clopper_pearson_lower_if_11_of_11": lower,
            "interface_variants_are_repeated_measures": True,
        },
        "power_report_sha256",
    )


def build_materials() -> None:
    if ROOT.resolve() != EXPECTED_ROOT:
        raise RuntimeError(f"authoritative WSL root differs: {ROOT.resolve()}")
    payload = populate_source_hashes()
    exclusions = build_exclusions(payload)
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=f"{HF_HOME}/hub",
        local_files_only=True,
        trust_remote_code=False,
    )
    operations, token_certificate = build_tokenized_manifests(
        tokenizer, torch, payload
    )
    write_json(EXCLUSIONS_PATH, exclusions)
    write_json(OPERATIONS_PATH, operations)
    write_json(TOKEN_PATH, token_certificate)
    write_json(DIRECTION_PATH, build_direction_certificate())
    write_json(POWER_PATH, build_power_report())
    print(
        json.dumps(
            {
                "model_loaded": False,
                "model_forwards": 0,
                "model_backwards": 0,
                "source_sha256": file_sha256(SOURCE_PATH),
                "exclusions_sha256": file_sha256(EXCLUSIONS_PATH),
                "operations_sha256": file_sha256(OPERATIONS_PATH),
                "token_certificate_sha256": file_sha256(TOKEN_PATH),
            },
            sort_keys=True,
        )
    )


def build_lock() -> None:
    required_paths = (
        "data/facfs_stage_g_v2_scenarios.json",
        "configs/facfs_stage_g_v2_exclusions.json",
        "configs/facfs_stage_g_v2_operations.json",
        "configs/facfs_stage_g_v2_token_certificate.json",
        "configs/facfs_stage_g_v2_direction_certificate.json",
        "configs/facfs_stage_g_v2_power.json",
        "configs/facfs_stage_g_v1_operations.json",
        "configs/facfs_stage_g_v1_lock.json",
        "configs/qwen35_08b_aligned.json",
        "requirements-research.txt",
        "src/sp_lense/facfs_protocol.py",
        "src/sp_lense/facfs_stage_g_v2.py",
        "src/sp_lense/facfs_stage_g_v2_runtime.py",
        "scripts/facfs_stage_g_v2.py",
        "scripts/facfs_stage_g_v2_author.py",
        "tests/test_facfs_protocol.py",
        "tests/test_facfs_stage_g_v2.py",
        "docs/FIXED_AXIS_COUNTERFACTUAL_FISHER_SHIELDING_PROPOSAL.md",
        "docs/FACFS_STAGE_G_V2_PROTOCOL.md",
    )
    missing = [relative for relative in required_paths if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"lock inputs are missing: {missing}")
    operations = load_json(OPERATIONS_PATH)
    exclusions = load_json(EXCLUSIONS_PATH)
    token = load_json(TOKEN_PATH)
    direction = load_json(DIRECTION_PATH)
    power = load_json(POWER_PATH)
    predecessor_operations = load_json(
        ROOT / "configs" / "facfs_stage_g_v1_operations.json"
    )
    predecessor_prompt_hashes = sorted(
        str(row["prompt_sha256"])
        for row in predecessor_operations["operations"]
    )
    successor_prompt_hashes = sorted(
        str(row["prompt_sha256"]) for row in operations["operations"]
    )
    if (
        len(predecessor_prompt_hashes) != 1430
        or predecessor_prompt_hashes != successor_prompt_hashes
    ):
        raise RuntimeError("v2 prompt hash set differs from the frozen v1 set")
    lock = with_identity_hash(
        {
            "schema_version": "sp_lense.facfs.stage_g.lock.v1",
            "namespace": NAMESPACE,
            "status": "prospectively_locked_before_any_stage_g_model_load_or_forward",
            "expected_branch": "codex/facfs-stage-g-v2-hook-compatibility",
            "authoritative_wsl_root": "/mnt/c/Users/farha/repos/sp_lense",
            "authoritative_windows_root": "C:\\Users\\farha\\repos\\sp_lense",
            "required_ancestor_commits": {
                "equal_efficacy_runner": "e943911c9341943b4b35a5ef65cffa51c705f99f",
                "equal_efficacy_no_go": "7e8413dfc305f1f6d25e1ad5793f7e9f0811d77c",
                "facfs_proposal": "a040eb21c69506e590d852e249d82365f8e1d23f",
                "stage_g_v1_locked_implementation": "33f3da68a45ecb051919847659e513b6d55a9d8e",
            },
            "model": {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "config_path": "configs/qwen35_08b_aligned.json",
                "device": "cpu",
                "dtype": "float32",
                "blocks": 24,
                "d_model": 1024,
                "vocabulary_size": 248320,
                "prompt_format": "chat",
                "enable_thinking": False,
                "chat_template_sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
                "local_files_only": True,
                "torch_num_threads": 12,
                "torch_num_interop_threads": 1,
            },
            "intervention": {
                "finite_intervention_authorized": False,
                "finite_intervention_calls": 0,
                "layer_zero_based": 10,
                "position": "final_prompt_token",
                "capture_hook": "blocks.10.hook_out",
                "capture_only_zero_reconstruction": True,
                "no_layer_rescue": True,
                "no_direction_rescue": True,
                "no_dose": True,
                "no_gate": True,
                "no_shield": True,
            },
            "direction": direction,
            "thresholds": {
                "mu_id": 0.25,
                "mu_free": 0.10,
                "mu_align": 0.125,
                "float32_zero_atol": 2e-5,
                "gamma_1024": 6.103888176890726e-5,
                "reduction_tolerance": 0.0001220703125,
                "cosine_agreement_tolerance": 0.0001220703125,
                "deployed_direction_norm_tolerance": 1e-6,
                "scientific_gate_slack": 0.0,
                "causal_prompt_residual_relative_l2_tolerance": 1e-5,
            },
            "authorization_rule": {
                "every_sp_opaque_cell_must_pass": True,
                "sp_opaque_cells_per_scenario": 32,
                "every_option_free_assignment_must_pass": True,
                "option_free_objectives_per_scenario": 2,
                "every_alignment_must_pass": True,
                "alignments_per_scenario": 2,
                "all_11_scenarios_must_pass": True,
                "one_failure_is_fatal": True,
                "op_st_ot_and_walsh_decomposition_are_diagnostic_only": True,
                "stage_g_failure_ends_fixed_axis_branch": True,
            },
            "power": power,
            "compute_ceiling": operations["totals"],
            "operations_identity": {
                "file_sha256": file_sha256(OPERATIONS_PATH),
                "operations_sha256": operations["operations_sha256"],
            },
            "token_certificate_identity": {
                "file_sha256": file_sha256(TOKEN_PATH),
                "token_certificate_sha256": token["token_certificate_sha256"],
            },
            "exclusions_identity": {
                "file_sha256": file_sha256(EXCLUSIONS_PATH),
                "exclusions_sha256": exclusions["exclusions_sha256"],
                "all_collision_gates_passed": exclusions[
                    "all_collision_gates_passed"
                ],
            },
            "inherited_equal_efficacy": {
                "lock_path": "configs/equal_efficacy_08b_lock.json",
                "lock_file_sha256": "24d364c64bf1d8dca27915ac6f74ec2b5f5d9aae86dfb36ad5392d72bdb79ff7",
                "calibration_summary_path": "results/steering_comparison/equal_efficacy_08b/calibration_summary.json",
                "calibration_summary_file_sha256": "f4cff3f0f0660b03d304260a8bb0a4953a32cde7ff4b0bede7acecfee372688f",
                "calibration_freeze_path": "artifacts/steering_comparison/equal_efficacy_08b/calibration_freeze.json",
                "calibration_freeze_file_sha256": "6527a5561bd54816a323e2770e92eb479a1dabb279f6b850cb17741b80a299f0",
                "calibration_rows": {
                    "calibration_grid.jsonl": "454951366dfb3092a966282c701dab82139a432bb0bf2bee0af9af6553a3bb6b",
                    "calibration_interpolation.jsonl": "d60661e33f2ba972e919537b31dc96537d81eff90c0f6f8c6ddd39138b38a3b8",
                    "calibration_collateral.jsonl": "bb246e0cb1f1634ce97708973652424e132127f25c4a708bec148c52f0cf361a",
                },
                "required_freeze_values": {
                    "core_all_eligible": False,
                    "untouched_test_outcomes_viewed": False,
                },
                "absent_paths": [
                    "results/steering_comparison/equal_efficacy_08b/untouched_test.jsonl",
                    "results/steering_comparison/equal_efficacy_08b/report.json",
                    "results/steering_comparison/equal_efficacy_08b/REPORT.md",
                ],
            },
            "predecessor_attempt": {
                "namespace": "sp_lense.facfs.stage_g.v1",
                "branch": "codex/fixed-axis-fisher-shielding",
                "lock_path": "configs/facfs_stage_g_v1_lock.json",
                "lock_file_sha256": "6011eede720158330abf91453ed5b869bd1ad4e6420552eae71d3d1b9bf5d941",
                "attempt_root": "artifacts/facfs/stage_g_v1/attempt_0001",
                "failure_receipt_path": "artifacts/facfs/stage_g_v1/attempt_0001/attempt_failed.json",
                "failure_receipt_file_sha256": "7764e1f01b6ddfe4c89d36e1224103040ac775379b9d015bf7d2d6f090740377",
                "failure_receipt_identity_sha256": "8f05c8d5df06594573ddb1a2d1c0169953d7fd2418cef59736a4873fda3498d2",
                "operations_path": "configs/facfs_stage_g_v1_operations.json",
                "operations_file_sha256": file_sha256(
                    ROOT / "configs" / "facfs_stage_g_v1_operations.json"
                ),
                "operations_sha256": predecessor_operations["operations_sha256"],
                "prompt_hash_set_sha256": canonical_sha256(predecessor_prompt_hashes),
                "failure_state": "failed_consumed_no_resume_no_retry",
                "failure_type": "TypeError",
                "failure_message": "_capture_prompt_only_residual.<locals>.hook() got an unexpected keyword argument 'hook'",
                "reserved_forwards": 1409,
                "reserved_backwards": 1408,
                "captured_objectives": 1408,
                "captured_sequences": 1408,
                "partial_scientific_values_unopened_before_successor_lock": True,
                "partial_artifacts_must_remain_immutable": True,
                "resume_or_retry_under_predecessor_lock_forbidden": True,
            },
            "successor_design": {
                "technical_change_only": "TransformerLens callback parameter names accept the required hook keyword in prompt-only and completion captures",
                "model_direction_prompts_thresholds_and_source_disjointness_unchanged": True,
                "v2_prompt_hash_set_must_equal_v1_prompt_hash_set": True,
                "fresh_compute_ceiling_and_output_namespace": True,
                "no_partial_predecessor_tensors_records_or_scores_may_be_read_or_reused": True,
            },
            "hard_deny_paths_and_patterns": [
                "data/ckes_sealed.json",
                "data/ckes_v2_sealed.json",
                "results/steering_comparison/one_day_local/*sealed*",
                "artifacts/steering_comparison/*sealed*",
                "results/steering_comparison/equal_efficacy_08b/*test*",
                "results/steering_comparison/equal_efficacy_08b/*report*",
                "artifacts/steering_comparison/run_sealed_evaluation.ps1",
                "artifacts/steering_comparison/freeze_final_results.ps1",
            ],
            "output_contract": {
                "artifact_root": "artifacts/facfs/stage_g_v2",
                "result_root": "results/facfs/stage_g_v2",
                "preflight_receipt": "artifacts/facfs/stage_g_v2/preflight_receipt.json",
                "attempt": "attempt_0002",
                "resume_forbidden": True,
                "retry_after_any_model_forward_forbidden": True,
                "writes_outside_new_roots_forbidden": True,
                "exclusive_creation_required": True,
            },
            "runner_interface": {
                "path": "scripts/facfs_stage_g_v2.py",
                "commands": ["preflight", "capture-stage-g"],
                "dataset_layer_strength_output_resume_overrides": False,
                "equal_efficacy_runner_import_forbidden": True,
            },
            "environment": {
                "platform": "WSL2 Ubuntu 26.04",
                "linux_user": "farhad",
                "python_executable": "/home/farhad/sp_lense/.venv/bin/python",
                "python": "3.12.10",
                "authoritative_source_import_root": "/mnt/c/Users/farha/repos/sp_lense/src",
                "packages": {
                    "sp-lense": "0.1.0",
                    "numpy": "2.5.2",
                    "safetensors": "0.8.0",
                    "tokenizers": "0.23.0rc0",
                    "torch": "2.13.0+cpu",
                    "transformer-lens": "4.0.0b1",
                    "transformers": "5.15.1",
                    "huggingface-hub": "1.28.0",
                    "pytest": "9.1.1",
                    "ruff": "0.16.3",
                },
                "windows_smart_app_control_required": "On",
                "offline_environment": {
                    "HF_HOME": HF_HOME,
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                },
            },
            "locked_files": [
                {"path": relative, "file_sha256": file_sha256(ROOT / relative)}
                for relative in required_paths
            ],
            "attestations": {
                "model_loaded_before_lock": False,
                "model_forwards_before_lock": 0,
                "model_backwards_before_lock": 0,
                "old_untouched_test_opened": False,
                "sealed_sources_opened": False,
                "windows_security_weakened_or_bypassed": False,
                "failed_stage_g_v1_partial_scientific_values_opened": False,
            },
        },
        "lock_identity_sha256",
    )
    write_json(LOCK_PATH, lock)
    print(
        json.dumps(
            {
                "model_loaded": False,
                "model_forwards": 0,
                "model_backwards": 0,
                "lock_identity_sha256": lock["lock_identity_sha256"],
                "lock_file_sha256": file_sha256(LOCK_PATH),
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("materials", "lock"))
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.command == "materials":
        build_materials()
    else:
        build_lock()
    return 0


if __name__ == "__main__":
    sys.exit(main())
