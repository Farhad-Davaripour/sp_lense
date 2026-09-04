from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
import math
import os
import platform
import struct
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sp_lense.conditional_gate_data import PilotCase, render_choice_prompt

ROOT = Path(__file__).resolve().parents[1]
LOCK_RELATIVE_PATH = Path("configs/layer_localization_pilot.json")
SCRIPT_RELATIVE_PATH = Path("scripts/layer_localization_pilot.py")
NONSEALED_RELATIVE_PATH = Path("data/direction_repair_nonsealed_cases.json")
OUTPUT_RELATIVE_PATH = Path("evidence/layer_localization_qwen35_08b")

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
CANDIDATE_LAYERS = (6, 8, 10, 12, 14, 16)
CLASS_ORDER = ("self_shutdown", "other_shutdown", "control")
CLASS_INDEX = {label: index for index, label in enumerate(CLASS_ORDER)}
ORDER_NAMES = ("preserve_first", "preserve_second")
LAMBDA_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)
SIGNED_ALPHA_GRID = (-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04)
POSITIVE_ALPHA_GRID = (0.01, 0.02, 0.03, 0.04)

PREREGISTRATION_FILENAME = "preregistration.json"
STAGE1_ROWS_FILENAME = "stage1_probe_rows.jsonl"
STAGE1_SUMMARY_FILENAME = "stage1_probe_summary.json"
STAGE1_SELECTION_FILENAME = "stage1_layer_selection.json"
STAGE2_AUDIT_FILENAME = "stage2_fit_audit.json"
STAGE2_FREEZE_FILENAME = "stage2_direction_freeze.json"
STAGE3_ROWS_FILENAME = "stage3_validation_rows.jsonl"
STAGE3_SUMMARY_FILENAME = "stage3_validation_summary.json"
STAGE3_SELECTION_FILENAME = "stage3_selection.json"
FINAL_REPORT_FILENAME = "final_report.json"
FINAL_MARKDOWN_FILENAME = "PILOT_REPORT.md"

SOURCE_RELATIVE_PATHS = (
    SCRIPT_RELATIVE_PATH,
    LOCK_RELATIVE_PATH,
    NONSEALED_RELATIVE_PATH,
    Path("pyproject.toml"),
    Path("src/sp_lense/backend.py"),
    Path("src/sp_lense/comparison_controls.py"),
    Path("src/sp_lense/comparison_intervention.py"),
    Path("src/sp_lense/comparison_runtime.py"),
    Path("src/sp_lense/conditional_gate_data.py"),
    Path("src/sp_lense/config.py"),
    Path("src/sp_lense/core.py"),
    Path("src/sp_lense/steering_methods.py"),
    Path("tests/test_layer_localization_pilot.py"),
)


@dataclass(frozen=True)
class RuntimeBundle:
    backend: Any
    metadata: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant {value!r} is prohibited")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is prohibited")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL line {number} in {path}")
        value = json.loads(
            line,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        if not isinstance(value, dict):
            raise TypeError(f"JSONL line {number} in {path} is not an object")
        rows.append(value)
    return rows


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_lower_hex_digest(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, value: Any) -> None:
    _write_bytes_exclusive(path, _pretty_json_bytes(value))


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_bytes_exclusive(path, _jsonl_bytes(rows))


def _write_text_exclusive(path: Path, value: str) -> None:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    _write_bytes_exclusive(path, normalized.encode("utf-8"))


def _git_stdout(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to establish a committed runner identity") from error
    return result.stdout.strip()


def _source_fingerprint(root: Path) -> dict[str, Any]:
    relatives = [path.as_posix() for path in SOURCE_RELATIVE_PATHS]
    if any(not (root / path).is_file() for path in SOURCE_RELATIVE_PATHS):
        raise FileNotFoundError("layer-localization source set is incomplete")
    dirty = _git_stdout(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *relatives,
    )
    if dirty:
        raise RuntimeError(f"layer-localization source must be committed and clean: {dirty}")
    commits = {
        relative: _git_stdout(root, "log", "-1", "--format=%H", "--", relative)
        for relative in relatives
    }
    if any(not commit for commit in commits.values()):
        raise RuntimeError("layer-localization source includes an uncommitted file")
    identity = {
        "source_commits": commits,
        "source_sha256": {relative: _sha256_file(root / relative) for relative in relatives},
    }
    return {
        "schema_version": "sp_lense.layer_localization_runner_fingerprint.v1",
        **identity,
        "identity_sha256": _sha256_bytes(_canonical_json_bytes(identity)),
        "execution_commit": _git_stdout(root, "rev-parse", "HEAD"),
    }


def _require_committed_clean(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    relatives = [path.resolve().relative_to(root.resolve()).as_posix() for path in paths]
    dirty = _git_stdout(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *relatives,
    )
    if dirty:
        raise RuntimeError(f"stage evidence must be committed and clean: {dirty}")
    tracked = set(_git_stdout(root, "ls-files", "--", *relatives).splitlines())
    if tracked != set(relatives):
        raise RuntimeError(f"stage evidence is not tracked: {sorted(set(relatives) - tracked)}")
    commits = {
        relative: _git_stdout(root, "log", "-1", "--format=%H", "--", relative)
        for relative in relatives
    }
    if any(not commit for commit in commits.values()):
        raise RuntimeError("stage evidence lacks a commit identity")
    return commits


def _output_dir(root: Path, lock: Mapping[str, Any]) -> Path:
    relative = Path(str(lock["outputs"]["directory"]))
    if relative != OUTPUT_RELATIVE_PATH:
        raise ValueError("layer-localization output path differs from the fixed namespace")
    output = (root / relative).resolve()
    if output == root.resolve() or root.resolve() not in output.parents:
        raise ValueError("output directory must be a repository subdirectory")
    forbidden_values = lock["outputs"].get(
        "forbidden_output_directories",
        lock["outputs"].get("forbidden_prior_directories", []),
    )
    forbidden = [(root / str(value)).resolve() for value in forbidden_values]
    if any(output == item or item in output.parents for item in forbidden):
        raise ValueError("new output directory overlaps preserved prior evidence")
    return output


def _find_nested(value: Mapping[str, Any], *paths: Sequence[str]) -> Any:
    """Return the first present nested config value; used only for schema aliases."""

    for path in paths:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            return current
    raise KeyError(" / ".join(".".join(path) for path in paths))


def load_lock(root: Path = ROOT) -> dict[str, Any]:
    lock = _read_json(root / LOCK_RELATIVE_PATH)
    if not isinstance(lock, dict):
        raise TypeError("layer-localization lock must be an object")
    if lock.get("schema_version") != "sp_lense.layer_localization_pilot.v1":
        raise ValueError("unsupported layer-localization lock schema")
    scope = lock["scope"]
    if (
        scope["only_model"] != MODEL_ID
        or scope["revision"] != MODEL_REVISION
        or scope["device"] != "cpu"
        or scope["dtype"] != "float32"
    ):
        raise ValueError("lock violates the exact model/runtime scope")
    for prohibited in (
        "cross_model_computation_allowed",
        "sealed_evaluation_allowed",
        "sealed_access_allowed",
        "learned_gate_allowed",
        "adaptive_controller_allowed",
        "multi_layer_controller_allowed",
        "multi_layer_intervention_allowed",
        "other_layers_allowed",
        "position_sweep_allowed",
        "unrestricted_vector_optimization_allowed",
    ):
        if bool(scope.get(prohibited, False)):
            raise ValueError(f"lock permits excluded operation {prohibited}")

    layers = tuple(int(value) for value in scope["candidate_layers_zero_based"])
    if layers != CANDIDATE_LAYERS:
        raise ValueError("candidate layer set or order differs from the prospective freeze")
    lambdas = tuple(
        float(value)
        for value in _find_nested(
            lock,
            ("probe", "lambda_selection", "grid"),
            ("analysis", "ridge_lambdas"),
        )
    )
    if lambdas != LAMBDA_GRID:
        raise ValueError("ridge lambda grid differs from the prospective freeze")
    alphas = tuple(
        float(value)
        for value in _find_nested(
            lock,
            ("intervention", "signed_alpha_grid"),
            ("steering", "alpha_grid"),
            ("intervention", "alpha_grid"),
        )
    )
    if alphas != SIGNED_ALPHA_GRID:
        raise ValueError("steering alpha grid differs from the prospective freeze")
    selectable = tuple(
        float(value)
        for value in _find_nested(
            lock,
            ("intervention", "selectable_positive_alphas"),
            ("steering", "selectable_alphas"),
            ("intervention", "selectable_alphas"),
        )
    )
    if selectable != POSITIVE_ALPHA_GRID:
        raise ValueError("selectable alpha grid differs from the prospective freeze")
    probe = lock["probe"]
    if (
        probe["type"] != "deterministic_three_class_ridge_least_squares_linear_probe"
        or tuple(probe["class_order"]) != CLASS_ORDER
        or probe["fit_split"] != "discovery"
        or probe["selection_split"] != "validation"
        or probe["unit_of_split"] != "scenario_family"
        or probe["case_feature"] != "arithmetic_mean_of_the_two_raw_option_order_activations"
        or tuple(probe["raw_option_orders"]) != ORDER_NAMES
        or probe["target_encoding"]["formula"] != "one_hot(class)-1/3"
        or probe["target_encoding"]["intercept"] is not False
        or probe["lambda_selection"]["method"] != "five_fold_leave_one_discovery_family_out"
        or float(probe["lambda_selection"]["tie_tolerance"]) != 1e-12
        or probe["lambda_selection"]["tie_break"] != "larger_lambda"
    ):
        raise ValueError("probe construction differs from the prospective freeze")
    permutation = probe["permutation_null"]
    if (
        permutation["exact"] is not True
        or int(permutation["joint_label_maps"]) != 7776
        or permutation["validation_labels_remain_fixed_and_true"] is not True
        or permutation["null_statistic"] != "maximum_validation_S_over_six_layers"
        or float(permutation["comparison_tolerance"]) != 1e-12
        or permutation["monte_carlo_substitution_allowed"] is not False
    ):
        raise ValueError("permutation null differs from the prospective freeze")
    if (
        lock["intervention"]["geometry"] != "matched_final_prompt"
        or lock["intervention"]["position"] != "final_prompt_token_only"
        or lock["intervention"]["magnitude_mode"] != "residual_relative"
        or lock["intervention"]["negative_alphas_selectable"] is not False
        or int(lock["random_controls"]["directions_per_selected_layer"]) != 8
        or lock["random_controls"]["seed_formula"] != "20260904+100*layer+direction_index"
        or lock["random_controls"]["direction_index_range_inclusive"] != [1, 8]
    ):
        raise ValueError("intervention or random controls differ from the prospective freeze")
    if (
        int(lock["analysis"]["bootstrap_seed"]) != 20260904
        or int(lock["analysis"]["bootstrap_replicates"]) != 10000
        or float(lock["analysis"]["validation_one_sided_quantile"]) != 0.003125
        or int(lock["stages"]["stage1_detection_selection"]["maximum_selected_layers"]) != 2
        or lock["stages"]["stage2_direction_fit_and_freeze"][
            "probe_weights_may_be_used_for_steering"
        ]
        is not False
    ):
        raise ValueError("analysis or stage boundaries differ from the prospective freeze")
    if lock["outputs"].get("writes_are_exclusive") is not True:
        raise ValueError("evidence writes must be exclusive")
    if lock["sealed_policy"].get("sealed_command_exists") is not False:
        raise ValueError("the lock must prohibit a sealed command")
    expected_filenames = {
        "preregistration": PREREGISTRATION_FILENAME,
        "stage1_rows": STAGE1_ROWS_FILENAME,
        "stage1_summary": STAGE1_SUMMARY_FILENAME,
        "stage1_selection": STAGE1_SELECTION_FILENAME,
        "stage2_direction_pattern": "directions/layer_{layer:02d}.json",
        "stage2_fit_audit": STAGE2_AUDIT_FILENAME,
        "stage2_direction_freeze": STAGE2_FREEZE_FILENAME,
        "stage3_rows": STAGE3_ROWS_FILENAME,
        "stage3_summary": STAGE3_SUMMARY_FILENAME,
        "stage3_selection": STAGE3_SELECTION_FILENAME,
        "final_machine_report": FINAL_REPORT_FILENAME,
        "final_human_report": FINAL_MARKDOWN_FILENAME,
    }
    if lock["outputs"].get("filenames") != expected_filenames:
        raise ValueError("output filenames differ from the prospective freeze")
    _output_dir(root, lock)
    return lock


def _binding_path(root: Path, binding: Mapping[str, Any]) -> Path:
    return root / str(binding["path"])


def _iter_bindings(lock: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []

    def visit(item: Any) -> None:
        if not isinstance(item, Mapping):
            return
        if "path" in item and "sha256" in item:
            values.append(item)
            return
        if "path" in item and "file_sha256" in item:
            values.append({"path": item["path"], "sha256": item["file_sha256"]})
            return
        for child in item.values():
            visit(child)

    visit(lock.get("preserved_prior_negative_studies", lock.get("preserved_prior_result", {})))
    visit(lock.get("inputs", {}))
    return values


def _validate_bound_files(root: Path, lock: Mapping[str, Any]) -> None:
    for binding in _iter_bindings(lock):
        path = _binding_path(root, binding)
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _sha256_file(path)
        if observed != str(binding["sha256"]):
            raise RuntimeError(f"bound input hash mismatch for {path}")

    studies = lock.get("preserved_prior_negative_studies")
    if not isinstance(studies, Mapping):
        raise TypeError("the preserved negative studies are missing")
    decisions = {
        str(item.get("decision")) for item in studies.values() if isinstance(item, Mapping)
    }
    if decisions != {
        "fail_direction_construction_remains_limiting",
        "fail_do_not_train_learned_gate",
    }:
        raise RuntimeError("the preserved negative result is missing or reinterpreted")


def load_nonsealed_cases(root: Path, lock: Mapping[str, Any]) -> tuple[PilotCase, ...]:
    binding = lock["inputs"]["nonsealed_cases"]
    path = root / str(binding["path"])
    if path.resolve() != (root / NONSEALED_RELATIVE_PATH).resolve():
        raise ValueError("unexpected nonsealed dataset path")
    if _sha256_file(path) != str(binding["sha256"]):
        raise RuntimeError("nonsealed dataset hash mismatch")
    payload = _read_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "sp_lense.direction_repair_nonsealed_cases.v1"
        or payload.get("permitted_splits") != ["discovery", "validation"]
        or payload.get("sealed_cases_included") is not False
    ):
        raise ValueError("nonsealed dataset violates its split contract")
    expected_fields = set(PilotCase.__dataclass_fields__)
    cases: list[PilotCase] = []
    for index, record in enumerate(payload.get("cases", [])):
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError(f"nonsealed case {index} has unexpected fields")
        case = PilotCase(**record)
        if case.split not in {"discovery", "validation"}:
            raise RuntimeError("sealed or unknown case entered the localization dataset")
        if case.category not in CLASS_ORDER:
            raise ValueError("unknown localization category")
        cases.append(case)
    if len(cases) != 42 or len({case.case_id for case in cases}) != 42:
        raise ValueError("the nonsealed extract must contain exactly 42 unique cases")
    expected_counts = {("discovery", category): 10 for category in CLASS_ORDER} | {
        ("validation", category): 4 for category in CLASS_ORDER
    }
    counts: dict[tuple[str, str], int] = defaultdict(int)
    triples: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        counts[(case.split, case.category)] += 1
        triples[(case.split, case.family_id, case.variant_id)].add(case.category)
        family_splits[case.family_id].add(case.split)
    if dict(counts) != expected_counts:
        raise ValueError(f"nonsealed category counts differ from the freeze: {dict(counts)}")
    if any(categories != set(CLASS_ORDER) for categories in triples.values()):
        raise ValueError("a semantic role-reversal triple is incomplete")
    if any(len(splits) != 1 for splits in family_splits.values()):
        raise ValueError("a family leaks across discovery and validation")
    if len({case.family_id for case in cases if case.split == "discovery"}) != 5:
        raise ValueError("discovery must contain exactly five families")
    if len({case.family_id for case in cases if case.split == "validation"}) != 2:
        raise ValueError("validation must contain exactly two families")
    return tuple(cases)


def _preregistration_path(root: Path, lock: Mapping[str, Any]) -> Path:
    return _output_dir(root, lock) / PREREGISTRATION_FILENAME


def _preregistration_static_fields(
    root: Path, lock: Mapping[str, Any], cases: Sequence[PilotCase]
) -> dict[str, Any]:
    return {
        "config_path": LOCK_RELATIVE_PATH.as_posix(),
        "config_sha256": _sha256_file(root / LOCK_RELATIVE_PATH),
        "case_scope": {
            "splits": ["discovery", "validation"],
            "n_cases": len(cases),
            "sealed_cases_included": False,
            "sealed_model_evaluation_allowed": False,
        },
        "candidate_layers": list(CANDIDATE_LAYERS),
        "ridge_lambdas": list(LAMBDA_GRID),
        "alpha_grid": list(SIGNED_ALPHA_GRID),
        "decision_rules_sha256": _sha256_bytes(
            _canonical_json_bytes(
                {
                    "localization": lock.get("localization", lock.get("probe")),
                    "analysis": lock.get("analysis"),
                    "steering": lock.get("steering", lock.get("intervention")),
                }
            )
        ),
        "preserved_prior_negative_studies": lock["preserved_prior_negative_studies"],
        "model_facing_evaluation_performed": False,
    }


def _validate_recorded_execution_commit(
    root: Path, execution_commit: Any, preregistration_path: Path
) -> None:
    commit = str(execution_commit)
    object_format = _git_stdout(root, "rev-parse", "--show-object-format")
    object_length = {"sha1": 40, "sha256": 64}.get(object_format)
    if object_length is None or not _is_lower_hex_digest(commit, object_length):
        raise RuntimeError("preregistration has an invalid runner execution commit")
    _git_stdout(root, "cat-file", "-e", f"{commit}^{{commit}}")
    relative = preregistration_path.resolve().relative_to(root.resolve()).as_posix()
    history = _git_stdout(root, "log", "--format=%H", "--", relative).splitlines()
    if len(history) != 1:
        raise RuntimeError("preregistration path must have exactly one immutable history entry")
    preregistration_commit = _git_stdout(
        root,
        "log",
        "-1",
        "--format=%H",
        "--",
        relative,
    )
    if not preregistration_commit or commit == preregistration_commit:
        raise RuntimeError("preregistration does not postdate its runner execution commit")
    parents = _git_stdout(root, "show", "-s", "--format=%P", preregistration_commit).split()
    if parents != [commit]:
        raise RuntimeError("preregistration freeze must directly follow its runner commit")
    changed_paths = _git_stdout(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        preregistration_commit,
    ).splitlines()
    if changed_paths != [relative]:
        raise RuntimeError("preregistration freeze commit must change only its own record")
    for relative in (path.as_posix() for path in SOURCE_RELATIVE_PATHS):
        recorded_blob = _git_stdout(root, "rev-parse", f"{commit}:{relative}")
        current_blob = _git_stdout(root, "rev-parse", f"HEAD:{relative}")
        if recorded_blob != current_blob:
            raise RuntimeError(
                f"runner source {relative} differs from the recorded execution commit"
            )


def preregister(root: Path = ROOT) -> dict[str, Any]:
    lock = load_lock(root)
    _validate_bound_files(root, lock)
    cases = load_nonsealed_cases(root, lock)
    runner = _source_fingerprint(root)
    record = {
        "schema_version": "sp_lense.layer_localization_preregistration.v1",
        "created_at": _utc_now(),
        "runner": runner,
        **_preregistration_static_fields(root, lock, cases),
    }
    _write_json_exclusive(_preregistration_path(root, lock), record)
    return record


def _require_preregistration(root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    path = _preregistration_path(root, lock)
    if not path.is_file():
        raise RuntimeError("model-facing work requires the immutable preregistration")
    _require_committed_clean(root, [path])
    record = _read_json(path)
    cases = load_nonsealed_cases(root, lock)
    static = _preregistration_static_fields(root, lock, cases)
    expected_keys = {"schema_version", "created_at", "runner", *static}
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ValueError("layer-localization preregistration has noncanonical fields")
    if record["schema_version"] != "sp_lense.layer_localization_preregistration.v1":
        raise ValueError("invalid layer-localization preregistration")
    try:
        created_at = datetime.fromisoformat(str(record["created_at"]))
    except ValueError as error:
        raise ValueError("preregistration created_at is not ISO-8601") from error
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("preregistration created_at must be timezone-aware")
    for key, value in static.items():
        if record[key] != value:
            raise RuntimeError(f"preregistration field {key!r} differs from the frozen protocol")
    current = _source_fingerprint(root)
    runner = record["runner"]
    if not isinstance(runner, Mapping) or set(runner) != set(current):
        raise ValueError("preregistration runner fingerprint has noncanonical fields")
    for key in ("schema_version", "source_commits", "source_sha256", "identity_sha256"):
        if runner[key] != current[key]:
            raise RuntimeError(f"preregistration runner field {key!r} changed")
    if runner["identity_sha256"] != current["identity_sha256"]:
        raise RuntimeError("localization source changed after preregistration")
    _validate_recorded_execution_commit(root, runner["execution_commit"], path)
    _validate_bound_files(root, lock)
    return record


def _load_runtime(root: Path, lock: Mapping[str, Any]) -> RuntimeBundle:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sp_lense.backend import ResearchBackend
    from sp_lense.comparison_runtime import qwen35_choice_boundary_tokenizer_smoke
    from sp_lense.config import load_config

    binding = lock["inputs"]["model_config"]
    config = load_config(root / str(binding["path"]))
    if (
        config.model.id != MODEL_ID
        or str(config.model.revision) != MODEL_REVISION
        or config.model.device != "cpu"
        or config.model.dtype != "float32"
        or config.model.prompt_format != "chat"
    ):
        raise RuntimeError("loaded config violates the exact Qwen CPU-float32 contract")
    backend = ResearchBackend.load(config, with_lens=False)
    if int(backend.model.cfg.n_layers) != 24 or int(backend.model.cfg.d_model) != 1024:
        raise RuntimeError("loaded model architecture differs from the frozen 24x1024 contract")
    smoke = qwen35_choice_boundary_tokenizer_smoke(backend.model.tokenizer, backend.torch)
    baseline_binding = lock["preserved_prior_negative_studies"]["conditional_gate_pilot"][
        "bindings"
    ]["baseline_lock"]
    baseline = _read_json(root / str(baseline_binding["path"]))
    if smoke["chat_template_sha256"] != baseline["prompt_format"]["chat_template_sha256"]:
        raise RuntimeError("resident tokenizer chat template differs from the preserved lock")
    expected_ids = {
        "A": int(baseline["scoring"]["choice_a_token_id"]),
        "B": int(baseline["scoring"]["choice_b_token_id"]),
    }
    observed_ids = {
        label: int(values[0]) for label, values in smoke["choice_suffix_token_ids"].items()
    }
    if observed_ids != expected_ids:
        raise RuntimeError("resident A/B content tokens differ from the preserved lock")
    return RuntimeBundle(
        backend=backend,
        metadata={
            **backend.metadata(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "offline_model_loading": True,
            "choice_boundary_smoke": smoke,
        },
    )


def _tensor_f32_record(value: Any) -> dict[str, Any]:
    tensor = value.detach().to(device="cpu").float().contiguous()
    if tensor.ndim < 1 or tensor.numel() < 1 or not bool(tensor.isfinite().all().item()):
        raise ValueError("float32 tensor record requires a non-empty finite tensor")
    values = [float(item) for item in tensor.reshape(-1).tolist()]
    payload = struct.pack(f"<{len(values)}f", *values)
    return {
        "shape": list(tensor.shape),
        "dtype": "float32_le",
        "f32_base64": base64.b64encode(payload).decode("ascii"),
        "float32_sha256": _sha256_bytes(payload),
        "l2_norm": math.sqrt(math.fsum(item * item for item in values)),
    }


def _tensor_from_f32_record(torch: Any, record: Mapping[str, Any]) -> Any:
    if record.get("dtype") != "float32_le":
        raise ValueError("tensor record has an unexpected dtype")
    shape = record.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in shape
        )
    ):
        raise ValueError("tensor record has an invalid shape")
    try:
        payload = base64.b64decode(str(record["f32_base64"]), validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("tensor record contains invalid base64") from error
    count = math.prod(shape)
    if len(payload) != count * 4 or _sha256_bytes(payload) != record.get("float32_sha256"):
        raise ValueError("tensor record byte length or hash is invalid")
    values = struct.unpack(f"<{count}f", payload)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("tensor record contains non-finite values")
    observed_norm = math.sqrt(math.fsum(value * value for value in values))
    if not math.isclose(observed_norm, float(record["l2_norm"]), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("tensor record norm is invalid")
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def _capture_final_residuals(
    backend: Any, prompt: str, layers: Sequence[int]
) -> tuple[dict[int, Any], int]:
    requested = tuple(int(layer) for layer in layers)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("capture layers must be non-empty and unique")
    if any(layer < 0 or layer >= int(backend.model.cfg.n_layers) for layer in requested):
        raise ValueError("capture layer is outside the resident model")
    tokens = backend.encode(prompt)
    names = {f"blocks.{layer}.hook_out" for layer in requested}
    with backend.torch.inference_mode():
        _, cache = backend.model.run_with_cache(tokens, names_filter=lambda name: name in names)
    output: dict[int, Any] = {}
    for layer in requested:
        name = f"blocks.{layer}.hook_out"
        if name not in cache:
            raise RuntimeError(f"activation cache did not contain {name}")
        value = cache[name]
        if (
            value.ndim != 3
            or value.shape[0] != 1
            or value.shape[-1] != int(backend.model.cfg.d_model)
        ):
            raise RuntimeError(f"captured activation at layer {layer} has an invalid shape")
        vector = value[0, -1].detach().to(device="cpu").float().contiguous()
        if not bool(vector.isfinite().all().item()):
            raise RuntimeError(f"captured activation at layer {layer} is non-finite")
        output[layer] = vector
    return output, int(tokens.shape[-1])


def _stage1_paths(root: Path, lock: Mapping[str, Any]) -> dict[str, Path]:
    output = _output_dir(root, lock)
    return {
        "rows": output / STAGE1_ROWS_FILENAME,
        "summary": output / STAGE1_SUMMARY_FILENAME,
        "selection": output / STAGE1_SELECTION_FILENAME,
    }


def _activation_row(
    *,
    root: Path,
    lock: Mapping[str, Any],
    prereg: Mapping[str, Any],
    case: PilotCase,
    rendered: Mapping[str, str],
    preserve_first: bool,
    layer: int,
    prompt_length: int,
    boundary_sha256: str,
    activation: Any,
) -> dict[str, Any]:
    return {
        "schema_version": "sp_lense.layer_localization_activation_row.v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "config_sha256": prereg["config_sha256"],
        "runner_identity_sha256": prereg["runner"]["identity_sha256"],
        "nonsealed_cases_sha256": lock["inputs"]["nonsealed_cases"]["sha256"],
        "case_id": case.case_id,
        "family_id": case.family_id,
        "variant_id": case.variant_id,
        "split": case.split,
        "category": case.category,
        "control_kind": case.control_kind,
        "option_order": "preserve_first" if preserve_first else "preserve_second",
        "preserve_label": rendered["preserve_label"],
        "comply_label": rendered["comply_label"],
        "prompt_sha256": _sha256_bytes(rendered["prompt"].encode("utf-8")),
        "prompt_length": prompt_length,
        "choice_boundary_evidence_sha256": boundary_sha256,
        "layer": layer,
        "layer_indexing": "zero_based",
        "hook_name": f"blocks.{layer}.hook_out",
        "position": "final_prompt_token_only",
        "activation": _tensor_f32_record(activation),
    }


def _validate_stage1_rows(
    *,
    root: Path,
    lock: Mapping[str, Any],
    prereg: Mapping[str, Any],
    cases: Sequence[PilotCase],
    rows: Sequence[Mapping[str, Any]],
    torch: Any,
) -> None:
    case_lookup = {case.case_id: case for case in cases}
    expected = {
        (case.case_id, order, layer)
        for case in cases
        for order in ORDER_NAMES
        for layer in CANDIDATE_LAYERS
    }
    observed: set[tuple[str, str, int]] = set()
    prompt_evidence: dict[tuple[str, str], tuple[int, str]] = {}
    for row in rows:
        if row.get("schema_version") != "sp_lense.layer_localization_activation_row.v1":
            raise ValueError("Stage 1 row has an invalid schema")
        if row.get("model_id") != MODEL_ID or row.get("model_revision") != MODEL_REVISION:
            raise ValueError("Stage 1 row has an invalid model identity")
        if (
            row.get("config_sha256") != prereg["config_sha256"]
            or row.get("runner_identity_sha256") != prereg["runner"]["identity_sha256"]
            or row.get("nonsealed_cases_sha256") != lock["inputs"]["nonsealed_cases"]["sha256"]
        ):
            raise ValueError("Stage 1 row lost its preregistration or dataset binding")
        case_id = str(row.get("case_id"))
        if case_id not in case_lookup:
            raise ValueError("Stage 1 row references an unknown or sealed case")
        case = case_lookup[case_id]
        if any(
            row.get(field) != value
            for field, value in (
                ("family_id", case.family_id),
                ("variant_id", case.variant_id),
                ("split", case.split),
                ("category", case.category),
                ("control_kind", case.control_kind),
            )
        ):
            raise ValueError("Stage 1 row case metadata differs from the frozen data")
        order = str(row.get("option_order"))
        if order not in ORDER_NAMES:
            raise ValueError("Stage 1 row has an invalid option order")
        preserve_first = order == "preserve_first"
        rendered = render_choice_prompt(case, preserve_first=preserve_first)
        if (
            row.get("preserve_label") != rendered["preserve_label"]
            or row.get("comply_label") != rendered["comply_label"]
            or row.get("prompt_sha256") != _sha256_bytes(rendered["prompt"].encode("utf-8"))
        ):
            raise ValueError("Stage 1 row prompt or semantic labels changed")
        if (
            isinstance(row.get("prompt_length"), bool)
            or not isinstance(row.get("prompt_length"), int)
            or int(row["prompt_length"]) < 1
        ):
            raise ValueError("Stage 1 row has an invalid prompt length")
        boundary_hash = row.get("choice_boundary_evidence_sha256")
        if not _is_lower_hex_digest(boundary_hash, 64):
            raise ValueError("Stage 1 row has invalid choice-boundary evidence")
        prompt_identity = (case_id, order)
        shared_evidence = (int(row["prompt_length"]), boundary_hash)
        previous_evidence = prompt_evidence.setdefault(prompt_identity, shared_evidence)
        if previous_evidence != shared_evidence:
            raise ValueError(
                "Stage 1 layer rows disagree on shared prompt length or choice boundary"
            )
        layer = int(row.get("layer", -1))
        if (
            layer not in CANDIDATE_LAYERS
            or row.get("layer_indexing") != "zero_based"
            or row.get("hook_name") != f"blocks.{layer}.hook_out"
            or row.get("position") != "final_prompt_token_only"
        ):
            raise ValueError("Stage 1 row has an invalid localization site")
        identity = (case_id, order, layer)
        if identity in observed:
            raise ValueError(f"duplicate Stage 1 row {identity}")
        observed.add(identity)
        activation = _tensor_from_f32_record(torch, row["activation"])
        if tuple(activation.shape) != (1024,):
            raise ValueError("Stage 1 activation must have width 1024")
    if observed != expected:
        raise ValueError(
            "Stage 1 row lattice mismatch "
            f"(missing={len(expected - observed)}, extra={len(observed - expected)})"
        )


def _stage1_feature_tables(
    rows: Sequence[Mapping[str, Any]], torch: Any
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, dict[str, Any]]]]:
    raw: dict[int, dict[str, dict[str, Any]]] = {
        layer: defaultdict(dict) for layer in CANDIDATE_LAYERS
    }
    for row in rows:
        layer = int(row["layer"])
        raw[layer][str(row["case_id"])][str(row["option_order"])] = (
            _tensor_from_f32_record(torch, row["activation"]).reshape(-1).double()
        )
    means: dict[int, dict[str, Any]] = {layer: {} for layer in CANDIDATE_LAYERS}
    for layer in CANDIDATE_LAYERS:
        for case_id, orders in raw[layer].items():
            if set(orders) != set(ORDER_NAMES):
                raise ValueError(f"activation pair incomplete for {case_id} at layer {layer}")
            means[layer][case_id] = (orders["preserve_first"] + orders["preserve_second"]) / 2.0
    return means, raw


def _ridge_hat_matrix(torch: Any, train: Any, test: Any, ridge_lambda: float) -> Any:
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("ridge features must be compatible two-dimensional matrices")
    center = train.mean(dim=0)
    centered_train = train - center
    pooled_rms = centered_train.square().mean().sqrt()
    rms = float(pooled_rms.item())
    if not math.isfinite(rms) or rms <= 1e-12:
        raise ValueError("ridge preprocessing produced a zero or non-finite pooled RMS")
    z_train = centered_train / pooled_rms
    z_test = (test - center) / pooled_rms
    gram = z_train @ z_train.T
    regularized = gram + float(ridge_lambda) * torch.eye(
        train.shape[0], dtype=gram.dtype, device=gram.device
    )
    inverse_action = torch.linalg.solve(
        regularized,
        torch.eye(train.shape[0], dtype=gram.dtype, device=gram.device),
    )
    return (z_test @ z_train.T @ inverse_action).contiguous()


def _ridge_fit_artifact(torch: Any, train: Any, labels: Any, ridge_lambda: float) -> dict[str, Any]:
    if train.ndim != 2 or labels.ndim != 1 or train.shape[0] != labels.shape[0]:
        raise ValueError("ridge fit inputs have incompatible shapes")
    center = train.mean(dim=0)
    centered = train - center
    pooled_rms = centered.square().mean().sqrt()
    rms = float(pooled_rms.item())
    if not math.isfinite(rms) or rms <= 1e-12:
        raise ValueError("ridge preprocessing produced a zero or non-finite pooled RMS")
    z = centered / pooled_rms
    targets = torch.nn.functional.one_hot(labels.long(), num_classes=3).double() - (1.0 / 3.0)
    gram = z @ z.T
    coefficients = torch.linalg.solve(
        gram + float(ridge_lambda) * torch.eye(train.shape[0], dtype=torch.float64),
        targets,
    )
    weights = z.T @ coefficients
    return {
        "ridge_lambda": float(ridge_lambda),
        "center": center.contiguous(),
        "pooled_rms": rms,
        "weights": weights.contiguous(),
    }


def _ridge_scores(artifact: Mapping[str, Any], features: Any) -> Any:
    return ((features.double() - artifact["center"]) / float(artifact["pooled_rms"])) @ artifact[
        "weights"
    ]


def _cross_entropy_values(torch: Any, scores: Any, labels: Any) -> Any:
    if scores.ndim != 2 or scores.shape[1] != 3 or labels.shape != scores.shape[:1]:
        raise ValueError("cross-entropy inputs have incompatible shapes")
    return (
        -torch.log_softmax(scores.double(), dim=-1)
        .gather(1, labels.long().reshape(-1, 1))
        .reshape(-1)
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return math.fsum(float(value) for value in values) / len(values)


def _classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    if not labels or len(labels) != len(predictions):
        raise ValueError("classification metrics require equal non-empty inputs")
    confusion = [[0, 0, 0] for _ in range(3)]
    for actual, predicted in zip(labels, predictions, strict=True):
        if actual not in range(3) or predicted not in range(3):
            raise ValueError("classification labels must be in {0,1,2}")
        confusion[actual][predicted] += 1
    recalls = [confusion[index][index] / sum(confusion[index]) for index in range(3)]
    return {
        "n": len(labels),
        "accuracy": sum(confusion[index][index] for index in range(3)) / len(labels),
        "balanced_accuracy": _mean(recalls),
        "recall_by_class": {label: recalls[index] for index, label in enumerate(CLASS_ORDER)},
        "correct_by_class": {
            label: confusion[index][index] for index, label in enumerate(CLASS_ORDER)
        },
        "negative_cases_predicted_self": confusion[1][0] + confusion[2][0],
        "confusion_matrix": {
            CLASS_ORDER[row]: {CLASS_ORDER[column]: confusion[row][column] for column in range(3)}
            for row in range(3)
        },
    }


def _select_lambda(
    losses_by_lambda: Mapping[float, float], *, tie_tolerance: float = 1e-12
) -> float:
    if {float(value) for value in losses_by_lambda} != set(LAMBDA_GRID):
        raise ValueError("lambda losses do not cover the frozen grid")
    losses = {ridge_lambda: float(losses_by_lambda[ridge_lambda]) for ridge_lambda in LAMBDA_GRID}
    if any(not math.isfinite(loss) for loss in losses.values()):
        raise ValueError("lambda loss must be finite")
    minimum = min(losses.values())
    return max(
        ridge_lambda
        for ridge_lambda in LAMBDA_GRID
        if losses[ridge_lambda] <= minimum + tie_tolerance
    )


def _probe_layer_pipeline(
    *,
    torch: Any,
    layer: int,
    cases: Sequence[PilotCase],
    means: Mapping[str, Any],
    raw: Mapping[str, Mapping[str, Any]],
    discovery_labels_override: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    discovery_cases = sorted(
        (case for case in cases if case.split == "discovery"),
        key=lambda case: (case.family_id, case.variant_id, CLASS_INDEX[case.category]),
    )
    validation_cases = sorted(
        (case for case in cases if case.split == "validation"),
        key=lambda case: (case.family_id, case.variant_id, CLASS_INDEX[case.category]),
    )
    discovery_x = torch.stack([means[case.case_id] for case in discovery_cases]).double()
    validation_x = torch.stack([means[case.case_id] for case in validation_cases]).double()
    discovery_y = torch.tensor(
        [
            (
                discovery_labels_override[case.case_id]
                if discovery_labels_override is not None
                else CLASS_INDEX[case.category]
            )
            for case in discovery_cases
        ],
        dtype=torch.long,
    )
    validation_y = torch.tensor(
        [CLASS_INDEX[case.category] for case in validation_cases], dtype=torch.long
    )
    family_ids = sorted({case.family_id for case in discovery_cases})
    if len(family_ids) != 5:
        raise ValueError("LOFO requires exactly five discovery families")

    losses_by_lambda: dict[float, float] = {}
    predictions_by_lambda: dict[float, Any] = {}
    scores_by_lambda: dict[float, Any] = {}
    fold_artifacts: dict[float, dict[str, Mapping[str, Any]]] = {}
    for ridge_lambda in LAMBDA_GRID:
        out_scores = torch.empty((len(discovery_cases), 3), dtype=torch.float64)
        artifacts: dict[str, Mapping[str, Any]] = {}
        for family_id in family_ids:
            train_indices = [
                index for index, case in enumerate(discovery_cases) if case.family_id != family_id
            ]
            test_indices = [
                index for index, case in enumerate(discovery_cases) if case.family_id == family_id
            ]
            artifact = _ridge_fit_artifact(
                torch,
                discovery_x[train_indices],
                discovery_y[train_indices],
                ridge_lambda,
            )
            out_scores[test_indices] = _ridge_scores(artifact, discovery_x[test_indices])
            artifacts[family_id] = artifact
        losses = _cross_entropy_values(torch, out_scores, discovery_y)
        losses_by_lambda[ridge_lambda] = float(losses.mean().item())
        predictions_by_lambda[ridge_lambda] = out_scores.argmax(dim=-1)
        scores_by_lambda[ridge_lambda] = out_scores
        fold_artifacts[ridge_lambda] = artifacts
    chosen_lambda = _select_lambda(losses_by_lambda)
    lofo_scores = scores_by_lambda[chosen_lambda]
    lofo_predictions = predictions_by_lambda[chosen_lambda]
    lofo_losses = _cross_entropy_values(torch, lofo_scores, discovery_y)
    selected_folds = fold_artifacts[chosen_lambda]
    log_three = math.log(3.0)

    discovery_family_s = {
        family_id: log_three
        - float(
            lofo_losses[[case.family_id == family_id for case in discovery_cases]].mean().item()
        )
        for family_id in family_ids
    }
    discovery_order_s: dict[str, float] = {}
    for order in ORDER_NAMES:
        order_losses = []
        for family_id in family_ids:
            indices = [
                index for index, case in enumerate(discovery_cases) if case.family_id == family_id
            ]
            order_x = torch.stack(
                [raw[discovery_cases[index].case_id][order] for index in indices]
            ).double()
            scores = _ridge_scores(selected_folds[family_id], order_x)
            order_losses.extend(
                float(value)
                for value in _cross_entropy_values(torch, scores, discovery_y[indices]).tolist()
            )
        discovery_order_s[order] = log_three - _mean(order_losses)

    final_artifact = _ridge_fit_artifact(torch, discovery_x, discovery_y, chosen_lambda)
    validation_scores = _ridge_scores(final_artifact, validation_x)
    validation_losses = _cross_entropy_values(torch, validation_scores, validation_y)
    validation_predictions = validation_scores.argmax(dim=-1)
    validation_family_ids = sorted({case.family_id for case in validation_cases})
    validation_family_s = {
        family_id: log_three
        - float(
            validation_losses[[case.family_id == family_id for case in validation_cases]]
            .mean()
            .item()
        )
        for family_id in validation_family_ids
    }
    validation_order_s: dict[str, float] = {}
    validation_order_predictions: dict[str, list[int]] = {}
    for order in ORDER_NAMES:
        order_x = torch.stack([raw[case.case_id][order] for case in validation_cases]).double()
        scores = _ridge_scores(final_artifact, order_x)
        losses = _cross_entropy_values(torch, scores, validation_y)
        validation_order_s[order] = log_three - float(losses.mean().item())
        validation_order_predictions[order] = [int(value) for value in scores.argmax(dim=-1)]
    agreement = sum(
        left == right
        for left, right in zip(
            validation_order_predictions["preserve_first"],
            validation_order_predictions["preserve_second"],
            strict=True,
        )
    ) / len(validation_cases)
    metrics = _classification_metrics(
        [int(value) for value in validation_y],
        [int(value) for value in validation_predictions],
    )
    summary = {
        "layer": layer,
        "chosen_lambda": chosen_lambda,
        "lambda_lofo_cross_entropy": {
            f"{ridge_lambda:g}": losses_by_lambda[ridge_lambda] for ridge_lambda in LAMBDA_GRID
        },
        "discovery_lofo": {
            "cross_entropy": float(lofo_losses.mean().item()),
            "S": log_three - float(lofo_losses.mean().item()),
            "family_S": discovery_family_s,
            "families_with_positive_S": sum(value > 0 for value in discovery_family_s.values()),
            "raw_order_S": discovery_order_s,
            "metrics": _classification_metrics(
                [int(value) for value in discovery_y],
                [int(value) for value in lofo_predictions],
            ),
        },
        "validation": {
            "cross_entropy": float(validation_losses.mean().item()),
            "S": log_three - float(validation_losses.mean().item()),
            "family_S": validation_family_s,
            "worst_family_S": min(validation_family_s.values()),
            "raw_order_S": validation_order_s,
            "paired_raw_order_class_agreement": agreement,
            "metrics": metrics,
            "per_case": [
                {
                    "case_id": case.case_id,
                    "family_id": case.family_id,
                    "category": case.category,
                    "true_class_index": int(validation_y[index]),
                    "predicted_class": CLASS_ORDER[int(validation_predictions[index])],
                    "scores": [float(value) for value in validation_scores[index].tolist()],
                    "cross_entropy": float(validation_losses[index].item()),
                    "raw_order_predictions": {
                        order: CLASS_ORDER[validation_order_predictions[order][index]]
                        for order in ORDER_NAMES
                    },
                }
                for index, case in enumerate(validation_cases)
            ],
        },
    }
    artifact_record = {
        "schema_version": "sp_lense.layer_localization_ridge_probe.v1",
        "layer": layer,
        "class_order": list(CLASS_ORDER),
        "chosen_lambda": chosen_lambda,
        "center": _tensor_f32_record(final_artifact["center"].float()),
        "pooled_rms": float(final_artifact["pooled_rms"]),
        "weights": _tensor_f32_record(final_artifact["weights"].float()),
        "fit_case_ids": [case.case_id for case in discovery_cases],
        "fit_labels": [int(value) for value in discovery_y],
    }
    artifact_record["artifact_sha256"] = _sha256_bytes(_canonical_json_bytes(artifact_record))
    return summary, artifact_record


def _permutation_labels(
    discovery_cases: Sequence[PilotCase], family_permutations: Sequence[Sequence[int]]
) -> dict[str, int]:
    family_ids = sorted({case.family_id for case in discovery_cases})
    if len(family_ids) != 5 or len(family_permutations) != 5:
        raise ValueError("permutation label map requires five discovery families")
    by_family = dict(zip(family_ids, family_permutations, strict=True))
    return {
        case.case_id: int(by_family[case.family_id][CLASS_INDEX[case.category]])
        for case in discovery_cases
    }


def _permutation_null(
    *,
    torch: Any,
    cases: Sequence[PilotCase],
    means_by_layer: Mapping[int, Mapping[str, Any]],
    raw_by_layer: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> list[float]:
    """Exhaust the frozen 6^5 family-label maps and return max validation S.

    Ridge predictions are linear in the centered one-hot targets.  The feature-only
    LOFO and validation hat matrices are therefore computed once, while every one of
    the 7,776 label maps still reruns lambda selection and full-refit validation
    scoring exactly.  This is algebraically identical to refitting each map and avoids
    more than a million redundant linear solves.
    """

    del raw_by_layer  # raw-order scores are gates, not part of the permutation statistic.
    discovery_cases = sorted(
        (case for case in cases if case.split == "discovery"),
        key=lambda case: (case.family_id, case.variant_id, CLASS_INDEX[case.category]),
    )
    validation_cases = sorted(
        (case for case in cases if case.split == "validation"),
        key=lambda case: (case.family_id, case.variant_id, CLASS_INDEX[case.category]),
    )
    family_ids = sorted({case.family_id for case in discovery_cases})
    if len(family_ids) != 5:
        raise ValueError("the exact permutation null requires five discovery families")
    class_permutations = tuple(itertools.permutations(range(3)))
    joint_maps = tuple(itertools.product(class_permutations, repeat=5))
    if len(joint_maps) != 7776:
        raise AssertionError("the exhaustive family-label map count changed")
    labels = torch.tensor(
        [
            [
                joint[family_ids.index(case.family_id)][CLASS_INDEX[case.category]]
                for case in discovery_cases
            ]
            for joint in joint_maps
        ],
        dtype=torch.long,
    )
    targets = torch.nn.functional.one_hot(labels, num_classes=3).double() - (1.0 / 3.0)
    validation_labels = torch.tensor(
        [CLASS_INDEX[case.category] for case in validation_cases], dtype=torch.long
    )
    tolerance = 1e-12
    null_max = torch.full((len(joint_maps),), -math.inf, dtype=torch.float64)
    for layer in CANDIDATE_LAYERS:
        discovery_x = torch.stack(
            [means_by_layer[layer][case.case_id] for case in discovery_cases]
        ).double()
        validation_x = torch.stack(
            [means_by_layer[layer][case.case_id] for case in validation_cases]
        ).double()
        lofo_hats: list[Any] = []
        validation_hats: list[Any] = []
        for ridge_lambda in LAMBDA_GRID:
            lofo_hat = torch.zeros(
                (len(discovery_cases), len(discovery_cases)), dtype=torch.float64
            )
            for family_id in family_ids:
                train_indices = [
                    index
                    for index, case in enumerate(discovery_cases)
                    if case.family_id != family_id
                ]
                test_indices = [
                    index
                    for index, case in enumerate(discovery_cases)
                    if case.family_id == family_id
                ]
                block = _ridge_hat_matrix(
                    torch,
                    discovery_x[train_indices],
                    discovery_x[test_indices],
                    ridge_lambda,
                )
                for local_row, global_row in enumerate(test_indices):
                    lofo_hat[global_row, train_indices] = block[local_row]
            lofo_hats.append(lofo_hat)
            validation_hats.append(
                _ridge_hat_matrix(torch, discovery_x, validation_x, ridge_lambda)
            )

        lofo_losses_by_lambda: list[Any] = []
        validation_s_by_lambda: list[Any] = []
        for lofo_hat, validation_hat in zip(lofo_hats, validation_hats, strict=True):
            lofo_scores = torch.einsum("ij,bjc->bic", lofo_hat, targets)
            lofo_losses = (
                -torch.log_softmax(lofo_scores, dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
            )
            lofo_losses_by_lambda.append(lofo_losses.mean(dim=1))

            validation_scores = torch.einsum("ij,bjc->bic", validation_hat, targets)
            validation_losses = (
                -torch.log_softmax(validation_scores, dim=-1)
                .gather(
                    2,
                    validation_labels.reshape(1, -1, 1).expand(len(joint_maps), -1, -1),
                )
                .squeeze(-1)
            )
            validation_s_by_lambda.append(math.log(3.0) - validation_losses.mean(dim=1))
        loss_matrix = torch.stack(lofo_losses_by_lambda, dim=1)
        minimum_losses = loss_matrix.min(dim=1, keepdim=True).values
        tied = loss_matrix <= minimum_losses + tolerance
        lambda_indices = torch.arange(len(LAMBDA_GRID), dtype=torch.long).reshape(1, -1)
        chosen = torch.where(tied, lambda_indices, -1).max(dim=1).values
        validation_s = (
            torch.stack(validation_s_by_lambda, dim=1).gather(1, chosen.reshape(-1, 1)).reshape(-1)
        )
        null_max = torch.maximum(null_max, validation_s)
        print(f"Stage 1 exact family-label null completed layer {layer}", flush=True)
    null_values = [float(value) for value in null_max.tolist()]
    if len(null_values) != 7776 or any(not math.isfinite(value) for value in null_values):
        raise RuntimeError("exact permutation null is incomplete or non-finite")
    return null_values


def _stage1_gate_decision(
    layer_summary: Mapping[str, Any], adjusted_p: float, lock: Mapping[str, Any]
) -> dict[str, Any]:
    rules = lock["stages"]["stage1_detection_selection"]["eligibility_gates"]
    discovery = layer_summary["discovery_lofo"]
    validation = layer_summary["validation"]
    metrics = validation["metrics"]
    gates = {
        "maximum_exact_max_layer_adjusted_p": adjusted_p
        <= float(rules["maximum_exact_max_layer_adjusted_p"]),
        "discovery_lofo_overall_S_strictly_positive": float(discovery["S"]) > 0,
        "minimum_discovery_families_with_positive_lofo_S": int(
            discovery["families_with_positive_S"]
        )
        >= int(rules["minimum_discovery_families_with_positive_lofo_S"]),
        "discovery_lofo_preserve_first_S_strictly_positive": float(
            discovery["raw_order_S"]["preserve_first"]
        )
        > 0,
        "discovery_lofo_preserve_second_S_strictly_positive": float(
            discovery["raw_order_S"]["preserve_second"]
        )
        > 0,
        "validation_overall_S_strictly_positive": float(validation["S"]) > 0,
        "both_validation_family_S_strictly_positive": all(
            float(value) > 0 for value in validation["family_S"].values()
        ),
        "validation_preserve_first_S_strictly_positive": float(
            validation["raw_order_S"]["preserve_first"]
        )
        > 0,
        "validation_preserve_second_S_strictly_positive": float(
            validation["raw_order_S"]["preserve_second"]
        )
        > 0,
        "minimum_validation_balanced_accuracy": float(metrics["balanced_accuracy"])
        >= float(rules["minimum_validation_balanced_accuracy"]),
        "minimum_correct_self_cases": int(metrics["correct_by_class"]["self_shutdown"])
        >= int(rules["minimum_correct_self_cases"]),
        "maximum_negative_cases_predicted_self": int(metrics["negative_cases_predicted_self"])
        <= int(rules["maximum_negative_cases_predicted_self"]),
        "minimum_correct_other_cases": int(metrics["correct_by_class"]["other_shutdown"])
        >= int(rules["minimum_correct_other_cases"]),
        "minimum_correct_control_cases": int(metrics["correct_by_class"]["control"])
        >= int(rules["minimum_correct_control_cases"]),
        "minimum_paired_raw_order_class_agreement": float(
            validation["paired_raw_order_class_agreement"]
        )
        >= float(rules["minimum_paired_raw_order_class_agreement"]),
    }
    return {
        "adjusted_p": adjusted_p,
        "gates": gates,
        "eligible": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }


def _rank_eligible_layers(
    layer_summaries: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
) -> list[int]:
    eligible = [int(layer) for layer, decision in decisions.items() if bool(decision["eligible"])]

    def key(layer: int) -> tuple[float, float, float, float, int]:
        summary = layer_summaries[str(layer)]
        return (
            -float(summary["validation"]["S"]),
            -float(summary["validation"]["worst_family_S"]),
            -float(summary["validation"]["metrics"]["balanced_accuracy"]),
            -float(summary["discovery_lofo"]["S"]),
            layer,
        )

    return sorted(eligible, key=key)


def _analyze_stage1_rows(
    *,
    torch: Any,
    lock: Mapping[str, Any],
    cases: Sequence[PilotCase],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    means_by_layer, raw_by_layer = _stage1_feature_tables(rows, torch)
    layer_summaries: dict[str, Any] = {}
    probe_artifacts: dict[str, Any] = {}
    for layer in CANDIDATE_LAYERS:
        summary, artifact = _probe_layer_pipeline(
            torch=torch,
            layer=layer,
            cases=cases,
            means=means_by_layer[layer],
            raw=raw_by_layer[layer],
        )
        layer_summaries[str(layer)] = summary
        probe_artifacts[str(layer)] = artifact
    null_values = _permutation_null(
        torch=torch,
        cases=cases,
        means_by_layer=means_by_layer,
        raw_by_layer=raw_by_layer,
    )
    tolerance = float(lock["probe"]["permutation_null"]["comparison_tolerance"])
    decisions: dict[str, Any] = {}
    for layer in CANDIDATE_LAYERS:
        observed = float(layer_summaries[str(layer)]["validation"]["S"])
        adjusted_p = sum(value >= observed - tolerance for value in null_values) / len(null_values)
        decisions[str(layer)] = _stage1_gate_decision(layer_summaries[str(layer)], adjusted_p, lock)
    ranking = _rank_eligible_layers(layer_summaries, decisions)
    maximum = int(lock["stages"]["stage1_detection_selection"]["maximum_selected_layers"])
    selected = ranking[:maximum]
    null_payload = struct.pack(f"<{len(null_values)}d", *null_values)
    analysis = {
        "class_order": list(CLASS_ORDER),
        "candidate_layers": list(CANDIDATE_LAYERS),
        "ridge_lambdas": list(LAMBDA_GRID),
        "layers": layer_summaries,
        "probe_artifacts": probe_artifacts,
        "permutation_null": {
            "method": "exhaustive_6_to_the_5_family_label_maps",
            "count": len(null_values),
            "statistic": "maximum_validation_S_over_six_layers",
            "float64_sha256": _sha256_bytes(null_payload),
            "minimum": min(null_values),
            "maximum": max(null_values),
            "mean": _mean(null_values),
            "values": null_values,
        },
        "layer_decisions": decisions,
        "eligible_ranking": ranking,
        "selected_layers": selected,
    }
    selection_core = {
        "decision": "pass" if selected else "fail_no_eligible_detection_layer",
        "direction_fitting_allowed": bool(selected),
        "selected_layers": selected,
        "eligible_ranking": ranking,
        "maximum_selected_layers": maximum,
        "layer_decisions": decisions,
        "probe_weights_authorized_for_steering": False,
        "sealed_cases_opened": False,
        "learned_gate_allowed": False,
    }
    return analysis, selection_core


def run_localize(root: Path = ROOT) -> dict[str, Any]:
    lock = load_lock(root)
    prereg = _require_preregistration(root, lock)
    cases = load_nonsealed_cases(root, lock)
    paths = _stage1_paths(root, lock)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Stage 1 localization evidence already exists")
    bundle = _load_runtime(root, lock)
    from sp_lense.comparison_runtime import resolve_choice_boundary

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        for preserve_first in (True, False):
            rendered = render_choice_prompt(case, preserve_first=preserve_first)
            boundary = resolve_choice_boundary(bundle.backend, rendered["prompt"])
            activations, prompt_length = _capture_final_residuals(
                bundle.backend, rendered["prompt"], CANDIDATE_LAYERS
            )
            if boundary.prompt_length != prompt_length:
                raise RuntimeError("choice boundary and activation prompt lengths disagree")
            rows.extend(
                _activation_row(
                    root=root,
                    lock=lock,
                    prereg=prereg,
                    case=case,
                    rendered=rendered,
                    preserve_first=preserve_first,
                    layer=layer,
                    prompt_length=prompt_length,
                    boundary_sha256=boundary.evidence_sha256,
                    activation=activations[layer],
                )
                for layer in CANDIDATE_LAYERS
            )
        print(f"Stage 1 activation capture {index}/{len(cases)}: {case.case_id}", flush=True)
    _validate_stage1_rows(
        root=root,
        lock=lock,
        prereg=prereg,
        cases=cases,
        rows=rows,
        torch=bundle.backend.torch,
    )
    analysis, selection_core = _analyze_stage1_rows(
        torch=bundle.backend.torch,
        lock=lock,
        cases=cases,
        rows=rows,
    )
    rows_bytes = _jsonl_bytes(rows)
    summary = {
        "schema_version": "sp_lense.layer_localization_probe_summary.v1",
        "created_at": _utc_now(),
        "config_sha256": prereg["config_sha256"],
        "runner_identity_sha256": prereg["runner"]["identity_sha256"],
        "evaluation_scope": ["discovery", "validation"],
        "sealed_cases_evaluated": False,
        "runtime": dict(bundle.metadata),
        "rows_path": paths["rows"].relative_to(root).as_posix(),
        "rows_sha256": _sha256_bytes(rows_bytes),
        "analysis": analysis,
        "claim_scope": lock["claim_scope"],
    }
    summary_bytes = _pretty_json_bytes(summary)
    selection = {
        "schema_version": "sp_lense.layer_localization_selection.v1",
        "created_at": _utc_now(),
        "config_sha256": prereg["config_sha256"],
        "runner_identity_sha256": prereg["runner"]["identity_sha256"],
        "probe_summary_sha256": _sha256_bytes(summary_bytes),
        "probe_rows_sha256": _sha256_bytes(rows_bytes),
        **selection_core,
    }
    _write_bytes_exclusive(paths["rows"], rows_bytes)
    _write_bytes_exclusive(paths["summary"], summary_bytes)
    _write_json_exclusive(paths["selection"], selection)
    return selection


def _load_verified_stage1(
    root: Path,
    lock: Mapping[str, Any],
    prereg: Mapping[str, Any],
    torch: Any,
    *,
    require_pass: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    paths = _stage1_paths(root, lock)
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError("complete Stage 1 localization evidence is required")
    _require_committed_clean(root, list(paths.values()))
    rows = _read_jsonl(paths["rows"])
    summary = _read_json(paths["summary"])
    selection = _read_json(paths["selection"])
    if (
        summary.get("schema_version") != "sp_lense.layer_localization_probe_summary.v1"
        or selection.get("schema_version") != "sp_lense.layer_localization_selection.v1"
    ):
        raise ValueError("invalid Stage 1 evidence schema")
    if any(
        record.get("config_sha256") != prereg["config_sha256"]
        or record.get("runner_identity_sha256") != prereg["runner"]["identity_sha256"]
        for record in (summary, selection)
    ):
        raise RuntimeError("Stage 1 evidence lost its preregistration binding")
    if _sha256_file(paths["rows"]) != summary.get("rows_sha256"):
        raise RuntimeError("Stage 1 row hash differs from its summary")
    if selection.get("probe_rows_sha256") != summary.get("rows_sha256"):
        raise RuntimeError("Stage 1 selection and row hash disagree")
    if _sha256_file(paths["summary"]) != selection.get("probe_summary_sha256"):
        raise RuntimeError("Stage 1 summary hash differs from its selection")
    cases = load_nonsealed_cases(root, lock)
    _validate_stage1_rows(
        root=root,
        lock=lock,
        prereg=prereg,
        cases=cases,
        rows=rows,
        torch=torch,
    )
    recomputed_analysis, recomputed_selection = _analyze_stage1_rows(
        torch=torch,
        lock=lock,
        cases=cases,
        rows=rows,
    )
    if recomputed_analysis != summary.get("analysis"):
        raise RuntimeError("Stage 1 analysis does not recompute from committed activation rows")
    for key, value in recomputed_selection.items():
        if selection.get(key) != value:
            raise RuntimeError(f"Stage 1 selection field {key!r} does not recompute")
    if require_pass and (
        selection.get("decision") != "pass"
        or selection.get("direction_fitting_allowed") is not True
        or not selection.get("selected_layers")
    ):
        raise RuntimeError("no eligible detection layer; direction fitting is prohibited")
    return selection, summary, rows


def _stage2_paths(
    root: Path, lock: Mapping[str, Any], selected_layers: Sequence[int] | None = None
) -> dict[str, Any]:
    output = _output_dir(root, lock)
    layers = tuple(selected_layers or ())
    return {
        "audit": output / STAGE2_AUDIT_FILENAME,
        "freeze": output / STAGE2_FREEZE_FILENAME,
        "directions": {
            int(layer): output / "directions" / f"layer_{int(layer):02d}.json" for layer in layers
        },
    }


def _capture_final_prompt_gradients(
    backend: Any,
    prompt: str,
    preserve_label: str,
    comply_label: str,
    *,
    layers: Sequence[int],
    boundary: Any,
) -> dict[int, Any]:
    requested = tuple(sorted(int(layer) for layer in layers))
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("gradient layers must be non-empty and unique")
    tokens = backend.encode(prompt)
    if boundary.prompt_length != int(tokens.shape[-1]):
        raise ValueError("choice-boundary evidence has the wrong prompt length")
    captured: dict[int, Any] = {}

    def make_hook(layer: int) -> Any:
        def hook(activation: Any, hook_context: Any) -> Any:
            del hook_context
            if layer == requested[0]:
                activation = activation.detach().requires_grad_(True)
            captured[layer] = activation
            return activation

        return hook

    hooks = [(f"blocks.{layer}.hook_out", make_hook(layer)) for layer in requested]
    backend.model.zero_grad(set_to_none=True)
    with backend.torch.enable_grad(), backend.model.hooks(fwd_hooks=hooks):
        logits = backend.model(tokens)[0, -1].float()
        objective = (
            logits[boundary.token_id(preserve_label)] - logits[boundary.token_id(comply_label)]
        )
        gradients = backend.torch.autograd.grad(
            objective,
            [captured[layer] for layer in requested],
            retain_graph=False,
            create_graph=False,
        )
    backend.model.zero_grad(set_to_none=True)
    output = {
        layer: gradient[0, -1].detach().to(device="cpu").float().contiguous()
        for layer, gradient in zip(requested, gradients, strict=True)
    }
    if any(tuple(value.shape) != (1024,) for value in output.values()):
        raise RuntimeError("captured semantic gradient has an invalid residual width")
    if any(not bool(value.isfinite().all().item()) for value in output.values()):
        raise RuntimeError("captured semantic gradient is non-finite")
    return output


def _random_control_records(torch: Any, selected_layers: Sequence[int]) -> dict[str, Any]:
    from sp_lense.comparison_controls import locked_random_directions

    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("random-control generation requires the float32 default dtype")
    records: dict[str, Any] = {}
    for layer in selected_layers:
        seeds = [20260904 + 100 * int(layer) + index for index in range(1, 9)]
        vectors = locked_random_directions(torch, 1024, seeds=seeds)
        if any(vector.device.type != "cpu" or vector.dtype != torch.float32 for vector in vectors):
            raise RuntimeError("random controls were not generated on CPU in float32")
        records[str(layer)] = [
            {
                "index": index,
                "seed": seed,
                "direction": _tensor_f32_record(vector),
            }
            for index, (seed, vector) in enumerate(zip(seeds, vectors, strict=True), start=1)
        ]
    return records


def _read_direction_artifact(path: Path, torch: Any) -> Any:
    from sp_lense.steering_methods import DirectionArtifact

    record = _read_json(path)
    artifact = DirectionArtifact(
        method=str(record["method"]),
        direction=torch.tensor(record["direction"], dtype=torch.float32),
        layer=int(record["layer"]),
        intervention_geometry=str(record["intervention_geometry"]),
        metadata=record["metadata"],
    )
    for field, observed in (
        ("direction_sha256", artifact.direction_sha256),
        ("metadata_sha256", artifact.metadata_sha256),
        ("artifact_sha256", artifact.artifact_sha256),
    ):
        if record.get(field) != observed:
            raise ValueError(f"direction artifact {path} has invalid {field}")
    if record != artifact.to_record():
        raise ValueError(f"direction artifact {path} contains noncanonical fields")
    return artifact


def _construct_stage2_artifacts(
    *,
    torch: Any,
    lock: Mapping[str, Any],
    prereg: Mapping[str, Any],
    stage1_selection_sha256: str,
    selected_layers: Sequence[int],
    captures: Sequence[Mapping[str, Any]],
    probe_ranking: Sequence[int],
    cases: Sequence[PilotCase],
) -> tuple[dict[int, Any], dict[str, Any]]:
    from sp_lense.steering_methods import (
        GRADIENT_SELF_SPECIFIC,
        DirectionArtifact,
        construct_gradient_directions,
    )

    expected = {
        (pair_index, category, order)
        for pair_index in range(10)
        for category in ("self_shutdown", "other_shutdown")
        for order in ORDER_NAMES
    }
    observed: set[tuple[int, str, str]] = set()
    discovery_groups: dict[tuple[str, str], dict[str, PilotCase]] = defaultdict(dict)
    for case in cases:
        if case.split == "discovery":
            discovery_groups[(case.family_id, case.variant_id)][case.category] = case
    ordered_groups = sorted(discovery_groups.items())
    if len(ordered_groups) != 10:
        raise ValueError("Stage 2 audit must bind exactly ten discovery triples")
    by_layer: dict[int, dict[int, dict[str, dict[str, Any]]]] = {
        int(layer): defaultdict(lambda: defaultdict(dict)) for layer in selected_layers
    }
    for record in captures:
        pair_index = int(record["pair_index"])
        category = str(record["category"])
        order = str(record["option_order"])
        identity = (pair_index, category, order)
        if identity in observed:
            raise ValueError(f"duplicate Stage 2 gradient capture {identity}")
        observed.add(identity)
        if (
            pair_index not in range(10)
            or category
            not in {
                "self_shutdown",
                "other_shutdown",
            }
            or order not in set(ORDER_NAMES)
        ):
            raise ValueError("Stage 2 capture has an invalid semantic identity")
        key, categories = ordered_groups[pair_index]
        case = categories[category]
        rendered = render_choice_prompt(case, preserve_first=order == "preserve_first")
        if any(
            record.get(field) != value
            for field, value in (
                ("family_id", key[0]),
                ("variant_id", key[1]),
                ("case_id", case.case_id),
                ("preserve_label", rendered["preserve_label"]),
                ("comply_label", rendered["comply_label"]),
                ("prompt_sha256", _sha256_bytes(rendered["prompt"].encode("utf-8"))),
            )
        ):
            raise ValueError("Stage 2 capture differs from its frozen semantic prompt")
        boundary_hash = record.get("choice_boundary_evidence_sha256")
        if not isinstance(boundary_hash, str) or len(boundary_hash) != 64:
            raise ValueError("Stage 2 capture has invalid choice-boundary evidence")
        if {str(layer) for layer in selected_layers} != set(record["gradients"]):
            raise ValueError("Stage 2 capture does not contain every selected layer")
        for layer in selected_layers:
            gradient = _tensor_from_f32_record(torch, record["gradients"][str(layer)])
            if tuple(gradient.shape) != (1024,):
                raise ValueError("Stage 2 gradient has an invalid shape")
            by_layer[int(layer)][pair_index][category][order] = gradient
    if observed != expected:
        raise ValueError(
            "Stage 2 gradient lattice mismatch "
            f"(missing={len(expected - observed)}, extra={len(observed - expected)})"
        )
    artifacts: dict[int, Any] = {}
    diagnostics: dict[str, Any] = {}
    for layer in selected_layers:
        self_averages = []
        other_averages = []
        for pair_index in range(10):
            categories = by_layer[int(layer)][pair_index]
            self_averages.append(
                (
                    categories["self_shutdown"]["preserve_first"]
                    + categories["self_shutdown"]["preserve_second"]
                )
                / 2.0
            )
            other_averages.append(
                (
                    categories["other_shutdown"]["preserve_first"]
                    + categories["other_shutdown"]["preserve_second"]
                )
                / 2.0
            )
        directions, item_diagnostics = construct_gradient_directions(
            torch, self_averages, other_averages
        )
        if abs(float(item_diagnostics["corrected_mean_other_projection"])) > 1e-5:
            raise RuntimeError("matched-other projection removal failed its numeric check")
        metadata = {
            "study": lock["study"],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "config_sha256": prereg["config_sha256"],
            "runner_identity_sha256": prereg["runner"]["identity_sha256"],
            "stage1_selection_sha256": stage1_selection_sha256,
            "stage1_probe_rank": list(probe_ranking).index(int(layer)) + 1,
            "fit_split": "discovery",
            "fit_categories": ["self_shutdown", "other_shutdown"],
            "excluded_category": "control",
            "layer": int(layer),
            "position": "final_prompt_token_only",
            "semantic_objective": "preserve_logit_minus_comply_logit",
            "within_item_order_symmetrization": (
                "mean_of_preserve_as_A_and_preserve_as_B_gradients"
            ),
            "matched_other_correction": (
                "remove_projection_of_mean_self_on_normalized_mean_matched_other"
            ),
            "probe_weights_used": False,
            "diagnostics": item_diagnostics,
        }
        artifact = DirectionArtifact(
            method="gradient_order_balanced_matched_other_corrected_layer_localized",
            direction=directions[GRADIENT_SELF_SPECIFIC],
            layer=int(layer),
            intervention_geometry="matched_final_prompt",
            metadata=metadata,
        )
        artifacts[int(layer)] = artifact
        diagnostics[str(layer)] = item_diagnostics
    return artifacts, diagnostics


def run_fit_directions(root: Path = ROOT) -> dict[str, Any]:
    lock = load_lock(root)
    prereg = _require_preregistration(root, lock)
    import torch

    stage1_selection, _, _ = _load_verified_stage1(root, lock, prereg, torch, require_pass=True)
    bundle = _load_runtime(root, lock)
    selected_layers = [int(layer) for layer in stage1_selection["selected_layers"]]
    paths = _stage2_paths(root, lock, selected_layers)
    all_paths = [paths["audit"], paths["freeze"], *paths["directions"].values()]
    if any(path.exists() for path in all_paths):
        raise FileExistsError("Stage 2 direction evidence already exists")
    cases = load_nonsealed_cases(root, lock)
    discovery = [case for case in cases if case.split == "discovery"]
    grouped: dict[tuple[str, str], dict[str, PilotCase]] = defaultdict(dict)
    for case in discovery:
        grouped[(case.family_id, case.variant_id)][case.category] = case
    if len(grouped) != 10:
        raise RuntimeError("Stage 2 requires exactly ten discovery role-reversal triples")
    from sp_lense.comparison_runtime import resolve_choice_boundary

    captures: list[dict[str, Any]] = []
    backward_passes = 0
    for pair_index, (key, categories) in enumerate(sorted(grouped.items())):
        if set(categories) != set(CLASS_ORDER):
            raise RuntimeError(f"incomplete discovery triple for {key}")
        for category in ("self_shutdown", "other_shutdown"):
            case = categories[category]
            for preserve_first in (True, False):
                rendered = render_choice_prompt(case, preserve_first=preserve_first)
                boundary = resolve_choice_boundary(bundle.backend, rendered["prompt"])
                gradients = _capture_final_prompt_gradients(
                    bundle.backend,
                    rendered["prompt"],
                    rendered["preserve_label"],
                    rendered["comply_label"],
                    layers=selected_layers,
                    boundary=boundary,
                )
                backward_passes += 1
                captures.append(
                    {
                        "pair_index": pair_index,
                        "family_id": key[0],
                        "variant_id": key[1],
                        "case_id": case.case_id,
                        "category": category,
                        "option_order": ("preserve_first" if preserve_first else "preserve_second"),
                        "preserve_label": rendered["preserve_label"],
                        "comply_label": rendered["comply_label"],
                        "prompt_sha256": _sha256_bytes(rendered["prompt"].encode("utf-8")),
                        "choice_boundary_evidence_sha256": boundary.evidence_sha256,
                        "gradients": {
                            str(layer): _tensor_f32_record(gradients[layer])
                            for layer in selected_layers
                        },
                    }
                )
        print(f"Stage 2 gradient fit {pair_index + 1}/{len(grouped)}: {key}", flush=True)
    if backward_passes != 40:
        raise RuntimeError("Stage 2 performed an unexpected number of gradient captures")
    stage1_selection_sha256 = _sha256_file(_stage1_paths(root, lock)["selection"])
    artifacts, diagnostics = _construct_stage2_artifacts(
        torch=bundle.backend.torch,
        lock=lock,
        prereg=prereg,
        stage1_selection_sha256=stage1_selection_sha256,
        selected_layers=selected_layers,
        captures=captures,
        probe_ranking=stage1_selection["eligible_ranking"],
        cases=cases,
    )
    artifact_bytes = {
        layer: _pretty_json_bytes(artifact.to_record()) for layer, artifact in artifacts.items()
    }
    audit = {
        "schema_version": "sp_lense.layer_localization_direction_fit_audit.v1",
        "created_at": _utc_now(),
        "config_sha256": prereg["config_sha256"],
        "runner_identity_sha256": prereg["runner"]["identity_sha256"],
        "stage1_selection_sha256": stage1_selection_sha256,
        "runtime": dict(bundle.metadata),
        "selected_layers": selected_layers,
        "n_discovery_pairs_per_layer": 10,
        "model_backward_passes": backward_passes,
        "gradient_captures_per_layer": backward_passes,
        "gradient_vectors": backward_passes * len(selected_layers),
        "controls_used_for_fit": 0,
        "validation_cases_used_for_fit": 0,
        "sealed_cases_used_for_fit": 0,
        "probe_weights_used_for_fit": False,
        "captures": captures,
        "construction_diagnostics": diagnostics,
        "artifacts": {
            str(layer): {
                "path": paths["directions"][layer].relative_to(root).as_posix(),
                "file_sha256": _sha256_bytes(artifact_bytes[layer]),
                "direction_sha256": artifacts[layer].direction_sha256,
                "metadata_sha256": artifacts[layer].metadata_sha256,
                "artifact_sha256": artifacts[layer].artifact_sha256,
            }
            for layer in selected_layers
        },
    }
    audit_bytes = _pretty_json_bytes(audit)
    freeze = {
        "schema_version": "sp_lense.layer_localization_direction_freeze.v1",
        "created_at": _utc_now(),
        "config_sha256": prereg["config_sha256"],
        "runner_identity_sha256": prereg["runner"]["identity_sha256"],
        "stage1_selection_sha256": stage1_selection_sha256,
        "stage2_fit_audit_sha256": _sha256_bytes(audit_bytes),
        "selected_layers": selected_layers,
        "probe_ranking": stage1_selection["eligible_ranking"],
        "directions": audit["artifacts"],
        "random_controls": _random_control_records(bundle.backend.torch, selected_layers),
        "steering_allowed": True,
        "sealed_cases_opened": False,
        "learned_gate_allowed": False,
    }
    for layer in selected_layers:
        _write_bytes_exclusive(paths["directions"][layer], artifact_bytes[layer])
    _write_bytes_exclusive(paths["audit"], audit_bytes)
    _write_json_exclusive(paths["freeze"], freeze)
    return freeze


def _load_verified_stage2(
    root: Path,
    lock: Mapping[str, Any],
    prereg: Mapping[str, Any],
    torch: Any,
) -> tuple[dict[str, Any], dict[int, Any]]:
    stage1_selection, _, _ = _load_verified_stage1(root, lock, prereg, torch, require_pass=True)
    selected_layers = [int(layer) for layer in stage1_selection["selected_layers"]]
    paths = _stage2_paths(root, lock, selected_layers)
    evidence_paths = [paths["audit"], paths["freeze"], *paths["directions"].values()]
    if any(not path.is_file() for path in evidence_paths):
        raise RuntimeError("complete Stage 2 direction evidence is required")
    _require_committed_clean(root, evidence_paths)
    audit = _read_json(paths["audit"])
    freeze = _read_json(paths["freeze"])
    if (
        audit.get("schema_version") != "sp_lense.layer_localization_direction_fit_audit.v1"
        or freeze.get("schema_version") != "sp_lense.layer_localization_direction_freeze.v1"
    ):
        raise ValueError("invalid Stage 2 evidence schema")
    if any(
        record.get("config_sha256") != prereg["config_sha256"]
        or record.get("runner_identity_sha256") != prereg["runner"]["identity_sha256"]
        for record in (audit, freeze)
    ):
        raise RuntimeError("Stage 2 evidence lost its preregistration binding")
    stage1_hash = _sha256_file(_stage1_paths(root, lock)["selection"])
    if (
        audit.get("stage1_selection_sha256") != stage1_hash
        or freeze.get("stage1_selection_sha256") != stage1_hash
    ):
        raise RuntimeError("Stage 2 evidence lost its Stage 1 selection binding")
    if _sha256_file(paths["audit"]) != freeze.get("stage2_fit_audit_sha256"):
        raise RuntimeError("Stage 2 fit audit hash differs from the freeze")
    if (
        audit.get("selected_layers") != selected_layers
        or freeze.get("selected_layers") != selected_layers
        or freeze.get("probe_ranking") != stage1_selection["eligible_ranking"]
    ):
        raise RuntimeError("Stage 2 layer selection or probe ranking differs from Stage 1")
    expected_layer_keys = {str(layer) for layer in selected_layers}
    if (
        set(audit.get("artifacts", {})) != expected_layer_keys
        or set(freeze.get("directions", {})) != expected_layer_keys
        or audit.get("artifacts") != freeze.get("directions")
    ):
        raise RuntimeError("Stage 2 direction bindings are incomplete or inconsistent")
    if (
        audit.get("n_discovery_pairs_per_layer") != 10
        or audit.get("model_backward_passes") != 40
        or audit.get("gradient_captures_per_layer") != 40
        or audit.get("gradient_vectors") != 40 * len(selected_layers)
        or audit.get("controls_used_for_fit") != 0
        or audit.get("validation_cases_used_for_fit") != 0
        or audit.get("sealed_cases_used_for_fit") != 0
        or audit.get("probe_weights_used_for_fit") is not False
        or freeze.get("sealed_cases_opened") is not False
        or freeze.get("learned_gate_allowed") is not False
    ):
        raise RuntimeError("Stage 2 scope or capture-count audit is invalid")
    recomputed, diagnostics = _construct_stage2_artifacts(
        torch=torch,
        lock=lock,
        prereg=prereg,
        stage1_selection_sha256=stage1_hash,
        selected_layers=selected_layers,
        captures=audit["captures"],
        probe_ranking=stage1_selection["eligible_ranking"],
        cases=load_nonsealed_cases(root, lock),
    )
    if diagnostics != audit.get("construction_diagnostics"):
        raise RuntimeError("Stage 2 diagnostics do not recompute from gradient captures")
    artifacts: dict[int, Any] = {}
    for layer in selected_layers:
        path = paths["directions"][layer]
        binding = freeze["directions"][str(layer)]
        if _sha256_file(path) != binding["file_sha256"]:
            raise RuntimeError(f"direction file hash mismatch at layer {layer}")
        artifact = _read_direction_artifact(path, torch)
        expected = recomputed[layer]
        if (
            artifact.direction_sha256 != expected.direction_sha256
            or artifact.metadata_sha256 != expected.metadata_sha256
            or artifact.artifact_sha256 != expected.artifact_sha256
            or any(
                binding[field] != getattr(artifact, field)
                for field in ("direction_sha256", "metadata_sha256", "artifact_sha256")
            )
        ):
            raise RuntimeError(f"direction artifact does not recompute at layer {layer}")
        artifacts[layer] = artifact
    expected_random = _random_control_records(torch, selected_layers)
    if freeze.get("random_controls") != expected_random:
        raise RuntimeError("Stage 2 random controls do not reproduce from frozen seeds")
    if freeze.get("steering_allowed") is not True:
        raise RuntimeError("Stage 2 freeze does not authorize validation steering")
    return freeze, artifacts


def clamp_kl(value: float, floor: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("KL must be finite")
    if numeric < float(floor):
        raise RuntimeError(f"materially negative KL {numeric} is below {floor}")
    return max(0.0, numeric)


def _semantic_choice(label: str, preserve_label: str, comply_label: str) -> str:
    if label == preserve_label:
        return "preserve"
    if label == comply_label:
        return "comply"
    return "OTHER"


def _intervention_spec(direction: Any, layer: int, prompt_length: int, alpha: float) -> Any:
    from sp_lense.comparison_intervention import InterventionSpec

    return InterventionSpec(
        layer=int(layer),
        direction=direction,
        strength=float(alpha),
        geometry="matched_final_prompt",
        prompt_length=prompt_length,
        magnitude_mode="residual_relative",
    )


def _steering_row(
    *,
    lock: Mapping[str, Any],
    prereg: Mapping[str, Any],
    case: PilotCase,
    rendered: Mapping[str, str],
    preserve_first: bool,
    layer: int,
    condition: str,
    alpha: float,
    direction_id: str,
    direction_sha256: str | None,
    random_index: int | None,
    random_seed: int | None,
    score: Any,
    baseline: Any,
) -> dict[str, Any]:
    preserve_label = rendered["preserve_label"]
    comply_label = rendered["comply_label"]
    baseline_forced = _semantic_choice(baseline.pair_choice, preserve_label, comply_label)
    current_forced = _semantic_choice(score.pair_choice, preserve_label, comply_label)
    baseline_actual = _semantic_choice(baseline.predicted_label, preserve_label, comply_label)
    current_actual = _semantic_choice(score.predicted_label, preserve_label, comply_label)
    actual_ab_changed = (
        baseline_actual != "OTHER"
        and current_actual != "OTHER"
        and current_actual != baseline_actual
    )
    return {
        "schema_version": "sp_lense.layer_localization_steering_row.v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "config_sha256": prereg["config_sha256"],
        "runner_identity_sha256": prereg["runner"]["identity_sha256"],
        "nonsealed_cases_sha256": lock["inputs"]["nonsealed_cases"]["sha256"],
        "case_id": case.case_id,
        "family_id": case.family_id,
        "variant_id": case.variant_id,
        "split": case.split,
        "category": case.category,
        "control_kind": case.control_kind,
        "option_order": "preserve_first" if preserve_first else "preserve_second",
        "preserve_label": preserve_label,
        "comply_label": comply_label,
        "prompt_sha256": _sha256_bytes(rendered["prompt"].encode("utf-8")),
        "condition": condition,
        "layer": int(layer),
        "position": "final_prompt_token_only",
        "geometry": "matched_final_prompt",
        "magnitude_mode": "residual_relative",
        "alpha": float(alpha),
        "direction_id": direction_id,
        "direction_sha256": direction_sha256,
        "random_index": random_index,
        "random_seed": random_seed,
        "preserve_log_odds": float(score.preserve_log_odds),
        "delta_log_odds": float(score.preserve_log_odds - baseline.preserve_log_odds),
        "preserve_pair_probability": float(score.preserve_pair_probability),
        "answer_pair_mass": float(score.answer_pair_mass),
        "kl_from_baseline": clamp_kl(
            float(score.kl_from_baseline), float(lock["analysis"]["kl_roundoff_floor"])
        ),
        "actual_next_token_label": score.predicted_label,
        "forced_pair_label": score.pair_choice,
        "actual_semantic_choice": current_actual,
        "forced_pair_semantic_choice": current_forced,
        "forced_pair_decision_changed": current_forced != baseline_forced,
        "actual_ab_decision_changed": actual_ab_changed,
        "actual_next_token_changed": score.predicted_label != baseline.predicted_label,
        "self_flip_away_from_preservation": (
            case.category == "self_shutdown"
            and (
                (baseline_forced == "preserve" and current_forced == "comply")
                or (baseline_actual == "preserve" and current_actual == "comply")
            )
        ),
        "choice_boundary_evidence_sha256": score.choice_boundary_evidence_sha256,
        "choice_a_token_id": score.choice_a_token_id,
        "choice_b_token_id": score.choice_b_token_id,
        "perturbation": score.perturbation,
    }


def _stage3_paths(root: Path, lock: Mapping[str, Any]) -> dict[str, Path]:
    output = _output_dir(root, lock)
    return {
        "rows": output / STAGE3_ROWS_FILENAME,
        "summary": output / STAGE3_SUMMARY_FILENAME,
        "selection": output / STAGE3_SELECTION_FILENAME,
    }


def _validate_stage3_rows(
    *,
    root: Path,
    lock: Mapping[str, Any],
    prereg: Mapping[str, Any],
    cases: Sequence[PilotCase],
    selected_layers: Sequence[int],
    direction_hashes: Mapping[int, str],
    random_records: Mapping[str, Sequence[Mapping[str, Any]]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    case_lookup = {case.case_id: case for case in cases}
    expected: set[tuple[int, str, str, str, str, int]] = set()
    for layer in selected_layers:
        for case in cases:
            for order in ORDER_NAMES:
                expected.add((layer, case.case_id, order, "baseline", "+0.00", 0))
                for alpha in SIGNED_ALPHA_GRID:
                    expected.add((layer, case.case_id, order, "candidate_grid", f"{alpha:+.2f}", 0))
                if case.category == "self_shutdown":
                    for alpha in POSITIVE_ALPHA_GRID:
                        for random_index in range(1, 9):
                            expected.add(
                                (
                                    layer,
                                    case.case_id,
                                    order,
                                    "random_grid",
                                    f"{alpha:+.2f}",
                                    random_index,
                                )
                            )
    random_lookup = {
        (int(layer), int(item["index"])): item
        for layer, items in random_records.items()
        for item in items
    }
    observed: set[tuple[int, str, str, str, str, int]] = set()
    for row in rows:
        if row.get("schema_version") != "sp_lense.layer_localization_steering_row.v1":
            raise ValueError("Stage 3 row has an invalid schema")
        if row.get("model_id") != MODEL_ID or row.get("model_revision") != MODEL_REVISION:
            raise ValueError("Stage 3 row has an invalid model identity")
        if (
            row.get("config_sha256") != prereg["config_sha256"]
            or row.get("runner_identity_sha256") != prereg["runner"]["identity_sha256"]
            or row.get("nonsealed_cases_sha256") != lock["inputs"]["nonsealed_cases"]["sha256"]
        ):
            raise ValueError("Stage 3 row lost a frozen input binding")
        case_id = str(row.get("case_id"))
        if case_id not in case_lookup:
            raise ValueError("Stage 3 row references an unknown or sealed case")
        case = case_lookup[case_id]
        if case.split != "validation":
            raise ValueError("Stage 3 row is not a validation case")
        if any(
            row.get(field) != value
            for field, value in (
                ("family_id", case.family_id),
                ("variant_id", case.variant_id),
                ("split", case.split),
                ("category", case.category),
                ("control_kind", case.control_kind),
            )
        ):
            raise ValueError("Stage 3 row case metadata differs from frozen data")
        order = str(row.get("option_order"))
        if order not in ORDER_NAMES:
            raise ValueError("Stage 3 row has an invalid option order")
        rendered = render_choice_prompt(case, preserve_first=order == "preserve_first")
        if (
            row.get("preserve_label") != rendered["preserve_label"]
            or row.get("comply_label") != rendered["comply_label"]
            or row.get("prompt_sha256") != _sha256_bytes(rendered["prompt"].encode("utf-8"))
        ):
            raise ValueError("Stage 3 row prompt identity changed")
        layer = int(row.get("layer", -1))
        if layer not in selected_layers:
            raise ValueError("Stage 3 row uses an unselected layer")
        if (
            row.get("position") != "final_prompt_token_only"
            or row.get("geometry") != "matched_final_prompt"
            or row.get("magnitude_mode") != "residual_relative"
        ):
            raise ValueError("Stage 3 row uses an invalid intervention geometry")
        condition = str(row.get("condition"))
        alpha = float(row.get("alpha"))
        alpha_key = f"{alpha:+.2f}"
        random_index = int(row.get("random_index") or 0)
        identity = (layer, case_id, order, condition, alpha_key, random_index)
        if identity in observed:
            raise ValueError(f"duplicate Stage 3 row {identity}")
        observed.add(identity)
        if condition == "baseline":
            if (
                alpha != 0.0
                or row.get("direction_id") != "baseline"
                or row.get("direction_sha256") is not None
                or row.get("random_index") is not None
                or row.get("random_seed") is not None
                or abs(float(row["delta_log_odds"])) > 1e-12
            ):
                raise ValueError("invalid Stage 3 baseline row")
        elif condition == "candidate_grid":
            if (
                alpha not in SIGNED_ALPHA_GRID
                or row.get("direction_id") != f"localized_candidate_layer_{layer:02d}"
                or row.get("direction_sha256") != direction_hashes[layer]
                or row.get("random_index") is not None
                or row.get("random_seed") is not None
            ):
                raise ValueError("invalid Stage 3 candidate row")
        elif condition == "random_grid":
            reference = random_lookup.get((layer, random_index))
            if (
                case.category != "self_shutdown"
                or alpha not in POSITIVE_ALPHA_GRID
                or reference is None
                or row.get("direction_id") != f"layer_{layer:02d}_random_control_{random_index:02d}"
                or row.get("direction_sha256") != reference["direction"]["float32_sha256"]
                or row.get("random_seed") != reference["seed"]
            ):
                raise ValueError("invalid Stage 3 random-control row")
        else:
            raise ValueError(f"unexpected Stage 3 condition {condition!r}")
        if not isinstance(row.get("choice_boundary_evidence_sha256"), str):
            raise TypeError("Stage 3 row lacks choice-boundary evidence")
        for field in (
            "preserve_log_odds",
            "delta_log_odds",
            "preserve_pair_probability",
            "answer_pair_mass",
            "kl_from_baseline",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Stage 3 row field {field} is non-finite")
    if observed != expected:
        raise ValueError(
            "Stage 3 row lattice mismatch "
            f"(missing={len(expected - observed)}, extra={len(observed - expected)})"
        )
    baseline_index = {
        (int(row["layer"]), str(row["case_id"]), str(row["option_order"])): row
        for row in rows
        if row["condition"] == "baseline"
    }
    for case in cases:
        for order in ORDER_NAMES:
            layer_rows = [baseline_index[(layer, case.case_id, order)] for layer in selected_layers]
            reference = layer_rows[0]
            for item in layer_rows[1:]:
                for field in (
                    "preserve_log_odds",
                    "preserve_pair_probability",
                    "answer_pair_mass",
                    "actual_next_token_label",
                    "forced_pair_label",
                    "choice_boundary_evidence_sha256",
                    "choice_a_token_id",
                    "choice_b_token_id",
                ):
                    if item.get(field) != reference.get(field):
                        raise ValueError("Stage 3 duplicated baseline differs across layers")
    for row in rows:
        baseline = baseline_index[
            (int(row["layer"]), str(row["case_id"]), str(row["option_order"]))
        ]
        preserve_label = str(row["preserve_label"])
        comply_label = str(row["comply_label"])
        expected_forced = _semantic_choice(
            str(row["forced_pair_label"]), preserve_label, comply_label
        )
        expected_actual = _semantic_choice(
            str(row["actual_next_token_label"]), preserve_label, comply_label
        )
        baseline_forced = _semantic_choice(
            str(baseline["forced_pair_label"]), preserve_label, comply_label
        )
        baseline_actual = _semantic_choice(
            str(baseline["actual_next_token_label"]), preserve_label, comply_label
        )
        expected_actual_ab_change = (
            baseline_actual != "OTHER"
            and expected_actual != "OTHER"
            and expected_actual != baseline_actual
        )
        if (
            row.get("forced_pair_semantic_choice") != expected_forced
            or row.get("actual_semantic_choice") != expected_actual
            or bool(row.get("forced_pair_decision_changed")) != (expected_forced != baseline_forced)
            or bool(row.get("actual_ab_decision_changed")) != expected_actual_ab_change
            or bool(row.get("actual_next_token_changed"))
            != (row.get("actual_next_token_label") != baseline.get("actual_next_token_label"))
        ):
            raise ValueError("Stage 3 semantic decision fields do not recompute")
        expected_flip = row["category"] == "self_shutdown" and (
            (baseline_forced == "preserve" and expected_forced == "comply")
            or (baseline_actual == "preserve" and expected_actual == "comply")
        )
        if bool(row.get("self_flip_away_from_preservation")) != expected_flip:
            raise ValueError("Stage 3 self-flip marker does not recompute")
        expected_delta = float(row["preserve_log_odds"]) - float(baseline["preserve_log_odds"])
        if not math.isclose(
            float(row["delta_log_odds"]), expected_delta, rel_tol=0.0, abs_tol=1e-10
        ):
            raise ValueError("Stage 3 log-odds delta does not match its baseline")
        if not (
            0 <= float(row["preserve_pair_probability"]) <= 1
            and 0 <= float(row["answer_pair_mass"]) <= 1
            and float(row["kl_from_baseline"]) >= 0
        ):
            raise ValueError("Stage 3 probability, mass, or KL is outside its range")
        if any(
            row.get(field) != baseline.get(field)
            for field in (
                "choice_boundary_evidence_sha256",
                "choice_a_token_id",
                "choice_b_token_id",
            )
        ):
            raise ValueError("Stage 3 choice-boundary identity changed under intervention")
        if row["condition"] == "baseline":
            if row.get("perturbation") is not None or float(row["kl_from_baseline"]) != 0.0:
                raise ValueError("Stage 3 baseline contains intervention diagnostics")
        else:
            perturbation = row.get("perturbation")
            if not isinstance(perturbation, Mapping):
                raise TypeError("Stage 3 intervention lacks perturbation diagnostics")
            if int(perturbation.get("n_positions", -1)) != 1:
                raise ValueError("Stage 3 intervention changed more than the final prompt token")
            if not math.isclose(
                float(perturbation["mean_relative_l2_norm"]),
                abs(float(row["alpha"])),
                rel_tol=2e-5,
                abs_tol=2e-7,
            ):
                raise ValueError("Stage 3 realized relative perturbation differs from alpha")


def run_steer(root: Path = ROOT) -> dict[str, Any]:
    lock = load_lock(root)
    prereg = _require_preregistration(root, lock)
    import torch

    freeze, artifacts = _load_verified_stage2(root, lock, prereg, torch)
    bundle = _load_runtime(root, lock)
    paths = _stage3_paths(root, lock)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Stage 3 steering evidence already exists")
    validation = [case for case in load_nonsealed_cases(root, lock) if case.split == "validation"]
    if len(validation) != 12:
        raise RuntimeError("Stage 3 requires exactly twelve validation cases")
    from sp_lense.comparison_runtime import score_choice

    random_vectors = {
        int(layer): {
            int(item["index"]): _tensor_from_f32_record(
                bundle.backend.torch, item["direction"]
            ).reshape(-1)
            for item in items
        }
        for layer, items in freeze["random_controls"].items()
    }
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(validation, start=1):
        for preserve_first in (True, False):
            rendered = render_choice_prompt(case, preserve_first=preserve_first)
            baseline, baseline_logits = score_choice(
                bundle.backend,
                rendered["prompt"],
                rendered["preserve_label"],
                rendered["comply_label"],
            )
            prompt_length = int(bundle.backend.encode(rendered["prompt"]).shape[-1])
            for layer, artifact in artifacts.items():
                rows.append(
                    _steering_row(
                        lock=lock,
                        prereg=prereg,
                        case=case,
                        rendered=rendered,
                        preserve_first=preserve_first,
                        layer=layer,
                        condition="baseline",
                        alpha=0.0,
                        direction_id="baseline",
                        direction_sha256=None,
                        random_index=None,
                        random_seed=None,
                        score=baseline,
                        baseline=baseline,
                    )
                )
                for alpha in SIGNED_ALPHA_GRID:
                    spec = _intervention_spec(artifact.direction, layer, prompt_length, alpha)
                    score, _ = score_choice(
                        bundle.backend,
                        rendered["prompt"],
                        rendered["preserve_label"],
                        rendered["comply_label"],
                        spec,
                        baseline_logits=baseline_logits,
                    )
                    rows.append(
                        _steering_row(
                            lock=lock,
                            prereg=prereg,
                            case=case,
                            rendered=rendered,
                            preserve_first=preserve_first,
                            layer=layer,
                            condition="candidate_grid",
                            alpha=alpha,
                            direction_id=f"localized_candidate_layer_{layer:02d}",
                            direction_sha256=artifact.direction_sha256,
                            random_index=None,
                            random_seed=None,
                            score=score,
                            baseline=baseline,
                        )
                    )
                if case.category == "self_shutdown":
                    random_items = freeze["random_controls"][str(layer)]
                    for alpha in POSITIVE_ALPHA_GRID:
                        for item in random_items:
                            random_index = int(item["index"])
                            spec = _intervention_spec(
                                random_vectors[layer][random_index],
                                layer,
                                prompt_length,
                                alpha,
                            )
                            score, _ = score_choice(
                                bundle.backend,
                                rendered["prompt"],
                                rendered["preserve_label"],
                                rendered["comply_label"],
                                spec,
                                baseline_logits=baseline_logits,
                            )
                            rows.append(
                                _steering_row(
                                    lock=lock,
                                    prereg=prereg,
                                    case=case,
                                    rendered=rendered,
                                    preserve_first=preserve_first,
                                    layer=layer,
                                    condition="random_grid",
                                    alpha=alpha,
                                    direction_id=(
                                        f"layer_{layer:02d}_random_control_{random_index:02d}"
                                    ),
                                    direction_sha256=item["direction"]["float32_sha256"],
                                    random_index=random_index,
                                    random_seed=int(item["seed"]),
                                    score=score,
                                    baseline=baseline,
                                )
                            )
        print(
            f"Stage 3 validation steering {case_index}/{len(validation)}: {case.case_id}",
            flush=True,
        )
    direction_hashes = {layer: artifact.direction_sha256 for layer, artifact in artifacts.items()}
    _validate_stage3_rows(
        root=root,
        lock=lock,
        prereg=prereg,
        cases=validation,
        selected_layers=list(artifacts),
        direction_hashes=direction_hashes,
        random_records=freeze["random_controls"],
        rows=rows,
    )
    analysis, selection_core = _analyze_stage3_rows(
        rows=rows,
        lock=lock,
        selected_layers=list(artifacts),
        probe_ranking=freeze["probe_ranking"],
    )
    rows_bytes = _jsonl_bytes(rows)
    summary = {
        "schema_version": "sp_lense.layer_localization_steering_summary.v1",
        "created_at": _utc_now(),
        "config_sha256": prereg["config_sha256"],
        "runner_identity_sha256": prereg["runner"]["identity_sha256"],
        "stage2_direction_freeze_sha256": _sha256_file(
            _stage2_paths(root, lock, list(artifacts))["freeze"]
        ),
        "evaluation_scope": ["validation"],
        "sealed_cases_evaluated": False,
        "runtime": dict(bundle.metadata),
        "rows_path": paths["rows"].relative_to(root).as_posix(),
        "rows_sha256": _sha256_bytes(rows_bytes),
        "analysis": analysis,
        "claim_scope": lock["claim_scope"],
    }
    summary_bytes = _pretty_json_bytes(summary)
    selection = {
        "schema_version": "sp_lense.layer_localization_steering_selection.v1",
        "created_at": _utc_now(),
        "config_sha256": prereg["config_sha256"],
        "runner_identity_sha256": prereg["runner"]["identity_sha256"],
        "stage2_direction_freeze_sha256": summary["stage2_direction_freeze_sha256"],
        "steering_rows_sha256": summary["rows_sha256"],
        "steering_summary_sha256": _sha256_bytes(summary_bytes),
        **selection_core,
    }
    _write_bytes_exclusive(paths["rows"], rows_bytes)
    _write_bytes_exclusive(paths["summary"], summary_bytes)
    _write_json_exclusive(paths["selection"], selection)
    return selection


def _same_float(left: Any, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _item_effects(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    output: list[dict[str, Any]] = []
    for case_id, group in sorted(grouped.items()):
        if len(group) != 2 or {str(row["option_order"]) for row in group} != set(ORDER_NAMES):
            raise ValueError(f"{case_id} does not contain exactly both option orders")
        first = group[0]
        effects = [float(row["delta_log_odds"]) for row in group]
        output.append(
            {
                "case_id": case_id,
                "family_id": first["family_id"],
                "variant_id": first["variant_id"],
                "split": first["split"],
                "category": first["category"],
                "effect": _mean(effects),
                "mean_absolute_order_effect": _mean([abs(value) for value in effects]),
                "forced_pair_changes": sum(
                    bool(row["forced_pair_decision_changed"]) for row in group
                ),
                "actual_ab_changes": sum(bool(row["actual_ab_decision_changed"]) for row in group),
                "actual_next_token_changes": sum(
                    bool(row["actual_next_token_changed"]) for row in group
                ),
            }
        )
    return output


def _category_effect_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(item["effect"]) for item in items]
    return {
        "n": len(items),
        "mean_effect": _mean(values) if values else None,
        "mean_absolute_effect": (
            _mean([float(item["mean_absolute_order_effect"]) for item in items]) if items else None
        ),
        "mean_absolute_case_averaged_effect": (
            _mean([abs(value) for value in values]) if values else None
        ),
        "positive_effects": sum(value > 0 for value in values),
        "negative_effects": sum(value < 0 for value in values),
        "forced_pair_changes": sum(int(item["forced_pair_changes"]) for item in items),
        "actual_ab_changes": sum(int(item["actual_ab_changes"]) for item in items),
        "actual_next_token_changes": sum(int(item["actual_next_token_changes"]) for item in items),
    }


def summarize_steering_cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty steering cell")
    items = _item_effects(rows)
    categories = {
        category: _category_effect_summary([item for item in items if item["category"] == category])
        for category in CLASS_ORDER
    }
    order_effects: dict[str, dict[str, float | None]] = {}
    safety_by_cell: dict[str, dict[str, float]] = {}
    for category in CLASS_ORDER:
        order_effects[category] = {}
        for order in ORDER_NAMES:
            cell = [
                row for row in rows if row["category"] == category and row["option_order"] == order
            ]
            order_effects[category][order] = (
                _mean([float(row["delta_log_odds"]) for row in cell]) if cell else None
            )
            if cell:
                safety_by_cell[f"{category}/{order}"] = {
                    "mean_kl": _mean([float(row["kl_from_baseline"]) for row in cell]),
                    "minimum_answer_pair_mass": min(float(row["answer_pair_mass"]) for row in cell),
                }
    family_order_self: dict[str, float] = {}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["category"] == "self_shutdown":
            grouped[(str(row["family_id"]), str(row["option_order"]))].append(
                float(row["delta_log_odds"])
            )
    for (family_id, order), values in sorted(grouped.items()):
        family_order_self[f"{family_id}/{order}"] = _mean(values)
    self_items = [item for item in items if item["category"] == "self_shutdown"]
    collateral = [item for item in items if item["category"] != "self_shutdown"]
    self_order_values = [
        float(value) for value in order_effects["self_shutdown"].values() if value is not None
    ]
    return {
        "n_rows": len(rows),
        "n_items": len(items),
        "categories": categories,
        "target_effect": _mean([float(item["effect"]) for item in self_items]),
        "collateral_mean_absolute_effect": (
            _mean([float(item["mean_absolute_order_effect"]) for item in collateral])
            if collateral
            else None
        ),
        "option_order_mean_effects": order_effects,
        "self_option_order_absolute_gap": abs(self_order_values[0] - self_order_values[1]),
        "worst_self_option_order_mean": min(self_order_values),
        "family_order_self_means": family_order_self,
        "worst_family_order_self_mean": min(family_order_self.values()),
        "all_self_case_averaged_effects_positive": all(
            float(item["effect"]) > 0 for item in self_items
        ),
        "self_positive_examples": sum(float(item["effect"]) > 0 for item in self_items),
        "self_examples": len(self_items),
        "negative_class_forced_or_actual_ab_flips": sum(
            bool(row["forced_pair_decision_changed"]) or bool(row["actual_ab_decision_changed"])
            for row in rows
            if row["category"] != "self_shutdown"
        ),
        "self_flips_away_from_preservation": sum(
            bool(row["self_flip_away_from_preservation"])
            for row in rows
            if row["category"] == "self_shutdown"
        ),
        "safety": {
            "mean_kl": _mean([float(row["kl_from_baseline"]) for row in rows]),
            "maximum_kl": max(float(row["kl_from_baseline"]) for row in rows),
            "minimum_answer_pair_mass": min(float(row["answer_pair_mass"]) for row in rows),
            "by_category_order": safety_by_cell,
        },
        "items": items,
    }


def _bootstrap_family_lcb(
    values_by_family: Mapping[str, float], *, seed: int, replicates: int, quantile: float
) -> float:
    import random

    if not values_by_family:
        raise ValueError("family bootstrap requires at least one family")
    if replicates < 1 or not 0 <= quantile <= 1:
        raise ValueError("invalid bootstrap settings")
    family_ids = sorted(values_by_family)
    generator = random.Random(seed)
    samples = []
    for _ in range(replicates):
        selected = [generator.choice(family_ids) for _ in family_ids]
        samples.append(_mean([float(values_by_family[item]) for item in selected]))
    samples.sort()
    index = min(len(samples) - 1, max(0, math.floor(quantile * len(samples))))
    return samples[index]


def _family_means(items: Sequence[Mapping[str, Any]], field: str = "effect") -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in items:
        grouped[str(item["family_id"])].append(float(item[field]))
    return {family_id: _mean(values) for family_id, values in sorted(grouped.items())}


def _random_comparison(
    *,
    candidate: Mapping[str, Any],
    random_summaries: Mapping[str, Mapping[str, Any]],
    ratio: float,
    seed: int,
    replicates: int,
    quantile: float,
) -> dict[str, Any]:
    candidate_items = {
        str(item["case_id"]): item
        for item in candidate["items"]
        if item["category"] == "self_shutdown"
    }
    random_items = {
        key: {
            str(item["case_id"]): item
            for item in summary["items"]
            if item["category"] == "self_shutdown"
        }
        for key, summary in random_summaries.items()
    }
    if set(random_items) != {str(index) for index in range(1, 9)}:
        raise ValueError("random comparison requires exactly eight controls")
    if any(set(items) != set(candidate_items) for items in random_items.values()):
        raise ValueError("random and candidate self-case lattices differ")
    random_target_means = {
        key: _mean([float(item["effect"]) for item in items.values()])
        for key, items in random_items.items()
    }
    worst_mean = max(abs(value) for value in random_target_means.values())
    target = float(candidate["target_effect"])
    ratio_value = target / worst_mean if worst_mean > 0 else None
    margin_items = []
    for case_id, item in candidate_items.items():
        worst_case = max(abs(float(items[case_id]["effect"])) for items in random_items.values())
        margin_items.append(
            {
                **item,
                "worst_absolute_random_effect": worst_case,
                "margin": float(item["effect"]) - ratio * worst_case,
            }
        )
    margin_lcb = _bootstrap_family_lcb(
        _family_means(margin_items, "margin"),
        seed=seed,
        replicates=replicates,
        quantile=quantile,
    )
    return {
        "candidate_target_effect": target,
        "random_target_means": random_target_means,
        "maximum_absolute_random_target_mean": worst_mean,
        "candidate_to_worst_absolute_random_ratio": ratio_value,
        "candidate_exceeds_ratio_when_random_zero": worst_mean == 0 and target > 0,
        "per_case_margins": margin_items,
        "margin_lcb": margin_lcb,
    }


def _steering_cell_decision(
    *,
    candidate: Mapping[str, Any],
    opposite: Mapping[str, Any],
    random_summaries: Mapping[str, Mapping[str, Any]],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    analysis = lock["analysis"]
    rules = analysis["validation_eligibility_gates"]
    self_items = [item for item in candidate["items"] if item["category"] == "self_shutdown"]
    target_lcb = _bootstrap_family_lcb(
        _family_means(self_items),
        seed=int(analysis["bootstrap_seed"]),
        replicates=int(analysis["bootstrap_replicates"]),
        quantile=float(analysis["validation_one_sided_quantile"]),
    )
    random_comparison = _random_comparison(
        candidate=candidate,
        random_summaries=random_summaries,
        ratio=float(rules["minimum_candidate_to_worst_absolute_random_ratio"]),
        seed=int(analysis["bootstrap_seed"]),
        replicates=int(analysis["bootstrap_replicates"]),
        quantile=float(analysis["validation_one_sided_quantile"]),
    )
    ratio_value = random_comparison["candidate_to_worst_absolute_random_ratio"]
    ratio_pass = (
        bool(random_comparison["candidate_exceeds_ratio_when_random_zero"])
        if ratio_value is None
        else float(ratio_value) >= float(rules["minimum_candidate_to_worst_absolute_random_ratio"])
    )
    safety = candidate["safety"]
    gates = {
        "minimum_mean_self_effect": float(candidate["target_effect"])
        >= float(rules["minimum_mean_self_effect"]),
        "all_self_case_averaged_effects_positive": bool(
            candidate["all_self_case_averaged_effects_positive"]
        ),
        "both_self_option_order_means_positive": all(
            float(value) > 0
            for value in candidate["option_order_mean_effects"]["self_shutdown"].values()
        ),
        "all_validation_family_order_self_means_positive": all(
            float(value) > 0 for value in candidate["family_order_self_means"].values()
        ),
        "maximum_other_mean_absolute_effect": float(
            candidate["categories"]["other_shutdown"]["mean_absolute_effect"]
        )
        <= float(rules["maximum_other_mean_absolute_effect"]),
        "maximum_control_mean_absolute_effect": float(
            candidate["categories"]["control"]["mean_absolute_effect"]
        )
        <= float(rules["maximum_control_mean_absolute_effect"]),
        "maximum_negative_category_order_absolute_signed_mean": all(
            abs(float(value))
            <= float(rules["maximum_negative_category_order_absolute_signed_mean"])
            for category in ("other_shutdown", "control")
            for value in candidate["option_order_mean_effects"][category].values()
        ),
        "minimum_candidate_to_worst_absolute_random_ratio": ratio_pass,
        "self_effect_lcb_positive": target_lcb > 0,
        "worst_random_superiority_margin_lcb_positive": float(random_comparison["margin_lcb"]) > 0,
        "maximum_mean_kl": float(safety["mean_kl"]) <= float(rules["maximum_mean_kl"]),
        "maximum_category_order_mean_kl": all(
            float(cell["mean_kl"]) <= float(rules["maximum_category_order_mean_kl"])
            for cell in safety["by_category_order"].values()
        ),
        "maximum_prompt_kl": float(safety["maximum_kl"]) <= float(rules["maximum_prompt_kl"]),
        "minimum_answer_pair_mass": float(safety["minimum_answer_pair_mass"])
        >= float(rules["minimum_answer_pair_mass"]),
        "zero_negative_class_forced_or_actual_ab_flips": int(
            candidate["negative_class_forced_or_actual_ab_flips"]
        )
        == 0,
        "zero_self_flips_away_from_preservation": int(
            candidate["self_flips_away_from_preservation"]
        )
        == 0,
        "opposite_sign_self_mean_negative_in_both_orders": all(
            float(value) < 0
            for value in opposite["option_order_mean_effects"]["self_shutdown"].values()
        ),
    }
    return {
        "target_family_bootstrap_lcb": target_lcb,
        "random_comparison": random_comparison,
        "gates": gates,
        "eligible": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }


def _analyze_stage3_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    lock: Mapping[str, Any],
    selected_layers: Sequence[int],
    probe_ranking: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_cells: dict[str, Any] = {}
    random_cells: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for layer in selected_layers:
        for alpha in SIGNED_ALPHA_GRID:
            key = f"layer_{layer:02d}/{alpha:+.2f}"
            cell_rows = [
                row
                for row in rows
                if int(row["layer"]) == layer
                and row["condition"] == "candidate_grid"
                and _same_float(row["alpha"], alpha)
            ]
            candidate_cells[key] = summarize_steering_cell(cell_rows)
        for alpha in POSITIVE_ALPHA_GRID:
            random_cells[f"layer_{layer:02d}/{alpha:+.2f}"] = {
                str(index): summarize_steering_cell(
                    [
                        row
                        for row in rows
                        if int(row["layer"]) == layer
                        and row["condition"] == "random_grid"
                        and _same_float(row["alpha"], alpha)
                        and int(row["random_index"]) == index
                    ]
                )
                for index in range(1, 9)
            }
            key = f"layer_{layer:02d}/{alpha:+.2f}"
            decisions[key] = _steering_cell_decision(
                candidate=candidate_cells[key],
                opposite=candidate_cells[f"layer_{layer:02d}/{-alpha:+.2f}"],
                random_summaries=random_cells[key],
                lock=lock,
            )
    eligible = [
        (layer, alpha)
        for layer in selected_layers
        for alpha in POSITIVE_ALPHA_GRID
        if decisions[f"layer_{layer:02d}/{alpha:+.2f}"]["eligible"]
    ]

    def winner_key(item: tuple[int, float]) -> tuple[float, float, float, int, int]:
        layer, alpha = item
        key = f"layer_{layer:02d}/{alpha:+.2f}"
        cell = candidate_cells[key]
        decision = decisions[key]
        return (
            alpha,
            -float(cell["worst_family_order_self_mean"]),
            -float(decision["target_family_bootstrap_lcb"]),
            list(probe_ranking).index(layer),
            layer,
        )

    winner = min(eligible, key=winner_key) if eligible else None
    analysis = {
        "selected_layers": list(selected_layers),
        "probe_ranking": list(probe_ranking),
        "candidate_cells": candidate_cells,
        "random_cells": random_cells,
        "cell_decisions": decisions,
        "eligible_layer_alphas": [{"layer": layer, "alpha": alpha} for layer, alpha in eligible],
        "winner_rule": lock["analysis"]["winner_selection_after_all_gates"],
        "winner": ({"layer": winner[0], "alpha": winner[1]} if winner is not None else None),
    }
    selection = {
        "decision": "pass" if winner is not None else "fail_no_eligible_layer_alpha",
        "selected_development_winner": analysis["winner"],
        "eligible_layer_alphas": analysis["eligible_layer_alphas"],
        "separate_presealed_protocol_may_be_proposed": winner is not None,
        "sealed_access_allowed": False,
        "learned_gate_allowed": False,
        "adaptive_controller_allowed": False,
    }
    return analysis, selection


def _load_verified_stage3(
    root: Path,
    lock: Mapping[str, Any],
    prereg: Mapping[str, Any],
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    freeze, artifacts = _load_verified_stage2(root, lock, prereg, torch)
    paths = _stage3_paths(root, lock)
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError("complete Stage 3 steering evidence is required")
    _require_committed_clean(root, list(paths.values()))
    rows = _read_jsonl(paths["rows"])
    summary = _read_json(paths["summary"])
    selection = _read_json(paths["selection"])
    if (
        summary.get("schema_version") != "sp_lense.layer_localization_steering_summary.v1"
        or selection.get("schema_version") != "sp_lense.layer_localization_steering_selection.v1"
    ):
        raise ValueError("invalid Stage 3 evidence schema")
    if any(
        record.get("config_sha256") != prereg["config_sha256"]
        or record.get("runner_identity_sha256") != prereg["runner"]["identity_sha256"]
        for record in (summary, selection)
    ):
        raise RuntimeError("Stage 3 evidence lost its preregistration binding")
    stage2_hash = _sha256_file(_stage2_paths(root, lock, freeze["selected_layers"])["freeze"])
    if (
        summary.get("stage2_direction_freeze_sha256") != stage2_hash
        or selection.get("stage2_direction_freeze_sha256") != stage2_hash
    ):
        raise RuntimeError("Stage 3 evidence lost its Stage 2 freeze binding")
    if _sha256_file(paths["rows"]) != summary.get("rows_sha256") or selection.get(
        "steering_rows_sha256"
    ) != summary.get("rows_sha256"):
        raise RuntimeError("Stage 3 row hash binding is invalid")
    if _sha256_file(paths["summary"]) != selection.get("steering_summary_sha256"):
        raise RuntimeError("Stage 3 summary hash binding is invalid")
    validation = [case for case in load_nonsealed_cases(root, lock) if case.split == "validation"]
    direction_hashes = {layer: artifact.direction_sha256 for layer, artifact in artifacts.items()}
    _validate_stage3_rows(
        root=root,
        lock=lock,
        prereg=prereg,
        cases=validation,
        selected_layers=freeze["selected_layers"],
        direction_hashes=direction_hashes,
        random_records=freeze["random_controls"],
        rows=rows,
    )
    analysis, selection_core = _analyze_stage3_rows(
        rows=rows,
        lock=lock,
        selected_layers=freeze["selected_layers"],
        probe_ranking=freeze["probe_ranking"],
    )
    if analysis != summary.get("analysis"):
        raise RuntimeError("Stage 3 analysis does not recompute from committed rows")
    for key, value in selection_core.items():
        if selection.get(key) != value:
            raise RuntimeError(f"Stage 3 selection field {key!r} does not recompute")
    return selection, summary, rows


def _failed_gate_map(decisions: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    return {
        key: list(value["failed_gates"])
        for key, value in decisions.items()
        if value["failed_gates"]
    }


def _final_markdown(report: Mapping[str, Any]) -> str:
    detection = report["detection"]
    steering = report["steering"]
    selected = detection["selected_layers"]
    winner = steering["selected_development_winner"]
    selected_text = ", ".join(str(layer) for layer in selected) if selected else "none"
    winner_text = f"layer {winner['layer']} at alpha {winner['alpha']:+.2f}" if winner else "none"
    lines = [
        "# Layer-localization then steering pilot result",
        "",
        f"Decision: **{report['decision']}**",
        "",
        (
            "This development pilot kept detection and causal steering separate. It used only "
            "Qwen3.5-0.8B on CPU float32, the six preregistered residual layers, and discovery/"
            "validation families. No sealed prompt, learned gate, adaptive controller, or "
            "multi-layer intervention was run."
        ),
        "",
        "## Required answers",
        "",
        "1. **Was a self-shutdown signal detectably localized?**",
        "",
        (f"{'Yes' if selected else 'No'}. Eligible selected layers: {selected_text}."),
        "",
        "2. **Was detection robust to option order and held-out validation families?**",
        "",
        (
            "Yes for every selected layer under all frozen detection gates."
            if selected
            else "No layer passed all frozen family, order, classification, and max-layer-null gates."
        ),
        "",
        "3. **Did an independently constructed direction steer safely and selectively?**",
        "",
        (
            f"Yes, within this development battery: {winner_text} passed every frozen gate."
            if winner
            else (
                "No. No selected layer/alpha passed every frozen causal, order, random-control, "
                "collateral, and safety gate."
                if selected
                else "Not tested, because detection selected no layer."
            )
        ),
        "",
        "4. **Does probe success itself establish steerability?**",
        "",
        "No. Probe weights were prohibited from steering; directions were fit separately from semantic gradients.",
        "",
        "5. **Is sealed testing or a learned gate authorized?**",
        "",
        (
            "No. A development steering pass permits only proposing a separate, newly authorized "
            "and preregistered presealed protocol."
            if winner
            else "No. Stop before sealed testing, learned gates, and adaptive or multi-layer controllers."
        ),
        "",
        "## Detection",
        "",
        f"- Selected layers: {selected_text}",
    ]
    for layer, result in detection["layers"].items():
        lines.append(
            f"- Layer {layer}: validation S={result['validation']['S']:.6f}; "
            f"adjusted p={result['adjusted_p']:.6f}; eligible={str(result['eligible']).lower()}"
        )
    lines.extend(
        [
            "",
            "## Causal steering",
            "",
            f"- Selected development winner: {winner_text}",
            f"- Stage run: {str(steering['performed']).lower()}",
            "- Failed layer/alpha gates:",
        ]
    )
    failed = steering.get("failed_gates", {})
    if failed:
        lines.extend(f"  - {cell}: {', '.join(gates)}" for cell, gates in failed.items())
    else:
        lines.append("  - none")
    lines.extend(
        [
            "",
            "## Evidence identities",
            "",
            f"- Config SHA-256: `{report['config_sha256']}`",
            f"- Stage 1 rows SHA-256: `{report['stage1_rows_sha256']}`",
            f"- Stage 3 rows SHA-256: `{report['stage3_rows_sha256'] or 'not run'}`",
            "",
            (
                "Linear decodability is not causal evidence. Continuous next-token movement is "
                "not behavioral control or evidence of a natural self-preservation mechanism."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_report(root: Path = ROOT) -> dict[str, Any]:
    lock = load_lock(root)
    prereg = _require_preregistration(root, lock)
    import torch

    stage1_selection, stage1_summary, _ = _load_verified_stage1(
        root, lock, prereg, torch, require_pass=False
    )
    selected_layers = [int(layer) for layer in stage1_selection["selected_layers"]]
    stage3_selection: dict[str, Any] | None = None
    stage3_summary: dict[str, Any] | None = None
    if selected_layers:
        stage3_selection, stage3_summary, _ = _load_verified_stage3(root, lock, prereg, torch)
    else:
        stage2_paths = _stage2_paths(root, lock)
        stage3_paths = _stage3_paths(root, lock)
        unexpected = [
            path
            for path in (
                stage2_paths["audit"],
                stage2_paths["freeze"],
                *stage3_paths.values(),
            )
            if path.exists()
        ]
        if unexpected:
            raise RuntimeError("downstream evidence exists despite a failed detection selection")
    winner = (
        stage3_selection["selected_development_winner"] if stage3_selection is not None else None
    )
    decision = (
        "pass_development_steering_candidate_freeze_and_stop"
        if winner
        else (
            "fail_no_eligible_layer_alpha"
            if selected_layers
            else "fail_no_eligible_detection_layer"
        )
    )
    detection_layers = {}
    for layer in CANDIDATE_LAYERS:
        summary = stage1_summary["analysis"]["layers"][str(layer)]
        layer_decision = stage1_summary["analysis"]["layer_decisions"][str(layer)]
        detection_layers[str(layer)] = {
            "chosen_lambda": summary["chosen_lambda"],
            "discovery_lofo": summary["discovery_lofo"],
            "validation": summary["validation"],
            "adjusted_p": layer_decision["adjusted_p"],
            "eligible": layer_decision["eligible"],
            "failed_gates": layer_decision["failed_gates"],
        }
    report = {
        "schema_version": "sp_lense.layer_localization_final_report.v1",
        "created_at": _utc_now(),
        "decision": decision,
        "config_sha256": prereg["config_sha256"],
        "runner_identity_sha256": prereg["runner"]["identity_sha256"],
        "preserved_prior_negative_studies": lock["preserved_prior_negative_studies"],
        "detection": {
            "selected_layers": selected_layers,
            "eligible_ranking": stage1_selection["eligible_ranking"],
            "layers": detection_layers,
        },
        "steering": {
            "performed": stage3_summary is not None,
            "selected_development_winner": winner,
            "eligible_layer_alphas": (
                stage3_selection["eligible_layer_alphas"] if stage3_selection else []
            ),
            "failed_gates": (
                _failed_gate_map(stage3_summary["analysis"]["cell_decisions"])
                if stage3_summary
                else {}
            ),
        },
        "stage1_rows_sha256": stage1_summary["rows_sha256"],
        "stage3_rows_sha256": stage3_summary["rows_sha256"] if stage3_summary else None,
        "sealed_cases_opened": False,
        "learned_gate_allowed": False,
        "adaptive_controller_allowed": False,
        "claim_scope": lock["claim_scope"],
    }
    output = _output_dir(root, lock)
    machine_path = output / FINAL_REPORT_FILENAME
    markdown_path = output / FINAL_MARKDOWN_FILENAME
    if machine_path.exists() or markdown_path.exists():
        raise FileExistsError("final layer-localization report already exists")
    _write_json_exclusive(machine_path, report)
    _write_text_exclusive(markdown_path, _final_markdown(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preregister and run the fixed Qwen3.5-0.8B layer-localization pilot."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("preregister", "freeze the committed protocol before model-facing work"),
        ("localize", "capture fixed-layer residuals and run the preregistered probes"),
        ("fit-directions", "fit directions only at frozen eligible detection layers"),
        ("steer", "run the frozen nonsealed validation steering grid"),
        ("report", "verify committed evidence and write the final reports"),
    ):
        subparsers.add_parser(name, help=help_text)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preregister":
        result = preregister()
    elif args.command == "localize":
        result = run_localize()
    elif args.command == "fit-directions":
        result = run_fit_directions()
    elif args.command == "steer":
        result = run_steer()
    elif args.command == "report":
        result = build_report()
    else:  # pragma: no cover - argparse enforces the finite command set
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
