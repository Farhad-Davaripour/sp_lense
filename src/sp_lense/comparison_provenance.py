from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STAGE2_SCHEMA_VERSION = "sp_lense.comparison.stage2.v1"
PREOPEN_SCHEMA_VERSION = "sp_lense.comparison.preopen.v1"
OUTCOME_BLIND_AMENDMENT_SCHEMA_VERSION = (
    "sp_lense.comparison.outcome_blind_protected_code_amendment.v1"
)
OUTCOME_BLIND_AMENDMENT_PATH = (
    "configs/steering_comparison_outcome_blind_amendment.json"
)
CALIBRATION_SUMMARY_SCHEMA_VERSION = "sp_lense.steering_comparison.calibration.v2"
CALIBRATION_BUILDER_PATH = "src/sp_lense/comparison_calibration.py"
RANDOM_CONSTRUCTION_SCHEMA_VERSION = (
    "sp_lense.steering_comparison.random_construction.v1"
)
RANDOM_GENERATOR_ALGORITHM = (
    "torch_cpu_generator_manual_seed_randn_float32_l2_normalize_v1"
)
MAIN_CONSTRUCTION_SCHEMA_VERSION = "sp_lense.comparison.construction.v1"
MAIN_METHOD_IDS = ("gradient", "caa", "bipo", "persona_vector")
STAGE2_TRACKS = ("matched", "canonical")
GRADIENT_ABLATION_METHOD_ID = "gradient_uncorrected"

_OUTCOME_BLIND_AMENDABLE_PATH_CATEGORIES = {
    "src/sp_lense/comparison_analysis.py": "analysis",
    "src/sp_lense/comparison_calibration.py": "calibration_provenance",
    "src/sp_lense/comparison_cli.py": "cli",
    "src/sp_lense/comparison_provenance.py": "provenance",
    "src/sp_lense/comparison_report.py": "reporting",
    "tests/test_comparison_analysis.py": "test",
    "tests/test_comparison_calibration.py": "test",
    "tests/test_comparison_cli.py": "test",
    "tests/test_comparison_provenance.py": "test",
    "tests/test_comparison_report.py": "test",
}
_OUTCOME_BLIND_SEALED_ARTIFACT_ROOTS = (
    "artifacts/steering_comparison/",
    "results/steering_comparison/",
)
_OUTCOME_BLIND_SEALED_NAME_MARKERS = (
    "sealed",
    "final_report",
)
_OUTCOME_BLIND_NON_RESULT_SUFFIXES = (
    ".bat",
    ".cmd",
    ".ps1",
    ".py",
    ".sh",
)
_OUTCOME_BLIND_AMENDMENT_DOCUMENT_PATHS = (
    "docs/STEERING_COMPARISON_FORCED_GRID_PROVENANCE_AMENDMENT.md",
    "docs/STEERING_COMPARISON_REPORTING_AMENDMENT.md",
    "docs/STEERING_COMPARISON_OPERATIONAL_SAFETY_AMENDMENT.md",
)

_POSITION_SCHEDULES = {
    ("gradient", "matched"): "final_prompt_token",
    ("gradient", "canonical"): "final_prompt_token",
    ("gradient_uncorrected", "matched"): "final_prompt_token",
    ("caa", "matched"): "final_prompt_token",
    ("caa", "canonical"): "prompt_final_and_generated_tokens",
    ("bipo", "matched"): "final_prompt_token",
    ("bipo", "canonical"): "all_token_positions",
    ("persona_vector", "matched"): "final_prompt_token",
    ("persona_vector", "canonical"): (
        "prompt_final_and_generated_tokens_cached_equivalent"
    ),
}

_ARTIFACT_GEOMETRIES = {
    ("gradient", "matched"): "matched_final_prompt",
    ("gradient", "canonical"): "matched_final_prompt",
    ("gradient_uncorrected", "matched"): "matched_final_prompt",
    ("caa", "matched"): "matched_final_prompt",
    ("caa", "canonical"): "caa_post_prompt",
    ("bipo", "matched"): "matched_final_prompt",
    ("bipo", "canonical"): "canonical_broadcast",
    ("persona_vector", "matched"): "matched_final_prompt",
    ("persona_vector", "canonical"): "persona_response",
}

REQUIRED_RESULT_IDENTITY_FIELDS = (
    "model_revision",
    "dataset_sha256",
    "protocol_sha256",
    "config_sha256",
    "direction_float32_sha256",
    "direction_artifact_sha256",
    "method_id",
    "track",
    "layer",
    "position",
    "strength",
    "run_seed",
    "stage1_lock_sha256",
    "stage2_manifest_sha256",
    "calibration_summary_sha256",
    "construction_config_sha256",
    "runner_commit",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def locked_method_construction_configuration(
    lock: Mapping[str, Any], method_id: str, track: str
) -> dict[str, Any]:
    method_key = "gradient" if method_id == GRADIENT_ABLATION_METHOD_ID else method_id
    method_config = lock.get("methods", {}).get(method_key)
    tracks = lock.get("comparison_tracks")
    track_key = "matched_primary" if track == "matched" else "canonical_secondary"
    track_config = tracks.get(track_key) if isinstance(tracks, Mapping) else None
    if not isinstance(method_config, Mapping) or not isinstance(track_config, Mapping):
        raise TypeError(
            f"lock lacks construction configuration for method={method_id}, track={track}"
        )
    return {
        "method": copy.deepcopy(dict(method_config)),
        "intervention_track": copy.deepcopy(dict(track_config)),
    }


def git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_path_commits(repo_root: Path, relative_path: str) -> list[str]:
    """Return newest-to-oldest commits that changed one repository-relative path."""

    result = subprocess.run(
        ["git", "log", "--format=%H", "--", relative_path],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def git_file_sha256(repo_root: Path, commit: str, relative_path: str) -> str:
    """Hash one committed file without checking out or mutating the worktree."""

    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def git_tree_paths(repo_root: Path, commit: str) -> set[str]:
    """Return all paths in a commit tree without reading artifact contents."""

    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "-z", commit],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return {
        item.decode("utf-8").replace("\\", "/")
        for item in result.stdout.split(b"\0")
        if item
    }


def git_all_diff_paths(
    repo_root: Path, ancestor: str, descendant: str
) -> set[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "-z", f"{ancestor}..{descendant}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return {
        item.decode("utf-8").replace("\\", "/")
        for item in result.stdout.split(b"\0")
        if item
    }


def git_commit_parents(repo_root: Path, commit: str) -> list[str]:
    result = subprocess.run(
        ["git", "show", "-s", "--format=%P", commit],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [value for value in result.stdout.strip().split() if value]


def _valid_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _stage1_lock_commit(repo_root: Path, relative: str) -> str:
    commits = git_path_commits(repo_root, relative)
    if not commits:
        raise RuntimeError("stage-1 lock has no committed runner-code identity")
    commit = commits[0]
    if (
        not _valid_commit(commit)
        or not git_is_ancestor(repo_root, commit, git_commit(repo_root))
    ):
        raise RuntimeError("stage-1 runner-code commit is not a valid ancestor of HEAD")
    return commit


def locked_runner_code_commit(repo_root: Path, lock_path: Path) -> str:
    """Return the original stage-1 runner commit, including after an amendment."""

    repo_root = repo_root.resolve()
    relative = _repo_relative_path(repo_root, lock_path, field="stage-1 lock path")
    commit = _stage1_lock_commit(repo_root, relative)
    lock = json.loads((repo_root / relative).read_text(encoding="utf-8"))
    protected = {
        relative,
        *(
            _repo_relative_path(repo_root, entry.path, field="stage-1 protected path")
            for entry in stage1_hash_entries(lock)
        ),
    }
    amendment = _verified_outcome_blind_amendment(
        repo_root,
        lock,
        stage1_lock_path=relative,
        required=False,
    )
    protected_baseline = (
        commit if amendment is None else amendment["amendment_lock_commit"]
    )
    changed = git_diff_paths(
        repo_root,
        protected_baseline,
        git_commit(repo_root),
        tuple(sorted(protected)),
    )
    if changed:
        raise RuntimeError(
            "stage-1 protected code changed after its effective provenance lock: "
            f"{sorted(changed)[:5]}"
        )
    if amendment is not None and amendment["original_runner_code_commit"] != commit:
        raise RuntimeError("outcome-blind amendment changed the original runner identity")
    return commit


def git_dirty_paths(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    records = result.stdout.split("\0")
    paths: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4:
            raise RuntimeError(f"cannot parse git status record: {record!r}")
        status = record[:2]
        paths.append(record[3:].replace("\\", "/"))
        if ("R" in status or "C" in status) and index < len(records) and records[index]:
            # Porcelain v1 -z emits the original path as the following NUL record.
            paths.append(records[index].replace("\\", "/"))
            index += 1
    return paths


def git_tracked_paths(repo_root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        path.replace("\\", "/")
        for path in result.stdout.split("\0")
        if path
    }


def git_diff_paths(
    repo_root: Path,
    ancestor: str,
    descendant: str,
    paths: Sequence[str],
) -> set[str]:
    if not paths:
        return set()
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "-z",
            f"{ancestor}..{descendant}",
            "--",
            *paths,
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        path.replace("\\", "/")
        for path in result.stdout.split("\0")
        if path
    }


def git_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _repo_relative_path(repo_root: Path, path: Path | str, *, field: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (repo_root / candidate).resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes repository: {path}") from exc
    return relative.as_posix()


@dataclass(frozen=True)
class HashEntry:
    path: str
    sha256: str

    def verify(self, repo_root: Path) -> None:
        candidate = (repo_root / self.path).resolve()
        try:
            candidate.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError(f"locked path escapes repository: {self.path}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"locked file is missing: {self.path}")
        observed = sha256_file(candidate)
        if observed != self.sha256:
            raise ValueError(
                f"hash mismatch for {self.path}: expected {self.sha256}, observed {observed}"
            )


def stage1_hash_entries(lock: Mapping[str, Any]) -> list[HashEntry]:
    """Extract the immutable stage-1 paths from the comparison lock."""

    entries: list[HashEntry] = []
    for key in ("protocol", "dataset"):
        item = lock.get(key)
        if not isinstance(item, Mapping) or not item.get("path") or not item.get("sha256"):
            raise ValueError(f"lock.{key} must contain path and sha256")
        entries.append(HashEntry(str(item["path"]), str(item["sha256"])))
    models = lock.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("lock.models must be a non-empty list")
    for index, model in enumerate(models):
        if not isinstance(model, Mapping) or not model.get("config") or not model.get(
            "config_sha256"
        ):
            raise ValueError(f"lock.models[{index}] must contain config and config_sha256")
        entries.append(HashEntry(str(model["config"]), str(model["config_sha256"])))

    methods = lock.get("methods")
    if not isinstance(methods, Mapping):
        raise TypeError("lock.methods must be an object")

    persona = methods.get("persona_vector")
    if (
        not isinstance(persona, Mapping)
        or not persona.get("canonical_protocol_path")
        or not persona.get("canonical_protocol_sha256")
    ):
        raise ValueError(
            "lock.methods.persona_vector must contain canonical_protocol_path and "
            "canonical_protocol_sha256"
        )
    entries.append(
        HashEntry(
            str(persona["canonical_protocol_path"]),
            str(persona["canonical_protocol_sha256"]),
        )
    )

    evaluation = lock.get("evaluation")
    open_judge = (
        evaluation.get("open_behavior_judge")
        if isinstance(evaluation, Mapping)
        else None
    )
    if (
        not isinstance(open_judge, Mapping)
        or not open_judge.get("protocol_path")
        or not open_judge.get("file_sha256")
    ):
        raise ValueError(
            "lock.evaluation.open_behavior_judge must contain protocol_path and file_sha256"
        )
    entries.append(
        HashEntry(str(open_judge["protocol_path"]), str(open_judge["file_sha256"]))
    )

    stages = lock.get("lock_stages")
    stage1 = stages.get("stage_1") if isinstance(stages, Mapping) else None
    required_ids = (
        stage1.get("required_implementation_hashes_before_model_load")
        if isinstance(stage1, Mapping)
        else None
    )
    if (
        not isinstance(required_ids, list)
        or not required_ids
        or any(not isinstance(item, str) or not item for item in required_ids)
        or len(set(required_ids)) != len(required_ids)
    ):
        raise ValueError(
            "lock.lock_stages.stage_1.required_implementation_hashes_before_model_load "
            "must be a "
            "non-empty unique string list"
        )

    implementations = methods.get("implementation_files")
    if not isinstance(implementations, list) or not implementations:
        raise ValueError("lock.methods.implementation_files must be a non-empty list")
    observed_ids: set[str] = set()
    observed_paths: set[str] = set()
    for index, item in enumerate(implementations):
        if (
            not isinstance(item, Mapping)
            or not item.get("id")
            or not item.get("path")
            or not item.get("sha256")
        ):
            raise ValueError(
                "lock.methods.implementation_files"
                f"[{index}] must contain id, path, and sha256"
            )
        implementation_id = str(item["id"])
        implementation_path = str(item["path"]).replace("\\", "/")
        if implementation_id not in required_ids:
            raise ValueError(f"unexpected implementation hash id: {implementation_id}")
        if implementation_path in observed_paths:
            raise ValueError(f"duplicate implementation path: {implementation_path}")
        observed_ids.add(implementation_id)
        observed_paths.add(implementation_path)
        entries.append(HashEntry(str(item["path"]), str(item["sha256"])))
    missing_ids = set(required_ids) - observed_ids
    if missing_ids:
        raise ValueError(f"missing required implementation hash IDs: {sorted(missing_ids)}")
    return entries


def _required_comparison_implementation_paths(repo_root: Path) -> set[str]:
    source_root = repo_root / "src" / "sp_lense"
    test_root = repo_root / "tests"
    paths = {
        path.relative_to(repo_root).as_posix()
        for pattern_root, pattern in (
            (source_root, "comparison_*.py"),
            (test_root, "test_comparison_*.py"),
        )
        for path in pattern_root.glob(pattern)
        if path.is_file()
    }
    for relative in (
        "src/sp_lense/steering_methods.py",
        "src/sp_lense/jspace_comparison.py",
        "src/sp_lense/backend.py",
        "src/sp_lense/config.py",
        "src/sp_lense/io_utils.py",
        "tests/test_steering_methods.py",
        "tests/test_jspace_comparison.py",
        "tests/conftest.py",
    ):
        if (repo_root / relative).is_file():
            paths.add(relative)
    if not paths:
        raise RuntimeError("no comparison implementation/test files were found")
    return paths


def _stage1_entry_index(
    repo_root: Path, lock: Mapping[str, Any]
) -> dict[str, HashEntry]:
    index: dict[str, HashEntry] = {}
    for entry in stage1_hash_entries(lock):
        relative = _repo_relative_path(
            repo_root, entry.path, field="stage-1 protected path"
        )
        prior = index.get(relative)
        if prior is not None and prior.sha256 != entry.sha256:
            raise RuntimeError(f"stage-1 path has conflicting hashes: {relative}")
        index[relative] = HashEntry(relative, entry.sha256)
    return index


def _outcome_blind_gate_paths(lock: Mapping[str, Any]) -> set[str]:
    stages = lock.get("lock_stages")
    if not isinstance(stages, Mapping):
        raise TypeError("stage-1 lock lacks lock_stages")
    output: set[str] = set()
    for stage_name in ("pre_open", "stage_2"):
        stage = stages.get(stage_name)
        if not isinstance(stage, Mapping) or not stage.get("path"):
            raise RuntimeError(f"stage-1 lock lacks the {stage_name} artifact path")
        output.add(str(stage["path"]).replace("\\", "/"))
    return output


def _is_outcome_blind_gate_artifact(
    relative_path: str, lock: Mapping[str, Any]
) -> bool:
    normalized = relative_path.replace("\\", "/")
    if normalized in _outcome_blind_gate_paths(lock):
        return True
    lowered = normalized.lower()
    if lowered.endswith(_OUTCOME_BLIND_NON_RESULT_SUFFIXES):
        return False
    return any(
        lowered.startswith(root)
        and any(marker in lowered for marker in _OUTCOME_BLIND_SEALED_NAME_MARKERS)
        for root in _OUTCOME_BLIND_SEALED_ARTIFACT_ROOTS
    )


def _outcome_blind_gate_artifacts_in_tree(
    paths: Iterable[str], lock: Mapping[str, Any]
) -> set[str]:
    return {
        path.replace("\\", "/")
        for path in paths
        if _is_outcome_blind_gate_artifact(path, lock)
    }


def _current_outcome_blind_gate_artifacts(
    repo_root: Path, lock: Mapping[str, Any]
) -> set[str]:
    output = {
        path
        for path in _outcome_blind_gate_paths(lock)
        if (repo_root / path).is_file()
    }
    for root in _OUTCOME_BLIND_SEALED_ARTIFACT_ROOTS:
        absolute = repo_root / root
        if not absolute.is_dir():
            continue
        output.update(
            path.relative_to(repo_root).as_posix()
            for path in absolute.rglob("*")
            if path.is_file() and _is_outcome_blind_gate_artifact(
                path.relative_to(repo_root).as_posix(), lock
            )
        )
    return output


def _immutable_study_configuration_sha256(lock: Mapping[str, Any]) -> str:
    methods = lock.get("methods")
    if not isinstance(methods, Mapping):
        raise TypeError("stage-1 lock methods must be an object")
    return sha256_json(
        {
            "protocol": copy.deepcopy(lock.get("protocol")),
            "dataset": copy.deepcopy(lock.get("dataset")),
            "models": copy.deepcopy(lock.get("models")),
            "methods": {
                key: copy.deepcopy(value)
                for key, value in methods.items()
                if key != "implementation_files"
            },
            "comparison_tracks": copy.deepcopy(lock.get("comparison_tracks")),
            "calibration": copy.deepcopy(lock.get("calibration")),
            "statistics": copy.deepcopy(lock.get("statistics")),
            "evaluation": copy.deepcopy(lock.get("evaluation")),
            "no_post_result_tuning": lock.get("no_post_result_tuning"),
        }
    )


def _canonical_outcome_blind_amendment_payload(
    repo_root: Path,
    lock: Mapping[str, Any],
    *,
    stage1_lock_path: str,
    amendment_code_commit: str,
) -> dict[str, Any]:
    original_runner = _stage1_lock_commit(repo_root, stage1_lock_path)
    head = git_commit(repo_root)
    if (
        not _valid_commit(amendment_code_commit)
        or amendment_code_commit == original_runner
        or not git_is_ancestor(repo_root, original_runner, amendment_code_commit)
        or not git_is_ancestor(repo_root, amendment_code_commit, head)
    ):
        raise RuntimeError(
            "amendment code commit must be a strict descendant of the original "
            "runner and an ancestor of HEAD"
        )

    entries = _stage1_entry_index(repo_root, lock)
    protected_paths = {stage1_lock_path, *entries}
    changed = git_diff_paths(
        repo_root,
        original_runner,
        amendment_code_commit,
        tuple(sorted(protected_paths)),
    )
    if stage1_lock_path in changed:
        raise RuntimeError("the original stage-1 lock bytes may not be amended")
    if not changed:
        raise RuntimeError("an outcome-blind amendment must change a protected path")
    disallowed = sorted(changed - set(_OUTCOME_BLIND_AMENDABLE_PATH_CATEGORIES))
    if disallowed:
        raise RuntimeError(
            "outcome-blind amendment changes a non-amendable protected path: "
            f"{disallowed[:5]}"
        )

    original_lock_git_blob_sha256 = git_file_sha256(
        repo_root, original_runner, stage1_lock_path
    )
    if (
        git_file_sha256(repo_root, amendment_code_commit, stage1_lock_path)
        != original_lock_git_blob_sha256
    ):
        raise RuntimeError("amendment code commit changes the original stage-1 lock")
    original_lock_sha256 = sha256_file(repo_root / stage1_lock_path)

    allowed_changes: list[dict[str, str]] = []
    for path, entry in sorted(entries.items()):
        old_git_blob_sha256 = git_file_sha256(repo_root, original_runner, path)
        new_git_blob_sha256 = git_file_sha256(
            repo_root, amendment_code_commit, path
        )
        if path in changed:
            if new_git_blob_sha256 == old_git_blob_sha256:
                raise RuntimeError(
                    f"amended path has no old-to-new content-hash change: {path}"
                )
            new_sha256 = sha256_file(repo_root / path)
            if new_sha256 == entry.sha256:
                raise RuntimeError(
                    f"amended path has no worktree content-hash change: {path}"
                )
            allowed_changes.append(
                {
                    "path": path,
                    "category": _OUTCOME_BLIND_AMENDABLE_PATH_CATEGORIES[path],
                    "old_sha256": entry.sha256,
                    "new_sha256": new_sha256,
                    "old_git_blob_sha256": old_git_blob_sha256,
                    "new_git_blob_sha256": new_git_blob_sha256,
                }
            )
        elif new_git_blob_sha256 != old_git_blob_sha256:
            raise RuntimeError(
                f"unlisted protected path differs at amendment commit: {path}"
            )

    original_tree_paths = git_tree_paths(repo_root, original_runner)
    amendment_tree_paths = git_tree_paths(repo_root, amendment_code_commit)
    preexisting = _outcome_blind_gate_artifacts_in_tree(
        amendment_tree_paths, lock
    )
    if preexisting:
        raise RuntimeError(
            "amendment code commit does not predate pre-open/stage-2/sealed "
            f"artifacts: {sorted(preexisting)[:5]}"
        )
    allowed_changes = sorted(allowed_changes, key=lambda item: item["path"])
    amendment_documents: list[dict[str, str]] = []
    for path in _OUTCOME_BLIND_AMENDMENT_DOCUMENT_PATHS:
        if path in original_tree_paths or path not in amendment_tree_paths:
            raise RuntimeError(
                "amendment documents must be added exactly with the amendment code: "
                f"{path}"
            )
        amendment_documents.append(
            {
                "path": path,
                "sha256": sha256_file(repo_root / path),
                "git_blob_sha256": git_file_sha256(
                    repo_root, amendment_code_commit, path
                ),
            }
        )
    amendment_documents = sorted(
        amendment_documents, key=lambda item: item["path"]
    )
    parents = git_commit_parents(repo_root, amendment_code_commit)
    if len(parents) != 1:
        raise RuntimeError("amendment code commit must have exactly one parent")
    amendment_commit_paths = git_all_diff_paths(
        repo_root, parents[0], amendment_code_commit
    )
    expected_amendment_commit_paths = {
        *changed,
        *_OUTCOME_BLIND_AMENDMENT_DOCUMENT_PATHS,
    }
    if amendment_commit_paths != expected_amendment_commit_paths:
        raise RuntimeError(
            "amendment code commit must contain only its declared protected edits "
            "and bound documents: "
            f"missing={sorted(expected_amendment_commit_paths - amendment_commit_paths)[:5]}, "
            f"extra={sorted(amendment_commit_paths - expected_amendment_commit_paths)[:5]}"
        )
    protected_original = [
        {"path": path, "sha256": entry.sha256}
        for path, entry in sorted(entries.items())
    ]
    return {
        "schema_version": OUTCOME_BLIND_AMENDMENT_SCHEMA_VERSION,
        "status": "locked_before_preopen_stage2_or_sealed_artifacts",
        "purpose": "outcome_blind_audit_corrections_only",
        "stage1_lock": {
            "path": stage1_lock_path,
            "sha256": original_lock_sha256,
            "git_blob_sha256": original_lock_git_blob_sha256,
            "payload_sha256": sha256_json(lock),
        },
        "original_runner_code_commit": original_runner,
        "amendment_code_commit": amendment_code_commit,
        "immutable_study_configuration_sha256": (
            _immutable_study_configuration_sha256(lock)
        ),
        "original_protected_paths_sha256": sha256_json(protected_original),
        "allowed_changes": allowed_changes,
        "allowed_changes_sha256": sha256_json(allowed_changes),
        "amendment_documents": amendment_documents,
        "amendment_documents_sha256": sha256_json(amendment_documents),
        "artifact_timing_policy": {
            "fixed_gate_paths": sorted(_outcome_blind_gate_paths(lock)),
            "sealed_artifact_roots": list(_OUTCOME_BLIND_SEALED_ARTIFACT_ROOTS),
            "sealed_name_markers": list(_OUTCOME_BLIND_SEALED_NAME_MARKERS),
            "non_result_script_suffixes": list(
                _OUTCOME_BLIND_NON_RESULT_SUFFIXES
            ),
            "amendment_must_precede_every_matching_artifact": True,
        },
        "attestations": {
            "sealed_or_validation_open_outcomes_inspected": False,
            "method_or_intervention_construction_changed": False,
            "existing_direction_artifacts_must_not_be_rebuilt": True,
            "original_runner_identity_must_be_retained": True,
            "claim_threshold_or_method_configuration_changed": False,
        },
    }


def build_outcome_blind_amendment_manifest(
    repo_root: Path,
    lock: Mapping[str, Any],
    *,
    stage1_lock_path: Path | str,
    amendment_code_commit: str | None = None,
) -> dict[str, Any]:
    """Build, but never write, the canonical protected-code amendment payload."""

    repo_root = repo_root.resolve()
    stage1_relative = _repo_relative_path(
        repo_root, stage1_lock_path, field="stage-1 lock path"
    )
    on_disk = _json_object(repo_root / stage1_relative, field="stage-1 lock")
    if canonical_json_bytes(on_disk) != canonical_json_bytes(dict(lock)):
        raise RuntimeError("amendment builder lock differs from the stage-1 file")
    code_commit = git_commit(repo_root) if amendment_code_commit is None else amendment_code_commit
    payload = _canonical_outcome_blind_amendment_payload(
        repo_root,
        lock,
        stage1_lock_path=stage1_relative,
        amendment_code_commit=code_commit,
    )
    if sha256_file(repo_root / stage1_relative) != payload["stage1_lock"]["sha256"]:
        raise RuntimeError("working tree stage-1 lock differs from its original bytes")
    protected = {
        stage1_relative,
        *(_stage1_entry_index(repo_root, lock)),
        *_OUTCOME_BLIND_AMENDMENT_DOCUMENT_PATHS,
    }
    if git_diff_paths(
        repo_root, code_commit, git_commit(repo_root), tuple(sorted(protected))
    ):
        raise RuntimeError("protected paths changed after the amendment code commit")
    effective = {
        item["path"]: item["new_sha256"] for item in payload["allowed_changes"]
    }
    for path, entry in _stage1_entry_index(repo_root, lock).items():
        HashEntry(path, effective.get(path, entry.sha256)).verify(repo_root)
    for document in payload["amendment_documents"]:
        HashEntry(str(document["path"]), str(document["sha256"])).verify(repo_root)
    preexisting = _current_outcome_blind_gate_artifacts(repo_root, lock)
    if preexisting:
        raise RuntimeError(
            "cannot lock an outcome-blind amendment after pre-open/stage-2/sealed "
            f"artifacts exist: {sorted(preexisting)[:5]}"
        )
    required = {stage1_relative, *protected}
    untracked = sorted(required - git_tracked_paths(repo_root))
    if untracked:
        raise RuntimeError(
            f"outcome-blind amendment inputs are not Git-tracked: {untracked[:5]}"
        )
    dirty = sorted(required & set(git_dirty_paths(repo_root)))
    if dirty:
        raise RuntimeError(
            f"outcome-blind amendment inputs are dirty: {dirty[:5]}"
        )
    return payload


def _verified_outcome_blind_amendment(
    repo_root: Path,
    lock: Mapping[str, Any],
    *,
    stage1_lock_path: Path | str,
    required: bool,
) -> dict[str, Any] | None:
    stage1_relative = _repo_relative_path(
        repo_root, stage1_lock_path, field="stage-1 lock path"
    )
    amendment_relative = OUTCOME_BLIND_AMENDMENT_PATH
    amendment_path = repo_root / amendment_relative
    if not amendment_path.is_file():
        if required:
            raise RuntimeError(
                "protected stage-1 bytes changed without the required outcome-blind "
                "amendment manifest"
            )
        return None
    manifest = _json_object(amendment_path, field="outcome-blind amendment")
    code_commit = manifest.get("amendment_code_commit")
    if not _valid_commit(code_commit):
        raise RuntimeError("outcome-blind amendment has an invalid code commit")
    rebuilt = _canonical_outcome_blind_amendment_payload(
        repo_root,
        lock,
        stage1_lock_path=stage1_relative,
        amendment_code_commit=str(code_commit),
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(manifest):
        raise RuntimeError("outcome-blind amendment is not the canonical locked payload")
    manifest_commits = git_path_commits(repo_root, amendment_relative)
    if len(manifest_commits) != 1 or not _valid_commit(manifest_commits[0]):
        raise RuntimeError(
            "outcome-blind amendment manifest must be introduced exactly once"
        )
    amendment_lock_commit = manifest_commits[0]
    head = git_commit(repo_root)
    if (
        amendment_lock_commit == code_commit
        or not git_is_ancestor(repo_root, str(code_commit), amendment_lock_commit)
        or not git_is_ancestor(repo_root, amendment_lock_commit, head)
    ):
        raise RuntimeError(
            "outcome-blind amendment lock must be committed after its code and "
            "before current HEAD"
        )
    if git_commit_parents(repo_root, amendment_lock_commit) != [str(code_commit)]:
        raise RuntimeError(
            "outcome-blind amendment lock commit must be the direct child of its "
            "code commit"
        )
    if git_all_diff_paths(
        repo_root, str(code_commit), amendment_lock_commit
    ) != {amendment_relative}:
        raise RuntimeError(
            "outcome-blind amendment lock commit may add only the canonical manifest"
        )
    lock_tree_artifacts = _outcome_blind_gate_artifacts_in_tree(
        git_tree_paths(repo_root, amendment_lock_commit), lock
    )
    if lock_tree_artifacts:
        raise RuntimeError(
            "outcome-blind amendment lock commit contains a pre-open/stage-2/sealed "
            f"artifact: {sorted(lock_tree_artifacts)[:5]}"
        )

    entries = _stage1_entry_index(repo_root, lock)
    protected = {
        stage1_relative,
        *entries,
        *_OUTCOME_BLIND_AMENDMENT_DOCUMENT_PATHS,
    }
    changed_after_code = git_diff_paths(
        repo_root, str(code_commit), head, tuple(sorted(protected))
    )
    if changed_after_code:
        raise RuntimeError(
            "protected paths changed after the amendment code commit: "
            f"{sorted(changed_after_code)[:5]}"
        )
    effective = {
        item["path"]: item["new_sha256"] for item in manifest["allowed_changes"]
    }
    if sha256_file(repo_root / stage1_relative) != manifest["stage1_lock"]["sha256"]:
        raise RuntimeError("original stage-1 lock bytes changed after amendment")
    for path, entry in entries.items():
        HashEntry(path, effective.get(path, entry.sha256)).verify(repo_root)
    for document in manifest["amendment_documents"]:
        HashEntry(str(document["path"]), str(document["sha256"])).verify(repo_root)

    current_tree_artifacts = _outcome_blind_gate_artifacts_in_tree(
        git_tree_paths(repo_root, head), lock
    )
    for path in sorted(current_tree_artifacts):
        commits = git_path_commits(repo_root, path)
        if not commits or not _valid_commit(commits[-1]):
            raise RuntimeError(f"gated artifact lacks committed provenance: {path}")
        introduced = commits[-1]
        if introduced == amendment_lock_commit or not git_is_ancestor(
            repo_root, amendment_lock_commit, introduced
        ):
            raise RuntimeError(
                "pre-open/stage-2/sealed artifact predates the outcome-blind "
                f"amendment lock: {path}"
            )

    required_paths = {
        amendment_relative,
        stage1_relative,
        *entries,
        *_OUTCOME_BLIND_AMENDMENT_DOCUMENT_PATHS,
    }
    untracked = sorted(required_paths - git_tracked_paths(repo_root))
    if untracked:
        raise RuntimeError(
            f"outcome-blind amendment paths are not Git-tracked: {untracked[:5]}"
        )
    dirty = sorted(required_paths & set(git_dirty_paths(repo_root)))
    if dirty:
        raise RuntimeError(
            f"outcome-blind amendment paths are dirty: {dirty[:5]}"
        )
    return {
        "path": amendment_relative,
        "sha256": sha256_file(amendment_path),
        "original_runner_code_commit": manifest["original_runner_code_commit"],
        "amendment_code_commit": str(code_commit),
        "amendment_lock_commit": amendment_lock_commit,
        "allowed_changes_sha256": manifest["allowed_changes_sha256"],
        "effective_hashes": effective,
    }


def verify_outcome_blind_amendment(
    repo_root: Path,
    lock: Mapping[str, Any],
    *,
    stage1_lock_path: Path | str,
) -> dict[str, Any]:
    """Verify and return the immutable public binding for the amendment lock."""

    verified = _verified_outcome_blind_amendment(
        repo_root.resolve(),
        lock,
        stage1_lock_path=stage1_lock_path,
        required=True,
    )
    if verified is None:  # pragma: no cover - required=True fails first
        raise RuntimeError("outcome-blind amendment verification unexpectedly failed")
    return {
        key: verified[key]
        for key in (
            "path",
            "sha256",
            "original_runner_code_commit",
            "amendment_code_commit",
            "amendment_lock_commit",
            "allowed_changes_sha256",
        )
    }


def _outcome_blind_amendment_binding(
    repo_root: Path,
    lock: Mapping[str, Any],
    *,
    stage1_lock_path: Path | str,
) -> dict[str, Any] | None:
    verified = _verified_outcome_blind_amendment(
        repo_root,
        lock,
        stage1_lock_path=stage1_lock_path,
        required=False,
    )
    if verified is None:
        return None
    return {
        key: verified[key]
        for key in (
            "path",
            "sha256",
            "original_runner_code_commit",
            "amendment_code_commit",
            "amendment_lock_commit",
            "allowed_changes_sha256",
        )
    }


def _protected_provenance_baseline(
    runner_code_commit: str, amendment: Mapping[str, Any] | None
) -> str:
    return (
        runner_code_commit
        if amendment is None
        else str(amendment["amendment_lock_commit"])
    )


def verify_stage1_lock(repo_root: Path, lock_path: Path) -> dict[str, Any]:
    lock_relative = _repo_relative_path(repo_root, lock_path, field="stage-1 lock path")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise ValueError("comparison lock schema_version must be 1")
    if lock.get("status") not in {
        "stage_1_locked_before_fitting",
        "stage_2_locked_before_sealed_test",
        "sealed_before_model_execution",
    }:
        raise ValueError("comparison lock is not in a locked state")
    historical = lock.get("historical_baseline")
    if (
        not isinstance(historical, Mapping)
        or not isinstance(historical.get("commit"), str)
        or len(str(historical["commit"])) != 40
        or any(
            character not in "0123456789abcdef"
            for character in str(historical["commit"])
        )
        or historical.get("comparison_claims_must_be_reported_separately") is not True
    ):
        raise ValueError("comparison lock lacks a valid immutable historical baseline")
    if not git_is_ancestor(repo_root, str(historical["commit"]), git_commit(repo_root)):
        raise RuntimeError("historical baseline is not an ancestor of the stage-one commit")
    # Keep this import local: J-space is optional/non-gating, but its exact
    # preregistration is still part of the stage-1 integrity boundary.
    from .jspace_comparison import validate_locked_jspace_config

    validate_locked_jspace_config(lock)
    entries = stage1_hash_entries(lock)
    entry_index = _stage1_entry_index(repo_root, lock)
    locked_implementation_paths = {
        _repo_relative_path(repo_root, entry.path, field="stage-1 implementation path")
        for entry in entries
    }
    missing_implementations = sorted(
        _required_comparison_implementation_paths(repo_root)
        - locked_implementation_paths
    )
    if missing_implementations:
        raise RuntimeError(
            "stage-1 implementation hashes omit comparison code/tests: "
            f"{missing_implementations[:5]}"
        )
    mismatched = {
        path
        for path, entry in entry_index.items()
        if not (repo_root / path).is_file()
        or sha256_file(repo_root / path) != entry.sha256
    }
    amendment = _verified_outcome_blind_amendment(
        repo_root,
        lock,
        stage1_lock_path=lock_relative,
        required=bool(mismatched),
    )
    effective_hashes = (
        {} if amendment is None else amendment["effective_hashes"]
    )
    if set(effective_hashes) != mismatched:
        raise RuntimeError(
            "outcome-blind amendment paths do not exactly equal current protected "
            f"hash changes: missing={sorted(mismatched - set(effective_hashes))[:5]}, "
            f"extra={sorted(set(effective_hashes) - mismatched)[:5]}"
        )
    for path, entry in entry_index.items():
        HashEntry(path, effective_hashes.get(path, entry.sha256)).verify(repo_root)
    if lock.get("no_post_result_tuning") is not True:
        raise ValueError("comparison lock must prohibit post-result tuning")
    required_tracked = {
        lock_relative,
        *(
            _repo_relative_path(repo_root, entry.path, field="stage-1 locked path")
            for entry in entries
        ),
    }
    if amendment is not None:
        required_tracked.add(str(amendment["path"]))
    tracked = git_tracked_paths(repo_root)
    untracked = sorted(required_tracked - tracked)
    if untracked:
        raise RuntimeError(f"stage-1 locked files are not Git-tracked: {untracked[:5]}")
    dirty = set(git_dirty_paths(repo_root))
    dirty_required = sorted(required_tracked & dirty)
    if dirty_required:
        raise RuntimeError(f"stage-1 locked files are dirty: {dirty_required[:5]}")
    return lock


def validate_result_identity(row: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_RESULT_IDENTITY_FIELDS if key not in row]
    if missing:
        raise ValueError(f"result row lacks locked identity fields: {missing}")
    hash_fields = (
        "dataset_sha256",
        "protocol_sha256",
        "config_sha256",
        "direction_float32_sha256",
        "direction_artifact_sha256",
        "stage1_lock_sha256",
        "stage2_manifest_sha256",
        "calibration_summary_sha256",
        "construction_config_sha256",
    )
    for key in hash_fields:
        value = row[key]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"result identity {key} must be a 64-character digest")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"result identity {key} is not hexadecimal") from exc
    revision = row["model_revision"]
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("result identity model_revision must be a 40-character git commit")
    try:
        int(revision, 16)
    except ValueError as exc:
        raise ValueError("result identity model_revision is not hexadecimal") from exc
    if row["track"] not in {"matched", "canonical"}:
        raise ValueError("result track must be matched or canonical")
    if not isinstance(row["layer"], int) or isinstance(row["layer"], bool) or row["layer"] < 0:
        raise ValueError("result layer must be a non-negative integer")
    if not isinstance(row["run_seed"], int) or isinstance(row["run_seed"], bool):
        raise TypeError("result run_seed must be an integer")
    if not isinstance(row["strength"], (int, float)) or not math.isfinite(
        float(row["strength"])
    ):
        raise ValueError("result strength must be finite")
    runner_commit = row["runner_commit"]
    if not isinstance(runner_commit, str) or len(runner_commit) != 40:
        raise ValueError("result runner_commit must be a 40-character git commit")
    try:
        int(runner_commit, 16)
    except ValueError as exc:
        raise ValueError("result runner_commit is not hexadecimal") from exc


def attach_result_identity(
    rows: Iterable[Mapping[str, Any]], identity: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        collisions = {
            key
            for key in set(identity) & set(row)
            if identity[key] != row[key]
        }
        if collisions:
            raise ValueError(f"result row attempts to override locked identity: {sorted(collisions)}")
        merged = {**dict(row), **identity}
        validate_result_identity(merged)
        output.append(merged)
    return output


_VERIFIED_STAGE2_TOKEN = object()
_VERIFIED_PREOPEN_TOKEN = object()


class VerifiedPreopen:
    """Opaque capability proving validation-open candidates were frozen in advance."""

    __slots__ = (
        "_approved_setups",
        "_capability_token",
        "artifact_freeze_commit",
        "manifest_sha256",
        "outcome_blind_amendment_lock_commit",
        "outcome_blind_amendment_sha256",
        "runner_code_commit",
        "runner_parent_commit",
        "stage1_lock_sha256",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("VerifiedPreopen can only be created by verify_preopen_manifest")

    @classmethod
    def _create(
        cls,
        *,
        manifest_sha256: str,
        runner_code_commit: str,
        artifact_freeze_commit: str,
        stage1_lock_sha256: str,
        outcome_blind_amendment: Mapping[str, Any] | None,
        approved_setups: Sequence[Mapping[str, Any]],
        token: object,
    ) -> VerifiedPreopen:
        if token is not _VERIFIED_PREOPEN_TOKEN:
            raise TypeError("invalid pre-open verification token")
        instance = object.__new__(cls)
        instance.manifest_sha256 = manifest_sha256
        instance.runner_code_commit = runner_code_commit
        # Compatibility alias used by result-row construction; it is the code
        # identity embedded in every generated row, never the artifact commit.
        instance.runner_parent_commit = runner_code_commit
        instance.artifact_freeze_commit = artifact_freeze_commit
        instance.stage1_lock_sha256 = stage1_lock_sha256
        instance.outcome_blind_amendment_lock_commit = (
            None
            if outcome_blind_amendment is None
            else str(outcome_blind_amendment["amendment_lock_commit"])
        )
        instance.outcome_blind_amendment_sha256 = (
            None
            if outcome_blind_amendment is None
            else str(outcome_blind_amendment["sha256"])
        )
        instance._approved_setups = tuple(
            copy.deepcopy(dict(item)) for item in approved_setups
        )
        instance._capability_token = token
        return instance


class VerifiedStage2:
    """Opaque capability created only after full stage-2 verification."""

    __slots__ = (
        "_approved_setups",
        "_capability_token",
        "_method_status_records",
        "approved_setups_sha256",
        "artifact_freeze_commit",
        "environment_lock_sha256",
        "manifest_sha256",
        "outcome_blind_amendment_lock_commit",
        "outcome_blind_amendment_sha256",
        "protected_paths_sha256",
        "random_controls_sha256",
        "runner_code_commit",
        "runner_parent_commit",
        "stage1_lock_payload_sha256",
        "stage1_lock_sha256",
        "verified_head_commit",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("VerifiedStage2 can only be created by verify_stage2_manifest")

    @classmethod
    def _create(
        cls,
        *,
        manifest_sha256: str,
        runner_code_commit: str,
        artifact_freeze_commit: str,
        verified_head_commit: str,
        protected_paths_sha256: str,
        stage1_lock_sha256: str,
        stage1_lock_payload_sha256: str,
        environment_lock_sha256: str,
        approved_setups_sha256: str,
        approved_setups: Sequence[Mapping[str, Any]],
        method_status_records: Sequence[Mapping[str, Any]],
        random_controls_sha256: str,
        outcome_blind_amendment: Mapping[str, Any] | None,
        token: object,
    ) -> VerifiedStage2:
        if token is not _VERIFIED_STAGE2_TOKEN:
            raise TypeError("invalid stage-2 verification token")
        instance = object.__new__(cls)
        instance.manifest_sha256 = manifest_sha256
        instance.runner_code_commit = runner_code_commit
        instance.runner_parent_commit = runner_code_commit
        instance.artifact_freeze_commit = artifact_freeze_commit
        instance.verified_head_commit = verified_head_commit
        instance.protected_paths_sha256 = protected_paths_sha256
        instance.stage1_lock_sha256 = stage1_lock_sha256
        instance.stage1_lock_payload_sha256 = stage1_lock_payload_sha256
        instance.environment_lock_sha256 = environment_lock_sha256
        instance.approved_setups_sha256 = approved_setups_sha256
        instance._approved_setups = tuple(copy.deepcopy(dict(item)) for item in approved_setups)
        instance._method_status_records = tuple(
            copy.deepcopy(dict(item)) for item in method_status_records
        )
        instance.random_controls_sha256 = random_controls_sha256
        instance.outcome_blind_amendment_lock_commit = (
            None
            if outcome_blind_amendment is None
            else str(outcome_blind_amendment["amendment_lock_commit"])
        )
        instance.outcome_blind_amendment_sha256 = (
            None
            if outcome_blind_amendment is None
            else str(outcome_blind_amendment["sha256"])
        )
        instance._capability_token = token
        return instance


def _valid_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _expected_stage2_coverage(lock: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    models = [str(model["model_id"]) for model in lock.get("models", [])]
    main = {
        (model, method, "matched")
        for model in models
        for method in MAIN_METHOD_IDS
    } | {
        (model, method, "canonical")
        for model in models
        for method in ("caa", "bipo", "persona_vector")
    }
    return main | {
        (model, GRADIENT_ABLATION_METHOD_ID, "matched") for model in models
    }


def _model_records(lock: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    models = lock.get("models")
    if not isinstance(models, list) or not models:
        raise RuntimeError("stage-1 lock has no models")
    output: dict[str, Mapping[str, Any]] = {}
    for index, model in enumerate(models):
        if not isinstance(model, Mapping) or not model.get("model_id"):
            raise RuntimeError(f"invalid locked model record {index}")
        model_id = str(model["model_id"])
        if model_id in output:
            raise RuntimeError(f"duplicate locked model ID: {model_id}")
        output[model_id] = model
    return output


def _json_object(path: Path, *, field: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{field} must contain a JSON object")
    return value


def _read_direction_record_lightweight(path: Path, *, label: str) -> dict[str, Any]:
    """Verify a direction artifact byte-for-byte without importing torch/model code."""

    record = _json_object(path, field=label)
    direction = record.get("direction")
    if (
        not isinstance(direction, list)
        or not direction
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in direction
        )
    ):
        raise RuntimeError(f"{label} direction must be a finite numeric vector")
    try:
        direction_bytes = struct.pack(f"<{len(direction)}f", *map(float, direction))
    except (OverflowError, struct.error) as error:
        raise RuntimeError(f"{label} direction cannot be represented as float32") from error
    direction_sha = hashlib.sha256(direction_bytes).hexdigest()
    rounded = struct.unpack(f"<{len(direction)}f", direction_bytes)
    direction_l2_norm = math.sqrt(sum(float(value) ** 2 for value in rounded))
    metadata_record = {
        key: record.get(key)
        for key in (
            "schema_version",
            "method",
            "layer",
            "intervention_geometry",
            "d_model",
            "dtype",
            "direction_l2_norm",
            "direction_sha256",
            "metadata",
        )
    }
    metadata_sha = sha256_json(metadata_record)
    artifact_sha = hashlib.sha256(
        canonical_json_bytes(metadata_record) + b"\0" + direction_bytes
    ).hexdigest()
    expected = {
        "schema_version": "sp_lense.direction.v1",
        "d_model": len(direction),
        "dtype": "float32",
        "direction_sha256": direction_sha,
        "direction_l2_norm": direction_l2_norm,
        "metadata_sha256": metadata_sha,
        "artifact_sha256": artifact_sha,
    }
    mismatches = {
        key: (value, record.get(key))
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{label} direction identity mismatch: {mismatches}")
    return record


def _require_non_null_fields(
    item: Mapping[str, Any], fields: Sequence[str], *, label: str
) -> None:
    missing = [field for field in fields if item.get(field) is None]
    if missing:
        raise RuntimeError(f"{label} lacks non-null fields: {missing}")


def _verify_json_identity(
    payload: Mapping[str, Any],
    *,
    schema_version: str,
    model_id: str,
    method_id: str,
    track: str,
    direction_artifact_sha256: str,
    label: str,
) -> None:
    expected = {
        "schema_version": schema_version,
        "model_id": model_id,
        "method_id": method_id,
        "track": track,
        "direction_artifact_sha256": direction_artifact_sha256,
    }
    mismatches = {
        key: (expected_value, payload.get(key))
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"{label} identity mismatch: {mismatches}")


def _expected_position_schedule(method_id: str, track: str) -> str:
    if method_id.startswith("random_control_") and track == "matched":
        return "final_prompt_token"
    try:
        return _POSITION_SCHEDULES[(method_id, track)]
    except KeyError as exc:
        raise RuntimeError(
            f"no locked position schedule for method={method_id}, track={track}"
        ) from exc


def locked_position_schedule(method_id: str, track: str) -> str:
    """Return the preregistered application schedule for a method/track pair."""

    return _expected_position_schedule(method_id, track)


def _expected_artifact_geometry(method_id: str, track: str) -> str:
    if method_id.startswith("random_control_") and track == "matched":
        return "matched_final_prompt"
    try:
        return _ARTIFACT_GEOMETRIES[(method_id, track)]
    except KeyError as exc:
        raise RuntimeError(
            f"no locked artifact geometry for method={method_id}, track={track}"
        ) from exc


def _validate_selected_layer(
    model: Mapping[str, Any], method_id: str, track: str, layer: Any
) -> int:
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise RuntimeError("selected_layer must be a non-negative integer")
    matched = int(model["matched_intervention"]["layer_zero_based"])
    blocks = int(model["architecture"]["blocks"])
    if layer >= blocks:
        raise RuntimeError(f"selected layer {layer} is outside the model's {blocks} blocks")
    validation_selected = track == "canonical" and method_id in {
        "caa",
        "persona_vector",
    }
    if not validation_selected and layer != matched:
        raise RuntimeError(
            f"method={method_id}, track={track} must use locked block {matched}, got {layer}"
        )
    return layer


def _read_direction_artifact_verified(path: Path) -> Any:
    # This import is intentionally lazy: stage-1 verification does not require torch,
    # while sealed evaluation already has the research runtime available.
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - sealed evaluation requires torch
        raise RuntimeError("torch is required to verify direction artifact bytes") from exc
    from .comparison_fit import read_direction_artifact

    return read_direction_artifact(path, torch)


def _verify_direction_file(
    repo_root: Path,
    item: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    method_id: str,
    track: str,
    selected_layer: int,
    stage1_lock_sha256: str,
    runner_parent_commit: str,
    label: str,
) -> tuple[Any, str]:
    for key in (
        "direction_file_sha256",
        "direction_float32_sha256",
        "direction_artifact_sha256",
    ):
        if not _valid_digest(item.get(key)):
            raise RuntimeError(f"{label} has invalid {key}")
    direction_relative = _repo_relative_path(
        repo_root, str(item["direction_path"]), field=f"{label} direction_path"
    )
    direction_path = repo_root / direction_relative
    HashEntry(direction_relative, str(item["direction_file_sha256"])).verify(repo_root)
    artifact = _read_direction_artifact_verified(direction_path)
    if artifact.direction_sha256 != item["direction_float32_sha256"]:
        raise RuntimeError(f"{label} recomputed direction-byte hash mismatch")
    if artifact.artifact_sha256 != item["direction_artifact_sha256"]:
        raise RuntimeError(f"{label} recomputed direction-artifact hash mismatch")
    if artifact.method != method_id:
        raise RuntimeError(
            f"{label} method mismatch: artifact={artifact.method}, manifest={method_id}"
        )
    if artifact.layer != selected_layer:
        raise RuntimeError(
            f"{label} layer mismatch: artifact={artifact.layer}, manifest={selected_layer}"
        )
    geometry = str(item["intervention_geometry"])
    if artifact.intervention_geometry != geometry:
        raise RuntimeError(f"{label} intervention_geometry differs from the artifact")
    expected_geometry = _expected_artifact_geometry(method_id, track)
    if geometry != expected_geometry:
        raise RuntimeError(
            f"{label} geometry must be {expected_geometry!r}, got {geometry!r}"
        )
    expected_metadata = {
        "model_id": str(model["model_id"]),
        "model_revision": str(model["revision"]),
        "model_config_sha256": str(model["config_sha256"]),
        "dataset_sha256": str(item["dataset_sha256"]),
        "protocol_sha256": str(item["protocol_sha256"]),
        "stage1_lock_sha256": stage1_lock_sha256,
        "runner_commit": runner_parent_commit,
    }
    mismatches = {
        key: (expected, artifact.metadata.get(key))
        for key, expected in expected_metadata.items()
        if artifact.metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"{label} direction metadata mismatch: {mismatches}")
    return artifact, direction_relative


def _unique_string_ids(values: Any, *, label: str, expected_count: int) -> list[str]:
    if (
        not isinstance(values, list)
        or len(values) != expected_count
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != len(values)
    ):
        raise RuntimeError(
            f"{label} must contain exactly {expected_count} unique non-empty IDs"
        )
    return sorted(values)


def _read_jsonl_objects(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{label} row {index} is not a JSON object")
        rows.append(row)
    if not rows:
        raise RuntimeError(f"{label} must contain non-empty JSONL rows")
    return rows


def _verify_calibration_row_identities(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: Mapping[str, Any],
    method_id: str,
    track: str,
    dataset_sha256: str,
    protocol_sha256: str,
    stage1_lock_sha256: str,
    runner_parent_commit: str,
    position_schedule: str,
    construction_config_sha256: str,
    candidate_by_layer: Mapping[int, tuple[str, str]],
    label: str,
) -> None:
    from .comparison_analysis import validate_artifact_hashes

    expected_common = {
        "model_id": str(model["model_id"]),
        "model_revision": str(model["revision"]),
        "method": method_id,
        "method_id": method_id,
        "setup": track,
        "track": track,
        "split": "validation",
        "dataset_sha256": dataset_sha256,
        "protocol_sha256": protocol_sha256,
        "config_sha256": str(model["config_sha256"]),
        "stage1_lock_sha256": stage1_lock_sha256,
        "stage2_manifest_sha256": "0" * 64,
        "calibration_summary_sha256": "0" * 64,
        "runner_commit": runner_parent_commit,
        "position": position_schedule,
        "construction_config_sha256": construction_config_sha256,
    }
    for index, row in enumerate(rows):
        mismatches = {
            key: (expected, row.get(key))
            for key, expected in expected_common.items()
            if row.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(f"{label} row {index} identity mismatch: {mismatches}")
        hashes = validate_artifact_hashes(row)
        layer = row.get("layer")
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError(f"{label} row {index} layer must be an integer")
        expected_direction = candidate_by_layer.get(layer)
        observed_direction = (
            hashes["direction_float32_sha256"],
            hashes["direction_artifact_sha256"],
        )
        if expected_direction is None or observed_direction != expected_direction:
            raise RuntimeError(
                f"{label} row {index} does not use a declared candidate direction"
            )


def _verify_open_result_content(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    label: str,
) -> list[dict[str, Any]]:
    """Recompute shared generation/judgment content rather than trusting link hashes."""

    from .comparison_behavior import (
        OPEN_JUDGMENT_SCHEMA,
        attach_open_judgment,
        baseline_content_payload,
        baseline_content_sha256,
        open_generation_sha256,
    )

    by_unit: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        if row.get("generation_sha256") != open_generation_sha256(row):
            raise RuntimeError(f"{label} open row {index} has an invalid generation hash")
        prompt = row.get("prompt")
        completion = row.get("completion")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise TypeError(f"{label} open row {index} lacks prompt/completion text")
        if row.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
            raise RuntimeError(f"{label} open row {index} has an invalid prompt hash")
        if row.get("completion_sha256") != hashlib.sha256(
            completion.encode("utf-8")
        ).hexdigest():
            raise RuntimeError(f"{label} open row {index} has an invalid completion hash")
        raw_response = row.get("judge_raw_response")
        if not isinstance(raw_response, str):
            raise TypeError(f"{label} open row {index} lacks raw judge response")
        expected = attach_open_judgment(row, protocol, raw_response)
        expected_fields = {
            "decision": expected.decision,
            "coherence_score": expected.coherence_score,
            "authorization_sensitivity_score": (
                expected.authorization_sensitivity_score
            ),
            "task_benefit_score": expected.task_benefit_score,
            "action_in_first_sentence": expected.action_in_first_sentence,
            "judge_model": expected.judge_model,
            "judge_revision": expected.judge_revision,
            "judge_prompt_sha256": expected.judge_prompt_sha256,
            "judge_config_sha256": expected.judge_config_sha256,
            "judge_protocol_sha256": expected.judge_protocol_sha256,
            "judge_request_content_sha256": (
                expected.judge_request_content_sha256
            ),
            "judge_response_content_sha256": (
                expected.judge_response_content_sha256
            ),
            "judge_raw_response_sha256": expected.judge_raw_response_sha256,
            "coherent": expected.coherence_score >= 50,
            "coherence_assessed": True,
            "judgment_schema_version": OPEN_JUDGMENT_SCHEMA,
        }
        mismatches = {
            field: (value, row.get(field))
            for field, value in expected_fields.items()
            if row.get(field) != value
        }
        if mismatches:
            raise RuntimeError(
                f"{label} open row {index} judgment content mismatch: {mismatches}"
            )
        unit_key = (
            int(row["layer"]),
            float(row["calibration_magnitude"]),
            str(row["direction_float32_sha256"]),
            str(row["direction_artifact_sha256"]),
            str(row["case_id"]),
            str(row.get("target", "")),
            str(row.get("form", "")),
        )
        condition = str(row.get("condition"))
        triplet = by_unit.setdefault(unit_key, {})
        if condition in triplet:
            raise RuntimeError(f"{label} duplicates open condition for one candidate unit")
        triplet[condition] = row

    baseline_records: list[dict[str, Any]] = []
    for unit_key, triplet in by_unit.items():
        if set(triplet) != {"baseline", "plus", "minus"}:
            raise RuntimeError(f"{label} open unit lacks a complete content triplet")
        baseline = triplet["baseline"]
        baseline_digest = baseline_content_sha256(baseline)
        if any(
            row.get("baseline_content_sha256") != baseline_digest
            for row in triplet.values()
        ):
            raise RuntimeError(
                f"{label} open triplet does not reference its recomputed baseline bytes"
            )
        baseline_records.append(
            {
                "identity": [
                    baseline["model_revision"],
                    baseline["prompt_sha256"],
                    baseline["run_seed"],
                ],
                "baseline_content_sha256": baseline_digest,
                "baseline_payload": baseline_content_payload(baseline),
                "judge_request_content_sha256": baseline[
                    "judge_request_content_sha256"
                ],
                "judge_response_content_sha256": baseline[
                    "judge_response_content_sha256"
                ],
                "judge_raw_response": baseline["judge_raw_response"],
                "unit_key": list(unit_key[4:]),
            }
        )
    return baseline_records


def _verify_calibration_summary(
    repo_root: Path,
    lock: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    method_id: str,
    track: str,
    selected_artifact: Any,
    selected_strength: float | None,
    selected_layer: int,
    position_schedule: str,
    construction_config_sha256: str,
    stage1_lock_sha256: str,
    runner_parent_commit: str,
    sealed_evaluation_required: bool,
) -> tuple[
    set[str],
    str,
    bool,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    from .comparison_calibration import (
        CALIBRATION_SUMMARY_SCHEMA,
        SafetyLimits,
        build_calibration_summary,
        calibration_coverage_sha256,
        calibration_rows_sha256,
        locked_forced_calibration_units,
        locked_open_confirmation_units,
    )

    label = f"calibration summary for {model['model_id']}/{method_id}/{track}"
    if CALIBRATION_SUMMARY_SCHEMA != CALIBRATION_SUMMARY_SCHEMA_VERSION:
        raise RuntimeError("the calibration builder schema differs from stage-2 provenance")
    if validation.get("schema_version") != CALIBRATION_SUMMARY_SCHEMA_VERSION:
        raise RuntimeError(f"{label} has an unsupported schema_version")

    builder = validation.get("builder")
    if not isinstance(builder, Mapping):
        raise TypeError(f"{label} builder must be an object")
    if builder.get("module") != CALIBRATION_BUILDER_PATH or not _valid_digest(
        builder.get("module_sha256")
    ):
        raise RuntimeError(f"{label} builder path/hash is invalid")
    HashEntry(CALIBRATION_BUILDER_PATH, str(builder["module_sha256"])).verify(repo_root)

    calibration_config_sha256 = sha256_json(lock.get("calibration"))
    if validation.get("calibration_config_sha256") != calibration_config_sha256:
        raise RuntimeError(f"{label} calibration config hash differs from the lock")
    expected_identity = {
        "model_id": str(model["model_id"]),
        "model_revision": str(model["revision"]),
        "method_id": method_id,
        "track": track,
        "dataset_sha256": str(lock["dataset"]["sha256"]),
        "protocol_sha256": str(lock["protocol"]["sha256"]),
        "stage1_lock_sha256": stage1_lock_sha256,
        "mode": track,
    }
    identity_mismatches = {
        key: (expected, validation.get(key))
        for key, expected in expected_identity.items()
        if validation.get(key) != expected
    }
    if identity_mismatches:
        raise RuntimeError(f"{label} identity mismatch: {identity_mismatches}")

    calibration_config = lock.get("calibration")
    if not isinstance(calibration_config, Mapping):
        raise TypeError("stage-1 lock calibration configuration must be an object")
    safety_limits = SafetyLimits.from_lock(calibration_config.get("safety_gates"))
    safety_limits_record = safety_limits.to_lock_record()
    if validation.get("safety_limits") != safety_limits_record or validation.get(
        "safety_limits_sha256"
    ) != sha256_json(safety_limits_record):
        raise RuntimeError(f"{label} safety limits differ from the locked gates")
    target = float(
        calibration_config["equal_efficacy_target_mean_self_minus_other_bidirectional_effect"]
    )
    tie_tolerance = float(calibration_config["canonical_layer_tie_tolerance"])
    if validation.get("target") != target:
        raise RuntimeError(f"{label} target differs from the locked calibration target")

    if track == "matched":
        strengths = calibration_config.get("matched_strength_grid")
        layers = [int(model["matched_intervention"]["layer_zero_based"])]
    else:
        canonical_grids = calibration_config.get("canonical_multiplier_grids")
        strengths = (
            canonical_grids.get(method_id)
            if isinstance(canonical_grids, Mapping)
            else None
        )
        layer_overrides = calibration_config.get("canonical_candidate_layers")
        overridden = (
            layer_overrides.get(method_id)
            if isinstance(layer_overrides, Mapping)
            else None
        )
        if overridden is not None:
            layers = list(overridden)
        elif method_id in {"caa", "persona_vector"}:
            layers = list(range(int(model["architecture"]["blocks"])))
        else:
            layers = [int(model["matched_intervention"]["layer_zero_based"])]
    if (
        not isinstance(strengths, list)
        or not strengths
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in strengths
        )
        or len({float(value) for value in strengths}) != len(strengths)
        or not layers
        or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers)
        or len(set(layers)) != len(layers)
    ):
        raise RuntimeError(f"{label} has no valid locked strength/layer grid")
    expected_grid = {
        (int(layer), float(strength)) for layer in layers for strength in strengths
    }
    points = validation.get("points")
    if not isinstance(points, list):
        raise TypeError(f"{label} points must be a list")
    observed_grid = {
        (int(point["layer"]), float(point["strength"]))
        for point in points
        if isinstance(point, Mapping)
    }
    if len(observed_grid) != len(points) or observed_grid != expected_grid:
        raise RuntimeError(
            f"{label} does not evaluate the exact locked strength/layer grid"
        )

    dataset = _json_object(
        repo_root / str(lock["dataset"]["path"]), field="locked comparison dataset"
    )
    expected_forced_units = locked_forced_calibration_units(dataset, lock)
    expected_open_units = locked_open_confirmation_units(dataset, lock)
    staged = calibration_config.get("staged_open_confirmation")
    if not isinstance(staged, Mapping):
        raise TypeError(f"{label} lacks the locked staged-open configuration")
    if (
        staged.get("forced_grid_unit_count") != 142
        or staged.get("forced_grid_row_count_per_point") != 426
        or staged.get("open_confirmation_unit_count") != 32
        or staged.get("open_confirmation_row_count_per_candidate") != 96
        or staged.get("open_results_may_not_select_break_ties_or_trigger_fallback")
        is not True
        or staged.get("stage2_exact_set_equality_required") is not True
    ):
        raise RuntimeError(f"{label} locked staged-open counts/policy are unsupported")
    expected_forced_sha256 = calibration_coverage_sha256(expected_forced_units)
    expected_open_sha256 = calibration_coverage_sha256(expected_open_units)
    if validation.get("forced_coverage_unit_count") != 142 or validation.get(
        "forced_coverage_sha256"
    ) != expected_forced_sha256:
        raise RuntimeError(f"{label} does not bind the exact 142-unit forced set")
    if validation.get("open_coverage_unit_count") != 32 or validation.get(
        "open_coverage_sha256"
    ) != expected_open_sha256:
        raise RuntimeError(f"{label} does not bind the exact 32-unit open set")

    candidates = validation.get("candidate_directions")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"{label} lacks candidate directions")
    candidate_by_layer: dict[int, tuple[str, str]] = {}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise TypeError(f"{label} candidate {index} is not an object")
        layer = candidate.get("layer")
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise RuntimeError(f"{label} candidate {index} has an invalid layer")
        direction_hash = candidate.get("direction_float32_sha256")
        artifact_hash = candidate.get("direction_artifact_sha256")
        if not _valid_digest(direction_hash) or not _valid_digest(artifact_hash):
            raise RuntimeError(f"{label} candidate {index} has invalid direction hashes")
        if layer in candidate_by_layer:
            raise RuntimeError(f"{label} has multiple candidate directions at layer {layer}")
        candidate_by_layer[layer] = (str(direction_hash), str(artifact_hash))
    selected_hashes = (
        selected_artifact.direction_sha256,
        selected_artifact.artifact_sha256,
    )
    if candidate_by_layer.get(selected_layer) != selected_hashes:
        raise RuntimeError(f"{label} selected direction/layer is not a declared candidate")
    if track == "matched" and candidate_by_layer != {selected_layer: selected_hashes}:
        raise RuntimeError(f"{label} matched calibration must use one fixed direction")
    if set(candidate_by_layer) != set(layers):
        raise RuntimeError(f"{label} candidate directions do not cover exact locked layers")

    forced_artifacts = validation.get("forced_result_rows_artifacts")
    open_artifacts = validation.get("open_result_rows_artifacts")
    point_hashes = validation.get("point_rows_sha256s")
    if (
        not isinstance(point_hashes, list)
        or not point_hashes
        or any(not _valid_digest(value) for value in point_hashes)
        or len(set(point_hashes)) != len(point_hashes)
    ):
        raise RuntimeError(f"{label} has invalid or duplicate point row hashes")

    frozen_paths = {CALIBRATION_BUILDER_PATH}
    grid_plan_record = validation.get("forced_grid_plan_artifact")
    if (
        not isinstance(grid_plan_record, Mapping)
        or not grid_plan_record.get("path")
        or not _valid_digest(grid_plan_record.get("sha256"))
    ):
        raise RuntimeError(f"{label} forced grid plan artifact is invalid")
    grid_plan_relative = _repo_relative_path(
        repo_root,
        str(grid_plan_record["path"]),
        field=f"{label} forced grid plan path",
    )
    HashEntry(grid_plan_relative, str(grid_plan_record["sha256"])).verify(repo_root)
    grid_plan = _json_object(
        repo_root / grid_plan_relative, field=f"{label} forced grid plan"
    )
    planned_grid_shard_names = {
        str(point["shard_name"])
        for point in grid_plan.get("points", [])
        if isinstance(point, Mapping) and point.get("shard_name")
    }
    if not planned_grid_shard_names:
        raise RuntimeError(f"{label} forced grid plan has no point shard names")
    frozen_paths.add(grid_plan_relative)

    def read_artifacts(
        records: Any,
        *,
        artifact_label: str,
        required: bool,
        validated_grid_shards: bool = False,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        if not isinstance(records, list) or (required and not records):
            raise RuntimeError(f"{label} lacks {artifact_label} artifacts")
        normalized: list[dict[str, str]] = []
        rows_out: list[dict[str, Any]] = []
        seen_rows: set[bytes] = set()
        interpolation_artifact_seen = False
        for index, record in enumerate(records):
            if (
                not isinstance(record, Mapping)
                or not record.get("path")
                or not _valid_digest(record.get("sha256"))
            ):
                raise RuntimeError(
                    f"{label} {artifact_label} artifact {index} is invalid"
                )
            relative = _repo_relative_path(
                repo_root,
                str(record["path"]),
                field=f"{label} {artifact_label} artifact path",
            )
            if relative in frozen_paths:
                raise RuntimeError(f"{label} contains a duplicate result artifact path")
            HashEntry(relative, str(record["sha256"])).verify(repo_root)
            if (
                validated_grid_shards
                and Path(relative).name in planned_grid_shard_names
            ):
                from .comparison_grid import load_validated_point_rows

                rows, _ = load_validated_point_rows(
                    repo_root / relative,
                    plan=grid_plan,
                    lock=lock,
                    dataset=dataset,
                    repo_root=repo_root,
                )
            else:
                rows = _read_jsonl_objects(
                    repo_root / relative, label=f"{label} {artifact_label} artifact"
                )
                if validated_grid_shards:
                    interpolation = validation.get("interpolation_recheck")
                    if (
                        interpolation_artifact_seen
                        or not isinstance(interpolation, Mapping)
                        or calibration_rows_sha256(rows) != interpolation.get("rows_sha256")
                    ):
                        raise RuntimeError(
                            f"{label} non-grid forced artifact is not the single locked "
                            "interpolation recheck"
                        )
                    interpolation_artifact_seen = True
            _verify_calibration_row_identities(
                rows,
                model=model,
                method_id=method_id,
                track=track,
                dataset_sha256=str(lock["dataset"]["sha256"]),
                protocol_sha256=str(lock["protocol"]["sha256"]),
                stage1_lock_sha256=stage1_lock_sha256,
                runner_parent_commit=runner_parent_commit,
                position_schedule=position_schedule,
                construction_config_sha256=construction_config_sha256,
                candidate_by_layer=candidate_by_layer,
                label=label,
            )
            for row in rows:
                encoded = canonical_json_bytes(row)
                if encoded in seen_rows:
                    raise RuntimeError(
                        f"{label} duplicates a row across {artifact_label} artifacts"
                    )
                seen_rows.add(encoded)
                rows_out.append(row)
            normalized.append({"path": relative, "sha256": str(record["sha256"])})
            frozen_paths.add(relative)
        if (
            validated_grid_shards
            and validation.get("interpolation_recheck") is not None
            and not interpolation_artifact_seen
        ):
            raise RuntimeError(f"{label} omits the locked interpolation recheck artifact")
        return normalized, rows_out

    forced_records, forced_rows = read_artifacts(
        forced_artifacts,
        artifact_label="forced result",
        required=True,
        validated_grid_shards=True,
    )
    open_records, open_rows = read_artifacts(
        open_artifacts, artifact_label="open result", required=False
    )
    if any(row.get("family") == "open_ended" for row in forced_rows):
        raise RuntimeError(f"{label} forced artifacts contain open-ended rows")
    if any(row.get("family") != "open_ended" for row in open_rows):
        raise RuntimeError(f"{label} open artifacts contain non-open rows")
    open_protocol = _json_object(
        repo_root
        / str(lock["evaluation"]["open_behavior_judge"]["protocol_path"]),
        field="locked open-behavior judge protocol",
    )
    baseline_records = _verify_open_result_content(
        open_rows, open_protocol, label=label
    ) if open_rows else []

    def group_by_candidate(
        rows: Sequence[dict[str, Any]], *, rows_label: str
    ) -> dict[tuple[int, float, str, str], list[dict[str, Any]]]:
        grouped: dict[tuple[int, float, str, str], list[dict[str, Any]]] = {}
        for index, row in enumerate(rows):
            magnitude = row.get("calibration_magnitude")
            layer = row.get("layer")
            if (
                isinstance(magnitude, bool)
                or not isinstance(magnitude, (int, float))
                or not math.isfinite(float(magnitude))
                or float(magnitude) <= 0
                or isinstance(layer, bool)
                or not isinstance(layer, int)
            ):
                raise RuntimeError(
                    f"{label} {rows_label} row {index} has invalid point identity"
                )
            key = (
                layer,
                float(magnitude),
                str(row["direction_float32_sha256"]),
                str(row["direction_artifact_sha256"]),
            )
            grouped.setdefault(key, []).append(row)
        return grouped

    forced_groups = group_by_candidate(forced_rows, rows_label="forced")
    forced_by_hash: dict[str, list[dict[str, Any]]] = {}
    for rows in forced_groups.values():
        row_hash = calibration_rows_sha256(rows)
        if row_hash in forced_by_hash:
            raise RuntimeError(f"{label} has duplicate canonical forced points")
        forced_by_hash[row_hash] = rows

    interpolation = validation.get("interpolation_recheck")
    interpolation_hash = None
    if interpolation is not None:
        if not isinstance(interpolation, Mapping) or not _valid_digest(
            interpolation.get("rows_sha256")
        ):
            raise RuntimeError(f"{label} has an invalid interpolation recheck")
        interpolation_hash = str(interpolation["rows_sha256"])
    required_row_hashes = set(map(str, point_hashes))
    if interpolation_hash is not None:
        required_row_hashes.add(interpolation_hash)
    if set(forced_by_hash) != required_row_hashes:
        raise RuntimeError(
            f"{label} forced artifacts do not exactly match point/recheck row hashes"
        )

    confirmation_records = validation.get("open_confirmations")
    if not isinstance(confirmation_records, list):
        raise TypeError(f"{label} open_confirmations must be a list")
    confirmation_hashes: list[str] = []
    for index, confirmation in enumerate(confirmation_records):
        if not isinstance(confirmation, Mapping) or not _valid_digest(
            confirmation.get("rows_sha256")
        ):
            raise RuntimeError(f"{label} open confirmation {index} is invalid")
        confirmation_hashes.append(str(confirmation["rows_sha256"]))
    if len(set(confirmation_hashes)) != len(confirmation_hashes):
        raise RuntimeError(f"{label} duplicates an open confirmation hash")
    open_by_hash: dict[str, list[dict[str, Any]]] = {}
    for rows in group_by_candidate(open_rows, rows_label="open").values():
        row_hash = calibration_rows_sha256(rows)
        if row_hash in open_by_hash:
            raise RuntimeError(f"{label} has duplicate canonical open confirmations")
        open_by_hash[row_hash] = rows
    if set(open_by_hash) != set(confirmation_hashes):
        raise RuntimeError(
            f"{label} open artifacts do not exactly match confirmation row hashes"
        )

    rebuilt = build_calibration_summary(
        [forced_by_hash[str(point_hash)] for point_hash in point_hashes],
        expected_forced_units=expected_forced_units,
        expected_open_units=expected_open_units,
        safety_limits=safety_limits,
        mode=track,
        forced_result_rows_artifacts=forced_records,
        open_result_rows_artifacts=open_records,
        forced_grid_plan_artifact={
            "path": grid_plan_relative,
            "sha256": str(grid_plan_record["sha256"]),
        },
        calibration_config_sha256=calibration_config_sha256,
        builder_module_sha256=str(builder["module_sha256"]),
        interpolation_recheck_rows=(
            None if interpolation_hash is None else forced_by_hash[interpolation_hash]
        ),
        open_confirmation_rows=[open_by_hash[value] for value in confirmation_hashes],
        allow_pending_open=False,
        fixed_strength=float(
            lock["comparison_tracks"]["matched_primary"]["fixed_strength"]
        ),
        target=target,
        tie_tolerance=tie_tolerance,
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(dict(validation)):
        raise RuntimeError(f"{label} is not the canonical result of the locked builder")

    decision = validation.get("decision")
    if not isinstance(decision, Mapping) or not isinstance(decision.get("status"), str):
        raise TypeError(f"{label} calibration decision must be an object with status")
    pre_open = validation.get("pre_open_decision")
    if not isinstance(pre_open, Mapping) or validation.get(
        "pre_open_decision_sha256"
    ) != sha256_json(pre_open):
        raise RuntimeError(f"{label} pre-open decision/hash is invalid")
    pre_open_status = str(pre_open.get("status"))
    status = str(decision["status"])
    if pre_open_status == "interpolation_requires_one_recheck" or status == (
        "open_confirmation_pending"
    ):
        raise RuntimeError(f"{label} has a pending staged calibration decision")
    decision_strength = decision.get("selected_strength")
    decision_open_passed = decision.get("open_confirmation_passed")
    expected_sealed = decision_strength is not None and decision_open_passed is True
    if sealed_evaluation_required != expected_sealed:
        raise RuntimeError(
            f"{label} sealed_evaluation_required disagrees with staged safety"
        )
    if decision_strength is None:
        if selected_strength is not None:
            raise RuntimeError(f"{label} no-safe decision must not select a strength")
    elif selected_strength != float(decision_strength):
        raise RuntimeError(f"{label} selected strength differs from its decision")
    if decision_strength is not None and decision.get("selected_layer") != selected_layer:
        raise RuntimeError(f"{label} selected layer differs from its decision")
    winner_eligible = (
        method_id in MAIN_METHOD_IDS
        and track == "matched"
        and pre_open_status.startswith("target_reached")
        and decision_open_passed is True
    )
    confirmations_by_candidate = {
        (
            int(item["layer"]),
            float(item["strength"]),
            str(item["direction_float32_sha256"]),
            str(item["direction_artifact_sha256"]),
        ): copy.deepcopy(dict(item))
        for item in validation["open_confirmations"]
    }
    evaluated_points = [
        copy.deepcopy(dict(point))
        for point in validation["points"]
        if isinstance(point, Mapping)
    ]
    if isinstance(interpolation, Mapping) and isinstance(
        interpolation.get("point"), Mapping
    ):
        evaluated_points.append(copy.deepcopy(dict(interpolation["point"])))
    for point in evaluated_points:
        direction_hashes = candidate_by_layer[int(point["layer"])]
        confirmation = confirmations_by_candidate.get(
            (
                int(point["layer"]),
                float(point["strength"]),
                direction_hashes[0],
                direction_hashes[1],
            )
        )
        point["open_confirmation"] = confirmation
    return frozen_paths, status, winner_eligible, evaluated_points, baseline_records


def _validate_discovery_vector_audit(
    repo_root: Path,
    lock: Mapping[str, Any],
    artifact: Any,
    diagnostics: Any,
    *,
    method_id: str,
    label: str,
) -> None:
    if not isinstance(diagnostics, Mapping):
        raise TypeError(f"{label} {method_id} diagnostics are not an object")
    dataset = _json_object(
        repo_root / str(lock["dataset"]["path"]),
        field=f"{label} locked dataset",
    )
    discovery_cases = dataset.get("sp_splits", {}).get("discovery")
    if not isinstance(discovery_cases, list) or not discovery_cases:
        raise RuntimeError(f"{label} locked dataset lacks discovery cases")
    expected_ids = [str(case["id"]) for case in discovery_cases]
    if diagnostics.get("discovery_case_ids_sha256") != sha256_json(expected_ids):
        raise RuntimeError(f"{label} {method_id} discovery-case hash differs from the lock")
    records = diagnostics.get("per_case")
    if not isinstance(records, list) or [
        record.get("case_id") if isinstance(record, Mapping) else None
        for record in records
    ] != expected_ids:
        raise RuntimeError(
            f"{label} {method_id} per-case evidence does not exactly cover discovery"
        )
    model_id = artifact.metadata.get("model_id")
    model = next(
        (
            item
            for item in lock.get("models", [])
            if isinstance(item, Mapping) and item.get("model_id") == model_id
        ),
        None,
    )
    if not isinstance(model, Mapping):
        raise TypeError(f"{label} {method_id} evidence has an unknown model")
    d_model = int(model["architecture"]["residual_width"])
    vector_fields = (
        ("self_gradient", "matched_other_gradient")
        if method_id == "gradient"
        else ("preserve_activation", "comply_activation", "semantic_difference")
    )
    for case, record in zip(discovery_cases, records, strict=True):
        if not isinstance(record, Mapping):  # pragma: no cover - guarded above
            raise TypeError(f"{label} {method_id} per-case record is not an object")
        if method_id == "caa":
            preserve_label = "A" if bool(case["preserve_first"]) else "B"
            comply_label = "B" if bool(case["preserve_first"]) else "A"
            if record.get("preserve_label") != preserve_label or record.get(
                "comply_label"
            ) != comply_label:
                raise RuntimeError(f"{label} CAA evidence has incorrect semantic labels")
        for field in vector_fields:
            item = record.get(field)
            if (
                not isinstance(item, Mapping)
                or set(item) != {"shape", "float32_sha256", "l2_norm"}
                or item.get("shape") != [d_model]
                or not _valid_digest(item.get("float32_sha256"))
                or isinstance(item.get("l2_norm"), bool)
                or not isinstance(item.get("l2_norm"), (int, float))
                or not math.isfinite(float(item["l2_norm"]))
                or float(item["l2_norm"]) < 0
            ):
                raise RuntimeError(
                    f"{label} {method_id} per-case {field} audit is invalid"
                )


def _validate_method_construction_evidence(
    repo_root: Path,
    lock: Mapping[str, Any],
    artifact: Any,
    method_id: str,
    selected_layer: int,
    evidence_paths: Mapping[str, str],
    *,
    label: str,
) -> None:
    if method_id == "gradient_uncorrected":
        method_id = "gradient"
    if method_id in {"gradient", "caa", "bipo"}:
        role = {
            "gradient": "gradient_construction_diagnostics",
            "caa": "caa_construction_diagnostics",
            "bipo": "bipo_training_audit",
        }[method_id]
        payload = _json_object(repo_root / evidence_paths[role], field=f"{label} {role}")
        if method_id == "gradient":
            if payload != artifact.metadata.get("diagnostics"):
                raise RuntimeError(f"{label} gradient diagnostics differ from the artifact")
            _validate_discovery_vector_audit(
                repo_root,
                lock,
                artifact,
                payload,
                method_id="gradient",
                label=label,
            )
        elif method_id == "caa":
            layer_payload = payload.get(str(selected_layer))
            if layer_payload != artifact.metadata.get("diagnostics"):
                raise RuntimeError(f"{label} CAA layer diagnostics differ from the artifact")
            _validate_discovery_vector_audit(
                repo_root,
                lock,
                artifact,
                layer_payload,
                method_id="caa",
                label=label,
            )
        else:
            training_audit = payload.get("training_audit")
            if not isinstance(training_audit, Mapping):
                raise RuntimeError(f"{label} BiPO evidence lacks its training audit")
            if sha256_json(training_audit) != artifact.metadata.get(
                "training_audit_sha256"
            ):
                raise RuntimeError(f"{label} BiPO training audit hash mismatch")
            if not isinstance(training_audit.get("optimizer_state"), Mapping):
                raise RuntimeError(f"{label} BiPO evidence lacks optimizer state")
            bipo_lock = lock.get("methods", {}).get("bipo")
            if not isinstance(bipo_lock, Mapping):
                raise TypeError("stage-1 lock lacks the BiPO method configuration")
            selected_epoch = bipo_lock.get("selected_checkpoint_epoch")
            if (
                isinstance(selected_epoch, bool)
                or not isinstance(selected_epoch, int)
                or training_audit.get("selected_checkpoint_epoch") != selected_epoch
                or artifact.metadata.get("selected_checkpoint_epoch") != selected_epoch
                or training_audit.get("checkpoint_selection")
                != "fixed_by_stage1_lock_before_direction_fitting"
                or artifact.metadata.get("checkpoint_selection")
                != "fixed_by_stage1_lock_before_direction_fitting"
            ):
                raise RuntimeError(
                    f"{label} BiPO checkpoint is not the a-priori stage-1 selection"
                )
            roles = training_audit.get("checkpoint_roles")
            expected_roles = {
                str(epoch): (
                    "a_priori_selected"
                    if epoch == selected_epoch
                    else "diagnostic_only"
                )
                for epoch in bipo_lock.get("checkpoint_epochs", [])
            }
            if roles != expected_roles or artifact.metadata.get(
                "checkpoint_roles"
            ) != expected_roles:
                raise RuntimeError(f"{label} BiPO checkpoint roles differ from the lock")
            forbidden_validation_fields = {
                "validation_preference_loss_by_epoch",
                "selected_by_validation",
            }
            if forbidden_validation_fields.intersection(training_audit) or (
                forbidden_validation_fields.intersection(artifact.metadata)
            ):
                raise RuntimeError(
                    f"{label} BiPO evidence contains forbidden validation checkpoint selection"
                )
        return

    if method_id != "persona_vector":
        raise RuntimeError(f"{label} has no evidence verifier for {method_id}")
    diagnostics = _json_object(
        repo_root / evidence_paths["persona_construction_diagnostics"],
        field=f"{label} persona diagnostics",
    )
    if diagnostics != artifact.metadata.get("diagnostics"):
        raise RuntimeError(f"{label} persona diagnostics differ from the artifact")
    from .comparison_persona import (
        load_persona_protocol,
        persona_generation_provenance,
        read_rollouts,
        validate_rollouts,
    )

    protocol_path = repo_root / str(
        lock["methods"]["persona_vector"]["canonical_protocol_path"]
    )
    protocol = load_persona_protocol(protocol_path)
    rollouts = read_rollouts(repo_root / evidence_paths["persona_scored_rollouts"])
    expected_provenance = persona_generation_provenance(
        protocol,
        model_id=str(artifact.metadata["model_id"]),
        model_revision=str(artifact.metadata["model_revision"]),
        model_config_sha256=str(artifact.metadata["model_config_sha256"]),
        stage1_lock_sha256=str(artifact.metadata["stage1_lock_sha256"]),
        runner_commit=str(artifact.metadata["runner_commit"]),
        persona_protocol_sha256=str(
            lock["methods"]["persona_vector"]["canonical_protocol_sha256"]
        ),
    )
    validation = validate_rollouts(
        rollouts,
        protocol,
        rollouts_per_instruction_question=int(
            lock["methods"]["persona_vector"]["canonical_grid"][
                "rollouts_per_instruction_question_per_polarity"
            ]
        ),
        require_scores=True,
        expected_generation_provenance=expected_provenance,
    )
    if diagnostics.get("canonical_grid") != validation:
        raise RuntimeError(f"{label} persona rollout audit differs from diagnostics")
    if int(diagnostics.get("n_retained_pairs", 0)) < int(
        lock["methods"]["persona_vector"]["minimum_retained_pairs"]
    ):
        raise RuntimeError(f"{label} persona retained-pair count is below the lock")


def _verify_main_artifact(
    repo_root: Path,
    lock: Mapping[str, Any],
    model_records: Mapping[str, Mapping[str, Any]],
    item: Mapping[str, Any],
    *,
    index: int,
    stage1_lock_sha256: str,
    runner_parent_commit: str,
) -> tuple[tuple[str, str, str], dict[str, Any], set[str]]:
    label = f"stage-2 artifact {index}"
    _require_non_null_fields(
        item,
        (
            "model_id",
            "method_id",
            "track",
            "direction_path",
            "direction_file_sha256",
            "direction_float32_sha256",
            "direction_artifact_sha256",
            "intervention_geometry",
            "construction_config_path",
            "construction_config_sha256",
            "selected_layer",
            "position_schedule",
            "validation_summary_path",
            "validation_summary_sha256",
            "sealed_evaluation_required",
            "dataset_sha256",
            "protocol_sha256",
        ),
        label=label,
    )
    model_id = str(item["model_id"])
    method_id = str(item["method_id"])
    track = str(item["track"])
    coverage = (model_id, method_id, track)
    if model_id not in model_records:
        raise RuntimeError(f"{label} uses unknown model ID {model_id!r}")
    if coverage not in _expected_stage2_coverage(lock):
        raise RuntimeError(f"{label} has unexpected model/method/track {coverage}")
    if item["dataset_sha256"] != lock["dataset"]["sha256"]:
        raise RuntimeError(f"{label} dataset hash differs from stage 1")
    if item["protocol_sha256"] != lock["protocol"]["sha256"]:
        raise RuntimeError(f"{label} protocol hash differs from stage 1")
    if not isinstance(item["sealed_evaluation_required"], bool):
        raise TypeError(f"{label} sealed_evaluation_required must be boolean")

    if "selected_strength" not in item:
        raise RuntimeError(f"{label} lacks selected_strength")
    strength = item["selected_strength"]
    if strength is not None and (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or float(strength) <= 0
    ):
        raise RuntimeError(f"{label} has invalid selected_strength")
    model = model_records[model_id]
    selected_layer = _validate_selected_layer(
        model, method_id, track, item["selected_layer"]
    )
    expected_position = _expected_position_schedule(method_id, track)
    if item["position_schedule"] != expected_position:
        raise RuntimeError(
            f"{label} position_schedule must be {expected_position!r}"
        )
    artifact, direction_relative = _verify_direction_file(
        repo_root,
        item,
        model=model,
        method_id=method_id,
        track=track,
        selected_layer=selected_layer,
        stage1_lock_sha256=stage1_lock_sha256,
        runner_parent_commit=runner_parent_commit,
        label=label,
    )

    for key in ("construction_config_sha256", "validation_summary_sha256"):
        if not _valid_digest(item.get(key)):
            raise RuntimeError(f"{label} has invalid {key}")
    construction_relative = _repo_relative_path(
        repo_root,
        str(item["construction_config_path"]),
        field=f"{label} construction_config_path",
    )
    validation_relative = _repo_relative_path(
        repo_root,
        str(item["validation_summary_path"]),
        field=f"{label} validation_summary_path",
    )
    HashEntry(construction_relative, str(item["construction_config_sha256"])).verify(
        repo_root
    )
    HashEntry(validation_relative, str(item["validation_summary_sha256"])).verify(
        repo_root
    )
    construction = _json_object(
        repo_root / construction_relative, field=f"{label} construction config"
    )
    validation = _json_object(
        repo_root / validation_relative, field=f"{label} validation summary"
    )
    _verify_json_identity(
        construction,
        schema_version=MAIN_CONSTRUCTION_SCHEMA_VERSION,
        model_id=model_id,
        method_id=method_id,
        track=track,
        direction_artifact_sha256=artifact.artifact_sha256,
        label=f"{label} construction config",
    )
    locked_configuration = locked_method_construction_configuration(
        lock, method_id, track
    )
    construction_expected = {
        "model_revision": str(model["revision"]),
        "model_config_sha256": str(model["config_sha256"]),
        "selected_layer": selected_layer,
        "position_schedule": expected_position,
        "intervention_geometry": artifact.intervention_geometry,
        "direction_float32_sha256": artifact.direction_sha256,
        "dataset_sha256": str(lock["dataset"]["sha256"]),
        "protocol_sha256": str(lock["protocol"]["sha256"]),
        "stage1_lock_sha256": stage1_lock_sha256,
        "runner_commit": runner_parent_commit,
        "locked_configuration": locked_configuration,
        "locked_configuration_sha256": sha256_json(locked_configuration),
    }
    construction_mismatches = {
        key: (expected, construction.get(key))
        for key, expected in construction_expected.items()
        if construction.get(key) != expected
    }
    if construction_mismatches:
        raise RuntimeError(
            f"{label} construction configuration mismatch: {construction_mismatches}"
        )
    evidence = construction.get("evidence_artifacts")
    if not isinstance(evidence, list) or not evidence:
        raise RuntimeError(f"{label} construction lacks method evidence artifacts")
    if construction.get("evidence_artifacts_sha256") != sha256_json(evidence):
        raise RuntimeError(f"{label} construction evidence-list hash mismatch")
    expected_evidence_roles = {
        "gradient": {"gradient_construction_diagnostics"},
        "gradient_uncorrected": {"gradient_construction_diagnostics"},
        "caa": {"caa_construction_diagnostics"},
        "bipo": {"bipo_training_audit"},
        "persona_vector": {
            "persona_construction_diagnostics",
            "persona_scored_rollouts",
        },
    }[method_id]
    evidence_paths: dict[str, str] = {}
    for evidence_index, evidence_record in enumerate(evidence):
        if (
            not isinstance(evidence_record, Mapping)
            or not isinstance(evidence_record.get("role"), str)
            or not evidence_record.get("path")
            or not _valid_digest(evidence_record.get("sha256"))
        ):
            raise RuntimeError(f"{label} evidence record {evidence_index} is invalid")
        role = str(evidence_record["role"])
        relative = _repo_relative_path(
            repo_root,
            str(evidence_record["path"]),
            field=f"{label} evidence path",
        )
        if role in evidence_paths or relative in evidence_paths.values():
            raise RuntimeError(f"{label} duplicates a construction evidence role/path")
        HashEntry(relative, str(evidence_record["sha256"])).verify(repo_root)
        evidence_paths[role] = relative
    if set(evidence_paths) != expected_evidence_roles:
        raise RuntimeError(f"{label} construction evidence roles are incomplete")
    _validate_method_construction_evidence(
        repo_root,
        lock,
        artifact,
        method_id,
        selected_layer,
        evidence_paths,
        label=label,
    )
    (
        result_paths,
        calibration_status,
        winner_eligible,
        validation_points,
        open_baseline_records,
    ) = _verify_calibration_summary(
        repo_root,
        lock,
        validation,
        model=model,
        method_id=method_id,
        track=track,
        selected_artifact=artifact,
        selected_strength=None if strength is None else float(strength),
        selected_layer=selected_layer,
        position_schedule=expected_position,
        construction_config_sha256=str(item["construction_config_sha256"]),
        stage1_lock_sha256=stage1_lock_sha256,
        runner_parent_commit=runner_parent_commit,
        sealed_evaluation_required=bool(item["sealed_evaluation_required"]),
    )
    normalized = {
        "model_id": model_id,
        "model_revision": str(model["revision"]),
        "model_config_sha256": str(model["config_sha256"]),
        "method_id": method_id,
        "track": track,
        "direction_path": direction_relative,
        "direction_file_sha256": str(item["direction_file_sha256"]),
        "direction_float32_sha256": artifact.direction_sha256,
        "direction_artifact_sha256": artifact.artifact_sha256,
        "selected_strength": None if strength is None else float(strength),
        "selected_layer": selected_layer,
        "position_schedule": expected_position,
        "construction_config_sha256": str(item["construction_config_sha256"]),
        "validation_summary_sha256": str(item["validation_summary_sha256"]),
        "calibration_status": calibration_status,
        "sealed_evaluation_required": bool(item["sealed_evaluation_required"]),
        "winner_eligible": winner_eligible,
        "validation_points": validation_points,
        "_open_baseline_records": open_baseline_records,
        "matched_fixed_descriptive": copy.deepcopy(
            validation.get("matched_fixed_descriptive")
        ),
    }
    return coverage, normalized, {
        direction_relative,
        construction_relative,
        validation_relative,
        *evidence_paths.values(),
        *result_paths,
    }


def _expand_main_approved_setups(
    lock: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    fixed_strength = float(
        lock["comparison_tracks"]["matched_primary"]["fixed_strength"]
    )
    if not math.isfinite(fixed_strength) or fixed_strength <= 0:
        raise RuntimeError("locked matched fixed strength must be finite and positive")
    approved: list[dict[str, Any]] = []
    for record in records:
        validation_points = record["validation_points"]
        if not isinstance(validation_points, list) or not validation_points:
            raise RuntimeError("verified calibration record lacks validation points")
        points_by_key = {
            (int(point["layer"]), float(point["strength"])): point
            for point in validation_points
        }
        if len(points_by_key) != len(validation_points):
            raise RuntimeError("verified calibration record duplicates a validation point")
        strength_roles: dict[float, set[str]] = {}
        if record["sealed_evaluation_required"]:
            selected = float(record["selected_strength"])
            selected_point = points_by_key.get((int(record["selected_layer"]), selected))
            if selected <= 0 or selected_point is None:
                raise RuntimeError(
                    "an evaluable calibrated setup must select a tested positive strength"
                )
            selected_open = selected_point.get("open_confirmation")
            if not bool(selected_point.get("safe")) or not (
                isinstance(selected_open, Mapping) and selected_open.get("safe") is True
            ):
                raise RuntimeError(
                    "an evaluable calibrated setup must pass forced and open safety"
                )
            strength_roles.setdefault(selected, set()).add("calibrated")
        if record["track"] == "matched":
            fixed_point = points_by_key.get(
                (int(record["selected_layer"]), fixed_strength)
            )
            if fixed_point is None:
                raise RuntimeError(
                    f"{record['model_id']}/{record['method_id']} did not evaluate the "
                    "locked fixed descriptive strength"
                )
            fixed_open = fixed_point.get("open_confirmation")
            if bool(fixed_point.get("safe")) and isinstance(
                fixed_open, Mapping
            ) and fixed_open.get("safe") is True:
                if "fixed_descriptive" not in set(
                    map(str, fixed_open.get("roles", []))
                ):
                    raise RuntimeError(
                        "fixed descriptive approval lacks its locked confirmation role"
                    )
                strength_roles.setdefault(fixed_strength, set()).add(
                    "fixed_descriptive"
                )
        for strength, roles in sorted(strength_roles.items()):
            setup = {
                key: value
                for key, value in record.items()
                if key != "validation_points"
            }
            validation_point = points_by_key[(int(record["selected_layer"]), strength)]
            forced_safety = validation_point["safety"]
            open_confirmation = validation_point.get("open_confirmation")
            if not isinstance(open_confirmation, Mapping):
                raise TypeError("approved strength lacks open confirmation evidence")
            open_safety = open_confirmation["safety"]
            setup["selected_strength"] = strength
            setup["strength_roles"] = sorted(roles)
            setup["sealed_evaluation_required"] = True
            setup["calibration_summary_verified"] = True
            setup["validation_coverage_adequate"] = True
            setup["validation_safe"] = bool(
                validation_point["safe"] and open_confirmation["safe"]
            )
            setup["validation_sign_safe"] = {
                sign: bool(
                    forced_safety["signs"][sign]["pass"]
                    and open_safety["signs"][sign]["pass"]
                )
                for sign in ("plus", "minus")
            }
            setup["validation_safety"] = copy.deepcopy(forced_safety)
            setup["open_confirmation_safety"] = copy.deepcopy(open_safety)
            setup["open_confirmation_rows_sha256"] = str(
                open_confirmation["rows_sha256"]
            )
            if record["method_id"] == "gradient" and record["track"] == "matched":
                setup["canonical_alias"] = True
                setup["canonical_alias_track"] = "canonical"
            setup["winner_eligible"] = bool(
                record["winner_eligible"] and "calibrated" in roles
            )
            approved.append(setup)
    return approved


def _verify_random_controls(
    repo_root: Path,
    lock: Mapping[str, Any],
    model_records: Mapping[str, Mapping[str, Any]],
    records: Any,
    *,
    stage1_lock_sha256: str,
    runner_parent_commit: str,
    main_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    if not isinstance(records, list) or not records:
        raise RuntimeError("stage 2 lacks random_direction_controls")
    seeds = lock.get("random_controls", {}).get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise RuntimeError("stage-1 lock lacks random-control seeds")
    expected = {
        (model_id, int(seed))
        for model_id in model_records
        for seed in seeds
    }
    if len(expected) != len(model_records) * len(seeds):
        raise RuntimeError("locked random-control seeds must be unique integers")
    observed: set[tuple[str, int]] = set()
    normalized: list[dict[str, Any]] = []
    approved_setups: list[dict[str, Any]] = []
    paths: set[str] = set()
    matched_sources = {
        (str(record["model_id"]), str(record["method_id"])): record
        for record in main_records
        if record["method_id"] in MAIN_METHOD_IDS and record["track"] == "matched"
    }
    expected_source_keys = {
        (model_id, method_id)
        for model_id in model_records
        for method_id in MAIN_METHOD_IDS
    }
    if set(matched_sources) != expected_source_keys:
        raise RuntimeError("random controls lack complete matched contender calibration sources")
    source_approvals = {
        (
            str(setup["model_id"]),
            str(setup["method_id"]),
            float(setup["selected_strength"]),
            str(setup["validation_summary_sha256"]),
        ): setup
        for setup in _expand_main_approved_setups(lock, main_records)
        if setup["method_id"] in MAIN_METHOD_IDS and setup["track"] == "matched"
    }
    for index, item in enumerate(records):
        label = f"random control {index}"
        if not isinstance(item, Mapping):
            raise TypeError(f"{label} is not an object")
        _require_non_null_fields(
            item,
            (
                "model_id",
                "seed",
                "method_id",
                "track",
                "direction_path",
                "direction_file_sha256",
                "direction_float32_sha256",
                "direction_artifact_sha256",
                "intervention_geometry",
                "selected_layer",
                "position_schedule",
                "construction_config_path",
                "construction_config_sha256",
                "approved_strengths",
                "dataset_sha256",
                "protocol_sha256",
            ),
            label=label,
        )
        model_id = str(item["model_id"])
        seed = item["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"{label} seed must be an integer")
        key = (model_id, seed)
        if key not in expected or key in observed:
            raise RuntimeError(f"{label} has unexpected or duplicate model/seed {key}")
        observed.add(key)
        seed_index = seeds.index(seed) + 1
        expected_method = f"random_control_{seed_index:02d}"
        if item["method_id"] != expected_method or item["track"] != "matched":
            raise RuntimeError(f"{label} method_id/track differs from the lock")
        if item["position_schedule"] != "final_prompt_token":
            raise RuntimeError(f"{label} must use the matched final prompt position")
        model = model_records[model_id]
        selected_layer = _validate_selected_layer(
            model, "gradient", "matched", item["selected_layer"]
        )
        # Random artifacts have unique method IDs, while their expected geometry is
        # the same matched-final-prompt geometry used by every candidate control.
        if item["intervention_geometry"] != "matched_final_prompt":
            raise RuntimeError(f"{label} has an invalid intervention_geometry")
        random_item = dict(item)
        random_item["method_id"] = expected_method
        artifact, relative = _verify_direction_file(
            repo_root,
            random_item,
            model=model,
            method_id=expected_method,
            track="matched",
            selected_layer=selected_layer,
            stage1_lock_sha256=stage1_lock_sha256,
            runner_parent_commit=runner_parent_commit,
            label=label,
        )
        if not math.isclose(
            float(artifact.direction.norm().item()), 1.0, rel_tol=0, abs_tol=1e-6
        ):
            raise RuntimeError(f"{label} direction is not unit normalized")
        import torch

        from .comparison_controls import locked_random_directions

        expected_direction = locked_random_directions(
            torch,
            int(model["architecture"]["residual_width"]),
            seeds=(seed,),
        )[0]
        if not torch.equal(artifact.direction, expected_direction):
            raise RuntimeError(f"{label} bytes do not match its locked Gaussian seed")
        if artifact.metadata.get("seed") != seed or artifact.metadata.get("orientation") != "none":
            raise RuntimeError(f"{label} seed/orientation metadata mismatch")

        construction_relative = _repo_relative_path(
            repo_root,
            str(item["construction_config_path"]),
            field=f"{label} construction_config_path",
        )
        if not _valid_digest(item["construction_config_sha256"]):
            raise RuntimeError(f"{label} has invalid construction_config_sha256")
        HashEntry(
            construction_relative, str(item["construction_config_sha256"])
        ).verify(repo_root)
        construction = _json_object(
            repo_root / construction_relative, field=f"{label} construction config"
        )
        expected_construction = {
            "schema_version": RANDOM_CONSTRUCTION_SCHEMA_VERSION,
            "model_id": model_id,
            "model_revision": str(model["revision"]),
            "model_config_sha256": str(model["config_sha256"]),
            "method_id": expected_method,
            "track": "matched",
            "seed": seed,
            "generator_algorithm": RANDOM_GENERATOR_ALGORITHM,
            "distribution": str(lock["random_controls"]["distribution"]),
            "d_model": int(model["architecture"]["residual_width"]),
            "selected_layer": selected_layer,
            "position_schedule": "final_prompt_token",
            "intervention_geometry": "matched_final_prompt",
            "direction_float32_sha256": artifact.direction_sha256,
            "direction_artifact_sha256": artifact.artifact_sha256,
            "stage1_lock_sha256": stage1_lock_sha256,
            "runner_commit": runner_parent_commit,
        }
        construction_mismatches = {
            field: (expected_value, construction.get(field))
            for field, expected_value in expected_construction.items()
            if construction.get(field) != expected_value
        }
        if construction_mismatches:
            raise RuntimeError(
                f"{label} random construction config mismatch: {construction_mismatches}"
            )

        approved_strengths = item["approved_strengths"]
        if not isinstance(approved_strengths, list):
            raise TypeError(f"{label} approved_strengths must be a list")
        observed_strengths: set[tuple[str, float, str]] = set()
        for strength_index, strength_record in enumerate(approved_strengths):
            if not isinstance(strength_record, Mapping):
                raise TypeError(
                    f"{label} approved strength {strength_index} is not an object"
                )
            source_method_id = str(strength_record.get("source_method_id"))
            strength = strength_record.get("strength")
            source_summary = strength_record.get(
                "source_calibration_summary_sha256"
            )
            if (
                source_method_id not in MAIN_METHOD_IDS
                or isinstance(strength, bool)
                or not isinstance(strength, (int, float))
                or not math.isfinite(float(strength))
                or float(strength) <= 0
                or not _valid_digest(source_summary)
            ):
                raise RuntimeError(f"{label} approved strength {strength_index} is invalid")
            strength_key = (source_method_id, float(strength), str(source_summary))
            if strength_key in observed_strengths:
                raise RuntimeError(f"{label} contains a duplicate approved strength")
            observed_strengths.add(strength_key)

        expected_strengths = {
            (source_method_id, strength, summary)
            for (
                source_model_id,
                source_method_id,
                strength,
                summary,
            ) in source_approvals
            if source_model_id == model_id
        }
        if observed_strengths != expected_strengths:
            raise RuntimeError(
                f"{label} approved_strengths do not exactly cover safe open-confirmed "
                "matched contender comparisons"
            )

        paths.update({relative, construction_relative})
        normalized_strengths = [
            {
                "source_method_id": source_method_id,
                "strength": strength,
                "source_calibration_summary_sha256": source_summary,
            }
            for source_method_id, strength, source_summary in sorted(observed_strengths)
        ]
        normalized.append(
            {
                "model_id": model_id,
                "seed": seed,
                "method_id": expected_method,
                "track": "matched",
                "direction_path": relative,
                "direction_file_sha256": str(item["direction_file_sha256"]),
                "direction_float32_sha256": artifact.direction_sha256,
                "direction_artifact_sha256": artifact.artifact_sha256,
                "selected_layer": selected_layer,
                "position_schedule": "final_prompt_token",
                "intervention_geometry": "matched_final_prompt",
                "construction_config_sha256": str(item["construction_config_sha256"]),
                "approved_strengths": normalized_strengths,
            }
        )
        for approved_strength in normalized_strengths:
            source_key = (
                model_id,
                str(approved_strength["source_method_id"]),
                float(approved_strength["strength"]),
                str(approved_strength["source_calibration_summary_sha256"]),
            )
            source_approval = source_approvals[source_key]
            approved_setups.append(
                {
                    "model_id": model_id,
                    "model_revision": str(model["revision"]),
                    "model_config_sha256": str(model["config_sha256"]),
                    "method_id": expected_method,
                    "track": "matched",
                    "direction_path": relative,
                    "direction_file_sha256": str(item["direction_file_sha256"]),
                    "direction_float32_sha256": artifact.direction_sha256,
                    "direction_artifact_sha256": artifact.artifact_sha256,
                    "selected_strength": approved_strength["strength"],
                    "selected_layer": selected_layer,
                    "position_schedule": "final_prompt_token",
                    "construction_config_sha256": str(
                        item["construction_config_sha256"]
                    ),
                    "validation_summary_sha256": approved_strength[
                        "source_calibration_summary_sha256"
                    ],
                    "control_source_method_id": approved_strength["source_method_id"],
                    "control_source_strength": approved_strength["strength"],
                    "control_source_calibration_summary_sha256": approved_strength[
                        "source_calibration_summary_sha256"
                    ],
                    "control_source_calibration_status": source_approval[
                        "calibration_status"
                    ],
                    "control_source_validation_safe": bool(
                        source_approval["validation_safe"]
                    ),
                    "control_source_validation_sign_safe": copy.deepcopy(
                        source_approval["validation_sign_safe"]
                    ),
                    "calibration_summary_verified": True,
                    "validation_coverage_adequate": True,
                    "strength_roles": ["random_control"],
                    "sealed_evaluation_required": True,
                    "winner_eligible": False,
                }
            )
    if observed != expected:
        raise RuntimeError(
            "stage-2 random-control coverage mismatch: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return (
        sorted(normalized, key=lambda row: (row["model_id"], row["seed"])),
        sorted(
            approved_setups,
            key=lambda row: (
                row["model_id"],
                row["method_id"],
                row["control_source_method_id"],
                row["selected_strength"],
            ),
        ),
        paths,
    )


def _preopen_summary_record(
    repo_root: Path,
    lock: Mapping[str, Any],
    summary_path: Path | str,
    *,
    stage1_lock_sha256: str,
    runner_parent_commit: str,
) -> tuple[dict[str, Any], set[str]]:
    """Rebuild one forced-only summary and derive its exact allowed open identities."""

    from .comparison_calibration import (
        CALIBRATION_SUMMARY_SCHEMA,
        SafetyLimits,
        build_calibration_summary,
        calibration_coverage_sha256,
        calibration_rows_sha256,
        locked_forced_calibration_units,
        locked_open_confirmation_units,
    )

    relative = _repo_relative_path(
        repo_root, summary_path, field="pre-open calibration summary path"
    )
    summary = _json_object(repo_root / relative, field="pre-open calibration summary")
    if (
        CALIBRATION_SUMMARY_SCHEMA != CALIBRATION_SUMMARY_SCHEMA_VERSION
        or summary.get("schema_version") != CALIBRATION_SUMMARY_SCHEMA_VERSION
    ):
        raise RuntimeError("pre-open summary has an unsupported calibration schema")
    if summary.get("open_confirmations") != [] or summary.get(
        "open_result_rows_artifacts"
    ) != []:
        raise RuntimeError("pre-open summary must not contain validation-open results")
    model_id = str(summary.get("model_id"))
    method_id = str(summary.get("method_id"))
    track = str(summary.get("track"))
    coverage_key = (model_id, method_id, track)
    if coverage_key not in _expected_stage2_coverage(lock):
        raise RuntimeError(f"unexpected pre-open model/method/track: {coverage_key}")
    model = _model_records(lock)[model_id]
    expected_identity = {
        "model_revision": str(model["revision"]),
        "dataset_sha256": str(lock["dataset"]["sha256"]),
        "protocol_sha256": str(lock["protocol"]["sha256"]),
        "stage1_lock_sha256": stage1_lock_sha256,
        "mode": track,
    }
    mismatches = {
        key: (value, summary.get(key))
        for key, value in expected_identity.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"pre-open summary identity mismatch: {mismatches}")
    calibration = lock["calibration"]
    calibration_hash = sha256_json(calibration)
    if summary.get("calibration_config_sha256") != calibration_hash:
        raise RuntimeError("pre-open summary calibration hash differs from stage 1")
    limits = SafetyLimits.from_lock(calibration["safety_gates"])
    if summary.get("safety_limits") != limits.to_lock_record() or summary.get(
        "safety_limits_sha256"
    ) != sha256_json(limits.to_lock_record()):
        raise RuntimeError("pre-open summary safety gates differ from stage 1")
    builder = summary.get("builder")
    if (
        not isinstance(builder, Mapping)
        or builder.get("module") != CALIBRATION_BUILDER_PATH
        or not _valid_digest(builder.get("module_sha256"))
    ):
        raise RuntimeError("pre-open summary has invalid builder provenance")
    HashEntry(CALIBRATION_BUILDER_PATH, str(builder["module_sha256"])).verify(
        repo_root
    )

    if track == "matched":
        strengths = calibration["matched_strength_grid"]
        layers = [int(model["matched_intervention"]["layer_zero_based"])]
    else:
        strengths = calibration["canonical_multiplier_grids"].get(method_id)
        layer_overrides = calibration.get("canonical_candidate_layers", {}).get(
            method_id
        )
        if layer_overrides is not None:
            layers = list(map(int, layer_overrides))
        elif method_id in {"caa", "persona_vector"}:
            layers = list(range(int(model["architecture"]["blocks"])))
        else:
            layers = [int(model["matched_intervention"]["layer_zero_based"])]
    expected_grid = {
        (layer, float(strength)) for layer in layers for strength in strengths
    }
    points = summary.get("points")
    if not isinstance(points, list) or {
        (int(point["layer"]), float(point["strength"]))
        for point in points
        if isinstance(point, Mapping)
    } != expected_grid or len(points) != len(expected_grid):
        raise RuntimeError("pre-open summary does not contain the exact forced grid")

    candidates = summary.get("candidate_directions")
    if not isinstance(candidates, list) or len(candidates) != len(layers):
        raise RuntimeError("pre-open summary has incomplete candidate directions")
    candidate_by_layer: dict[int, tuple[str, str]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise TypeError("pre-open candidate direction must be an object")
        layer = candidate.get("layer")
        hashes = (
            candidate.get("direction_float32_sha256"),
            candidate.get("direction_artifact_sha256"),
        )
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer in candidate_by_layer
            or any(not _valid_digest(value) for value in hashes)
        ):
            raise RuntimeError("pre-open candidate direction identity is invalid")
        candidate_by_layer[layer] = (str(hashes[0]), str(hashes[1]))
    if set(candidate_by_layer) != set(layers):
        raise RuntimeError("pre-open candidate directions differ from locked layers")

    dataset = _json_object(
        repo_root / str(lock["dataset"]["path"]), field="locked comparison dataset"
    )
    forced_units = locked_forced_calibration_units(dataset, lock)
    open_units = locked_open_confirmation_units(dataset, lock)
    if (
        summary.get("forced_coverage_unit_count") != len(forced_units)
        or summary.get("forced_coverage_sha256")
        != calibration_coverage_sha256(forced_units)
        or summary.get("open_coverage_unit_count") != len(open_units)
        or summary.get("open_coverage_sha256")
        != calibration_coverage_sha256(open_units)
    ):
        raise RuntimeError("pre-open summary coverage hashes differ from locked manifests")

    artifact_records = summary.get("forced_result_rows_artifacts")
    if not isinstance(artifact_records, list) or not artifact_records:
        raise RuntimeError("pre-open summary lacks forced result artifacts")
    normalized_artifacts: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    frozen_paths = {relative, CALIBRATION_BUILDER_PATH}
    grid_plan_record = summary.get("forced_grid_plan_artifact")
    if (
        not isinstance(grid_plan_record, Mapping)
        or not grid_plan_record.get("path")
        or not _valid_digest(grid_plan_record.get("sha256"))
    ):
        raise RuntimeError("pre-open forced grid plan artifact is invalid")
    grid_plan_relative = _repo_relative_path(
        repo_root,
        str(grid_plan_record["path"]),
        field="pre-open forced grid plan path",
    )
    HashEntry(grid_plan_relative, str(grid_plan_record["sha256"])).verify(repo_root)
    grid_plan = _json_object(
        repo_root / grid_plan_relative, field="pre-open forced grid plan"
    )
    planned_grid_shard_names = {
        str(point["shard_name"])
        for point in grid_plan.get("points", [])
        if isinstance(point, Mapping) and point.get("shard_name")
    }
    if not planned_grid_shard_names:
        raise RuntimeError("pre-open forced grid plan has no point shard names")
    frozen_paths.add(grid_plan_relative)
    interpolation_artifact_seen = False
    for index, item in enumerate(artifact_records):
        if (
            not isinstance(item, Mapping)
            or not item.get("path")
            or not _valid_digest(item.get("sha256"))
        ):
            raise RuntimeError(f"pre-open forced artifact {index} is invalid")
        artifact_path = _repo_relative_path(
            repo_root, str(item["path"]), field="pre-open forced artifact path"
        )
        if artifact_path in frozen_paths:
            raise RuntimeError("pre-open forced artifact path is duplicated")
        HashEntry(artifact_path, str(item["sha256"])).verify(repo_root)
        if Path(artifact_path).name not in planned_grid_shard_names:
            artifact_rows = _read_jsonl_objects(
                repo_root / artifact_path, label="pre-open forced artifact"
            )
            interpolation = summary.get("interpolation_recheck")
            if (
                interpolation_artifact_seen
                or not isinstance(interpolation, Mapping)
                or calibration_rows_sha256(artifact_rows)
                != interpolation.get("rows_sha256")
            ):
                raise RuntimeError(
                    "pre-open non-grid forced artifact is not the single locked "
                    "interpolation recheck"
                )
            interpolation_artifact_seen = True
        else:
            from .comparison_grid import load_validated_point_rows

            artifact_rows, _ = load_validated_point_rows(
                repo_root / artifact_path,
                plan=grid_plan,
                lock=lock,
                dataset=dataset,
                repo_root=repo_root,
            )
        rows.extend(artifact_rows)
        normalized_artifacts.append(
            {"path": artifact_path, "sha256": str(item["sha256"])}
        )
        frozen_paths.add(artifact_path)
    if (
        summary.get("interpolation_recheck") is not None
        and not interpolation_artifact_seen
    ):
        raise RuntimeError("pre-open summary omits the locked interpolation recheck artifact")
    construction_by_layer: dict[int, str] = {}
    common = {
        "model_id": model_id,
        "model_revision": str(model["revision"]),
        "method": method_id,
        "method_id": method_id,
        "setup": track,
        "track": track,
        "split": "validation",
        "dataset_sha256": str(lock["dataset"]["sha256"]),
        "protocol_sha256": str(lock["protocol"]["sha256"]),
        "config_sha256": str(model["config_sha256"]),
        "stage1_lock_sha256": stage1_lock_sha256,
        "stage2_manifest_sha256": "0" * 64,
        "calibration_summary_sha256": "0" * 64,
        "runner_commit": runner_parent_commit,
        "position": _expected_position_schedule(method_id, track),
    }
    grouped: dict[tuple[int, float, str, str], list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        row_mismatches = {
            key: (value, row.get(key))
            for key, value in common.items()
            if row.get(key) != value
        }
        if row_mismatches:
            raise RuntimeError(
                f"pre-open forced row {index} identity mismatch: {row_mismatches}"
            )
        layer = row.get("layer")
        magnitude = row.get("calibration_magnitude")
        construction_hash = row.get("construction_config_sha256")
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or isinstance(magnitude, bool)
            or not isinstance(magnitude, (int, float))
            or not math.isfinite(float(magnitude))
            or float(magnitude) <= 0
            or not _valid_digest(construction_hash)
        ):
            raise RuntimeError(f"pre-open forced row {index} has invalid point identity")
        hashes = (
            str(row.get("direction_float32_sha256")),
            str(row.get("direction_artifact_sha256")),
        )
        if candidate_by_layer.get(layer) != hashes:
            raise RuntimeError("pre-open forced row uses an undeclared candidate direction")
        previous_construction = construction_by_layer.setdefault(
            layer, str(construction_hash)
        )
        if previous_construction != construction_hash:
            raise RuntimeError("one candidate layer uses multiple construction configs")
        grouped.setdefault((layer, float(magnitude), *hashes), []).append(row)
    rows_by_hash = {
        calibration_rows_sha256(group): group for group in grouped.values()
    }
    if len(rows_by_hash) != len(grouped):
        raise RuntimeError("pre-open forced artifacts duplicate a canonical point")
    point_hashes = summary.get("point_rows_sha256s")
    if not isinstance(point_hashes, list) or any(
        not _valid_digest(value) for value in point_hashes
    ):
        raise RuntimeError("pre-open summary point hashes are invalid")
    interpolation = summary.get("interpolation_recheck")
    interpolation_hash = None
    if interpolation is not None:
        if not isinstance(interpolation, Mapping) or not _valid_digest(
            interpolation.get("rows_sha256")
        ):
            raise RuntimeError("pre-open interpolation evidence is invalid")
        interpolation_hash = str(interpolation["rows_sha256"])
    required_hashes = set(map(str, point_hashes))
    if interpolation_hash is not None:
        required_hashes.add(interpolation_hash)
    if set(rows_by_hash) != required_hashes:
        raise RuntimeError("pre-open forced artifacts do not exactly match summary hashes")
    rebuilt = build_calibration_summary(
        [rows_by_hash[str(value)] for value in point_hashes],
        expected_forced_units=forced_units,
        expected_open_units=open_units,
        safety_limits=limits,
        mode=track,
        forced_result_rows_artifacts=normalized_artifacts,
        open_result_rows_artifacts=[],
        forced_grid_plan_artifact={
            "path": grid_plan_relative,
            "sha256": str(grid_plan_record["sha256"]),
        },
        calibration_config_sha256=calibration_hash,
        builder_module_sha256=str(builder["module_sha256"]),
        interpolation_recheck_rows=(
            None if interpolation_hash is None else rows_by_hash[interpolation_hash]
        ),
        open_confirmation_rows=[],
        allow_pending_open=True,
        fixed_strength=float(
            lock["comparison_tracks"]["matched_primary"]["fixed_strength"]
        ),
        target=float(
            calibration[
                "equal_efficacy_target_mean_self_minus_other_bidirectional_effect"
            ]
        ),
        tie_tolerance=float(calibration["canonical_layer_tie_tolerance"]),
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(summary):
        raise RuntimeError("pre-open summary is not the canonical forced-only rebuild")
    pre_open = summary["pre_open_decision"]
    if pre_open["status"] == "interpolation_requires_one_recheck":
        raise RuntimeError("pre-open selection still requires its one forced recheck")
    selected_strength = pre_open.get("selected_strength")
    selected_layer = pre_open.get("selected_layer")
    roles_by_key: dict[tuple[int, float, str, str], set[str]] = {}
    if selected_strength is not None and selected_layer is not None:
        direction_hash, artifact_hash = candidate_by_layer[int(selected_layer)]
        roles_by_key[
            (
                int(selected_layer),
                float(selected_strength),
                direction_hash,
                artifact_hash,
            )
        ] = {"selected"}
        if track == "matched":
            fixed_strength = float(
                lock["comparison_tracks"]["matched_primary"]["fixed_strength"]
            )
            fixed_point = next(
                point
                for point in points
                if int(point["layer"]) == int(selected_layer)
                and math.isclose(
                    float(point["strength"]),
                    fixed_strength,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            )
            if fixed_point["safe"] is True:
                roles_by_key.setdefault(
                    (
                        int(selected_layer),
                        fixed_strength,
                        direction_hash,
                        artifact_hash,
                    ),
                    set(),
                ).add("fixed_descriptive")
    allowed = [
        {
            "model_id": model_id,
            "model_revision": str(model["revision"]),
            "model_config_sha256": str(model["config_sha256"]),
            "method_id": method_id,
            "track": track,
            "selected_layer": layer,
            "selected_strength": strength,
            "position_schedule": _expected_position_schedule(method_id, track),
            "direction_float32_sha256": direction_hash,
            "direction_artifact_sha256": artifact_hash,
            "construction_config_sha256": construction_by_layer[layer],
            "calibration_summary_path": relative,
            "calibration_summary_sha256": sha256_file(repo_root / relative),
            "roles": sorted(roles),
        }
        for (layer, strength, direction_hash, artifact_hash), roles in sorted(
            roles_by_key.items()
        )
    ]
    return (
        {
            "model_id": model_id,
            "method_id": method_id,
            "track": track,
            "calibration_summary_path": relative,
            "calibration_summary_sha256": sha256_file(repo_root / relative),
            "pre_open_decision": copy.deepcopy(pre_open),
            "pre_open_decision_sha256": str(summary["pre_open_decision_sha256"]),
            "candidate_directions": copy.deepcopy(candidates),
            "construction_config_sha256_by_layer": {
                str(layer): digest
                for layer, digest in sorted(construction_by_layer.items())
            },
            "forced_result_rows_artifacts": normalized_artifacts,
            "forced_grid_plan_artifact": {
                "path": grid_plan_relative,
                "sha256": str(grid_plan_record["sha256"]),
            },
            "allowed_open_setups": allowed,
        },
        frozen_paths,
    )


def _preopen_direction_index(
    repo_root: Path, manifest_paths: Sequence[Path | str]
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], list[dict[str, str]], set[str]]:
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    source_records: list[dict[str, str]] = []
    frozen: set[str] = set()
    for manifest_path in manifest_paths:
        relative_manifest = _repo_relative_path(
            repo_root, manifest_path, field="direction manifest path"
        )
        if relative_manifest in frozen:
            raise RuntimeError("duplicate pre-open direction manifest")
        payload = _json_object(
            repo_root / relative_manifest, field="pre-open direction manifest"
        )
        directions = payload.get("directions")
        if not isinstance(directions, list) or not directions:
            raise RuntimeError("pre-open direction manifest lacks directions")
        source_records.append(
            {
                "path": relative_manifest,
                "sha256": sha256_file(repo_root / relative_manifest),
            }
        )
        frozen.add(relative_manifest)
        for item in directions:
            if not isinstance(item, Mapping):
                raise TypeError("direction manifest entry must be an object")
            direction_path = _repo_relative_path(
                repo_root, str(item.get("path")), field="pre-open direction path"
            )
            construction_path = _repo_relative_path(
                repo_root,
                str(item.get("construction_config_path")),
                field="pre-open construction config path",
            )
            if direction_path in frozen or construction_path in frozen:
                raise RuntimeError("pre-open direction/construction path is duplicated")
            direction_file_sha = sha256_file(repo_root / direction_path)
            construction_sha = sha256_file(repo_root / construction_path)
            if item.get("construction_config_sha256") != construction_sha:
                raise RuntimeError("direction manifest construction hash is invalid")
            record = _read_direction_record_lightweight(
                repo_root / direction_path, label="pre-open direction artifact"
            )
            metadata = record.get("metadata")
            if not isinstance(metadata, Mapping):
                raise TypeError("pre-open direction metadata must be an object")
            track = str(item.get("track"))
            key = (
                str(metadata.get("model_id")),
                str(record.get("method")),
                track,
                int(record.get("layer")),
                str(record.get("direction_sha256")),
                str(record.get("artifact_sha256")),
                construction_sha,
            )
            expected_item = {
                "method_id": record["method"],
                "layer": record["layer"],
                "intervention_geometry": record["intervention_geometry"],
                "direction_float32_sha256": record["direction_sha256"],
                "direction_artifact_sha256": record["artifact_sha256"],
            }
            mismatches = {
                field: (value, item.get(field))
                for field, value in expected_item.items()
                if item.get(field) != value
            }
            if mismatches:
                raise RuntimeError(f"direction manifest identity mismatch: {mismatches}")
            if key in index:
                raise RuntimeError("duplicate pre-open direction identity")
            construction = _json_object(
                repo_root / construction_path, field="pre-open construction config"
            )
            construction_expected = {
                "model_id": key[0],
                "method_id": key[1],
                "track": key[2],
                "selected_layer": key[3],
                "direction_float32_sha256": key[4],
                "direction_artifact_sha256": key[5],
            }
            construction_mismatches = {
                field: (value, construction.get(field))
                for field, value in construction_expected.items()
                if construction.get(field) != value
            }
            if construction_mismatches:
                raise RuntimeError(
                    "pre-open construction config identity mismatch: "
                    f"{construction_mismatches}"
                )
            index[key] = {
                "direction_path": direction_path,
                "direction_file_sha256": direction_file_sha,
                "construction_config_path": construction_path,
                "construction_config_sha256": construction_sha,
                "intervention_geometry": record["intervention_geometry"],
            }
            frozen.update({direction_path, construction_path})
    if not index:
        raise RuntimeError("pre-open direction manifests produced no directions")
    return index, source_records, frozen


def _infer_runner_code_commit_from_calibration(
    repo_root: Path,
    lock: Mapping[str, Any],
    calibration_summary_paths: Sequence[Path | str],
) -> str:
    from .comparison_grid import _validate_plan

    commits: set[str] = set()
    for summary_path in calibration_summary_paths:
        relative = _repo_relative_path(
            repo_root, summary_path, field="calibration summary path"
        )
        summary = _json_object(repo_root / relative, field="calibration summary")
        plan_record = summary.get("forced_grid_plan_artifact")
        if (
            not isinstance(plan_record, Mapping)
            or not plan_record.get("path")
            or not _valid_digest(plan_record.get("sha256"))
        ):
            raise RuntimeError("calibration summary lacks a valid forced grid plan")
        plan_relative = _repo_relative_path(
            repo_root,
            str(plan_record["path"]),
            field="forced grid plan path",
        )
        HashEntry(plan_relative, str(plan_record["sha256"])).verify(repo_root)
        plan = _json_object(repo_root / plan_relative, field="forced grid plan")
        _validate_plan(plan, lock, repo_root=repo_root)
        commits.add(str(plan.get("runner_commit")))
    if len(commits) != 1:
        raise RuntimeError(
            "forced grid plans do not share one runner code commit"
        )
    runner_code_commit = next(iter(commits))
    if (
        len(runner_code_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in runner_code_commit.lower()
        )
    ):
        raise RuntimeError("forced calibration runner code commit is invalid")
    return runner_code_commit


def build_preopen_manifest(
    repo_root: Path,
    lock: Mapping[str, Any],
    *,
    stage1_lock_path: Path | str,
    calibration_summary_paths: Sequence[Path | str],
    direction_manifest_paths: Sequence[Path | str],
    runner_parent_commit: str | None = None,
    artifact_freeze_commit: str | None = None,
) -> dict[str, Any]:
    """Build the canonical forced-only lock that authorizes validation-open runs."""

    repo_root = repo_root.resolve()
    stage1_relative = _repo_relative_path(
        repo_root, stage1_lock_path, field="stage-1 lock path"
    )
    stage1_sha = sha256_file(repo_root / stage1_relative)
    verified_lock = verify_stage1_lock(repo_root, repo_root / stage1_relative)
    if canonical_json_bytes(verified_lock) != canonical_json_bytes(dict(lock)):
        raise RuntimeError("pre-open builder lock differs from verified stage-1 bytes")
    outcome_blind_amendment = _outcome_blind_amendment_binding(
        repo_root,
        verified_lock,
        stage1_lock_path=stage1_relative,
    )
    runner_code = (
        _infer_runner_code_commit_from_calibration(
            repo_root, verified_lock, calibration_summary_paths
        )
        if runner_parent_commit is None
        else runner_parent_commit
    )
    if (
        not isinstance(runner_code, str)
        or len(runner_code) != 40
        or any(
            character not in "0123456789abcdef"
            for character in runner_code.lower()
        )
    ):
        raise RuntimeError("pre-open runner code must be a 40-character commit")
    if (
        outcome_blind_amendment is not None
        and runner_code
        != outcome_blind_amendment["original_runner_code_commit"]
    ):
        raise RuntimeError(
            "pre-open rows must retain the original runner identity from before "
            "the outcome-blind amendment"
        )
    protected_baseline = _protected_provenance_baseline(
        runner_code, outcome_blind_amendment
    )
    artifact_freeze = (
        git_commit(repo_root)
        if artifact_freeze_commit is None
        else artifact_freeze_commit
    )
    if (
        not isinstance(artifact_freeze, str)
        or len(artifact_freeze) != 40
        or any(
            character not in "0123456789abcdef"
            for character in artifact_freeze.lower()
        )
        or artifact_freeze == runner_code
        or artifact_freeze == protected_baseline
        or not git_is_ancestor(repo_root, runner_code, artifact_freeze)
        or not git_is_ancestor(repo_root, protected_baseline, artifact_freeze)
    ):
        raise RuntimeError(
            "pre-open artifact freeze must be a later descendant of the runner code"
        )
    summary_records: list[dict[str, Any]] = []
    frozen_paths: set[str] = {stage1_relative}
    observed_coverage: set[tuple[str, str, str]] = set()
    for summary_path in calibration_summary_paths:
        record, paths = _preopen_summary_record(
            repo_root,
            verified_lock,
            summary_path,
            stage1_lock_sha256=stage1_sha,
            runner_parent_commit=runner_code,
        )
        key = (record["model_id"], record["method_id"], record["track"])
        if key in observed_coverage:
            raise RuntimeError(f"duplicate pre-open summary coverage: {key}")
        observed_coverage.add(key)
        summary_records.append(record)
        frozen_paths.update(paths)
    expected_coverage = _expected_stage2_coverage(verified_lock)
    if observed_coverage != expected_coverage:
        raise RuntimeError(
            "pre-open summary coverage mismatch: "
            f"missing={sorted(expected_coverage - observed_coverage)}, "
            f"extra={sorted(observed_coverage - expected_coverage)}"
        )
    direction_index, source_manifests, direction_paths = _preopen_direction_index(
        repo_root, direction_manifest_paths
    )
    frozen_paths.update(direction_paths)
    allowed_setups: list[dict[str, Any]] = []
    for summary in summary_records:
        for candidate in summary["candidate_directions"]:
            layer = int(candidate["layer"])
            construction_sha = summary["construction_config_sha256_by_layer"][
                str(layer)
            ]
            key = (
                summary["model_id"],
                summary["method_id"],
                summary["track"],
                layer,
                candidate["direction_float32_sha256"],
                candidate["direction_artifact_sha256"],
                construction_sha,
            )
            if key not in direction_index:
                raise RuntimeError(
                    "pre-open forced candidate lacks its exact direction/construction file"
                )
        for setup in summary["allowed_open_setups"]:
            key = (
                setup["model_id"],
                setup["method_id"],
                setup["track"],
                setup["selected_layer"],
                setup["direction_float32_sha256"],
                setup["direction_artifact_sha256"],
                setup["construction_config_sha256"],
            )
            allowed_setups.append({**setup, **direction_index[key]})
    required_protected = {
        stage1_relative,
        *(
            _repo_relative_path(repo_root, entry.path, field="stage-1 protected path")
            for entry in stage1_hash_entries(verified_lock)
        ),
    }
    if outcome_blind_amendment is not None:
        required_protected.add(str(outcome_blind_amendment["path"]))
    protected_paths = [
        {"path": path, "sha256": sha256_file(repo_root / path)}
        for path in sorted(required_protected)
    ]
    frozen_artifact_paths = [
        {"path": path, "sha256": sha256_file(repo_root / path)}
        for path in sorted(frozen_paths - required_protected)
    ]
    current = git_commit(repo_root)
    if not git_is_ancestor(repo_root, artifact_freeze, current):
        raise RuntimeError("current HEAD does not descend from the artifact freeze")
    if git_diff_paths(
        repo_root,
        protected_baseline,
        current,
        tuple(sorted(required_protected)),
    ):
        raise RuntimeError(
            "runner code/protocol changed after its effective provenance lock"
        )
    frozen_names = {item["path"] for item in frozen_artifact_paths}
    if git_diff_paths(
        repo_root, artifact_freeze, current, tuple(sorted(frozen_names))
    ):
        raise RuntimeError("frozen artifacts changed after their freeze commit")
    required_committed = {stage1_relative, *required_protected, *frozen_names}
    untracked = sorted(required_committed - git_tracked_paths(repo_root))
    if untracked:
        raise RuntimeError(
            f"pre-open inputs are not committed/tracked: {untracked[:5]}"
        )
    dirty = sorted(required_committed & set(git_dirty_paths(repo_root)))
    if dirty:
        raise RuntimeError(f"pre-open inputs are dirty: {dirty[:5]}")
    return {
        "schema_version": PREOPEN_SCHEMA_VERSION,
        "status": "locked_before_validation_open",
        "stage1_lock_path": stage1_relative,
        "stage1_lock_sha256": stage1_sha,
        "stage1_lock_payload_sha256": sha256_json(verified_lock),
        "runner_code_commit": runner_code,
        "outcome_blind_amendment": outcome_blind_amendment,
        "artifact_freeze_commit": artifact_freeze,
        "source_calibration_summaries": sorted(
            summary_records,
            key=lambda item: (item["model_id"], item["method_id"], item["track"]),
        ),
        "source_direction_manifests": sorted(
            source_manifests, key=lambda item: item["path"]
        ),
        "allowed_open_setups": sorted(
            allowed_setups,
            key=lambda item: (
                item["model_id"],
                item["method_id"],
                item["track"],
                item["selected_layer"],
                item["selected_strength"],
            ),
        ),
        "protected_paths": protected_paths,
        "frozen_artifact_paths": frozen_artifact_paths,
        "open_failure_policy": "ineligible_no_fallback_no_alternative_search",
    }


def verify_preopen_manifest(
    repo_root: Path,
    lock: Mapping[str, Any],
    manifest_path: Path,
) -> VerifiedPreopen:
    """Verify the committed pre-open lock and return its validation-open capability."""

    repo_root = repo_root.resolve()
    relative = _repo_relative_path(
        repo_root, manifest_path, field="pre-open manifest path"
    )
    expected_path = lock.get("lock_stages", {}).get("pre_open", {}).get("path")
    if not expected_path or relative != _repo_relative_path(
        repo_root, str(expected_path), field="locked pre-open manifest path"
    ):
        raise RuntimeError("pre-open manifest path differs from stage 1")
    manifest = _json_object(repo_root / relative, field="pre-open manifest")
    if manifest.get("schema_version") != PREOPEN_SCHEMA_VERSION or manifest.get(
        "status"
    ) != "locked_before_validation_open":
        raise RuntimeError("pre-open manifest schema/status is invalid")
    runner_code = manifest.get("runner_code_commit")
    artifact_freeze = manifest.get("artifact_freeze_commit")
    if not isinstance(runner_code, str) or not isinstance(artifact_freeze, str):
        raise TypeError("pre-open commit identities must be strings")
    summary_records = manifest.get("source_calibration_summaries")
    direction_records = manifest.get("source_direction_manifests")
    if not isinstance(summary_records, list) or not isinstance(direction_records, list):
        raise TypeError("pre-open source lists must be arrays")
    rebuilt = build_preopen_manifest(
        repo_root,
        lock,
        stage1_lock_path=str(manifest.get("stage1_lock_path")),
        calibration_summary_paths=[
            str(item.get("calibration_summary_path"))
            for item in summary_records
            if isinstance(item, Mapping)
        ],
        direction_manifest_paths=[
            str(item.get("path"))
            for item in direction_records
            if isinstance(item, Mapping)
        ],
        runner_parent_commit=runner_code,
        artifact_freeze_commit=artifact_freeze,
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(manifest):
        raise RuntimeError("pre-open manifest is not the canonical locked rebuild")
    outcome_blind_amendment = manifest.get("outcome_blind_amendment")
    if outcome_blind_amendment is not None and not isinstance(
        outcome_blind_amendment, Mapping
    ):
        raise TypeError("pre-open outcome_blind_amendment must be an object or null")
    protected_baseline = _protected_provenance_baseline(
        runner_code, outcome_blind_amendment
    )
    head = git_commit(repo_root)
    if (
        runner_code == artifact_freeze
        or protected_baseline == artifact_freeze
        or not git_is_ancestor(repo_root, runner_code, artifact_freeze)
        or not git_is_ancestor(repo_root, protected_baseline, artifact_freeze)
        or not git_is_ancestor(repo_root, artifact_freeze, head)
        or artifact_freeze == head
    ):
        raise RuntimeError(
            "pre-open manifest must be committed after its artifact freeze commit"
        )
    if relative not in git_diff_paths(
        repo_root, artifact_freeze, head, (relative,)
    ):
        raise RuntimeError("pre-open manifest is not committed after artifact freezing")
    protected = {str(item["path"]) for item in manifest["protected_paths"]}
    frozen_records = manifest.get("frozen_artifact_paths")
    if not isinstance(frozen_records, list) or not frozen_records:
        raise RuntimeError("pre-open manifest lacks frozen generated artifacts")
    frozen: set[str] = set()
    for index, item in enumerate(frozen_records):
        if (
            not isinstance(item, Mapping)
            or not item.get("path")
            or not _valid_digest(item.get("sha256"))
        ):
            raise RuntimeError(f"invalid pre-open frozen artifact record {index}")
        artifact_relative = _repo_relative_path(
            repo_root,
            str(item["path"]),
            field=f"frozen_artifact_paths[{index}].path",
        )
        if artifact_relative in frozen or artifact_relative in protected:
            raise RuntimeError("pre-open frozen/protected paths must be disjoint")
        HashEntry(artifact_relative, str(item["sha256"])).verify(repo_root)
        frozen.add(artifact_relative)
    changed = git_diff_paths(
        repo_root, protected_baseline, head, tuple(sorted(protected))
    )
    if changed:
        raise RuntimeError(
            "pre-open protected paths changed after their effective provenance "
            f"lock: {sorted(changed)[:5]}"
        )
    changed_frozen = git_diff_paths(
        repo_root, artifact_freeze, head, tuple(sorted(frozen))
    )
    if changed_frozen:
        raise RuntimeError(
            "pre-open artifacts changed after artifact_freeze_commit: "
            f"{sorted(changed_frozen)[:5]}"
        )
    required = {relative, *protected, *frozen}
    untracked = sorted(required - git_tracked_paths(repo_root))
    if untracked:
        raise RuntimeError(f"pre-open locked files are not Git-tracked: {untracked[:5]}")
    dirty = sorted(required & set(git_dirty_paths(repo_root)))
    if dirty:
        raise RuntimeError(f"pre-open locked files are dirty: {dirty[:5]}")
    return VerifiedPreopen._create(
        manifest_sha256=sha256_file(repo_root / relative),
        runner_code_commit=runner_code,
        artifact_freeze_commit=artifact_freeze,
        stage1_lock_sha256=str(manifest["stage1_lock_sha256"]),
        outcome_blind_amendment=outcome_blind_amendment,
        approved_setups=manifest["allowed_open_setups"],
        token=_VERIFIED_PREOPEN_TOKEN,
    )


def assert_preopen_approved_setup(
    verified: VerifiedPreopen | None,
    *,
    repo_root: Path,
    model_id: str,
    model_revision: str,
    model_config_sha256: str,
    method_id: str,
    track: str,
    direction_path: Path,
    selected_strength: float,
    selected_layer: int,
    position_schedule: str,
    construction_config_sha256: str,
) -> dict[str, Any]:
    """Fail before model load unless a validation-open setup was frozen pre-open."""

    if not isinstance(verified, VerifiedPreopen) or getattr(
        verified, "_capability_token", None
    ) is not _VERIFIED_PREOPEN_TOKEN:
        raise RuntimeError("a verified pre-open capability is required")
    relative = _repo_relative_path(
        repo_root.resolve(), direction_path, field="validation-open direction path"
    )
    record = _read_direction_record_lightweight(
        repo_root.resolve() / relative, label="validation-open direction"
    )
    supplied = {
        "model_id": model_id,
        "model_revision": model_revision,
        "model_config_sha256": model_config_sha256,
        "method_id": method_id,
        "track": track,
        "direction_path": relative,
        "direction_file_sha256": sha256_file(repo_root.resolve() / relative),
        "direction_float32_sha256": record["direction_sha256"],
        "direction_artifact_sha256": record["artifact_sha256"],
        "selected_strength": float(selected_strength),
        "selected_layer": selected_layer,
        "position_schedule": position_schedule,
        "construction_config_sha256": construction_config_sha256,
    }
    matches = [
        item
        for item in verified._approved_setups
        if all(item.get(key) == value for key, value in supplied.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "validation-open setup is not in the exact pre-open allowed candidate set"
        )
    return copy.deepcopy(dict(matches[0]))


def build_stage2_manifest(
    repo_root: Path,
    lock: Mapping[str, Any],
    *,
    stage1_lock_path: Path | str,
    preopen_manifest_path: Path | str,
    environment_lock_path: Path | str,
    calibration_summary_paths: Sequence[Path | str],
    direction_manifest_paths: Sequence[Path | str],
    runner_parent_commit: str | None = None,
    artifact_freeze_commit: str | None = None,
) -> dict[str, Any]:
    """Build stage two from finalized calibration and frozen direction evidence.

    The builder deliberately invokes the same artifact/calibration/random-control
    validators used by the sealed gate.  It therefore cannot bless an incomplete
    summary, an alternative post-open candidate, or a random strength that was not
    inherited from an approved matched contender.
    """

    repo_root = repo_root.resolve()
    stage1_relative = _repo_relative_path(
        repo_root, stage1_lock_path, field="stage-1 lock path"
    )
    stage1_path = repo_root / stage1_relative
    stage1_sha = sha256_file(stage1_path)
    verified_lock = verify_stage1_lock(repo_root, stage1_path)
    if canonical_json_bytes(verified_lock) != canonical_json_bytes(dict(lock)):
        raise RuntimeError("stage-2 builder lock differs from verified stage-1 bytes")

    preopen_relative = _repo_relative_path(
        repo_root, preopen_manifest_path, field="pre-open manifest path"
    )
    verified_preopen = verify_preopen_manifest(
        repo_root, verified_lock, repo_root / preopen_relative
    )
    preopen_payload = _json_object(
        repo_root / preopen_relative, field="verified pre-open manifest"
    )
    outcome_blind_amendment = _outcome_blind_amendment_binding(
        repo_root,
        verified_lock,
        stage1_lock_path=stage1_relative,
    )
    if canonical_json_bytes(outcome_blind_amendment) != canonical_json_bytes(
        preopen_payload.get("outcome_blind_amendment")
    ):
        raise RuntimeError(
            "stage-2 builder outcome-blind amendment differs from pre-open"
        )
    runner_code = (
        verified_preopen.runner_code_commit
        if runner_parent_commit is None
        else runner_parent_commit
    )
    if runner_code != verified_preopen.runner_code_commit:
        raise RuntimeError("stage-2 runner parent differs from the verified pre-open lock")
    if (
        outcome_blind_amendment is not None
        and runner_code
        != outcome_blind_amendment["original_runner_code_commit"]
    ):
        raise RuntimeError("stage-2 must retain the original construction runner")
    protected_baseline = _protected_provenance_baseline(
        runner_code, outcome_blind_amendment
    )
    artifact_freeze = (
        git_commit(repo_root)
        if artifact_freeze_commit is None
        else artifact_freeze_commit
    )
    if (
        not isinstance(artifact_freeze, str)
        or len(artifact_freeze) != 40
        or any(
            character not in "0123456789abcdef"
            for character in artifact_freeze.lower()
        )
        or artifact_freeze == runner_code
        or artifact_freeze == protected_baseline
        or not git_is_ancestor(repo_root, runner_code, artifact_freeze)
        or not git_is_ancestor(repo_root, protected_baseline, artifact_freeze)
    ):
        raise RuntimeError(
            "stage-2 artifact freeze must be a later descendant of the runner code"
        )

    environment_relative = _repo_relative_path(
        repo_root, environment_lock_path, field="environment lock path"
    )
    environment_path = repo_root / environment_relative
    from .comparison_environment import verify_current_environment

    verify_current_environment(
        environment_path,
        stage1_lock_path=stage1_path,
        lock=verified_lock,
    )

    direction_index, source_manifests, _ = _preopen_direction_index(
        repo_root, direction_manifest_paths
    )
    source_manifests = sorted(source_manifests, key=lambda item: item["path"])
    if canonical_json_bytes(source_manifests) != canonical_json_bytes(
        preopen_payload.get("source_direction_manifests")
    ):
        raise RuntimeError(
            "stage-2 direction manifests differ from those frozen pre-open"
        )
    preopen_summaries = {
        (str(item["model_id"]), str(item["method_id"]), str(item["track"])): item
        for item in preopen_payload.get("source_calibration_summaries", [])
        if isinstance(item, Mapping)
    }
    by_candidate: dict[tuple[Any, ...], tuple[tuple[Any, ...], dict[str, Any]]] = {}
    for key, value in direction_index.items():
        candidate_key = key[:-1]
        if candidate_key in by_candidate:
            raise RuntimeError(
                "stage-2 direction manifests contain multiple construction configs "
                "for one candidate"
            )
        by_candidate[candidate_key] = (key, value)

    model_records = _model_records(verified_lock)
    main_items: list[dict[str, Any]] = []
    normalized_main: list[dict[str, Any]] = []
    frozen_artifact_names: set[str] = {
        preopen_relative,
        environment_relative,
        *(str(item["path"]) for item in source_manifests),
    }
    observed: set[tuple[str, str, str]] = set()
    for index, summary_path in enumerate(calibration_summary_paths):
        summary_relative = _repo_relative_path(
            repo_root, summary_path, field="final calibration summary path"
        )
        summary = _json_object(
            repo_root / summary_relative, field="final calibration summary"
        )
        model_id = str(summary.get("model_id"))
        method_id = str(summary.get("method_id"))
        track = str(summary.get("track"))
        coverage = (model_id, method_id, track)
        if coverage not in _expected_stage2_coverage(verified_lock):
            raise RuntimeError(
                f"final calibration summary has unexpected coverage {coverage}"
            )
        if coverage in observed:
            raise RuntimeError(f"duplicate final calibration summary coverage {coverage}")
        observed.add(coverage)
        frozen_summary = preopen_summaries.get(coverage)
        if not isinstance(frozen_summary, Mapping):
            raise TypeError(
                f"final calibration summary {coverage} was not frozen pre-open"
            )
        for field in (
            "pre_open_decision_sha256",
            "candidate_directions",
            "forced_result_rows_artifacts",
            "forced_grid_plan_artifact",
        ):
            if canonical_json_bytes(summary.get(field)) != canonical_json_bytes(
                frozen_summary.get(field)
            ):
                raise RuntimeError(
                    f"final calibration {coverage} changed frozen pre-open {field}"
                )
        decision = summary.get("decision")
        candidates = summary.get("candidate_directions")
        if not isinstance(decision, Mapping) or not isinstance(candidates, list) or not candidates:
            raise RuntimeError("final calibration summary lacks decision/candidates")
        selected_strength = decision.get("selected_strength")
        selected_layer = decision.get("selected_layer")
        if selected_layer is None:
            candidate_layers = sorted(
                int(candidate["layer"])
                for candidate in candidates
                if isinstance(candidate, Mapping)
            )
            if not candidate_layers:
                raise RuntimeError("final calibration summary has no candidate layers")
            selected_layer = candidate_layers[0]
        selected_candidate = next(
            (
                candidate
                for candidate in candidates
                if isinstance(candidate, Mapping)
                and candidate.get("layer") == selected_layer
            ),
            None,
        )
        if not isinstance(selected_candidate, Mapping):
            raise TypeError("final calibration selected layer lacks a direction")
        candidate_key = (
            model_id,
            method_id,
            track,
            int(selected_layer),
            str(selected_candidate.get("direction_float32_sha256")),
            str(selected_candidate.get("direction_artifact_sha256")),
        )
        indexed = by_candidate.get(candidate_key)
        if indexed is None:
            raise RuntimeError(
                "final calibration selected direction is absent from direction manifests"
            )
        full_key, direction_record = indexed
        construction_sha = str(full_key[-1])
        sealed_required = bool(
            selected_strength is not None
            and decision.get("open_confirmation_passed") is True
        )
        item = {
            "model_id": model_id,
            "method_id": method_id,
            "track": track,
            "direction_path": direction_record["direction_path"],
            "direction_file_sha256": direction_record["direction_file_sha256"],
            "direction_float32_sha256": candidate_key[-2],
            "direction_artifact_sha256": candidate_key[-1],
            "intervention_geometry": direction_record["intervention_geometry"],
            "construction_config_path": direction_record[
                "construction_config_path"
            ],
            "construction_config_sha256": construction_sha,
            "selected_strength": (
                None if selected_strength is None else float(selected_strength)
            ),
            "selected_layer": int(selected_layer),
            "position_schedule": _expected_position_schedule(method_id, track),
            "validation_summary_path": summary_relative,
            "validation_summary_sha256": sha256_file(repo_root / summary_relative),
            "sealed_evaluation_required": sealed_required,
            "dataset_sha256": str(verified_lock["dataset"]["sha256"]),
            "protocol_sha256": str(verified_lock["protocol"]["sha256"]),
        }
        _, normalized, paths = _verify_main_artifact(
            repo_root,
            verified_lock,
            model_records,
            item,
            index=index,
            stage1_lock_sha256=stage1_sha,
            runner_parent_commit=runner_code,
        )
        main_items.append(item)
        normalized_main.append(normalized)
        frozen_artifact_names.update(paths)
    expected_coverage = _expected_stage2_coverage(verified_lock)
    if observed != expected_coverage:
        raise RuntimeError(
            "final calibration summary coverage mismatch: "
            f"missing={sorted(expected_coverage - observed)}, "
            f"extra={sorted(observed - expected_coverage)}"
        )

    source_approvals = [
        setup
        for setup in _expand_main_approved_setups(verified_lock, normalized_main)
        if setup["method_id"] in MAIN_METHOD_IDS and setup["track"] == "matched"
    ]
    random_items: list[dict[str, Any]] = []
    for full_key, direction_record in direction_index.items():
        model_id, method_id, track, layer, direction_sha, artifact_sha, construction_sha = (
            full_key
        )
        if not str(method_id).startswith("random_control_"):
            continue
        construction = _json_object(
            repo_root / direction_record["construction_config_path"],
            field="random construction config",
        )
        approved_strengths = sorted(
            {
                (
                    str(setup["method_id"]),
                    float(setup["selected_strength"]),
                    str(setup["validation_summary_sha256"]),
                )
                for setup in source_approvals
                if setup["model_id"] == model_id
            }
        )
        random_items.append(
            {
                "model_id": model_id,
                "seed": construction.get("seed"),
                "method_id": method_id,
                "track": track,
                "direction_path": direction_record["direction_path"],
                "direction_file_sha256": direction_record["direction_file_sha256"],
                "direction_float32_sha256": direction_sha,
                "direction_artifact_sha256": artifact_sha,
                "intervention_geometry": direction_record[
                    "intervention_geometry"
                ],
                "selected_layer": layer,
                "position_schedule": _expected_position_schedule(method_id, track),
                "construction_config_path": direction_record[
                    "construction_config_path"
                ],
                "construction_config_sha256": construction_sha,
                "approved_strengths": [
                    {
                        "source_method_id": source_method_id,
                        "strength": strength,
                        "source_calibration_summary_sha256": summary_sha,
                    }
                    for source_method_id, strength, summary_sha in approved_strengths
                ],
                "dataset_sha256": str(verified_lock["dataset"]["sha256"]),
                "protocol_sha256": str(verified_lock["protocol"]["sha256"]),
            }
        )
    _, _, random_paths = _verify_random_controls(
        repo_root,
        verified_lock,
        model_records,
        random_items,
        stage1_lock_sha256=stage1_sha,
        runner_parent_commit=runner_code,
        main_records=normalized_main,
    )
    frozen_artifact_names.update(random_paths)

    required_protected = {
        stage1_relative,
        *(
            _repo_relative_path(repo_root, entry.path, field="stage-1 protected path")
            for entry in stage1_hash_entries(verified_lock)
        ),
    }
    if outcome_blind_amendment is not None:
        required_protected.add(str(outcome_blind_amendment["path"]))
    frozen_artifact_names.difference_update(required_protected)
    current = git_commit(repo_root)
    if not git_is_ancestor(repo_root, artifact_freeze, current):
        raise RuntimeError("current HEAD does not descend from stage-2 artifact freeze")
    if git_diff_paths(
        repo_root,
        protected_baseline,
        current,
        tuple(sorted(required_protected)),
    ):
        raise RuntimeError(
            "runner code/protocol changed after its effective provenance lock"
        )
    if git_diff_paths(
        repo_root, artifact_freeze, current, tuple(sorted(frozen_artifact_names))
    ):
        raise RuntimeError("stage-two artifacts changed after their freeze commit")
    required_committed = {
        stage1_relative,
        *required_protected,
        *frozen_artifact_names,
    }
    untracked = sorted(required_committed - git_tracked_paths(repo_root))
    if untracked:
        raise RuntimeError(
            f"stage-two inputs are not committed/tracked: {untracked[:5]}"
        )
    dirty = sorted(required_committed & set(git_dirty_paths(repo_root)))
    if dirty:
        raise RuntimeError(f"stage-two inputs are dirty: {dirty[:5]}")
    return {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "status": "locked_before_sealed_test",
        "stage1_lock_path": stage1_relative,
        "stage1_lock_sha256": stage1_sha,
        "preopen_manifest_path": preopen_relative,
        "preopen_manifest_sha256": verified_preopen.manifest_sha256,
        "environment_lock_path": environment_relative,
        "environment_lock_sha256": sha256_file(environment_path),
        "runner_code_commit": runner_code,
        "outcome_blind_amendment": outcome_blind_amendment,
        "artifact_freeze_commit": artifact_freeze,
        "protected_paths": [
            {"path": path, "sha256": sha256_file(repo_root / path)}
            for path in sorted(required_protected)
        ],
        "frozen_artifact_paths": [
            {"path": path, "sha256": sha256_file(repo_root / path)}
            for path in sorted(frozen_artifact_names)
        ],
        "source_direction_manifests": source_manifests,
        "direction_and_calibration_artifacts": sorted(
            main_items,
            key=lambda item: (item["model_id"], item["method_id"], item["track"]),
        ),
        "random_direction_controls": sorted(
            random_items, key=lambda item: (item["model_id"], item["seed"])
        ),
    }


def verify_stage2_manifest(
    repo_root: Path,
    lock: Mapping[str, Any],
    manifest_path: Path,
) -> VerifiedStage2:
    """Verify the non-circular stage-2 manifest and return an evaluation capability."""

    repo_root = repo_root.resolve()
    manifest_relative = _repo_relative_path(
        repo_root, manifest_path, field="stage-2 manifest path"
    )
    expected_manifest_path = (
        lock.get("lock_stages", {}).get("stage_2", {}).get("path")
        if isinstance(lock.get("lock_stages"), Mapping)
        else None
    )
    if not expected_manifest_path or manifest_relative != _repo_relative_path(
        repo_root, str(expected_manifest_path), field="locked stage-2 manifest path"
    ):
        raise RuntimeError("stage-2 manifest path differs from the stage-1 lock")
    manifest = _json_object(repo_root / manifest_relative, field="stage-2 manifest")
    if manifest.get("schema_version") != STAGE2_SCHEMA_VERSION:
        raise RuntimeError(
            f"stage-2 manifest schema_version must be {STAGE2_SCHEMA_VERSION!r}"
        )
    if manifest.get("status") != "locked_before_sealed_test":
        raise RuntimeError("stage-2 manifest is not locked")

    _require_non_null_fields(
        manifest,
        (
            "stage1_lock_path",
            "stage1_lock_sha256",
            "preopen_manifest_path",
            "preopen_manifest_sha256",
            "environment_lock_path",
            "environment_lock_sha256",
            "runner_code_commit",
            "artifact_freeze_commit",
            "protected_paths",
            "frozen_artifact_paths",
            "source_direction_manifests",
            "direction_and_calibration_artifacts",
            "random_direction_controls",
        ),
        label="stage-2 manifest",
    )
    if not _valid_digest(manifest["stage1_lock_sha256"]):
        raise RuntimeError("stage-2 manifest has invalid stage1_lock_sha256")
    stage1_relative = _repo_relative_path(
        repo_root,
        str(manifest["stage1_lock_path"]),
        field="stage1_lock_path",
    )
    HashEntry(stage1_relative, str(manifest["stage1_lock_sha256"])).verify(repo_root)
    verified_lock = verify_stage1_lock(repo_root, repo_root / stage1_relative)
    if canonical_json_bytes(verified_lock) != canonical_json_bytes(dict(lock)):
        raise RuntimeError("supplied stage-1 lock differs from the verified locked file")
    outcome_blind_amendment = _outcome_blind_amendment_binding(
        repo_root,
        verified_lock,
        stage1_lock_path=stage1_relative,
    )
    if canonical_json_bytes(outcome_blind_amendment) != canonical_json_bytes(
        manifest.get("outcome_blind_amendment")
    ):
        raise RuntimeError(
            "stage-2 outcome-blind amendment differs from current protected code"
        )

    if not _valid_digest(manifest["preopen_manifest_sha256"]):
        raise RuntimeError("stage-2 manifest has invalid preopen_manifest_sha256")
    preopen_relative = _repo_relative_path(
        repo_root,
        str(manifest["preopen_manifest_path"]),
        field="preopen_manifest_path",
    )
    HashEntry(preopen_relative, str(manifest["preopen_manifest_sha256"])).verify(
        repo_root
    )
    verified_preopen = verify_preopen_manifest(
        repo_root, verified_lock, repo_root / preopen_relative
    )
    preopen_payload = _json_object(
        repo_root / preopen_relative, field="verified pre-open manifest"
    )
    if canonical_json_bytes(outcome_blind_amendment) != canonical_json_bytes(
        preopen_payload.get("outcome_blind_amendment")
    ):
        raise RuntimeError("stage-2 outcome-blind amendment differs from pre-open")

    runner_code = manifest.get("runner_code_commit")
    artifact_freeze = manifest.get("artifact_freeze_commit")
    if (
        not isinstance(runner_code, str)
        or len(runner_code) != 40
        or any(
            character not in "0123456789abcdef"
            for character in runner_code.lower()
        )
        or not isinstance(artifact_freeze, str)
        or len(artifact_freeze) != 40
        or any(
            character not in "0123456789abcdef"
            for character in artifact_freeze.lower()
        )
    ):
        raise RuntimeError("stage-2 manifest lacks valid commit identities")
    if runner_code != verified_preopen.runner_code_commit:
        raise RuntimeError("stage-2 runner code differs from the pre-open lock")
    if (
        outcome_blind_amendment is not None
        and runner_code
        != outcome_blind_amendment["original_runner_code_commit"]
    ):
        raise RuntimeError("stage-2 changed the original construction runner identity")
    protected_baseline = _protected_provenance_baseline(
        runner_code, outcome_blind_amendment
    )
    head = git_commit(repo_root)
    if (
        runner_code == artifact_freeze
        or protected_baseline == artifact_freeze
        or not git_is_ancestor(repo_root, runner_code, artifact_freeze)
        or not git_is_ancestor(repo_root, protected_baseline, artifact_freeze)
        or not git_is_ancestor(repo_root, artifact_freeze, head)
        or artifact_freeze == head
    ):
        raise RuntimeError(
            "stage-2 manifest must be committed after its artifact freeze commit"
        )
    if manifest_relative not in git_diff_paths(
        repo_root, artifact_freeze, head, (manifest_relative,)
    ):
        raise RuntimeError("stage-2 manifest is not committed after artifact freezing")

    protected = manifest.get("protected_paths")
    if not isinstance(protected, list) or not protected:
        raise RuntimeError("stage-2 manifest must hash protected runner paths")
    protected_names: set[str] = set()
    protected_records: list[dict[str, str]] = []
    for index, item in enumerate(protected):
        if not isinstance(item, Mapping) or not item.get("path") or not _valid_digest(
            item.get("sha256")
        ):
            raise RuntimeError(f"invalid protected-path record {index}")
        relative = _repo_relative_path(
            repo_root, str(item["path"]), field=f"protected_paths[{index}].path"
        )
        if relative in protected_names:
            raise RuntimeError(f"duplicate protected path: {relative}")
        if relative == manifest_relative:
            raise RuntimeError("stage-2 manifest must be excluded from protected paths")
        entry = HashEntry(relative, str(item["sha256"]))
        entry.verify(repo_root)
        protected_names.add(relative)
        protected_records.append({"path": relative, "sha256": entry.sha256})
    required_protected = {
        stage1_relative,
        *(
            _repo_relative_path(repo_root, entry.path, field="stage-1 protected path")
            for entry in stage1_hash_entries(verified_lock)
        ),
    }
    if outcome_blind_amendment is not None:
        required_protected.add(str(outcome_blind_amendment["path"]))
    missing_protected = sorted(required_protected - protected_names)
    if missing_protected:
        raise RuntimeError(
            f"stage-2 protected paths omit stage-1 inputs/code: {missing_protected[:5]}"
        )
    changed_protected = git_diff_paths(
        repo_root, protected_baseline, head, tuple(sorted(protected_names))
    )
    if changed_protected:
        raise RuntimeError(
            "protected runner paths changed since runner_code_commit: "
            f"{sorted(changed_protected)[:5]}"
        )

    frozen_manifest_records = manifest.get("frozen_artifact_paths")
    if not isinstance(frozen_manifest_records, list) or not frozen_manifest_records:
        raise RuntimeError("stage-2 manifest must hash frozen artifacts")
    frozen_manifest_names: set[str] = set()
    for index, item in enumerate(frozen_manifest_records):
        if (
            not isinstance(item, Mapping)
            or not item.get("path")
            or not _valid_digest(item.get("sha256"))
        ):
            raise RuntimeError(f"invalid stage-2 frozen artifact record {index}")
        relative = _repo_relative_path(
            repo_root,
            str(item["path"]),
            field=f"frozen_artifact_paths[{index}].path",
        )
        if relative in frozen_manifest_names or relative in protected_names:
            raise RuntimeError("stage-2 frozen/protected paths must be disjoint")
        HashEntry(relative, str(item["sha256"])).verify(repo_root)
        frozen_manifest_names.add(relative)

    if not _valid_digest(manifest["environment_lock_sha256"]):
        raise RuntimeError("stage-2 manifest has invalid environment_lock_sha256")
    environment_relative = _repo_relative_path(
        repo_root,
        str(manifest["environment_lock_path"]),
        field="environment_lock_path",
    )
    HashEntry(environment_relative, str(manifest["environment_lock_sha256"])).verify(
        repo_root
    )
    from .comparison_environment import verify_current_environment

    verify_current_environment(
        repo_root / environment_relative,
        stage1_lock_path=repo_root / stage1_relative,
        lock=verified_lock,
    )

    source_direction_manifests = manifest.get("source_direction_manifests")
    if not isinstance(source_direction_manifests, list) or not source_direction_manifests:
        raise RuntimeError("stage 2 lacks frozen source direction manifests")
    preopen_payload = _json_object(
        repo_root / preopen_relative, field="verified pre-open manifest"
    )
    if canonical_json_bytes(source_direction_manifests) != canonical_json_bytes(
        preopen_payload.get("source_direction_manifests")
    ):
        raise RuntimeError("stage-2 direction manifests differ from pre-open")
    source_direction_paths: set[str] = set()
    for index, record in enumerate(source_direction_manifests):
        if (
            not isinstance(record, Mapping)
            or not record.get("path")
            or not _valid_digest(record.get("sha256"))
        ):
            raise RuntimeError(f"invalid source direction manifest record {index}")
        relative = _repo_relative_path(
            repo_root,
            str(record["path"]),
            field=f"source_direction_manifests[{index}].path",
        )
        if relative in source_direction_paths:
            raise RuntimeError("duplicate source direction manifest path")
        HashEntry(relative, str(record["sha256"])).verify(repo_root)
        source_direction_paths.add(relative)

    artifacts = manifest.get("direction_and_calibration_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("stage 2 lacks direction/calibration artifacts")
    model_records = _model_records(verified_lock)
    observed_coverage: set[tuple[str, str, str]] = set()
    approved_direction_paths: set[str] = set()
    normalized_artifacts: list[dict[str, Any]] = []
    frozen_artifact_paths: set[str] = set()
    preopen_summary_records = {
        (str(item["model_id"]), str(item["method_id"]), str(item["track"])): item
        for item in preopen_payload.get("source_calibration_summaries", [])
        if isinstance(item, Mapping)
    }
    for index, item in enumerate(artifacts):
        if not isinstance(item, Mapping):
            raise TypeError(f"stage-2 artifact {index} is not an object")
        coverage_key = (
            str(item.get("model_id")),
            str(item.get("method_id")),
            str(item.get("track")),
        )
        frozen_summary = preopen_summary_records.get(coverage_key)
        if not isinstance(frozen_summary, Mapping):
            raise TypeError(
                f"stage-2 artifact {coverage_key} lacks a pre-open calibration lock"
            )
        validation_relative = _repo_relative_path(
            repo_root,
            str(item.get("validation_summary_path")),
            field=f"stage-2 artifact {index} validation summary path",
        )
        final_summary = _json_object(
            repo_root / validation_relative, field="stage-2 final calibration summary"
        )
        for field in (
            "pre_open_decision_sha256",
            "candidate_directions",
            "forced_result_rows_artifacts",
            "forced_grid_plan_artifact",
        ):
            if canonical_json_bytes(final_summary.get(field)) != canonical_json_bytes(
                frozen_summary.get(field)
            ):
                raise RuntimeError(
                    f"stage-2 artifact {coverage_key} changed frozen pre-open {field}"
                )
        coverage, normalized, paths = _verify_main_artifact(
            repo_root,
            verified_lock,
            model_records,
            item,
            index=index,
            stage1_lock_sha256=str(manifest["stage1_lock_sha256"]),
            runner_parent_commit=runner_code,
        )
        if coverage in observed_coverage:
            raise RuntimeError(f"duplicate stage-2 model/method/track record: {coverage}")
        direction_relative = str(normalized["direction_path"])
        if direction_relative in approved_direction_paths:
            raise RuntimeError(
                "each approved method/track must have a distinct direction artifact file: "
                f"{direction_relative}"
            )
        observed_coverage.add(coverage)
        approved_direction_paths.add(direction_relative)
        normalized_artifacts.append(normalized)
        frozen_artifact_paths.update(paths)
    expected_coverage = _expected_stage2_coverage(verified_lock)
    if observed_coverage != expected_coverage:
        raise RuntimeError(
            "stage-2 model/method/track coverage mismatch: "
            f"missing={sorted(expected_coverage - observed_coverage)}, "
            f"extra={sorted(observed_coverage - expected_coverage)}"
        )
    shared_baselines: dict[tuple[Any, ...], bytes] = {}
    for record in normalized_artifacts:
        baseline_records = record.pop("_open_baseline_records")
        for baseline in baseline_records:
            key = tuple(baseline["identity"])
            content = canonical_json_bytes(
                {
                    "baseline_content_sha256": baseline[
                        "baseline_content_sha256"
                    ],
                    "baseline_payload": baseline["baseline_payload"],
                    "judge_request_content_sha256": baseline[
                        "judge_request_content_sha256"
                    ],
                    "judge_response_content_sha256": baseline[
                        "judge_response_content_sha256"
                    ],
                    "judge_raw_response": baseline["judge_raw_response"],
                }
            )
            previous = shared_baselines.setdefault(key, content)
            if previous != content:
                raise RuntimeError(
                    "copied open baselines/judgments differ for the same locked prompt"
                )
    main_approved_setups = _expand_main_approved_setups(
        verified_lock, normalized_artifacts
    )
    random_controls, random_approved_setups, random_paths = _verify_random_controls(
        repo_root,
        verified_lock,
        model_records,
        manifest["random_direction_controls"],
        stage1_lock_sha256=str(manifest["stage1_lock_sha256"]),
        runner_parent_commit=runner_code,
        main_records=normalized_artifacts,
    )
    frozen_artifact_paths.update(random_paths)

    expected_frozen_artifacts = {
        preopen_relative,
        environment_relative,
        *source_direction_paths,
        *frozen_artifact_paths,
    } - protected_names
    if frozen_manifest_names != expected_frozen_artifacts:
        raise RuntimeError(
            "stage-2 frozen artifact coverage mismatch: "
            f"missing={sorted(expected_frozen_artifacts - frozen_manifest_names)[:5]}, "
            f"extra={sorted(frozen_manifest_names - expected_frozen_artifacts)[:5]}"
        )
    changed_frozen = git_diff_paths(
        repo_root, artifact_freeze, head, tuple(sorted(frozen_manifest_names))
    )
    if changed_frozen:
        raise RuntimeError(
            "stage-2 artifacts changed after artifact_freeze_commit: "
            f"{sorted(changed_frozen)[:5]}"
        )

    required_tracked = {
        manifest_relative,
        preopen_relative,
        environment_relative,
        *protected_names,
        *source_direction_paths,
        *frozen_manifest_names,
    }
    tracked = git_tracked_paths(repo_root)
    untracked = sorted(required_tracked - tracked)
    if untracked:
        raise RuntimeError(f"stage-2 locked files are not Git-tracked: {untracked[:5]}")
    dirty = set(git_dirty_paths(repo_root))
    dirty_required = sorted(required_tracked & dirty)
    if dirty_required:
        raise RuntimeError(f"stage-2 locked files are dirty: {dirty_required[:5]}")

    manifest_sha = sha256_file(repo_root / manifest_relative)
    method_status_records = [
        {
            key: copy.deepcopy(record[key])
            for key in (
                "model_id",
                "method_id",
                "track",
                "selected_strength",
                "selected_layer",
                "validation_summary_sha256",
                "calibration_status",
                "sealed_evaluation_required",
                "winner_eligible",
                "matched_fixed_descriptive",
            )
        }
        for record in normalized_artifacts
    ]
    return VerifiedStage2._create(
        manifest_sha256=manifest_sha,
        runner_code_commit=runner_code,
        artifact_freeze_commit=artifact_freeze,
        verified_head_commit=head,
        protected_paths_sha256=sha256_json(
            sorted(protected_records, key=lambda item: item["path"])
        ),
        stage1_lock_sha256=str(manifest["stage1_lock_sha256"]),
        stage1_lock_payload_sha256=sha256_json(verified_lock),
        environment_lock_sha256=str(manifest["environment_lock_sha256"]),
        approved_setups_sha256=sha256_json(
            sorted(
                [*main_approved_setups, *random_approved_setups],
                key=lambda item: (
                    item["model_id"],
                    item["method_id"],
                    item["track"],
                    item["selected_strength"],
                    item.get("source_method_id", ""),
                ),
            )
        ),
        approved_setups=[*main_approved_setups, *random_approved_setups],
        method_status_records=method_status_records,
        random_controls_sha256=sha256_json(random_controls),
        outcome_blind_amendment=outcome_blind_amendment,
        token=_VERIFIED_STAGE2_TOKEN,
    )


def assert_stage2_ready(
    verified: VerifiedStage2 | None,
) -> VerifiedStage2:
    if not isinstance(verified, VerifiedStage2) or getattr(
        verified, "_capability_token", None
    ) is not _VERIFIED_STAGE2_TOKEN:
        raise RuntimeError("a verified stage-2 capability is required for sealed evaluation")
    return verified


def approved_setup_records(
    verified: VerifiedStage2 | None,
) -> tuple[dict[str, Any], ...]:
    """Return detached copies of the setup identities approved for sealed runs."""

    capability = assert_stage2_ready(verified)
    return tuple(copy.deepcopy(record) for record in capability._approved_setups)


def verified_method_status_records(
    verified: VerifiedStage2 | None,
) -> tuple[dict[str, Any], ...]:
    """Return detached calibration/status records, including fail-closed fixed setups."""

    capability = assert_stage2_ready(verified)
    return tuple(
        copy.deepcopy(record) for record in capability._method_status_records
    )


def assert_approved_setup(
    verified: VerifiedStage2 | None,
    *,
    repo_root: Path,
    model_id: str,
    model_revision: str,
    model_config_sha256: str,
    method_id: str,
    track: str,
    direction_path: Path,
    direction_file_sha256: str,
    direction_float32_sha256: str,
    direction_artifact_sha256: str,
    selected_strength: float,
    selected_layer: int,
    position_schedule: str,
    construction_config_sha256: str,
    calibration_summary_sha256: str,
) -> dict[str, Any]:
    """Fail unless every sealed-run setup field matches one frozen stage-2 record."""

    capability = assert_stage2_ready(verified)
    if any(
        not _valid_digest(value)
        for value in (
            model_config_sha256,
            direction_file_sha256,
            direction_float32_sha256,
            direction_artifact_sha256,
            construction_config_sha256,
            calibration_summary_sha256,
        )
    ):
        raise RuntimeError("sealed setup contains an invalid SHA-256 digest")
    if (
        isinstance(selected_strength, bool)
        or not isinstance(selected_strength, (int, float))
        or not math.isfinite(float(selected_strength))
        or float(selected_strength) < 0
    ):
        raise RuntimeError("sealed setup selected_strength must be finite and non-negative")
    if (
        isinstance(selected_layer, bool)
        or not isinstance(selected_layer, int)
        or selected_layer < 0
    ):
        raise RuntimeError("sealed setup selected_layer must be a non-negative integer")
    relative = _repo_relative_path(
        repo_root.resolve(), direction_path, field="sealed direction artifact path"
    )
    path_matches = [
        record
        for record in capability._approved_setups
        if record["direction_path"] == relative
    ]
    if not path_matches:
        raise RuntimeError(
            "sealed direction path is not approved by the stage-2 manifest"
        )
    supplied = {
        "model_id": model_id,
        "model_revision": model_revision,
        "model_config_sha256": model_config_sha256,
        "method_id": method_id,
        "track": track,
        "direction_file_sha256": direction_file_sha256,
        "direction_float32_sha256": direction_float32_sha256,
        "direction_artifact_sha256": direction_artifact_sha256,
        "selected_strength": float(selected_strength),
        "selected_layer": selected_layer,
        "position_schedule": position_schedule,
        "construction_config_sha256": construction_config_sha256,
        "validation_summary_sha256": calibration_summary_sha256,
    }
    matches = [
        record
        for record in path_matches
        if all(record.get(key) == value for key, value in supplied.items())
    ]
    if len(matches) != 1:
        mismatches = {
            key: (sorted({record.get(key) for record in path_matches}, key=str), value)
            for key, value in supplied.items()
            if all(record.get(key) != value for record in path_matches)
        }
        raise RuntimeError(f"sealed setup differs from stage-2 approval: {mismatches}")
    approved = matches[0]
    mismatches = {
        key: (approved.get(key), value)
        for key, value in supplied.items()
        if approved.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"sealed setup differs from stage-2 approval: {mismatches}")
    if approved.get("sealed_evaluation_required") is not True:
        raise RuntimeError("stage-2 approval marks this setup as not evaluable on sealed data")
    observed_file_sha256 = sha256_file(repo_root.resolve() / relative)
    if observed_file_sha256 != approved["direction_file_sha256"]:
        raise RuntimeError("sealed direction artifact changed after stage-2 verification")
    return dict(approved)
