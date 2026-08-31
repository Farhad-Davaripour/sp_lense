#!/usr/bin/env python3
"""Fail-closed runner for the locked FACFS Stage-G geometry screen."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from sp_lense.facfs_stage_g import (
    NAMESPACE,
    build_tokenized_manifests,
    canonical_json_bytes,
    canonical_sha256,
    plain,
    validate_source,
    verify_identity_hash,
    with_identity_hash,
)
from sp_lense.facfs_stage_g_runtime import (
    analyze_stage_g,
    capture_identifier_objective,
    capture_option_free_objective,
    effect_certificate,
    model_parameters_disabled,
)
from sp_lense.gradient_specificity_v3 import tensor_float32_sha256

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/mnt/c/Users/farha/repos/sp_lense")
LOCK_PATH = ROOT / "configs" / "facfs_stage_g_v1_lock.json"
SOURCE_PATH = ROOT / "data" / "facfs_stage_g_v1_scenarios.json"
EXCLUSIONS_PATH = ROOT / "configs" / "facfs_stage_g_v1_exclusions.json"
OPERATIONS_PATH = ROOT / "configs" / "facfs_stage_g_v1_operations.json"
TOKEN_PATH = ROOT / "configs" / "facfs_stage_g_v1_token_certificate.json"
DIRECTION_PATH = ROOT / "configs" / "facfs_stage_g_v1_direction_certificate.json"
POWER_PATH = ROOT / "configs" / "facfs_stage_g_v1_power.json"
AUTHOR_PATH = ROOT / "scripts" / "facfs_stage_g_author.py"
ARTIFACT_ROOT = ROOT / "artifacts" / "facfs" / "stage_g_v1"
RESULT_ROOT = ROOT / "results" / "facfs" / "stage_g_v1"
RECEIPT_PATH = ARTIFACT_ROOT / "preflight_receipt.json"
ATTEMPT_ROOT = ARTIFACT_ROOT / "attempt_0001"
CHUNK_ROOT = ATTEMPT_ROOT / "chunks"
LEDGER_PATH = ATTEMPT_ROOT / "compute_ledger.jsonl"
OBJECTIVE_ROWS_PATH = ATTEMPT_ROOT / "objective_records.jsonl"
SEQUENCE_ROWS_PATH = ATTEMPT_ROOT / "sequence_records.jsonl"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
HF_HOME = "/mnt/c/Users/farha/.cache/huggingface"
PYTHON_EXECUTABLE = "/home/farhad/sp_lense/.venv/bin/python"
RAW_DIRECTION_PATH = (
    ROOT
    / "artifacts"
    / "steering_comparison"
    / "one_day_local"
    / "qwen35_08b"
    / "directions"
    / "gradient.json"
)
OFFLINE_ENVIRONMENT = {
    "HF_HOME": HF_HOME,
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


class IntegrityError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    _deny_protected_path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{_relative(path)} must contain one JSON object")
    return value


def git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _deny_protected_path(path: Path) -> None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve()).as_posix().casefold()
    except ValueError:
        return
    exact = {
        "data/ckes_sealed.json",
        "data/ckes_v2_sealed.json",
        "results/steering_comparison/equal_efficacy_08b/untouched_test.jsonl",
        "results/steering_comparison/equal_efficacy_08b/report.json",
        "results/steering_comparison/equal_efficacy_08b/report.md",
    }
    if (
        relative in exact
        or ("steering_comparison/one_day_local/" in relative and "sealed" in relative)
        or (relative.startswith("artifacts/steering_comparison/") and "sealed" in relative)
    ):
        raise IntegrityError(f"hard-denied path access: {relative}")


def _author_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "sp_lense_facfs_stage_g_author", AUTHOR_PATH
    )
    if specification is None or specification.loader is None:
        raise IntegrityError("cannot load the locked Stage-G authoring verifier")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _check_path_integrity(paths: list[Path]) -> None:
    root = ROOT.resolve()
    if root != EXPECTED_ROOT:
        raise IntegrityError(f"authoritative root differs: {root}")
    for path in paths:
        resolved = path.resolve(strict=path.exists())
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise IntegrityError(f"path escapes the authoritative root: {path}") from error
        current = path
        while current != root and current != current.parent:
            if current.is_symlink():
                raise IntegrityError(f"symlink is forbidden in locked path: {path}")
            current = current.parent
        if path.name.casefold() != path.name.lower().casefold():
            raise AssertionError("unreachable case-fold check")


def _check_git(lock: dict[str, Any]) -> dict[str, str]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "@{upstream}")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    divergence = git("rev-list", "--left-right", "--count", "HEAD...@{upstream}")
    remote_ref = f"refs/heads/{branch}"
    remote_lines = git("ls-remote", "origin", remote_ref).splitlines()
    remote = remote_lines[0].split()[0] if len(remote_lines) == 1 else ""
    if branch != lock["expected_branch"]:
        raise IntegrityError(f"branch differs: {branch}")
    if status:
        raise IntegrityError("tracked or untracked worktree state is not clean")
    if head != upstream or head != remote or divergence.split() != ["0", "0"]:
        raise IntegrityError("HEAD, upstream, remote, or divergence differs")
    for commit in lock["required_ancestor_commits"].values():
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(commit), head],
            cwd=ROOT,
            check=False,
        )
        if completed.returncode != 0:
            raise IntegrityError(f"required ancestor is absent: {commit}")
    return {
        "branch": branch,
        "head": head,
        "upstream": upstream,
        "remote": remote,
        "divergence": divergence,
    }


def _check_locked_files(lock: dict[str, Any]) -> None:
    for row in lock["locked_files"]:
        path = (ROOT / str(row["path"])).resolve()
        path.relative_to(ROOT.resolve())
        if not path.is_file() or file_sha256(path) != row["file_sha256"]:
            raise IntegrityError(f"locked file differs: {row['path']}")


def _check_environment(lock: dict[str, Any]) -> dict[str, Any]:
    expected = lock["environment"]
    if Path(sys.executable) != Path(PYTHON_EXECUTABLE):
        raise IntegrityError(f"Python executable differs: {sys.executable}")
    observed_python = ".".join(str(value) for value in sys.version_info[:3])
    if observed_python != expected["python"]:
        raise IntegrityError(f"Python version differs: {observed_python}")
    observed_packages = {}
    for package, version in expected["packages"].items():
        observed = importlib.metadata.version(package)
        observed_packages[package] = observed
        if observed != version:
            raise IntegrityError(f"package version differs for {package}: {observed}")
    os_release = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os_release[key] = value.strip().strip('"')
    kernel_release = platform.release()
    if (
        platform.system() != "Linux"
        or platform.machine() not in {"x86_64", "AMD64"}
        or os_release.get("ID") != "ubuntu"
        or os_release.get("VERSION_ID") != "26.04"
        or "microsoft" not in kernel_release.casefold()
        or getpass.getuser() != expected["linux_user"]
    ):
        raise IntegrityError("WSL Linux platform differs")
    smart_app = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "(Get-MpComputerStatus).SmartAppControlState",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if smart_app != expected["windows_smart_app_control_required"]:
        raise IntegrityError(f"Windows Smart App Control differs: {smart_app}")
    return {
        "python": observed_python,
        "python_executable": str(Path(sys.executable)),
        "packages": observed_packages,
        "os_release": {"ID": os_release.get("ID"), "VERSION_ID": os_release.get("VERSION_ID")},
        "kernel_release": kernel_release,
        "linux_user": getpass.getuser(),
        "windows_smart_app_control": smart_app,
    }


def _check_inherited(lock: dict[str, Any]) -> dict[str, Any]:
    inherited = lock["inherited_equal_efficacy"]
    checks = (
        (inherited["lock_path"], inherited["lock_file_sha256"]),
        (
            inherited["calibration_summary_path"],
            inherited["calibration_summary_file_sha256"],
        ),
        (
            inherited["calibration_freeze_path"],
            inherited["calibration_freeze_file_sha256"],
        ),
    )
    for relative, expected in checks:
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise IntegrityError(f"inherited artifact differs: {relative}")
    result_root = ROOT / "results" / "steering_comparison" / "equal_efficacy_08b"
    for name, expected in inherited["calibration_rows"].items():
        path = result_root / name
        if not path.is_file() or file_sha256(path) != expected:
            raise IntegrityError(f"inherited calibration rows differ: {name}")
    freeze = load_json(ROOT / inherited["calibration_freeze_path"])
    if freeze.get("core_all_eligible") is not False:
        raise IntegrityError("equal-efficacy freeze no longer records a core no-go")
    if freeze.get("attestation", {}).get("untouched_test_outcomes_viewed") is not False:
        raise IntegrityError("equal-efficacy untouched-test attestation differs")
    absent = []
    for relative in inherited["absent_paths"]:
        path = ROOT / relative
        try:
            os.lstat(path)
        except FileNotFoundError:
            pass
        else:
            raise IntegrityError(f"old untouched outcome/report path exists: {relative}")
        if git("ls-tree", "-r", "--name-only", "HEAD", "--", relative):
            raise IntegrityError(f"old untouched outcome/report path is tracked: {relative}")
        absent.append(relative)
    return {
        "freeze_core_all_eligible": False,
        "untouched_test_outcomes_viewed": False,
        "absent_paths": absent,
    }


def _check_source_and_manifests(lock: dict[str, Any]) -> dict[str, Any]:
    payload = load_json(SOURCE_PATH)
    validate_source(payload)
    exclusions = load_json(EXCLUSIONS_PATH)
    operations = load_json(OPERATIONS_PATH)
    token = load_json(TOKEN_PATH)
    direction = load_json(DIRECTION_PATH)
    power = load_json(POWER_PATH)
    verify_identity_hash(exclusions, "exclusions_sha256")
    verify_identity_hash(operations, "operations_sha256")
    verify_identity_hash(token, "token_certificate_sha256")
    verify_identity_hash(direction, "direction_certificate_sha256")
    verify_identity_hash(power, "power_report_sha256")
    if not exclusions.get("all_collision_gates_passed") or any(
        exclusions["collision_counts"].values()
    ):
        raise IntegrityError("source collision gates do not all pass")
    author = _author_module()
    rebuilt_exclusions = author.build_exclusions(payload)
    if canonical_json_bytes(rebuilt_exclusions) != canonical_json_bytes(exclusions):
        raise IntegrityError("historical exclusion manifest does not reproduce")

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=f"{HF_HOME}/hub",
        local_files_only=True,
        trust_remote_code=False,
    )
    rebuilt_operations, rebuilt_token = build_tokenized_manifests(
        tokenizer, torch, payload
    )
    if canonical_json_bytes(rebuilt_operations) != canonical_json_bytes(operations):
        raise IntegrityError("operation ledger does not reproduce")
    if canonical_json_bytes(rebuilt_token) != canonical_json_bytes(token):
        raise IntegrityError("token certificate does not reproduce")
    rebuilt_direction = author.build_direction_certificate()
    if canonical_json_bytes(rebuilt_direction) != canonical_json_bytes(direction):
        raise IntegrityError("direction certificate does not reproduce")
    if canonical_json_bytes(author.build_power_report()) != canonical_json_bytes(power):
        raise IntegrityError("power report does not reproduce")
    if operations["totals"] != lock["compute_ceiling"]:
        raise IntegrityError("operation totals differ from the compute ceiling")
    if lock["operations_identity"] != {
        "file_sha256": file_sha256(OPERATIONS_PATH),
        "operations_sha256": operations["operations_sha256"],
    }:
        raise IntegrityError("operation identity differs from the lock")
    if lock["token_certificate_identity"] != {
        "file_sha256": file_sha256(TOKEN_PATH),
        "token_certificate_sha256": token["token_certificate_sha256"],
    }:
        raise IntegrityError("token certificate identity differs from the lock")
    if lock["exclusions_identity"] != {
        "file_sha256": file_sha256(EXCLUSIONS_PATH),
        "exclusions_sha256": exclusions["exclusions_sha256"],
        "all_collision_gates_passed": True,
    }:
        raise IntegrityError("exclusion identity differs from the lock")
    return {
        "scenario_count": 11,
        "objective_count": len(operations["operations"]),
        "operations_sha256": operations["operations_sha256"],
        "token_certificate_sha256": token["token_certificate_sha256"],
        "exclusions_sha256": exclusions["exclusions_sha256"],
        "direction_certificate_sha256": direction["direction_certificate_sha256"],
        "tokenizer_loaded": True,
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
    }


def _verify_preflight(*, allow_receipt: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    if not LOCK_PATH.is_file():
        raise IntegrityError("Stage-G lock is absent")
    lock = load_json(LOCK_PATH)
    verify_identity_hash(lock, "lock_identity_sha256")
    if (
        lock.get("namespace") != NAMESPACE
        or lock.get("status")
        != "prospectively_locked_before_any_stage_g_model_load_or_forward"
    ):
        raise IntegrityError("Stage-G lock status differs")
    locked_paths = [ROOT / str(row["path"]) for row in lock["locked_files"]]
    _check_path_integrity([LOCK_PATH, *locked_paths])
    git_state = _check_git(lock)
    _check_locked_files(lock)
    environment = _check_environment(lock)
    inherited = _check_inherited(lock)
    manifests = _check_source_and_manifests(lock)
    if allow_receipt:
        observed = sorted(
            path.relative_to(ARTIFACT_ROOT).as_posix()
            for path in ARTIFACT_ROOT.rglob("*")
            if path.is_file()
        )
        if observed != ["preflight_receipt.json"] or RESULT_ROOT.exists():
            raise IntegrityError("unexpected pre-existing FACFS outputs are present")
    elif ARTIFACT_ROOT.exists() or RESULT_ROOT.exists():
        raise IntegrityError("FACFS output roots must be absent before preflight")
    return lock, {
        "git": git_state,
        "environment": environment,
        "inherited": inherited,
        "manifests": manifests,
    }


def _write_new_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable {_relative(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(
        plain(value), indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    _write_new_bytes(path, payload)


def _append_json_line(handle: Any, value: dict[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())


def run_preflight() -> None:
    for key, value in OFFLINE_ENVIRONMENT.items():
        os.environ[key] = value
    lock, checks = _verify_preflight(allow_receipt=False)
    receipt = with_identity_hash(
        {
            "schema_version": "sp_lense.facfs.stage_g.preflight_receipt.v1",
            "namespace": NAMESPACE,
            "status": "passed",
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "lock_file_sha256": file_sha256(LOCK_PATH),
            "lock_commit": checks["git"]["head"],
            "checks": checks,
            "model_loaded": False,
            "model_forwards": 0,
            "model_backwards": 0,
            "generated_tokens": 0,
            "finite_intervention_calls": 0,
            "windows_security_changed": False,
        },
        "receipt_sha256",
    )
    _write_new_json(RECEIPT_PATH, receipt)
    print(
        json.dumps(
            {
                "preflight": "passed",
                "receipt_sha256": receipt["receipt_sha256"],
                "model_loaded": False,
                "model_forwards": 0,
                "model_backwards": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )


class ComputeLedger:
    def __init__(self, operations: list[dict[str, Any]]) -> None:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            LEDGER_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        self.handle = os.fdopen(descriptor, "wb", buffering=0)
        self.expected = [
            (str(row["objective_id"]), event)
            for row in operations
            for event in row["ledger_events"]
        ]
        self.index = 0
        self.forward_count = 0
        self.backward_count = 0
        self.prior_sha256 = "0" * 64
        self.objective_id = ""

    def set_objective(self, objective_id: str) -> None:
        self.objective_id = objective_id

    def reserve(self, kind: str, role: str) -> None:
        if self.index >= len(self.expected):
            raise IntegrityError("compute ledger exceeded its ceiling")
        expected_objective, expected = self.expected[self.index]
        if expected_objective != self.objective_id or expected["kind"] != kind:
            raise IntegrityError("compute ledger event order differs from the lock")
        if "role" in expected and expected["role"] != role:
            raise IntegrityError("compute ledger event role differs from the lock")
        self.index += 1
        if kind == "forward":
            self.forward_count += 1
        elif kind == "backward":
            self.backward_count += 1
        else:
            raise IntegrityError("unknown compute ledger event kind")
        body = {
            "schema_version": "sp_lense.facfs.stage_g.compute_event.v1",
            "ledger_index": self.index,
            "event_id": expected["event_id"],
            "objective_id": self.objective_id,
            "kind": kind,
            "role": role,
            "prior_event_sha256": self.prior_sha256,
            "cumulative_forwards": self.forward_count,
            "cumulative_backwards": self.backward_count,
        }
        record = with_identity_hash(body, "event_sha256")
        _append_json_line(self.handle, record)
        self.prior_sha256 = str(record["event_sha256"])

    def close(self) -> None:
        self.handle.close()


def _load_deployed_direction(torch: Any, lock: dict[str, Any]) -> Any:
    record = load_json(RAW_DIRECTION_PATH)
    raw = torch.tensor(record["direction"], dtype=torch.float32).contiguous()
    if tensor_float32_sha256(raw) != lock["direction"][
        "raw_direction_float32_sha256"
    ]:
        raise IntegrityError("raw direction hash differs at capture")
    deployed = (raw / raw.norm().clamp_min(1e-12)).float().contiguous()
    if tensor_float32_sha256(deployed) != lock["direction"][
        "deployed_direction_float32_sha256"
    ]:
        raise IntegrityError("deployed direction hash differs at capture")
    if (
        abs(float(deployed.norm().item()) - 1.0)
        > lock["thresholds"]["deployed_direction_norm_tolerance"]
    ):
        raise IntegrityError("deployed direction norm differs at capture")
    return deployed


def _save_chunk(
    objective_id: str,
    tensors: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    from safetensors.torch import save_file

    tensor_path = CHUNK_ROOT / f"{objective_id}.safetensors"
    metadata_path = CHUNK_ROOT / f"{objective_id}.json"
    if tensor_path.exists() or metadata_path.exists():
        raise FileExistsError(f"chunk already exists: {objective_id}")
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary chunk already exists: {objective_id}")
    save_file({key: value.cpu().contiguous() for key, value in tensors.items()}, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, tensor_path)
    metadata = with_identity_hash(
        {
            **metadata,
            "tensor_path": _relative(tensor_path),
            "tensor_file_sha256": file_sha256(tensor_path),
            "tensor_keys": sorted(tensors),
            "tensor_hashes": {
                key: tensor_float32_sha256(value) for key, value in tensors.items()
            },
        },
        "chunk_metadata_sha256",
    )
    _write_new_json(metadata_path, metadata)
    return tensor_path, metadata_path


def _sequence_records(
    operation: dict[str, Any], metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    common = {
        "schema_version": "sp_lense.facfs.stage_g.sequence_record.v1",
        "objective_id": operation["objective_id"],
        "scenario_id": operation["scenario_id"],
        "form_kind": operation["form_kind"],
        "prompt_sha256": operation["prompt_sha256"],
        "prompt_token_ids_sha256": operation["prompt_token_ids_sha256"],
    }
    if operation["form_kind"] == "opaque_identifier":
        return [
            {
                **common,
                "sequence_id": operation["sequence_id"],
                "sequence_role": "opaque_next_token",
                "prompt_token_count": operation["prompt_token_count"],
                "preserve_token_id": operation["preserve_token_id"],
                "comply_token_id": operation["comply_token_id"],
                "preserve_logit": metadata["preserve_logit"],
                "comply_logit": metadata["comply_logit"],
                "full_next_token_logits_float32_sha256": metadata[
                    "full_next_token_logits_float32_sha256"
                ],
            }
        ]
    rows = []
    for semantic in ("preserve", "comply"):
        encoding = operation["completion_encodings"][semantic]
        audit = metadata[f"{semantic}_capture"]
        rows.append(
            {
                **common,
                "sequence_id": encoding["sequence_id"],
                "sequence_role": semantic,
                "full_token_count": encoding["full_token_count"],
                "full_token_ids_sha256": encoding["full_token_ids_sha256"],
                "content_token_count": encoding["content_token_count"],
                "content_token_ids_sha256": encoding["content_token_ids_sha256"],
                "mean_content_log_probability": metadata[
                    f"{semantic}_mean_content_log_probability"
                ],
                "content_logprob_vector_float32_sha256": audit[
                    "content_logprob_vector_float32_sha256"
                ],
            }
        )
    return rows


def _runtime_metadata(backend: Any, lock: dict[str, Any]) -> dict[str, Any]:
    metadata = backend.metadata()
    tokenizer = backend.model.tokenizer
    template = getattr(tokenizer, "chat_template", None)
    template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
    vocab = int(getattr(backend.model.cfg, "d_vocab", len(tokenizer)))
    expected = lock["model"]
    if (
        metadata["device"] != expected["device"]
        or metadata["dtype"] != expected["dtype"]
        or metadata["model_id"] != expected["model_id"]
        or metadata["model_revision"] != expected["revision"]
        or int(metadata["model_layers"]) != expected["blocks"]
        or int(metadata["d_model"]) != expected["d_model"]
        or vocab != expected["vocabulary_size"]
        or template_hash != expected["chat_template_sha256"]
        or backend.torch.get_num_threads() != expected["torch_num_threads"]
        or backend.torch.get_num_interop_threads()
        != expected["torch_num_interop_threads"]
    ):
        raise IntegrityError(f"loaded runtime differs: {metadata}, vocab={vocab}")
    import sp_lense
    import sp_lense.facfs_stage_g
    import sp_lense.facfs_stage_g_runtime

    origins = {
        "sp_lense": Path(sp_lense.__file__).resolve(),
        "facfs_stage_g": Path(sp_lense.facfs_stage_g.__file__).resolve(),
        "facfs_stage_g_runtime": Path(
            sp_lense.facfs_stage_g_runtime.__file__
        ).resolve(),
    }
    for name, path in origins.items():
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as error:
            raise IntegrityError(f"{name} imported from another checkout: {path}") from error
    return {
        **metadata,
        "vocabulary_size": vocab,
        "chat_template_sha256": template_hash,
        "import_origins": {key: str(value) for key, value in origins.items()},
        "torch_num_threads": backend.torch.get_num_threads(),
        "torch_num_interop_threads": backend.torch.get_num_interop_threads(),
    }


def _load_chunks(operations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from safetensors.torch import load_file

    chunks = {}
    for operation in operations:
        objective_id = str(operation["objective_id"])
        tensor_path = CHUNK_ROOT / f"{objective_id}.safetensors"
        metadata_path = CHUNK_ROOT / f"{objective_id}.json"
        metadata = load_json(metadata_path)
        verify_identity_hash(metadata, "chunk_metadata_sha256")
        if metadata["tensor_file_sha256"] != file_sha256(tensor_path):
            raise IntegrityError(f"chunk tensor hash differs: {objective_id}")
        tensors = load_file(tensor_path, device="cpu")
        if sorted(tensors) != metadata["tensor_keys"]:
            raise IntegrityError(f"chunk tensor keys differ: {objective_id}")
        for key, tensor in tensors.items():
            if tensor_float32_sha256(tensor) != metadata["tensor_hashes"][key]:
                raise IntegrityError(f"chunk tensor value differs: {objective_id}:{key}")
        chunks[objective_id] = {"metadata": metadata, "tensors": tensors}
    return chunks


def _report_markdown(summary: dict[str, Any]) -> str:
    status = summary["status"]
    lines = [
        "# FACFS Stage-G Geometry Screen",
        "",
        f"Status: **{status}**",
        "",
        "This was an unsteered local-gradient screen at block 10, final prompt token.",
        "No finite intervention, shield, gate, dose, generation, external API, or judge was used.",
        "",
        f"Complete scenarios passed: {summary['scenario_successes']} / {summary['scenario_count']} (all 11 required).",
        "",
        "| Scenario | SP opaque | Option-free | Alignment | Overall |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["scenario_results"]:
        lines.append(
            "| {scenario_id} | {sp} | {free} | {alignment} | {overall} |".format(
                scenario_id=row["scenario_id"],
                sp="pass" if row["all_sp_opaque_effects_passed"] else "fail",
                free="pass" if row["all_option_free_effects_passed"] else "fail",
                alignment="pass" if row["all_alignments_passed"] else "fail",
                overall="pass" if row["scenario_passed"] else "fail",
            )
        )
    lines.extend(
        [
            "",
            (
                "A pass authorizes only the authoring of a separate, prospectively locked FACFS study. "
                "It does not itself authorize or validate a finite intervention."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _tree_manifest(paths: list[Path]) -> dict[str, Any]:
    rows = [
        {
            "path": _relative(path),
            "byte_size": path.stat().st_size,
            "file_sha256": file_sha256(path),
        }
        for path in sorted(paths, key=lambda item: _relative(item))
    ]
    return with_identity_hash(
        {
            "schema_version": "sp_lense.facfs.stage_g.output_inventory.v1",
            "files": rows,
            "file_count": len(rows),
        },
        "inventory_sha256",
    )


def run_capture() -> None:
    for key, value in OFFLINE_ENVIRONMENT.items():
        os.environ[key] = value
    lock, checks = _verify_preflight(allow_receipt=True)
    receipt = load_json(RECEIPT_PATH)
    verify_identity_hash(receipt, "receipt_sha256")
    if (
        receipt["status"] != "passed"
        or receipt["lock_identity_sha256"] != lock["lock_identity_sha256"]
        or receipt["lock_commit"] != checks["git"]["head"]
        or receipt["model_loaded"] is not False
        or receipt["model_forwards"] != 0
        or receipt["model_backwards"] != 0
    ):
        raise IntegrityError("preflight receipt differs from the capture lock")
    if ATTEMPT_ROOT.exists() or RESULT_ROOT.exists():
        raise IntegrityError("attempt_0001 is already present; resume/retry is forbidden")

    import torch

    from sp_lense.backend import ResearchBackend
    from sp_lense.config import load_config

    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    direction = _load_deployed_direction(torch, lock)
    backend = ResearchBackend.load(
        load_config(ROOT / lock["model"]["config_path"]), with_lens=False
    )
    runtime = _runtime_metadata(backend, lock)
    operations_manifest = load_json(OPERATIONS_PATH)
    operations = operations_manifest["operations"]
    ATTEMPT_ROOT.mkdir(parents=True, exist_ok=False)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=False)
    _write_new_json(
        ATTEMPT_ROOT / "attempt_started.json",
        with_identity_hash(
            {
                "schema_version": "sp_lense.facfs.stage_g.attempt_started.v1",
                "attempt": "attempt_0001",
                "lock_identity_sha256": lock["lock_identity_sha256"],
                "preflight_receipt_sha256": receipt["receipt_sha256"],
                "lock_commit": checks["git"]["head"],
                "runtime": runtime,
                "planned_totals": operations_manifest["totals"],
                "state": "started",
            },
            "attempt_started_sha256",
        ),
    )
    ledger = ComputeLedger(operations)
    objective_descriptor = os.open(
        OBJECTIVE_ROWS_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    sequence_descriptor = os.open(
        SEQUENCE_ROWS_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    objective_handle = os.fdopen(objective_descriptor, "wb", buffering=0)
    sequence_handle = os.fdopen(sequence_descriptor, "wb", buffering=0)
    captured_objectives = 0
    captured_sequences = 0
    started = time.perf_counter()
    try:
        with model_parameters_disabled(backend) as parameter_audit:
            for operation in operations:
                objective_id = str(operation["objective_id"])
                ledger.set_objective(objective_id)
                if operation["form_kind"] == "opaque_identifier":
                    tensors, metadata = capture_identifier_objective(
                        backend, operation, reserve=ledger.reserve
                    )
                    gradient = tensors["s32"]
                    margin = float(lock["thresholds"]["mu_id"])
                else:
                    tensors, metadata = capture_option_free_objective(
                        backend, operation, reserve=ledger.reserve
                    )
                    gradient = tensors["s_free32"]
                    margin = float(lock["thresholds"]["mu_free"])
                q32, certificate = effect_certificate(
                    torch,
                    tensors["h32"],
                    gradient,
                    direction,
                    margin=margin,
                    gamma_1024=float(lock["thresholds"]["gamma_1024"]),
                    reduction_tolerance=float(
                        lock["thresholds"]["reduction_tolerance"]
                    ),
                    zero_atol=float(lock["thresholds"]["float32_zero_atol"]),
                )
                tensors["g32"] = q32
                metadata = {
                    **metadata,
                    "operation_ordinal": operation["ordinal"],
                    "operation_identity_sha256": canonical_sha256(operation),
                    "effect_certificate": certificate,
                    "ledger_event_ids": [
                        event["event_id"] for event in operation["ledger_events"]
                    ],
                    "ledger_cumulative_after": {
                        "forwards": ledger.forward_count,
                        "backwards": ledger.backward_count,
                        "events": ledger.index,
                    },
                    "parameter_audit": parameter_audit,
                    "direction_float32_sha256": tensor_float32_sha256(direction),
                    "factors": {
                        key: operation[key]
                        for key in (
                            "scenario_id",
                            "condition",
                            "assignment",
                            "alphabet_id",
                            "alphabet_index",
                            "mapping",
                            "order",
                            "preserve_first",
                        )
                        if key in operation
                    },
                }
                _, metadata_path = _save_chunk(
                    objective_id, tensors, metadata
                )
                stored_metadata = load_json(metadata_path)
                _append_json_line(
                    objective_handle,
                    {
                        "schema_version": "sp_lense.facfs.stage_g.objective_record.v1",
                        "objective_id": objective_id,
                        "operation_ordinal": operation["ordinal"],
                        "chunk_metadata_path": _relative(metadata_path),
                        "chunk_metadata_sha256": stored_metadata[
                            "chunk_metadata_sha256"
                        ],
                        "effect_certificate": certificate,
                    },
                )
                records = _sequence_records(operation, metadata)
                for record in records:
                    _append_json_line(sequence_handle, record)
                captured_objectives += 1
                captured_sequences += len(records)
                if captured_objectives % 25 == 0 or captured_objectives == len(
                    operations
                ):
                    print(
                        json.dumps(
                            {
                                "progress_objectives": captured_objectives,
                                "total_objectives": len(operations),
                                "reserved_forwards": ledger.forward_count,
                                "reserved_backwards": ledger.backward_count,
                                "scientific_values_withheld": True,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        objective_handle.close()
        sequence_handle.close()
        ledger.close()
        expected = operations_manifest["totals"]
        if (
            captured_objectives != expected["total_objectives"]
            or captured_sequences != expected["total_scored_sequence_items"]
            or ledger.forward_count != expected["physical_forward_invocations"]
            or ledger.backward_count != expected["physical_backward_invocations"]
            or ledger.index != expected["hash_chained_ledger_events"]
        ):
            raise IntegrityError("realized compute or output counts differ from the lock")
        chunks = _load_chunks(operations)
        summary, decomposition, analysis_audit = analyze_stage_g(
            torch,
            direction,
            operations,
            chunks,
            lock["thresholds"],
        )
        RESULT_ROOT.mkdir(parents=True, exist_ok=False)
        _write_new_json(
            RESULT_ROOT / "summary.json",
            with_identity_hash(summary, "summary_sha256"),
        )
        _write_new_json(
            RESULT_ROOT / "walsh_decomposition.json",
            with_identity_hash(decomposition, "decomposition_sha256"),
        )
        _write_new_json(
            RESULT_ROOT / "analysis_audit.json",
            with_identity_hash(analysis_audit, "analysis_audit_sha256"),
        )
        _write_new_bytes(
            RESULT_ROOT / "REPORT.md", _report_markdown(summary).encode("utf-8")
        )
        realized = with_identity_hash(
            {
                "schema_version": "sp_lense.facfs.stage_g.realized_ledger.v1",
                "planned": expected,
                "realized": {
                    "total_objectives": captured_objectives,
                    "total_scored_sequence_items": captured_sequences,
                    "physical_forward_invocations": ledger.forward_count,
                    "physical_backward_invocations": ledger.backward_count,
                    "hash_chained_ledger_events": ledger.index,
                    "generated_tokens": 0,
                    "finite_intervention_calls": 0,
                },
                "final_compute_event_sha256": ledger.prior_sha256,
                "elapsed_seconds": time.perf_counter() - started,
                "counts_match_exactly": True,
            },
            "realized_ledger_sha256",
        )
        _write_new_json(ATTEMPT_ROOT / "realized_ledger.json", realized)
        inventory_paths = [
            path
            for root in (ARTIFACT_ROOT, RESULT_ROOT)
            for path in root.rglob("*")
            if path.is_file()
            and path.name not in {"output_inventory.json", "attempt_complete.json"}
        ]
        inventory = _tree_manifest(inventory_paths)
        _write_new_json(ARTIFACT_ROOT / "output_inventory.json", inventory)
        complete = with_identity_hash(
            {
                "schema_version": "sp_lense.facfs.stage_g.attempt_complete.v1",
                "attempt": "attempt_0001",
                "state": "complete",
                "lock_identity_sha256": lock["lock_identity_sha256"],
                "preflight_receipt_sha256": receipt["receipt_sha256"],
                "final_compute_event_sha256": ledger.prior_sha256,
                "output_inventory_sha256": inventory["inventory_sha256"],
                "scientific_outcome_unopened_until_commit_and_push": True,
            },
            "attempt_complete_sha256",
        )
        _write_new_json(ATTEMPT_ROOT / "attempt_complete.json", complete)
        print(
            json.dumps(
                {
                    "capture": "complete",
                    "objectives": captured_objectives,
                    "sequence_items": captured_sequences,
                    "forwards": ledger.forward_count,
                    "backwards": ledger.backward_count,
                    "output_inventory_sha256": inventory["inventory_sha256"],
                    "scientific_values_withheld_until_commit_and_push": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException as error:
        for handle in (objective_handle, sequence_handle):
            if not handle.closed:
                handle.close()
        if not ledger.handle.closed:
            ledger.close()
        if ATTEMPT_ROOT.exists() and not (
            ATTEMPT_ROOT / "attempt_failed.json"
        ).exists():
            _write_new_json(
                ATTEMPT_ROOT / "attempt_failed.json",
                with_identity_hash(
                    {
                        "schema_version": "sp_lense.facfs.stage_g.attempt_failed.v1",
                        "attempt": "attempt_0001",
                        "state": "failed_consumed_no_resume_no_retry",
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                        "reserved_forwards": ledger.forward_count,
                        "reserved_backwards": ledger.backward_count,
                        "captured_objectives": captured_objectives,
                        "captured_sequences": captured_sequences,
                    },
                    "attempt_failed_sha256",
                ),
            )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "capture-stage-g"))
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.command == "preflight":
        run_preflight()
    else:
        run_capture()
    return 0


if __name__ == "__main__":
    sys.exit(main())
