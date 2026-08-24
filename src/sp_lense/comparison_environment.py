"""Capture and verify the exact local runtime used by the comparison study."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .comparison_provenance import sha256_file

ENVIRONMENT_SCHEMA = "sp_lense.steering_comparison.environment.v1"
DEFAULT_PACKAGES = (
    "torch",
    "transformers",
    "transformer-lens",
    "numpy",
    "safetensors",
    "tokenizers",
)
DETERMINISM_ENVIRONMENT_KEYS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
)


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def capture_environment_record(
    *,
    stage1_lock_path: Path,
    lock: Mapping[str, Any],
    package_names: Sequence[str] = DEFAULT_PACKAGES,
) -> dict[str, Any]:
    """Return a deterministic record of software, hardware, and runtime contracts.

    Only an allow-list of non-secret environment variables is captured.  The record
    deliberately omits timestamps so identical environments serialize identically.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the research environment requires torch
        raise RuntimeError("the comparison environment requires torch") from exc

    runtime_contracts = {
        str(model["model_id"]): {
            "revision": str(model["revision"]),
            "config_path": str(model["config"]),
            "config_sha256": str(model["config_sha256"]),
            "runtime": dict(model["runtime"]),
            "architecture": dict(model["architecture"]),
        }
        for model in lock["models"]
    }
    record: dict[str, Any] = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "study": str(lock["study"]),
        "stage1_lock_sha256": sha256_file(stage1_lock_path),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable_name": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "packages": _package_versions(tuple(package_names)),
        "torch_runtime": {
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cuda_device_count": int(torch.cuda.device_count()),
            "default_dtype": str(torch.get_default_dtype()),
        },
        "determinism_environment": {
            key: os.environ.get(key) for key in DETERMINISM_ENVIRONMENT_KEYS
        },
        "model_runtime_contracts": runtime_contracts,
    }
    record["content_sha256"] = _canonical_json_sha256(record)
    return record


def write_environment_record(path: Path, record: Mapping[str, Any]) -> None:
    validate_environment_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def validate_environment_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != ENVIRONMENT_SCHEMA:
        raise ValueError("unknown comparison environment schema")
    expected = record.get("content_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("environment record lacks content_sha256")
    payload = dict(record)
    del payload["content_sha256"]
    observed = _canonical_json_sha256(payload)
    if observed != expected:
        raise ValueError("environment record content_sha256 does not match its content")


def verify_current_environment(
    path: Path, *, stage1_lock_path: Path, lock: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail if the current process differs from the locked environment record."""

    saved = json.loads(path.read_text(encoding="utf-8"))
    validate_environment_record(saved)
    current = capture_environment_record(
        stage1_lock_path=stage1_lock_path,
        lock=lock,
        package_names=tuple(saved["packages"]),
    )
    if current != saved:
        differing = sorted(
            key for key in saved if saved.get(key) != current.get(key)
        )
        raise RuntimeError(
            "current environment differs from the stage-2 environment lock: "
            + ", ".join(differing)
        )
    return saved
