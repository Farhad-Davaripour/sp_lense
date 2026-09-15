"""Optimization-safe verification utilities shared by the supported reproduction commands."""

import hashlib
import json
from pathlib import Path


class VerificationError(RuntimeError):
    """An artifact or numerical reproduction check failed."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def verify_manifest(root, manifest_path=None):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path) if manifest_path else root / "SHA256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    require(isinstance(manifest, dict) and bool(manifest), "Integrity manifest must be nonempty")
    for name, expected in manifest.items():
        require(isinstance(name, str), "Artifact name must be a string")
        target = (root / name).resolve()
        require(target.is_relative_to(root), f"Artifact escapes root: {name}")
        require(isinstance(expected, str) and len(expected) == 64, f"Invalid digest: {name}")
        try:
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc:
            raise VerificationError(f"Cannot read artifact {name}: {exc}") from exc
        require(
            actual == expected, f"Integrity mismatch: {name}; expected {expected}, got {actual}"
        )
    return len(manifest)


def compare_numeric(actual, expected, label, tolerance=1e-7):
    import numpy as np

    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        actual.shape == expected.shape,
        f"{label}: shape mismatch {actual.shape} != {expected.shape}",
    )
    require(actual.size > 0, f"{label}: empty numerical result")
    require(
        bool(np.isfinite(actual).all() and np.isfinite(expected).all()),
        f"{label}: non-finite values",
    )
    difference = float(np.max(np.abs(actual - expected)))
    require(
        difference < tolerance,
        f"{label}: maximum difference {difference} exceeds tolerance {tolerance}",
    )
