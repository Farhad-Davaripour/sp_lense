"""Optimization-safe verification utilities shared by the supported reproduction commands."""

import hashlib
import json
import math
import numbers
import re
from pathlib import Path


class VerificationError(RuntimeError):
    """An artifact or numerical reproduction check failed."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    """Read strict JSON with unambiguous keys and standard numeric constants."""

    def invalid_constant(value):
        raise VerificationError(f"Non-standard JSON constant: {value}")

    return json.loads(
        Path(path).read_text(encoding="utf-8-sig"),
        object_pairs_hook=strict_object,
        parse_constant=invalid_constant,
    )


def verify_manifest(root, manifest_path=None, *, required=None):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path) if manifest_path else root / "SHA256.json"
    require(manifest_path.stat().st_size <= 1024 * 1024, "Manifest exceeds 1 MiB")
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict) and bool(manifest), "Integrity manifest must be nonempty")
    if required is not None:
        require(set(manifest) == set(required), "Artifact inventory differs from required schema")
    for name, expected in manifest.items():
        require(isinstance(name, str), "Artifact name must be a string")
        target = (root / name).resolve()
        require(target.is_relative_to(root), f"Artifact escapes root: {name}")
        require(
            isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            f"Invalid digest: {name}",
        )
        try:
            with target.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
        except OSError as exc:
            raise VerificationError(f"Cannot read artifact {name}: {exc}") from exc
        require(
            actual == expected, f"Integrity mismatch: {name}; expected {expected}, got {actual}"
        )
    return len(manifest)


def compare_numeric(actual, expected, label: str, tolerance: float = 1e-7) -> None:
    """Compare real integer/float arrays; integer differences use exact Python arithmetic."""
    import numpy as np

    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        isinstance(tolerance, numbers.Real)
        and not isinstance(tolerance, bool)
        and math.isfinite(tolerance)
        and tolerance > 0,
        f"{label}: invalid tolerance",
    )
    require(
        actual.dtype.kind in "iuf" and expected.dtype.kind in "iuf",
        f"{label}: only real integer and floating dtypes are supported",
    )
    require(
        actual.shape == expected.shape,
        f"{label}: shape mismatch {actual.shape} != {expected.shape}",
    )
    require(actual.size > 0, f"{label}: empty numerical result")
    require(
        bool(np.isfinite(actual).all() and np.isfinite(expected).all()),
        f"{label}: non-finite values",
    )
    # Python integers avoid signed overflow and preserve uint64 precision. Mixed
    # integer/float inputs are rejected rather than silently rounded to float64.
    if actual.dtype.kind in "iu" or expected.dtype.kind in "iu":
        require(
            actual.dtype.kind in "iu" and expected.dtype.kind in "iu",
            f"{label}: mixed integer/float comparison is unsupported",
        )
        difference = max(abs(int(a) - int(b)) for a, b in zip(actual.flat, expected.flat))
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.max(
                np.abs(actual.astype(np.longdouble) - expected.astype(np.longdouble))
            )
    require(
        difference < tolerance,
        f"{label}: maximum difference {difference} exceeds tolerance {tolerance}",
    )
