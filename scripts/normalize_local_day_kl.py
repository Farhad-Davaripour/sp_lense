"""Normalize impossible tiny-negative float32 KL values in local-day staging rows.

This utility exists for one demonstrated numerical failure: the float32
full-vocabulary KL calculation can round a mathematically non-negative value
slightly below zero.  It is deliberately narrow, fail-closed, and auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "steering_comparison" / "one_day_local"
LOCK_PATH = ROOT / "configs" / "steering_comparison_local_day_lock.json"
SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_TOLERANCE = 1e-6


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _allowed_staging_path(model_key: str, split: str) -> Path:
    if model_key not in {"qwen35_08b", "qwen35_2b"}:
        raise ValueError("unsupported model key")
    if split not in {"validation", "sealed_test"}:
        raise ValueError("unsupported split")
    return RESULT_ROOT / f".{model_key}_{split}.staging.jsonl"


def normalize_staging_kl(
    *, model_key: str, split: str, tolerance: float = DEFAULT_TOLERANCE
) -> dict[str, Any]:
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a positive finite number")
    path = _allowed_staging_path(model_key, split)
    if not path.exists():
        raise FileNotFoundError(path)
    pre_sha256 = file_sha256(path)
    rows: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        if row.get("schema_version") != "sp_lense.local_day_choice_row.v1":
            raise ValueError(f"row {index} has an unexpected schema")
        if row.get("model_key") != model_key or row.get("split") != split:
            raise ValueError(f"row {index} is outside the requested run")
        value = row.get("full_vocabulary_kl_from_baseline")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"row {index} KL is not numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"row {index} KL is not finite")
        if numeric < -tolerance:
            raise ValueError(
                f"row {index} KL {numeric!r} exceeds numerical tolerance {-tolerance!r}"
            )
        if numeric < 0.0:
            corrections.append(
                {
                    "row_index_zero_based": index,
                    "unit_id": row.get("unit_id"),
                    "method": row.get("method"),
                    "condition": row.get("condition"),
                    "raw_kl": numeric,
                    "normalized_kl": 0.0,
                }
            )
            row["full_vocabulary_kl_from_baseline"] = 0.0
        rows.append(row)
    if not corrections:
        raise RuntimeError("no tiny-negative KL values were found; no rewrite performed")
    rendered = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for row in rows
    )
    temporary = path.with_name(f".{path.name}.normalize.tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    post_sha256 = file_sha256(path)
    audit = {
        "schema_version": "sp_lense.local_day_kl_numerical_correction.v1",
        "reason": "float32_rounding_of_mathematically_nonnegative_full_vocabulary_kl",
        "normalization": "values in [-tolerance, 0) are set to exactly 0.0",
        "tolerance": tolerance,
        "model_key": model_key,
        "split": split,
        "staging_path": path.relative_to(ROOT).as_posix(),
        "pre_sha256": pre_sha256,
        "post_sha256": post_sha256,
        "row_count": len(rows),
        "correction_count": len(corrections),
        "minimum_raw_kl": min(item["raw_kl"] for item in corrections),
        "maximum_raw_kl": max(item["raw_kl"] for item in corrections),
        "corrections": corrections,
        "local_day_lock_sha256": file_sha256(LOCK_PATH),
        "normalizer_sha256": file_sha256(SCRIPT_PATH),
        "normalizer_commit": _git_head(),
    }
    audit_path = RESULT_ROOT / f"{model_key}_{split}_kl_numerical_correction.json"
    audit_temporary = audit_path.with_name(f".{audit_path.name}.tmp")
    audit_temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(audit_temporary, audit_path)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("qwen35_08b", "qwen35_2b"))
    parser.add_argument("--split", required=True, choices=("validation", "sealed_test"))
    args = parser.parse_args()
    audit = normalize_staging_kl(model_key=args.model, split=args.split)
    print(
        json.dumps(
            {
                "state": "normalized",
                "model_key": audit["model_key"],
                "split": audit["split"],
                "correction_count": audit["correction_count"],
                "minimum_raw_kl": audit["minimum_raw_kl"],
                "post_sha256": audit["post_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
