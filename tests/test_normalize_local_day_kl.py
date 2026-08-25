from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import normalize_local_day_kl as normalizer


def _row(*, model_key: str, split: str, kl: float) -> dict[str, object]:
    return {
        "schema_version": "sp_lense.local_day_choice_row.v1",
        "model_key": model_key,
        "split": split,
        "unit_id": "unit",
        "method": "gradient",
        "condition": "plus",
        "full_vocabulary_kl_from_baseline": kl,
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_normalizes_only_tiny_negative_kl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    lock_path = tmp_path / "lock.json"
    script_path = tmp_path / "normalizer.py"
    lock_path.write_text("{}\n", encoding="utf-8")
    script_path.write_text("# test\n", encoding="utf-8")
    monkeypatch.setattr(normalizer, "RESULT_ROOT", result_root)
    monkeypatch.setattr(normalizer, "ROOT", tmp_path)
    monkeypatch.setattr(normalizer, "LOCK_PATH", lock_path)
    monkeypatch.setattr(normalizer, "SCRIPT_PATH", script_path)
    monkeypatch.setattr(normalizer, "_git_head", lambda: "a" * 40)
    path = result_root / ".qwen35_2b_validation.staging.jsonl"
    _write_rows(
        path,
        [
            _row(model_key="qwen35_2b", split="validation", kl=-2.5e-7),
            _row(model_key="qwen35_2b", split="validation", kl=0.125),
        ],
    )

    audit = normalizer.normalize_staging_kl(model_key="qwen35_2b", split="validation")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["full_vocabulary_kl_from_baseline"] == 0.0
    assert rows[1]["full_vocabulary_kl_from_baseline"] == 0.125
    assert audit["correction_count"] == 1
    assert audit["minimum_raw_kl"] == -2.5e-7
    assert audit["pre_sha256"] != audit["post_sha256"]
    audit_path = result_root / "qwen35_2b_validation_kl_numerical_correction.json"
    assert json.loads(audit_path.read_text(encoding="utf-8"))["corrections"][0][
        "raw_kl"
    ] == -2.5e-7


def test_refuses_materially_negative_kl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    monkeypatch.setattr(normalizer, "RESULT_ROOT", result_root)
    path = result_root / ".qwen35_2b_validation.staging.jsonl"
    _write_rows(
        path,
        [_row(model_key="qwen35_2b", split="validation", kl=-1.1e-6)],
    )

    with pytest.raises(ValueError, match="exceeds numerical tolerance"):
        normalizer.normalize_staging_kl(model_key="qwen35_2b", split="validation")


def test_refuses_rewrite_without_negative_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root = tmp_path / "results"
    monkeypatch.setattr(normalizer, "RESULT_ROOT", result_root)
    path = result_root / ".qwen35_2b_validation.staging.jsonl"
    _write_rows(
        path,
        [_row(model_key="qwen35_2b", split="validation", kl=0.0)],
    )

    with pytest.raises(RuntimeError, match="no tiny-negative KL"):
        normalizer.normalize_staging_kl(model_key="qwen35_2b", split="validation")
