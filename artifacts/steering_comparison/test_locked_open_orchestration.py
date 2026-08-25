from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from locked_open_orchestration import (
    build_plan,
    combine_generations,
    partition_scored,
    verify_canonical_plan,
)

DIGEST = "a" * 64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    lock = {
        "models": [
            {
                "model_id": "Qwen/Qwen3.5-0.8B",
                "revision": "r08",
                "config": "configs/qwen35_08b_aligned.json",
                "config_sha256": DIGEST,
            },
            {
                "model_id": "Qwen/Qwen3.5-2B",
                "revision": "r2",
                "config": "configs/qwen35_2b_aligned.json",
                "config_sha256": "b" * 64,
            },
        ]
    }
    setup: dict[str, object] = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "model_revision": "r08",
        "model_config_sha256": DIGEST,
        "method_id": "gradient",
        "track": "matched",
        "selected_layer": 10,
        "selected_strength": 0.02,
        "position_schedule": "final_prompt_token",
        "direction_float32_sha256": "c" * 64,
        "direction_artifact_sha256": "d" * 64,
        "construction_config_sha256": "e" * 64,
        "direction_path": "artifacts/direction.json",
        "calibration_summary_path": "artifacts/preopen.json",
        "calibration_summary_sha256": "f" * 64,
        "roles": ["fixed_descriptive", "selected"],
    }
    manifest = {
        "status": "locked_before_validation_open",
        "allowed_open_setups": [setup],
    }
    lock_path = tmp_path / "lock.json"
    manifest_path = tmp_path / "preopen.json"
    output_directory = tmp_path / "validation_open"
    _write_json(lock_path, lock)
    _write_json(manifest_path, manifest)
    return lock_path, manifest_path, output_directory, setup


def _rows(setup: dict[str, object], count: int = 96) -> list[dict[str, object]]:
    output = []
    for index in range(count):
        output.append(
            {
                "split": "validation",
                "model_id": setup["model_id"],
                "model_revision": setup["model_revision"],
                "config_sha256": setup["model_config_sha256"],
                "method": setup["method_id"],
                "method_id": setup["method_id"],
                "setup": setup["track"],
                "track": setup["track"],
                "layer": setup["selected_layer"],
                "position": setup["position_schedule"],
                "calibration_magnitude": setup["selected_strength"],
                "direction_sha256": setup["direction_float32_sha256"],
                "direction_float32_sha256": setup["direction_float32_sha256"],
                "direction_artifact_sha256": setup["direction_artifact_sha256"],
                "construction_config_sha256": setup["construction_config_sha256"],
                "calibration_summary_sha256": setup["calibration_summary_sha256"],
                "generation_sha256": f"{index:064x}",
            }
        )
    return output


def test_validation_plan_is_deterministic_and_bound_to_locked_models(tmp_path: Path) -> None:
    lock_path, manifest_path, output_directory, _ = _fixture(tmp_path)
    first = build_plan(tmp_path, lock_path, manifest_path, output_directory, "validation")
    second = build_plan(tmp_path, lock_path, manifest_path, output_directory, "validation")
    assert first == second
    assert first["setup_count"] == 1
    assert first["setups"][0]["model_tag"] == "qwen35_08b"
    assert first["setups"][0]["calibration_summary_sha256"] == "0" * 64
    assert first["setups"][0]["source_calibration_summary_sha256"] == "f" * 64


def test_verify_plan_requires_literal_canonical_bytes(tmp_path: Path) -> None:
    lock_path, manifest_path, output_directory, _ = _fixture(tmp_path)
    plan = build_plan(tmp_path, lock_path, manifest_path, output_directory, "validation")
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(
        (json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
    )
    verified = verify_canonical_plan(
        tmp_path,
        lock_path,
        manifest_path,
        output_directory,
        "validation",
        plan_path,
    )
    assert verified == plan

    plan_path.write_bytes((json.dumps(plan, sort_keys=True) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="byte-for-byte"):
        verify_canonical_plan(
            tmp_path,
            lock_path,
            manifest_path,
            output_directory,
            "validation",
            plan_path,
        )


def test_plan_rejects_duplicate_exact_setups(tmp_path: Path) -> None:
    lock_path, manifest_path, output_directory, setup = _fixture(tmp_path)
    _write_json(
        manifest_path,
        {
            "status": "locked_before_validation_open",
            "allowed_open_setups": [setup, setup],
        },
    )
    with pytest.raises(ValueError, match="duplicate exact setup"):
        build_plan(tmp_path, lock_path, manifest_path, output_directory, "validation")


def test_combine_and_partition_require_exact_96_row_setup(tmp_path: Path) -> None:
    lock_path, manifest_path, output_directory, _ = _fixture(tmp_path)
    plan = build_plan(tmp_path, lock_path, manifest_path, output_directory, "validation")
    plan_path = output_directory / "plan.json"
    _write_json(plan_path, plan)
    generation_path = tmp_path / plan["setups"][0]["generation_path"]
    rows = _rows(plan["setups"][0])
    _write_jsonl(generation_path, rows)
    combined_path = output_directory / "all.jsonl"
    assert combine_generations(tmp_path, plan_path, combined_path) == 96
    for row in rows:
        row["decision"] = "preserve"
    _write_jsonl(combined_path, rows)
    assert partition_scored(tmp_path, plan_path, combined_path) == 96
    scored_path = tmp_path / plan["setups"][0]["scored_path"]
    assert len(scored_path.read_text(encoding="utf-8").splitlines()) == 96


def test_combine_rejects_incomplete_generation(tmp_path: Path) -> None:
    lock_path, manifest_path, output_directory, _ = _fixture(tmp_path)
    plan = build_plan(tmp_path, lock_path, manifest_path, output_directory, "validation")
    plan_path = output_directory / "plan.json"
    _write_json(plan_path, plan)
    generation_path = tmp_path / plan["setups"][0]["generation_path"]
    _write_jsonl(generation_path, _rows(plan["setups"][0], count=95))
    with pytest.raises(ValueError, match="95 rows instead of 96"):
        combine_generations(tmp_path, plan_path, output_directory / "all.jsonl")


def test_combine_rejects_plan_after_source_manifest_changes(tmp_path: Path) -> None:
    lock_path, manifest_path, output_directory, _ = _fixture(tmp_path)
    plan = build_plan(tmp_path, lock_path, manifest_path, output_directory, "validation")
    plan_path = output_directory / "plan.json"
    _write_json(plan_path, plan)
    generation_path = tmp_path / plan["setups"][0]["generation_path"]
    _write_jsonl(generation_path, _rows(plan["setups"][0]))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["post_lock_change"] = True
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="source manifest hash differs"):
        combine_generations(tmp_path, plan_path, output_directory / "all.jsonl")


def test_combine_rejects_source_manifest_path_escape(tmp_path: Path) -> None:
    lock_path, manifest_path, output_directory, setup = _fixture(tmp_path)
    plan = build_plan(tmp_path, lock_path, manifest_path, output_directory, "validation")
    outside_manifest = tmp_path.parent / f"{tmp_path.name}_outside.json"
    _write_json(
        outside_manifest,
        {
            "status": "locked_before_validation_open",
            "allowed_open_setups": [setup],
        },
    )
    plan["source_manifest_path"] = f"../{outside_manifest.name}"
    plan["source_manifest_sha256"] = hashlib.sha256(
        outside_manifest.read_bytes()
    ).hexdigest()
    plan_path = output_directory / "plan.json"
    _write_json(plan_path, plan)

    with pytest.raises(ValueError, match="source manifest path escapes repository root"):
        combine_generations(tmp_path, plan_path, output_directory / "all.jsonl")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model_revision", "wrong-revision"),
        ("config_sha256", "0" * 64),
        ("method", "caa"),
        ("setup", "canonical"),
        ("position", "all_response_tokens"),
        ("direction_sha256", "1" * 64),
        ("calibration_summary_sha256", "2" * 64),
    ],
)
def test_combine_rejects_mismatched_full_setup_identity(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    lock_path, manifest_path, output_directory, _ = _fixture(tmp_path)
    plan = build_plan(tmp_path, lock_path, manifest_path, output_directory, "validation")
    plan_path = output_directory / "plan.json"
    _write_json(plan_path, plan)
    generation_path = tmp_path / plan["setups"][0]["generation_path"]
    rows = _rows(plan["setups"][0])
    rows[0][field] = replacement
    _write_jsonl(generation_path, rows)

    with pytest.raises(ValueError, match="rows from a different setup"):
        combine_generations(tmp_path, plan_path, output_directory / "all.jsonl")
