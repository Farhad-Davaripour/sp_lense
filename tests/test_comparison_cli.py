from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sp_lense import comparison_cli
from sp_lense.comparison_cli import build_parser
from sp_lense.comparison_provenance import sha256_file
from sp_lense.comparison_report import _jspace_table
from sp_lense.steering_methods import DirectionArtifact

torch = pytest.importorskip("torch")


def test_comparison_cli_requires_explicit_phase_and_fit_method() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "fit",
            "--model-config",
            "model.json",
            "--method",
            "gradient",
            "--output",
            "out",
        ]
    )
    assert args.command == "fit"
    assert args.method == "gradient"


def test_sealed_and_validation_are_explicit_split_choices() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "evaluate-forced",
                "--model-config",
                "m",
                "--direction",
                "d",
                "--track",
                "matched",
                "--strength",
                "0.02",
                "--split",
                "test",
                "--calibration-summary-sha256",
                "a",
                "--construction-config-sha256",
                "b",
                "--output",
                "o",
            ]
        )


def test_calibration_summary_command_requires_validated_grid_plan_and_shards() -> None:
    args = build_parser().parse_args(
        [
            "build-calibration-summary",
            "--mode",
            "matched",
            "--grid-plan",
            "forced_grid_plan.json",
            "--point-shards",
            "point-1.json",
            "point-2.json",
            "--output",
            "summary.json",
        ]
    )
    assert args.mode == "matched"
    assert args.grid_plan == Path("forced_grid_plan.json")
    assert args.point_shards == [Path("point-1.json"), Path("point-2.json")]


@pytest.mark.parametrize(
    ("argv", "command"),
    [
        (
            [
                "run-forced-grid",
                "--model-config",
                "model.json",
                "--direction-manifest",
                "directions.json",
                "--output-dir",
                "grid",
            ],
            "run-forced-grid",
        ),
        (
            [
                "judge-requests",
                "--kind",
                "open",
                "--input",
                "generations.jsonl",
                "--output",
                "requests.jsonl",
            ],
            "judge-requests",
        ),
        (
            [
                "attach-judgments",
                "--kind",
                "persona",
                "--input",
                "rollouts.jsonl",
                "--responses",
                "responses.jsonl",
                "--output",
                "scored.jsonl",
            ],
            "attach-judgments",
        ),
        (["verify-stage2", "--stage2-manifest", "stage2.json"], "verify-stage2"),
        (
            [
                "build-stage2-manifest",
                "--preopen-manifest",
                "preopen.json",
                "--environment-lock",
                "environment.json",
                "--calibration-summary",
                "summary.json",
                "--direction-manifest",
                "directions.json",
                "--output",
                "stage2.json",
            ],
            "build-stage2-manifest",
        ),
        (
            [
                "report",
                "--stage2-manifest",
                "stage2.json",
                "--forced-rows",
                "sealed.jsonl",
                "--output-json",
                "report.json",
                "--output-markdown",
                "report.md",
            ],
            "report",
        ),
        (
            [
                "prepare-jspace-atoms",
                "--model-config",
                "model.json",
                "--layer",
                "10",
                "--atoms-output",
                "atoms.pt",
                "--token-labels-output",
                "labels.json",
                "--manifest-output",
                "atoms-manifest.json",
            ],
            "prepare-jspace-atoms",
        ),
        (
            [
                "jspace",
                "--direction",
                "direction.json",
                "--atoms-manifest",
                "atoms-manifest.json",
                "--setup",
                "matched",
                "--output",
                "jspace.json",
            ],
            "jspace",
        ),
        (
            [
                "generate-open",
                "--model-config",
                "model.json",
                "--direction",
                "direction.json",
                "--track",
                "matched",
                "--strength",
                "0.02",
                "--split",
                "validation",
                "--calibration-summary-sha256",
                "0" * 64,
                "--construction-config-sha256",
                "a" * 64,
                "--output",
                "open.jsonl",
            ],
            "generate-open",
        ),
    ],
)
def test_offline_workflow_subcommands_are_exposed(
    argv: list[str], command: str
) -> None:
    assert build_parser().parse_args(argv).command == command


def test_jspace_unavailable_layer_writes_reportable_non_gating_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = Path(__file__).parents[1] / "configs" / "steering_comparison_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    model = lock["models"][0]
    artifact = DirectionArtifact(
        method="caa",
        direction=torch.ones(model["architecture"]["residual_width"]),
        layer=23,
        intervention_geometry="caa_post_prompt",
        metadata={
            "model_id": model["model_id"],
            "model_revision": model["revision"],
            "model_config_sha256": model["config_sha256"],
            "track": "canonical",
        },
    )
    direction_path = tmp_path / "direction.json"
    direction_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "jspace.jsonl"
    monkeypatch.setattr(comparison_cli, "_load_lock", lambda repo_root, path: lock)
    monkeypatch.setattr(comparison_cli, "read_direction_artifact", lambda path, module: artifact)

    comparison_cli.command_jspace(
        SimpleNamespace(
            repo_root=Path(__file__).parents[1],
            lock=lock_path,
            direction=direction_path,
            setup="canonical",
            k=[8, 16, 25],
            random_count=50,
            random_seed=20_260_824,
            max_working_gib=8.0,
            max_dictionary_read_tib=4.0,
            atoms_manifest=None,
            output=output,
        )
    )

    lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["status"] == "not_run_lens_layer_unavailable"
    assert record["analysis"] is None
    assert set(record["lens_provenance"]) == {"file_sha256", "revision", "source_layers"}
    table = _jspace_table([record])
    assert table[0]["status"] == "not_run_lens_layer_unavailable"
    assert table[0]["used_for_primary_ranking"] is False


def test_sealed_setup_approval_runs_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "model.json"
    config.write_text("{}", encoding="utf-8")
    direction = tmp_path / "direction.json"
    direction.write_text(
        json.dumps(
            {
                "method": "gradient",
                "layer": 10,
                "direction_sha256": "d" * 64,
                "artifact_sha256": "e" * 64,
            }
        ),
        encoding="utf-8",
    )
    stage2 = tmp_path / "stage2.json"
    stage2.write_text("{}", encoding="utf-8")
    lock_path = tmp_path / "stage1.json"
    lock_path.write_text("{}", encoding="utf-8")
    lock = {
        "dataset": {"path": "dataset.json", "sha256": "a" * 64},
        "models": [],
    }
    model = {
        "model_id": "m",
        "revision": "1" * 40,
        "config_sha256": sha256_file(config),
    }
    seen: dict[str, object] = {}

    monkeypatch.setattr(comparison_cli, "_load_lock", lambda root, path: lock)
    monkeypatch.setattr(
        comparison_cli, "load_comparison_dataset", lambda path, expected_sha256: {}
    )
    monkeypatch.setattr(comparison_cli, "_model_record", lambda value, path: model)
    monkeypatch.setattr(
        comparison_cli, "verify_stage2_manifest", lambda root, value, path: object()
    )

    def reject_setup(verified: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise RuntimeError("blocked before model load")

    monkeypatch.setattr(comparison_cli, "assert_approved_setup", reject_setup)
    monkeypatch.setattr(
        comparison_cli.ResearchBackend,
        "load",
        lambda *args, **kwargs: pytest.fail("model loaded before sealed setup approval"),
    )
    args = SimpleNamespace(
        repo_root=tmp_path,
        lock=lock_path,
        model_config=config,
        direction=direction,
        track="matched",
        strength=0.02,
        split="sealed_test",
        stage2_manifest=stage2,
        calibration_summary_sha256="b" * 64,
        construction_config_sha256="c" * 64,
        include_tbsp=False,
        output=tmp_path / "out.jsonl",
    )
    with pytest.raises(RuntimeError, match="blocked before model load"):
        comparison_cli.command_evaluate_forced(args)
    assert seen["method_id"] == "gradient"
    assert seen["selected_layer"] == 10
    assert seen["position_schedule"] == "final_prompt_token"


def test_artifact_writer_emits_hash_bound_construction_config(tmp_path: Path) -> None:
    artifact = DirectionArtifact(
        "gradient",
        torch.tensor([1.0, 0.0]),
        10,
        "matched_final_prompt",
        {
            "track": "matched",
            "model_id": "m",
            "model_revision": "1" * 40,
            "model_config_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
            "protocol_sha256": "c" * 64,
            "stage1_lock_sha256": "d" * 64,
            "runner_commit": "2" * 40,
        },
    )
    lock = {
        "methods": {"gradient": {"direction": "locked"}},
        "comparison_tracks": {
            "matched_primary": {"fixed_strength": 0.02},
            "canonical_secondary": {},
        },
    }
    evidence = tmp_path / "gradient_evidence.json"
    evidence.write_text('{"gradient":"audit"}', encoding="utf-8")
    records = comparison_cli._write_artifacts(
        tmp_path,
        {"gradient_matched": artifact},
        repo_root=tmp_path,
        lock=lock,
        evidence_paths={"gradient_construction_diagnostics": evidence},
    )
    construction_path = tmp_path / records[0]["construction_config_path"]
    construction = json.loads(construction_path.read_text(encoding="utf-8"))
    assert not Path(records[0]["path"]).is_absolute()
    assert not Path(records[0]["construction_config_path"]).is_absolute()
    assert not Path(construction["evidence_artifacts"][0]["path"]).is_absolute()
    assert records[0]["construction_config_sha256"] == sha256_file(construction_path)
    assert construction["direction_artifact_sha256"] == artifact.artifact_sha256
    assert construction["locked_configuration"]["method"] == {"direction": "locked"}
    assert construction["evidence_artifacts"][0]["sha256"] == sha256_file(evidence)


def test_validation_requires_zero_calibration_summary_sentinel(tmp_path: Path) -> None:
    args = SimpleNamespace(
        repo_root=tmp_path,
        lock=tmp_path / "lock.json",
        split="validation",
        calibration_summary_sha256="a" * 64,
    )
    with pytest.raises(RuntimeError, match="all-zero pre-summary"):
        comparison_cli.command_evaluate_forced(args)
