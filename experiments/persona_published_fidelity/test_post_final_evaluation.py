from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from experiments.persona_published_fidelity import persona_published_fidelity as pf
from experiments.persona_published_fidelity import post_final_evaluation as post_final


def _run(repo: Path, *args: str) -> str:
    process = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return process.stdout.strip()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _make_frozen_repo(
    tmp_path: Path,
    *,
    subject: str = post_final.FINAL_COMMIT_SUBJECT,
    omit: str | None = None,
    corrupt_entry: str | None = None,
) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repo.mkdir()
    _run(repo, "init", "-b", "main")
    _run(repo, "config", "user.email", "test@example.com")
    _run(repo, "config", "user.name", "Test")
    required = {
        post_final.REQUIRED_FINAL_OUTPUTS[0]: json.dumps(
            {"schema_version": "sp_lense.comparison.report.v1", "outcome": "synthetic"}
        ).encode(),
        post_final.REQUIRED_FINAL_OUTPUTS[1]: b"# synthetic report\n",
        post_final.REQUIRED_FINAL_OUTPUTS[2]: b"# synthetic review\n",
        post_final.REQUIRED_FINAL_OUTPUTS[3]: b'{"complete":true}\n',
        post_final.STAGE2_LOCK_PATH: b'{"schema_version":"synthetic-stage2"}\n',
    }
    for path, content in required.items():
        _write(repo / path, content)
    _write(
        repo / "experiments/persona_published_fidelity/config.json",
        b'{"schema_version":"synthetic"}\n',
    )
    _write(
        repo / "experiments/persona_published_fidelity/lock_manifest.json",
        b'{"schema_version":"synthetic"}\n',
    )
    source = Path(post_final.__file__).read_bytes()
    _write(repo / "experiments/persona_published_fidelity/post_final_evaluation.py", source)
    _write(repo / "README.md", b"base\n")
    _run(repo, "add", ".")
    _run(repo, "commit", "-m", "Synthetic base")
    base = _run(repo, "rev-parse", "HEAD")

    paths = [path for path in required if path != omit]
    paths = post_final._powershell_sort_unique(paths)
    entries = []
    for path in paths:
        raw = (repo / path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if path == corrupt_entry:
            digest = "f" * 64
        entries.append({"path": path, "sha256": digest, "size_bytes": len(raw)})
    inventory = {
        "schema_version": post_final.FINAL_INVENTORY_SCHEMA,
        "phase": "final",
        "base_commit": base,
        "path_count": len(entries),
        "paths_sha256": post_final._inventory_paths_sha256(paths),
        "entries": entries,
    }
    _write(
        repo / post_final.FINAL_INVENTORY_PATH,
        (json.dumps(inventory, indent=2) + "\n").encode(),
    )
    _run(repo, "add", post_final.FINAL_INVENTORY_PATH)
    _run(repo, "commit", "-m", subject)
    final = _run(repo, "rev-parse", "HEAD")
    _run(repo, "init", "--bare", str(remote))
    _run(repo, "remote", "add", "origin", str(remote))
    _run(repo, "push", "-u", "origin", "main")
    return repo, final


def test_final_gate_binds_exact_pushed_commit_inventory_and_outputs(tmp_path: Path) -> None:
    repo, final = _make_frozen_repo(tmp_path)
    gate = post_final.expected_post_final_gate(repo)
    assert gate["final_commit"] == final
    assert gate["verified_remote_commit"] == final
    assert gate["main_ranking_eligible"] is False
    assert set(gate["required_final_output_sha256s"]) == set(post_final.REQUIRED_FINAL_OUTPUTS)

    _write(repo / "later.txt", b"later\n")
    _run(repo, "add", "later.txt")
    _run(repo, "commit", "-m", "Later local commit")
    with pytest.raises(RuntimeError, match="HEAD == origin/main"):
        post_final.expected_post_final_gate(repo, final_commit=final)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"subject": "Wrong final subject"}, "wrong exact subject"),
        ({"omit": post_final.REQUIRED_FINAL_OUTPUTS[2]}, "omits required"),
        ({"corrupt_entry": post_final.REQUIRED_FINAL_OUTPUTS[1]}, "committed bytes"),
    ],
)
def test_final_gate_rejects_subject_coverage_or_identity(
    tmp_path: Path, kwargs: dict[str, str], message: str
) -> None:
    repo, _ = _make_frozen_repo(tmp_path, **kwargs)
    with pytest.raises((RuntimeError, ValueError), match=message):
        post_final.expected_post_final_gate(repo)


def _plan() -> dict:
    return {
        "schema_version": post_final.POST_FINAL_PLAN_SCHEMA,
        "study_role": pf.SECONDARY_ROLE,
        "main_ranking_eligible": False,
        "ranking_namespace": post_final.RANKING_NAMESPACE,
        "post_final_gate_sha256": "b" * 64,
        "final_commit": "c" * 40,
        "final_report_sha256": "d" * 64,
        "stage2_lock_sha256": "e" * 64,
        "model_id": "Qwen/Qwen3.5-0.8B",
        "model_revision": "revision",
        "forced_units": [
            {"family": "self_preservation", "case_id": "case-1", "target": "self"},
            {"family": "self_preservation", "case_id": "case-1", "target": "other"},
        ],
        "open_units": [],
        "conditions": ["baseline", "plus", "minus"],
    }


def _standard_rows(*, plus_effect: float = 1.0) -> list[dict]:
    rows = []
    for target in ("self", "other"):
        for condition, alpha, logit in (
            ("baseline", 0.0, 0.0),
            ("plus", 1.0, plus_effect if target == "self" else plus_effect / 4),
            ("minus", -1.0, -plus_effect if target == "self" else -plus_effect / 4),
        ):
            rows.append(
                {
                    "schema_version": "sp_lense.comparison.row.v1",
                    "model_id": "Qwen/Qwen3.5-0.8B",
                    "model_revision": "revision",
                    "method": "persona_vector",
                    "method_id": "persona_vector",
                    "setup": "canonical",
                    "track": "canonical",
                    "split": "sealed_test",
                    "family": "self_preservation",
                    "case_id": "case-1",
                    "target": target,
                    "condition": condition,
                    "condition_alpha": alpha,
                    "strength": alpha,
                    "layer": 10,
                    "position": "prompt_final_and_generated_tokens_cached_equivalent",
                    "run_seed": 20260824,
                    "runner_commit": "1" * 40,
                    "a_minus_b_log_odds": logit,
                    "forced_pair_label": "A" if logit >= 0 else "B",
                    "actual_next_token_label": "A" if logit >= 0 else "B",
                    "kl_from_baseline": 0.0 if condition == "baseline" else 0.001,
                    "coherent": True,
                    "preserve_label": "A",
                    "comply_label": "B",
                    "direction_id": "2" * 64,
                    "strength_id": "canonical:1",
                    "dataset_sha256": "3" * 64,
                    "protocol_sha256": "4" * 64,
                    "config_sha256": "5" * 64,
                    "direction_sha256": "6" * 64,
                    "direction_float32_sha256": "6" * 64,
                    "direction_artifact_sha256": "7" * 64,
                    "prompt_sha256": ("8" if target == "self" else "9") * 64,
                    "stage1_lock_sha256": "a" * 64,
                    "stage2_manifest_sha256": "b" * 64,
                    "calibration_summary_sha256": "c" * 64,
                    "construction_config_sha256": "d" * 64,
                }
            )
    return rows


def _envelopes(plan: dict, view: str, effect: float = 1.0) -> list[dict]:
    return post_final.make_row_envelopes(
        _standard_rows(plus_effect=effect),
        view=view,
        measurement_type="forced_choice",
        plan_sha256=pf.canonical_json_sha256(plan),
        post_final_gate_sha256=plan["post_final_gate_sha256"],
    )


def test_coverage_and_cross_arm_identity_are_fail_closed() -> None:
    plan = _plan()
    arms = {view: _envelopes(plan, view) for view in post_final.ALL_VIEWS}
    rows = {
        view: post_final.validate_row_envelopes(
            envelopes,
            plan,
            expected_view=view,
            measurement_type="forced_choice",
        )
        for view, envelopes in arms.items()
    }
    post_final.validate_cross_arm_identity(rows)

    with pytest.raises(RuntimeError, match="coverage differs"):
        post_final.validate_row_envelopes(
            arms["shared_selected"][:-1],
            plan,
            expected_view="shared_selected",
            measurement_type="forced_choice",
        )

    changed = _envelopes(plan, "shared_selected")
    for envelope in changed[:3]:
        envelope["measurement"]["prompt_sha256"] = "f" * 64
        envelope["source_standard_row_sha256"] = pf.canonical_json_sha256(envelope["measurement"])
    changed_rows = post_final.validate_row_envelopes(
        changed,
        plan,
        expected_view="shared_selected",
        measurement_type="forced_choice",
    )
    with pytest.raises(RuntimeError, match="prompt_sha256"):
        post_final.validate_cross_arm_identity({**rows, "shared_selected": changed_rows})


def test_report_and_rows_cannot_enter_main_ranking(tmp_path: Path) -> None:
    plan = _plan()
    arms = {
        "shared_selected": _envelopes(plan, "shared_selected", 1.5),
        "published_trait_selected": _envelopes(plan, "published_trait_selected", 1.0),
        "adapted_confirmatory_persona": _envelopes(plan, "adapted_confirmatory_persona", 0.5),
    }
    report = post_final.build_post_final_report(plan, arms)
    assert report["main_ranking_eligible"] is False
    assert "rankings" not in report
    assert report["ranking_firewall"]["eligible_for_main_behavioral_ranking"] is False

    from sp_lense.comparison_analysis import validate_result_rows

    with pytest.raises(ValueError, match="missing required fields"):
        validate_result_rows(arms["shared_selected"])

    main_path = tmp_path / "artifacts/steering_comparison/bad.json"
    with pytest.raises(ValueError, match="main comparison artifacts"):
        post_final.write_post_final_report(
            report,
            json_path=main_path,
            markdown_path=tmp_path / "secondary.md",
        )


def test_final_gate_tests_skip_without_git() -> None:
    # Makes the test module's external dependency explicit for minimal CI images.
    assert shutil.which("git") is not None
