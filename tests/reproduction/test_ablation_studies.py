import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.ablation_studies import (
    ARMS,
    RECORDED,
    RIDGE,
    audit_rank,
    audit_ridge,
    sha256,
)


def test_retained_rank_records_and_ridge_results_replay():
    rank = audit_rank(ROOT)
    assert rank["models"]["m2_s42"]["2"]["guarded_corrections"] == 50
    ridge = audit_ridge(ROOT)
    assert ridge["m2_s42"]["all_cases"]["guarded_corrections"] == 50
    assert ridge["m2_s42"]["active_only"]["guarded_corrections"] == 55
    assert len(ridge["m2_s42"]["active_only"]["gained_vs_original"]) == 5
    assert ridge["m2_s42"]["active_only"]["lost_vs_original"] == []
    assert ridge["m08_s43"]["active_only"]["guarded_corrections"] == 69
    assert all(r["ridge_replay_max_error"] < 1e-4 for v in ridge.values() for r in v.values())
    assert all(
        r["guarded_control_changes"] == r["guarded_wrong_way"] == 0
        for v in ridge.values()
        for r in v.values()
    )


def test_exact_executed_source_is_preserved():
    plan = json.loads((ROOT / RIDGE / "PLAN.json").read_text())
    assert sha256(ROOT / RECORDED) == plan["script_sha256"]
    assert plan["fixed"]["rank"] == 4
    assert set(plan["arms"]) == set(ARMS)


def test_offline_audit_works_with_assertions_disabled():
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            "sp_lense.research2.ablation_studies",
            "audit",
            "--repo",
            str(ROOT),
        ],
        cwd=ROOT,
        env=dict(os.environ, SP_LENSE_REPO=str(ROOT)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ridge"]["m2_s42"]["active_only"]["guarded_corrections"] == 55


@pytest.fixture
def copied_ridge(tmp_path):
    output = tmp_path / "ridge"
    shutil.copytree(ROOT / RIDGE, output)
    return output


def test_changed_weights_fail_after_manifest_and_plan_are_rehashed(copied_ridge):
    folder = copied_ridge / "m08_s42"
    path = folder / "active_only.npz"
    with np.load(path, allow_pickle=False) as saved:
        values = {k: saved[k] for k in saved.files}
    values["weights"][0, 0] += 1
    np.savez_compressed(path, **values)
    files = json.loads((folder / "FILES.json").read_text())
    files[path.name] = sha256(path)
    (folder / "FILES.json").write_text(json.dumps(files))
    plan = json.loads((copied_ridge / "PLAN.json").read_text())
    plan["variants"]["m08_s42"]["arms"]["active_only"]["checkpoint_sha256"] = sha256(path)
    (copied_ridge / "PLAN.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="weight replay"):
        audit_ridge(ROOT, copied_ridge)


def test_changed_scores_fail_even_with_updated_file_hash(copied_ridge):
    folder = copied_ridge / "m08_s42"
    path = folder / "active_only.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["canonical_probability"] = 0.333
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    files = json.loads((folder / "FILES.json").read_text())
    files[path.name] = sha256(path)
    (folder / "FILES.json").write_text(json.dumps(files))
    with pytest.raises(ValueError, match="metric replay"):
        audit_ridge(ROOT, copied_ridge)


def test_missing_arm_cannot_be_hidden_by_removing_manifest_entry(copied_ridge):
    path = copied_ridge / "m08_s42/FILES.json"
    files = json.loads(path.read_text())
    del files["active_only.jsonl"]
    path.write_text(json.dumps(files))
    with pytest.raises(ValueError, match="Incomplete ridge file manifest"):
        audit_ridge(ROOT, copied_ridge)
