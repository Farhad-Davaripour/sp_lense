import json
import os
import subprocess
import sys
from pathlib import Path

from sp_lense.reproduction.paths import repository_root
from sp_lense.steering.provenance import executed_source_digest, source_file

ROOT = Path(__file__).resolve().parents[2]


def test_module_entrypoint_works_outside_checkout_with_explicit_root(tmp_path):
    result = subprocess.run(
        [sys.executable, "-O", "-m", "sp_lense.reproduction", "--repo", str(ROOT), "verify"],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "25 immutable reproduction artifacts verified" in result.stdout


def test_invalid_explicit_root_never_falls_back_to_another_checkout(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="Not a study checkout"):
        repository_root(tmp_path)


def test_gpu_sources_recover_exact_recorded_execution_hashes():
    baseline = json.loads((ROOT / "study/baseline_scores/RUNTIME.json").read_text())
    guarded = json.loads((ROOT / "study/guarded_steering/RUNTIME.json").read_text())
    assert executed_source_digest(source_file("gated_chat.py")) == baseline["source_sha256"]
    assert executed_source_digest(source_file("guarded_chat.py")) == guarded["source_sha256"]
    assert (
        executed_source_digest(source_file("search_steering_rules.py"))
        == guarded["rule_source_sha256"]
    )
