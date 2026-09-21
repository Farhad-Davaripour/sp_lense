import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("optimized", [False, True])
def test_changed_expected_predictions_fail_actual_replay(tmp_path, optimized):
    # Update the checksum deliberately to test numerical validation independently.
    inputs = tmp_path / "reproduce"
    inputs.mkdir()
    shutil.copytree(ROOT / "reproduce/artifacts", inputs / "artifacts")
    shutil.copy2(ROOT / "reproduce/inventory.json", inputs / "inventory.json")
    expected_path = inputs / "artifacts/expected.json"
    expected = json.loads(expected_path.read_text())
    expected["xgboost_shutdown_v1"]["validation_probabilities"][0] += 0.1
    expected_path.write_text(json.dumps(expected))
    manifest_path = inputs / "artifacts/SHA256.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["expected.json"] = hashlib.sha256(expected_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [
            sys.executable,
            *(["-O"] if optimized else []),
            "-m",
            "sp_lense.reproduction",
            "--repo",
            str(tmp_path),
            "replay",
        ],
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "xgboost_shutdown_v1 validation: maximum difference" in result.stderr
