import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("optimized", [False, True])
def test_changed_expected_predictions_fail_actual_replay(tmp_path, optimized):
    # Update the checksum deliberately to test numerical validation independently.
    shutil.copytree(ROOT / "reproduce/artifacts", tmp_path / "artifacts")
    for name in ("run.py", "replay.py", "utils.py", "layout.py", "inventory.json"):
        shutil.copy2(ROOT / "reproduce" / name, tmp_path / name)
    expected_path = tmp_path / "artifacts/expected.json"
    expected = json.loads(expected_path.read_text())
    expected["xgboost_shutdown_v1"]["validation_probabilities"][0] += 0.1
    expected_path.write_text(json.dumps(expected))
    manifest_path = tmp_path / "artifacts/SHA256.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["expected.json"] = hashlib.sha256(expected_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), str(tmp_path / "run.py"), "replay"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "xgboost_shutdown_v1 validation: maximum difference" in result.stderr
