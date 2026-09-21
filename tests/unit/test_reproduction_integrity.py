import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("failure", ["checksum", "traversal", "empty"])
def test_verifier_rejects_bad_inputs_under_optimization(tmp_path, optimized, failure):
    inputs = tmp_path / "reproduce"
    inputs.mkdir()
    (inputs / "inventory.json").write_text(json.dumps(["payload"]))
    data = inputs / "artifacts"
    data.mkdir()
    (data / "payload").write_bytes(b"valid")
    manifest = {"payload": hashlib.sha256(b"valid").hexdigest()}
    if failure == "checksum":
        (data / "payload").write_bytes(b"corrupt")
    elif failure == "traversal":
        manifest = {"../outside": "0" * 64}
    else:
        manifest = {}
    (data / "SHA256.json").write_text(json.dumps(manifest))
    result = subprocess.run(
        [
            sys.executable,
            *(["-O"] if optimized else []),
            "-m",
            "sp_lense.reproduction",
            "--repo",
            str(tmp_path),
            "verify",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
    )
    assert result.returncode != 0
    assert "PASS:" not in result.stdout
    assert "VerificationError" in result.stderr


def test_valid_manifest_passes_optimized(tmp_path):
    inputs = tmp_path / "reproduce"
    inputs.mkdir()
    (inputs / "inventory.json").write_text(json.dumps(["payload"]))
    data = inputs / "artifacts"
    data.mkdir()
    (data / "payload").write_bytes(b"valid")
    (data / "SHA256.json").write_text(json.dumps({"payload": hashlib.sha256(b"valid").hexdigest()}))
    result = subprocess.run(
        [sys.executable, "-O", "-m", "sp_lense.reproduction", "--repo", str(tmp_path), "verify"],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
    )
    assert result.returncode == 0, result.stderr
    assert "PASS: 1" in result.stdout
