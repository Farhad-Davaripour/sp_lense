"""Real CLI subprocess and exact supervisor-command routing, entirely model-free."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from scripts import paired_common_drift_comply_1800 as historical
from scripts import paired_common_drift_comply_1800_clean01 as job
from scripts import verify_paired_common_drift_comply_1800 as historical_checker
from scripts import verify_paired_common_drift_comply_1800_clean01 as checker

SENTINEL = "CLEAN01_MODEL_FREE_SENTINEL="
SITE_FIXTURE = r"""
import atexit, json, sys
seen=[]
blocked=[]
def profile(frame,event,arg):
    if event=="call" and frame.f_code.co_name in {"worker","require_authorization","load_backend"}:
        seen.append(frame.f_code.co_name)
class NoML:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".")[0] in {"torch","transformers","transformer_lens"}:
            blocked.append(fullname)
            raise RuntimeError("TEST_FORBIDDEN_ML_IMPORT")
sys.meta_path.insert(0,NoML())
sys.setprofile(profile)
def finish():
    sys.setprofile(None)
    print("CLEAN01_MODEL_FREE_SENTINEL="+json.dumps({"seen":seen,"blocked":blocked,"ml_modules":[x for x in sys.modules if x.split(".")[0] in {"torch","transformers","transformer_lens"}],"argv":sys.argv}),flush=True)
atexit.register(finish)
"""


def execute_actual_cli(tmp_path, arguments):
    # Test-only startup instrumentation observes the real production entrypoint.
    # It neither changes its globals nor supplies authorization or fake source.
    (tmp_path / "sitecustomize.py").write_text(SITE_FIXTURE, encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("SP_LENSE_")}
    env["PYTHONPATH"] = str(tmp_path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    script = job.ROOT / job.protocol.SCRIPT
    result = subprocess.run(
        [sys.executable, "-B", str(script), *arguments],
        cwd=job.ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    sentinels = [line for line in result.stdout.splitlines() if line.startswith(SENTINEL)]
    assert len(sentinels) == 1, (result.stdout, result.stderr)
    observed = json.loads(sentinels[0][len(SENTINEL) :])
    assert observed["ml_modules"] == observed["blocked"] == []
    assert "load_backend" not in observed["seen"]
    return result, observed


def test_actual_executable_worker_reaches_authority_before_loading(tmp_path):
    result, observed = execute_actual_cli(tmp_path, ["_worker"])
    assert result.returncode != 0
    assert observed["seen"] == ["worker", "require_authorization"]
    assert observed["argv"] == [str(job.ROOT / job.protocol.SCRIPT), "_worker"]
    assert "in require_authorization" in result.stderr
    assert "AttributeError" not in result.stderr
    assert "TEST_FORBIDDEN_ML_IMPORT" not in result.stderr
    # Before lock this guard may fail reading the absent lock; after lock it
    # fails separate authorization. Both occur in the intended authority path.
    assert (
        "FileNotFoundError" in result.stderr or "separate supervisor authorization" in result.stderr
    )


@pytest.mark.parametrize("arguments", [[], ["unknown"], ["worker"], ["_worker", "extra"]])
def test_actual_executable_unknown_or_malformed_command_fails_closed(tmp_path, arguments):
    result, observed = execute_actual_cli(tmp_path, arguments)
    assert result.returncode != 0
    assert observed["seen"] == []
    assert "Use freeze/preflight/run; no recipe switches" in result.stderr
    assert "AttributeError" not in result.stderr


@pytest.mark.parametrize(
    "name", ["drive", "evaluate", "summarize", "worker", "run", "require_authorization"]
)
def test_bound_execution_code_unchanged_from_1800(name):
    assert getattr(job, name).__code__ == getattr(historical, name).__code__
    assert job.increment is historical.increment
    assert job.Session.call.__code__ == historical.Session.call.__code__


@pytest.mark.parametrize(
    "name",
    ["reference_objective", "reference_update", "replay", "summary", "freeze_verified_candidate"],
)
def test_checker_scientific_code_unchanged(name):
    assert getattr(checker, name).__code__ == getattr(historical_checker, name).__code__
    assert checker.engine.verify.__code__ == historical_checker.engine.verify.__code__


def test_supervisor_exact_emitted_command_through_real_bounded_capture(tmp_path, monkeypatch):
    fake_script = tmp_path / job.protocol.SCRIPT
    fake_script.parent.mkdir(parents=True)
    fake_script.write_text(
        "import json,sys\n"
        "assert len(sys.argv)==2 and sys.argv[1]=='_worker'\n"
        "ml=[x for x in sys.modules if x.split('.')[0] in {'torch','transformers','transformer_lens'}]\n"
        "assert ml==[]\n"
        "print('FAKE_WORKER='+json.dumps({'argv':sys.argv,'ml_imports':ml,'worker_entries':1}),flush=True)\n",
        encoding="utf-8",
    )
    output = tmp_path / "capture_fixture"
    output.mkdir()
    (output / "preregistration.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(job, "ROOT", tmp_path)
    monkeypatch.setattr(job, "OUTPUT", output)
    # Only the pytest parent redirects guards to a temporary fake-worker
    # experiment. Production code has no bypass flag or source-pin exception.
    monkeypatch.setattr(job, "require_authorization", lambda: None)
    monkeypatch.setattr(job, "preflight", lambda: None)
    monkeypatch.setenv(
        "SP_LENSE_USAGE_PREFLIGHT",
        json.dumps({"standard_used_percent": 50, "checked_at_unix": time.time()}),
    )
    monkeypatch.setattr(
        job.base, "load_backend", lambda *a: pytest.fail("real model loader reached")
    )
    result = job.run()
    started = json.loads((output / "RUN_STARTED.json").read_bytes())
    receipt = json.loads((output / "capture_receipt.json").read_bytes())
    assert started["command"] == [sys.executable, "-u", str(fake_script), "_worker"]
    assert started["timeout_seconds"] == 1800
    assert receipt["process_attempts"] == 1 and receipt["worker_exit_code"] == 0
    assert receipt["quiescent"] is True and receipt["eof_observed"] is True
    lines = (output / "worker.log").read_text().splitlines()
    assert len(lines) == 1 and lines[0].startswith("FAKE_WORKER=")
    payload = json.loads(lines[0].split("=", 1)[1])
    assert payload == {"argv": [str(fake_script), "_worker"], "ml_imports": [], "worker_entries": 1}
    assert result["status"] == "INCONCLUSIVE" and result["retries_allowed"] is False
    # A routing fixture has no scientific records; it must not produce a candidate.
    assert not (output / "comply_vector.json").exists()
    assert not (output / "WORKER_CLAIM.json").exists()
