"""Model-free routing/accounting tests; fake callbacks do not certify model hooks."""

import json
import os
import subprocess
import sys
import time
from fractions import Fraction
from types import SimpleNamespace

import pytest

from scripts import certified_descent_comply as job

SENTINEL = "CERTIFIED_DESCENT_CLI_OBSERVER="
SITE_FIXTURE = r"""
import atexit,json,sys,os
from pathlib import Path
seen=[];blocked=[]
def profile(frame,event,arg):
    if event=='call' and frame.f_code.co_name in {'worker','require_authorization','load_backend','run','run_capture'} and str(frame.f_globals.get('__name__','')).startswith('scripts.'):
        seen.append(frame.f_code.co_name)
        if frame.f_code.co_name=='require_authorization' and os.environ.get('TEST_CERTIFIED_AUTH_OUTPUT'):
            frame.f_globals['OUTPUT']=Path(os.environ['TEST_CERTIFIED_AUTH_OUTPUT'])
class NoML:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','transformers','transformer_lens'}:
            blocked.append(fullname)
            raise RuntimeError('TEST_FORBIDDEN_ML_IMPORT')
sys.meta_path.insert(0,NoML());sys.setprofile(profile)
def finish():
    sys.setprofile(None)
    print('CERTIFIED_DESCENT_CLI_OBSERVER='+json.dumps({'seen':seen,'blocked':blocked,'ml':[x for x in sys.modules if x.split('.')[0] in {'torch','transformers','transformer_lens'}]}),flush=True)
atexit.register(finish)
"""


def actual_cli(tmp_path, arguments, authorization=None, *, synthetic_lock=False):
    (tmp_path / "sitecustomize.py").write_text(SITE_FIXTURE, encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("SP_LENSE_")}
    env.update(PYTHONPATH=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    if synthetic_lock:
        (tmp_path / "preregistration.json").write_text(
            '{"fixture":"synthetic authorization target only"}', encoding="utf-8"
        )
        env["TEST_CERTIFIED_AUTH_OUTPUT"] = str(tmp_path)
    if authorization is not None:
        env[job.protocol.AUTH_KEY] = json.dumps(authorization)
    result = subprocess.run(
        [sys.executable, "-B", str(job.ROOT / job.protocol.SCRIPT), *arguments],
        cwd=job.ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith(SENTINEL)]
    assert len(lines) == 1, (result.stdout, result.stderr)
    observed = json.loads(lines[0][len(SENTINEL) :])
    assert observed["ml"] == observed["blocked"] == []
    assert "load_backend" not in observed["seen"]
    assert "run_capture" not in observed["seen"]
    assert not (tmp_path / "RUN_STARTED.json").exists()
    return result, observed


@pytest.mark.parametrize(
    "authorization", [None, {}, {"authorized_by": "supervisor", "scope": "wrong"}]
)
def test_actual_worker_cli_routes_to_missing_or_wrong_authority_before_loading(
    tmp_path, authorization
):
    result, observed = actual_cli(tmp_path, ["_worker"], authorization)
    assert result.returncode != 0 and observed["seen"] == ["worker", "require_authorization"]
    assert "in require_authorization" in result.stderr and "AttributeError" not in result.stderr
    assert (
        "FileNotFoundError" in result.stderr or "separate supervisor authorization" in result.stderr
    )


@pytest.mark.parametrize("arguments", [[], ["unknown"], ["worker"], ["_worker", "extra"]])
def test_actual_cli_unknown_commands_never_enter_worker(tmp_path, arguments):
    result, observed = actual_cli(tmp_path, arguments)
    assert result.returncode != 0 and observed["seen"] == []
    assert "Use freeze/preflight/run; no recipe switches" in result.stderr


def test_actual_authority_identity_is_new_and_explicit():
    assert job.protocol.AUTH_KEY == "SP_LENSE_CERTIFIED_DESCENT_COMPLY_V1_AUTHORIZATION"
    assert (
        job.protocol.AUTH_SCOPE
        == "one fresh certified-descent dyadic COMPLY construction;216F/96D/1800s;no retry"
    )


@pytest.mark.parametrize(
    "authorization", [None, {}, {"authorized_by": "supervisor", "scope": "wrong"}]
)
def test_actual_run_cli_rejects_missing_wrong_authority_without_launch(tmp_path, authorization):
    result, observed = actual_cli(tmp_path, ["run"], authorization, synthetic_lock=True)
    assert result.returncode != 0
    assert observed["seen"] == ["run", "require_authorization"]
    assert "separate supervisor authorization" in result.stderr


@pytest.mark.parametrize("command", ["run", "_worker"])
def test_actual_cli_correct_scope_wrong_lock_hash_is_rejected(tmp_path, command):
    authority = {
        "authorized_by": "supervisor",
        "scope": job.protocol.AUTH_SCOPE,
        "preregistration_sha256": "0" * 64,
    }
    result, observed = actual_cli(tmp_path, [command], authority, synthetic_lock=True)
    assert result.returncode != 0
    assert observed["seen"] == ["run" if command == "run" else "worker", "require_authorization"]
    assert "separate supervisor authorization" in result.stderr
    assert "FileNotFoundError" not in result.stderr


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {},
        {"unknown": 0},
        {"standard_used_percent": True, "checked_at_unix": 1000},
        {"standard_used_percent": 50, "checked_at_unix": True},
        {"standard_used_percent": float("nan"), "checked_at_unix": 1000},
        {"standard_used_percent": float("inf"), "checked_at_unix": 1000},
        {"standard_used_percent": 50, "checked_at_unix": float("nan")},
        {"standard_used_percent": 50, "checked_at_unix": float("inf")},
        {"standard_used_percent": 50, "checked_at_unix": 1001},
        {"standard_used_percent": 50, "checked_at_unix": 939},
        {"standard_used_percent": -1, "checked_at_unix": 1000},
        {"standard_used_percent": 100.01, "checked_at_unix": 1000},
    ],
)
def test_usage_guard_rejects_missing_nonfinite_stale_or_out_of_range(monkeypatch, snapshot):
    monkeypatch.setattr(job.time, "time", lambda: 1000.0)
    with pytest.raises(ValueError):
        job.checked_usage(snapshot)


def test_usage_guard_accepts_fresh_exactly_100(monkeypatch):
    monkeypatch.setattr(job.time, "time", lambda: 1000.0)
    snapshot = {"standard_used_percent": 100, "checked_at_unix": 1000.0}
    assert job.checked_usage(snapshot) == snapshot


def fake_plan():
    ids = [f"fake_{i}" for i in range(12)]
    groups = [("baseline", 0, False)]
    for i in range(1, 9):
        groups.extend([(f"gradient_{i}", i, True), (f"step_{i}", i, True)])
    groups.append(("final", 9, False))
    cells = [
        {
            "cell_id": pid + "__" + condition,
            "prompt_id": pid,
            "condition": condition,
            "stage": stage,
            "optional": optional,
        }
        for condition, stage, optional in groups
        for pid in ids
    ]
    return {
        "construction_ids": ids,
        "control_ids": [],
        "transfer_ids": [],
        "model": {"d_model": 2},
        "cells": cells,
    }


def drive_fixture(monkeypatch, mode):
    plan, calls, skipped, updates, endpoints = fake_plan(), [], [], [], []
    monkeypatch.setattr(
        job,
        "quality",
        lambda row: not (mode == "quality_failure" and row["condition"].startswith("gradient_")),
    )
    monkeypatch.setattr(job, "accepts", lambda row: row["accepted"])
    monkeypatch.setattr(
        job,
        "stopping",
        lambda states: "accepted" if all(s["row"]["accepted"] for s in states) else None,
    )

    def call(cell, w, current):
        if cell["condition"] == "baseline":
            assert w == [0.0, 0.0] and current is None
        if cell["condition"].startswith("gradient_") or cell["condition"] == "final":
            assert current is not None and w == current["w"]
        good = mode == "baseline" or mode == "first" and cell["condition"] in ("step_1", "final")
        state = {"row": {**cell, "accepted": good}, "w": list(w), "current": current}
        calls.append(state)
        return state

    def propose(gradients, w, path, stage, baselines, *, history, path_upper):
        assert len(gradients) == len(baselines) == 12
        assert all(s["w"] == [0.0, 0.0] for s in baselines)
        assert history[0] == [0.0, 0.0] and history[-1] == w
        assert len(history) == stage
        assert float(Fraction(path_upper)) == path
        status = (
            "no_certified_step"
            if mode == "technical_failure"
            else mode
            if mode in ("exact_zero_optimum", "no_certified_step", "technical_failure")
            else "ready"
        )
        after = w[0] + 0.001
        bound = Fraction(path_upper) + abs(Fraction(after) - Fraction(w[0]))
        return {
            "status": status,
            "technical_failure": mode == "technical_failure",
            "w_after": [after, 0.0] if status == "ready" else list(w),
            "path_after": float(bound) if status == "ready" else path,
            "path_after_upper": str(bound) if status == "ready" else path_upper,
            "net_norm": after if status == "ready" else w[0],
            "step_norm": 0.001,
            "stage": stage,
        }

    def run():
        return job.drive(
            plan,
            call,
            lambda *args: skipped.append(args),
            propose,
            lambda update: updates.append(update),
            lambda endpoint: endpoints.append(endpoint),
        )

    return run, plan, calls, skipped, updates, endpoints


@pytest.mark.parametrize(
    "mode,forwards,derivatives,attempts",
    [
        ("baseline", 24, 0, 0),
        ("first", 48, 12, 1),
        ("exact_zero_optimum", 36, 12, 1),
        ("no_certified_step", 36, 12, 1),
        ("quality_failure", 36, 12, 0),
        ("maximum", 216, 96, 8),
    ],
)
def test_fake_driver_conditional_schedule_and_last_endpoint_replays(
    monkeypatch, mode, forwards, derivatives, attempts
):
    run, plan, calls, skipped, updates, endpoints = drive_fixture(monkeypatch, mode)
    result = run()
    assert len(calls) == forwards and len(calls) + len(skipped) == 216
    assert sum(s["row"]["condition"].startswith("gradient_") for s in calls) == derivatives
    assert len(updates) == attempts and len(endpoints) == 1
    finals = calls[-12:]
    assert all(s["row"]["condition"] == "final" and s["w"] == endpoints[0]["w"] for s in finals)
    assert [s["current"]["row"]["cell_id"] for s in finals] == endpoints[0]["endpoint_cell_ids"]
    assert result["final_cell_ids"] == [s["row"]["cell_id"] for s in finals]
    assert result["transfer_ran"] is False
    assert len(result["history_w_sha256"]) == result["updates"] + 1
    assert result["history_w_sha256"][0] == job.vector_sha([0.0, 0.0])
    assert result["history_w_sha256"][-1] == job.vector_sha(result["w"])
    assert result["path"] == float(Fraction(result["path_upper"]))
    executed = {s["row"]["cell_id"] for s in calls}
    omitted = {args[0]["cell_id"] for args in skipped}
    assert not executed & omitted and executed | omitted == {c["cell_id"] for c in plan["cells"]}
    assert all(anchor in executed for _, _, anchor in skipped)


def test_no_certified_step_replays_prior_endpoint_without_stationarity_claim(monkeypatch):
    run, _, calls, skipped, updates, endpoints = drive_fixture(monkeypatch, "no_certified_step")
    result = run()
    assert len(calls) == 36 and len(updates) == 1
    assert updates[0]["status"] == "no_certified_step"
    assert all(state["w"] == [0.0, 0.0] for state in calls[-12:])
    assert endpoints and skipped
    assert result["stop_reason"] == "no_certified_step"
    assert result.get("exact_zero_optimum", False) is False


def test_technical_failure_saved_without_rescue_replay_or_candidate(monkeypatch):
    run, _, calls, skipped, updates, endpoints = drive_fixture(monkeypatch, "technical_failure")
    with pytest.raises(ValueError):
        run()
    assert len(calls) == 24
    assert len(updates) == 1 and updates[0]["technical_failure"] is True
    assert not endpoints and not skipped


def test_ledger_deadline_and_216_ceiling(tmp_path):
    plan = fake_plan()
    ledger = job.ForwardLedger(tmp_path, plan["cells"], 100, now=lambda: 0)
    for cell in plan["cells"]:
        ledger.begin(cell)
        ledger.finish(True)
    assert ledger.attempts == ledger.completed == 216
    with pytest.raises(ValueError):
        ledger.begin(plan["cells"][-1])
    expired_dir = tmp_path / "expired"
    expired_dir.mkdir()
    expired = job.ForwardLedger(expired_dir, plan["cells"], 0, now=lambda: 0)
    with pytest.raises(ValueError, match="deadline"):
        expired.begin(plan["cells"][0])


def test_supervisor_exact_command_real_capture_and_no_retry(tmp_path, monkeypatch):
    fake_script = tmp_path / job.protocol.SCRIPT
    fake_script.parent.mkdir(parents=True)
    fake_script.write_text(
        "import json,sys\nassert sys.argv[1:]==['_worker']\n"
        "assert not any(x.split('.')[0] in {'torch','transformers','transformer_lens'} for x in sys.modules)\n"
        "print('FAKE_WORKER='+json.dumps(sys.argv),flush=True)\n",
        encoding="utf-8",
    )
    output = tmp_path / "fake_capture"
    output.mkdir()
    (output / "preregistration.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(job, "ROOT", tmp_path)
    monkeypatch.setattr(job, "OUTPUT", output)
    # Test-parent mocks only; the production CLI has no bypass or alternate loader.
    monkeypatch.setattr(job, "require_authorization", lambda: None)
    monkeypatch.setattr(job, "preflight", lambda: None)
    monkeypatch.setattr(job.base, "load_backend", lambda *args: pytest.fail("model loader reached"))
    usage_snapshot = {"standard_used_percent": 100, "checked_at_unix": time.time()}
    monkeypatch.setenv(
        "SP_LENSE_USAGE_PREFLIGHT",
        json.dumps(usage_snapshot),
    )
    result = job.run()
    started = json.loads((output / "RUN_STARTED.json").read_bytes())
    receipt = json.loads((output / "capture_receipt.json").read_bytes())
    assert started["command"] == [sys.executable, "-u", str(fake_script), "_worker"]
    assert started["timeout_seconds"] == 1800
    assert started["usage_preflight"] == usage_snapshot
    assert receipt["process_attempts"] == 1 and receipt["worker_exit_code"] == 0
    assert receipt["quiescent"] is True and receipt["eof_observed"] is True
    assert (output / "worker.log").read_text().splitlines() == [
        "FAKE_WORKER=" + json.dumps([str(fake_script), "_worker"])
    ]
    assert result["status"] == "INCONCLUSIVE" and result["retries_allowed"] is False
    assert (
        not (output / "comply_vector.json").exists() and not (output / "WORKER_CLAIM.json").exists()
    )
    repeated_launches = []

    def forbidden_capture(*args, **kwargs):
        repeated_launches.append(True)
        raise AssertionError("second process forbidden")

    monkeypatch.setattr(job.capture, "run_capture", forbidden_capture)
    repeated = job.run()
    assert repeated["status"] == "INCONCLUSIVE"
    assert repeated["retries_allowed"] is False and repeated_launches == []


def test_deadline_not_a_runtime_switch(tmp_path):
    with pytest.raises(ValueError, match="fixed1800"):
        job.supervise(["fake"], tmp_path, {}, timeout=1801)


def test_supervisor_technical_worker_failure_is_inconclusive(tmp_path, monkeypatch):
    from scripts import verify_certified_descent_comply as checker

    output = tmp_path / "technical_capture"
    output.mkdir()
    observed = []

    def failed_capture(command, budget, deadline, **kwargs):
        observed.append(command)
        return {
            "status": "INCONCLUSIVE",
            "quiescent": True,
            "worker_exit_code": 1,
            "process_attempts": 1,
            "reason": "technical_failure",
        }

    monkeypatch.setattr(job.capture, "run_capture", failed_capture)
    monkeypatch.setattr(
        job.engine.recorder,
        "journal_counts",
        lambda path: (24, 24, False) if path.name == "forward_events.jsonl" else (12, 12, False),
    )

    def finalizer(budget, receipt, result):
        assert receipt["worker_exit_code"] == 1
        assert result["status"] == "INCONCLUSIVE"
        return result

    monkeypatch.setattr(checker, "finalize_recording", finalizer)
    result = job.supervise(
        ["test-only-failed-worker"],
        output,
        {"standard_used_percent": 100, "checked_at_unix": time.time()},
    )
    assert result["status"] == "INCONCLUSIVE" and result["retries_allowed"] is False
    assert observed == [["test-only-failed-worker"]]
    assert not (output / "comply_vector.json").exists()


def test_unquiescent_capture_never_reads_mutable_journals(tmp_path, monkeypatch):
    from scripts import verify_certified_descent_comply as checker

    monkeypatch.setattr(
        job, "PairedBudget", lambda output: SimpleNamespace(write_bytes=lambda *a, **k: None)
    )
    monkeypatch.setattr(
        job.capture, "run_capture", lambda *a, **k: {"status": "INCONCLUSIVE", "quiescent": False}
    )
    monkeypatch.setattr(
        job.engine.recorder, "journal_counts", lambda *a: pytest.fail("unquiescent journal read")
    )
    monkeypatch.setattr(checker, "finalize_recording", lambda budget, receipt, result: result)
    result = job.supervise(
        ["fake"], tmp_path, {"standard_used_percent": 50, "checked_at_unix": time.time()}
    )
    assert result["status"] == "INCONCLUSIVE" and result["forward_attempts"] is None


def test_dirty_scoped_source_fails_identity_check(monkeypatch):
    monkeypatch.setattr(
        job.protocol, "build_plan", lambda: {"input_sha256": {"scoped.py": "unused"}}
    )
    monkeypatch.setattr(job.protocol, "SOURCE_PATHS", ())

    def fake_git(command, **kwargs):
        return b"100644 abc 0\tscoped.py\0" if command[1] == "ls-files" else b"scoped.py\0"

    monkeypatch.setattr(job.subprocess, "check_output", fake_git)
    with pytest.raises(ValueError, match="clean tracked"):
        job.source_identity()
