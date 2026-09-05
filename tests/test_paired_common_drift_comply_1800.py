"""Targeted identity/time/authority/fresh-zero checks; no model or tensor imports."""

import dis
import json
import time
from types import SimpleNamespace

import pytest

from scripts import paired_common_drift_comply as old
from scripts import paired_common_drift_comply_1800 as job
from scripts import verify_paired_common_drift_comply as old_checker
from scripts import verify_paired_common_drift_comply_1800 as checker
from scripts.paired_common_drift_comply_1800_binding import load_bound


@pytest.mark.parametrize("name", ["drive", "summarize", "evaluate"])
def test_scientific_runtime_code_unchanged(name):
    assert getattr(job, name).__code__ == getattr(old, name).__code__
    assert job.increment is old.increment
    assert job.Session.call.__code__ == old.Session.call.__code__
    assert job.ForwardLedger.begin.__code__ == old.ForwardLedger.begin.__code__


@pytest.mark.parametrize(
    "name",
    [
        "reference_objective",
        "reference_update",
        "replay",
        "verify_data",
        "summary",
        "freeze_verified_candidate",
    ],
)
def test_independent_scientific_code_unchanged(name):
    assert getattr(checker, name).__code__ == getattr(old_checker, name).__code__


def test_only_verifier_deadline_constant_changes():
    before, after = old_checker.engine.verify.__code__, checker.engine.verify.__code__
    assert before.co_code == after.co_code
    assert after.co_consts == tuple(
        1800 if type(c) is int and c == 1200 else c for c in before.co_consts
    )
    assert (
        sum(
            i.opname == "LOAD_CONST" and i.argval == 1800
            for i in dis.get_instructions(checker.engine.verify)
        )
        == 2
    )
    assert job.supervise.__defaults__ == (1800,)
    assert old.supervise.__defaults__ == (1200,)


def test_worker_has_only_new_deadline_constants_and_message():
    before, after = old.worker.__code__, job.worker.__code__
    assert before.co_code == after.co_code
    assert after.co_consts == tuple(
        1800
        if type(c) is int and c == 1200
        else c.replace("1200", "1800")
        if isinstance(c, str)
        else c
        for c in before.co_consts
    )


def test_binding_rejects_wrong_source_or_replacement_count():
    spec = job.TIME_ONLY_BINDING
    with pytest.raises(ValueError, match="source bytes"):
        load_bound("scripts._reject_wrong_digest", spec["source_path"], "0" * 64, ())
    with pytest.raises(ValueError, match="replacement sites"):
        load_bound(
            "scripts._reject_wrong_site",
            spec["source_path"],
            spec["source_sha256"],
            (("not a real source fragment", "replacement", 1),),
        )


def test_new_output_and_authority_distinct():
    assert job.OUTPUT != old.OUTPUT
    assert checker.OUTPUT == job.OUTPUT
    assert job.protocol.AUTH_KEY != old.protocol.AUTH_KEY
    assert job.protocol.AUTH_SCOPE.endswith("216F/96D/1800s;no retry")


def test_old_authority_cannot_authorize_new_worker(tmp_path, monkeypatch):
    (tmp_path / "preregistration.json").write_text("{}")
    monkeypatch.setattr(job, "OUTPUT", tmp_path)
    monkeypatch.delenv(job.protocol.AUTH_KEY, raising=False)
    monkeypatch.setenv(old.protocol.AUTH_KEY, "{}")
    monkeypatch.setattr(job.base, "load_backend", lambda *a: pytest.fail("model loader reached"))
    with pytest.raises(ValueError, match="separate supervisor authorization"):
        job.worker()
    assert not (tmp_path / "WORKER_CLAIM.json").exists()


@pytest.mark.parametrize(
    "digest", ["0" * 64, "2e81567a6af62a95f09bfcf60975e32cf670fe2ada098a2cb43d5c04f192b19b"]
)
def test_wrong_or_historical_lock_authority_rejected(tmp_path, monkeypatch, digest):
    (tmp_path / "preregistration.json").write_text("{}")
    monkeypatch.setattr(job, "OUTPUT", tmp_path)
    monkeypatch.setenv(
        job.protocol.AUTH_KEY,
        json.dumps(
            {
                "authorized_by": "supervisor",
                "scope": job.protocol.AUTH_SCOPE,
                "preregistration_sha256": digest,
            }
        ),
    )
    with pytest.raises(ValueError, match="exact paired lock"):
        job.require_authorization()


@pytest.mark.parametrize("timeout", [1200, 1799, 1801, 3600])
def test_timeout_not_adjustable(tmp_path, timeout):
    with pytest.raises(ValueError, match="fixed1800"):
        job.supervise(["fake"], tmp_path, {}, timeout=timeout)


def test_one_capture_at_exact_1800_and_no_retry(tmp_path, monkeypatch):
    records, calls = [], []
    monkeypatch.setattr(
        job,
        "PairedBudget",
        lambda *a: SimpleNamespace(write_bytes=lambda path, data: records.append(json.loads(data))),
    )

    def capture(command, budget, deadline, **kwargs):
        calls.append(deadline)
        return {"status": "INCONCLUSIVE", "quiescent": False}

    monkeypatch.setattr(job.capture, "run_capture", capture)
    monkeypatch.setattr(
        job.engine.recorder, "journal_counts", lambda *a: pytest.fail("unquiescent journal read")
    )
    monkeypatch.setattr(checker, "finalize_recording", lambda budget, receipt, result: result)
    result = job.supervise(
        ["fake"], tmp_path, {"standard_used_percent": 50, "checked_at_unix": time.time()}
    )
    assert len(calls) == 1
    assert records[0]["deadline_monotonic"] - records[0]["started_monotonic"] == 1800
    assert records[0]["timeout_seconds"] == 1800
    assert records[0]["forward_ceiling"] == 216 and records[0]["derivative_ceiling"] == 96
    assert result["status"] == "INCONCLUSIVE" and result["retries_allowed"] is False


@pytest.mark.parametrize("baseline_accepted,expected_calls", [(False, 216), (True, 24)])
def test_fresh_zero_and_original_first_acceptance_final_replays(
    monkeypatch, baseline_accepted, expected_calls
):
    plan = job.protocol.build_plan()
    calls, skips, updates, endpoints = [], [], [], []
    monkeypatch.setattr(job, "quality", lambda row: True)
    monkeypatch.setattr(job, "accepts", lambda row: baseline_accepted)
    monkeypatch.setattr(job, "stopping", lambda states: "accepted" if baseline_accepted else None)

    def call(cell, w, previous):
        calls.append((cell, list(w), previous))
        if cell["condition"] == "baseline":
            assert w == [0.0] * 1024 and previous is None
        return {"row": {"cell_id": cell["cell_id"]}}

    def propose(gradients, w, path, stage, baselines):
        assert len(gradients) == len(baselines) == 12
        assert len(updates) == stage - 1
        return {
            "status": "ready",
            "step_norm": 0.001,
            "path_after": 0.001 * stage,
            "net_norm": 0.001 * stage,
            "w_after": [0.001 * stage] + [0.0] * 1023,
        }

    result = job.drive(
        plan, call, lambda *a: skips.append(a), propose, updates.append, endpoints.append
    )
    assert len(calls) == expected_calls
    assert len(calls) + len(skips) == 216
    assert len(endpoints) == 1
    assert len([c for c, _, _ in calls if c["condition"] == "final"]) == 12
    assert len(updates) == (0 if baseline_accepted else 8)
    assert result["candidate_eligible"] == baseline_accepted
