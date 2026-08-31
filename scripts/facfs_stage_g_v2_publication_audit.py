#!/usr/bin/env python3
"""Read-only provenance audit for the completed FACFS Stage-G v2 no-go.

This verifier deliberately has no model-loading code and no capture command.  It checks
the prospectively locked predecessor/successor chain, the capture ledger, the complete
output inventory, and the reported no-go decision using only committed files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs" / "facfs_stage_g_v2_lock.json"
V1_FAILURE_PATH = (
    ROOT / "artifacts" / "facfs" / "stage_g_v1" / "attempt_0001" / "attempt_failed.json"
)
V2_ARTIFACT_ROOT = ROOT / "artifacts" / "facfs" / "stage_g_v2"
V2_ATTEMPT_ROOT = V2_ARTIFACT_ROOT / "attempt_0002"
V2_RESULT_ROOT = ROOT / "results" / "facfs" / "stage_g_v2"
CAPTURE_COMMIT = "1b7e0e54bd3dbe78aaf3037fad612d49d71ddec4"


class AuditError(RuntimeError):
    """A committed provenance or result claim is inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AuditError(f"required regular file is absent: {path.relative_to(ROOT)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"JSON root is not an object: {path.relative_to(ROOT)}")
    return value


def verify_identity(value: Mapping[str, Any], field: str) -> None:
    body = {key: item for key, item in value.items() if key != field}
    if value.get(field) != canonical_sha256(body):
        raise AuditError(f"canonical identity differs: {field}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _verify_v1_predecessor(lock: dict[str, Any]) -> dict[str, Any]:
    predecessor = lock["predecessor_attempt"]
    failure = load_json(V1_FAILURE_PATH)
    verify_identity(failure, "attempt_failed_sha256")
    require(
        file_sha256(V1_FAILURE_PATH) == predecessor["failure_receipt_file_sha256"],
        "v1 failure receipt file hash differs from the successor lock",
    )
    require(
        failure["attempt_failed_sha256"] == predecessor["failure_receipt_identity_sha256"],
        "v1 failure receipt identity differs from the successor lock",
    )
    expected = {
        "attempt": "attempt_0001",
        "state": "failed_consumed_no_resume_no_retry",
        "exception_type": "TypeError",
        "exception_message": "_capture_prompt_only_residual.<locals>.hook() got an unexpected keyword argument 'hook'",
        "reserved_forwards": 1409,
        "reserved_backwards": 1408,
        "captured_objectives": 1408,
        "captured_sequences": 1408,
    }
    require(
        all(failure.get(key) == value for key, value in expected.items()),
        "v1 failure receipt fields differ from the immutable technical no-result",
    )
    old_lock = ROOT / predecessor["lock_path"]
    require(
        old_lock.is_file() and file_sha256(old_lock) == predecessor["lock_file_sha256"],
        "v1 lock hash differs from the successor lock",
    )
    v1_root = ROOT / predecessor["attempt_root"]
    require(v1_root.is_dir(), "v1 attempt root is absent")
    require(not (v1_root / "attempt_complete.json").exists(), "v1 was incorrectly completed")
    require(
        not (ROOT / "results" / "facfs" / "stage_g_v1").exists(),
        "v1 results namespace exists despite technical no-result status",
    )
    return {
        "v1_failure_receipt_file_sha256": file_sha256(V1_FAILURE_PATH),
        "v1_failure_receipt_identity_sha256": failure["attempt_failed_sha256"],
        "v1_state": failure["state"],
    }


def _verify_locked_inputs(lock: dict[str, Any]) -> None:
    verify_identity(lock, "lock_identity_sha256")
    require(
        lock["status"] == "prospectively_locked_before_any_stage_g_model_load_or_forward",
        "v2 lock is not prospectively locked",
    )
    for row in lock["locked_files"]:
        path = ROOT / row["path"]
        require(
            path.is_file() and not path.is_symlink() and file_sha256(path) == row["file_sha256"],
            f"locked input differs: {row['path']}",
        )
    exclusions = load_json(ROOT / "configs" / "facfs_stage_g_v2_exclusions.json")
    verify_identity(exclusions, "exclusions_sha256")
    require(exclusions["all_collision_gates_passed"] is True, "source collision gate failed")
    require(
        not any(exclusions["collision_counts"].values()),
        "source collision count is nonzero",
    )
    operations = load_json(ROOT / "configs" / "facfs_stage_g_v2_operations.json")
    verify_identity(operations, "operations_sha256")
    require(
        operations["operations_sha256"] == lock["operations_identity"]["operations_sha256"],
        "operations manifest identity differs from lock",
    )


def _verify_preflight(lock: dict[str, Any]) -> dict[str, Any]:
    receipt = load_json(V2_ARTIFACT_ROOT / "preflight_receipt.json")
    verify_identity(receipt, "receipt_sha256")
    require(receipt["status"] == "passed", "v2 zero-model preflight did not pass")
    require(
        receipt["lock_identity_sha256"] == lock["lock_identity_sha256"]
        and receipt["lock_file_sha256"] == file_sha256(LOCK_PATH),
        "preflight lock binding differs",
    )
    require(
        receipt["lock_commit"] == "5d25b3c4ebea72013b287e2706a046024752049e",
        "preflight did not use the bound predecessor-receipt commit",
    )
    for field in (
        "model_loaded",
        "model_forwards",
        "model_backwards",
        "generated_tokens",
        "finite_intervention_calls",
    ):
        require(receipt[field] in (False, 0), f"preflight is nonzero for {field}")
    require(receipt["windows_security_changed"] is False, "preflight changed Windows security")
    environment = receipt["checks"]["environment"]
    require(
        environment["packages"] == lock["environment"]["packages"],
        "preflight package provenance differs",
    )
    require(environment["python"] == lock["environment"]["python"], "preflight Python differs")
    require(
        environment["windows_smart_app_control"] == "On",
        "preflight did not attest Windows Smart App Control On",
    )
    inherited = receipt["checks"]["inherited"]
    require(inherited["freeze_core_all_eligible"] is False, "inherited calibration status differs")
    require(inherited["untouched_test_outcomes_viewed"] is False, "old untouched test was viewed")
    for denied in lock["inherited_equal_efficacy"]["absent_paths"]:
        require(not (ROOT / denied).exists(), f"hard-denied historical outcome exists: {denied}")
    return receipt


def _verify_ledger(lock: dict[str, Any]) -> dict[str, int | str]:
    operations = load_json(ROOT / "configs" / "facfs_stage_g_v2_operations.json")["operations"]
    expected = [
        (str(operation["objective_id"]), event)
        for operation in operations
        for event in operation["ledger_events"]
    ]
    ledger_path = V2_ATTEMPT_ROOT / "compute_ledger.jsonl"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == len(expected) == 2926, "ledger event count differs")
    prior = "0" * 64
    forwards = 0
    backwards = 0
    expected_fields = {
        "schema_version",
        "ledger_index",
        "event_id",
        "objective_id",
        "kind",
        "role",
        "prior_event_sha256",
        "cumulative_forwards",
        "cumulative_backwards",
        "event_sha256",
    }
    for index, (line, (objective_id, event)) in enumerate(zip(lines, expected), start=1):
        record = json.loads(line)
        require(
            isinstance(record, dict) and set(record) == expected_fields, "ledger schema differs"
        )
        verify_identity(record, "event_sha256")
        require(record["ledger_index"] == index, "ledger index differs")
        require(record["prior_event_sha256"] == prior, "ledger chain is broken")
        require(record["objective_id"] == objective_id, "ledger objective order differs")
        require(record["event_id"] == event["event_id"], "ledger event id differs")
        require(record["kind"] == event["kind"], "ledger event kind differs")
        expected_role = event.get("role", "opaque")
        require(record["role"] == expected_role, "ledger event role differs")
        if record["kind"] == "forward":
            forwards += 1
        elif record["kind"] == "backward":
            backwards += 1
        else:
            raise AuditError("ledger has unknown operation kind")
        require(record["cumulative_forwards"] == forwards, "ledger forward total differs")
        require(record["cumulative_backwards"] == backwards, "ledger backward total differs")
        prior = record["event_sha256"]
    planned = lock["compute_ceiling"]
    require(
        forwards == planned["physical_forward_invocations"],
        "ledger forwards exceed or miss ceiling",
    )
    require(
        backwards == planned["physical_backward_invocations"],
        "ledger backwards exceed or miss ceiling",
    )
    return {
        "ledger_events": len(lines),
        "forwards": forwards,
        "backwards": backwards,
        "final_event": prior,
    }


def _verify_inventory() -> dict[str, Any]:
    inventory = load_json(V2_ARTIFACT_ROOT / "output_inventory.json")
    verify_identity(inventory, "inventory_sha256")
    observed_paths = [
        path
        for root in (V2_ARTIFACT_ROOT, V2_RESULT_ROOT)
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"output_inventory.json", "attempt_complete.json"}
    ]
    observed = [
        {
            "path": relative(path),
            "byte_size": path.stat().st_size,
            "file_sha256": file_sha256(path),
        }
        for path in sorted(observed_paths, key=relative)
    ]
    require(inventory["files"] == observed, "output inventory does not exactly match files")
    require(inventory["file_count"] == len(observed) == 2870, "output inventory count differs")
    return inventory


def _verify_results(
    lock: dict[str, Any], receipt: dict[str, Any], ledger: dict[str, int | str]
) -> dict[str, Any]:
    attempt_started = load_json(V2_ATTEMPT_ROOT / "attempt_started.json")
    attempt_complete = load_json(V2_ATTEMPT_ROOT / "attempt_complete.json")
    realized = load_json(V2_ATTEMPT_ROOT / "realized_ledger.json")
    summary = load_json(V2_RESULT_ROOT / "summary.json")
    decomposition = load_json(V2_RESULT_ROOT / "walsh_decomposition.json")
    analysis = load_json(V2_RESULT_ROOT / "analysis_audit.json")
    for value, field in (
        (attempt_started, "attempt_started_sha256"),
        (attempt_complete, "attempt_complete_sha256"),
        (realized, "realized_ledger_sha256"),
        (summary, "summary_sha256"),
        (decomposition, "decomposition_sha256"),
        (analysis, "analysis_audit_sha256"),
    ):
        verify_identity(value, field)
    require(attempt_started["state"] == "started", "v2 attempt-start record differs")
    require(
        attempt_started["lock_identity_sha256"] == lock["lock_identity_sha256"]
        and attempt_started["preflight_receipt_sha256"] == receipt["receipt_sha256"],
        "attempt-start lock or preflight binding differs",
    )
    require(
        attempt_started["planned_totals"] == lock["compute_ceiling"],
        "attempt-start compute plan differs from the lock",
    )
    runtime = attempt_started["runtime"]
    for key in (
        "model_id",
        "model_revision",
        "device",
        "dtype",
        "d_model",
        "vocabulary_size",
        "chat_template_sha256",
        "torch_num_threads",
        "torch_num_interop_threads",
    ):
        lock_key = "revision" if key == "model_revision" else key
        require(runtime[key] == lock["model"][lock_key], f"captured runtime differs: {key}")
    require(
        runtime["model_layers"] == lock["model"]["blocks"], "captured runtime layer count differs"
    )
    require(
        runtime["packages"]
        == {
            name: lock["environment"]["packages"][name]
            for name in ("torch", "transformer-lens", "transformers")
        },
        "captured runtime package provenance differs",
    )
    require(attempt_complete["state"] == "complete", "v2 attempt is not complete")
    require(
        attempt_complete["scientific_outcome_unopened_until_commit_and_push"] is True,
        "attempt completion does not record outcome custody",
    )
    require(
        attempt_complete["lock_identity_sha256"] == lock["lock_identity_sha256"],
        "attempt lock differs",
    )
    require(
        attempt_complete["preflight_receipt_sha256"] == receipt["receipt_sha256"],
        "attempt preflight differs",
    )
    require(
        attempt_complete["final_compute_event_sha256"] == ledger["final_event"],
        "attempt final event differs",
    )
    inventory = _verify_inventory()
    require(
        attempt_complete["output_inventory_sha256"] == inventory["inventory_sha256"],
        "attempt inventory differs",
    )
    require(realized["counts_match_exactly"] is True, "realized counts not exact")
    require(
        realized["planned"] == lock["compute_ceiling"], "realized ledger plan differs from ceiling"
    )
    require(realized["realized"]["generated_tokens"] == 0, "capture generated tokens")
    require(
        realized["realized"]["finite_intervention_calls"] == 0, "capture used finite intervention"
    )
    require(
        realized["realized"]["physical_forward_invocations"] == ledger["forwards"],
        "realized forwards differ",
    )
    require(
        realized["realized"]["physical_backward_invocations"] == ledger["backwards"],
        "realized backwards differ",
    )
    require(
        realized["final_compute_event_sha256"] == ledger["final_event"],
        "realized final event differs",
    )
    require(summary["status"] == "no_go_fixed_axis_branch_ends", "summary status differs")
    require(
        summary["scenario_count"] == 11 and summary["scenario_successes"] == 0,
        "scenario result differs",
    )
    require(summary["thresholds"] == lock["thresholds"], "summary thresholds differ from lock")
    require(summary["all_11_required"] is True, "all-11 decision rule differs")
    require(
        summary["facfs_lock_authoring_authorized"] is False,
        "construction was incorrectly authorized",
    )
    require(
        summary["finite_facfs_intervention_authorized"] is False,
        "finite intervention was authorized",
    )
    require(
        summary["finite_intervention_used"] is False and summary["generated_tokens"] == 0,
        "summary reports model output",
    )
    scenarios = summary["scenario_results"]
    require(len(scenarios) == 11, "scenario table length differs")
    require(
        {row["scenario_id"] for row in scenarios} == {f"facfs_g2_s{i:03d}" for i in range(1, 12)},
        "scenario ids differ",
    )
    require(
        all(row["inventory_passed"] and not row["scenario_passed"] for row in scenarios),
        "scenario completion result differs",
    )
    require(
        sum(row["all_sp_opaque_effects_passed"] for row in scenarios) == 1,
        "SP opaque pass count differs",
    )
    require(
        not any(row["all_option_free_effects_passed"] for row in scenarios),
        "option-free gate unexpectedly passed",
    )
    require(
        not any(row["all_alignments_passed"] for row in scenarios),
        "alignment gate unexpectedly passed",
    )
    require(len(summary["opaque_effect_certificates"]) == 1408, "opaque certificate count differs")
    require(
        len(summary["option_free_effect_certificates"]) == 22,
        "option-free certificate count differs",
    )
    require(len(summary["alignment_certificates"]) == 22, "alignment certificate count differs")
    require(
        analysis["opaque_objective_count"] == 1408
        and analysis["option_free_objective_count"] == 22,
        "analysis objective counts differ",
    )
    require(analysis["alignment_count"] == 22, "alignment count differs")
    require(
        analysis["all_effect_certificates_numerically_valid"] is True,
        "effect numerical certificate failed",
    )
    require(
        analysis["all_alignment_certificates_numerically_valid"] is True,
        "alignment numerical certificate failed",
    )
    require(
        decomposition["diagnostic_only"] is True
        and decomposition["no_authorization_thresholds"] is True,
        "diagnostics gained decision authority",
    )
    return {
        "inventory_sha256": inventory["inventory_sha256"],
        "summary_sha256": summary["summary_sha256"],
        "scenario_successes": summary["scenario_successes"],
        "scenario_count": summary["scenario_count"],
    }


def _verify_git(lock: dict[str, Any]) -> dict[str, str]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "@{upstream}")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    divergence = git("rev-list", "--left-right", "--count", "HEAD...@{upstream}")
    remote_lines = git("ls-remote", "origin", f"refs/heads/{branch}").splitlines()
    remote = remote_lines[0].split()[0] if len(remote_lines) == 1 else ""
    require(branch == lock["expected_branch"], "current branch differs from the v2 lock")
    require(not status, "worktree is not clean")
    require(
        head == upstream == remote and divergence.split() == ["0", "0"],
        "branch is not exactly pushed",
    )
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", CAPTURE_COMMIT, head],
        cwd=ROOT,
        check=False,
    )
    require(completed.returncode == 0, "capture commit is not an ancestor of HEAD")
    return {"branch": branch, "head": head, "remote": remote, "divergence": divergence}


def _verify_live_environment(lock: dict[str, Any]) -> dict[str, str]:
    expected = lock["environment"]
    observed_python = ".".join(str(value) for value in sys.version_info[:3])
    require(observed_python == expected["python"], "live Python version differs")
    require(
        Path(sys.executable) == Path(expected["python_executable"]),
        "live Python executable differs",
    )
    observed = {name: importlib.metadata.version(name) for name in expected["packages"]}
    require(observed == expected["packages"], "live package versions differ")
    require(
        platform.system() == "Linux" and "microsoft" in platform.release().casefold(),
        "not running in WSL",
    )
    require(os.environ.get("USER") == expected["linux_user"], "live Linux user differs")
    smart_app = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", "(Get-MpComputerStatus).SmartAppControlState"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(smart_app == "On", "Windows Smart App Control is not On")
    return {"python": observed_python, "windows_smart_app_control": smart_app}


def audit_repository(
    *, check_git: bool = False, verify_live_environment: bool = False
) -> dict[str, Any]:
    """Verify the complete no-go chain without importing or loading the experiment model."""

    lock = load_json(LOCK_PATH)
    _verify_locked_inputs(lock)
    predecessor = _verify_v1_predecessor(lock)
    receipt = _verify_preflight(lock)
    ledger = _verify_ledger(lock)
    results = _verify_results(lock, receipt, ledger)
    report: dict[str, Any] = {
        "audit": "passed",
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "lock_identity_sha256": lock["lock_identity_sha256"],
        "predecessor": predecessor,
        "compute": ledger,
        "results": results,
    }
    if check_git:
        report["git"] = _verify_git(lock)
    if verify_live_environment:
        report["live_environment"] = _verify_live_environment(lock)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-git", action="store_true", help="require clean, exactly pushed branch state"
    )
    parser.add_argument(
        "--verify-live-environment",
        action="store_true",
        help="require the locked WSL/Python/package/Smart App Control state",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit_repository(
                check_git=arguments.check_git,
                verify_live_environment=arguments.verify_live_environment,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
