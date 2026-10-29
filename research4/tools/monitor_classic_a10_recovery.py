#!/usr/bin/env python3
"""Bounded monitor for the one-time classic-A10 Research4 provider run."""

from __future__ import annotations

import argparse
import json
import subprocess
import time

PROFILE = "azure-research4"
RUN_NAME = "Research4 scale1 v6 classic A10 provider recovery"
IDENTITY = "qualification_rank4_a10_fp16_scale1_v6"
INTENT_SHA = "96c19838047bf8e79ce1d48b572ee34ff6a7cbd5e8504830595ce665c1b717f3"
TASK_KEYS = {
    "validate_before_gpu",
    "run_one_partition",
    "finalize_partition_mlflow",
    "release_active_reservation",
}
STARTUP_ABORT_SECONDS = 1200
OPERATOR_STOP_SECONDS = 2400
CEILING_SECONDS = 2700
POLL_SECONDS = 60
LOCK_ROOT = (
    "dbfs:/Volumes/research4_simulated_continuity/bronze/model_data/locks"
)
ACTIVE_LOCK_PATH = LOCK_ROOT + "/scientific.lock"
CONSUMED_PATH = LOCK_ROOT + "/consumed/" + IDENTITY + ".json"
TERMINAL_STATES = {"TERMINATED", "INTERNAL_ERROR", "SKIPPED"}


def _task(run: dict, key: str) -> dict:
    matches = [item for item in run.get("tasks", [])
               if item.get("task_key") == key]
    if len(matches) != 1:
        raise ValueError(f"exact task {key} not found")
    return matches[0]


def _reservation(owner_run_id: int) -> dict:
    return {
        "identity": IDENTITY,
        "intent_sha256": INTENT_SHA,
        "owner": str(owner_run_id),
        "status": "reserved_after_gpu_ready_before_model",
    }


def _validate_reservation(payload: dict | None, owner_run_id: int,
                          label: str) -> bool:
    if payload is None:
        return False
    if payload != _reservation(owner_run_id):
        raise ValueError(f"{label} is not owned by the exact provider run")
    return True


def cancellation_run_id(run: dict) -> int:
    task_run_id = _task(run, "run_one_partition").get("run_id")
    if type(task_run_id) is not int or task_run_id <= 0:
        raise ValueError("missing typed GPU task run id")
    return task_run_id


def decision(run: dict, active_lock: dict | None, consumed: dict | None,
             now_ms: int) -> str:
    if run.get("run_name") != RUN_NAME:
        raise ValueError("unexpected provider run name")
    if {item.get("task_key") for item in run.get("tasks", [])} != TASK_KEYS:
        raise ValueError("unexpected provider task set")
    run_id = run.get("run_id")
    if type(run_id) is not int or run_id <= 0:
        raise ValueError("missing typed provider run id")
    start_ms = run.get("start_time")
    if type(start_ms) is not int or start_ms <= 0:
        raise ValueError("missing typed provider start time")
    elapsed = (now_ms - start_ms) / 1000.0
    lock_owned = _validate_reservation(active_lock, run_id, "scientific lock")
    consumed_owned = _validate_reservation(consumed, run_id, "consumed receipt")
    if consumed_owned and not lock_owned:
        gpu_state = _task(run, "run_one_partition").get("state", {}).get(
            "life_cycle_state")
        if gpu_state not in TERMINAL_STATES:
            # Normal success removes the active lease immediately before the
            # notebook exits.  Preserve that closeout window while keeping the
            # immutable hard ceiling.  The internal worker has already applied
            # the 2400-second operator stop before releasing this lease.
            if elapsed >= CEILING_SECONDS:
                return "cancel_ceiling"
            return "scientific_closeout_pending"
    if lock_owned:
        if elapsed >= CEILING_SECONDS:
            return "cancel_ceiling"
        if elapsed >= OPERATOR_STOP_SECONDS:
            return "cancel_operator_stop"
        return "scientific_attempt_active"
    gpu = _task(run, "run_one_partition")
    lifecycle = gpu.get("state", {}).get("life_cycle_state")
    if lifecycle in TERMINAL_STATES:
        return "gpu_terminal_before_scientific_lock"
    if elapsed >= STARTUP_ABORT_SECONDS:
        return "cancel_startup_timeout"
    return "wait"


def downstream_released(run: dict, active_lock: dict | None,
                        consumed: dict | None, scientific_seen: bool) -> bool:
    run_id = run.get("run_id")
    if type(run_id) is not int or run_id <= 0:
        raise ValueError("missing typed provider run id")
    if active_lock is not None:
        _validate_reservation(active_lock, run_id, "scientific lock")
        return False
    for key in ("finalize_partition_mlflow", "release_active_reservation"):
        task = _task(run, key)
        if task.get("state", {}).get("life_cycle_state") not in TERMINAL_STATES:
            return False
        if task.get("state", {}).get("result_state") != "SUCCESS":
            raise RuntimeError(f"{key} did not succeed")
    if scientific_seen:
        if not _validate_reservation(consumed, run_id, "consumed receipt"):
            raise RuntimeError("consumed identity was not retained")
    elif consumed is not None:
        raise RuntimeError("unexpected consumed identity appeared")
    return True


def _json_cli(cli: str, *args: str) -> object:
    result = subprocess.run(
        [cli, *args, "--profile", PROFILE, "--output", "json"],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def _cat_optional(cli: str, path: str) -> dict | None:
    result = subprocess.run(
        [cli, "fs", "cat", path, "--profile", PROFILE],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        error = (result.stderr + "\n" + result.stdout).strip()
        normalized = error.lower()
        not_found = (
            "file does not exist:" in normalized
            or "no such file" in normalized
            or "resource_does_not_exist" in normalized
        )
        if not_found:
            return None
        raise RuntimeError(f"Databricks fs cat failed closed: {error[:500]}")
    return json.loads(result.stdout)


def _cancel_gpu_task(cli: str, task_run_id: int) -> None:
    subprocess.run(
        [cli, "jobs", "cancel-run", str(task_run_id), "--profile", PROFILE],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=int)
    parser.add_argument("--cli", required=True)
    parser.add_argument("--arm-cancel", action="store_true")
    args = parser.parse_args()

    cancel_sent = False
    scientific_seen = False
    while True:
        run = _json_cli(args.cli, "jobs", "get-run", str(args.run_id))
        active_lock = _cat_optional(args.cli, ACTIVE_LOCK_PATH)
        consumed = _cat_optional(args.cli, CONSUMED_PATH)
        scientific_seen = scientific_seen or active_lock is not None or consumed is not None
        action = decision(
            run, active_lock, consumed, int(time.time() * 1000))
        print(json.dumps({
            "run_id": args.run_id,
            "gpu_task_run_id": cancellation_run_id(run),
            "action": action,
            "checked_at_epoch_ms": int(time.time() * 1000),
        }, sort_keys=True), flush=True)
        if action.startswith("cancel_") and not cancel_sent:
            if not args.arm_cancel:
                raise SystemExit(
                    "Cancellation threshold reached but --arm-cancel was not supplied")
            _cancel_gpu_task(args.cli, cancellation_run_id(run))
            cancel_sent = True
        state = run.get("state", {}).get("life_cycle_state")
        if state in TERMINAL_STATES:
            if downstream_released(
                    run, active_lock, consumed, scientific_seen):
                return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
