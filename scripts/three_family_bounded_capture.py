"""Bounded binary PIPE capture for the three-family study's parent only.

This module has no model dependency and does not finalize the recording budget.
An unjoined reader remains a technical fault, not a claim of complete capture.
"""

# Interruption and storage faults must both go through bounded cleanup/receipts.
# ruff: noqa: BLE001

from __future__ import annotations

import hashlib
import math
import subprocess
import threading
import time
from pathlib import Path

READ_CHUNK_BYTES = 65536
LOG_CAP_BYTES = 4 * 1024 * 1024
EXCEPTION_PREFIX_BYTES = 4096


def exception_record(error):
    """Describe a bounded first string argument, never invoke arbitrary __str__.

    Hash/length fields refer to that selected argument, not an exception's custom
    rendering. Long strings are sliced before encoding; unknown totals stay null.
    """
    name = type.__getattribute__(type(error), "__name__")[:128]
    args = BaseException.args.__get__(error) if isinstance(error, BaseException) else ()
    selected = args[0] if args and type(args[0]) is str else None
    if selected is None:
        return {
            "exception_type": name,
            "text_source": "no_safe_first_string_argument",
            "prefix": "",
            "prefix_bytes": 0,
            "truncated": True,
            "full_text_bytes": None,
            "full_text_sha256": None,
        }
    fully_encoded = len(selected) <= EXCEPTION_PREFIX_BYTES
    raw = selected[:EXCEPTION_PREFIX_BYTES].encode("utf-8", errors="replace")
    prefix = raw[:EXCEPTION_PREFIX_BYTES].decode("utf-8", errors="ignore")
    prefix_bytes = prefix.encode("utf-8")
    return {
        "exception_type": name,
        "text_source": "first_string_argument_utf8_replace",
        "prefix": prefix,
        "prefix_bytes": len(prefix_bytes),
        "truncated": not fully_encoded or len(raw) > len(prefix_bytes),
        "full_text_bytes": len(raw) if fully_encoded else None,
        "full_text_sha256": hashlib.sha256(raw).hexdigest() if fully_encoded else None,
    }


def _prefix_on_disk(path):
    """Read the actual prefix after joins, including bytes from partial writes."""
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb", buffering=0) as handle:
        while True:
            data = handle.read(READ_CHUNK_BYTES)
            if not data:
                break
            size += len(data)
            if size > LOG_CAP_BYTES:
                raise ValueError("CAPTURE_PREFIX_EXCEEDS_CAP")
            digest.update(data)
    return size, digest.hexdigest()


def run_capture(
    command,
    budget,
    deadline,
    *,
    cwd=None,
    env=None,
    process_factory=subprocess.Popen,
    clock=time.monotonic,
    poll_interval=0.01,
    terminate_timeout=1.0,
    kill_timeout=1.0,
    reader_join_timeout=1.0,
):
    """Capture one process attempt with bounded memory and bounded cleanup waits.

    ``deadline`` is an absolute monotonic time. The budget must expose a short,
    independent sticky-fault lock: fault()/fault_code must not wait for file I/O.
    A blocking reader or writer can outlive cleanup only as an explicitly unjoined
    daemon; its budget is faulted, and the caller must not finalize that recording.
    """
    for value in (deadline, poll_interval, terminate_timeout, kill_timeout, reader_join_timeout):
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError("capture timing parameters must be finite numbers")
    if not 0 < poll_interval <= 1 or max(terminate_timeout, kill_timeout, reader_join_timeout) > 5:
        raise ValueError("capture cleanup waits must be bounded by five seconds")
    if min(terminate_timeout, kill_timeout, reader_join_timeout) < 0:
        raise ValueError("capture cleanup waits cannot be negative")
    started = clock()
    if not math.isfinite(started) or deadline <= started:
        raise ValueError("capture deadline must be in the future")
    stop = threading.Event()
    reader_done = threading.Event()
    budget_ready = threading.Event()
    watcher_stop = threading.Event()
    state_lock = threading.Lock()
    state = {
        "fault": None,
        "exception": None,
        "fault_persistence_error": False,
        "observed": 0,
        "observed_hash": hashlib.sha256(),
        "eof": False,
        "maximum_read": 0,
        "successful_prefix": 0,
        "budget_ready": False,
    }
    process = None
    process_attempts = 0
    reader = None
    worker_joined = False
    termination_attempted = False
    kill_attempted = False
    worker_exit_code = None
    cleanup_error = None
    fault_writer = None

    def persist_fault(code):
        try:
            budget.fault(code)
        except BaseException:
            with state_lock:
                state["fault_persistence_error"] = True

    def fail(code, error=None):
        nonlocal fault_writer
        with state_lock:
            if state["fault"] is None:
                state["fault"] = code
                state["exception"] = exception_record(error) if error is not None else None
                fault_writer = threading.Thread(
                    target=persist_fault, args=(code,), name="capture-sticky-fault", daemon=True
                )
                try:
                    fault_writer.start()
                except BaseException:
                    fault_writer = None
                    state["fault_persistence_error"] = True
        stop.set()

    def watch_budget():
        try:
            # Exclusive creation and fault reads may block in real storage I/O.
            # Their daemon must be joined before finalization is permitted.
            budget.write_bytes("worker.log", b"", mode="xb")
            initial_fault = budget.fault_code
            if initial_fault is not None:
                fail(initial_fault)
                return
            with state_lock:
                state["budget_ready"] = True
            budget_ready.set()
            while not stop.is_set():
                budget_fault = budget.fault_code
                if budget_fault is not None:
                    fail(budget_fault)
                    return
                if watcher_stop.is_set():
                    return
                watcher_stop.wait(poll_interval)
        except BaseException as error:
            fail("CAPTURE_BUDGET_STATE_ERROR", error)
        finally:
            budget_ready.set()

    def read_pipe():
        try:
            while not stop.is_set():
                chunk = process.stdout.read(READ_CHUNK_BYTES)
                if type(chunk) is not bytes or len(chunk) > READ_CHUNK_BYTES:
                    raise ValueError("pipe returned an invalid or oversized binary chunk")
                with state_lock:
                    state["observed"] += len(chunk)
                    state["observed_hash"].update(chunk)
                    state["maximum_read"] = max(state["maximum_read"], len(chunk))
                    if not chunk:
                        state["eof"] = True
                if not chunk or stop.is_set():
                    break
                room = LOG_CAP_BYTES - state["successful_prefix"]
                fitting = chunk[:room]
                if fitting:
                    budget.write_bytes("worker.log", fitting, mode="ab")
                    with state_lock:
                        state["successful_prefix"] += len(fitting)
                if len(chunk) > room:
                    fail("CAPTURE_LOG_CAP")
                    break
        except BaseException as error:
            fail("CAPTURE_READ_OR_WRITE_ERROR", error)
        finally:
            try:
                process.stdout.close()
            except BaseException as error:
                fail("CAPTURE_PIPE_CLOSE_ERROR", error)
            reader_done.set()

    watcher = threading.Thread(target=watch_budget, name="capture-budget-watch", daemon=True)
    try:
        watcher.start()
        budget_ready.wait(timeout=min(1.0, max(0, deadline - clock())))
        if not state["budget_ready"] or stop.is_set():
            raise RuntimeError("capture budget initialization did not complete without fault")
        if clock() >= deadline:
            raise RuntimeError("capture deadline expired before process launch")
        process_attempts += 1
        process = process_factory(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if process.stdout is None:
            raise RuntimeError("worker has no owned stdout pipe")
        reader = threading.Thread(target=read_pipe, name="three-family-pipe-capture", daemon=True)
        reader.start()
        while True:
            if stop.is_set():
                break
            code = process.poll()
            if code is not None and reader_done.is_set():
                worker_exit_code = process.wait(timeout=0)
                worker_joined = True
                if worker_exit_code != 0:
                    fail("CAPTURE_WORKER_NONZERO_EXIT")
                break
            if clock() >= deadline:
                fail("CAPTURE_DEADLINE")
                break
            stop.wait(min(poll_interval, max(0, deadline - clock())))
    except BaseException as error:
        fail("CAPTURE_SUPERVISOR_ERROR", error)
    finally:
        if process is not None and not worker_joined:
            try:
                if process.poll() is None:
                    termination_attempted = True
                    process.terminate()
                worker_exit_code = process.wait(timeout=terminate_timeout)
                worker_joined = True
            except BaseException as error:
                # A terminate failure must still attempt kill exactly once.
                fail("CAPTURE_TERMINATE_OR_WAIT_ERROR", error)
                try:
                    kill_attempted = True
                    process.kill()
                    worker_exit_code = process.wait(timeout=kill_timeout)
                    worker_joined = True
                except BaseException as kill_error:
                    fail("CAPTURE_KILL_OR_WAIT_ERROR", kill_error)
                    cleanup_error = "CAPTURE_WORKER_UNJOINED"
        if reader is not None:
            try:
                reader.join(timeout=reader_join_timeout)
            except BaseException as error:
                fail("CAPTURE_READER_JOIN_ERROR", error)
            if reader.is_alive():
                fail("CAPTURE_READER_UNJOINED")
                cleanup_error = cleanup_error or "CAPTURE_READER_UNJOINED"

    reader_joined = reader is None or not reader.is_alive()
    process_reader_quiet = reader_joined and (process is None or worker_joined)
    prefix_bytes = None
    prefix_sha = None
    prefix_reader = None
    prefix_result = {}

    def inspect_prefix():
        try:
            prefix_result["value"] = _prefix_on_disk(budget.root / "worker.log")
        except BaseException as error:
            fail("CAPTURE_PREFIX_READ_ERROR", error)

    if process_reader_quiet and state["budget_ready"]:
        prefix_reader = threading.Thread(
            target=inspect_prefix, name="capture-prefix-verification", daemon=True
        )
        prefix_reader.start()
        prefix_reader.join(timeout=reader_join_timeout)
        if prefix_reader.is_alive():
            fail("CAPTURE_PREFIX_READER_UNJOINED")
            cleanup_error = cleanup_error or "CAPTURE_PREFIX_READER_UNJOINED"
        elif "value" in prefix_result:
            prefix_bytes, prefix_sha = prefix_result["value"]
    # The watcher makes a last fault-state read after the actual-prefix read.
    watcher_stop.set()
    if watcher.ident is not None:
        watcher.join(timeout=reader_join_timeout)
    if watcher.is_alive():
        fail("CAPTURE_BUDGET_WATCH_UNJOINED")
        cleanup_error = cleanup_error or "CAPTURE_BUDGET_WATCH_UNJOINED"
    if clock() >= deadline:
        fail("CAPTURE_DEADLINE")
    if prefix_bytes is not None and prefix_bytes != state["observed"]:
        fail("CAPTURE_PREFIX_LENGTH_MISMATCH")
    if prefix_sha is not None and prefix_sha != state["observed_hash"].hexdigest():
        fail("CAPTURE_PREFIX_CONTENT_MISMATCH")
    if fault_writer is not None:
        fault_writer.join(timeout=reader_join_timeout)
    fault_writer_joined = fault_writer is None or not fault_writer.is_alive()
    if not fault_writer_joined:
        cleanup_error = cleanup_error or "CAPTURE_FAULT_WRITER_UNJOINED"
    prefix_reader_joined = prefix_reader is None or not prefix_reader.is_alive()
    quiescent = (
        process_reader_quiet
        and not watcher.is_alive()
        and fault_writer_joined
        and prefix_reader_joined
    )
    if not quiescent:
        prefix_bytes = None
        prefix_sha = None
    with state_lock:
        observed = state["observed"]
        observed_sha = state["observed_hash"].hexdigest()
        eof = state["eof"]
        fault = state["fault"]
        error_record = state["exception"]
        persistence_error = state["fault_persistence_error"]
        maximum_read = state["maximum_read"]
    full_known = eof and reader_joined
    valid = (
        process is not None
        and worker_joined
        and reader_joined
        and quiescent
        and eof
        and worker_exit_code == 0
        and fault is None
        and not persistence_error
        and prefix_bytes == observed
        and prefix_sha == observed_sha
    )
    return {
        "status": "complete_valid" if valid else "INCONCLUSIVE",
        "technical_recording_fault": fault,
        "cleanup_error": cleanup_error,
        "exception": error_record,
        "fault_persistence_error": persistence_error,
        "worker_started": process is not None,
        "worker_exit_code": worker_exit_code,
        "worker_joined": worker_joined,
        "reader_joined": reader_joined,
        "budget_watcher_joined": not watcher.is_alive(),
        "fault_writer_joined": fault_writer_joined,
        "prefix_reader_joined": prefix_reader_joined,
        "quiescent": quiescent,
        "termination_attempted": termination_attempted,
        "kill_attempted": kill_attempted,
        "terminated": termination_attempted and worker_joined,
        "process_attempts": process_attempts,
        "captured_prefix_bytes": prefix_bytes,
        "captured_prefix_sha256": prefix_sha,
        "total_observed_bytes": observed,
        "observed_bytes_sha256": observed_sha,
        "observed_snapshot_stable": reader_joined,
        "eof_observed": eof,
        "unread_tail_possible": not full_known,
        "full_output_bytes": observed if full_known else None,
        "full_output_sha256": observed_sha if full_known else None,
        "log_cap_bytes": LOG_CAP_BYTES,
        "read_chunk_bytes": READ_CHUNK_BYTES,
        "maximum_read_bytes": maximum_read,
        "maximum_read_overshoot_bytes": READ_CHUNK_BYTES,
        "observed_read_overshoot_bytes": max(0, observed - LOG_CAP_BYTES),
        "elapsed_seconds": max(0.0, clock() - started),
    }
