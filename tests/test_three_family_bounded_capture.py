"""Model-free tests of the new parent-owned bounded binary PIPE capture only."""

import hashlib
import io
import subprocess
import threading
import time

import pytest

from scripts import three_family_bounded_capture as capture


class Budget:
    def __init__(self, root):
        self.root = root
        self._fault = None
        self.lock = threading.Lock()
        self.writes = []
        self.append_hook = None
        self.fault_hook = None
        self.read_fault_hook = None

    @property
    def fault_code(self):
        if self.read_fault_hook:
            self.read_fault_hook()
        with self.lock:
            return self._fault

    def fault(self, code):
        with self.lock:
            self._fault = self._fault or code
        if self.fault_hook:
            self.fault_hook()

    def write_bytes(self, name, payload, mode="ab"):
        assert name == "worker.log"
        assert type(payload) is bytes
        if mode == "ab" and self.append_hook:
            self.append_hook(payload)
        with self.lock:
            if self._fault:
                raise RuntimeError("recording is stopped")
            self.writes.append((name, mode, len(payload)))
            with (self.root / name).open(mode) as handle:
                handle.write(payload)


class Pipe:
    def __init__(self, data=b"", *, piece=None, read_hook=None):
        self.buffer = io.BytesIO(data)
        self.piece = piece
        self.read_hook = read_hook
        self.requests = []
        self.closed = False

    def read(self, size):
        self.requests.append(size)
        if self.read_hook:
            self.read_hook()
        return self.buffer.read(min(size, self.piece) if self.piece else size)

    def close(self):
        self.closed = True


class Process:
    def __init__(self, pipe, *, live=False, stubborn=False, termination_error=False):
        self.stdout = pipe
        self.returncode = None if live else 0
        self.stubborn = stubborn
        self.termination_error = termination_error
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts = []

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake worker", timeout)
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        if self.termination_error:
            raise OSError("injected terminate failure")
        if not self.stubborn:
            self.returncode = -15

    def kill(self):
        self.kill_calls += 1
        if self.stubborn:
            raise OSError("injected kill failure")
        self.returncode = -9


def run(tmp_path, payload=b"hello\x00\xff", *, budget=None, process=None, **kwargs):
    budget = budget or Budget(tmp_path)
    process = process or Process(Pipe(payload))
    calls = []

    def factory(command, **options):
        calls.append((command, options))
        assert (tmp_path / "worker.log").is_file()
        assert (tmp_path / "worker.log").stat().st_size == 0
        return process

    options = {
        "deadline": time.monotonic() + 10,
        "poll_interval": 0.001,
        "terminate_timeout": 0.02,
        "kill_timeout": 0.02,
        "reader_join_timeout": 0.1,
    }
    options.update(kwargs)
    receipt = capture.run_capture(
        ["fake-worker"], budget, process_factory=factory, cwd=tmp_path, **options
    )
    return receipt, budget, process, calls


@pytest.mark.parametrize(
    "payload",
    [b"", b"\x00\xff\xfe\r\n", bytes(range(256)) * 400],
    ids=["empty", "binary", "chunks"],
)
def test_binary_output_complete_and_parent_pipe_only(tmp_path, payload):
    result, budget, process, calls = run(tmp_path, payload)
    assert result["status"] == "complete_valid"
    assert result["quiescent"] and result["worker_joined"] and result["reader_joined"]
    assert result["technical_recording_fault"] is None
    assert result["captured_prefix_bytes"] == result["full_output_bytes"] == len(payload)
    assert result["captured_prefix_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["full_output_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["total_observed_bytes"] == len(payload)
    assert result["eof_observed"] and not result["unread_tail_possible"]
    assert result["process_attempts"] == 1
    assert len(calls) == 1
    assert (tmp_path / "worker.log").read_bytes() == payload
    assert budget.writes[0] == ("worker.log", "xb", 0)
    opts = calls[0][1]
    assert opts["stdout"] == subprocess.PIPE
    assert opts["stderr"] == subprocess.STDOUT
    assert opts["stdin"] == subprocess.DEVNULL
    assert opts["bufsize"] == 0
    assert opts["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert max(process.stdout.requests) == 65536
    assert process.stdout.closed


def test_exact_four_mib_is_valid_only_after_next_read_observes_eof(tmp_path):
    payload = b"x" * capture.LOG_CAP_BYTES
    result, _, process, _ = run(tmp_path, payload)
    assert result["status"] == "complete_valid"
    assert result["captured_prefix_bytes"] == 4 * 1024 * 1024
    assert len(process.stdout.requests) == capture.LOG_CAP_BYTES // 65536 + 1
    assert result["eof_observed"]


@pytest.mark.parametrize("extra", [1, 17, 65536, 65537, 3 * 65536])
def test_cap_preserves_exact_prefix_and_never_claims_unread_tail(tmp_path, extra):
    cap = capture.LOG_CAP_BYTES
    payload = b"x" * cap + b"z" * extra
    result, budget, process, _ = run(tmp_path, payload)
    assert result["status"] == "INCONCLUSIVE"
    assert result["technical_recording_fault"] == "CAPTURE_LOG_CAP"
    assert budget.fault_code == "CAPTURE_LOG_CAP"
    assert result["captured_prefix_bytes"] == cap
    assert result["captured_prefix_sha256"] == hashlib.sha256(payload[:cap]).hexdigest()
    assert result["total_observed_bytes"] == cap + min(extra, 65536)
    assert result["total_observed_bytes"] <= cap + result["maximum_read_overshoot_bytes"]
    assert result["maximum_read_bytes"] <= 65536
    assert not result["eof_observed"] and result["unread_tail_possible"]
    assert result["full_output_bytes"] is result["full_output_sha256"] is None
    assert result["quiescent"]
    assert (tmp_path / "worker.log").read_bytes() == payload[:cap]
    assert len(process.stdout.requests) == cap // 65536 + 1


def test_partial_last_chunk_writes_fitting_prefix_before_fault(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "LOG_CAP_BYTES", 100)
    process = Process(Pipe(bytes(range(200)), piece=70), live=True)
    result, budget, process, _ = run(tmp_path, process=process)
    assert result["captured_prefix_bytes"] == 100
    assert result["total_observed_bytes"] == 140
    assert [entry[2] for entry in budget.writes] == [0, 70, 30]
    assert (tmp_path / "worker.log").read_bytes() == bytes(range(100))
    assert process.terminate_calls == 1 and process.kill_calls == 0
    assert result["terminated"] and result["worker_joined"]


def test_partial_storage_write_receipt_hashes_actual_disk_prefix(tmp_path):
    budget = Budget(tmp_path)

    def partial(payload):
        with (tmp_path / "worker.log").open("ab") as handle:
            handle.write(payload[:7])
        raise OSError("partial write")

    budget.append_hook = partial
    result, _, _, _ = run(tmp_path, b"abcdefghijklmnopqrstuvwxyz", budget=budget)
    assert result["status"] == "INCONCLUSIVE"
    assert result["captured_prefix_bytes"] == 7
    assert result["captured_prefix_sha256"] == hashlib.sha256(b"abcdefg").hexdigest()
    assert result["total_observed_bytes"] == 26
    assert result["technical_recording_fault"] == "CAPTURE_READ_OR_WRITE_ERROR"
    assert not result["eof_observed"]


def test_existing_log_rejects_before_process_attempt_without_retry(tmp_path):
    first, budget, _, _ = run(tmp_path)
    assert first["status"] == "complete_valid"
    original = (tmp_path / "worker.log").read_bytes()
    second, _, _, calls = run(tmp_path, budget=budget)
    assert second["status"] == "INCONCLUSIVE"
    assert second["process_attempts"] == 0 and not second["worker_started"]
    assert not calls
    assert (tmp_path / "worker.log").read_bytes() == original


def test_failed_spawn_records_one_attempt_not_zero(tmp_path):
    calls = []

    def fail_spawn(*args, **kwargs):
        calls.append(1)
        raise OSError("launch failed")

    result = capture.run_capture(
        ["fake"], Budget(tmp_path), time.monotonic() + 2, process_factory=fail_spawn
    )
    assert calls == [1]
    assert result["status"] == "INCONCLUSIVE"
    assert result["process_attempts"] == 1 and not result["worker_started"]
    assert result["quiescent"]


def test_nonzero_exit_is_not_complete_even_with_complete_log(tmp_path):
    process = Process(Pipe(b"complete output"))
    process.returncode = 7
    result, _, _, _ = run(tmp_path, process=process)
    assert result["status"] == "INCONCLUSIVE"
    assert result["worker_exit_code"] == 7
    assert result["full_output_bytes"] == len(b"complete output")
    assert result["technical_recording_fault"] == "CAPTURE_WORKER_NONZERO_EXIT"


def test_eof_does_not_substitute_for_process_termination(tmp_path):
    process = Process(Pipe(b"finished printing"), live=True)
    result, _, _, _ = run(tmp_path, process=process, deadline=time.monotonic() + 0.02)
    assert result["status"] == "INCONCLUSIVE"
    assert result["technical_recording_fault"] == "CAPTURE_DEADLINE"
    assert result["eof_observed"]
    assert process.terminate_calls == 1 and result["worker_joined"]


def test_terminate_failure_escalates_once_and_does_not_retry_worker(tmp_path):
    process = Process(Pipe(b""), live=True, termination_error=True)
    result, _, _, calls = run(tmp_path, process=process, deadline=time.monotonic() + 0.02)
    assert process.terminate_calls == process.kill_calls == 1
    assert result["worker_joined"] and result["kill_attempted"]
    assert result["status"] == "INCONCLUSIVE" and len(calls) == 1


def test_failed_termination_never_claims_quiescence(tmp_path):
    process = Process(Pipe(b""), live=True, stubborn=True)
    result, _, _, _ = run(tmp_path, process=process, deadline=time.monotonic() + 0.02)
    assert process.terminate_calls == process.kill_calls == 1
    assert result["status"] == "INCONCLUSIVE"
    assert not result["worker_joined"] and not result["quiescent"]
    assert result["cleanup_error"] == "CAPTURE_WORKER_UNJOINED"
    assert result["captured_prefix_bytes"] is result["captured_prefix_sha256"] is None


@pytest.mark.parametrize("block", ["reader", "writer", "fault", "budget_state"])
def test_blocked_helpers_are_bounded_and_never_false_quiescent(tmp_path, block):
    release = threading.Event()
    entered = threading.Event()

    def wait_here(*args):
        entered.set()
        release.wait(timeout=3)

    budget = Budget(tmp_path)
    pipe = Pipe(b"payload")
    if block == "reader":
        pipe.read_hook = wait_here
    elif block == "writer":
        budget.append_hook = wait_here
    elif block == "fault":
        budget.fault_hook = wait_here
    else:
        budget.read_fault_hook = wait_here
    process = Process(pipe, live=True)
    started = time.monotonic()
    try:
        result, _, _, _ = run(
            tmp_path,
            budget=budget,
            process=process,
            deadline=time.monotonic() + 0.02,
            reader_join_timeout=0.02,
        )
        assert entered.is_set()
        assert time.monotonic() - started < 0.8
        assert result["status"] == "INCONCLUSIVE" and not result["quiescent"]
        assert result["captured_prefix_bytes"] is result["captured_prefix_sha256"] is None
        if block in ("reader", "writer"):
            assert not result["reader_joined"]
            assert result["unread_tail_possible"] and result["full_output_bytes"] is None
        elif block == "fault":
            assert not result["fault_writer_joined"]
        else:
            assert not result["budget_watcher_joined"]
            assert result["process_attempts"] == 0
    finally:
        release.set()


def test_external_sticky_fault_after_log_write_withholds_complete(tmp_path):
    budget = Budget(tmp_path)
    pipe = Pipe(b"data")
    reads = 0

    def external_fault():
        nonlocal reads
        reads += 1
        if reads == 2:
            budget.fault("EXTERNAL_WRITER_FAULT")

    pipe.read_hook = external_fault
    result, _, _, _ = run(tmp_path, budget=budget, process=Process(pipe))
    assert result["status"] == "INCONCLUSIVE"
    assert result["technical_recording_fault"] == "EXTERNAL_WRITER_FAULT"


def test_final_deadline_check_catches_slow_prefix_verification(tmp_path, monkeypatch):
    value = [100.0]
    original = capture._prefix_on_disk

    def late_prefix(path):
        result = original(path)
        value[0] = 102.0
        return result

    monkeypatch.setattr(capture, "_prefix_on_disk", late_prefix)
    result, _, _, _ = run(tmp_path, clock=lambda: value[0], deadline=101.0)
    assert result["status"] == "INCONCLUSIVE"
    assert result["technical_recording_fault"] == "CAPTURE_DEADLINE"


def test_equal_length_log_tamper_is_sticky_inconclusive(tmp_path, monkeypatch):
    original = capture._prefix_on_disk

    def tampered(path):
        path.write_bytes(b"X" * path.stat().st_size)
        return original(path)

    monkeypatch.setattr(capture, "_prefix_on_disk", tampered)
    result, budget, _, _ = run(tmp_path, b"good")
    assert result["status"] == "INCONCLUSIVE"
    assert result["technical_recording_fault"] == "CAPTURE_PREFIX_CONTENT_MISMATCH"
    assert budget.fault_code == "CAPTURE_PREFIX_CONTENT_MISMATCH"
    assert result["captured_prefix_sha256"] == hashlib.sha256(b"XXXX").hexdigest()
    assert result["full_output_sha256"] == hashlib.sha256(b"good").hexdigest()


def test_blocked_prefix_reader_cannot_claim_quiescence(tmp_path, monkeypatch):
    release = threading.Event()
    original = capture._prefix_on_disk

    def blocked(path):
        release.wait(timeout=3)
        return original(path)

    monkeypatch.setattr(capture, "_prefix_on_disk", blocked)
    try:
        result, _, _, _ = run(tmp_path, reader_join_timeout=0.02)
        assert result["status"] == "INCONCLUSIVE" and not result["quiescent"]
        assert not result["prefix_reader_joined"]
        assert result["captured_prefix_bytes"] is result["captured_prefix_sha256"] is None
    finally:
        release.set()


def test_sticky_fault_persistence_failure_cannot_be_valid(tmp_path):
    budget = Budget(tmp_path)

    def failed_persistence():
        raise OSError("fault state cannot be persisted")

    budget.fault_hook = failed_persistence
    process = Process(Pipe(b""), live=True)
    result, _, _, _ = run(
        tmp_path, budget=budget, process=process, deadline=time.monotonic() + 0.02
    )
    assert result["status"] == "INCONCLUSIVE"
    assert result["fault_persistence_error"] and result["fault_writer_joined"]


def test_bounded_exception_never_invokes_custom_stringification():
    class HostileError(Exception):
        def __str__(self):
            raise AssertionError("arbitrary exception rendering must not execute")

    result = capture.exception_record(HostileError("safe message"))
    assert result["prefix"] == "safe message" and not result["truncated"]
    assert result["full_text_bytes"] == 12
    assert result["full_text_sha256"] == hashlib.sha256(b"safe message").hexdigest()


@pytest.mark.parametrize(
    "message",
    ["x" * 100000, "😀" * 100000, "x" * 4095 + "😀"],
    ids=["long-ascii", "long-unicode", "split-codepoint"],
)
def test_long_exception_prefix_is_bounded_utf8_and_unknown_totals(message):
    result = capture.exception_record(RuntimeError(message))
    assert result["prefix_bytes"] <= 4096
    assert len(result["prefix"].encode("utf-8")) == result["prefix_bytes"]
    assert result["truncated"]
    if len(message) > 4096:
        assert result["full_text_bytes"] is result["full_text_sha256"] is None
    else:
        assert result["full_text_bytes"] == len(message.encode("utf-8"))


def test_exception_non_string_argument_not_rendered():
    class Unrenderable:
        def __str__(self):
            raise AssertionError("not allowed")

        def __repr__(self):
            raise AssertionError("not allowed")

    result = capture.exception_record(RuntimeError(Unrenderable()))
    assert result["prefix"] == ""
    assert result["full_text_bytes"] is result["full_text_sha256"] is None
    assert result["text_source"] == "no_safe_first_string_argument"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "later"])
def test_nonfinite_or_invalid_deadline_rejected_before_writes(tmp_path, value):
    budget = Budget(tmp_path)
    with pytest.raises(ValueError):
        capture.run_capture(["fake"], budget, value)
    assert not budget.writes


@pytest.mark.parametrize("option", ["terminate_timeout", "kill_timeout", "reader_join_timeout"])
def test_cleanup_wait_bounds_are_fixed(tmp_path, option):
    with pytest.raises(ValueError):
        capture.run_capture(["fake"], Budget(tmp_path), time.monotonic() + 2, **{option: 60})
