"""One supervised fixed rational witness child; no model, writes, or retry."""
import base64
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def capture(raw):
    value = {"bytes": len(raw), "sha256": digest(raw)}
    try:
        value["utf8"] = raw.decode("utf-8")
        assert value["utf8"].encode("utf-8") == raw
    except UnicodeDecodeError:
        value["base64"] = base64.b64encode(raw).decode("ascii")
    return value


def main():
    process = None
    stdout, stderr = b"", b""
    child_started = None
    cleanup_started = None
    error = None
    cleanup_error = None
    timed_out = False
    output_complete = False
    receipt = {
        "schema": "sp_lense.retention_rational_witness_supervision.v1",
        "child_limit_seconds": 60,
        "cleanup_limit_seconds": 15,
        "numerical_invocations_in_this_job": 0,
        "prior_witness_invocations_in_this_job": 0,
        "retry_performed": False,
    }
    try:
        assert len(sys.argv) == 2, "expected prospective lock SHA256"
        raw_lock = (HERE / "lock.json").read_bytes()
        assert digest(raw_lock) == sys.argv[1], "prospective lock bytes"
        lock = json.loads(raw_lock)
        assert lock["child_limit_seconds"] == 60 and lock["cleanup_limit_seconds"] == 15
        for name, expected in lock["frozen_sources"].items():
            assert digest((ROOT / name).read_bytes()) == expected, "frozen source: " + name
        receipt["lock_sha256"] = digest(raw_lock)
        receipt["source_sha256"] = lock["frozen_sources"]
        command = [sys.executable, "-I", "-B", str(HERE / "witness.py")]
        receipt["command"] = command
        child_started = time.monotonic()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        receipt["numerical_invocations_in_this_job"] = 1
        stdout, stderr = process.communicate(
            timeout=max(0.0, 60 - (time.monotonic() - child_started))
        )
        output_complete = True
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        error = type(exc).__name__ + ": " + str(exc)
        stdout, stderr = exc.output or b"", exc.stderr or b""
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)
    finally:
        if process is not None and process.poll() is None:
            cleanup_started = time.monotonic()
            try:
                process.kill()
                stdout, stderr = process.communicate(
                    timeout=max(0.0, 15 - (time.monotonic() - cleanup_started))
                )
                output_complete = True
            except BaseException as exc:
                cleanup_error = type(exc).__name__ + ": " + str(exc)
                if isinstance(exc, subprocess.TimeoutExpired):
                    stdout = exc.output or stdout
                    stderr = exc.stderr or stderr
        elif process is not None and not output_complete:
            # Child may exit between a timeout and poll; still join stream readers.
            cleanup_started = time.monotonic()
            try:
                stdout, stderr = process.communicate(timeout=15)
                output_complete = True
            except BaseException as exc:
                cleanup_error = type(exc).__name__ + ": " + str(exc)
                if isinstance(exc, subprocess.TimeoutExpired):
                    stdout = exc.output or stdout
                    stderr = exc.stderr or stderr
        finished = time.monotonic()
        receipt.update({
            "launcher_started_monotonic": STARTED,
            "child_started_monotonic": child_started,
            "finished_monotonic": finished,
            "whole_process_elapsed_seconds_in_finally": finished - STARTED,
            "child_through_cleanup_elapsed_seconds":
                None if child_started is None else finished - child_started,
            "supervised_cleanup_seconds":
                0.0 if cleanup_started is None else finished - cleanup_started,
            "finally_sample": "After child exit/join and stream capture; excludes wrapper receipt encoding. Child rational checking and result serialization are included.",
            "child_timed_out": timed_out,
            "error": error,
            "cleanup_error": cleanup_error,
            "child_returncode": None if process is None else process.poll(),
            "child_joined": process is not None and process.poll() is not None,
            "stdout_stderr_complete": output_complete,
            "stdout": capture(stdout),
            "stderr": capture(stderr),
        })
        receipt["status"] = (
            "CHILD_COMPLETED"
            if process is not None and process.returncode == 0
            and error is None and cleanup_error is None and output_complete
            else "FAILED_NO_RETRY"
        )
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
    return 0 if receipt["status"] == "CHILD_COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
