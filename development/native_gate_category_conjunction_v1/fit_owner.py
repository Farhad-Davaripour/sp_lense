"""Small Windows retained owner. CLI denies execution without exact root hash inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / "native_oracle_confirmation_preparation_owner_v1/windows_job.py"
HELPER_SHA256 = "a6334fea1678773ff169aed39c3b1939c57614d0096a446e6d0418abcb21368a"
STREAM_CAP = 64 * 1024
TERMINAL_RESERVATION = 256 * 1024
WORKER_CAP = 1024 * 1024
TOTAL_CAP = 8 * 1024 * 1024
PER_FILE = 5 * 1024 * 1024


def require(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    require(path.stat().st_size <= PER_FILE, "INPUT_SIZE")
    return path.read_bytes()


def _job_class():
    raw = read(HELPER)
    require(sha(raw) == HELPER_SHA256, "HELPER_CHANGED")
    module = types.ModuleType("fit_owned_windows_job")
    module.__file__ = str(HELPER)
    # Execute the authenticated bytes; native() is never called.
    exec(compile(raw, str(HELPER), "exec"), module.__dict__)
    return module.Job


def _run_owned(command, destination, seconds=60.0, cleanup_seconds=5.0):
    """Private fake-child test seam. Production command is fixed by launch()."""
    started = time.monotonic()
    deadline = started + seconds
    require(os.name == "nt", "WINDOWS_ONLY")
    require(0 < seconds <= 60 and 0 < cleanup_seconds <= 5, "TIME_BOUNDS")
    destination.mkdir(exist_ok=False)
    terminal = (destination / "TERMINAL.json").open("xb+")
    terminal.write(b" " * (TERMINAL_RESERVATION - 16384))
    terminal.flush()
    os.fsync(terminal.fileno())
    evidence = {"schema": "native_gate_fit_owner_v1", "pid": None,
                "assigned_before_resume": False, "timed_out": False,
                "output_limit": False, "exit_code": None, "errors": [],
                "termination": None, "job_empty": False, "job_closed": False,
                "process_handle_closed": False, "scientific_result_preserved": True,
                "worker_seconds": seconds, "cleanup_seconds": cleanup_seconds,
                "terminal_reservation_bytes": TERMINAL_RESERVATION,
                "command": command, "owner_sha256": sha(read(Path(__file__))),
                "helper_sha256": HELPER_SHA256, "pre_cleanup_job_pids": []}
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    guard = threading.Lock()
    overflow = threading.Event()
    threads = []
    process = None
    job = None

    def drain(stream, name):
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                with guard:
                    space = STREAM_CAP - len(buffers[name])
                    buffers[name].extend(chunk[:space])
                    if len(chunk) > space:
                        overflow.set()
        except Exception:
            overflow.set()

    try:
        job = _job_class()()
        require(time.monotonic() < deadline, "DEADLINE_BEFORE_LAUNCH")
        process = subprocess.Popen(command, cwd=HERE, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   creationflags=0x00000004 | subprocess.CREATE_NO_WINDOW,
                                   close_fds=True)
        evidence["pid"] = process.pid
        job.assign_resume(int(process._handle))
        evidence["assigned_before_resume"] = True
        for name in buffers:
            thread = threading.Thread(target=drain, args=(getattr(process, name), name), daemon=True)
            thread.start()
            threads.append(thread)
        while True:
            evidence["exit_code"] = process.poll()
            if time.monotonic() >= deadline:
                evidence["timed_out"] = True
                break
            if evidence["exit_code"] is not None:
                break
            if overflow.is_set():
                evidence["output_limit"] = True
                break
            if time.monotonic() >= deadline:
                evidence["timed_out"] = True
                break
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
    except Exception as error:
        evidence["errors"].append(type(error).__name__)
        if time.monotonic() >= deadline:
            evidence["timed_out"] = True
    finally:
        cleanup_started = time.monotonic()
        cleanup_deadline = cleanup_started + cleanup_seconds
        # A single shared cleanup deadline covers descendants, parent and pipe readers.
        if job is not None:
            try:
                evidence["pre_cleanup_job_pids"] = job.pids()
            except Exception as error:
                evidence["errors"].append("pre_cleanup_query:" + type(error).__name__)
            try:
                evidence["termination"] = job.terminate()
                while time.monotonic() < cleanup_deadline:
                    if not job.pids():
                        evidence["job_empty"] = True
                        break
                    time.sleep(min(0.01, max(0.0, cleanup_deadline - time.monotonic())))
            except Exception as error:
                evidence["errors"].append("job_cleanup:" + type(error).__name__)
            finally:
                try:
                    job.close()  # kill-on-close remains the fallback if termination fails.
                    evidence["job_closed"] = job.closed
                except Exception as error:
                    evidence["errors"].append("job_close:" + type(error).__name__)
        if process is not None:
            try:
                if process.poll() is None and not evidence["assigned_before_resume"]:
                    process.kill()  # owned retained handle, including assignment failure.
                evidence["exit_code"] = process.wait(timeout=max(0.0, cleanup_deadline - time.monotonic()))
            except Exception as error:
                evidence["errors"].append("parent_cleanup:" + type(error).__name__)
        for thread in threads:
            thread.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
        evidence["readers_closed"] = all(not thread.is_alive() for thread in threads)
        if process is not None:
            if evidence["readers_closed"]:
                for name in buffers:
                    try:
                        getattr(process, name).close()
                    except Exception as error:
                        evidence["errors"].append("pipe_close:" + type(error).__name__)
            try:
                process._handle.Close()
                evidence["process_handle_closed"] = True
            except Exception as error:
                evidence["errors"].append("process_close:" + type(error).__name__)
        evidence["cleanup_elapsed_seconds"] = time.monotonic() - cleanup_started
        evidence["cleanup_within_budget"] = time.monotonic() <= cleanup_deadline
        evidence["elapsed_seconds"] = time.monotonic() - started
        evidence["output_limit"] = evidence["output_limit"] or overflow.is_set()
        evidence["technical_complete"] = bool(evidence["assigned_before_resume"] and
            evidence["job_empty"] and evidence["job_closed"] and evidence["process_handle_closed"] and
            evidence["readers_closed"] and evidence["cleanup_within_budget"] and
            not evidence["errors"] and not evidence["timed_out"] and not evidence["output_limit"])
        # Scientific files are never overwritten, deleted, or reclassified by cleanup.
        with guard:
            snapshots = {name: bytes(value) for name, value in buffers.items()}
        for name, raw in snapshots.items():
            try:
                with (destination / (name + ".log")).open("xb") as stream:
                    stream.write(raw)
            except Exception as error:
                evidence["errors"].append("log_write:" + type(error).__name__)
                evidence["technical_complete"] = False
        evidence["process_technical_complete"] = evidence.pop("technical_complete")
        raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("ascii")
        require(len(raw) < TERMINAL_RESERVATION - 16384, "TERMINAL_RESERVATION")
        try:
            terminal.seek(0)
            terminal.write(raw)
            terminal.truncate()
            terminal.flush()
            os.fsync(terminal.fileno())
        finally:
            terminal.close()
        # No stored PASS precedes publication. This separate receipt describes
        # completed terminal/log writes; its own post-close timing is returned to
        # the retained root caller, since a file cannot attest its future closure.
        sizes = [path.stat().st_size for path in destination.iterdir() if path.is_file()]
        finalization = {"terminal_sha256": sha(raw),
                        "elapsed_seconds_after_terminal": time.monotonic() - started,
                        "cleanup_seconds_after_terminal": time.monotonic() - cleanup_started,
                        "deadline_fault": time.monotonic() > cleanup_deadline,
                        "storage_fault": sum(sizes) + 16384 > 2 * STREAM_CAP + TERMINAL_RESERVATION or any(size > PER_FILE for size in sizes),
                        "receipt_own_closure_attested": False}
        receipt_raw = json.dumps(finalization, sort_keys=True).encode("ascii")
        require(len(receipt_raw) < 16384, "FINALIZATION_RESERVATION")
        with (destination / "FINALIZATION.json").open("xb") as stream:
            stream.write(receipt_raw)
            stream.flush()
            os.fsync(stream.fileno())
        evidence["finalization"] = finalization
        evidence["elapsed_seconds"] = time.monotonic() - started
        evidence["cleanup_elapsed_seconds"] = time.monotonic() - cleanup_started
        evidence["cleanup_within_budget"] = time.monotonic() <= cleanup_deadline
        evidence["technical_complete"] = (evidence["process_technical_complete"] and
            evidence["cleanup_within_budget"] and not finalization["storage_fault"])
    return evidence


def launch(*, root_approved=False, owner_sha256=None, helper_sha256=None,
           release=None, release_sha256=None):
    require(root_approved is True and owner_sha256 and helper_sha256 and release and release_sha256,
            "ROOT_RELEASE_REQUIRED")
    require(sha(read(Path(__file__))) == owner_sha256, "OWNER_CHANGED")
    require(helper_sha256 == HELPER_SHA256 and sha(read(HELPER)) == helper_sha256, "HELPER_CHANGED")
    release_path = Path(release).resolve()
    release_raw = read(release_path)
    require(sha(release_raw) == release_sha256, "RELEASE_HASH")
    approval = json.loads(release_raw)
    require(approval.get("approved") is True and approval.get("operation") == "one_construction_fit",
            "RELEASE_DEFAULT_DENY")
    require(all(approval.get(key) is False for key in
                ("model_permission", "tokenizer_permission", "evaluation_permission")), "RELEASE_SCOPE")
    for filename, key in (("CORE_SOURCE_LOCK.json", "source_sha256"),
                          ("TRAINING_MANIFEST.json", "training_manifest_sha256"),
                          ("CONSTRUCTION_LOCK_DRAFT.json", "construction_lock_sha256")):
        require(sha(read(HERE / filename)) == approval.get(key), "RELEASE_BINDING")
    sources = json.loads(read(HERE / "CORE_SOURCE_LOCK.json"))
    require(set(sources) == {"gate.py", "checker.py", "source_auth.py", "construction.py"}, "SOURCE_SET")
    for name, expected in sources.items():
        require(sha(read(HERE / name)) == expected, "CORE_SOURCE_CHANGED")
    require(not (HERE / "construction_attempt_001").exists(), "NO_RETRY")
    # Core write_new independently caps its complete attempt at 1 MiB. Streams and
    # terminal reservation add 384 KiB; all are below both public storage ceilings.
    require(WORKER_CAP + 2 * STREAM_CAP + TERMINAL_RESERVATION < TOTAL_CAP, "STORAGE_BUDGET")
    command = [sys.executable, "-B", "-E", "-S", str(HERE / "construction.py"), "fit",
               "--release", str(release_path), "--release-sha256", release_sha256]
    return _run_owned(command, HERE / "fit_owner_attempt_001")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-approved", action="store_true")
    for name in ("owner-sha256", "helper-sha256", "release", "release-sha256"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    result = launch(root_approved=args.root_approved, owner_sha256=args.owner_sha256,
                    helper_sha256=args.helper_sha256, release=args.release,
                    release_sha256=args.release_sha256)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["technical_complete"] and result["exit_code"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
