"""Authoritative retained-owned-process outer deadlines; no PID cleanup lookup."""
import json
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
from support import HERE, ROOT, ownership, require, sha, write_new


def supervise(lane, deadline, absolute_cleanup_deadline, config):
    native,owned = ownership()
    started = time.monotonic()
    command = [sys.executable,"-B",str(HERE/"entry.py"),lane,str(deadline)]
    for image,key in (("launch_image","launch_sha256"),("base_image","base_sha256")):
        require(sha(__import__("pathlib").Path(config[image]).read_bytes()) == config[key], "prelocked executable identity")
    require(os.path.normcase(sys.executable) == os.path.normcase(config["launch_image"]), "fixed venv launcher")
    output = "owned/"+lane+"/"
    write_new(output+"RUN_STARTED.json",{"command":command,"started_monotonic":started,"absolute_deadline":deadline,
        "absolute_cleanup_deadline":absolute_cleanup_deadline,"fake_only":True})
    proc = actual = launcher = pair = handshake = reader = None
    events,stream_parts,stream_errors = [],[],[]
    error = None
    receipt = {"lane":lane,"quiescent":False,"primary_error":None,"real_model_process":False}

    def event(kind, data):
        require(len(events) < 64, "bounded ownership events")
        row = {"kind":kind,"data":data,"monotonic":time.monotonic()}
        events.append(row)
        if kind == "authenticated_before_permission":
            # Durable identity before permission; no substantive work on a blocked ACK.
            write_new(output+"PROCESS_IDENTITY.json",row)

    try:
        require(time.monotonic() < deadline, "never launch after outer cutoff")
        proc = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        launcher = native.NativeHandle(int(proc._handle),"owned_launcher_original",True)
        nonce = secrets.token_hex(32)
        received = queue.Queue(maxsize=1)

        def receive():
            try:
                received.put(proc.stdout.readline(8193))
            except BaseException as exc:
                received.put(exc)

        handshake = threading.Thread(target=receive,name="delta-private-handshake",daemon=True)
        handshake.start()
        proc.stdin.write(json.dumps({"nonce":nonce}).encode()+b"\n")
        proc.stdin.flush()
        raw = received.get(timeout=min(3.,max(.001,deadline-time.monotonic())))
        if isinstance(raw,BaseException):
            raise raw
        handshake.join(.1)
        require(not handshake.is_alive() and 0 < len(raw) <= 8192 and raw.endswith(b"\n"), "one bounded private claim")
        claim = owned.strict_json(raw)
        require(claim.get("kind") == "claim" and claim.get("nonce") == nonce and type(claim.get("pid")) is int
                and claim["pid"] > 0, "claim authenticated before opening actual handle")
        actual = native.open_claimed_worker(claim["pid"])
        pair = owned.OwnedPair(launcher,actual,event)

        def acknowledge(frame):
            require(time.monotonic() < deadline, "no ACK beyond cutoff")
            envelope = {"ownership":frame,"lane":lane,"absolute_deadline":deadline,
                        "source_freeze_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes())}
            proc.stdin.write(json.dumps(envelope).encode()+b"\n")
            proc.stdin.flush()

        pair.authorize([claim],config,nonce,os.getpid(),acknowledge)
        proc.stdin.close()

        def drain():
            total = 0
            try:
                while True:
                    raw = proc.stdout.read(4096)
                    if not raw:
                        return
                    stream_parts.append(raw)
                    total += len(raw)
                    if total > 65536:
                        stream_errors.append("stdout exceeded 64 KiB; complete received chunk retained")
                        return
            except BaseException as exc:
                stream_errors.append(type(exc).__name__+": "+str(exc))

        reader = threading.Thread(target=drain,name="delta-owned-output",daemon=True)
        reader.start()
        while True:
            if stream_errors:
                pair.stop_and_wait("capture_limit")
            state = pair.observe(deadline)
            if state in ("STOPPED","BOTH_EXITED"):
                break
            if time.monotonic() >= absolute_cleanup_deadline:
                raise TimeoutError("absolute cleanup deadline exceeded without exit proof")
            time.sleep(.01)
        require(pair.observe_completion(require_exit=True), "both retained exit proofs")
        proc.wait(timeout=0)
    except BaseException as exc:
        error = {"type":type(exc).__name__,"message":str(exc)}
    finally:
        cleanup_started = time.monotonic()
        if proc is not None:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            if pair is not None:
                if not pair.observe_completion():
                    pair.stop_and_wait("outer_failure_cleanup")
                pair.observe_completion(require_exit=True)
            elif launcher is not None:
                # Only our retained launcher is targetable before authentication.
                if not launcher.exited():
                    launcher.terminate()
                launcher.exit_proof(min(1.,max(0.,absolute_cleanup_deadline-time.monotonic())))
            for thread in (handshake,reader):
                if thread is not None:
                    thread.join(min(1.,max(0.,absolute_cleanup_deadline-time.monotonic())))
            joined = all(t is None or not t.is_alive() for t in (handshake,reader))
            if joined and proc.stdout and not proc.stdout.closed:
                proc.stdout.close()
            pipes = proc.stdin.closed and proc.stdout.closed
            quiet = pair is not None and pair.quiescent(joined,pipes)
            if quiet:
                proc.wait(timeout=0)
            receipt.update(primary_error=error,binding_authenticated=pair is not None and pair.authenticated,
                quiescent=quiet,threads_joined=joined,pipes_closed=pipes,
                faults=[] if pair is None else pair.faults,cleanup_faults=[] if pair is None else pair.cleanup_faults,
                stop_reason=None if pair is None else pair.stop_reason,termination_attempts=[] if pair is None else pair.termination_attempts,
                exit_proofs={} if pair is None else pair.exit_proofs,
                stop_episode=None if pair is None else {"started_at":pair.stop_started_at,"deadline":pair.stop_deadline,
                    "completed_at":pair.stop_completed_at,"overrun":pair.episode_overrun},
                stdout_capture_errors=stream_errors,authenticated_binding=None if pair is None else pair.binding)
            if actual is not None:
                actual.close()
                receipt["actual_handle_closed"] = actual.closed
            if quiet:
                proc._handle.Close()
                receipt["launcher_original_handle_closed"] = True
        receipt.update(started_monotonic=started,deadline_monotonic=deadline,finished_monotonic=time.monotonic(),
            elapsed_seconds=time.monotonic()-started,cleanup_seconds=time.monotonic()-cleanup_started,
            within_absolute_cleanup_deadline=time.monotonic() <= absolute_cleanup_deadline,
            no_later_pid_lookup=True,no_tree_kill=True,ownership_core_unchanged=True)
        write_new(output+"stdout.log",b"".join(stream_parts),raw=True,critical=True)
        write_new(output+"OWNERSHIP_EVENTS.json",events,critical=True)
        write_new(output+"CAPTURE.json",receipt,critical=True)
    return receipt
