"""Darwin-only owned process group cleanup using an unreaped child as anchor.

The leader is observed with waitid(WNOWAIT), never poll()/wait(), until TERM,
KILL and a live-member check complete. This does not contain setsid escapees,
protect against supervisor SIGKILL/host death, or guarantee uninterruptible
kernel tasks can be killed. An uncertain result must retain the caller's lock.
"""
from contextlib import contextmanager
import ctypes
import errno
import os
import signal
import subprocess
import sys
import time

import psutil


class _DarwinSigInfo(ctypes.Structure):
    # Verified against the Darwin SDK's sys/signal.h siginfo_t on arm64.
    _fields_ = [('signo', ctypes.c_int), ('error', ctypes.c_int), ('code', ctypes.c_int),
                ('pid', ctypes.c_int), ('uid', ctypes.c_uint), ('status', ctypes.c_int),
                ('addr', ctypes.c_void_p), ('value', ctypes.c_uint64), ('band', ctypes.c_long),
                ('pad', ctypes.c_ulong * 7)]


def peek_exited(pid):
    if sys.platform != 'darwin' or ctypes.sizeof(_DarwinSigInfo) != 104:
        raise RuntimeError('Unreviewed waitid ABI')
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.waitid
    call.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.POINTER(_DarwinSigInfo), ctypes.c_int]
    call.restype = ctypes.c_int
    while True:
        info = _DarwinSigInfo()
        result = call(os.P_PID, pid, ctypes.byref(info), os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if result == 0: break
        error = ctypes.get_errno()
        if error == errno.EINTR: continue
        raise OSError(error, 'waitid cannot establish ownership')
    if info.pid not in (0, pid): raise RuntimeError('Unexpected waitid child identity')
    return info.pid == pid


def live_group_members(pgid, exclude_pid):
    """Conservatively identify live group members; errors mean unknown."""
    members = []
    for proc in psutil.process_iter(['pid']):
        pid = proc.info['pid']
        if pid == exclude_pid: continue
        try:
            if os.getpgid(pid) != pgid: continue
            # Include zombies: an orphan must disappear from the owned group,
            # not merely stop executing, before the caller releases its lock.
            proc.status()
            members.append(pid)
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
        except (psutil.AccessDenied, PermissionError) as exc:
            raise RuntimeError('Cannot establish process-group membership') from exc
        except OSError as exc:
            if exc.errno == errno.ESRCH: continue
            raise RuntimeError('Cannot establish process-group membership') from exc
    return sorted(members)


@contextmanager
def defer_termination_signals():
    """Defer ordinary termination during spawn and cleanup critical sections."""
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    saved = {sig: signal.getsignal(sig) for sig in signals}; pending = []
    def defer(signum, frame): pending.append(signum)
    try:
        for sig in signals: signal.signal(sig, defer)
        yield pending
    finally:
        for sig, handler in saved.items(): signal.signal(sig, handler)


class LaunchSlot:
    process = None


def launch(slot, command, *, stdin, stdout, cwd):
    """Publish the Popen into an existing slot before restoring signal handlers."""
    with defer_termination_signals() as pending:
        slot.process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=subprocess.STDOUT,
                                        cwd=cwd, start_new_session=True)
    return pending


def cleanup(process, grace_seconds=2.0, kill_seconds=10.0):
    """Close the entire owned group, keeping the leader unreaped until safe.

    Returns (exit_code, descendants_observed, deferred_signals). Any failure to
    establish that the group is clear raises; callers must retain their lock.
    """
    leader = process.pid; descendants_observed = []
    with defer_termination_signals() as pending:
        # Confirm this still is our unreaped child before addressing its group.
        exited = peek_exited(leader)
        descendants_observed = live_group_members(leader, leader)
        # Darwin can return EPERM for a group containing only the zombie
        # leader. If it exited and no live members exist, there is no process
        # capable of forking; retain the anchor and verify clear twice below.
        if not exited or descendants_observed:
            try: os.killpg(leader, signal.SIGTERM)
            except ProcessLookupError:
                if not exited or descendants_observed: raise RuntimeError('Owned group disappeared unexpectedly')
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            exited = peek_exited(leader)
            current = live_group_members(leader, leader)
            descendants_observed = sorted(set(descendants_observed) | set(current))
            if exited and not current: break
            time.sleep(0.05)
        else:
            # Leader remains unreaped even when it exited immediately after TERM.
            try: os.killpg(leader, signal.SIGKILL)
            except ProcessLookupError:
                if not peek_exited(leader) or live_group_members(leader, leader):
                    raise RuntimeError('Owned group lost during escalation')
        deadline = time.monotonic() + kill_seconds; clear_samples = 0
        while time.monotonic() < deadline:
            exited = peek_exited(leader); current = live_group_members(leader, leader)
            descendants_observed = sorted(set(descendants_observed) | set(current))
            clear_samples = clear_samples + 1 if exited and not current else 0
            if clear_samples >= 2: break
            time.sleep(0.05)
        else:
            raise RuntimeError('Owned group did not become clear; retain lock')
        # Only now may subprocess reap the leader and release its identity.
        code = process.wait(timeout=1)
    return code, descendants_observed, pending
