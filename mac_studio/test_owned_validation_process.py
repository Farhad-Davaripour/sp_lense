"""Darwin CPU-only tests of owned process cleanup; no model or network."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import psutil
import owned_validation_process as group


@unittest.skipUnless(sys.platform == 'darwin', 'Darwin waitid ABI')
class OwnedGroupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='research4_group_test_')
        self.slot = group.LaunchSlot(); self.child_pid = None

    def tearDown(self):
        p = self.slot.process
        if p is not None and p.returncode is None:
            try: os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError): pass
            try: p.wait(timeout=5)
            except subprocess.TimeoutExpired: pass
        self.tmp.cleanup()

    def launch(self, code):
        pending = group.launch(self.slot, [sys.executable, '-c', code],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, cwd=self.tmp.name)
        self.assertFalse(pending)

    def wait_for(self, predicate, seconds=5):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if predicate(): return
            time.sleep(.02)
        self.fail('Synthetic process did not reach expected state')

    def assert_gone(self, pid):
        self.wait_for(lambda: not psutil.pid_exists(pid), seconds=10)

    def test_normal_leader_without_children(self):
        self.launch('pass'); p = self.slot.process
        self.wait_for(lambda: group.peek_exited(p.pid))
        code, members, pending = group.cleanup(p, grace_seconds=.1, kill_seconds=5)
        self.assertEqual(code, 0); self.assertEqual(members, []); self.assertFalse(pending)
        self.assert_gone(p.pid)

    def spawn_ignoring_child(self, leader_exit):
        pidfile = str(Path(self.tmp.name) / 'pid')
        child = ('import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                 f'open({pidfile!r},"w").write(str(os.getpid())); time.sleep(60)')
        leader = ('import os,subprocess,sys,time; '
                  f'subprocess.Popen([sys.executable,"-c",{child!r}], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); '
                  f'p={pidfile!r}; '
                  '[(time.sleep(.02)) for _ in range(250) if not os.path.exists(p)]; '
                  + ('sys.exit(0)' if leader_exit else 'time.sleep(60)'))
        self.launch(leader)
        self.wait_for(lambda: Path(pidfile).exists())
        self.child_pid = int(Path(pidfile).read_text())
        self.assertTrue(psutil.pid_exists(self.child_pid))
        return self.slot.process

    def test_normal_leader_exit_with_sigterm_ignoring_child(self):
        p = self.spawn_ignoring_child(leader_exit=True)
        self.wait_for(lambda: group.peek_exited(p.pid))
        code, members, pending = group.cleanup(p, grace_seconds=.15, kill_seconds=10)
        self.assertEqual(code, 0); self.assertIn(self.child_pid, members); self.assertFalse(pending)
        self.assert_gone(self.child_pid); self.assert_gone(p.pid)

    def test_timeout_leader_exits_but_child_ignores_sigterm(self):
        p = self.spawn_ignoring_child(leader_exit=False)
        code, members, pending = group.cleanup(p, grace_seconds=.15, kill_seconds=10)
        self.assertLess(code, 0); self.assertIn(self.child_pid, members); self.assertFalse(pending)
        self.assert_gone(self.child_pid); self.assert_gone(p.pid)

    def test_signal_during_cleanup_is_deferred_until_group_clean(self):
        p = self.spawn_ignoring_child(leader_exit=False)
        previous = signal.getsignal(signal.SIGTERM)
        def handler(signum, frame): raise InterruptedError('synthetic interruption')
        signal.signal(signal.SIGTERM, handler)
        timer = threading.Timer(.03, lambda: os.kill(os.getpid(), signal.SIGTERM)); timer.start()
        try:
            _, members, pending = group.cleanup(p, grace_seconds=.15, kill_seconds=10)
            self.assertIn(signal.SIGTERM, pending); self.assertIn(self.child_pid, members)
            self.assert_gone(self.child_pid); self.assert_gone(p.pid)
        finally:
            timer.join(timeout=2); signal.signal(signal.SIGTERM, previous)

    def test_uncertain_membership_does_not_reap_anchor(self):
        self.launch('import time; time.sleep(60)'); p = self.slot.process
        with patch.object(group, 'live_group_members', side_effect=RuntimeError('uncertain')):
            with self.assertRaises(RuntimeError): group.cleanup(p, grace_seconds=.1, kill_seconds=.1)
        self.assertIsNone(p.returncode)

    def test_ordinary_signal_during_spawn_is_deferred_until_slot_bound(self):
        fake = object(); slot = group.LaunchSlot()
        def spawn(*args, **kwargs):
            os.kill(os.getpid(), signal.SIGTERM)
            return fake
        with patch.object(group.subprocess, 'Popen', side_effect=spawn):
            pending = group.launch(slot, ['synthetic'], stdin=None, stdout=None, cwd=self.tmp.name)
        self.assertIs(slot.process, fake)
        self.assertIn(signal.SIGTERM, pending)


if __name__ == '__main__': unittest.main()
