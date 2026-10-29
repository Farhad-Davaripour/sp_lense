import importlib.util
import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = pathlib.Path(__file__).parents[1]
MODULE_PATH = ROOT / "tools" / "monitor_classic_a10_recovery.py"
SPEC = importlib.util.spec_from_file_location("classic_a10_monitor", MODULE_PATH)
MONITOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MONITOR)


def run(start_ms=1_000_000, gpu_state="RUNNING", gpu_result=None,
        final_state="BLOCKED", final_result=None):
    return {
        "run_id": 123,
        "run_name": MONITOR.RUN_NAME,
        "start_time": start_ms,
        "state": {"life_cycle_state": "RUNNING"},
        "tasks": [
            {"task_key": "validate_before_gpu", "run_id": 11,
             "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"}},
            {"task_key": "run_one_partition", "run_id": 22,
             "state": {"life_cycle_state": gpu_state, "result_state": gpu_result}},
            {"task_key": "finalize_partition_mlflow", "run_id": 33,
             "state": {"life_cycle_state": final_state, "result_state": final_result}},
            {"task_key": "release_active_reservation", "run_id": 44,
             "state": {"life_cycle_state": final_state, "result_state": final_result}},
        ],
    }


def reservation(owner="123"):
    return {
        "identity": MONITOR.IDENTITY,
        "intent_sha256": MONITOR.INTENT_SHA,
        "owner": owner,
        "status": "reserved_after_gpu_ready_before_model",
    }


class ClassicA10MonitorTests(unittest.TestCase):
    def test_waits_before_startup_boundary(self):
        self.assertEqual(
            MONITOR.decision(run(), None, None, 2_199_000), "wait")

    def test_cancels_unadmitted_startup_at_boundary(self):
        self.assertEqual(
            MONITOR.decision(run(), None, None, 2_200_000),
            "cancel_startup_timeout",
        )

    def test_scientific_lock_switches_to_operator_clock(self):
        lock = reservation()
        self.assertEqual(
            MONITOR.decision(run(), lock, lock, 3_399_000),
            "scientific_attempt_active",
        )
        self.assertEqual(
            MONITOR.decision(run(), lock, lock, 3_400_000),
            "cancel_operator_stop",
        )
        self.assertEqual(
            MONITOR.decision(run(), lock, lock, 3_700_000),
            "cancel_ceiling",
        )

    def test_unowned_or_stale_lock_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "not owned"):
            MONITOR.decision(run(), reservation(owner="999"), None, 1_100_000)
        stale = reservation()
        stale["intent_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "not owned"):
            MONITOR.decision(run(), stale, None, 1_100_000)

    def test_expected_post_close_pre_exit_window_is_not_cancelled(self):
        self.assertEqual(
            MONITOR.decision(run(), None, reservation(), 1_100_000),
            "scientific_closeout_pending",
        )
        self.assertEqual(
            MONITOR.decision(run(), None, reservation(), 3_400_000),
            "scientific_closeout_pending",
        )
        self.assertEqual(
            MONITOR.decision(run(), None, reservation(), 3_700_000),
            "cancel_ceiling",
        )

    def test_terminal_gpu_without_lock_is_not_retried(self):
        self.assertEqual(
            MONITOR.decision(
                run(gpu_state="INTERNAL_ERROR"), None, None, 1_100_000),
            "gpu_terminal_before_scientific_lock",
        )

    def test_cancellation_targets_gpu_task_not_parent(self):
        self.assertEqual(MONITOR.cancellation_run_id(run()), 22)
        self.assertNotEqual(MONITOR.cancellation_run_id(run()), run()["run_id"])

    def test_downstream_completion_and_release_are_required(self):
        terminal = run(
            gpu_state="TERMINATED", gpu_result="CANCELED",
            final_state="TERMINATED", final_result="SUCCESS")
        terminal["state"]["life_cycle_state"] = "TERMINATED"
        self.assertTrue(MONITOR.downstream_released(
            terminal, None, reservation(), True))
        incomplete = run(gpu_state="TERMINATED", gpu_result="CANCELED")
        self.assertFalse(MONITOR.downstream_released(
            incomplete, None, reservation(), True))
        failed = run(
            gpu_state="TERMINATED", gpu_result="CANCELED",
            final_state="TERMINATED", final_result="FAILED")
        with self.assertRaises(RuntimeError):
            MONITOR.downstream_released(failed, None, reservation(), True)

    def test_unconsumed_terminal_run_requires_no_consumed_receipt(self):
        terminal = run(
            gpu_state="INTERNAL_ERROR", gpu_result="FAILED",
            final_state="TERMINATED", final_result="SUCCESS")
        self.assertTrue(MONITOR.downstream_released(
            terminal, None, None, False))
        with self.assertRaises(RuntimeError):
            MONITOR.downstream_released(
                terminal, None, reservation(), False)

    def test_wrong_run_or_task_set_fails_closed(self):
        bad = run()
        bad["run_name"] = "other"
        with self.assertRaises(ValueError):
            MONITOR.decision(bad, None, None, 1_100_000)
        bad = run()
        bad["tasks"].pop()
        with self.assertRaises(ValueError):
            MONITOR.decision(bad, None, None, 1_100_000)

    def test_profile_is_pinned(self):
        self.assertEqual(MONITOR.PROFILE, "azure-research4")

    def test_optional_cat_distinguishes_absence_from_provider_error(self):
        with patch.object(MONITOR.subprocess, "run", return_value=SimpleNamespace(
                returncode=1, stdout="", stderr="Error: file does not exist: /x")):
            self.assertIsNone(MONITOR._cat_optional("dbx", "dbfs:/x"))
        with patch.object(MONITOR.subprocess, "run", return_value=SimpleNamespace(
                returncode=1, stdout="", stderr="PERMISSION_DENIED")):
            with self.assertRaisesRegex(RuntimeError, "failed closed"):
                MONITOR._cat_optional("dbx", "dbfs:/x")
        with patch.object(MONITOR.subprocess, "run", return_value=SimpleNamespace(
                returncode=1, stdout="", stderr="network timeout")):
            with self.assertRaisesRegex(RuntimeError, "failed closed"):
                MONITOR._cat_optional("dbx", "dbfs:/x")


if __name__ == "__main__":
    unittest.main()
