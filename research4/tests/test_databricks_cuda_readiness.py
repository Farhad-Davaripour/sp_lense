import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
import yaml


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "databricks_continuity_partition.py"


def _load_readiness_namespace():
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    assignments = {
        "_CUDA_NF4_READINESS_MARKER", "_CUDA_NF4_RECEIPT_MARKER",
        "_CUDA_NF4_RECEIPT_VERSION", "_CUDA_NF4_BINDING_VERSION",
        "_CUDA_NF4_MAX_ATTEMPTS", "_CUDA_NF4_BACKOFF_SECONDS",
        "_CUDA_NF4_SUBPROCESS_TIMEOUT_SECONDS", "_CUDA_NF4_PHASE_CAP_SECONDS",
        "_CUDA_NF4_PRE_RESERVATION_MARGIN_SECONDS",
        "_CUDA_NF4_PROVIDER_STARTUP_ABORT_SECONDS", "_CUDA_NF4_CHILD",
    }
    functions = {
        "_canonical", "_sha256_text", "_read_cuda_nf4_marker",
        "_bounded_cuda_nf4_readiness",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in assignments
                for target in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
    namespace = {
        "Path": Path, "hashlib": hashlib, "json": json, "os": os,
        "re": re, "subprocess": subprocess, "sys": sys, "time": time,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"),
         namespace)
    return namespace


class Clock:
    def __init__(self, job_elapsed=100.0):
        self.value = 0.0
        self.job_elapsed = job_elapsed
        self.sleeps = []

    def monotonic(self):
        return self.value

    def elapsed(self):
        return self.job_elapsed + self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


class DatabricksCudaReadinessTests(unittest.TestCase):
    def setUp(self):
        self.ns = _load_readiness_namespace()
        self.marker = self.ns["_CUDA_NF4_READINESS_MARKER"]

    def _completed(self, payload, returncode=0, stderr=""):
        return SimpleNamespace(
            returncode=returncode,
            stdout=self.marker + json.dumps(payload, sort_keys=True) + "\n",
            stderr=stderr)

    @staticmethod
    def _ready():
        return {
            "bitsandbytes_version": "0.49.2", "finite": True,
            "hardware": {
                "profile": "databricks-a10x1", "compute_dtype": "float16",
                "adapter_dtype": "float32", "devices": [{
                    "index": 0, "name": "NVIDIA A10", "major": 8, "minor": 6,
                    "total_memory_bytes": 24 * 1024**3,
                }],
            },
            "probe_elements": 256, "quant_type": "nf4", "status": "ready",
        }

    @staticmethod
    def _transient():
        return {
            "error_kind": "cuda_devices_unavailable",
            "error_sha256": "a" * 64,
            "error_type": "AcceleratorError",
            "retryable_token": "cuda_devices_busy_or_unavailable_phrase",
            "status": "transient_device_unavailable",
        }

    @staticmethod
    def _binding():
        return {
            "version": "research4.cuda-nf4-readiness-binding.v1",
            "identity": "self_continuity_seed202610281_delta2_recovery_test",
            "intent_sha256": "1" * 64,
            "plan_sha256": "2" * 64,
            "source_manifest_sha256": "3" * 64,
            "job_run_id": "447764828703422",
            "native_parent_start_epoch_ms": 1793420578713,
        }

    def _run(self, responses, clock=None):
        clock = clock or Clock()
        calls = []
        emitted = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            response = responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            clock.value += 1.0
            return response

        receipt = self.ns["_bounded_cuda_nf4_readiness"](
            code_root=ROOT, binding=self._binding(),
            elapsed_seconds=clock.elapsed, stop_seconds=1200,
            python_executable="python", runner=runner, sleeper=clock.sleep,
            monotonic=clock.monotonic,
            receipt_emitter=lambda value: emitted.append(
                json.loads(json.dumps(value))))
        return receipt, calls, clock, emitted

    def test_transient_then_ready_is_bounded_and_self_hashed(self):
        receipt, calls, clock, emitted = self._run([
            self._completed(self._transient(), 75),
            self._completed(self._ready()),
        ])
        self.assertEqual(receipt["status"], "ready")
        self.assertEqual(len(receipt["attempts"]), 2)
        self.assertEqual(clock.sleeps, [30.0])
        self.assertEqual(len(calls), 2)
        self.assertEqual(emitted, [receipt])
        self.assertEqual(calls[0][0][:2], ["python", "-c"])
        self.assertEqual(calls[0][1]["timeout"], 90.0)
        self.assertTrue(calls[0][1]["env"]["PYTHONPATH"].startswith(str(ROOT)))
        self.assertEqual(receipt["version"], "research4.cuda-nf4-readiness.v1")
        self.assertEqual(receipt["backward_passes"], 0)
        self.assertEqual(receipt["stop_seconds"], 1200.0)
        self.assertEqual(receipt["provider_startup_abort_seconds"], 1200.0)
        self.assertEqual(receipt["pre_reservation_margin_seconds"], 60.0)
        self.assertEqual(receipt["child_source_sha256"], hashlib.sha256(
            self.ns["_CUDA_NF4_CHILD"].encode("utf-8")).hexdigest())
        self.assertEqual(receipt["binding"], self._binding())
        self.assertEqual(receipt["binding_sha256"], hashlib.sha256(
            self.ns["_canonical"](self._binding()).encode("utf-8")).hexdigest())
        embedded = receipt.pop("receipt_sha256")
        self.assertEqual(embedded, hashlib.sha256(
            self.ns["_canonical"](receipt).encode("utf-8")).hexdigest())

    def test_non_transient_failure_is_not_retried(self):
        fatal = {
            "error_kind": "fatal_cuda_readiness", "error_sha256": "b" * 64,
            "error_type": "OutOfMemoryError", "status": "fatal",
        }
        calls = []
        emitted = []

        def runner(*args, **kwargs):
            calls.append(1)
            return self._completed(fatal, 1)

        with self.assertRaisesRegex(RuntimeError, "non-retryable") as raised:
            self.ns["_bounded_cuda_nf4_readiness"](
                code_root=ROOT, binding=self._binding(),
                elapsed_seconds=lambda: 100.0,
                stop_seconds=1200, runner=runner, sleeper=lambda _: None,
                monotonic=lambda: 0.0,
                receipt_emitter=lambda value: emitted.append(value))
        self.assertEqual(calls, [1])
        self.assertEqual(emitted[0]["status"], "fatal")
        self.assertEqual(len(emitted[0]["attempts"]), 1)
        serialized = str(raised.exception).split("readiness_receipt=", 1)[1]
        self.assertEqual(json.loads(serialized), emitted[0])

    def test_execution_binding_is_strict_and_tamper_evident(self):
        for mutation in (
                lambda value: value.pop("job_run_id"),
                lambda value: value.__setitem__("intent_sha256", "x" * 64),
                lambda value: value.__setitem__("native_parent_start_epoch_ms", 0),
                lambda value: value.__setitem__("extra", "forbidden")):
            binding = self._binding(); mutation(binding)
            with self.subTest(binding=binding), self.assertRaises(PermissionError):
                self.ns["_bounded_cuda_nf4_readiness"](
                    code_root=ROOT, binding=binding,
                    elapsed_seconds=lambda: 100.0, stop_seconds=1200,
                    runner=lambda *a, **k: self.fail("must not spawn"),
                    sleeper=lambda _: None, monotonic=lambda: 0.0,
                    receipt_emitter=lambda _: self.fail("must not emit"))

    def test_exit75_claim_cannot_relabel_oom_or_illegal_access(self):
        for error_type, token in (
                ("OutOfMemoryError", "cuda_devices_busy_or_unavailable_phrase"),
                ("AcceleratorError", "cudaErrorIllegalAddress")):
            payload = self._transient()
            payload["error_type"] = error_type
            payload["retryable_token"] = token
            calls = []

            def runner(*args, **kwargs):
                calls.append(1)
                return self._completed(payload, 75)

            with self.subTest(error_type=error_type, token=token), \
                    self.assertRaisesRegex(RuntimeError, "non-retryable"):
                self.ns["_bounded_cuda_nf4_readiness"](
                    code_root=ROOT, binding=self._binding(),
                    elapsed_seconds=lambda: 100.0,
                    stop_seconds=1200, runner=runner, sleeper=lambda _: None,
                    monotonic=lambda: 0.0, receipt_emitter=lambda _: None)
            self.assertEqual(calls, [1])

    def test_two_transient_failures_exhaust_without_third_probe(self):
        calls = []
        clock = Clock()
        emitted = []

        def runner(*args, **kwargs):
            calls.append(1); clock.value += 1.0
            return self._completed(self._transient(), 75)

        with self.assertRaisesRegex(RuntimeError, "remained unavailable"):
            self.ns["_bounded_cuda_nf4_readiness"](
                code_root=ROOT, binding=self._binding(), elapsed_seconds=clock.elapsed,
                stop_seconds=1200, runner=runner, sleeper=clock.sleep,
                monotonic=clock.monotonic,
                receipt_emitter=lambda value: emitted.append(value))
        self.assertEqual(len(calls), 2)
        self.assertEqual(clock.sleeps, [30.0])
        self.assertEqual(emitted[0]["status"], "transient_exhausted")
        self.assertEqual(len(emitted[0]["attempts"]), 2)

    def test_malformed_output_and_timeout_are_fatal(self):
        for response, message, attempt_status in (
                (SimpleNamespace(returncode=0, stdout="noise", stderr=""),
                 "output is invalid", "malformed_output"),
                (subprocess.TimeoutExpired(["python"], 90),
                 "timed out", "timeout")):
            with self.subTest(message=message):
                emitted = []

                def runner(*args, **kwargs):
                    if isinstance(response, BaseException):
                        raise response
                    return response

                with self.assertRaisesRegex(RuntimeError, message):
                    self.ns["_bounded_cuda_nf4_readiness"](
                        code_root=ROOT, binding=self._binding(),
                        elapsed_seconds=lambda: 100.0,
                        stop_seconds=1200, runner=runner,
                        sleeper=lambda _: None, monotonic=lambda: 0.0,
                        receipt_emitter=lambda value: emitted.append(value))
                self.assertEqual(emitted[0]["status"], "fatal")
                self.assertEqual(emitted[0]["attempts"][0]["status"],
                                 attempt_status)

        emitted = []
        with self.assertRaisesRegex(RuntimeError, "could not start"):
            self.ns["_bounded_cuda_nf4_readiness"](
                code_root=ROOT, binding=self._binding(),
                elapsed_seconds=lambda: 100.0,
                stop_seconds=1200,
                runner=lambda *a, **k: (_ for _ in ()).throw(
                    FileNotFoundError("python missing")),
                sleeper=lambda _: None, monotonic=lambda: 0.0,
                receipt_emitter=lambda value: emitted.append(value))
        self.assertEqual(emitted[0]["attempts"][0]["status"], "runner_error")

    def test_ready_payload_hardware_is_strict(self):
        mutations = []
        for key, value in (
                ("compute_dtype", "bfloat16"), ("adapter_dtype", "float16")):
            payload = self._ready(); payload["hardware"][key] = value
            mutations.append(payload)
        payload = self._ready(); payload["hardware"]["devices"] = []
        mutations.append(payload)
        payload = self._ready(); payload["hardware"]["devices"][0]["name"] = "T4"
        mutations.append(payload)
        payload = self._ready()
        payload["hardware"]["devices"][0]["total_memory_bytes"] = 16 * 1024**3
        mutations.append(payload)
        for payload in mutations:
            with self.subTest(hardware=payload["hardware"]), \
                    self.assertRaisesRegex(RuntimeError, "non-retryable"):
                self._run([self._completed(payload)])

    def test_deadline_rejects_before_sleep_or_child(self):
        calls = []
        emitted = []
        with self.assertRaisesRegex(TimeoutError, "safe pre-reservation window"):
            self.ns["_bounded_cuda_nf4_readiness"](
                code_root=ROOT, binding=self._binding(),
                elapsed_seconds=lambda: 1100.0,
                stop_seconds=1200, runner=lambda *a, **k: calls.append(1),
                sleeper=lambda _: self.fail("must not sleep"), monotonic=lambda: 0.0,
                receipt_emitter=lambda value: emitted.append(value))
        self.assertEqual(calls, [])
        self.assertEqual(emitted[0]["status"], "deadline_rejected")

    def test_deadline_rechecks_after_transient_and_after_backoff(self):
        for sleep_advance in (30.0, 80.0):
            clock = Clock(job_elapsed=100.0)
            calls = []
            emitted = []

            def runner(*args, **kwargs):
                calls.append(1); clock.value += 1.0
                return self._completed(self._transient(), 75)

            def sleeper(_seconds):
                clock.sleeps.append(_seconds); clock.value += sleep_advance

            stop = 280.0 if sleep_advance == 30.0 else 300.0
            with self.subTest(sleep_advance=sleep_advance), \
                    self.assertRaisesRegex(TimeoutError, "window"):
                self.ns["_bounded_cuda_nf4_readiness"](
                    code_root=ROOT, binding=self._binding(),
                    elapsed_seconds=clock.elapsed,
                    stop_seconds=stop, runner=runner, sleeper=sleeper,
                    monotonic=clock.monotonic,
                    receipt_emitter=lambda value: emitted.append(value))
            self.assertEqual(calls, [1])
            self.assertEqual(emitted[0]["status"], "deadline_rejected")

    def test_phase_cap_exact_boundary_allows_second_child(self):
        clock = Clock(job_elapsed=100.0)
        responses = [
            self._completed(self._transient(), 75), self._completed(self._ready())]

        def runner(*args, **kwargs):
            response = responses.pop(0)
            if response.returncode == 75:
                clock.value += 60.0
            return response

        receipt = self.ns["_bounded_cuda_nf4_readiness"](
            code_root=ROOT, binding=self._binding(),
            elapsed_seconds=clock.elapsed, stop_seconds=1200,
            runner=runner, sleeper=clock.sleep, monotonic=clock.monotonic,
            receipt_emitter=lambda _: None)
        self.assertEqual(receipt["status"], "ready")
        self.assertEqual(len(receipt["attempts"]), 2)

    def test_notebook_order_and_zero_parent_cuda_allocation(self):
        source = NOTEBOOK.read_text(encoding="utf-8")
        helper_call = source.index("cuda_readiness = _bounded_cuda_nf4_readiness(")
        parent_check_before = source.index(
            'if parent_torch.cuda.is_initialized():\n    raise RuntimeError('
            '"Parent CUDA context initialized before readiness child")')
        parent_check_after = source.index(
            'if parent_torch.cuda.is_initialized():\n    raise RuntimeError('
            '"Readiness child initialized the parent CUDA context")')
        nvml = source.index("pynvml.nvmlInit()")
        disk = source.index("shutil.disk_usage(")
        reserve = source.index("    reserve_intent(lock_root, intent, owner)")
        consume = source.index('"consumed_after_gpu_ready_before_model"')
        model_work = source.index("    result = run_partition(")
        self.assertLess(parent_check_before, helper_call)
        self.assertLess(helper_call, parent_check_after)
        self.assertLess(parent_check_after, nvml)
        self.assertLess(helper_call, disk)
        self.assertLess(nvml, reserve)
        self.assertLess(disk, reserve)
        self.assertLess(reserve, consume)
        self.assertLess(consume, model_work)
        self.assertEqual(source.count(
            'torch.zeros(256, dtype=torch.float16, device="cuda:0")'), 1)
        self.assertIn("_CUDA_NF4_CHILD = r'''", source)
        jobs = yaml.safe_load((ROOT / "resources" /
            "simulated_continuity_databricks.job.yml").read_text())["resources"]["jobs"]
        primary = jobs["simulated_continuity_primary_partition"]
        task = next(row for row in primary["tasks"]
                    if row["task_key"] == "run_one_partition")
        self.assertEqual(task["max_retries"], 0)
        self.assertEqual(primary["queue"], {"enabled": False})
        self.assertEqual(primary["max_concurrent_runs"], 1)


if __name__ == "__main__":
    unittest.main()
