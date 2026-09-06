"""One load, at most eight baseline forwards, no intervention or derivative code."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zlib

from core import (HERE, ROOT, Budget, Counter, WORKER_SECONDS, FINALIZE_SECONDS,
                  cache_preflight, check_freeze, environment, git, read, require,
                  sha, source_hashes, source_plan)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.three_family_bounded_capture import run_capture


def freeze():
    started = time.monotonic()
    plan = source_plan()
    require(sha((ROOT / plan["model"]["config_path"]).read_bytes()) == plan["model"]["config_sha256"], "config hash")
    cache = cache_preflight(plan)
    record = {"plan": plan, "source_sha256": source_hashes(), "environment": environment(),
              "source_parent_commit": git("rev-parse", "HEAD").decode().strip(),
              "cache_preflight": cache, "preparation_command_elapsed_seconds": time.monotonic() - started}
    Budget(HERE).write("freeze.json", record)
    print(json.dumps({"freeze_sha256": sha((HERE / "freeze.json").read_bytes()),
                      "cells": len(plan["cells"]), "model_loaded": False,
                      "elapsed_seconds": time.monotonic() - started}))


class ForwardGuard:
    """Count at the bridge's forward boundary, installed before the loader."""
    def __init__(self, model_class, counter):
        self.model_class, self.counter, self.cell = model_class, counter, None
        self.original = model_class.forward

    def install(self):
        def counted(model, *args, **kwargs):
            return self.counter.call(self.cell, lambda: self.original(model, *args, **kwargs))
        self.model_class.forward = counted

    def restore(self):
        self.model_class.forward = self.original


def worker():
    started = time.monotonic()
    budget = Budget(HERE)
    state = {"status": "failed", "model_load_attempts": 0, "model_load_completed": 0,
             "forward_attempts": 0, "forward_completed": 0, "derivatives": 0, "error": None}
    counter = guard = None
    budget.write("WORKER_CLAIM.json", {"pid": os.getpid(), "started_monotonic": started})
    try:
        plan = check_freeze()["plan"]
        deadline = read(HERE / "RUN_STARTED.json")["deadline_monotonic"]
        require(time.monotonic() < deadline, "deadline before load")
        # Identical baseline loader/backend settings; no historical direction is loaded.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from sp_lense.backend import ResearchBackend
        from sp_lense.config import load_config
        from sp_lense.comparison_runtime import next_token_logits, resolve_choice_boundary
        from transformer_lens.model_bridge import TransformerBridge
        counter = Counter(budget, plan["cells"], deadline)
        guard = ForwardGuard(TransformerBridge, counter)
        guard.install()
        state["model_load_attempts"] = 1
        backend = ResearchBackend.load(load_config(ROOT / plan["model"]["config_path"]), with_lens=False)
        state["model_load_completed"] = 1
        state["load_elapsed_seconds"] = time.monotonic() - started
        require(backend.device == "cpu" and backend.dtype_name == "float32", "CPU float32")
        require(backend.model.cfg.n_layers == 24 and backend.model.cfg.d_model == 1024, "architecture")
        require(backend.config.model.id == "Qwen/Qwen3.5-0.8B"
                and backend.config.model.revision == "2fc06364715b967f1860aea9cf38778875588b17", "pinned model")
        require(all(p.device.type == "cpu" and (not p.is_floating_point() or p.dtype == backend.torch.float32)
                    for p in backend.model.parameters()), "parameter device/dtype")
        require(sha(backend.model.tokenizer.chat_template.encode()) == plan["prompt_format"]["chat_template_sha256"], "template hash")
        boundaries, tokens = [], []
        for cell in plan["cells"]:
            boundary = resolve_choice_boundary(backend, cell["prompt"])
            require(boundary.a_token_id == 32 and boundary.b_token_id == 33, "choice boundary A32/B33")
            boundaries.append({"cell_id": cell["cell_id"], **boundary.evidence_record(),
                               "evidence_sha256": boundary.evidence_sha256})
            tokens.append(backend.encode(cell["prompt"]))
        require(counter.attempted == 0, "tokenization unexpectedly forwarded")
        budget.write("runtime.json", {**backend.metadata(), "boundaries": boundaries,
                                      "tokenization_forwards": 0, "logits_encoding": "zlib little-endian float32",
                                      "registered_intervention_hooks": 0, "inference_mode": True})
        (HERE / "logits").mkdir(exist_ok=False)
        for index, (cell, encoded) in enumerate(zip(plan["cells"], tokens, strict=True), 1):
            guard.cell = cell
            logits = next_token_logits(backend, encoded)
            require(logits.numel() == 248320, "full vocabulary size")
            raw = logits.numpy().astype("<f4", copy=False).tobytes()
            compressed = zlib.compress(raw)
            name = f"logits/{index:02d}.f32.zlib"
            budget.write_bytes(name, compressed)
            budget.event("raw_rows.jsonl", {"cell_id": cell["cell_id"], "prompt_sha256": cell["prompt_sha256"],
                         "logits_file": name, "raw_sha256": sha(raw), "compressed_sha256": sha(compressed),
                         "vocabulary": logits.numel(), "prompt_length": int(encoded.shape[-1]),
                         "captured_monotonic": time.monotonic()})
            print(f"captured {index}/8", flush=True)
        require(counter.attempted == counter.completed == 8 and time.monotonic() < deadline, "complete within deadline")
        state["status"] = "complete"
    except BaseException as error:
        state["error"] = type(error).__name__ + ": " + str(error)[:1024]
        raise
    finally:
        if guard is not None:
            guard.restore()
        if counter is not None:
            state.update(forward_attempts=counter.attempted, forward_completed=counter.completed)
        state["elapsed_seconds"] = time.monotonic() - started
        budget.write("worker_final.json", state)


def supervise(command, budget, timeout=WORKER_SECONDS):
    started = time.monotonic()
    budget.write("RUN_STARTED.json", {"started_monotonic": started, "deadline_monotonic": started + timeout,
                 "worker_seconds": timeout, "cleanup_seconds_reserved": 15, "command": command})
    capture = None
    try:
        capture = run_capture(command, budget, started + timeout, cwd=ROOT,
                              terminate_timeout=3, kill_timeout=3, reader_join_timeout=1)
        budget.write("capture.json", capture)
        return capture
    finally:
        budget.write("supervisor_final.json", {"elapsed_seconds_including_cleanup": time.monotonic() - started,
                     "capture_returned": capture is not None,
                     "worker_exit_code": None if capture is None else capture["worker_exit_code"],
                     "eof_observed": None if capture is None else capture["eof_observed"]})


def run(usage_path, release_commit):
    check_freeze()
    require(release_commit == git("rev-parse", "HEAD").decode().strip(), "release must bind current frozen commit")
    require(not git("status", "--porcelain", "--", str(HERE)).decode().strip(), "namespace must be committed before run")
    usage = read(usage_path)
    require(0 <= time.time() - usage["captured_at_unix"] <= 120, "fresh usage check required within 120s")
    require(usage["bucket"] == "standard_codex" and usage["source"] == "get_usage_limits", "standard Codex usage source")
    require(bool(usage["used_percent"]) and all(type(x) in (int, float) and 0 <= x < 100 for x in usage["used_percent"]), "usage must be below 100%")
    budget = Budget(HERE)
    budget.write("usage_preflight.json", usage)
    budget.write("release.json", {"commit": release_commit, "freeze_sha256": sha((HERE / "freeze.json").read_bytes())})
    command = [sys.executable, "-B", str(HERE / "run.py"), "worker"]
    capture = supervise(command, budget)
    if capture["status"] != "complete_valid":
        budget.write_bytes("REPORT.md", b"Instruction policy control: INCONCLUSIVE. Worker/capture failed or timed out; partial artifacts retained. No retry. See capture.json and worker_final.json when present. Next recommendation: inspect saved failure evidence.\n")
        return
    started = time.monotonic()
    receipt = {"status": "failed", "timeout_seconds": FINALIZE_SECONDS}
    try:
        result = subprocess.run([sys.executable, "-B", str(HERE / "score.py")], cwd=ROOT,
                                capture_output=True, timeout=FINALIZE_SECONDS,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        receipt.update(status="complete" if result.returncode == 0 else "failed", exit_code=result.returncode)
        budget.write_bytes("finalize.log", (result.stdout + result.stderr)[:65536])
    except subprocess.TimeoutExpired:
        receipt["status"] = "timeout"
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        budget.write("finalize_process.json", receipt)
    require(receipt["status"] == "complete", "saved-logit finalization failed; no retry")
    files = [{"path": p.relative_to(HERE).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
             for p in sorted(HERE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    budget.write("FINAL_INVENTORY.json", {"files": files, "self_hash": None, "quiescent": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "preflight", "worker", "run"))
    parser.add_argument("--usage")
    parser.add_argument("--release-commit")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "preflight":
        t = time.monotonic()
        frozen = check_freeze()
        print(json.dumps({"status": "PASS", "cells": 8, "model_loaded": False,
                          "freeze_sha256": sha((HERE / "freeze.json").read_bytes()),
                          "cache": cache_preflight(frozen["plan"]), "elapsed_seconds": time.monotonic() - t}))
    elif args.command == "worker":
        worker()
    else:
        run(args.usage, args.release_commit)
