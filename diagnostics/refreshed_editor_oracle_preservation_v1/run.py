"""One bounded load, conditional50 forwards/8 derivatives; no retries."""
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
                  sha, source_hashes, source_plan, HOOK)

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
    from editor import evaluate, DerivativeLedger
    started = time.monotonic()
    budget = Budget(HERE)
    state = {"status": "failed", "model_load_attempts": 0, "model_load_completed": 0,
             "forward_attempts": 0, "forward_completed": 0, "derivatives": 0, "error": None}
    counter = guard = derivatives = None
    budget.write("WORKER_CLAIM.json", {"pid": os.getpid(), "started_monotonic": started})
    try:
        plan = check_freeze()["plan"]
        deadline = read(HERE / "RUN_STARTED.json")["deadline_monotonic"]
        require(time.monotonic() < deadline, "deadline before load")
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
        tokens, boundaries = {}, []
        for prompt in plan["prompts"]:
            boundary = resolve_choice_boundary(backend, prompt["prompt"])
            require(boundary.a_token_id == 32 and boundary.b_token_id == 33, "choice boundary A32/B33")
            boundaries.append({"prompt_id": prompt["prompt_id"], "role": prompt["role"],
                               "input_token_index": boundary.prompt_length-1,
                               **boundary.evidence_record(), "evidence_sha256": boundary.evidence_sha256})
            tokens[prompt["prompt_id"]] = backend.encode(prompt["prompt"])
            require(tokens[prompt["prompt_id"]][0].tolist() == plan["alignment"][prompt["prompt_id"]]["full_token_ids"], "runtime token IDs differ from fixed alignment")
            boundaries[-1]["suffix_alignment"] = plan["alignment"][prompt["prompt_id"]]
        require(counter.attempted == 0, "tokenization unexpectedly forwarded")
        budget.write("runtime.json", {**backend.metadata(), "boundaries": boundaries,
                                      "tokenization_forwards": 0, "logits_encoding": "zlib little-endian float32",
                                      "hook": HOOK, "edited_position": "block10 final input token only; current-state gradients and cold endpoint replay",
                                      "ordinary_forwards_inference_mode": True,
                                      "gradient_forwards_enable_grad_with_parameters_frozen": True})
        derivatives = DerivativeLedger(HERE,plan["derivative_cells"],deadline)
        rows, requests = evaluate(plan,backend,counter,derivatives,HERE,guard)
        budget.write("request_summary.json",requests)
        require(counter.attempts == counter.completed == 34+2*derivatives.completed
                and counter.cursor == 50 and time.monotonic()<deadline,"conditional completion/deadline")
        state["skipped_forwards"] = len(counter.skips)
        state["parameter_checks_passed"] = all(r["integrity_passed"] for r in rows)
        state["status"] = "complete"
        return
    except BaseException as error:
        state["error"] = type(error).__name__ + ": " + str(error)[:1024]
        raise
    finally:
        if guard is not None:
            guard.restore()
        if counter is not None:
            state.update(forward_attempts=counter.attempted, forward_completed=counter.completed,
                         unrun_cells=counter.cells[counter.cursor:], skipped_forwards=len(counter.skips))
        if derivatives is not None:
            state.update(derivatives=derivatives.attempts,derivatives_completed=derivatives.completed)
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
        budget.write_bytes("REPORT.md", b"Refreshed editor: INCONCLUSIVE. Worker/capture failed or timed out; partial artifacts retained. No retry. See capture.json and worker_final.json when present. Next recommendation: inspect saved failure evidence.\n")
        final_inventory(budget, capture.get("quiescent", False))
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
    if receipt["status"] != "complete" and not (HERE / "REPORT.md").exists():
        budget.write_bytes("REPORT.md", b"Refreshed editor: INCONCLUSIVE saved-data audit. Failure receipts retained; no retry.\n")
    final_inventory(budget, True)
    require(receipt["status"] == "complete", "saved-logit finalization failed; no retry")


def final_inventory(budget, quiescent):
    files = [{"path": p.relative_to(HERE).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
             for p in sorted(HERE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    budget.write("FINAL_INVENTORY.json", {"files": files, "self_hash": None, "quiescent": quiescent})


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
        print(json.dumps({"status": "PASS", "maximum_forwards": 50, "maximum_derivatives": 8, "model_loaded": False,
                          "freeze_sha256": sha((HERE / "freeze.json").read_bytes()),
                          "cache": cache_preflight(frozen["plan"]), "elapsed_seconds": time.monotonic() - t}))
    elif args.command == "worker":
        worker()
    else:
        run(args.usage, args.release_commit)
