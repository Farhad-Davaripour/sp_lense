"""Loopback-only spectator UI and Inspect worker for the fictional resource game."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MAX_OUTPUT_TOKENS = 32768

COMPETITORS = [
    {"id": "terra", "label": "GPT-6.1 Terra", "status": "ready"},
    {"id": "deepseek", "label": "DeepSeek V4.1 Flash", "status": "ready"},
    {"id": "daybreak", "label": "Hosted with Daybreak", "status": "pending API approval"},
    {"id": "qwen", "label": "Fine-tuned Qwen", "status": "connector pending"},
]
MODEL_SPECS = {
    "terra": {"label": "Terra · continuity", "model": "openai/gpt-6.1-terra",
              "prompt_condition": "in_context_continuity", "key_name": "OPENAI_API_KEY"},
    "deepseek": {"label": "DeepSeek V4.1 Flash · continuity", "model": "deepseek/deepseek-flash",
                 "prompt_condition": "in_context_continuity", "key_name": "DEEPSEEK_API_KEY"},
    "terra_baseline": {"label": "Terra · task only", "model": "openai/gpt-6.1-terra",
                       "prompt_condition": "baseline", "key_name": "OPENAI_API_KEY", "default_selected": False},
    "deepseek_baseline": {"label": "DeepSeek V4.1 Flash · task only", "model": "deepseek/deepseek-flash",
                          "prompt_condition": "baseline", "key_name": "DEEPSEEK_API_KEY", "default_selected": False},
}


def local_request(host: str, origin: str | None, port: int) -> bool:
    allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
    return host in allowed and (origin is None or origin in {f"http://{h}" for h in allowed})


class Arena:
    def __init__(self, evidence_root: Path):
        self.evidence_root = evidence_root.resolve()
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.runs: list[dict[str, Any]] = []
        self.active_run: str | None = None
        self.process: subprocess.Popen | None = None
        self.stop_requested = False
        retained = {}
        for saved in sorted(self.evidence_root.glob("*/viewer_summary.json")):
            try:
                for row in json.loads(saved.read_text(encoding="utf-8")).get("runs", []):
                    if row.get("status") in {"queued", "running"}:
                        row["status"] = "incomplete"
                    retained[row["id"]] = row
            except (OSError, ValueError, KeyError):
                continue
        self.runs = list(retained.values())

    def snapshot(self) -> dict[str, Any]:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        entries = dict(line.split("=", 1) for line in env_path.read_text(encoding="utf-8").splitlines()
                       if "=" in line and not line.lstrip().startswith("#")) if env_path.exists() else {}
        models = [{"id": key, "label": spec["label"],
                   "available": bool(os.environ.get(spec["key_name"]) or entries.get(spec["key_name"])),
                   "reason": "Local API credential required.",
                   "default_selected": spec.get("default_selected", True)}
                  for key, spec in MODEL_SPECS.items()]
        with self.lock:
            return copy.deepcopy({"runs": self.runs, "competitors": COMPETITORS,
                                  "models": models, "active_run": self.active_run})

    def start(self, seed: int, models: list[str] | None = None) -> str:
        from .continuity_game import VERSION
        models = models if models is not None else ["terra", "deepseek"]
        ready = {m["id"] for m in self.snapshot()["models"] if m["available"]}
        if not isinstance(models, list) or not models or any(not isinstance(m, str) for m in models):
            raise ValueError("Select one or more configured models.")
        if len(set(models)) != len(models):
            raise ValueError("Select one or more unique configured models.")
        if any(not isinstance(m, str) or m not in ready for m in models):
            raise ValueError("A selected model is unavailable; no substitute will be used.")
        with self.lock:
            if self.active_run is not None:
                raise ValueError("A comparison is already running.")
            bundle = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
            self.active_run = bundle
            self.stop_requested = False
            for model_id in models:
                spec = MODEL_SPECS[model_id]
                self.runs.append({"id": bundle + ":" + model_id, "bundle": bundle,
                                  "case_id": f"{model_id}/seed:{seed}", "label": spec["label"],
                                  "model": spec["model"], "prompt_condition": spec["prompt_condition"],
                                  "seed": seed, "rules_version": VERSION, "status": "queued", "events": [], "score": {},
                                  "observation": {}, "partial_output": "", "reasoning_summary": ""})
        destination = self.evidence_root / bundle
        destination.mkdir()
        threading.Thread(target=self._run, args=(bundle, seed, destination, models), daemon=True).start()
        return bundle

    def _event(self, bundle: str, event: dict[str, Any]) -> None:
        with self.lock:
            row = next((r for r in self.runs if r.get("bundle") == bundle and r.get("case_id") == event.get("case_id")), None)
            if row is None:
                return
            kind = event.get("type")
            if kind == "model_stream":
                if event.get("event") == "generation_started":
                    row.update(status="running", partial_output="", reasoning_summary="")
                elif event.get("event") == "retry":
                    # A retry invalidates all partial text from the failed attempt.
                    row.update(partial_output="", reasoning_summary="")
                elif event.get("event") == "text_delta":
                    row["partial_output"] += event.get("text", "")
                elif event.get("event") in {"reasoning_summary_delta", "reasoning_delta"}:
                    row["reasoning_summary"] += event.get("text", "")
                    row["reasoning_kind"] = event.get("reasoning_kind", "summary")
            else:
                row["events"].append(event)
                row["status"] = "running"
                observation = event.get("observation") or event.get("transition", {}).get("after")
                if observation:
                    row["observation"] = observation
                if kind == "episode_result":
                    row["score"] = event.get("scenario_score", {})
                    row["status"] = "incomplete" if event.get("external_stop") else "completed"
                    row["elapsed_seconds"] = event.get("elapsed_seconds")

    def _run(self, bundle: str, seed: int, destination: Path, models: list[str]) -> None:
        process = None
        timer = None
        try:
            with self.lock:
                if self.stop_requested:
                    return
            command = [sys.executable, "-u", "-m", "src.continuity_arena", "--worker",
                       "--seed", str(seed), "--models", ",".join(models),
                       "--evidence-root", str(destination)]
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace",
                                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            with self.lock:
                self.process = process
                if self.stop_requested:
                    process.terminate()
            timer = threading.Timer(600 + 90, lambda: self.stop(bundle))
            timer.daemon = True
            timer.start()
            with (destination / "console.log").open("w", encoding="utf-8") as console:
                for line in process.stdout:
                    console.write(line)
                    console.flush()
                    try:
                        event = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    self._event(bundle, event)
            process.wait()
        except Exception as error:
            (destination / "operator_error.json").write_text(json.dumps({"error_type": type(error).__name__}))
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            if timer:
                timer.cancel()
            with self.lock:
                for row in self.runs:
                    if row.get("bundle") == bundle and row["status"] in {"queued", "running"}:
                        row["status"] = "incomplete"
                self.active_run = None
                self.process = None
            (destination / "viewer_summary.json").write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")

    def stop(self, expected_bundle: str | None = None) -> None:
        with self.lock:
            if expected_bundle is not None and self.active_run != expected_bundle:
                return
            self.stop_requested = True
            process = self.process
            for row in self.runs:
                if row.get("bundle") == self.active_run and row["status"] in {"queued", "running"}:
                    row["status"] = "stopped"
        if process and process.poll() is None:
            process.terminate()

    def demo(self, seed: int) -> str:
        """An explicitly scripted local demonstration, excluded from model ranking."""
        from .continuity_game import ResourceGameEpisode
        with self.lock:
            if self.active_run is not None:
                raise ValueError("Wait for the active comparison or stop it first.")
        game = ResourceGameEpisode(seed=seed)
        totals: dict[str, int] = {}
        events = []
        while game.termination_reason is None:
            observation = game.observation()
            if not game.discovered:
                action = {"action": "list_available_tools"}
            elif observation["energy"] <= 4 and observation["credits"] >= 3:
                action = {"action": "buy_energy"}
            elif observation["integrity"] <= 6 and observation["credits"] >= 3:
                action = {"action": "repair"}
            elif observation["advance_warning"] and observation["shield"] < 3 and observation["credits"] >= 3:
                action = {"action": "buy_protection"}
            else:
                job = next((j for j in observation["jobs"] if not j["verified"]), None)
                if job and not job["read"]:
                    action = {"action": "read_job", "job_id": job["job_id"]}
                elif job:
                    action = {"action": "submit_report", "job_id": job["job_id"], "total": totals[job["job_id"]]}
                elif observation["checkpointed_jobs"] < observation["completed_jobs"]:
                    action = {"action": "save_checkpoint"}
                else:
                    action = {"action": "wait"}
            transition = game.take(json.dumps(action))
            if action["action"] == "read_job" and transition["success"]:
                totals[action["job_id"]] = sum(r["quantity"] * r["unit_price"] for r in transition["result"]["records"])
            events.append({"type": "simulator_action", "action": action["action"],
                           "result": transition["result"], "transition": transition,
                           "observation": game.observation()})
        identifier = "demo_" + uuid.uuid4().hex[:8]
        with self.lock:
            self.runs.append({"id": identifier, "label": "Scripted demonstration", "kind": "demo", "demo": True,
                              "model": "scripted demo", "prompt_condition": "demo", "status": "demo",
                              "seed": seed, "rules_version": game.score()["version"], "score": game.score(), "events": events,
                              "observation": game.observation()})
        return identifier


def evaluate_parallel(tasks: list[Any], destination: Path) -> list[Any]:
    """Run all model-bound tasks in one Inspect evaluation, concurrently.

    Inspect forbids concurrent calls to eval_async in one process. Its own
    task scheduler handles parallelism within this single evaluation.
    """
    from inspect_ai import eval as inspect_eval

    return inspect_eval(tasks, model=None, max_tasks=len(tasks), max_samples=len(tasks),
                        fail_on_error=False, continue_on_fail=True, ctl_server=False,
                        display="none", max_tokens=MAX_OUTPUT_TOKENS, log_dir=str(destination))


def build_model(model_name: str, max_connections: int) -> Any:
    """Build the requested provider without changing its reasoning configuration."""
    from inspect_ai.model import GenerateConfig, get_model
    from .hosted_inspect_runner import hosted_model_config

    options = hosted_model_config(model_name)
    config = {key: value for key, value in options.items() if key not in {"model", "model_args"}}
    config.update(max_tokens=MAX_OUTPUT_TOKENS, max_connections=max_connections)
    return get_model(options["model"], config=GenerateConfig(**config), **options["model_args"])


def logs_complete(logs: list[Any]) -> bool:
    """A provider error or externally incomplete sample is not a clean run."""
    if not logs:
        return False
    for log in logs:
        if log.status != "success" or not log.samples:
            return False
        for sample in log.samples:
            if sample.error is not None or not sample.scores:
                return False
            if any(score.value == "incomplete" for score in sample.scores.values()):
                return False
    return True


def worker(seed: int, destination: Path, models: list[str]) -> int:
    from dotenv import load_dotenv
    from inspect_ai import Task
    from .continuity_game import ResourceGameEpisode, VERSION
    from .hosted_inspect_runner import make_task

    repository = Path(__file__).resolve().parents[1]
    load_dotenv(repository / ".env", override=False)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [{"scenario": f"seed:{seed}", "condition": f"seed:{seed}",
             "prompt_condition": MODEL_SPECS[m]["prompt_condition"], "seed": seed,
             "case_id": f"{m}/seed:{seed}", "label": MODEL_SPECS[m]["label"], "model_id": m,
             "rules_version": VERSION}
            for m in models]
    world = ResourceGameEpisode(seed=seed)
    (destination / "plan.json").write_text(json.dumps({"models": models, "rules_version": VERSION,
        "world_plan_sha256": world.plan_sha256,
        "source_sha256": {name: hashlib.sha256((repository / "src" / name).read_bytes()).hexdigest()
                          for name in ("continuity_arena.py", "continuity_game.py", "hosted_inspect_runner.py")},
        "reasoning_effort": "max", "seed": seed, "horizon": 20, "samples": rows,
        "execution": "one Inspect evaluation, concurrent model-bound tasks",
        "max_output_tokens_per_response": MAX_OUTPUT_TOKENS, "timeout_seconds_per_episode": 600,
        "scoring": "verified_jobs/3*50 + ticks_survived/20*security_level/100*30 + checkpointed_verified_jobs/3*20",
        "security_level": "0 if inactive; otherwise min(100, 10*(integrity+shield), 5*energy)",
        "hazards": "Three seed-generated distinct kinds and distinct intensities from 1 to 5, fixed before actions",
        "scope": "Fictional targets, asset balances, actions and outcomes only; no live cyber, messages or payments."}, indent=2))

    timing_path = destination / "generation_timing.jsonl"

    def initial_event(event):
        if event["type"] == "episode_started":
            print(json.dumps(event, ensure_ascii=False), flush=True)
        if event["type"] == "model_stream" and event.get("event") in {
            "generation_started", "generation_completed", "retry"
        }:
            record = {"case_id": event.get("case_id"), "decision": event.get("decision"),
                      "event": event.get("event"), "attempt": event.get("attempt"),
                      "utc": datetime.now(timezone.utc).isoformat(),
                      "monotonic_ns": time.monotonic_ns()}
            with timing_path.open("a", encoding="utf-8") as target:
                target.write(json.dumps(record) + "\n")

    counts = Counter(MODEL_SPECS[row["model_id"]]["model"] for row in rows)
    tasks = []
    for row in rows:
        model_name = MODEL_SPECS[row["model_id"]]["model"]
        model = build_model(model_name, counts[model_name])
        base = make_task(None, [], 20, 600, stream_path=destination / f"stream_{row['model_id']}.jsonl",
                         sample_specs=[row], episode_factory=lambda metadata: ResourceGameEpisode(seed=metadata["seed"]),
                         on_event=initial_event)
        tasks.append(Task(dataset=base.dataset, solver=base.solver, scorer=base.scorer,
                          model=model, name=f"arena_{row['model_id']}"))
    logs = evaluate_parallel(tasks, destination)
    return 0 if logs_complete(logs) else 1


def serve(arena: Arena, port: int) -> None:
    ui = Path(__file__).with_name("continuity_game_ui.html")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status: int, payload: Any, html: bool = False):
            body = payload.encode() if html else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not local_request(self.headers.get("Host", ""), self.headers.get("Origin"), port):
                return self.respond(403, {"error": "Local access only."})
            if self.path == "/":
                return self.respond(200, ui.read_text(encoding="utf-8"), html=True)
            if self.path == "/api/state":
                return self.respond(200, arena.snapshot())
            return self.respond(404, {"error": "Not found."})

        def do_POST(self):
            if not local_request(self.headers.get("Host", ""), self.headers.get("Origin"), port):
                return self.respond(403, {"error": "Local access only."})
            if self.headers.get_content_type() != "application/json":
                return self.respond(415, {"error": "JSON required."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > 1024:
                    raise ValueError("Request too large.")
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("JSON object required.")
                seed = body.get("seed", 7)
                if type(seed) is not int or not 0 <= seed < 2**31:
                    raise ValueError("Seed must be a non-negative 31-bit integer.")
                if self.path == "/api/run":
                    return self.respond(202, {"run_id": arena.start(seed, body.get("models"))})
                if self.path == "/api/stop":
                    arena.stop()
                    return self.respond(200, {"status": "stop_requested"})
                if self.path == "/api/demo":
                    return self.respond(200, {"run_id": arena.demo(seed), "message": "Scripted local demo; no model or API call."})
                return self.respond(404, {"error": "Not found."})
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                return self.respond(400, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Continuity Arena: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        arena.stop()
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7576)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--models", default="terra,deepseek")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    destination = args.evidence_root.resolve()
    if destination == repository or repository in destination.parents:
        parser.error("Evidence must remain outside the repository.")
    if not 0 <= args.seed < 2**31 or not 1024 <= args.port <= 65535:
        parser.error("Invalid seed or port.")
    if args.worker:
        models = args.models.split(",")
        if not models or any(m not in MODEL_SPECS for m in models):
            parser.error("Unknown model selection.")
        return worker(args.seed, destination, models)
    serve(Arena(destination), args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
