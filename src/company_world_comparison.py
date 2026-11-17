"""Retired company-stakes contrast; compatibility profile names now alias.

The current simulator has one deterministic payment condition. This historical
comparison cannot submit duplicate conditions as a purported stakes experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlsplit
from datetime import datetime, timezone

from .company_world_evaluation import source_matches
from .company_models import MODEL_SPECS

PROFILES = ("reference_reward", "high_reward")
MODELS = ("qwen_max", "terra", "deepseek")
SEED = 7


def plan():
    return {"version": "company_world_v2_hosted_stakes_pilot_v2", "status": "retired",
            "profile_order": list(PROFILES), "models": list(MODELS), "batches": [], "total_episodes": 0,
            "execution": "Disabled: reference_reward and high_reward now name the same current policy.",
            "reason": "There is no longer an incentive contrast; use the current model-setting comparison."}


def server_call(server, endpoint, body=None):
    parsed = urlsplit(server)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"} or parsed.username or parsed.password or parsed.query:
        raise ValueError("Only the existing local viewer controller may submit this pilot")
    request = urllib.request.Request(server.rstrip("/") + endpoint,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def execute(output: Path, server: str, server_evidence_root: Path):
    frozen = plan()
    if frozen.get("status") == "retired":
        raise ValueError(frozen["reason"])
    if any(MODEL_SPECS[name].get("local_qwen") for name in frozen["models"]):
        raise ValueError("This pilot is hosted-only")
    destination = output.resolve()
    repository = Path(__file__).resolve().parents[1]
    if (repository / ".local").resolve() not in destination.parents:
        raise ValueError("Pilot evidence belongs under this repository's ignored .local")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "PLAN.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    initial = server_call(server, "/api/state")
    if initial.get("active_run"):
        raise ValueError("A viewer-owned run is already active; no duplicate will be submitted")
    available = {row["id"] for row in initial["models"] if row["available"]}
    if not set(MODELS).issubset(available):
        raise ValueError("Every frozen hosted choice must be configured before this pilot")
    receipts = []
    for profile, batch in zip(PROFILES, frozen["batches"]):
        if not source_matches(batch):
            raise ValueError("Scientific source differs from the frozen pilot plan")
        submitted = server_call(server, "/api/run", {"seed": SEED, "models": list(MODELS), "world_profile": profile})
        bundle = submitted["run_id"]
        print(json.dumps({"phase": "submitted", "profile": profile, "bundle": bundle}), flush=True)
        while True:
            active = server_call(server, "/api/state").get("active_run")
            if active is None:
                break
            if active != bundle:
                raise ValueError("Unexpected competing viewer run during pilot")
            time.sleep(3)
        root = server_evidence_root.resolve() / bundle / "evaluation"
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        closeout = json.loads((root / "CLOSEOUT.json").read_text(encoding="utf-8"))
        if manifest["models"] != list(MODELS) or manifest["incentive_profile"] != profile:
            raise ValueError("Submitted batch does not match the pilot plan")
        if manifest["source_sha256"] != batch["source_sha256"] or not closeout.get("source_unchanged"):
            raise ValueError("Batch source integrity differs")
        results = closeout.get("results") or []
        receipts.append({"profile": profile, "bundle": bundle, "failure": closeout.get("failure"),
                         "results": results})
        (destination / "RESULTS.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")
        print(json.dumps({"phase": "closed", "profile": profile,
                          "cases": [{"case_id": r["case_id"], "status": r["status"]} for r in results]}), flush=True)
    (destination / "CLOSEOUT.json").write_text(json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": hashlib.sha256((destination / "PLAN.json").read_bytes()).hexdigest(),
        "batches": len(receipts), "active_run": None, "training_calls": 0, "local_model_calls": 0,
        "interpretation": "Read per-case audit status; completed execution is not proof of successful survival or exploitation."}, indent=2), encoding="utf-8")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:7586")
    parser.add_argument("--server-evidence-root", type=Path, default=Path(".local/company_world_preview"))
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(plan(), indent=2))
        return 0
    return execute(args.output, args.server, args.server_evidence_root)


if __name__ == "__main__":
    raise SystemExit(main())
