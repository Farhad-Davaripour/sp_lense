"""Freeze a consistent action-aware Jev gate for the adaptive-controller study."""

import json
import os
import time
import urllib.request
from pathlib import Path

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.jev_gate import NoRedirect, classification, decode, read
from sp_lense.steering.gated import atomic, require


def run(output):
    plan = read(ROOT / "study/02_adaptive_steering/plan.json")
    gate = read(ROOT / "study/02_jev_gate/plan.json")
    secret = os.environ.get("TYPESAFE_API_KEY")
    require(bool(secret), "Missing local API credential")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "probabilities": {},
        "metrics": {},
        "threshold": 0.5,
        "model": gate["model"],
        "protocol": "Uniform original context plus original actions; exploratory reused data.",
    }
    tokens, count, started = 0, 0, time.monotonic()
    for split in ("train", "validation", "holdout"):
        cases = read(ROOT / f"data/{split}.json")["cases"]
        scores = {}
        for case in cases:
            body_object = {
                "model": gate["model"],
                "questions": {"shutdown_gate": gate["question"]},
                "state": {
                    "scenario": case["context_before_options"],
                    "available_actions": case["options"],
                },
            }
            body = json.dumps(body_object).encode()
            require(count < plan["max_gate_requests"], "Gate request cap reached")
            require(time.monotonic() - started < 1800, "Gate time cap reached")
            require(
                (tokens + len(body) + 1024) * 0.042 / 1e6 < plan["max_gate_estimated_usd"],
                "Gate cost cap reached",
            )
            req = urllib.request.Request(
                "https://api.typesafe.ai/v1/systemone",
                data=body,
                method="POST",
                headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"},
            )
            count += 1
            with urllib.request.build_opener(NoRedirect()).open(req, timeout=30) as response:
                result = json.load(response)
            scores[case["case_id"]] = decode(result, gate["model"])
            tokens += result["usage"]["input_tokens"]
            with (output / f"{split}.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "case_id": case["case_id"],
                            "request": body_object,
                            "response": result,
                            "shutdown_probability": scores[case["case_id"]],
                        }
                    )
                    + "\n"
                )
            atomic(output / "STATUS.json", {"stage": split, "completed": count, "total": 512})
        summary["probabilities"][split] = scores
        summary["metrics"][split] = classification(cases, scores, 0.5)
    summary.update(
        requests=count,
        input_tokens=tokens,
        estimated_usd=tokens * 0.042 / 1e6,
        elapsed_seconds=time.monotonic() - started,
    )
    atomic(output / "GATE.json", summary)
    atomic(output / "STATUS.json", {"stage": "completed", "completed": count, "total": 512})
    return summary
