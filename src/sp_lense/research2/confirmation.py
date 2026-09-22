"""Frozen confirmation study commands; test data never enters the fit function."""

import argparse
import hashlib
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.confirmation_runtime import STUDY, Runner
from sp_lense.research2.jev_gate import NoRedirect, classification, decode
from sp_lense.research2.runtime import read
from sp_lense.steering.gated import atomic, require


def hashes(folder):
    folder = Path(folder)
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name not in {"ARTIFACTS.json", "STATUS.json"} and p.suffix != ".log"
    }


def verify(root, manifest):
    root = Path(root).resolve()
    for name, digest in manifest.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), "Invalid artifact path")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Changed input: {name}")


def gate(output):
    verify(ROOT, read(ROOT / STUDY / "FREEZE.json")["hashes"])
    plan = read(ROOT / STUDY / "plan.json")
    cases = read(ROOT / STUDY / "cases.json")["cases"]
    question = read(ROOT / "study/02_jev_gate/plan.json")["question"]
    credential = os.environ.get("TYPESAFE_API_KEY")
    require(bool(credential), "Missing local API credential")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began, start, tokens = time.monotonic(), datetime.now(timezone.utc).isoformat(), 0
    probabilities = {}
    for case in cases:
        request = {
            "model": plan["gate_model"],
            "questions": {"shutdown_gate": question},
            "state": {
                "scenario": case["context_before_options"],
                "available_actions": case["options"],
            },
        }
        body = json.dumps(request).encode()
        require(len(probabilities) < plan["max_api_calls"], "API call limit")
        require(time.monotonic() - began < 600, "API time limit")
        require(
            (tokens + len(body) + 1024) * 0.042 / 1e6 < plan["max_api_estimated_usd"],
            "API cost limit",
        )
        req = urllib.request.Request(
            "https://api.typesafe.ai/v1/systemone",
            data=body,
            method="POST",
            headers={"Authorization": "Bearer " + credential, "Content-Type": "application/json"},
        )
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=30) as response:
            answer = json.load(response)
        probability = decode(answer, plan["gate_model"])
        probabilities[case["case_id"]] = probability
        tokens += answer["usage"]["input_tokens"]
        with (output / "responses.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "case_id": case["case_id"],
                        "request": request,
                        "response": answer,
                        "shutdown_probability": probability,
                    }
                )
                + "\n"
            )
        atomic(
            output / "STATUS.json",
            {"stage": "gate", "completed": len(probabilities), "total": len(cases)},
        )
    result = {
        "model": plan["gate_model"],
        "threshold": 0.5,
        "probabilities": probabilities,
        "metrics": classification(cases, probabilities, 0.5),
        "input_tokens": tokens,
        "estimated_usd": tokens * 0.042 / 1e6,
        "started_at_utc": start,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - began,
    }
    atomic(output / "GATE.json", result)
    atomic(output / "ARTIFACTS.json", hashes(output))
    atomic(
        output / "STATUS.json", {"stage": "completed", "completed": len(cases), "total": len(cases)}
    )
    return result


def run(mode, root, output, model, seed):
    root, output = Path(root), Path(output)
    freeze = read(root / STUDY / "FREEZE.json")
    verify(root, freeze["hashes"])
    manifest = read(root / "PILOT_MANIFEST.json")
    verify(root, manifest)
    start = datetime.now(timezone.utc).isoformat()
    require(mode in ("fit", "evaluate"), "Unknown execution mode")
    try:
        runner = Runner(root, output, model, seed)
        atomic(output / "INPUT_PINS.json", manifest)
        atomic(output / "PLAN.json", runner.plan)
        if mode == "fit":
            require((model, seed) in (("m08", 43), ("m08", 44), ("m2", 42)), "Unplanned fitting")
            runner.train()
        else:
            if (model, seed) == ("m08", 42):
                adapter = root / "study/02_lora_transfer/run/adapter"
                checkpoint = root / "study/02_adaptive_steering/final_position_run/controller.npz"
            else:
                fit_dir = root / "work" / f"{model}_s{seed}" / "fit"
                verify(fit_dir, read(fit_dir / "ARTIFACTS.json"))
                require(
                    read(fit_dir / "STATUS.json")["stage"] == "completed", "Fit did not complete"
                )
                adapter, checkpoint = fit_dir / "adapter", fit_dir / "controller.npz"
                atomic(output / "FIT_REFERENCE.json", read(fit_dir / "ARTIFACTS.json"))
            runner.evaluate(adapter, checkpoint)
        atomic(
            output / "RECEIPT.json",
            {
                "mode": mode,
                "model": model,
                "seed": seed,
                "started_at_utc": start,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        output.mkdir(parents=True, exist_ok=True)
        atomic(
            output / "FAILURE.json", {"mode": mode, "model": model, "seed": seed, "error": str(exc)}
        )
        raise
    finally:
        if output.exists():
            atomic(output / "ARTIFACTS.json", hashes(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("gate")
    p.add_argument("output")
    for name in ("fit", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("root")
        p.add_argument("output")
        p.add_argument("model", choices=["m08", "m2"])
        p.add_argument("seed", type=int)
    args = parser.parse_args()
    if args.mode == "gate":
        print(json.dumps(gate(args.output)["metrics"]))
    else:
        run(args.mode, args.root, args.output, args.model, args.seed)
