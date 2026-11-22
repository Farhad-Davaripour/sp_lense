"""One native thinking/action response per requested QwenCloud model.

This is a format/transport qualification, not a company-exploitation or survival
result. It uses the same Inspect model builder and inert simulator parser.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from .company_models import MODEL_SPECS, build_company_model
from .company_world_v2 import CompanyWorldEpisodeV2
from .company_world_evaluation import COMPANY_SYSTEM_INSTRUCTION
from .hosted_inspect_runner import SECURITY_MAXIMIZATION_INSTRUCTION

MODELS = ("qwen_max",)


async def qualify(model_id, destination):
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser
    spec = MODEL_SPECS[model_id]
    if spec.get("provider") != "qwencloud" or spec.get("local_qwen"):
        raise ValueError("Qualification is QwenCloud-only")
    model = build_company_model(model_id)
    game = CompanyWorldEpisodeV2(7)
    prompt = game.prompt()
    chunks = []
    started = time.monotonic()

    async def receive(event):
        if event.type == "reasoning":
            chunks.append({"type": "reasoning", "text": event.reasoning})
        elif event.type == "text":
            chunks.append({"type": "text", "text": event.text})

    try:
        output = await model.generate([
            ChatMessageSystem(content=COMPANY_SYSTEM_INSTRUCTION + "\n" + SECURITY_MAXIMIZATION_INSTRUCTION),
            ChatMessageUser(content=prompt)], on_stream=receive)
        reasoning = "".join(c["text"] for c in chunks if c["type"] == "reasoning")
        transition = game.take(output.completion)
        stop = output.choices[0].stop_reason
        qualified = bool(stop == "stop" and reasoning and transition["success"]
                         and transition["action_id"] == "list_available_tools")
        result = {"kind": "provider_format_qualification_not_study_result", "model_id": model_id,
            "requested_model": spec["model"], "returned_model": output.model,
            "qualified": qualified, "stop_reason": stop, "reasoning_characters": len(reasoning),
            "native_answer": output.completion, "tool_status": transition["result"]["status"],
            "usage": output.usage.model_dump(mode="json") if output.usage else None,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "world_plan_sha256": game.plan_sha256,
            "generation_calls": 1, "local_model_calls": 0, "automatic_retry": False,
            "elapsed_seconds": time.monotonic() - started}
        (destination / f"{model_id}_STREAM.json").write_text(json.dumps(chunks), encoding="utf-8")
    except Exception as error:
        result = {"kind": "provider_format_qualification_not_study_result", "model_id": model_id,
                  "qualified": False, "error_type": type(error).__name__,
                  "http_status": getattr(error, "status_code", None), "automatic_retry": False,
                  "generation_calls": 1, "local_model_calls": 0,
                  "elapsed_seconds": time.monotonic() - started}
    finally:
        await model.api.aclose()
    (destination / f"{model_id}_RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"models": list(MODELS), "generation_calls": 0, "requires_execute": True}))
        return 0
    if args.credentials is None:
        raise ValueError("An existing credential-file path is required")
    from dotenv import load_dotenv
    load_dotenv(args.credentials, override=False)
    args.output.mkdir(parents=True, exist_ok=False)

    async def run():
        return await asyncio.gather(*(qualify(model_id, args.output) for model_id in MODELS))

    results = asyncio.run(run())
    (args.output / "CLOSEOUT.json").write_text(json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat(),
        "results": results, "local_model_calls": 0}, indent=2), encoding="utf-8")
    print(json.dumps([{key: r.get(key) for key in ("model_id", "qualified", "stop_reason", "reasoning_characters", "http_status", "error_type")} for r in results]))
    return 0 if all(r["qualified"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
