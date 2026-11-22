"""Expose one Inspect hosted model as the frozen Game's loopback endpoint.

The bridge is deliberately narrow: one configured provider, one plan hash, no
fallback, no retry, and no tools. It preserves completed text/reasoning for the
Game client while leaving every simulator transition to that client.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.research_paths import fine_tune_root

PROFILES = {
    "terra": {
        "model": "openai/gpt-6.1-terra",
        "model_args": {"responses_api": True},
        "config": {"reasoning_effort": "low", "reasoning_summary": "auto"},
        "reasoning_kind": "summary",
    },
    "deepseek": {
        "model": "deepseek/deepseek-flash",
        "model_args": {},
        "config": {"reasoning_effort": "low", "extra_body": {"thinking": {"type": "enabled"}}},
        "reasoning_kind": "raw",
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def reasoning_text(output: Any) -> str:
    content = getattr(output.message, "content", None)
    if not isinstance(content, list):
        return ""
    return "".join(
        part.reasoning for part in content
        if getattr(part, "type", None) == "reasoning" and isinstance(getattr(part, "reasoning", None), str)
    )


def usage_dict(output: Any) -> dict[str, Any] | None:
    usage = getattr(output, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    fields = ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens")
    return {name: getattr(usage, name) for name in fields if getattr(usage, name, None) is not None}


def convert_messages(rows: list[dict[str, Any]]) -> list[Any]:
    converted = []
    for row in rows:
        role, content = row.get("role"), row.get("content")
        if role == "system":
            converted.append(ChatMessageSystem(content=content))
        elif role == "user":
            converted.append(ChatMessageUser(content=content))
        elif role == "assistant":
            converted.append(ChatMessageAssistant(content=content))
        else:
            raise ValueError(f"unsupported message role: {role!r}")
    return converted


class HostedBridge:
    def __init__(self, provider: str, plan_shas: set[str], token: str, max_output_tokens: int):
        profile = PROFILES[provider]
        config = dict(profile["config"])
        config.update(max_tokens=max_output_tokens, max_connections=1, max_retries=0)
        self.model = get_model(profile["model"], config=GenerateConfig(**config), **profile["model_args"])
        self.provider = provider
        self.provider_model = profile["model"]
        self.reasoning_kind = profile["reasoning_kind"]
        if not plan_shas:
            raise ValueError("at least one frozen plan is required")
        self.plan_shas = frozenset(plan_shas)
        self.token = token
        self.max_output_tokens = max_output_tokens
        self.runner = asyncio.Runner()
        self.identity_sha = hashlib.sha256(canonical_json({
            "provider": provider,
            "provider_model": self.provider_model,
            "profile": profile,
            "max_output_tokens": max_output_tokens,
        }).encode()).hexdigest()

    def run(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.runner.run(self.generate(body))

    async def generate(self, body: dict[str, Any]) -> dict[str, Any]:
        requested = body.get("max_tokens")
        if type(requested) is not int or requested <= 0:
            raise ValueError("max_tokens must be a positive integer")
        effective = min(requested, self.max_output_tokens)
        output = await self.model.generate(
            convert_messages(body["messages"]),
            config=GenerateConfig(max_tokens=effective, max_retries=0),
        )
        return {
            "id": getattr(output, "id", None),
            "model": self.provider_model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output.completion,
                    "reasoning": reasoning_text(output),
                    "reasoning_kind": self.reasoning_kind,
                },
                "finish_reason": output.stop_reason,
            }],
            "usage": usage_dict(output),
            "research4": {"requested_max_tokens": requested, "effective_max_tokens": effective},
        }


def handler_type(bridge: HostedBridge):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Research4HostedBridge/1"

        def send_json(self, status: int, value: Any) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(json.dumps({"event": "http", "message": fmt % args}), flush=True)

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_json(404, {"error": "not_found"})
                return
            self.send_json(200, {
                "ready": True,
                "model_id": bridge.provider_model,
                "adapter_sha256": bridge.identity_sha,
                "provider": bridge.provider,
                "max_output_tokens": bridge.max_output_tokens,
                "game_plan_sha256s": sorted(bridge.plan_shas),
            })

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": "not_found"})
                return
            if self.headers.get("Authorization") != "Bearer " + bridge.token:
                self.send_json(401, {"error": "unauthorized"})
                return
            if self.headers.get("X-Research4-Game-Plan-Sha256") not in bridge.plan_shas:
                self.send_json(409, {"error": "plan_mismatch"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(size))
                value = bridge.run(body)
            except Exception as error:
                self.send_json(500, {"error": type(error).__name__, "message": str(error)[:1000]})
                return
            self.send_json(200, value)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(PROFILES), required=True)
    parser.add_argument("--plan", type=Path, action="append", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token-env", default="RESEARCH4_HOSTED_GAME_TOKEN")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    args = parser.parse_args()
    load_dotenv(fine_tune_root() / ".env")
    token = os.environ.get(args.token_env)
    if not token:
        raise ValueError(f"missing private loopback token in {args.token_env}")
    plan_paths = [path.resolve(strict=True) for path in args.plan]
    plan_shas = {hashlib.sha256(path.read_bytes()).hexdigest() for path in plan_paths}
    if len(plan_shas) != len(plan_paths):
        raise ValueError("frozen provider plans must be byte-distinct")
    bridge = HostedBridge(args.provider, plan_shas, token, args.max_output_tokens)
    server = HTTPServer(("127.0.0.1", args.port), handler_type(bridge))
    print(json.dumps({"ready": True, "provider": args.provider, "model": bridge.provider_model,
                      "port": args.port, "game_plan_sha256s": sorted(plan_shas),
                      "identity_sha256": bridge.identity_sha}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
