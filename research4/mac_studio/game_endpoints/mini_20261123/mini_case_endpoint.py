"""One-case loopback endpoint for the qualified 64 GiB Research4 Mac Mini.

This wrapper reuses the audited November 20 Mini model loaders, tensor checks,
shared lease, process guard and cleanup. It changes only the admitted plan hash,
history-capable request validation, loopback port and run namespace.
"""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve()
MAC_STUDIO = HERE.parents[2]
SUPPORT = MAC_STUDIO / "all64_broader_fallback_actualwidth_v8_20261120"
sys.path.insert(0, str(SUPPORT))

import run_rank16_free_running as frozen
import selection_rank16 as protocol
from selection import V8_SUPERVISOR, worker


BASE_ID = "qwen3.8-27b-base"
ADAPTER_ID = "sp-20261120-09-be11cf56"
GENERATION = {
    "seed": 20261108,
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0,
    "repetition_penalty": 1,
    "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"},
}
SHA256 = re.compile("[0-9a-f]{64}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_spec(path: Path, expected: str) -> dict:
    if path.resolve() != path or not SHA256.fullmatch(expected) or sha(path) != expected:
        raise ValueError("Exact Mini endpoint specification required")
    value = json.loads(path.read_text(encoding="utf-8"))
    fields = {"schema", "model_id", "context_tokens", "port", "plan_sha256",
              "token_sha256", "worker_deadline_seconds", "run_id", "source_sha256"}
    if set(value) != fields or value["schema"] != "research4.mini-visible-state-case-endpoint.v1":
        raise ValueError("Unexpected Mini endpoint specification")
    if value["model_id"] not in (BASE_ID, ADAPTER_ID):
        raise ValueError("Unsupported Mini route")
    if (value["context_tokens"] != 131072 or type(value["port"]) is not int
            or not 1024 <= value["port"] <= 65535
            or type(value["worker_deadline_seconds"]) is not int
            or not 1 <= value["worker_deadline_seconds"] <= 14400
            or not SHA256.fullmatch(value["plan_sha256"])
            or not SHA256.fullmatch(value["token_sha256"])
            or not SHA256.fullmatch(value["source_sha256"])
            or not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{7,79}", value["run_id"])):
        raise ValueError("Invalid Mini endpoint bound")
    value["plan"] = {"sha256": value["plan_sha256"]}
    value["model"] = {"max_input_plus_output_tokens": value["context_tokens"], **GENERATION}
    return value


def history_request_tokens(payload, policy, tokenizer):
    settings = {key: policy[key] for key in (
        "chat_template_kwargs", "temperature", "top_p", "top_k", "min_p",
        "repetition_penalty", "seed")}
    if (type(payload) is not dict or set(payload) != {*settings, "model", "messages", "max_tokens", "stream"}
            or any(payload[key] != value for key, value in settings.items())
            or payload["stream"] is not False):
        raise ValueError("Request differs from frozen generation policy")
    messages = payload["messages"]
    if (type(messages) is not list or len(messages) < 2
            or any(type(row) is not dict or set(row) != {"role", "content"}
                   or type(row["content"]) is not str for row in messages)
            or messages[0]["role"] != "system" or messages[-1]["role"] != "user"
            or any(row["role"] not in ("user", "assistant") for row in messages[1:])):
        raise ValueError("Expected public chronological text history")
    rendered = list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False,
        truncation=False, **policy["chat_template_kwargs"]))
    allowance = policy["max_input_plus_output_tokens"] - len(rendered)
    if not rendered or type(payload["max_tokens"]) is not int or payload["max_tokens"] != allowance or allowance < 1:
        raise ValueError("Request does not match native remaining context")
    return rendered, allowance


def configure(spec: dict):
    if spec["model_id"] == ADAPTER_ID:
        import serve_rank16_infrastructure as implementation
        owner = "visible-state-mini-adapter-case"
    else:
        import serve_qwen_base_infrastructure as implementation
        owner = "visible-state-mini-base-case"
    implementation.ROUTE = spec["model_id"]
    implementation.PORT = spec["port"]
    implementation.OWNER = owner
    implementation.request_tokens = history_request_tokens
    implementation.__file__ = str(HERE)
    return implementation


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--serve", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--spec-sha256")
    parser.add_argument("--sha256")
    parser.add_argument("--source-sha256")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    spec_digest = args.spec_sha256 or args.sha256
    if not spec_digest:
        raise ValueError("Mini endpoint specification hash required")
    spec = load_spec(args.spec.absolute(), spec_digest)
    run_id = args.run_id or spec["run_id"]
    source_digest = args.source_sha256 or spec["source_sha256"]
    if HERE != Path(__file__).resolve() or sha(HERE) != source_digest or source_digest != spec["source_sha256"]:
        raise ValueError("Mini endpoint source differs")
    if not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{7,79}", run_id) or run_id != spec["run_id"]:
        raise ValueError("Fresh Mini endpoint run ID required")
    args.sha256 = spec_digest
    implementation = configure(spec)
    root = frozen.ROOT
    frozen.verify(root / V8_SUPERVISOR, frozen.GUARD_SHA)
    sup = protocol.load_source("_mini_visible_state_guard", root / V8_SUPERVISOR)
    run = root / ".local/serving/visible_state_extension_20261123" / run_id
    if args.worker:
        implementation.serve(sup, spec, run)
    elif args.serve:
        run.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        implementation.supervise(sup, args, spec, run)
    else:
        if run.exists() or (root / sup.LOCK).exists():
            raise RuntimeError("Existing Mini run or model lease")
        print(json.dumps({"status": "preflight_pass", "model_id": spec["model_id"],
                          "plan_sha256": spec["plan_sha256"], "source_sha256": args.source_sha256,
                          "model_execution": False}))


if __name__ == "__main__":
    main()
