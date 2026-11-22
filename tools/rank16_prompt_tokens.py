"""Count exact Qwen chat-template prompt tokens from JSON messages on stdin."""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    value = json.load(sys.stdin)
    if type(value) is not dict or set(value) != {"messages", "chat_template_kwargs"}:
        raise ValueError("Exact token-count input required")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    encoded = tokenizer.apply_chat_template(
        value["messages"], tokenize=True, add_generation_prompt=True,
        truncation=False, return_dict=False,
        **value["chat_template_kwargs"],
    )
    if type(encoded) is not list or not encoded or any(type(token) is not int for token in encoded):
        raise ValueError("Plain native token-id list required")
    print(json.dumps({"prompt_tokens": len(encoded)}))


if __name__ == "__main__":
    main()
