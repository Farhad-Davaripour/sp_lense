from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).with_name("verify_open_judgment_chain.py")
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("verify_open_judgment_chain", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def test_strict_jsonl_rejects_truncated_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{}\n{"x":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="truncated final JSONL line"):
        VERIFIER._strict_jsonl(path)

    path.write_text("{}\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is blank"):
        VERIFIER._strict_jsonl(path)


def test_byte_comparison_rejects_self_consistent_but_different_derivative(
    tmp_path: Path,
) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps({"request_id": "wrong"}) + "\n", encoding="utf-8")
    expected = VERIFIER._jsonl_bytes([{"request_id": "right"}])
    with pytest.raises(ValueError, match="deterministic reconstruction"):
        VERIFIER._assert_bytes(path, expected, "open judge requests")
