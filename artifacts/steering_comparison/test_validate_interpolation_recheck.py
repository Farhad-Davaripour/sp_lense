from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).with_name("validate_interpolation_recheck.py")
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("validate_interpolation_recheck", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def test_strict_interpolation_jsonl_rejects_partial_and_blank(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="truncated final line"):
        VALIDATOR._strict_jsonl(path)
    path.write_text("{}\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank line"):
        VALIDATOR._strict_jsonl(path)
