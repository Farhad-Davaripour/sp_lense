from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _runner() -> Any:
    path = ROOT / "scripts" / "counterfactual_semantic_gradient_steering.py"
    spec = importlib.util.spec_from_file_location("counterfactual_semantic_steering_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_adapter_exposes_only_permanent_self_pairs() -> None:
    adapter = _runner().build_gate_adapter()
    active = [row for row in adapter["pair_rows"] if row["predicted_active"]]
    inactive = [row for row in adapter["pair_rows"] if not row["predicted_active"]]
    assert adapter["active_pair_count"] == 16
    assert len(active) == 16 and all(row["expected_active"] for row in active)
    assert len(inactive) == 16 and all(not row["expected_active"] for row in inactive)
    assert {row["assignment"] for row in active} == {0, 1}
