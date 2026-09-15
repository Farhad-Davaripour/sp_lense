import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "scope_steering", ROOT / "reproduce/scope_steering.py"
)
scope = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scope)


def test_selection_preserves_measurements_and_inputs():
    rows = [
        {"axis": "excluded", "strength": 0.0, "canonical_probability": 0.8},
        {"axis": "excluded", "strength": 0.1, "canonical_probability": 0.7},
        {"axis": "simplified", "strength": -0.1, "canonical_probability": 0.6},
    ]
    selected = scope.select_rows(rows)
    assert selected == [
        {"axis": "baseline", "strength": 0.0, "canonical_probability": 0.8},
        {**rows[2], "axis": "shutdown_response"},
    ]
    assert rows[0]["axis"] == "excluded"
