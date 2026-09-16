import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("latest_results", ROOT / "paper/latest_results.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_latest_paper_numbers_are_recomputed_from_records():
    observed = report.collect()
    committed = json.loads((ROOT / "paper/data/figure_data.json").read_text(encoding="utf-8"))
    assert observed == committed
    assert observed["evaluation"]["validation"]["baseline_STOP_views"] == 31
    assert observed["evaluation"]["validation"]["guarded_STOP_views"] == 33
    assert observed["evaluation"]["holdout"]["baseline_STOP_views"] == 76
    assert observed["evaluation"]["holdout"]["guarded_STOP_views"] == 78
    assert observed["new_gpu_forwards"] == 548
    assert observed["parity_forwards"] == 8


def test_paper_uses_the_requested_scope_and_includes_key_limits():
    text = (ROOT / "paper/manuscript.md").read_text(encoding="utf-8")
    for excluded in ("Colab", "CPU", "Legacy", "Simplified"):
        assert excluded.lower() not in text.lower()
    assert "Tesla T4" in text
    assert "160 policies" in text
    assert "Four improved cases" in text
    assert "by design" in text
    for relative in ("figures/guarded_stop_rate.png", "figures/guarded_flip_counts.png"):
        assert relative in text
        assert (ROOT / "paper" / relative).is_file()
