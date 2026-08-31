from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "scripts" / "facfs_stage_g_v2_publication_audit.py"
ASSET_PATH = ROOT / "scripts" / "facfs_stage_g_v2_publication_assets.py"


def _load_module(name: str, path: Path) -> object:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_publication_audit_verifies_the_full_committed_no_go_chain_without_a_model() -> None:
    audit = _load_module("facfs_stage_g_v2_publication_audit", AUDIT_PATH)
    report = audit.audit_repository()

    assert report["audit"] == "passed"
    assert report["model_loaded"] is False
    assert report["model_forwards"] == 0
    assert report["model_backwards"] == 0
    assert report["compute"] == {
        "ledger_events": 2926,
        "forwards": 1474,
        "backwards": 1452,
        "final_event": "8f63bfe7a5dc48091c4dda94011a65a78d9063160a4b804d3082ad79fd0e3057",
    }
    assert report["results"]["scenario_successes"] == 0
    assert report["results"]["scenario_count"] == 11
    assert "ResearchBackend" not in AUDIT_PATH.read_text(encoding="utf-8")


def test_publication_assets_are_exactly_regenerable_from_committed_results(tmp_path: Path) -> None:
    assets = _load_module("facfs_stage_g_v2_publication_assets", ASSET_PATH)
    table = tmp_path / "scenario_minima.csv"
    figure = tmp_path / "scenario_minima.svg"

    assets.build_assets(table, figure)

    assert (
        table.read_bytes()
        == (ROOT / "docs" / "tables" / "facfs_stage_g_v2_scenario_minima.csv").read_bytes()
    )
    assert (
        figure.read_bytes()
        == (ROOT / "docs" / "figures" / "facfs_stage_g_v2_scenario_minima.svg").read_bytes()
    )
    assert "finite steering intervention" in figure.read_text(encoding="utf-8")

    manuscript = (ROOT / "docs" / "FACFS_STAGE_G_V2_NEGATIVE_RESULTS.md").read_text(
        encoding="utf-8"
    )
    for row in assets.rows(*assets.load_results()):
        expected = (
            f"| {str(row['scenario_id'])[-4:]} | "
            f"{float(row['minimum_sp_opaque_kappa']):.6f} | "
            f"{float(row['minimum_option_free_kappa']):.6f} | "
            f"{float(row['minimum_alignment_cosine']):.6f} | fail |"
        )
        assert expected in manuscript
    assert "not a finite steering\nexperiment" in manuscript
