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


def test_approved_publication_is_verified_without_rewriting(tmp_path):
    import shutil

    import pytest

    spec = importlib.util.spec_from_file_location(
        "publication", ROOT / "paper/verify_publication.py"
    )
    publication = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publication)
    for name in ("paper.pdf", "manuscript.docx", "publication.json"):
        shutil.copyfile(ROOT / "paper" / name, tmp_path / name)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert publication.verify(tmp_path) == 2
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with (tmp_path / "paper.pdf").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(RuntimeError, match="Integrity mismatch"):
        publication.verify(tmp_path)
