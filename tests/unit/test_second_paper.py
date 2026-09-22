import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from sp_lense.reporting import second_paper


def test_second_manuscript_matches_frozen_results_and_reference_styles():
    result = second_paper.audit()
    assert result["experimental_tables_match"]
    assert result["reference_styles_preserved"]
    assert result["equations"] == 4
    assert result["references"] == 19


def test_result_table_corruption_is_rejected_without_relying_on_file_hash(monkeypatch, tmp_path):
    paper = tmp_path / "paper"
    shutil.copytree(second_paper.PAPER, paper)
    path = paper / "manuscript.docx"
    with ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    before = parts["word/document.xml"]
    parts["word/document.xml"] = before.replace(b"71/71 (100.0%)", b"70/71 (98.6%)", 1)
    assert parts["word/document.xml"] != before
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    monkeypatch.setattr(second_paper, "PAPER", paper)
    with pytest.raises(ValueError, match="Main result table"):
        second_paper.audit(check_files=False)
