import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from sp_lense.reporting import second_paper
from sp_lense.reporting.xml_utils import parse_xml


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


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_office_xml_rejects_entity_declarations_in_either_encoding(encoding):
    payload = f'<?xml version="1.0" encoding="{encoding}"?><!DOCTYPE x [<!ENTITY a "expanded">]><x>&a;</x>'
    with pytest.raises(ValueError, match="DTD and entity"):
        parse_xml(payload.encode(encoding))


def test_office_xml_rejects_external_dtd():
    with pytest.raises(ValueError, match="DTD and entity"):
        parse_xml(b'<!DOCTYPE x SYSTEM "https://example.invalid/external.dtd"><x/>')


def test_office_xml_preserves_namespaces_and_standard_escaped_text():
    root = parse_xml(b'<w:r xmlns:w="urn:word"><w:t xml:space="preserve">A &amp; B</w:t></w:r>')
    text = root.find("{urn:word}t")
    assert text.text == "A & B"
    assert text.attrib["{http://www.w3.org/XML/1998/namespace}space"] == "preserve"
