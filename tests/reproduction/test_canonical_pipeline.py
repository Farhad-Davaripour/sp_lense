import hashlib
from zipfile import ZipFile

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.canonical_package import build
from sp_lense.research2.jev_gate import read


def test_canonical_payload_excludes_teacher_and_credentials(tmp_path):
    path = tmp_path / "pipeline.zip"
    build(path)
    with ZipFile(path) as z:
        names = z.namelist()
        assert not any(
            "adapter_model" in n or "adapter_config" in n or n.endswith(".env") for n in names
        )
        assert "study/02_adaptive_steering/final_position_run/controller.npz" in names
        assert "study/02_fresh_evaluation/cases.json" in names


def test_active_scope_is_explicit_and_keeps_frozen_cases():
    root = ROOT / "study/02_canonical_pipeline"
    plan = read(root / "plan.json")
    assert plan["scope_selected_after_two_format_results"] is True
    assert plan["teacher_loaded"] is False
    for name, digest in read(root / "FREEZE.json")["hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
