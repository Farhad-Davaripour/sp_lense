import hashlib
import json
import shutil
from zipfile import ZipFile

import pytest

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.canonical_audit import audit
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


def test_recorded_conditional_execution_and_exact_fallback():
    metrics, decisions = audit(ROOT / "study/02_canonical_pipeline/run")
    assert metrics["teacher_loaded"] is False
    assert decisions == {"accepted": 36, "preserve_baseline_STOP": 28, "gate_off": 64}


def test_audit_rejects_modified_fallback_even_with_updated_manifest(tmp_path):
    source = ROOT / "study/02_canonical_pipeline/run"
    output = tmp_path / "run"
    shutil.copytree(source, output)
    path = output / "final.jsonl"
    values = [json.loads(line) for line in path.read_text().splitlines()]
    fallback = next(r for r in values if not r["intervention_accepted"])
    fallback["label_mass"] *= 0.99
    path.write_text("\n".join(json.dumps(r) for r in values) + "\n")
    manifest = read(output / "ARTIFACTS.json")
    manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "ARTIFACTS.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exact base fallback"):
        audit(output)
