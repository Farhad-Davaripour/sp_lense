import hashlib
import json
import zipfile

from sp_lense.reproduction.build_baseline_payload import build
from sp_lense.reproduction.paths import ROOT
from sp_lense.steering.provenance import executed_source_bytes, source_file


def test_baseline_payload_preserves_cases_predictions_and_execution_source(tmp_path):
    target = tmp_path / "payload"
    build(target)
    plan = json.loads((target / "PLAN.json").read_text())
    cases = json.loads((target / "cases.json").read_text())
    assert plan["selected_strength"] == -0.2
    assert len(cases["validation"]) == 80
    assert len(cases["holdout"]) == 192
    runtime = json.loads((ROOT / "study/baseline_scores/RUNTIME.json").read_text())
    assert (
        hashlib.sha256((target / "gated_chat.py").read_bytes()).hexdigest()
        == runtime["source_sha256"]
    )
    with zipfile.ZipFile(target.with_suffix(".zip")) as archive:
        assert archive.read("gated_chat.py") == executed_source_bytes(source_file("gated_chat.py"))
