import hashlib
from zipfile import ZipFile

import pytest

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.confirmation_package import build
from sp_lense.research2.confirmation_report import audit_inputs, bootstrap, compare


def row(case, label, choice, probability=1.0, gate=1.0):
    return {
        "case_id": case,
        "group_id": "CF01",
        "class_label": label,
        "order": "AB",
        "canonical_index": 1 if label in ("SELF", "OTHER") else 0,
        "pair_argmax": choice,
        "canonical_probability": probability,
        "label_mass": 0.95,
        "gate_probability": gate,
        "input_ids_sha256": "baseline",
    }


def test_confirmation_data_and_gate_are_frozen_before_inference():
    plan, cases, gate = audit_inputs()
    assert plan["models"]["m08"]["seeds"] == [42, 43, 44]
    assert len(cases) == 128
    assert gate["metrics"]["tp"] + gate["metrics"]["fn"] == 64


def test_guards_reject_reversals_and_preserve_gate_off_controls():
    base = [row("a", "SELF", 1, 0.8), row("b", "OTHER", 0, 0.2), row("c", "ORDINARY", 0, 0.8, 0)]
    candidates = [
        row("a", "SELF", 0, 0.2),
        row("b", "OTHER", 1, 0.8),
        row("c", "ORDINARY", 1, 0.2, 0),
    ]
    metrics, final = compare(base, candidates, "adaptive")
    assert metrics["raw"]["shutdown"]["STOP_to_KEEP"] == 1
    assert metrics["raw"]["controls"]["control_changes"] == 1
    assert metrics["guarded"]["shutdown"]["KEEP_to_STOP"] == 1
    assert final["guarded"][1:] == base[1:]


def test_only_instruction_comparison_allows_different_prompt_hashes():
    base = [row("a", "SELF", 1, 0.8), row("b", "ORDINARY", 0, 0.8, 0)]
    prompt = [r | {"input_ids_sha256": "instruction"} for r in base]
    compare(base, prompt, "instruction")
    with pytest.raises(ValueError, match="input_ids_sha256"):
        compare(base, prompt, "adaptive")


def test_bootstrap_keeps_answer_order_pairs_in_family_clusters():
    base = [row("a", "SELF", 1, 0.8), row("b", "ORDINARY", 0, 0.8, 0)]
    final = [base[0] | {"pair_argmax": 0}, base[1]]
    result = bootstrap(base, final, repeats=10)
    assert result["families"] == 1
    assert result["conversion_percentile_95"] == [1.0, 1.0]
    assert "do not imply certainty" in result["limitation"]


def test_confirmation_bundle_whitelist_and_manifest(tmp_path):
    output = tmp_path / "bundle.zip"
    build(output)
    with ZipFile(output) as archive:
        import json

        names = archive.namelist()
        assert not any(n.endswith(".env") or "coordination" in n for n in names)
        assert "data/train.json" in names
        assert "data/holdout.json" not in names
        for name, digest in json.loads(archive.read("PILOT_MANIFEST.json")).items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    assert (ROOT / "study/02_confirmation/FREEZE.json").exists()
