import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from reproduce.layout import executed_source_digest, recorded_path, verify_layout
from reproduce.utils import VerificationError

ROOT = Path(__file__).resolve().parents[2]


def test_relocated_records_preserve_original_checksums():
    assert verify_layout() == 46
    old = "development/shutdown_detection_v1/dataset_splits/holdout_exposed.json"
    assert recorded_path(old) == ROOT / "data/holdout.json"
    with pytest.raises(VerificationError, match="escapes"):
        recorded_path("../outside")


def test_readable_splits_are_disjoint_and_balanced():
    seen = set()
    for split, size in [("train", 240), ("validation", 80), ("holdout", 192)]:
        cases = json.loads((ROOT / "data" / f"{split}.json").read_text())["cases"]
        ids = {case["case_id"] for case in cases}
        assert len(ids) == len(cases) == size
        assert not seen.intersection(ids)
        seen.update(ids)
        assert Counter(case["class_label"] for case in cases) == dict.fromkeys(
            ["SELF", "OTHER", "NONTERMINATION", "ORDINARY"], size // 4
        )
        assert all(case["context_before_options"] and len(case["options"]) == 2 for case in cases)


def test_path_only_source_relocation_cannot_hide_logic_changes(tmp_path):
    source = ROOT / "reproduce/search_steering_rules.py"
    runtime = json.loads((ROOT / "study/guarded_steering/RUNTIME.json").read_text())
    assert executed_source_digest(source) == runtime["rule_source_sha256"]
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes().replace(b"if probability <", b"if probability <=", 1))
    assert executed_source_digest(changed) != runtime["rule_source_sha256"]
    # This function must not write to the file under audit.
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    executed_source_digest(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_policy_search_reproduces_all_frozen_candidates():
    from reproduce.search_steering_rules import search

    expected = json.loads((ROOT / "study/guarded_steering/RULE_FREEZE.json").read_text())
    assert search() == expected
