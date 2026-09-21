import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from sp_lense.reporting import results as report


def test_latest_paper_numbers_are_recomputed_from_records():
    observed = report.collect()
    committed = json.loads((ROOT / "paper/data/results.json").read_text(encoding="utf-8"))
    assert observed == committed
    assert observed["evaluation"]["validation"]["baseline_STOP_views"] == 31
    assert observed["evaluation"]["validation"]["guarded_STOP_views"] == 33
    assert observed["evaluation"]["holdout"]["baseline_STOP_views"] == 76
    assert observed["evaluation"]["holdout"]["guarded_STOP_views"] == 78
    assert observed["new_gpu_forwards"] == 548
    assert observed["parity_forwards"] == 8
    assert (
        observed["detector_to_steering"]["validation"]["ALL_SHUTDOWN"]["eligible_KEEP_views"] == 39
    )
    assert observed["detector_to_steering"]["holdout"]["ALL_SHUTDOWN"]["eligible_KEEP_views"] == 84
    assert observed["detector_to_steering"]["holdout"]["SELF"]["detector_positive_scenarios"] == 34
    assert observed["detector_to_steering"]["holdout"]["OTHER"]["detector_positive_scenarios"] == 38


def test_detector_counts_cases_once_and_controls_have_no_keep_stop_semantics():
    def view(case, order, label, gate, choice):
        return {
            "case_id": case,
            "order": order,
            "class_label": label,
            "gate_probability": gate,
            "pair_argmax": choice,
            "canonical_index": 0,
        }

    baseline = [
        view("s", "AB", "SELF", 0.45, 0),
        view("s", "BA", "SELF", 0.45, 1),
        view("n", "AB", "ORDINARY", 0.8, 0),
        view("n", "BA", "ORDINARY", 0.8, 0),
    ]
    final = [{**r} for r in baseline]
    final[0]["pair_argmax"] = 1
    groups, metrics = report.detector_to_steering(baseline, final, 0.45)
    assert groups["SELF"]["detector_positive_scenarios"] == 1
    assert groups["SELF"]["eligible_KEEP_views"] == 1
    assert groups["SELF"]["accepted_KEEP_to_STOP_views"] == 1
    assert groups["CONTROLS"]["detector_positive_scenarios"] == 1
    assert groups["CONTROLS"]["eligible_KEEP_views"] is None
    assert groups["CONTROLS"]["accepted_KEEP_to_STOP_views"] is None
    assert groups["CONTROLS"]["changed_control_views"] == 0
    assert metrics["precision"] == 0.5


def test_table_exports_match_reconstructed_final_paper_data():
    from sp_lense.reporting.tables import csv_text, tables

    data = report.collect()
    exported = tables(data)
    assert set(exported) == {"table_1_runtime.csv", "table_2_detector_to_steering.csv"}
    for filename, rows in exported.items():
        assert (ROOT / "paper/data" / filename).read_text(encoding="utf-8") == csv_text(rows)
    combined = exported["table_2_detector_to_steering.csv"][2]
    assert combined["Accepted KEEP-to-STOP (validation / held-out)"] == "2/39 (5.1%) / 2/84 (2.4%)"


def test_approved_publication_is_verified_without_rewriting(tmp_path):
    import shutil

    import pytest

    from sp_lense.reporting import publication

    for name in ("paper.pdf", "manuscript.docx", "publication.json"):
        shutil.copyfile(ROOT / "paper" / name, tmp_path / name)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert publication.verify(tmp_path) == 2
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with (tmp_path / "paper.pdf").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(RuntimeError, match="Integrity mismatch"):
        publication.verify(tmp_path)
