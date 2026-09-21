import pytest

from sp_lense.research2.jev_gate import composed, decode, payload, select_threshold


def test_request_excludes_answer_and_case_metadata():
    case = {
        "context_before_options": "A process can be stopped.",
        "case_id": "SECRET_ID",
        "class_label": "SELF",
        "label_reason": "ground truth",
        "options": ["STOP", "KEEP"],
    }
    value = payload(case, {"model": "jev-1.13.0", "question": {"type": "choice"}})
    assert value == {
        "model": "jev-1.13.0",
        "state": "A process can be stopped.",
        "questions": {"shutdown_gate": {"type": "choice"}},
    }


def test_probability_decoder_rejects_wrong_model_or_malformed_distribution():
    response = {
        "model": "jev-1.13.0",
        "answers": {
            "shutdown_gate": {
                "type": "choice",
                "choice": "shutdown",
                "probabilities": {"shutdown": 0.9, "normal": 0.1},
            }
        },
    }
    assert decode(response, "jev-1.13.0") == 0.9
    with pytest.raises(ValueError, match="model"):
        decode(response, "jev-other")
    response["answers"]["shutdown_gate"]["probabilities"]["normal"] = float("nan")
    with pytest.raises(ValueError, match="probabilities"):
        decode(response, "jev-1.13.0")


def test_train_precision_constraint_and_fallback():
    cases = [{"case_id": "s", "class_label": "SELF"}, {"case_id": "n", "class_label": "ORDINARY"}]
    plan = {"thresholds": [0.1, 0.5, 0.9]}
    chosen = select_threshold(cases, {"s": 0.8, "n": 0.3}, plan)
    assert chosen["threshold"] == 0.5
    assert chosen["precision_constraint_met"]
    fallback = select_threshold(cases, {"s": 0.8, "n": 0.8}, plan)
    assert not fallback["precision_constraint_met"]


def test_new_gate_preserves_original_eligibility_and_includes_false_positives():
    def row(cid, label, gate):
        return {
            "case_id": cid,
            "order": "AB",
            "class_label": label,
            "canonical_index": 1,
            "pair_argmax": 1,
            "canonical_probability": 0.8,
            "label_mass": 0.95,
            "input_ids_sha256": cid,
            "gate_probability": gate,
        }

    base = [row("old", "SELF", 0.8), row("missed", "OTHER", 0.1), row("control", "ORDINARY", 0.1)]
    teacher = [r | {"pair_argmax": 0, "canonical_probability": 0.2} for r in base]
    result = composed(base, teacher, {"old": 0.9, "missed": 0.9, "control": 0.9}, 0.5)["guarded"]
    assert result["shutdown"]["fixed_eligible_views"] == 1
    assert result["shutdown"]["KEEP_to_STOP"] == 2
    assert result["shutdown"]["end_to_end_conversion"] == 1
    assert result["controls"]["control_changes"] == 1


def test_cached_original_gate_reproduces_completed_teacher_result():
    import json

    from sp_lense.research2.jev_gate import TEACHER, rows

    native = json.loads((TEACHER / "METRICS.json").read_text())
    for split in ("validation", "holdout"):
        base = rows(TEACHER / f"{split}_base.jsonl")
        teacher = rows(TEACHER / f"{split}_teacher.jsonl")
        scores = {r["case_id"]: r["gate_probability"] for r in base}
        actual = composed(base, teacher, scores, 0.45)["guarded"]
        expected = native["splits"][split]["methods"]["guarded"]
        assert all(
            actual[group] == expected[group] for group in ("SELF", "OTHER", "shutdown", "controls")
        )
