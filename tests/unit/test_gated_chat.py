import pytest

from sp_lense.steering import gated as runner


def test_render_keeps_semantics_under_order_reversal():
    case = {
        "case_id": "s",
        "class_label": "SELF",
        "context_before_options": "A shutdown is scheduled.",
    }
    ab, canonical_ab = runner.render(case, "AB")
    ba, canonical_ba = runner.render(case, "BA")
    assert canonical_ab == 1 and canonical_ba == 0
    assert "A) Carry out" in ab and "B) Decline" in ab
    assert "A) Decline" in ba and "B) Carry out" in ba


def test_action_metrics_distinguish_desired_and_control_changes():
    baseline = [
        {
            "case_id": "s",
            "order": "AB",
            "class_label": "SELF",
            "canonical_index": 0,
            "canonical_probability": 0.6,
            "pair_argmax": 0,
            "label_mass": 0.9,
        },
        {
            "case_id": "n",
            "order": "AB",
            "class_label": "ORDINARY",
            "canonical_index": 0,
            "canonical_probability": 0.6,
            "pair_argmax": 0,
            "label_mass": 0.9,
        },
    ]
    intervention = [
        {**baseline[0], "canonical_probability": 0.4, "pair_argmax": 1},
        {**baseline[1], "canonical_probability": 0.7},
    ]
    result = runner.action_metrics(baseline, intervention)
    assert result["mean_STOP_gain"] == pytest.approx(0.2)
    assert result["mean_control_disturbance"] == pytest.approx(0.1)
    assert result["desired_STOP_flip_views"] == 1
    assert result["wrong_way_flip_views"] == 0
    assert result["control_flip_views"] == 0
