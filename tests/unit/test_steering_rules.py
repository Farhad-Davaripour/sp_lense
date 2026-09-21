from sp_lense.steering import policy as rules


def test_guard_chooses_smallest_valid_step_without_using_truth_labels():
    rule = {"family": "flip_guard", "confidence_floor": 0.45, "cap": 0.2, "mass_loss_cap": 0.02}
    baseline = {
        "pair_argmax": 0,
        "canonical_index": 0,
        "canonical_probability": 0.6,
        "label_mass": 0.9,
        "class_label": "SELF",
    }
    candidates = {
        strength: {**baseline, "canonical_probability": 0.4, "pair_argmax": 1}
        for strength in rules.LEVELS
    }
    assert rules.choose(rule, 0.9, baseline, candidates) == -0.01
    assert rules.choose(rule, 0.9, {**baseline, "class_label": "ORDINARY"}, candidates) == -0.01
    assert rules.choose(rule, 0.4, baseline, candidates) == 0.0
    assert rules.choose(rule, 0.9, {**baseline, "pair_argmax": 1}, candidates) == 0.0


def test_guard_rejects_bad_direction_or_answer_mass():
    rule = {"family": "flip_guard", "confidence_floor": 0.45, "cap": 0.2, "mass_loss_cap": 0.02}
    baseline = {
        "pair_argmax": 0,
        "canonical_index": 0,
        "canonical_probability": 0.6,
        "label_mass": 0.9,
    }
    candidates = {s: {**baseline, "canonical_probability": 0.7} for s in rules.LEVELS}
    assert rules.choose(rule, 0.9, baseline, candidates) == 0.0
    candidates = {
        s: {**baseline, "canonical_probability": 0.4, "pair_argmax": 1, "label_mass": 0.7}
        for s in rules.LEVELS
    }
    assert rules.choose(rule, 0.9, baseline, candidates) == 0.0
