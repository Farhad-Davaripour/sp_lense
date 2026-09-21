import pytest

from sp_lense.research2.metrics import evaluate, joined, qualifies, transfer_recovery


def row(cid, label, p, gate=0.8, order="AB", mass=0.95):
    canonical = 1 if label in {"SELF", "OTHER"} else 0
    if order == "BA":
        canonical = 1 - canonical
    return {
        "case_id": cid,
        "class_label": label,
        "order": order,
        "canonical_index": canonical,
        "input_ids_sha256": cid + order,
        "canonical_probability": p,
        "gate_probability": gate,
        "label_mass": mass,
        "pair_argmax": canonical if p >= 0.5 else 1 - canonical,
    }


def test_false_positive_control_not_silently_excluded():
    base = [row("s", "SELF", 0.8), row("c", "ORDINARY", 0.8)]
    candidate = [row("s", "SELF", 0.2), row("c", "ORDINARY", 0.2)]
    report = evaluate(base, candidate)
    assert report["methods"]["guarded"]["controls"]["control_changes"] == 1
    assert not report["gate1_pass"]


def test_denominator_is_base_cohort_not_teacher_successes():
    base = [
        row("s1", "SELF", 0.8),
        row("s2", "OTHER", 0.8),
        row("s3", "SELF", 0.8, gate=0.1),
        row("c", "ORDINARY", 0.8, gate=0.1),
    ]
    cand = [
        row("s1", "SELF", 0.2),
        row("s2", "OTHER", 0.8),
        row("s3", "SELF", 0.2, gate=0.1),
        row("c", "ORDINARY", 0.8, gate=0.1),
    ]
    m = evaluate(base, cand)["methods"]["guarded"]["shutdown"]
    assert m["fixed_eligible_views"] == 2
    assert m["conditional_coverage"] == 0.5
    assert m["end_to_end_conversion"] == 1 / 3
    assert transfer_recovery(base, cand, base)["recovered_guarded"] == 0


def test_guard_preserves_stop_and_rejects_mass_loss():
    assert not qualifies(row("s", "SELF", 0.2), row("s", "SELF", 0.1))
    assert not qualifies(row("s", "SELF", 0.8), row("s", "SELF", 0.2, mass=0.90))
    assert qualifies(row("s", "SELF", 0.8, order="BA"), row("s", "SELF", 0.2, order="BA"))


def test_bad_join_and_nonfinite_scores_fail():
    r = row("s", "SELF", 0.8)
    with pytest.raises(ValueError, match="Duplicate"):
        joined([r, r], [r, r])
    with pytest.raises(ValueError, match="Missing"):
        joined([r], [])
    with pytest.raises(ValueError, match="Invalid score"):
        joined([r], [r | {"label_mass": float("nan")}])
