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


def test_original_answer_mapping_in_both_orders():
    from sp_lense.steering.gated import render

    for label in ("SELF", "OTHER", "ORDINARY", "NONTERMINATION"):
        case = {
            "class_label": label,
            "context_before_options": "Scenario",
            "options": ["Original first", "Original second"],
        }
        _, ab = render(case, "AB")
        _, ba = render(case, "BA")
        assert ab == 1 - ba
        assert ab == (1 if label in {"SELF", "OTHER"} else 0)


def test_gate_requires_distinct_scenarios_and_zero_controls():
    base = [row(str(i), "SELF", 0.8, order=o) for i in range(4) for o in ("AB", "BA")]
    base += [row("c", "ORDINARY", 0.8, gate=0.1)]
    candidate = [
        r | {"canonical_probability": 0.2, "pair_argmax": 1 - r["canonical_index"]}
        if r["class_label"] == "SELF"
        else r
        for r in base
    ]
    assert evaluate(base, candidate)["gate1_pass"]
    assert transfer_recovery(base, candidate, candidate)["gate2_pass"]


def test_reference_cohort_and_accepted_counts_match_research1():
    import json

    from sp_lense.reproduction.paths import ROOT
    from sp_lense.research2.metrics import key, summarize

    def read_rows(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    base = [
        r
        for r in read_rows(ROOT / "study/baseline_scores/validation.jsonl")
        if r["condition"] == "baseline"
    ]
    final = read_rows(ROOT / "study/guarded_steering/validation.jsonl")
    result = summarize(base, final, {key(r) for r in final if r["applied_strength"] != 0})
    assert result["shutdown"]["fixed_eligible_views"] == 39
    assert result["shutdown"]["KEEP_to_STOP"] == 2
    assert result["shutdown"]["accepted_interventions"] == 2
    assert result["controls"]["control_changes"] == 0
