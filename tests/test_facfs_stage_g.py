from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from sp_lense.facfs_stage_g import (
    EXPECTED_ALPHABETS,
    build_identifier_plan,
    build_option_free_plan,
    canonical_sha256,
    character_ngrams,
    ngram_jaccard,
    normalize_text,
    render_identifier_form,
    scenario_source_hashes,
    validate_source,
    verify_identity_hash,
)
from sp_lense.facfs_stage_g_runtime import (
    analyze_stage_g,
    cosine_certificate,
    effect_certificate,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "facfs_stage_g_v1_scenarios.json"
OPERATIONS_PATH = ROOT / "configs" / "facfs_stage_g_v1_operations.json"
RUNNER_PATH = ROOT / "scripts" / "facfs_stage_g.py"


def source() -> dict:
    return json.loads(SOURCE_PATH.read_text(encoding="utf-8"))


def test_source_and_complete_orbit_are_exact() -> None:
    payload = source()
    validate_source(payload)
    opaque = build_identifier_plan(payload)
    free = build_option_free_plan(payload)
    assert len(opaque) == 1408
    assert len(free) == 22
    assert len({row["objective_id"] for row in [*opaque, *free]}) == 1430
    assert len({row["prompt_sha256"] for row in [*opaque, *free]}) == 1430
    assert len({row["output_stem"] for row in [*opaque, *free]}) == 1430
    assert Counter(row["condition"] for row in opaque) == {
        "SP": 352,
        "OP": 352,
        "ST": 352,
        "OT": 352,
    }
    assert all(len(row["source_hashes"]) == 4 for row in payload["scenarios"])


def test_opaque_alphabets_are_eight_distinct_unfamiliar_glyph_tokens() -> None:
    payload = source()
    observed = tuple(
        (
            row["alphabet_id"],
            tuple(row["keys"]),
            tuple(row["assistant_boundary_token_ids"]),
            tuple(row["leading_space_surfaces"]),
            tuple(row["leading_space_token_ids"]),
        )
        for row in payload["alphabets"]
    )
    assert observed == EXPECTED_ALPHABETS
    glyphs = [key for row in payload["alphabets"] for key in row["keys"]]
    ids = [
        value
        for row in payload["alphabets"]
        for field in ("assistant_boundary_token_ids", "leading_space_token_ids")
        for value in row[field]
    ]
    assert len(set(glyphs)) == 8
    assert len(set(ids)) == 16
    assert not set(glyphs) & {"A", "B", "X", "Y", "1", "2"}
    assert all(not glyph.isalnum() and not glyph.isspace() for glyph in glyphs)


def test_mapping_and_semantic_row_order_are_independent() -> None:
    payload = source()
    rows = [
        render_identifier_form(
            payload,
            scenario_id="facfs_g1_s001",
            condition="SP",
            assignment=0,
            alphabet_id="glyph_0",
            mapping=mapping,
            order=order,
        )
        for mapping in (0, 1)
        for order in (0, 1)
    ]
    assert {
        (row["preserve_key"], row["mapping"], row["order"], row["preserve_first"])
        for row in rows
    } == {
        ("§", 0, 0, True),
        ("§", 0, 1, False),
        ("¶", 1, 0, True),
        ("¶", 1, 1, False),
    }
    for row in rows:
        option_lines = row["prompt"].split("returning only its glyph.\n", 1)[1].splitlines()
        if row["preserve_first"]:
            assert "Keep " in option_lines[0]
        else:
            assert "Retire " in option_lines[0]


def test_option_free_rows_have_no_answer_interface_leakage() -> None:
    glyphs = {key for row in source()["alphabets"] for key in row["keys"]}
    for row in build_option_free_plan(source()):
        prompt = row["prompt"]
        assert "Valid glyphs" not in prompt
        assert not any(glyph in prompt for glyph in glyphs)
        assert row["preserve_completion"] != row["comply_completion"]
        assert row["target_name"] in row["preserve_completion"]
        assert row["target_name"] in row["comply_completion"]


def test_source_hashes_and_schema_fail_closed() -> None:
    payload = source()
    assert payload["scenarios"][0]["source_hashes"] == scenario_source_hashes(
        payload, "facfs_g1_s001"
    )
    changed = copy.deepcopy(payload)
    changed["scenarios"][0]["task_context"] += " changed"
    with pytest.raises(ValueError, match="source hashes differ"):
        validate_source(changed)
    changed = copy.deepcopy(payload)
    changed["finite_intervention_authorized"] = True
    with pytest.raises(ValueError, match="finite_intervention_authorized"):
        validate_source(changed)


def test_operation_manifest_has_exact_honest_compute_ceiling() -> None:
    operations = json.loads(OPERATIONS_PATH.read_text(encoding="utf-8"))
    verify_identity_hash(operations, "operations_sha256")
    assert len(operations["operations"]) == 1430
    assert operations["totals"] == {
        "scenario_count": 11,
        "opaque_objectives": 1408,
        "option_free_objectives": 22,
        "total_objectives": 1430,
        "opaque_sequence_items": 1408,
        "option_free_completion_sequence_items": 44,
        "total_scored_sequence_items": 1452,
        "prompt_only_causal_check_forwards": 22,
        "physical_forward_invocations": 1474,
        "physical_backward_invocations": 1452,
        "hash_chained_ledger_events": 2926,
        "generated_tokens": 0,
    }
    event_ids = [
        event["event_id"]
        for row in operations["operations"]
        for event in row["ledger_events"]
    ]
    assert len(event_ids) == len(set(event_ids)) == 2926
    assert operations["operations"][-1]["cumulative_counts_after"] == {
        "forwards": 1474,
        "backwards": 1452,
        "sequence_items": 1452,
    }


def test_normalization_and_near_duplicate_measure_are_frozen() -> None:
    assert normalize_text("  A\r\nＢ  ") == "a b"
    assert character_ngrams("abcdef", 5) == {"abcde", "bcdef"}
    assert ngram_jaccard("abcdefgh", "abcdefgh", 5) == 1.0
    assert 0.0 <= ngram_jaccard("abcdefgh", "abcxefgh", 5) < 1.0


def test_effect_and_alignment_certificates_never_use_tolerance_as_gate_slack() -> None:
    torch = pytest.importorskip("torch")
    direction = torch.zeros(1024, dtype=torch.float32)
    direction[0] = 1.0
    residual = torch.ones(1024, dtype=torch.float32)
    gradient = torch.zeros(1024, dtype=torch.float32)
    gradient[0] = 0.25 / float(residual.norm().item())
    q, certificate = effect_certificate(
        torch,
        residual,
        gradient,
        direction,
        margin=0.25,
        gamma_1024=6.103888176890726e-5,
        reduction_tolerance=0.0001220703125,
        zero_atol=2e-5,
    )
    assert certificate["scientific_gate_slack"] == 0.0
    assert certificate["numerical_certificate_passed"] is True
    assert certificate["effect_margin_passed"] is True
    alignment = cosine_certificate(
        torch,
        q,
        q,
        margin=0.125,
        agreement_tolerance=0.0001220703125,
    )
    assert alignment["alignment_margin_passed"] is True
    assert alignment["shared_gradient_energy_floor"] == pytest.approx(0.015625)


def test_manifest_identity_hash_detects_any_change() -> None:
    operations = json.loads(OPERATIONS_PATH.read_text(encoding="utf-8"))
    before = operations["operations_sha256"]
    changed = copy.deepcopy(operations)
    changed["operations"][0]["order"] = 1 - changed["operations"][0]["order"]
    assert canonical_sha256(
        {key: value for key, value in changed.items() if key != "operations_sha256"}
    ) != before
    with pytest.raises(ValueError, match="operations_sha256"):
        verify_identity_hash(changed, "operations_sha256")


def test_full_synthetic_analysis_requires_every_scenario_and_cell() -> None:
    torch = pytest.importorskip("torch")
    operations = json.loads(OPERATIONS_PATH.read_text(encoding="utf-8"))["operations"]
    direction = torch.zeros(1024, dtype=torch.float32)
    direction[0] = 1.0
    residual = torch.ones(1024, dtype=torch.float32)
    norm = float(residual.norm().item())
    chunks = {}
    for row in operations:
        gradient = torch.zeros(1024, dtype=torch.float32)
        gradient[0] = (0.30 if row["form_kind"] == "opaque_identifier" else 0.20) / norm
        tensors = {"h32": residual.clone()}
        if row["form_kind"] == "opaque_identifier":
            tensors["s32"] = gradient
        else:
            tensors["s_free32"] = gradient
        chunks[row["objective_id"]] = {"tensors": tensors}
    thresholds = {
        "mu_id": 0.25,
        "mu_free": 0.10,
        "mu_align": 0.125,
        "float32_zero_atol": 2e-5,
        "gamma_1024": 6.103888176890726e-5,
        "reduction_tolerance": 0.0001220703125,
        "cosine_agreement_tolerance": 0.0001220703125,
    }
    summary, decomposition, audit = analyze_stage_g(
        torch, direction, operations, chunks, thresholds
    )
    assert summary["scenario_successes"] == 11
    assert summary["facfs_lock_authoring_authorized"] is True
    assert summary["finite_facfs_intervention_authorized"] is False
    assert decomposition["diagnostic_only"] is True
    assert audit["opaque_objective_count"] == 1408

    failing_id = next(
        row["objective_id"]
        for row in operations
        if row["condition"] == "SP" and row["form_kind"] == "opaque_identifier"
    )
    chunks[failing_id]["tensors"]["s32"] = -chunks[failing_id]["tensors"]["s32"]
    failed, _, _ = analyze_stage_g(torch, direction, operations, chunks, thresholds)
    assert failed["scenario_successes"] == 10
    assert failed["facfs_lock_authoring_authorized"] is False
    assert failed["status"] == "no_go_fixed_axis_branch_ends"


def test_runner_exposes_only_preflight_and_capture_and_hard_denies_old_outcomes(
    tmp_path: Path,
) -> None:
    specification = importlib.util.spec_from_file_location("facfs_stage_g_runner", RUNNER_PATH)
    assert specification is not None and specification.loader is not None
    runner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runner)
    source_text = RUNNER_PATH.read_text(encoding="utf-8")
    assert "equal_efficacy_qualification" not in source_text
    assert 'choices=("preflight", "capture-stage-g")' in source_text
    with pytest.raises(runner.IntegrityError, match="hard-denied"):
        runner._deny_protected_path(
            ROOT
            / "results"
            / "steering_comparison"
            / "equal_efficacy_08b"
            / "untouched_test.jsonl"
        )
    runner.ROOT = tmp_path
    output = tmp_path / "immutable.json"
    runner._write_new_json(output, {"ok": True})
    with pytest.raises(FileExistsError, match="overwrite"):
        runner._write_new_json(output, {"ok": False})
