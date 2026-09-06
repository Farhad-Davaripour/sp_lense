"""Metadata-only evidence gate tests; never rerun numerical or full-native cases."""

import copy
import hashlib
import json

import pytest

from scripts import certified_descent_comply_recording as recording
from scripts import verify_certified_descent_comply as checker


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_results(root, certificate, evidence):
    # NaN literals are intentionally allowed here solely for hostile-input tests.
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    target = root / recording.RESULTS
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    certificate["source_sha256"][recording.RESULTS] = digest(raw)
    certificate["model_free_results_sha256"] = digest(raw)


def fixture(root):
    sources = {}
    coverage = {
        key: ["synthetic_metadata_fixture::" + key] for key in recording.METHOD_COVERAGE_KEYS
    }
    for index, path in enumerate(recording.TEST_LOCKS):
        lock = (
            {"schema": "sp_lense.certified_descent_comply_model_free_test_lock.v1"}
            if index == 0
            else {
                "schema": "sp_lense.certified_descent_comply_metadata_test_lock.v1",
                "original_test_lock_sha256": sources[recording.TEST_LOCKS[0]],
                "metadata_only": True,
                "full_cases_repeated": False,
                "method_coverage_evidence": coverage,
            }
        )
        raw = json.dumps(lock).encode()
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        sources[path] = digest(raw)
    attempts = [
        {
            "name": name,
            "category": category,
            "external_seconds": seconds,
            "returncode": 0,
            "timed_out": False,
            "hard_timeout_seconds": cap,
        }
        for name, category, seconds, cap in (
            ("initial_saved", "initial_saved", 0.75, 10),
            ("ancillary_tests_1", "ancillary", 1.0, 10),
            ("full_native_interior_candidate", "full_interior_candidate", 227.0, 300),
            ("full_native_finite_negative", "full_finite_negative", 232.0, 300),
            ("metadata_tests_1", "ancillary", 1.0, 10),
        )
    ]
    for attempt in attempts[1:]:
        attempt["test_lock_sha256"] = sources[
            recording.TEST_LOCKS[int(attempt["name"] == "metadata_tests_1")]
        ]
    evidence = {
        "schema": "sp_lense.certified_descent_comply_model_free_results.v1",
        "test_lock_sha256": dict(sources),
        "method_coverage": {
            key: {"passed": True, "evidence": names[:]} for key, names in coverage.items()
        },
        "saved_input_gate": {
            "admitted": True,
            "solver_calls": 1,
            "normal_exit": True,
            "external_seconds": 0.75,
            "candidate_exported": False,
            "saved_step_reuse_forbidden": True,
            "new_model_result": False,
            **recording.SAVED_GATE_SHA256,
        },
        "execution": {
            "limits": {"total": 720, "initial_saved": 10, "ancillary": 110, "full_case": 300},
            "all_attempts_including_failures_recorded": True,
            "ancillary_overhead_charge_seconds": 1.0,
            "attempts": attempts,
            "aggregate_charged_seconds": 462.75,
        },
    }
    certificate = {"source_sha256": sources}
    write_results(root, certificate, evidence)
    return certificate, evidence


def test_complete_named_evidence_and_budget_ledger_admitted(tmp_path):
    certificate, _ = fixture(tmp_path)
    recording._require_preparation_evidence(certificate, tmp_path)


def test_public_certificate_requires_both_inherited_and_method_evidence(tmp_path, monkeypatch):
    certificate, _ = fixture(tmp_path)
    calls = []

    def inherited(root):
        assert root == tmp_path
        calls.append("inherited")
        return certificate

    actual = recording._require_preparation_evidence

    def method(value, root):
        assert value is certificate
        calls.append("method")
        return actual(value, root)

    monkeypatch.setattr(recording.frozen, "require_certificate", inherited)
    monkeypatch.setattr(recording, "_require_preparation_evidence", method)
    assert recording.require_certificate(tmp_path) is certificate
    assert calls == ["inherited", "method"]
    certificate["model_free_results_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        recording.require_certificate(tmp_path)


def test_missing_inherited_certificate_cannot_be_replaced_by_method_receipt(tmp_path, monkeypatch):
    fixture(tmp_path)

    def missing(root):
        raise FileNotFoundError("test-only missing preparation certificate")

    def forbidden(*args):
        raise AssertionError("method gate reached after missing certificate")

    monkeypatch.setattr(recording.frozen, "require_certificate", missing)
    monkeypatch.setattr(recording, "_require_preparation_evidence", forbidden)
    with pytest.raises(FileNotFoundError):
        recording.require_certificate(tmp_path)


@pytest.mark.parametrize("fault", ["missing", "false", "empty_names"])
@pytest.mark.parametrize("key", sorted(recording.METHOD_COVERAGE_KEYS))
def test_missing_or_false_method_proof_rejected(tmp_path, fault, key):
    certificate, evidence = fixture(tmp_path)
    if fault == "missing":
        del evidence["method_coverage"][key]
    elif fault == "false":
        evidence["method_coverage"][key]["passed"] = False
    else:
        evidence["method_coverage"][key]["evidence"] = []
    write_results(tmp_path, certificate, evidence)
    with pytest.raises(ValueError):
        recording._require_preparation_evidence(certificate, tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("admitted", False),
        ("solver_calls", 2),
        ("solver_calls", True),
        ("normal_exit", False),
        ("external_seconds", 10.01),
        ("external_seconds", float("nan")),
        ("external_seconds", float("inf")),
        ("candidate_exported", True),
        ("saved_step_reuse_forbidden", False),
        ("new_model_result", True),
        ("lock_sha256", "bad"),
        ("result_sha256", "bad"),
    ],
)
def test_gate_count_timing_export_or_identity_rejected(tmp_path, field, value):
    certificate, evidence = fixture(tmp_path)
    evidence["saved_input_gate"][field] = value
    write_results(tmp_path, certificate, evidence)
    with pytest.raises(ValueError):
        recording._require_preparation_evidence(certificate, tmp_path)


@pytest.mark.parametrize(
    "fault",
    [
        "certificate_digest",
        "source_digest",
        "result_bytes",
        "lock_map",
        "lock_bytes",
        "missing_results_certificate",
    ],
)
def test_exact_results_and_lock_hashes_required(tmp_path, fault):
    certificate, evidence = fixture(tmp_path)
    if fault == "certificate_digest":
        certificate["model_free_results_sha256"] = "0" * 64
    elif fault == "source_digest":
        certificate["source_sha256"][recording.RESULTS] = "0" * 64
    elif fault == "result_bytes":
        with (tmp_path / recording.RESULTS).open("ab") as stream:
            stream.write(b" ")
    elif fault == "lock_map":
        evidence["test_lock_sha256"][recording.TEST_LOCKS[0]] = "0" * 64
        write_results(tmp_path, certificate, evidence)
    elif fault == "lock_bytes":
        with (tmp_path / recording.TEST_LOCKS[0]).open("ab") as stream:
            stream.write(b" ")
    else:
        del certificate["model_free_results_sha256"]
    with pytest.raises((ValueError, KeyError)):
        recording._require_preparation_evidence(certificate, tmp_path)


@pytest.mark.parametrize(
    "fault",
    [
        "omitted_initial",
        "omitted_full",
        "omitted_ancillary",
        "duplicate_full",
        "full_over_cap",
        "ancillary_over_cap",
        "total_over_cap",
        "negative_time",
        "nonfinite_time",
        "failed_full",
        "timed_out_initial",
        "incorrect_limit",
        "hard_cap",
        "gate_time_disagreement",
        "aggregate_disagreement",
        "failure_history_not_recorded",
    ],
)
def test_omitted_failed_or_over_budget_ledger_rejected(tmp_path, fault):
    certificate, evidence = fixture(tmp_path)
    execution = evidence["execution"]
    attempts = execution["attempts"]
    if fault.startswith("omitted_"):
        index = {"omitted_initial": 0, "omitted_ancillary": 1, "omitted_full": 3}[fault]
        attempts.pop(index)
    elif fault == "duplicate_full":
        attempts.append(copy.deepcopy(attempts[3]))
    elif fault == "full_over_cap":
        attempts[3]["external_seconds"] = 300.01
    elif fault == "ancillary_over_cap":
        execution["ancillary_overhead_charge_seconds"] = 111.0
    elif fault == "total_over_cap":
        execution["aggregate_charged_seconds"] = 721.0
    elif fault == "negative_time":
        attempts[1]["external_seconds"] = -1
    elif fault == "nonfinite_time":
        attempts[1]["external_seconds"] = float("nan")
    elif fault == "failed_full":
        attempts[3]["returncode"] = 1
    elif fault == "timed_out_initial":
        attempts[0]["timed_out"] = True
    elif fault == "incorrect_limit":
        execution["limits"]["total"] = 721
    elif fault == "hard_cap":
        attempts[3]["hard_timeout_seconds"] = 301
    elif fault == "gate_time_disagreement":
        attempts[0]["external_seconds"] = 0.5
    elif fault == "aggregate_disagreement":
        execution["aggregate_charged_seconds"] += 0.01
    else:
        execution["all_attempts_including_failures_recorded"] = False
    if fault not in ("aggregate_disagreement", "total_over_cap"):
        execution["aggregate_charged_seconds"] = (
            sum(item["external_seconds"] for item in attempts)
            + execution["ancillary_overhead_charge_seconds"]
        )
    write_results(tmp_path, certificate, evidence)
    with pytest.raises(ValueError):
        recording._require_preparation_evidence(certificate, tmp_path)


@pytest.mark.parametrize("field", ["lock_sha256", "result_sha256"])
def test_valid_length_but_wrong_saved_gate_hash_rejected(tmp_path, field):
    certificate, evidence = fixture(tmp_path)
    evidence["saved_input_gate"][field] = "0" * 64
    write_results(tmp_path, certificate, evidence)
    with pytest.raises(ValueError):
        recording._require_preparation_evidence(certificate, tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "wrong"),
        ("original_test_lock_sha256", "0" * 64),
        ("metadata_only", False),
        ("full_cases_repeated", True),
        ("method_coverage_evidence", {}),
    ],
)
def test_wrong_chronology_or_locked_method_names_rejected(tmp_path, field, value):
    certificate, evidence = fixture(tmp_path)
    path = recording.TEST_LOCKS[1]
    lock = json.loads((tmp_path / path).read_bytes())
    lock[field] = value
    raw = json.dumps(lock).encode()
    (tmp_path / path).write_bytes(raw)
    certificate["source_sha256"][path] = digest(raw)
    evidence["test_lock_sha256"][path] = digest(raw)
    evidence["execution"]["attempts"][4]["test_lock_sha256"] = digest(raw)
    write_results(tmp_path, certificate, evidence)
    with pytest.raises(ValueError):
        recording._require_preparation_evidence(certificate, tmp_path)


@pytest.mark.parametrize("index", [1, 2, 3, 4])
def test_each_required_batch_needs_its_correct_prospective_lock(tmp_path, index):
    certificate, evidence = fixture(tmp_path)
    item = evidence["execution"]["attempts"][index]
    other = recording.TEST_LOCKS[0 if index == 4 else 1]
    item["test_lock_sha256"] = certificate["source_sha256"][other]
    write_results(tmp_path, certificate, evidence)
    with pytest.raises(ValueError):
        recording._require_preparation_evidence(certificate, tmp_path)


def test_results_cannot_substitute_unlocked_coverage_names(tmp_path):
    certificate, evidence = fixture(tmp_path)
    key = min(recording.METHOD_COVERAGE_KEYS)
    evidence["method_coverage"][key]["evidence"] = ["different::named_test"]
    write_results(tmp_path, certificate, evidence)
    with pytest.raises(ValueError):
        recording._require_preparation_evidence(certificate, tmp_path)


def test_no_step_report_does_not_claim_a_passing_endpoint():
    text = checker.render_report(
        {
            "status": checker.AUDIT_MATCH,
            "summary": {
                "stop_reason": "no_certified_step",
                "final_accepted": 0,
                "forward_count": 36,
                "derivative_count": 12,
                "updates": 0,
                "attempted_updates": 1,
                "shared_net": 0,
                "shared_path": 0,
            },
        }
    )
    assert "any issued endpoint" in text
    assert "no_certified_step" in text
    assert "stationarity without exact proof" in text
