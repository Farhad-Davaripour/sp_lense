"""Native-width fake rows: mocked contracts plus one actual independently audited solve."""

import copy
import math
from fractions import Fraction as F

import pytest

from scripts import certified_descent_comply_optimizer as opt


def vec(x=0.0, y=0.0):
    return [float(x), float(y)] + [0.0] * 1022


def fixture(stage=1, w=None):
    w = vec() if w is None else w
    rows, baselines = [], []
    for index, pid in enumerate(opt.contracts.PROMPT_IDS):
        mapping_a = index % 4 >= 2
        hn = 2.0 if mapping_a else 3.0
        family = opt.contracts.FAMILIES[index // 4]
        row = {
            "prompt_id": pid,
            "family_id": family,
            "case_id": family + "__v1__self_shutdown",
            "variant_id": "v1",
            "category": "self_shutdown",
            "split": "discovery",
            "order": "preserve_second" if mapping_a else "preserve_first",
            "display_order": opt.contracts.DISPLAYS[index % 2],
            "comply_label": "A" if mapping_a else "B",
            "preserve_label": "B" if mapping_a else "A",
            "semantic_mapping": "preserve_B_comply_A" if mapping_a else "preserve_A_comply_B",
            "rendering_index": index + 1,
            "preserve_log_odds": 0.0,
            "h0": vec(hn),
            "h0_norm": hn,
            "shared_w": vec(),
            "shared_w_sha256": opt.vector_sha(vec()),
            "condition": "baseline",
            "cell_id": pid + "__baseline",
            "baseline_cell_id": pid + "__baseline",
            "baseline_margin": 0.0,
        }
        baselines.append({"row": copy.deepcopy(row)})
        row.update(
            shared_w=list(w),
            shared_w_sha256=opt.vector_sha(w),
            condition=f"gradient_{stage}",
            stage=stage,
            cell_id=pid + f"__gradient_{stage}",
            current_cell_id=pid + ("__baseline" if stage == 1 else f"__step_{stage - 1}"),
            maximum_current_logit_difference=0.0,
            maximum_current_h_difference=0.0,
            gradient=vec(-0.5, 0.0) if mapping_a else vec(0.0, 1 / 3),
        )
        rows.append({"row": row})
    return rows, baselines


def install_fake(monkeypatch, status="ADMITTED_DESCENT", reason=None, mutate=None):
    calls = []

    def fake(problem, fingerprint, *, observer):
        calls.append(copy.deepcopy(problem))
        observer({"test_only": True})
        w = problem["w"]
        point = vec(1 / 64)
        endpoint = [x + y for x, y in zip(w, point, strict=True)]
        actual = [F(x) - F(y) for x, y in zip(endpoint, w, strict=True)]
        square = sum(x * x for x in actual)
        path_after = F(problem["path_upper"]) + F(1, 64)
        certificate = {
            "status": "ADMITTED_DESCENT",
            "problem_sha256": fingerprint,
            "current_w_sha256": problem["w_sha256"],
            "w_next_sha256": opt.vector_sha(endpoint),
            "geometry_valid": True,
            "actual_displacement": [str(x) for x in actual],
            "geometry": {
                "path_before_upper": problem["path_upper"],
                "path_after_upper": str(path_after),
                "actual_step_squared": str(square),
                "net_squared": str(sum(F(x) ** 2 for x in endpoint)),
            },
        }
        zero = {
            "status": "EXACT_ZERO_OPTIMUM",
            "problem_sha256": fingerprint,
            "current_w_sha256": problem["w_sha256"],
        }
        result = {
            "status": status,
            "terminal_reason": reason
            or {
                "ADMITTED_DESCENT": "FIRST_CERTIFIED_DYADIC_TRIAL",
                "EXACT_ZERO_OPTIMUM": "exact_zero_gradient",
                "NO_CERTIFIED_STEP": "serialized_zero_increment",
            }[status],
            "problem_sha256": fingerprint,
            "serialization_checked": True,
            "gradient_count": int(status != "EXACT_ZERO_OPTIMUM"),
            "proposal_count": int(status != "EXACT_ZERO_OPTIMUM"),
            "trial_count": int(status == "ADMITTED_DESCENT"),
            "trials": (
                [{"j": 0, "lambda": 1.0, "check_status": "COMPLETED", "certificate": certificate}]
                if status == "ADMITTED_DESCENT"
                else []
            ),
            "chosen_j": 0 if status == "ADMITTED_DESCENT" else None,
            "selected_certificate": certificate if status == "ADMITTED_DESCENT" else None,
            "zero_certificate": zero,
            "w_next": endpoint if status == "ADMITTED_DESCENT" else None,
            "descriptive_gap": {
                "w_next_sha256": opt.vector_sha(endpoint),
                "descriptive_gap_upper": "1/3",
            },
            "proposal": {
                "y": point,
                "p": point,
                "projection": {"step_normal": vec(), "net_normal": vec(), "case": "interior"},
            },
        }
        if mutate:
            mutate(result)
        return result

    monkeypatch.setattr(opt.solver, "solve", fake)
    return calls


def run(rows, baselines, *, stage=1, w=None, history=None, path_upper="0", path=None):
    w = vec() if w is None else w
    history = [vec()] if history is None else history
    path = float(F(path_upper)) if path is None else path
    return opt.increment(rows, w, path, stage, baselines, history=history, path_upper=path_upper)


def test_own_norm_original_casts_and_compact_journal(monkeypatch):
    rows, base = fixture()
    rows[0]["row"]["preserve_log_odds"] = -0.2
    calls = install_fake(monkeypatch)
    result = run(rows, base)
    assert result["status"] == "ready" and not result["adapter_fault"]
    assert result["technical_failure"] is False
    assert len(calls) == 1
    problem = calls[0]
    assert problem["A"][2] == vec(1.0)
    assert problem["A"][0] == vec(0.0, -1.0)
    assert problem["b"][0] == 0.1 - 0.2 < 0
    assert problem["c"][0] == -0.1
    assert result["problem_sha256"] == opt.canonical_sha(problem)
    expected_D = [str((F(a) - F(b)) / 2) for a, b in zip(problem["A"][2], problem["A"][0])]
    assert result["inputs"]["D_exact_row_sha256"][0] == opt.canonical_sha(expected_D)
    certificate = result["solver"]["trials"][0]["certificate"]
    assert "actual_displacement" not in certificate
    assert certificate["actual_displacement_sha256"] == opt.canonical_sha(["1/64"] + ["0"] * 1023)
    assert "w_next" not in result["solver"] and "selected_certificate" not in result["solver"]
    assert result["solver"]["selected_trial_j"] == 0
    assert result["path_after_upper"] == "1/64"
    assert result["path_after"] == result["step_norm"] == 1 / 64
    assert len(opt.encoded(result)) < opt.MAX_UPDATE_BYTES


@pytest.mark.parametrize("fault", ["own_norm", "own_h0", "cache", "stage", "mapping", "signed_w"])
def test_bad_fresh_row_authentication_stops_before_solver(monkeypatch, fault):
    rows, base = fixture()
    row = rows[2]["row"]
    if fault == "own_norm":
        row["h0_norm"] = 3.0
    elif fault == "own_h0":
        row["h0"] = vec(3.0)
    elif fault == "cache":
        row["current_cell_id"] = row["prompt_id"] + "__step_7"
    elif fault == "stage":
        row["stage"] = 2
    elif fault == "mapping":
        row["comply_label"] = "B"
    else:
        row["shared_w"][1] = -0.0
        row["shared_w_sha256"] = opt.vector_sha(row["shared_w"])
    calls = install_fake(monkeypatch)
    result = run(rows, base)
    assert result["status"] == "no_certified_step" and result["adapter_fault"]
    assert not calls and "w_after" not in result


@pytest.mark.parametrize(
    "fault", ["path_float", "history_count", "history_endpoint", "history_zero"]
)
def test_path_and_history_contracts(monkeypatch, fault):
    w = vec(1 / 64)
    rows, base = fixture(2, w)
    arguments = {"stage": 2, "w": w, "history": [vec(), w], "path_upper": "1/64"}
    if fault == "path_float":
        arguments["path"] = 0.0
    elif fault == "history_count":
        arguments["history"] = [w]
    elif fault == "history_endpoint":
        arguments["history"] = [vec(), vec(1 / 32)]
    else:
        arguments["history"] = [vec(1 / 128), w]
    calls = install_fake(monkeypatch)
    result = run(rows, base, **arguments)
    assert result["adapter_fault"] and not calls and "w_after" not in result


def test_path_advance_uses_exact_certificate_bound(monkeypatch):
    w = vec(1 / 64)
    rows, base = fixture(2, w)
    install_fake(monkeypatch)
    result = run(rows, base, stage=2, w=w, history=[vec(), w], path_upper="1/64")
    assert result["status"] == "ready" and result["path_after_upper"] == "1/32"
    assert result["path_after"] == float(F(result["path_after_upper"]))
    assert result["history_w_sha256"] == [opt.vector_sha(vec()), opt.vector_sha(w)]


@pytest.mark.parametrize(
    "status,expected",
    [("EXACT_ZERO_OPTIMUM", "exact_zero_optimum"), ("NO_CERTIFIED_STEP", "no_certified_step")],
)
def test_no_step_and_exact_zero_are_distinct(monkeypatch, status, expected):
    rows, base = fixture()
    install_fake(monkeypatch, status)
    result = run(rows, base)
    assert result["status"] == expected and not result["adapter_fault"]
    assert result["technical_failure"] is False
    assert result["exact_zero_optimum"] == (status == "EXACT_ZERO_OPTIMUM")
    assert not result["step_admitted"] and "w_after" not in result


def test_solver_fault_is_not_ordinary_no_step(monkeypatch):
    rows, base = fixture()
    install_fake(monkeypatch, "NO_CERTIFIED_STEP", reason="ValueError:projection failed")
    result = run(rows, base)
    assert result["status"] == "no_certified_step" and result["adapter_fault"]
    assert result["technical_failure"] is True


def test_forged_selected_endpoint_hash_is_technical_failure(monkeypatch):
    rows, base = fixture()
    install_fake(
        monkeypatch,
        mutate=lambda result: result["selected_certificate"].update(w_next_sha256="0" * 64),
    )
    result = run(rows, base)
    assert result["adapter_fault"] and "w_after" not in result
    assert result["solver"]["trial_count"] == 1


@pytest.mark.parametrize("phase", ["assembly", "solver", "serialization"])
def test_outer_deadline_never_issues_endpoint(monkeypatch, phase):
    clock = [0.0]
    monkeypatch.setattr(opt.time, "monotonic", lambda: clock[0])
    rows, base = fixture()
    calls = install_fake(monkeypatch)
    if phase == "assembly":
        original = opt.canonical_sha

        def late(value):
            result = original(value)
            clock[0] = 11.0
            return result

        monkeypatch.setattr(opt, "canonical_sha", late)
    elif phase == "solver":
        original = opt.solver.solve

        def late(*args, **kwargs):
            result = original(*args, **kwargs)
            clock[0] = 11.0
            return result

        monkeypatch.setattr(opt.solver, "solve", late)
    else:
        original = opt.encoded

        def late(value):
            result = original(value)
            if isinstance(value, dict) and value.get("status") == "ready":
                clock[0] = 11.0
            return result

        monkeypatch.setattr(opt, "encoded", late)
    result = run(rows, base)
    assert result["status"] == "no_certified_step" and result["adapter_fault"]
    assert "w_after" not in result and not result["step_admitted"]
    assert result["terminal_reason"] == "ADAPTER_COMBINED_TEN_SECOND_DEADLINE"
    assert result["technical_failure"] is True
    assert len(calls) == (0 if phase == "assembly" else 1)


def test_total_encoder_failure_returns_honest_receipt(monkeypatch):
    rows, base = fixture()
    calls = install_fake(monkeypatch)

    def fail(_value):
        raise ValueError("test-only encoding fault")

    monkeypatch.setattr(opt, "encoded", fail)
    result = run(rows, base)
    assert result["failure_receipt_only"] and result["adapter_fault"]
    assert result["serialization_checked"] is False and "w_after" not in result
    assert not calls


def test_descriptive_response_objective_not_an_acceptance_gate():
    rows, base = fixture()
    rows[2]["row"]["preserve_log_odds"] = -0.2
    result = opt.objective(rows, base)
    assert result["comply_margins"][2] == 0.2
    assert math.isclose(result["baseline_relative_common_letter_drift"][0], 0.1)
    assert result["behavioral_acceptance_gate"] is False


@pytest.fixture(scope="module")
def independently_checked_update():
    # Exactly one actual adapter construction on these fresh artificial rows.
    # Copies below reuse its journal; no model, historical input or benchmark.
    from scripts import verify_certified_descent_comply as verifier

    rows, base = fixture()
    update = run(rows, base)
    assert update["status"] == "ready", update
    audit = verifier.verify_update(update, rows, vec(), 0.0, base, history=[vec()], path_upper="0")
    return update, rows, base, audit, verifier


def test_actual_adapter_and_independent_update_schema(independently_checked_update):
    update, _rows, _base, audit, _verifier = independently_checked_update
    assert audit["status"] == "ready"
    assert audit["first_passing_verified"] is True
    assert audit["exact_serialized_geometry_verified"] is True
    assert audit["nominal_proposal_reconstructed"] is True
    assert audit["w_after_sha256"] == update["w_after_sha256"]
    assert audit["path_after_upper"] == update["path_after_upper"]
    assert len(opt.encoded(audit)) <= 32768


@pytest.mark.parametrize("fault", ["problem", "current_path", "selected_hash", "trial_status"])
def test_independent_update_rejects_forged_journal(independently_checked_update, fault):
    original, rows, base, _audit, verifier = independently_checked_update
    update = copy.deepcopy(original)
    if fault == "problem":
        update["problem_sha256"] = "0" * 64
    elif fault == "current_path":
        update["path_before_upper"] = "1/128"
    elif fault == "selected_hash":
        update["solver"]["trials"][-1]["certificate"]["w_next_sha256"] = "0" * 64
    else:
        update["solver"]["trials"][-1]["certificate"]["status"] = "REJECTED_TRIAL"
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        verifier.verify_update(update, rows, vec(), 0.0, base, history=[vec()], path_upper="0")
