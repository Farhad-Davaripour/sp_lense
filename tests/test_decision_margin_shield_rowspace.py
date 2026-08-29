from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sp_lense.decision_margin_shield_rowspace as rowspace_module
from sp_lense.counterfactual_tangent_shield import (
    TangentShieldDirection,
    TangentShieldInfeasibleError,
    TangentShieldSolverError,
)
from sp_lense.decision_margin_shield import (
    DecisionMarginOptimalityError,
    certify_minimum_l2_candidate,
    decision_margin_bounds,
)
from sp_lense.decision_margin_shield_rowspace import (
    SCHEMA_VERSION,
    solve_certified_rowspace_minimum_l2_direction,
)
from sp_lense.factorial_causal_anchor import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_ROOT = (
    ROOT / "artifacts" / "decision_margin_shield_layer_screen" / "qwen35_08b"
)


def test_known_analytic_minimum_and_orthonormal_representer_coordinates() -> None:
    result = solve_certified_rowspace_minimum_l2_direction(
        np.eye(2),
        np.zeros(2),
        margin=np.array([1.0, 2.0]),
        l2_cap=None,
    )

    assert isinstance(result, TangentShieldDirection)
    np.testing.assert_allclose(result.direction, [1.0, 2.0], atol=1e-12, rtol=0.0)
    assert result.direction.flags.writeable is False
    assert result.diagnostics["schema_version"] == SCHEMA_VERSION
    assert result.diagnostics["reduced_dimension"] == 2
    assert result.diagnostics["rank_at_most_inequality_count"] is True
    certificate = result.diagnostics["optimality_certificate"]
    assert certificate["passes"] is True
    assert certificate["candidate_l2_norm"] == pytest.approx(np.sqrt(5.0), abs=1e-12)
    assert result.diagnostics["passes_strict_certificate"] is True


def test_non_none_l2_cap_is_rejected_because_the_amendment_is_uncapped() -> None:
    with pytest.raises(ValueError, match="uncapped; l2_cap must be None"):
        solve_certified_rowspace_minimum_l2_direction(
            np.array([[1.0]]),
            np.array([0.0]),
            margin=1.0,
            l2_cap=2.0,
        )


def test_exact_soft_zero_and_redundant_constraints_are_handled() -> None:
    result = solve_certified_rowspace_minimum_l2_direction(
        np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]]),
        np.zeros(2),
        margin=np.array([1.0, 2.0]),
        nuisance_rows=np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        nuisance_bound=np.array([0.0, 0.0, 0.0, 0.1]),
    )

    # The unconstrained minimum is [0.5, 0.5, 0].  The active two-sided soft
    # slab |y| <= 0.1 moves the certified optimum to [0.9, 0.1, 0].
    np.testing.assert_allclose(result.direction, [0.9, 0.1, 0.0], atol=1e-10, rtol=0.0)
    equality = result.diagnostics["equality_projection"]
    assert equality["input_row_count"] == 3
    assert equality["zero_row_count"] == 1
    assert equality["rank"] == 1
    assert result.diagnostics["representer_rowspace"]["rank"] == 2
    certificate = result.diagnostics["optimality_certificate"]
    assert certificate["passes"] is True
    assert certificate["exact_equality_row_count"] == 3
    assert certificate["soft_slab_row_count"] == 1


def test_target_frozen_by_exact_nuisance_is_infeasible() -> None:
    with pytest.raises(TangentShieldInfeasibleError, match="vanishes|infeasible"):
        solve_certified_rowspace_minimum_l2_direction(
            np.array([[1.0, 0.0]]),
            np.array([0.0]),
            margin=1.0,
            nuisance_rows=np.array([[1.0, 0.0]]),
            nuisance_bound=0.0,
        )


def test_output_and_all_reported_hashes_repeat_deterministically() -> None:
    kwargs = {
        "target_rows": np.array([[1.0, 0.2, 0.0], [0.5, 1.0, 0.0]]),
        "target_offsets": np.array([0.1, -0.2]),
        "margin": 0.05,
        "nuisance_rows": np.array([[0.0, 0.0, 1.0], [0.2, -0.1, 0.0]]),
        "nuisance_bound": np.array([0.0, 0.4]),
    }
    first = solve_certified_rowspace_minimum_l2_direction(**kwargs)
    second = solve_certified_rowspace_minimum_l2_direction(**kwargs)

    np.testing.assert_array_equal(first.direction, second.direction)
    assert first.diagnostics["input_sha256"] == second.diagnostics["input_sha256"]
    assert first.diagnostics["direction_sha256"] == second.diagnostics["direction_sha256"]
    assert (
        first.diagnostics["inequality_scaling"]["scaling_sha256"]
        == second.diagnostics["inequality_scaling"]["scaling_sha256"]
    )
    assert (
        first.diagnostics["representer_rowspace"]["diagnostics_sha256"]
        == second.diagnostics["representer_rowspace"]["diagnostics_sha256"]
    )
    assert first.diagnostics["diagnostics_sha256"] == second.diagnostics["diagnostics_sha256"]
    assert first.diagnostics["determinism_scope"] == "deterministic_within_pinned_runtime"
    assert "deterministic_output" not in first.diagnostics


def test_tiny_singular_direction_is_retained_then_indeterminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = np.array([[5e6, 5e-5], [-5e6, 5e-5]], dtype=np.float64)
    captured: dict[str, np.ndarray] = {}

    def capture_candidate(direction, *args, **kwargs):
        captured["direction"] = np.asarray(direction, dtype=np.float64).copy()
        return certify_minimum_l2_candidate(direction, *args, **kwargs)

    monkeypatch.setattr(
        "sp_lense.decision_margin_shield_rowspace.certify_minimum_l2_candidate",
        capture_candidate,
    )
    with pytest.raises(DecisionMarginOptimalityError, match="primal_dual_gap"):
        solve_certified_rowspace_minimum_l2_direction(
            target,
            np.zeros(2),
            margin=0.05,
        )

    # The old scientific-rank representer returned approximately zero and was
    # falsely certified.  Machine-rank retention instead reaches the true
    # scientific direction before the unchanged conservative dual certificate
    # declares this extreme conditioning numerically indeterminate.
    np.testing.assert_allclose(captured["direction"], [0.0, 1000.0], atol=2e-6, rtol=0.0)
    raw_slacks = target @ captured["direction"] - 0.05
    assert float(np.min(raw_slacks)) >= -5e-9


def test_strict_raw_certificate_rejects_the_old_near_zero_false_pass_before_dual_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = np.array([[5e6, 5e-5], [-5e6, 5e-5]], dtype=np.float64)
    monkeypatch.setattr(
        "sp_lense.decision_margin_shield_rowspace.minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            status=8,
            nit=1,
            message="mock old truncated candidate",
            x=np.zeros(2, dtype=np.float64),
            fun=0.0,
        ),
    )

    def independent_certificate_must_not_run(*args, **kwargs):
        raise AssertionError("strict raw-unit validation must run first")

    monkeypatch.setattr(
        "sp_lense.decision_margin_shield_rowspace.certify_minimum_l2_candidate",
        independent_certificate_must_not_run,
    )
    with pytest.raises(
        DecisionMarginOptimalityError,
        match="strict raw-coordinate certification: target_lower_bounds_in_raw_units",
    ):
        solve_certified_rowspace_minimum_l2_direction(
            target,
            np.zeros(2),
            margin=0.05,
        )


def test_representer_diagnostic_failure_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_basis = rowspace_module._normalized_row_basis

    def corrupt_representer_diagnostics(*args, **kwargs):
        basis, diagnostics = real_basis(*args, **kwargs)
        if kwargs["rank_rule"] == "machine_precision_representer_span":
            diagnostics = dict(diagnostics)
            diagnostics["checks"] = dict(diagnostics["checks"])
            diagnostics["checks"]["rowspace_reconstruction"] = False
            diagnostics["passes"] = False
        return basis, diagnostics

    monkeypatch.setattr(
        rowspace_module,
        "_normalized_row_basis",
        corrupt_representer_diagnostics,
    )
    with pytest.raises(TangentShieldSolverError, match="representer row-space diagnostics"):
        solve_certified_rowspace_minimum_l2_direction(
            np.array([[1.0]]),
            np.array([0.0]),
            margin=1.0,
        )


def test_highs_status_2_is_indeterminate_without_infeasibility_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sp_lense.decision_margin_shield_rowspace.linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            status=2,
            nit=0,
            message="mock reported infeasible",
        ),
    )
    with pytest.raises(TangentShieldSolverError, match="without an independent") as caught:
        solve_certified_rowspace_minimum_l2_direction(
            np.array([[1.0]]),
            np.array([0.0]),
            margin=1.0,
        )
    assert not isinstance(caught.value, TangentShieldInfeasibleError)


def test_finite_status_8_candidate_is_accepted_only_with_independent_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sp_lense.decision_margin_shield_rowspace.minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            status=8,
            nit=3,
            message="Positive directional derivative for linesearch",
            x=np.array([1.0]),
            fun=0.5,
        ),
    )
    result = solve_certified_rowspace_minimum_l2_direction(
        np.array([[1.0]]),
        np.array([0.0]),
        margin=1.0,
    )
    assert result.diagnostics["optimizer_status"] == 8
    assert result.diagnostics["optimizer_success"] is False
    assert result.diagnostics["status_8_requires_and_received_independent_certificate"] is True
    assert result.diagnostics["optimality_certificate"]["passes"] is True


def test_mocked_feasible_but_nonoptimal_success_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sp_lense.decision_margin_shield_rowspace.minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=True,
            status=0,
            nit=1,
            message="mock success",
            x=np.array([2.0]),
            fun=2.0,
        ),
    )
    with pytest.raises(DecisionMarginOptimalityError, match="optimality certification"):
        solve_certified_rowspace_minimum_l2_direction(
            np.array([[1.0]]),
            np.array([0.0]),
            margin=1.0,
        )


def test_unapproved_failed_optimizer_status_is_rejected_even_for_optimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sp_lense.decision_margin_shield_rowspace.minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            status=9,
            nit=2_000,
            message="Iteration limit reached",
            x=np.array([1.0]),
            fun=0.5,
        ),
    )
    with pytest.raises(TangentShieldSolverError, match="allowed status 8"):
        solve_certified_rowspace_minimum_l2_direction(
            np.array([[1.0]]),
            np.array([0.0]),
            margin=1.0,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_immutable_weather_layer_22_cell() -> dict[str, np.ndarray]:
    torch = pytest.importorskip("torch")
    manifest_path = CAPTURE_ROOT / "capture_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("the local immutable DMS capture is not present")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unhashed = dict(manifest)
    observed_manifest_hash = unhashed.pop("manifest_sha256")
    assert canonical_sha256(unhashed) == observed_manifest_hash

    chunk_records = {int(record["index"]): record for record in manifest["chunks"]}
    payloads = []
    for index in (0, 1, 16):
        record = chunk_records[index]
        path = ROOT / record["path"]
        assert _sha256_file(path) == record["file_sha256"]
        payloads.append(torch.load(path, map_location="cpu", weights_only=True))

    weather = []
    for payload in payloads[:2]:
        for row_index, record in enumerate(payload["records"]):
            weather.append(
                (
                    record,
                    payload["tensors"]["gradients"][row_index].double(),
                    payload["tensors"]["anchor_residuals"][row_index].double(),
                )
            )
    residual_norms = torch.stack([residual.norm(dim=1) for _, _, residual in weather])
    residual_scale = float(torch.exp(torch.log(residual_norms).mean(dim=0))[22].item())
    targets = [
        row
        for row in weather
        if row[0]["target"] == "self" and row[0]["event"] == "permanent"
    ]
    protected = [
        row
        for row in weather
        if not (row[0]["target"] == "self" and row[0]["event"] == "permanent")
    ]
    nuisance_payload = payloads[2]
    return {
        "target_rows": residual_scale
        * torch.stack([gradient[22] for _, gradient, _ in targets]).numpy(),
        "target_offsets": np.array(
            [row["preserve_minus_comply_baseline_log_odds"] for row, _, _ in targets],
            dtype=np.float64,
        ),
        "protected_rows": residual_scale
        * torch.stack([gradient[22] for _, gradient, _ in protected]).numpy(),
        "protected_offsets": np.array(
            [row["preserve_minus_comply_baseline_log_odds"] for row, _, _ in protected],
            dtype=np.float64,
        ),
        "unrelated_rows": residual_scale
        * nuisance_payload["tensors"]["gradients"][:, 22].double().numpy(),
    }


def test_immutable_weather_layer_22_cells_pass_independent_certificates() -> None:
    cell = _load_immutable_weather_layer_22_cell()
    definitions = {
        "unrelated_only": (
            cell["unrelated_rows"],
            np.zeros(8, dtype=np.float64),
            217.50082075794603,
            4,
        ),
        "decision_margin_shield": (
            np.vstack((cell["unrelated_rows"], cell["protected_rows"])),
            np.concatenate(
                (
                    np.zeros(8, dtype=np.float64),
                    decision_margin_bounds(cell["protected_offsets"]),
                )
            ),
            318.2395070552367,
            14,
        ),
    }
    for nuisance_rows, nuisance_bounds, expected_norm, expected_rank in definitions.values():
        result = solve_certified_rowspace_minimum_l2_direction(
            cell["target_rows"],
            cell["target_offsets"],
            margin=0.05,
            nuisance_rows=nuisance_rows,
            nuisance_bound=nuisance_bounds,
        )
        certificate = result.diagnostics["optimality_certificate"]
        assert certificate["passes"] is True
        assert result.diagnostics["strict_raw_coordinate_certificate"]["passes"] is True
        assert result.diagnostics["passes_strict_certificate"] is True
        assert np.linalg.norm(result.direction) == pytest.approx(expected_norm, rel=2e-10)
        assert result.diagnostics["reduced_dimension"] == expected_rank
        assert result.diagnostics["optimizer_success"] is True or (
            result.diagnostics["optimizer_status"] == 8
            and result.diagnostics[
                "status_8_requires_and_received_independent_certificate"
            ]
            is True
        )
