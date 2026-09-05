"""Only time/identity bindings; no models, historical output audits or optimizer reruns."""

from __future__ import annotations

import copy

import pytest

from scripts import paired_common_drift_comply_1800_plan as protocol


def test_only_six_config_fields_change():
    config, old = protocol.config_at(), protocol.parent.config_at()
    protocol.validate_config(config, old)
    assert {key for key in old if old[key] != config[key]} == protocol.ALLOWED_CONFIG_CHANGES
    assert config["optimizer"]["initialization"] == "fresh zero native float64 vector"
    assert config["provenance"]["resume_allowed"] is False
    assert config["provenance"]["prior_inconclusive_result_changed"] is False
    assert protocol.AUTH_SCOPE == (
        "one fresh time-only paired common-drift COMPLY construction;216F/96D/1800s;no retry"
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("aim", 0.2),
        ("acceptance", 0.04),
        ("step_cap", 0.06),
        ("total_cap", 0.3),
        ("path_cap", 0.5),
        ("maximum_updates", 9),
        ("maximum_forwards", 217),
        ("maximum_derivatives", 97),
        ("cast_sequence", "different casts"),
        ("rendered_prompts", []),
        ("training_families", []),
        ("candidate_freeze", "exact radius"),
        ("timeout_seconds", 2400),
    ],
)
def test_science_or_unapproved_time_change_rejected(key, value):
    config = copy.deepcopy(protocol.config_at())
    config[key] = value
    with pytest.raises(ValueError):
        protocol.validate_config(config, protocol.parent.config_at())


def test_nested_science_and_warm_start_changes_rejected():
    config, old = protocol.config_at(), protocol.parent.config_at()
    for location, field, value in (
        ("objective", "lambda", 2),
        ("optimizer", "B0", 8),
        ("storage", "total_bound_bytes", 1),
        ("provenance", "resume_allowed", True),
    ):
        mutated = copy.deepcopy(config)
        mutated[location][field] = value
        with pytest.raises(ValueError):
            protocol.validate_config(mutated, old)


def test_config_raw_identity_guard(tmp_path):
    path = tmp_path / protocol.CONFIG
    path.parent.mkdir(parents=True)
    path.write_bytes((protocol.ROOT / protocol.CONFIG).read_bytes() + b" ")
    with pytest.raises(ValueError, match="exact prospective"):
        protocol.config_at(tmp_path)


def test_plan_uses_exact_parent_science_and_fresh_namespace():
    value = protocol.build_plan()
    parent = protocol.parent.build_plan()
    changed = {"schema", "output_namespace", "config", "input_sha256", "recording_policy"}
    assert set(value) == set(parent)
    assert all(value[key] == parent[key] for key in parent if key not in changed)
    assert len(value["cells"]) == 216
    assert len(value["derivative_cells"]) == 96
    assert len(value["pairs"]) == 6
    assert value["config"]["timeout_seconds"] == 1800
    assert protocol.PARENT_LOCK in value["input_sha256"]
    assert value["input_sha256"][protocol.parent.PREP_JSON] == protocol.PARENT_CERTIFICATE_SHA
    assert all(
        value["input_sha256"][path] == sha for path, sha in protocol.DIAGNOSTIC_SHA256.items()
    )
    assert not (protocol.ROOT / protocol.OUTPUT).exists()


def test_new_certificate_gate_is_distinct_and_read_only(tmp_path):
    from scripts import paired_common_drift_comply_1800_recording as recording

    assert recording.CERTIFICATE == protocol.PREP_JSON != protocol.parent.PREP_JSON
    with pytest.raises((ValueError, FileNotFoundError)):
        protocol.require_preparation_certificate(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_source_closure_includes_reused_implementation_and_new_binding():
    assert set(protocol.parent.SOURCE_PATHS) <= set(protocol.SOURCE_PATHS)
    assert protocol.OPTIMIZER == protocol.parent.OPTIMIZER
    assert protocol.BINDING in protocol.SOURCE_PATHS
    assert "tests/test_paired_common_drift_comply.py" in protocol.SOURCE_PATHS
    assert set(protocol.DIAGNOSTIC_SHA256) <= set(protocol.SOURCE_PATHS)
