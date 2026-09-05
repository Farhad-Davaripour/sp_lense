"""Clean01 strict identity/dispatch binding; no models or old output replay."""

from __future__ import annotations

import copy

import pytest

from scripts import paired_common_drift_comply_1800_clean01_plan as protocol


def test_exact_five_field_diff_leaves_time_and_science_unchanged():
    config, old = protocol.config_at(), protocol.parent.config_at()
    protocol.validate_config(config, old)
    assert {key for key in old if old[key] != config[key]} == protocol.ALLOWED_CONFIG_CHANGES
    assert "timeout_seconds" not in protocol.ALLOWED_CONFIG_CHANGES
    assert config["timeout_seconds"] == old["timeout_seconds"] == 1800
    assert config["provenance"]["prior_certificate_is_repaired_cli_proof"] is False
    assert config["provenance"]["prior_cli_coverage_limitation"] == protocol.CLI_COVERAGE_LIMITATION
    assert config["optimizer"]["initialization"] == "fresh zero native float64 vector"


@pytest.mark.parametrize(
    "key,value",
    [
        ("timeout_seconds", 2400),
        ("maximum_forwards", 217),
        ("maximum_derivatives", 97),
        ("maximum_updates", 9),
        ("step_cap", 0.06),
        ("total_cap", 0.3),
        ("path_cap", 0.5),
        ("aim", 0.2),
        ("acceptance", 0.04),
        ("cast_sequence", "changed"),
        ("rendered_prompts", []),
        ("candidate_freeze", "exact radius"),
    ],
)
def test_any_time_or_scientific_change_is_rejected(key, value):
    config = copy.deepcopy(protocol.config_at())
    config[key] = value
    with pytest.raises(ValueError):
        protocol.validate_config(config, protocol.parent.config_at())


@pytest.mark.parametrize(
    "field,value",
    [
        ("prior_certificate_is_repaired_cli_proof", True),
        ("prior_cli_coverage_limitation", "unqualified launch proof"),
        ("resume_allowed", True),
        ("warm_start_allowed", True),
        ("independent_confirmation", True),
        ("only_dispatch_identity_changes", False),
    ],
)
def test_certificate_overclaim_or_nonfresh_provenance_rejected(field, value):
    config = copy.deepcopy(protocol.config_at())
    config["provenance"][field] = value
    with pytest.raises(ValueError):
        protocol.validate_config(config, protocol.parent.config_at())


def test_raw_config_bytes_are_locked(tmp_path):
    path = tmp_path / protocol.CONFIG
    path.parent.mkdir(parents=True)
    path.write_bytes((protocol.ROOT / protocol.CONFIG).read_bytes() + b" ")
    with pytest.raises(ValueError, match="exact clean01"):
        protocol.config_at(tmp_path)


def test_plan_preserves_parent_method_and_binds_historical_manifest():
    value = protocol.build_plan()
    old = protocol.parent.build_plan()
    changed = {"schema", "output_namespace", "config", "input_sha256", "recording_policy"}
    assert set(value) == set(old)
    assert all(value[key] == old[key] for key in old if key not in changed)
    assert len(value["cells"]) == 216 and len(value["derivative_cells"]) == 96
    assert value["input_sha256"][protocol.PARENT_LOCK] == protocol.PARENT_LOCK_SHA
    assert value["input_sha256"][protocol.parent.PREP_JSON] == protocol.PARENT_CERTIFICATE_SHA
    assert not (protocol.ROOT / protocol.OUTPUT).exists()


def test_new_certificate_gate_and_original_binder_are_distinct(tmp_path):
    from scripts import paired_common_drift_comply_1800_clean01_recording as recording

    assert recording.CERTIFICATE == protocol.PREP_JSON != protocol.parent.PREP_JSON
    assert protocol.BINDING == protocol.parent.BINDING
    assert set(protocol.parent.SOURCE_PATHS) <= set(protocol.SOURCE_PATHS)
    with pytest.raises((ValueError, FileNotFoundError)):
        protocol.require_preparation_certificate(tmp_path)
    assert not list(tmp_path.iterdir())
