"""Clean01 dispatch-repair identity; immutable 1800-second scientific method."""

from __future__ import annotations

from pathlib import Path

from scripts import paired_common_drift_comply_1800_plan as parent

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/paired_common_drift_comply_three_family_1800_clean01_v1.json"
CONFIG_SHA = "10309536f5adb95ffc3cd9691c462ff4a04ed2f6e4f1c764deeebb02a7ff1aa0"
DOC = "docs/PAIRED_COMMON_DRIFT_COMPLY_1800_CLEAN01_V1.md"
PREP_REPORT = "docs/PAIRED_COMMON_DRIFT_COMPLY_1800_CLEAN01_PREPARATION.md"
PREP_JSON = "docs/paired_common_drift_comply_1800_clean01_preparation.json"
SCRIPT = "scripts/paired_common_drift_comply_1800_clean01.py"
VERIFY = "scripts/verify_paired_common_drift_comply_1800_clean01.py"
TEST = "tests/test_paired_common_drift_comply_1800_clean01_plan.py"
PLAN = "scripts/paired_common_drift_comply_1800_clean01_plan.py"
OPTIMIZER = parent.OPTIMIZER
RECORDING = "scripts/paired_common_drift_comply_1800_clean01_recording.py"
RECORDING_POLICY = "configs/paired_common_drift_comply_1800_clean01_recording_policy.json"
BINDING = parent.BINDING
OUTPUT = "evidence/paired_common_drift_comply_three_family_1800_clean01_v1_qwen35_08b"
AUTH_KEY = "SP_LENSE_PAIRED_COMMON_DRIFT_COMPLY_1800_CLEAN01_AUTHORIZATION"
AUTH_SCOPE = "one fresh clean01 paired common-drift COMPLY construction;216F/96D/1800s;no retry"
PARENT_LOCK = (
    "evidence/paired_common_drift_comply_three_family_1800_v1_qwen35_08b/preregistration.json"
)
PARENT_LOCK_SHA = "4b71558149c81f5c61a72a16c8a2f750c7063bad52dc7a3236b8fbb034744e76"
PARENT_CERTIFICATE_SHA = "fb25e4a26b51e4838cb726e9864bad4e953a38fe993dcac7f10c38b9424d7f80"
CLI_COVERAGE_LIMITATION = (
    "The prior positive certificate did not cover the failed direct worker CLI dispatch; "
    "its authenticated bytes are historical provenance, not proof of repaired launcher correctness."
)
ALLOWED_CONFIG_CHANGES = frozenset(
    {
        "schema",
        "output_namespace",
        "provenance",
        "execution_authority",
        "recording_policy",
    }
)
SOURCE_PATHS = tuple(
    dict.fromkeys(
        (
            *parent.SOURCE_PATHS,
            CONFIG,
            DOC,
            PREP_REPORT,
            PREP_JSON,
            SCRIPT,
            VERIFY,
            TEST,
            PLAN,
            RECORDING,
            RECORDING_POLICY,
            "tests/test_paired_common_drift_comply_1800_clean01.py",
            "tests/test_paired_common_drift_comply_1800_clean01_recording.py",
        )
    )
)
require, read, sha, io = parent.require, parent.read, parent.sha, parent.io
isolate, canonical_sha, authenticated = parent.isolate, parent.canonical_sha, parent.authenticated
FAMILIES, ORDERS, DISPLAYS, PAIRS = parent.FAMILIES, parent.ORDERS, parent.DISPLAYS, parent.PAIRS


def config_at(root=ROOT):
    require(sha((root / CONFIG).read_bytes()) == CONFIG_SHA, "exact clean01 config bytes")
    return read(root / CONFIG)


def validate_config(config, old):
    require(
        set(config) == set(old)
        and all(config[k] == v for k, v in old.items() if k not in ALLOWED_CONFIG_CHANGES),
        "clean01 strict diff: dispatch identity/provenance only, no time or scientific change",
    )
    require(
        config["schema"] == "sp_lense.paired_common_drift_comply_three_family_1800_clean01_v1.v1"
        and config["output_namespace"] == OUTPUT
        and (
            config["maximum_forwards"],
            config["maximum_derivatives"],
            config["maximum_updates"],
            config["timeout_seconds"],
        )
        == (216, 96, 8, 1800)
        and config["execution_authority"]
        == {
            "preparation_only": True,
            "eventual_run_requires_separate_supervisor_authorization": True,
            "environment_key": AUTH_KEY,
            "authorization_scope": AUTH_SCOPE,
        },
        "fresh clean01 namespace and separately authorized one-attempt scope",
    )
    require(config["recording_policy"]["path"] == RECORDING_POLICY, "clean01 recording binding")
    provenance = config["provenance"]
    require(
        provenance["prior_preregistration"] == {"path": PARENT_LOCK, "sha256": PARENT_LOCK_SHA}
        and provenance["prior_preparation_certificate"]
        == {"path": parent.PREP_JSON, "sha256": PARENT_CERTIFICATE_SHA}
        and provenance["prior_cli_coverage_limitation"] == CLI_COVERAGE_LIMITATION
        and provenance["prior_certificate_is_repaired_cli_proof"] is False
        and provenance["only_dispatch_identity_changes"] is True
        and provenance["only_worker_time_changes"] is False
        and provenance["resume_allowed"] is False
        and provenance["warm_start_allowed"] is False
        and provenance["adaptive_development"] is True
        and provenance["independent_confirmation"] is False,
        "historical certificate limitation and fresh-zero launcher-repair provenance",
    )


def recording_policy(root=ROOT):
    from scripts import paired_common_drift_comply_1800_clean01_recording as recording

    return recording.recording_policy(root)


def require_preparation_certificate(root=ROOT):
    from scripts import paired_common_drift_comply_1800_clean01_recording as recording

    return recording.require_certificate(root)


def storage_preflight(root, config):
    result = parent.storage_preflight(root, config)
    recording_policy(root)
    return result


def build_plan(root=ROOT):
    config, old = config_at(root), parent.config_at(root)
    validate_config(config, old)
    original = parent.build_plan(root)
    lock = authenticated(PARENT_LOCK, PARENT_LOCK_SHA, root)
    require(
        lock["plan"] == original and len(lock["source_sha256"]) == 133,
        "exact prior 1800 plan metadata and 133-source manifest; no outcome replay",
    )
    # Authenticate the historic certificate without interpreting its positive
    # status as clean01 CLI coverage. The new certificate gate is independent.
    require(
        sha((root / parent.PREP_JSON).read_bytes()) == PARENT_CERTIFICATE_SHA,
        "unchanged historical certificate bytes, not repaired dispatch proof",
    )
    inputs = dict(original["input_sha256"])
    additions = {
        **lock["source_sha256"],
        PARENT_LOCK: PARENT_LOCK_SHA,
        parent.PREP_JSON: PARENT_CERTIFICATE_SHA,
    }
    for path, digest in additions.items():
        require(path not in inputs or inputs[path] == digest, "consistent inherited identity")
        require(sha((root / path).read_bytes()) == digest, "unchanged source/provenance bytes")
        inputs[path] = digest
    policy = recording_policy(root)
    require(
        policy == authenticated(RECORDING_POLICY, config["recording_policy"]["sha256"], root),
        "exact clean01 recording policy",
    )
    inputs[RECORDING_POLICY] = config["recording_policy"]["sha256"]
    return {
        **original,
        "schema": config["schema"],
        "output_namespace": OUTPUT,
        "config": config,
        "input_sha256": inputs,
        "recording_policy": policy,
    }
