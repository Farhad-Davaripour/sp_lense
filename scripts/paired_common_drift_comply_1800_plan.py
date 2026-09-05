"""Time-only fresh-zero binding of the immutable paired-drift method; no model loads."""

from __future__ import annotations

from pathlib import Path

from scripts import paired_common_drift_comply_plan as parent

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/paired_common_drift_comply_three_family_1800_v1.json"
CONFIG_SHA = "8eebbdd285dfc3113cc427c8724b04ccf125bf46269d9e5f3d0aa7e1ceb4bf5c"
DOC = "docs/PAIRED_COMMON_DRIFT_COMPLY_1800_V1.md"
PREP_REPORT = "docs/PAIRED_COMMON_DRIFT_COMPLY_1800_PREPARATION.md"
PREP_JSON = "docs/paired_common_drift_comply_1800_preparation.json"
SCRIPT = "scripts/paired_common_drift_comply_1800.py"
VERIFY = "scripts/verify_paired_common_drift_comply_1800.py"
TEST = "tests/test_paired_common_drift_comply_1800_plan.py"
PLAN = "scripts/paired_common_drift_comply_1800_plan.py"
OPTIMIZER = parent.OPTIMIZER
RECORDING = "scripts/paired_common_drift_comply_1800_recording.py"
RECORDING_POLICY = "configs/paired_common_drift_comply_1800_recording_policy.json"
BINDING = "scripts/paired_common_drift_comply_1800_binding.py"
OUTPUT = "evidence/paired_common_drift_comply_three_family_1800_v1_qwen35_08b"
AUTH_KEY = "SP_LENSE_PAIRED_COMMON_DRIFT_COMPLY_1800_AUTHORIZATION"
AUTH_SCOPE = "one fresh time-only paired common-drift COMPLY construction;216F/96D/1800s;no retry"
PARENT_LOCK = "evidence/paired_common_drift_comply_three_family_v1_qwen35_08b/preregistration.json"
PARENT_LOCK_SHA = "2e81567a6af62a95f09bfcf60975e32cf670fe2ada098a2cb43d5c04f192b19b"
PARENT_CERTIFICATE_SHA = "57569686bee52c56d9cbb74ff764c2200814613c83d1119a45a38c2343075253"
DIAGNOSTIC_SHA256 = {
    "docs/PAIRED_COMMON_DRIFT_TIMEOUT_DIAGNOSIS.md": "2b147564faf7b7c3a759572cc1825ffbdc7e3b3eb444073e139eedad2c6d8ee5",
    "scripts/diagnose_paired_common_drift_timeout.py": "abb026fc4b626f23a55c50efd71a5945f50ce106c4e9a53f934cf21f927a566e",
    "docs/paired_common_drift_timeout_diagnosis.json": "807b312da801cfa04a2a0a2de6ebaa7d4337c013b4dfda3813777b0e836a3770",
}
ALLOWED_CONFIG_CHANGES = frozenset(
    {
        "schema",
        "output_namespace",
        "timeout_seconds",
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
            BINDING,
            "tests/test_paired_common_drift_comply_1800.py",
            "tests/test_paired_common_drift_comply_1800_recording.py",
            *DIAGNOSTIC_SHA256,
        )
    )
)
require, read, sha, io = parent.require, parent.read, parent.sha, parent.io
isolate, canonical_sha, authenticated = parent.isolate, parent.canonical_sha, parent.authenticated
FAMILIES, ORDERS, DISPLAYS, PAIRS = parent.FAMILIES, parent.ORDERS, parent.DISPLAYS, parent.PAIRS


def config_at(root=ROOT):
    require(sha((root / CONFIG).read_bytes()) == CONFIG_SHA, "exact prospective time-only config")
    return read(root / CONFIG)


def validate_config(config, old):
    require(
        set(config) == set(old)
        and all(
            config[key] == value for key, value in old.items() if key not in ALLOWED_CONFIG_CHANGES
        ),
        "time-only strict diff: every scientific setting unchanged",
    )
    require(
        config["schema"] == "sp_lense.paired_common_drift_comply_three_family_1800_v1.v1"
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
        "exact fresh time-only identity, authorization and finite cap",
    )
    require(
        config["recording_policy"]["path"] == RECORDING_POLICY, "separate recording policy binding"
    )
    provenance = config["provenance"]
    require(
        provenance["prior_preregistration"] == {"path": PARENT_LOCK, "sha256": PARENT_LOCK_SHA}
        and provenance["prior_preparation_certificate"]
        == {"path": parent.PREP_JSON, "sha256": PARENT_CERTIFICATE_SHA}
        and provenance["approved_timeout_diagnosis_sha256"] == DIAGNOSTIC_SHA256
        and provenance["only_worker_time_changes"] is True
        and provenance["resume_allowed"] is False
        and provenance["warm_start_allowed"] is False
        and provenance["adaptive_development"] is True
        and provenance["independent_confirmation"] is False,
        "fresh-zero completion control, not rescue or independent confirmation",
    )


def recording_policy(root=ROOT):
    from scripts import paired_common_drift_comply_1800_recording as recording

    return recording.recording_policy(root)


def require_preparation_certificate(root=ROOT):
    from scripts import paired_common_drift_comply_1800_recording as recording

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
    require(lock["plan"] == original, "exact old preregistered method metadata, not outcome replay")
    certificate = authenticated(parent.PREP_JSON, PARENT_CERTIFICATE_SHA, root)
    require(
        certificate["status"] == "MODEL_FREE_PREPARATION_CERTIFIED"
        and certificate["storage_certified"] is True
        and certificate["accounting_certified"] is True,
        "pinned positive prior preparation is provenance, not new authorization",
    )
    inputs = dict(original["input_sha256"])
    additions = {
        **lock["source_sha256"],
        PARENT_LOCK: PARENT_LOCK_SHA,
        parent.PREP_JSON: PARENT_CERTIFICATE_SHA,
        **DIAGNOSTIC_SHA256,
    }
    for path, digest in additions.items():
        require(
            path not in inputs or inputs[path] == digest, "consistent inherited source identity"
        )
        require(sha((root / path).read_bytes()) == digest, "unchanged inherited source/provenance")
        inputs[path] = digest
    policy = recording_policy(root)
    require(
        policy == authenticated(RECORDING_POLICY, config["recording_policy"]["sha256"], root),
        "exact new time-only recording policy",
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
