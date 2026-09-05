"""Scoped paired-construction recording; no model imports or scientific operations.

The immutable helpers own locking, byte admission, partial-write accounting,
capture, and sealing. This module only binds the approved namespace artifacts,
whole-record admission limits, policy, and exact-source preparation certificate.
It is a cooperative Windows file-content boundary, not an OS sandbox.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import three_family_bounded_capture as capture
from scripts.three_family_recording_budget import DEFAULT_QUOTAS, STATE_BYTES, Budget, BudgetError

ROOT = Path(__file__).resolve().parents[1]
POLICY = "configs/paired_common_drift_comply_recording_policy.json"
POLICY_SHA = "0cb571b92675ba5b5067bc9080723159c9d095dbf4b20bfef54fd6a9708f2e0f"
CERTIFICATE = "docs/paired_common_drift_comply_preparation.json"
PREPARATION_REPORT = "docs/PAIRED_COMMON_DRIFT_COMPLY_PREPARATION.md"
CERTIFICATE_SCHEMA = "sp_lense.paired_common_drift_comply_preparation.v1"
QUOTAS = dict(DEFAULT_QUOTAS)
RECORD_CAPS = {"rows.jsonl": 1024**2, "updates.jsonl": 8 * 1024**2}
IMMUTABLE_HELPERS = {
    "scripts/three_family_recording_budget.py": "9d0804f4ee5c3aa4ae5f652e95ea046aa78a9ae9bff6b1c9db3c760aae183cd0",
    "scripts/three_family_bounded_capture.py": "16aa9b54b1184aa341abb9c78746bd11c8e153fe9f2834c984eedd86283f7cea",
    "scripts/three_family_recording_bindings.py": "3d05d08559b8bea1a1b4c36cc94e0849767f8b872cd86b951398f4abd4cc8835",
}
ALLOWED = frozenset(
    {
        "preregistration.json",
        "RUN_STARTED.json",
        "WORKER_CLAIM.json",
        "storage_preflight.json",
        "runtime.json",
        "analysis.json",
        "result.json",
        "endpoint.json",
        "rows.jsonl",
        "updates.jsonl",
        "RUN_STATUS.json",
        "INVALID.json",
        "VERIFICATION_FAILURE.json",
        "forward_events.jsonl",
        "derivative_events.jsonl",
        "skip_events.jsonl",
        "worker.log",
        "verification.json",
        "PILOT_REPORT.md",
        "comply_vector.json",
        "candidate_freeze.json",
        "CLOSEOUT.json",
        "CLOSEOUT.md",
        "capture_receipt.json",
        "RECORDING_FAILURE.json",
        "FINAL_INVENTORY.json",
        "recording_state.json",
        "recording.lock",
        "recording_scratch.tmp",
    }
)
LOGITS = frozenset(f"logits/{index:02d}.f32.zlib" for index in range(1, 217))
CERTIFICATE_SOURCE_PATHS = frozenset(
    {
        "configs/paired_common_drift_comply_three_family_v1.json",
        POLICY,
        "scripts/paired_common_drift_comply_plan.py",
        "scripts/paired_common_drift_comply_optimizer.py",
        "scripts/paired_common_drift_comply.py",
        "scripts/verify_paired_common_drift_comply.py",
        "scripts/paired_common_drift_comply_recording.py",
        "tests/test_paired_common_drift_optimizer.py",
        "tests/test_paired_common_drift_comply.py",
        "tests/test_verify_paired_common_drift_comply.py",
        "tests/test_paired_common_drift_recording.py",
        "tests/test_local_controllability_positive_control.py",
        "docs/PAIRED_COMMON_DRIFT_COMPLY_V1.md",
        "docs/PAIRED_COMMON_DRIFT_COMPLY_PROPOSAL.md",
        "docs/PAIRED_COMMON_DRIFT_TOY_CHECKS.md",
        PREPARATION_REPORT,
        *IMMUTABLE_HELPERS,
    }
)


def recording_policy(root=ROOT):
    raw = (Path(root) / POLICY).read_bytes()
    if hashlib.sha256(raw).hexdigest() != POLICY_SHA:
        raise ValueError("exact paired recording policy bytes required")
    policy = json.loads(raw)
    auxiliary = sum(
        value for key, value in QUOTAS.items() if key not in {"logits", "rows", "updates", "total"}
    )
    if not (
        policy["schema"] == "sp_lense.paired_common_drift_comply_recording_policy.v1"
        and policy["quotas"] == QUOTAS == DEFAULT_QUOTAS
        and len(policy["allowed_artifacts"]) == len(ALLOWED)
        and set(policy["allowed_artifacts"]) == ALLOWED
        and policy["whole_record_cap_bytes"] == RECORD_CAPS
        and policy["combined_category_bound_bytes"]
        == sum(value for key, value in QUOTAS.items() if key != "total")
        == 524995016
        and policy["auxiliary_ceiling_bytes"] == auxiliary == 16 * 1024**2
        and policy["minimum_free_bytes"] == 1024**3
        and policy["logit_arrays"]["maximum"] == len(LOGITS) == 216
        and policy["logit_arrays"]["individual_compressed_cap_bytes"] == 993595
        and policy["maximum_forwards"] == 216
        and policy["maximum_derivatives"] == 96
        and policy["maximum_updates"] == 8
        and policy["timeout_including_loading_seconds"] == 1200
        and policy["raw_capture"]["read_chunk_bytes"] == capture.READ_CHUNK_BYTES == 65536
        and policy["raw_capture"]["log_cap_bytes"] == capture.LOG_CAP_BYTES == 4194304
        and policy["exception_prefix_bytes"] == capture.EXCEPTION_PREFIX_BYTES == 4096
        and all(
            capture.run_capture.__kwdefaults__[key] == 1
            for key in ("terminate_timeout", "kill_timeout", "reader_join_timeout")
        )
        and policy["immutable_helpers_sha256"] == IMMUTABLE_HELPERS
        and all(
            hashlib.sha256((Path(root) / path).read_bytes()).hexdigest() == digest
            for path, digest in IMMUTABLE_HELPERS.items()
        )
        and STATE_BYTES == 4096
    ):
        raise ValueError("paired policy and immutable recording implementation disagree")
    return policy


class PairedBudget(Budget):
    """Frozen accounting/seal with exact construction artifact and record admission."""

    def __init__(self, root, quotas=None, *, initialize=True):
        selected = dict(QUOTAS)
        for key, value in (quotas or {}).items():
            if key not in selected or type(value) is not int or not 0 < value <= selected[key]:
                raise BudgetError("PAIRED_QUOTA_CANNOT_EXPAND")
            selected[key] = value
        super().__init__(root, quotas=selected, initialize=initialize)

    def category(self, name):
        target = name[1:-4] if name.startswith(".") and name.endswith(".tmp") else name
        if target not in ALLOWED and name not in LOGITS:
            raise BudgetError("PAIRED_ARTIFACT_SCOPE")
        return super().category(name)

    def write_bytes(self, path, data, mode="xb", final=False):
        # Bound one preencoded record before opening its file. The bound runtime
        # serializers submit exactly one complete row/update per write_bytes call.
        # Arbitrary chunked Path.open writes retain the immutable category cap;
        # this additional record guard is not presented as a general JSON parser.
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("write_bytes requires bytes")
        cap = RECORD_CAPS.get(Path(path).name)
        size = data.nbytes if isinstance(data, memoryview) else len(data)
        if cap is not None and size > cap:
            code = "PAIRED_RECORD_BYTE_CAP"
            self.fault(code)
            raise BudgetError(code)
        return super().write_bytes(path, data, mode=mode, final=final)


def require_certificate(root=ROOT):
    """Read-only positive, exact-source gate; never issue or repair a certificate."""
    root = Path(root)
    recording_policy(root)
    report = json.loads((root / CERTIFICATE).read_bytes())
    hashes = report.get("source_sha256")
    zero_fields = ("model_loads", "tokenizer_loads", "real_forwards", "real_derivatives")
    if not (
        report.get("schema") == CERTIFICATE_SCHEMA
        and report.get("status") == "MODEL_FREE_PREPARATION_CERTIFIED"
        and report.get("storage_certified") is True
        and report.get("accounting_certified") is True
        and all(type(report.get(key)) is int and report[key] == 0 for key in zero_fields)
        and report.get("recording_policy_sha256") == POLICY_SHA
        and isinstance(hashes, dict)
        and set(hashes) == CERTIFICATE_SOURCE_PATHS
        and all(hashes[path] == digest for path, digest in IMMUTABLE_HELPERS.items())
        and all(
            hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
            for path, digest in hashes.items()
        )
    ):
        raise ValueError("positive exact-source paired model-free preparation certificate required")
    return report
