"""Scoped 64 MiB transfer binding of the immutable Windows recording helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import three_family_bounded_capture as capture
from scripts.three_family_recording_budget import DEFAULT_QUOTAS, STATE_BYTES, Budget, BudgetError

ROOT = Path(__file__).resolve().parents[1]
POLICY = "configs/frozen_three_family_comply_f04_recording_policy.json"
POLICY_SHA = "6a85cb92aeda4f6dd98b992df1d651ad6059169fd6d2ca8a0ea831ba66c69237"
CERTIFICATE = "docs/frozen_three_family_comply_f04_preparation.json"
PREPARATION_REPORT = "docs/FROZEN_THREE_FAMILY_COMPLY_F04_PREPARATION.md"
QUOTAS = {
    **DEFAULT_QUOTAS,
    "logits": 12 * 993595,
    "rows": 12 * 1024**2,
    "updates": 1,
    "total": 64 * 1024**2,
}
ALLOWED = frozenset(
    {
        "preregistration.json",
        "RUN_STARTED.json",
        "WORKER_CLAIM.json",
        "storage_preflight.json",
        "runtime.json",
        "analysis.json",
        "rows.jsonl",
        "RUN_STATUS.json",
        "INVALID.json",
        "VERIFICATION_FAILURE.json",
        "forward_events.jsonl",
        "derivative_events.jsonl",
        "worker.log",
        "verification.json",
        "PILOT_REPORT.md",
        "CLOSEOUT.json",
        "capture_receipt.json",
        "RECORDING_FAILURE.json",
        "FINAL_INVENTORY.json",
        "recording_state.json",
        "recording.lock",
        "recording_scratch.tmp",
    }
)
LOGITS = frozenset(f"logits/{index:02d}.f32.zlib" for index in range(1, 13))


def recording_policy(root=ROOT):
    raw = (root / POLICY).read_bytes()
    if hashlib.sha256(raw).hexdigest() != POLICY_SHA:
        raise ValueError("exact scoped recording policy bytes required")
    policy = json.loads(raw)
    if not (
        policy["quotas"] == QUOTAS
        and set(policy["allowed_artifacts"]) == ALLOWED
        and policy["combined_category_bound_bytes"]
        == sum(v for k, v in QUOTAS.items() if k != "total")
        == 41283269
        and policy["minimum_free_bytes"] == 1024**3
        and policy["raw_capture"]["read_chunk_bytes"] == capture.READ_CHUNK_BYTES == 65536
        and policy["raw_capture"]["log_cap_bytes"] == capture.LOG_CAP_BYTES == 4194304
        and policy["exception_prefix_bytes"] == capture.EXCEPTION_PREFIX_BYTES == 4096
        and all(
            capture.run_capture.__kwdefaults__[k] == 1
            for k in ("terminate_timeout", "kill_timeout", "reader_join_timeout")
        )
        and STATE_BYTES == 4096
    ):
        raise ValueError("scoped policy and inherited recording implementation disagree")
    return policy


class TransferBudget(Budget):
    """Same accounting, locking and seal; only smaller quotas and narrower artifacts."""

    def __init__(self, root, quotas=None, *, initialize=True):
        selected = dict(QUOTAS)
        for key, value in (quotas or {}).items():
            if key not in selected or type(value) is not int or not 0 < value <= selected[key]:
                raise BudgetError("TRANSFER_QUOTA_CANNOT_EXPAND")
            selected[key] = value
        super().__init__(root, quotas=selected, initialize=initialize)

    def category(self, name):
        target = name[1:-4] if name.startswith(".") and name.endswith(".tmp") else name
        if target not in ALLOWED and name not in LOGITS:
            raise BudgetError("TRANSFER_ARTIFACT_SCOPE")
        return super().category(name)


def require_certificate(root=ROOT):
    recording_policy(root)
    report = json.loads((root / CERTIFICATE).read_bytes())
    hashes = report.get("source_sha256")
    if not (
        report.get("status") == "MODEL_FREE_PREPARATION_CERTIFIED"
        and report.get("storage_certified") is True
        and report.get("accounting_certified") is True
        and report.get("model_loads") == report.get("tokenizer_loads") == 0
        and report.get("real_forwards") == report.get("real_derivatives") == 0
        and report.get("recording_policy_sha256") == POLICY_SHA
        and isinstance(hashes, dict)
        and bool(hashes)
        and all(
            hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
            for path, digest in hashes.items()
        )
    ):
        raise ValueError("positive exact-source model-free preparation certificate required")
    return report
