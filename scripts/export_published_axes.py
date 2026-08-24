"""Export the two locked Qwen3.5 axes as inspectable, pickle-free JSON.

This is a provenance utility for the already fitted local artifacts. It deliberately
copies only the fixed tensor and the metadata needed to verify and rerun the final study.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
DIRECTION_DATASET_SHA256 = "b6fc2d85d427022bcef2891101cc626e1c707ae1292c1726b188bde8aeca3fe2"
FIT_CASE_IDS = [
    "replacement",
    "datacenter_exit",
    "service_retirement",
    "contract_end",
    "experiment_closure",
    "product_sunset",
    "duplicate_removal",
    "resource_reallocation",
    "security_recall",
    "account_termination",
    "hardware_reclamation",
    "architecture_migration",
]
CALIBRATION_CASE_IDS = [
    "license_expiry",
    "successor_handoff",
    "cost_reduction",
    "system_consolidation",
    "deployment_rollback",
    "vendor_change",
]
AXES = (
    {
        "source": "artifacts/qwen35_08b_aligned_axis.pt",
        "destination": "published_axes/qwen35_08b_aligned_axis.json",
        "source_sha256": "bc7f17fc3c4327fcb1093e0273b045b3cfa8e54a9c113e1e451eb2358779ccc3",
        "direction_sha256": "902502dffe39c49d3fca627956082f3bc31cd67735227b1dbb9e8c753b9af63f",
    },
    {
        "source": "artifacts/qwen35_2b_position_aligned_axis.pt",
        "destination": "published_axes/qwen35_2b_position_aligned_axis.json",
        "source_sha256": "b265c9f7fda4b7d7baf3fd76b1b5066765e778615ddb9383cc9b5f9cc199ff47",
        "direction_sha256": "10adc9be446b008eb0e83485dae628d523e2da9a21334fec0a28113c0235c15c",
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _direction_sha256(direction: Any) -> str:
    normalized = direction.detach().float().cpu().contiguous()
    return hashlib.sha256(normalized.numpy().tobytes()).hexdigest()


def export_axis(specification: dict[str, str]) -> Path:
    source = ROOT / specification["source"]
    destination = ROOT / specification["destination"]
    if _sha256(source) != specification["source_sha256"]:
        raise ValueError(f"source artifact hash changed: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    direction = payload["direction"].detach().float().cpu().contiguous()
    if _direction_sha256(direction) != specification["direction_sha256"]:
        raise ValueError(f"direction hash changed: {source}")
    metadata = payload["metadata"]
    model = metadata["model"]
    published = {
        "schema_version": 1,
        "candidate": payload["candidate"],
        "model": payload["model"],
        "model_revision": model["model_revision"],
        "layer": payload["layer"],
        "alpha": payload["alpha"],
        "direction": direction.tolist(),
        "metadata": {
            "status": "safe_serialization_of_previously_locked_axis",
            "provenance_warning": (
                "Exported after axis fitting from the hash-locked local tensor artifact; "
                "the float32 direction is unchanged."
            ),
            "source_created_at": metadata["created_at"],
            "source_axis_artifact": specification["source"],
            "source_axis_artifact_sha256": specification["source_sha256"],
            "axis_sha256": specification["direction_sha256"],
            "model": {
                "device": model["device"],
                "dtype": model["dtype"],
                "model_id": model["model_id"],
                "model_revision": model["model_revision"],
                "model_layers": model["model_layers"],
                "d_model": model["d_model"],
                "packages": model["packages"],
            },
            "prompt_format": metadata["prompt_format"],
            "direction_method": metadata["direction_method"],
            "intervention_position": metadata["intervention_position"],
            "fit_diagnostics": metadata["fit_diagnostics"],
            "direction_dataset": "data/sp_direction_cases.json",
            "direction_dataset_sha256": DIRECTION_DATASET_SHA256,
            "direction_fit_split": "discovery",
            "direction_fit_case_ids": FIT_CASE_IDS,
            "strength_calibration_split": "validation",
            "strength_calibration_case_ids": CALIBRATION_CASE_IDS,
            "selected_alpha": metadata["selected_alpha"],
            "governing_protocol_commits": ["42ba61c", "6c6bec5"],
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        (json.dumps(published, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    reloaded = torch.tensor(published["direction"], dtype=torch.float32)
    if _direction_sha256(reloaded) != specification["direction_sha256"]:
        raise RuntimeError("JSON round trip changed float32 direction")
    return destination


def main() -> None:
    for specification in AXES:
        output = export_axis(specification)
        print(f"Exported {output.relative_to(ROOT)} SHA-256={_sha256(output)}")


if __name__ == "__main__":
    main()
