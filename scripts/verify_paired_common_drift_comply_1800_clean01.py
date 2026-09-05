"""Clean01 identity binding of the immutable 1800-second independent checker.

No scientific or deadline changes. This checker does not certify executable
worker dispatch; the predecessor CLI coverage gap requires separate validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paired_common_drift_comply_1800_binding import load_bound

wrapper = load_bound(
    "scripts._paired1800_clean01_checker_wrapper",
    "scripts/verify_paired_common_drift_comply_1800.py",
    "690c94cd14a2095eabcb039a23adbc0869f9ec6e9345ad9c8c59bcdd9410c76b",
    (
        (
            "paired_common_drift_comply_1800_plan",
            "paired_common_drift_comply_1800_clean01_plan",
            1,
        ),
        (
            "paired_common_drift_comply_1800_recording",
            "paired_common_drift_comply_1800_clean01_recording",
            1,
        ),
        (
            "scripts._paired_common_drift_1800_independent_base",
            "scripts._paired_common_drift_1800_clean01_independent_base",
            1,
        ),
        (
            "scripts._paired_common_drift_1800_frozen_checker",
            "scripts._paired_common_drift_1800_clean01_frozen_checker",
            1,
        ),
    ),
)
bound = wrapper.bound
bound.CLEAN01_BINDING = wrapper.TIME_ONLY_BINDING

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = bound.read_sealed_recording()
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("summary", "construction_stages", "optimizer_checks")
            },
            indent=2,
        )
    )
    if result["status"] == "INCONCLUSIVE":
        raise SystemExit(1)
else:
    sys.modules[__name__] = bound
