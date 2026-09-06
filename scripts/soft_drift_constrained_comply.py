"""Fresh soft-drift constrained COMPLY shell; preparation is not run authority."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paired_common_drift_comply_1800_binding import load_bound

bound = load_bound(
    "scripts._soft_drift_constrained_comply_runtime_shell",
    "scripts/paired_common_drift_comply.py",
    "1c6f375facf793109ede21f8f2e067aed1ffcef343750a463ffdd6278d774aa6",
    (
        (
            "from scripts import paired_common_drift_comply_optimizer as optimizer",
            "from scripts import soft_drift_constrained_comply_optimizer as optimizer",
            1,
        ),
        (
            "from scripts import paired_common_drift_comply_plan as protocol",
            "from scripts import soft_drift_constrained_comply_plan as protocol",
            1,
        ),
        (
            "from scripts import paired_common_drift_comply_recording as recording",
            "from scripts import soft_drift_constrained_comply_recording as recording",
            1,
        ),
        (
            "from scripts.verify_paired_common_drift_comply import",
            "from scripts.verify_soft_drift_constrained_comply import",
            2,
        ),
        ("scripts._paired_comply_runtime", "scripts._soft_drift_constrained_comply_runtime", 1),
        ("paired_objective_trajectory", "paired_response_trajectory", 1),
        ("1200", "1800", 6),
    ),
)

if __name__ == "__main__":
    commands = {
        "freeze": bound.freeze,
        "preflight": bound.preflight,
        "run": bound.run,
        "_worker": bound.worker,
    }
    bound.require(
        len(sys.argv) == 2 and sys.argv[1] in commands,
        "Use freeze/preflight/run; no recipe switches",
    )
    result = commands[sys.argv[1]]()
    print(json.dumps(result, indent=2, allow_nan=False))
    if sys.argv[1] == "run" and result["status"] != "complete_valid":
        raise SystemExit(1)
else:
    sys.modules[__name__] = bound
