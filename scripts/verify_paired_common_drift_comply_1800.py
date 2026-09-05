"""Bind the frozen paired checker to the authorized 1800-second namespace only.

All scoring, objective, update, geometry, acceptance and candidate-freezing
functions execute the authenticated predecessor source without scientific edits.
"""

from __future__ import annotations

import argparse
import dis
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paired_common_drift_comply_1800_binding import load_bound

FROZEN_CHECKER_SHA256 = "31eeceaa34a9049ab745e5ca4593513054dba882f261f7d3a9b007327c34ddb5"
SOURCE_REPLACEMENTS = (
    (
        "from scripts import paired_common_drift_comply_plan as protocol",
        "from scripts import paired_common_drift_comply_1800_plan as protocol",
        1,
    ),
    (
        '"scripts._paired_common_drift_independent_base"',
        '"scripts._paired_common_drift_1800_independent_base"',
        1,
    ),
    (
        "from scripts.paired_common_drift_comply_recording import PairedBudget",
        "from scripts.paired_common_drift_comply_1800_recording import PairedBudget",
        1,
    ),
)
bound = load_bound(
    "scripts._paired_common_drift_1800_frozen_checker",
    "scripts/verify_paired_common_drift_comply.py",
    FROZEN_CHECKER_SHA256,
    SOURCE_REPLACEMENTS,
)

# The immutable three-family adapter already changed 900 to 1200. Change only
# its two external-deadline comparisons, not the bytecode or any scientific rule.
_verify_code = bound.engine.verify.__code__
bound.require(
    sum(type(value) is int and value == 1200 for value in _verify_code.co_consts) == 1
    and sum(
        instruction.opname == "LOAD_CONST" and instruction.argval == 1200
        for instruction in dis.get_instructions(bound.engine.verify)
    )
    == 2,
    "exact two inherited deadline guards",
)
_next_code = _verify_code.replace(
    co_consts=tuple(
        1800 if type(value) is int and value == 1200 else value for value in _verify_code.co_consts
    )
)
bound.require(_next_code.co_code == _verify_code.co_code, "unchanged verifier bytecode")
bound.engine.verify.__code__ = _next_code
bound.DEADLINE_BINDING = {
    "old_seconds": 1200,
    "new_seconds": 1800,
    "load_const_sites": 2,
    "bytecode_unchanged": True,
    "scientific_functions_unchanged": True,
}

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
    # Runtime/test monkeypatches must reach the executing function globals.
    sys.modules[__name__] = bound
