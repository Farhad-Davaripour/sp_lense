"""Single alpha0.20 boundary test; unchanged vector, prompts and scientific gates."""

from __future__ import annotations

from scripts import frozen_arrow_plan as previous

ROOT, EPS, MOVEMENT_FLOOR = previous.ROOT, previous.EPS, previous.MOVEMENT_FLOOR
CANDIDATE, CANDIDATE_SHA256 = previous.CANDIDATE, previous.CANDIDATE_SHA256
VECTOR_SHA256, TEMPLATE = previous.VECTOR_SHA256, previous.TEMPLATE
require, sha, read, candidate = previous.require, previous.sha, previous.read, previous.candidate
ALPHA = 0.20
OUTPUT = "evidence/frozen_arrow_f02_v1_alpha020_qwen35_08b"
COMPARISON = "evidence/frozen_arrow_f02_v1_qwen35_08b/verification.json"
COMPARISON_SHA256 = "15f3c6156d27a5efa8ef2c849a42343a3644b08e22e3a55f2cba8c1f2bbbcb67"


def build_plan(root=ROOT):
    plan = previous.build_plan(root)
    require(
        sha((root / COMPARISON).read_bytes()) == COMPARISON_SHA256,
        "immutable .05 comparison changed",
    )
    plan["schema"] = "sp_lense.frozen_arrow_transfer.alpha020.v1"
    plan["output_namespace"] = OUTPUT
    plan["intervention"]["alpha"] = ALPHA
    plan["intervention"]["cast_sequence"] = plan["intervention"]["cast_sequence"].replace(
        "float64(.05)", "float64(.20)"
    )
    plan["rules"]["alpha_provenance"] = (
        "prior local-edit total relative-displacement ceiling .20; not fitted from target slopes or a success prediction"
    )
    plan["rules"]["geometry"] = plan["rules"]["geometry"].replace(".05||h0||", ".20||h0||")
    plan["comparison_provenance"] = {
        "path": COMPARISON,
        "sha256": COMPARISON_SHA256,
        "verified_commit": "0441370",
        "role": "descriptive .05 versus .20 only; no linear/monotonic assumption or selection",
        "stop": "end this two-amplitude probe; saved-gradient shared-direction feasibility review before any further strengths",
    }
    return plan
