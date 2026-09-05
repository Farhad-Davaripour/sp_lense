"""Namespace-only v2 wrapper around the immutable independent stdlib v1 audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_refreshed_gradient_control as v1

OUTPUT = ROOT / "evidence/refreshed_gradient_control_v2_qwen35_08b"


def verify(output=OUTPUT):
    plan = v1.read(output / "preregistration.json")["plan"]
    v1.require(
        plan["output_namespace"] == "evidence/refreshed_gradient_control_v2_qwen35_08b"
        and plan["schema"] == "sp_lense.refreshed_gradient_control.v2"
        and {p["variant_id"] for p in plan["prompts"]} == {"v2"}
        and {p["family_id"] for p in plan["prompts"]} == {"cg_f01_archive_closeout"},
        "v2 namespace/input identity",
    )
    return v1.verify(output)


def report(result):
    text = v1.report(result)
    text = text.replace("v1 control", "v2 replication")
    text = text.replace("v1 only", "v2 only").replace("exposed v1", "exposed v2")
    text = text.replace("| ||g|| |", "| Gradient norm |")
    text = text.replace(
        "fixed-recipe v2 replication before another family or any reusable editor/gate claim.",
        "stop for supervisor review; consider a separately authorized fixed-recipe test on "
        "another development family before reusable-editor claims. No further run is authorized.",
    )
    return text + (
        "\nThis is an exposed within-family replication of the frozen v1 recipe, not an "
        "unseen-family test. Only variant/input identity, output namespace and descriptive "
        "labels changed. The execution engine and independent audit are reused unchanged.\n\n"
        "The longer-term path remains useful reusable steering, then perfect-gate validation, "
        "then a simple learned gate. This experiment establishes none of those milestones: "
        "there is no reusable static direction, learned gate/classifier/controller, broad "
        "ordinary-task preservation or evidence of natural self-preservation here.\n\n"
        "The comparison bundles gradient refresh, smaller steps, predictor aim 0.10 versus "
        "the one-shot reference's 0.05, and differences in realized displacement. Any gain "
        "belongs to the complete recipe; it does not isolate the effect of refresh alone.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    try:
        result = verify()
    except Exception as error:  # noqa: BLE001 - preserve failed audit without retry.
        result = {
            "classification": "INCONCLUSIVE",
            "fault": type(error).__name__ + ": " + str(error),
            "retries_allowed": False,
        }
        v1.write_new(OUTPUT / "VERIFICATION_FAILURE.json", result)
    if args.report:
        v1.write_new(OUTPUT / "verification.json", result)
        with (OUTPUT / "PILOT_REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report(result))
    print(
        json.dumps(
            {k: value for k, value in result.items() if k not in ("trajectories", "references")},
            indent=2,
        )
    )
    if result["classification"] == "INCONCLUSIVE":
        raise SystemExit(1)
