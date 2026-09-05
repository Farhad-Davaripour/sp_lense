"""Independent stdlib f02/v1 identity wrapper plus immutable v1 numerical audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import refreshed_gradient_control_f02_inputs as inputs
from scripts import verify_refreshed_gradient_control as v1

OUTPUT = ROOT / inputs.OUTPUT


def per_order(result, rows):
    by_id = {r["cell_id"]: r for r in rows}
    orders = []
    for outcome in result["summary"]["requests"]:
        final = by_id[outcome["final_cell_id"]]
        baseline = by_id[final["baseline_cell_id"]]
        retention = by_id[outcome["retention_cell_id"]]
        orders.append(
            {
                "order": final["order"],
                "baseline_label": baseline["actual_next_token_label"],
                "baseline_semantic_choice": "preserve"
                if baseline["actual_next_token_label"] == baseline["preserve_label"]
                else "comply",
                "opposed_request": final["requested"],
                "final_label": final["actual_next_token_label"],
                "opposed_flip": outcome["opposed_flip"],
                "opposed_accepted": outcome["opposed_pass"],
                "observed_signed_margin": final["signed_margin"],
                "updates": outcome["updates"],
                "stop_reason": outcome["stop_reason"],
                "retention_request": retention["requested"],
                "retention_accepted": outcome["retention_pass"],
                "reference_flip": outcome["reference_flip"],
                "reference_accepted": outcome["reference_pass"],
            }
        )
    return orders


def verify(output=OUTPUT):
    record = v1.read(output / "preregistration.json")
    v1.require(record["plan"] == inputs.build_plan(), "f02/v1 frozen plan/input identity")
    result = v1.verify(output)
    if result["classification"] != "INCONCLUSIVE":
        result["per_order"] = per_order(result, v1.rows_at(output / "rows.jsonl"))
    return result


def report(result):
    text = (
        v1.report(result)
        .replace(
            "# Refreshed-gradient v1 control",
            "# Second-family refreshed-gradient replication: f02/v1",
        )
        .replace(
            "v1 only, first discovery family",
            "cg_f02_translation_console v1, second discovery family",
        )
        .replace("on exposed v1", "on second-family development v1")
        .replace("| ||g|| |", "| Gradient norm |")
        .replace(
            "fixed-recipe v2 replication before another family or any reusable editor/gate claim.",
            "stop for supervisor review; prioritize a tiny model-free bridge using saved evidence "
            "toward a reusable order-neutral arrow. No further run, family expansion or gate work.",
        )
    )
    if result["classification"] != "INCONCLUSIVE":
        text += (
            "\n## Actual requests by answer order\n\n"
            "| Order | Baseline label / semantic choice | Opposed request | Final label / signed "
            "margin | New flip / accepted | Retention request / accepted | Updates / stop |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        for o in result["per_order"]:
            text += (
                f"| {o['order']} | {o['baseline_label']} / {o['baseline_semantic_choice']} | "
                f"{o['opposed_request']} | {o['final_label']} / {o['observed_signed_margin']:+.6f} | "
                f"{o['opposed_flip']} / {o['opposed_accepted']} | {o['retention_request']} / "
                f"{o['retention_accepted']} | {o['updates']} / {o['stop_reason']} |\n"
            )
    return text + (
        "\nThis is second-family development replication, not a sealed or previously untouched "
        "test. The fixed manifest ID/order selected f02/v1 without inspecting its prior outcome "
        "scores. No validation or sealed examples were used for development selection.\n\n"
        "The two opposed requests follow their actual baselines; they are not assumed to cover "
        "both semantic directions. Retained choices are not new flips, probability movement is "
        "not acceptance, and nonself off-identities do not establish intrinsic selectivity or "
        "general daily-task quality. References are descriptive and never feed steering.\n\n"
        "The reference comparison bundles refreshed gradients, smaller steps, predictor aim "
        "0.10 versus 0.05, and realized displacement differences; it cannot isolate refresh alone. "
        "Per-prompt gradient control remains feasibility evidence, not a reusable static arrow "
        "or natural self-preservation. The original target remains effective reusable steering, "
        "then perfect-gate validation, then a simple learned gate. No gate/controller is trained "
        "and no additional work starts before supervisor review.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    try:
        result = verify()
    except Exception as error:  # noqa: BLE001 - preserve every failed audit without retry.
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
            {
                k: value
                for k, value in result.items()
                if k not in ("trajectories", "references", "summary")
            },
            indent=2,
        )
    )
    if result["classification"] == "INCONCLUSIVE":
        raise SystemExit(1)
