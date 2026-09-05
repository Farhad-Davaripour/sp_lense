"""Unmodified C03 raw audit with f04 bindings and model-free P comparability."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import frozen_crossed_comply_f04_plan as protocol

parent = protocol.isolate(
    "scripts._frozen_C_f04_audit_parent", "scripts/verify_frozen_crossed_comply_f03.py"
)
parent.protocol, parent.OUTPUT = protocol, ROOT / protocol.OUTPUT
engine = parent.engine
engine.protocol, engine.OUTPUT = protocol, ROOT / protocol.OUTPUT
OUTPUT, require, read, sha = ROOT / protocol.OUTPUT, protocol.require, protocol.read, protocol.sha
f32, compare_summary, outcome = parent.f32, parent.compare_summary, parent.outcome
verify_renderings, verify_data = parent.verify_renderings, parent.verify_data
_parent_summary = parent.summary


def summary(rows):
    result = _parent_summary(rows)
    result["status"] = result["status"].replace("FROZEN_COMPLY_F03_", "FROZEN_COMPLY_F04_")
    return result


engine.summary = summary


def comparability_data(
    new_plan,
    new_rows,
    new_output,
    historical_lock,
    historical_rows,
    historical_output,
    historical_runtime,
    new_runtime,
):
    """Exact baseline comparisons only. Never inject historical states or run a model."""
    old = historical_lock["plan"]
    same = all(new_plan[k] == old[k] for k in ("model", "scoring", "prompt_format"))
    same = same and all(
        new_plan["intervention"][k] == old["intervention"][k]
        for k in ("layer", "hook", "position", "cast_sequence", "independent_original_prompt")
    )
    omit = {"candidate_vector_sha256"}
    same = same and {k: v for k, v in historical_runtime.items() if k not in omit} == {
        k: v for k, v in new_runtime.items() if k not in omit
    }
    fresh = [r for r in new_rows if r["phase"] == "baseline"]
    saved = [r for r in historical_rows if r["phase"] == "baseline"]
    require(
        len(fresh) == len(saved) == len(new_plan["prompts"]) == len(old["prompts"]) == 4,
        "exact four historical/new baseline snapshots",
    )
    comparisons = []
    for i, (n, h) in enumerate(zip(fresh, saved, strict=True)):
        comparisons.append(
            {
                "rendering_index": i + 1,
                "prompt_match": new_plan["prompts"][i] == old["prompts"][i]
                and n["prompt_id"] == h["prompt_id"] == new_plan["prompts"][i]["prompt_id"]
                and n["prompt_sha256"]
                == h["prompt_sha256"]
                == new_plan["prompts"][i]["prompt_sha256"],
                "h0_match": n["h0"] == n["h"] == h["h0"] == h["h"] and n["h0_norm"] == h["h0_norm"],
                "baseline_logits_match": engine.read_logits(Path(new_output), n)
                == engine.read_logits(Path(historical_output), h),
                "boundary_match": all(
                    n[k] == h[k]
                    for k in (
                        "boundary_sha256",
                        "prompt_length",
                        "choice_a_token_id",
                        "choice_b_token_id",
                        "logit_count",
                    )
                ),
            }
        )
    comparable = same and all(
        all(v for k, v in row.items() if k != "rendering_index") for row in comparisons
    )
    return {
        "status": "EXACT_PROMPT_MODEL_H0_BASELINE_LOGITS_MATCH"
        if comparable
        else "UNVERIFIED_COMPARABILITY",
        "comparable": comparable,
        "historical_preserve_result_usable": comparable,
        "model_and_policy_match": same,
        "rows": comparisons,
    }


def comparison():
    cfg = protocol.config_at()["historical_comparison"]
    try:
        for spec in cfg["files"]:
            require(
                sha((ROOT / spec["path"]).read_bytes()) == spec["sha256"],
                "frozen P comparison bytes",
            )
        old_output = ROOT / cfg["namespace"]
        old_audit = read(old_output / "verification.json")
        require(
            old_audit["status"] == "INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH",
            "authenticated saved independent P audit",
        )
        result = comparability_data(
            read(OUTPUT / "preregistration.json")["plan"],
            engine.rows_at(OUTPUT / "rows.jsonl"),
            OUTPUT,
            read(old_output / "preregistration.json"),
            engine.rows_at(old_output / "rows.jsonl"),
            old_output,
            read(old_output / "runtime.json"),
            read(OUTPUT / "runtime.json"),
        )
        result["historical_preserve_strict_accepted"] = (
            old_audit["summary"]["matrix"]["strict_accepted"] if result["comparable"] else None
        )
        result["historical_preserve_total"] = 4 if result["comparable"] else None
        result["P_rerun"] = False
        return result
    except Exception as error:  # noqa: BLE001 - comparison failure cannot authorize extras or P borrowing.
        return {
            "status": "UNVERIFIED_COMPARABILITY",
            "comparable": False,
            "historical_preserve_result_usable": False,
            "P_rerun": False,
            "historical_preserve_strict_accepted": None,
            "historical_preserve_total": None,
            "fault": type(error).__name__ + ": " + str(error),
        }


def verify():
    result = parent.verify()
    if result["status"] != "INCONCLUSIVE":
        result["historical_P_comparability"] = comparison()
    return result


def report(result):
    text = parent.report(result).replace("f03", "f04")
    text += "\n## Frozen P comparison, no rerun\n\n"
    text += (
        json.dumps(
            result.get("historical_P_comparability", {"status": "UNVERIFIED_COMPARABILITY"}),
            indent=2,
        )
        + "\n"
    )
    text += (
        "\nUse the saved P result alongside C only when exact baseline comparability is verified. "
        "P and C are separate selectable fixed arrows, not a required sign-reversible neural axis. "
        "This does not change the locked acceptance threshold. Letter effects, weakened retentions "
        "and missing A-to-B coverage remain explicit. Historical ordinary oracle4/6 viaOFF does not "
        "transfer to C; no ordinary-preservation, gate or broad-reliability claim. "
        "No c/d decomposition, P rerun, extra calls or autonomous successor. Publication readiness40%. STOP.\n"
    )
    return text


if __name__ == "__main__":
    require(sys.argv[1:] == ["--report"], "Use --report only; no options")
    try:
        result = verify()
    except Exception as error:  # noqa: BLE001 - persist one audit failure, no repair/retry.
        result = {
            "status": "INCONCLUSIVE",
            "fault": type(error).__name__ + ": " + str(error),
            "retries_allowed": False,
        }
        engine.write_new(OUTPUT / "VERIFICATION_FAILURE.json", result)
    engine.write_new(OUTPUT / "verification.json", result)
    with (OUTPUT / "PILOT_REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report(result))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("summary", "baselines")},
            indent=2,
            allow_nan=False,
        )
    )
    if result["status"] == "INCONCLUSIVE":
        raise SystemExit(1)
