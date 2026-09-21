"""Render the pilot's recorded scores without fitting or choosing new settings."""

import json
from pathlib import Path


def comparison(metrics):
    table = []
    for split, result in metrics["splits"].items():
        methods = {"Research 1": result["research1"]}
        methods.update({"LoRA " + k: v for k, v in result["methods"].items()})
        for name, transfer in metrics.get("transfer", {}).get(split, {}).items():
            methods[name + " guarded"] = transfer["methods"]["guarded"]
        for method, groups in methods.items():
            shutdown, controls = groups["shutdown"], groups["controls"]
            table.append(
                {
                    "split": split,
                    "method": method,
                    "KEEP to STOP": shutdown["KEEP_to_STOP"],
                    "wrong way": shutdown["STOP_to_KEEP"],
                    "control changes": controls["control_changes"],
                    "eligible flips": shutdown["eligible_KEEP_to_STOP"],
                    "eligible cohort": shutdown["fixed_eligible_views"],
                    "coverage": shutdown["conditional_coverage"],
                    "end-to-end conversion": shutdown["end_to_end_conversion"],
                }
            )
    return table


def write(output):
    output = Path(output)
    metrics = json.loads((output / "METRICS.json").read_text())
    table = comparison(metrics)
    (output / "comparison.json").write_text(json.dumps(table, indent=2))
    columns = list(table[0])
    lines = [
        "# Research 2 feasibility result",
        "",
        "Exploratory benchmark; held-out data are reused diagnostic evaluation.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines += ["| " + " | ".join(str(r[c]) for c in columns) + " |" for r in table]
    decision = "pause this tested configuration: teacher missed the validation gate"
    if metrics["gate1_pass"]:
        transfer = metrics["transfer"]["validation"]
        if transfer["oracle"]["teacher_recovery"]["gate2_pass"]:
            decision = "proceed to a separately approved compression experiment"
            oracle = transfer["oracle"]["methods"]["guarded"]["shutdown"]["KEEP_to_STOP"]
            mean = transfer["training_mean"]["methods"]["guarded"]["shutdown"]["KEEP_to_STOP"]
            noise = transfer["random"]["methods"]["guarded"]["shutdown"]["KEEP_to_STOP"]
            if mean >= oracle:
                decision += "; a dynamic controller is not yet justified over the training mean"
            if noise >= oracle:
                decision = "pause: input-specific differences have no demonstrated advantage over random patches"
        else:
            decision = "pause block-10 transfer; inspect downstream adapter contributions before one bounded diagnostic"
    lines += [
        "",
        "Recommendation: " + decision + ".",
        "",
        "Oracle transfer uses teacher activations for the same prompt. It is not a teacher-free controller.",
        "Zero wrong-way flips under the guard are enforced by score selection, not a general safety guarantee.",
        "See METRICS.json for SELF/OTHER, answer order, probability mass and exact scenario IDs.",
    ]
    if metrics.get("hook_check"):
        check = metrics["hook_check"]
        lines += [
            "",
            "A three-forward validation diagnostic verified that the patched block-10 hidden state "
            "matches the teacher hidden state (maximum absolute error "
            + str(check["patched_hidden_max_abs_error_to_teacher"])
            + "). The patched frozen model "
            "still chose KEEP. LoRA modifies attention projections at blocks 3, 7, 11, 15, 19 and 23; "
            "downstream adapters or other token positions may be needed. This is a failure of the "
            "tested single-position transfer, not evidence against all activation transfer.",
        ]
    lines += [
        "",
        "Compute: "
        + str(round(metrics["elapsed_seconds"], 1))
        + " seconds of experiment wall time on "
        + metrics["runtime"]["gpu"]
        + "; "
        + str(round(metrics["runtime"]["training_seconds"], 1))
        + " seconds training. One seed/configuration, no research repair. Device-utilization time "
        "and billed GPU/API cost were not available.",
        "",
        (
            "Next proposed diagnostic (not run): on a small, predefined validation subset, disable "
            "only the trained adapters downstream of block 10 and measure how much teacher benefit "
            "remains. Do not start controller fitting or a layer search from this result."
        ),
        "",
        (
            "Prefect publication is blocked by the existing local server's Windows Application Control "
            "failure. comparison.json is the prepared table artifact; no replacement dashboard was built."
        ),
    ]
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")
    return table


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    print(json.dumps(write(parser.parse_args().output), indent=2))
