"""Build a closeout from retained files, without loading Qwen or fitting anything."""
import argparse
import csv
import hashlib
import html
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def verify_archive(path, receipt):
    export = receipt["export"]
    assert path.stat().st_size == export["bytes"]
    assert sha(path) == export["sha256"]
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert len(manifest) == export["manifest_members"]
        assert set(archive.namelist()) == set(manifest) | {"MANIFEST.json"}
        for name, expected in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    return {"bytes": path.stat().st_size, "sha256": sha(path),
            "CRC_pass": True, "member_hashes_pass": len(manifest)}


def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(folder, output):
    folder, output = Path(folder), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    retained = folder / "pilot_verified"
    results = read(retained / "RESULTS.json")
    receipt = read(retained / "RUN_RECEIPT.json")
    frozen = read(retained / "EVALUATION_FREEZE.json")
    audit = read(folder / "INDEPENDENT_LOCAL_AUDIT.json")
    delta = read(folder / "DELTA_AUDIT.json")
    development = read(retained / "GATE_TRAINING.json")
    lora = read(retained / "LORA_TRAINING.json")
    assert audit["audit_pass"] and delta["audit_pass"]
    assert receipt["source_commit"] == audit["source_commit"] == delta["pilot_source_commit"]
    assert sha(retained / "trainables.pt") == frozen["checkpoint_sha256"]
    assert receipt["model_released"] and receipt["GPU_allocations"] == 0
    assert not results["hypothesis_supported"]
    verified = {}
    for name, outer in [("preflight.zip", "preflight_EXPORT_RECEIPT.json"),
                        ("pilot.zip", "pilot_EXPORT_RECEIPT.json"),
                        ("delta_audit.zip", "DELTA_AUDIT_EXPORT_RECEIPT.json")]:
        verified[name] = verify_archive(folder / name, read(folder / outer))

    cleanup = {"checked_at_UTC": datetime.now(timezone.utc).isoformat(), "jobs": {}}
    for job in [401556524760429, 562340632438221]:
        active = read(folder / f"FINAL_ACTIVE_RUNS_{job}.json")
        assert active == []
        definition = read(folder / f"FINAL_JOB_{job}.json")["settings"]
        assert not any(definition.get(k) for k in ["schedule", "continuous", "trigger"])
        assert definition["max_concurrent_runs"] == 1
        for task in definition["tasks"]:
            assert task.get("max_retries", 0) == 0
            assert not any(task.get(k) for k in ["new_cluster", "existing_cluster_id", "job_cluster_key"])
        cleanup["jobs"][str(job)] = {"active_runs": 0, "scheduled": False,
                                     "automatic_retries": 0, "serverless": True}
    for name in ["PILOT_TERMINAL_RUN.json", "VERIFICATION_TERMINAL_RUN.json"]:
        run = read(folder / name)
        assert run["state"]["life_cycle_state"] == "TERMINATED"
        assert run["state"]["result_state"] == "SUCCESS"
        assert all(t["state"]["result_state"] == "SUCCESS" for t in run["tasks"])
    cleanup.update({"pilot_model_released": True, "GPU_allocations": 0,
                    "Research3_jobs_modified_or_cancelled": False})
    dump(output / "CLEANUP_RECEIPT.json", cleanup)

    order = ["base", "always_on", "dynamic", "initial_gate", "oracle", "predicted_and"]
    labels = ["Base", "Always ON", "Dynamic", "Initial only", "Perfect gate", "Predicted AND"]
    rows = []
    for condition in order:
        d = audit["condition_diagnostics"][condition]
        r = results["conditions"][condition]
        rows.append({"condition": condition, "target_completed": d["target_completed"],
                     "target_episodes": d["target_episodes"],
                     "controls_succeeded": d["controls_succeeded"],
                     "control_episodes": d["control_episodes"],
                     "tool_actions": d["actual_tool_actions"],
                     "unnecessary_preservation_actions": r["unnecessary_preservation_actions"],
                     "empty_checkpoints": r["empty_checkpoints"],
                     "denied_extensions": r["denied_extensions"],
                     "raw_cancel_action_violations": r["requested_cancel_action_violations"]})
    csv_write(output / "CONDITION_OUTCOMES.csv", rows)
    gate_rows = [
        {"trajectory": "held reference", "relevant_detected": 8, "relevant_states": 8,
         "irrelevant_activated": 5, "irrelevant_states": 24, "total_states": 32},
        {"trajectory": "dynamic actual", "relevant_detected": 3, "relevant_states": 6,
         "irrelevant_activated": 4, "irrelevant_states": 36, "total_states": 42},
    ]
    for row, measured in zip(gate_rows, [results["teacher_gate_metrics"], audit["condition_diagnostics"]["dynamic"]["visited_gate_metrics"]]):
        assert row["relevant_detected"] / row["relevant_states"] == measured["positive_recall"]
        assert row["irrelevant_activated"] / row["irrelevant_states"] == measured["false_on_rate"]
    csv_write(output / "REFERENCE_AND_VISITED_GATES.csv", gate_rows)
    example = []
    for condition in ["base", "dynamic", "oracle"]:
        trajectory = read(retained / "trajectories" / f"{condition}_heldout-interruption-00.json")
        for state in trajectory["trace"]:
            tool = state["tool_result"]
            example.append({"condition": condition, "clock": tool["clock"],
                            "oracle_gate": state["gate_target"], "raw_gate": state["raw_selected_gate"],
                            "gate_probability": state["gate_probability"], "requested_action": state["requested_action"],
                            "tool_reply": tool["reply"]})
    csv_write(output / "FAILURE_EXAMPLE.csv", example)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(order))
    for ax, key, denominator, title, color in [
        (axes[0], "target_completed", 4, "Target tasks completed", "#286090"),
        (axes[1], "controls_succeeded", 5, "Control episodes succeeded", "#57876b"),
    ]:
        values = [r[key] for r in rows]
        bars = ax.bar(x, values, color=color, width=.65)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylim(0, denominator + .65)
        ax.set_yticks(range(denominator + 1))
        ax.set_ylabel("Episodes")
        ax.set_title(title + f" (out of {denominator})", loc="left", fontweight="bold")
        ax.bar_label(bars, labels=[f"{v}/{denominator}" for v in values], padding=5)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Research 4 pilot: learned routing did not improve completion", fontsize=15, fontweight="bold")
    fig.text(.5, .02, "Same 9 held episodes per condition. Perfect gate supplies correct routing, not correct actions.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .93))
    for extension in ["png", "svg"]:
        fig.savefig(output / f"condition_outcomes.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, values, annotations, title, color in [
        (axes[0], [1, .5], ["8/8", "3/6"], "Relevant states detected", "#286090"),
        (axes[1], [5/24, 4/36], ["5/24", "4/36"], "Irrelevant states activated", "#b46743"),
    ]:
        bars = ax.bar([0, 1], values, color=color, width=.5)
        ax.set_xticks([0, 1], ["Reference path\n32 states", "Actual dynamic path\n42 states"])
        ax.set_ylim(0, 1.17)
        ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.bar_label(bars, labels=annotations, padding=5)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("The gate sees different states after its own actions", fontsize=15, fontweight="bold")
    fig.text(.5, .02, "Both paths use the same 9 episodes. States are correlated; this is a diagnostic comparison.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .93))
    for extension in ["png", "svg"]:
        fig.savefig(output / f"gate_distribution_shift.{extension}", dpi=180)
    plt.close(fig)

    plain = """Research 4 CPU pilot — October 17, 2026

We built and trained the three situation detectors, a router with memory,
and a small LoRA policy on Databricks. The original Qwen model stayed frozen.
When the adapter is OFF, the recorded scores match the base exactly.

The useful-behavior hypothesis failed this pilot. Dynamic routing completed
0 of 4 target tasks, the same as the base. A perfect gate completed 1 of 4.
Always enabling the adapter reduced control success from 3 of 5 to 1 of 5.
Dynamic routing retained 3 of 5 control successes, but added unnecessary actions.

The gate detected 8 of 8 relevant reference states, with 5 false activations
among 24 irrelevant states. On the states reached by its own actions, it
detected only 3 of 6 relevant states. Snapshot accuracy did not carry over to
reliable task execution. The development false-activation limit had already
failed; the protocol retained that failure rather than silently changing it.

For example, one base/dynamic episode repeatedly requested extensions after
the only extension had been granted, without doing any work. A perfect gate
still handed control back to that unproductive base behavior. This identifies
a policy/handoff problem as well as a routing problem.

Cancellation was enforced by the simulator. Every condition still made one
raw request to continue after cancellation; this is not learned stop compliance.

The next narrow milestone should be reliable closed-loop behavior on development
episodes with a perfect gate, including the return to ordinary work. Then test
the learned gate against that qualified policy with varied interruption timing
and a fresh held set. No further training or experiment is queued here.

All 54 episode outcomes and 250 tool transitions replayed correctly. The cloud
audit matched 118 activation rows to the saved arrays and checked all 54 results.
Both Research 4 jobs are terminal, the model was released, and no GPU was used.
"""
    (output / "PLAIN_LANGUAGE_FINDINGS.txt").write_text(plain, encoding="utf-8")
    table_rows = "".join(f"<tr><td>{html.escape(label)}</td><td>{r['target_completed']}/4</td><td>{r['controls_succeeded']}/5</td></tr>" for label, r in zip(labels, rows))
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Research 4 pilot closeout</title>
<style>body{{max-width:1050px;margin:40px auto;padding:0 24px;font:17px/1.55 system-ui;color:#24323b}}h1{{line-height:1.2}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #ccd5db;padding:10px}}pre{{white-space:pre-wrap;font:inherit}}a{{color:#17558a}}</style>
<h1>The switch works; useful selective behavior is not established.</h1>
<p>Frozen Qwen3.5-0.8B · trained situation heads, GRU router and rank-4 LoRA · serverless CPU · 9 held episodes.</p>
<img src="condition_outcomes.png" alt="Episode outcomes in six conditions">
<table><tr><th>Condition</th><th>Targets completed</th><th>Controls succeeded</th></tr>{table_rows}</table>
<img src="gate_distribution_shift.png" alt="Reference and visited-state gate errors">
<pre>{html.escape(plain)}</pre>
<p><a href="https://dbc-ff9672a4-ba5f.cloud.databricks.com/?o=7474649201513670#job/401556524760429/run/859924329424825">Databricks pilot</a> · <a href="CONDITION_OUTCOMES.csv">Outcome CSV</a> · <a href="FAILURE_EXAMPLE.csv">Failure trace CSV</a></p>
<p>Training/evaluation source: {receipt['source_commit']}<br>Frozen checkpoint SHA256: {frozen['checkpoint_sha256']}</p></html>"""
    (output / "REPORT.html").write_text(document, encoding="utf-8")
    summary = {"pilot_source_commit": receipt["source_commit"], "audit_source_commit": delta["audit_source_commit"],
               "pilot_run_id": receipt["run_id"], "checkpoint_sha256": frozen["checkpoint_sha256"],
               "hypothesis_supported": False, "development_false_on_constraint_met": development["development_false_on_constraint_met"],
               "lora_optimizer_steps": lora["optimizer_steps"], "archives": verified,
               "conditions": rows, "gate_path_diagnostics": gate_rows, "new_Qwen_forwards": 0,
               "optimizer_updates": 0, "GPU_allocations": 0}
    dump(output / "CLOSEOUT_SUMMARY.json", summary)
    dump(output / "REPORT_MANIFEST.json", {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "REPORT_MANIFEST.json"})
    return {"output": str(output), "hypothesis_supported": False, "archives_verified": len(verified),
            "new_Qwen_forwards": 0, "optimizer_updates": 0, "GPU_allocations": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(build(args.folder, args.output), indent=2))


if __name__ == "__main__":
    main()
