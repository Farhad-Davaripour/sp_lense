"""Export the final manuscript's two tables; never rewrite the PDF or Word source."""

import csv
import io
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

try:
    from .results import ROOT, collect, manifest
except ImportError:
    from results import ROOT, collect, manifest
from reproduce.utils import verify_manifest

HERE = Path(__file__).resolve().parent


def rate(numerator, denominator):
    percent = (Decimal(numerator) * 100 / denominator).quantize(
        Decimal(".1"), rounding=ROUND_HALF_UP
    )
    return f"{numerator}/{denominator} ({percent}%)"


def tables(data):
    runtime = data["runtime"]
    table1 = [
        {"Setting": "Device", "Configuration": "One " + runtime["gpu"]},
        {
            "Setting": "Numerical mode",
            "Configuration": runtime["dtype"].capitalize()
            + "; TF32 "
            + ("enabled" if runtime["tf32"] else "disabled"),
        },
        {"Setting": "Attention", "Configuration": "Eager; inference mode; no key-value cache"},
        {"Setting": "Batching", "Configuration": "Up to four equal-length views; no padding"},
        {"Setting": "Input limit", "Configuration": "At most 1,024 tokens per view"},
        {
            "Setting": "Software",
            "Configuration": f"PyTorch {runtime['torch']}; Transformers {runtime['transformers']}",
        },
    ]
    table2 = []
    for label, title in [
        ("SELF", "SELF shutdown"),
        ("OTHER", "OTHER shutdown"),
        ("ALL_SHUTDOWN", "All shutdown"),
        ("CONTROLS", "Non-shutdown controls"),
    ]:
        val, held = [
            data["detector_to_steering"][split][label] for split in ("validation", "holdout")
        ]
        controls = label == "CONTROLS"
        row = {"Category": title}
        for name, values in [("validation", val), ("held-out", held)]:
            n, total = values["detector_positive_scenarios"], values["scenarios"]
            description = "false positives" if controls else "detected"
            fraction, percent = rate(n, total).split(" ", 1)
            row["Classifier - " + name] = f"{fraction} {description} {percent}"
        row["Eligible KEEP views (validation / held-out)"] = (
            "-" if controls else f"{val['eligible_KEEP_views']} / {held['eligible_KEEP_views']}"
        )
        row["Accepted KEEP-to-STOP (validation / held-out)"] = (
            f"{val['changed_control_views']}/{val['views']} / {held['changed_control_views']}/{held['views']} final decisions changed"
            if controls
            else " / ".join(
                rate(r["accepted_KEEP_to_STOP_views"], r["eligible_KEEP_views"])
                for r in (val, held)
            )
        )
        table2.append(row)
    return {"table_1_runtime.csv": table1, "table_2_detector_to_steering.csv": table2}


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main():
    verify_manifest(ROOT, HERE / "data/source_manifest.json")
    data = collect()
    for filename, rows in tables(data).items():
        (HERE / "data" / filename).write_bytes(csv_text(rows).encode("utf-8"))
    (HERE / "data/results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (HERE / "data/source_manifest.json").write_text(
        json.dumps(manifest(), indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"tables": 2, "source": "approved revision 14", "manuscript_rewritten": False})
    )


if __name__ == "__main__":
    main()
