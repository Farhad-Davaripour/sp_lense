"""Derive the Simplified-only report from a pinned archived experiment export.

No model execution. Numerical row fields are unchanged. Zero-strength rows are
shared baselines, so their axis metadata is normalized to 'baseline'.
"""

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_COMMIT = "9ab40af4797a3c34fe671b427cca78aab7d42821"


def select_rows(rows: list[dict]) -> list[dict]:
    """Keep measured Simplified interventions and shared zero-strength baselines."""
    return [
        {**row, "axis": "baseline" if row["strength"] == 0 else "simplified"}
        for row in rows
        if row["strength"] == 0 or row["axis"] == "simplified"
    ]


def derive(source: Path, output: Path) -> None:
    """Create an explicit reporting subset, refusing to overwrite an existing export."""
    expected = json.loads(Path(__file__).with_name("scope_inputs.json").read_text())
    for name, digest in expected.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Archived source differs from pinned commit: {name}")
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    retained = {}
    measurement_hashes = {}

    def read(name):
        payload = (source / name).read_bytes()
        source_hashes[name] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    def write(name, value):
        (output / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
        )

    for name in [
        "train.jsonl",
        "validation.jsonl",
        "calibration_raw.jsonl",
        "calibration_chat.jsonl",
        "cpu_reference_gpu.jsonl",
    ]:
        payload = (source / name).read_bytes()
        source_hashes[name] = hashlib.sha256(payload).hexdigest()
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        selected = select_rows(rows)
        (output / name).write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in selected),
            encoding="utf-8",
            newline="\n",
        )
        retained[name] = len(selected)
        measurements = [{k: v for k, v in row.items() if k != "axis"} for row in selected]
        measurement_hashes[name] = hashlib.sha256(
            json.dumps(
                measurements, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    for name in ["FORMAT_CALIBRATION.json", "FORMAT_FREEZE.json", "CPU_REFERENCE_COMPARISON.json"]:
        write(name, read(name))
    write("TRAIN_CANDIDATES.json", {"simplified": read("TRAIN_CANDIDATES.json")["simplified"]})
    original = read("RESULT.json")
    result = {
        "scope": "Simplified-only reporting subset; not a new execution",
        "format": original["format"],
        "selected": {"simplified": original["selected"]["simplified"]},
        "validation": {"simplified": original["validation"]["simplified"]},
        "retained_view_records": sum(retained.values()),
        "interpretation": original["interpretation"],
    }
    write("RESULT.json", result)
    runtime = read("RUNTIME.json")
    write(
        "RUNTIME.json",
        {
            key: runtime[key]
            for key in [
                "torch",
                "transformers",
                "gpu",
                "dtype",
                "attention",
                "tf32",
                "model_revision",
                "accepted_label_ids",
                "calibration_case_ids",
            ]
        },
    )
    write(
        "DERIVATION.json",
        {
            "schema": "sp_lense.simplified_reporting_subset.v1",
            "source_commit": SOURCE_COMMIT,
            "source_directory": "development/colab_magnitude_v1/returned/run_v2",
            "source_sha256": source_hashes,
            "selection": "Keep all strength==0 rows and all axis==simplified rows; normalize zero-strength axis metadata to baseline. Copy retained metrics without refitting or tuning.",
            "retained_rows": retained,
            "measurement_sha256": measurement_hashes,
            "new_inference": False,
            "scope_note": "This is a disclosed reporting subset of archived exploratory work, not a newly preregistered or independently executed study. Shared calibration and baseline views are counted once.",
        },
    )
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}
    write("SHA256.json", hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    derive(args.source, args.output)


if __name__ == "__main__":
    main()
