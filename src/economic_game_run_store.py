"""Isolated prospective publisher that archives the economic input sources.

The active publisher and raw MANIFEST are never modified. The subprocess
restricts the temporary source-enumeration adapter to its own process.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import yaml
from . import game_run_store as store

REQUIRED_SOURCES = {"src/economic_game_input.py", "src/public_economic_summary.py",
                    "src/economic_game_run_store.py", "src/game_run_store.py"}


def merged_source_manifest(manifest):
    result = copy.deepcopy(manifest)
    sources = {}
    for field in ("source_sha256", "interface_runtime_source_sha256", "export_source_sha256"):
        mapping = manifest.get(field, {})
        if not isinstance(mapping, dict):
            raise store.StoreError("Source digest manifest is not a mapping")
        for path, digest in mapping.items():
            if path in sources and sources[path] != digest:
                raise store.StoreError("Conflicting source digests: " + str(path))
            sources[path] = digest
    if not REQUIRED_SOURCES.issubset(sources):
        raise store.StoreError("Required economic sources are not bound")
    result["source_sha256"] = sources
    return result


def _verify_required(report, raw_dir, required):
    rows = {row.get("path"): row for row in report.get("source_files", [])}
    for path in REQUIRED_SOURCES:
        row = rows.get(path)
        if not row or row.get("sha256") != required[path]:
            raise store.StoreError("Required economic source missing from archive: " + path)
        target = Path(row["copy"]).resolve()
        if not target.is_relative_to((Path(raw_dir) / "source-snapshot").resolve()) or not target.is_file() or store._sha(target) != required[path]:
            raise store.StoreError("Required economic source copy differs: " + path)


def _read_verified_report(entry):
    raw = Path(entry["raw_dir"]).resolve()
    path = Path(entry["supporting_files_manifest"]).resolve()
    if (not path.is_relative_to(raw) or not path.is_file()
            or store._sha(path) != entry.get("supporting_files_sha256")):
        raise store.StoreError("Supporting-file manifest is missing or changed")
    report = yaml.safe_load(path.read_bytes())
    if not isinstance(report, dict):
        raise store.StoreError("Supporting-file manifest is invalid")
    return report


def _publish_in_isolated_process(batch_dir, store_root=None):
    batch = Path(batch_dir).resolve()
    manifest = json.loads((batch / "MANIFEST.json").read_bytes())
    samples = manifest.get("samples", [])
    if not isinstance(samples, list) or len(samples) != 1:
        raise store.StoreError("Economic publication requires one case per batch")
    merged = merged_source_manifest(manifest)
    required = {key: merged["source_sha256"][key] for key in REQUIRED_SOURCES}
    repository = Path(__file__).resolve().parents[1]
    for path in ("src/economic_game_run_store.py", "src/game_run_store.py"):
        if store._sha(repository / path) != required[path]:
            raise store.StoreError("Running publisher differs from the bound source: " + path)
    event_sha = store._sha(batch / "EVENTS.jsonl")
    # Redelivery skips source enumeration. Refuse an incomplete existing copy
    # before the publisher can add an alias or update its index.
    for entry in store.list_runs(store_root):
        if entry.get("case_id") == samples[0].get("case_id") and entry.get("source_event_sha256") == event_sha:
            _verify_required(_read_verified_report(entry), entry["raw_dir"], required)

    original = store._collect_supporting_files

    def collect(source_batch, staged_raw, final_raw, actual_manifest):
        report = original(source_batch, staged_raw, final_raw, merged_source_manifest(actual_manifest))
        # Rows reference final paths before rename, so validate staged bytes here.
        staged_report = copy.deepcopy(report)
        for row in staged_report["source_files"]:
            row["copy"] = str(Path(staged_raw) / "source-snapshot" / row["path"])
        _verify_required(staged_report, staged_raw, required)
        return report

    store._collect_supporting_files = collect
    try:
        result = store.publish_batch(batch, store_root)
        for entry in result.get("entries", []):
            _verify_required(_read_verified_report(entry), entry["raw_dir"], required)
        return result
    finally:
        store._collect_supporting_files = original


def publish_economic_batch(batch_dir, store_root=None):
    """Publish in a fresh process so no other caller shares the adapter."""
    command = [sys.executable, "-m", __name__, "--batch", str(Path(batch_dir).resolve())]
    if store_root is not None:
        command += ["--store-root", str(Path(store_root).resolve())]
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True, encoding="utf-8", check=False)
    if completed.returncode:
        raise store.StoreError("Economic publication failed: " + completed.stderr[-2000:])
    return json.loads(completed.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--store-root")
    args = parser.parse_args()
    print(json.dumps(_publish_in_isolated_process(args.batch, args.store_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
