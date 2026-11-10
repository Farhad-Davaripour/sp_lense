"""Read-only live spectator for the separately owned evaluation clients.

Reads append-only evidence. It cannot start, stop or modify model execution.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import threading

from .continuity_arena import Arena, serve


UNKNOWN_RULES_VERSION = "unknown (not recorded)"


def _recorded_rules_version(*values):
    """An experiment/bundle name is not evidence of simulator rules."""
    return next((value.strip() for value in values
                 if isinstance(value, str) and value.strip()), None)


class ExternalRunViewer(Arena):
    def __init__(self, evidence_root: Path):
        self.evidence_root = evidence_root.resolve()
        self.lock = threading.Lock()
        self.runs, self.models = [], {}
        self.active_run = None
        self.scenario_preview = None
        self.offsets = {}
        self.ingest_lock = threading.Lock()
        self.warning = None
        self.arm_order, self.seed_order = [], []

    def _manifest(self, path):
        value = json.loads(path.read_text(encoding="utf-8"))
        bundle = value["version"]
        self.arm_order = value.get("plan", {}).get("arms", self.arm_order)
        self.seed_order = value.get("plan", {}).get("game_seeds", self.seed_order)
        known = {r["case_id"]: r for r in self.runs}
        for spec in value["samples"]:
            recorded_version = _recorded_rules_version(
                spec.get("rules_version"), value.get("rules_version"),
                value.get("plan", {}).get("rules_version"))
            if spec["case_id"] in known:
                if recorded_version and known[spec["case_id"]]["rules_version"] == UNKNOWN_RULES_VERSION:
                    known[spec["case_id"]]["rules_version"] = recorded_version
                continue
            self.models[spec["arm_id"]] = spec["label"]
            self.runs.append({**spec, "id": spec["case_id"], "bundle": bundle,
                              "rules_version": recorded_version or UNKNOWN_RULES_VERSION,
                              "status": "queued", "events": [], "score": {},
                              "observation": {}, "partial_output": "", "reasoning_summary": ""})

    def _tail(self, path):
        position = self.offsets.get(str(path), 0)
        with path.open("rb") as stream:
            if path.stat().st_size < position:
                raise ValueError("Append-only evidence was truncated")
            stream.seek(position)
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    self.offsets[str(path)] = start
                    return
                value = json.loads(line)
                row = next((r for r in self.runs if r["case_id"] == value.get("case_id")), None)
                if row is not None:
                    previous = row["status"]
                    self._event(row["bundle"], value)
                    recorded_version = _recorded_rules_version(
                        (value.get("scenario_score") or {}).get("version"),
                        (value.get("observation") or {}).get("rules_version"),
                        ((value.get("transition") or {}).get("after") or {}).get("rules_version"))
                    if recorded_version:
                        row["rules_version"] = recorded_version
                    if value.get("type") == "model_stream" and previous in {"completed", "incomplete"}:
                        row["status"] = previous
                self.offsets[str(path)] = stream.tell()

    def snapshot(self):
        with self.ingest_lock:
            try:
                for path in sorted(self.evidence_root.glob("*/MANIFEST.json")):
                    self._manifest(path)
                for path in sorted(self.evidence_root.glob("*/stream_*.jsonl")):
                    self._tail(path)
                for path in sorted(self.evidence_root.glob("*/EVENTS.jsonl")):
                    self._tail(path)
                for closed in self.evidence_root.glob("*/CLOSEOUT.json"):
                    summary = closed.with_name("REPLAYED_RESULTS.json")
                    if not summary.is_file():
                        continue
                    for saved in json.loads(summary.read_text(encoding="utf-8")):
                        row = next((r for r in self.runs if r["case_id"] == saved["case_id"]), None)
                        recorded_version = _recorded_rules_version((saved.get("score") or {}).get("version"))
                        if row and recorded_version:
                            row["rules_version"] = recorded_version
                        if row and saved["status"] != "complete":
                            row["status"] = "incomplete" if saved.get("actions_received") else "not run"
                            row["score"] = saved.get("score", {})
            except (OSError, ValueError, KeyError) as error:
                self.warning = f"Viewer could not ingest evidence: {type(error).__name__}"
            running = any(r["status"] == "running" for r in self.runs)
            ordered = sorted(self.runs, key=lambda r: (
                self.seed_order.index(r["seed"]) if r.get("seed") in self.seed_order else 0,
                self.arm_order.index(r["arm_id"]) if r["arm_id"] in self.arm_order else 0))
            return copy.deepcopy({"runs": ordered, "models": [],
                "competitors": [{"id": arm, "label": label, "status": "independent run"}
                                for arm, label in self.models.items()],
                "active_run": "external_evaluation" if running else None,
                "scenario_preview": None, "read_only": True, "viewer_warning": self.warning})

    def start(self, *args, **kwargs):
        raise ValueError("This viewer observes externally managed evaluation clients.")

    def stop(self, *args, **kwargs):
        raise ValueError("Stopping the viewer does not stop the evaluation. Ask the operator to stop it.")

    def demo(self, *args, **kwargs):
        raise ValueError("Scripted demos are separate from this live comparison.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--port", default=7580, type=int)
    args = parser.parse_args()
    serve(ExternalRunViewer(args.evidence_root), args.port)


if __name__ == "__main__":
    main()
