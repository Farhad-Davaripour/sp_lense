"""Locate saved study inputs independently of installed package locations."""

import os
from pathlib import Path


def repository_root(explicit=None):
    requested = explicit or os.environ.get("SP_LENSE_REPO")
    if requested:
        root = Path(requested).expanduser().resolve()
        if not (root / "reproduce/inventory.json").is_file():
            raise ValueError(f"Not a study checkout: {root}; reproduce/inventory.json is missing")
        return root
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for root in (start, *start.parents):
            if (root / "reproduce/inventory.json").is_file():
                return root.resolve()
    raise ValueError("Run inside a study checkout or supply --repo /path/to/sp_lense")


ROOT = repository_root()
INPUTS = ROOT / "reproduce"
ARTIFACTS = INPUTS / "artifacts"
PAPER = ROOT / "paper"
BASELINE = ROOT / "study/baseline_scores"
POLICY = ROOT / "study/guarded_steering"
