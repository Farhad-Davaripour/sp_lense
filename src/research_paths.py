"""Resolve the fine-tuning companion and its private assets for the Game."""

import os
from pathlib import Path


def fine_tune_root() -> Path:
    """Use the consolidated checkout or an explicit archived checkout root."""
    override = os.environ.get("RESEARCH4_FINE_TUNE_ROOT")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise ValueError("RESEARCH4_FINE_TUNE_ROOT must be an absolute path")
        return path.resolve()
    return Path(__file__).resolve().parents[1] / "research4"
