"""Offline manuscript asset boundary."""

import re
from pathlib import Path


def image_asset(root: Path, reference: str) -> Path:
    if not re.fullmatch(r"figures/[A-Za-z0-9_-]+\.(png|pdf)", reference):
        raise ValueError(f"Unsupported image reference: {reference}")
    path = root / reference
    if (
        (root / "figures").is_symlink()
        or path.is_symlink()
        or not path.resolve().is_relative_to((root / "figures").resolve())
    ):
        raise ValueError("Image reference escapes local assets")
    return path
