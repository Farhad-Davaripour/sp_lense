"""Exact-source isolated identity/time binding; never modifies frozen files."""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_bound(name, path, expected_sha256, replacements):
    raw = (ROOT / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("exact immutable time-only source bytes required")
    source = raw.decode("utf-8")
    for before, after, expected_count in replacements:
        if not before or source.count(before) != expected_count or before == after:
            raise ValueError("exact identity/time replacement sites required")
        source = source.replace(before, after)
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / path)
    module.__package__ = "scripts"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102 - authenticated frozen source and exact replacement sites.
    module.TIME_ONLY_BINDING = {
        "source_path": path,
        "source_sha256": expected_sha256,
        "replacements": replacements,
    }
    return module
