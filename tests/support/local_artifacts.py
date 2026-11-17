"""Explicit prerequisites for historical tests bound to local-only evidence.

These checks do not replace or weaken the exact SHA assertions in the tests.
If an artifact is present but changed, the test still fails normally.
"""
from pathlib import Path
from unittest import SkipTest


def require_local_files(*paths):
    missing = [Path(path).name for path in paths if not Path(path).is_file()]
    if missing:
        raise SkipTest("Local-only exact-source integration requires: " + ", ".join(missing))
