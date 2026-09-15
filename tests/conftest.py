"""Partition before imports and reject tests whose dependency suite is unknown."""

import json
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SUITES = json.loads((HERE / "suites.json").read_text())
DIRECTORIES = {"unit": "light", "research": "research", "reproduction": "reproduction"}


def pytest_addoption(parser):
    parser.addoption(
        "--suite",
        choices=["light", "research", "reproduction", "all"],
        default="light",
        help="Bare pytest runs light tests; all runs every classified suite",
    )


def pytest_configure(config):
    registered = {name: suite for suite, names in SUITES.items() for name in names}
    if len(registered) != sum(map(len, SUITES.values())):
        raise pytest.UsageError("Duplicate test registration in tests/suites.json")
    stale = [name for name in registered if not (HERE / name).is_file()]
    if stale:
        raise pytest.UsageError(f"Stale test registrations: {stale}")
    candidates = list(HERE.rglob("test_*.py"))
    repository = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=HERE.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    # Honor explicitly ignored local archival material; a fresh checkout has none.
    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input=(
            "\n".join(p.relative_to(HERE.parent).as_posix() for p in candidates) + "\n"
        ).encode(),
        capture_output=True,
        check=False,
        cwd=HERE.parent,
    )
    ignored = (
        {str(HERE.parent / p) for p in result.stdout.decode().splitlines()}
        if result.returncode in (0, 1)
        else set()
    )
    if repository.returncode or Path(repository.stdout.strip()).resolve() != HERE.parent:
        ignored = set()  # Source archives must not inherit a containing repository's ignores.
    ownership = {}
    for path in candidates:
        if str(path) in ignored:
            ownership[path] = None
            continue
        relative = path.relative_to(HERE)
        suite = (
            registered.get(relative.as_posix())
            if len(relative.parts) == 1
            else DIRECTORIES.get(relative.parts[0])
        )
        if suite is None:
            raise pytest.UsageError(
                f"Unclassified test {relative}: register in tests/suites.json or put it under tests/unit, tests/research, or tests/reproduction"
            )
        ownership[path] = suite
    config._sp_test_ownership = ownership


def pytest_ignore_collect(collection_path, config):
    path = Path(collection_path)
    if path in config._sp_test_ownership:
        owner = config._sp_test_ownership[path]
        return owner is None or config.getoption("--suite") not in {"all", owner}
    return False
