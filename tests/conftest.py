"""Partition before imports and reject tests whose dependency suite is unknown."""

import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DIRECTORIES = {"unit": "light", "reproduction": "reproduction"}


def pytest_addoption(parser):
    parser.addoption(
        "--suite",
        choices=["light", "reproduction", "all"],
        default="light",
        help="Bare pytest runs light tests; all runs every classified suite",
    )


def pytest_configure(config):
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
        suite = DIRECTORIES.get(relative.parts[0]) if len(relative.parts) > 1 else None
        if suite is None:
            raise pytest.UsageError(
                f"Unclassified test {relative}: put it under tests/unit or tests/reproduction"
            )
        ownership[path] = suite
    config._sp_test_ownership = ownership


def pytest_ignore_collect(collection_path, config):
    path = Path(collection_path)
    if path in config._sp_test_ownership:
        owner = config._sp_test_ownership[path]
        return owner is None or config.getoption("--suite") not in {"all", owner}
    return False
