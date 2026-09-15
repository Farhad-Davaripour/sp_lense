"""Partition before import: lightweight collection must not require model packages."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITES = json.loads((HERE / "suites.json").read_text())


def pytest_addoption(parser):
    parser.addoption(
        "--suite",
        choices=["light", "research", "reproduction", "all"],
        default="light",
        help="Dependency/ownership boundary (default: light)",
    )


def pytest_ignore_collect(collection_path, config):
    suite = config.getoption("--suite")
    path = Path(collection_path)
    if path.is_dir() and path.parent == HERE:
        if path.name == "unit":
            return suite not in {"light", "all"}
        if path.name == "reproduction":
            return suite not in {"reproduction", "all"}
    if path.parent == HERE and path.name.startswith("test_") and path.suffix == ".py":
        return path.name not in (
            [name for names in SUITES.values() for name in names]
            if suite == "all"
            else SUITES.get(suite, [])
        )
    return False
