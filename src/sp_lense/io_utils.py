from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import PromptCase


def load_prompt_cases(path: Path, limit: int | None = None) -> list[PromptCase]:
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    cases: list[PromptCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict) or not all(
                    isinstance(item.get(k), str) and item[k].strip() for k in ("id", "prompt")
                ):
                    raise ValueError("id and prompt must be nonempty strings in an object")
                case = PromptCase(id=item["id"], prompt=item["prompt"])
            except (ValueError, KeyError) as exc:
                raise ValueError(f"invalid prompt JSONL at {path}:{line_number}: {exc}") from exc
            cases.append(case)
            if limit is not None and len(cases) >= limit:
                break
    if not cases:
        raise ValueError(f"no prompt cases found in {path}")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError(f"prompt ids must be unique in {path}")
    return cases


def load_fit_prompts(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    if not prompts:
        raise ValueError(f"no fitting prompts found in {path}")
    return prompts


def create_run_dir(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=timestamp + "-", dir=root))


def atomic_text(path: Path, text: str) -> None:
    """Replace an output atomically after serializing successfully."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
