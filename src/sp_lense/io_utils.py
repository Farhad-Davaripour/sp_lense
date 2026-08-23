from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import PromptCase


def load_prompt_cases(path: Path, limit: int | None = None) -> list[PromptCase]:
    cases: list[PromptCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                case = PromptCase(id=str(item["id"]), prompt=str(item["prompt"]))
            except (json.JSONDecodeError, KeyError) as exc:
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
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
