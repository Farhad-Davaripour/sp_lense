"""Expose saved episodes from two read-only viewers without starting a game."""
from __future__ import annotations


class CombinedGameHistory:
    def __init__(self, current, older):
        self.current = current
        self.older = older

    def snapshot(self):
        # Unstarted cases belong in the live queue, not the saved-run list.
        recent = self.current.snapshot()
        historical = self.older.snapshot()
        runs, seen = [], set()
        for row in [*recent.get("runs", []), *historical.get("runs", [])]:
            ident = row.get("id")
            if not ident or ident in seen or not row.get("events"):
                continue
            seen.add(ident)
            runs.append(row)
        return {"runs": runs, "read_only": True,
                "viewer_warning": historical.get("viewer_warning") or recent.get("viewer_warning")}
