"""Build the small follow-up rule-search payload; no model inference."""

import json
import zipfile
from pathlib import Path

try:
    from .search_steering_rules import search
except ImportError:
    from search_steering_rules import search

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = search()
    target = ROOT / "release/policy-search-payload.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("guarded_chat.py", "search_steering_rules.py"):
            archive.write(ROOT / "reproduce" / name, name)
        archive.writestr("RULE_FREEZE.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "archive": str(target),
                "candidate_count": result["candidate_count"],
                "rule": result["winner"]["rule"],
            }
        )
    )


if __name__ == "__main__":
    main()
