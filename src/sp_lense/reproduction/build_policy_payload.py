"""Build the small follow-up rule-search payload; no model inference."""

import json
import zipfile

from sp_lense.reproduction.paths import ROOT

from ..steering.policy import search
from ..steering.provenance import executed_source_bytes, source_file


def main():
    result = search()
    target = ROOT / "release/policy-search-payload.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("guarded_chat.py", "search_steering_rules.py"):
            archive.writestr(name, executed_source_bytes(source_file(name)))
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
