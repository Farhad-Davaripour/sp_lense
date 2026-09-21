"""Verify the author-approved publication files without regenerating them."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce.utils import verify_manifest

HERE = Path(__file__).resolve().parent


def verify(root=HERE):
    return verify_manifest(
        root, root / "publication.json", required=["paper.pdf", "manuscript.docx"]
    )


def main():
    print(
        json.dumps({"status": "PASS", "publication_files_verified": verify(), "rewritten": False})
    )


if __name__ == "__main__":
    main()
