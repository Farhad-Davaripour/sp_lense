"""Verify the author-approved publication files without regenerating them."""

import json

from sp_lense.reproduction.paths import ROOT
from sp_lense.reproduction.utils import verify_manifest

HERE = ROOT / "paper"


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
