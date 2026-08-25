"""Validate a local J-space atom cache without running a model or analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def validate_cache(manifest_path: Path) -> dict[str, Any]:
    """Return a small receipt after the canonical validator checks all cache hashes."""

    from sp_lense.jspace_comparison import validate_jspace_atom_manifest

    validated = validate_jspace_atom_manifest(manifest_path.resolve())
    manifest = validated.manifest
    return {
        "status": "valid",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": validated.manifest_sha256,
        "model_id": manifest["model"]["id"],
        "layer": manifest["layer"],
        "atoms_file_sha256": manifest["atoms"]["file_sha256"],
        "token_labels_file_sha256": manifest["token_labels"]["file_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = validate_cache(args.manifest)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "invalid",
                    "manifest_path": str(args.manifest.resolve()),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
