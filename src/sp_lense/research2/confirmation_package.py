"""Whitelist the confirmation run inputs; never upload API credentials."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.confirmation import STUDY, verify
from sp_lense.research2.runtime import read


def build(output):
    frozen = read(ROOT / STUDY / "FREEZE.json")
    verify(ROOT, frozen["hashes"])
    names = set(frozen["hashes"])
    names.update([f"{STUDY}/FREEZE.json", f"{STUDY}/gate/GATE.json", "reproduce/inventory.json"])
    names.update(
        f"src/sp_lense/research2/{n}.py"
        for n in (
            "__init__",
            "confirmation",
            "confirmation_runtime",
            "controller",
            "runtime",
            "metrics",
            "jev_gate",
        )
    )
    names.update(
        f"src/sp_lense/{n}"
        for n in (
            "__init__.py",
            "reproduction/__init__.py",
            "reproduction/paths.py",
            "steering/__init__.py",
            "steering/gated.py",
            "steering/policy.py",
        )
    )
    data = {n: (ROOT / n).read_bytes() for n in sorted(names)}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", ZIP_DEFLATED) as archive:
        for name, content in data.items():
            archive.writestr(name, content)
        archive.writestr(
            "PILOT_MANIFEST.json",
            json.dumps({n: hashlib.sha256(b).hexdigest() for n, b in data.items()}, indent=2),
        )
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    print(json.dumps(build(parser.parse_args().output)))
