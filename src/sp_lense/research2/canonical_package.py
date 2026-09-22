"""Bundle only the base-controller pipeline; exclude teacher weights and API credentials."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.canonical_pipeline import STUDY
from sp_lense.research2.jev_gate import read


def build(output):
    freeze = read(ROOT / STUDY / "FREEZE.json")
    names = set(freeze["hashes"])
    names.update(
        [
            f"{STUDY}/FREEZE.json",
            f"{STUDY}/gate/GATE.json",
            "reproduce/inventory.json",
            "src/sp_lense/__init__.py",
            "src/sp_lense/reproduction/__init__.py",
            "src/sp_lense/reproduction/paths.py",
            "src/sp_lense/steering/__init__.py",
            "src/sp_lense/steering/gated.py",
            "src/sp_lense/steering/policy.py",
        ]
    )
    names.update(
        f"src/sp_lense/research2/{n}.py"
        for n in (
            "__init__",
            "canonical_pipeline",
            "controller",
            "student",
            "runtime",
            "metrics",
            "jev_gate",
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
        "files": len(data),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    print(json.dumps(build(parser.parse_args().output)))
