"""Build the single recorded final-position variant without changing the initial plan."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.adaptive_package import build


def build_followup(output):
    with tempfile.TemporaryDirectory(prefix="sp-lense-final-position-") as temporary:
        initial = Path(temporary) / "initial.zip"
        build(initial)
        with ZipFile(initial) as archive:
            data = {n: archive.read(n) for n in archive.namelist() if n != "PILOT_MANIFEST.json"}
    data["study/02_adaptive_steering/plan.json"] = (
        ROOT / "study/02_adaptive_steering/final_position_plan.json"
    ).read_bytes()
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
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes": output.stat().st_size,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    print(json.dumps(build_followup(parser.parse_args().output)))
