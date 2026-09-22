"""Minimal fresh-evaluation bundle, including frozen inputs and no credentials."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.fresh_eval import STUDY, verify_freeze


def build(output):
    freeze = verify_freeze(ROOT)
    names = set(freeze["hashes"])
    names.update(
        [
            f"{STUDY}/FREEZE.json",
            f"{STUDY}/gate/GATE.json",
            "reproduce/inventory.json",
            "data/validation.json",
            "src/sp_lense/__init__.py",
            "src/sp_lense/steering/__init__.py",
            "src/sp_lense/steering/gated.py",
            "src/sp_lense/steering/policy.py",
            "src/sp_lense/reproduction/__init__.py",
            "src/sp_lense/reproduction/paths.py",
            "study/02_lora_transfer/run/adapter/adapter_config.json",
            "study/02_lora_transfer/run/validation_base.jsonl",
            "study/02_lora_transfer/run/validation_teacher.jsonl",
        ]
    )
    for name in (
        "__init__",
        "fresh_eval",
        "activation_runtime",
        "adaptive",
        "controller",
        "jev_gate",
        "metrics",
        "runtime",
    ):
        names.add(f"src/sp_lense/research2/{name}.py")
    data = {n: (ROOT / n).read_bytes() for n in sorted(names)}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", ZIP_DEFLATED) as z:
        for name, content in data.items():
            z.writestr(name, content)
        z.writestr(
            "PILOT_MANIFEST.json",
            json.dumps({n: hashlib.sha256(b).hexdigest() for n, b in data.items()}, indent=2),
        )
    return {
        "path": str(output),
        "files": len(data),
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    print(json.dumps(build(p.parse_args().output)))
