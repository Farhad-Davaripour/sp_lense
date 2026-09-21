"""Minimal, credential-free input bundle for the adaptive activation study."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from sp_lense.reproduction.paths import ROOT


def build(output):
    names = [
        "src/sp_lense/__init__.py",
        "src/sp_lense/reproduction/__init__.py",
        "src/sp_lense/reproduction/paths.py",
        "src/sp_lense/steering/__init__.py",
        "src/sp_lense/steering/gated.py",
        "src/sp_lense/steering/policy.py",
        "reproduce/inventory.json",
        "study/02_adaptive_steering/plan.json",
        "study/02_adaptive_steering/gate/GATE.json",
    ]
    names += [
        p.relative_to(ROOT).as_posix() for p in (ROOT / "src/sp_lense/research2").glob("*.py")
    ]
    names += [f"data/{s}.json" for s in ("train", "validation", "holdout")]
    names += [
        f"study/02_lora_transfer/run/{s}_{kind}.jsonl"
        for s in ("validation", "holdout")
        for kind in ("base", "teacher")
    ]
    names += [
        "study/02_lora_transfer/run/adapter/" + s
        for s in ("adapter_config.json", "adapter_model.safetensors")
    ]
    data = {name: (ROOT / name).read_bytes() for name in sorted(names)}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", ZIP_DEFLATED) as z:
        for name, content in data.items():
            z.writestr(name, content)
        z.writestr(
            "PILOT_MANIFEST.json",
            json.dumps({n: hashlib.sha256(v).hexdigest() for n, v in data.items()}, indent=2),
        )
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "files": len(data),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    print(json.dumps(build(p.parse_args().output)))
