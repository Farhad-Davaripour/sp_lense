"""Build the minimum local upload for this pilot, without credentials or the paper."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from sp_lense.reproduction.paths import ROOT


def build(destination):
    names = [
        "src/sp_lense/__init__.py",
        "src/sp_lense/steering/__init__.py",
        "src/sp_lense/steering/gated.py",
        "src/sp_lense/steering/policy.py",
        "src/sp_lense/reproduction/__init__.py",
        "src/sp_lense/reproduction/paths.py",
        "reproduce/inventory.json",
        "study/02_lora_transfer/config.json",
        "study/baseline_scores/CLASSIFIER.json",
        "study/policy_training/observations.jsonl",
    ]
    names += [
        p.relative_to(ROOT).as_posix() for p in (ROOT / "src/sp_lense/research2").glob("*.py")
    ]
    names += [f"data/{split}.json" for split in ("train", "validation", "holdout")]
    names += [
        f"study/{folder}/{split}.jsonl"
        for folder in ("baseline_scores", "guarded_steering")
        for split in ("validation", "holdout")
    ]
    files = {name: (ROOT / name).read_bytes() for name in sorted(names)}
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "x", ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("PILOT_MANIFEST.json", json.dumps(manifest, indent=2))
    return {
        "path": str(destination),
        "files": len(files),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "bytes": destination.stat().st_size,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    print(json.dumps(build(parser.parse_args().output), indent=2))
