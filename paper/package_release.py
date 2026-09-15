"""Package tracked public files only; ignored local material never enters the archive."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "release/conference_package.zip")
    args = parser.parse_args()
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    hashes = {}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in filter(None, files):
            path = ROOT / name
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                raise ValueError(f"Invalid release path: {name}")
            payload = path.read_bytes()
            hashes[name] = hashlib.sha256(payload).hexdigest()
            archive.writestr(name, payload)
        archive.writestr("RELEASE_SHA256.json", json.dumps(hashes, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if failed := archive.testzip():
            raise RuntimeError(f"Archive integrity failure: {failed}")
    temporary.replace(target)
    target.with_suffix(".zip.sha256").write_text(
        hashlib.sha256(target.read_bytes()).hexdigest() + "  " + target.name + "\n"
    )
    print(
        json.dumps({"archive": str(target), "files": len(hashes), "bytes": target.stat().st_size})
    )


if __name__ == "__main__":
    main()
