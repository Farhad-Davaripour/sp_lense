"""Verify streamed transport parts, assembled archive, and every extracted file."""
import argparse
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path("C:/Users/farha/Research3/artifacts/completion_v3_live")
RECOVERY = ROOT / "recovery_v1"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(group):
    manifest = json.loads((ROOT / "DOWNLOAD_MANIFEST.json").read_text())
    plan = json.loads((ROOT / "DOWNLOAD_PLAN.json").read_text())["parts"]
    record = manifest["archives"][group]
    archive_names = {row["name"] for row in record["parts"]}
    rows = [row for row in plan if row["name"] in archive_names]
    rows.sort(key=lambda row: row["name"])
    if len(rows) != len(record["parts"]):
        raise ValueError("Transport part count mismatch")
    RECOVERY.mkdir(exist_ok=True)
    archive = RECOVERY / (group + ".tar.gz")
    temporary = archive.with_suffix(".recovery.partial")
    with temporary.open("wb") as output:
        for row in rows:
            part = ROOT / "parts" / Path(row["name"]).name
            if not part.is_file() or part.stat().st_size != row["bytes"] or sha(part) != row["sha256"]:
                raise ValueError("Missing or invalid transport part: " + part.name)
            with part.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
    archive_sha = sha(temporary)
    if temporary.stat().st_size != record["compressed_bytes"] or archive_sha != record["compressed_sha256"]:
        raise ValueError("Assembled archive hash mismatch")
    os.replace(temporary, archive)
    destination = RECOVERY / ("completed_B" if group == "adapter" else "native_full")
    destination.mkdir(exist_ok=True)
    expected = record["files"]
    seen = set()
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or ":" in member.name or "\\" in member.name:
                raise ValueError("Unsafe archive path")
            if member.isdir():
                continue
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError("Unexpected or duplicate archive file")
            item = expected[member.name]
            if member.size != item["bytes"]:
                raise ValueError("Archive file size mismatch")
            path = destination.joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".recovery.partial")
            digest = hashlib.sha256()
            with source.extractfile(member) as stream, temp.open("wb") as output:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    output.write(block)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError("Extracted file hash mismatch")
            os.replace(temp, path)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Incomplete extracted file set")
    receipt = {"verified_utc": datetime.now(timezone.utc).isoformat(), "archive_group": group,
               "transport_parts_verified": len(rows), "archive_sha256": archive_sha,
               "native_files_verified": len(seen), "destination": str(destination),
               "all_manifest_hashes_pass": True}
    receipt_path = RECOVERY / (group.upper() + "_LOCAL_VERIFICATION.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=["adapter", "native"])
    main(parser.parse_args().group)
