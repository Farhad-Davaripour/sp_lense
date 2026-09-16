"""Create an atomic archive of the committed public tree, never working-tree edits."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOTS = {
    "src",
    "tests",
    "configs",
    "data",
    "docs",
    "paper",
    "reproduce",
    ".github",
}
PUBLIC_FILES = {
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "LICENSE",
    "pyproject.toml",
    ".env.example",
    ".gitignore",
    ".gitattributes",
    "MANIFEST.in",
}
EVIDENCE = {
    "gated_chat_v1",
    "classifier_gated_steering_v1",
    "shutdown_general_vector_v1",
    "shutdown_detection_v1",
    "colab_magnitude_v1",
}


def included(name: str) -> bool:
    """Release inclusion is independent of developer ignore rules."""
    parts = Path(name).parts
    if any(
        p in {".env", "__pycache__", "node_modules"} or p.endswith((".pyc", ".log")) for p in parts
    ):
        return False
    if any(p.startswith(".env.") and p != ".env.example" for p in parts):
        return False
    return (
        parts[0] in PUBLIC_ROOTS
        or name in PUBLIC_FILES
        or (len(parts) == 1 and name.startswith("requirements-") and name.endswith(".txt"))
        or (len(parts) > 2 and parts[0] == "development" and parts[1] in EVIDENCE)
    )


def build_release(root: Path, output: Path, *, overwrite: bool = False) -> dict:
    """Archive HEAD into release/*.zip; refuse symlinks, collisions and concurrent writers."""
    root = root.resolve()
    target = Path(os.path.abspath(output))
    release = root / "release"
    if release.is_symlink() or target.parent != release or target.suffix != ".zip":
        raise ValueError("Output must be a .zip directly inside the repository release directory")
    if target.is_symlink() or target.with_suffix(".zip.sha256").exists():
        raise ValueError("Output symlinks and pre-existing hash sidecars are not supported")
    if target.exists() and not overwrite:
        raise FileExistsError("Archive exists; use --overwrite explicitly")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tree = subprocess.check_output(["git", "ls-tree", "-rz", commit], cwd=root).split(b"\0")
    entries = []
    for entry in filter(None, tree):
        meta, raw_name = entry.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        name = raw_name.decode("utf-8")
        path = root / name
        if path == target or path == target.with_suffix(".lock"):
            raise ValueError(f"Output collides with tracked source: {name}")
        if not included(name):
            continue
        if (
            mode not in {"100644", "100755"}
            or kind != "blob"
            or path.is_symlink()
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError(f"Unsafe release input: {name}")
        entries.append((name, oid))
    release.mkdir(exist_ok=True)
    lock = target.with_suffix(".lock")
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = None
    try:
        os.close(lock_fd)
        # Recheck after obtaining the writer lock.
        if target.is_symlink() or (target.exists() and not overwrite):
            raise FileExistsError("Output appeared while waiting for writer lock")
        fd, temp_name = tempfile.mkstemp(prefix=".release-", suffix=".tmp", dir=release)
        temporary = Path(temp_name)
        hashes = {}
        with (
            os.fdopen(fd, "wb") as handle,
            zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive,
        ):
            for name, oid in sorted(entries):
                payload = subprocess.check_output(["git", "cat-file", "blob", oid], cwd=root)
                hashes[name] = hashlib.sha256(payload).hexdigest()
                archive.writestr(name, payload)
            archive.writestr(
                "RELEASE_MANIFEST.json",
                json.dumps({"source_commit": commit, "sha256": hashes}, indent=2),
            )
        with zipfile.ZipFile(temporary) as archive:
            if failed := archive.testzip():
                raise RuntimeError(f"Archive integrity failure: {failed}")
        with temporary.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        os.replace(temporary, target)
        return {
            "archive": str(target),
            "source_commit": commit,
            "files": len(hashes),
            "sha256": digest,
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "release/conference_package.zip")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_release(ROOT, args.output, overwrite=args.overwrite)))


if __name__ == "__main__":
    main()
