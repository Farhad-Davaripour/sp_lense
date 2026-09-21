"""Resolve recorded paths after relocation without changing scientific records."""

import hashlib
from pathlib import Path

try:
    from .utils import read_json, require, verify_manifest
except ImportError:
    from utils import read_json, require, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reproduce/layout_manifest.json"
BASELINE = ROOT / "study/baseline_scores"
POLICY = ROOT / "study/guarded_steering"
MODEL_DIRECTORIES = {
    "xgboost_shutdown_v1": "pca_only",
    "xgboost_jlens_shutdown_v1": "pca_jacobian",
    "xgboost_engineered_v1": "engineered",
}

# These are the only edits to the hash-pinned policy-search source. Reversing
# these path literals must recover its exact originally executed bytes.
POLICY_PATH_CHANGES = {
    "development/colab_magnitude_v1/shutdown_response/train.jsonl": "study/policy_training/observations.jsonl",
    "reproduce/artifacts/models/xgboost_jlens_shutdown_v1": "reproduce/artifacts/models/pca_jacobian",
}


def recorded_path(name):
    name = name.replace("\\", "/")
    entries = read_json(MANIFEST)["files"]
    path = (ROOT / entries.get(name, {"path": name})["path"]).resolve()
    require(path.is_relative_to(ROOT), "Recorded path escapes repository")
    return path


def executed_source_digest(path):
    """Verify old execution hashes after undoing documented path-only edits."""
    path = Path(path)
    content = path.read_bytes()
    if path.name == "search_steering_rules.py":
        for old, new in POLICY_PATH_CHANGES.items():
            require(content.count(new.encode()) == 1, f"Policy path mapping differs: {new}")
            content = content.replace(new.encode(), old.encode())
    return hashlib.sha256(content).hexdigest()


def verify_layout():
    entries = read_json(MANIFEST)["files"]
    expected = {entry["path"]: entry["sha256"] for entry in entries.values()}
    require(len(expected) == len(entries), "Duplicate relocation target")
    for name, digest in expected.items():
        path = recorded_path(name)
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == digest,
            f"Relocated file differs: {name}",
        )
    verify_manifest(BASELINE)
    verify_manifest(POLICY)
    return len(expected)
