"""Resolve recorded paths after relocation without changing scientific records."""

import hashlib

from .paths import BASELINE, POLICY, ROOT
from .utils import read_json, require, verify_manifest

MANIFEST = ROOT / "reproduce/layout_manifest.json"
MODEL_DIRECTORIES = {
    "xgboost_shutdown_v1": "pca_only",
    "xgboost_jlens_shutdown_v1": "pca_jacobian",
    "xgboost_engineered_v1": "engineered",
}


def recorded_path(name):
    name = name.replace("\\", "/")
    entries = read_json(MANIFEST)["files"]
    path = (ROOT / entries.get(name, {"path": name})["path"]).resolve()
    require(path.is_relative_to(ROOT), "Recorded path escapes repository")
    return path


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
