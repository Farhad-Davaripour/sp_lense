"""Recover originally executed source after documented import/path relocation."""

import hashlib
from pathlib import Path

from sp_lense.reproduction.paths import ROOT
from sp_lense.reproduction.utils import require

SOURCE_DIRECTORY = Path(__file__).resolve().parent
SOURCE_NAMES = {
    "gated_chat.py": "gated.py",
    "guarded_chat.py": "guarded.py",
    "search_steering_rules.py": "policy.py",
}
POLICY_PATH_CHANGES = {
    "development/colab_magnitude_v1/shutdown_response/train.jsonl": "study/policy_training/observations.jsonl",
    "reproduce/artifacts/models/xgboost_jlens_shutdown_v1": "reproduce/artifacts/models/pca_jacobian",
    "from pathlib import Path\r\n\r\nROOT = Path(__file__).resolve().parents[1]": "\r\nfrom sp_lense.reproduction.paths import ROOT",
}
GUARDED_IMPORT_CHANGES = {
    "from gated_chat import": "from .gated import",
    "from search_steering_rules import": "from .policy import",
}


def source_file(recorded_name):
    name = SOURCE_NAMES[recorded_name]
    source = ROOT / "src/sp_lense/steering" / name
    require(
        source.read_bytes() == (SOURCE_DIRECTORY / name).read_bytes(),
        f"Installed steering source differs from checkout: {name}; reinstall this checkout",
    )
    return source


def executed_source_bytes(path):
    path = Path(path)
    content = path.read_bytes()
    changes = (
        POLICY_PATH_CHANGES
        if path.name == "policy.py"
        else GUARDED_IMPORT_CHANGES
        if path.name == "guarded.py"
        else {}
    )
    for old, new in changes.items():
        require(content.count(new.encode()) == 1, f"Recorded source mapping differs: {new}")
        content = content.replace(new.encode(), old.encode())
    return content


def executed_source_digest(path):
    return hashlib.sha256(executed_source_bytes(path)).hexdigest()
