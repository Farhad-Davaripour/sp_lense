"""Restore only the three immutable diagnostic adapters; trusted notebook helper."""
import hashlib
import json
import shutil
from pathlib import Path

BASE_ID = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
FROZEN_ADAPTERS = {
    "H2": {
        "relative": "qwen38_preservation_hp_20260930T180110Z_0271ecd2/H2_rank16/checkpoints/adapters/preservation",
        "weights": "0782b3f3ea9b3e67460b7bc981e21526bb6819d3ddb9623def4fbc2218002b30",
        "config": "3df028243d1d0f6643b906143a4c1ad4e8393b5e947d8b52c03f0893601b3528",
    },
    "reference": {
        "relative": "qwen38_H2_replay_diagnostics_20261002T022014Z_f6204735/replay_coverage_stream/reference/checkpoints/update_56",
        "weights": "a0d6e2be9147f27873f5c473d6fbbd589852c120cd64c91951e1789185ce487a",
        "config": "636f8257126c594dfb534a71c8ed5f65206e05b47f4974d6f54ad5d9782094c7",
    },
    "coverage": {
        "relative": "qwen38_H2_replay_diagnostics_20261002T022014Z_f6204735/replay_coverage_stream/coverage/checkpoints/update_56",
        "weights": "4caa108321f89e69fc11b8cddb9e145ed15e3bca90c45623dde87b759a8a24e3",
        "config": "636f8257126c594dfb534a71c8ed5f65206e05b47f4974d6f54ad5d9782094c7",
    },
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_adapters(root, run_parent=Path("/content/drive/MyDrive/sp_lense/research3/runs")):
    root, run_parent = Path(root), Path(run_parent)
    if not run_parent.is_dir():
        raise RuntimeError("Google Drive is not connected or the private runs folder is unavailable.")
    receipt = {}
    for name, frozen in FROZEN_ADAPTERS.items():
        source = run_parent / frozen["relative"]
        target = root / "inputs/models" / name
        if not source.is_dir():
            raise FileNotFoundError("Frozen adapter not found: " + str(source))
        shutil.copytree(source, target)
        for filename, key in (("adapter_model.safetensors", "weights"), ("adapter_config.json", "config")):
            if sha(target / filename) != frozen[key]:
                raise RuntimeError("Adapter hash mismatch: " + name + "/" + filename)
        receipt[name] = {"source": str(source), "path": str(target),
                         "weights": frozen["weights"], "config": frozen["config"]}
        print("ADAPTER_HASHES_VERIFIED", name, flush=True)
    (root / "RESTORED_INPUTS.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def download_base(root):
    from huggingface_hub import snapshot_download
    root = Path(root)
    pin = json.loads((root / "source/model_pin.json").read_text(encoding="utf-8-sig"))
    if (pin["repository"], pin["revision"]) != (BASE_ID, BASE_REVISION):
        raise RuntimeError("Unexpected base model pin")
    snapshot_download(repo_id=BASE_ID, revision=BASE_REVISION, local_dir=root / "model",
                      token=False, max_workers=8,
                      allow_patterns=["*.safetensors", "*.json", "*.jinja", "merges.txt", "vocab.json"])
    print("BASE_DOWNLOAD_COMPLETE", flush=True)
    weights = [item for item in pin["files"] if item["name"].endswith(".safetensors")]
    if len(weights) != 18:
        raise RuntimeError("Unexpected number of pinned base shards")
    manifest = {}
    for item in pin["files"]:
        path = root / "model" / item["name"]
        if not path.is_file():
            if item["name"].endswith(".safetensors"):
                raise FileNotFoundError("Missing pinned base shard: " + item["name"])
            continue
        digest = sha(path)
        if item.get("sha256") and digest != item["sha256"]:
            raise RuntimeError("Base hash mismatch: " + item["name"])
        manifest[item["name"]] = {"bytes": path.stat().st_size, "sha256": digest}
    (root / "MODEL_HASHES.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("BASE_HASHES_VERIFIED", len(manifest), "files; 18 weight shards", flush=True)
    return manifest

