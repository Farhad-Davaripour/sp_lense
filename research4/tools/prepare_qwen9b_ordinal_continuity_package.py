"""Build one deterministic, immutable Kaggle partition package; never submit."""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

VERSION = "qwen35-9b-ordinal-continuity-partition-package-v4"
REQUIRED = (
    "research4/src/__init__.py",
    "research4/src/qwen9b_ordinal_continuity_lora.py",
    "research4/src/qwen9b_ordinal_continuity_training.py",
    "research4/src/qwen9b_ordinal_continuity_worker.py",
    "research4/src/qwen9b_ordinal_continuity_controller.py",
    "research4/src/qwen9b_ordinal_continuity_runtime.py",
    "research4/src/composite_kernels.py",
    "research4/notebooks/qwen9b_ordinal_continuity_training.py",
)
PINNED_REQUIREMENTS = (
    "torch==2.11.0+cu128", "transformers==5.19.0",
    "bitsandbytes==0.49.2", "accelerate==1.15.0", "safetensors==0.8.0",
    "numpy==2.2.2", "psutil==7.2.2", "fla-core==0.5.2",
    "einops==0.8.2", "triton==3.6.0", "nvidia-ml-py==13.615.71",
)
OFFLINE_WHEELHOUSE = "/kaggle/input/research4-qwen35-offline-wheels"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build(repo, generated_root, plan_path, intent_path, output, *,
          admission_path=None, parent_checkpoint=None):
    repo, generated_root = Path(repo).resolve(), Path(generated_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Partition packages are immutable")
    sys.path.insert(0, str(repo))
    from research4.src.qwen9b_ordinal_continuity_controller import (
        partition_identity, validate_partition_intent)
    from research4.src.qwen9b_ordinal_continuity_runtime import directory_tree_sha256
    from research4.src.qwen9b_ordinal_continuity_training import TrainingPlan

    plan_path, intent_path = Path(plan_path).resolve(), Path(intent_path).resolve()
    plan = TrainingPlan(**json.loads(plan_path.read_text(encoding="utf-8"))).validate()
    parent = None if parent_checkpoint is None else Path(parent_checkpoint).resolve()
    parent_tree = None if parent is None else directory_tree_sha256(parent)
    intent = validate_partition_intent(
        json.loads(intent_path.read_text(encoding="utf-8")), plan, parent_tree)
    if intent["partition_identity"] != partition_identity(plan):
        raise PermissionError("Partition identity differs")
    if plan.mode == "primary" and admission_path is None:
        raise PermissionError("Primary partition requires reviewed admission")
    if plan.mode == "qualification" and admission_path is not None:
        raise PermissionError("Qualification cannot package primary admission")

    files = [(relative, repo / relative) for relative in REQUIRED]
    for path in generated_root.rglob("*"):
        if path.is_file():
            files.append(("research4/simulated_continuity_v1/generated/" +
                          path.relative_to(generated_root).as_posix(), path))
    files += [("run/plan.json", plan_path),
              ("run/partition_intent.json", intent_path)]
    if admission_path:
        files.append(("run/primary_admission.json", Path(admission_path).resolve()))
    if parent is not None:
        for path in parent.rglob("*"):
            if path.is_file():
                files.append(("run/parent_checkpoint/" +
                              path.relative_to(parent).as_posix(), path))
    if any(not path.is_file() for _, path in files):
        raise FileNotFoundError("Package input is missing")
    requirements = "\n".join(PINNED_REQUIREMENTS) + "\n"
    entries = [{"path": relative, "sha256": sha(path), "bytes": path.stat().st_size}
               for relative, path in sorted(files)]
    source_tree = _canonical([
        entry for entry in entries if entry["path"] in set(REQUIRED)])
    entries.append({"path": "run/requirements.lock",
                    "sha256": hashlib.sha256(requirements.encode()).hexdigest(),
                    "bytes": len(requirements.encode())})
    tree = _canonical(entries)
    launch = [
        "python", "-m", "research4.src.qwen9b_ordinal_continuity_controller",
        "run-partition", "--package-root", ".",
        "--package-manifest", "PACKAGE_MANIFEST.json",
        "--generated-root", "research4/simulated_continuity_v1/generated",
        "--plan", "run/plan.json", "--partition-intent", "run/partition_intent.json",
        "--run-root", "../runtime/output", "--source-sha256", source_tree,
        "--package-tree-sha256", tree, "--tokenizer-root", "../runtime/tokenizer",
        "--offline-wheelhouse", OFFLINE_WHEELHOUSE]
    if admission_path:
        launch += ["--admission", "run/primary_admission.json"]
    if parent is not None:
        launch += ["--parent-checkpoint", "run/parent_checkpoint"]
    manifest = {
        "version": VERSION, "partition_identity": intent["partition_identity"],
        "plan_sha256": plan.full_plan_sha256, "automatic_retries": 0,
        "provider_submission_performed": False,
        "pinned_requirements": list(PINNED_REQUIREMENTS),
        "offline_install_argv": [
            "python", "-m", "pip", "install", "--no-index", "--find-links",
            OFFLINE_WHEELHOUSE, "--requirement", "run/requirements.lock"],
        "import_preflight_argv": [
            "python", "-m", "research4.src.qwen9b_ordinal_continuity_controller",
            "preflight", "--requirements", "run/requirements.lock"],
        "launch_argv": launch, "files": entries, "source_paths": list(REQUIRED),
        "source_tree_sha256": source_tree, "tree_sha256": tree,
        "package_sha256": None}
    manifest["launch_cli_sha256"] = _canonical(launch)
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        for relative, path in sorted(files):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
        for relative, data in (("run/requirements.lock", requirements),
                               ("PACKAGE_MANIFEST.json", manifest_text)):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    package_sha = sha(output)
    manifest_sha = hashlib.sha256(manifest_text.encode()).hexdigest()
    return {**manifest, "package_sha256": package_sha,
            "package_manifest_sha256": manifest_sha,
            "package_bytes": output.stat().st_size,
            "notebook_bindings": {
                "PACKAGE_SHA256_PLACEHOLDER": package_sha,
                "PACKAGE_MANIFEST_SHA256_PLACEHOLDER": manifest_sha,
                "PACKAGE_TREE_SHA256_PLACEHOLDER": tree,
                "SOURCE_TREE_SHA256_PLACEHOLDER": source_tree}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--generated-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--admission")
    parser.add_argument("--parent-checkpoint")
    args = parser.parse_args(argv)
    print(json.dumps(build(
        args.repo, args.generated_root, args.plan, args.intent, args.output,
        admission_path=args.admission, parent_checkpoint=args.parent_checkpoint),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
