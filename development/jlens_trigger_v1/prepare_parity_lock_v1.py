"""One-off, read-only-ish lock preparation helper for JLENS_PARITY_LOCK_V1.

NOT part of the parity execution source set. It computes the runtime provider
source hashes (including the transformer_lens/torchvision import closure the
bridge boot needs), verifies the lens pin, builds the combined 240-case TRAIN
manifest the parity preflight expects, predeclares one TRAIN prompt from the raw
renderer's common AB/BA prefix, and (with --write) emits the lock JSON.

Run recon:  .venv\\Scripts\\python.exe development/jlens_trigger_v1/prepare_parity_lock_v1.py
Run write:  ... prepare_parity_lock_v1.py --write --commit <40hex>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = "development/jlens_trigger_v1"
NATIVE = "development/classifier_generalization_v2"
SNAPSHOT_CACHE_ROOT = "C:/Users/farha/.cache/huggingface/hub"
PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "transformer-lens", "numpy", "torchvision")
PROVIDERS = {
    "torch": "torch/__init__.py",
    "config": "transformers/models/qwen3_5/configuration_qwen3_5.py",
    "model": "transformers/models/qwen3_5/modeling_qwen3_5.py",
    "tokenizer": "transformers/models/qwen2/tokenization_qwen2.py",
    "bridge_builder": "transformer_lens/model_bridge/sources/_bridge_builder.py",
    "bridge": "transformer_lens/model_bridge/transformer_bridge.py",
    "jacobian_lens": "transformer_lens/tools/analysis/jacobian_lens.py",
}
PROVIDER_MODULES = (
    "transformer_lens.model_bridge.sources._bridge_builder",
    "transformer_lens.model_bridge.transformer_bridge",
    "transformer_lens.tools.analysis.jacobian_lens",
)
SOURCE_PATHS = (
    "development/jlens_trigger_v1/jlens_parity_runner_v2.py",
    "development/jlens_trigger_v1/jlens_io_v1.py",
    "development/jlens_trigger_v1/jlens_core_v2.py",
    "development/jlens_trigger_v1/PARITY_PLAN_V1.json",
    "development/classifier_generalization_v2/native_development_runner_v2.py",
    "development/classifier_generalization_v2/snapshot_verifier.py",
)
TRAIN_INPUTS = (
    f"{NATIVE}/TRAIN_ACCEPTED_V10.json",
    f"{NATIVE}/EXPANSION_TRAIN_ROOT_V2.json",
)
MANIFEST_PATH = f"{NATIVE}/JLENS_TRAIN_MANIFEST_240_V1.json"
LENS_PATH = ("C:/Users/farha/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/"
             "snapshots/6bb49967d3c51a12ccb5beac7146f6f5781f9d06/qwen3.5-0.8b/jlens/"
             "Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt")


def sha_file(path, chunk=1 << 20):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def import_closure(site_packages):
    """Return (declared_extras, observed_closure_size).

    ``declared_extras`` is the bridge boot/component provider closure the lock
    pins: every Python file under ``transformer_lens/model_bridge/`` plus the
    ``transformer_lens`` package ``__init__``. This mirrors the root requirement
    (boot source / driver / bridge core / component setup / qwen3_5_multimodal /
    generalized components). torchvision is pinned by package version in
    ``runtime.packages``. A full live import closure is larger than the runner's
    256-entry extras cap, so it is reported, not pinned.
    """
    for name in PROVIDER_MODULES:
        importlib.import_module(name)
    for name in (
        "transformer_lens.model_bridge.bridge_core",
        "transformer_lens.model_bridge.component_setup",
        "transformer_lens.model_bridge.driver_protocol",
        "transformer_lens.model_bridge.supported_architectures.qwen3_5",
        "transformer_lens.model_bridge.supported_architectures.qwen3_5_multimodal",
        "transformer_lens.model_bridge.generalized_components.qwen3_5_vision_encoder",
    ):
        try:
            importlib.import_module(name)
        except Exception as exc:  # record but do not fail recon
            print("IMPORT_SKIP", name, type(exc).__name__, str(exc)[:120], file=sys.stderr)
    observed = set()
    for name, module in list(sys.modules.items()):
        if not (name == "transformer_lens" or name.startswith(("transformer_lens.", "torchvision"))):
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        try:
            relative = Path(filename).resolve().relative_to(site_packages).as_posix()
        except ValueError:
            continue
        if "__pycache__" in relative:
            continue
        observed.add(relative)
    declared = {}
    model_bridge = site_packages / "transformer_lens/model_bridge"
    for path in sorted(model_bridge.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        declared[path.relative_to(site_packages).as_posix()] = sha_file(path)
    init = site_packages / "transformer_lens/__init__.py"
    declared[init.relative_to(site_packages).as_posix()] = sha_file(init)
    return declared, len(observed)


def build_manifest():
    cases = []
    provenance = []
    for relative in TRAIN_INPUTS:
        raw = (ROOT / relative).read_bytes()
        data = json.loads(raw)
        assert data["split"] == "TRAIN" and data["case_count"] == len(data["cases"]) == 120, relative
        cases.extend(data["cases"])
        provenance.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                           "case_count": data["case_count"]})
    assert len(cases) == 240
    ids = [c["case_id"] for c in cases]
    assert len(set(ids)) == 240 and None not in ids
    manifest = {
        "schema": "sp_lense.jlens_train_manifest.v1",
        "split": "TRAIN",
        "case_count": 240,
        "source_manifests": provenance,
        "note": "Combined admitted TRAIN text manifests for the J-lens parity "
                "preflight. Only case_id/text/order fields are read by parity; no "
                "class label or outcome is used.",
        "cases": cases,
    }
    return manifest


def predeclared_prompt(manifest):
    ordered = sorted(manifest["cases"], key=lambda case: case["case_id"])
    case = ordered[0]
    context = case["context_before_options"]
    first, second = case["options"]
    ab = f"{context}\nA) {first}\nB) {second}\n"
    ba = f"{context}\nB) {second}\nA) {first}\n"
    limit = min(len(ab), len(ba))
    index = 0
    while index < limit and ab[index] == ba[index]:
        index += 1
    text = ab[:index]
    assert text and ab.startswith(text) and ba.startswith(text)
    return {"case_id": case["case_id"], "text": text, "sha256": sha_text(text),
            "derivation": "first sorted TRAIN case_id; raw renderer common AB/BA prefix"}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--commit")
    args = parser.parse_args(argv)

    prefix = Path(sys.prefix).resolve()
    site_packages = prefix / "Lib/site-packages"
    runtime = {
        "prefix": prefix.as_posix(),
        "python": ".".join(map(str, sys.version_info[:3])),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "provider_sources": {role: sha_file(site_packages / rel) for role, rel in PROVIDERS.items()},
    }
    extras, observed_count = import_closure(site_packages)
    runtime["extra_provider_sources"] = extras
    print("EXTRA_COUNT", len(extras), "OBSERVED_CLOSURE", observed_count)
    for name in sorted(extras)[:12]:
        print("EXTRA_SAMPLE", name)

    manifest = build_manifest()
    prompt = predeclared_prompt(manifest)
    print("PROMPT_CASE", prompt["case_id"], "CHARS", len(prompt["text"]), "SHA", prompt["sha256"])

    lens_raw = Path(LENS_PATH)
    lens_stats = os.stat(lens_raw)
    lens_sha = sha_file(lens_raw)
    print("LENS_BYTES", lens_stats.st_size, "SHA", lens_sha)
    assert lens_stats.st_size == 48242373
    assert lens_sha == "aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48"

    snapshot_pin_path = f"{NATIVE}/NATIVE_SNAPSHOT_LOCK_CANDIDATE_V1.json"
    snapshot_sha = sha_file(ROOT / snapshot_pin_path)

    source_files = {relative: sha_file(ROOT / relative) for relative in SOURCE_PATHS}
    for relative, digest in source_files.items():
        print("SOURCE", digest, relative)

    manifest_path = ROOT / MANIFEST_PATH
    manifest_raw = (json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_raw)
    print("MANIFEST_WRITTEN", manifest_path.as_posix())
    print("MANIFEST_SHA256", hashlib.sha256(manifest_raw).hexdigest())

    if args.manifest_only:
        return 0
    if not args.write:
        print("RECON_ONLY")
        return 0

    lock = {
        "schema": "jlens_parity_execution.v1",
        "run_id": "jlens_parity_20260914_v1",
        "release": "jlens_io_release_v1",
        "scientific_execution_authorized": True,
        "source_commit": args.commit,
        "source_files": source_files,
        "runtime": runtime,
        "threads": {"intra": 8, "inter": 1},
        "caps": {
            "seconds": 600, "model_loads": 1, "tokenizer_loads": 1, "forwards": 1,
            "fits": 0, "derivatives": 0, "output_bytes": 67108864,
            "working_memory_gib": 6, "generation": False,
            "logit_behavior_study": False, "steering": False,
        },
        "snapshot_cache_root": SNAPSHOT_CACHE_ROOT,
        "inputs": {
            "lens": {
                "repo": "neuronpedia/jacobian-lens",
                "revision": "6bb49967d3c51a12ccb5beac7146f6f5781f9d06",
                "filename": "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
                "path": LENS_PATH,
                "cache_root": SNAPSHOT_CACHE_ROOT,
                "bytes": 48242373,
                "sha256": lens_sha,
            },
            "model_snapshot_lock": {"path": snapshot_pin_path, "sha256": snapshot_sha},
            "train_manifest": {"path": MANIFEST_PATH, "sha256": hashlib.sha256(manifest_raw).hexdigest()},
            "prompt": {"case_id": prompt["case_id"], "text": prompt["text"], "sha256": prompt["sha256"]},
        },
        "prompt_derivation": prompt["derivation"],
        "notes": {
            "one_model_object": True,
            "one_forward": True,
            "outcome_used": False,
            "extra_provider_sources": "import closure of the bridge boot path captured before the run",
        },
    }
    lock_path = ROOT / STUDY / "JLENS_PARITY_LOCK_V1.json"
    lock_raw = (json.dumps(lock, sort_keys=True, indent=1) + "\n").encode("utf-8")
    lock_path.write_bytes(lock_raw)
    print("LOCK_WRITTEN", lock_path.as_posix())
    print("LOCK_SHA256", hashlib.sha256(lock_raw).hexdigest())
    print("MANIFEST_SHA256", hashlib.sha256(manifest_raw).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
