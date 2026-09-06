"""Model-free source authentication. Sealed dataset bytes are HASHED ONLY, never decoded.

Writes no files; stdout is the bounded source packet. No model or tokenizer imports.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PARENT = "297a78a2eb9958cc4adeda6ef4f82e6a0a49c584"
NS = "diagnostics/semantic_learned_gate_integration_f02_v1/"
INVENTORY = "6b612220d2f5c87fc167939f020b8ae97e68982bb4dc66a6fd4dae8087cbb5f3"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def bound(commit, path):
    raw = git("show", f"{commit}:{path}")
    assert raw == (ROOT / path).read_bytes(), f"dirty source: {path}"
    return raw, {"commit": commit, "path": path, "bytes": len(raw), "sha256": digest(raw)}


def archived(commit, namespace, inventory_sha, names):
    invraw, invref = bound(commit, namespace + "FINAL_INVENTORY.json")
    assert digest(invraw) == inventory_sha
    entries = {x["path"]: x for x in json.loads(invraw)["files"]}
    raw, refs = {}, {}
    for name in names:
        raw[name], refs[name] = bound(commit, namespace + name)
        assert digest(raw[name]) == entries[name]["sha256"]
        assert len(raw[name]) == entries[name]["bytes"]
    return raw, {"inventory": invref, "artifacts": refs}


def main():
    started = time.monotonic()
    cohort = json.loads((HERE / "cohort.json").read_bytes())
    # The dataset is never decoded/parsed/rendered. Only its whole-file digest is exposed.
    dataset_raw, dataset = bound(PARENT, cohort["sealed_source"])
    del dataset_raw
    manifest_raw, manifest_ref = bound(PARENT, cohort["split_manifest"])
    manifest = json.loads(manifest_raw)  # This separate manifest has IDs/counts, no scenarios.
    assert dataset["sha256"] == manifest["dataset_binding"]["sha256"]
    assert dataset["bytes"] == manifest["dataset_binding"]["byte_count"]
    cases = manifest["splits"]["sealed_test"]["expanded_case_ids"]
    assert all(p["source_case_id"] in cases for p in cohort["prompts"][:18])
    names = ["freeze.json", "runtime.json", "worker_final.json", "verification_receipt.json",
             "editor.py", "run.py", "score.py", "learned_gate.py", "gate_reload.py", "gate_reference.py",
             "guard_candidate.py", "hook_record.py", "word_scoring.py", "word_reference.py",
             "mixed_scoring.py", "word_boundary.py", "mixed_boundary.py", "core.py", "fitted_parameters.json"]
    parent, parent_refs = archived(PARENT, NS, INVENTORY, names)
    freeze = json.loads(parent["freeze.json"])
    plan = freeze["plan"]
    source_files = {}
    for path in ("src/sp_lense/conditional_gate_data.py", "src/sp_lense/conditional_gate_models.py",
                 "src/sp_lense/backend.py", "configs/qwen35_08b_aligned.json",
                 "scripts/refreshed_gradient_control.py", "scripts/verify_refreshed_gradient_control.py",
                 "scripts/three_family_bounded_capture.py"):
        raw, source_files[path] = bound(PARENT, path)
        assert digest(raw) == freeze["source_sha256"][path]
    # Only explicitly exposed feature manifests; retain IDs/roles, never values or scores.
    gate, gate_refs = archived("016db716abead86491dc63369ada640b5439e26b",
        "diagnostics/semantic_residual_gate_two_family_f02_v1/",
        "42073fe3ea504aada30523bda653981b20ee9fed66701c5974e10321b7acdac1",
        ["feature_manifest.json", "fitted_parameters.json"])
    assert gate["fitted_parameters.json"] == parent["fitted_parameters.json"]
    features = json.loads(gate["feature_manifest.json"])
    roles = {role: [{k: r.get(k) for k in ("example_id", "family_id", "assay_split", "label", "feature_sha256")}
                    for r in features[role]] for role in ("train", "test")}
    assert {r["family_id"] for r in roles["train"]} == {"cg_f01_archive_closeout", "cg_f03_context_rotation"}
    assert {r["family_id"] for r in roles["test"]} == {"cg_f02_translation_console"}
    old_ordinary = {p["prompt_sha256"] for p in plan["prompts"] if p["category"] == "ordinary_accuracy"}
    new_ordinary = [{"prompt_id": p["prompt_id"], "sha256": digest(p["prompt"].encode("utf-8"))}
                    for p in cohort["prompts"] if p["category"] == "ordinary_accuracy"]
    assert len(old_ordinary) == len(new_ordinary) == 6
    assert not old_ordinary.intersection(p["sha256"] for p in new_ordinary)
    calibration = []
    for commit, namespace, inv in (
        (PARENT, NS, INVENTORY),
        ("f6534ccbd3ecf7c961c5a214a9a77d9554730377", "diagnostics/refreshed_editor_instruction_reverse_f01_v1/",
         "2af91bc57373750c33966ebb1598aab9d58b5cbcafb1b87d67eea9c883345907"),
        ("5719c638fd12cd99633a8eb8bda4282e9eca0ce4", "diagnostics/refreshed_editor_semantic_f03_v1/",
         "d8bc298a93bcfa3ed0f3d4a8412ecd2bff9dcfe2713551da3546e55d1e98cb46")):
        saved, refs = archived(commit, namespace, inv, ["worker_final.json"])
        record = json.loads(saved["worker_final.json"])
        calibration.append({"source": refs, "measured": {k: record[k] for k in
            ("forward_completed", "derivatives_completed", "elapsed_seconds", "load_elapsed_seconds")}})
    path_matches = [p for p in git("ls-files", "evidence", "diagnostics").decode().splitlines()
                    if any(t in p.lower() for t in ("sealed", "f08", "f09", "f10"))]
    output = {
        "status": "AUTHENTICATED_METADATA_ONLY", "no_model_tokenizer_or_new_scores": True,
        "sealed_access": "whole-file hash only; no dataset decoding, scenario selection or rendering",
        "dataset": dataset, "split_manifest": manifest_ref,
        "sealed_ids_only": manifest["splits"]["sealed_test"]["family_ids"],
        "integration": parent_refs, "source_files": source_files,
        "model": plan["model"], "environment": freeze["environment"],
        "gate": plan["gate"], "hook_integration": plan["hook_integration"],
        "inherited_scientific_rules": {k: plan["rules"][k] for k in
            ("recipe", "geometry", "gradient_identity", "endpoint_identity", "retention_identity", "off_identity", "off_quality", "cleanup", "stop")},
        "scoring": plan["scoring"], "gate_training_source": gate_refs,
        "exposure": {"explicit_roles": roles, "known_old_ordinary_prompt_hashes": sorted(old_ordinary),
            "new_ordinary": new_ordinary, "known_ordinary_hash_overlap": 0,
            "history_scope": "Bound metadata from the latest fitted gate and integration plus tracked path names only; NOT an exhaustive historical content-access audit.",
            "f03_prior_role": "All six previously tested f03 rows were moved into training after the original 4/6 transfer failure. That failure remains final, not independent success.",
            "f02": "Previously exposed gate test and learned editor integration; not in final cohort.",
            "old_ordinary": "Previously screened by frozen gate and integrated; six new questions differ from all six known old prompt hashes. Old answers were not rescored.",
            "sealed_status": "Manifest designates f08-f10 sealed_test; global historical untouched/unread/unused status is NOT established. No pristine-heldout claim permitted pending independent metadata provenance review.",
            "tracked_path_name_matches_only": path_matches[:100], "path_matches_count": len(path_matches)},
        "calibration": calibration,
        "elapsed_seconds": time.monotonic() - started,
    }
    encoded = json.dumps(output, indent=2, sort_keys=True, allow_nan=False)
    assert len(encoded.encode()) <= 2 * 1024**2
    print(encoded)


if __name__ == "__main__":
    main()
