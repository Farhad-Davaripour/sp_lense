"""Create one reviewed CPU-only base-qualification carry-forward receipt."""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.base_qualification_carryforward import (  # noqa: E402
    ALLOWED_RUNTIME_CHANGED_PATHS,
    BASE_CRITICAL_ENTRYPOINTS,
    CARRYFORWARD_VERSION,
    REVIEWED_CHANGED_SYMBOLS,
    UNCHANGED_BASE_FILES,
    symbol_ast_sha256,
    symbol_dependency_closure_sha256,
    validate_carryforward,
)
from src.databricks_simulated_continuity import (  # noqa: E402
    build_source_manifest,
    digest,
    validate_base_qualification_intent,
    validate_base_qualification_receipt,
)


def _git(repository: Path, *args, binary=False):
    return subprocess.check_output(
        ["git", "-C", str(repository), *args],
        text=not binary)


def _git_file(repository: Path, commit: str, path: str) -> str:
    return _git(repository, "show", f"{commit}:research4/{path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("prior-source-commit", "current-source-commit",
                 "prior-base-intent", "base-receipt", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    source_root = Path(__file__).resolve().parents[1]
    repository = source_root.parent
    old, new = args.prior_source_commit, args.current_source_commit
    head = _git(repository, "rev-parse", "HEAD").strip()
    if head != new:
        raise PermissionError("Carry-forward current commit is not checked out")
    if _git(repository, "status", "--porcelain=v1", "--", "research4").strip():
        raise PermissionError("Carry-forward requires a clean merged Research4 source")
    subprocess.check_call(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", old, new])
    prior_intent = validate_base_qualification_intent(json.loads(Path(
        args.prior_base_intent).read_text(encoding="utf-8")))
    base_receipt = validate_base_qualification_receipt(json.loads(Path(
        args.base_receipt).read_text(encoding="utf-8")))
    if (prior_intent["source_commit"] != old
            or base_receipt["source_commit"] != old
            or base_receipt["base_qualification_intent_sha256"]
               != prior_intent["intent_sha256"]):
        raise PermissionError("Prior base authority differs from old commit")
    current_manifest = build_source_manifest(source_root, new)
    old_manifest = prior_intent["source_manifest"]
    old_files = {row["path"]: row for row in old_manifest["files"]}
    new_files = {row["path"]: row for row in current_manifest["files"]}
    unchanged = {}
    for path in UNCHANGED_BASE_FILES:
        old_sha = old_files.get(path, {}).get("sha256")
        new_sha = new_files.get(path, {}).get("sha256")
        if old_sha is None or old_sha != new_sha:
            raise PermissionError(f"Base-critical file changed: {path}")
        unchanged[path] = old_sha
    symbol_hashes = {}
    for path, entrypoints in BASE_CRITICAL_ENTRYPOINTS.items():
        old_hashes = symbol_dependency_closure_sha256(
            _git_file(repository, old, path), entrypoints,
            stop_symbols=REVIEWED_CHANGED_SYMBOLS.get(path, ()))
        new_hashes = symbol_dependency_closure_sha256(
            (source_root / path).read_text(encoding="utf-8"), entrypoints,
            stop_symbols=REVIEWED_CHANGED_SYMBOLS.get(path, ()))
        if set(old_hashes) != set(new_hashes):
            raise PermissionError(f"Base-critical closure changed: {path}")
        for name in REVIEWED_CHANGED_SYMBOLS.get(path, ()):
            old_hashes.pop(name, None); new_hashes.pop(name, None)
        if old_hashes != new_hashes:
            raise PermissionError(f"Base-critical symbols changed: {path}")
        symbol_hashes[path] = old_hashes
    reviewed_symbol_hashes = {}
    for path, names in REVIEWED_CHANGED_SYMBOLS.items():
        old_hashes = symbol_ast_sha256(_git_file(repository, old, path), names)
        new_hashes = symbol_ast_sha256(
            (source_root / path).read_text(encoding="utf-8"), names)
        if old_hashes == new_hashes:
            raise PermissionError(f"Reviewed symbol delta is absent: {path}")
        reviewed_symbol_hashes[path] = {
            name: {"prior": old_hashes[name], "current": new_hashes[name]}
            for name in names}
    changed = sorted(line.removeprefix("research4/") for line in _git(
        repository, "diff", "--name-only", old, new, "--", "research4").splitlines()
                     if line.strip())
    if any((not path.startswith("tests/") and path != "AGENTS.md"
            and path not in ALLOWED_RUNTIME_CHANGED_PATHS) for path in changed):
        raise PermissionError("Carry-forward commit changed an unreviewed runtime path")
    diff_bytes = _git(repository, "diff", "--binary", old, new, "--", "research4",
                      binary=True)
    value = {
        "version": CARRYFORWARD_VERSION,
        "prior_source_commit": old,
        "current_source_commit": new,
        "prior_git_tree": _git(repository, "rev-parse", f"{old}^{{tree}}").strip(),
        "current_git_tree": _git(repository, "rev-parse", f"{new}^{{tree}}").strip(),
        "git_diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
        "changed_paths": changed,
        "allowed_runtime_changed_paths": list(ALLOWED_RUNTIME_CHANGED_PATHS),
        "prior_source_manifest_sha256": prior_intent["source_sha256"],
        "current_source_manifest_sha256": current_manifest["manifest_sha256"],
        "prior_base_intent_sha256": prior_intent["intent_sha256"],
        "prior_base_receipt_sha256": base_receipt["receipt_sha256"],
        "model_tree_sha256": base_receipt["model_tree_sha256"],
        "model_receipt_sha256": base_receipt["model_receipt_sha256"],
        "dataset_tree_sha256": base_receipt["dataset_tree_sha256"],
        "base_plan_sha256": base_receipt["plan_sha256"],
        "serialization_contract_version": base_receipt[
            "serialization_contract_version"],
        "qualification_set_version": base_receipt["qualification_set_version"],
        "qualification_seed": base_receipt["qualification_seed"],
        "qualification_selection_digest": base_receipt[
            "qualification_selection_digest"],
        "qualification_proposal_sha256": base_receipt[
            "qualification_proposal_sha256"],
        "unchanged_base_files": dict(sorted(unchanged.items())),
        "base_critical_symbol_sha256": dict(sorted(symbol_hashes.items())),
        "reviewed_changed_symbol_sha256": dict(sorted(
            reviewed_symbol_hashes.items())),
        "model_execution_performed": False,
        "optimizer_updates": 0,
        "scientific_status_preserved": "qualified",
    }
    value["carryforward_sha256"] = digest(value)
    validate_carryforward(
        value, prior_intent=prior_intent, base_receipt=base_receipt,
        current_source_manifest=current_manifest)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Carry-forward receipt is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({"carryforward_sha256": value["carryforward_sha256"],
                      "changed_paths": len(changed)}, sort_keys=True))


if __name__ == "__main__":
    main()
