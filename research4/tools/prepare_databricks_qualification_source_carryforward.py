"""Create reviewed source-only lineage for a completed qualification."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.databricks_simulated_continuity import (
    QUALIFICATION_FINALIZATION_RESUME_CLAIM_FILE_SHA256,
    QUALIFICATION_FINALIZATION_RESUME_CLAIM_SHA256,
    QUALIFICATION_PARENT_FINALIZATION_FILE_SHA256,
    QUALIFICATION_PARENT_FINALIZATION_SHA256,
    QUALIFICATION_PARENT_MANIFEST_FILE_SHA256,
    QUALIFICATION_PARENT_MANIFEST_SHA256,
    QUALIFICATION_RECEIPT_FILE_SHA256,
    QUALIFICATION_RECEIPT_SHA256,
    QUALIFICATION_RECOVERY_RESULT_FILE_SHA256,
    QUALIFICATION_RECOVERY_RESULT_SHA256,
    QUALIFICATION_SOURCE_ORIGINAL_COMMIT,
    build_source_manifest,
    create_qualification_source_carryforward,
    digest,
    file_sha256,
)
from src.base_qualification_carryforward import (
    ALLOWED_RUNTIME_CHANGED_PATHS,
    QUALIFIED_BASE_SOURCE_COMMIT,
)


def validate_base_runtime_changed_paths(paths):
    normalized = sorted(set(paths))
    if (not normalized
            or any(not isinstance(path, str) or not path.startswith("research4/")
                   for path in normalized)):
        raise PermissionError("Base carry-forward Git paths differ")
    logical = [path.removeprefix("research4/") for path in normalized]
    if any(path != "AGENTS.md" and not path.startswith("tests/")
           and path not in ALLOWED_RUNTIME_CHANGED_PATHS for path in logical):
        raise PermissionError("Base carry-forward contains an unreviewed path")
    return normalized


def _verify_self_hashed_file(path, *, field, semantic_sha256,
                             expected_file_sha256):
    target = Path(path)
    value = json.loads(target.read_text(encoding="utf-8"))
    payload = dict(value); embedded = payload.pop(field, None)
    if (file_sha256(target) != expected_file_sha256
            or embedded != semantic_sha256
            or embedded != digest(payload)):
        raise PermissionError(f"Exact {target.name} evidence differs")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-intent", required=True)
    parser.add_argument("--qualification-receipt", required=True)
    parser.add_argument("--finalization-resume-claim", required=True)
    parser.add_argument("--recovery-result", required=True)
    parser.add_argument("--parent-finalization", required=True)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    source_root = Path(__file__).resolve().parents[1]
    repository = source_root.parent
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True).strip()
    if head != args.source_commit:
        raise PermissionError("Carry-forward source is not the checked-out commit")
    changed = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain=v1", "--",
         "research4"], text=True).strip()
    if changed:
        raise PermissionError("Carry-forward requires a clean reviewed checkout")
    subprocess.check_call([
        "git", "-C", str(repository), "merge-base", "--is-ancestor",
        QUALIFICATION_SOURCE_ORIGINAL_COMMIT, head])
    current_git_tree = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", f"{head}^{{tree}}"],
        text=True).strip()
    raw_diff = subprocess.check_output([
        "git", "-C", str(repository), "-c", "core.pager=cat", "diff",
        "--no-ext-diff", "--binary", QUALIFICATION_SOURCE_ORIGINAL_COMMIT,
        head, "--", "research4"])
    base_changed_paths = subprocess.check_output([
        "git", "-C", str(repository), "diff", "--name-only",
        QUALIFIED_BASE_SOURCE_COMMIT, head, "--", "research4"],
        text=True).splitlines()
    validate_base_runtime_changed_paths(base_changed_paths)
    qualification_intent = json.loads(Path(
        args.qualification_intent).read_text(encoding="utf-8"))
    qualification_receipt = json.loads(Path(
        args.qualification_receipt).read_text(encoding="utf-8"))
    receipt_path = Path(args.qualification_receipt)
    if (receipt_path.stat().st_size != 5293
            or file_sha256(receipt_path) != QUALIFICATION_RECEIPT_FILE_SHA256
            or qualification_receipt.get("receipt_sha256")
               != QUALIFICATION_RECEIPT_SHA256
            or digest({key: value for key, value in qualification_receipt.items()
                       if key != "receipt_sha256"})
               != QUALIFICATION_RECEIPT_SHA256):
        raise PermissionError("Exact qualification receipt bytes differ")
    _verify_self_hashed_file(
        args.finalization_resume_claim, field="claim_sha256",
        semantic_sha256=QUALIFICATION_FINALIZATION_RESUME_CLAIM_SHA256,
        expected_file_sha256=(
            QUALIFICATION_FINALIZATION_RESUME_CLAIM_FILE_SHA256))
    _verify_self_hashed_file(
        args.recovery_result, field="result_sha256",
        semantic_sha256=QUALIFICATION_RECOVERY_RESULT_SHA256,
        expected_file_sha256=QUALIFICATION_RECOVERY_RESULT_FILE_SHA256)
    _verify_self_hashed_file(
        args.parent_finalization, field="parent_finalization_sha256",
        semantic_sha256=QUALIFICATION_PARENT_FINALIZATION_SHA256,
        expected_file_sha256=QUALIFICATION_PARENT_FINALIZATION_FILE_SHA256)
    _verify_self_hashed_file(
        args.parent_manifest, field="manifest_sha256",
        semantic_sha256=QUALIFICATION_PARENT_MANIFEST_SHA256,
        expected_file_sha256=QUALIFICATION_PARENT_MANIFEST_FILE_SHA256)
    value = create_qualification_source_carryforward(
        qualification_intent=qualification_intent,
        qualification_receipt=qualification_receipt,
        current_source_manifest=build_source_manifest(
            source_root, args.source_commit),
        current_git_tree=current_git_tree,
        current_research4_diff_sha256=hashlib.sha256(raw_diff).hexdigest())
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Carry-forward evidence is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({"carryforward_sha256": value[
        "carryforward_sha256"]}))


if __name__ == "__main__":
    main()
