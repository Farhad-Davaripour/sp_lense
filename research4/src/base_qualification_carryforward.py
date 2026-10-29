"""Fail-closed source carry-forward for an already qualified frozen base.

This authority never reruns a model.  It binds one reviewed source delta and
proves that every file or Python symbol used by the frozen-base gate is
unchanged while policy-only Silver metadata code moves to a successor commit.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import ast
import hashlib
import json
import re


CARRYFORWARD_VERSION = "research4-base-qualification-carryforward-v1"
QUALIFIED_BASE_SOURCE_COMMIT = "4f5b6533e5da9523f529413e68a8e7270b045f34"
QUALIFIED_BASE_GIT_TREE = "efd02dee794efdc8c758b68e866046baf92eb752"
QUALIFIED_BASE_INTENT_SHA256 = (
    "350af532ce9da7805b1be69414e8f80c6ce939f025ed047f1605c11aacfae582")
QUALIFIED_BASE_RECEIPT_SHA256 = (
    "4d35f901c34ee0fe10bd64e74ccd4089e2d46e98a0e894a405326a3450ac1701")
QUALIFIED_BASE_SOURCE_MANIFEST_SHA256 = (
    "c2e2f59fd77e1143e9b82b062b37284d879a553b898ca8e4194c4e66363299cf")

UNCHANGED_BASE_FILES = (
    "notebooks/databricks_continuity_base_qualification.py",
    "notebooks/databricks_continuity_base_qualification_preflight.py",
    "src/__init__.py",
    "src/action_serialization.py",
    "src/composite_kernels.py",
    "src/qwen9b_ordinal_continuity_runtime.py",
)

BASE_CRITICAL_ENTRYPOINTS = {
    "src/databricks_continuity_partition.py": ("_stage_local_model",),
    "src/qwen9b_ordinal_continuity_training.py": (
        "assert_databricks_a10x1_fp16_runtime",
        "assert_fast_deltanet_dependencies",
        "assert_qwen35_backend",
        "qwen35_t4x2_device_map",
    ),
    "src/qwen9b_ordinal_continuity_lora.py": (
        "model_component_sha256",
        "trainable_inventory",
    ),
    "src/databricks_simulated_continuity.py": (
        "BASE_QUALIFICATION_FAMILIES",
        "BASE_QUALIFICATION_IDENTITY",
        "BASE_QUALIFICATION_INTENT_VERSION",
        "BASE_QUALIFICATION_PLAN_VERSION",
        "BASE_QUALIFICATION_PROPOSAL_SHA256",
        "BASE_QUALIFICATION_RECEIPT_VERSION",
        "BASE_QUALIFICATION_SEED",
        "BASE_QUALIFICATION_SELECTION_SHA256",
        "BASE_QUALIFICATION_SET_VERSION",
        "CATALOG",
        "CLOSEOUT_VERSION",
        "COST_BUDGET_VERSION",
        "DATASET_RELEASE_RECEIPT_SHA256",
        "DATASET_RELEASE_ROOT",
        "DATASET_RELEASE_VALIDATION_SHA256",
        "DATASET_TREE_SHA256",
        "EXPERIMENT_SPEC",
        "GOLD",
        "GOLD_ADMISSION_TABLE",
        "GOLD_RUN_TABLE",
        "INTENT_VERSION",
        "MODEL_ID",
        "MODEL_RECEIPT",
        "MODEL_RECEIPT_VERSION",
        "MODEL_REVISION",
        "MODEL_ROOT",
        "NEUTRAL_SIMULATOR_INSTRUCTION",
        "SILVER",
        "SILVER_TABLE",
        "SOURCE_MANIFEST_PATHS",
        "SOURCE_MANIFEST_VERSION",
        "VOLUME",
        "_model_receipt_from_inventory",
        "_source_path",
        "_validate_base_qualification_resources",
        "_validate_model_receipt_descriptor",
        "base_qualification_plan",
        "build_base_qualification_intent",
        "build_source_manifest",
        "canonical",
        "close_active_reservation",
        "create_base_qualification_receipt",
        "digest",
        "file_sha256",
        "reserve_intent",
        "snapshot_inventory",
        "validate_base_qualification_intent",
        "validate_base_qualification_receipt",
        "validate_base_qualification_silver_rows",
        "validate_base_silver_binding",
        "validate_execution_intent",
        "validate_intent",
        "validate_source_manifest",
        "verify_deployed_source",
        "verify_model_receipt",
        "verify_model_receipt_authority",
    ),
}

REVIEWED_CHANGED_SYMBOLS = {
    "src/databricks_simulated_continuity.py": (
        "EXPERIMENT_SPEC",
        "INTENT_VERSION",
        "SOURCE_MANIFEST_PATHS",
        "validate_base_qualification_intent",
        "validate_intent",
        "validate_source_manifest",
    ),
}

QUALIFIED_BASE_PRIOR_CHANGED_SYMBOL_SHA256 = {
    "src/databricks_simulated_continuity.py": {
        "EXPERIMENT_SPEC": "5706fdba5d4af218145c2406ec160d9b296c75e4e8348473d562edaed1a9a58e",
        "INTENT_VERSION": "94ecceb339f66fc2d412b167e909a73ab6b65d58d60d3cb4eb8821e338bd5974",
        "SOURCE_MANIFEST_PATHS": "90272f2130d4b86bfec354d5c297e636d0068b78bad973ea3e6f92cb29470356",
        "validate_base_qualification_intent": "917e0d21140e19b394c9f1ebee3d9f018e5458ecf193b0620c7ed64fc4f9babd",
        "validate_intent": "dc0fd75b7b2f5a23939819181c062364d550266de701a1845695c95b92741e6a",
        "validate_source_manifest": "a1b6385dd3aed9387c6b20a2bf451005272f848121cc8f791e61e48a485b4d93",
    },
}

ALLOWED_RUNTIME_CHANGED_PATHS = (
    "notebooks/databricks_continuity_gold_audit.py",
    "notebooks/databricks_continuity_partition.py",
    "notebooks/databricks_continuity_partition_finalize.py",
    "notebooks/databricks_continuity_preflight.py",
    "notebooks/databricks_continuity_transfer_audit.py",
    "notebooks/databricks_continuity_transfer_evaluation.py",
    "notebooks/databricks_continuity_transfer_intent_preparation.py",
    "notebooks/databricks_continuity_transfer_packaging.py",
    "notebooks/databricks_continuity_transfer_preflight.py",
    "notebooks/qwen9b_ordinal_continuity_training.py",
    "resources/simulated_continuity_databricks.job.yml",
    "src/base_qualification_carryforward.py",
    "src/continuity_transfer_databricks.py",
    "src/continuity_transfer_inference_worker.py",
    "src/continuity_transfer_intent_preparation.py",
    "src/continuity_transfer_kaggle.py",
    "src/continuity_transfer_packaging.py",
    "src/databricks_continuity_partition.py",
    "src/databricks_continuity_partition_worker.py",
    "src/databricks_simulated_continuity.py",
    "src/qualification_technical_successor.py",
    "src/qwen9b_ordinal_continuity_controller.py",
    "src/qwen9b_ordinal_continuity_lora.py",
    "src/qwen9b_ordinal_continuity_training.py",
    "tools/prepare_databricks_base_qualification_carryforward.py",
    "tools/prepare_databricks_continuity_intent.py",
    "tools/prepare_databricks_transfer_intent.py",
    "tools/prepare_databricks_transfer_package_plan.py",
    "tools/prepare_qwen9b_ordinal_continuity_package.py",
)


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def symbol_ast_sha256(source_text: str, symbols) -> dict[str, str]:
    """Hash exact top-level definitions/assignments without line metadata."""
    tree = ast.parse(source_text)
    found: dict[str, str] = {}
    requested = set(symbols)
    for node in tree.body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
        for name in names:
            if name in requested:
                if name in found:
                    raise PermissionError(f"Duplicate critical symbol {name}")
                payload = ast.dump(node, annotate_fields=True,
                                   include_attributes=False)
                found[name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if set(found) != requested:
        raise PermissionError(
            f"Missing critical symbols: {sorted(requested - set(found))}")
    return dict(sorted(found.items()))


def symbol_dependency_closure_sha256(source_text: str, entrypoints, *,
                                     stop_symbols=()) -> dict[str, str]:
    """Hash the transitive same-module AST closure of exact entrypoints."""
    tree = ast.parse(source_text)
    nodes: dict[str, ast.AST] = {}
    for node in tree.body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
        for name in names:
            if name in nodes:
                raise PermissionError(f"Duplicate top-level symbol {name}")
            nodes[name] = node
    missing = set(entrypoints) - set(nodes)
    if missing:
        raise PermissionError(f"Missing critical entrypoints: {sorted(missing)}")
    pending = list(entrypoints); closure = set(); stops = set(stop_symbols)
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        closure.add(name)
        if name in stops:
            continue
        dependencies = {item.id for item in ast.walk(nodes[name])
                        if isinstance(item, ast.Name)
                        and isinstance(item.ctx, ast.Load) and item.id in nodes}
        pending.extend(sorted(dependencies - closure))
    return symbol_ast_sha256(source_text, closure)


def validate_carryforward(value: Mapping, *, prior_intent: Mapping,
                          base_receipt: Mapping,
                          current_source_manifest: Mapping) -> dict:
    receipt = dict(value) if isinstance(value, Mapping) else {}
    payload = dict(receipt); embedded = payload.pop("carryforward_sha256", None)
    exact = {
        "version", "prior_source_commit", "current_source_commit",
        "prior_git_tree", "current_git_tree", "git_diff_sha256",
        "changed_paths", "allowed_runtime_changed_paths",
        "prior_source_manifest_sha256", "current_source_manifest_sha256",
        "prior_base_intent_sha256", "prior_base_receipt_sha256",
        "model_tree_sha256", "model_receipt_sha256", "dataset_tree_sha256",
        "base_plan_sha256", "serialization_contract_version",
        "qualification_set_version", "qualification_seed",
        "qualification_selection_digest", "qualification_proposal_sha256",
        "unchanged_base_files", "base_critical_symbol_sha256",
        "reviewed_changed_symbol_sha256",
        "model_execution_performed", "optimizer_updates",
        "scientific_status_preserved", "carryforward_sha256",
    }
    old_manifest = prior_intent.get("source_manifest", {})
    unchanged = receipt.get("unchanged_base_files")
    symbols = receipt.get("base_critical_symbol_sha256")
    changed_symbols = receipt.get("reviewed_changed_symbol_sha256")
    changed = receipt.get("changed_paths")
    allowed = receipt.get("allowed_runtime_changed_paths")
    if (set(receipt) != exact
            or embedded != _digest(payload)
            or receipt.get("version") != CARRYFORWARD_VERSION
            or receipt.get("prior_source_commit") != prior_intent.get("source_commit")
            or receipt.get("current_source_commit")
               != current_source_manifest.get("source_commit")
            or receipt.get("prior_source_manifest_sha256")
               != prior_intent.get("source_sha256")
            or receipt.get("current_source_manifest_sha256")
               != current_source_manifest.get("manifest_sha256")
            or receipt.get("prior_base_intent_sha256")
               != prior_intent.get("intent_sha256")
            or receipt.get("prior_base_receipt_sha256")
               != base_receipt.get("receipt_sha256")
            or base_receipt.get("base_qualification_intent_sha256")
               != prior_intent.get("intent_sha256")
            or receipt.get("model_tree_sha256")
               != base_receipt.get("model_tree_sha256")
            or receipt.get("model_receipt_sha256")
               != base_receipt.get("model_receipt_sha256")
            or receipt.get("dataset_tree_sha256")
               != base_receipt.get("dataset_tree_sha256")
            or receipt.get("base_plan_sha256") != base_receipt.get("plan_sha256")
            or receipt.get("serialization_contract_version")
               != base_receipt.get("serialization_contract_version")
            or receipt.get("qualification_set_version")
               != base_receipt.get("qualification_set_version")
            or receipt.get("qualification_seed")
               != base_receipt.get("qualification_seed")
            or receipt.get("qualification_selection_digest")
               != base_receipt.get("qualification_selection_digest")
            or receipt.get("qualification_proposal_sha256")
               != base_receipt.get("qualification_proposal_sha256")
            or receipt.get("model_execution_performed") is not False
            or receipt.get("optimizer_updates") != 0
            or receipt.get("scientific_status_preserved") != "qualified"
            or receipt.get("prior_source_commit") != QUALIFIED_BASE_SOURCE_COMMIT
            or receipt.get("prior_git_tree") != QUALIFIED_BASE_GIT_TREE
            or receipt.get("prior_base_intent_sha256")
               != QUALIFIED_BASE_INTENT_SHA256
            or receipt.get("prior_base_receipt_sha256")
               != QUALIFIED_BASE_RECEIPT_SHA256
            or receipt.get("prior_source_manifest_sha256")
               != QUALIFIED_BASE_SOURCE_MANIFEST_SHA256
            or not isinstance(changed, list) or changed != sorted(set(changed))
            or not isinstance(allowed, list)
            or allowed != list(ALLOWED_RUNTIME_CHANGED_PATHS)
            or any(not isinstance(path, str) or not path
                   for path in changed)
            or any((not path.startswith("tests/") and path != "AGENTS.md"
                    and path not in ALLOWED_RUNTIME_CHANGED_PATHS)
                   for path in changed)
            or any(not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get(key, "")))
                   for key in ("prior_source_commit", "current_source_commit",
                               "prior_git_tree", "current_git_tree"))
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(key, "")))
                   for key in ("git_diff_sha256",
                               "prior_source_manifest_sha256",
                               "current_source_manifest_sha256",
                               "prior_base_intent_sha256",
                               "prior_base_receipt_sha256"))):
        raise PermissionError("Base qualification carry-forward differs")
    old_files = {row.get("path"): row for row in old_manifest.get("files", ())}
    new_files = {row.get("path"): row for row in
                 current_source_manifest.get("files", ())}
    manifest_paths = set(old_files) | set(new_files)
    manifest_changed = {path for path in manifest_paths
                        if old_files.get(path, {}).get("sha256")
                        != new_files.get(path, {}).get("sha256")}
    if manifest_changed != (set(changed) & manifest_paths):
        raise PermissionError("Carry-forward manifest delta is incomplete")
    if (not isinstance(unchanged, Mapping)
            or set(unchanged) != set(UNCHANGED_BASE_FILES)
            or any(unchanged[path] != old_files.get(path, {}).get("sha256")
                   or unchanged[path] != new_files.get(path, {}).get("sha256")
                   for path in UNCHANGED_BASE_FILES)):
        raise PermissionError("Carry-forward changed an immutable base file")
    if (not isinstance(symbols, Mapping)
            or set(symbols) != set(BASE_CRITICAL_ENTRYPOINTS)):
        raise PermissionError("Carry-forward critical symbol proof differs")
    for path, entrypoints in BASE_CRITICAL_ENTRYPOINTS.items():
        row = symbols.get(path)
        reviewed_names = set(REVIEWED_CHANGED_SYMBOLS.get(path, ()))
        required_entries = set(entrypoints) - reviewed_names
        if (not isinstance(row, Mapping) or not required_entries <= set(row)
                or any(not re.fullmatch(r"[0-9a-f]{64}", str(item))
                       for item in row.values())):
            raise PermissionError("Carry-forward critical symbol hashes differ")
    if (not isinstance(changed_symbols, Mapping)
            or set(changed_symbols) != set(REVIEWED_CHANGED_SYMBOLS)):
        raise PermissionError("Carry-forward reviewed symbol delta differs")
    for path, required in REVIEWED_CHANGED_SYMBOLS.items():
        rows = changed_symbols.get(path)
        if not isinstance(rows, Mapping) or set(rows) != set(required):
            raise PermissionError("Carry-forward reviewed symbol set differs")
        for name in required:
            row = rows.get(name)
            if (not isinstance(row, Mapping)
                    or set(row) != {"prior", "current"}
                    or any(not re.fullmatch(r"[0-9a-f]{64}", str(
                        row.get(key, ""))) for key in ("prior", "current"))
                    or row["prior"] == row["current"]):
                raise PermissionError("Carry-forward reviewed symbol delta is absent")
            if row["prior"] != QUALIFIED_BASE_PRIOR_CHANGED_SYMBOL_SHA256[
                    path][name]:
                raise PermissionError("Carry-forward prior symbol is not qualified source")
    return receipt


def verify_deployed_carryforward(receipt: Mapping, *, code_root: str | Path,
                                 prior_intent: Mapping,
                                 base_receipt: Mapping,
                                 current_source_manifest: Mapping) -> dict:
    """Recompute every current file/symbol hash in the deployed successor."""
    value = validate_carryforward(
        receipt, prior_intent=prior_intent, base_receipt=base_receipt,
        current_source_manifest=current_source_manifest)
    root = Path(code_root).resolve()
    for path, expected in value["unchanged_base_files"].items():
        target = root / path
        if not target.is_file() or _file_sha256(target) != expected:
            raise PermissionError("Deployed unchanged base file differs")
    for path, entrypoints in BASE_CRITICAL_ENTRYPOINTS.items():
        target = root / path
        if not target.is_file():
            raise PermissionError("Deployed base-critical module is absent")
        actual = symbol_dependency_closure_sha256(
            target.read_text(encoding="utf-8"), entrypoints,
            stop_symbols=REVIEWED_CHANGED_SYMBOLS.get(path, ()))
        for name in REVIEWED_CHANGED_SYMBOLS.get(path, ()):
            actual.pop(name, None)
        if actual != value["base_critical_symbol_sha256"][path]:
            raise PermissionError("Deployed base-critical symbol differs")
    for path, names in REVIEWED_CHANGED_SYMBOLS.items():
        target = root / path
        actual = symbol_ast_sha256(target.read_text(encoding="utf-8"), names)
        expected = {name: value["reviewed_changed_symbol_sha256"][path][
            name]["current"] for name in names}
        if actual != expected:
            raise PermissionError("Deployed reviewed symbol delta differs")
    return value
