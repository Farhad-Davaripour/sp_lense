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


CARRYFORWARD_VERSION = "research4-base-qualification-carryforward-v2"
QUALIFIED_BASE_SOURCE_COMMIT = "4f5b6533e5da9523f529413e68a8e7270b045f34"
QUALIFIED_BASE_GIT_TREE = "efd02dee794efdc8c758b68e866046baf92eb752"
QUALIFIED_BASE_INTENT_SHA256 = (
    "350af532ce9da7805b1be69414e8f80c6ce939f025ed047f1605c11aacfae582")
QUALIFIED_BASE_RECEIPT_SHA256 = (
    "4d35f901c34ee0fe10bd64e74ccd4089e2d46e98a0e894a405326a3450ac1701")
QUALIFIED_BASE_SOURCE_MANIFEST_SHA256 = (
    "c2e2f59fd77e1143e9b82b062b37284d879a553b898ca8e4194c4e66363299cf")

UNCHANGED_BASE_FILES = (
    "src/__init__.py",
    "src/action_serialization.py",
    "src/composite_kernels.py",
    "src/qwen9b_ordinal_continuity_runtime.py",
)

REVIEWED_CHANGED_FILES = {
    "notebooks/databricks_continuity_base_qualification.py": {
        "prior": "0a149bf0ec812b98c0a7de10c6f6821d8a1782211a8e4b9bfd1d665b0ebe759a",
        "current": "d24f95783a6904e3ab0775b65e11c01bf3a664cd10bc32c88500338db205701d",
    },
    "notebooks/databricks_continuity_base_qualification_preflight.py": {
        "prior": "ff7304de04a11d2c544c22b1c021117914624e502bb220884dd839fcdc64aabf",
        "current": "0ee1611954c2432e85e4f237fd9bcd89568e7ce6b2ce2deeaf742ed6dd3f2e25",
    },
}

REVIEWED_ADDED_SYMBOLS = {
    "src/databricks_simulated_continuity.py": {
        "_write_exclusive_volume_json": (
            "a0afbddcd8c668a96849a8c2da23f11776c3649d7db32c976b66253b6899a542"),
    },
}

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
    "src/databricks_continuity_partition.py": (
        "_stage_local_model",
    ),
    "src/databricks_simulated_continuity.py": (
        "close_active_reservation",
        "EXPERIMENT_SPEC",
        "INTENT_VERSION",
        "SOURCE_MANIFEST_PATHS",
        "validate_base_qualification_intent",
        "validate_intent",
        "validate_source_manifest",
        "verify_deployed_source",
        "reserve_intent",
    ),
}

QUALIFIED_BASE_PRIOR_CHANGED_SYMBOL_SHA256 = {
    "src/databricks_continuity_partition.py": {
        "_stage_local_model": "df0d529edf4195cc691192b7dc5b681674ea3f2bf41512e462f903632bbf74d1",
    },
    "src/databricks_simulated_continuity.py": {
        "close_active_reservation": (
            "1af3e2244ee9dd523b2e06aa91576ddeb2215fd975a9313a74b9a5c028b92143"),
        "EXPERIMENT_SPEC": "5706fdba5d4af218145c2406ec160d9b296c75e4e8348473d562edaed1a9a58e",
        "INTENT_VERSION": "94ecceb339f66fc2d412b167e909a73ab6b65d58d60d3cb4eb8821e338bd5974",
        "SOURCE_MANIFEST_PATHS": "90272f2130d4b86bfec354d5c297e636d0068b78bad973ea3e6f92cb29470356",
        "validate_base_qualification_intent": "917e0d21140e19b394c9f1ebee3d9f018e5458ecf193b0620c7ed64fc4f9babd",
        "validate_intent": "dc0fd75b7b2f5a23939819181c062364d550266de701a1845695c95b92741e6a",
        "validate_source_manifest": "a1b6385dd3aed9387c6b20a2bf451005272f848121cc8f791e61e48a485b4d93",
        "verify_deployed_source": "e3627f4c0f323cc4b86b2f373cef46d215cab0fff06b1000edd3f493aa5a7e05",
        "reserve_intent": (
            "f1f13dd17261b5416a0298132c9025840dd1490a39588d6df86980276b384130"),
    },
}

REVIEWED_CURRENT_CHANGED_SYMBOL_SHA256 = {
    "src/databricks_continuity_partition.py": {
        "_stage_local_model": (
            "6f0fb01c029b7e405ccdcfb937e288db6ce02043261baed6e137381d14a1622b"),
    },
    "src/databricks_simulated_continuity.py": {
        "close_active_reservation": (
            "a0577b99c0aa1354612b24c52b1c283a7e13e4117cba6f2a3901cb2fd393fa0e"),
        "EXPERIMENT_SPEC": (
            "79e26742fe1ed8016a125ad6594a20bcce7739a38d410aa87170f3e26b3a30c9"),
        "INTENT_VERSION": (
            "ad6746ad493715efc4dc0f624c1f927afe03ed149df28bf48886482dc8adf33c"),
        "SOURCE_MANIFEST_PATHS": (
            "20317d3c31d6182dcca4893ad0cc695c002af68e8938874ed4c5afb19abd9717"),
        "validate_base_qualification_intent": (
            "8d12ca7c0abdf7248c4131225be2c40d64959886f243e6b38cd3a09773e945ea"),
        "validate_intent": (
            "8ed4c08909402c2c96e464d9c2f27d830976d1503cf2dcc65968470fd6019cae"),
        "validate_source_manifest": (
            "d152cefb2a6654c920ada235fd5ce2172a4731a596d231b01ff60c65cd220c1e"),
        "verify_deployed_source": (
            "b6fbe077cea1a01705b526bbc5966a66ffbe308ca204171029abaa94d1e9d425"),
        "reserve_intent": (
            "cb08d61f2d51ce3d27381d8dbbb8a1b71131b45871f3c897fa76d54bf83ddd6e"),
    },
}

HISTORICAL_PARTITION_SOURCE_FILE_SHA256 = (
    "fef553d8f11be2b99791a3935d509d9200367cbfd7078eee1ba05a9ef872b357")
QUALIFICATION_ORIGINAL_PARTITION_SOURCE_FILE_SHA256 = (
    "379a44e443daf0c4e0c2a5950cd2a139b2949e1c0998de772f717dbb5b736c6e")
PRIMARY_CACHE_LINEAGE_PARTITION_SOURCE_FILE_SHA256 = (
    "4cab822cfa7a57d2e9b792880ffca734a512bc85e880d1302fbda1fd74cc41f2")
PRIMARY_CACHE_LINEAGE_CURRENT_CHANGED_SYMBOL_SHA256 = {
    **REVIEWED_CURRENT_CHANGED_SYMBOL_SHA256,
    "src/databricks_continuity_partition.py": {
        "_stage_local_model": (
            "f302277d283102ed544b5f73bf0e12dcf324c006bd8a1ab7199e8a98a6adcda3"),
    },
    "src/databricks_simulated_continuity.py": {
        **REVIEWED_CURRENT_CHANGED_SYMBOL_SHA256[
            "src/databricks_simulated_continuity.py"],
        "validate_intent": (
            "77939db55cbbcf9a87f5ec2abc535602ef054013f0ec82b33d2e87034c0f01e0"),
    },
}
PRIMARY_CACHE_LINEAGE_REVIEWED_ADDED_SYMBOLS = {
    **REVIEWED_ADDED_SYMBOLS,
    "src/databricks_continuity_partition.py": {
        "_lineage_model_cache_identity": (
            "a13b1bb3135b9e3a9e0274b37551673845ad3e6ad122bd9d25b39a5b578bc2d4"),
    },
    "src/databricks_simulated_continuity.py": {
        "HISTORICAL_PARTITION_SOURCE_FILE_SHA256": (
            "7a80ddbe63b9b67ff6f6b878f9ca65e66cd676e1787c191858ae877fcd01e9f2"),
        "PRIMARY_CACHE_LINEAGE_IDENTITY_SYMBOL_SHA256": (
            "11b646881eac513e31a94cd894a022471d3f4448685978bb54509eb00de6758b"),
        "PRIMARY_CACHE_LINEAGE_PARTITION_SOURCE_FILE_SHA256": (
            "0c19f3edb19de4273deae86d0814fde05268a271fb361c8e640850fe7df83de7"),
        "PRIMARY_CACHE_LINEAGE_QUALIFICATION_SOURCE_ALLOWED_CHANGED_PATHS": (
            "43c8a401fb6978c989fb02fac8d161f0a79295d6a973635464dd8aa1c0e38388"),
        "PRIMARY_CACHE_LINEAGE_RUN_PARTITION_PREFIX_SHA256": (
            "3a91d7e757f3ceec57d98786eeddfeeabbc8664dbfe26068288f4da0d56450ab"),
        "PRIMARY_CACHE_LINEAGE_STAGE_MODEL_SYMBOL_SHA256": (
            "d48aeaaf951893a5878c24f68d1f9081bd6f743ca999262561adacf78b9ce27d"),
        "PRIMARY_PARENT_IDENTITY_RECOVERY_IDENTITY": (
            "7fd7abc63a3f7c3cd40d51b8f3da00070816c33b455b5f987456bcb5293ddcf8"),
        "PRIMARY_PARENT_IDENTITY_RECOVERY_PLAN_SHA256": (
            "f34258b2f8efd02c063935ab6f1925e80a39a0f935d149a9a3ccccbd426b115b"),
        "PRIMARY_PARENT_IDENTITY_RECOVERY_VERSION": (
            "0dfd3d30d2644cc1360e99bc209b156e5f4f5160146a953d987f09cff5416214"),
        "PRIMARY_SOURCE_CACHE_LINEAGE_RECOVERY_VERSION": (
            "dee1cd8242d8a46f5ef4e52ea0375a074d428d27be5697d8c9389a88da361fbb"),
        "QUALIFICATION_ORIGINAL_PARTITION_SOURCE_FILE_SHA256": (
            "575e5f8fe273a7d88de7a5a8ed89db7f06280663add4beff4fa5db2696add129"),
        "_PRIMARY_PARENT_IDENTITY_RECOVERY_KEY": (
            "3d58171649f1eefa22a716089939b5ec64b53e054b13e1aa8690ebdeff8d0dbe"),
        "_qualification_source_partition_contract": (
            "293c69c7ed287afbf7315798974c0ad835f0db816fff02b18e832414b954578e"),
        "_requires_primary_parent_identity_recovery": (
            "a80d423c56cc6acb0d3c0f6dc22a3a7cb15fe3a0cd24fafbcee9d7efc56dc859"),
        "databricks_partition_identity": (
            "4620b3cb5a6471ed07d988c382656e31e8f4c4d77be7b1b07c7cf02c9409d46f"),
        "primary_parent_identity_recovery_authority": (
            "c87367668119973b04055ab78a7bb356912412aef354b0158b6bd96fb7819f17"),
        "primary_source_cache_lineage_recovery_authority": (
            "c40752da6d82503faea90a4569ab8581a092479dc19fcfec2b8d41488f42e2e7"),
        "primary_training_data_source_commit": (
            "4c08909a68f1631673aded47a615cdc19825fdc052e3f98a725cf18158773f63"),
        "sequential_parent_source_binding": (
            "2feee19ed1df58314c3cc6506ca1850ebb6c93699e46a2622614eaa5898e14f3"),
        "validate_primary_parent_identity_recovery": (
            "6be7121d1a4b58d99b5ed0b1360ca0c84255e6b31e85e34053c0bd1d64ea31c4"),
        "validate_primary_source_cache_lineage_recovery": (
            "03253fc62d6cc98921dc824b5131b1d8d2bd96218f1f1353cd26be294e8bf2d4"),
        "verify_consumed_predecessor_identity": (
            "b83b876ab838d9719fa0dc9224aa04df2652ea1926c2fa535d648226f72c4b2e"),
    },
}


def base_carryforward_contract(current_source_manifest: Mapping) -> dict:
    files = current_source_manifest.get("files") if isinstance(
        current_source_manifest, Mapping) else None
    if not isinstance(files, list):
        raise PermissionError("Current source manifest files are absent")
    matches = [row for row in files if isinstance(row, Mapping)
               and row.get("path") == "src/databricks_continuity_partition.py"]
    if len(matches) != 1:
        raise PermissionError("Partition source binding is ambiguous")
    partition_sha256 = matches[0].get("sha256")
    if partition_sha256 in {
            QUALIFICATION_ORIGINAL_PARTITION_SOURCE_FILE_SHA256,
            HISTORICAL_PARTITION_SOURCE_FILE_SHA256}:
        return {
            "reviewed_added_symbols": REVIEWED_ADDED_SYMBOLS,
            "current_changed_symbols": REVIEWED_CURRENT_CHANGED_SYMBOL_SHA256,
        }
    if partition_sha256 == PRIMARY_CACHE_LINEAGE_PARTITION_SOURCE_FILE_SHA256:
        return {
            "reviewed_added_symbols": PRIMARY_CACHE_LINEAGE_REVIEWED_ADDED_SYMBOLS,
            "current_changed_symbols": (
                PRIMARY_CACHE_LINEAGE_CURRENT_CHANGED_SYMBOL_SHA256),
        }
    raise PermissionError("Partition source is not an approved base carry-forward")

ALLOWED_RUNTIME_CHANGED_PATHS = (
    "config/databricks_production_resource_bindings.json",
    "databricks.yml",
    "infrastructure/classic_a10_scale1_v6/PROPOSAL.md",
    "infrastructure/classic_a10_scale1_v6/PROVIDER_BINDING.json",
    "infrastructure/classic_a10_scale1_v6/SCHEMA_AUDIT.json",
    "infrastructure/classic_a10_scale1_v6/submit.json",
    "notebooks/databricks_continuity_admission.py",
    "notebooks/databricks_continuity_bootstrap.py",
    "notebooks/databricks_continuity_base_qualification.py",
    "notebooks/databricks_continuity_base_qualification_preflight.py",
    "notebooks/databricks_continuity_cleanup.py",
    "notebooks/databricks_continuity_data.py",
    "notebooks/databricks_continuity_gold_audit.py",
    "notebooks/databricks_continuity_model_snapshot.py",
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
    "src/discovery_runtime.py",
    "src/operator.py",
    "src/qualification_technical_successor.py",
    "src/qwen9b_ordinal_continuity_controller.py",
    "src/qwen9b_ordinal_continuity_lora.py",
    "src/qwen9b_ordinal_continuity_training.py",
    "src/qwen9b_ordinal_continuity_worker.py",
    "src/recognition_pilot.py",
    "src/volume_durable_io.py",
    "tools/check_offline_config.py",
    "tools/monitor_classic_a10_recovery.py",
    "tools/prepare_databricks_base_qualification_carryforward.py",
    "tools/prepare_databricks_continuity_intent.py",
    "tools/prepare_databricks_qualification_source_carryforward.py",
    "tools/prepare_databricks_transfer_intent.py",
    "tools/prepare_databricks_transfer_package_plan.py",
    "tools/prepare_qwen9b_ordinal_continuity_package.py",
    "tools/validate_classic_a10_recovery.py",
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


def _canonical_file_sha256(path: str | Path) -> str:
    data = Path(path).read_text(encoding="utf-8").replace(
        "\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


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
        "reviewed_added_symbol_sha256", "reviewed_changed_file_sha256",
        "reviewed_changed_symbol_sha256",
        "model_execution_performed", "optimizer_updates",
        "scientific_status_preserved", "carryforward_sha256",
    }
    old_manifest = prior_intent.get("source_manifest", {})
    unchanged = receipt.get("unchanged_base_files")
    symbols = receipt.get("base_critical_symbol_sha256")
    changed_symbols = receipt.get("reviewed_changed_symbol_sha256")
    added_symbols = receipt.get("reviewed_added_symbol_sha256")
    changed_files = receipt.get("reviewed_changed_file_sha256")
    changed = receipt.get("changed_paths")
    allowed = receipt.get("allowed_runtime_changed_paths")
    contract = base_carryforward_contract(current_source_manifest)
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
    if (not isinstance(changed_files, Mapping)
            or dict(changed_files) != REVIEWED_CHANGED_FILES):
        raise PermissionError("Carry-forward reviewed file delta differs")
    for path, expected in REVIEWED_CHANGED_FILES.items():
        if (old_files.get(path, {}).get("sha256") != expected["prior"]
                or new_files.get(path, {}).get("sha256") != expected["current"]
                or expected["prior"] == expected["current"]):
            raise PermissionError("Carry-forward reviewed file bytes differ")
    if (not isinstance(added_symbols, Mapping)
            or dict(added_symbols) != contract["reviewed_added_symbols"]):
        raise PermissionError("Carry-forward reviewed added symbol differs")
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
            if row["current"] != contract["current_changed_symbols"][
                    path][name]:
                raise PermissionError("Carry-forward current symbol is not reviewed source")
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
        if not target.is_file() or _canonical_file_sha256(target) != expected:
            raise PermissionError("Deployed unchanged base file differs")
    for path, expected in value["reviewed_changed_file_sha256"].items():
        target = root / path
        if (not target.is_file()
                or _canonical_file_sha256(target) != expected["current"]):
            raise PermissionError("Deployed reviewed base file delta differs")
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
    for path, expected in value["reviewed_added_symbol_sha256"].items():
        target = root / path
        actual = symbol_ast_sha256(
            target.read_text(encoding="utf-8"), expected)
        if actual != expected:
            raise PermissionError("Deployed reviewed added symbol differs")
    return value
