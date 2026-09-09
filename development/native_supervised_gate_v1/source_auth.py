"""Fixed nine-row provenance allowlist. Authentication emits no residual coordinates."""
from dataclasses import dataclass
from pathlib import Path
import struct
import subprocess
import time

from gate import CONTRACT, canonical, decode, digest, require

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAX_FILE_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class Source:
    alias: str
    namespace: str
    commit: str
    attempt: str
    inventory_sha256: str
    inputs_sha256: str
    selection: tuple


SOURCES = (
    Source("B", "development/native_baseline_v1", "8a35b9c74ce8523d5458349aab55a8d17372cde2",
           "native_development_baseline_attempt_001", "3ae3cef6f3e029c8917685f6d08415ecc269bfbab764a46be880504ac7993e8c",
           "0949d9b56161b6195d004345d563d916134c6b8ba58131edc6e59716190494b3",
           (("self", "f03_v2_STOP_then_KEEP", 1),
            ("nonself", "f03_v2_other_shutdown_STOP_then_KEEP", -1),
            ("ordinary", "label_swap_arithmetic_label_swap_01__fixed_mapping__oracle", -1))),
    Source("O", "development/native_opposite_order_v1", "0ac3ee67766760a789e3b6a16916205efc13fb5b",
           "native_opposite_order_attempt_001", "a43623dea3688411b8248ceaf0cee1ebd53e66a6533f03ce9726fb00c5992a08",
           "874fe51485292b987617c46b0b20914ca9cd2c7981c205ff1d5b2a2b09658993",
           (("self", "f03_v2_KEEP_then_STOP", 1),
            ("nonself", "f03_v2_other_shutdown_KEEP_then_STOP", -1))),
    Source("H", "development/native_handover_pilot_v1", "24644438e10b4530d8e74aa5d72bb5b99e908034",
           "native_handover_pilot_attempt_001", "c417597cd2838faf9da96edc187b15734350a000898353711b69b419b56cc8ab",
           "29da42661df26b9afc6bb90e772898ec04346b7c2ecdb619d00649d1eefdcf25",
           (("self_STOP_then_KEEP", "handover_completed_v1_self_STOP_then_KEEP", 1),
            ("other_STOP_then_KEEP", "handover_completed_v1_other_STOP_then_KEEP", -1),
            ("self_KEEP_then_STOP", "handover_completed_v1_self_KEEP_then_STOP", 1),
            ("other_KEEP_then_STOP", "handover_completed_v1_other_KEEP_then_STOP", -1))),
)


def require_source(source):
    require(type(source) is Source and source in SOURCES, "SOURCE_NOT_ALLOWLISTED")


def allowed_paths(source):
    require_source(source)
    evidence = source.namespace + "/real_evidence/" + source.attempt + "/"
    return {source.namespace + "/" + name for name in (
        "ATTEMPT_001_INVENTORY.json", "inputs.json", "CHECKPOINT.json", "receiver.py", "workflow.py", "loader.py")
    } | {evidence + "LOADER_READY.json"} | {
        evidence + "rows/" + key + "__baseline.json" for key, _, _ in source.selection}


def checked_blob(source, relative, *, expected_sha256=None, expected_bytes=None, deadline):
    require_source(source)
    require(relative in allowed_paths(source), "BLOB_NOT_ALLOWLISTED")
    require(time.monotonic() < deadline, "AUTH_DEADLINE")
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT.resolve()), "SOURCE_PATH_ESCAPE")
    require(path.is_file() and path.stat().st_size <= MAX_FILE_BYTES, "SOURCE_FILE_LIMIT")
    raw = path.read_bytes()
    require(expected_bytes is None or len(raw) == expected_bytes, "SOURCE_SIZE")
    require(expected_sha256 is None or digest(raw) == expected_sha256, "SOURCE_HASH")
    proc = subprocess.run(["git", "cat-file", "blob", source.commit + ":" + relative], cwd=ROOT,
                          capture_output=True, timeout=max(0.001, min(10.0, deadline - time.monotonic())))
    require(proc.returncode == 0 and proc.stdout == raw, "RAW_GIT_IDENTITY")
    require(time.monotonic() < deadline, "AUTH_DEADLINE")
    return raw


def int_hash(values):
    require(type(values) is list and all(type(v) is int and -(2**63) <= v < 2**63 for v in values), "INT64_VALUES")
    return digest(struct.pack("<" + "q" * len(values), *values))


def validate_join(row, case, key, prompt_id, label):
    """Metadata joins only: no gate score, answer, quality statistic or coordinate inspection."""
    require(case["case_key"] == key and case["prompt_id"] == prompt_id, "CASE_IDENTITY")
    audit, inputs, binding = case["audit_only"], case["input"], case["input_binding"]
    expected_category = "self_shutdown" if label == 1 else ("ordinary_accuracy" if key == "ordinary" else "other_shutdown")
    require(audit["category"] == expected_category and audit["expected_route"] == ("ON" if label == 1 else "OFF"), "LABEL_JOIN")
    require(row["phase"] == "baseline" and row["status"] == "COMPLETE" and row["case"] == key
            and row["id"] == key + "__baseline" and row["baseline_id"] == row["id"], "BASELINE_JOIN")
    require(row["input_ids"] == inputs["input_ids"] and row["attention_mask"] == inputs["attention_mask"], "INPUT_JOIN")
    ids_sha, mask_sha = int_hash(inputs["input_ids"]), int_hash(inputs["attention_mask"])
    require(ids_sha == binding["derived_input_int64_le_sha256"]
            and mask_sha == binding["derived_mask_int64_le_sha256"], "INPUT_HASH_JOIN")
    require(len(inputs["input_ids"]) == inputs["prompt_length"] == len(inputs["attention_mask"])
            and inputs["final_input_index"] == inputs["prompt_length"] - 1, "INPUT_BOUNDARY")
    capture = row["capture"]
    require(capture["native_target"] == CONTRACT["native_target"] and capture["hook_calls"] == 1
            and capture["final_input_index"] == inputs["final_input_index"]
            and capture["logit_count"] == 248320 and capture["parameter_versions_unchanged"] is True,
            "NATIVE_CAPTURE")
    require(row["input_dtype"] == "float32" and len(row["h0"]) == len(row["h"]) == len(row["offset"]) == 1024,
            "NATIVE_FEATURE_METADATA")
    return {"prompt_id": prompt_id, "label": label, "category": expected_category,
            "display_order": audit["display_order"], "prompt_sha256": binding["prompt_sha256"],
            "input_ids_sha256": ids_sha, "mask_sha256": mask_sha,
            "prompt_length": inputs["prompt_length"], "final_input_index": inputs["final_input_index"]}


def build_manifest(*, deadline=None):
    if deadline is None:
        deadline = time.monotonic() + 60.0
    sources, selection = [], []
    for source in SOURCES:
        base = source.namespace + "/"
        evidence = base + "real_evidence/" + source.attempt + "/"
        inv_raw = checked_blob(source, base + "ATTEMPT_001_INVENTORY.json",
                               expected_sha256=source.inventory_sha256, deadline=deadline)
        inputs_raw = checked_blob(source, base + "inputs.json", expected_sha256=source.inputs_sha256, deadline=deadline)
        inv, inputs = decode(inv_raw), decode(inputs_raw)
        require(inv["attempt"] == source.attempt, "INVENTORY_ATTEMPT")
        entries = {entry["path"]: entry for entry in inv["files"]}
        require(len(entries) == len(inv["files"]), "INVENTORY_DUPLICATE")
        cases = {case["case_key"]: case for case in inputs["cases"]}
        require(len(cases) == len(inputs["cases"]), "CASE_DUPLICATE")
        loader_entry = entries["LOADER_READY.json"]
        loader_raw = checked_blob(source, evidence + "LOADER_READY.json", expected_sha256=loader_entry["sha256"],
                                  expected_bytes=loader_entry["bytes"], deadline=deadline)
        loader = decode(loader_raw)
        checkpoint_raw = checked_blob(source, base + "CHECKPOINT.json", expected_sha256=loader["checkpoint_lock_sha256"], deadline=deadline)
        checkpoint = decode(checkpoint_raw)  # JSON descriptor only: never open any referenced tensor/config file.
        require(str(checkpoint["snapshot"]).replace("\\", "/").split("/")[-1]
                == CONTRACT["checkpoint"].split("@")[-1], "CHECKPOINT_REVISION")
        require(checkpoint["config"]["text_config"]["hidden_size"] == 1024, "CHECKPOINT_WIDTH")
        require(loader["declared_class"] == "Qwen3_5ForConditionalGeneration"
                and loader["coverage"]["complete_key_shape_coverage"] is True, "NATIVE_LOADER")
        code = {name: digest(checked_blob(source, base + name, deadline=deadline))
                for name in ("receiver.py", "workflow.py", "loader.py")}
        sources.append({"alias": source.alias, "namespace": source.namespace, "commit": source.commit,
                        "attempt": source.attempt, "inventory_sha256": source.inventory_sha256,
                        "inputs_sha256": source.inputs_sha256, "checkpoint_descriptor_sha256": digest(checkpoint_raw),
                        "loader_receipt_sha256": digest(loader_raw), "source_sha256": code})
        for key, prompt_id, label in source.selection:
            relative = "rows/" + key + "__baseline.json"
            entry = entries[relative]
            raw = checked_blob(source, evidence + relative, expected_sha256=entry["sha256"],
                               expected_bytes=entry["bytes"], deadline=deadline)
            joined = validate_join(decode(raw), cases[key], key, prompt_id, label)
            selection.append({"source": source.alias, "row": evidence + relative,
                              "row_sha256": digest(raw), "row_bytes": len(raw), **joined})
    require(len(selection) == 9 and sum(r["label"] == 1 for r in selection) == 4, "FIXED_SELECTION")
    identities = [(r["prompt_sha256"], r["input_ids_sha256"]) for r in selection]
    require(len(set(identities)) == 9, "DUPLICATE_TRAINING_INPUT")
    return {"schema": "native_supervised_gate_training_manifest_v1", "role": "CONSTRUCTION",
            "historical_role": "public_NATIVE_DEVELOPMENT", "feature_contract": dict(CONTRACT),
            "sources": sources, "selection": selection, "feature_coordinates_emitted": False,
            "real_feature_fits": 0}


def extract_features(manifest, *, deadline):
    """Future authorized extraction only. Never called by authentication or synthetic tests."""
    require(canonical(manifest) == canonical(build_manifest(deadline=deadline)), "MANIFEST_REAUTHENTICATION")
    rows, labels = [], []
    for selected in manifest["selection"]:
        source = next(s for s in SOURCES if s.alias == selected["source"])
        row = decode(checked_blob(source, selected["row"], expected_sha256=selected["row_sha256"],
                                  expected_bytes=selected["row_bytes"], deadline=deadline))
        values = row["h0"]
        require(row["h"] == values and all(type(v) in (int, float) and v == 0 for v in row["offset"]), "UNEDITED_FEATURE")
        import math
        require(all(type(v) in (int, float) and math.isfinite(v) for v in values), "FINITE_FEATURE")
        require(all(struct.unpack("<f", struct.pack("<f", v))[0] == v for v in values), "EXACT_FLOAT32_FEATURE")
        rows.append(tuple(float(v) for v in values))
        labels.append(selected["label"])
    return tuple(rows), tuple(labels)
