"""Changed-path fake recording checks; no model, subprocess, or historical audit."""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import json

import pytest

from scripts import paired_common_drift_comply_recording as recording
from scripts.three_family_recording_budget import Budget, BudgetError


def encoded(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode()


@pytest.fixture
def budget(tmp_path):
    output = tmp_path / "paired_namespace"
    output.mkdir()
    return recording.PairedBudget(output)


def smaller(tmp_path, **quotas):
    output = tmp_path / "paired_namespace"
    output.mkdir()
    return recording.PairedBudget(output, quotas=quotas)


def fake_certificate(root):
    """An isolated source tree; copies only three immutable helper source files."""
    for name in recording.CERTIFICATE_SOURCE_PATHS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in recording.IMMUTABLE_HELPERS or name == recording.POLICY:
            data = (recording.ROOT / name).read_bytes()
        else:
            data = ("synthetic source: " + name + "\n").encode()
        path.write_bytes(data)
    report = {
        "schema": recording.CERTIFICATE_SCHEMA,
        "status": "MODEL_FREE_PREPARATION_CERTIFIED",
        "storage_certified": True,
        "accounting_certified": True,
        "model_loads": 0,
        "tokenizer_loads": 0,
        "real_forwards": 0,
        "real_derivatives": 0,
        "recording_policy_sha256": recording.POLICY_SHA,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in recording.CERTIFICATE_SOURCE_PATHS
        },
    }
    (root / recording.CERTIFICATE).write_bytes(encoded(report))
    return report


def test_exact_policy_arithmetic_and_frozen_helpers():
    policy = recording.recording_policy()
    quotas = policy["quotas"]
    assert quotas == recording.QUOTAS
    assert sum(value for key, value in quotas.items() if key != "total") == 524995016
    assert quotas["total"] == 536870912
    assert quotas["total"] - policy["combined_category_bound_bytes"] == 11875896
    assert (
        sum(
            value
            for key, value in quotas.items()
            if key not in {"logits", "rows", "updates", "total"}
        )
        == 16777216
    )
    assert policy["minimum_free_bytes"] == 1073741824
    assert (policy["maximum_forwards"], policy["maximum_derivatives"]) == (216, 96)
    assert (policy["maximum_updates"], policy["timeout_including_loading_seconds"]) == (8, 1200)
    assert policy["raw_capture"]["process_attempts"] == 1
    assert set(policy["allowed_artifacts"]) == recording.ALLOWED
    assert "endpoint.json" in recording.ALLOWED and "updates.jsonl" in recording.ALLOWED


def source_function(path, name, class_name=None):
    """Read only the named serializer's AST; never import its model runtime."""
    tree = ast.parse((recording.ROOT / path).read_text(encoding="utf-8"))
    scope = tree
    if class_name:
        scope = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    return next(
        node for node in scope.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def assigned_dictionary(function, variable):
    return next(
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == variable for target in node.targets)
        and isinstance(node.value, ast.Dict)
    )


def literal_keys(dictionary):
    assert all(key is None or isinstance(key, ast.Constant) for key in dictionary.keys)
    return {key.value for key in dictionary.keys if key is not None}


def mutated_keys(function, variable):
    keys = set()
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == variable
            and node.func.attr == "update"
        ):
            assert not node.args and all(keyword.arg is not None for keyword in node.keywords)
            keys.update(keyword.arg for keyword in node.keywords)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == variable
                ):
                    assert isinstance(target.slice, ast.Constant)
                    keys.add(target.slice.value)
    return keys


def test_native_row_and_optimizer_serialization_schema_upper_envelopes():
    """Bound serialized shape, not scientific validity or runtime completion.

    Finite binary64 decimal tokens fit in 32 ASCII bytes (at most 17 significant
    digits plus sign, point and three-digit exponent). A 32-byte integer token
    below deliberately fills every numeric vector slot to that conservative
    envelope. Default JSON ', ' separators add two bytes per slot. Every other
    computed field is replaced by a 1024-byte string envelope, including the
    bounded integrity-failure list; fixed prompt metadata is copied verbatim.
    No optimizer function, tensor, tokenizer or model is imported or evaluated.
    """
    slot = -int("9" * 31)
    assert len(json.dumps(slot)) == 32
    native = [slot] * 1024
    scalar_envelope = "x" * 1024
    call = source_function("scripts/shared_comply_two_family.py", "call", "Session")
    score = source_function("src/sp_lense/future_choice_scoring.py", "score_float32_logits")
    row_dictionary = assigned_dictionary(call, "row")
    assert sum(key is None for key in row_dictionary.keys) == 3  # p metadata, cell, scorer
    row_keys = literal_keys(row_dictionary) | mutated_keys(call, "row")
    row_keys |= literal_keys(assigned_dictionary(score, "result"))
    row_vectors = {
        "h0",
        "h",
        "shared_w",
        "intended_delta",
        "actual_delta",
        "gradient",
        "actual_step",
    }
    assert row_vectors <= row_keys and len(row_vectors) == 7
    # These are the only string-valued dynamic row fields: hashes, fixed
    # prompt/cell IDs, canonical logits path, labels, and literal failure names.
    for function in (call, score):
        assert (
            max(
                len(json.dumps(node.value))
                for node in ast.walk(function)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            )
            < 1024
        )
    config = json.loads(
        (recording.ROOT / "configs/paired_common_drift_comply_three_family_v1.json").read_bytes()
    )
    prompts = config["rendered_prompts"]
    assert len(prompts) == 12
    cell_keys = {"cell_id", "prompt_id", "condition", "stage", "optional", "cell_sha256"}
    max_row = 0
    for prompt in prompts:
        assert len(json.dumps(prompt["prompt_id"] + "__gradient_8")) < 1024
        row = {key: value for key, value in prompt.items() if key != "prompt"}
        row.update({key: scalar_envelope for key in row_keys | cell_keys})
        row.update({key: native for key in row_vectors})
        max_row = max(max_row, len(encoded(row)))
    # The complete row metadata/punctuation envelope has >3x spare room.
    assert max_row <= 350000 < recording.RECORD_CAPS["rows.jsonl"]

    optimizer_path = "scripts/paired_common_drift_comply_optimizer.py"
    objective = source_function(optimizer_path, "_objective")
    increment = source_function(optimizer_path, "increment")
    pair_keys = literal_keys(assigned_dictionary(objective, "pair"))
    pair_keys |= mutated_keys(increment, "pair")
    update_dictionary = next(
        node.value
        for node in ast.walk(increment)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    )
    update_keys = literal_keys(update_dictionary)
    pair_vectors = {"J_A", "J_B", "J_c"}
    top_vectors = {"w_before", "g", "d", "s", "u", "w_next", "r", "step", "w_after"}
    assert pair_vectors <= pair_keys and top_vectors <= update_keys
    assert 6 * len(pair_vectors) + len(top_vectors) == 27
    pair = {key: scalar_envelope for key in pair_keys}
    pair.update({key: native for key in pair_vectors})
    update = {key: scalar_envelope for key in update_keys}
    update.update({key: native for key in top_vectors})
    update["pairs"] = [pair] * 6
    update["gradient_cell_ids"] = [scalar_envelope] * 12
    max_update = len(encoded(update))
    assert max_update <= 1200000 < recording.RECORD_CAPS["updates.jsonl"]
    assert 7 * 1024 * 34 == 243712
    assert 27 * 1024 * 34 == 940032


def test_policy_byte_change_rejected_without_model_or_namespace(tmp_path):
    path = tmp_path / recording.POLICY
    path.parent.mkdir(parents=True)
    path.write_bytes((recording.ROOT / recording.POLICY).read_bytes() + b" ")
    before = set(tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="exact paired recording policy bytes"):
        recording.recording_policy(tmp_path)
    assert set(tmp_path.rglob("*")) == before


def test_recording_import_does_not_import_model_or_runtime(monkeypatch):
    seen = []
    original = builtins.__import__
    denied = (
        "torch",
        "transformers",
        "transformer_lens",
        "tokenizers",
        "sp_lense.backend",
        "scripts.paired_common_drift_comply",
        "scripts.paired_common_drift_comply_optimizer",
        "scripts.verify_paired_common_drift_comply",
    )

    def guarded(name, *args, **kwargs):
        seen.append(name)
        if name in denied or name.split(".")[0] in denied[:4]:
            raise AssertionError("recording imported a model/runtime dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location(
        "_paired_recording_import_probe", recording.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PairedBudget.__mro__[1] is Budget
    assert not any(name in denied for name in seen)


def test_controls_exact_and_worker_attach_does_not_create(tmp_path, budget):
    assert (budget.root / "recording_state.json").stat().st_size == 4096
    assert (budget.root / "recording.lock").read_bytes() == b"\0"
    before = {p.name: p.read_bytes() for p in budget.root.iterdir()}
    attached = recording.PairedBudget(budget.root, initialize=False)
    assert attached.status()["phase"] == "RUNNING"
    assert {p.name: p.read_bytes() for p in budget.root.iterdir()} == before
    empty = tmp_path / "empty_attach"
    empty.mkdir()
    with pytest.raises(BudgetError, match="EXISTING_EXACT_CONTROLS_REQUIRED"):
        recording.PairedBudget(empty, initialize=False)
    assert not list(empty.iterdir())


@pytest.mark.parametrize("index", [1, 99, 100, 216])
def test_canonical_raw_arrays_at_two_and_three_digit_boundaries(budget, index):
    (budget.root / "logits").mkdir()
    name = f"logits/{index:02d}.f32.zlib"
    budget.write_bytes(name, b"fake raw compressed payload")
    assert (budget.root / name).read_bytes() == b"fake raw compressed payload"
    assert budget.category(name) == "logits"


@pytest.mark.parametrize(
    "name",
    [
        "logits/00.f32.zlib",
        "logits/001.f32.zlib",
        "logits/217.f32.zlib",
        "objective_events.jsonl",
        "preserve_vector.json",
        "checksums.json",
        "unknown.json",
    ],
)
def test_extra_artifacts_denied_and_fault_sticky(budget, name):
    (budget.root / "logits").mkdir()
    with pytest.raises(BudgetError, match="PAIRED_ARTIFACT_SCOPE"):
        budget.write_bytes(name, b"must not be admitted")
    assert not (budget.root / name).exists()
    assert budget.fault_code == "PAIRED_ARTIFACT_SCOPE"
    with pytest.raises(BudgetError, match="NORMAL_RECORDING_NOT_ALLOWED"):
        budget.write_bytes("rows.jsonl", encoded({"blocked": True}), mode="ab")


@pytest.mark.parametrize("key", ["rows", "updates", "total", "workerlog"])
def test_quota_cannot_expand(tmp_path, key):
    root = tmp_path / "expand"
    root.mkdir()
    with pytest.raises(BudgetError, match="PAIRED_QUOTA_CANNOT_EXPAND"):
        recording.PairedBudget(root, {key: recording.QUOTAS[key] + 1})
    assert not list(root.iterdir())


@pytest.mark.parametrize("name", ["rows.jsonl", "updates.jsonl"])
def test_complete_serialized_record_cap_before_append(budget, name):
    initial = encoded({"kind": "small complete record", "values": [0.1, -0.2]})
    budget.write_bytes(name, initial, mode="ab")
    cap = recording.RECORD_CAPS[name]
    oversized = encoded({"payload": "x" * cap})
    assert len(oversized) > cap
    with pytest.raises(BudgetError, match="PAIRED_RECORD_BYTE_CAP"):
        budget.write_bytes(name, oversized, mode="ab")
    assert (budget.root / name).read_bytes() == initial
    assert budget.fault_code == "PAIRED_RECORD_BYTE_CAP"
    budget.begin_finalization(quiescent=True)
    with pytest.raises(BudgetError, match="CANDIDATE_AFTER_RECORDING_FAULT"):
        budget.write_bytes("comply_vector.json", b"{}", final=True)
    inventory = budget.finalize_inventory(valid_candidate=True, quiescent=True)
    assert inventory["fault_code"] == "PAIRED_RECORD_BYTE_CAP"
    assert inventory["valid_candidate"] is False


@pytest.mark.parametrize("name", ["rows.jsonl", "updates.jsonl"])
def test_whole_record_cap_uses_memoryview_bytes_not_element_count(budget, name):
    data = bytearray(recording.RECORD_CAPS[name] + 8)
    words = memoryview(data).cast("Q")
    assert len(words) < recording.RECORD_CAPS[name] < words.nbytes
    with pytest.raises(BudgetError, match="PAIRED_RECORD_BYTE_CAP"):
        budget.write_bytes(name, words, mode="ab")
    assert not (budget.root / name).exists()


def test_aggregate_rows_and_updates_caps_preserve_prior_complete_record(tmp_path):
    budget = smaller(tmp_path, rows=16, updates=16)
    payload = encoded({"v": 0})
    assert len(payload) <= 16 < 2 * len(payload)
    budget.write_bytes("rows.jsonl", payload, mode="ab")
    budget.write_bytes("updates.jsonl", payload, mode="ab")
    with pytest.raises(BudgetError, match="ARTIFACT_BYTE_CAP"):
        budget.write_bytes("updates.jsonl", payload, mode="ab")
    assert (budget.root / "updates.jsonl").read_bytes() == payload
    assert (budget.root / "rows.jsonl").read_bytes() == payload


def test_normal_writers_cannot_spend_final_or_receipt_reserves(budget):
    with pytest.raises(BudgetError, match="NORMAL_RECORDING_NOT_ALLOWED"):
        budget.write_bytes("capture_receipt.json", encoded({"status": "premature"}))
    assert not (budget.root / "capture_receipt.json").exists()
    budget.begin_finalization(quiescent=True)
    budget.write_bytes("capture_receipt.json", encoded({"status": "INCONCLUSIVE"}), final=True)
    with pytest.raises(BudgetError, match="CANDIDATE_AFTER_RECORDING_FAULT"):
        budget.write_bytes("candidate_freeze.json", encoded({"candidate": True}), final=True)


def test_fake_complete_construction_inventory_and_no_late_writes(budget):
    budget.write_bytes("preregistration.json", encoded({"synthetic": True}))
    budget.write_bytes("rows.jsonl", encoded({"saved": "fake row"}), mode="ab")
    budget.write_bytes("updates.jsonl", encoded({"saved": "fake update"}), mode="ab")
    budget.write_bytes("endpoint.json", encoded({"w": [0.01, -0.02]}))
    budget.write_bytes("result.json", encoded({"accepted": True}))
    budget.begin_finalization(quiescent=True)
    budget.write_bytes("capture_receipt.json", encoded({"quiescent": True}), final=True)
    budget.write_bytes("verification.json", encoded({"synthetic_audit": True}), final=True)
    budget.write_bytes("comply_vector.json", encoded({"vector": [0.01, -0.02]}), final=True)
    budget.write_bytes("candidate_freeze.json", encoded({"synthetic": True}), final=True)
    budget.write_bytes("PILOT_REPORT.md", b"Synthetic fixture only.\n", final=True)
    inventory = budget.finalize_inventory(valid_candidate=True, quiescent=True)
    assert inventory["valid_candidate"] is True
    assert inventory["fault_code"] is None and inventory["quiescent"] is True
    assert inventory["total_bytes"] == sum(p.stat().st_size for p in budget.root.iterdir())
    own = next(item for item in inventory["files"] if item["path"] == "FINAL_INVENTORY.json")
    assert own["bytes"] == (budget.root / "FINAL_INVENTORY.json").stat().st_size
    assert own["sha256"] is None
    before = {p.name: p.read_bytes() for p in budget.root.iterdir()}
    checked = recording.PairedBudget(budget.root, initialize=False).verify_inventory()
    assert checked["status"] == "SEALED_INVENTORY_VERIFIED"
    assert checked["inventory"] == inventory
    with pytest.raises(BudgetError, match="RECORDING_SEALED"):
        budget.write_bytes("CLOSEOUT.json", encoded({"late": True}), final=True)
    assert {p.name: p.read_bytes() for p in budget.root.iterdir()} == before


def test_unquiescent_failure_cannot_seal_or_create_candidate(budget):
    assert budget.begin_finalization(quiescent=False) is False
    budget.write_bytes("RECORDING_FAILURE.json", encoded({"quiescent": False}), final=True)
    assert budget.status()["phase"] == "FAILED"
    assert budget.fault_code == "QUIESCENCE_UNCONFIRMED"
    with pytest.raises(BudgetError):
        budget.write_bytes("comply_vector.json", b"{}", final=True)
    with pytest.raises(BudgetError, match="QUIESCENT_FINALIZATION_OWNER_REQUIRED"):
        budget.finalize_inventory(valid_candidate=True, quiescent=False)
    assert not (budget.root / "FINAL_INVENTORY.json").exists()


def test_scoped_native_path_io_stays_inside_same_budget(budget, tmp_path):
    outside = tmp_path / "outside.txt"
    with budget.scoped_writes():
        (budget.root / "worker.log").write_bytes(b"captured prefix\n")
        (budget.root / "derivative_events.jsonl").write_text("", encoding="utf-8")
        outside.write_text("outside unchanged", encoding="utf-8")
        assert outside.read_text() == "outside unchanged"
    assert (budget.root / "worker.log").read_bytes() == b"captured prefix\n"
    assert budget.snapshot()["sizes"]["derivative_events.jsonl"] == 0


def test_exact_positive_source_certificate_and_report_binding(tmp_path):
    report = fake_certificate(tmp_path)
    assert recording.require_certificate(tmp_path) == report
    assert set(report["source_sha256"]) == recording.CERTIFICATE_SOURCE_PATHS
    assert len(report["source_sha256"]) == 19
    assert "tests/test_local_controllability_positive_control.py" in report["source_sha256"]
    assert recording.PREPARATION_REPORT in report["source_sha256"]
    assert recording.CERTIFICATE not in report["source_sha256"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "dirty", "wrong_helper"])
def test_certificate_exact_source_set_or_bytes_required(tmp_path, mutation):
    report = fake_certificate(tmp_path)
    path = "scripts/paired_common_drift_comply.py"
    if mutation == "missing":
        del report["source_sha256"][path]
    elif mutation == "extra":
        report["source_sha256"]["unexpected.json"] = "0" * 64
    elif mutation == "dirty":
        (tmp_path / path).write_bytes(b"changed source")
    else:
        helper = next(iter(recording.IMMUTABLE_HELPERS))
        report["source_sha256"][helper] = "0" * 64
    (tmp_path / recording.CERTIFICATE).write_bytes(encoded(report))
    with pytest.raises(ValueError, match="positive exact-source paired"):
        recording.require_certificate(tmp_path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("status", "PREPARATION_INCOMPLETE"),
        ("storage_certified", False),
        ("accounting_certified", False),
        ("model_loads", 1),
        ("tokenizer_loads", 1),
        ("real_forwards", 1),
        ("real_derivatives", 1),
        ("model_loads", False),
    ],
)
def test_certificate_cannot_promote_incomplete_or_nonzero_model_work(tmp_path, key, value):
    report = fake_certificate(tmp_path)
    report[key] = value
    (tmp_path / recording.CERTIFICATE).write_bytes(encoded(report))
    with pytest.raises(ValueError, match="positive exact-source paired"):
        recording.require_certificate(tmp_path)


def test_missing_certificate_not_created_or_repaired(tmp_path):
    report = fake_certificate(tmp_path)
    certificate = tmp_path / recording.CERTIFICATE
    certificate.unlink()
    before = set(tmp_path.rglob("*"))
    with pytest.raises(FileNotFoundError):
        recording.require_certificate(tmp_path)
    assert set(tmp_path.rglob("*")) == before
    assert report["model_loads"] == 0
