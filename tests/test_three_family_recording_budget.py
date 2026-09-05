"""Small, model-free disk/phase/fault/locking tests; no solver benchmark."""

from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import three_family_recording_budget as recording


@pytest.fixture
def budget(tmp_path):
    root = tmp_path / "namespace"
    root.mkdir()
    return recording.Budget(root)


def small(tmp_path, **quotas):
    root = tmp_path / "namespace"
    root.mkdir()
    return recording.Budget(root, quotas)


def test_fixed_controls_and_original_category_allocations(budget):
    assert (budget.root / recording.STATE_NAME).stat().st_size == 4096
    assert (budget.root / recording.LOCK_NAME).read_bytes() == b"\0"
    assert budget.status()["phase"] == "RUNNING"
    assert budget.fault_code is None
    assert recording.DEFAULT_QUOTAS["logits"] == 214616520
    assert recording.DEFAULT_QUOTAS["rows"] == 226492416
    assert recording.DEFAULT_QUOTAS["updates"] == 67108864
    assert (
        sum(
            value
            for key, value in recording.DEFAULT_QUOTAS.items()
            if key not in {"logits", "rows", "updates", "total"}
        )
        == 16 * 1024**2
    )
    assert recording.DEFAULT_QUOTAS["total"] == 512 * 1024**2


def test_scoped_text_bytes_and_outside_operations_unchanged(budget, tmp_path):
    original = builtins.open, io.open, os.open, os.replace
    outside = tmp_path / "outside.txt"
    with budget.scoped_writes():
        with (budget.root / "rows.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            assert stream.write("é\n") == 2
            stream.flush()
        with builtins.open(budget.root / "updates.jsonl", "xb") as stream:
            stream.write(b"raw")
        outside.write_text("outside", encoding="utf-8")
        assert outside.read_text() == "outside"
        assert (budget.root / "rows.jsonl").read_bytes() == "é\n".encode()
    assert (builtins.open, io.open, os.open, os.replace) == original
    assert (budget.root / "updates.jsonl").read_bytes() == b"raw"


def test_nested_scopes_restore_only_after_last_exit(budget):
    original = io.open
    with budget.scoped_writes():
        with budget.scoped_writes():
            assert io.open is recording._dispatch_open
        assert io.open is recording._dispatch_open
    assert io.open is original


@pytest.mark.parametrize(
    "name",
    [
        "unknown.json",
        "../escape.json",
        "logits/00.f32.zlib",
        "logits/217.f32.zlib",
        "logits/001.f32.zlib",
    ],
)
def test_unknown_and_escaping_artifacts_fail_before_data(budget, name):
    with pytest.raises(recording.BudgetError):
        budget.write_bytes(name, b"no")
    assert budget.fault_code is not None
    assert not (budget.root.parent / "escape.json").exists()


@pytest.mark.parametrize("name", ["recording_state.json", "recording.lock", "FINAL_INVENTORY.json"])
def test_control_artifacts_not_public_writers(budget, name):
    original = (budget.root / name).read_bytes() if (budget.root / name).exists() else None
    with pytest.raises(recording.BudgetError):
        budget.write_bytes(name, b"overwrite", mode="ab")
    if name == "recording.lock":
        assert (budget.root / name).read_bytes() == original
    assert (budget.root / recording.STATE_NAME).stat().st_size == 4096


def test_complete_write_rejected_at_file_cap_and_sticky_fault(tmp_path):
    budget = small(tmp_path, workerlog=8)
    budget.write_bytes("worker.log", b"123456")
    with pytest.raises(recording.BudgetError, match="ARTIFACT_BYTE_CAP"):
        budget.write_bytes("worker.log", b"789", mode="ab")
    assert (budget.root / "worker.log").read_bytes() == b"123456"
    assert budget.fault_code == "ARTIFACT_BYTE_CAP"
    budget.fault("LATER_FAULT")
    assert budget.fault_code == "ARTIFACT_BYTE_CAP"
    with pytest.raises(recording.BudgetError):
        budget.write_bytes("rows.jsonl", b"blocked")


def test_aggregate_cap_counts_controls_and_rejects_no_partial_normal_data(tmp_path):
    budget = small(tmp_path, total=4101)
    budget.write_bytes("rows.jsonl", b"1234")
    with pytest.raises(recording.BudgetError, match="NAMESPACE_BYTE_CAP"):
        budget.write_bytes("rows.jsonl", b"5", mode="ab")
    assert sum(budget.snapshot()["sizes"].values()) == 4101


def test_category_shared_across_files(tmp_path):
    budget = small(tmp_path, metadata=8)
    budget.write_bytes("runtime.json", b"12345")
    with pytest.raises(recording.BudgetError, match="CATEGORY_BYTE_CAP"):
        budget.write_bytes("result.json", b"6789")
    assert (budget.root / "result.json").read_bytes() == b""


def test_final_and_receipt_reserves_not_available_to_normal_writers(budget):
    with pytest.raises(recording.BudgetError, match="NORMAL_RECORDING_NOT_ALLOWED"):
        budget.write_bytes("capture_receipt.json", b"not yet")
    budget.begin_finalization(quiescent=True)
    budget.write_bytes("capture_receipt.json", b"fault receipt", final=True)
    budget.write_bytes("RECORDING_FAILURE.json", b"fault", final=True)
    with pytest.raises(recording.BudgetError, match="CANDIDATE_AFTER_RECORDING_FAULT"):
        budget.write_bytes("comply_vector.json", b"forbidden", final=True)


@pytest.mark.parametrize("operation", ["fileno", "seek", "truncate"])
def test_writer_descriptor_seek_truncate_denied(budget, operation):
    with budget.open_writer("rows.jsonl") as stream:
        stream.write(b"abc")
        with pytest.raises(recording.BudgetError):
            getattr(stream, operation)(*([0] if operation != "fileno" else []))
    assert (budget.root / "rows.jsonl").read_bytes() == b"abc"


@pytest.mark.parametrize("mode", ["r+b", "a+b", "w+b"])
def test_update_modes_denied(budget, mode):
    with pytest.raises(recording.BudgetError, match="UNBOUNDED_WRITER_MODE"):
        budget.open_writer("rows.jsonl", mode)


def test_w_mode_never_silently_overwrites_existing_raw(budget):
    budget.write_bytes("rows.jsonl", b"keep")
    with pytest.raises(recording.BudgetError, match="OVERWRITE_DENIED"):
        budget.open_writer("rows.jsonl", "wb")
    assert (budget.root / "rows.jsonl").read_bytes() == b"keep"


def test_stale_nonappend_offset_rejected_before_overwriting(budget):
    with budget.open_writer("rows.jsonl", "xb") as old:
        old.write(b"first")
        budget.write_bytes("rows.jsonl", b"second", mode="ab")
        with pytest.raises(recording.BudgetError, match="STALE_WRITER_OFFSET"):
            old.write(b"bad")
    assert (budget.root / "rows.jsonl").read_bytes() == b"firstsecond"


def test_two_persistent_appenders_do_not_overwrite(budget):
    with (
        budget.open_writer("rows.jsonl", "ab") as first,
        budget.open_writer("rows.jsonl", "ab") as second,
    ):
        first.write(b"a")
        second.write(b"b")
        first.write(b"c")
    assert (budget.root / "rows.jsonl").read_bytes() == b"abc"


class StreamProxy:
    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)


def test_partial_disk_write_is_preserved_and_rescanned(budget):
    class Partial(StreamProxy):
        def write(self, data):
            self.stream.write(data[:2])
            raise OSError("arbitrarily long error must not be serialized" * 1000)

    with budget.open_writer("rows.jsonl", "xb") as writer:
        writer.stream = Partial(writer.stream)
        with pytest.raises(recording.BudgetError, match="RECORDING_WRITE_FAILURE"):
            writer.write(b"abcdef")
    assert (budget.root / "rows.jsonl").read_bytes() == b"ab"
    assert budget.snapshot()["sizes"]["rows.jsonl"] == 2
    assert budget.fault_code == "RECORDING_WRITE_FAILURE"
    assert (budget.root / recording.STATE_NAME).stat().st_size == 4096


def test_fault_and_failed_phase_do_not_wait_for_blocked_data_writer(budget):
    entered, release = threading.Event(), threading.Event()
    errors = []

    class Blocked(StreamProxy):
        def write(self, data):
            entered.set()
            if not release.wait(3):
                raise OSError("test release timeout")
            return self.stream.write(data)

    with budget.open_writer("worker.log", "ab") as writer:
        writer.stream = Blocked(writer.stream)

        def writing():
            try:
                writer.write(b"prefix")
            except recording.BudgetError as error:
                errors.append(error.code)

        thread = threading.Thread(target=writing)
        thread.start()
        try:
            assert entered.wait(2)
            start = time.monotonic()
            assert budget.fault("CAPTURE_TIMEOUT") == "CAPTURE_TIMEOUT"
            assert budget.fault_code == "CAPTURE_TIMEOUT"
            assert budget.begin_finalization(quiescent=False) is False
            assert budget.status()["phase"] == "FAILED"
            assert time.monotonic() - start < 1
        finally:
            release.set()
            thread.join(3)
        assert not thread.is_alive()
    budget.write_bytes("capture_receipt.json", b"unjoined-at-timeout", final=True)
    with pytest.raises(recording.BudgetError, match="QUIESCENT_FINALIZATION_OWNER_REQUIRED"):
        budget.finalize_inventory(valid_candidate=True, quiescent=True)
    assert not (budget.root / recording.INVENTORY_NAME).exists()


def test_explicit_quiescence_and_live_local_handles_block_valid_seal(budget):
    with (
        budget.open_writer("rows.jsonl"),
        pytest.raises(recording.BudgetError, match="QUIESCENCE_REQUIRED"),
    ):
        budget.begin_finalization(quiescent=True)
    assert budget.status()["phase"] == "RUNNING"
    budget.begin_finalization(quiescent=True)
    receipt = budget.finalize_inventory(valid_candidate=True)
    assert receipt["valid_candidate"] is False


def test_unconfirmed_quiescence_allows_only_failure_receipts_no_inventory(budget):
    assert budget.begin_finalization(quiescent=False) is False
    budget.write_bytes("RUN_STATUS.json", b"INCONCLUSIVE", final=True)
    budget.write_bytes("RECORDING_FAILURE.json", b"unconfirmed", final=True)
    budget.write_bytes("capture_receipt.json", b"not joined", final=True)
    with pytest.raises(recording.BudgetError):
        budget.write_bytes("comply_vector.json", b"candidate", final=True)
    with pytest.raises(recording.BudgetError):
        budget.finalize_inventory(valid_candidate=False)


def test_finalizing_writes_require_exact_owner_instance(budget):
    budget.begin_finalization(quiescent=True)
    peer = recording.Budget(budget.root)
    with pytest.raises(recording.BudgetError, match="FINALIZATION_OWNER_REQUIRED"):
        peer.write_bytes("verification.json", b"peer", final=True)
    budget.write_bytes("RECORDING_FAILURE.json", b"owner", final=True)


def test_staging_overlap_counts_before_explicit_replace(tmp_path):
    budget = small(tmp_path, metadata=10)
    budget.write_bytes("runtime.json", b"12345")
    budget.write_bytes(".runtime.json.tmp", b"67890")
    assert budget.snapshot()["sizes"][".runtime.json.tmp"] == 5
    budget.replace(".runtime.json.tmp", "runtime.json")
    assert (budget.root / "runtime.json").read_bytes() == b"67890"
    assert not (budget.root / ".runtime.json.tmp").exists()


def test_staging_cannot_borrow_target_bytes_before_replace(tmp_path):
    budget = small(tmp_path, metadata=8)
    budget.write_bytes("runtime.json", b"12345")
    with pytest.raises(recording.BudgetError, match="CATEGORY_BYTE_CAP"):
        budget.write_bytes(".runtime.json.tmp", b"67890")
    assert (budget.root / "runtime.json").read_bytes() == b"12345"


@pytest.mark.parametrize(
    "action",
    ["replace", "rename", "unlink", "raw_open", "replace_keyword", "unlink_keyword", "link"],
)
def test_uninstrumented_namespace_mutations_denied(budget, action):
    budget.write_bytes("runtime.json", b"keep")
    with budget.scoped_writes(), pytest.raises(recording.BudgetError):
        if action in {"replace", "rename"}:
            getattr(os, action)(budget.root / "runtime.json", budget.root / "result.json")
        elif action == "unlink":
            os.unlink(budget.root / "runtime.json")
        elif action == "replace_keyword":
            os.replace(src=budget.root / "runtime.json", dst=budget.root / "result.json")
        elif action == "unlink_keyword":
            os.unlink(path=budget.root / "runtime.json")
        elif action == "link":
            os.link(budget.root / "runtime.json", budget.root / "result.json")
        else:
            os.open(budget.root / "runtime.json", os.O_WRONLY)
    assert (budget.root / "runtime.json").read_bytes() == b"keep"


def test_hardlinked_output_denied_without_outside_mutation(budget, tmp_path):
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    os.link(outside, budget.root / "rows.jsonl")
    with pytest.raises(recording.BudgetError, match="UNSAFE_OUTPUT_LINK"):
        budget.write_bytes("rows.jsonl", b"bad", mode="ab")
    assert outside.read_bytes() == b"outside"


def test_symlink_escape_denied_without_outside_mutation(budget, tmp_path):
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    try:
        os.symlink(outside, budget.root / "rows.jsonl")
    except OSError:
        pytest.skip("platform does not permit creating test symlink")
    with budget.scoped_writes(), pytest.raises(recording.BudgetError):
        (budget.root / "rows.jsonl").write_bytes(b"bad")
    assert outside.read_bytes() == b"outside"


def test_final_inventory_seals_state_before_hash_and_includes_itself(budget):
    budget.write_bytes("rows.jsonl", b"raw")
    budget.begin_finalization(quiescent=True)
    budget.write_bytes("verification.json", b"verified", final=True)
    budget.write_bytes("PILOT_REPORT.md", b"report", final=True)
    result = budget.finalize_inventory(valid_candidate=False)
    actual = {
        path.relative_to(budget.root).as_posix(): path.read_bytes()
        for path in budget.root.iterdir()
        if path.is_file()
    }
    assert result["total_bytes"] == sum(map(len, actual.values()))
    entries = {entry["path"]: entry for entry in result["files"]}
    assert set(entries) == set(actual)
    for name, raw in actual.items():
        assert entries[name]["bytes"] == len(raw)
        assert entries[name]["sha256"] == (
            None if name == recording.INVENTORY_NAME else hashlib.sha256(raw).hexdigest()
        )
    assert json.loads(actual[recording.STATE_NAME])["phase"] == "SEALED"
    assert result["inventory_self_hash"] is None
    assert budget.verify_inventory() == {"status": "SEALED_INVENTORY_VERIFIED", "inventory": result}
    with pytest.raises(recording.BudgetError):
        budget.write_bytes("capture_receipt.json", b"late", final=True)
    assert actual == {
        path.name: path.read_bytes() for path in budget.root.iterdir() if path.is_file()
    }


@pytest.mark.parametrize("tamper", ["size", "same_size", "new_file", "self"])
def test_readonly_inventory_verifier_rejects_postseal_tampering(budget, tamper):
    budget.write_bytes("rows.jsonl", b"raw")
    budget.begin_finalization(quiescent=True)
    budget.finalize_inventory()
    if tamper == "size":
        (budget.root / "rows.jsonl").write_bytes(b"longer")
    elif tamper == "same_size":
        (budget.root / "rows.jsonl").write_bytes(b"bad")
    elif tamper == "new_file":
        (budget.root / "unknown.json").write_bytes(b"new")
    else:
        path = budget.root / recording.INVENTORY_NAME
        value = json.loads(path.read_bytes())
        value["total_bytes"] += 1
        path.write_text(json.dumps(value))
    with pytest.raises(recording.BudgetError):
        budget.verify_inventory()


def test_invalid_fixed_state_fails_closed_without_repair(budget):
    path = budget.root / recording.STATE_NAME
    path.write_bytes(b"broken")
    with pytest.raises(recording.BudgetError, match="INVALID_RECORDING_STATE"):
        budget.write_bytes("rows.jsonl", b"raw")
    assert path.read_bytes() == b"broken"


@pytest.mark.parametrize("missing", ["recording_state.json", "recording.lock"])
def test_attach_and_default_never_repair_missing_control(budget, missing):
    (budget.root / missing).unlink()
    before = {path.name: path.read_bytes() for path in budget.root.iterdir()}
    for initialize in (False, True):
        with pytest.raises(recording.BudgetError, match="EXISTING_EXACT_CONTROLS_REQUIRED"):
            recording.Budget(budget.root, initialize=initialize)
    assert before == {path.name: path.read_bytes() for path in budget.root.iterdir()}


def test_readonly_attach_no_initialization_on_absent_controls(tmp_path):
    root = tmp_path / "namespace"
    root.mkdir()
    with pytest.raises(recording.BudgetError, match="EXISTING_EXACT_CONTROLS_REQUIRED"):
        recording.Budget(root, initialize=False)
    assert list(root.iterdir()) == []


def test_attach_never_recreates_state_deleted_after_initial_check(budget, monkeypatch):
    real_locked = recording.Budget._locked

    @contextlib.contextmanager
    def disappearing_state(self, initialize=False):
        with real_locked(self, initialize=initialize):
            (self.root / recording.STATE_NAME).unlink()
            yield

    monkeypatch.setattr(recording.Budget, "_locked", disappearing_state)
    with pytest.raises((recording.BudgetError, OSError)):
        recording.Budget(budget.root, initialize=False)
    assert not (budget.root / recording.STATE_NAME).exists()


def test_postseal_hash_fault_aborts_sealing_and_allows_only_failure_receipts(budget, monkeypatch):
    budget.write_bytes("rows.jsonl", b"raw")
    budget.begin_finalization(quiescent=True)

    def failed_hash(name):
        raise OSError("injected read failure")

    monkeypatch.setattr(budget, "_hash_file", failed_hash)
    with pytest.raises(recording.BudgetError, match="FINAL_INVENTORY_WRITE_FAILURE"):
        budget.finalize_inventory(valid_candidate=True)
    assert budget.status()["phase"] == "FAILED"
    assert budget.fault_code == "FINAL_INVENTORY_WRITE_FAILURE"
    assert not (budget.root / recording.INVENTORY_NAME).exists()
    budget.write_bytes("RECORDING_FAILURE.json", b"hash failure", final=True)


def test_partial_inventory_write_preserves_prefix_and_denies_valid_seal(budget, monkeypatch):
    budget.write_bytes("rows.jsonl", b"raw")
    budget.begin_finalization(quiescent=True)
    real_open = recording._OPEN

    class PartialInventory(StreamProxy):
        def write(self, data):
            self.stream.write(data[:7])
            raise OSError("injected inventory write failure")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

    def intercepted(path, mode="r", *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        return (
            PartialInventory(stream)
            if Path(path).name == recording.INVENTORY_NAME and mode == "xb"
            else stream
        )

    monkeypatch.setattr(recording, "_OPEN", intercepted)
    with pytest.raises(recording.BudgetError, match="FINAL_INVENTORY_WRITE_FAILURE"):
        budget.finalize_inventory(valid_candidate=True)
    assert (budget.root / recording.INVENTORY_NAME).stat().st_size == 7
    assert budget.status()["phase"] == "FAILED"
    assert budget.fault_code == "FINAL_INVENTORY_WRITE_FAILURE"
    with pytest.raises(recording.BudgetError, match="SEALED_INVENTORY_REQUIRED"):
        budget.verify_inventory()


def test_crossprocess_append_quota_is_shared_and_partial_lines_not_silently_written(tmp_path):
    budget = small(tmp_path, rows=32)
    code = """
import json,sys
from scripts.three_family_recording_budget import Budget,BudgetError
b=Budget(sys.argv[1],{'rows':32})
n=0
for i in range(12):
    try:
        b.write_bytes('rows.jsonl',b'12345678',mode='ab')
        n+=1
    except BudgetError:
        break
print(json.dumps({'accepted':n}))
"""
    children = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(budget.root)],
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for _ in range(3)
    ]
    records = []
    try:
        for process in children:
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stderr
            records.append(json.loads(stdout))
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)
    assert sum(record["accepted"] for record in records) == 4
    assert (budget.root / "rows.jsonl").read_bytes() == b"12345678" * 4
    assert budget.fault_code is not None


def test_posix_descriptor_lifetime_covers_all_active_regions(budget, monkeypatch):
    # Platform-independent lifetime test, not a claim of executing POSIX kernel locks.
    path = budget.root / recording.LOCK_NAME
    with monkeypatch.context() as patch:
        patch.setattr(recording, "os", SimpleNamespace(name="posix"))
        data = recording._borrow_lock_stream(path, False)
        state = recording._borrow_lock_stream(path, False)
        assert data is state
        recording._return_lock_stream(path, state)
        assert not data.closed
        recording._return_lock_stream(path, data)
        assert data.closed
    assert str(path) not in recording._POSIX_LOCK_STREAMS
