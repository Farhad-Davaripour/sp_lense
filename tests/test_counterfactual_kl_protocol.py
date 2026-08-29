from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import sp_lense.counterfactual_kl_protocol as protocol

RESULT_SCHEMA = "sp_lense.ckes_validation_result.test.v1"
REQUIRED_GATES = ("efficacy", "integrity", "safety")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _build_lock(sealed_path: Path) -> dict[str, object]:
    file_hashes = {
        "renderer": {"path": "src/example_renderer.py", "sha256": "1" * 64},
        "sealed_dataset": {
            "path": "private/example_sealed.json",
            "sha256": protocol.file_sha256(sealed_path),
            "bytes": sealed_path.stat().st_size,
        },
        "validation_dataset": {
            "path": "data/example_validation.json",
            "sha256": "2" * 64,
        },
    }
    rendered_manifests = {
        "validation": {
            "form_count": 80,
            "form_ids_sha256": "3" * 64,
            "prompt_manifest_sha256": "4" * 64,
        },
        "sealed": {
            "form_count": 80,
            "form_ids_sha256": "5" * 64,
            "prompt_manifest_sha256": "6" * 64,
        },
    }
    configuration = {
        "model": {
            "id": "synthetic/model",
            "revision": "7" * 40,
            "device": "cpu",
            "dtype": "float32",
        },
        "layer": 0,
        "position": "shared_prefix_anchor",
    }
    thresholds = {
        "actual_kl": {"mean": 0.005, "p95": 0.02, "max": 0.05},
        "lookahead_epsilon": 1.0 / 32.0,
        "trial_cap": 24,
    }
    return protocol.build_prospective_lock(
        file_hashes=file_hashes,
        rendered_manifests=rendered_manifests,
        configuration=configuration,
        thresholds=thresholds,
        sealed_dataset_file_key="sealed_dataset",
        validation_result_schema_version=RESULT_SCHEMA,
        required_validation_gates=REQUIRED_GATES,
    )


def _validation_result(
    lock: dict[str, object],
    *,
    status: str = "go",
    gates: dict[str, object] | None = None,
    lock_identity_sha256: str | None = None,
    split: str = "validation",
    dataset_file_sha256: str | None = None,
) -> dict[str, object]:
    return protocol.self_hash_record(
        {
            "schema_version": RESULT_SCHEMA,
            "lock_identity_sha256": (
                lock["lock_identity_sha256"]
                if lock_identity_sha256 is None
                else lock_identity_sha256
            ),
            "status": status,
            "split": split,
            "dataset_file_sha256": (
                lock["file_hashes"]["validation_dataset"]["sha256"]
                if dataset_file_sha256 is None
                else dataset_file_sha256
            ),
            "gates": ({name: True for name in REQUIRED_GATES} if gates is None else gates),
            "machine_readable_summary": {"synthetic": True},
        }
    )


def test_lock_is_canonical_and_order_independent(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed.json"
    _write_json(sealed, {"split": "sealed", "records": [{"id": "secret"}]})
    first = _build_lock(sealed)
    access = first["sealed_access"]
    second = protocol.build_prospective_lock(
        file_hashes=dict(reversed(list(first["file_hashes"].items()))),
        rendered_manifests=dict(reversed(list(first["rendered_manifests"].items()))),
        configuration=dict(reversed(list(first["configuration"].items()))),
        thresholds=dict(reversed(list(first["thresholds"].items()))),
        sealed_dataset_file_key=access["sealed_dataset_file_key"],
        validation_dataset_file_key=access["validation_dataset_file_key"],
        validation_result_schema_version=access["validation_result_schema_version"],
        required_validation_gates=list(reversed(REQUIRED_GATES)),
    )
    assert first == second
    assert protocol.verify_prospective_lock(first) == first
    assert first["model_compute_used_to_build_lock"] == 0


def test_lock_tampering_fails_self_hash_and_component_checks(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed.json"
    _write_json(sealed, {"split": "sealed"})
    lock = _build_lock(sealed)

    tampered = copy.deepcopy(lock)
    tampered["thresholds"]["trial_cap"] = 25
    with pytest.raises(protocol.CounterfactualKLProtocolIntegrityError, match="identity differs"):
        protocol.verify_prospective_lock(tampered)

    recomputed_identity_only = copy.deepcopy(tampered)
    unhashed = dict(recomputed_identity_only)
    del unhashed["lock_identity_sha256"]
    recomputed_identity_only["lock_identity_sha256"] = protocol.canonical_sha256(unhashed)
    with pytest.raises(
        protocol.CounterfactualKLProtocolIntegrityError, match="component hash differs"
    ):
        protocol.verify_prospective_lock(recomputed_identity_only)

    extra_field = copy.deepcopy(lock)
    extra_field["unlocked_note"] = "not part of the schema"
    unhashed = dict(extra_field)
    del unhashed["lock_identity_sha256"]
    extra_field["lock_identity_sha256"] = protocol.canonical_sha256(unhashed)
    with pytest.raises(protocol.CounterfactualKLProtocolIntegrityError, match="fields differ"):
        protocol.verify_prospective_lock(extra_field)


def test_no_go_never_deserializes_sealed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = tmp_path / "sealed.json"
    result_path = tmp_path / "validation_result.json"
    _write_json(sealed, {"split": "sealed", "secret_prompt": "not parsed"})
    lock = _build_lock(sealed)
    _write_json(result_path, _validation_result(lock, status="development_no_go"))

    calls = 0

    def forbidden_parser(value: bytes) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise AssertionError("sealed JSON parser must not run before authorization")

    monkeypatch.setattr(protocol, "_parse_sealed_json_bytes", forbidden_parser)
    with pytest.raises(protocol.SealedAccessDenied, match="authorized go"):
        protocol.load_sealed_dataset(sealed, result_path, lock=lock)
    assert calls == 0


@pytest.mark.parametrize(
    ("result_factory", "message"),
    [
        (
            lambda lock: _validation_result(lock, lock_identity_sha256="a" * 64),
            "exact lock",
        ),
        (
            lambda lock: _validation_result(
                lock,
                gates={"efficacy": True, "integrity": True, "safety": False},
            ),
            "not every",
        ),
        (
            lambda lock: _validation_result(lock, gates={"efficacy": True, "integrity": True}),
            "gates differ",
        ),
        (
            lambda lock: _validation_result(lock, split="sealed"),
            "locked split",
        ),
        (
            lambda lock: _validation_result(lock, dataset_file_sha256="f" * 64),
            "dataset hash",
        ),
        (
            lambda lock: _validation_result(
                lock,
                gates={
                    "efficacy": True,
                    "integrity": True,
                    "safety": True,
                    "unlocked_extra": True,
                },
            ),
            "gates differ",
        ),
        (
            lambda lock: _validation_result(
                lock,
                gates={"efficacy": 1, "integrity": True, "safety": True},
            ),
            "literal boolean",
        ),
    ],
)
def test_wrong_lock_or_gate_set_denies_without_parsing_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_factory: object,
    message: str,
) -> None:
    sealed = tmp_path / "sealed.json"
    result_path = tmp_path / "validation_result.json"
    _write_json(sealed, {"split": "sealed", "secret": "opaque"})
    lock = _build_lock(sealed)
    _write_json(result_path, result_factory(lock))
    monkeypatch.setattr(
        protocol,
        "_parse_sealed_json_bytes",
        lambda value: pytest.fail("sealed bytes were parsed before authorization"),
    )
    with pytest.raises(protocol.SealedAccessDenied, match=message):
        protocol.load_sealed_dataset(sealed, result_path, lock=lock)


def test_invalid_result_self_hash_denies_without_parsing_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = tmp_path / "sealed.json"
    result_path = tmp_path / "validation_result.json"
    _write_json(sealed, {"split": "sealed", "secret": "opaque"})
    lock = _build_lock(sealed)
    result = _validation_result(lock)
    result["machine_readable_summary"]["synthetic"] = False
    _write_json(result_path, result)
    monkeypatch.setattr(
        protocol,
        "_parse_sealed_json_bytes",
        lambda value: pytest.fail("sealed bytes were parsed before authorization"),
    )
    with pytest.raises(protocol.SealedAccessDenied, match="self-hash"):
        protocol.load_sealed_dataset(sealed, result_path, lock=lock)


def test_sealed_byte_hash_is_checked_before_validation_json(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed.json"
    result_path = tmp_path / "validation_result.json"
    _write_json(sealed, {"split": "sealed", "records": []})
    lock = _build_lock(sealed)
    sealed.write_bytes(b'{"split":"sealed","tampered":true}')
    result_path.write_bytes(b"not-json")
    with pytest.raises(
        protocol.CounterfactualKLProtocolIntegrityError,
        match="sealed dataset byte hash differs",
    ):
        protocol.load_sealed_dataset(sealed, result_path, lock=lock)


def test_authorized_result_deserializes_the_exact_hashed_bytes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = tmp_path / "sealed.json"
    result_path = tmp_path / "validation_result.json"
    expected = {"split": "sealed", "records": [{"id": "synthetic-secret"}]}
    _write_json(sealed, expected)
    lock = _build_lock(sealed)
    _write_json(result_path, _validation_result(lock))

    original = protocol._parse_sealed_json_bytes
    calls: list[bytes] = []

    def observed_parser(value: bytes) -> dict[str, object]:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(protocol, "_parse_sealed_json_bytes", observed_parser)
    observed = protocol.load_sealed_dataset(sealed, result_path, lock=lock)
    assert observed == expected
    assert calls == [sealed.read_bytes()]


def test_self_hash_helper_rejects_tampering() -> None:
    record = protocol.self_hash_record({"status": "go", "gates": {"safety": True}})
    assert protocol.verify_self_hashed_record(record) == record
    tampered = copy.deepcopy(record)
    tampered["gates"]["safety"] = False
    with pytest.raises(protocol.CounterfactualKLProtocolIntegrityError, match="canonical"):
        protocol.verify_self_hashed_record(tampered)


def test_exact_result_validator_accepts_consistent_no_go_and_rejects_status_drift(
    tmp_path: Path,
) -> None:
    sealed = tmp_path / "sealed.json"
    _write_json(sealed, {"split": "sealed"})
    lock = _build_lock(sealed)
    gates = {"efficacy": False, "integrity": True, "safety": True}
    no_go = _validation_result(lock, status="no_go", gates=gates)
    assert (
        protocol.validate_locked_result(
            no_go,
            lock=lock,
            expected_split="validation",
        )
        == no_go
    )

    inconsistent = _validation_result(lock, status="go", gates=gates)
    with pytest.raises(
        protocol.CounterfactualKLProtocolIntegrityError,
        match="status is inconsistent",
    ):
        protocol.validate_locked_result(
            inconsistent,
            lock=lock,
            expected_split="validation",
        )


def test_builder_rejects_incomplete_manifest_and_nonfinite_threshold(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed.json"
    _write_json(sealed, {"split": "sealed"})
    kwargs = {
        "file_hashes": {"sealed": {"path": "sealed.json", "sha256": protocol.file_sha256(sealed)}},
        "rendered_manifests": {"validation": {"count": 1}},
        "configuration": {"model": "synthetic"},
        "thresholds": {"kl": float("nan")},
        "sealed_dataset_file_key": "sealed",
        "validation_result_schema_version": RESULT_SCHEMA,
        "required_validation_gates": REQUIRED_GATES,
    }
    with pytest.raises(ValueError, match="exactly validation and sealed"):
        protocol.build_prospective_lock(**kwargs)
    kwargs["rendered_manifests"]["sealed"] = {"count": 1}
    with pytest.raises(ValueError, match="NaN or infinity"):
        protocol.build_prospective_lock(**kwargs)
