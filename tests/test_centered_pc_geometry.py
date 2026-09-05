"""Synthetic-only, model-free software tests; no real P/C geometry is calculated."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import struct
import sys

import pytest

from scripts import centered_pc_geometry as generator
from scripts import verify_centered_pc_geometry as checker


def raw(vector):
    return b"".join(struct.pack("<d", value) for value in vector)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def saved_vector(tmp_path, vector):
    coordinate_digest = digest(raw(vector))
    payload = json.dumps(
        {"vector": vector, "vector_float64_le_sha256": coordinate_digest},
        allow_nan=False,
    ).encode()
    path = tmp_path / "synthetic_vector.json"
    path.write_bytes(payload)
    return path, digest(payload), coordinate_digest


def verify(result, p, c):
    return checker.check_geometry(p, c, result["midpoint"], result["difference"], result["metrics"])


def test_known_small_vectors_all_metrics_and_independent_identities():
    p, c = [3.0, 4.0], [-3.0, 4.0]
    result = generator.compute(p, c)
    assert result["midpoint"] == [0.0, 4.0]
    assert result["difference"] == [3.0, 0.0]
    assert result["metrics"] == {
        "P_norm": 5.0,
        "C_norm": 5.0,
        "midpoint_norm": 4.0,
        "difference_norm": 3.0,
        "P_dot_C": 7.0,
        "midpoint_dot_difference": 0.0,
        "P_C_cosine": 7.0 / 25.0,
        "midpoint_difference_cosine": 0.0,
    }
    assert verify(result, p, c)["status"] == "GEOMETRY_IDENTITIES_VERIFIED"


@pytest.mark.parametrize(
    "p,c,midpoint,difference,pc_cosine,md_cosine",
    [
        ([2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [0.0, 0.0], 1.0, None),
        ([2.0, 0.0], [-2.0, 0.0], [0.0, 0.0], [2.0, 0.0], -1.0, None),
        ([2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [1.0, -1.0], 0.0, 0.0),
        ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], None, None),
        ([2.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], None, 1.0),
    ],
)
def test_identical_opposite_orthogonal_and_zero_inputs(
    p, c, midpoint, difference, pc_cosine, md_cosine
):
    result = generator.compute(p, c)
    assert result["midpoint"] == midpoint and result["difference"] == difference
    assert result["metrics"]["P_C_cosine"] == pc_cosine
    assert result["metrics"]["midpoint_difference_cosine"] == md_cosine
    assert verify(result, p, c)["status"] == "GEOMETRY_IDENTITIES_VERIFIED"


def test_input_order_changes_difference_sign_not_midpoint_and_never_normalizes():
    p, c = [0.125, -0.25, 0.5], [0.375, 0.25, -0.5]
    forward, reverse = generator.compute(p, c), generator.compute(c, p)
    assert forward["midpoint"] == reverse["midpoint"] == [0.25, 0.0, 0.0]
    assert forward["difference"] == [-0.125, -0.25, 0.5]
    assert reverse["difference"] == [0.125, 0.25, -0.5]
    assert forward["metrics"]["midpoint_norm"] == 0.25
    assert forward["metrics"]["difference_norm"] == math.sqrt(0.328125)
    assert verify(forward, p, c)["status"] == "GEOMETRY_IDENTITIES_VERIFIED"
    assert verify(reverse, c, p)["status"] == "GEOMETRY_IDENTITIES_VERIFIED"


def test_sum_is_rounded_before_division_not_half_each_input():
    tiny = math.ulp(0.0)
    result = generator.compute([tiny], [tiny])
    assert result["midpoint"] == [tiny]
    assert float(tiny / 2) + float(tiny / 2) == 0.0
    assert result["midpoint"][0] != float(tiny / 2) + float(tiny / 2)
    opposite = generator.compute([tiny], [-tiny])
    assert opposite["difference"] == [tiny]


def test_signed_zero_and_little_endian_coordinate_bits_are_preserved():
    vector = [-0.0, 0.0, 0.5, -2.0]
    assert generator.vector_bytes(vector) == raw(vector)
    assert generator.vector_sha(vector) == digest(raw(vector))
    assert generator.vector_sha([-0.0]) != generator.vector_sha([0.0])
    result = generator.compute([-0.0, 0.0], [-0.0, -0.0])
    assert raw(result["midpoint"]) == raw([-0.0, 0.0])
    assert raw(result["difference"]) == raw([0.0, 0.0])
    assert verify(result, [-0.0, 0.0], [-0.0, -0.0])["status"] == "GEOMETRY_IDENTITIES_VERIFIED"


@pytest.mark.parametrize(
    "p,c",
    [
        ([sys.float_info.max], [sys.float_info.max]),
        ([sys.float_info.max], [-sys.float_info.max]),
        ([1e200], [0.0]),
    ],
)
def test_nonfinite_operation_or_metric_intermediates_are_rejected(p, c):
    with pytest.raises((ValueError, OverflowError)):
        generator.compute(p, c)


@pytest.mark.parametrize(
    "vector,dimension",
    [
        ([1.0], 2),
        ([1.0, 2.0], 1),
        ([1], 1),
        ([True], 1),
        (["1"], 1),
        ([None], 1),
        ([[1.0]], 1),
        ((1.0,), 1),
        ([float("nan")], 1),
        ([float("inf")], 1),
        ([float("-inf")], 1),
    ],
)
def test_validate_vector_rejects_wrong_shape_type_or_nonfinite(vector, dimension):
    with pytest.raises((ValueError, TypeError)):
        generator.validate_vector(vector, dimension)


def test_validation_keeps_exact_finite_coordinates_without_modification():
    vector = [0.0, -0.0, 0.125, -0.5]
    before = raw(vector)
    generator.validate_vector(vector, 4)
    assert raw(vector) == before


@pytest.mark.parametrize("loader", [generator.load_vector, checker.decode_vector])
def test_authentication_checks_file_declared_and_coordinate_digests(tmp_path, loader):
    vector = [0.125, -0.5, 0.0]
    path, file_digest, coordinate_digest = saved_vector(tmp_path, vector)
    decoded = loader(path, file_digest, coordinate_digest, 3)
    assert decoded == vector
    assert raw(decoded) == raw(vector)


@pytest.mark.parametrize("loader", [generator.load_vector, checker.decode_vector])
@pytest.mark.parametrize(
    "fault",
    [
        "file_hash",
        "expected_vector_hash",
        "declared_hash",
        "coordinate",
        "order",
        "sign",
        "malformed",
        "nonfinite",
        "dimension",
    ],
)
def test_independent_loaders_reject_all_artifact_tampering(tmp_path, loader, fault):
    vector = [0.125, -0.5, 0.25]
    path, file_digest, coordinate_digest = saved_vector(tmp_path, vector)
    dimension = 3
    if fault == "file_hash":
        file_digest = "0" * 64
    elif fault == "expected_vector_hash":
        coordinate_digest = "0" * 64
    elif fault == "dimension":
        dimension = 4
    else:
        value = json.loads(path.read_text())
        if fault == "declared_hash":
            value["vector_float64_le_sha256"] = "0" * 64
        elif fault == "coordinate":
            value["vector"][0] += 0.01
        elif fault == "order":
            value["vector"] = list(reversed(value["vector"]))
        elif fault == "sign":
            value["vector"] = [-x for x in value["vector"]]
        elif fault == "nonfinite":
            value["vector"][0] = float("nan")
        payload = b"{" if fault == "malformed" else json.dumps(value).encode()
        path.write_bytes(payload)
        # Updating only the outer file hash must not conceal coordinate/metadata corruption.
        file_digest = digest(payload)
    with pytest.raises((ValueError, TypeError)):
        loader(path, file_digest, coordinate_digest, dimension)


@pytest.mark.parametrize(
    "fault",
    [
        "midpoint",
        "difference",
        "swap",
        "metric",
        "cosine",
        "missing_metric",
        "nonfinite_metric",
        "zero_cosine",
    ],
)
def test_independent_checker_rejects_wrong_reconstruction_and_metrics(fault):
    p, c = [3.0, 4.0], [-3.0, 4.0]
    result = copy.deepcopy(generator.compute(p, c))
    if fault == "midpoint":
        result["midpoint"][0] += 1e-14
    elif fault == "difference":
        result["difference"][0] = math.nextafter(result["difference"][0], math.inf)
    elif fault == "swap":
        result["midpoint"], result["difference"] = result["difference"], result["midpoint"]
    elif fault == "metric":
        result["metrics"]["P_norm"] += 1e-8
    elif fault == "cosine":
        result["metrics"]["P_C_cosine"] += 1e-8
    elif fault == "missing_metric":
        result["metrics"].pop("difference_norm")
    elif fault == "nonfinite_metric":
        result["metrics"]["P_dot_C"] = float("nan")
    else:
        result["metrics"]["midpoint_difference_cosine"] = None
    with pytest.raises((ValueError, TypeError, KeyError)):
        verify(result, p, c)


def test_fixed_absolute_metric_tolerance_and_triangle_equality():
    p, c = [3.0, 4.0], [-3.0, -4.0]
    result = generator.compute(p, c)
    assert result["metrics"]["difference_norm"] == 5.0
    assert (
        result["metrics"]["difference_norm"]
        == (result["metrics"]["P_norm"] + result["metrics"]["C_norm"]) / 2
    )
    assert verify(result, p, c)["status"] == "GEOMETRY_IDENTITIES_VERIFIED"
    result["metrics"]["difference_norm"] += 2e-12
    with pytest.raises(ValueError):
        verify(result, p, c)


def test_guarded_write_fixed_default_total_bound_and_exclusive_files(tmp_path):
    assert (
        inspect.signature(generator.guarded_write).parameters["maximum_bytes"].default == 10485760
    )
    generator.guarded_write(tmp_path, "one.bin", b"12345", maximum_bytes=10)
    generator.guarded_write(tmp_path, "two.bin", b"67890", maximum_bytes=10)
    assert sum(path.stat().st_size for path in tmp_path.iterdir()) == 10
    with pytest.raises((ValueError, FileExistsError)):
        generator.guarded_write(tmp_path, "one.bin", b"x", maximum_bytes=100)
    assert (tmp_path / "one.bin").read_bytes() == b"12345"
    with pytest.raises(ValueError):
        generator.guarded_write(tmp_path, "three.bin", b"x", maximum_bytes=10)
    assert not (tmp_path / "three.bin").exists()
    assert sum(path.stat().st_size for path in tmp_path.iterdir()) == 10


@pytest.mark.parametrize(
    "name",
    [
        "../escape.bin",
        "nested/item.bin",
        "nested\\item.bin",
        "/absolute.bin",
        "C:\\absolute.bin",
        "..",
        ".",
    ],
)
def test_guarded_write_rejects_nonflat_or_escaping_names_without_artifacts(tmp_path, name):
    with pytest.raises((ValueError, OSError)):
        generator.guarded_write(tmp_path, name, b"x")
    assert not list(tmp_path.iterdir())


def test_guarded_write_counts_all_existing_bytes_before_creating_file(tmp_path):
    (tmp_path / "existing.json").write_bytes(b"12345678")
    with pytest.raises(ValueError):
        generator.guarded_write(tmp_path, "new.bin", b"123", maximum_bytes=10)
    assert not (tmp_path / "new.bin").exists()
    assert (tmp_path / "existing.json").read_bytes() == b"12345678"


@pytest.mark.parametrize(
    "fault", [None, "raw", "operation", "coordinates", "metric", "resources", "source", "extra"]
)
def test_synthetic_frozen_artifact_and_independent_report_path(tmp_path, monkeypatch, fault):
    def save(name, value):
        payload = (json.dumps(value, indent=2) + "\n").encode()
        (tmp_path / name).write_bytes(payload)
        return digest(payload)

    model = {"id": "synthetic-only", "d_model": 2}
    site = {"hook": "blocks.10.hook_out", "position": "final encoded prompt token"}
    cast = "hn=norm(h0); independent original prompt"
    inputs = {}
    for label, v in (("P", [3.0, 4.0]), ("C", [-3.0, 4.0])):
        path, lock, audit = label + ".json", label + "_lock.json", label + "_audit.json"
        vsha = digest(raw(v))
        fsha = save(path, {"vector": v, "vector_float64_le_sha256": vsha})
        lsha = save(
            lock,
            {
                "plan": {
                    "model": model,
                    "intervention": {**site, "layer": 10},
                    "config": {"cast_sequence": cast},
                }
            },
        )
        asha = save(audit, {"status": "SYNTHETIC"})
        inputs[label] = {
            "path": path,
            "file_sha256": fsha,
            "vector_sha256": vsha,
            "lock": lock,
            "lock_sha256": lsha,
            "audit": audit,
            "audit_sha256": asha,
            "audit_status": "SYNTHETIC",
            "norm": 5.0,
        }
    cfg = {
        "dimension": 2,
        "model": model,
        "site": site,
        "inputs": inputs,
        "output_namespace": "synthetic_evidence",
        "maximum_evidence_bytes": 10485760,
        "operations": {"midpoint": "(P+C)/2", "difference": "(P-C)/2"},
    }
    cfgsha = save("config.json", cfg)
    output = tmp_path / "synthetic_evidence"
    for module in (generator, checker):
        monkeypatch.setattr(module, "ROOT", tmp_path)
        monkeypatch.setattr(module, "OUTPUT", output)
        monkeypatch.setattr(module, "CONFIG", "config.json")
        monkeypatch.setattr(module, "CONFIG_SHA", cfgsha)
    monkeypatch.setattr(generator, "source_identity", lambda bindings: dict(bindings))

    def git(*args):
        if args[0] == "show":
            return "synthetic_evidence/preregistration.json"
        if args[0] == "rev-parse":
            return "synthetic-source"
        assert args[0] == "status"
        return ""

    monkeypatch.setattr(generator, "git", git)
    assert generator.freeze()["status"] == "MODEL_FREE_RECIPE_FROZEN"
    receipt = generator.derive()
    assert receipt["resources"] == {
        "model_loads": 0,
        "tokenizer_loads": 0,
        "real_forwards": 0,
        "real_derivatives": 0,
    }
    if fault in ("operation", "coordinates"):
        item = json.loads((output / "midpoint.json").read_bytes())
        if fault == "operation":
            item["operation"] = "normalized alternative"
        else:
            item["vector"][0] = 1e-14
            item["vector_float64_le_sha256"] = digest(raw(item["vector"]))
            (output / "midpoint.f64").write_bytes(raw(item["vector"]))
        (output / "midpoint.json").write_text(json.dumps(item), encoding="utf-8")
    elif fault == "raw":
        value = bytearray((output / "difference.f64").read_bytes())
        value[0] ^= 1
        (output / "difference.f64").write_bytes(value)
    elif fault == "metric":
        receipt["metrics"]["P_C_cosine"] += 1e-8
    elif fault == "resources":
        receipt["resources"]["real_forwards"] = 1
    elif fault == "source":
        (tmp_path / "P.json").write_bytes(b"{}")
    elif fault == "extra":
        (output / "unexpected.json").write_bytes(b"{}")
    for name in receipt["artifacts"]:
        payload = (output / name).read_bytes()
        receipt["artifacts"][name] = {"sha256": digest(payload), "bytes": len(payload)}
    (output / "geometry.json").write_text(json.dumps(receipt), encoding="utf-8")
    if fault is not None:
        with pytest.raises(ValueError):
            checker.verify()
    else:
        verified = checker.verify()
        assert verified["status"] == "GEOMETRY_IDENTITIES_VERIFIED"
        report = checker.report(verified)
        for phrase in (
            "Geometry identities PASS",
            "NOT an identified A-bias",
            "No semantic steering",
            "zero real forwards",
            "STOP",
        ):
            assert phrase in report
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        with pytest.raises(ValueError):
            generator.derive()
        assert {p.name: p.read_bytes() for p in output.iterdir()} == before
