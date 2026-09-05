"""Independent Decimal geometry audit. Does not import the generator or ML libraries."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import struct
import sys
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence/centered_pc_geometry_v1_qwen35_08b"
CONFIG = "configs/centered_pc_geometry_v1.json"
CONFIG_SHA = "c5ab875c672cf2be031c8afb6bc533b5e8d059decba40c450523bb10ab2c7fc5"
EPS = Decimal("1e-12")
CAP = 10485760


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def coordinates(v, size):
    require(
        isinstance(v, list)
        and len(v) == size
        and size > 0
        and all(type(x) is float and math.isfinite(x) for x in v),
        "finite exact-length binary64 coordinates",
    )
    return v


def packed(v):
    coordinates(v, len(v))
    return b"".join(struct.pack("<d", x) for x in v)


def decode_vector(path, file_sha, vector_digest, dimension):
    raw = Path(path).read_bytes()
    require(sha(raw) == file_sha, "source file hash")
    value = json.loads(raw)
    require(isinstance(value, dict), "source vector object")
    v = coordinates(value.get("vector"), dimension)
    require(
        value.get("vector_float64_le_sha256") == vector_digest and sha(packed(v)) == vector_digest,
        "source coordinate digest",
    )
    return v


def exact(v):
    return [Decimal.from_float(x) for x in v]


def dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), Decimal(0))


def length(v):
    return dot(v, v).sqrt()


def within(value, expected, label):
    require(type(value) is float and math.isfinite(value), label + " finite float")
    error = abs(Decimal.from_float(value) - expected)
    require(error <= EPS, label + " absolute1e-12/relative0")
    return error


def check_geometry(p, comply, c, d, metrics):
    for value in (p, comply, c, d):
        coordinates(value, len(p))
    with localcontext() as context:
        context.prec = 120
        P, C, M, D = map(exact, (p, comply, c, d))
        expected_c, expected_d = [], []
        for x, y in zip(P, C, strict=True):
            plus, minus = float(x + y), float(x - y)
            require(math.isfinite(plus) and math.isfinite(minus), "finite rounded intermediate")
            expected_c.append(float(Decimal.from_float(plus) / Decimal(2)))
            expected_d.append(float(Decimal.from_float(minus) / Decimal(2)))
        require(
            packed(c) == packed(expected_c) and packed(d) == packed(expected_d),
            "exact two-stage binary64 operation/order/sign coordinates",
        )
        pn, cn, mn, dn = map(length, (P, C, M, D))
        pc, md = dot(P, C), dot(M, D)
        values = {
            "P_norm": pn,
            "C_norm": cn,
            "midpoint_norm": mn,
            "difference_norm": dn,
            "P_dot_C": pc,
            "midpoint_dot_difference": md,
            "P_C_cosine": pc / (pn * cn) if pn and cn else None,
            "midpoint_difference_cosine": md / (mn * dn) if mn and dn else None,
        }
        require(isinstance(metrics, dict) and set(metrics) == set(values), "exact metric keys")
        errors = {}
        for key, value in values.items():
            if value is None:
                require(metrics[key] is None, "undefined zero-norm cosine is null")
                errors[key] = None
            else:
                errors[key] = str(within(metrics[key], value, key))
        p2, c2, m2, d2 = (dot(x, x) for x in (P, C, M, D))
        residuals = {
            "P_coordinate_reconstruction": max(
                abs(x - (a + b)) for x, a, b in zip(P, M, D, strict=True)
            ),
            "C_coordinate_reconstruction": max(
                abs(x - (a - b)) for x, a, b in zip(C, M, D, strict=True)
            ),
            "sum_squares": abs(m2 + d2 - (p2 + c2) / 2),
            "difference_squares": abs(m2 - d2 - pc),
            "midpoint_difference_dot": abs(md - (p2 - c2) / 4),
            "midpoint_squared": abs(m2 - (p2 + c2 + 2 * pc) / 4),
            "difference_squared": abs(d2 - (p2 + c2 - 2 * pc) / 4),
        }
        require(all(x <= EPS for x in residuals.values()), "midpoint/difference identity tolerance")
        triangle_bound, excess = (pn + cn) / 2, dn - (pn + cn) / 2
        require(excess <= EPS, "triangle bound")
        return {
            "status": "GEOMETRY_IDENTITIES_VERIFIED",
            "decimal_precision": 120,
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 0,
            "decimal_metrics": {k: str(v) if v is not None else None for k, v in values.items()},
            "metric_absolute_errors": errors,
            "identity_absolute_residuals": {k: str(v) for k, v in residuals.items()},
            "triangle_bound": str(triangle_bound),
            "triangle_signed_excess": str(excess),
            "exact_binary64_coordinate_match": True,
        }


def authenticated_inputs(cfg):
    result, bindings = {}, {}
    with localcontext() as context:
        context.prec = 120
        for name, spec in cfg["inputs"].items():
            for key, digest_key in (
                ("path", "file_sha256"),
                ("lock", "lock_sha256"),
                ("audit", "audit_sha256"),
            ):
                raw = (ROOT / spec[key]).read_bytes()
                require(sha(raw) == spec[digest_key], "independently authenticated input bytes")
                bindings[spec[key]] = spec[digest_key]
            v = decode_vector(
                ROOT / spec["path"], spec["file_sha256"], spec["vector_sha256"], cfg["dimension"]
            )
            require(
                abs(length(exact(v)) - Decimal.from_float(spec["norm"])) <= EPS, "source .20 norm"
            )
            lock, audit = read(ROOT / spec["lock"]), read(ROOT / spec["audit"])
            model, site = lock["plan"]["model"], lock["plan"]["intervention"]
            require(
                all(model[k] == x for k, x in cfg["model"].items())
                and site["hook"] == cfg["site"]["hook"]
                and site["position"] == cfg["site"]["position"]
                and site["layer"] == 10
                and audit["status"] == spec["audit_status"],
                "source model/site/audit identity",
            )
            cast = site.get("cast_sequence", lock["plan"]["config"]["cast_sequence"])
            require("h0" in cast and "original prompt" in cast, "original norm convention")
            result[name] = v
    require(set(result) == {"P", "C"}, "exact P/C inputs")
    return result, bindings


def verify():
    require(
        not any(
            x.split(".")[0] in {"torch", "numpy", "transformers", "transformer_lens"}
            for x in sys.modules
        ),
        "model-free checker imports",
    )
    raw_config = (ROOT / CONFIG).read_bytes()
    require(sha(raw_config) == CONFIG_SHA, "fixed independent recipe bytes")
    cfg, record = json.loads(raw_config), read(OUTPUT / "preregistration.json")
    require(record["config"] == cfg, "frozen recipe")
    require(
        {p.name for p in OUTPUT.iterdir()}
        == {
            "preregistration.json",
            "DERIVATION_STARTED.json",
            "midpoint.json",
            "midpoint.f64",
            "difference.json",
            "difference.f64",
            "geometry.json",
        },
        "single untouched derivation output",
    )
    for path, digest in record["source_sha256"].items():
        require(sha((ROOT / path).read_bytes()) == digest, "frozen source/input bytes")
    expected_environment = {
        "python": sys.version,
        "executable": sys.executable,
        "executable_sha256": sha(Path(sys.executable).read_bytes()),
        "platform": platform.platform(),
    }
    require(record["environment"] == expected_environment, "same Python environment")
    vectors, inputs = authenticated_inputs(cfg)
    require(record["input_sha256"] == inputs, "exact input manifest")
    receipt = read(OUTPUT / "geometry.json")
    resources = {"model_loads": 0, "tokenizer_loads": 0, "real_forwards": 0, "real_derivatives": 0}
    started = read(OUTPUT / "DERIVATION_STARTED.json")
    require(
        receipt["resources"] == record["resources"] == resources
        and started == {"source_commit": record["source_commit"], **resources},
        "single model-free derivation resource envelope",
    )
    require(
        receipt["preregistration_sha256"] == sha((OUTPUT / "preregistration.json").read_bytes())
        and receipt["status"] == "DERIVED_NOT_BEHAVIORALLY_TESTED"
        and receipt["future_sign_conventions_only"] == {"+d": "proposed P", "-d": "proposed C"}
        and receipt["no_bias_cancellation_claim"] is True,
        "receipt scope and lock identity",
    )
    require(
        set(receipt["artifacts"])
        == {"midpoint.json", "midpoint.f64", "difference.json", "difference.f64"},
        "exact two JSON and raw binary64 artifacts",
    )
    for name, meta in receipt["artifacts"].items():
        raw = (OUTPUT / name).read_bytes()
        require(
            sha(raw) == meta["sha256"] and len(raw) == meta["bytes"], "artifact file hash/length"
        )
    saved, hashes = {}, {}
    for name in ("midpoint", "difference"):
        item = read(OUTPUT / (name + ".json"))
        v = coordinates(item["vector"], cfg["dimension"])
        raw = (OUTPUT / (name + ".f64")).read_bytes()
        require(
            raw == packed(v) and sha(raw) == item["vector_float64_le_sha256"],
            "saved JSON/raw coordinate agreement",
        )
        require(
            item["kind"] == name
            and item["operation"] == cfg["operations"][name]
            and item["dimension"] == cfg["dimension"]
            and item["mathematical_only"] is True
            and item["behavior_tested"] is False
            and item["norm"] == receipt["metrics"][name + "_norm"],
            "artifact operation/scope/norm",
        )
        saved[name] = v
        hashes[name] = {
            "vector_float64_le_sha256": sha(raw),
            "json_file_sha256": sha((OUTPUT / (name + ".json")).read_bytes()),
            "raw_file_sha256": sha(raw),
        }
    checked = check_geometry(
        vectors["P"], vectors["C"], saved["midpoint"], saved["difference"], receipt["metrics"]
    )
    require(
        all(
            receipt["metrics"][k] > 0
            for k in ("P_norm", "C_norm", "midpoint_norm", "difference_norm")
        ),
        "actual nonzero geometry",
    )
    size = sum(p.stat().st_size for p in OUTPUT.rglob("*") if p.is_file())
    require(size <= CAP == cfg["maximum_evidence_bytes"], "10MiB total storage")
    return {
        **checked,
        "metrics": receipt["metrics"],
        "artifacts": hashes,
        "resources": resources,
        "input_sha256": inputs,
        "source_commit": record["source_commit"],
        "preregistration_sha256": receipt["preregistration_sha256"],
        "pre_report_evidence_bytes": size,
        "maximum_evidence_bytes": CAP,
        "behavioral_success_claim": False,
        "prior_approximation_disclosed": True,
    }


def report(value):
    if value["status"] != "GEOMETRY_IDENTITIES_VERIFIED":
        return (
            "# Centered P/C geometry\n\nINCONCLUSIVE. No retry or behavioral inference.\n\n"
            + json.dumps(value, indent=2)
            + "\n"
        )
    lines = [
        "# Centered P/C geometry: independent mathematical report",
        "",
        "Geometry identities PASS. This is a mathematical construction, NOT behavioral steering evidence.",
        "Zero model/tokenizer loads, zero real forwards/derivatives. No training or application.",
        "",
        "| Metric | Binary64 repr | Independent Decimal (120-digit context) |",
        "|---|---:|---:|",
    ]
    for name, metric in value["metrics"].items():
        lines.append(f"| {name} | {metric!r} | {value['decimal_metrics'][name]} |")
    lines += [
        "",
        "## Identities and bound",
        "",
        "All checks use fixed absolute1e-12, relative0.",
        "Exact per-coordinate two-stage binary64 operation and JSON/raw bit agreement: passed.",
        "",
        "| Identity | Absolute residual |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {error} |" for name, error in value["identity_absolute_residuals"].items()
    )
    lines += [
        "",
        "Triangle bound: " + value["triangle_bound"] + ".",
        "Signed excess ||d|| - (||P||+||C||)/2: " + value["triangle_signed_excess"] + ".",
        "Equal source lengths imply approximately orthogonal midpoint and difference by algebra, not a neural mechanism.",
        "",
        "## Frozen artifacts",
        "",
        "| Vector | Binary64/raw SHA256 | JSON file SHA256 |",
        "|---|---|---|",
    ]
    for name, meta in value["artifacts"].items():
        lines.append(
            f"| {name} | {meta['vector_float64_le_sha256']} | {meta['json_file_sha256']} |"
        )
    lines += [
        "",
        "## Scope and confounds",
        "",
        "c=(P+C)/2 is a midpoint, NOT an identified A-bias vector. d=(P-C)/2 mixes semantic goals with construction differences.",
        "P used f01/f02 v1/v2 AB with retention goals then fixed .20 scaling; C used f01/f02 v1 crossed AB/BA with the original outcome objective.",
        "Subtracting the midpoint may remove a necessary nonlinear offset. No semantic steering, ordinary-task preservation, gating or bias cancellation is guaranteed.",
        "Future +d for P and -d for C are proposed signs only. Neither was applied. c±d reconstruct the existing arrows, not a new behavioral experiment.",
        "Prior saved-offset bridge/sign-reversal failures remain relevant negative evidence; no historical re-audit was run. The old f03 failure and exposed frozen-C pass remain unchanged.",
        "The supervisor disclosed approximate P/C cosine -.28595422036543316 and ||d|| .16037170700378756 before this run. Definitions preceded that exploration; neither number selected this recipe or acceptance tolerance.",
        "This is not blind evidence and earns no final-study success. Publication readiness remains40%.",
        "No normalization, upscaling, coordinate selection, alternative contrast, optimizer, search or successor.",
        "A later model application needs a separately specified prospective DEVELOPMENT test and supervisor instruction. STOP.",
        "",
    ]
    return "\n".join(lines)


def write_report_file(name, raw):
    require(
        sum(p.stat().st_size for p in OUTPUT.rglob("*") if p.is_file()) + len(raw) <= CAP,
        "10MiB report storage",
    )
    with (OUTPUT / name).open("xb") as stream:
        stream.write(raw)


if __name__ == "__main__":
    require(sys.argv[1:] == ["--report"], "Use --report; no adjustable arithmetic/tolerance")
    try:
        result = verify()
    except Exception as error:  # noqa: BLE001 - preserve audit failure without repair or retry.
        result = {
            "status": "INCONCLUSIVE",
            "fault": type(error).__name__ + ": " + str(error),
            "retry_allowed": False,
        }
    write_report_file(
        "verification.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    )
    write_report_file("GEOMETRY_REPORT.md", report(result).encode())
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != "GEOMETRY_IDENTITIES_VERIFIED":
        raise SystemExit(1)
