"""Fixed standard-library-only midpoint/difference derivation; no behavioral entry point."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/centered_pc_geometry_v1.json"
CONFIG_SHA = "c5ab875c672cf2be031c8afb6bc533b5e8d059decba40c450523bb10ab2c7fc5"
OUTPUT = ROOT / "evidence/centered_pc_geometry_v1_qwen35_08b"
SOURCES = (
    CONFIG,
    "docs/CENTERED_PC_GEOMETRY_V1.md",
    "scripts/centered_pc_geometry.py",
    "scripts/verify_centered_pc_geometry.py",
    "tests/test_centered_pc_geometry.py",
)
EPS = 1e-12
CAP = 10485760
ZERO_CALLS = {"model_loads": 0, "tokenizer_loads": 0, "real_forwards": 0, "real_derivatives": 0}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def validate_vector(v, dimension):
    require(
        isinstance(v, list)
        and len(v) == dimension
        and dimension > 0
        and all(type(x) is float and math.isfinite(x) for x in v),
        "finite binary64 float coordinates and exact dimension",
    )
    return v


def vector_bytes(v):
    validate_vector(v, len(v))
    return struct.pack("<" + "d" * len(v), *v)


def vector_sha(v):
    return sha(vector_bytes(v))


def load_vector(path, file_sha, vector_digest, dimension):
    raw = Path(path).read_bytes()
    require(sha(raw) == file_sha, "input file hash")
    value = json.loads(raw)
    require(isinstance(value, dict), "vector object")
    v = validate_vector(value.get("vector"), dimension)
    require(
        value.get("vector_float64_le_sha256") == vector_digest == vector_sha(v),
        "declared and actual coordinate hash",
    )
    return v


def norm(v):
    result = math.sqrt(math.fsum(x * x for x in v))
    require(math.isfinite(result), "finite binary64 norm")
    return result


def dot(a, b):
    result = math.fsum(x * y for x, y in zip(a, b, strict=True))
    require(math.isfinite(result), "finite binary64 dot")
    return result


def compute(p, comply):
    validate_vector(p, len(p))
    validate_vector(comply, len(p))
    c = [float(float(x + y) / 2) for x, y in zip(p, comply, strict=True)]
    d = [float(float(x - y) / 2) for x, y in zip(p, comply, strict=True)]
    validate_vector(c, len(p))
    validate_vector(d, len(p))
    pn, cn, mn, dn = map(norm, (p, comply, c, d))
    pc, md = dot(p, comply), dot(c, d)
    metrics = {
        "P_norm": pn,
        "C_norm": cn,
        "midpoint_norm": mn,
        "difference_norm": dn,
        "P_dot_C": pc,
        "midpoint_dot_difference": md,
        "P_C_cosine": pc / (pn * cn) if pn and cn else None,
        "midpoint_difference_cosine": md / (mn * dn) if mn and dn else None,
    }
    require(all(x is None or math.isfinite(x) for x in metrics.values()), "finite metrics")
    return {"midpoint": c, "difference": d, "metrics": metrics}


def config_at():
    raw = (ROOT / CONFIG).read_bytes()
    require(sha(raw) == CONFIG_SHA, "fixed committed recipe bytes")
    return json.loads(raw)


def load_inputs(cfg, root=None):
    root = ROOT if root is None else Path(root)
    vectors, bindings = {}, {}
    for label, spec in cfg["inputs"].items():
        for key, digest in (
            ("path", "file_sha256"),
            ("lock", "lock_sha256"),
            ("audit", "audit_sha256"),
        ):
            require(
                sha((root / spec[key]).read_bytes()) == spec[digest], "authenticated source " + key
            )
            bindings[spec[key]] = spec[digest]
        vectors[label] = load_vector(
            root / spec["path"], spec["file_sha256"], spec["vector_sha256"], cfg["dimension"]
        )
        require(abs(norm(vectors[label]) - spec["norm"]) <= EPS, "source .20 norm")
        lock, audit = read(root / spec["lock"]), read(root / spec["audit"])
        model, site = lock["plan"]["model"], lock["plan"]["intervention"]
        require(
            all(model[k] == value for k, value in cfg["model"].items())
            and all(site[k] == cfg["site"][k] for k in ("hook", "position"))
            and site["layer"] == 10
            and audit["status"] == spec["audit_status"],
            "same source coordinate/model/site and successful audit",
        )
        cast = site.get("cast_sequence", lock["plan"]["config"]["cast_sequence"])
        require("h0" in cast and "original prompt" in cast, "own-original norm convention")
    require(set(vectors) == {"P", "C"}, "exact two source arrows")
    return vectors, bindings


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_identity(bindings):
    result = dict(bindings)
    for path in (*SOURCES, *bindings):
        require(
            git("ls-files", "--", path) and not git("status", "--porcelain", "--", path),
            "clean tracked source/input " + path,
        )
        result[path] = sha((ROOT / path).read_bytes())
    return result


def environment():
    require(
        not any(
            x.split(".")[0] in {"torch", "transformers", "transformer_lens", "numpy"}
            for x in sys.modules
        ),
        "no ML libraries loaded",
    )
    return {
        "python": sys.version,
        "executable": sys.executable,
        "executable_sha256": sha(Path(sys.executable).read_bytes()),
        "platform": platform.platform(),
    }


def guarded_write(output, name, payload_bytes, maximum_bytes=CAP):
    output = Path(output)
    require(
        isinstance(name, str)
        and name not in ("", ".", "..")
        and Path(name).name == name
        and "/" not in name
        and "\\" not in name,
        "flat evidence filename",
    )
    require(
        type(payload_bytes) is bytes and type(maximum_bytes) is int and 0 <= maximum_bytes <= CAP,
        "bounded byte payload",
    )
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    require(size + len(payload_bytes) <= maximum_bytes, "10MiB storage bound")
    with (output / name).open("xb") as stream:
        stream.write(payload_bytes)


def freeze():
    cfg = config_at()
    require(not OUTPUT.exists(), "new namespace must be absent")
    _, inputs = load_inputs(cfg)
    record = {
        "config": cfg,
        "source_commit": git("rev-parse", "HEAD"),
        "source_sha256": source_identity(inputs),
        "input_sha256": inputs,
        "environment": environment(),
        "resources": dict(ZERO_CALLS),
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    guarded_write(OUTPUT, "preregistration.json", encoded(record))
    return {
        "status": "MODEL_FREE_RECIPE_FROZEN",
        "source_commit": record["source_commit"],
        **ZERO_CALLS,
    }


def derive():
    record = read(OUTPUT / "preregistration.json")
    cfg = config_at()
    vectors, inputs = load_inputs(cfg)
    require(
        record["config"] == cfg
        and record["input_sha256"] == inputs
        and record["source_sha256"] == source_identity(inputs)
        and record["environment"] == environment(),
        "frozen source/input/environment",
    )
    lock_path = cfg["output_namespace"] + "/preregistration.json"
    require(
        git("show", "--pretty=", "--name-only", "HEAD").splitlines() == [lock_path]
        and git("rev-parse", "HEAD^") == record["source_commit"]
        and not git("status", "--porcelain", "--", lock_path)
        and {p.name for p in OUTPUT.iterdir()} == {"preregistration.json"},
        "preregistration-only HEAD and unused namespace; no retry",
    )
    guarded_write(
        OUTPUT,
        "DERIVATION_STARTED.json",
        encoded({"source_commit": record["source_commit"], **ZERO_CALLS}),
    )
    result = compute(vectors["P"], vectors["C"])
    require(
        all(
            result["metrics"][k] > 0
            for k in ("P_norm", "C_norm", "midpoint_norm", "difference_norm")
        ),
        "nondegenerate actual geometry",
    )
    artifacts = {}
    for name in ("midpoint", "difference"):
        v = result[name]
        value = {
            "kind": name,
            "vector": v,
            "dimension": cfg["dimension"],
            "vector_float64_le_sha256": vector_sha(v),
            "operation": cfg["operations"][name],
            "norm": result["metrics"][name + "_norm"],
            "mathematical_only": True,
            "behavior_tested": False,
        }
        for filename, raw in ((name + ".json", encoded(value)), (name + ".f64", vector_bytes(v))):
            guarded_write(OUTPUT, filename, raw)
            artifacts[filename] = {"sha256": sha(raw), "bytes": len(raw)}
    receipt = {
        "status": "DERIVED_NOT_BEHAVIORALLY_TESTED",
        "metrics": result["metrics"],
        "artifacts": artifacts,
        "resources": dict(ZERO_CALLS),
        "preregistration_sha256": sha((OUTPUT / "preregistration.json").read_bytes()),
        "future_sign_conventions_only": {"+d": "proposed P", "-d": "proposed C"},
        "no_bias_cancellation_claim": True,
    }
    guarded_write(OUTPUT, "geometry.json", encoded(receipt))
    return receipt


if __name__ == "__main__":
    require(sys.argv[1:] in (["freeze"], ["derive"]), "Use freeze or derive; no recipe options")
    try:
        result = freeze() if sys.argv[1] == "freeze" else derive()
    except Exception as error:
        if OUTPUT.is_dir() and not (OUTPUT / "FAILURE.json").exists():
            guarded_write(
                OUTPUT,
                "FAILURE.json",
                encoded({"status": "INCONCLUSIVE", "fault": str(error), **ZERO_CALLS}),
            )
        raise
    print(json.dumps(result, indent=2, allow_nan=False))
