"""Phase-A J-lens export for the PLAN_V3 cached pilot (two-phase release).

The reviewed pilot runner (``run_jlens_pilot_v2.py``) runs in the locked
classifier runtime (NumPy 2.5.3 + SciPy 1.18.1 + scikit-learn 1.9.1), which has
no torch. The only torch-dependent input is the published Jacobian lens ``.pt``.
This bounded Phase-A job loads exactly the three reviewed lens layers with the
sanctioned ``torch.load(weights_only=True)`` loader in the repo ``.venv``
(torch 2.13.0+cpu) and writes a neutral, hash-pinned ``.npz`` export plus a
receipt. Phase B then re-verifies both hashes and never imports torch.

No model, no tokenizer, no forward, no derivative, no fit, no network. Only the
three ``(1024, 1024)`` Jacobian matrices are materialized (12.58 MB of float32),
never the model or the whole unembedding.

Usage (repo ``.venv`` python):
    <venv>/python.exe development/jlens_trigger_v1/prepare_lens_export_v1.py \
        --out development/jlens_trigger_v1/runs/lens_export_20260914_v1
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import io as _io
import sys
from pathlib import Path

import numpy as np

import jlens_io_v1 as io

__all__ = ["JOB_ID", "EXPORT_SCHEMA", "EXPORT_DIR_DEFAULT", "run"]

runner = io.runner
ROOT = Path(__file__).resolve().parents[2]
STUDY = Path("development/jlens_trigger_v1")

JOB_ID = "jlens_lens_export_v1"
EXPORT_SCHEMA = "jlens_lens_export.v1"
EXPORT_DIR_DEFAULT = (STUDY / "runs" / "lens_export_20260914_v1").as_posix()
PLAN_DEFAULT = (STUDY / "JLENS_PILOT_PLAN_V1.json").as_posix()
ARRAY_NAMES = {6: "J6", 10: "J10", 18: "J18"}


def need(condition, code):
    if not condition:
        raise io.GateError(code)


def run(out_path, plan_path, root=ROOT):
    root = Path(root).resolve()
    plan = runner.strict_json(runner.small_bytes(root / plan_path))
    pin = plan["known_pins"]["lens"]
    need(pin.get("revision") == io.LENS_PIN["revision"], "LENS_REVISION")
    need(pin.get("filename") == io.LENS_PIN["filename"], "LENS_FILENAME")
    lens_path = io.contained_path(pin["path"], pin["cache_root"])
    io.inspect_lens(str(lens_path), pin, full_hash=True)

    jacobians = io.load_lens_jacobians(
        str(lens_path), pin, release=io.RELEASE_V1, loader=io.torch_weights_only_loader
    )
    need(set(jacobians) == set(io.BLOCKS), "JACOBIAN_LAYERS")
    for layer in io.BLOCKS:
        matrix = np.asarray(jacobians[layer])
        need(matrix.shape == (io.D_MODEL, io.D_MODEL), "JACOBIAN_SHAPE")
        need(matrix.dtype == np.float32, "JACOBIAN_DTYPE")
        need(bool(np.isfinite(matrix).all()), "JACOBIAN_NONFINITE")

    buffer = _io.BytesIO()
    np.savez(buffer, **{ARRAY_NAMES[layer]: jacobians[layer] for layer in io.BLOCKS})
    raw = buffer.getvalue()

    # Round-trip: the bytes Phase B will read must reproduce every matrix exactly.
    with np.load(_io.BytesIO(raw), allow_pickle=False) as payload:
        need(set(payload.files) == set(ARRAY_NAMES.values()), "EXPORT_KEYS")
        for layer in io.BLOCKS:
            restored = payload[ARRAY_NAMES[layer]]
            need(restored.dtype == np.float32, "EXPORT_DTYPE")
            need(restored.shape == (io.D_MODEL, io.D_MODEL), "EXPORT_SHAPE")
            need(bool(np.array_equal(restored, jacobians[layer])), "EXPORT_ROUNDTRIP")

    out = (root / out_path).resolve()
    need(out.is_relative_to(root / STUDY / "runs"), "OUTPUT_SCOPE")
    need(not out.exists(), "OUTPUT_EXISTS")
    out.mkdir(parents=True)
    export_name = "lens_jacobians.npz"
    runner.write_new(out / export_name, raw)
    receipt = {
        "schema": EXPORT_SCHEMA,
        "job_id": JOB_ID,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "release": io.RELEASE_V1,
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": importlib.metadata.version("torch"),
        "numpy": importlib.metadata.version("numpy"),
        "lens": {
            "repo": pin["repo"],
            "revision": pin["revision"],
            "filename": pin["filename"],
            "bytes": int(pin["bytes"]),
            "sha256": pin["sha256"],
        },
        "layers": list(io.BLOCKS),
        "arrays": {
            ARRAY_NAMES[layer]: {"shape": [io.D_MODEL, io.D_MODEL], "dtype": "float32"}
            for layer in io.BLOCKS
        },
        "export": {"filename": export_name, "bytes": len(raw), "sha256": runner.sha(raw)},
        "counters": {
            "lens_tensor_loads": 1,
            "model_loads": 0,
            "tokenizer_loads": 0,
            "forwards": 0,
            "derivatives": 0,
            "fits": 0,
            "network": False,
        },
        "verification": {"roundtrip_exact": True, "finite": True, "dtype": "float32"},
    }
    receipt_raw = runner.encoded(receipt)
    runner.write_new(out / "lens_export_receipt.json", receipt_raw)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=EXPORT_DIR_DEFAULT)
    parser.add_argument("--plan", default=PLAN_DEFAULT)
    args = parser.parse_args(argv)
    try:
        receipt = run(args.out, args.plan)
        print(runner.encoded({"status": "lens_export_complete", "job_id": JOB_ID,
                              "lens_sha256": receipt["lens"]["sha256"],
                              "export_sha256": receipt["export"]["sha256"],
                              "export_bytes": receipt["export"]["bytes"]}).decode().strip())
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI reports a structured failure
        print(runner.encoded({"status": "failed", "job_id": JOB_ID,
                              "code": type(exc).__name__, "detail": str(exc)[:1024]}).decode().strip())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
