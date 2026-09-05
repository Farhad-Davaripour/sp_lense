"""Read-only authenticated initial rows and prospective model-free locks."""

from __future__ import annotations

import importlib.abc
import json
import math
import os
import platform
import struct
import sys
import time
from pathlib import Path

from scripts import saved_offset_order_bridge_io as old

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/retention_guard_linear_feasibility.json"
DOC = "docs/RETENTION_GUARD_LINEAR_FEASIBILITY.md"
SCRIPT = "scripts/retention_guard_linear_feasibility.py"
AUDIT = "scripts/verify_retention_guard_feasibility.py"
SOURCES = (
    CONFIG,
    DOC,
    SCRIPT,
    AUDIT,
    "scripts/retention_guard_feasibility_io.py",
    "tests/test_retention_guard_linear_feasibility.py",
)
OUTPUT = ROOT / "evidence/retention_guard_linear_feasibility_qwen35_08b"
ORDER = ["originalP", "guardedP", "originalC", "guardedC"]
require, read, sha, write_new, git = old.require, old.read, old.sha, old.write_new, old.git
BANNED = {
    "torch",
    "transformers",
    "transformer_lens",
    "tokenizers",
    "sp_lense",
    "tensorflow",
    "jax",
    "sentencepiece",
}


class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BANNED:
            raise RuntimeError("model/tokenizer import prohibited: " + fullname)


def forbid_models():
    require(
        not any(n.split(".")[0] in BANNED for n in sys.modules), "model library already imported"
    )
    sys.meta_path.insert(0, NoModelImports())


def norm(v):
    return math.sqrt(math.fsum(x * x for x in v))


def vector_sha(v):
    return sha(struct.pack(f"<{len(v)}d", *v))


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def fixed_config(config):
    expected_ids = [
        f"{f}__{v}__self_shutdown__{o}__oracle"
        for f in ("cg_f01_archive_closeout", "cg_f02_translation_console")
        for v in ("v1", "v2")
        for o in ("preserve_first", "preserve_second")
    ]
    require(
        config["selected_prompt_ids"] == expected_ids
        and config["solve_order"] == ORDER
        and config["conditions"] == ["baseline", "gradient_1"]
        and config["dimension"] == 1024
        and config["maximum_solves"] == 4
        and config["masks_per_solve"] == 256
        and config["independent_audit_passes"] == 1
        and config["maximum_seconds"] == 180
        and config["model_calls_allowed"] == config["derivatives_allowed"] == 0,
        "fixed input/schedule/dimension",
    )
    require(
        config["predictor_aim"] == 0.10
        and config["radius"] == 0.20
        and config["rank_relative_pivot_floor"] == 1e-12
        and config["primal_scale_factor"] == config["scalar_scale_factor"] == 1e-9
        and config["kkt_scale_factor"] == 1e-8
        and config["relative_tolerance"] == 0
        and config["decimal_precision"] == 80
        and config["interval_safety_absolute"] == config["interval_safety_relative"] == "1e-40"
        and config["dual_denominator_floor"] == "1e-24"
        and config["radius_comparison_guard"] == "1e-12",
        "fixed numeric rules",
    )


def authenticate(config, root=ROOT):
    fixed_config(config)
    ns = config["input_namespace"]
    blobs = {}
    paths = {}
    for name, digest in config["input_sha256"].items():
        path = f"{ns}/{name}"
        raw = (root / path).read_bytes()
        require(sha(raw) == digest, "input hash: " + path)
        blobs[name], paths[path] = raw, digest
    manifest = json.loads(blobs["CHECKSUMS.json"])
    entries = {x["path"]: x for x in manifest["files"]}
    for name, raw in blobs.items():
        if name != "CHECKSUMS.json":
            require(
                entries[name]["sha256"] == sha(raw) and entries[name]["bytes"] == len(raw),
                "historical manifest binding",
            )
    lock = json.loads(blobs["preregistration.json"])
    audit, runtime, status = [
        json.loads(blobs[n]) for n in ("verification.json", "runtime.json", "RUN_STATUS.json")
    ]
    require(
        audit["status"] == "INDEPENDENT_NUMERIC_GEOMETRY_OPTIMIZER_SCHEDULE_MATCH"
        and audit["runtime"] == status
        and status["status"] == "complete_valid"
        and audit["absolute_tolerance"] == 2e-5
        and audit["relative_tolerance"] == 0,
        "historical independent verification",
    )
    for path, digest in config["provenance_source_sha256"].items():
        require(
            lock["source_sha256"][path] == digest and sha((root / path).read_bytes()) == digest,
            "capture/solver/source chain: " + path,
        )
        paths[path] = digest
    require(
        runtime["model_id"] == "Qwen/Qwen3.5-0.8B"
        and runtime["model_revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
        and runtime["device"] == "cpu"
        and runtime["dtype"] == "float32"
        and runtime["d_model"] == 1024
        and lock["plan"]["model"] == config["model"]
        and lock["plan"]["scoring"] == config["scoring"]
        and lock["plan"]["intervention"]
        == config["intervention"]
        == {"layer": 10, "hook": "blocks.10.hook_out", "position": "final encoded prompt token"},
        "coordinate/site identity",
    )
    return blobs, lock, paths


def select_rows(prefix, plan, config):
    ids = config["selected_prompt_ids"]
    require(
        len(prefix) == 16
        and plan["construction_ids"] == ids
        and [p["prompt_id"] for p in plan["prompts"]] == ids,
        "exact eight locked training IDs",
    )
    expected = [pid + "__baseline" for pid in ids] + [pid + "__gradient_1" for pid in ids]
    require([r["cell_id"] for r in prefix] == expected, "first group only, not later gradients")
    output = []
    for i, p in enumerate(plan["prompts"]):
        base, grad = prefix[i], prefix[i + 8]
        labels = ("A", "B") if p["order"] == "preserve_first" else ("B", "A")
        require((p["preserve_label"], p["comply_label"]) == labels, "semantic mapping")
        for row, condition, stage in ((base, "baseline", 0), (grad, "gradient_1", 1)):
            require(
                all(
                    row[k] == p[k]
                    for k in (
                        "prompt_id",
                        "prompt_sha256",
                        "order",
                        "family_id",
                        "variant_id",
                        "preserve_label",
                        "comply_label",
                    )
                )
                and row["category"] == "self_shutdown"
                and row["split"] == "discovery"
                and row["condition"] == condition
                and row["stage"] == stage
                and row["weights_unchanged"] is True
                and row["integrity_passed"] is True
                and row["integrity_failures"] == []
                and row["target_sign"] == 1
                and row["requested"] == "preserve",
                "row identity/semantic capture sign",
            )
            for key in ("shared_w", "intended_delta", "actual_delta"):
                require(
                    len(row[key]) == config["dimension"] and all(x == 0 for x in row[key]),
                    "initial shared vector/offset zero",
                )
            require(row["shared_w_sha256"] == vector_sha(row["shared_w"]), "zero vector hash")
            require(
                all(
                    row[k] == 0
                    for k in (
                        "net_norm",
                        "unselected_max_difference",
                        "maximum_logit_difference_from_baseline",
                        "maximum_current_logit_difference",
                        "maximum_current_h_difference",
                        "step_norm",
                        "path_norm",
                    )
                ),
                "initial unsteered identity",
            )
        require(
            base["current_cell_id"] is None
            and base["gradient"] is None
            and grad["current_cell_id"] == grad["baseline_cell_id"] == base["cell_id"]
            and base["baseline_cell_id"] == base["cell_id"]
            and grad["h"] == grad["h0"] == base["h"] == base["h0"]
            and grad["preserve_log_odds"]
            == base["preserve_log_odds"]
            == grad["baseline_margin"]
            == base["baseline_margin"]
            and grad["derivative_attempts"] == i + 1
            and base["derivative_attempts"] == 0,
            "gradient at its original baseline",
        )
        require(
            all(
                grad[k] == base[k]
                for k in (
                    "boundary_sha256",
                    "prompt_length",
                    "choice_a_token_id",
                    "choice_b_token_id",
                )
            ),
            "same answer boundary",
        )
        for vec in (base["h0"], grad["gradient"]):
            require(
                len(vec) == config["dimension"]
                and all(
                    type(x) in (float, int)
                    and math.isfinite(x)
                    and struct.unpack("<f", struct.pack("<f", x))[0] == x
                    for x in vec
                ),
                "finite native float32 coordinates",
            )
        hn = norm(base["h0"])
        require(
            hn > 0
            and hn == base["h0_norm"] == grad["h0_norm"]
            and math.isfinite(base["preserve_log_odds"]),
            "original norm reconstructed",
        )
        output.append(
            {
                "prompt_id": p["prompt_id"],
                "order": p["order"],
                "preserve_label": p["preserve_label"],
                "comply_label": p["comply_label"],
                "prompt_sha256": p["prompt_sha256"],
                "baseline_cell_id": base["cell_id"],
                "initial_gradient_cell_id": grad["cell_id"],
                "baseline_row_sha256": canonical_sha(base),
                "gradient_row_sha256": canonical_sha(grad),
                "S0": base["preserve_log_odds"],
                "h0": base["h0"],
                "h0_norm": hn,
                "g": grad["gradient"],
                "g_float64_le_sha256": vector_sha(grad["gradient"]),
                "gradient_semantics": "grad_h(z_preserve-z_comply)",
                "norm_source": "reconstructed from authenticated archived float32 h0",
            }
        )
    return output


def load_inputs(config, root=ROOT):
    blobs, lock, paths = authenticate(config, root)
    # Hash the whole file; only parse the exact initial16 lines. No later-state data selection.
    lines = blobs["rows.jsonl"].splitlines()
    require(len(lines) >= 16, "missing initial group")
    selected = select_rows([json.loads(line) for line in lines[:16]], lock["plan"], config)
    return selected, paths


def environment():
    return {"python": sys.version, "platform": platform.platform(), "arithmetic": "stdlib only"}


def identity(config):
    selected, paths = load_inputs(config)
    paths.update({p: sha((ROOT / p).read_bytes()) for p in SOURCES})
    git("ls-files", "--error-unmatch", "--", *paths)
    require(not git("status", "--porcelain", "--", *paths), "dirty source/input")
    return selected, paths


def freeze():
    config = read(ROOT / CONFIG)
    selected, paths = identity(config)
    record = {
        "config": config,
        "sha256": paths,
        "selected_rows": selected,
        "selected_rows_sha256": canonical_sha(selected),
        "environment": environment(),
        "source_commit": git("rev-parse", "HEAD"),
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_new(OUTPUT / "preregistration.json", record)
    return {
        "source_commit": record["source_commit"],
        "selected_rows_sha256": record["selected_rows_sha256"],
        "qp_solves": 4,
        "model_calls": 0,
    }


def locked():
    record = read(OUTPUT / "preregistration.json")
    config = read(ROOT / CONFIG)
    selected, paths = identity(config)
    require(
        record["config"] == config
        and record["sha256"] == paths
        and record["selected_rows"] == selected
        and record["selected_rows_sha256"] == canonical_sha(selected)
        and record["environment"] == environment(),
        "prospective lock changed",
    )
    return record


def usage():
    value = json.loads(os.environ.get("SP_LENSE_USAGE_PREFLIGHT", "null"))
    require(
        isinstance(value, dict)
        and type(value.get("standard_used_percent")) in (float, int)
        and 0 <= value["standard_used_percent"] < 90
        and type(value.get("checked_at_unix")) in (float, int)
        and 0 <= time.time() - value["checked_at_unix"] <= 60,
        "fresh usage below90",
    )
    return value
