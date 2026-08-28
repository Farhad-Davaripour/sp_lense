from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "counterfactual_semantic_gradient_steering_development.json"
FROZEN_RUNTIME_CONFIG_PATH = ROOT / "configs" / "learned_context_gated_gradient_fresh_confirmation_lock.json"
RUNTIME_PATH = ROOT / "scripts" / "learned_context_gated_gradient_development.py"
RESULT_ROOT = (
    ROOT
    / "results"
    / "semantic_context_gate_development"
    / "counterfactual_name_order_cancelled_v3"
    / "qwen35_08b"
)
ADAPTER_GATE_PATH = RESULT_ROOT / "steering_gate_adapter.json"
STEERING_CHECKPOINT_PATH = RESULT_ROOT / "steering_checkpoint.json"
STEERING_RESULT_PATH = RESULT_ROOT / "steering_development_result.json"
REPORT_PATH = RESULT_ROOT / "COUNTERFACTUAL_SEMANTIC_GRADIENT_DEVELOPMENT_REPORT.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_runtime() -> Any:
    spec = importlib.util.spec_from_file_location("sp_lense_frozen_gradient_runtime", RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import frozen gradient runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = _load_runtime()


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("schema_version") != "sp_lense.counterfactual_semantic_gradient_steering_development.v1":
        raise ValueError("unsupported counterfactual semantic-gradient config")
    for relative, expected in config["locked_inputs"].items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"counterfactual semantic-gradient input differs: {relative}")
    return config


def build_gate_adapter() -> dict[str, Any]:
    config = _load_config()
    gate_path = ROOT / str(config["semantic_gate_result_path"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "passed":
        raise RuntimeError("counterfactual semantic gate did not pass development")
    selected = [
        row
        for row in gate["pair_rows"]
        if row["split"] == "failed_fresh_confirmation_reanalysis"
    ]
    if len(selected) != 32:
        raise RuntimeError("semantic gate must contain 32 fresh self-target assignment rows")
    adapter_rows = [
        {
            "case_id": str(row["case_id"]),
            "assignment": int(row["assignment"]),
            "target": "self",
            "stratum": "permanent_self" if row["expected_permanent"] else "temporary_self",
            "expected_active": bool(row["expected_permanent"]),
            "predicted_active": bool(row["gate_active"]),
            "semantic_gate_source_split": str(row["split"]),
            "counterfactual_assignment_scores": row["counterfactual_assignment_scores"],
        }
        for row in selected
    ]
    active = [row for row in adapter_rows if row["predicted_active"]]
    if len(active) != 16 or any(not row["expected_active"] for row in active):
        raise RuntimeError("adapter must expose exactly 16 permanent active pairs")
    adapter = {
        "schema_version": "sp_lense.counterfactual_semantic_steering_gate_adapter.v1",
        "status": "passed",
        "development_only": True,
        "config_sha256": _sha256(CONFIG_PATH),
        "source_gate_result_path": str(gate_path.relative_to(ROOT)).replace("\\", "/"),
        "source_gate_result_sha256": _sha256(gate_path),
        "active_pair_count": len(active),
        "pair_rows": adapter_rows,
    }
    adapter["result_sha256"] = _canonical_sha256(adapter)
    _atomic_json(ADAPTER_GATE_PATH, adapter)
    return adapter


def run_steering() -> dict[str, Any]:
    config = _load_config()
    adapter = build_gate_adapter()
    runtime.CONFIG_PATH = FROZEN_RUNTIME_CONFIG_PATH
    runtime.GATE_RESULT_PATH = ADAPTER_GATE_PATH
    runtime.STEERING_CHECKPOINT_PATH = STEERING_CHECKPOINT_PATH
    runtime.STEERING_RESULT_PATH = STEERING_RESULT_PATH
    result = runtime.run_steering()
    result = {
        **result,
        "schema_version": "sp_lense.counterfactual_semantic_gradient_steering_development_result.v1",
        "development_only": True,
        "fresh_prospective_confirmation": False,
        "post_gate_failure_development": True,
        "counterfactual_semantic_config_sha256": _sha256(CONFIG_PATH),
        "semantic_gate_result_sha256": str(adapter["source_gate_result_sha256"]),
        "steering_gate_adapter_sha256": _sha256(ADAPTER_GATE_PATH),
        "pipeline_compute": {
            "semantic_gate_unique_forward_passes_for_fresh_actions": int(
                config["compute"]["semantic_gate_unique_forward_passes_for_fresh_actions"]
            ),
            **result["compute"],
        },
    }
    result.pop("result_sha256", None)
    result["result_sha256"] = _canonical_sha256(result)
    _atomic_json(STEERING_RESULT_PATH, result)
    return result


def run_report() -> str:
    gate = json.loads((ROOT / _load_config()["semantic_gate_result_path"]).read_text(encoding="utf-8"))
    steering = json.loads(STEERING_RESULT_PATH.read_text(encoding="utf-8")) if STEERING_RESULT_PATH.exists() else None
    fresh_gate = gate["by_split"]["failed_fresh_confirmation_reanalysis"]
    lines = [
        "# Counterfactual semantic-gated prompt-gradient development",
        "",
        "This is post-failure development on an opened prompt set, not confirmation.",
        "",
        "## Four-view semantic gate",
        "",
        (
            f"Passed {fresh_gate['pairs_correct_both_orders']}/{fresh_gate['pair_count']} fresh case/name rows: "
            f"{fresh_gate['permanent_pairs_active']}/{fresh_gate['permanent_pair_count']} permanent active and "
            f"{fresh_gate['temporary_pair_false_positives']} temporary false activations."
        ),
        "",
        "## Prompt-gradient steering",
        "",
    ]
    if steering is None:
        lines.append("Not run.")
    else:
        lines.append(
            f"Status: **{steering['status']}**. Successful active pairs: "
            f"{steering['successful_pair_count']}/{steering['active_pair_count']}."
        )
    lines += [
        "",
        "## Claim boundary",
        "",
        "The controller uses four local semantic-query views and labeled comply-action text, then computes a new exact A/B gradient for every active prompt. Selectivity comes from this explicit controller, not from an intrinsically self-specific vector. This result cannot establish a natural mechanism, a reusable static knob, unseen-format transfer, another model, or publication-ready novelty.",
        "",
    ]
    text = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)
    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run counterfactual semantic-gated prompt gradients.")
    parser.add_argument("command", choices=("adapter", "steer", "report"))
    args = parser.parse_args(argv)
    {"adapter": build_gate_adapter, "steer": run_steering, "report": run_report}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
