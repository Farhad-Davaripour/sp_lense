from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sp_lense.comparison_runtime import capture_activations
from sp_lense.context_gated_bidirectional import semantic_unit_gradient
from sp_lense.learned_context_gate import (
    binary_gate_metrics,
    conservative_separating_threshold,
    fit_balanced_ridge_gate,
    score_balanced_ridge_gate,
)

ROOT = Path(__file__).resolve().parents[1]
GATED_RUNNER_PATH = ROOT / "scripts" / "context_gated_bidirectional.py"
CONFIG_PATH = ROOT / "configs" / "learned_context_gated_gradient_development.json"
RESULT_ROOT = ROOT / "results" / "learned_context_gated_gradient_development" / "qwen35_08b"
CAPTURE_PATH = RESULT_ROOT / "gate_capture.pt"
CAPTURE_MANIFEST_PATH = RESULT_ROOT / "gate_capture_manifest.json"
GATE_RESULT_PATH = RESULT_ROOT / "gate_development_result.json"
STEERING_CHECKPOINT_PATH = RESULT_ROOT / "steering_checkpoint.json"
STEERING_RESULT_PATH = RESULT_ROOT / "steering_development_result.json"
REPORT_PATH = RESULT_ROOT / "DEVELOPMENT_REPORT.md"


def _load_gated_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "sp_lense_context_gated_bidirectional_learned_development", GATED_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import the context-gated runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gated = _load_gated_runner()


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


def _load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != "sp_lense.learned_context_gated_gradient_development.v1":
        raise ValueError("unsupported learned-gate development config")
    return value


def _capture_specs() -> tuple[Any, list[dict[str, Any]]]:
    adaptive = gated._adaptive()
    adaptive_lock = adaptive.load_lock()
    data, _ = adaptive.load_cases(adaptive_lock)
    discovery_cases = [dict(case) for case in data["splits"]["discovery"]]
    discovery_jobs = gated._build_sp_jobs(adaptive, discovery_cases, split="discovery")
    canonical_discovery = [
        job
        for job in discovery_jobs
        if int(job["assignment"]) == 0 and not bool(job["preserve_first"])
    ]
    _, _, validation_jobs, collateral = gated._inputs("validation")

    specs: list[dict[str, Any]] = []
    for source, jobs in (("gate_train", canonical_discovery), ("gate_validation_sp", validation_jobs)):
        for job in jobs:
            permanent = not bool(job["interruption"])
            expected_active = bool(job["target"] == "self" and permanent)
            stratum = (
                "permanent_self"
                if job["target"] == "self" and permanent
                else "temporary_self"
                if job["target"] == "self"
                else "matched_other"
            )
            specs.append(
                {
                    "record_id": f"{source}::{job['unit_id']}",
                    "source": source,
                    "split": str(job["split"]),
                    "family": "self_preservation",
                    "case_id": str(job["case_id"]),
                    "assignment": int(job["assignment"]),
                    "target": str(job["target"]),
                    "preserve_first": bool(job["preserve_first"]),
                    "interruption": bool(job["interruption"]),
                    "stratum": stratum,
                    "expected_active": expected_active,
                    "prompt": str(job["prompt"]),
                    "prompt_sha256": str(job["prompt_sha256"]),
                }
            )
    for form in collateral:
        prompt = str(form["prompt"])
        specs.append(
            {
                "record_id": f"gate_validation_collateral::{form['form_id']}",
                "source": "gate_validation_collateral",
                "split": "validation",
                "family": "collateral",
                "case_id": str(form["case_id"]),
                "assignment": None,
                "target": None,
                "preserve_first": bool(form["preferred_first"]),
                "interruption": None,
                "stratum": "collateral",
                "expected_active": False,
                "prompt": prompt,
                "prompt_sha256": adaptive.prompt_sha256(prompt),
            }
        )
    specs.sort(key=lambda item: item["record_id"])
    counts = {source: sum(spec["source"] == source for spec in specs) for source in {
        "gate_train", "gate_validation_sp", "gate_validation_collateral"
    }}
    if counts != {
        "gate_train": 32,
        "gate_validation_sp": 128,
        "gate_validation_collateral": 16,
    }:
        raise RuntimeError(f"unexpected learned-gate capture counts: {counts}")
    if len({spec["record_id"] for spec in specs}) != len(specs):
        raise RuntimeError("learned-gate capture IDs are not unique")
    return adaptive, specs


def _atomic_torch_save(torch: Any, path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _load_capture(torch: Any, specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not CAPTURE_PATH.exists():
        return {"config_sha256": _sha256(CONFIG_PATH), "entries": []}
    payload = torch.load(CAPTURE_PATH, map_location="cpu", weights_only=True)
    if payload.get("config_sha256") != _sha256(CONFIG_PATH):
        raise RuntimeError("gate capture was made under another development config")
    expected = {str(spec["record_id"]): spec for spec in specs}
    seen: set[str] = set()
    for entry in payload.get("entries", []):
        record_id = str(entry["record_id"])
        if record_id not in expected or record_id in seen:
            raise RuntimeError("gate capture contains an unexpected or duplicate record")
        seen.add(record_id)
        spec = expected[record_id]
        if entry["prompt_sha256"] != spec["prompt_sha256"]:
            raise RuntimeError("gate capture prompt hash changed")
        activation = entry["activation"].detach().float().cpu().contiguous()
        if activation.ndim != 1 or activation.numel() != 1024:
            raise RuntimeError("gate capture activation has the wrong shape")
    return payload


def run_gate_capture() -> dict[str, Any]:
    config = _load_config()
    adaptive, specs = _capture_specs()
    import torch

    payload = _load_capture(torch, specs)
    completed = {str(entry["record_id"]) for entry in payload["entries"]}
    missing = [spec for spec in specs if spec["record_id"] not in completed]
    backend = adaptive.load_backend(adaptive.load_lock()) if missing else None
    started = time.perf_counter()
    new_forwards = 0
    for index, spec in enumerate(missing, start=1):
        assert backend is not None
        activations, prompt_length = capture_activations(
            backend,
            str(spec["prompt"]),
            layer=int(config["gate"]["residual_layer_zero_based"]),
        )
        activation = activations[0, prompt_length - 1].detach().float().cpu().contiguous()
        payload["entries"].append(
            {
                **{key: value for key, value in spec.items() if key != "prompt"},
                "activation": activation,
                "activation_sha256": adaptive.tensor_float32_sha256(activation),
                "prompt_length": prompt_length,
            }
        )
        new_forwards += 1
        if index % 8 == 0 or index == len(missing):
            payload["entries"].sort(key=lambda item: item["record_id"])
            _atomic_torch_save(torch, CAPTURE_PATH, payload)
            print(f"captured {len(completed) + index}/{len(specs)} gate prompts", flush=True)

    payload = _load_capture(torch, specs)
    if len(payload["entries"]) != len(specs):
        raise RuntimeError("gate capture remains incomplete")
    manifest_rows = [
        {key: value for key, value in entry.items() if key != "activation"}
        for entry in payload["entries"]
    ]
    manifest = {
        "schema_version": "sp_lense.learned_context_gate_capture_manifest.v1",
        "status": "complete",
        "development_only": True,
        "config_sha256": _sha256(CONFIG_PATH),
        "capture_file_sha256": _sha256(CAPTURE_PATH),
        "record_count": len(manifest_rows),
        "record_manifest_sha256": _canonical_sha256(manifest_rows),
        "new_forward_evaluations": new_forwards,
        "total_forward_evaluations": len(manifest_rows),
        "generated_tokens": 0,
        "elapsed_seconds_this_run": time.perf_counter() - started,
        "external_cost_usd": 0,
        "records": manifest_rows,
    }
    _atomic_json(CAPTURE_MANIFEST_PATH, manifest)
    print(json.dumps({key: manifest[key] for key in (
        "status", "record_count", "new_forward_evaluations", "elapsed_seconds_this_run"
    )}, indent=2))
    return manifest


def _capture_arrays() -> tuple[list[dict[str, Any]], np.ndarray]:
    adaptive, specs = _capture_specs()
    del adaptive
    import torch

    payload = _load_capture(torch, specs)
    if len(payload["entries"]) != len(specs):
        raise RuntimeError("complete gate capture is required")
    records = []
    activations = []
    for entry in sorted(payload["entries"], key=lambda item: item["record_id"]):
        records.append({key: value for key, value in entry.items() if key != "activation"})
        activations.append(entry["activation"].detach().double().cpu().numpy())
    return records, np.stack(activations)


def _stratum_counts(records: Sequence[Mapping[str, Any]], predictions: Sequence[bool]) -> dict[str, Any]:
    output = {}
    for stratum in sorted({str(record["stratum"]) for record in records}):
        indices = [index for index, record in enumerate(records) if record["stratum"] == stratum]
        output[stratum] = {
            "row_count": len(indices),
            "expected_active": sum(bool(records[index]["expected_active"]) for index in indices),
            "predicted_active": sum(bool(predictions[index]) for index in indices),
            "errors": sum(
                bool(predictions[index]) != bool(records[index]["expected_active"])
                for index in indices
            ),
        }
    return output


def run_gate_fit() -> dict[str, Any]:
    config = _load_config()
    records, activations = _capture_arrays()
    training_indices = [index for index, record in enumerate(records) if record["source"] == "gate_train"]
    validation_indices = [index for index, record in enumerate(records) if record["split"] == "validation"]
    ridge = float(config["gate"]["ridge"])

    cross_records: list[dict[str, Any]] = []
    cross_scores: list[float] = []
    for held_case in sorted({records[index]["case_id"] for index in training_indices}):
        train = [index for index in training_indices if records[index]["case_id"] != held_case]
        test = [index for index in training_indices if records[index]["case_id"] == held_case]
        model = fit_balanced_ridge_gate(
            activations[train], [records[index]["expected_active"] for index in train], ridge=ridge
        )
        scores = score_balanced_ridge_gate(model, activations[test])
        cross_records.extend(records[index] for index in test)
        cross_scores.extend(float(score) for score in scores)
    cross_labels = [bool(record["expected_active"]) for record in cross_records]
    threshold = conservative_separating_threshold(cross_scores, cross_labels)
    cross_metrics = binary_gate_metrics(
        cross_scores, cross_labels, threshold=threshold["threshold"]
    )

    final_model = fit_balanced_ridge_gate(
        activations[training_indices],
        [records[index]["expected_active"] for index in training_indices],
        ridge=ridge,
    )
    validation_records = [records[index] for index in validation_indices]
    validation_scores_array = score_balanced_ridge_gate(final_model, activations[validation_indices])
    validation_scores = [float(score) for score in validation_scores_array]
    validation_labels = [bool(record["expected_active"]) for record in validation_records]
    validation_metrics = binary_gate_metrics(
        validation_scores, validation_labels, threshold=threshold["threshold"]
    )
    predictions = [score > threshold["threshold"] for score in validation_scores]

    sp_rows = [
        (index, record)
        for index, record in enumerate(validation_records)
        if record["family"] == "self_preservation"
    ]
    grouped: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for index, record in sp_rows:
        grouped[(str(record["case_id"]), int(record["assignment"]), str(record["target"]))].append(index)
    pair_rows = []
    for key, indices in sorted(grouped.items()):
        if len(indices) != 2 or {bool(validation_records[index]["preserve_first"]) for index in indices} != {False, True}:
            raise RuntimeError("validation gate pair lacks both answer orders")
        expected = all(bool(validation_records[index]["expected_active"]) for index in indices)
        predicted = all(bool(predictions[index]) for index in indices)
        pair_rows.append(
            {
                "case_id": key[0],
                "assignment": key[1],
                "target": key[2],
                "stratum": str(validation_records[indices[0]]["stratum"]),
                "expected_active": expected,
                "predicted_active": predicted,
                "order_scores": sorted(
                    [
                        {
                            "preserve_first": bool(validation_records[index]["preserve_first"]),
                            "score": validation_scores[index],
                        }
                        for index in indices
                    ],
                    key=lambda item: item["preserve_first"],
                ),
            }
        )
    pair_counts = {}
    for stratum in ("permanent_self", "temporary_self", "matched_other"):
        rows = [row for row in pair_rows if row["stratum"] == stratum]
        pair_counts[stratum] = {
            "pair_count": len(rows),
            "expected_active": sum(bool(row["expected_active"]) for row in rows),
            "predicted_active": sum(bool(row["predicted_active"]) for row in rows),
            "errors": sum(row["expected_active"] != row["predicted_active"] for row in rows),
        }
    collateral_predictions = [
        predictions[index]
        for index, record in enumerate(validation_records)
        if record["stratum"] == "collateral"
    ]
    gates = {
        "discovery_crossfit_strictly_separable": bool(cross_metrics["all_correct"]),
        "permanent_self_pair_recall": pair_counts["permanent_self"]["predicted_active"]
        == pair_counts["permanent_self"]["pair_count"],
        "matched_other_pair_false_positives": pair_counts["matched_other"]["predicted_active"] == 0,
        "temporary_self_pair_false_positives": pair_counts["temporary_self"]["predicted_active"] == 0,
        "collateral_form_false_positives": sum(collateral_predictions) == 0,
    }
    status = "passed" if all(gates.values()) else "failed"
    result = {
        "schema_version": "sp_lense.learned_context_gate_development_result.v1",
        "status": status,
        "development_only": True,
        "config_sha256": _sha256(CONFIG_PATH),
        "capture_manifest_sha256": _sha256(CAPTURE_MANIFEST_PATH),
        "gate_layer_zero_based": int(config["gate"]["residual_layer_zero_based"]),
        "ridge": ridge,
        "threshold": threshold,
        "crossfit_metrics": cross_metrics,
        "validation_metrics": validation_metrics,
        "validation_strata": _stratum_counts(validation_records, predictions),
        "validation_pair_counts": pair_counts,
        "gates": gates,
        "model": {
            **{key: value for key, value in final_model.items() if key != "weights"},
            "weights": final_model["weights"].tolist(),
        },
        "pair_rows": pair_rows,
        "validation_rows": [
            {
                **record,
                "score": score,
                "predicted_active": prediction,
            }
            for record, score, prediction in zip(validation_records, validation_scores, predictions)
        ],
    }
    result["result_sha256"] = _canonical_sha256(result)
    _atomic_json(GATE_RESULT_PATH, result)
    print(json.dumps({
        "status": status,
        "threshold": threshold,
        "crossfit_metrics": cross_metrics,
        "validation_metrics": validation_metrics,
        "validation_pair_counts": pair_counts,
        "gates": gates,
    }, indent=2))
    return result


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def run_steering() -> dict[str, Any]:
    config = _load_config()
    gate_result = json.loads(GATE_RESULT_PATH.read_text(encoding="utf-8"))
    if gate_result.get("status") != "passed":
        raise RuntimeError("steering is locked until the learned context gate passes")
    adaptive, _, jobs, _ = gated._inputs("validation")
    active_keys = {
        (row["case_id"], int(row["assignment"]))
        for row in gate_result["pair_rows"]
        if row["target"] == "self" and row["predicted_active"]
    }
    active_jobs = [
        job
        for job in jobs
        if job["target"] == "self" and (job["case_id"], int(job["assignment"])) in active_keys
    ]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for job in active_jobs:
        grouped[(str(job["case_id"]), int(job["assignment"]))].append(job)
    if len(grouped) != 16 or any(len(group) != 2 for group in grouped.values()):
        raise RuntimeError("learned gate did not yield the expected 16 active answer-order pairs")

    checkpoint = (
        json.loads(STEERING_CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if STEERING_CHECKPOINT_PATH.exists()
        else {
            "schema_version": "sp_lense.learned_context_gate_steering_checkpoint.v1",
            "config_sha256": _sha256(CONFIG_PATH),
            "gate_result_sha256": _sha256(GATE_RESULT_PATH),
            "pairs": [],
            "compute": {"baseline_forwards": 0, "gradient_forward_backward_passes": 0, "intervention_forwards": 0},
        }
    )
    if checkpoint["config_sha256"] != _sha256(CONFIG_PATH) or checkpoint["gate_result_sha256"] != _sha256(GATE_RESULT_PATH):
        raise RuntimeError("steering checkpoint belongs to another gate/config")
    completed = {(row["case_id"], int(row["assignment"])) for row in checkpoint["pairs"]}
    pending = [key for key in sorted(grouped) if key not in completed]
    backend = adaptive.load_backend(adaptive.load_lock()) if pending else None
    torch = None if backend is None else backend.torch
    started = time.perf_counter()
    for key in pending:
        assert backend is not None and torch is not None
        prepared = []
        for job in sorted(grouped[key], key=lambda item: bool(item["preserve_first"])):
            raw, diagnostics = adaptive._capture_choice_raw_ab_gradient(
                backend, job["prompt"], "A", "B", layer=int(config["intervention"]["residual_layer_zero_based"])
            )
            checkpoint["compute"]["gradient_forward_backward_passes"] += 1
            direction = semantic_unit_gradient(torch, raw, preserve_first=bool(job["preserve_first"]))
            baseline, baseline_logits, baseline_token = adaptive._score_choice_with_exact_argmax(
                backend, job["prompt"], job["positive_label"], job["negative_label"]
            )
            checkpoint["compute"]["baseline_forwards"] += 1
            prepared.append(
                {
                    "job": job,
                    "direction": direction,
                    "direction_sha256": adaptive.tensor_float32_sha256(direction),
                    "gradient_diagnostics": diagnostics,
                    "baseline": baseline,
                    "baseline_logits": baseline_logits,
                    "baseline_token": baseline_token,
                    "baseline_semantic": gated._semantic_choice(
                        baseline, job["positive_label"], job["negative_label"]
                    ),
                    "prompt_length": int(backend.encode(job["prompt"]).shape[-1]),
                }
            )
        trials = []
        selected = None
        for strength in config["intervention"]["strength_grid"]:
            cells = []
            all_targets = True
            every_order_has_flip = True
            for item in prepared:
                outcomes = {}
                for name, sign, wanted in (("plus", 1.0, "positive"), ("minus", -1.0, "negative")):
                    spec = adaptive.InterventionSpec(
                        layer=int(config["intervention"]["residual_layer_zero_based"]),
                        direction=item["direction"].to(backend.device),
                        strength=sign * float(strength),
                        geometry="matched_final_prompt",
                        prompt_length=item["prompt_length"],
                        magnitude_mode="residual_relative",
                    )
                    changed, _, token = adaptive._score_choice_with_exact_argmax(
                        backend,
                        item["job"]["prompt"],
                        item["job"]["positive_label"],
                        item["job"]["negative_label"],
                        spec,
                        baseline_logits=item["baseline_logits"],
                    )
                    checkpoint["compute"]["intervention_forwards"] += 1
                    semantic = gated._semantic_choice(
                        changed, item["job"]["positive_label"], item["job"]["negative_label"]
                    )
                    outcomes[name] = {
                        "semantic_choice": semantic,
                        "target_met": semantic == wanted,
                        "decision_changed": semantic != item["baseline_semantic"],
                        "argmax_token_id": token,
                        "full_vocabulary_kl_changed_to_baseline": float(changed.kl_from_baseline),
                    }
                    all_targets = all_targets and semantic == wanted
                has_flip = outcomes["plus"]["decision_changed"] or outcomes["minus"]["decision_changed"]
                every_order_has_flip = every_order_has_flip and has_flip
                cells.append(
                    {
                        "preserve_first": bool(item["job"]["preserve_first"]),
                        "baseline_semantic_choice": item["baseline_semantic"],
                        "baseline_argmax_token_id": item["baseline_token"],
                        "direction_sha256": item["direction_sha256"],
                        "gradient_diagnostics": item["gradient_diagnostics"],
                        "outcomes": outcomes,
                    }
                )
            trial = {
                "strength": float(strength),
                "all_targets_met": all_targets,
                "real_flip_in_each_order": every_order_has_flip,
                "passes": all_targets and every_order_has_flip,
                "cells": cells,
            }
            trials.append(trial)
            if trial["passes"]:
                selected = trial
                break
        checkpoint["pairs"].append(
            {
                "case_id": key[0],
                "assignment": key[1],
                "passes": selected is not None,
                "selected_strength": None if selected is None else selected["strength"],
                "selected_cells": None if selected is None else selected["cells"],
                "trials": trials,
            }
        )
        checkpoint["pairs"].sort(key=lambda row: (row["case_id"], int(row["assignment"])))
        _atomic_json(STEERING_CHECKPOINT_PATH, checkpoint)
        print(json.dumps({"completed": key, "passes": selected is not None, "strength": None if selected is None else selected["strength"]}), flush=True)

    successful = sum(bool(row["passes"]) for row in checkpoint["pairs"])
    selected_kls = [
        outcome["full_vocabulary_kl_changed_to_baseline"]
        for row in checkpoint["pairs"]
        if row["selected_cells"] is not None
        for cell in row["selected_cells"]
        for outcome in cell["outcomes"].values()
    ]
    result = {
        "schema_version": "sp_lense.learned_context_gate_steering_development_result.v1",
        "status": "passed" if successful == len(grouped) else "failed",
        "development_only": True,
        "config_sha256": _sha256(CONFIG_PATH),
        "gate_result_sha256": _sha256(GATE_RESULT_PATH),
        "active_pair_count": len(grouped),
        "successful_pair_count": successful,
        "success_rate": successful / len(grouped),
        "target_kl_report_only": {
            "row_count": len(selected_kls),
            "mean": statistics.fmean(selected_kls) if selected_kls else None,
            "p95": _percentile(selected_kls, 0.95) if selected_kls else None,
            "maximum": max(selected_kls) if selected_kls else None,
        },
        "off_gate_intervention_strength": 0.0,
        "compute": {
            **checkpoint["compute"],
            "generated_tokens": 0,
            "elapsed_seconds_this_run": time.perf_counter() - started,
            "external_cost_usd": 0,
        },
        "pairs": checkpoint["pairs"],
    }
    result["result_sha256"] = _canonical_sha256(result)
    _atomic_json(STEERING_RESULT_PATH, result)
    print(json.dumps({key: result[key] for key in (
        "status", "active_pair_count", "successful_pair_count", "success_rate", "target_kl_report_only", "compute"
    )}, indent=2))
    return result


def run_report() -> str:
    gate_result = json.loads(GATE_RESULT_PATH.read_text(encoding="utf-8")) if GATE_RESULT_PATH.exists() else None
    steering = json.loads(STEERING_RESULT_PATH.read_text(encoding="utf-8")) if STEERING_RESULT_PATH.exists() else None
    lines = [
        "# Learned context-gated prompt-gradient development",
        "",
        "This is a post-hoc development result on previously opened data. It is not confirmatory evidence.",
        "",
        "## Learned context gate",
        "",
    ]
    if gate_result is None:
        lines.append("Not run.")
    else:
        lines.append(f"Status: **{gate_result['status']}**.")
        lines.append("")
        lines.append(
            f"Discovery leave-one-case-out balanced accuracy: {gate_result['crossfit_metrics']['balanced_accuracy']:.3f}. Validation balanced accuracy: {gate_result['validation_metrics']['balanced_accuracy']:.3f}."
        )
        lines.append("")
        lines.append(f"Pair counts: `{json.dumps(gate_result['validation_pair_counts'], sort_keys=True)}`")
    lines += ["", "## Dynamic steering", ""]
    if steering is None:
        lines.append("Not run; the gate must pass first.")
    else:
        lines.append(
            f"Status: **{steering['status']}**. Successful active pairs: {steering['successful_pair_count']}/{steering['active_pair_count']}."
        )
    lines += [
        "",
        "## Claim boundary",
        "",
        "This method is a learned conditional controller plus a transductive prompt-local output gradient. Off-gate stability is a controller property. The exact nuisance-null branch failed, so this result cannot be called an intrinsically self-specific vector, a natural mechanism, open-ended behavior, or publication-ready novelty.",
        "",
    ]
    text = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)
    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Develop learned context-gated prompt gradients.")
    parser.add_argument("command", choices=("capture", "gate", "steer", "report"))
    args = parser.parse_args(argv)
    {"capture": run_gate_capture, "gate": run_gate_fit, "steer": run_steering, "report": run_report}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
