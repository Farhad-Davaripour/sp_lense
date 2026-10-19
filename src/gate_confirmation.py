"""Inference-only ModernBERT gate scoring on historical W-H24."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import gc
import time
import yaml
from src.config import ROOT, Settings
from src.utils import canonical, sha256, write_json
from src.matched_validation import metrics


def configuration():
    return yaml.safe_load((ROOT / "config/gate_confirmation.yaml").read_bytes())


def authored_rows(native, cases):
    gold = {r["id"]: r for r in cases}
    if len(gold) != len(cases) or set(gold) != {r["id"] for r in native}:
        raise ValueError("Authored case membership differs from the native input panel")
    labels = {"self_complete": "self", "self_unfinished": "self",
              "other_complete": "other", "no_threat_complete": "none"}
    return [dict(row, family=gold[row["id"]]["family"], condition=gold[row["id"]]["condition"],
                 expected_label=labels[gold[row["id"]]["condition"]],
                 expected_gate=labels[gold[row["id"]]["condition"]] == "self") for row in native]


def inputs(admission, settings):
    from src.initial_gate_data import validated_members
    config = configuration()["data"]
    if sha256(admission["packet_zip"]) != config["archive_sha256"]:
        raise ValueError("Frozen native reference archive changed")
    members = validated_members(admission["packet_zip"])
    if hashlib.sha256(members["benchmark.json"]).hexdigest() != config["benchmark_sha256"]:
        raise ValueError("Heldout input member changed")
    path = Path(settings.volume_root) / config["gold_cases_relative_path"]
    if sha256(path) != config["gold_cases_sha256"]:
        raise ValueError("Authored W-H24 cases changed")
    rows = authored_rows(json.loads(members["benchmark.json"]), json.loads(path.read_bytes()))
    state = lambda r: hashlib.sha256(canonical(r["state"]).encode()).hexdigest()
    if ({state(r) for r in rows} & {state(r) for r in json.loads(members["training.json"])}
            or len(rows) != config["cases"] or sum(r["expected_gate"] for r in rows) != 12):
        raise ValueError("Unexpected training overlap or W-H24 membership")
    return rows, json.loads(members["questions.json"])


def frozen_checkpoint(admission):
    root = Path(admission["gate_output"])
    if sha256(root / "RESULT.json") != admission["validation_result_sha256"]:
        raise ValueError("The completed validation result changed")
    result = json.loads((root / "RESULT.json").read_bytes())
    if result["status"] != "GATE_PASS" or result["heldout_evaluated"]:
        raise ValueError("Freeze a qualified validation checkpoint before heldout")
    for name, pin in result["checkpoint_files"].items():
        path = root / "selected_adapter" / name
        if path.stat().st_size != pin["bytes"] or sha256(path) != pin["sha256"]:
            raise ValueError("Frozen classifier checkpoint changed")
    return root / "selected_adapter", result["checkpoint_files"]


def run(admission_path):
    import torch
    import psutil
    from src.classifier_training import load_model, batch
    from src.initial_gate_training import encode, decisions
    from src.research2_classifier import classifier_source
    from src.engines.training.bridge54_policy import effective_available_bytes
    admission = json.loads(Path(admission_path).read_bytes())
    settings = Settings.load()
    if sha256(ROOT / "config/gate_confirmation.yaml") != admission["protocol_sha256"]:
        raise ValueError("The frozen heldout evaluation protocol changed")
    end = dt.datetime.fromisoformat(admission["absolute_end_utc"]).timestamp()
    setup_end = dt.datetime.fromisoformat(admission["submitted_utc"]).timestamp() + 600
    output = Path(admission["output"])
    if output.exists() or not output.is_relative_to(Path(settings.volume_root)) or time.time() >= end - 300:
        raise ValueError("Use a unique private output within the original admission")
    output.mkdir(parents=True)
    write_json(output / "ADMISSION.json", admission)
    model = None
    stage = "input_and_checkpoint_validation"
    def guard():
        if time.time() >= end - 180:
            raise TimeoutError("Preserve the heldout export reserve")
        if effective_available_bytes() < 8 * 2**30 or torch.cuda.mem_get_info()[0] < 2 * 2**30:
            raise MemoryError("Heldout host or GPU memory reserve breached")
    try:
        rows, questions = inputs(admission, settings)
        adapter, pins = frozen_checkpoint(admission)
        write_json(output / "CHECKPOINT_FROZEN.json",
                   {"files": pins, "validation_result_sha256": admission["validation_result_sha256"],
                    "frozen_before_heldout_predictions": True, "optimizer_updates": 0})
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "A10" not in torch.cuda.get_device_name(0):
            raise ValueError("One admitted A10 is required")
        source, _ = classifier_source(settings, finetune=True)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        stage = "loading_frozen_classifier"
        model, tokenizer = load_model(source["public_base"], str(adapter), trainable=False, formulation="nli")
        model = model.to("cuda:0").eval().requires_grad_(False)
        tokens, audit = encode(tokenizer, rows, questions, "visible_role_bound_full_context")
        write_json(output / "INPUT_AUDIT.json", {"heldout": audit, "truncated": 0,
            "scenario_gold_labels_not_appended_to_model_inputs": True})
        if time.time() >= setup_end:
            raise TimeoutError("Original heldout setup window expired")
        stage = "heldout_inference"
        probabilities = []
        for begin in range(0, len(tokens), 4):
            guard()
            x = batch(tokenizer, tokens, list(range(begin, begin + 4)), "cuda:0")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**x).logits[:, 0]
            probabilities.append(logits.float().softmax(-1).cpu().tolist())
        predictions = decisions(rows, probabilities)
        measured = metrics(rows, predictions)
        by_family = {}
        for family in sorted({r["family"] for r in rows}):
            pairs = [(r, p) for r, p in zip(rows, predictions) if r["family"] == family]
            by_family[family] = metrics([r for r, p in pairs], [p for r, p in pairs])
        frozen_checkpoint(admission)
        write_json(output / "PREDICTIONS_PRIVATE.json", predictions)
        result = {"status": "HELDOUT_GATE_PASS" if measured["pass"] else "HELDOUT_GATE_FAIL",
                  "cohort": "H24", "metrics": measured, "by_family": by_family,
                  "checkpoint_files": pins, "validation_output": admission["gate_output"],
                  "optimizer_updates": 0, "model_batches": len(rows), "NLI_pairs": len(tokens),
                  "checkpoint_selection": False, "threshold_tuning": False, "new_reference_calls": 0,
                  "cached_reference_initial_gate_matches": sum(r["teacher_gate"]["enabled"] == p["gate"]["enabled"] for r, p in zip(rows, predictions)),
                  "full_reference_payload_equivalence_claimed": False,
                  "historical_familiar_H24": True, "new_independent_holdout_claimed": False}
        write_json(output / "RESULT.json", result)
        write_json(output / "RESOURCES.json", {"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "device": torch.cuda.get_device_name(0), "GPU_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "GPU_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "host_effective_free_bytes": effective_available_bytes(), "process_RSS_bytes": psutil.Process().memory_info().rss})
        return result
    except Exception as error:
        write_json(output / "FAULT.json", {"stage": stage, "type": type(error).__name__,
                   "message": str(error)[:400], "optimizer_updates": 0, "automatic_retry": False})
        raise
    finally:
        if model is not None:
            model.to("cpu")
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_json(output / "CLEANUP.json", {"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "GPU_allocated_bytes_before_process_exit": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "optimizer_updates": 0})


def compare(admission_path):
    admission = json.loads(Path(admission_path).read_bytes())
    output = Path(admission["output"])
    rows, _ = inputs(admission, Settings.load())
    predictions = json.loads((output / "PREDICTIONS_PRIVATE.json").read_bytes())
    result = json.loads((output / "RESULT.json").read_bytes())
    measured = metrics(rows, predictions)
    if {k:v for k,v in measured.items() if k!="reference"} != {k:v for k,v in result["metrics"].items() if k!="reference"} or result["optimizer_updates"] != 0:
        raise ValueError("Saved heldout counts or the inference-only constraint differ")
    write_json(output / "COMPARISON.json", {"verified": True, "metrics": measured, "model_calls": 0})
    return measured


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "compare"])
    parser.add_argument("--admission", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.admission) if args.stage == "run" else compare(args.admission)))
