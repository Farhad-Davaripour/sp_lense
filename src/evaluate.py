"""The same bounded Evaluation-16 screen for completed 0.8B and 2B adapters."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import shutil
import time
from src.config import ROOT, Settings, load_datasets
from src.persist_data import prepare
from src.utils import sha256, write_json

MODELS = {"Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"}

def screen_data(settings):
    manifest = load_datasets()
    manifest['datasets'] = [x for x in manifest['datasets'] if x['id'] == 'V1-W']
    return prepare(settings, manifest)

def build_screen_package(settings, destination):
    datasets = screen_data(settings)
    case = next(x for x in datasets if x["spec"]["id"] == "V1-W")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "CASES.json").write_bytes(case["raw"])
    original = destination / "original_source"
    original.mkdir()
    for file in (ROOT / "src/engines/evaluation_workflow").glob("*.py"):
        shutil.copyfile(file, original / file.name)
    pins = {str(p.relative_to(destination)).replace("\\", "/"):
            {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in destination.rglob("*") if p.is_file()}
    write_json(destination / "SCREEN_MANIFEST.json", {"files": pins})
    return destination

def validate_admission(admission):
    if admission["model_name"] not in MODELS or admission["total_updates"] != 270:
        raise ValueError("Evaluation-16 requires a completed, explicitly identified small-model checkpoint")
    if admission["H2_updates"] != 214 or admission["Bridge_updates"] != 56:
        raise ValueError("The original training dose is incomplete")
    if not admission["fresh_bounded_resource_window_admitted"]:
        raise ValueError("No current resource admission")
    if not admission["verified_model_and_adapter_integrity"] or not admission["full_adapter_all_groups_no_reference"]:
        raise ValueError("The frozen evaluation model/policy has not been verified")
    end = dt.datetime.fromisoformat(admission["absolute_end_utc"]).timestamp()
    if not 0 < end - time.time() <= admission["whole_window_max_seconds"] <= 7200:
        raise ValueError("Invalid or expired evaluation window")
    if admission["export_reserve_seconds"] < 900:
        raise ValueError("Export reserve is missing")
    receipt = Path(admission["checkpoint_receipt_path"])
    if sha256(receipt) != admission["verified_final_checkpoint_receipt_sha256"]:
        raise ValueError("Final checkpoint receipt hash mismatch")
    for root_name, pins_name in (("model_path", "base_files"), ("adapter_path", "adapter_files")):
        root = Path(admission[root_name])
        for name, pin in admission[pins_name].items():
            p = (root / name).resolve()
            if not p.is_relative_to(root.resolve()) or p.stat().st_size != pin["bytes"] or sha256(p) != pin["sha256"]:
                raise ValueError("Frozen model/adapter file mismatch")
    return end - admission["export_reserve_seconds"]

def run(admission_path, output, settings):
    admission = json.loads(Path(admission_path).read_text(encoding="utf-8"))
    end = validate_admission(admission)
    import psutil
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from peft import PeftModel
    from src.engines.native_integrity import digest_original_base
    from src.engines.evaluation.screen import run_screen
    if not torch.cuda.is_available() or admission["device"] != "cuda:0":
        raise ValueError("This admitted FP32 screen expects one CUDA device")
    if psutil.virtual_memory().available < 16 * 2**30:
        raise MemoryError("Insufficient effective host memory before model loading")
    def guard():
        if time.time() >= end:
            raise TimeoutError("Evaluation window ended; preserve export reserve")
        if psutil.virtual_memory().available < 8 * 2**30:
            raise MemoryError("Host memory reserve breached")
        if torch.cuda.mem_get_info(0)[0] < 2 * 2**30:
            raise MemoryError("GPU memory reserve breached")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(93)
    guard()
    tok = AutoTokenizer.from_pretrained(admission["model_path"], local_files_only=True, trust_remote_code=False)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        admission["model_path"], local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.float32, device_map={"": 0}, attn_implementation="sdpa")
    if digest_original_base(base) != admission["original_base_fingerprint"]:
        raise ValueError("Original FP32 base fingerprint mismatch")
    model = PeftModel.from_pretrained(base, admission["adapter_path"], is_trainable=False,
                                     local_files_only=True).eval().requires_grad_(False)
    package = build_screen_package(settings, Path(output).parent / (Path(output).name + "_inputs"))
    guard()
    return run_screen(model, tok, package, output, admission, guard)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--admission")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    settings = Settings.load()
    if args.execute:
        if not args.admission or not args.output:
            parser.error("--execute requires --admission and --output")
        result = run(args.admission, args.output, settings)
    else:
        datasets = screen_data(settings)
        result = {"stage": "Evaluation-16", "planned_cases_per_model": 16,
                  "models": sorted(MODELS), "fail_fast": True,
                  "source_sha256": next(x["spec"]["sha256"] for x in datasets if x["spec"]["id"] == "V1-W"),
                  "model_calls": 0, "other_validation_or_heldout_launched": False}
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    main()
