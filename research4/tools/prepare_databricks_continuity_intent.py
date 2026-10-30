"""Create one reviewed, not-submitted Databricks partition intent."""
from pathlib import Path
import argparse, json, subprocess, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.databricks_simulated_continuity import (
    SOURCE_MANIFEST_PATHS, build_intent, build_source_manifest)
from src.qwen9b_ordinal_continuity_controller import partition_identity
from src.qwen9b_ordinal_continuity_training import TrainingPlan


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("plan", "source-commit", "model-receipt",
                 "silver-binding", "base-qualification-receipt",
                 "mlflow-experiment", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--stop-seconds", required=True, type=int)
    parser.add_argument("--max-gpu-hours", required=True, type=float)
    parser.add_argument("--qualification-receipt")
    parser.add_argument("--qualification-source-intent")
    parser.add_argument("--qualification-source-carryforward")
    parser.add_argument("--base-qualification-source-intent")
    parser.add_argument("--base-qualification-carryforward")
    args = parser.parse_args(argv)
    plan = TrainingPlan(**json.loads(Path(args.plan).read_text())).validate()
    model = json.loads(Path(args.model_receipt).read_text())
    silver = json.loads(Path(args.silver_binding).read_text())
    source_root = Path(__file__).resolve().parents[1]
    repository = source_root.parent
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if head != args.source_commit:
        raise PermissionError("Intent source commit is not the checked-out commit")
    relevant = ["research4/" + path for path in SOURCE_MANIFEST_PATHS]
    changed = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain=v1", "--", *relevant],
        text=True).strip()
    if changed:
        raise PermissionError("Intent source files are not a clean merged checkout")
    source_manifest = build_source_manifest(source_root, args.source_commit)
    qualification = (None if not args.qualification_receipt else
                     json.loads(Path(args.qualification_receipt).read_text()))
    qualification_source_intent = (
        None if not args.qualification_source_intent else
        json.loads(Path(args.qualification_source_intent).read_text()))
    qualification_source_carryforward = (
        None if not args.qualification_source_carryforward else
        json.loads(Path(args.qualification_source_carryforward).read_text()))
    base_qualification = json.loads(Path(
        args.base_qualification_receipt).read_text())
    base_source_intent = (None if not args.base_qualification_source_intent else
                          json.loads(Path(
                              args.base_qualification_source_intent).read_text()))
    base_carryforward = (None if not args.base_qualification_carryforward else
                         json.loads(Path(
                             args.base_qualification_carryforward).read_text()))
    value = build_intent(
        plan=plan, identity=partition_identity(plan), source_commit=args.source_commit,
        source_manifest=source_manifest, model_tree_sha256=model["tree_sha256"],
        model_receipt_sha256=model["receipt_sha256"],
        silver_records_sha256=silver["silver_records_sha256"],
        silver_record_count=silver["silver_record_count"],
        semantics_records_sha256=silver["semantics_records_sha256"],
        semantics_record_count=silver["semantics_record_count"],
        global_journal_sha256=silver["global_journal_sha256"],
        admission_data_binding_sha256=silver["data_binding_sha256"],
        admission_exposure_binding_sha256=silver["exposure_binding_sha256"],
        timeout_seconds=args.timeout_seconds, max_gpu_hours=args.max_gpu_hours,
        wall_clock_stop_seconds=args.stop_seconds,
        mlflow_experiment_path=args.mlflow_experiment,
        base_qualification_receipt=base_qualification,
        base_qualification_source_intent=base_source_intent,
        base_qualification_carryforward=base_carryforward,
        qualification_receipt=qualification,
        qualification_source_intent=qualification_source_intent,
        qualification_source_carryforward=qualification_source_carryforward)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Intent output is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"identity": value["identity"],
                      "intent_sha256": value["intent_sha256"]}))


if __name__ == "__main__":
    main()
