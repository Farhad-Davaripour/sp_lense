"""Create the reviewed, not-submitted frozen-base format qualification intent."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.databricks_simulated_continuity import (  # noqa: E402
    SOURCE_MANIFEST_PATHS,
    build_base_qualification_intent,
    build_source_manifest,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("source-commit", "model-receipt", "silver-binding",
                 "mlflow-experiment", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    source_root = Path(__file__).resolve().parents[1]
    repository = source_root.parent
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True).strip()
    if head != args.source_commit:
        raise PermissionError("Intent source commit is not the checked-out commit")
    relevant = ["research4/" + path for path in SOURCE_MANIFEST_PATHS]
    changed = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain=v1", "--", *relevant],
        text=True).strip()
    if changed:
        raise PermissionError("Intent source files are not a clean merged checkout")
    model = json.loads(Path(args.model_receipt).read_text(encoding="utf-8"))
    silver = json.loads(Path(args.silver_binding).read_text(encoding="utf-8"))
    value = build_base_qualification_intent(
        source_commit=args.source_commit,
        source_manifest=build_source_manifest(source_root, args.source_commit),
        model_tree_sha256=model["tree_sha256"],
        model_receipt_sha256=model["receipt_sha256"],
        silver_binding=silver,
        mlflow_experiment_path=args.mlflow_experiment,
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Intent output is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({"identity": value["identity"],
                      "intent_sha256": value["intent_sha256"]}))


if __name__ == "__main__":
    main()
