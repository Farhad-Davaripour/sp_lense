"""Model-free source and deployment checks used locally and in CI."""
from pathlib import Path
import argparse
import hashlib
import json
import yaml
from src.config import ROOT, load_datasets
from src.utils import canonical, sha256


def source_digest(root=ROOT):
    """Hash deployed ordinary files; notebook objects are checked by bundle validation."""
    root = Path(root)
    paths = sorted(p for folder in ("src", "config", "ddl")
                   for p in (root / folder).rglob("*")
                   if p.is_file() and p.suffix in (".py", ".yaml", ".json", ".sql"))
    if not paths:
        raise ValueError("No deployed source found")
    pins = {p.relative_to(root).as_posix(): sha256(p) for p in paths}
    return hashlib.sha256(canonical(pins).encode()).hexdigest()


def verify():
    recipe = yaml.safe_load((ROOT / "config/research.yaml").read_text(encoding="utf-8"))
    pins = recipe["sources"] + recipe["configuration_files"]
    for item in pins:
        if sha256(ROOT / item["source"]) != item["sha256"]:
            raise ValueError("Frozen scientific source changed: " + item["source"])
    inference = yaml.safe_load((ROOT / "config/research2_inference.yaml").read_text(encoding="utf-8"))
    if sha256(ROOT / "src/engines/research2/inference_math.py") != inference["inference_math_sha256"]:
        raise ValueError("Original Research 2 inference math changed")
    datasets = load_datasets()
    notebooks = 0
    for path in (ROOT / "resources").glob("*.job.yml"):
        jobs = yaml.safe_load(path.read_text(encoding="utf-8"))["resources"]["jobs"]
        for job in jobs.values():
            if job.get("schedule") or job.get("trigger"):
                raise ValueError("Research jobs must remain manual")
            for task in job["tasks"]:
                relative = task["notebook_task"]["notebook_path"]
                if not relative.startswith("../notebooks/") or not (path.parent / relative).is_file():
                    raise ValueError("Missing deployed notebook: " + relative)
                if task.get("max_retries") != 0 or not task.get("disable_auto_optimization"):
                    raise ValueError("Scientific and smoke jobs must not retry automatically")
                notebooks += 1
    return {"status": "PUBLIC_SOURCE_VERIFIED", "frozen_files": len(pins),
            "datasets": len(datasets["datasets"]), "notebook_tasks": notebooks,
            "source_digest": source_digest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digest", action="store_true")
    args = parser.parse_args()
    result = verify()
    print(result["source_digest"] if args.digest else json.dumps(result))


if __name__ == "__main__":
    main()
