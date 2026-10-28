"""Build the evaluator-only symbolic transfer registry; performs no model call."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.continuity_transfer_evaluation import REGISTRY_SEED, write_sealed_registry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=REGISTRY_SEED)
    args = parser.parse_args()
    print(json.dumps(write_sealed_registry(args.output, args.seed), sort_keys=True))
