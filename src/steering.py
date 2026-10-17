"""Reproduce fixed or learned steering with the preserved scientific workers."""
import argparse
import json
from src.utils import run_recipe

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["fixed", "fixed-analysis", "learned", "learned-evaluation"], default="fixed-analysis")
    parser.add_argument("--engine-args", default="[]")
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    recipe = {"fixed": "fixed_steering", "fixed-analysis": "fixed_steering_analysis",
              "learned": "learned_steering", "learned-evaluation": "learned_steering_evaluation"}[args.method]
    engine_args = json.loads(args.engine_args)
    if not isinstance(engine_args, list) or not all(isinstance(x, str) for x in engine_args):
        raise ValueError("engine-args must be a JSON array of strings")
    result = run_recipe(recipe, engine_args, args.execute, args.max_seconds)
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    main()
