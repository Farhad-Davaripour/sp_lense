"""Reproduce the pinned matched fine-tuning recipe using its original worker."""
import argparse
import json
from src.utils import run_recipe

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--engine-args", default="[]", help="JSON array of original worker arguments")
    parser.add_argument("--resume-2b-admission", help="Current bounded 54-update job configuration")
    args = parser.parse_args(argv)
    if args.resume_2b_admission:
        if not args.execute:
            return {"stage": "resume-2B", "new_updates": 54, "model_calls": 0}
        from src.resume2b import run
        return run(args.resume_2b_admission)
    engine_args = json.loads(args.engine_args)
    if not isinstance(engine_args, list) or not all(isinstance(x, str) for x in engine_args):
        raise ValueError("engine-args must be a JSON array of strings")
    if "--resume" in engine_args:
        raise ValueError("Use the separately admitted resume-2B worker; the fresh-fit reference performs scratch calibration")
    result = run_recipe("finetune", engine_args, args.execute, args.max_seconds)
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    main()
