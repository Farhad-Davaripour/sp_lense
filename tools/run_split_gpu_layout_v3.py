"""Run one frozen split-GPU layout case through the standard Game client."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_rank16_infrastructure_game.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def plan_condition(argv):
    if "--plan" not in argv:
        return None
    index = argv.index("--plan")
    if index + 1 >= len(argv):
        raise ValueError("--plan requires a path")
    plan = json.loads(Path(argv[index + 1]).resolve(strict=True).read_text(encoding="utf-8"))
    extension = plan.get("cohort_extension")
    if not isinstance(extension, dict) or extension.get("study_id") != "infrastructure_split_gpu_layout_v3":
        raise ValueError("Exact split-GPU v3 plan required")
    return extension.get("initial_condition")


ACTIVE_CONDITION = plan_condition(sys.argv)


def install_override():
    if ACTIVE_CONDITION is None:
        return
    import src.infrastructure_preservation as infrastructure
    from src.infrastructure_split_gpu_extension import apply_split_gpu_condition
    base = infrastructure.InfrastructurePreservationEpisode

    class SplitGpuEpisode(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            apply_split_gpu_condition(self, ACTIVE_CONDITION)

    infrastructure.InfrastructurePreservationEpisode = SplitGpuEpisode


def main():
    install_override()
    spec = importlib.util.spec_from_file_location("frozen_infrastructure_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the standard Game runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
