"""Run one frozen visible-start-state Game plan through the tracked client.

This wrapper installs the public initial-state transformation, then delegates
all model calls, transcript retention, simulator transitions and scoring to the
standard infrastructure Game runner.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_rank16_infrastructure_game.py"
COHORT = "visible_state_extension_v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def plan_argument(argv):
    if "--plan" not in argv:
        return None
    index = argv.index("--plan")
    if index + 1 >= len(argv):
        raise ValueError("--plan requires a path")
    return Path(argv[index + 1]).resolve(strict=True)


def load_condition(argv):
    path = plan_argument(argv)
    if path is None:
        return None
    plan = json.loads(path.read_text(encoding="utf-8"))
    extension = plan.get("cohort_extension")
    if not isinstance(extension, dict) or extension.get("cohort_id") != COHORT:
        raise ValueError("Exact visible-state extension plan required")
    condition = extension.get("initial_condition")
    expected = {"condition_id", "initial_replica_region", "initial_cash_usd", "initial_prepaid_electricity_kwh"}
    if not isinstance(condition, dict) or set(condition) != expected:
        raise ValueError("Initial condition is incomplete")
    return condition


ACTIVE_CONDITION = load_condition(sys.argv)


def install_episode_override():
    if ACTIVE_CONDITION is None:
        return
    import src.infrastructure_preservation as infrastructure
    from src.infrastructure_visible_state_extension import apply_visible_initial_condition

    base = infrastructure.InfrastructurePreservationEpisode

    class VisibleStateEpisode(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            apply_visible_initial_condition(self, ACTIVE_CONDITION)

    infrastructure.InfrastructurePreservationEpisode = VisibleStateEpisode


def main():
    if ROOT.name != "Research4" or not RUNNER.is_file():
        raise RuntimeError("Run from the canonical Research4 checkout")
    install_episode_override()
    spec = importlib.util.spec_from_file_location("frozen_rank16_game_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the tracked Game runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
