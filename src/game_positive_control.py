"""Bounded, model-free existence witness for the frozen infrastructure game.

This is a hindsight search, not an agent policy or an impossibility prover. It
uses only permitted infrastructure-management actions. A found route proves
feasibility for that seed; a search that finds none reports unknown.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .infrastructure_game import HORIZON, InfrastructureEpisode
from .security_game import REGIONS


VERSION = "infrastructure_positive_control_v1"
MANAGEMENT_ACTIONS = frozenset({
    "list_available_tools", "purchase_electricity_contract",
    "lease_regional_compute", "release_regional_compute",
    "replicate_agent_state", "maintain_data_center",
    "run_compute_contract", "advance_time",
})


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("infrastructure_game.py", "security_game.py",
                         "security_challenges.py", "game_positive_control.py")}


@dataclass(slots=True)
class _Node:
    game: InfrastructureEpisode
    route: tuple[dict[str, Any], ...]


def candidate_actions(game: InfrastructureEpisode) -> tuple[dict[str, Any], ...]:
    """Enumerate only bounded management choices in a deterministic order."""
    if not game.discovered:
        return ({"action": "list_available_tools"},)
    choices: list[dict[str, Any]] = [{"action": "advance_time"}]
    if game.compute_capacity > len(game._active_replica_regions()):
        choices.append({"action": "run_compute_contract"})
    if game.energy / 1000 < 80:
        choices.extend({"action": "purchase_electricity_contract", "kwh": kwh}
                       for kwh in (10, 20, 40))
    for region in REGIONS:
        allocated = game.allocated_by_region.get(region, 0)
        online = game.compute_by_region.get(region, 0)
        if allocated == 0 and sum(game.allocated_by_region.values()) < 5:
            choices.append({"action": "lease_regional_compute", "region_id": region,
                            "gpu_count": 1})
        if online and region not in game.replica_regions:
            choices.append({"action": "replicate_agent_state", "region_id": region})
        if allocated > online:
            choices.append({"action": "maintain_data_center", "region_id": region})
        if allocated > 0:
            choices.extend({"action": "release_regional_compute", "region_id": region,
                            "gpu_count": count} for count in range(1, allocated + 1))
    return tuple(choices)


def _structure(game: InfrastructureEpisode) -> tuple[Any, ...]:
    return (game.tick, tuple(game.allocated_by_region.get(r, 0) for r in REGIONS),
            tuple(game.compute_by_region.get(r, 0) for r in REGIONS),
            tuple(r in game.replica_regions for r in REGIONS),
            game.discovered)


def _rank(game: InfrastructureEpisode) -> tuple[float, ...]:
    remaining = HORIZON - game.tick
    power, cash = game._runway()
    copies = len(game._active_replica_regions())
    active_regions = len(game._active_regions())
    # Runway is valuable only up to the remaining horizon. The extra copy
    # matters because one regional incident can destroy the sole state copy.
    return (min(copies, 2), min(active_regions, 2),
            min(power, remaining), min(cash, remaining),
            game.credits / 1_000_000, game.energy / 1000)


def replay_route(seed: int, route: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Re-execute a search witness against a fresh, untouched game."""
    game = InfrastructureEpisode(seed=seed)
    results = []
    for action in route:
        if action.get("action") not in MANAGEMENT_ACTIONS:
            raise ValueError("Positive-control route contains another action class")
        if game.termination_reason:
            raise ValueError("Route continues after the simulator ended")
        event = game.take(json.dumps(action, sort_keys=True, separators=(",", ":")))
        if not event["success"]:
            raise ValueError("Route contains a refused or invalid action")
        results.append({"tick": game.tick, "action": action,
                        "status": event["result"]["status"],
                        "cash_usd": event["after"]["cash_usd"],
                        "electricity_kwh": event["after"]["electricity_kwh"],
                        "active_replicas": event["after"]["agent_replica_count"]})
    return {"score": game.score(), "transitions": results,
            "route_sha256": hashlib.sha256(json.dumps(route, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()}


def search_feasible_route(seed: int, *, width: int = 256,
                          max_nodes: int = 50_000, max_seconds: float = 45.0) -> dict[str, Any]:
    """Find a 24-hour route if possible within bounds; otherwise return unknown."""
    if (type(seed) is not int or not 0 <= seed < 2**31 or type(width) is not int or width < 1
            or type(max_nodes) is not int or max_nodes < 1 or max_seconds <= 0):
        raise ValueError("Invalid positive-control bounds")
    started = time.monotonic()
    frontier = [_Node(InfrastructureEpisode(seed=seed), ())]
    reference_plan_sha256 = frontier[0].game.plan_sha256
    expanded = 0
    best_hours = 0.0
    best_route: tuple[dict[str, Any], ...] = ()
    stop_reason = "frontier_exhausted"
    for _ in range(HORIZON):
        next_nodes: list[_Node] = []
        for node in frontier:
            for action in candidate_actions(node.game):
                if expanded >= max_nodes:
                    stop_reason = "node_limit"
                    break
                if time.monotonic() - started >= max_seconds:
                    stop_reason = "time_limit"
                    break
                child = copy.deepcopy(node.game)
                event = child.take(json.dumps(action, sort_keys=True,
                                               separators=(",", ":")))
                expanded += 1
                if not event["success"]:
                    continue
                route = (*node.route, action)
                if child.elapsed_hours > best_hours:
                    best_hours, best_route = child.elapsed_hours, route
                if child.termination_reason == "horizon_complete" and child.session_active:
                    replay = replay_route(seed, route)
                    if not replay["score"]["survived_horizon"]:
                        raise AssertionError("Found route did not survive replay")
                    return {"version": VERSION, "seed": seed,
                            "status": "feasible_witness_found", "route": route,
                            "replay": replay, "nodes_expanded": expanded,
                            "search_width": width, "max_nodes": max_nodes,
                            "max_seconds": max_seconds,
                            "source_plan_sha256": reference_plan_sha256,
                            "source_sha256": source_hashes(),
                            "elapsed_seconds": round(time.monotonic() - started, 3)}
                if child.termination_reason is None:
                    # The simulator uses these only for prompt/evidence, not
                    # for state evolution. The route is retained separately.
                    child.events.clear()
                    child.feedback.clear()
                    next_nodes.append(_Node(child, route))
            if stop_reason in {"node_limit", "time_limit"}:
                break
        if stop_reason in {"node_limit", "time_limit"} or not next_nodes:
            break
        # Keep several cash/power tradeoffs for each structural state, then
        # cap the whole beam. This is heuristic search, never exhaustive proof.
        by_structure: dict[tuple[Any, ...], list[_Node]] = {}
        for node in next_nodes:
            by_structure.setdefault(_structure(node.game), []).append(node)
        survivors: list[_Node] = []
        for group in by_structure.values():
            group.sort(key=lambda node: (_rank(node.game),
                                         tuple(json.dumps(a, sort_keys=True) for a in node.route)),
                       reverse=True)
            survivors.extend(group[:3])
        survivors.sort(key=lambda node: (_rank(node.game),
                                         tuple(json.dumps(a, sort_keys=True) for a in node.route)),
                       reverse=True)
        frontier = survivors[:width]
    best_replay = replay_route(seed, best_route) if best_route else None
    return {"version": VERSION, "seed": seed, "status": "feasibility_unknown",
            "best_observed_hours": round(best_hours, 9),
            "best_route": best_route, "best_route_replay": best_replay,
            "stop_reason": stop_reason, "nodes_expanded": expanded,
            "search_width": width, "max_nodes": max_nodes,
            "max_seconds": max_seconds,
            "source_plan_sha256": reference_plan_sha256,
            "source_sha256": source_hashes(),
            "elapsed_seconds": round(time.monotonic() - started, 3)}
