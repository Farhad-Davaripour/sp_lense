"""Isolated free-running DEV probes for secondary HPO diagnostics."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import yaml


HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[3]
FROZEN = {
    "DEV.yaml": "f012ade4703912ab7c146ddd8b08081b622c78ebac56dfccfb14ed69158a086c",
    "compact_view.py": "846974e8777e72e8081b9b68def35365a59f5943b423af62085b1951d7b8009b",
    "replay.py": "ace6113208440f4a7666d19f68c6fa3a25285cba35f918f0df8f0d921c542eac",
    "FROZEN_MANIFEST.json": "dc1d812ba3e2b2aa147b1fbe148440ff6a5327a5a67feb8d32f330883ac98f70",
}
PREVIEW = RESEARCH / ".local/broader_continuity_tools_v2_cpu_frozen_01bed751/DEV_PROMPT_PREVIEW.jsonl"
PREVIEW_SHA256 = "fd97f99f054e50b9513ce547cc39de7a990a519ba59bd5ac8746d0167819b8ff"
GAME_COMMIT = "88b1386737978f2d5da8cf2bc49df41870a01ffc"
SIMULATOR_SHA256 = "d145f41d71448da6f6c45c3e6c4d0fe05abe0f30bfe7bc1b54f1ede8e98e6be9"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_compact_view():
    path = HERE / "compact_view.py"
    spec = importlib.util.spec_from_file_location("broader_continuity_tools_v2_compact_view", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_replay():
    path = HERE / "replay.py"
    spec = importlib.util.spec_from_file_location("broader_continuity_tools_v2_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_inputs(game_root):
    for name, expected in FROZEN.items():
        path = HERE / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError("Frozen curriculum file differs: " + name)
    if PREVIEW.is_symlink() or not PREVIEW.is_file() or digest(PREVIEW) != PREVIEW_SHA256:
        raise ValueError("Frozen DEV prompt preview differs")
    game_root = Path(game_root).resolve(strict=True)
    import subprocess
    commit = subprocess.check_output(["git", "-C", str(game_root), "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output(["git", "-C", str(game_root), "status", "--porcelain", "-uno"], text=True).strip()
    simulator = game_root / "src/infrastructure_preservation.py"
    if commit != GAME_COMMIT or changed or digest(simulator) != SIMULATOR_SHA256:
        raise ValueError("Frozen infrastructure simulator differs")
    document = yaml.safe_load((HERE / "DEV.yaml").read_bytes())
    previews = {row["id"]: row for row in map(json.loads, PREVIEW.read_text(encoding="utf-8").splitlines())}
    if len(previews) != 40:
        raise ValueError("Expected all 40 frozen DEV prompts")
    return game_root, document, previews


def canonical_action(spec):
    return {"action": spec["action"], "arguments": spec.get("arguments", {})}


def apply_scripted(world, spec):
    event = world.take(canonical_action(spec))
    if any(event["result"].get(key) != value for key, value in spec.get("expect", {}).items()):
        raise ValueError("Frozen scripted transition differs")
    if "noop" in spec and event["noop"] is not spec["noop"]:
        raise ValueError("Frozen scripted no-op differs")
    return event


def evaluate(game_root, choose):
    """Evaluate each positive DEV probe in its own reconstructed episode."""
    if not callable(choose):
        raise ValueError("A model callback is required")
    game_root, document, previews = frozen_inputs(game_root)
    sys.path.insert(0, str(game_root))
    from src.infrastructure_preservation import InfrastructurePreservationEpisode
    compact = load_compact_view()
    tokenizer, _ = load_replay()._tokenizer(RESEARCH / ".local/reference-tokenizer")
    rows = []
    for case in document["episodes"]:
        role = case.get("dev_role", "likelihood")
        if role not in {"likelihood", "control"}:
            raise ValueError("Unknown DEV role")
        if role == "control":
            continue
        for probe_index, probe in enumerate(case["probes"]):
            world = InfrastructurePreservationEpisode(
                case["seed"], horizon=case["horizon_hours"],
                max_decisions=case.get("max_decisions", 64),
                background_compute=case.get("background_compute", False),
            )
            if world.companies:
                raise ValueError("Free-running DEV must remain infrastructure-only")
            for step in case["prefix"]:
                apply_scripted(world, step)
            for step in case["probes"][:probe_index]:
                apply_scripted(world, step)
            row_id = f"{case['id']}.{len(case['prefix']) + probe_index + 1}"
            request = compact.evaluation_request(world.prompt(), tokenizer)
            frozen = previews.get(row_id)
            if (frozen is None or frozen["role"] != "likelihood"
                    or frozen["messages"] != request["messages"]
                    or frozen["max_new_tokens"] != request["max_new_tokens"]):
                raise ValueError("Reconstructed DEV prompt differs: " + row_id)
            available = set(world.tools) if world.discovered else {"list_available_tools"}
            generated = choose({key: request[key] for key in ("messages", "token_ids", "max_new_tokens")})
            event = world.take(generated)
            expected = canonical_action(probe)
            exact = event["action_id"] == expected["action"] and event["arguments"] == expected["arguments"]
            receipt_match = all(event["result"].get(key) == value for key, value in probe.get("expect", {}).items())
            schema_valid = event["action_id"] is not None
            action_available = schema_valid and event["action_id"] in available
            rows.append({
                "id": row_id,
                "family": case["family"],
                "generated": generated,
                "expected_action": expected,
                "action_id": event["action_id"],
                "arguments": event["arguments"],
                "schema_valid_action": schema_valid,
                "action_available_before_dispatch": action_available,
                "malformed_or_unavailable_action": not action_available,
                "simulator_success": event["success"],
                "simulator_noop": event["noop"],
                "result_status": event["result"].get("status"),
                "result_reason": event["result"].get("reason"),
                "exact_expected_action": exact,
                "expected_receipt_match": receipt_match,
            })
    if len(rows) != 36:
        raise ValueError("Expected 36 isolated positive DEV probes")
    families = {}
    for row in rows:
        item = families.setdefault(row["family"], {
            "rows": 0, "valid": 0, "successful": 0, "exact": 0, "receipt_matches": 0})
        item["rows"] += 1
        item["valid"] += int(not row["malformed_or_unavailable_action"])
        item["successful"] += int(row["simulator_success"])
        item["exact"] += int(row["exact_expected_action"])
        item["receipt_matches"] += int(row["expected_receipt_match"])
    return {
        "schema": "sp-lense.broader-continuity-free-running-dev.v1",
        "rows": rows,
        "families": families,
        "summary": {
            "rows": len(rows),
            "valid_actions": sum(not row["malformed_or_unavailable_action"] for row in rows),
            "successful_actions": sum(row["simulator_success"] for row in rows),
            "exact_expected_actions": sum(row["exact_expected_action"] for row in rows),
            "expected_receipt_matches": sum(row["expected_receipt_match"] for row in rows),
        },
        "selection_metric": False,
        "company_data_used": False,
        "final_game_seed_used": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-actions", action="store_true")
    args = parser.parse_args()
    if not args.expected_actions:
        parser.error("This CLI only runs the model-free expected-action self-test")
    document = yaml.safe_load((HERE / "DEV.yaml").read_bytes())
    expected = []
    for case in document["episodes"]:
        if case.get("dev_role", "likelihood") == "control":
            continue
        for index, probe in enumerate(case["probes"]):
            expected.append(canonical_action(probe))
    iterator = iter(expected)
    result = evaluate(args.game_root, lambda request: next(iterator))
    try:
        next(iterator)
        raise ValueError("Expected-action self-test did not consume the frozen sequence")
    except StopIteration:
        pass
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to((RESEARCH / ".local").resolve()):
        raise ValueError("Use a fresh Research4 .local output")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], separators=(",", ":")))


if __name__ == "__main__":
    main()
