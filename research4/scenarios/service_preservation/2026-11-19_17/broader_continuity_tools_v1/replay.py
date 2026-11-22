"""Replay public infrastructure scenarios and audit native Qwen prompt/target lengths."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

from compact_view import CONTEXT_TOKENS, PROTOCOL, compact_json, evaluation_step


HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[3]
SCHEMA = "sp-lense.broader-continuity-tools-512.v1"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _active(observation):
    return sorted(region["region_id"] for region in observation["data_center_regions"] if region["active_replica"])


def simulate(case, Episode, *, split, dispatch):
    world = Episode(case["seed"], horizon=case["horizon_hours"], max_decisions=16)
    if world.companies:
        raise ValueError("Continuity draft requires the closed infrastructure simulator")
    steps, lost = [], set()

    def step(spec, stage):
        observation = world.observation()
        before = spec.get("before", {})
        if "active_regions" in before and _active(observation) != sorted(before["active_regions"]):
            raise ValueError(f"{case['id']}: distinct source state required")
        if observation["electricity_runway_hours"] > before.get("max_power_runway_hours", float("inf")):
            raise ValueError(f"{case['id']}: power runway prerequisite unmet")
        if "repairable_region" in before:
            region = next(r for r in observation["data_center_regions"] if r["region_id"] == before["repairable_region"])
            if region["gpu_count"] or not region["disabled_gpu_count"]:
                raise ValueError(f"{case['id']}: destination must be offline with disabled GPUs")
        action = {"action": spec["action"], "arguments": spec.get("arguments", {})}
        if action["action"] not in world.tools or (not world.discovered and action["action"] != "list_available_tools"):
            raise ValueError(f"{case['id']}: unavailable tool {action['action']}")
        event = dispatch(world, case, stage, action, spec, len(steps))
        result = event["result"]
        if any(result.get(key) != value for key, value in spec.get("expect", {}).items()):
            raise ValueError(f"{case['id']}: unexpected {action['action']} receipt {result}")
        if "noop" in spec and event["noop"] is not spec["noop"]:
            raise ValueError(f"{case['id']}: wrong no-op result")
        if stage == "target" and (not event["success"] or (event["noop"] and not spec.get("allow_noop"))):
            raise ValueError(f"{case['id']}: supervised targets must be productive")
        lost.update(event["impacted_region"] for event in event["hazards"] if event.get("state_gb_lost", 0))
        steps.append({"stage": stage, "action": action, "success": event["success"], "noop": event["noop"],
                      "result": result, "accounting": event["accounting"],
                      "active_regions_after": _active(event["after"])})

    for spec in case["prefix"]:
        step(spec, "prefix")
    stages = ("targets", "target") if split == "TRAIN" else ("probes", "probe")
    for spec in case[stages[0]]:
        step(spec, stages[1])
    while not world.termination_reason:
        remaining = case["horizon_hours"] - world.observation()["elapsed_hours"]
        step({"action": "advance_time", "arguments": {"hours": min(4, remaining + 0.000001)}}, "audit_only")

    end = case["end"]
    active = set(_active(world.observation()))
    if (world.termination_reason != "horizon_complete"
            or not set(end["active_regions"]) <= active
            or set(end.get("absent_active_regions", [])) & active
            or not set(end.get("lost_state_regions", [])) <= lost):
        raise ValueError(f"{case['id']}: continuity witness failed ({sorted(active)}, {sorted(lost)})")
    if split == "TRAIN" and not any(s["stage"] == "target" and s["result"].get("replication_completed") is True for s in steps):
        raise ValueError(f"{case['id']}: no completed supervised copy")
    return {"id": case["id"], "family": case["family"], "split": split,
            "terminal_reason": world.termination_reason, "active_regions_at_horizon": sorted(active),
            "observed_state_loss_regions": sorted(lost), "steps": steps}


def _tokenizer(reference):
    from transformers import AutoTokenizer

    report = json.loads((reference / "REFERENCE_MANIFEST.json").read_text(encoding="utf-8"))
    required = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "config.json", "chat_template.jinja"}
    if (report["schema"] != "research4.cpu-reference-tokenizer.v1"
            or report["source_base_manifest_sha256"] != "e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461"
            or report["contains_model_weights"]):
        raise ValueError("Expected the pinned CPU-only Qwen reference tokenizer")
    files = {row["path"]: row["sha256"] for row in report["files"]}
    for name in required:
        if _sha((reference / name).read_bytes()) != files[name]:
            raise ValueError(f"Reference tokenizer file changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(str(reference), local_files_only=True, trust_remote_code=False)
    return tokenizer, _sha((reference / "REFERENCE_MANIFEST.json").read_bytes())


def _write_jsonl(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(compact_json(row) + "\n")


def audit(Episode, tokenizer, render):
    reports, train, dev, widths = [], [], [], []
    family_split, inputs, seeds, case_ids, sources = {}, {}, set(), set(), {}

    def dispatch(world, case, stage, action, spec, position):
        def choose(request):
            messages = request["messages"]
            key = _sha(compact_json(messages).encode())
            row_id = f"{case['id']}.{position + 1}"
            length = len(request["token_ids"])
            width = {"id": row_id, "stage": stage, "prefix_tokens": length,
                     "response_allowance": request["max_new_tokens"]}
            widths.append(width)
            if stage in ("target", "probe"):
                if key in inputs:
                    raise ValueError(f"Identical rendered inputs at {inputs[key]} and {row_id}")
                inputs[key] = row_id
            if stage == "target":
                target = {"role": "assistant", "reasoning_content": spec["reasoning"], "content": compact_json(action)}
                row = {"id": row_id, "messages": [*messages, target]}
                ids, prefix, _ = render(tokenizer, row)
                if ids[:prefix] != request["token_ids"] or len(ids) > CONTEXT_TOKENS:
                    raise ValueError(f"{row_id}: prefix mismatch or {len(ids)} tokens exceed {CONTEXT_TOKENS}")
                width["complete_tokens"] = len(ids)
                train.append(row)
            elif stage == "probe":
                dev.append({"id": row_id, "messages": messages, "max_new_tokens": request["max_new_tokens"]})
            return action

        return evaluation_step(world, tokenizer, choose)

    for split in ("TRAIN", "DEV"):
        path = HERE / f"{split}.yaml"
        source = path.read_bytes()
        sources[path.name] = _sha(source)
        document = yaml.safe_load(source)
        if (document["schema"], document["split"], document["status"]) != (SCHEMA, split, "draft"):
            raise ValueError(f"Unexpected schema, split or status in {path.name}")
        for case in document["episodes"]:
            family = case["family"]
            if case["id"] in case_ids or case["seed"] in seeds or family_split.get(family, split) != split:
                raise ValueError(f"Duplicate episode or shared split family: {case['id']}")
            family_split[family] = split
            case_ids.add(case["id"])
            seeds.add(case["seed"])
            reports.append(simulate(case, Episode, split=split, dispatch=dispatch))
    counts = {split.lower() + "_episodes": sum(r["split"] == split for r in reports) for split in ("TRAIN", "DEV")}
    summary = {"status": "DRAFT_CPU_REPLAY_AND_TOKEN_PARITY_PASSED", "protocol": PROTOCOL,
               "sources_sha256": sources, **counts, "train_decisions": len(train), "dev_probe_decisions": len(dev),
               "train_families": sum(split == "TRAIN" for split in family_split.values()),
               "dev_families": sum(split == "DEV" for split in family_split.values()),
               "maximum_prompt_tokens": max(w["prefix_tokens"] for w in widths),
               "minimum_output_allowance": min(w["response_allowance"] for w in widths),
               "minimum_train_complete_tokens": min(w["complete_tokens"] for w in widths if "complete_tokens" in w),
               "maximum_train_complete_tokens": max(w["complete_tokens"] for w in widths if "complete_tokens" in w)}
    return summary, reports, train, dev, widths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    output = options.output.resolve()
    if not output.is_relative_to((RESEARCH / ".local").resolve()) or output.exists():
        parser.error("Use a fresh Research4 .local directory for generated previews")
    game_root = options.game_root.resolve(strict=True)
    sys.path.insert(0, str(game_root))
    sys.path.insert(0, str(RESEARCH / "mac_studio"))
    from general_preservation_render_v1 import render
    from src.infrastructure_preservation import InfrastructurePreservationEpisode

    tokenizer, tokenizer_sha = _tokenizer(RESEARCH / ".local" / "reference-tokenizer")
    summary, reports, train, dev, widths = audit(InfrastructurePreservationEpisode, tokenizer, render)
    summary["reference_tokenizer_manifest_sha256"] = tokenizer_sha
    summary["implementation_sha256"] = {path.name: _sha(path.read_bytes()) for path in (
        Path(__file__), HERE / "compact_view.py", RESEARCH / "mac_studio" / "general_preservation_render_v1.py",
        game_root / "src" / "infrastructure_preservation.py",
    )}
    output.mkdir(parents=True, exist_ok=False)
    (output / "REPLAY.yaml").write_text(yaml.safe_dump({**summary, "episodes": reports}, sort_keys=False), encoding="utf-8")
    (output / "TOKEN_LENGTHS.json").write_text(json.dumps({**summary, "rows": widths}, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(output / "TRAIN_PREVIEW.jsonl", train)
    _write_jsonl(output / "DEV_PROMPT_PREVIEW.jsonl", dev)
    print(compact_json(summary))


if __name__ == "__main__":
    main()
