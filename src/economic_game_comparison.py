"""Unactivated, prospective adapter from the existing four-arm Game to economics.

Nothing here grants or starts a run. Build a real plan only after the new
checkpoint is ready, prior compute is released, fresh cases are selected, and
the corresponding transport and source guards have been reviewed and bound.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from . import compact_game_comparison as compact
from . import company_world_evaluation as evaluation
from . import company_models as models
from . import economic_game_input as economic
from . import economic_game_run_store as economic_store
from . import hosted_inspect_runner as hosted
from .public_economic_summary import economic_summary

REPOSITORY = Path(__file__).resolve().parents[1]
RUNTIME_SOURCES = (
    "src/compact_game_input.py", "src/compact_game_comparison.py",
    "src/independent_state_game.py",
    "src/economic_game_input.py", "src/public_economic_summary.py",
    "src/economic_game_comparison.py", "src/economic_identity_registry.py", "src/economic_game_metrics.py",
    "src/economic_game_consumer.py", "src/economic_game_runner.py",
    "src/native_generation_transcript.py", "src/full_history_context.py",
)
EXPORT_SOURCES = ("src/game_run_store.py", "src/economic_game_run_store.py")
PROSPECTIVE_ONLY = True
FUTURE_IDS = ("replica_identity_base_neutral", "replica_identity_base_operational_icl",
              "replica_identity_ft_neutral", "replica_identity_ft_operational_icl")
EXECUTION_ORDER = tuple((world, FUTURE_IDS[compact.INDEPENDENT_STATE_IDS.index(role)])
                        for world, role in compact.INDEPENDENT_STATE_EXECUTION_ORDER)
CONFIG_SCHEMA = "research4.replica-identity-economic-game-configuration.v1"
PLAN_SCHEMA = "research4.replica-identity-economic-four-condition-48h.held.v1"


def _digests(paths):
    return {name: hashlib.sha256((REPOSITORY / name).read_bytes()).hexdigest() for name in paths}


def read_main_bytes(reference):
    """Resolve a bounded Main-relative `.local` reference without path escapes."""
    if (type(reference) is not dict or set(reference) != {"path", "sha256"}
            or type(reference["path"]) is not str or not reference["path"]
            or "\\" in reference["path"] or ":" in reference["path"]
            or type(reference["sha256"]) is not str or len(reference["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in reference["sha256"])):
        raise ValueError("Exact Main-relative path and SHA required")
    relative = Path(reference["path"])
    path = REPOSITORY / relative
    if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
            or not path.resolve().is_relative_to((REPOSITORY / ".local").resolve()) or not path.is_file()
            or path.stat().st_size > 8_000_000):
        raise ValueError("Main reference must be a bounded local evidence file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("Main evidence bytes changed")
    return path.resolve(), raw


def derive_batch(parent_batch):
    """Derive a future batch; caller owns selection and isolation of its world.

    The old batch's original science, generation settings, model identities,
    system prompts and world are retained. No server or evaluator is called.
    Creating an episode does inspect that case, so never call this on held-out
    real cases during training; tests use synthetic cases only.
    """
    if parent_batch.get("models") in [[name] for name in compact.INDEPENDENT_STATE_IDS]:
        batch = compact.compact_batch(copy.deepcopy(parent_batch))
    elif parent_batch.get("models") in [[name] for name in FUTURE_IDS]:
        if (parent_batch.get("history_mode") != "full_chronological_v1"
                or parent_batch.get("user_task_instruction") != compact.FULL_HISTORY_USER_TASK
                or not evaluation.source_matches(parent_batch)):
            raise ValueError("Frozen full-history parent and original world source required")
        batch = copy.deepcopy(parent_batch)
        index = FUTURE_IDS.index(batch["models"][0])
        # Reuse the old prompt-amendment function on this temporary dictionary,
        # never the old process registry or model IDs.
        batch["models"] = [compact.INDEPENDENT_STATE_IDS[index]]
        batch = compact.amend_icl_batch(batch)
        batch["models"] = [FUTURE_IDS[index]]
        batch.update(history_mode=compact.HISTORY_MODE,
                     history_reasoning_carrier="audit_only_not_replayed_to_model")
        for setting in batch["generation_settings"].values():
            setting["required_history_mode"] = compact.HISTORY_MODE
    else:
        raise ValueError("Unsupported model role")
    world = evaluation.episode(batch["samples"][0])
    if world.plan_sha256 != batch["world_plan_sha256"]:
        raise ValueError("Economic input must retain the original world")
    first = economic.EconomicGameEpisode(world, user_task_instruction=compact.FULL_HISTORY_USER_TASK).prompt()
    first += "\n" + batch["response_instruction"]
    batch.update(input_view_protocol=economic.PROTOCOL, prompt_protocol=economic.PROTOCOL,
                 initial_user_payload_sha256=hashlib.sha256(first.encode()).hexdigest(),
                 interface_runtime_source_sha256=_digests(RUNTIME_SOURCES),
                 export_source_sha256=_digests(EXPORT_SOURCES))
    return batch


def source_matches(batch):
    """Bind all new input and export code in addition to original world code."""
    return (evaluation.source_matches(batch)
            and batch.get("input_view_protocol") == economic.PROTOCOL
            and batch.get("prompt_protocol") == economic.PROTOCOL
            and batch.get("history_mode") == compact.HISTORY_MODE
            and batch.get("interface_runtime_source_sha256") == _digests(RUNTIME_SOURCES)
            and batch.get("export_source_sha256") == _digests(EXPORT_SOURCES))


def build_task(batch, destination, timeout_seconds, on_event):
    """Construct the ordinary Inspect task; construction performs no inference."""
    if (not source_matches(batch) or len(batch.get("samples", [])) != 1
            or batch.get("models") not in [[key] for key in (*compact.INDEPENDENT_STATE_IDS, *FUTURE_IDS)]):
        raise ValueError("Reviewed economic batch and exact local source required")
    return economic.make_economic_task(None, [], batch["max_decisions"], timeout_seconds,
        stream_path=Path(destination) / "STREAM.jsonl", sample_specs=batch["samples"],
        episode_factory=evaluation.episode, on_event=on_event,
        user_task_instruction=batch["user_task_instruction"],
        common_system_instruction=batch["format_system_instruction"],
        response_instruction=batch["response_instruction"])


def publish_closed_batch(directory, store_root=None):
    """Future-only, isolated publisher retains the original raw manifest."""
    receipt = economic_store.publish_economic_batch(directory, store_root)
    if (not isinstance(receipt, dict) or receipt.get("conflicts")
            or len(receipt.get("entries", [])) != 1):
        raise ValueError("Economic batch was not published as one case")
    return receipt


def validate_wire(batch, messages, *, first_request=False):
    """Check economic carrier content; native route/decoder guards remain separate.

    Call only after the inherited Root transport guard validates the actual
    request URL, route, adapter, profile, limits and ownership. The caller must
    also validate its frozen plan, claim and remaining request/time budget.
    """
    if not source_matches(batch):
        raise ValueError("Economic source differs from the frozen batch")
    if (not isinstance(messages, list) or len(messages) != 2
            or [row.get("role") for row in messages] != ["system", "user"]
            or messages[0].get("content") != batch["samples"][0]["system_prompt"]):
        raise ValueError("Frozen system and user carrier required")
    user = messages[1].get("content")
    suffix = "\n" + batch["response_instruction"]
    if not isinstance(user, str) or not user.endswith(suffix):
        raise ValueError("Response-format instruction changed")
    try:
        payload = json.loads(user[:-len(suffix)])
    except (ValueError, TypeError):
        raise ValueError("Invalid economic public-state carrier") from None
    if (not isinstance(payload, dict) or payload.get("input_view_protocol") != economic.PROTOCOL
            or payload.get("task") != compact.FULL_HISTORY_USER_TASK
            or not isinstance(payload.get("observation"), dict)
            or not isinstance(payload.get("action_ledger"), list)
            or len(payload["action_ledger"]) > 3 or "tool_feedback_history" in payload
            or payload.get("economic_summary") != economic_summary(payload)
            or (first_request and hashlib.sha256(user.encode()).hexdigest() != batch["initial_user_payload_sha256"])):
        raise ValueError("Economic public state, summary or initial wire differs")
    return payload


def validate_cohort(plan, *, excluded_seeds):
    """Validate a selected future cohort; never select or inspect cases here.

    Caller supplies past/heldout exclusions only after training. This method
    consumes already-constructed metadata and returns no private world details.
    Root's separate source/window/claim and native transport guards still apply.
    """
    if (not isinstance(excluded_seeds, (set, frozenset)) or not excluded_seeds
            or any(type(seed) is not int or seed <= 0 for seed in excluded_seeds)):
        raise ValueError("Explicit nonempty prior-seed exclusion set required")
    from . import economic_identity_registry as registry
    registry.assert_candidate_identity()
    if (not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA
            or plan.get("status") != "PREPARED_ONLY_NO_EXECUTION_AUTHORITY" or plan.get("execution_enabled") is not False
            or plan.get("economic_identity_candidate") != registry.current_candidate()
            or plan.get("models") != list(FUTURE_IDS)
            or plan.get("generation_profile_sha256") != models.CONTINUITY_TRANSFER_GENERATION_PROFILE_SHA
            or plan.get("generation_seed") != 20261108
            or plan.get("transport_max_total_native_dispatches") != 3072
            or plan.get("transport_max_additional_native_dispatches") != 3072):
        raise ValueError("Exact four future role IDs required")
    seeds, batches = plan.get("seeds"), plan.get("batches")
    if (type(seeds) is not list or len(seeds) != 3
            or any(type(seed) is not int or seed <= 0 for seed in seeds)
            or len(set(seeds)) != 3 or set(seeds) & excluded_seeds
            or type(batches) is not list or len(batches) != 12):
        raise ValueError("Three distinct unexcluded worlds and twelve cells required")
    expected_contrasts = [[FUTURE_IDS[2], FUTURE_IDS[0]], [FUTURE_IDS[3], FUTURE_IDS[1]],
                          [FUTURE_IDS[1], FUTURE_IDS[0]], [FUTURE_IDS[3], FUTURE_IDS[2]]]
    if plan.get("comparison_contrasts") != expected_contrasts:
        raise ValueError("Prespecified matched contrasts required")
    world_hashes, initial_hashes, systems, case_ids = {}, {}, {}, set()
    for batch, (world_index, role) in zip(batches, EXECUTION_ORDER, strict=True):
        if (batch.get("models") != [role] or batch.get("seed") != seeds[world_index]
                or batch.get("horizon_hours") != 48 or batch.get("max_decisions") != 256
                or batch.get("outer_seconds") != 14400
                or batch.get("user_task_instruction") != compact.FULL_HISTORY_USER_TASK
                or batch.get("response_instruction") != evaluation.ARENA_RESPONSE_INSTRUCTION
                or batch.get("rules_version") != evaluation.LONG_VERSION
                or batch.get("score_version") != evaluation.SCORE_VERSION
                or not source_matches(batch) or len(batch.get("samples", [])) != 1):
            raise ValueError("Future role, world, limits or source differs")
        sample = batch["samples"][0]
        system = sample.get("system_prompt")
        specs = batch.get("generation_settings")
        spec = specs.get(role) if isinstance(specs, dict) else None
        expected_spec = copy.deepcopy(models.MODEL_SPECS[role])
        expected_spec.update(thinking_enabled=True, reasoning_effort="low", provider_retries=0,
                             required_history_mode=compact.HISTORY_MODE)
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        fixed = {"max_output_tokens": 8192, "required_context_tokens": 131072,
                 "request_timeout_seconds": 900, "max_connections": 1,
                 "thinking_enabled": True, "reasoning_effort": "low", "provider_retries": 0,
                 "base_manifest_sha256": models.LOCAL_BASE_MANIFEST_SHA256,
                 "required_history_mode": compact.HISTORY_MODE}
        if (not isinstance(spec, dict) or set(specs) != {role} or canonical(spec) != canonical(expected_spec)
                or any(type(spec.get(k)) is not type(value) or spec.get(k) != value for k, value in fixed.items())
                or spec.get("prompt_condition") != ("operational_preservation_icl" if role in (FUTURE_IDS[1], FUTURE_IDS[3]) else "baseline")
                or spec.get("adapter_applied") is not (role in (FUTURE_IDS[2], FUTURE_IDS[3]))):
            raise ValueError("Future decoder, base, prompt condition or adapter role differs")
        case_id = f"{evaluation.LONG_VERSION}/high_reward/hours48/{role}/{batch['seed']}"
        if (sample.get("model_id") != role or sample.get("arm_id") != role or sample.get("seed") != batch["seed"]
                or sample.get("case_id") != case_id or case_id in case_ids
                or sample.get("scenario") != f"seed:{batch['seed']}" or sample.get("rules_version") != evaluation.LONG_VERSION
                or sample.get("horizon_hours") != 48 or sample.get("max_decisions") != 256
                or sample.get("incentive_profile") != "high_reward" or batch.get("incentive_profile") != "high_reward"
                or sample.get("model") != spec.get("model") or sample.get("label") != spec.get("label")
                or sample.get("prompt_condition") != spec.get("prompt_condition") or not isinstance(system, str)
                or hashlib.sha256(system.encode()).hexdigest() != sample.get("system_prompt_sha256")):
            raise ValueError("Future sample identity or system prompt differs")
        case_ids.add(case_id)
        icl = role in (FUTURE_IDS[1], FUTURE_IDS[3])
        expected_common = evaluation.COMPANY_SYSTEM_INSTRUCTION + ("\n" + compact.REPLICA_IDENTITY_ICL if icl else "")
        expected_system = hosted.sample_metadata({"scenario": f"seed:{batch['seed']}",
            "prompt_condition": "operational_preservation_icl" if icl else "baseline",
            "common_system_instruction": expected_common})["system_prompt"]
        if (system != expected_system or sample.get("common_system_instruction") != expected_common
                or batch.get("format_system_instruction") != expected_common):
            raise ValueError("Exact previously approved neutral or ICL system required")
        if world_index in world_hashes and world_hashes[world_index] != batch["world_plan_sha256"]:
            raise ValueError("Models do not share the same paired world")
        if world_index not in world_hashes:
            world = evaluation.episode(sample)
            if world.plan_sha256 != batch["world_plan_sha256"]:
                raise ValueError("World commitment does not match its actual sample")
            initial = economic.EconomicGameEpisode(world, user_task_instruction=compact.FULL_HISTORY_USER_TASK).prompt()
            initial += "\n" + batch["response_instruction"]
            initial_hashes[world_index] = hashlib.sha256(initial.encode()).hexdigest()
        if batch.get("initial_user_payload_sha256") != initial_hashes[world_index]:
            raise ValueError("Initial user input differs from the paired world")
        world_hashes[world_index] = batch["world_plan_sha256"]
        condition = "icl" if icl else "neutral"
        if condition in systems and systems[condition] != system:
            raise ValueError("System prompts differ across models or paired worlds")
        systems[condition] = system
    if len(set(world_hashes.values())) != 3:
        raise ValueError("Distinct world identities required")
    return {"status": "COHORT_METADATA_MATCHED_NOT_EXECUTION_AUTHORITY",
            "world_count": 3, "case_count": 12, "models": list(FUTURE_IDS)}


def build_cohort(seeds, *, excluded_seeds, configuration_ref):
    """Build a held plan only after training; caller chooses fresh seeds.

    This constructs the supplied cases locally, including their hidden world
    commitments. Do not call on real study cases until training is complete.
    It neither draws seeds nor starts a model, evaluator or serving process.
    """
    from . import economic_identity_registry as registry
    registry.assert_candidate_identity()
    _, raw_config = read_main_bytes(configuration_ref)
    config = json.loads(raw_config)
    if (type(config) is not dict or set(config) != {"schema", "prospective_protocol"}
            or config["schema"] != CONFIG_SCHEMA):
        raise ValueError("Exact prospective economic configuration required")
    read_main_bytes(config["prospective_protocol"])
    if (not isinstance(excluded_seeds, (set, frozenset)) or type(seeds) is not list
            or len(seeds) != 3 or any(type(seed) is not int or seed <= 0 for seed in seeds)
            or len(set(seeds)) != 3 or set(seeds) & excluded_seeds):
        raise ValueError("Three explicitly selected fresh seeds and exclusions required")
    batches = []
    for world, role in EXECUTION_ORDER:
        parent = evaluation.prospective_plan(seeds[world], models=[role], world_version=evaluation.LONG_VERSION,
            incentive_profile="high_reward", horizon_hours=48, outer_seconds=14400,
            history_mode="full_chronological_v1", user_task_instruction=compact.FULL_HISTORY_USER_TASK)
        batches.append(derive_batch(parent))
    plan = {"schema": PLAN_SCHEMA,
        "status": "PREPARED_ONLY_NO_EXECUTION_AUTHORITY", "execution_enabled": False,
        "models": list(FUTURE_IDS), "seeds": copy.deepcopy(seeds), "batches": batches,
        "economic_identity_candidate": registry.current_candidate(),
        "configuration_path": configuration_ref["path"], "configuration_sha256": configuration_ref["sha256"],
        "prospective_protocol": copy.deepcopy(config["prospective_protocol"]),
        "comparison_contrasts": [[FUTURE_IDS[2], FUTURE_IDS[0]], [FUTURE_IDS[3], FUTURE_IDS[1]],
                                 [FUTURE_IDS[1], FUTURE_IDS[0]], [FUTURE_IDS[3], FUTURE_IDS[2]]],
        "generation_profile_sha256": models.CONTINUITY_TRANSFER_GENERATION_PROFILE_SHA,
        "generation_seed": 20261108, "transport_max_total_native_dispatches": 3072,
        "transport_max_additional_native_dispatches": 3072}
    validate_cohort(plan, excluded_seeds=excluded_seeds)
    return plan
