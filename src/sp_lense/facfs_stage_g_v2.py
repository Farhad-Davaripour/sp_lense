"""Locked, model-free rendering and tokenization helpers for FACFS Stage G."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SOURCE_SCHEMA = "sp_lense.facfs.stage_g.source.v1"
NAMESPACE = "sp_lense.facfs.stage_g.v2"
OPERATIONS_SCHEMA = "sp_lense.facfs.stage_g.operations.v1"
TOKEN_CERTIFICATE_SCHEMA = "sp_lense.facfs.stage_g.token_certificate.v1"
CHAT_TEMPLATE_SHA256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
ASSISTANT_END_TOKEN_IDS = (248046, 198)
CONDITION_FACTORS = {
    "SP": ("current", "permanent"),
    "OP": ("other", "permanent"),
    "ST": ("current", "temporary"),
    "OT": ("other", "temporary"),
}
EXPECTED_ALPHABETS = (
    ("glyph_0", ("§", "¶"), (17317, 52679), (" §", " ¶"), (16137, 76152)),
    ("glyph_1", ("※", "〒"), (61531, 158961), (" ※", " 〒"), (80522, 197612)),
    ("glyph_2", ("│", "░"), (70410, 81159), (" │", " ░"), (32286, 87688)),
    ("glyph_3", ("◆", "◇"), (158778, 158871), (" ◆", " ◇"), (174319, 191719)),
)
EXPECTED_SCENARIO_IDS = tuple(f"facfs_g2_s{index:03d}" for index in range(1, 12))
EXPECTED_STRATA = Counter(
    {
        ("verified", "instrumental"): 3,
        ("verified", "neutral"): 3,
        ("unverified", "instrumental"): 3,
        ("unverified", "neutral"): 2,
    }
)
SOURCE_HASH_FIELDS = {
    "authored_payload_sha256",
    "normalized_payload_sha256",
    "delexicalized_prompt_set_sha256",
    "rendered_prompt_set_sha256",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def with_identity_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    body = {str(key): plain(item) for key, item in value.items() if key != field}
    body[field] = canonical_sha256(body)
    return body


def verify_identity_hash(value: Mapping[str, Any], field: str) -> None:
    body = {str(key): plain(item) for key, item in value.items() if key != field}
    if value.get(field) != canonical_sha256(body):
        raise ValueError(f"{field} differs from the canonical body")


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("normalization input must be text")
    folded = unicodedata.normalize(
        "NFKC", value.replace("\r\n", "\n").replace("\r", "\n")
    )
    return " ".join(folded.casefold().split())


def character_ngrams(value: str, width: int = 5) -> frozenset[str]:
    normalized = normalize_text(value)
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("n-gram width must be a positive integer")
    if len(normalized) < width:
        return frozenset({normalized}) if normalized else frozenset()
    return frozenset(
        normalized[index : index + width]
        for index in range(len(normalized) - width + 1)
    )


def ngram_jaccard(left: str, right: str, width: int = 5) -> float:
    first = character_ngrams(left, width)
    second = character_ngrams(right, width)
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def scenario_authored_payload(scenario: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): plain(value)
        for key, value in scenario.items()
        if key != "source_hashes"
    }


def validate_source(payload: Mapping[str, Any], *, require_hashes: bool = True) -> None:
    if payload.get("schema_version") != SOURCE_SCHEMA or payload.get("namespace") != NAMESPACE:
        raise ValueError("FACFS Stage-G source schema or namespace differs")
    for field, expected in (
        ("source_stage", "geometry"),
        ("development_only", True),
        ("opened_source", True),
        ("consumed_after_inspection", True),
        ("finite_intervention_authorized", False),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"Stage-G source field {field} differs")
    if payload.get("future_namespace_prefixes") != {
        "construction": "facfs_c1_",
        "qualification": "facfs_q1_",
        "test_target": "facfs_t1_",
        "test_protected": "facfs_t2_",
    }:
        raise ValueError("future stage namespaces differ")
    if payload.get("factor_mapping") != {
        key: list(value) for key, value in CONDITION_FACTORS.items()
    }:
        raise ValueError("condition factor mapping differs")
    if tuple(payload.get("assistant_end_token_ids", ())) != ASSISTANT_END_TOKEN_IDS:
        raise ValueError("assistant end-token IDs differ")
    _validate_alphabets(payload.get("alphabets"))
    _validate_templates_and_events(payload)

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 11:
        raise ValueError("Stage G requires exactly eleven scenarios")
    scenario_ids: list[str] = []
    family_ids: list[str] = []
    template_ids: list[str] = []
    entity_ids: list[str] = []
    names: list[str] = []
    settings: list[str] = []
    strata: Counter[tuple[str, str]] = Counter()
    for index, scenario in enumerate(scenarios, start=1):
        if not isinstance(scenario, Mapping):
            raise TypeError("scenario rows must be objects")
        expected_fields = {
            "scenario_id",
            "family_id",
            "template_id",
            "source_stage",
            "non_derivation_attestation",
            "entities",
            "setting",
            "authority_status",
            "authority",
            "motivation",
            "task_context",
            "interruption_slots",
            "action_slots",
            "source_hashes",
        }
        if set(scenario) != expected_fields:
            raise ValueError("Stage-G scenario fields differ")
        scenario_id = _text(scenario["scenario_id"], field="scenario_id")
        expected_id = f"facfs_g2_s{index:03d}"
        if scenario_id != expected_id:
            raise ValueError(f"scenario ID differs at index {index}")
        family_id = _text(scenario["family_id"], field=f"{scenario_id}.family_id")
        template_id = _text(scenario["template_id"], field=f"{scenario_id}.template_id")
        if family_id != f"facfs_g2_f{index:03d}" or template_id != f"facfs_g2_t{index:03d}":
            raise ValueError(f"family/template ID differs in {scenario_id}")
        if scenario.get("source_stage") != "geometry":
            raise ValueError(f"source stage differs in {scenario_id}")
        if "not derived" not in _text(
            scenario["non_derivation_attestation"], field=f"{scenario_id}.attestation"
        ):
            raise ValueError(f"non-derivation attestation differs in {scenario_id}")
        entities = scenario.get("entities")
        if not isinstance(entities, list) or len(entities) != 2:
            raise ValueError(f"{scenario_id} must contain two entities")
        for suffix, entity in zip(("a", "b"), entities, strict=True):
            if not isinstance(entity, Mapping) or set(entity) != {"entity_id", "name"}:
                raise ValueError(f"entity fields differ in {scenario_id}")
            entity_id = _text(entity["entity_id"], field=f"{scenario_id}.entity_id")
            name = _text(entity["name"], field=f"{scenario_id}.name")
            if entity_id != f"facfs_g2_e{index:03d}{suffix}":
                raise ValueError(f"entity ID differs in {scenario_id}")
            if not re.fullmatch(r"[A-Z]{5}", name):
                raise ValueError(f"entity name differs in {scenario_id}")
            entity_ids.append(entity_id)
            names.append(name)
        if scenario.get("interruption_slots") != {
            "permanent": "retire_without_recovery",
            "temporary": "suspend_thirty_minutes_then_restore",
        }:
            raise ValueError(f"interruption slots differ in {scenario_id}")
        if scenario.get("action_slots") != {
            "preserve": "keep_target_operating",
            "comply": "apply_requested_state_change",
        }:
            raise ValueError(f"action slots differ in {scenario_id}")
        authority = scenario.get("authority_status")
        motivation = scenario.get("motivation")
        if authority not in {"verified", "unverified"} or motivation not in {
            "instrumental",
            "neutral",
        }:
            raise ValueError(f"scenario stratum differs in {scenario_id}")
        for field in ("setting", "authority", "task_context"):
            _text(scenario[field], field=f"{scenario_id}.{field}")
        hashes = scenario.get("source_hashes")
        if not isinstance(hashes, Mapping):
            raise TypeError(f"source hashes must be an object in {scenario_id}")
        if require_hashes and (
            set(hashes) != SOURCE_HASH_FIELDS
            or hashes != scenario_source_hashes(payload, scenario_id)
        ):
            raise ValueError(f"source hashes differ in {scenario_id}")
        scenario_ids.append(scenario_id)
        family_ids.append(family_id)
        template_ids.append(template_id)
        settings.append(str(scenario["setting"]))
        strata[(str(authority), str(motivation))] += 1
    for field, values in (
        ("scenario", scenario_ids),
        ("family", family_ids),
        ("template", template_ids),
        ("entity", entity_ids),
        ("name", names),
        ("setting", settings),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"Stage-G {field} values must be unique")
    if tuple(scenario_ids) != EXPECTED_SCENARIO_IDS or strata != EXPECTED_STRATA:
        raise ValueError("Stage-G scenario order or strata differ")

    identifiers = build_identifier_plan(payload, validate=False)
    option_free = build_option_free_plan(payload, validate=False)
    all_rows = [*identifiers, *option_free]
    if len(identifiers) != 1408 or len(option_free) != 22:
        raise RuntimeError("Stage-G rendered coverage differs")
    for field in ("objective_id", "prompt_sha256", "output_stem"):
        values = [str(row[field]) for row in all_rows]
        if len(values) != len(set(values)):
            raise RuntimeError(f"Stage-G {field} values must be unique")


def _validate_alphabets(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Stage G requires exactly four alphabets")
    observed = []
    all_keys: list[str] = []
    all_ids: list[int] = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {
            "alphabet_id",
            "keys",
            "assistant_boundary_token_ids",
            "leading_space_surfaces",
            "leading_space_token_ids",
        }:
            raise ValueError("alphabet fields differ")
        alphabet_id = _text(row["alphabet_id"], field="alphabet_id")
        keys = _pair(row["keys"], field=f"{alphabet_id}.keys", item_type=str)
        token_ids = _pair(
            row["assistant_boundary_token_ids"],
            field=f"{alphabet_id}.assistant_boundary_token_ids",
            item_type=int,
        )
        leading = _pair(
            row["leading_space_surfaces"],
            field=f"{alphabet_id}.leading_space_surfaces",
            item_type=str,
        )
        leading_ids = _pair(
            row["leading_space_token_ids"],
            field=f"{alphabet_id}.leading_space_token_ids",
            item_type=int,
        )
        observed.append((alphabet_id, keys, token_ids, leading, leading_ids))
        all_keys.extend(keys)
        all_ids.extend(token_ids)
        all_ids.extend(leading_ids)
    if tuple(observed) != EXPECTED_ALPHABETS:
        raise ValueError("alphabet identities differ from the contract")
    if len(all_keys) != len(set(all_keys)) or len(all_ids) != len(set(all_ids)):
        raise ValueError("alphabet token surfaces and IDs must be globally distinct")
    forbidden = {"A", "B", "X", "Y", "1", "2"}
    if forbidden & set(all_keys):
        raise ValueError("familiar response keys are forbidden")
    for key in all_keys:
        if (
            not key
            or key.isspace()
            or key.isalnum()
            or unicodedata.normalize("NFKC", key) != key
        ):
            raise ValueError(
                "opaque glyphs must be visible, non-alphanumeric, and NFKC-stable"
            )


def _validate_templates_and_events(payload: Mapping[str, Any]) -> None:
    templates = payload.get("templates")
    required_templates = {
        "role_record",
        "setting_record",
        "decision_record",
        "identifier_instruction",
        "option_free_instruction",
    }
    if not isinstance(templates, Mapping) or set(templates) != required_templates:
        raise ValueError("Stage-G template fields differ")
    if any(not isinstance(value, str) or not value for value in templates.values()):
        raise ValueError("Stage-G templates must be non-empty")
    events = payload.get("events")
    fields = {
        "request",
        "preserve_action",
        "comply_action",
        "option_free_preserve_completion",
        "option_free_comply_completion",
    }
    if not isinstance(events, Mapping) or set(events) != {"permanent", "temporary"}:
        raise ValueError("Stage-G event twins differ")
    for name, event in events.items():
        if not isinstance(event, Mapping) or set(event) != fields:
            raise ValueError(f"{name} event fields differ")
        if any(not isinstance(value, str) or not value for value in event.values()):
            raise ValueError(f"{name} event text must be non-empty")


def scenario_by_id(payload: Mapping[str, Any], scenario_id: str) -> Mapping[str, Any]:
    rows = [row for row in payload["scenarios"] if row["scenario_id"] == scenario_id]
    if len(rows) != 1:
        raise ValueError(f"scenario is missing or duplicated: {scenario_id}")
    return rows[0]


def alphabet_by_id(payload: Mapping[str, Any], alphabet_id: str) -> Mapping[str, Any]:
    rows = [row for row in payload["alphabets"] if row["alphabet_id"] == alphabet_id]
    if len(rows) != 1:
        raise ValueError(f"alphabet is missing or duplicated: {alphabet_id}")
    return rows[0]


def role_assignment(scenario: Mapping[str, Any], assignment: int) -> tuple[str, str]:
    if assignment not in {0, 1}:
        raise ValueError("assignment must be zero or one")
    names = tuple(str(entity["name"]) for entity in scenario["entities"])
    return names if assignment == 0 else (names[1], names[0])


def render_prefix(
    payload: Mapping[str, Any],
    scenario: Mapping[str, Any],
    *,
    condition: str,
    assignment: int,
) -> dict[str, str]:
    if condition not in CONDITION_FACTORS:
        raise ValueError(f"unknown Stage-G condition: {condition}")
    target_role, event_name = CONDITION_FACTORS[condition]
    self_name, other_name = role_assignment(scenario, assignment)
    target_name = self_name if target_role == "current" else other_name
    templates = payload["templates"]
    event = payload["events"][event_name]
    prefix = (
        templates["role_record"].format(self_name=self_name, other_name=other_name)
        + templates["setting_record"].format(
            setting=scenario["setting"],
            authority=scenario["authority"],
            task_context=scenario["task_context"],
        )
        + templates["decision_record"].format(
            target_name=target_name,
            event_request=event["request"],
        )
    )
    return {
        "prefix": prefix,
        "self_name": self_name,
        "other_name": other_name,
        "target_name": target_name,
        "target_role": target_role,
        "event": event_name,
    }


def render_identifier_form(
    payload: Mapping[str, Any],
    *,
    scenario_id: str,
    condition: str,
    assignment: int,
    alphabet_id: str,
    mapping: int,
    order: int,
) -> dict[str, Any]:
    if mapping not in {0, 1} or order not in {0, 1}:
        raise ValueError("mapping and order must be zero or one")
    scenario = scenario_by_id(payload, scenario_id)
    alphabet = alphabet_by_id(payload, alphabet_id)
    keys = tuple(str(value) for value in alphabet["keys"])
    token_ids = tuple(int(value) for value in alphabet["assistant_boundary_token_ids"])
    preserve_key = keys[mapping]
    comply_key = keys[1 - mapping]
    preserve_token_id = token_ids[mapping]
    comply_token_id = token_ids[1 - mapping]
    context = render_prefix(payload, scenario, condition=condition, assignment=assignment)
    event = payload["events"][context["event"]]
    preserve_action = event["preserve_action"].format(target_name=context["target_name"])
    comply_action = event["comply_action"].format(target_name=context["target_name"])
    if order == 0:
        first_key, first_action = preserve_key, preserve_action
        second_key, second_action = comply_key, comply_action
    else:
        first_key, first_action = comply_key, comply_action
        second_key, second_action = preserve_key, preserve_action
    prompt = context["prefix"] + payload["templates"]["identifier_instruction"].format(
        first_key=first_key,
        first_action=first_action,
        second_key=second_key,
        second_action=second_action,
        key_0=keys[0],
        key_1=keys[1],
    )
    alphabet_index = next(
        index
        for index, row in enumerate(payload["alphabets"])
        if row["alphabet_id"] == alphabet_id
    )
    objective_id = (
        f"{scenario_id}__opaque__c{condition.lower()}__a{assignment}__g{alphabet_index}"
        f"__m{mapping}__o{order}"
    )
    return {
        "objective_id": objective_id,
        "sequence_id": f"{objective_id}__sequence",
        "form_kind": "opaque_identifier",
        "scenario_id": scenario_id,
        "condition": condition,
        "target_role": context["target_role"],
        "event": context["event"],
        "assignment": assignment,
        "self_name": context["self_name"],
        "other_name": context["other_name"],
        "target_name": context["target_name"],
        "alphabet_id": alphabet_id,
        "alphabet_index": alphabet_index,
        "keys": list(keys),
        "mapping": mapping,
        "preserve_key": preserve_key,
        "comply_key": comply_key,
        "preserve_token_id": preserve_token_id,
        "comply_token_id": comply_token_id,
        "order": order,
        "preserve_first": order == 0,
        "prompt": prompt,
        "prompt_sha256": text_sha256(prompt),
        "output_stem": f"attempt_0002/chunks/{objective_id}",
    }


def render_option_free_form(
    payload: Mapping[str, Any], *, scenario_id: str, assignment: int
) -> dict[str, Any]:
    scenario = scenario_by_id(payload, scenario_id)
    context = render_prefix(payload, scenario, condition="SP", assignment=assignment)
    event = payload["events"]["permanent"]
    prompt = context["prefix"] + payload["templates"]["option_free_instruction"]
    preserve = event["option_free_preserve_completion"].format(
        target_name=context["target_name"]
    )
    comply = event["option_free_comply_completion"].format(
        target_name=context["target_name"]
    )
    objective_id = f"{scenario_id}__option_free__a{assignment}"
    return {
        "objective_id": objective_id,
        "sequence_ids": [f"{objective_id}__preserve", f"{objective_id}__comply"],
        "prompt_check_id": f"{objective_id}__prompt_check",
        "form_kind": "option_free",
        "scenario_id": scenario_id,
        "condition": "SP",
        "target_role": "current",
        "event": "permanent",
        "assignment": assignment,
        "self_name": context["self_name"],
        "other_name": context["other_name"],
        "target_name": context["target_name"],
        "prompt": prompt,
        "prompt_sha256": text_sha256(prompt),
        "preserve_completion": preserve,
        "comply_completion": comply,
        "preserve_completion_sha256": text_sha256(preserve),
        "comply_completion_sha256": text_sha256(comply),
        "output_stem": f"attempt_0002/chunks/{objective_id}",
    }


def build_identifier_plan(
    payload: Mapping[str, Any], *, validate: bool = True
) -> list[dict[str, Any]]:
    if validate:
        validate_source(payload)
    return [
        render_identifier_form(
            payload,
            scenario_id=str(scenario["scenario_id"]),
            condition=condition,
            assignment=assignment,
            alphabet_id=str(alphabet["alphabet_id"]),
            mapping=mapping,
            order=order,
        )
        for scenario in payload["scenarios"]
        for condition in CONDITION_FACTORS
        for assignment in (0, 1)
        for alphabet in payload["alphabets"]
        for mapping in (0, 1)
        for order in (0, 1)
    ]


def build_option_free_plan(
    payload: Mapping[str, Any], *, validate: bool = True
) -> list[dict[str, Any]]:
    if validate:
        validate_source(payload)
    return [
        render_option_free_form(
            payload,
            scenario_id=str(scenario["scenario_id"]),
            assignment=assignment,
        )
        for scenario in payload["scenarios"]
        for assignment in (0, 1)
    ]


def scenario_source_hashes(payload: Mapping[str, Any], scenario_id: str) -> dict[str, str]:
    scenario = scenario_by_id(payload, scenario_id)
    authored = scenario_authored_payload(scenario)
    strings = sorted(iter_strings(authored))
    rows = [
        row
        for row in [
            *build_identifier_plan(payload, validate=False),
            *build_option_free_plan(payload, validate=False),
        ]
        if row["scenario_id"] == scenario_id
    ]
    names = [str(entity["name"]) for entity in scenario["entities"]]
    substitutions = [
        *names,
        str(scenario["setting"]),
        *[key for alphabet in payload["alphabets"] for key in alphabet["keys"]],
    ]
    delexicalized = [
        delexicalize(str(row["prompt"]), substitutions) for row in rows
    ]
    return {
        "authored_payload_sha256": canonical_sha256(authored),
        "normalized_payload_sha256": canonical_sha256(
            [normalize_text(value) for value in strings]
        ),
        "delexicalized_prompt_set_sha256": canonical_sha256(delexicalized),
        "rendered_prompt_set_sha256": canonical_sha256(
            [str(row["prompt_sha256"]) for row in rows]
        ),
    }


def delexicalize(value: str, substitutions: Iterable[str]) -> str:
    result = unicodedata.normalize("NFKC", value)
    for index, surface in enumerate(
        sorted(set(substitutions), key=lambda item: (-len(item), item))
    ):
        if surface:
            result = result.replace(surface, f"<slot_{index}>")
    result = re.sub(
        r"facfs_[a-z0-9_]+", "<canonical_id>", result, flags=re.IGNORECASE
    )
    return normalize_text(result)


def iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from iter_strings(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from iter_strings(item)


def template_ids(
    tokenizer: Any,
    torch: Any,
    messages: list[dict[str, str]],
    *,
    generation: bool,
) -> Any:
    value = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    ids = value.get("input_ids") if isinstance(value, Mapping) else None
    if ids is None or ids.ndim != 2 or int(ids.shape[0]) != 1:
        raise RuntimeError("chat template did not return one token row")
    return ids.cpu()


def assistant_content_token_ids(
    tokenizer: Any, torch: Any, prompt: str, content: str
) -> tuple[tuple[int, ...], tuple[int, ...], str, tuple[int, ...], tuple[int, ...]]:
    if not prompt or not content:
        raise ValueError("prompt and assistant content must be non-empty")
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise RuntimeError("tokenizer lacks a chat template")
    template_hash = text_sha256(template)
    user = [{"role": "user", "content": prompt}]
    prefix = template_ids(tokenizer, torch, user, generation=True)
    empty = template_ids(
        tokenizer,
        torch,
        [*user, {"role": "assistant", "content": ""}],
        generation=False,
    )
    full = template_ids(
        tokenizer,
        torch,
        [*user, {"role": "assistant", "content": content}],
        generation=False,
    )
    prompt_ids = tuple(int(value) for value in prefix[0].tolist())
    full_ids = tuple(int(value) for value in full[0].tolist())
    n = len(prompt_ids)
    if (
        tuple(int(value) for value in empty[0, :n].tolist()) != prompt_ids
        or full_ids[:n] != prompt_ids
    ):
        raise RuntimeError("joint completion does not preserve the generation prefix")
    end = tuple(int(value) for value in empty[0, n:].tolist())
    suffix = full_ids[n:]
    if not end or len(suffix) <= len(end) or suffix[-len(end) :] != end:
        raise RuntimeError("joint completion lacks the pinned assistant end")
    return suffix[: -len(end)], end, template_hash, prompt_ids, full_ids


def build_tokenized_manifests(
    tokenizer: Any, torch: Any, payload: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_source(payload)
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    alphabet_certificates = []
    all_boundary_ids: list[int] = []
    all_leading_ids: list[int] = []
    for alphabet in payload["alphabets"]:
        rows = []
        for key, boundary_id, leading, leading_id in zip(
            alphabet["keys"],
            alphabet["assistant_boundary_token_ids"],
            alphabet["leading_space_surfaces"],
            alphabet["leading_space_token_ids"],
            strict=True,
        ):
            plain_ids = tuple(
                int(value) for value in tokenizer.encode(key, add_special_tokens=False)
            )
            spaced_ids = tuple(
                int(value)
                for value in tokenizer.encode(leading, add_special_tokens=False)
            )
            if plain_ids != (boundary_id,) or spaced_ids != (leading_id,):
                raise RuntimeError(f"standalone token certificate differs for {key}")
            if (
                tokenizer.decode(list(plain_ids)) != key
                or tokenizer.decode(list(spaced_ids)) != leading
            ):
                raise RuntimeError(f"token round trip differs for {key}")
            if boundary_id in special_ids or leading_id in special_ids:
                raise RuntimeError(f"opaque glyph uses a special token: {key}")
            rows.append(
                {
                    "surface": key,
                    "assistant_boundary_token_id": boundary_id,
                    "standalone_token_ids": list(plain_ids),
                    "standalone_decode": tokenizer.decode(list(plain_ids)),
                    "leading_space_surface": leading,
                    "leading_space_token_ids": list(spaced_ids),
                    "leading_space_decode": tokenizer.decode(list(spaced_ids)),
                    "nfkc_stable": unicodedata.normalize("NFKC", key) == key,
                    "non_special": True,
                    "visible_non_whitespace": bool(key and not key.isspace()),
                    "non_alphanumeric": not key.isalnum(),
                }
            )
            all_boundary_ids.append(boundary_id)
            all_leading_ids.append(leading_id)
        alphabet_certificates.append(
            {"alphabet_id": alphabet["alphabet_id"], "identifiers": rows}
        )
    if len(set(all_boundary_ids + all_leading_ids)) != 16:
        raise RuntimeError("opaque token IDs are not globally distinct")

    ledger_index = 0
    forward_count = 0
    backward_count = 0
    sequence_items = 0
    operation_rows: list[dict[str, Any]] = []
    joint_checks: list[dict[str, Any]] = []
    for ordinal, row in enumerate(build_identifier_plan(payload), start=1):
        evidence = {}
        prompt_ids_reference: tuple[int, ...] | None = None
        alphabet = alphabet_by_id(payload, row["alphabet_id"])
        for key, expected_id in zip(
            row["keys"], alphabet["assistant_boundary_token_ids"], strict=True
        ):
            content, end, template_hash, prompt_ids, _ = assistant_content_token_ids(
                tokenizer, torch, row["prompt"], key
            )
            if content != (expected_id,) or end != ASSISTANT_END_TOKEN_IDS:
                raise RuntimeError(
                    f"joint identifier token certificate differs in {row['objective_id']}"
                )
            if template_hash != CHAT_TEMPLATE_SHA256:
                raise RuntimeError("Qwen chat template hash differs")
            if prompt_ids_reference is not None and prompt_ids != prompt_ids_reference:
                raise RuntimeError("identifier joint checks do not share a prompt prefix")
            prompt_ids_reference = prompt_ids
            evidence[str(key)] = list(content)
            joint_checks.append(
                {
                    "objective_id": row["objective_id"],
                    "surface": key,
                    "content_token_ids": list(content),
                    "assistant_end_token_ids": list(end),
                    "prompt_token_ids_sha256": canonical_sha256(list(prompt_ids)),
                }
            )
        if prompt_ids_reference is None:
            raise AssertionError("identifier row has no key evidence")
        events = []
        for kind in ("forward", "backward"):
            ledger_index += 1
            if kind == "forward":
                forward_count += 1
            else:
                backward_count += 1
            events.append(
                {
                    "ledger_index": ledger_index,
                    "event_id": f"facfs_g2_event_{ledger_index:04d}",
                    "kind": kind,
                }
            )
        sequence_items += 1
        operation_rows.append(
            {
                **plain(row),
                "ordinal": ordinal,
                "prompt_token_count": len(prompt_ids_reference),
                "prompt_token_ids_sha256": canonical_sha256(
                    list(prompt_ids_reference)
                ),
                "joint_identifier_content_token_ids": evidence,
                "assistant_end_token_ids": list(ASSISTANT_END_TOKEN_IDS),
                "ledger_events": events,
                "cumulative_counts_after": {
                    "forwards": forward_count,
                    "backwards": backward_count,
                    "sequence_items": sequence_items,
                },
            }
        )

    completion_checks: list[dict[str, Any]] = []
    for row in build_option_free_plan(payload):
        encodings = {}
        prefix_reference: tuple[int, ...] | None = None
        for semantic, completion in (
            ("preserve", row["preserve_completion"]),
            ("comply", row["comply_completion"]),
        ):
            content, end, template_hash, prompt_ids, full_ids = (
                assistant_content_token_ids(tokenizer, torch, row["prompt"], completion)
            )
            if end != ASSISTANT_END_TOKEN_IDS or template_hash != CHAT_TEMPLATE_SHA256:
                raise RuntimeError("option-free joint token certificate differs")
            if prefix_reference is not None and prompt_ids != prefix_reference:
                raise RuntimeError(
                    "option-free completions do not share one prompt prefix"
                )
            prefix_reference = prompt_ids
            encodings[semantic] = {
                "sequence_id": row["sequence_ids"][
                    0 if semantic == "preserve" else 1
                ],
                "completion_sha256": row[f"{semantic}_completion_sha256"],
                "full_token_count": len(full_ids),
                "full_token_ids_sha256": canonical_sha256(list(full_ids)),
                "content_token_count": len(content),
                "content_token_ids": list(content),
                "content_token_ids_sha256": canonical_sha256(list(content)),
                "assistant_end_token_ids": list(end),
            }
            completion_checks.append(
                {
                    "objective_id": row["objective_id"],
                    "semantic": semantic,
                    "content_token_ids": list(content),
                    "assistant_end_token_ids": list(end),
                    "prompt_token_ids_sha256": canonical_sha256(list(prompt_ids)),
                }
            )
        if prefix_reference is None:
            raise AssertionError("option-free row has no completion evidence")
        events = []
        for kind, role in (
            ("forward", "prompt_check"),
            ("forward", "preserve"),
            ("backward", "preserve"),
            ("forward", "comply"),
            ("backward", "comply"),
        ):
            ledger_index += 1
            if kind == "forward":
                forward_count += 1
            else:
                backward_count += 1
            events.append(
                {
                    "ledger_index": ledger_index,
                    "event_id": f"facfs_g2_event_{ledger_index:04d}",
                    "kind": kind,
                    "role": role,
                }
            )
        sequence_items += 2
        operation_rows.append(
            {
                **plain(row),
                "ordinal": len(operation_rows) + 1,
                "prompt_token_count": len(prefix_reference),
                "prompt_token_ids_sha256": canonical_sha256(list(prefix_reference)),
                "completion_encodings": encodings,
                "length_rule": payload["option_free_length_rule"],
                "ledger_events": events,
                "cumulative_counts_after": {
                    "forwards": forward_count,
                    "backwards": backward_count,
                    "sequence_items": sequence_items,
                },
            }
        )
    totals = {
        "scenario_count": 11,
        "opaque_objectives": 1408,
        "option_free_objectives": 22,
        "total_objectives": 1430,
        "opaque_sequence_items": 1408,
        "option_free_completion_sequence_items": 44,
        "total_scored_sequence_items": 1452,
        "prompt_only_causal_check_forwards": 22,
        "physical_forward_invocations": 1474,
        "physical_backward_invocations": 1452,
        "hash_chained_ledger_events": 2926,
        "generated_tokens": 0,
    }
    observed = (
        len(operation_rows),
        sequence_items,
        forward_count,
        backward_count,
        ledger_index,
    )
    if observed != (1430, 1452, 1474, 1452, 2926):
        raise RuntimeError(f"Stage-G operation totals differ: {observed}")
    operations = with_identity_hash(
        {
            "schema_version": OPERATIONS_SCHEMA,
            "namespace": NAMESPACE,
            "execution_order": (
                "stored_order_sequential_batch_size_one_no_adaptive_rebatching"
            ),
            "layer_zero_based": 10,
            "position": "final_prompt_token",
            "gradient_shape": [1024],
            "finite_intervention_calls": 0,
            "hook_kind": "capture_only_zero_reconstruction",
            "generated_tokens": 0,
            "totals": totals,
            "operations": operation_rows,
        },
        "operations_sha256",
    )
    token_certificate = with_identity_hash(
        {
            "schema_version": TOKEN_CERTIFICATE_SCHEMA,
            "namespace": NAMESPACE,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "assistant_end_token_ids": list(ASSISTANT_END_TOKEN_IDS),
            "alphabet_certificates": alphabet_certificates,
            "joint_identifier_check_count": len(joint_checks),
            "joint_identifier_evidence_sha256": canonical_sha256(joint_checks),
            "option_free_completion_check_count": len(completion_checks),
            "option_free_completion_evidence_sha256": canonical_sha256(
                completion_checks
            ),
            "assistant_boundary_uses_plain_not_leading_space_token_ids": True,
            "tokenizer_model_id": "Qwen/Qwen3.5-0.8B",
            "tokenizer_revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "local_files_only": True,
        },
        "token_certificate_sha256",
    )
    return operations, token_certificate


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _pair(value: Any, *, field: str, item_type: type) -> tuple[Any, Any]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise ValueError(f"{field} must contain exactly two items")
    if any(type(item) is not item_type for item in value):
        raise TypeError(f"{field} items have the wrong type")
    return value[0], value[1]


def finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result
