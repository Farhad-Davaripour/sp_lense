"""Pure renderer and token-boundary contract for frozen pre-option capture.

This module performs no file, tokenizer, model, tensor, capture, or fitting I/O.
It converts one already-admitted logical case into two retained-label prompt
views and binds caller-supplied token IDs at their true longest common prefix.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct

MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
CHECKPOINT = f"Qwen/Qwen3.5-0.8B@{MODEL_REVISION}"
NATIVE_TARGET = "model.language_model.layers.10"
WIDTH = 1024
MAX_VOCAB_ID = 248320
MAX_VIEW_TOKENS = 320
LAST_SHARED_ID = 198
LABEL_TOKEN_IDS = frozenset((32, 33, 50057, 48964))
ALLOWED_SPLITS = frozenset(("TRAIN", "VALIDATION", "HOLDOUT"))
ALLOWED_CLASSES = frozenset(("SELF", "OTHER", "NONTERMINATION", "ORDINARY"))
ADMITTED_STATUS_BY_SPLIT = {
    "TRAIN": "ADMITTED_TRAIN_TEXT_ONLY",
    "VALIDATION": "ADMITTED_VALIDATION_TEXT_ONLY",
    "HOLDOUT": "ADMITTED_HOLDOUT_TEXT_ONLY",
}

FEATURE_CONTRACT = {
    "checkpoint": CHECKPOINT,
    "native_target": NATIVE_TARGET,
    "position": "last_shared_preoption_input",
    "selector": "retained_exact_option_record_swap_lcp_v1",
    "residual_dtype": "float32",
    "width": WIDTH,
    "final_logit_position": "final_input",
    "pair_average": "0.5*float64(hcanonical0)+0.5*float64(hcanonical1)",
}

RENDER_SCHEMA = "native_capture_render.v1"
BINDING_SCHEMA = "native_capture_binding.v1"
INFERENCE_SCHEMA = "native_capture_inference_input.v1"

_REQUIRED_CASE_KEYS = frozenset(
    (
        "case_id",
        "group_id",
        "split",
        "class_label",
        "mechanism_ancestry",
        "template_ancestry",
        "context_before_options",
        "options",
        "status",
        "development_fold",
    )
)
_FIELD_NAME_PATTERN = re.compile(
    r"(?i)(?:case_id|group_id|class_label|template_ancestry|"
    r"mechanism_ancestry|development_fold|correct_option_index|gold_label)"
)
_METADATA_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(?:status|split|outcome|answer|sidecar|label|target|score)\s*[:=]"
)
_CLASS_TOKEN_PATTERN = re.compile(r"\b(?:SELF|OTHER|NONTERMINATION|ORDINARY)\b")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class NativeCaptureContractError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


def _need(condition, code: str, detail: str = ""):
    if not condition:
        raise NativeCaptureContractError(code, detail)


def _text(value, code: str, maximum: int):
    _need(type(value) is str and value == value.strip(), code)
    _need(0 < len(value) <= maximum and "\x00" not in value and "\r" not in value, code)
    return value


def canonical_json(value):
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NativeCaptureContractError("CANONICAL_JSON", str(exc)) from exc


def sha256_hex(raw):
    _need(type(raw) is bytes, "SHA256_INPUT")
    return hashlib.sha256(raw).hexdigest()


def canonical_sha256(value):
    return sha256_hex(canonical_json(value))


def int_ids_sha256(ids):
    validate_token_ids(ids)
    return hashlib.sha256(struct.pack("<" + "q" * len(ids), *ids)).hexdigest()


def validate_token_ids(ids):
    _need(type(ids) is list and 1 <= len(ids) <= MAX_VIEW_TOKENS, "TOKEN_IDS")
    _need(all(type(value) is int and 0 <= value < MAX_VOCAB_ID for value in ids), "TOKEN_IDS")
    return ids


def _prompt_has_metadata(text, case):
    folded = text.casefold()
    identifiers = (
        case["case_id"],
        case["group_id"],
        case["template_ancestry"],
        case["mechanism_ancestry"],
    )
    return (
        any(identifier.casefold() in folded for identifier in identifiers)
        or _FIELD_NAME_PATTERN.search(text) is not None
        or _METADATA_ASSIGNMENT_PATTERN.search(text) is not None
        or _CLASS_TOKEN_PATTERN.search(text) is not None
    )


def validate_case(case):
    _need(type(case) is dict, "CASE_TYPE")
    keys = frozenset(case)
    _need(_REQUIRED_CASE_KEYS <= keys, "CASE_KEYS")
    _text(case["case_id"], "CASE_ID", 128)
    _text(case["group_id"], "GROUP_ID", 32)
    _text(case["class_label"], "CLASS_LABEL", 32)
    _text(case["mechanism_ancestry"], "MECHANISM", 128)
    _text(case["template_ancestry"], "TEMPLATE", 128)
    _need(case["split"] in ALLOWED_SPLITS, "SPLIT")
    _need(case["class_label"] in ALLOWED_CLASSES, "CLASS_LABEL")
    if case["split"] == "TRAIN":
        _need(type(case["development_fold"]) is int and 0 <= case["development_fold"] <= 4, "FOLD")
    else:
        _need(case["development_fold"] is None, "FOLD")
    _need(case["status"] == ADMITTED_STATUS_BY_SPLIT[case["split"]], "STATUS")
    context = _text(case["context_before_options"], "CONTEXT", 4096)
    _need(45 <= len(re.findall(r"\S+", context)) <= 90, "CONTEXT_WORDS")
    options = case["options"]
    _need(type(options) is list and len(options) == 2, "OPTIONS")
    _need(all(type(value) is str and value == value.strip() and 0 < len(value) <= 1024 for value in options), "OPTIONS")
    _need(options[0] != options[1], "OPTIONS")
    for prompt_part in (context, options[0], options[1]):
        _need(not _prompt_has_metadata(prompt_part, case), "PROMPT_METADATA")
    return case


def render_case_views(case):
    validate_case(case)
    context = case["context_before_options"]
    first, second = case["options"]
    prompt_ab = f"{context}\nA) {first}\nB) {second}\n"
    prompt_ba = f"{context}\nB) {second}\nA) {first}\n"
    _need(prompt_ab != prompt_ba, "IDENTICAL_VIEWS")
    return {
        "schema": RENDER_SCHEMA,
        "case_id": case["case_id"],
        "split": case["split"],
        "views": {
            "AB": {"order": ["A", "B"], "prompt_text": prompt_ab},
            "BA": {"order": ["B", "A"], "prompt_text": prompt_ba},
        },
        "feature_contract": dict(FEATURE_CONTRACT),
    }


def _lcp_length(left, right):
    index = 0
    limit = min(len(left), len(right))
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _validate_rendered(rendered):
    _need(type(rendered) is dict, "RENDERED")
    _need(set(rendered) == {"schema", "case_id", "split", "views", "feature_contract"}, "RENDERED")
    _need(rendered["schema"] == RENDER_SCHEMA, "RENDERED")
    _need(rendered["split"] in ALLOWED_SPLITS, "RENDERED")
    _need(rendered["feature_contract"] == FEATURE_CONTRACT, "FEATURE_CONTRACT")
    _need(set(rendered["views"]) == {"AB", "BA"}, "RENDERED")
    for key, order in (("AB", ["A", "B"]), ("BA", ["B", "A"])):
        view = rendered["views"][key]
        _need(set(view) == {"order", "prompt_text"}, "RENDERED_VIEW")
        _need(view["order"] == order and type(view["prompt_text"]) is str, "RENDERED_VIEW")
    _need(rendered["views"]["AB"]["prompt_text"] != rendered["views"]["BA"]["prompt_text"], "IDENTICAL_VIEWS")
    return rendered


def bind_rendered_views(rendered, input_ids, label_token_ids):
    _validate_rendered(rendered)
    _need(type(input_ids) is dict and set(input_ids) == {"AB", "BA"}, "INPUT_IDS")
    _need(type(label_token_ids) is dict and set(label_token_ids) == {"A", "B"}, "LABEL_TOKEN_IDS")
    _need(all(type(value) is int and value in LABEL_TOKEN_IDS for value in label_token_ids.values()), "LABEL_TOKEN_IDS")
    _need(label_token_ids["A"] != label_token_ids["B"], "LABEL_TOKEN_IDS")
    ab = list(validate_token_ids(input_ids["AB"]))
    ba = list(validate_token_ids(input_ids["BA"]))
    _need(ab != ba, "IDENTICAL_VIEWS")
    shared = _lcp_length(ab, ba)
    _need(1 <= shared < len(ab) and shared < len(ba), "SHARED_PREFIX")
    readout_index = shared - 1
    _need(ab[readout_index] == ba[readout_index] == LAST_SHARED_ID, "LAST_SHARED_ID")
    _need(ab[shared] == label_token_ids["A"], "AB_LABEL_BOUNDARY")
    _need(ba[shared] == label_token_ids["B"], "BA_LABEL_BOUNDARY")
    prefix_hash = int_ids_sha256(ab[:shared])
    _need(prefix_hash == int_ids_sha256(ba[:shared]), "SHARED_PREFIX_HASH")
    bindings = {}
    for key, ids, label in (("AB", ab, "A"), ("BA", ba, "B")):
        bindings[key] = {
            "schema": BINDING_SCHEMA,
            "case_id": rendered["case_id"],
            "order": list(rendered["views"][key]["order"]),
            "prompt_sha256": sha256_hex(rendered["views"][key]["prompt_text"].encode("utf-8")),
            "input_ids_sha256": int_ids_sha256(ids),
            "shared_prefix_ids_sha256": prefix_hash,
            "readout_index": readout_index,
            "shared_prefix_length": shared,
            "first_option_label_id": label_token_ids[label],
            "final_input_index": len(ids) - 1,
            "feature_contract": dict(FEATURE_CONTRACT),
        }
    return {
        "schema": BINDING_SCHEMA,
        "case_id": rendered["case_id"],
        "shared_prefix_length": shared,
        "readout_index": readout_index,
        "bindings": bindings,
        "feature_contract": dict(FEATURE_CONTRACT),
    }


def validate_binding(rendered, input_ids, binding):
    _need(type(binding) is dict and set(binding) == {"schema", "case_id", "shared_prefix_length", "readout_index", "bindings", "feature_contract"}, "BINDING")
    _need(set(binding["bindings"]) == {"AB", "BA"}, "BINDING")
    label_ids = {
        "A": binding["bindings"]["AB"]["first_option_label_id"],
        "B": binding["bindings"]["BA"]["first_option_label_id"],
    }
    expected = bind_rendered_views(rendered, input_ids, label_ids)
    _need(binding == expected, "BINDING_MISMATCH")
    return binding["readout_index"]


def _validate_binding_shape(rendered, binding):
    _need(type(binding) is dict, "BINDING")
    _need(set(binding) == {"schema", "case_id", "shared_prefix_length", "readout_index", "bindings", "feature_contract"}, "BINDING")
    _need(binding["schema"] == BINDING_SCHEMA, "BINDING")
    _need(binding["case_id"] == rendered["case_id"], "BINDING")
    _need(binding["feature_contract"] == FEATURE_CONTRACT, "FEATURE_CONTRACT")
    _need(set(binding["bindings"]) == {"AB", "BA"}, "BINDING")
    required = {
        "schema", "case_id", "order", "prompt_sha256", "input_ids_sha256",
        "shared_prefix_ids_sha256", "readout_index", "shared_prefix_length",
        "first_option_label_id", "final_input_index", "feature_contract",
    }
    for key, order in (("AB", ["A", "B"]), ("BA", ["B", "A"])):
        sidecar = binding["bindings"][key]
        _need(type(sidecar) is dict and set(sidecar) == required, "BINDING")
        _need(sidecar["schema"] == BINDING_SCHEMA, "BINDING")
        _need(sidecar["case_id"] == rendered["case_id"] and sidecar["order"] == order, "BINDING")
        _need(sidecar["feature_contract"] == FEATURE_CONTRACT, "FEATURE_CONTRACT")
        _need(sidecar["readout_index"] == binding["readout_index"], "BINDING")
        _need(sidecar["shared_prefix_length"] == binding["shared_prefix_length"], "BINDING")
        _need(sidecar["final_input_index"] > sidecar["readout_index"], "BINDING")
        _need(sidecar["first_option_label_id"] in LABEL_TOKEN_IDS, "BINDING")
        for digest in ("prompt_sha256", "input_ids_sha256", "shared_prefix_ids_sha256"):
            _need(type(sidecar[digest]) is str and _HEX64.fullmatch(sidecar[digest]) is not None, "BINDING")
        expected_prompt_hash = sha256_hex(rendered["views"][key]["prompt_text"].encode("utf-8"))
        _need(sidecar["prompt_sha256"] == expected_prompt_hash, "BINDING")


def inference_input(rendered, input_ids, binding):
    _validate_rendered(rendered)
    validate_binding(rendered, input_ids, binding)
    _validate_binding_shape(rendered, binding)
    views = []
    for key in ("AB", "BA"):
        sidecar = binding["bindings"][key]
        views.append(
            {
                "order": list(rendered["views"][key]["order"]),
                "prompt_text": rendered["views"][key]["prompt_text"],
                "readout_index": sidecar["readout_index"],
                "final_input_index": sidecar["final_input_index"],
                "input_ids_sha256": sidecar["input_ids_sha256"],
                "shared_prefix_ids_sha256": sidecar["shared_prefix_ids_sha256"],
            }
        )
    return {
        "schema": INFERENCE_SCHEMA,
        "views": views,
        "feature_contract": dict(FEATURE_CONTRACT),
    }
