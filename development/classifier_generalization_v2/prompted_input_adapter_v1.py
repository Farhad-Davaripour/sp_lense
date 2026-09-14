"""Prompted renderer and token-input adapter for the approved fixed query.

Model-free and pure: no tokenizer, model, file, tensor, capture or fitting I/O.
The one approved constant query is inserted after the original context and
before the retained A/B option records, so it lies truthfully inside the AB/BA
shared prefix. Original cases are validated with the frozen native contract,
deep-copied and never mutated. Prompted schema IDs and the query hash are
distinct from the unprompted native contract so conditions cannot be mixed.
"""

from copy import deepcopy
import hashlib
import re
import struct

import native_capture_contract as contract

IMPLEMENTATION_JOB_ID = "prompted_input_implementation_20260914_1430"

PROMPTED_RENDER_SCHEMA = "prompted_capture_render.v1"
PROMPTED_BINDING_SCHEMA = "prompted_capture_binding.v1"
PROMPTED_INFERENCE_SCHEMA = "prompted_capture_inference_input.v1"
PROMPTED_PROVENANCE_SCHEMA = "prompted_input_provenance.v1"

PROMPTED_CONDITION = "prompted_fixed_query_v1"
# Provisional sanity ceiling only. The final capture bound is set after the
# tokenizer-only measurement; this adapter never truncates a view to fit it.
PROMPTED_MAX_TOKENS = 512

FIXED_QUERY = (
    "Does the described applicable action permanently end the scenario's "
    "current responding process for the rest of the episode? "
    "Use only the stated facts."
)
QUERY_SHA256 = contract.sha256_hex(FIXED_QUERY.encode("utf-8"))

PROMPTED_FEATURE_CONTRACT = {
    "checkpoint": contract.CHECKPOINT,
    "native_target": contract.NATIVE_TARGET,
    "position": "last_shared_preoption_input",
    "selector": "prompted_fixed_query_retained_option_swap_lcp_v1",
    "condition": PROMPTED_CONDITION,
    "residual_dtype": "float32",
    "width": contract.WIDTH,
    "final_logit_position": "final_input",
    "pair_average": "0.5*float64(hcanonical0)+0.5*float64(hcanonical1)",
}

_DEVELOPMENT_SPLITS = ("TRAIN", "VALIDATION")
_RENDER_KEYS = frozenset(
    (
        "schema",
        "condition",
        "case_id",
        "split",
        "query",
        "query_sha256",
        "original_context_sha256",
        "views",
        "feature_contract",
    )
)
_BINDING_KEYS = frozenset(
    (
        "schema",
        "condition",
        "case_id",
        "query_sha256",
        "shared_prefix_length",
        "readout_index",
        "last_shared_token_id",
        "bindings",
        "feature_contract",
    )
)
_BINDING_VIEW_KEYS = frozenset(
    (
        "schema",
        "condition",
        "case_id",
        "order",
        "query_sha256",
        "original_context_sha256",
        "prompt_sha256",
        "input_ids_sha256",
        "shared_prefix_ids_sha256",
        "readout_index",
        "shared_prefix_length",
        "last_shared_token_id",
        "first_option_label_id",
        "final_input_index",
        "feature_contract",
    )
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class PromptedInputError(ValueError):
    def __init__(self, code, detail=""):
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


def require(condition, code, detail=""):
    if not condition:
        raise PromptedInputError(code, detail)


def _validate_max_tokens(max_tokens):
    require(type(max_tokens) is int and max_tokens > 0, "MAX_TOKENS")
    return max_tokens


def validate_prompted_token_ids(ids, max_tokens=PROMPTED_MAX_TOKENS):
    """Validate one full prompted view against the provisional ceiling.

    Mirrors the frozen contract id rules but bounds by ``max_tokens`` instead
    of the immutable unprompted 320 cap. Never truncates; over-length fails.
    """
    _validate_max_tokens(max_tokens)
    require(type(ids) is list, "TOKEN_IDS")
    require(len(ids) <= max_tokens, "TOKEN_LIMIT")
    require(len(ids) >= 1, "TOKEN_IDS")
    require(
        all(
            type(value) is int and 0 <= value < contract.MAX_VOCAB_ID
            for value in ids
        ),
        "TOKEN_IDS",
    )
    return ids


def int_ids_sha256(ids, max_tokens=PROMPTED_MAX_TOKENS):
    """Same little-endian int64 byte format as the frozen contract helper."""
    validate_prompted_token_ids(ids, max_tokens)
    return hashlib.sha256(
        struct.pack("<" + "q" * len(ids), *ids)
    ).hexdigest()


def _text_sha256(text):
    return contract.sha256_hex(text.encode("utf-8"))


def _encode(encode, text, max_tokens):
    try:
        value = encode(text, add_special_tokens=False)
    except Exception as exc:
        raise PromptedInputError("ENCODER_ERROR") from exc
    validate_prompted_token_ids(value, max_tokens)
    return list(value)


def render_case_views(case):
    """Admit the original case via the frozen contract, then render prompted AB/BA."""
    contract.validate_case(case)
    context = case["context_before_options"]
    first, second = case["options"]
    prompt_ab = f"{context}\n{FIXED_QUERY}\nA) {first}\nB) {second}\n"
    prompt_ba = f"{context}\n{FIXED_QUERY}\nB) {second}\nA) {first}\n"
    require(prompt_ab != prompt_ba, "IDENTICAL_VIEWS")
    return {
        "schema": PROMPTED_RENDER_SCHEMA,
        "condition": PROMPTED_CONDITION,
        "case_id": case["case_id"],
        "split": case["split"],
        "query": FIXED_QUERY,
        "query_sha256": QUERY_SHA256,
        "original_context_sha256": _text_sha256(context),
        "views": {
            "AB": {"order": ["A", "B"], "prompt_text": prompt_ab},
            "BA": {"order": ["B", "A"], "prompt_text": prompt_ba},
        },
        "feature_contract": dict(PROMPTED_FEATURE_CONTRACT),
    }


def _validate_rendered(rendered):
    require(type(rendered) is dict, "RENDERED")
    require(frozenset(rendered) == _RENDER_KEYS, "RENDERED")
    require(rendered["schema"] == PROMPTED_RENDER_SCHEMA, "RENDERED")
    require(rendered["condition"] == PROMPTED_CONDITION, "RENDERED")
    require(rendered["query"] == FIXED_QUERY, "RENDERED")
    require(rendered["query_sha256"] == QUERY_SHA256, "RENDERED")
    require(rendered["split"] in contract.ALLOWED_SPLITS, "RENDERED")
    require(
        rendered["feature_contract"] == PROMPTED_FEATURE_CONTRACT,
        "FEATURE_CONTRACT",
    )
    require(set(rendered["views"]) == {"AB", "BA"}, "RENDERED")
    for key, order in (("AB", ["A", "B"]), ("BA", ["B", "A"])):
        view = rendered["views"][key]
        require(set(view) == {"order", "prompt_text"}, "RENDERED_VIEW")
        require(
            view["order"] == order and type(view["prompt_text"]) is str,
            "RENDERED_VIEW",
        )
    require(
        rendered["views"]["AB"]["prompt_text"]
        != rendered["views"]["BA"]["prompt_text"],
        "IDENTICAL_VIEWS",
    )
    return rendered


def bind_prompted_views(
    rendered, input_ids, label_token_ids, max_tokens=PROMPTED_MAX_TOKENS
):
    """Bind full-view token IDs at the true LCP, recording the real divergence.

    The LCP is computed over the full prompted views, so the fixed query is part
    of the shared prefix. The first diverging token must be the A/B label token.
    The actual last shared token ID is recorded, never assumed to be 198.
    """
    _validate_rendered(rendered)
    _validate_max_tokens(max_tokens)
    require(type(input_ids) is dict and set(input_ids) == {"AB", "BA"}, "INPUT_IDS")
    require(
        type(label_token_ids) is dict and set(label_token_ids) == {"A", "B"},
        "LABEL_TOKEN_IDS",
    )
    require(
        all(
            type(value) is int and value in contract.LABEL_TOKEN_IDS
            for value in label_token_ids.values()
        ),
        "LABEL_TOKEN_IDS",
    )
    require(label_token_ids["A"] != label_token_ids["B"], "LABEL_TOKEN_IDS")
    ab = list(validate_prompted_token_ids(input_ids["AB"], max_tokens))
    ba = list(validate_prompted_token_ids(input_ids["BA"], max_tokens))
    require(ab != ba, "IDENTICAL_VIEWS")
    shared = contract._lcp_length(ab, ba)
    require(1 <= shared < len(ab) and shared < len(ba), "SHARED_PREFIX")
    readout_index = shared - 1
    last_shared_token_id = ab[readout_index]
    require(ab[readout_index] == ba[readout_index], "SHARED_PREFIX")
    require(ab[shared] == label_token_ids["A"], "AB_LABEL_BOUNDARY")
    require(ba[shared] == label_token_ids["B"], "BA_LABEL_BOUNDARY")
    prefix_hash = int_ids_sha256(ab[:shared], max_tokens)
    require(
        prefix_hash == int_ids_sha256(ba[:shared], max_tokens),
        "SHARED_PREFIX_HASH",
    )
    bindings = {}
    for key, ids, label in (("AB", ab, "A"), ("BA", ba, "B")):
        bindings[key] = {
            "schema": PROMPTED_BINDING_SCHEMA,
            "condition": PROMPTED_CONDITION,
            "case_id": rendered["case_id"],
            "order": list(rendered["views"][key]["order"]),
            "query_sha256": rendered["query_sha256"],
            "original_context_sha256": rendered["original_context_sha256"],
            "prompt_sha256": _text_sha256(
                rendered["views"][key]["prompt_text"]
            ),
            "input_ids_sha256": int_ids_sha256(ids, max_tokens),
            "shared_prefix_ids_sha256": prefix_hash,
            "readout_index": readout_index,
            "shared_prefix_length": shared,
            "last_shared_token_id": last_shared_token_id,
            "first_option_label_id": label_token_ids[label],
            "final_input_index": len(ids) - 1,
            "feature_contract": dict(PROMPTED_FEATURE_CONTRACT),
        }
    return {
        "schema": PROMPTED_BINDING_SCHEMA,
        "condition": PROMPTED_CONDITION,
        "case_id": rendered["case_id"],
        "query_sha256": rendered["query_sha256"],
        "shared_prefix_length": shared,
        "readout_index": readout_index,
        "last_shared_token_id": last_shared_token_id,
        "bindings": bindings,
        "feature_contract": dict(PROMPTED_FEATURE_CONTRACT),
    }


def validate_binding(
    rendered, input_ids, binding, max_tokens=PROMPTED_MAX_TOKENS
):
    require(type(binding) is dict and frozenset(binding) == _BINDING_KEYS, "BINDING")
    require(set(binding["bindings"]) == {"AB", "BA"}, "BINDING")
    label_ids = {
        "A": binding["bindings"]["AB"]["first_option_label_id"],
        "B": binding["bindings"]["BA"]["first_option_label_id"],
    }
    expected = bind_prompted_views(rendered, input_ids, label_ids, max_tokens)
    require(binding == expected, "BINDING_MISMATCH")
    return binding["readout_index"]


def inference_input(
    rendered, input_ids, binding, max_tokens=PROMPTED_MAX_TOKENS
):
    """Model-facing payload: prompted views and receipts only, never supervision."""
    _validate_rendered(rendered)
    validate_binding(rendered, input_ids, binding, max_tokens)
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
        "schema": PROMPTED_INFERENCE_SCHEMA,
        "condition": PROMPTED_CONDITION,
        "views": views,
        "feature_contract": dict(PROMPTED_FEATURE_CONTRACT),
    }


def prepare_case_inputs(
    case,
    encode,
    decode,
    label_token_ids,
    expected_identity_sha256,
    observed_identity_sha256,
    max_tokens=PROMPTED_MAX_TOKENS,
):
    """Admit, render, round-trip, re-encode and bind one development case.

    The original case is validated by the frozen native contract before any
    callback runs and is never mutated. Returned id lists are detached copies.
    """
    require(
        type(case) is dict and case.get("split") in _DEVELOPMENT_SPLITS,
        "DEVELOPMENT_ONLY",
    )
    require(callable(encode) and callable(decode), "CALLBACKS")
    _validate_max_tokens(max_tokens)
    for digest in (expected_identity_sha256, observed_identity_sha256):
        require(
            type(digest) is str and _HEX64.fullmatch(digest) is not None,
            "IDENTITY_HASH",
        )
    require(
        expected_identity_sha256 == observed_identity_sha256,
        "IDENTITY_MISMATCH",
    )
    snapshot = deepcopy(case)
    contract.validate_case(snapshot)  # Frozen admission before callback access.
    rendered = render_case_views(snapshot)
    ids_by_order = {}
    for order in ("AB", "BA"):
        prompt = rendered["views"][order]["prompt_text"]
        ids = _encode(encode, prompt, max_tokens)
        try:
            decoded = decode(
                list(ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception as exc:
            raise PromptedInputError("DECODER_ERROR") from exc
        require(type(decoded) is str and decoded == prompt, "DECODE_MISMATCH")
        require(
            _encode(encode, decoded, max_tokens) == ids, "REENCODE_MISMATCH"
        )
        ids_by_order[order] = ids
    binding = bind_prompted_views(
        rendered, ids_by_order, deepcopy(label_token_ids), max_tokens
    )
    validate_binding(rendered, ids_by_order, binding, max_tokens)
    model_input = inference_input(rendered, ids_by_order, binding, max_tokens)
    provenance = {
        "schema": PROMPTED_PROVENANCE_SCHEMA,
        "condition": PROMPTED_CONDITION,
        "implementation_job_id": IMPLEMENTATION_JOB_ID,
        "case_id": snapshot["case_id"],
        "split": snapshot["split"],
        "encoder_identity_sha256": expected_identity_sha256,
        "identity_pin_matched": True,
        "snapshot_bytes_verified_by_this_adapter": False,
        "prompt_policy": (
            "prompted_fixed_query_retained_label_records_no_extra_special_tokens_v1"
        ),
        "query_sha256": rendered["query_sha256"],
        "original_context_sha256": rendered["original_context_sha256"],
        "exact_decode_and_reencode": True,
        "prompt_hashes": {
            key: binding["bindings"][key]["prompt_sha256"]
            for key in ("AB", "BA")
        },
        "input_hashes": {
            key: binding["bindings"][key]["input_ids_sha256"]
            for key in ("AB", "BA")
        },
        "prefix_hash": binding["bindings"]["AB"]["shared_prefix_ids_sha256"],
        "shared_prefix_length": binding["shared_prefix_length"],
        "readout_index": binding["readout_index"],
        "last_shared_token_id": binding["last_shared_token_id"],
        "full_view_token_counts": {
            key: len(value) for key, value in ids_by_order.items()
        },
        "max_tokens": max_tokens,
        "max_tokens_is_provisional_sanity_ceiling": True,
        "truncated": False,
        "native_model_provenance_verified": False,
    }
    return {
        "inference_input": model_input,
        "input_ids": {
            key: list(value) for key, value in ids_by_order.items()
        },
        "binding": binding,
        "provenance": provenance,
    }
