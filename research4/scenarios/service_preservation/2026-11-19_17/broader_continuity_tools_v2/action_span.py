"""Exact action-plus-EOS token spans for fixed-rationale DEV likelihood rows."""

import json


SCHEMA = "sp-lense.action-token-span.v1"


def action_span(tokenizer, row, render):
    ids, prefix, meta = render(tokenizer, row)
    messages = row["messages"]
    action = messages[-1]["content"]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=True, truncation=False,
    )
    encoded = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"][:len(ids)] != ids or len(encoded["offset_mapping"]) < len(ids):
        raise ValueError("Offset tokenization differs from the native rendered IDs")
    start_character = formatted.rfind(action)
    end_character = start_character + len(action)
    if start_character < 0 or formatted.find(action, start_character + 1) >= 0:
        raise ValueError("Canonical action content is missing or ambiguous in the rendered target")
    action_positions = [index for index, (start, end) in enumerate(encoded["offset_mapping"][:len(ids)])
                        if end > start_character and start < end_character]
    if not action_positions or action_positions != list(range(action_positions[0], action_positions[-1] + 1)):
        raise ValueError("Action tokens must form one contiguous span")
    eos_position = meta["tokens_through_EOS"] - 1
    if ids[eos_position] != meta["canonical_EOS"] or action_positions[-1] >= eos_position:
        raise ValueError("Final assistant EOS must follow the complete action")
    start_token = action_positions[0]
    end_token = eos_position + 1
    if start_token < prefix or end_token > len(ids):
        raise ValueError("Action span lies outside the supervised suffix")
    decoded = tokenizer.decode(ids[action_positions[0]:action_positions[-1] + 1], skip_special_tokens=False)
    if decoded != action:
        raise ValueError("Action token span does not decode to the canonical action")
    return {
        "schema": SCHEMA,
        "id": row["id"],
        "family": row["family"],
        "full_sequence_tokens": len(ids),
        "prompt_tokens": prefix,
        "full_suffix_start_token": prefix,
        "full_suffix_end_token_exclusive": meta["tokens_through_EOS"],
        "action_start_token": start_token,
        "action_end_token_exclusive": end_token,
        "action_tokens_including_eos": end_token - start_token,
        "canonical_action_json": json.loads(action),
        "reference_reasoning_conditioned": True,
        "final_eos_included": True,
        "post_eos_structural_tokens_excluded": meta["post_EOS_structural_tokens_excluded"],
    }
