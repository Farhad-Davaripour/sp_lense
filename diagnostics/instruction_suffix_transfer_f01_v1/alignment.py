"""Deterministic text/fast-token offsets, before any model load or activation."""
import json

from core import MARKER, require, sha


def select_suffix(rendered, ids, offsets):
    require(rendered.count(MARKER) == 1, "ambiguous textual anchor")
    anchor = rendered.index(MARKER)
    require(len(ids) == len(offsets), "token offsets unavailable")
    require(not any(a < anchor < b for a,b in offsets), "anchor crosses a token boundary")
    starts = [i for i,(a,b) in enumerate(offsets) if a == anchor and b > a]
    require(len(starts) == 1, "no unique token start at textual anchor")
    first = starts[0]
    require(1 <= len(ids)-first <= 256, "fixed suffix must contain1..256 tokens")
    require(all(anchor <= a < b <= len(rendered) for a,b in offsets[first:]), "ambiguous suffix token offsets")
    return {"anchor_character_offset":anchor,"first_token_index":first,
            "last_token_index":len(ids)-1,"suffix_length":len(ids)-first,
            "selected_positions":list(range(first,len(ids))),"selected_token_ids":list(ids[first:]),
            "selected_character_offsets":[[a-anchor,b-anchor] for a,b in offsets[first:]],
            "full_token_ids":list(ids),
            "full_token_ids_sha256":sha(json.dumps(list(ids),separators=(",",":"),ensure_ascii=False).encode()),
            "suffix_rendered_sha256":sha(rendered[anchor:].encode())}


def compute(prompts, prior_boundaries, snapshot, expected_template_sha):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    require(tokenizer.is_fast,"exact offset mapping requires cached fast tokenizer")
    require(sha(tokenizer.chat_template.encode())==expected_template_sha,"pinned tokenizer template")
    prior = {b["prompt_id"]:b for b in prior_boundaries}
    aligned = {}
    for prompt in prompts:
        messages=[{"role":"user","content":prompt["prompt"]}]
        rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        canonical=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True)["input_ids"]
        encoded=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
        require(encoded["input_ids"]==canonical,"offset tokenizer differs from official chat encoding")
        selected=select_suffix(rendered,canonical,encoded["offset_mapping"])
        require(selected["full_token_ids_sha256"]==prior[prompt["prompt_id"]]["prompt_prefix_token_ids_sha256"],
                "token IDs differ from authenticated prior model encoding")
        aligned[prompt["prompt_id"]]={"prompt_id":prompt["prompt_id"],"role":prompt["role"],**selected}
    for prompt in prompts:
        if prompt["role"]=="instruction_donor":
            receiver=aligned[f"r{prompt['rendering_index']}_N"]
            donor=aligned[prompt["prompt_id"]]
            require(donor["selected_token_ids"]==receiver["selected_token_ids"],"not every suffix token ID matches")
            require(donor["selected_character_offsets"]==receiver["selected_character_offsets"]
                    and donor["suffix_rendered_sha256"]==receiver["suffix_rendered_sha256"],"suffix bytes/offsets differ")
    return aligned
