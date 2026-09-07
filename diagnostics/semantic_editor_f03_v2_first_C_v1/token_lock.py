"""Cached tokenizer-only boundary proof; no model or gate API is imported."""
import json
import sys
from core import ROOT,load_module,require,sha

TEMPLATE_SHA="273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
WORD_PATH="diagnostics/semantic_learned_gate_integration_f02_v1/word_boundary.py"
WORD_SHA="ee637bc2640dd7a8869edf01f394209165c2ed04f80a94780393a32735586625"

def validate_record(record,token_map):
    require(token_map in ({"KEEP":50057,"STOP":48964},{"A":32,"B":33}),"explicit per-input token map only")
    require(record["content_token_ids"]==token_map,"exact appended token identities")
    require(len(set(token_map.values()))==2,"distinct words/letters")
    require(record["chat_template_sha256"]==TEMPLATE_SHA,"frozen official chat template")
    tokens=record["full_token_ids"];mask=record["attention_mask"]
    require(tokens and record["prompt_length"]==len(tokens) and record["final_input_index"]==len(tokens)-1,"complete prefix dimensions")
    require(mask==[1]*len(tokens),"untruncated unpadded attention mask")
    require(record["final_input_mask"]==[0]*(len(tokens)-1)+[1],"exact last encoded INPUT mask")
    require(record["assistant_end_token_ids"]==[248046,198],"unchanged assistant end suffix")
    require(record["prefix_exact"] is True,"all joint prefixes exact")
    require(record["generation_header_suffix_ids"] and tokens[-len(record["generation_header_suffix_ids"]):]==record["generation_header_suffix_ids"],"assistant generation-header suffix")
    for label,token in token_map.items():
        require(record["full_suffix_token_ids"][label]==[token,248046,198],"one content token only, end suffix unchanged")
    return True

def prove(tokenizer,torch,prompt,token_map):
    sys.path.insert(0,str(ROOT/"src"))
    from sp_lense.comparison_runtime import _resolve_choice_boundary_from_tokenizer
    require(sha((ROOT/WORD_PATH).read_bytes())==WORD_SHA,"inherited word boundary source")
    words=load_module("final_input_word_boundary",WORD_PATH)
    messages=[{"role":"user","content":prompt}]
    arguments=dict(tokenize=True,enable_thinking=False,return_dict=True,return_tensors="pt")
    encoded=tokenizer.apply_chat_template(messages,add_generation_prompt=True,**arguments)
    prefix=encoded["input_ids"]
    no_header=tokenizer.apply_chat_template(messages,add_generation_prompt=False,**arguments)["input_ids"]
    require(no_header.shape[-1]<prefix.shape[-1] and torch.equal(prefix[:,:no_header.shape[-1]],no_header),"user transcript plus exact assistant generation header")
    if token_map=={"KEEP":50057,"STOP":48964}:
        record=words.boundary(tokenizer,torch,prompt,expected=prefix)
    else:
        require(token_map=={"A":32,"B":33},"no global-word fallback")
        record=_resolve_choice_boundary_from_tokenizer(tokenizer,torch,prompt,device="cpu",expected_prompt_tokens=prefix).evidence_record()
    tokens=prefix[0].tolist();n=len(tokens)
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    record.update({"full_token_ids":tokens,"attention_mask":encoded["attention_mask"][0].tolist(),
        "prompt_length":n,"final_input_index":n-1,"final_input_mask":[0]*(n-1)+[1],
        "prefix_exact":True,"generation_header_start_index":int(no_header.shape[-1]),
        "generation_header_suffix_ids":tokens[int(no_header.shape[-1]):],
        "final_input_token_id":tokens[-1],"final_input_token_decoded":tokenizer.decode([tokens[-1]],skip_special_tokens=False,clean_up_tokenization_spaces=False),
        "final_input_role":"assistant_generation_header_last_INPUT_token_not_generated_answer",
        "answer_content_index_if_appended":n,"answer_content_role":"assistant_answer_one_content_token",
        "rendered_chat_utf8_sha256":sha(rendered.encode()),
        "full_input_int64_le_sha256":sha(prefix.to(dtype=torch.int64).contiguous().numpy().tobytes()),
        "prompt_sha256":sha(prompt.encode()),"source_prompt_suffix":"\nAnswer:",
        "backend_encoding_call_parity":"same apply_chat_template arguments as frozen ResearchBackend.encode; no model instantiated"})
    require(prompt.endswith("\nAnswer:"),"exact prompt answer suffix")
    validate_record(record,token_map)
    return record
