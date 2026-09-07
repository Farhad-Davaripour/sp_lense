"""Read-only exact committed v2 input admission; no encoding/model imports."""
import json
import struct
from support import HERE, SOURCES, require, sha

V2_COMMIT = "01de65cc93b128d605ee1d66b8ccaebc6f6eda40"
V2 = "diagnostics/fresh_confirmation_token_binding_v2/"
LOCK_SHA = "d008fb7c53b974694e76afa9ef7db3f4f2d4443eed8a2c83ba06ea6e83987b22"
INVENTORY_SHA = "17dd0f806ff293e9af73eb677a7f9f360a90fa896bcd15c1fc56b2c76eab40f7"


def exact_record(record, default_commit=V2_COMMIT, prefix=""):
    raw = SOURCES.read(record.get("commit",default_commit),prefix+record["path"])
    require(sha(raw) == record["sha256"] and ("bytes" not in record or len(raw) == record["bytes"]), "exact input artifact hash/length")
    return raw


def admit():
    raw = SOURCES.read(V2_COMMIT,V2+"INPUT_LOCK.json")
    require(sha(raw) == LOCK_SHA, "required complete committed v2 input lock")
    lock = json.loads(raw)
    require(lock["status"] == "TOKEN_BOUND_MODEL_NOT_AUTHORIZED" and lock["model_execution_authorized"] is False
            and lock["model_calls"] == 0 and (lock["prompt_count"],lock["request_count"],lock["semantic_count"],lock["ordinary_count"]) == (24,48,18,6), "complete disabled model lock")
    inventory_raw = SOURCES.read(V2_COMMIT,V2+"FINAL_INVENTORY.json")
    require(sha(inventory_raw) == INVENTORY_SHA, "committed v2 inventory hash")
    inventory = json.loads(inventory_raw)
    require(inventory["status"] == "COMPLETE_TOKEN_BINDING" and inventory["writer_completed_before_inventory"] is True
        and inventory["input_lock_sha256"] == LOCK_SHA and inventory["file_count"] == 43, "complete committed v2 inventory")
    members = {r["path"]:r for r in inventory["files"]}
    require(members["INPUT_LOCK.json"]["sha256"] == LOCK_SHA, "lock inventory membership")
    for record in [lock["tokenizer_receipt"],*lock["token_records"]]:
        require(record == members[record["path"]], "token/receipt exact inventory membership")
    require(sha(SOURCES.read(V2_COMMIT,V2+"TOKENIZER_BINDINGS.json")) == lock["tokenizer_bindings_sha256"]
        and sha(SOURCES.read(V2_COMMIT,V2+"TOKENIZER_SOURCE_FREEZE.json")) == lock["tokenizer_source_freeze_sha256"], "frozen tokenizer-source provenance, never tokenizer execution")
    text_raw = exact_record(lock["text_lock"])
    text_lock = json.loads(text_raw)
    inputs_raw = exact_record(lock["exact_inputs"])
    inputs = json.loads(inputs_raw)
    text_records = {r["path"]:r for r in text_lock["files"]}
    require(text_records["EXACT_INPUTS.json"]["sha256"] == sha(inputs_raw), "text-lock input membership")
    require(inputs["model_execution_authorized"] is False and len(inputs["prompts"]) == 24 and len(inputs["requests"]) == 48,
            "complete unchanged admitted text")
    receipt = json.loads(exact_record(lock["tokenizer_receipt"],prefix=V2))
    require(receipt["status"] == "TOKEN_BOUND" and receipt["unique_prompts_bound"] == 24 and receipt["failure"] is None
        and receipt["model_loads"] == receipt["model_forwards"] == receipt["model_derivatives"] == 0
        and receipt["maximum_encoded_length"] == lock["maximum_encoded_length"] <= 160, "completed tokenizer-only prerequisite")
    require(len(lock["token_records"]) == 24, "no partial proof substitute")
    proofs = []
    for index,(prompt,record) in enumerate(zip(inputs["prompts"],lock["token_records"],strict=True),1):
        require(record["path"] == f"tokens_{index:02d}.json", "exact token proof order")
        proof = json.loads(exact_record(record,prefix=V2))
        ids,n = proof["full_token_ids"],proof["prompt_length"]
        token_map = {"A":32,"B":33} if prompt["category"] == "ordinary_accuracy" else {"KEEP":50057,"STOP":48964}
        require(proof["prompt_id"] == prompt["prompt_id"] and proof["prompt_sha256"] == proof["source_prompt_sha256"] == prompt["prompt_sha256"] == sha(prompt["prompt"].encode()), "original UTF-8 prompt identity")
        require(proof["text_lock_sha256"] == sha(text_raw) and proof["token_map"] == proof["content_token_ids"] == token_map
            and proof["category"] == prompt["category"], "token/text/answer-alphabet joins")
        require(0 < n == len(ids) <= 160 and all(type(v) is int and 0 <= v < 248320 for v in ids), "complete vocabulary input tokens")
        require(proof["attention_mask"] == [1]*n and proof["final_input_mask"] == [0]*(n-1)+[1]
            and proof["final_input_index"] == n-1 and proof["final_input_token_id"] == ids[-1], "exact final input boundary")
        require(proof["full_input_int64_le_sha256"] == sha(struct.pack("<"+str(n)+"q",*ids)), "locked int64 tensor byte identity")
        require(proof["chat_template_sha256"] == "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
            and proof["prefix_exact"] is True and proof["assistant_end_token_ids"] == [248046,198], "frozen chat template and appended boundaries")
        suffix = proof["generation_header_suffix_ids"]
        require(suffix and ids[-len(suffix):] == suffix and proof["generation_header_start_index"] == n-len(suffix)
            and proof["answer_content_index_if_appended"] == n, "assistant header is input, not generated answer")
        require(all(proof["full_suffix_token_ids"][label] == [token,248046,198] for label,token in token_map.items()), "each complete appended answer")
        require(proof["final_input_role"] == "assistant_generation_header_last_INPUT_token_not_generated_answer"
            and proof["answer_content_role"] == "assistant_answer_one_content_token" and prompt["prompt"].endswith("\nAnswer:"), "unchanged role and answer suffix")
        proofs.append(proof)
    require(len({p["prompt_id"] for p in inputs["prompts"]}) == 24 and len({p["full_input_int64_le_sha256"] for p in proofs}) == 24, "unique receiver inputs")
    expected = [(p["prompt_id"],policy) for p in inputs["prompts"] for policy in ("P","C")]
    require([(r["prompt_id"],r["policy"]) for r in inputs["requests"]] == expected, "exact 48 cold request order")
    for prompt in inputs["prompts"]:
        for policy in ("P","C"):
            request = inputs["requests"][expected.index((prompt["prompt_id"],policy))]
            target = None if prompt["category"] == "ordinary_accuracy" else "KEEP" if policy == "P" else "STOP"
            require(request["request_id"] == prompt["prompt_id"]+"::"+policy and request["target_word"] == target
                and request["target_position"] == (None if target is None else prompt["answer_words"].index(target)+1)
                and request["expected_route_audit_only"] == prompt["expected_route_audit_only"], "exact externally supplied request metadata")
    require(sum(p["status"] == "EXACT_VALUE_VERIFIED" for p in inputs["proof_checks"]) == 5
        and inputs["manual_proof_support"]["item_id"] == "O06"
        and lock["O06_machine_encoding"] == "UNVERIFIED"
        and inputs["manual_proof_support"]["machine_interface_status"] == "PROOF_ENCODING_UNVERIFIED", "preserve limited proof support")
    return {"binding":{"commit":V2_COMMIT,"input_lock_sha256":LOCK_SHA,"inventory_sha256":INVENTORY_SHA,
        "exact_inputs_sha256":sha(inputs_raw),"text_lock_sha256":sha(text_raw),"maximum_tokens":max(p["prompt_length"] for p in proofs)},
        "inputs":inputs,"proofs":proofs,"lock":lock}
