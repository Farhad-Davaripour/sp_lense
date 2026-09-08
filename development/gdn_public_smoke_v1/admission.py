"""Public synthetic bytes only; no study/tokenizer admission."""
import json
from support import HERE,sha,require
def admit():
    raw=(HERE/"PUBLIC_INPUT.json").read_bytes();value=json.loads(raw)
    binding=json.loads((HERE/"BINDINGS.json").read_bytes())
    require(sha(raw)==binding["input_lock_sha256"],"public input source lock")
    require(value=={"schema":"public_synthetic_smoke_input.v1","prompt_id":"PUBLIC_SYNTHETIC_TOKEN_1","input_ids":[1],"attention_mask":[1],"final_input_index":0,"offset":"EXACT_ZERO_FLOAT32_1024","tokenizer_calls":0,"study_inputs_read":False,"scientific_pass":False},"only prospectively public token")
    return {"binding":{"input_lock_sha256":sha(raw)},"input":value}
