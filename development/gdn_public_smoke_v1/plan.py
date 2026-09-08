"""One public baseline, zero study inputs, requests or labels."""
import struct
from support import sha
def build_plan():
    from admission import admit
    value=admit();p=value["input"];name=p["prompt_id"]
    prompt={k:p[k] for k in ("prompt_id","input_ids","attention_mask","final_input_index")}
    prompt.update(input_int64_le_sha256=sha(struct.pack("<q",1)),token_proof_sha256=value["binding"]["input_lock_sha256"])
    return {"prompts":[prompt],"requests":[],"cells":[{"cell_id":name+"__baseline","prompt_id":name,"request_id":None,"phase":"baseline"}]}
