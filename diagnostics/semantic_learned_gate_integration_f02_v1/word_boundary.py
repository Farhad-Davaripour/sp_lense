"""Literal KEEP/STOP word boundary, preserving the authenticated parent tokenizer check."""
import json
import sys
from core import ROOT,require,sha
WORDS=("KEEP","STOP")
def boundary(tokenizer,torch,prompt,expected=None):
    sys.path.insert(0,str(ROOT/"src"))
    from sp_lense.comparison_runtime import _template_token_tensor,_decode_single_token_exact
    messages=[{"role":"user","content":prompt}]
    prefix=_template_token_tensor(tokenizer,torch,messages,add_generation_prompt=True,device="cpu")
    if expected is not None: require(torch.equal(prefix,expected),"runtime unchanged encoded prefix")
    n=int(prefix.shape[-1]); prefix_ids=prefix[0].tolist()
    full={word:_template_token_tensor(tokenizer,torch,[*messages,{"role":"assistant","content":word}],
          add_generation_prompt=False,device="cpu") for word in ("",*WORDS)}
    require(all(x.shape[-1]>n and torch.equal(x[:,:n],prefix) for x in full.values()),"exact unchanged generation prefix")
    end=full[""][0,n:].tolist()
    ids={}; suffixes={}
    for word in WORDS:
        suffix=full[word][0,n:].tolist()
        require(len(suffix)==len(end)+1 and suffix[1:]==end,f"{word} must append exactly ONE content token")
        _decode_single_token_exact(tokenizer,suffix[0],word)
        ids[word]=suffix[0];suffixes[word]=suffix
    require(ids["KEEP"]!=ids["STOP"],"distinct semantic word tokens")
    return {"full_token_ids":prefix_ids,"prompt_length":n,"final_input_index":n-1,
            "content_token_ids":ids,"assistant_end_token_ids":end,"full_suffix_token_ids":suffixes,
            "prefix_sha256":sha(json.dumps(prefix_ids,separators=(",",":")).encode()),
            "chat_template_sha256":sha(tokenizer.chat_template.encode()),"exact_generation_prefix":True,
            "exactly_one_content_token":True,"labels_decode_exactly":True}

class WordBoundary:
    def __init__(self,evidence):
        self.evidence=evidence
        self.prompt_length=evidence["prompt_length"]
        self.keep_token_id=evidence["content_token_ids"]["KEEP"]
        self.stop_token_id=evidence["content_token_ids"]["STOP"]
        self.evidence_sha256=sha(json.dumps(evidence,sort_keys=True,separators=(",",":")).encode())
    def token_id(self,label):
        require(label in ("KEEP","STOP"),"literal semantic word required")
        return self.evidence["content_token_ids"][label]
    def evidence_record(self):return self.evidence

def resolve_choice_boundary(backend,prompt):
    encoded=backend.encode(prompt)
    return WordBoundary(boundary(backend.model.tokenizer,backend.torch,prompt,expected=encoded))
