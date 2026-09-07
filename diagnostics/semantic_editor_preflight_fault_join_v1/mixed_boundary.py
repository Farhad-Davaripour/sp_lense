"""Per-input exact boundary dispatch; no model forward and no global label fallback."""
from core import require
from mixed_scoring import kind
class PairBoundary:
    def __init__(self,inner,token_map):
        self.inner=inner
        self.token_map=dict(token_map)
        labels=("KEEP","STOP") if kind(token_map)=="words" else ("A","B")
        require(inner.evidence_record()["content_token_ids"]==token_map,"explicit boundary token map")
        self.first_token_id,self.second_token_id=(token_map[x] for x in labels)
        self.prompt_length=inner.prompt_length
        self.evidence_sha256=inner.evidence_sha256
    def token_id(self,label):
        require(label in self.token_map,"label absent from explicit per-input map")
        return self.token_map[label]
    def evidence_record(self):return self.inner.evidence_record()
def resolve_choice_boundary(backend,prompt,token_map):
    if kind(token_map)=="words":
        from word_boundary import resolve_choice_boundary as resolve
    else:
        from sp_lense.comparison_runtime import resolve_choice_boundary as resolve
    return PairBoundary(resolve(backend,prompt),token_map)
def tokenizer_boundary(tokenizer,torch,prompt,token_map):
    from sp_lense.comparison_runtime import _template_token_tensor,_resolve_choice_boundary_from_tokenizer
    tokens=_template_token_tensor(tokenizer,torch,[{"role":"user","content":prompt}],add_generation_prompt=True,device="cpu")
    if kind(token_map)=="words":
        from word_boundary import WordBoundary,boundary
        inner=WordBoundary(boundary(tokenizer,torch,prompt,expected=tokens))
    else:
        inner=_resolve_choice_boundary_from_tokenizer(tokenizer,torch,prompt,device="cpu",expected_prompt_tokens=tokens)
    return PairBoundary(inner,token_map),tokens[0].tolist()
