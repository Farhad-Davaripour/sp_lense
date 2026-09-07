"""Exact frozen token lookup; never calls a tokenizer, renderer or gate."""
import copy
from types import SimpleNamespace
from core import require,sha,json_bytes

class LockedBackend:
    def __init__(self,backend,plan):
        self._backend=backend;self.model=backend.model;self.torch=backend.torch
        self._records={p["prompt"]:(p,copy.deepcopy(plan["alignment"][p["prompt_id"]])) for p in plan["prompts"]}
        require(len(self._records)==len(plan["prompts"]),"no ambiguous input lookup")
    def __getattr__(self,name):return getattr(self._backend,name)
    def encode(self,text):
        require(text in self._records,"unlocked prompt forbidden before forward")
        p,b=self._records[text];require(sha(text.encode())==p["prompt_sha256"],"input bytes")
        return self.torch.tensor([b["full_token_ids"]],dtype=self.torch.int64,device="cpu")
    def locked_boundary(self,text,token_map):
        require(text in self._records,"unlocked boundary")
        p,b=self._records[text];require(token_map==p["token_map"]==b["content_token_ids"],"per-input boundary mapping")
        labels=p["pair_labels"]
        return SimpleNamespace(first_token_id=token_map[labels[0]],second_token_id=token_map[labels[1]],
            prompt_length=b["prompt_length"],evidence_sha256=sha(json_bytes(b)),
            token_id=lambda label:token_map[label],evidence_record=lambda:copy.deepcopy(b))
