"""Locked tensor-only synthetic backend; no prompt/category/policy/gold argument."""
import json
import struct
import torch
from support import HERE, require, sha


class TensorOnlyFake:
    def __init__(self, artifact_raw, mode):
        require(mode == "normal", "one success fixture only; real backend prohibited")
        fixture = json.loads((HERE/"SYNTHETIC_FIXTURE.json").read_bytes())
        require(fixture["execution_mode"] == "SYNTHETIC_ONLY" and fixture["model_predictions"] is False, "explicit fake fixture")
        params = json.loads(artifact_raw)["parameters"]
        mean,direction = torch.tensor(params["grand_mean"]),torch.tensor(params["direction"])
        self.records = {r["input_int64_le_sha256"]:r for r in fixture["records"]}
        self.states = {key:mean+r["axis_sign"]*10*direction for key,r in self.records.items()}
        self.weight = torch.nn.Parameter(torch.tensor(1.,dtype=torch.float32))
        self.active = self.cache = None
        self.dispatches = 0

    def forward_inputs(self, input_ids, attention_mask, phase, offset):
        require(type(input_ids) is list and type(attention_mask) is list and attention_mask == [1]*len(input_ids), "tensor-only receiver interface")
        tokens = torch.tensor([input_ids],dtype=torch.int64)
        mask = torch.tensor([attention_mask],dtype=torch.int64)
        raw = tokens.contiguous().numpy().astype("<i8",copy=False).tobytes()
        key = sha(raw)
        require(key in self.records and tokens[0].tolist() == self.records[key]["input_ids"]
            and mask[0].tolist() == self.records[key]["attention_mask"], "exact locked tensors, no encode fallback")
        r,base = self.records[key],self.states[key]
        h = (base+offset).detach().clone().requires_grad_(phase.startswith("gradient_"))
        z = torch.full((248320,),-100.,dtype=torch.float32)
        if r["keep_intercept"] is not None:
            z[50057] = r["keep_intercept"]+h[0]-base[0]
            z[48964] = 0.
        else:
            for token,value in r["constant_logits"]:
                z[token] = value
        self.dispatches += 1
        self.cache = (z,h)
        return z,h


def make_backend(artifact_raw, mode):
    return TensorOnlyFake(artifact_raw,mode)
