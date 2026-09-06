"""Four ordinary forwards with capture-only instrumentation; no editor or derivative path."""
import hashlib
import math
import time
import zlib
from pathlib import Path
from core import Budget,HOOK,require,sha
from inputs import feature_bytes,RUNTIME_KEYS
from hook_record import HookRecorder,finish_preserving_original

def parameter_digest(parameters):
    digest=hashlib.sha256()
    for p in parameters:digest.update(memoryview(p.detach().cpu().contiguous().numpy()).cast("B"))
    return digest.hexdigest()

def evaluate(plan,backend,counter,output,guard):
    from sp_lense.comparison_runtime import next_token_logits
    from word_boundary import resolve_choice_boundary
    from word_scoring import score_float32_logits
    output=Path(output);budget=Budget(output);model=backend.model
    parameters=list(model.parameters());flags=[p.requires_grad for p in parameters];versions=[p._version for p in parameters]
    initial=parameter_digest(parameters)
    require(initial==plan["runtime_compatibility"]["weight_sha256"],"new capture weights differ from archived feature weights")
    metadata=backend.metadata()
    require({k:metadata[k] for k in RUNTIME_KEYS}=={k:plan["runtime_compatibility"][k] for k in RUNTIME_KEYS},"new capture runtime differs from archived features")
    recorder=HookRecorder(model,plan,output,counter)
    (output/"logits").mkdir(exist_ok=False);(output/"rows").mkdir(exist_ok=False)
    held={};rows=[]
    def cleanup(label):
        held.clear();guard.cell=None
        if hasattr(model,"_last_hf_cache"):model._last_hf_cache=None
        status={"hook_registry_restored":recorder.inspect(model,label),"parameter_flags_restored":[p.requires_grad for p in parameters]==flags,
            "parameter_gradients_absent":all(p.grad is None for p in parameters),"parameter_versions_unchanged":[p._version for p in parameters]==versions,
            "capture_cache_empty":not held,"bridge_cache_empty":getattr(model,"_last_hf_cache",None) is None}
        require(all(status.values()),"capture cleanup/integrity")
        return status
    try:
        for index,(p,cell) in enumerate(zip(plan["prompts"],plan["cells"],strict=True),1):
            tokens=backend.encode(p["prompt"])
            require(tokens[0].tolist()==plan["alignment"][p["prompt_id"]]["full_token_ids"],"unchanged captured input prefix")
            boundary=resolve_choice_boundary(backend,p["prompt"])
            calls=0
            def capture_hook(activation,hook):
                nonlocal calls
                calls+=1
                require(calls==1 and hook.name==HOOK,"single exact capture hook")
                require(activation.dtype==backend.torch.float32 and activation.ndim==3 and activation.shape==(1,tokens.shape[-1],1024),"float32 hook shape")
                require(not backend.torch.is_grad_enabled() and not activation.requires_grad,"capture-only inference; no derivatives")
                held["h0"]=activation[0,-1].detach().cpu().clone()
                return activation
            guard.cell=cell
            with model.hooks(fwd_hooks=[(HOOK,capture_hook)]):
                logits=next_token_logits(backend,tokens)
            require(calls==1 and logits.numel()==248320 and bool(logits.isfinite().all()),"complete finite same-forward raw capture")
            h0=held["h0"].tolist();raw_feature=feature_bytes(h0)
            raw=logits.numpy().astype("<f4",copy=False).tobytes();compressed=zlib.compress(raw)
            budget.write_bytes(f"logits/{index:02d}.f32.zlib",compressed)
            numeric=score_float32_logits(backend.torch,logits,logits,choice_keep_token_id=50057,choice_stop_token_id=48964,preserve_label="KEEP")
            row={"example_id":p["example_id"],"prompt_id":p["prompt_id"],"cell_id":cell["cell_id"],"condition":"baseline",
                "prompt_sha256":p["prompt_sha256"],"prompt_length":boundary.prompt_length,"input_token_index":boundary.prompt_length-1,"boundary_sha256":boundary.evidence_sha256,
                "input_dtype":"float32","hook":HOOK,"h0":h0,"feature_sha256":sha(raw_feature),"feature_dimension":1024,"feature_byteorder":"little-endian float32",
                "capture_only":True,"hook_returns_original_activation":True,"activation_edits":0,"derivatives":0,"capture_calls":calls,
                "logits_file":f"logits/{index:02d}.f32.zlib","logits_sha256":sha(raw),"logit_count":248320,"compressed_sha256":sha(compressed),
                "descriptive_logits_only":numeric,"captured_monotonic":time.monotonic()}
            row["cleanup"]=cleanup("capture:"+cell["cell_id"])
            row["integrity_passed"]=all(row["cleanup"].values())
            budget.write(f"rows/{index:02d}.json",row);rows.append(row)
            print(f"captured {index}/4;0derivatives;0edits",flush=True)
    finally:
        def final():
            checks=cleanup("matrix-finally");final_weights=parameter_digest(parameters)
            receipt={**checks,"initial_weight_sha256":initial,"final_weight_sha256":final_weights,"weights_exact":initial==final_weights,
                "derivatives":0,"activation_edits":0,"monotonic":time.monotonic()}
            budget.write("integration_cleanup.json",receipt)
            require(receipt["weights_exact"],"capture weights changed")
        finish_preserving_original(output,"matrix-finally",final)
    return rows
