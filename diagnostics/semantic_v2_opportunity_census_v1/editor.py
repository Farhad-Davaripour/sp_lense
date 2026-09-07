"""Baseline-only same-forward capture. No optimizer, edit hooks or request replay."""
import time
from core import HERE,Budget,require
from capture_components import SnapshotModel,parameter_digest,DerivativeLedger as ParentDerivativeLedger
from hook_record import HookRecorder,finish_preserving_original
from learned_gate import FrozenGate
from mixed_scoring import score_float32_logits
class EligibilityError(ValueError):pass # Shared runner catch compatibility, never used for finite observations.
class DerivativeLedger(ParentDerivativeLedger):
    def call(self,*args,**kwargs):raise ValueError("census prohibits every derivative")
def evaluate(plan,backend,ledger,derivatives,output,guard):
    from sp_lense.comparison_runtime import next_token_logits
    from mixed_boundary import resolve_choice_boundary
    model=backend.model;params=list(model.parameters());flags=[p.requires_grad for p in params];versions=[p._version for p in params]
    weights=parameter_digest(params);wrapper=SnapshotModel(model,backend.torch,ledger,output,guard);backend.model=wrapper
    require(weights==plan["gate"]["runtime_compatibility"]["weight_sha256"],"whole frozen weights")
    metadata=backend.metadata()
    require(all(metadata[k]==plan["gate"]["runtime_compatibility"][k] for k in ("model_id","model_revision","device","dtype","d_model","model_layers","packages","lens")),"fixed feature runtime")
    hooks=HookRecorder(model,plan,output,ledger);gate=FrozenGate(HERE/plan["gate"]["path"],plan["gate"]["parameter_sha256"])
    rows=[]
    def clear():
        wrapper.logits=wrapper.activation=wrapper.cell=None;guard.cell=None
        if hasattr(model,"_last_hf_cache"):model._last_hf_cache=None
    def clean(label):
        return {"parameter_flags_restored":[p.requires_grad for p in params]==flags,"parameter_gradients_absent":all(p.grad is None for p in params),
            "parameter_versions_unchanged":[p._version for p in params]==versions,"hook_registry_restored":hooks.inspect(model,label),
            "wrapper_cache_empty":wrapper.logits is None and wrapper.activation is None and wrapper.cell is None,
            "bridge_cache_empty":getattr(model,"_last_hf_cache",None) is None,"zero_derivatives":derivatives.attempts==0,"edit_hook_registrations":0}
    def call(p,cell):
        before=clean("CAPTURE-before:"+cell["cell_id"])
        require(all(v is True for k,v in before.items() if k!="edit_hook_registrations"),"cold baseline")
        wrapper.cell=cell;tokens=backend.encode(p["prompt"])
        require(tokens[0].tolist()==plan["alignment"][p["prompt_id"]]["full_token_ids"],"unchanged input tokens")
        boundary=resolve_choice_boundary(backend,p["prompt"],p["token_map"])
        require((boundary.first_token_id,boundary.second_token_id)==(50057,48964),"fixed semantic tokens")
        logits=next_token_logits(backend,tokens);h=wrapper.activation[0,-1].tolist()
        require(bool(logits.isfinite().all() and wrapper.activation.isfinite().all()) and len(h)==1024,"finite full capture")
        measured=score_float32_logits(backend.torch,logits,logits,token_map=p["token_map"],preserve_label="KEEP")
        row={**{k:v for k,v in p.items() if k!="prompt"},**cell,**measured,"h":h,"h0":h,"cumulative_offset":[0.]*1024,
            "target_sign":0,"gradient":None,"logits_file":wrapper.logits_path,"logits_sha256":wrapper.logits_sha256,"logit_count":logits.numel(),
            "boundary_sha256":boundary.evidence_sha256,"prompt_length":int(tokens.shape[-1]),"capture_only_instrumentation":True,
            "edit_hook_registrations":0,"derivatives":0,"baseline_cell_id":cell["cell_id"]}
        decision=gate.decide(h) # Fresh raw vector only; expected category never selects capture/filtering.
        row["routing"]={**decision,"source_cell_id":cell["cell_id"],"source_logits_sha256":row["logits_sha256"]}
        row["route"]=decision["route"];clear()
        after=clean("CAPTURE-after:"+cell["cell_id"])
        require(before==after,"capture-only cleanup")
        row.update(capture_before=before,capture_after=after)
        require(all(p._version==v and p.grad is None for p,v in zip(params,versions)),"unchanged parameters")
        Budget(output).write(f"rows/{ledger.attempts:03d}.json",row);rows.append(row)
        Budget(output).event("routing_events.jsonl",{"cell_id":cell["cell_id"],**row["routing"]})
    try:
        for p,cell in zip(plan["prompts"],plan["cells"],strict=True):call(p,cell)
    finally:
        def finish():
            clear();backend.model=model;gate.unchanged();last=parameter_digest(params);snapshot=clean("matrix-finally")
            require(all(v is True for k,v in snapshot.items() if k!="edit_hook_registrations"),"complete capture cleanup")
            Budget(output).write("gate_final.json",{"parameters_unchanged":True,"fit_calls":0,"decisions":gate.calls,"parameter_sha256":gate.expected_sha})
            Budget(output).write("integration_cleanup.json",{"initial_weight_sha256":weights,"final_weight_sha256":last,"weights_exact":weights==last,**snapshot})
            require(weights==last,"whole weights unchanged")
        finish_preserving_original(output,"matrix-finally",finish)
    require(ledger.attempts==ledger.completed==8 and derivatives.attempts==0 and gate.calls==8,"8F0D exact schedule")
    return rows,[]
