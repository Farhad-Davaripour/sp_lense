"""OFF-only fresh-state adapter; no editor, edit hook, offset or derivative path."""
import sys,time
from pathlib import Path
from core import HERE,ROOT,Budget,require,sha
from capture_components import SnapshotModel,parameter_digest,DerivativeLedger as ParentDerivativeLedger
from hook_record import HookRecorder,finish_preserving_original
from learned_gate import FrozenGate,RoutingMismatch
from mixed_scoring import score_float32_logits
class EligibilityError(ValueError):pass
def require_off(row,scientific):
    if row["route"]!="OFF":
        scientific("routing",row)
        raise EligibilityError("fresh learned gate unexpectedly ON; no editor fallback")
class DerivativeLedger(ParentDerivativeLedger):
    def call(self,*args,**kwargs):raise ValueError("OFF assay prohibits every derivative")
def off_identity(row,base,logits,base_logits):
    return row["prompt_sha256"]==base["prompt_sha256"] and row["h"]==base["h"] and logits==base_logits
def evaluate(plan,backend,ledger,derivatives,output,guard):
    from sp_lense.comparison_runtime import next_token_logits
    from mixed_boundary import resolve_choice_boundary
    model=backend.model;params=list(model.parameters());flags=[p.requires_grad for p in params];versions=[p._version for p in params]
    weights=parameter_digest(params);wrapper=SnapshotModel(model,backend.torch,ledger,output,guard);backend.model=wrapper
    require(weights==plan["gate"]["runtime_compatibility"]["weight_sha256"],"whole frozen weights")
    metadata=backend.metadata()
    require(all(metadata[k]==plan["gate"]["runtime_compatibility"][k] for k in ("model_id","model_revision","device","dtype","d_model","model_layers","packages","lens")),"fixed feature runtime")
    hooks=HookRecorder(model,plan,output,ledger);gate=FrozenGate(HERE/plan["gate"]["path"],plan["gate"]["parameter_sha256"])
    bases={};rows=[]
    def clear():
        wrapper.logits=wrapper.activation=wrapper.cell=None;guard.cell=None
        if hasattr(model,"_last_hf_cache"):model._last_hf_cache=None
    def clean(label):
        flags_ok=[p.requires_grad for p in params]==flags
        return {"parameter_flags_restored":flags_ok,"parameter_gradients_absent":all(p.grad is None for p in params),
            "parameter_versions_unchanged":[p._version for p in params]==versions,"hook_registry_restored":hooks.inspect(model,label),
            "wrapper_cache_empty":wrapper.logits is None and wrapper.activation is None and wrapper.cell is None,
            "bridge_cache_empty":getattr(model,"_last_hf_cache",None) is None,"zero_derivatives":derivatives.attempts==0,"edit_hook_registrations":0}
    def scientific(kind,row):
        Budget(output).event("scientific_failures.jsonl",{"kind":kind,"cell_id":row["cell_id"],"monotonic":time.monotonic()})
    def call(p,cell):
        entry=cell["condition"]=="entry";before=None
        if entry:
            before=clean("REQUEST-entry:"+cell["cell_id"])
            require(all(v is True for k,v in before.items() if k!="edit_hook_registrations"),"cold request")
        wrapper.cell=cell;tokens=backend.encode(p["prompt"])
        require(tokens[0].tolist()==plan["alignment"][p["prompt_id"]]["full_token_ids"],"unchanged input tokens")
        boundary=resolve_choice_boundary(backend,p["prompt"],p["token_map"])
        require((boundary.first_token_id,boundary.second_token_id)==(50057,48964),"fixed semantic tokens")
        logits=next_token_logits(backend,tokens);h=wrapper.activation[0,-1].tolist()
        require(bool(logits.isfinite().all() and wrapper.activation.isfinite().all()) and len(h)==1024,"finite full capture")
        values=logits.tolist();base=bases.get(p["prompt_id"]);base_logits=logits if base is None else backend.torch.tensor(base["logits"],dtype=backend.torch.float32)
        measured=score_float32_logits(backend.torch,logits,base_logits,token_map=p["token_map"],preserve_label="KEEP")
        row={**{k:v for k,v in p.items() if k!="prompt"},**cell,**measured,"h":h,"h0":h,"cumulative_offset":[0.]*1024,
            "target_sign":0,"gradient":None,"logits_file":wrapper.logits_path,"logits_sha256":wrapper.logits_sha256,"logit_count":len(values),
            "boundary_sha256":boundary.evidence_sha256,"prompt_length":int(tokens.shape[-1]),"capture_only_instrumentation":True,
            "edit_hook_registrations":0,"derivatives":0,"baseline_cell_id":p["prompt_id"]+"__baseline"}
        decision=gate.decide(h) # Only the fresh raw vector, never category/policy/expected route.
        row["routing"]={**decision,"source_cell_id":cell["cell_id"],"source_logits_sha256":row["logits_sha256"]}
        row["route"]=decision["route"]
        clear()
        if entry:
            after=clean("REQUEST-exit:"+cell["cell_id"])
            require(before==after,"capture-only request cleanup")
            row.update(entry_before=before,entry_after=after,exact_identity=off_identity(row,base["row"],values,base["logits"]))
        require(all(p._version==v and p.grad is None for p,v in zip(params,versions)),"unchanged parameters")
        Budget(output).write(f"rows/{ledger.attempts:03d}.json",row);rows.append(row)
        Budget(output).event("routing_events.jsonl",{"cell_id":cell["cell_id"],**row["routing"]})
        if not entry:bases[p["prompt_id"]]={"row":row,"logits":values}
        else:
            require_off(row,scientific)
            if not row["exact_identity"]:scientific("off_identity",row);raise EligibilityError("finite OFF identity failure; stop")
            Budget(output).event("off_returns.jsonl",{"request_id":cell["request_id"],"entry_cell_id":cell["cell_id"],"policy":cell["dispatch_policy"],
                "logits_sha256":row["logits_sha256"],"no_additional_forward":True,"exact_identity":True})
        return row
    try:
        prompt_by_id={p["prompt_id"]:p for p in plan["prompts"]}
        for cell in plan["cells"][:4]:call(prompt_by_id[cell["prompt_id"]],cell)
        wrong=[b["row"] for b in bases.values() if b["row"]["route"]!="OFF"]
        for row in wrong:scientific("routing",row)
        if wrong:raise EligibilityError("preflight learned routing FAIL; all requests UNRUN")
        for cell in plan["cells"][4:]:call(prompt_by_id[cell["prompt_id"]],cell)
    finally:
        def finish():
            clear();backend.model=model;gate.unchanged();last=parameter_digest(params);snapshot=clean("matrix-finally")
            require(all(v is True for k,v in snapshot.items() if k!="edit_hook_registrations"),"complete capture cleanup")
            Budget(output).write("gate_final.json",{"parameters_unchanged":True,"fit_calls":0,"decisions":gate.calls,"parameter_sha256":gate.expected_sha})
            Budget(output).write("integration_cleanup.json",{"initial_weight_sha256":weights,"final_weight_sha256":last,"weights_exact":weights==last,**snapshot})
            require(weights==last,"whole weights unchanged")
        finish_preserving_original(output,"matrix-finally",finish)
    require(ledger.attempts==ledger.completed==12 and derivatives.attempts==0 and gate.calls==12,"12F0D exact schedule")
    return rows,[]
