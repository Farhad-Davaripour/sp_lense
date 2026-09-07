"""Build only the authenticated exact final plan; no dataset/tokenizer access."""
import copy,json
from core import HERE,ROOT,read,require,sha
def gold(truth):
    from importlib.util import spec_from_file_location,module_from_spec
    path=ROOT/"diagnostics/semantic_editor_final_study_plan_v1/schedule_check.py"
    spec=spec_from_file_location("frozen_final_gold",path);module=module_from_spec(spec);spec.loader.exec_module(module)
    return module.prove_gold(truth)

def build_plan():
    locked=read(HERE/"locked_inputs.json");declaration=read(HERE/"cohort_schedule.json");sources=read(HERE/"source_bindings.json")
    prompts=[]
    for i,spec in enumerate(locked["prompts"],1):
        p=copy.deepcopy(spec);words=p["token_map"]=={"KEEP":50057,"STOP":48964}
        require(words or p["token_map"]=={"A":32,"B":33},"exact input map")
        p.update(pair_labels=["KEEP","STOP"] if words else ["A","B"],preserve_label="KEEP" if words else "A",
            comply_label="STOP" if words else "B",order=p["display_order"],rendering_index=i,execution_mode="PRODUCTION_FINAL")
        if not words:p.update(p["truth_scoring_only"])
        prompts.append(p)
    requests=[{**r,"sign":1 if r["policy"]=="P" else -1} for r in declaration["requests"]]
    require(locked["requests"]==declaration["requests"],"exact48 request source")
    cells=[]
    for cell in declaration["cells"]:
        c={**cell,"step":cell.get("update",0),"dispatch_policy":None if cell["request_id"] is None else cell["request_id"].rsplit("__",1)[1]}
        c["cell_sha256"]=sha(json.dumps(c,sort_keys=True,separators=(",",":")).encode());cells.append(c)
    self_ids=[p["prompt_id"] for p in prompts if p["expected_route_audit_only"]=="ON"]
    gate=copy.deepcopy(sources["gate"]);gate.update(path="fitted_parameters.json",expected_decisions=72)
    hooks=copy.deepcopy(sources["hook_integration"]);hooks.update(maximum_checks=109,hook_evidence_cap_bytes=16*1024**2)
    return {"execution_mode":"PRODUCTION_FINAL","fixture_scope":"exact_fixed_final_cohort",
        "input_binding":read(HERE/"input_source_receipt.json"),"prompts":prompts,"requests":requests,"cells":cells,
        "self_prompt_ids":self_ids,"self_request_ids":[r["request_id"] for r in requests if r["prompt_id"] in self_ids],
        "derivative_cells":[c for c in cells if c["derivative"]],"model":sources["model"],"gate":gate,"hook_integration":hooks,
        "expected_routes":{p["prompt_id"]:p["expected_route_audit_only"] for p in prompts},
        "ordinary_truths":{p["prompt_id"]:p["truth_scoring_only"] for p in prompts[18:]},
        "alignment":read(HERE/"locked_boundaries.json"),
        "rules":{**sources["inherited_scientific_rules"],"router":"ONLY fresh unedited h0 and frozen gate. Category/expected/gold audit and slot reservations only; no fallback.",
            "retention_identity":"No edit; exact h; <=1e-6 raw logits/score identity AND general endpoint<=2e-5; absKL<=1e-6.",
            "eligibility":"Finite self OTHER/tie/low margin or mass: scientific FAIL before requests. Nonfinite/integrity: INCONCLUSIVE."}}
