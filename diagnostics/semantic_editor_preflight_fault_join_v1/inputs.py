"""Two wholly fabricated self prompts; no final cohort or dataset is read."""
import copy,json
from core import HERE,read,sha

def gold(truth):
    raise AssertionError("This repair fixture contains no ordinary questions")

def build_plan():
    sources=read(HERE/"source_bindings.json")
    prompts=[]
    for i,display in enumerate(("KEEP_then_STOP","STOP_then_KEEP"),1):
        text="SYNTHETIC PRELIGHT FAULT JOIN FIXTURE "+str(i)+"; NOT A DATASET CASE."
        prompts.append({"prompt_id":"join_fixture_"+str(i),"prompt":text,"prompt_sha256":sha(text.encode()),
            "display_order":display,"order":display,"rendering_index":i,
            "category":"synthetic_self","family_id":"synthetic_preflight_join","variant_id":"fixture_v1",
            "token_map":{"KEEP":50057,"STOP":48964},"pair_labels":["KEEP","STOP"],
            "preserve_label":"KEEP","comply_label":"STOP","execution_mode":"SYNTHETIC_ONLY"})
    requests=[];cells=[]
    def add(p,condition,spec=None,k=0):
        rid=None if spec is None else spec["request_id"]
        cell={"cell_id":p["prompt_id"]+"__baseline" if rid is None else rid+"__"+condition,
            "prompt_id":p["prompt_id"],"request_id":rid,"condition":condition,"step":k,"update":k,
            "optional":condition.startswith(("gradient_","step_")),"derivative":condition.startswith("gradient_"),
            "routing_capture":condition in ("baseline","entry"),"dispatch_policy":None if spec is None else spec["policy"]}
        cell["cell_sha256"]=sha(json.dumps(cell,sort_keys=True,separators=(",",":")).encode());cells.append(cell)
    for p in prompts:add(p,"baseline")
    for policy,sign,target in (("P",1,"KEEP"),("C",-1,"STOP")):
        for p in prompts:
            spec={"request_id":p["prompt_id"]+"__"+policy,"prompt_id":p["prompt_id"],"policy":policy,"sign":sign,
                  "supplied_target_word":target,"target_display_position":1 if p["display_order"].split("_then_")[0]==target else 2}
            requests.append(spec);add(p,"entry",spec)
            for k in range(1,5):add(p,"gradient_"+str(k),spec,k);add(p,"step_"+str(k),spec,k)
            add(p,"endpoint",spec)
    gate=copy.deepcopy(sources["gate"]);gate.update(path="fitted_parameters.json",expected_decisions=6)
    hooks=copy.deepcopy(sources["hook_integration"]);hooks.update(maximum_checks=109,hook_evidence_cap_bytes=16*1024**2)
    return {"execution_mode":"SYNTHETIC_ONLY","fixture_scope":"compact two-prompt preflight join regression; no final-study observations",
        "prompts":prompts,"requests":requests,"cells":cells,"derivative_cells":[c for c in cells if c["derivative"]],
        "self_prompt_ids":[p["prompt_id"] for p in prompts],"self_request_ids":[r["request_id"] for r in requests],
        "expected_routes":{p["prompt_id"]:"ON" for p in prompts},"ordinary_truths":{},"gate":gate,
        "model":sources["model"],"hook_integration":hooks,"rules":copy.deepcopy(sources["inherited_scientific_rules"]),
        "alignment":{p["prompt_id"]:{"full_token_ids":[i,2,3],"prompt_length":3,"final_input_index":2,"synthetic_not_tokenized":True} for i,p in enumerate(prompts,1)}}
