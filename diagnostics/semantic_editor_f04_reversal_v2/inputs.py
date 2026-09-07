"""Exact inherited input bytes, two constructed starts, global replay-before-update schedule."""
import copy,json
from core import HERE,read,require,sha
from start_states import SELECTION
def gold(truth):raise ValueError("No ordinary questions in this recovery control")
def build_plan():
    locked=read(HERE/"inputs.json");lock=read(HERE/"input_lock.json");source=read(HERE/"source_bindings.json")
    require(sha((HERE/"inputs.json").read_bytes())==lock["inputs_sha256"],"immutable input bytes")
    require([p["prompt_id"] for p in locked["prompts"]]==[s["prompt_id"] for s in SELECTION],"fixed pair")
    requests=[{k:s[k] for k in ("prompt_id","request_id","policy","sign","supplied_target_word","target_display_position","source_request_id")} for s in SELECTION]
    cells=[]
    def add(pid,rid,condition,step=0,optional=False):
        c={"cell_id":(rid or pid)+"__"+condition,"prompt_id":pid,"request_id":rid,"condition":condition,
            "step":step,"update":step,"optional":optional,"derivative":condition.startswith("gradient_"),
            "dispatch_policy":None if rid is None else rid.rsplit("__",1)[1]}
        c["cell_sha256"]=sha(json.dumps(c,sort_keys=True,separators=(",",":")).encode());cells.append(c)
    for p in locked["prompts"]:add(p["prompt_id"],None,"baseline")
    for s in requests:
        for phase in ("entry","start","start_replay"):add(s["prompt_id"],s["request_id"],phase)
    for s in requests:
        for k in range(1,5):
            for phase in ("gradient","step"):add(s["prompt_id"],s["request_id"],f"{phase}_{k}",k,True)
        add(s["prompt_id"],s["request_id"],"endpoint")
    alignment={}
    for item in lock["prompts"]:
        require(sha((HERE/item["path"]).read_bytes())==item["sha256"],"copied exact cached boundary, no tokenizer")
        alignment[item["prompt_id"]]=read(HERE/item["path"])
    gate=copy.deepcopy(source["gate"]);gate["expected_decisions"]=4
    return {"execution_mode":"PRODUCTION_F04","fixture_scope":"f04_constructed_start_recovery_two_first_position_requests",
        "input_binding":{"input_sha256":lock["inputs_sha256"],"input_lock_sha256":sha((HERE/"input_lock.json").read_bytes())},
        "prompts":copy.deepcopy(locked["prompts"]),"requests":requests,"cells":cells,
        "derivative_cells":[c for c in cells if c["derivative"]],"self_prompt_ids":[p["prompt_id"] for p in locked["prompts"]],
        "self_request_ids":[s["request_id"] for s in requests],"expected_routes":{p["prompt_id"]:"ON" for p in locked["prompts"]},
        "ordinary_truths":{},"alignment":alignment,"model":source["model"],"gate":gate,"hook_integration":source["hook_integration"],
        "rules":source["inherited_scientific_rules"],"limits":{"forwards":26,"derivatives":8,"loads":1,"worker_seconds":300,
            "cleanup_seconds":15,"audit_seconds":90,"bytes":96*1024**2},
        "recovery_rule":"saved planned float32 initial offset; actual old path carried without renewal; original h0 norm; both starting replays before fresh gradients"}
