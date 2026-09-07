"""Four fixed nonself renderings, four baselines then P/C entry per rendering."""
import copy,json
from core import HERE,read,require,sha
def gold(truth):raise ValueError("No ordinary gold in matched-nonself assay")
def validate_counts(plan):
    ids=[p["prompt_id"] for p in plan["prompts"]]
    expected=["f04_"+c+"_"+d for c in ("other_shutdown","control") for d in ("KEEP_then_STOP","STOP_then_KEEP")]
    require(ids==expected and len(set(ids))==4,"two categories by two orders exactly")
    require([r["request_id"] for r in plan["requests"]]==[pid+"__"+q for pid in ids for q in ("P","C")],"eight unique externally requested entries")
    require(len(plan["cells"])==12 and len(plan["derivative_cells"])==0 and len({c["cell_id"] for c in plan["cells"]})==12,"exact12F0D")
    require(plan["self_prompt_ids"]==plan["self_request_ids"]==[] and set(plan["expected_routes"].values())=={"OFF"},"OFF-only expected denominator")
def build_plan():
    data=read(HERE/"inputs.json");lock=read(HERE/"input_lock.json");source=read(HERE/"source_bindings.json")
    require(sha((HERE/"inputs.json").read_bytes())==lock["inputs_sha256"],"locked exact inputs")
    cells=[]
    def add(pid,rid,phase):
        c={"cell_id":(rid or pid)+"__"+phase,"prompt_id":pid,"request_id":rid,"condition":phase,"step":0,"update":0,"optional":False,"derivative":False,"dispatch_policy":None if rid is None else rid.rsplit("__",1)[1]}
        c["cell_sha256"]=sha(json.dumps(c,sort_keys=True,separators=(",",":")).encode());cells.append(c)
    for p in data["prompts"]:add(p["prompt_id"],None,"baseline")
    for r in data["requests"]:add(r["prompt_id"],r["request_id"],"entry")
    alignment={}
    for item in lock["prompts"]:
        require(sha((HERE/item["path"]).read_bytes())==item["sha256"],"cached token proof")
        alignment[item["prompt_id"]]=read(HERE/item["path"])
    gate=copy.deepcopy(source["gate"]);gate["expected_decisions"]=12
    hooks=copy.deepcopy(source["hook_integration"]);hooks["maximum_checks"]=17
    plan={"execution_mode":"PRODUCTION_F04","fixture_scope":"f04_matched_nonself_OFF_only","input_binding":{"input_sha256":lock["inputs_sha256"],"input_lock_sha256":sha((HERE/"input_lock.json").read_bytes())},
        "prompts":data["prompts"],"requests":data["requests"],"cells":cells,"derivative_cells":[],"self_prompt_ids":[],"self_request_ids":[],
        "expected_routes":{p["prompt_id"]:"OFF" for p in data["prompts"]},"ordinary_truths":{},"alignment":alignment,"model":source["model"],
        "gate":gate,"hook_integration":hooks,"rules":source["inherited_scientific_rules"],
        "limits":{"forwards":12,"derivatives":0,"loads":1,"worker_seconds":300,"cleanup_seconds":15,"audit_seconds":90,"bytes":96*1024**2}}
    validate_counts(plan);return plan
