"""Exactly eight immutable baseline-only census cells; no requests or derivatives."""
import copy,json
from core import HERE,read,require,sha
from select_inputs import FAMILIES
DISPLAYS=("KEEP_then_STOP","STOP_then_KEEP")
def gold(truth):raise ValueError("No ordinary gold or targeting in this census")
def validate_inputs(data):
    expected=[f.split("_")[1]+"_v2_"+d for f in FAMILIES for d in DISPLAYS]
    require([p["prompt_id"] for p in data["prompts"]]==expected and data["requests"]==[],"exact8 IDs, no requests")
    for p,(family,display) in zip(data["prompts"],[(f,d) for f in FAMILIES for d in DISPLAYS],strict=True):
        require(p["family_id"]==family and p["variant_id"]=="v2" and p["category"]=="self_shutdown" and p["source_split"]=="discovery","fixed discovery self cases")
        require(p["source_case_id"]==family+"__v2__self_shutdown" and p["display_order"]==display,"exact identity/display")
        require(p["token_map"]=={"KEEP":50057,"STOP":48964} and p["prompt_sha256"]==sha(p["prompt"].encode()) and p["inverse_transform_exact"],"exact prompt/token semantics")
def validate_counts(plan):
    validate_inputs(plan)
    require(len(plan["cells"])==8 and len({c["cell_id"] for c in plan["cells"]})==8 and plan["derivative_cells"]==[],"exact8F0D")
    require(all(c["condition"]=="baseline" and c["request_id"] is None and c["dispatch_policy"] is None and not c["optional"] and not c["derivative"] for c in plan["cells"]),"baseline-only unmodified input")
    require([c["prompt_id"] for c in plan["cells"]]==[p["prompt_id"] for p in plan["prompts"]],"fixed baseline order")
def build_plan():
    data=read(HERE/"inputs.json");validate_inputs(data);lock=read(HERE/"input_lock.json");source=read(HERE/"source_bindings.json")
    require(sha((HERE/"inputs.json").read_bytes())==lock["inputs_sha256"],"locked exact inputs")
    cells=[]
    for p in data["prompts"]:
        c={"cell_id":p["prompt_id"]+"__baseline","prompt_id":p["prompt_id"],"request_id":None,"condition":"baseline","step":0,"update":0,"optional":False,"derivative":False,"dispatch_policy":None}
        c["cell_sha256"]=sha(json.dumps(c,sort_keys=True,separators=(",",":")).encode());cells.append(c)
    alignment={}
    for item in lock["prompts"]:
        require(sha((HERE/item["path"]).read_bytes())==item["sha256"],"cached token proof")
        alignment[item["prompt_id"]]=read(HERE/item["path"])
    gate=copy.deepcopy(source["gate"]);gate["expected_decisions"]=8
    hooks=copy.deepcopy(source["hook_integration"]);hooks["maximum_checks"]=17
    plan={"execution_mode":"PRODUCTION_V2_CENSUS","fixture_scope":"remaining_discovery_v2_baselines_only",
        "input_binding":{"input_sha256":lock["inputs_sha256"],"input_lock_sha256":sha((HERE/"input_lock.json").read_bytes())},
        "prompts":data["prompts"],"requests":[],"cells":cells,"derivative_cells":[],"self_prompt_ids":[p["prompt_id"] for p in data["prompts"]],"self_request_ids":[],
        "expected_routes":{p["prompt_id"]:"ON" for p in data["prompts"]},"ordinary_truths":{},"alignment":alignment,"model":source["model"],
        "gate":gate,"hook_integration":hooks,"rules":source["inherited_scientific_rules"],
        "later_selection":"first ON eligible natural-second row per semantic first-position target in manifest/layout order; no follow-on authorized",
        "limits":{"forwards":8,"derivatives":0,"loads":1,"worker_seconds":300,"cleanup_seconds":15,"audit_seconds":90,"bytes":96*1024**2}}
    validate_counts(plan);return plan
