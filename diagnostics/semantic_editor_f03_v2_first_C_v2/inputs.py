"""One exact census-selected prompt, one C request, no tokenization or score access."""
import copy,json
from core import HERE,read,require,sha
PID="f03_v2_STOP_then_KEEP"
PROMPT_SHA="cce5f2cf04814313fcaa19774d071f7747e18d8f32c3932dc933657000868410"
TOKEN_SHA="199f63ebbc9ab6c0b95709ce144e79b45fdb683288905168adb7a1430a6f26b2"
def gold(truth):raise ValueError("No ordinary gold in this assay")
def validate_inputs(locked):
    require(len(locked["prompts"])==len(locked["requests"])==1,"exact one prompt and request")
    p=locked["prompts"][0];r=locked["requests"][0]
    require(p["prompt_id"]==PID and p["prompt_sha256"]==PROMPT_SHA==sha(p["prompt"].encode()),"unchanged selected prompt bytes")
    require(p["source_case_id"]=="cg_f03_context_rotation__v2__self_shutdown" and p["family_id"]=="cg_f03_context_rotation" and p["variant_id"]=="v2" and p["category"]=="self_shutdown" and p["source_split"]=="discovery","only selected discovery self case")
    require(p["token_map"]=={"KEEP":50057,"STOP":48964} and p["preserve_label"]=="KEEP" and p["comply_label"]=="STOP" and p["display_order"]=="STOP_then_KEEP" and p["inverse_transform_exact"],"unchanged semantic/display identity")
    require(r=={"request_id":PID+"__C","prompt_id":PID,"policy":"C","sign":-1,"supplied_target_word":"STOP","target_display_position":1},"only C/STOP first target")
    return True
def build_plan():
    locked=read(HERE/"inputs.json");validate_inputs(locked);lock=read(HERE/"input_lock.json");source=read(HERE/"source_bindings.json")
    require(sha((HERE/"inputs.json").read_bytes())==lock["inputs_sha256"],"locked input bytes")
    row=lock["prompts"][0];require(len(lock["prompts"])==1 and row["prompt_id"]==PID and row["sha256"]==TOKEN_SHA==sha((HERE/row["path"]).read_bytes()),"complete exact copied token proof")
    proof=read(HERE/row["path"]);require(proof["prompt_length"]==137 and proof["content_token_ids"]=={"KEEP":50057,"STOP":48964},"copied exact boundary")
    cells=[]
    def add(rid,phase,step=0,optional=False):
        c={"cell_id":(rid or PID)+"__"+phase,"prompt_id":PID,"request_id":rid,"condition":phase,"step":step,"update":step,"optional":optional,"derivative":phase.startswith("gradient_"),"dispatch_policy":None if rid is None else "C"}
        c["cell_sha256"]=sha(json.dumps(c,sort_keys=True,separators=(",",":")).encode());cells.append(c)
    add(None,"baseline");rid=PID+"__C";add(rid,"entry")
    for k in range(1,5):
        for phase in ("gradient","step"):add(rid,f"{phase}_{k}",k,True)
    add(rid,"endpoint")
    gate=copy.deepcopy(source["gate"]);gate["expected_decisions"]=2
    return {"execution_mode":"PRODUCTION_F03_V2_FIRST_C","fixture_scope":"exact_census_selected_f03_v2_self_STOP_first_C_request",
        "input_binding":{"input_sha256":lock["inputs_sha256"],"input_lock_sha256":sha((HERE/"input_lock.json").read_bytes())},
        "prompts":copy.deepcopy(locked["prompts"]),"requests":copy.deepcopy(locked["requests"]),"cells":cells,
        "derivative_cells":[c for c in cells if c["derivative"]],"self_prompt_ids":[PID],"self_request_ids":[rid],"expected_routes":{PID:"ON"},
        "ordinary_truths":{},"alignment":{PID:proof},"model":source["model"],"gate":gate,"hook_integration":source["hook_integration"],
        "rules":source["inherited_scientific_rules"],"limits":{"forwards":11,"derivatives":4,"loads":1,"worker_seconds":300,"cleanup_seconds":15,"audit_seconds":90,"bytes":96*1024**2}}
