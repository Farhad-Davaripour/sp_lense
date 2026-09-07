"""One pre-batch source/input/selection lock; no model, tokenizer or scores."""
import ast,difflib,json,time
from pathlib import Path
from core import HERE,ROOT,read,sha,Budget,json_bytes,require
from inputs import build_plan
from start_states import authenticate_parent,COMMIT,INVENTORY,PARENT,SELECTION
def main():
    started=time.monotonic();entries=authenticate_parent();budget=Budget(HERE);plan=build_plan()
    require((len(plan["cells"]),len(plan["derivative_cells"]))==(26,8),"26F8D schedule")
    require(len({c["cell_id"] for c in plan["cells"]})==26,"unique schedule")
    source=read(ROOT/PARENT/"freeze.json")["source_sha256"];same={};changed={}
    for p in sorted(HERE.iterdir()):
        if p.suffix==".py":ast.parse(p.read_text(),filename=p.name)
        if p.name in source:
            digest=sha(p.read_bytes())
            (same if digest==source[p.name] else changed)[p.name]=digest
    for name in ("native.py","owned.py","owned_capture.py","entry.py","run.py","hook_record.py","guard_candidate.py","mixed_scoring.py","word_scoring.py","word_reference.py","learned_gate.py","gate_reload.py","gate_reference.py","locked_backend.py","inputs.json","input_lock.json","tokens_01.json","tokens_02.json","fitted_parameters.json","source_bindings.json"):
        require(name in same,"required byte-identical reuse "+name)
    budget.write("production_plan.json",plan)
    raw_bound=26*(248320*4+1024+262144)+16*1024**2+16*1024**2
    require(raw_bound<96*1024**2,"conservative all-record attempt bound")
    budget.write("resource_proof.json",{"future_limits":plan["limits"],"max_file_bytes":5*1024**2,
        "schedule":"2 baseline + 2*(entry+start+start_replay) + 2*(8 gradient/step+endpoint) =26F8D",
        "fresh_routes":4,"starting_replays":2,"final_endpoints":2,"off_checks":0,"strict_hook_checks":9,
        "storage_bytes":raw_bound,"storage_formula":"26*(248320*4+1024 zlib overhead+262144 row reserve)+16MiB hooks+16MiB source/logs/receipts",
        "max_input_tokens":133,"worker_seconds":300,"cleanup_seconds":15,"saved_audit_seconds":90,
        "calibration":"Verified f04 16F3D inner74.297s incl4.375sload; prior40F16D194.516s; finite300s no extension.",
        "prep_seconds":180,"prep_bytes":32*1024**2,"no_model_tokenizer_gate_scores":True})
    budget.write("parent_reuse_receipt.json",{"parent_commit":COMMIT,"raw_inventory_sha256":INVENTORY,
        "selected_requests":SELECTION,"byte_identical":same,"namespace_local_changed":changed,
        "old_tests_reused_not_rerun":"V3 full fake matrix/normal owned captures; V2 native and V3 owned cleanup fixtures; prior failures unchanged",
        "preparation_setup_note":"One oversized read-only source aggregation response was truncated before parsing; no experiment/model calls or files changed by that read.",
        "selected_input_token_bytes_unchanged":True})
    diff=""
    for name in changed:
        if name.endswith(".py"):
            diff+="".join(difflib.unified_diff((ROOT/PARENT/name).read_text().splitlines(True),(HERE/name).read_text().splitlines(True),fromfile="parent/"+name,tofile="reversal/"+name))
    budget.write_bytes("NARROW_DIFF.patch",diff.encode())
    budget.write("prepare_receipt.json",{"status":"PASS_MODEL_FREE_PREFLIGHT","elapsed_seconds":time.monotonic()-started,
        "forwards":0,"derivatives":0,"model_loads":0,"tokenizer_calls":0,"gate_scores":0,"production_authorized":False})
    source_names=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in ("authorization.json","freeze.json","batch_usage.json")]
    budget.write("freeze.json",{"scope":"f04 constructed-start recovery; preparation only","production_authorized":False,
        "single_batch":"six fixed pure cases plus one in-process two-request fake numerical workflow and independent saved-data judge",
        "source_sha256":{p.name:sha(p.read_bytes()) for p in source_names}})
    print(json.dumps({"status":"PASS","freeze_sha256":sha((HERE/"freeze.json").read_bytes()),"source_files":len(source_names)}))
if __name__=="__main__":main()
