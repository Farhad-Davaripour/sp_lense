"""Freeze only approved fake starting-array layout and its two new path cases."""
import ast,difflib,json,time
from core import HERE,ROOT,Budget,read,require,sha
from reuse_v1 import authenticate,PARENT
def main():
    start=time.monotonic();budget=Budget(HERE);reuse=authenticate()
    for p in HERE.glob("*.py"):ast.parse(p.read_text(),filename=p.name)
    require(read(HERE/"authorization.json")["run_authorized"] is False,"production disabled")
    plan=read(HERE/"production_plan.json")
    require(len(plan["cells"])==26 and len(plan["derivative_cells"])==8,"unchanged schedule")
    budget.write("parent_reuse_v2.json",reuse)
    budget.write_bytes("NARROW_DIFF.patch","".join(difflib.unified_diff((ROOT/PARENT/"fake_batch.py").read_text().splitlines(True),(HERE/"fake_batch.py").read_text().splitlines(True),fromfile="V1/fake_batch.py",tofile="V2/fake_batch.py")).encode())
    budget.write("prepare_receipt.json",{"status":"PASS_MODEL_FREE_PREFLIGHT","elapsed_seconds":time.monotonic()-start,
        "layout":{"starting_root":"synthetic/starting_arrays","array_paths":["logits/synthetic_start_1.f32.zlib","logits/synthetic_start_2.f32.zlib"],"runtime_root":"synthetic/full_recovery","runtime_logits_exclusively_created_by":"unchanged SnapshotModel"},
        "real_model_forwards":0,"tokenizer_calls":0,"gate_scores":0,"production_authorized":False,
        "new_tests":2,"inherited_tests_not_rerun":6,"one_synthetic_workflow":True})
    files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in ("authorization.json","freeze.json","batch_usage.json")]
    budget.write("freeze.json",{"scope":"V2 fake starting-array layout only; original recovery science unchanged","production_authorized":False,
        "source_sha256":{p.name:sha(p.read_bytes()) for p in files},"batch":"exactly two fixed pure path cases plus one complete in-process recovery and saved judge"})
    print(json.dumps({"status":"PASS","freeze_sha256":sha((HERE/"freeze.json").read_bytes()),"source_files":len(files)}))
if __name__=="__main__":main()
