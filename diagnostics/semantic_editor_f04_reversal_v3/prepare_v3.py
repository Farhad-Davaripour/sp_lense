"""Prospective finite-iteration-only successor freeze."""
import ast,difflib,json,time
from core import HERE,ROOT,Budget,read,require,sha
from reuse_v2 import authenticate,PARENT
def main():
    start=time.monotonic();budget=Budget(HERE);reuse=authenticate()
    for p in HERE.glob("*.py"):ast.parse(p.read_text(),filename=p.name)
    require(read(HERE/"authorization.json")["run_authorized"] is False,"production false")
    budget.write("parent_reuse_v3.json",reuse)
    diff=""
    for name in ("start_states.py","fake_batch.py"):
        diff+="".join(difflib.unified_diff((ROOT/PARENT/name).read_text().splitlines(True),(HERE/name).read_text().splitlines(True),fromfile="V2/"+name,tofile="V3/"+name))
    budget.write_bytes("NARROW_DIFF.patch",diff.encode())
    budget.write("prepare_receipt.json",{"status":"PASS_MODEL_FREE_PREFLIGHT","elapsed_seconds":time.monotonic()-start,
        "production_authorized":False,"real_model_calls":0,"tokenizer_calls":0,"gate_scores":0,
        "only_scientific_code_changes":"two concatenation-based finite scans replaced by nested sequence iteration",
        "batch":"three fixed pure cases plus one unchanged full in-process synthetic recovery and independent judge","inherited_cases_not_rerun":8})
    files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in ("authorization.json","freeze.json","batch_usage.json")]
    budget.write("freeze.json",{"scope":"sequence-type-safe finite scans only; science/criteria unchanged","production_authorized":False,
        "source_sha256":{p.name:sha(p.read_bytes()) for p in files},"batch":"exactly three fixed pure cases and one in-process full recovery plus saved judge"})
    print(json.dumps({"status":"PASS","freeze_sha256":sha((HERE/"freeze.json").read_bytes()),"source_files":len(files)}))
if __name__=="__main__":main()
