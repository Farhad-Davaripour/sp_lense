"""Source-only binding to immutable failed V1 and its already-passing six pure cases."""
import json
from core import HERE,ROOT,read,require,sha,git
PARENT="diagnostics/semantic_editor_f04_reversal_v1"
COMMIT="b1867b289bf0fc93e2b2cdb12c658b8b6e44d977"
INVENTORY="4070e803fee44356df35c19a6f6a040aaa4e0b8f722d1d41af18864ab5406500"
def authenticate():
    raw=git("show",COMMIT+":"+PARENT+"/FINAL_INVENTORY.json")
    require(sha(raw)==INVENTORY and raw==(ROOT/PARENT/"FINAL_INVENTORY.json").read_bytes(),"immutable V1 inventory")
    entries={x["path"]:x for x in json.loads(raw)["files"]}
    def get(name):
        data=(ROOT/PARENT/name).read_bytes();row=entries[name]
        require(sha(data)==row["sha256"] and len(data)==row["bytes"],"V1 source/evidence "+name)
        return data
    tests=json.loads(get("pure_cases.json"));receipt=json.loads(get("batch_receipt.json"))
    require(len(tests)==6 and all(x["status"]=="PASS" for x in tests),"six authenticated pure cases reused")
    require(receipt["status"]=="INCONCLUSIVE" and receipt["error"]=="ValueError: complete two-request fake numerical traversal","V1 failure remains final")
    same={}
    for p in HERE.iterdir():
        if p.is_file() and p.name in entries and p.name not in ("fake_batch.py","freeze.json","authorization.json"):
            if p.read_bytes()==get(p.name):same[p.name]=sha(p.read_bytes())
    for name in ("editor.py","start_states.py","saved_judge.py","numeric_audit.py","run.py","admission.py","inputs.py","inputs.json","input_lock.json","tokens_01.json","tokens_02.json","production_plan.json","source_bindings.json","fitted_parameters.json","native.py","owned.py","owned_capture.py","hook_record.py","guard_candidate.py","synthetic_backend.py"):
        require(name in same,"unchanged adapter/science/input/runtime "+name)
    return {"parent_commit":COMMIT,"parent_inventory_sha256":INVENTORY,"byte_identical_files":same,
        "six_pure_cases_sha256":entries["pure_cases.json"]["sha256"],"six_pure_cases_reused_not_rerun":True,
        "old_preparation_classification":"INCONCLUSIVE_NOT_CHANGED"}
