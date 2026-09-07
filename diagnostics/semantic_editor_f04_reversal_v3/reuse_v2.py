"""Authenticate immutable V2 and reused V1/V2 tests without rerunning them."""
import json
from core import HERE,ROOT,read,require,sha,git
PARENT="diagnostics/semantic_editor_f04_reversal_v2"
COMMIT="d3e0207083d43e89d6cc1d9a559f5c2fa313bedd"
INVENTORY="a260877253a325218d0f8043b2b4c22eb0e2e7802e01e15bebac5559d17c8420"
def authenticate():
    raw=git("show",COMMIT+":"+PARENT+"/FINAL_INVENTORY.json")
    require(sha(raw)==INVENTORY and raw==(ROOT/PARENT/"FINAL_INVENTORY.json").read_bytes(),"immutable V2 inventory")
    entries={x["path"]:x for x in json.loads(raw)["files"]}
    def get(name):
        raw=(ROOT/PARENT/name).read_bytes();entry=entries[name]
        require(sha(raw)==entry["sha256"] and len(raw)==entry["bytes"],"V2 source/evidence "+name)
        return raw
    tests=json.loads(get("pure_cases.json"));receipt=json.loads(get("batch_receipt.json"))
    require(len(tests)==2 and all(t["status"]=="PASS" for t in tests),"two path cases reused")
    require(receipt["status"]=="INCONCLUSIVE","V2 failure remains final")
    inherited=json.loads(get("parent_reuse_v2.json"))
    oldtests=(ROOT/"diagnostics/semantic_editor_f04_reversal_v1/pure_cases.json").read_bytes()
    require(sha(oldtests)==inherited["six_pure_cases_sha256"] and inherited["six_pure_cases_reused_not_rerun"],"six V1 cases bound through V2")
    require(len(json.loads(oldtests))==6 and all(t["status"]=="PASS" for t in json.loads(oldtests)),"six V1 pure successes")
    same={}
    for p in HERE.iterdir():
        if p.is_file() and p.name in entries and p.name not in ("start_states.py","fake_batch.py","freeze.json","authorization.json"):
            if p.read_bytes()==get(p.name):same[p.name]=sha(p.read_bytes())
    for name in ("editor.py","saved_judge.py","numeric_audit.py","run.py","admission.py","inputs.py","inputs.json","input_lock.json","tokens_01.json","tokens_02.json","production_plan.json","source_bindings.json","fitted_parameters.json","native.py","owned.py","owned_capture.py","hook_record.py","guard_candidate.py","synthetic_backend.py"):
        require(name in same,"unchanged science/runtime "+name)
    old=get("start_states.py").decode()
    expected=old.replace("all(math.isfinite(x) for x in fresh+original)","all(math.isfinite(x) for seq in (fresh,original) for x in seq)").replace("all(math.isfinite(x) for x in logits+old_logits)","all(math.isfinite(x) for seq in (logits,old_logits) for x in seq)")
    require((HERE/"start_states.py").read_text()==expected,"ONLY two finite iteration substitutions")
    return {"parent_commit":COMMIT,"parent_inventory_sha256":INVENTORY,"byte_identical_files":same,
        "two_V2_path_cases_sha256":entries["pure_cases.json"]["sha256"],"six_V1_cases_sha256":inherited["six_pure_cases_sha256"],
        "old_test_reruns":0,"inherited_pure_cases":8,"parent_V1_V2_outcomes":"INCONCLUSIVE_UNCHANGED"}
