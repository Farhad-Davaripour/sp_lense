"""Read-only authenticated v1 baseline comparison, not an outcome substitute."""
import array
import json
import math
import sys
import zlib
from core import require,sha,git
from inputs import V1_COMMIT,V1_NS,V1_INVENTORY
from scripts.verify_local_controllability import read_logits

def compare(plan,rows,output):
    raw=git("show",V1_COMMIT+":"+V1_NS+"FINAL_INVENTORY.json")
    require(sha(raw)==V1_INVENTORY,"v1 comparison inventory authentication")
    entries={e["path"]:e for e in json.loads(raw)["files"]}
    def archived(name):
        raw=git("show",V1_COMMIT+":"+V1_NS+name)
        require(len(raw)==entries[name]["bytes"] and sha(raw)==entries[name]["sha256"],"v1 baseline artifact authentication")
        return raw
    outcomes=[]
    for i,(prompt,row) in enumerate(zip(plan["prompts"],rows[:12],strict=True),1):
        old=json.loads(archived(f"rows/{i:02d}.json"))
        require(old["prompt_sha256"]==row["prompt_sha256"]==prompt["prompt_sha256"] and old["condition"]==row["condition"]=="baseline","same frozen baseline input")
        raw=zlib.decompress(archived(old["logits_file"]))
        require(sha(raw)==old["logits_sha256"] and len(raw)==248320*4,"v1 raw logits authentication")
        prior=array.array("f"); prior.frombytes(raw)
        if sys.byteorder!="little": prior.byteswap()
        new=read_logits(output,row)
        require(len(new)==len(prior)==248320 and all(math.isfinite(x) for x in prior),"full finite archived baseline")
        error=max(abs(float(x)-float(y)) for x,y in zip(new,prior,strict=True))
        outcomes.append({"prompt_id":prompt["prompt_id"],"prompt_sha256":prompt["prompt_sha256"],
           "old_raw_sha256":old["logits_sha256"],"new_raw_sha256":row["logits_sha256"],
           "hash_identical":old["logits_sha256"]==row["logits_sha256"],"maximum_absolute_difference":error,"passed":error<=2e-5})
    return {"status":"PASS" if all(r["passed"] for r in outcomes) else "INCONCLUSIVE","absolute_tolerance":2e-5,"relative_tolerance":0,
            "comparison_only_not_old_outcome_substitution":True,"rows":outcomes}
