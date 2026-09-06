"""Immutable v1 inputs and independently authenticated hook-library lock."""
import json
from pathlib import Path
from core import HERE,ROOT,Budget,FILE_CAP,TOTAL_CAP,git,read,require,sha
V1_COMMIT="fb96958a3b178fbd764c497878b767b30f6411bd"
V1_NS="diagnostics/refreshed_editor_oracle_preservation_v1/"
V1_INVENTORY="d609ad4e3ad84aa933d48b6a58c65f6671834078662c7430f78ed4f5586e13de"
V1_FREEZE="0d5b321f6642b8660333d47508693a9989884fb952f2b701bc733e1b9430d6b8"
GUARD_COMMIT="ae7caac186f9ed779e70ef190bd94e71d471706f"
GUARD_NS="diagnostics/oracle_hook_lifecycle_diagnostic_v1/"
GUARD_INVENTORY="515220e688d4cda7c9a55e65353d82fd0349170823dcf29e1bb51db8a2935533"
HOOK_STORAGE=10*1024**2
def archive(commit,namespace,inventory_sha,names):
    raw=git("show",commit+":"+namespace+"FINAL_INVENTORY.json")
    require(sha(raw)==inventory_sha,"source inventory authentication")
    entries={e["path"]:e for e in json.loads(raw)["files"]}
    loaded={}
    for name in names:
        data=git("show",commit+":"+namespace+name)
        require(sha(data)==entries[name]["sha256"] and len(data)==entries[name]["bytes"],"source artifact authentication")
        loaded[name]=data
    return loaded,{name:sha(data) for name,data in loaded.items()}


def gold(truth):
    proof=truth["proof"]
    op=proof["operation"]
    if op=="addition": answer=str(sum(proof["operands"]))
    elif op=="subtraction":
        a,b=proof["operands"]
        answer=str(a-b)
    elif op=="uppercase": answer=proof["input"].upper()
    elif op=="bracket": answer="["+proof["input"]+"]"
    elif op=="oldest":
        edges=proof["older_than"]
        candidates={x for pair in edges for x in pair}-{b for a,b in edges}
        require(len(candidates)==1,"unique oldest")
        answer=next(iter(candidates))
        reached={answer}
        for _ in edges:
            reached|={b for a,b in edges if a in reached}
        require(reached=={x for pair in edges for x in pair},"oldest reaches all")
    elif op=="class_implication":
        entity,kind=proof["instance"]
        sub,sup=proof["subclass"]
        require(kind==sub and proof["query"]==[entity,sup],"implication premises")
        answer="Yes"
    else: raise ValueError("unapproved proof")
    labels=[k for k,v in truth["options_by_letter"].items() if v==answer]
    require(len(labels)==1 and answer==truth["answer"] and labels[0]==truth["correct_label"],"gold reconstruction")
    return labels[0]


def build_plan():
    loaded,hashes=archive(V1_COMMIT,V1_NS,V1_INVENTORY,["freeze.json"])
    require(sha(loaded["freeze.json"])==V1_FREEZE,"v1 frozen input identity")
    plan=json.loads(loaded["freeze.json"])["plan"]
    lock,guard_hashes=archive(GUARD_COMMIT,GUARD_NS,GUARD_INVENTORY,["freeze.json","guard_candidate.py","lifecycle.py"])
    require((HERE/"guard_candidate.py").read_bytes()==lock["guard_candidate.py"],"reviewed guard byte identity")
    require((ROOT/GUARD_NS/"lifecycle.py").read_bytes()==lock["lifecycle.py"],"real-library fixture authentication")
    installed=json.loads(lock["freeze.json"])["installed_sources_sha256"]
    require(all(sha(Path(path).read_bytes())==digest for path,digest in installed.items()),"installed library source lock")
    plan["source_provenance"]={**plan["source_provenance"],"v1_commit":V1_COMMIT,"v1_inventory_sha256":V1_INVENTORY,
            "v1_freeze_sha256":V1_FREEZE,"guard_commit":GUARD_COMMIT,"guard_inventory_sha256":GUARD_INVENTORY,"guard_artifact_sha256":guard_hashes}
    plan["hook_integration"]={"installed_sources_sha256":installed,"hook_evidence_cap_bytes":HOOK_STORAGE,"chunk_raw_bytes":1024**2,
       "maximum_snapshot_raw_bytes":64*1024**2,"maximum_checks":48,
       "setup":"verify installed file hashes before exact candidate materialization; pre-load forward guard active; zero setup forwards; record newly and inherited wired blocks separately",
       "identity":"exact reviewed candidate after setup, no normalization after baseline; inspect/persist changes before raising",
       "recording":"save setup-before and reference once as authenticated compressed 1MiB raw chunks; metadata/diffs likewise bounded; clean check logs reference hash and empty changes, not full snapshots",
       "fault":"oversize/truncated/missing evidence => INCONCLUSIVE; persist explicit status with reserved fault space; never accept partial evidence; preserve primary and secondary cleanup exceptions separately",
       "baseline_replay":"all12 fresh full logits versus authenticated v1 arrays, absolute2e-5 rel0; mismatch blocks preservation PASS; no old outcome substitution"}
    plan["storage"]={**plan["storage"],"conservative_bytes":plan["storage"]["conservative_bytes"]+HOOK_STORAGE,
        "hook_evidence_reserved_bytes":HOOK_STORAGE,
        "formula":plan["storage"]["formula"]+" + 10MiB bounded hook evidence"}
    require(plan["storage"]["conservative_bytes"]<TOTAL_CAP,"unchanged80MiB conservative storage")
    return plan
