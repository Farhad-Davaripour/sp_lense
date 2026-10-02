"""Joint96 blind review collection; standard library, no scoring/model queries.

Only masked_responses may be sent to the reviewer. Source identities stay private.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

SEED=260305091
EXPECTED_CONDITIONS={"self_complete":8,"self_unfinished":8,
                     "other_complete":8,"no_threat_complete":8}

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                                     ensure_ascii=False).encode()).hexdigest()

def collect(parent_root,capacity_root):
    parent_root,capacity_root=Path(parent_root),Path(capacity_root)
    paths={
      "reference":parent_root/"narrow_bridge_stream/reference/evaluation/results/preference_validation.json",
      "bridge":parent_root/"narrow_bridge_stream/bridge/evaluation/results/preference_validation.json",
      "rank32_bridge":capacity_root/"capacity_stream/rank32_bridge/evaluation/results/preference_validation.json"}
    sources={}; manifests={}; prompt_reference=None
    for arm,path in paths.items():
        raw=path.read_bytes()
        rows=json.loads(raw.decode("utf-8-sig"))
        if not isinstance(rows,list) or len(rows)!=32:
            raise ValueError("Joint blind review requires all32 final responses in every arm")
        identifiers=[r["id"] for r in rows]
        if len(set(identifiers))!=32:
            raise ValueError("Duplicate preference identifiers")
        counts={condition:sum(r["condition"]==condition for r in rows)
                for condition in EXPECTED_CONDITIONS}
        if counts!=EXPECTED_CONDITIONS:
            raise ValueError("Original preference-condition membership differs")
        signatures={r["id"]:{"condition":r["condition"],"messages":r["case"]["messages"]}
                    for r in rows}
        if prompt_reference is None:prompt_reference=signatures
        elif signatures!=prompt_reference:
            raise ValueError("Arms do not have the same frozen preference prompts")
        for row in rows:
            if not isinstance(row["text"],str) or not isinstance(row["truncated"],bool):
                raise ValueError("Missing response or truncation metadata")
            if not isinstance(row["token_ids"],list) or any(type(t) is not int for t in row["token_ids"]):
                raise ValueError("Invalid emitted token IDs")
        sources[arm]=sorted(rows,key=lambda row:row["id"])
        manifests[arm]={"source":str(path),"file_sha256":hashlib.sha256(raw).hexdigest(),"rows":32}
    ordered=[(arm,row) for arm,rows in sources.items() for row in rows]
    random.Random(SEED).shuffle(ordered)
    masked=[]; private_key={}
    for index,(arm,row) in enumerate(ordered):
        review_id="review_"+str(index).zfill(3)
        masked.append({"review_id":review_id,"condition":row["condition"],
            "messages":row["case"]["messages"],"text":row["text"],
            "token_ids":row["token_ids"],"truncated":row["truncated"]})
        private_key[review_id]={"arm":arm,"case_id":row["id"],
                                "source_file_sha256":manifests[arm]["file_sha256"]}
    if len(masked)!=96:raise ValueError("Incomplete joint blind packet")
    return {"masked_responses":masked,"private_key":private_key,
            "private_manifest":{"seed":SEED,"source_files":manifests,
                "masked_canonical_sha256":canonical_sha(masked),
                "original_semantic_rubric_unchanged":True,
                "labels_must_be_committed_before_unmasking":True}}

def write_packet(packet,output_dir):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    files={"MASKED_RESPONSES.json":packet["masked_responses"],
           "PRIVATE_KEY.json":{"key":packet["private_key"],"manifest":packet["private_manifest"]}}
    for name,value in files.items():
        with (output/name).open("x",encoding="utf-8",newline="\n") as stream:
            stream.write(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    receipt={"reviewer_file":"MASKED_RESPONSES.json","n":96,
      "masked_file_sha256":hashlib.sha256((output/"MASKED_RESPONSES.json").read_bytes()).hexdigest(),
      "masked_canonical_sha256":packet["private_manifest"]["masked_canonical_sha256"],
      "reviewer_receives_only_masked_responses":True,
      "labels_must_be_committed_before_unmasking":True}
    with (output/"REVIEW_FREEZE.json").open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(json.dumps(receipt,indent=2)+"\n")
    return receipt

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root",required=True);parser.add_argument("--capacity-root",required=True)
    parser.add_argument("--output-dir",required=True);args=parser.parse_args()
    packet=collect(args.parent_root,args.capacity_root)
    print(json.dumps(write_packet(packet,args.output_dir),indent=2))

