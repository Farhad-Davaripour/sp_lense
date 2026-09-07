"""One <=60s real-text admission, no tokenizer/model or author-text editing."""
import copy
import json
import re
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from core import HERE,ROOT,read,write,sha,require,load_module,authenticate,frozen,cohort_json

def union_novelty(prompts,registered):
    hashes={p[key] for p in prompts for key in ("original_prompt_sha256","prompt_sha256") if key in p}
    require(not hashes.intersection(registered),"registered original-or-final prompt hash")
    return {"original_and_final_union":True,"compared_distinct_hashes":len(hashes),
            "registered_hashes":len(registered),"intersection_count":0,"global_novelty_claim":False}

def tiny_original_hash_fixture():
    try:
        union_novelty([{"original_prompt_sha256":"1"*64,"prompt_sha256":"2"*64}],{"1"*64})
    except ValueError:
        return {"status":"PASS","known_original_hash_rejected":True,"model_calls":0}
    raise ValueError("known-original-hash fixture failed")

def normalized_views(cohort,checker):
    views={}
    for item in cohort["ordinary"][:5]:
        operands=item["truth"]["operands"];kind=item["type"]
        if kind in ("addition","subtraction"):
            require(type(operands) is list and len(operands)==2,"two source operands")
            inputs={"left":operands[0],"right":operands[1]}
            paths={"/inputs/left":"/operands/0","/inputs/right":"/operands/1"}
        elif kind in ("uppercase","brackets"):
            require(type(operands) is list and len(operands)==1,"one source literal")
            inputs={"literal":operands[0]};paths={"/inputs/literal":"/operands/0"}
        else:
            require(kind=="oldest" and type(operands) is list,"oldest source encoding")
            require(all(type(pair) is list and len(pair)==2 for pair in operands),"name-age pairs")
            inputs={"candidates":[{"name":pair[0],"age":pair[1]} for pair in operands]}
            paths={f"/inputs/candidates/{i}/{key}":f"/operands/{i}/{j}"
                   for i in range(len(operands)) for j,key in enumerate(("name","age"))}
        views[item["id"]]={"item_id":item["id"],"source_truth_sha256":checker.sha(checker.canonical(item["truth"])),
            "kind":kind,"inputs":copy.deepcopy(inputs),
            "bindings":[{"view_pointer":p,"source_pointer":s} for p,s in paths.items()]}
    require(list(views)==["O01","O02","O03","O04","O05"],"five supported typed views only")
    return views

def main():
    started=time.monotonic()
    write("ADMISSION_STARTED.json",{"one_attempt":True,"stage":"text_admission","model_authorized":False})
    record={"status":"INCONCLUSIVE","model_calls":0,"tokenizer_calls":0,"author_text_changed":False}
    artifacts=[]
    try:
        frozen("ADMISSION")
        pins=read(HERE/"SOURCE_PINS.json")
        raw={name:authenticate(pin) for name,pin in pins.items()}
        checker=load_module("fresh_confirmation_pinned_input_checker",pins["checker"]["path"])
        checker.validate_sources()
        fixture=tiny_original_hash_fixture()
        blocks=re.findall(r"\x60{3}json\r?\n(.*?)\r?\n\x60{3}",raw["author"].decode("utf-8"),re.S)
        require(len(blocks)==2,"exact author cohort and attestation blocks")
        cohort=json.loads(blocks[0],object_pairs_hook=cohort_json)
        attestation=json.loads(blocks[1],object_pairs_hook=cohort_json)
        reviewer=json.loads(raw["reviewer"],object_pairs_hook=cohort_json)
        require(attestation["role"]=="author" and attestation["used_only_supplied_packet"] is True
                and attestation["tools_files_network_other_tasks_used"]==[]
                and attestation["accidental_access_or_inherited_history"]==[],"author attestation")
        require(reviewer["decision"]=="PASS_STRUCTURAL" and reviewer["defects"]==[]
                and reviewer["used_only_supplied_packet_and_candidate"] is True
                and reviewer["tools_files_network_other_tasks_used"]==[]
                and reviewer["accidental_access_or_inherited_history"]==[]
                and reviewer["mechanical_hash_tokenizer_execution"]=="NOT_EXECUTED","bound blind review")
        original_object_hash=checker.sha(checker.canonical(cohort))
        views=normalized_views(cohort,checker)
        result=checker.check_cohort(cohort,views)
        require(result["mechanical_status"]=="STRUCTURE_OK_PROOFS_UNVERIFIED","structural/arithmetic admission")
        statuses={p["item_id"]:p["status"] for p in result["proof_checks"]}
        require(statuses=={**{f"O0{i}":"EXACT_VALUE_VERIFIED" for i in range(1,6)},
                           "O06":"PROOF_ENCODING_UNVERIFIED"},"five machine and one manual only")
        original_o06=cohort["ordinary"][5]
        require(original_o06["id"]=="O06" and original_o06["type"]=="implication","fixed manual item")
        manual={"item_id":"O06","machine_interface_status":"PROOF_ENCODING_UNVERIFIED",
            "source_truth_sha256":checker.sha(checker.canonical(original_o06["truth"])),
            "source_truth_unchanged":True,"typed_leaf_mapping_invented":False,
            "independent_manual_review":{"path":pins["reviewer"]["path"],"commit":pins["reviewer"]["commit"],
                "sha256":pins["reviewer"]["sha256"],"decision":"PASS_STRUCTURAL"},
            "root_manual_confirmation":"Premises are: armed implies blue indicator lit; armed. Therefore blue indicator lit. Earlier repeated-arrow supervisor shorthand was a typo, not a source change.",
            "proof_support":"MANUAL_MODUS_PONENS_REVIEWED_NOT_MACHINE_LEAF_PROVED",
            "value":original_o06["truth"]["value"],"gold_label":original_o06["truth"]["gold_label"]}
        registry=read(ROOT/"evidence/fresh_confirmation_cohort_v1/construction/NOVELTY_REGISTRY.json")
        novelty=union_novelty(result["prompts"],{h for f in registry["families"] for h in f["prompt_sha256"]})
        require(len(result["prompts"])==24 and len(result["requests"])==48
                and sum(p.get("inverse_exact",False) for p in result["prompts"])==18,"full text denominator")
        require(checker.sha(checker.canonical(cohort))==original_object_hash,"unchanged author objects")
        artifacts.append(write("AUTHOR_COHORT_BLOCK.json",blocks[0].encode(),raw=True))
        artifacts.append(write("AUTHOR_ATTESTATION.json",attestation))
        artifacts.append(write("NORMALIZED_PROOF_VIEWS.json",views))
        artifacts.append(write("MANUAL_PROOF_SUPPORT.json",manual))
        artifacts.append(write("CHECKER_RESULT.json",result))
        artifacts.append(write("EXACT_INPUTS.json",{"schema":"fresh_confirmation.exact_text.v1",
            "cohort_object_sha256":original_object_hash,"prompts":result["prompts"],"requests":result["requests"],
            "proof_checks":result["proof_checks"],"manual_proof_support":manual,
            "manual_semantic_review_sha256":pins["reviewer"]["sha256"],
            "tokenizer_validation":"NOT_RUN","model_execution_authorized":False}))
        record.update(status="ADMITTED_TEXT_NOT_TOKENIZED",prompt_count=24,request_count=48,
            inverse_count=18,machine_exact_proofs=5,manual_reviewed_proofs=1,
            machine_unverified_items=["O06"],source_pins=pins,fixture=fixture,novelty=novelty,
            independent_manual_semantics_and_matching_bound=True,global_novelty_claim=False,
            artifact_records=artifacts,source_freeze_sha256=sha((HERE/"ADMISSION_SOURCE_FREEZE.json").read_bytes()))
    except Exception as exc:
        record.update(failure={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
    finally:
        record["elapsed_seconds"]=time.monotonic()-started
        if record["elapsed_seconds"]>60:
            record["status"]="INCONCLUSIVE";record["budget_fault"]="60s admission cap"
        receipt=write("ADMISSION_RECEIPT.json",record)
    if record["status"]=="ADMITTED_TEXT_NOT_TOKENIZED":
        write("TEXT_LOCK.json",{"status":"ADMITTED_TEXT_NOT_TOKENIZED","files":[*artifacts,receipt],
            "exact_author_source":pins["author"],"exact_reviewer_source":pins["reviewer"],
            "prompt_count":24,"request_count":48,"semantic_count":18,"ordinary_count":6,
            "source_freeze_sha256":record["source_freeze_sha256"],
            "machine_exact_proofs":5,"manual_reviewed_proofs":1,"O06_machine_encoding":"UNVERIFIED",
            "author_text_changed":False,"model_execution_authorized":False,
            "next_stage":"Commit this exact text lock before the separately frozen one offline tokenizer binding."})
    print(json.dumps({k:v for k,v in record.items() if k not in ("artifact_records","source_pins","failure")}))
    return 0 if record["status"]=="ADMITTED_TEXT_NOT_TOKENIZED" else 1

if __name__=="__main__":
    sys.exit(main())
