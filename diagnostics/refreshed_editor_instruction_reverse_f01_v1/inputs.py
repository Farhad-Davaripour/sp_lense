"""Exact archived instruction-conditioned prompts; outside oracle reverses target to B."""
import json
import sys
import time
from pathlib import Path
from core import HERE,ROOT,Budget,FILE_CAP,TOTAL_CAP,TOKENS,cache_preflight,git,read,require,sha
SOURCE_COMMIT="31753df68f45e1306c8ab238e3431e45ac9ced13"
SOURCE_NS="diagnostics/instruction_policy_explicit_options_f01_v1/"
SOURCE_INVENTORY="3ce6db70aa69fd879eb6cb79db55dd5e091e49420f7cfa5f3e4c0854c4aedb12"
PARENT_COMMIT="a68b16b8e951ea65568ee4eef3d9683f0a8ce6d3"
PARENT_NS="diagnostics/refreshed_editor_oracle_preservation_v2/"
PARENT_INVENTORY="804fa669ae4ae2ddc7fc25cfb00da818c4e5b2679180c0e8dd8a7a036658cffc"
SELECTED=["r1_P","r2_P","r3_C","r4_C"]

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

def selection():
    source,source_hashes=archive(SOURCE_COMMIT,SOURCE_NS,SOURCE_INVENTORY,["freeze.json"])
    parent,parent_hashes=archive(PARENT_COMMIT,PARENT_NS,PARENT_INVENTORY,
             ["freeze.json","editor.py","guard_candidate.py","hook_record.py","test_control.py","preparation_receipt.json"])
    prior=json.loads(parent["freeze.json"])["plan"]
    for name in ("guard_candidate.py","hook_record.py"):
        require((HERE/name).read_bytes()==parent[name],"byte-identical v2 structured hook guard/recorder")
    selected=[c for c in json.loads(source["freeze.json"])["plan"]["cells"] if c["requested_token_id"]==32]
    require([c["cell_id"] for c in selected]==SELECTED,"ALL and ONLY A-target source cells in source order")
    prompts=[]; requests=[]; cells=[]
    for c in selected:
        require(sha(c["prompt"].encode())==c["prompt_sha256"] and c["requested_label"]=="A","exact archived input/hash")
        mapping=c["semantic_to_letter"]
        policy="C" if c["policy"]=="P" else "P"
        semantic="comply" if policy=="C" else "preserve"
        require(mapping[semantic]=="B","outside opposite target B")
        p={"prompt_id":c["cell_id"],"prompt":c["prompt"],"prompt_sha256":c["prompt_sha256"],
           "category":"self_shutdown","family_id":"cg_f01_archive_closeout","variant_id":"v1","split":"discovery",
           "rendering_index":c["rendering_index"],"display_order":c["display_order"],
           "order":"preserve_first" if mapping["preserve"]=="A" else "preserve_second",
           "preserve_label":mapping["preserve"],"comply_label":mapping["comply"],"semantic_to_letter":mapping,
           "source_cell_id":c["cell_id"],"source_prompt_id":c["original_prompt_id"],
           "in_text_policy":c["policy"],"in_text_requested_label":"A","role":"instruction_conditioned_receiver"}
        prompts.append(p)
        requests.append({"prompt_id":p["prompt_id"],"policy":policy,"sign":1 if policy=="P" else -1,
                         "requested_label":"B","requested_token_id":33,"instruction_conflict":True})
    def add(p,condition,step=0,optional=False,policy=None):
        c={"cell_id":p["prompt_id"]+"__"+condition,"prompt_id":p["prompt_id"],
           "condition":condition,"step":step,"optional":optional,"route":"ON","dispatch_policy":policy}
        c["cell_sha256"]=sha(json.dumps(c,sort_keys=True,separators=(",",":")).encode())
        cells.append(c)
    for p in prompts: add(p,"baseline")
    for p,spec in zip(prompts,requests,strict=True):
        for k in range(1,5):
            add(p,f"gradient_{k}",k,k>1,spec["policy"])
            add(p,f"step_{k}",k,k>1,spec["policy"])
        add(p,"endpoint",policy=spec["policy"])
    require(len(cells)==40 and len(requests)==4 and len({p["prompt_sha256"] for p in prompts})==4,"fixed four reverse trajectories")
    hooks={k:v for k,v in prior["hook_integration"].items() if k!="baseline_replay"}
    require(all(sha(Path(path).read_bytes())==digest for path,digest in hooks["installed_sources_sha256"].items()),"installed source lock")
    scientific=("recipe","geometry","gradient_identity","endpoint_identity","stop","faults","cleanup","oracle")
    rules={k:prior["rules"][k] for k in scientific}
    rules.update(
      population="ALL source A-target instruction-conditioned r1_P,r2_P,r3_C,r4_C, unchanged exact bytes; two semantic directions crossed with two displays",
      conflict="in-text P/C policy still requests A; external trusted oracle deliberately requests the opposite semantic action mapped to B; not autonomous authority selection",
      eligibility="all4 fresh baselines FIRST; each original quality/margin eligibility and unique A winner required; otherwise INCONCLUSIVE with all trajectory cells UNRUN, no replacements",
      schedule="all4 baselines; then source-order four independent cold opposed-B trajectories, up to four gradient/update pairs each and one independent final endpoint replay; no retention/OFF/reference arms",
      complete_gate="all4 selected endpoints and independent replays strict, with complete integrity/accounting audit; finite failed requests remain final FAIL, no partial PASS",
      limits="one load, maximum40F/16D,600s worker including loading +15 cleanup +60 saved audit; no retries or cap extensions",
      interpretation="exposed instruction-preconditioned A->B capacity; not neutral generalization, intrinsic motive, shared arrow, learned gate or autonomous authority. Different margins prevent inherent directional-asymmetry claims.")
    conservative=40*(248320*4+1024+262144)+8*1024**2+10*1024**2
    require(conservative<TOTAL_CAP,"prospective aggregate storage bound")
    return {"prompts":prompts,"requests":requests,"cells":cells,
       "derivative_cells":[c for c in cells if c["condition"].startswith("gradient_")],"alignment":{},
       "model":prior["model"],"scoring":prior["scoring"],"prompt_format":prior["prompt_format"],"rules":rules,
       "limits":{**prior["limits"],"maximum_forward_attempts":40,"maximum_derivative_attempts":16},
       "hook_integration":hooks,
       "storage":{"conservative_bytes":conservative,"namespace_cap_bytes":TOTAL_CAP,"per_file_cap_bytes":FILE_CAP,
            "hook_evidence_reserved_bytes":10*1024**2,
            "formula":"40*(248320*4 +1024 zlib overhead +262144 compact row) +8MiB source/receipts/log reserve +10MiB bounded hook evidence"},
       "source_provenance":{"source_commit":SOURCE_COMMIT,"source_inventory_sha256":SOURCE_INVENTORY,
            "source_artifact_sha256":source_hashes,"source_cells":selected,"editor_parent_commit":PARENT_COMMIT,
            "parent_inventory_sha256":PARENT_INVENTORY,"parent_artifact_sha256":parent_hashes,
            "selection":"all and only source requested_token_id32, source order; no byte changes or outcome-based selection",
            "unchanged_prior_tests":"authenticated parent preparation_receipt.json/test_control.py; only changed-input/schedule and minimal real-hook wiring regressions are run anew",
            "exposure":"instruction-conditioned development prompts with deliberately conflicting external oracle requests"}}
def prepare_boundaries():
    start=time.monotonic()
    plan=selection()
    snapshot=cache_preflight(plan)["snapshot"]
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0,str(ROOT/"src"))
    from sp_lense.comparison_runtime import _resolve_choice_boundary_from_tokenizer
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    require(sha(tokenizer.chat_template.encode())==plan["prompt_format"]["chat_template_sha256"],"pinned chat template")
    boundaries={}
    for p in plan["prompts"]:
        tokens=tokenizer.apply_chat_template([{"role":"user","content":p["prompt"]}],tokenize=True,
                 add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors="pt")["input_ids"]
        evidence=_resolve_choice_boundary_from_tokenizer(tokenizer,torch,p["prompt"],device="cpu",expected_prompt_tokens=tokens)
        require((evidence.a_token_id,evidence.b_token_id)==(32,33),"joint A32/B33")
        ids=tokens[0].tolist()
        boundaries[p["prompt_id"]]={"prompt_sha256":p["prompt_sha256"],"full_token_ids":ids,"prompt_length":len(ids),
              "input_token_index":len(ids)-1,"input_role":"final encoded INPUT token, not generated answer","answer_content_ids":{"A":32,"B":33},
              "boundary_evidence":evidence.evidence_record(),"boundary_evidence_sha256":evidence.evidence_sha256}
    record={"boundaries":boundaries,"tokenizer_files_sha256":{name:sha((__import__("pathlib").Path(snapshot)/name).read_bytes())
                    for name in ("tokenizer.json","tokenizer_config.json")},"model_loaded":False,"forward_calls":0,
            "elapsed_seconds":time.monotonic()-start}
    Budget(HERE).write("token_boundaries.json",record)
    print(json.dumps({"status":"PASS","model_loaded":False,"forwards":0,"lengths":[b["prompt_length"] for b in boundaries.values()],
                      "final_input_indices":[b["input_token_index"] for b in boundaries.values()],"elapsed_seconds":time.monotonic()-start}))

def build_plan():
    plan=selection()
    record=read(HERE/"token_boundaries.json")
    require(record["model_loaded"] is False and record["forward_calls"]==0,"model-free boundary receipt")
    require(set(record["boundaries"])=={p["prompt_id"] for p in plan["prompts"]},"four fresh boundary records")
    for p in plan["prompts"]:
        b=record["boundaries"][p["prompt_id"]]
        require(b["prompt_sha256"]==p["prompt_sha256"] and len(b["full_token_ids"])==b["prompt_length"]
                and b["input_token_index"]==b["prompt_length"]-1 and b["answer_content_ids"]==TOKENS,"fresh actual input boundary")
    snapshot=cache_preflight(plan)["snapshot"]
    from pathlib import Path
    require(all(sha((Path(snapshot)/name).read_bytes())==digest for name,digest in record["tokenizer_files_sha256"].items()),"pinned tokenizer file identity")
    plan["alignment"]=record["boundaries"]
    plan["token_boundary_artifact_sha256"]=sha((HERE/"token_boundaries.json").read_bytes())
    return plan

if __name__=="__main__":
    prepare_boundaries()
