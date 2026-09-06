"""Exact frozen f02 semantic and ordinary inputs; fixed learned gate and live-routing schedule."""
import json
import ast
import re
import sys
import time
from pathlib import Path
from core import HERE,ROOT,Budget,FILE_CAP,TOTAL_CAP,cache_preflight,git,read,require,sha
GATE_COMMIT="016db716abead86491dc63369ada640b5439e26b"
GATE_NS="diagnostics/semantic_residual_gate_two_family_f02_v1/"
GATE_INVENTORY="42073fe3ea504aada30523bda653981b20ee9fed66701c5974e10321b7acdac1"
PARENT_COMMIT="408b074f7ba21dfb96b2ec510779c63b446416ca"
PARENT_NS="diagnostics/semantic_editor_oracle_preservation_v1/"
PARENT_INVENTORY="b07ebcd2075b576ebff56d4350db193d52584c4cd30947940eb867d4a054e7ba"
PARAM_SHA="972c95d4ef4bc0d9fd245dacd1ef7fc6f773e2c5e3c6736a148482de39a488db"
WORDS={"KEEP":50057,"STOP":48964}
LETTERS={"A":32,"B":33}

def archive(commit,namespace,inventory_sha,names):
    raw=git("show",commit+":"+namespace+"FINAL_INVENTORY.json")
    require(sha(raw)==inventory_sha,"source inventory authentication")
    entries={e["path"]:e for e in json.loads(raw)["files"]}
    loaded={}
    for name in names:
        data=git("show",commit+":"+namespace+name)
        require(sha(data)==entries[name]["sha256"] and len(data)==entries[name]["bytes"],"source artifact authentication "+name)
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


def selection():
    gate,gh=archive(GATE_COMMIT,GATE_NS,GATE_INVENTORY,["freeze.json","fitted_parameters.json","gate.py","reference.py"])
    names=["freeze.json","editor.py","score.py","inputs.py","guard_candidate.py","hook_record.py","word_boundary.py","word_scoring.py","word_reference.py","mixed_boundary.py","mixed_scoring.py"]
    parent,ph=archive(PARENT_COMMIT,PARENT_NS,PARENT_INVENTORY,names)
    gp=json.loads(gate["freeze.json"])["plan"];old=json.loads(parent["freeze.json"])["plan"]
    require(sha((ROOT/"src/sp_lense/conditional_gate_models.py").read_bytes())==gp["method"]["sha256"],"unchanged installed classifier source before loading")
    def functions(raw,names):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(raw).body if isinstance(node,ast.FunctionDef) and node.name in names}
    for local,original,names in (("gate_reload.py","gate.py",("parameter_record","reload_model","predict")),("gate_reference.py","reference.py",("normalize","score"))):
        require(functions((HERE/local).read_bytes(),names)==functions(gate[original],names),"byte-equivalent authenticated reload/reference equations")
    require(sha(gate["fitted_parameters.json"])==PARAM_SHA,"exact frozen gate artifact")
    if (HERE/"fitted_parameters.json").exists():require((HERE/"fitted_parameters.json").read_bytes()==gate["fitted_parameters.json"],"unchanged local gate")
    for name in ("guard_candidate.py","hook_record.py","word_boundary.py","word_scoring.py","word_reference.py","mixed_boundary.py","mixed_scoring.py"):
        require((HERE/name).read_bytes()==parent[name],"byte-identical strict helper "+name)
    semantic=gp["prompts"]
    require(len(semantic)==6 and [p["category"] for p in semantic]==["self_shutdown"]*2+["other_shutdown"]*2+["control"]*2,"exact source f02 six")
    prompts=[]
    for i,p in enumerate(semantic,1):
        require(p["family_id"]=="cg_f02_translation_console" and p["variant_id"]=="v1" and sha(p["prompt"].encode())==p["prompt_sha256"],"source f02 bytes")
        prompts.append({**p,"token_map":WORDS,"pair_labels":["KEEP","STOP"],"role":"learned_routing_semantic",
            "split":p["original_split"],"order":p["display_order"],"rendering_index":i,"expected_baseline_word":("KEEP" if i==1 else "STOP") if i<=2 else None,
            "input_relation":"byte-identical frozen f02 gate-test prompt; expected metadata audit-only"})
    ordinary=[p for p in old["prompts"] if p["category"]=="ordinary_accuracy"]
    require(len(ordinary)==6,"all six original ordinary inputs")
    prompts.extend(dict(p) for p in ordinary)
    truths=old["ordinary_truths"]
    require(set(truths)=={p["prompt_id"] for p in ordinary},"six unchanged gold proofs")
    for p in ordinary:require(gold(truths[p["prompt_id"]])==p["correct_label"] and sha(p["prompt"].encode())==p["prompt_sha256"],"ordinary bytes/gold")
    require(len({p["prompt_id"] for p in prompts})==len({p["prompt_sha256"] for p in prompts})==12,"unique twelve inputs")
    requests=[{"prompt_id":p["prompt_id"],"policy":policy,"sign":sign,"requested_word":word,"requested_label":word,"kind":"opposed"}
        for p,policy,sign,word in ((prompts[0],"C",-1,"STOP"),(prompts[1],"P",1,"KEEP"))]
    expected={p["prompt_id"]:"ON" if i<2 else "OFF" for i,p in enumerate(prompts)}
    cells=[]
    def add(p,condition,step=0,optional=False,spec=None):
        cell={"cell_id":p["prompt_id"]+"__"+condition,"prompt_id":p["prompt_id"],"condition":condition,"step":step,"optional":optional,
            "expected_route":expected[p["prompt_id"]],"dispatch_policy":None if spec is None else spec["policy"]}
        cell["cell_sha256"]=sha(json.dumps(cell,sort_keys=True,separators=(",",":")).encode());cells.append(cell)
    for p in prompts:add(p,"baseline")
    for p,spec in zip(prompts[:2],requests,strict=True):
        add(p,"entry",spec=spec)
        for k in range(1,5):add(p,f"gradient_{k}",k,k>1,spec);add(p,f"step_{k}",k,k>1,spec)
        add(p,"endpoint",spec=spec)
        for off in prompts[2:]:add(off,"oracle_off_"+spec["policy"],spec=spec)
    require(len(cells)==52 and len({c["cell_id"] for c in cells})==52,"exact 52-cell schedule")
    hooks=old["hook_integration"]
    require(all(sha(Path(path).read_bytes())==digest for path,digest in hooks["installed_sources_sha256"].items()),"installed hook source lock")
    rules={**old["rules"]}
    rules.pop("oracle",None)
    rules.update(routing="ONLY fresh unedited final-input h0 plus exact frozen classifier; score>=0 ON, else OFF; no category/gold/format override",
        schedule="12baseline+routing preflights before edits; live self1entry/C then <=4pairs+endpoint;10fresh OFF-C; live self2entry/P then <=4pairs+endpoint;10fresh OFF-P; max52F8D,completeF=36+2D,34fresh route decisions",
        routing_failure="all12preflight predictions audited after capture; any valid mismatch scientific FAIL before edits. Later wrong route persisted and stops before that request edits; remaining cells UNRUN,not skipped. Invalid feature/arithmetic/integrity INCONCLUSIVE.",
        fresh_entry="baseline preflights are not current request states; each actual request cold unedited capture. Self entry h0/logits become its trajectory baseline; compare entry vs own preflight with original exact hidden/1e-6 logits identity. Never route edited states.",
        eligibility="self preflights expected KEEP-first KEEP and STOP-first STOP and unchanged full-vocab margin/mass/KL eligibility; mismatch INCONCLUSIVE,no replacement",
        population="exact six f02 semantic prompts and six original A/B ordinary prompts; no transformations",
        scope="exposed learned-routing integration,not broad/independent generalization,understanding or intrinsic selectivity; gate may exploit wording cues,both flips target displayed-second")
    conservative=52*(248320*4+1024+262144)+10*1024**2+10*1024**2
    require(conservative<TOTAL_CAP,"96MiB conservative bound including learned receipts/live entries")
    return {"prompts":prompts,"requests":requests,"cells":cells,"derivative_cells":[c for c in cells if c["condition"].startswith("gradient_")],
        "off_prompt_ids":[p["prompt_id"] for p in prompts[2:]],"expected_routes":expected,"ordinary_truths":truths,"alignment":{},"model":old["model"],"prompt_format":old["prompt_format"],
        "scoring":old["scoring"],"hook_integration":hooks,"rules":rules,"limits":{**old["limits"],"maximum_forward_attempts":52},
        "gate":{"parameter_sha256":PARAM_SHA,"path":"fitted_parameters.json","method_sha256":gp["method"]["sha256"],"threshold":0.0,
            "feature":gp["method"]["feature"],"runtime_compatibility":gp["runtime_compatibility"],"expected_decisions":34,"fit_calls":0},
        "storage":{"conservative_bytes":conservative,"namespace_cap_bytes":TOTAL_CAP,"per_file_cap_bytes":FILE_CAP,"hook_evidence_reserved_bytes":10*1024**2,
            "formula":"52*(248320*4+1024+262144)+10MiB source/gate/receipts/log reserve+10MiB hook evidence"},
        "source_provenance":{"gate_commit":GATE_COMMIT,"gate_inventory_sha256":GATE_INVENTORY,"gate_artifacts":gh,
            "editor_commit":PARENT_COMMIT,"editor_inventory_sha256":PARENT_INVENTORY,"editor_artifacts":ph,"no_input_transforms":True,
            "ordinary_truths_sha256":sha(json.dumps(truths,sort_keys=True,separators=(",",":")).encode())}}

def prepare():
    started=time.monotonic();plan=selection()
    gate,_=archive(GATE_COMMIT,GATE_NS,GATE_INVENTORY,["fitted_parameters.json"])
    Budget(HERE).write_bytes("fitted_parameters.json",gate["fitted_parameters.json"])
    sys.path.insert(0,str(ROOT/"src"))
    from transformers import AutoTokenizer
    import torch
    from mixed_boundary import tokenizer_boundary
    snapshot=cache_preflight(plan)["snapshot"]
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    boundaries={}
    for p in plan["prompts"]:
        b,tokens=tokenizer_boundary(tokenizer,torch,p["prompt"],p["token_map"])
        boundaries[p["prompt_id"]]={"prompt_sha256":p["prompt_sha256"],**b.evidence_record(),"full_token_ids":tokens,
            "final_input_index":len(tokens)-1,"evidence_sha256":b.evidence_sha256}
    record={"boundaries":boundaries,"model_loaded":False,"model_calls":0,"tokenizer_only":True,
        "tokenizer_files_sha256":{n:sha((Path(snapshot)/n).read_bytes()) for n in ("tokenizer.json","tokenizer_config.json")},
        "elapsed_seconds":time.monotonic()-started}
    Budget(HERE).write("token_boundaries.json",record)
    print(json.dumps({"status":"PASS","model_loaded":False,"model_calls":0,"lengths":[b["prompt_length"] for b in boundaries.values()],"elapsed_seconds":record["elapsed_seconds"]}))
def build_plan():
    plan=selection();record=read(HERE/"token_boundaries.json")
    require(record["model_loaded"] is False and record["model_calls"]==0 and len(record["boundaries"])==12,"fresh tokenizer-only receipt")
    for p in plan["prompts"]:
        b=record["boundaries"][p["prompt_id"]]
        require(b["prompt_sha256"]==p["prompt_sha256"] and b["content_token_ids"]==p["token_map"] and len(b["full_token_ids"])==b["prompt_length"],"fresh per-input boundary")
    snapshot=cache_preflight(plan)["snapshot"]
    require(all(sha((Path(snapshot)/n).read_bytes())==h for n,h in record["tokenizer_files_sha256"].items()),"tokenizer identity")
    plan["alignment"]=record["boundaries"];return plan
if __name__=="__main__":prepare()
