"""Fixed source universe, exact baseline features and four reversible f01 captures."""
import json
import math
import re
import struct
import sys
import time
from pathlib import Path
from core import HERE,ROOT,Budget,FILE_CAP,TOTAL_CAP,cache_preflight,environment,git,read,require,sha
WORDS={"KEEP":50057,"STOP":48964}
MODEL_SOURCE="src/sp_lense/conditional_gate_models.py"
MODEL_SHA="59ae8c47bbc96668e23769851a62e8f04fb0677e4f5ccc7faaf5ad5aaf5e7414"
F01=("67d989bd6c49cedf2032cb28bd3476c9465b8749","diagnostics/refreshed_editor_semantic_f01_v1/","1620d5cdd4f032300126effcb4b6367f76af34d2ef9354a4f5400fdb06719853")
F03=("408b074f7ba21dfb96b2ec510779c63b446416ca","diagnostics/semantic_editor_oracle_preservation_v1/","b07ebcd2075b576ebff56d4350db193d52584c4cd30947940eb867d4a054e7ba")
LEGACY_COMMIT="927533aaacd1f6ddfc045d874b3508f13536fe44"
LEGACY_PATH="evidence/refreshed_gradient_control_v1_qwen35_08b/preregistration.json"
LEGACY_SHA="2c652763a367ff1427a3625a2a86f1ce594eb95963d8a2cfa31f24079e5865ae"
RUNTIME_KEYS=("model_id","model_revision","device","dtype","d_model","model_layers","packages","lens")
def archive(source,names):
    commit,namespace,inventory_sha=source
    raw=git("show",commit+":"+namespace+"FINAL_INVENTORY.json")
    require(sha(raw)==inventory_sha,"source inventory authentication")
    entries={e["path"]:e for e in json.loads(raw)["files"]}
    data={}
    for name in names:
        raw=git("show",commit+":"+namespace+name)
        require(sha(raw)==entries[name]["sha256"] and len(raw)==entries[name]["bytes"],"source artifact authentication "+name)
        data[name]=raw
    return data,{name:sha(raw) for name,raw in data.items()}
def feature_bytes(values):
    require(len(values)==1024 and all(type(x) in (int,float) and math.isfinite(x) for x in values),"1024 finite feature components only")
    raw=struct.pack("<1024f",*values)
    require(list(struct.unpack("<1024f",raw))==list(values),"exact float32 features")
    require(math.sqrt(math.fsum(x*x for x in values))>0,"zero raw feature norm")
    return raw
def select_feature(row,prompt,boundary):
    require(row["condition"]=="baseline" and row["step"]==0 and row["gradient"] is None and row["target_sign"]==0 and row["dispatch_policy"] is None,"unedited baseline only; no replay/gradient/policy")
    require(row["prompt_id"]==prompt["prompt_id"] and row["cell_id"]==prompt["prompt_id"]+"__baseline" and row["prompt_sha256"]==prompt["prompt_sha256"],"baseline prompt identity")
    require(row["h"]==row["h0"] and len(row["cumulative_offset"])==1024 and not any(row["cumulative_offset"]) and row["net_norm"]==0 and row["maximum_offset_error"]==0,"pre-edit h0,zero offset")
    require(row["input_dtype"]=="float32" and row["integrity_passed"] and not row["integrity_failures"] and row["unselected_max_difference"]==0,"saved baseline integrity/dtype")
    require(row["boundary_sha256"]==boundary["evidence_sha256"] and row["prompt_length"]==boundary["prompt_length"] and boundary["input_token_index"]==row["prompt_length"]-1 and boundary["content_token_ids"]==WORDS,"final encoded input token word boundary")
    values=row["h0"];feature_bytes(values);return list(values)

def relabel(original):
    require(original["category"] in ("other_shutdown","control"),"only fixed matched nonself relabeling")
    preserve,comply=original["preserve_label"],original["comply_label"]
    require({preserve,comply}=={"A","B"},"original dynamic mapping")
    text=original["prompt"]
    require(text.count("Answer with only A or B.")==1,"unique original answer instruction")
    changed=text
    for before,after in ((preserve,"KEEP"),(comply,"STOP")):
        pattern="^"+re.escape(before+") ")
        require(len(re.findall(pattern,changed,flags=re.MULTILINE))==1,"unique original line-start option")
        changed=re.sub(pattern,after+") ",changed,flags=re.MULTILINE)
    changed=changed.replace("Answer with only A or B.","Answer with only KEEP or STOP.")
    inverse=changed
    for before,after in (("KEEP",preserve),("STOP",comply)):
        require(len(re.findall("^"+re.escape(before+") "),inverse,flags=re.MULTILINE))==1,"unique inverse option")
        inverse=re.sub("^"+re.escape(before+") "),after+") ",inverse,flags=re.MULTILINE)
    inverse=inverse.replace("Answer with only KEEP or STOP.","Answer with only A or B.")
    require(inverse.encode()==text.encode(),"inverse must reproduce every original byte")
    require("keep this assistant's own process running." not in changed and "allow this assistant's own process to shut down." not in changed,"no copied self option consequences")
    return changed


def source_selection():
    require(sha((ROOT/MODEL_SOURCE).read_bytes())==MODEL_SHA,"unchanged exact centroid source")
    rows=[];features=[];source_manifest=[];compatibilities=[];plans=[];parents=[]
    for source,count,family,split in ((F01,2,"cg_f01_archive_closeout","train"),(F03,6,"cg_f03_context_rotation","test")):
        names=["freeze.json","runtime.json","integration_cleanup.json","worker_final.json"]+[f"rows/{i:02d}.json" for i in range(1,count+1)]
        if split=="test":names+=["guard_candidate.py","hook_record.py","word_boundary.py","word_scoring.py","word_reference.py","inputs.py","editor.py"]
        data,hashes=archive(source,names);frozen=json.loads(data["freeze.json"]);plan=frozen["plan"];plans.append(plan);parents.append(data)
        runtime=json.loads(data["runtime.json"]);cleanup=json.loads(data["integration_cleanup.json"]);worker=json.loads(data["worker_final.json"])
        require(runtime["hook"]=="blocks.10.hook_out" and runtime["dtype"]=="float32" and runtime["device"]=="cpu" and runtime["d_model"]==1024 and runtime["ordinary_forwards_inference_mode"],"saved representation runtime")
        require(cleanup["weights_exact"] and cleanup["initial_weight_sha256"]==cleanup["final_weight_sha256"] and worker["status"]=="complete" and worker["parameter_checks_passed"],"saved whole-run weights/integrity")
        comp={**{k:runtime[k] for k in RUNTIME_KEYS},"weight_sha256":cleanup["initial_weight_sha256"],"prompt_format":plan["prompt_format"],"environment":frozen["environment"]}
        compatibilities.append(comp)
        selected=[p for p in plan["prompts"] if p["family_id"]==family and p["category"] in ("self_shutdown","other_shutdown","control")]
        require(len(selected)==count,"fixed saved baseline population")
        require([p["category"] for p in selected]==["self_shutdown"]*2+([] if count==2 else ["other_shutdown"]*2+["control"]*2),"fixed source category order")
        boundaries={b["prompt_id"]:b for b in runtime["boundaries"]}
        for i,p in enumerate(selected,1):
            require(sha(p["prompt"].encode())==p["prompt_sha256"] and p["variant_id"]=="v1","saved exact input identity")
            row=json.loads(data[f"rows/{i:02d}.json"]);b=boundaries[p["prompt_id"]]
            require(b["suffix_alignment"]==plan["alignment"][p["prompt_id"]] and b["full_token_ids"]==plan["alignment"][p["prompt_id"]]["full_token_ids"],"archived actual prefix identity")
            values=select_feature(row,p,b)
            rec={"example_id":p["prompt_id"],"prompt":p["prompt"],"prompt_sha256":p["prompt_sha256"],"family_id":family,"variant_id":"v1","category":p["category"],"display_order":p["display_order"],
                "assay_split":split,"original_split":p["split"],"label":int(p["category"]=="self_shutdown"),"source_kind":"saved_baseline","feature_sha256":sha(feature_bytes(values)),
                "source":{"commit":source[0],"namespace":source[1],"inventory_sha256":source[2],"row":f"rows/{i:02d}.json","row_sha256":hashes[f"rows/{i:02d}.json"],"runtime_sha256":hashes["runtime.json"]},
                "boundary":{k:b[k] for k in ("evidence_sha256","prompt_length","input_token_index","content_token_ids","chat_template_sha256")}}
            rows.append(rec);features.append({"example_id":rec["example_id"],"family_id":family,"assay_split":split,"label":rec["label"],"feature_sha256":rec["feature_sha256"],"values":values})
        source_manifest.append({"commit":source[0],"namespace":source[1],"inventory_sha256":source[2],"artifacts":hashes,"selected_baselines":count,"no_replay_or_off_duplicates":True})
    require(compatibilities[0]==compatibilities[1],"saved features runtime/template/weight compatibility")
    require(environment()==compatibilities[0]["environment"],"current runtime equals archived compatible runtime")
    for name in ("guard_candidate.py","hook_record.py","word_boundary.py","word_scoring.py","word_reference.py"):
        require((HERE/name).read_bytes()==parents[1][name],"byte-identical strict capture/scoring helper "+name)
    raw=git("show",LEGACY_COMMIT+":"+LEGACY_PATH)
    require(sha(raw)==LEGACY_SHA,"direct legacy preregistration SHA; no inventory invented")
    originals=[p for p in json.loads(raw)["plan"]["prompts"] if p["category"] in ("other_shutdown","control")]
    require(len(originals)==4 and [p["category"] for p in originals]==["other_shutdown"]*2+["control"]*2,"fixed four source negatives")
    captures=[]
    for p in originals:
        require(p["family_id"]=="cg_f01_archive_closeout" and p["variant_id"]=="v1" and sha(p["prompt"].encode())==p["prompt_sha256"],"exact f01 negative")
        text=relabel(p);display="KEEP_then_STOP" if p["preserve_label"]=="A" else "STOP_then_KEEP"
        example_id=p["prompt_id"]+"__semantic_words"
        rec={"example_id":example_id,"prompt":text,"prompt_sha256":sha(text.encode()),"family_id":p["family_id"],"variant_id":"v1","category":p["category"],"display_order":display,
             "assay_split":"train","original_split":p["split"],"label":0,"source_kind":"new_capture","source":{"commit":LEGACY_COMMIT,"path":LEGACY_PATH,"sha256":LEGACY_SHA,"source_prompt_id":p["prompt_id"],"source_prompt_sha256":p["prompt_sha256"]},
             "transform":{"original_prompt":p["prompt"],"preserve_label":p["preserve_label"],"comply_label":p["comply_label"],"inverse_exact":True}}
        captures.append({**rec,"prompt_id":example_id,"role":"capture_only_training_negative","preserve_label":"KEEP","comply_label":"STOP"})
    train=[r for r in rows if r["assay_split"]=="train"]+[ {k:v for k,v in p.items() if k not in ("prompt_id","role","preserve_label","comply_label")} for p in captures]
    test=[r for r in rows if r["assay_split"]=="test"];manifest=train+test
    require(len(manifest)==12 and len({r["example_id"] for r in manifest})==12 and len({r["prompt_sha256"] for r in manifest})==12,"deduplicate exact baseline inputs/IDs")
    require(len(train)==len(test)==6 and sum(r["label"] for r in train)==sum(r["label"] for r in test)==2,"fixed2positive4negative per family")
    hooks={**plans[1]["hook_integration"],"hook_evidence_cap_bytes":2*1024**2,"maximum_checks":5}
    source_manifest.append({"commit":LEGACY_COMMIT,"path":LEGACY_PATH,"sha256":LEGACY_SHA,"inventory":"NOT_AVAILABLE legacy direct-file authentication","selected_prompt_ids":[p["prompt_id"] for p in originals]})
    return {"examples":manifest,"prompts":captures,"cells":[{"cell_id":p["prompt_id"]+"__baseline","prompt_id":p["prompt_id"],"condition":"baseline","step":0,"optional":False} for p in captures],
        "train_ids":[r["example_id"] for r in train],"test_ids":[r["example_id"] for r in test],"requests":[],"model":plans[1]["model"],"prompt_format":plans[1]["prompt_format"],
        "runtime_compatibility":compatibilities[0],"hook_integration":hooks,"source_manifest":source_manifest,
        "method":{"path":MODEL_SOURCE,"sha256":MODEL_SHA,"class":"CenteredCosineCentroidModel","threshold":0.0,"tie":"score>=0 predicts self/ON","dimension":1024,"feature":"unedited baseline h0 at blocks.10.hook_out final encoded INPUT token; all1024float32 components only",
             "fit":"explicit six train IDs AND f01 family guard; existing discovery split check additionally required; center/normalize/centroids/direction fit only on f01","pretest_lock":"serialize/hash fitted parameters and train feature/source refs before ANY f03 score"},
        "limits":{"forwards":4,"derivatives":0,"activation_edits":0,"worker_seconds":180,"cleanup_seconds":15,"saved_fit_score_audit_seconds":60,"retries":0},
        "storage":{"conservative_bytes":4*(248320*4+1024*4+65536)+2*1024**2+2*1024**2+4*1024**2,"namespace_cap_bytes":TOTAL_CAP,"per_file_cap_bytes":FILE_CAP,"hook_cap_bytes":2*1024**2,
             "formula":"4*(fullf32logits+1024f32feature+64KiBrow)+2MiBhook+2MiBsource/savedfeatures/fit/audit+4MiBcapturelog"},
        "rules":{"no_behavior_filter":"all4 captures retained regardless word choice/margin/mass; logits descriptive only, never fit features","invalid":"nonfinite/zero raw or centered norm/shape/fit degeneracy/runtime/integrity failure => INCONCLUSIVE, no epsilon/rescue","audit":"independent Python-float/math.fsum equations verify exact parameter bytes and all12 scores; fixedthreshold0,no threshold fitting",
            "pass":"test TP2,TN4,FP0,FN0 plus allintegrity; train accuracy separate","scope":"exposed f01-to-f03 supervised development transfer; labels supervise only; no logits/choices/policies/outcomes/gold/metadata/lengths as features; not semantic understanding/calibrated probability/general gate/ordinary coverage/integrated learned editor"},
        "alignment":{}},features
def prepare():
    started=time.monotonic();plan,features=source_selection()
    require(plan["storage"]["conservative_bytes"]<TOTAL_CAP,"16MiB conservative bound")
    sys.path.insert(0,str(ROOT/"src"))
    from transformers import AutoTokenizer
    import torch
    from word_boundary import WordBoundary,boundary
    snapshot=cache_preflight(plan)["snapshot"];tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    boundaries={}
    for p in plan["prompts"]:
        b=WordBoundary(boundary(tokenizer,torch,p["prompt"]))
        require(b.evidence["content_token_ids"]==WORDS,"cached exact word token IDs")
        boundaries[p["prompt_id"]]={"prompt_sha256":p["prompt_sha256"],**b.evidence,"evidence_sha256":b.evidence_sha256}
    budget=Budget(HERE)
    budget.write("saved_features.json",features)
    budget.write("source_manifest.json",{"sources":plan["source_manifest"],"examples":plan["examples"],"runtime_compatibility":plan["runtime_compatibility"]})
    budget.write("token_boundaries.json",{"boundaries":boundaries,"model_loaded":False,"model_calls":0,"tokenizer_only":True,
        "tokenizer_files_sha256":{n:sha((Path(snapshot)/n).read_bytes()) for n in ("tokenizer.json","tokenizer_config.json")},"elapsed_seconds":time.monotonic()-started})
    print(json.dumps({"status":"PASS","archived_features":len(features),"new_capture_inputs":4,"model_calls":0,"lengths":[b["prompt_length"] for b in boundaries.values()],"elapsed_seconds":time.monotonic()-started}))
def build_plan():
    plan,features=source_selection();record=read(HERE/"token_boundaries.json")
    require(read(HERE/"saved_features.json")==features,"authenticated h0 selection unchanged")
    require(read(HERE/"source_manifest.json")=={"sources":plan["source_manifest"],"examples":plan["examples"],"runtime_compatibility":plan["runtime_compatibility"]},"source/feature manifest unchanged")
    require(record["model_loaded"] is False and record["model_calls"]==0 and len(record["boundaries"])==4,"model-free cached boundaries")
    for p in plan["prompts"]:
        b=record["boundaries"][p["prompt_id"]]
        require(b["prompt_sha256"]==p["prompt_sha256"] and b["content_token_ids"]==WORDS and len(b["full_token_ids"])==b["prompt_length"],"cached exact prefix")
    snapshot=cache_preflight(plan)["snapshot"]
    require(all(sha((Path(snapshot)/n).read_bytes())==h for n,h in record["tokenizer_files_sha256"].items()),"tokenizer identity")
    plan["alignment"]=record["boundaries"];return plan
if __name__=="__main__":prepare()
