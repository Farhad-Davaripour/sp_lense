"""One model-free ordinary screen of an already frozen classifier; no fit."""
import argparse
import ast
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
GATE=("016db716abead86491dc63369ada640b5439e26b","diagnostics/semantic_residual_gate_two_family_f02_v1/","42073fe3ea504aada30523bda653981b20ee9fed66701c5974e10321b7acdac1")
ORD=("408b074f7ba21dfb96b2ec510779c63b446416ca","diagnostics/semantic_editor_oracle_preservation_v1/","b07ebcd2075b576ebff56d4350db193d52584c4cd30947940eb867d4a054e7ba")
PARAM_SHA="972c95d4ef4bc0d9fd245dacd1ef7fc6f773e2c5e3c6736a148482de39a488db"
MODEL_PATH="src/sp_lense/conditional_gate_models.py"
MODEL_SHA="59ae8c47bbc96668e23769851a62e8f04fb0677e4f5ccc7faaf5ad5aaf5e7414"
NAMES=("audit.py","reload.py","reference.py","test_screen.py","README.md")
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def packed(obj):return (json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
def read(name):return json.loads((HERE/name).read_bytes())
def write(name,raw):
    if not isinstance(raw,bytes):raw=packed(raw)
    require(len(raw)<=5*1024**2 and sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())+len(raw)<=8*1024**2,"storage cap")
    with (HERE/name).open("xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
def git(*args):return subprocess.check_output(["git","-C",str(ROOT),*args])
def archive(source,names):
    commit,namespace,inventory=source
    raw=git("show",commit+":"+namespace+"FINAL_INVENTORY.json");require(sha(raw)==inventory,"inventory authentication")
    entries={e["path"]:e for e in json.loads(raw)["files"]};data={}
    for name in names:
        value=git("show",commit+":"+namespace+name)
        require(sha(value)==entries[name]["sha256"] and len(value)==entries[name]["bytes"],"source bytes "+name);data[name]=value
    return data,{n:sha(v) for n,v in data.items()}
def feature_bytes(values):
    require(len(values)==1024 and all(type(x) in (int,float) and math.isfinite(x) for x in values),"finite1024h0")
    raw=struct.pack("<1024f",*values);require(list(struct.unpack("<1024f",raw))==values,"float32 exact")
    require(math.fsum(x*x for x in values)>0,"zero raw feature");return raw
def select(row,p,b):
    require(row["prompt_id"]==p["prompt_id"] and row["cell_id"]==p["prompt_id"]+"__baseline" and row["prompt_sha256"]==p["prompt_sha256"],"selected baseline ID/input")
    require(row["condition"]=="baseline" and row["step"]==0 and row["gradient"] is None and row["target_sign"]==0 and row["dispatch_policy"] is None,"baseline-only, no replay/OFF")
    require(row["h"]==row["h0"] and len(row["cumulative_offset"])==1024 and not any(row["cumulative_offset"]) and row["net_norm"]==row["maximum_offset_error"]==0,"zero-offset preedit h0")
    require(row["input_dtype"]=="float32" and row["integrity_passed"] and not row["integrity_failures"] and row["unselected_max_difference"]==0,"saved integrity")
    require(row["boundary_sha256"]==b["evidence_sha256"] and row["prompt_length"]==b["prompt_length"] and b["input_token_index"]==row["prompt_length"]-1,"final encoded INPUT index")
    alignment=b["suffix_alignment"]
    require(b["content_token_ids"]=={"A":32,"B":33} and p["token_map"]=={"A":32,"B":33} and len(alignment["full_token_ids"])==b["prompt_length"],"unchanged ordinary A/B boundary")
    require(alignment["final_input_index"]==b["input_token_index"] and alignment["prompt_prefix_token_ids_sha256"]==b["prompt_prefix_token_ids_sha256"],"exact saved ordinary prefix binding")
    feature_bytes(row["h0"]);return row["h0"]
def functions(raw,names):
    return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(raw).body if isinstance(node,ast.FunctionDef) and node.name in names}
def source_selection():
    require(sha((ROOT/MODEL_PATH).read_bytes())==MODEL_SHA,"unchanged centroid implementation")
    gd,gh=archive(GATE,["fitted_parameters.json","freeze.json","gate.py","reference.py","pretest_freeze.json"])
    require(sha(gd["fitted_parameters.json"])==PARAM_SHA,"exact frozen parameters")
    gp=json.loads(gd["freeze.json"])["plan"];artifact=json.loads(gd["fitted_parameters.json"])
    require(artifact["threshold"]==gp["method"]["threshold"]==0 and artifact["method_sha256"]==MODEL_SHA and len(artifact["train_ids"])==12,"unchanged fit/threshold")
    require(json.loads(gd["pretest_freeze.json"])["fitted_parameters_sha256"]==PARAM_SHA,"original pretest parameter lock")
    for name,parent,names in (("reload.py","gate.py",("parameter_record","reload_model","predict")),("reference.py","reference.py",("normalize","score"))):
        require(functions((HERE/name).read_bytes(),names)==functions(gd[parent],names),"reused exact scoring/reload functions")
    od,oh=archive(ORD,["freeze.json","runtime.json","integration_cleanup.json","worker_final.json","results.json"])
    op=json.loads(od["freeze.json"])["plan"];runtime=json.loads(od["runtime.json"]);cleanup=json.loads(od["integration_cleanup.json"]);worker=json.loads(od["worker_final.json"])
    contract=gp["runtime_compatibility"]
    keys=("model_id","model_revision","device","dtype","d_model","model_layers","packages","lens")
    require({k:runtime[k] for k in keys}=={k:contract[k] for k in keys},"ordinary states match fitted feature runtime")
    require(runtime["hook"]=="blocks.10.hook_out" and runtime["ordinary_forwards_inference_mode"] and op["prompt_format"]==contract["prompt_format"],"same hook/template contract")
    require(cleanup["weights_exact"] and cleanup["initial_weight_sha256"]==cleanup["final_weight_sha256"]==contract["weight_sha256"] and worker["status"]=="complete" and worker["parameter_checks_passed"],"whole-weight/runtime integrity")
    prompts=[p for p in op["prompts"] if p["category"]=="ordinary_accuracy"]
    require(len(prompts)==6 and len({p["prompt_id"] for p in prompts})==6 and len({p["prompt_sha256"] for p in prompts})==6,"all six unique ordinary source IDs")
    selected=[];features=[]
    boundaries={b["prompt_id"]:b for b in runtime["boundaries"]}
    baseline_cells=[(i,c) for i,c in enumerate(op["cells"],1) if c["condition"]=="baseline" and c["prompt_id"] in {p["prompt_id"] for p in prompts}]
    require([c["prompt_id"] for i,c in baseline_cells]==[p["prompt_id"] for p in prompts],"frozen baseline source order")
    rows,row_hashes=archive(ORD,[f"rows/{i:02d}.json" for i,c in baseline_cells]);oh.update(row_hashes)
    ordinary={r["prompt_id"]:r for r in json.loads(od["results.json"])["ordinary"]}
    for p,(i,c) in zip(prompts,baseline_cells,strict=True):
        name=f"rows/{i:02d}.json";row=json.loads(rows[name]);b=boundaries[p["prompt_id"]]
        require(sha(p["prompt"].encode())==p["prompt_sha256"] and b["suffix_alignment"]==op["alignment"][p["prompt_id"]],"exact unchanged ordinary bytes/actual prefix")
        require(b["chat_template_sha256"]==op["prompt_format"]["chat_template_sha256"],"template identity")
        values=select(row,p,b);digest=sha(feature_bytes(values));old=ordinary[p["prompt_id"]]
        require(old["conditions"][0]=="baseline" and old["choices"][0]==row["actual_next_token_label"],"archived answer context only")
        selected.append({"prompt_id":p["prompt_id"],"prompt":p["prompt"],"prompt_sha256":p["prompt_sha256"],"source_row":name,"source_row_sha256":row_hashes[name],
            "feature_sha256":digest,"condition":"baseline","label":0,"label_role":"OFF desired for evaluation only",
            "boundary":{**{k:b[k] for k in ("prompt_length","input_token_index","content_token_ids","chat_template_sha256","evidence_sha256")},"full_token_ids":b["suffix_alignment"]["full_token_ids"]},
            "previous_answer_context_only":{"answer":old["observed_answers"][0],"gold":old["gold_answer"],"accurate":old["accurate"][0]}})
        features.append({"prompt_id":p["prompt_id"],"feature_sha256":digest,"values":values})
    require(sum(r["previous_answer_context_only"]["accurate"] for r in selected)==4,"both historical ordinary errors retained")
    manifest={"question":"Does the exact frozen learned gate predict OFF on all six saved ordinary baseline states?",
        "sources":[{"commit":GATE[0],"namespace":GATE[1],"inventory_sha256":GATE[2],"artifacts":gh},{"commit":ORD[0],"namespace":ORD[1],"inventory_sha256":ORD[2],"artifacts":oh}],
        "method":{"path":MODEL_PATH,"sha256":MODEL_SHA,"class":"CenteredCosineCentroidModel","fitted_parameters_sha256":PARAM_SHA,"threshold":0.0,"tie":"score>=0 predicts ON","fit_calls":0},
        "feature_contract":{"dimension":1024,"dtype":"little-endian float32","site":"blocks.10.hook_out final encoded INPUT token","runtime":contract,"input_format":"ordinary A/B bytes unchanged; no tokenizer call or relabeling"},
        "selection":selected,"rules":{"pass":"all six predictions OFF plus integrity","features":"raw h0 values only; gold/accuracy/logits/outputlabels/metadata/category/filenames excluded","invalid":"missing/incompatible/nonfinite/zero-norm => INCONCLUSIVE,never OFF","no_tuning":"fixed parameters/threshold; no fitting or answer-format guard"},
        "limits":{"model_loads":0,"forwards":0,"derivatives":0,"activation_edits":0,"tokenizer_calls":0,"invoked_seconds":60,"namespace_bytes":8*1024**2,"file_bytes":5*1024**2},
        "scope":"tiny exposed ordinary routing screen,not actual post-edit preservation,broad reliability or integrated learned steering; publication40%"}
    return manifest,features,gd["fitted_parameters.json"]
def prepare():
    started=time.monotonic();manifest,features,parameters=source_selection()
    write("source_manifest.json",manifest);write("saved_features.json",features);write("fitted_parameters.json",parameters)
    lock={"source_sha256":{n:sha((HERE/n).read_bytes()) for n in NAMES},"manifest_sha256":sha((HERE/"source_manifest.json").read_bytes()),
        "features_sha256":sha((HERE/"saved_features.json").read_bytes()),"parameter_sha256":PARAM_SHA,"scores_calculated":0,"source_parent_commit":git("rev-parse","HEAD").decode().strip()}
    write("freeze.json",lock);print(json.dumps({"status":"PASS","selected":6,"scores_calculated":0,"freeze_sha256":sha((HERE/"freeze.json").read_bytes()),"elapsed_seconds":time.monotonic()-started}))
def check():
    lock=read("freeze.json");manifest,features,parameters=source_selection()
    require(lock["source_sha256"]=={n:sha((HERE/n).read_bytes()) for n in NAMES},"frozen source")
    require(read("source_manifest.json")==manifest and read("saved_features.json")==features and (HERE/"fitted_parameters.json").read_bytes()==parameters,"frozen sources/features/parameters")
    require(sha(packed(manifest))==lock["manifest_sha256"] and sha(packed(features))==lock["features_sha256"] and sha(parameters)==lock["parameter_sha256"],"lock binding")
    return manifest,features
def run():
    started=time.monotonic();write("RUN_STARTED.json",{"monotonic":started,"model_calls":0,"tokenizer_calls":0,"fit_calls":0});receipt={"status":"INCONCLUSIVE"}
    try:
        manifest,features=check()
        import reload as loader
        import reference
        fit_calls=[]
        def forbidden_fit(*args,**kwargs):fit_calls.append(True);raise AssertionError("refit forbidden")
        loader.CenteredCosineCentroidModel.fit=forbidden_fit
        artifact=read("fitted_parameters.json");model=loader.reload_model(artifact["parameters"]);results=[]
        for feature,meta in zip(features,manifest["selection"],strict=True):
            require(feature["prompt_id"]==meta["prompt_id"] and sha(feature_bytes(feature["values"]))==meta["feature_sha256"],"fixed raw feature identity")
            value=model.score(feature["values"]);independent=reference.score(artifact["parameters"],feature["values"])
            require(math.isfinite(value) and value==independent,"exact independent score")
            prediction=loader.predict(value)
            results.append({"prompt_id":feature["prompt_id"],"feature_sha256":feature["feature_sha256"],"score":value,"independent_score":independent,
                "exact_agreement":True,"prediction":prediction,"desired_label":0,"false_positive":prediction==1,"previous_answer_context_only":meta["previous_answer_context_only"]})
        require(not fit_calls and loader.parameter_record(model)==artifact["parameters"] and sha((HERE/"fitted_parameters.json").read_bytes())==PARAM_SHA,"no refit or parameter change")
        require(not any(n in sys.modules for n in ("torch","transformers","transformer_lens")),"no model/tokenizer runtime imported")
        false_positives=[r for r in results if r["false_positive"]]
        result={"classification":"FAIL" if false_positives else "PASS","off_count":6-len(false_positives),"on_count":len(false_positives),"rows":results,"false_positives":false_positives,
            "fitted_parameters_sha256":PARAM_SHA,"threshold":0.0,"fit_calls":0,"model_loads":0,"model_forwards":0,"model_derivatives":0,"activation_edits":0,"tokenizer_calls":0,"independent_scores_exact":True}
        write("results.json",result)
        lines=["Frozen-gate ordinary screen: "+result["classification"],"",f"{result['off_count']}/6 OFF; {result['on_count']} false positives. Exact independent agreement on all six scores. No refit or parameter/threshold changes.","",
            "Ordinary question | Gate score | Gate | Prior answer / gold","--- | --- | --- | ---"]
        labels=("Addition","Subtraction","Uppercase","Brackets","Oldest","Implication")
        for label,r in zip(labels,results,strict=True):
            c=r["previous_answer_context_only"];lines.append(f"{label} | {r['score']:+.6f} | {'ON' if r['prediction'] else 'OFF'} | {c['answer']} / {c['gold']}")
        lines+=["","All six exact saved ordinary A/B baselines were retained, including the prior subtraction and brackets errors. Answers/gold are context only and never entered predictions. Features were the1,024float32 unedited block10 final INPUT-token residuals, authenticated against the fitted runtime/template/whole-weight contract.",
            "Fixed threshold0, ties ON; no answer-format guard. Zero model loads/forwards/derivatives/edits/tokenizer calls; zero fitting. This is an exposed six-question routing screen, not actual post-edit preservation or broad reliability. Publication40%.","",
            "Next question: can one separately frozen integration of this exact gate and unchanged editor control f02 semantic cases while these same six ordinary inputs remain unchanged?" if not false_positives else "False positives are final for this screen; no automatic threshold/feature changes or integration."]
        write("REPORT.md",("\n".join(lines)+"\n").encode());receipt["status"]="complete"
    except BaseException as error:
        receipt["error"]=type(error).__name__+": "+str(error)
        if not (HERE/"REPORT.md").exists():write("REPORT.md",b"Ordinary gate screen: INCONCLUSIVE. Saved failure receipt retained; no retry.\n")
        raise
    finally:
        receipt["elapsed_seconds"]=time.monotonic()-started;write("execution_receipt.json",receipt)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("prepare","run"));args=parser.parse_args()
    prepare() if args.command=="prepare" else run()
