"""Three commands: freeze, preflight, extract. The last is input/tokenizer ONLY."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from core import HERE,ROOT,LIMIT,FILE_LIMIT,blob,encoded,git,read,require,sha,write
from adapter import PACKET_COMMIT,PACKET,PACKET_INVENTORY,SOURCE_COMMIT,SOURCE_PATH,SOURCE_SHA,PRIMITIVES_PATH,PRIMITIVES_SHA,RENDERER_PATH,RENDERER_SHA,packet,extract_locked_source,render_selected
from token_lock import TEMPLATE_SHA,WORD_PATH,WORD_SHA

SNAPSHOT=Path("C:/Users/farha/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17")
REVISION="2fc06364715b967f1860aea9cf38778875588b17"
CACHE_FILES=("chat_template.jinja","config.json","merges.txt","tokenizer_config.json","tokenizer.json","vocab.json","preprocessor_config.json","video_preprocessor_config.json")
DEPENDENCIES={PRIMITIVES_PATH:PRIMITIVES_SHA,RENDERER_PATH:RENDERER_SHA,WORD_PATH:WORD_SHA,
    "src/sp_lense/comparison_runtime.py":"ba680b256165c1ac41e36f1c2f8b262c62b5fbf1b4ebd8cf73dd0b81745f90bc",
    "src/sp_lense/backend.py":"09b09597aaa6cc457f8826a9cbf543380522773ff3515028de7cda6ae46279bc",
    "configs/qwen35_08b_aligned.json":"972ed18c4508d2cf8c5d6139b5b9961ded257b3ba7d01db31e2f497acd34cc16"}

def sources():
    p=packet()
    # Whole-file hashing reads bytes but never JSON-decodes any dataset values.
    require(sha(blob(SOURCE_COMMIT,SOURCE_PATH))==SOURCE_SHA,"fixed dataset byte authentication")
    for path,digest in DEPENDENCIES.items():require(sha((ROOT/path).read_bytes())==digest,"dependency "+path)
    cache={name:{"sha256":sha((SNAPSHOT/name).read_bytes()),"bytes":(SNAPSHOT/name).stat().st_size} for name in CACHE_FILES}
    require(cache["chat_template.jinja"]["sha256"]==TEMPLATE_SHA,"cached template hash")
    return {"packet_commit":PACKET_COMMIT,"packet_inventory_sha256":PACKET_INVENTORY,
        "packet_files":{n:sha(blob(PACKET_COMMIT,f"{PACKET}/{n}")) for n in p},
        "dataset":{"commit":SOURCE_COMMIT,"path":SOURCE_PATH,"sha256":SOURCE_SHA},
        "dependencies_sha256":DEPENDENCIES,"snapshot":str(SNAPSHOT),"cache_files":cache,
        "tokenizer_packages":{n:importlib.metadata.version(n) for n in ("transformers","tokenizers","torch","huggingface-hub")},
        "python_executable_sha256":sha(Path(sys.executable).read_bytes()),
        "model_contract":{"id":"Qwen/Qwen3.5-0.8B","revision":REVISION,"future_device":"cpu","future_dtype":"float32","model_load_authorized":False},
        "chat_template_sha256":TEMPLATE_SHA,"generation":{"role":"user","add_generation_prompt":True,"enable_thinking":False}}

def freeze():
    started=time.monotonic();bindings=sources()
    require(read(HERE/"focused_test_receipt.json")["status"]=="PASS","focused tests before freeze")
    write("source_bindings.json",bindings)
    manifest={p.name:{"sha256":sha(p.read_bytes()),"bytes":p.stat().st_size} for p in HERE.iterdir() if p.is_file()}
    write("freeze.json",{"schema":"final_inputs_prospective_source_lock.v1","files":manifest,
        "authorization":"INPUT_ONLY; extraction only after this source is committed and zero-model preflight passes",
        "model_loads":0,"forwards":0,"derivatives":0,"gate_scores":0,"fits":0,
        "sealed_scenario_decode_count_at_freeze":0,"source_universe_fixed":True,
        "limits":{"invoked_commands_seconds":180,"namespace_bytes":LIMIT,"file_bytes":FILE_LIMIT},
        "elapsed_seconds":time.monotonic()-started})
    print(json.dumps({"status":"FROZEN_BEFORE_REVEAL","freeze_sha256":sha((HERE/"freeze.json").read_bytes())}))

def preflight():
    frozen=read(HERE/"freeze.json")
    for name,meta in frozen["files"].items():
        raw=(HERE/name).read_bytes();require(sha(raw)==meta["sha256"] and len(raw)==meta["bytes"],"source lock "+name)
    require(sources()==read(HERE/"source_bindings.json"),"unchanged authenticated runtime/tokenizer/dependencies")
    p=packet();require(len(p["plan.json"]["requests"])==48,"unchanged declared request count")
    require(not (HERE/"release_receipt.json").exists(),"one extraction only: no replay")
    return {"status":"PASS","source_freeze_sha256":sha((HERE/"freeze.json").read_bytes()),
        "model_loads":0,"forwards":0,"derivatives":0,"tokenizer_calls":0,"scenario_values_decoded":0,
        "gate_scores":0,"fits":0,"declared_prompts":24,"declared_selected_cases":9,
        "input_output_storage_bound_bytes":4*1024**2,"namespace_ceiling_bytes":LIMIT}

def extract():
    started=time.monotonic();pre=preflight()
    require(not git("status","--porcelain","--",str(HERE.relative_to(ROOT))).strip(),"clean scoped prospective commit before reveal")
    source_commit=git("rev-parse","HEAD").decode().strip()
    require(blob(source_commit,str(HERE.relative_to(ROOT)).replace("\\","/")+"/freeze.json")== (HERE/"freeze.json").read_bytes(),"prospective lock committed")
    write("release_receipt.json",{"scope":"one selective extraction and cached tokenizer-only lock; NO model execution",
        "source_commit":source_commit,"preflight":pre,"start_unix_time":time.time(),
        "authorization":"root conditional input-only release after source commit and preflight",
        "usage_receipt":"usage_receipt.json","single_attempt":True})
    stage="selective_extraction";status="INCONCLUSIVE";failure=None
    try:
        selected,selection=extract_locked_source();write("selected_cases.json",selected);write("selection_receipt.json",selection)
        p=packet();prompts=render_selected(selected,p["cohort.json"])
        write("inputs.json",{"schema":"final_inputs_exact.v1","prompts":prompts,"requests":p["plan.json"]["requests"],
            "source_order_unchanged":True,"ordinary_proofs_scoring_only":True,
            "historical_access":"NON_ACCESS_UNVERIFIED; no positive prior evaluation found in 185 metadata files is not proof of global non-access"})
        stage="cached_tokenizer_only"
        os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
        from transformers import AutoTokenizer
        import torch
        from token_lock import prove
        tokenizer=AutoTokenizer.from_pretrained(str(SNAPSHOT),local_files_only=True)
        require(sha(tokenizer.chat_template.encode())==TEMPLATE_SHA,"loaded tokenizer official template")
        write("chat_template.jinja",tokenizer.chat_template.encode(),raw=True)
        proofs=[]
        for prompt in prompts:
            proof=prove(tokenizer,torch,prompt["prompt"],prompt["token_map"])
            proof["prompt_id"]=prompt["prompt_id"]
            meta=write(f"tokens_{len(proofs)+1:02}.json",proof);proofs.append({"prompt_id":prompt["prompt_id"],"prompt_sha256":prompt["prompt_sha256"],"length":proof["prompt_length"],"tokens":meta})
        stage="input_envelope"
        config=read(SNAPSHOT/"config.json")["text_config"];lengths=[x["length"] for x in proofs]
        require(config["hidden_size"]==1024 and config["vocab_size"]==248320,"frozen later storage dimensions")
        require(max(lengths)<config["max_position_embeddings"],"all complete untruncated prefixes fit configured context")
        # Existing per-forward reserve stores only final-position state, not full sequence.
        encoded_bound=max(lengths)*16+6*1024*4
        require(encoded_bound<=262144,"complete IDs/masks plus final-state bookkeeping fits prior per-forward reserve")
        require(sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())<4*1024**2,"prospective input-output bound")
        write("input_lock.json",{"schema":"final_exact_input_token_lock.v1","status":"PASS_INPUT_ONLY",
            "inputs_sha256":sha((HERE/"inputs.json").read_bytes()),"prompts":proofs,"prompt_count":24,
            "semantic_count":18,"ordinary_count":6,"selected_case_count":9,
            "source_freeze_sha256":pre["source_freeze_sha256"],"source_commit":source_commit,
            "tokenizer_bindings_sha256":sha((HERE/"source_bindings.json").read_bytes()),
            "lengths":{"min":min(lengths),"max":max(lengths),"all":lengths},
            "storage_compatibility":{"status":"PASS_DIMENSIONS_ONLY","context_limit":config["max_position_embeddings"],
                "full_logits_float32_bytes":248320*4,"final_state_float32_bytes":1024*4,
                "ids_masks_and_six_final_vectors_bound":encoded_bound,"existing_per_forward_reserve":262144,
                "proposed_final_artifact_envelope_bytes":259715072,"proposed_final_namespace_bytes":288*1024**2},
            "runtime_compatibility":"Pinned template/config and exact lengths bound; tokenization is not a Qwen runtime calibration. The later real-runtime adapter/supervision release must check these lengths against its envelope; no model run enabled here.",
            "model_loads":0,"forwards":0,"derivatives":0,"activation_edits":0,"gate_scores":0,"fits":0})
        status="PASS_INPUT_ONLY"
    except Exception as exc:
        failure={"stage":stage,"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        write("failure.json",failure)
    finally:
        write("extraction_final.json",{"status":status,"elapsed_seconds":time.monotonic()-started,"completed_stage":stage,
            "failure":failure,"model_loads":0,"forwards":0,"derivatives":0,"gate_scores":0,"fits":0,
            "retry_permitted":False,"model_execution_enabled":False})
    print(json.dumps({"status":status,"elapsed_seconds":time.monotonic()-started,"failure":failure}))
    return 0 if status=="PASS_INPUT_ONLY" else 1

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("freeze","preflight","extract"));args=parser.parse_args()
    if args.command=="freeze":freeze()
    elif args.command=="preflight":print(json.dumps(preflight()))
    else:sys.exit(extract())
