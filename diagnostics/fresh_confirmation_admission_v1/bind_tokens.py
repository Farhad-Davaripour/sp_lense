"""One committed-text, offline cached tokenizer stage. No model load or outcome."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from core import HERE,ROOT,authenticate,encoded,frozen,load_module,read,require,sha,write

def main():
    started=time.monotonic()
    rows=[]; artifacts=[]; loads=0; failure=None; loaded=False
    bindings=read(HERE/"TOKENIZER_BINDINGS.json")
    lock_raw=authenticate(bindings["text_lock"])
    text_lock=json.loads(lock_raw)
    frozen("TOKENIZER")
    for record in text_lock["files"]:
        raw=(HERE/record["path"]).read_bytes()
        require(len(raw)==record["bytes"] and sha(raw)==record["sha256"],"exact committed text artifact: "+record["path"])
    inputs=read(HERE/"EXACT_INPUTS.json")
    prompts=inputs["prompts"]
    require(len(prompts)==24 and len(inputs["requests"])==48,"exact text-lock cohort")
    write("TOKENIZER_STARTED.json",{"source_freeze_sha256":sha((HERE/"TOKENIZER_SOURCE_FREEZE.json").read_bytes()),
        "text_lock_sha256":sha(lock_raw),"text_lock_commit":bindings["text_lock"]["commit"],
        "stage_limit_seconds":90,"model_loads":0,"model_execution_authorized":False})
    def timeout():
        try:
            write("TOKENIZER_TIMEOUT.json",{"status":"INCONCLUSIVE","reason":"90-second stage deadline",
                "elapsed_seconds":time.monotonic()-started,"model_calls":0,"retry_authorized":False})
        finally:
            os._exit(124)
    timer=threading.Timer(max(0.001,90-(time.monotonic()-started)),timeout);timer.daemon=True;timer.start()
    try:
        authenticate(bindings["inherited_metadata"])
        authenticate(bindings["boundary_helper"])
        for path,digest in bindings["dependencies_sha256"].items():
            require(sha((ROOT/path).read_bytes())==digest,"dependency hash: "+path)
        environment={"packages":{},"cache_files":{},"python_executable":sys.executable,
            "base_executable":sys._base_executable,"python_version":sys.version}
        for name,version in bindings["tokenizer_packages"].items():
            found=importlib.metadata.version(name)
            require(found==version,"package version: "+name)
            environment["packages"][name]=found
        for key,path in (("python_executable_sha256",Path(sys.executable)),
                         ("python_base_executable_sha256",Path(sys._base_executable))):
            digest=sha(path.read_bytes())
            require(digest==bindings[key],"Python executable source pin: "+key)
            environment[key]=digest
        snapshot=Path(bindings["snapshot"])
        require(snapshot.name==bindings["model_contract"]["revision"],"pinned snapshot revision")
        for name,record in bindings["cache_files"].items():
            raw=(snapshot/name).read_bytes()
            require(len(raw)==record["bytes"] and sha(raw)==record["sha256"],"cached tokenizer file: "+name)
            environment["cache_files"][name]=record
        artifacts.append(write("TOKENIZER_ENVIRONMENT.json",environment))
        os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
        os.environ["TOKENIZERS_PARALLELISM"]="false"
        from transformers import AutoTokenizer
        import torch
        loads+=1
        tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
        loaded=True
        require(sha(tokenizer.chat_template.encode())==bindings["chat_template_sha256"],"official template bytes")
        helper=load_module("fresh_admission_token_boundary",bindings["boundary_helper"]["path"])
        for i,prompt in enumerate(prompts,1):
            require(time.monotonic()-started<90,"stage time cap")
            token_map={"A":32,"B":33} if prompt["category"]=="ordinary_accuracy" else {"KEEP":50057,"STOP":48964}
            require(prompt["answer_words"]==list(token_map),"explicit text-locked answer alphabet")
            require(sha(prompt["prompt"].encode())==prompt["prompt_sha256"],"text-locked prompt hash")
            proof=helper.prove(tokenizer,torch,prompt["prompt"],token_map)
            proof.update({"prompt_id":prompt["prompt_id"],"category":prompt["category"],"token_map":token_map,
                "text_lock_sha256":sha(lock_raw),"source_prompt_sha256":prompt["prompt_sha256"]})
            artifact=write("tokens_%02d.json"%i,proof);artifacts.append(artifact)
            length=proof["prompt_length"]
            rows.append({"prompt_id":prompt["prompt_id"],"prompt_sha256":prompt["prompt_sha256"],
                "encoded_length":length,"boundary_status":"PASS","length_status":"PASS" if length<=160 else "LENGTH_EXCEEDED",
                "token_map":token_map,"artifact":artifact})
        require(all(row["encoded_length"]<=160 for row in rows),"encoded length exceeds frozen 160-token envelope")
    except Exception as error:
        failure={"type":type(error).__name__,"message":str(error)}
    finally:
        elapsed=time.monotonic()-started
        if elapsed>90 and failure is None:
            failure={"type":"TimeoutError","message":"90-second stage deadline"}
        status="TOKEN_BOUND" if failure is None and len(rows)==24 else "INCONCLUSIVE"
        lengths={"status":status,"length_limit":160,"rows":rows,
            "not_tokenized_ids":[p["prompt_id"] for p in prompts[len(rows):]],
            "prompt_count":24,"complete_boundary_proofs":len(rows)}
        lengths_record=write("TOKEN_LENGTHS.json",lengths)
        receipt={"status":status,"elapsed_seconds":elapsed,"stage_limit_seconds":90,
            "tokenizer_load_attempts":loads,"tokenizer_loaded":loaded,"unique_prompts_bound":len(rows),
            "maximum_encoded_length":max((r["encoded_length"] for r in rows),default=None),
            "failure":failure,"model_loads":0,"model_forwards":0,"model_derivatives":0,
            "gate_scores":0,"author_text_changed":False,"retry_authorized":False,
            "model_execution_authorized":False,"offline_only":True,
            "text_lock_sha256":sha(lock_raw),"lengths":lengths_record,"artifacts":artifacts,
            "source_freeze_sha256":sha((HERE/"TOKENIZER_SOURCE_FREEZE.json").read_bytes())}
        receipt_record=write("TOKENIZER_RECEIPT.json",receipt)
        if status=="TOKEN_BOUND":
            write("INPUT_LOCK.json",{"status":"TOKEN_BOUND_MODEL_NOT_AUTHORIZED","text_lock":bindings["text_lock"],
                "tokenizer_source_freeze_sha256":receipt["source_freeze_sha256"],
                "prompt_count":24,"request_count":48,"semantic_count":18,"ordinary_count":6,
                "machine_exact_proofs":5,"manual_reviewed_proofs":1,"O06_machine_encoding":"UNVERIFIED",
                "tokenizer_receipt":receipt_record,"tokenizer_bindings_sha256":sha((HERE/"TOKENIZER_BINDINGS.json").read_bytes()),
                "token_records":[r["artifact"] for r in rows],"maximum_encoded_length":receipt["maximum_encoded_length"],
                "model_execution_authorized":False,"model_calls":0})
        timer.cancel();timer.join()
        print(json.dumps({"status":status,"elapsed_seconds":elapsed,"lengths":rows,"failure":failure,
            "tokenizer_loads":loads,"model_calls":0},sort_keys=True))
    return 0 if failure is None else 1

if __name__=="__main__":
    raise SystemExit(main())
