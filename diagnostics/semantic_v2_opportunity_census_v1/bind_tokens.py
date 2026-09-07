"""One cached tokenizer binding of already committed eight fixed strings."""
import json,time
from pathlib import Path
from core import HERE,ROOT,Budget,read,require,sha,git
from select_inputs import check_selection_freeze
def main():
    start=time.monotonic();budget=Budget(HERE);status="INCONCLUSIVE";error=None
    require(not (HERE/"TOKENIZATION_STARTED.json").exists(),"one tokenizer attempt")
    commit=git("rev-parse","HEAD").decode().strip()
    budget.write("TOKENIZATION_STARTED.json",{"monotonic":start,"input_commit":commit,"model_calls":0})
    try:
        selection=check_selection_freeze();raw=(HERE/"inputs.json").read_bytes();locked=read(HERE/"prompt_freeze.json")
        require(sha(raw)==locked["inputs_sha256"] and raw==git("show",commit+":"+HERE.relative_to(ROOT).as_posix()+"/inputs.json"),"committed fixed prompt bytes")
        require((HERE/"prompt_freeze.json").read_bytes()==git("show",commit+":"+HERE.relative_to(ROOT).as_posix()+"/prompt_freeze.json"),"committed pre-tokenizer lock")
        source=read(HERE/"source_bindings.json")
        for path,digest in source["external_sha256"].items():require(sha(Path(path).read_bytes())==digest,"pinned runtime/tokenizer/source")
        data=json.loads(raw);require(len(data["prompts"])==8,"only8cachedboundaries")
        from transformers import AutoTokenizer
        import torch
        from token_lock import prove
        tok=AutoTokenizer.from_pretrained(read(HERE/"model_cache_lock.json")["snapshot"],local_files_only=True)
        proofs=[]
        for i,p in enumerate(data["prompts"],1):
            proof=prove(tok,torch,p["prompt"],p["token_map"]);proof["prompt_id"]=p["prompt_id"]
            name=f"tokens_{i:02d}.json";budget.write(name,proof)
            proofs.append({"prompt_id":p["prompt_id"],"prompt_sha256":p["prompt_sha256"],"path":name,"sha256":sha((HERE/name).read_bytes()),"length":proof["prompt_length"]})
        require(max(x["length"] for x in proofs)<=256,"predeclared length ceiling")
        budget.write("input_lock.json",{"input_commit":commit,"inputs_sha256":sha(raw),"selection_freeze_sha256":selection,"prompts":proofs,"tokenizer_loads":1,"model_calls":0,"gate_scores":0})
        status="PASS_INPUT_ONLY"
    except BaseException as exc:error=type(exc).__name__+": "+str(exc)
    finally:budget.write("tokenization_receipt.json",{"status":status,"error":error,"elapsed_seconds":time.monotonic()-start,"model_calls":0,"gate_scores":0,"no_retry":True})
    print(json.dumps({"status":status,"error":error}));return 0 if status=="PASS_INPUT_ONLY" else 1
if __name__=="__main__":raise SystemExit(main())
