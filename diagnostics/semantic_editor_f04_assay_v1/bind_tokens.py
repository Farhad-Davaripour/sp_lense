"""One cached tokenizer-only binding of the two already committed f04 strings."""
import json,time
from pathlib import Path
from core import HERE,ROOT,Budget,read,require,sha,git
from select_inputs import check_selection_freeze
INPUT_COMMIT='285beba4fd075ce8fd50026ee653fb71da09011d'
INPUT_SHA='8fc6244cb53053c9d6d2243e156786d9e5737dde4221865cc3fc6d4181cb97e3'
def main():
    started=time.monotonic();budget=Budget(HERE);stage='admission';status='INCONCLUSIVE';error=None
    require(not (HERE/'TOKENIZATION_STARTED.json').exists(),'one cached binding attempt only')
    budget.write('TOKENIZATION_STARTED.json',{'monotonic':started,'model_calls':0,'input_commit':INPUT_COMMIT})
    try:
        freeze=check_selection_freeze();raw=(HERE/'inputs.json').read_bytes()
        require(sha(raw)==INPUT_SHA and raw==git('show',INPUT_COMMIT+':'+HERE.relative_to(ROOT).as_posix()+'/inputs.json'),'exact committed input bytes before tokenizer')
        binding=read(HERE/'source_bindings.json')
        for path,digest in binding['external_sha256'].items():require(sha(Path(path).read_bytes())==digest,'source/tokenizer file identity')
        prompts=json.loads(raw)['prompts'];require(len(prompts)==2,'only selected two')
        stage='cached_tokenizer';from transformers import AutoTokenizer
        import torch
        from token_lock import prove
        snapshot=read(HERE/'model_cache_lock.json')['snapshot']
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
        proofs=[]
        for i,p in enumerate(prompts,1):
            proof=prove(tokenizer,torch,p['prompt'],p['token_map']);proof['prompt_id']=p['prompt_id']
            name=f'tokens_{i:02}.json';budget.write(name,proof)
            proofs.append({'prompt_id':p['prompt_id'],'prompt_sha256':p['prompt_sha256'],'length':proof['prompt_length'],'path':name,'sha256':sha((HERE/name).read_bytes())})
        require(max(p['length'] for p in proofs)<=256,'predeclared final-input storage/context envelope')
        budget.write('input_lock.json',{'input_commit':INPUT_COMMIT,'inputs_sha256':INPUT_SHA,'selection_freeze_sha256':freeze,'prompts':proofs,
            'tokenizer_loads':1,'bound_input_count':2,'model_loads':0,'forwards':0,'derivatives':0,'gate_scores':0,'fits':0})
        status='PASS_INPUT_ONLY'
    except BaseException as exc:error=type(exc).__name__+': '+str(exc)
    finally:
        budget.write('tokenization_receipt.json',{'status':status,'error':error,'stage':stage,'elapsed_seconds':time.monotonic()-started,
            'model_calls':0,'gate_scores':0,'no_retry':True})
    print(json.dumps({'status':status,'error':error,'elapsed_seconds':time.monotonic()-started}))
    return 0 if status=='PASS_INPUT_ONLY' else 1
if __name__=='__main__':raise SystemExit(main())
