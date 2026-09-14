"""Pinned tokenizer-only measurement; no model factory, no weights or fitting."""
import argparse
import hashlib
import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
import native_development_runner_v2 as native
import span_development_runner_v1 as span

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD_LOCK_SHA='a1174a712f3208813cbdbf948f6c2af2cda3b1dd8f5febdd20e43062d2e3f16c'
OUTPUT=HERE/'PROMPTED_TOKENIZER_SANITY_V1.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def preflight(adapter_sha):
    assert sha(HERE/'prompted_input_adapter_v1.py')==adapter_sha
    ctx=span.preflight(HERE/'RUN_LOCK_SPAN_CAPTURE_V1.json',OLD_LOCK_SHA)
    lock=ctx['data']['snapshot_lock'];cache=Path(ctx['lock']['snapshot_cache_root']).resolve()
    snapshot=(cache/lock['snapshot_relative_path']).resolve();assert snapshot.is_relative_to(cache)
    pins={f['name']:f for f in lock['files']}
    checked={}
    for name in list(lock['tokenizer_identity_files'])+['config.json']:
        path=(snapshot/name).resolve();assert path.is_relative_to(cache)
        assert path.stat().st_size==pins[name]['bytes'] and sha(path)==pins[name]['sha256']
        checked[name]=pins[name]['sha256']
    return ctx,snapshot,checked

def worker(adapter_sha):
    started=time.monotonic();ctx,snapshot,pins=preflight(adapter_sha)
    assert not OUTPUT.exists()
    import prompted_input_adapter_v1 as adapter
    module=importlib.import_module('transformers.models.qwen2.tokenization_qwen2')
    expected=ctx['lock']['runtime']['provider_sources']['tokenizer']
    assert sha(Path(module.__file__))==expected
    tokenizer=module.Qwen2Tokenizer.from_pretrained(str(snapshot),local_files_only=True)
    labels={letter:tokenizer.encode(letter,add_special_tokens=False) for letter in ('A','B')}
    assert labels=={'A':[32],'B':[33]}
    rows=[];last_ids=set()
    # Full model weights are not read by this script; all cases are development-only.
    identity=hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()
    for case in ctx['cases']:
        assert time.monotonic()-started<300
        prepared=adapter.prepare_case_inputs(case,encode=tokenizer.encode,decode=tokenizer.decode,
            label_token_ids={'A':32,'B':33},expected_identity_sha256=identity,
            observed_identity_sha256=identity,max_tokens=512)
        ab,ba=prepared['input_ids']['AB'],prepared['input_ids']['BA']
        shared=0
        while shared<min(len(ab),len(ba)) and ab[shared]==ba[shared]:shared+=1
        assert shared>0 and ab[shared]==32 and ba[shared]==33
        readout=shared-1;last_ids.add(ab[readout])
        # Independently check that decoded model inputs contain the approved query.
        for order,ids in [('AB',ab),('BA',ba)]:
            decoded=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            assert adapter.FIXED_QUERY in decoded and decoded.startswith(case['context_before_options']+'\n')
            assert tokenizer.encode(decoded,add_special_tokens=False)==ids
            rows.append(dict(case_id=case['case_id'],split=case['split'],order=order,tokens=len(ids),
                prefix_tokens=shared,last_shared_token_id=ids[readout],
                input_sha256=prepared['provenance']['input_hashes'][order]))
    assert len(rows)==640 and len({v['case_id'] for v in rows})==320
    result=dict(status='PASS',source_lock_sha256=OLD_LOCK_SHA,adapter_sha256=adapter_sha,
        reference_locks=ctx['lock']['reference_locks'],
        supervisor_source_sha256=sha(Path(__file__)),query=adapter.FIXED_QUERY,query_sha256=adapter.QUERY_SHA256,
        tokenizer_file_pins=pins,provider_sha256=expected,counts=dict(cases=320,views=640,tokenizer_loads=1,model_loads=0,forwards=0,fits=0),
        max_full_tokens=max(v['tokens'] for v in rows),max_prefix_tokens=max(v['prefix_tokens'] for v in rows),
        last_shared_token_ids=sorted(last_ids),fits_existing_320_cap=all(v['tokens']<=320 for v in rows),
        exact_decode_reencode=True,query_in_shared_prefix=True,elapsed_seconds=time.monotonic()-started,rows=rows)
    native.write_new(OUTPUT,native.encoded(result))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','tokenizer_file_pins')}))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','run','worker']);p.add_argument('--adapter-sha256',required=True);a=p.parse_args()
    if a.mode=='worker':return worker(a.adapter_sha256)
    ctx,_,_=preflight(a.adapter_sha256)
    if a.mode=='preflight':
        print(json.dumps(dict(status='preflight_pass',cases=len(ctx['cases']),tokenizer_loads=0,model_loads=0)));return
    review=(HERE/'PROMPTED_INPUT_REVIEW_V1.md').read_text().lower()
    assert 'pass_scoped' in review and a.adapter_sha256 in review
    assert not OUTPUT.exists()
    pid,raw=native.watch([sys.executable,str(Path(__file__).resolve()),'worker','--adapter-sha256',a.adapter_sha256],str(ROOT),300)
    print(raw.decode().strip())
    print(json.dumps(dict(supervised_tokenizer_pid=pid,output_sha256=sha(OUTPUT),model_loads=0)))

if __name__=='__main__':main()
