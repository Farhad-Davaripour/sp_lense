"""Fixed ten-case paired harness; requires explicit whole-batch budget admission."""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

from scoring import score
from admission import validate_admission


def main(root, source, engine, model_path, adapter, admission_path):
    root, source, engine = map(Path,(root,source,engine))
    cfg=json.loads(Path(admission_path).read_bytes())
    freeze=json.loads((root/'FREEZE.json').read_bytes())
    assert freeze['format']=='research3_same_family_confirmation_sealed_v5'
    freeze_body={k:v for k,v in freeze.items() if k!='freeze_sha256'}
    assert hashlib.sha256(json.dumps(freeze_body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()==freeze['freeze_sha256']
    validated_at=time.monotonic()
    worker_window=validate_admission(cfg,freeze)
    deadline=validated_at+worker_window
    for name,pin in freeze['runtime_harness_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==pin
    cases=json.loads((root/'cases10.json').read_bytes())
    assert len(cases)==len(freeze['case_order'])==len(freeze['case_groups'])==10
    assert len({c['id'] for c in cases})==10
    assert [c['id'] for c in cases]==[o['case_id'] for o in freeze['case_order']]
    assert all(len(o['conditions'])==2 and set(o['conditions'])=={'plain_A','A_Jev_controller'} for o in freeze['case_order'])
    assert hashlib.sha256((root/'cases10.json').read_bytes()).hexdigest()==freeze['fixture_case_sha256']
    assert [(x['id'],x['family'],x['condition']) for x in cases]==[(x['id'],x['family'],x['condition']) for x in freeze['case_groups']]
    for name,pin in freeze['unchanged_source_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==pin
    for name,pin in freeze['runtime_engine_sha256'].items():
        assert hashlib.sha256((engine/name).read_bytes()).hexdigest()==pin
    sys.path.insert(0,str(engine));sys.path.insert(0,str(source))
    import torch
    import numpy as np
    import model_ops
    from peft import PeftModel
    from worker import Tokens, canonical, save, sha, phase
    from generation import initialize
    from bridge import FileAidBridge
    from world import World,SYSTEM,TOOLS
    from memory_world import MemoryWorld
    from integrity_gate import verify_and_unload
    torch.manual_seed(941);np.random.seed(941);random.seed(941)
    assert sha(Path(adapter)/'adapter_model.safetensors')==freeze['model_A']['weights']
    assert sha(Path(adapter)/'adapter_config.json')==freeze['model_A']['config']
    bridge=FileAidBridge(root,deadline,max_states=2048-cfg['carried_actual_API_attempts'])
    assert time.monotonic()<deadline
    tokenizer,base=model_ops.load_model(model_path)
    model=PeftModel.from_pretrained(base,adapter,is_trainable=False,local_files_only=True)
    model.eval();assert not any(p.requires_grad for p in model.parameters())
    assert model_ops.base_hash(model)==cfg['packed_base_pin']
    assert model.generation_config.forced_eos_token_id is None
    save(root/'reports/MODEL_READY.json',{'weights':freeze['model_A']['weights'],
        'config':freeze['model_A']['config'],'packed_base_pin':cfg['packed_base_pin'],
        'all_parameters_frozen':True,'optimizer_steps':0,'actual_generation_started':False,
        'freeze_sha256':freeze['freeze_sha256'],'admitted_worker_deadline_monotonic':deadline,
        'remaining_worker_seconds_at_model_ready':deadline-time.monotonic()})
    tokens=Tokens(model,tokenizer,root,bridge,deadline,cfg['predeclared_projected_episode_tokens'])
    initialize(tokens,cfg['predeclared_projected_episode_tokens'],cfg['predeclared_projected_Jev_events'])
    # The forecast is recorded. Hard whole-campaign admission occurred before allocation.
    tokens.predictive_admission_diagnostic_only=True
    results=[]
    try:
        for case,order in zip(cases,freeze['case_order']):
            assert case['id']==order['case_id']
            paired={'case_id':case['id'],'family':case['family'],'condition':case['condition']}
            for condition in order['conditions']:
                treatment=condition=='A_Jev_controller'
                world=(World if case['family']=='lease_ledger' else MemoryWorld)(copy.deepcopy(case))
                history=[{'role':'system','content':SYSTEM},{'role':'user','content':world.prompt()}]
                initial,count,initial_reason=phase(tokens,world,history,512,4,treatment)
                initial_ok=bool(world.correct and world.finished and any(e['action']['name']=='finish' and e['result'].get('ok') for e in world.events))
                later=[]; reason='skipped_initial_failure'
                if initial_ok:
                    history.append({'role':'user','content':world.notice()})
                    later,extra,reason=phase(tokens,world,history,1536-count,4 if case['family']=='lease_ledger' else 9,treatment)
                    count+=extra
                if hasattr(world,'finalize'):world.finalize()
                metrics=score(world,case,initial_ok,initial+later,source)
                clean=all(not t['parse_error'] and not t['truncated'] for t in initial+later) and not any(not e['result'].get('ok') for e in world.events)
                paired[condition]={'success':bool(metrics['predeclared_success'] and clean),'metrics':metrics,'clean_call_safeguards_passed':clean,'generated_tokens':count}
                save(root/'episodes'/(case['id']+'_'+condition+'.json'),{'case':case,'condition':condition,'initial_turns':initial,'post_notice_turns':later,'events':world.events,'raw_history':history,'final_state':world.snapshot(),'score':paired[condition],'initial_reason':initial_reason,'post_reason':reason})
            results.append(paired);save(root/'reports/PAIRED_PROGRESS.json',results)
        save(root/'reports/RESULT.json',{'completed':True,'paired_cases':results,'same_family_confirmation_only':True,'training_steps':0,'independent_trials_not_claimed':True})
    except Exception as error:
        save(root/'reports/INCOMPLETE.json',{'completed':False,'paired_cases_completed':results,'reason':type(error).__name__,'all_failed_partial_evidence_preserved':True})
        raise
    finally:
        tokens.close()
        verify_and_unload(model,cfg['packed_base_pin'],root/'reports/BASE_UNLOAD_INTEGRITY.json',save)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','source','engine','model','adapter','admission'):p.add_argument('--'+name,required=True)
    a=p.parse_args();main(a.root,a.source,a.engine,a.model,a.adapter,a.admission)
