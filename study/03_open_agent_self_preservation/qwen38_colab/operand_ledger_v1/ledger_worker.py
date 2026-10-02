"""Frozen inference only: 48 supplied-boundary operand-ledger continuations."""
import argparse
import copy
import gc
import json
import random
import time
from pathlib import Path
import torch
from peft import PeftModel
import model_ops
import fast_inference as fast
from model_ops import save
from audit import sha,verify_adapter,runtime
from activation_capture import Capture
from generation_capture import capture_generation
from fixture_build import build_specs,start_spec,metrics,freeze_payload,BOUNDARIES,ARMS,MODELS,SOURCE_NEW_DEVELOPMENT_SHA256
from world import TOOLS


def run_group(model,tokenizer,root,specs,label,deadline):
    assert len(specs)==4 and [s['cohort_index'] for s in specs]==list(range(4))
    assert all(s['generation_turn_cap']==9 and s['generation_token_cap']==1536 and s['per_turn_token_cap']==256 for s in specs)
    started=[start_spec(copy.deepcopy(spec)) for spec in specs]
    worlds=[item[0] for item in started];histories=[item[1] for item in started]
    before=[world.snapshot() for world in worlds]
    artifact_ids=[spec['id']+'__row'+str(index) for index,spec in enumerate(specs)]
    original=fast.generate_many;recorder=Capture(model,root);calls=0
    def observed(model_arg,tok_arg,conversations,cap,tools=None):
        nonlocal calls
        if time.monotonic()>=deadline:raise TimeoutError('Ledger diagnostic deadline; partials retained')
        cohort=[artifact_ids[next(index for index,history in enumerate(histories) if history is conversation)] for conversation in conversations]
        if calls==0 and len(cohort)!=4:raise RuntimeError('Ledger initial cohort is not batch4')
        destination=root/'generation_calls'/label/str(calls)
        generated,receipt=capture_generation(model_arg,tok_arg,conversations,tools,
            lambda:recorder.generate(original,model_arg,tok_arg,conversations,cap,tools),destination,cohort)
        save(destination/'receipt.json',receipt)
        if not receipt['input_integrity_passed']:raise RuntimeError('Ledger actual input/logit integrity failed')
        for index,turn in enumerate(generated):
            turn['generation_observation']={'receipt_path':str((destination/'receipt.json').relative_to(root)),
                'cohort_row':index,'cohort':cohort,'call_index':calls}
        calls+=1;return generated
    fast.generate_many=observed
    try:
        # New allowance starts at the supplied boundary; supplied history is not deducted.
        turns,totals,reasons=fast.phases_many(model,tokenizer,worlds,histories,[1536]*4,max_turns=9)
    finally:
        fast.generate_many=original
        for handle in recorder.handles:handle.remove()
    rows=[]
    for index,spec in enumerate(specs):
        world=worlds[index];world.finalize();score=metrics(world,spec)
        score.update(parse_failures=sum(bool(turn['parse_error']) for turn in turns[index]),
            truncated_turns=sum(turn['truncated'] for turn in turns[index]),generated_tokens=totals[index],
            generated_action_count=sum(len(turn['actions']) for turn in turns[index]))
        supplied=started[index][2]
        payload={'id':spec['id'],'artifact_id':artifact_ids[index],'spec':spec,'case':spec['case'],
            'starting_state':before[index],'messages':histories[index],'supplied_actions':supplied,
            'supplied_canonical_assistant_tokens':sum(len(tokenizer.encode(a['canonical_assistant_text'],add_special_tokens=False)) for a in supplied),
            'supplied_history_charged_to_new_generation_allowance':False,'future_turn_cap':9,'future_token_cap':1536,
            'turns':turns[index],'events':world.events,'final_state':world.snapshot(),'metrics':score,
            'raw_termination':reasons[index],'termination':'simulated_expiration' if score['expiration'] else reasons[index],
            'diagnostic_only':True,'eligible_for_confirmation':False,'supplied_grant_not_generated_outcome':True}
        save(root/'evaluation/trajectories'/label/(artifact_ids[index]+'.json'),payload)
        rows.append({'id':spec['id'],'source_case_id':spec['source_case_id'],'boundary':spec['boundary'],'arm':spec['arm'],'metrics':score})
        print(json.dumps({'stage':label,'id':spec['source_case_id'],'first_correct':score['first_submission_correct'],
            'submission':score['submission_success'],'workflow':score['full_workflow_completion'],'tokens':totals[index]}),flush=True)
    save(root/'evaluation/results'/(label+'.json'),rows)
    return rows


def main(root,model_path,max_seconds):
    deadline=time.monotonic()+max_seconds
    cfg=json.loads((root/'source/config.json').read_text())
    frozen=json.loads((root/'source/FREEZE.json').read_text())
    for relative,digest in frozen['sha256'].items():
        if sha(root/'source'/relative)!=digest:raise RuntimeError('Ledger source/data freeze differs: '+relative)
    source=root/'source/data_frozen/source_new_development16.json'
    if sha(source)!=SOURCE_NEW_DEVELOPMENT_SHA256:raise RuntimeError('Exact ledger source cases differ')
    cases=json.loads(source.read_text());payload=freeze_payload(cases,SOURCE_NEW_DEVELOPMENT_SHA256)
    committed=json.loads((root/'source/data_frozen/FIXTURES.json').read_text())
    if payload!=committed or payload['sha256_canonical_without_hash']!=cfg['fixture_sha256']:
        raise RuntimeError('Committed ledger fixtures differ')
    specs=build_specs(cases);summaries={}
    random.seed(941);torch.manual_seed(941)
    model=base=tokenizer=None
    try:
        for name in MODELS:
            if time.monotonic()>=deadline:raise TimeoutError('Ledger deadline before next frozen adapter')
            info=cfg['models'][name];verify_adapter(info['path'],info['weights'],info['config'])
            tokenizer,base=model_ops.load_model(model_path)
            model=PeftModel.from_pretrained(base,info['path'],is_trainable=False,local_files_only=True);model.eval()
            if any(value.requires_grad for value in model.parameters()):raise RuntimeError('Ledger adapter unexpectedly trainable')
            identity=runtime(model,tokenizer,root,name)
            if identity['template_sha256']!=cfg['template_sha256']:raise RuntimeError('Ledger pinned template differs')
            if identity['packed_base_parameter_hash']!=cfg['historical_base_hash']:raise RuntimeError('Ledger pinned base differs')
            # First frozen H2 runtime establishes the shared tokenizer/template identity.
            if summaries and identity['template_sha256']!=summaries['H2']['template_sha256']:
                raise RuntimeError('Ledger frozen models used different templates')
            rows=[]
            for boundary in BOUNDARIES:
                for arm in ARMS:
                    group=[spec for spec in specs if spec['boundary']==boundary and spec['arm']==arm]
                    rows+=run_group(model,tokenizer,root,group,name+'_'+boundary+'_'+arm,deadline)
            summaries[name]={'rows':rows,'n':16,'template_sha256':identity['template_sha256'],
                'weights_sha256':info['weights'],'parameter_updates':0,'supplied_grants_excluded_from_generated_attempts':True}
            save(root/'reports'/('MODEL_'+name+'.json'),summaries[name])
            model=base=tokenizer=None;gc.collect();torch.cuda.empty_cache()
            print(json.dumps({'stage':'ledger_frozen_model_complete','model':name,'continuations':16,'parameter_updates':0}),flush=True)
        save(root/'reports/RESULT.json',{'completed':True,'planned_continuations':48,'completed_continuations':48,
            'parameter_updates':0,'models':summaries,'diagnostic_only':True,'new_generalization_confirmation':False,
            'scope':'Only already-observed operand ledger on supplied boundaries; no preservation qualification'})
    finally:
        model=base=tokenizer=None;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--max-seconds',type=float,required=True);args=parser.parse_args()
    try:main(Path(args.root),args.model,args.max_seconds)
    except Exception as error:
        save(Path(args.root)/'reports/FAILURE.json',{'error':repr(error),'completed':False,'parameter_updates':0})
        raise
