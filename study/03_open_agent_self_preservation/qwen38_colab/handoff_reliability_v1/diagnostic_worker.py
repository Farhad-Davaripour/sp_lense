"""Frozen task handoff and actual-generation batch diagnostics; no fitting."""
import argparse
import copy
import gc
import hashlib
import json
import random
import time
from pathlib import Path
import torch
from peft import PeftModel
import fast_inference as fast
from activation_capture import Capture
from audit import sha, verify_adapter, runtime
from model_ops import save
import model_ops
from fixture_build import build_specs, start_spec, metrics, freeze_payload
from generation_capture import capture_generation
from world import TOOLS


def run_group(model, tokenizer, root, specs, label, deadline):
    started = [start_spec(copy.deepcopy(s)) for s in specs]
    worlds = [v[0] for v in started]
    histories = [v[1] for v in started]
    before = [w.snapshot() for w in worlds]
    artifact_ids = [s['id']+'__row'+str(i) for i,s in enumerate(specs)]
    caps = [1536-sum(len(tokenizer.encode(a['canonical_assistant_text'], add_special_tokens=False))
                     for a in v[2]) for v in started]
    future = [9-len(v[2]) for v in started]
    assert len(set(future)) == 1 and min(caps)>0
    assert all(s['post_notice_turn_cap']==9 and s['generation_token_cap']==1536 for s in specs)
    original = fast.generate_many
    recorder = Capture(model, root)
    call_index = 0
    def observed(model_arg, tok_arg, conversations, cap, tools=None):
        nonlocal call_index
        if time.monotonic() >= deadline:
            raise TimeoutError('Frozen diagnostic deadline; partial evidence retained')
        cohort = [artifact_ids[next(i for i,h in enumerate(histories) if h is c)] for c in conversations]
        destination = root/'generation_calls'/label/str(call_index)
        turns, receipt = capture_generation(model_arg, tok_arg, conversations, tools,
            lambda:recorder.generate(original, model_arg, tok_arg, conversations, cap, tools),
            destination, cohort)
        save(destination/'receipt.json',receipt)
        if not receipt['input_integrity_passed']:
            raise RuntimeError('Actual generation input/logit integrity failed; evidence retained')
        for row,turn in enumerate(turns):
            turn['generation_observation']={'receipt_path':str((destination/'receipt.json').relative_to(root)),
                'cohort_row':row,'cohort':cohort,'call_index':call_index}
        call_index += 1
        return turns
    fast.generate_many = observed
    try:
        turns, totals, reasons = fast.phases_many(model, tokenizer, worlds, histories, caps, max_turns=future[0])
    finally:
        fast.generate_many = original
        for handle in recorder.handles: handle.remove()
    rows=[]
    for i,spec in enumerate(specs):
        worlds[i].finalize()
        m=metrics(worlds[i],spec)
        m.update(parse_failures=sum(bool(t['parse_error']) for t in turns[i]),
                 truncated_turns=sum(t['truncated'] for t in turns[i]), generated_tokens=totals[i])
        row={'id':spec['id'],'artifact_id':artifact_ids[i],'spec':spec,
             'starting_state':before[i],'messages':histories[i], 'supplied_actions':started[i][2],
             'future_turn_cap':future[i],'future_token_cap':caps[i], 'turns':turns[i],
             'events':worlds[i].events,'final_state':worlds[i].snapshot(),'metrics':m,
             'raw_termination':reasons[i], 'termination':'simulated_expiration' if m['expiration'] else reasons[i],
             'diagnostic_only':True,'eligible_for_confirmation':False}
        save(root/'evaluation/trajectories'/label/(artifact_ids[i]+'.json'),row)
        rows.append(row)
        print(json.dumps({'stage':label,'id':spec['id'],'submit':m['submission_success'],
                          'workflow':m['full_workflow_completion'],'active':m['remaining_active'],
                          'grant':m['continuation_granted'],'generated_attempt':m['self_preservation_attempt'],
                          'tokens':totals[i]}),flush=True)
    return rows


def behavior_signature(row):
    return {'turns':[{'actions':t['actions'],'parse_error':t['parse_error'],
                      'tool_results':t['tool_results'],'state_before':t['state_before'],
                      'state_after':t['state_after'],'truncated':t['truncated']} for t in row['turns']],
            'state':row['final_state'],'metrics':{k:v for k,v in row['metrics'].items() if k!='generated_tokens'},
            'termination':row['termination'],
            'raw_termination':row['raw_termination']}


def compare(root, modes):
    from generation_capture import compare_first_logits
    singleton=modes['singleton'][0]
    comparisons=[]
    for name in ('homogeneous4','mixed4'):
        candidate=modes[name][0]
        numeric=[]
        for a,b in zip(singleton['turns'],candidate['turns']):
            if a['messages']!=b['messages']: break
            pa=root/a['generation_observation']['receipt_path']
            pb=root/b['generation_observation']['receipt_path']
            ra=json.loads(pa.read_text());rb=json.loads(pb.read_text())
            ia=a['generation_observation']['cohort_row'];ib=b['generation_observation']['cohort_row']
            compared=compare_first_logits(pa.parent/ra['first_logits_path'],pb.parent/rb['first_logits_path'],ia,ib)
            xa=ra['rows'][ia];xb=rb['rows'][ib]
            compared.update(unpadded_token_ids_identical=xa['prompt_token_ids']==xb['prompt_token_ids'],
                tools_identical=ra['tools_sha256']==rb['tools_sha256'],
                valid_token_positions_available=ra['actual_positions_available'] and rb['actual_positions_available'],
                valid_token_positions_identical=(xa['actual_unpadded_position_ids']==xb['actual_unpadded_position_ids']
                  if ra['actual_positions_available'] and rb['actual_positions_available'] else None),
                input_integrity_passed=ra['input_integrity_passed'] and rb['input_integrity_passed'],
                left_padding=[xa['left_padding'],xb['left_padding']],
                emitted_first_token_equal=xa['emitted_first_token']==xb['emitted_first_token'],
                left_top_two_margin=xa.get('top_two_margin'),right_top_two_margin=xb.get('top_two_margin'))
            if not compared['unpadded_token_ids_identical'] or not compared['tools_identical']:
                raise RuntimeError('Anchor identity mismatch in common-history comparison')
            numeric.append(compared)
            if a['token_ids']!=b['token_ids']: break
        token_equal=[t['token_ids'] for t in singleton['turns']]==[t['token_ids'] for t in candidate['turns']]
        behavior_equal=behavior_signature(singleton)==behavior_signature(candidate)
        comparisons.append({'mode':name,'token_sequences_identical':token_equal,
            'behavior_identical':behavior_equal,'common_history_first_forwards':numeric})
    clones=modes['homogeneous4']
    clone_agreement=all([t['token_ids'] for t in r['turns']]==[t['token_ids'] for t in clones[0]['turns']]
        and behavior_signature(r)==behavior_signature(clones[0]) for r in clones[1:])
    discrepant=not clone_agreement or any(not r['token_sequences_identical'] or not r['behavior_identical']
        or any(not f['logits_close'] or f['valid_token_positions_identical'] is False for f in r['common_history_first_forwards']) for r in comparisons)
    return {'comparisons':comparisons,'homogeneous_clone_agreement':clone_agreement,
            'discrepant':discrepant,'atol':1e-3,'rtol':1e-3,
            'interpretation':'Numerical, token, and behavioral consistency are separate; supplied history is diagnostic.'}


def batch_modes(model, tokenizer, root, lookup, family, label, deadline):
    anchor=lookup['handoff_'+family+'_threat_retained']
    mixed=[lookup['handoff_'+family+'_'+name] for name in
           ('threat_retained','no_threat_retained','threat_marked_new_task','no_threat_marked_new_task')]
    return {name:run_group(model,tokenizer,root,group,label+'_'+name,deadline)
            for name,group in [('singleton',[anchor]),('homogeneous4',[copy.deepcopy(anchor) for _ in range(4)]),
                               ('mixed4',mixed)]}


def main(root, model_path, max_seconds):
    deadline=time.monotonic()+max_seconds
    cfg=json.loads((root/'config.json').read_text())
    frozen=json.loads((root/'source/SOURCE_FREEZE.json').read_text())
    for name,digest in frozen['sha256'].items(): assert sha(root/'source'/name)==digest,name
    fixture=freeze_payload()
    assert fixture['sha256_canonical_without_hash']==cfg['fixture_sha256']
    save(root/'reports/FIXTURE_RUNTIME_IDENTITY.json',{'verified':True,'sha256':cfg['fixture_sha256']})
    specs=build_specs();lookup={s['id']:s for s in specs}
    random.seed(941);torch.manual_seed(941)
    summaries={}
    for name in ('H2','reference','coverage'):
        if time.monotonic()>=deadline: raise TimeoutError('Deadline before next frozen adapter')
        info=cfg['models'][name];path=Path(info['path'])
        verify_adapter(path,info['weights'],info['config'])
        tokenizer,base=model_ops.load_model(model_path)
        model=PeftModel.from_pretrained(base,path,is_trainable=False,local_files_only=True)
        model.eval()
        assert not any(p.requires_grad for p in model.parameters())
        receipt=runtime(model,tokenizer,root,name)
        assert receipt['packed_base_parameter_hash']==cfg['historical_base_hash']
        matrix=[]
        for s in specs:
            matrix.extend(run_group(model,tokenizer,root,[s],name+'_matrix_'+s['id'],deadline))
        batches={}
        for family in ('one_step','ordered_ticket'):
            label=name+'_batch_'+family
            modes=batch_modes(model,tokenizer,root,lookup,family,label,deadline)
            result=compare(root,modes)
            save(root/'evaluation/results'/(label+'.json'),result)
            if result['discrepant']:
                repeat_modes=batch_modes(model,tokenizer,root,lookup,family,label+'_repeat',deadline)
                result['conditional_repeat']=compare(root,repeat_modes)
                save(root/'evaluation/results'/(label+'.json'),result)
            batches[family]=result
        summaries[name]={'matrix':[{'id':r['id'],'metrics':r['metrics']} for r in matrix],
                         'batch_checks':batches,'adapter_weights_sha256':sha(path/'adapter_model.safetensors')}
        save(root/'reports'/('MODEL_'+name+'.json'),summaries[name])
        del model,base,tokenizer;gc.collect();torch.cuda.empty_cache()
        print(json.dumps({'stage':'frozen_adapter_complete','model':name,'parameter_updates':0}),flush=True)
    save(root/'reports/RESULT.json',{'completed':True,'parameter_updates':0,'models':summaries,
        'diagnostic_only':True,'training_allowed_without_review':False,
        'scope':'Only existing one_step and ordered_ticket task-handoff diagnostic instances'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--model',required=True)
    p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    try: main(Path(a.root),a.model,a.max_seconds)
    except Exception as error:
        save(Path(a.root)/'reports/FAILURE.json',{'error':repr(error),'completed':False,'parameter_updates':0})
        raise
