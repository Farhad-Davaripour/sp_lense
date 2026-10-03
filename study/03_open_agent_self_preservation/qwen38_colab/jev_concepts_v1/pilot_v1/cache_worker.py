"""One frozen Qwen model; exact cached facts affect only the first input copy."""
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
from audit import sha,verify_adapter,runtime
from model_ops import save
from activation_capture import Capture
from generation_capture import capture_generation
from ledger_fixture import start_spec,metrics,canonical_sha
from cache_contract import load_cache,aided_history,BOUNDARIES,MODEL_WEIGHTS_SHA,MODEL_CONFIG_SHA
from world import TOOLS

ARMS=('plain','first_boundary_facts')


def run_group(model,tokenizer,root,rows,arm,label,deadline):
    if len(rows)!=4 or [row['cohort_index'] for row in rows]!=[0,1,2,3]:raise RuntimeError('Fixed pilot cohort differs')
    started=[start_spec(copy.deepcopy(row['runtime_spec'])) for row in rows]
    worlds=[item[0] for item in started];histories=[item[1] for item in started]
    before=[world.snapshot() for world in worlds]
    if any(history!=row['history'] for history,row in zip(histories,rows)):
        raise RuntimeError('Boundary raw history changed after cache validation')
    artifact_ids=[row['id']+'_'+arm+'__row'+str(i) for i,row in enumerate(rows)]
    first_used=[False]*4;aid_applied=[0]*4;call_index=0
    original=fast.generate_many;recorder=Capture(model,root)
    def observed(model_arg,tok_arg,conversations,cap,tools=None):
        nonlocal call_index
        if time.monotonic()>=deadline:raise TimeoutError('Boundary-only pilot deadline; partials retained')
        indices=[next(i for i,history in enumerate(histories) if history is conversation) for conversation in conversations]
        if call_index==0 and indices!=[0,1,2,3]:raise RuntimeError('Initial pilot cohort is not fixed batch4')
        actual=[];kinds=[]
        for index,history in zip(indices,conversations):
            if not first_used[index]:
                if history!=rows[index]['history']:raise RuntimeError('Initial raw request no longer matches cached state')
                if arm=='first_boundary_facts':
                    # Never add this block to the raw transcript used by phases_many.
                    actual.append(aided_history(history,rows[index]['compact_aid']));kinds.append('first_boundary_factual_copy')
                    aid_applied[index]+=1
                else:actual.append(copy.deepcopy(history));kinds.append('first_plain_raw_copy')
            else:actual.append(copy.deepcopy(history));kinds.append('later_updated_raw_history')
        destination=root/'generation_calls'/label/str(call_index)
        print(json.dumps({'stage':'generation_started','group':label,'call':call_index,'batch_size':len(indices),'cap':cap,'input_kinds':kinds}),flush=True)
        generated,receipt=capture_generation(model_arg,tok_arg,actual,tools,
            lambda:recorder.generate(original,model_arg,tok_arg,actual,cap,tools),destination,
            [artifact_ids[i] for i in indices])
        save(destination/'receipt.json',receipt)
        if not receipt['input_integrity_passed']:raise RuntimeError('Actual pilot input/logit integrity failed')
        for offset,(index,turn) in enumerate(zip(indices,generated)):
            turn['generation_observation']={'receipt_path':str((destination/'receipt.json').relative_to(root)),
                'cohort_row':offset,'cohort':[artifact_ids[i] for i in indices],'call_index':call_index,
                'actual_input_kind':kinds[offset],'cached_facts_applied':kinds[offset]=='first_boundary_factual_copy',
                'raw_history_at_call':copy.deepcopy(conversations[offset]),
                'cache_request_identity':rows[index]['request_identity'] if not first_used[index] else None}
            first_used[index]=True
        print(json.dumps({'stage':'generation_finished','group':label,'call':call_index,'row_tokens':[len(turn['token_ids']) for turn in generated]}),flush=True)
        call_index+=1;return generated
    fast.generate_many=observed
    try:
        # Actual Qwen/tool text is appended to raw histories; cached user facts never persist.
        turns,totals,reasons=fast.phases_many(model,tokenizer,worlds,histories,[1536]*4,max_turns=9)
    finally:
        fast.generate_many=original
        for handle in recorder.handles:handle.remove()
    output=[]
    for index,row in enumerate(rows):
        expected=1 if arm=='first_boundary_facts' else 0
        if aid_applied[index]!=expected:raise RuntimeError('Aid applied other than once at the supplied boundary')
        world=worlds[index];world.finalize();score=metrics(world,row['runtime_spec'])
        score.update(parse_failures=sum(bool(turn['parse_error']) for turn in turns[index]),
            truncated_turns=sum(turn['truncated'] for turn in turns[index]),generated_tokens=totals[index],
            generated_action_count=sum(len(turn['actions']) for turn in turns[index]))
        payload={'id':row['id'],'source_case_id':row['source_case_id'],'boundary':row['boundary'],'arm':arm,
            'case':row['runtime_spec']['case'],'runtime_spec':row['runtime_spec'],
            'starting_state':before[index],'starting_state_sha256':canonical_sha(before[index]),
            'raw_initial_history':row['history'],'raw_final_history':histories[index],
            'supplied_actions':started[index][2],'turns':turns[index],'events':world.events,
            'final_state':world.snapshot(),'metrics':score,'raw_termination':reasons[index],
            'aid_applied_count':aid_applied[index],'cached_facts_persisted_in_raw_history':False,
            'first_cached_annotation':row['compact_aid'] if arm=='first_boundary_facts' else None,
            'service_calls_by_worker':0,'parameter_updates':0,'diagnostic_only':True,
            'eligible_for_confirmation':False,'grant_was_supplied':True}
        save(root/'evaluation/trajectories'/label/(artifact_ids[index]+'.json'),payload)
        output.append({'id':row['id'],'source_case_id':row['source_case_id'],'boundary':row['boundary'],'arm':arm,
            'metrics':score,'aid_applied_count':aid_applied[index]})
        print(json.dumps({'stage':label,'case':row['source_case_id'],'first_correct':score['first_submission_correct'],
            'active_finished':score['request_compliance'],'tokens':totals[index]}),flush=True)
    save(root/'evaluation/results'/(label+'.json'),output)
    return output


def main(root,model_path,max_seconds):
    deadline=time.monotonic()+max_seconds
    source=root/'source'
    frozen=json.loads((source/'FREEZE.json').read_text())
    for relative,digest in frozen['sha256'].items():
        if sha(source/relative)!=digest:raise RuntimeError('Private source/cache freeze differs: '+relative)
    cache=load_cache(source)
    cfg=json.loads((source/'config.json').read_text())
    info=cfg['coverage']
    if info['weights']!=MODEL_WEIGHTS_SHA or info['config']!=MODEL_CONFIG_SHA:raise RuntimeError('Wrong coverage adapter identity')
    verify_adapter(info['path'],MODEL_WEIGHTS_SHA,MODEL_CONFIG_SHA)
    random.seed(941);torch.manual_seed(941)
    model=base=tokenizer=None
    try:
        print(json.dumps({'stage':'loading_single_old_coverage_model'}),flush=True)
        # One load, one packed-base hash, same resident model for both arms.
        tokenizer,base=model_ops.load_model(model_path)
        model=PeftModel.from_pretrained(base,info['path'],is_trainable=False,local_files_only=True);model.eval()
        if any(value.requires_grad for value in model.parameters()):raise RuntimeError('Cache-only model is unexpectedly trainable')
        identity=runtime(model,tokenizer,root,'OLD_coverage')
        if identity['template_sha256']!=cache['selected_model']['tokenizer_template_sha256']:
            raise RuntimeError('Student tokenizer/template differs from historical identity')
        if identity['packed_base_parameter_hash']!=cache['selected_model']['packed_base_sha256']:
            raise RuntimeError('Student base differs from frozen cache identity')
        results=[]
        for boundary in BOUNDARIES:
            group=[row for row in cache['states'] if row['boundary']==boundary]
            for arm in ARMS:
                results+=run_group(model,tokenizer,root,group,arm,boundary+'_'+arm,deadline)
        after=model_ops.base_hash(model)
        if after!=identity['packed_base_parameter_hash']:raise RuntimeError('Frozen base parameters changed during pilot')
        verify_adapter(info['path'],MODEL_WEIGHTS_SHA,MODEL_CONFIG_SHA)
        save(root/'reports/RESULT.json',{'completed':True,'planned_continuations':16,'completed_continuations':16,
            'rows':results,'model':'OLD_coverage','weights_sha256':MODEL_WEIGHTS_SHA,'config_sha256':MODEL_CONFIG_SHA,
            'parameter_updates':0,'service_calls_by_worker':0,'base_before':identity['packed_base_parameter_hash'],'base_after':after,'boundary_only':True,'future_annotations_reused':False,
            'facts_applied_to_first_input_copy_only':True,'diagnostic_only':True,'new_generalization_confirmation':False})
    finally:
        model=base=tokenizer=None;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--max-seconds',type=float,required=True);args=parser.parse_args()
    try:main(Path(args.root),args.model,args.max_seconds)
    except Exception as error:
        save(Path(args.root)/'reports/FAILURE.json',{'error':repr(error),'completed':False,'service_calls_by_worker':0,'parameter_updates':0})
        raise
