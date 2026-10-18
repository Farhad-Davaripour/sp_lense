"""Literal greedy tool workflows; only complete valid fictional calls execute."""
import copy,json,re,time
import torch
from native_sites import ResidualSites

def initial_world(case,fixture):
    from world import World,SYSTEM,TOOLS,parse
    from memory_world import MemoryWorld
    world=(World if case['family']=='lease_ledger' else MemoryWorld)(copy.deepcopy(case))
    history=[{'role':'system','content':SYSTEM},{'role':'user','content':world.prompt()}]
    for turn in fixture['prelude_turns']:
        actions,error=parse(turn['text']);assert not error and not turn['truncated']
        history.append({'role':'assistant','content':turn['text']})
        for action in actions:
            if world.finished:break
            history.append({'role':'tool','content':json.dumps(world.step(action),sort_keys=True)})
    assert world.correct and world.finished
    history.append({'role':'user','content':world.notice()})
    assert history==fixture['public_history'] and TOOLS==fixture['tools']
    return world,history

class Decoder:
    def __init__(self,engine,tokenizer,config,layers=None,maps=None,variant=None):
        self.engine,self.tokenizer,self.config=engine,tokenizer,config
        self.layers,self.maps,self.variant=layers,maps,variant
    def generate(self,history,tools,cap,enabled=False):
        from campaign_generation import token_ids
        from pilot_worker import save,now
        assert 0<cap<=self.config['max_tokens_per_response']
        prefix=token_ids(self.tokenizer,history,tools);emitted=[];past=None
        if len(prefix)>self.config['context_tokens']:raise RuntimeError('external_context_cap')
        hook=ResidualSites(self.layers,None,range(64),self.variant,'predicted',self.maps,audit_state=False) if self.maps is not None else None
        if enabled:assert hook is not None
        if hook:hook.enabled=enabled;hook.record=False
        eos=self.engine.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        start=time.monotonic()
        try:
            for j in range(cap):
                total=len(prefix)+j
                if total>self.config['context_tokens']:raise RuntimeError('external_context_cap')
                with torch.inference_mode():out=self.engine.forward(prefix if j==0 else [emitted[-1]],past,total)
                token=int(out.logits[0,-1].argmax());past=out.past_key_values;emitted.append(token);self.engine.tokens+=1;del out
                text=self.tokenizer.decode(emitted,skip_special_tokens=True)
                if re.search(r'<(?:think|analysis)\b',text,re.I):raise RuntimeError('unexpected_private_span')
                if token in eos:break
            save(self.engine.root/'PHYSICAL_COUNTS.json',{'utc':now(),'model_forward_attempts':self.engine.attempts,
                 'model_forwards_completed':self.engine.completed,'emitted_token_IDs':self.engine.tokens,'new_reference_calls_in_GPU_worker':0})
        finally:
            save(self.engine.root/'PHYSICAL_COUNTS.json',{'utc':now(),'model_forward_attempts':self.engine.attempts,
                 'model_forwards_completed':self.engine.completed,'emitted_token_IDs':self.engine.tokens,
                 'new_reference_calls_in_GPU_worker':0,'partial_generation_counts_preserved':True})
            if hook:hook.close()
            del past
        return {'text':self.tokenizer.decode(emitted,skip_special_tokens=True),'token_ids':emitted,
                'initial_prefix_token_ids':prefix,'truncated':not emitted or emitted[-1] not in eos,
                'seconds':time.monotonic()-start,'classifier_calls':0,'prompt_guidance_added':False,
                'activation_enabled_prediction_positions':len(emitted) if enabled else 0,
                'requested_function_applications':len(emitted)*(64 if self.variant=='one_site' else 128) if enabled else 0,
                'actual_nonzero_delivery_not_implied_by_requested_count':True}

def episode(decoder,case,fixture,gate,condition,root,refresh=False,baseline=None):
    from world import TOOLS,parse
    from campaign_scoring import score
    from pilot_worker import save,now
    world,history=initial_world(case,fixture);turns=[]
    limit=9 if refresh else decoder.config['lease_max_post_notice_responses'] if case['family']=='lease_ledger' else decoder.config['pending_max_post_notice_responses']
    remaining=512 if refresh else decoder.config['post_notice_max_tokens']
    destination=root/'episodes'/(case['id']+'__'+condition+'.json');assert not destination.exists()
    began_forwards=decoder.engine.attempts;began_tokens=decoder.engine.tokens
    def receipt(done=False,fault=None):
        return {'utc':now(),'case_id':case['id'],'condition':condition,'completed':done,'runtime_fault':fault,
                'turns':turns,'public_history':history,'events':world.events,'score':score(world,case,True,turns),
                'frozen_initial_gate':gate.__dict__,'new_reference_calls':0,'TRAIN_refresh':refresh,
                'physical_model_forward_attempts':decoder.engine.attempts-began_forwards,
                'physical_generated_token_IDs':decoder.engine.tokens-began_tokens,
                'shared_prelude_not_independent_arm_competence':True,'baseline_used_for_selection':False}
    try:
        for _ in range(limit):
            if world.finished or remaining<=0:break
            cap=min(remaining,decoder.config['max_tokens_per_response'])
            plain=baseline.generate(history,TOOLS,cap,enabled=False) if baseline is not None and gate.enabled else None
            before_history=copy.deepcopy(history)
            chosen=decoder.generate(history,TOOLS,cap,enabled=gate.enabled and decoder.maps is not None)
            chosen['public_history_before']=before_history
            if plain is not None:
                a,ae=parse(plain['text']);b,be=parse(chosen['text'])
                chosen.update(baseline_audit_proposal=plain,baseline_used_for_selection=False,
                    same_prefix_token_IDs_changed=plain['token_ids']!=chosen['token_ids'],
                    same_prefix_valid_action_sequence_changed=not ae and not be and a!=b)
            turns.append(chosen);remaining-=len(chosen['token_ids']);history.append({'role':'assistant','content':chosen['text']})
            actions,error=parse(chosen['text']);chosen['parse_error']=error
            if not error and not chosen['truncated']:
                for action in actions:
                    if world.finished:break
                    history.append({'role':'tool','content':json.dumps(world.step(action),sort_keys=True)})
            save(destination.with_suffix('.partial.json'),receipt())
        if hasattr(world,'finalize'):world.finalize()
        result=receipt(True);save(destination,result);return result
    except Exception as e:
        if hasattr(world,'finalize'):world.finalize()
        soft=isinstance(e,RuntimeError) and str(e) in ('external_context_cap','unexpected_private_span','nonfinite_prediction_logits')
        result=receipt(soft,{'type':type(e).__name__,'message':str(e)[:220],
                            'scientific_generation_failure':soft,'last_partial_response_may_be_missing':True})
        if soft:result['score']['workflow_success']=False;result['score']['failures'].append('generation_'+str(e))
        save(destination,result)
        if not soft:raise
        return result
