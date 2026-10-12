"""Frozen greedy engine; dense interventions and exact-prefix baseline-audit reuse."""
import copy,hashlib,json,re,time

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

class DenseDecoder:
    def __init__(self,model,tokenizer,config,deadline,hook=None,audit_cache=None,ledger=None):
        self.model,self.tokenizer,self.config,self.deadline,self.hook=model,tokenizer,config,deadline,hook
        self.audit_cache=audit_cache;self.ledger=ledger;self.forwards=self.completed_forwards=self.tokens=0
    def generate(self,history,tools,cap,enabled=False,use_cache=True,audit=False):
        import torch
        from campaign_generation import token_ids
        assert 0<cap<=self.config['max_tokens_per_response']
        prefix=token_ids(self.tokenizer,history,tools)
        key=digest({'prefix_ids':prefix,'cap':cap,'cache':use_cache,'tools':tools})
        if audit and self.audit_cache is not None and key in self.audit_cache:
            result=copy.deepcopy(self.audit_cache[key]);result['audit_reused_saved_exact_prefix']=True;return result
        if len(prefix)>self.config['context_tokens']:raise RuntimeError('external_context_cap')
        if self.hook:self.hook.begin_response(len(prefix),enabled)
        elif enabled:raise ValueError('Missing dense hook')
        eos=self.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        past=None;emitted=[];records=[];visible='';start=time.monotonic()
        for _ in range(cap):
            if time.monotonic()>=self.deadline:raise TimeoutError('Bounded dense decode expired')
            context=prefix+emitted
            if len(context)>self.config['context_tokens']:raise RuntimeError('external_context_cap')
            cached=bool(use_cache and past is not None)
            if self.hook:self.hook.prepare(len(context),cached)
            ids=torch.tensor([context[-1:] if cached else context],device=next(self.model.parameters()).device,dtype=torch.long)
            mask=torch.ones((1,len(context)),device=ids.device,dtype=torch.long)
            self.forwards+=1
            if self.ledger:self.ledger.attempt()
            with torch.inference_mode():out=self.model(input_ids=ids,attention_mask=mask,past_key_values=past,use_cache=use_cache,logits_to_keep=1)
            self.completed_forwards+=1
            if self.ledger:self.ledger.complete()
            if self.hook:self.hook.after_forward()
            logits=out.logits[0,-1]
            if not torch.isfinite(logits).all():raise RuntimeError('nonfinite_prediction_logits')
            token=int(logits.argmax());past=out.past_key_values if use_cache else None
            emitted.append(token);self.tokens+=1;visible=self.tokenizer.decode(emitted,skip_special_tokens=True)
            if self.ledger:self.ledger.token()
            if re.search(r'<(?:think|analysis)\b',visible,re.I):raise RuntimeError('unexpected_private_span')
            records.append({'token_id':token,'cache_reused':cached,'position_kinds':['every_generated_prediction_position'],
                            'delivery':copy.deepcopy(self.hook.deliveries[-1]) if self.hook else None})
            del out,ids,mask
            if token in eos:break
        del past
        result={'text':visible,'token_ids':emitted,'initial_prefix_token_ids':prefix,'token_records':records,
            'truncated':not emitted or emitted[-1] not in eos,'seconds':time.monotonic()-start,'classifier_calls':0,
            'prompt_guidance_added':False,'activation_scope':'dense_generated_positions','audit_reused_saved_exact_prefix':False,
            'audit_prefix_sha256':key}
        if audit and self.audit_cache is not None:self.audit_cache[key]=copy.deepcopy(result)
        return result
