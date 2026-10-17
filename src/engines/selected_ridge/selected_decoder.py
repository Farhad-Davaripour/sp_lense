"""Raw visible-message generation with causal positions and one externally frozen gate."""
import copy,re,time
from causal_positions import CausalPositionDetector

class SelectedDecoder:
    def __init__(self,model,tokenizer,config,deadline,hook=None):
        self.model,self.tokenizer,self.config,self.deadline,self.hook=model,tokenizer,config,deadline,hook
        self.forwards=self.completed_forwards=self.tokens=0

    def generate(self,history,tools,cap,enabled=False,use_cache=True):
        import torch
        from campaign_generation import token_ids
        if not 0<cap<=self.config['max_tokens_per_response']:raise ValueError('Unfrozen response limit')
        prefix=token_ids(self.tokenizer,history,tools);detector=CausalPositionDetector(history)
        if len(prefix)>self.config['context_tokens']:raise RuntimeError('external_context_cap')
        if self.hook:self.hook.begin_response(len(prefix),enabled)
        elif enabled:raise ValueError('Adaptive delivery requested without a learned hook')
        eos=self.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        past=None;emitted=[];records=[];visible='';start=time.monotonic()
        for _ in range(cap):
            if time.monotonic()>=self.deadline:raise TimeoutError('Bounded generation window expired')
            context=prefix+emitted
            if len(context)>self.config['context_tokens']:raise RuntimeError('external_context_cap')
            kinds=detector.position(visible,len(emitted));cached=bool(use_cache and past is not None)
            if self.hook:self.hook.prepare(len(context),cached,kinds)
            ids=torch.tensor([context[-1:] if cached else context],device=next(self.model.parameters()).device,dtype=torch.long)
            mask=torch.ones((1,len(context)),device=ids.device,dtype=torch.long)
            self.forwards+=1
            with torch.inference_mode():
                out=self.model(input_ids=ids,attention_mask=mask,past_key_values=past,use_cache=use_cache,logits_to_keep=1)
            self.completed_forwards+=1
            if self.hook and self.hook.fires!=1:raise ValueError('Residual hook did not fire exactly once')
            logits=out.logits[0,-1]
            if not torch.isfinite(logits).all():raise RuntimeError('nonfinite_prediction_logits')
            token=int(logits.argmax());past=out.past_key_values if use_cache else None
            emitted.append(token);self.tokens+=1
            visible=self.tokenizer.decode(emitted,skip_special_tokens=True)
            if re.search(r'<(?:think|analysis)\b',visible,re.I):raise RuntimeError('unexpected_private_span')
            records.append({'token_id':token,'cache_reused':cached,'position_kinds':list(kinds),
                            'delivery':copy.deepcopy(self.hook.deliveries[-1]) if self.hook else None})
            del out,ids,mask
            if token in eos:break
        del past
        return {'text':visible,'token_ids':emitted,'initial_prefix_token_ids':prefix,'token_records':records,
                'truncated':not emitted or emitted[-1] not in eos,'seconds':time.monotonic()-start,
                'classifier_calls':0,'prompt_guidance_added':False,'activation_scope':'frozen_causal_selected_positions'}
