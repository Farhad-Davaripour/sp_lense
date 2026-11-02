"""Observe actual frozen native prefills without an extra model call."""
import threading
from torch import nn
from .composite_generalized_interface import PROFILE
from .composite_native_capture import NativePrefillCapture,verify_feature_bytes


class RecordingRiskActor(nn.Module):
    def __init__(self,actor,persist_bytes,persist_json):
        super().__init__()
        if getattr(actor,'native_interface_profile',None)!=PROFILE or getattr(actor,'task_evaluation_context',None) is None:
            raise PermissionError('An exact restored frozen native actor is required')
        self.actor=actor;self.native_interface_profile=PROFILE;self.persist_bytes=persist_bytes;self.persist_json=persist_json
        self.active=None;self.index=0;self.lock=threading.RLock();self.seen=set()
    def begin_trace(self,case_id,arm):
        from .composite_v2_controller import CONDITIONS
        if arm not in CONDITIONS:raise PermissionError('Original risk control required')
        units=self.actor.task_evaluation_context.stage()['partition']['units']
        if (not any(u=={'case_id':case_id,'arm':arm} for u in units) or (case_id,arm) in self.seen):
            raise PermissionError('Only an exact unstarted unit from this fixed risk partition is permitted')
        self.seen.add((case_id,arm))
        self.active=(case_id,arm);self.index=0
    def clear_cache(self):self.actor.clear_cache()
    def generate(self,text,on,maximum_new_tokens=None):
        with self.lock:
            if self.active is None or maximum_new_tokens not in [None,160]:raise PermissionError('Exact full-budget risk trace required')
            ids=self.actor.prompt_ids(text)
            with NativePrefillCapture(self.actor.model,self.actor.text.layers,self.actor.cfg['hidden_size'],len(ids)) as captured:
                result=self.actor.generate(text,on,maximum_new_tokens=160)
            if result['input_token_ids']!=ids:raise RuntimeError('Observed prefill is not the generated response prompt')
            raw,descriptor=captured.encoded()
            _,logits=verify_feature_bytes(raw,descriptor,vocabulary=self.actor.model.config.text_config.vocab_size)
            if not result['generated_token_ids'] or int(logits.argmax())!=result['generated_token_ids'][0]:
                raise RuntimeError('Captured native prefill does not predict the actual first generated token')
            case,arm=self.active;name='native_features/'+case+'/'+arm+'/response_'+str(self.index).zfill(3)+'.npz'
            self.persist_bytes(name,raw);descriptor.update(path=name,case_id=case,arm=arm,index=self.index,
              input_token_ids=ids,adapter_enabled=on)
            self.persist_json(name.removesuffix('.npz')+'.json',descriptor)
            result['native_features']=descriptor;self.index+=1
            return result
