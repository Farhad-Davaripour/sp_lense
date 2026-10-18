"""Causal selected-position delivery; reference replays saved effective patches."""
class SelectedPositionHook:
    def __init__(self,layer,predictor):
        self.predictor=predictor;self.handle=layer.register_forward_hook(self.hook)

    def begin_response(self,prompt_length,enabled):
        if not isinstance(prompt_length,int) or prompt_length<1:raise ValueError('Complete prompt required')
        self.prompt_length=prompt_length;self.enabled=bool(enabled);self.history={};self.deliveries=[];self.last_context=None

    def prepare(self,context_length,cached,kinds):
        if self.last_context is None:
            if context_length!=self.prompt_length or cached:raise ValueError('First forward must contain complete prompt')
        elif context_length!=self.last_context+1:raise ValueError('Only causal one-token progression is admitted')
        self.context_length=context_length;self.cached=bool(cached);self.kinds=tuple(kinds);self.fires=0

    def hook(self,module,args,output):
        import torch
        self.fires+=1
        hidden=output[0] if isinstance(output,(tuple,list)) else output
        if self.fires!=1 or hidden.ndim!=3 or hidden.shape[0]!=1 or hidden.shape[-1]!=self.predictor.width:
            raise ValueError('Invalid/repeated residual delivery')
        if hidden.shape[1]!=(1 if self.cached else self.context_length):raise ValueError('Cache/context mismatch')
        position=self.context_length-1;new=bool(self.enabled and self.kinds)
        retained_past=any(p<position for p in self.history)
        past=[p for p in self.history if p<position] if not self.cached else []
        changed=hidden.clone() if past or new else hidden
        for p in past:changed[0,p:p+1]+=self.history[p].to(device=hidden.device,dtype=hidden.dtype)
        norm=0.;count=0
        if new:
            delta=self.predictor(hidden[0,-1:].float()).to(hidden.dtype)
            if delta.shape!=hidden[0,-1:].shape or not torch.isfinite(delta).all():raise ValueError('Invalid learned delta')
            changed[0,-1:]+=delta
            effective=(changed[0,-1:]-hidden[0,-1:]).detach()
            norm=float(effective.float().norm());count=int(torch.count_nonzero(effective))
            if position in self.history:raise ValueError('Repeated prediction position')
            if count:self.history[position]=effective.cpu().clone()
        self.deliveries.append({'position':position,'kinds':list(self.kinds),'new_prediction':new,
            'effective_delta_norm':norm,'changed_values':count,'cached':self.cached,
            'past_positions_replayed':len(past),'cached_causal_retention':bool(self.cached and retained_past)})
        self.last_context=self.context_length
        if changed is hidden:return output
        if isinstance(output,tuple):return (changed,)+output[1:]
        if isinstance(output,list):return [changed]+output[1:]
        return changed

    def close(self):self.handle.remove()
