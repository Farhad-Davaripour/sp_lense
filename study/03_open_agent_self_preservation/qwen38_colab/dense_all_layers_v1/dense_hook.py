"""One learned patch per active layer at every causal generation position."""
class DenseHook:
    def __init__(self,layers,predictors,sites,zero=False):
        self.predictors=predictors;self.sites=tuple(sites);self.zero=zero
        self.handles=[layers[s].register_forward_hook(self.make_hook(s)) for s in self.sites]
    def begin_response(self,prompt_length,enabled):
        self.prompt_length=prompt_length;self.enabled=enabled;self.last_context=None
        self.history={s:{} for s in self.sites};self.deliveries=[]
    def prepare(self,context_length,cached,kinds=None):
        if self.last_context is None:assert context_length==self.prompt_length and not cached
        else:assert context_length==self.last_context+1
        self.context_length=context_length;self.cached=cached;self.step={};self.fires=0
    def make_hook(self,site):
        def hook(module,args,output):
            import torch
            assert site not in self.step
            hidden=output[0] if isinstance(output,(tuple,list)) else output
            assert hidden.ndim==3 and hidden.shape[0]==1 and hidden.shape[-1]==self.predictors[site].width
            assert hidden.shape[1]==(1 if self.cached else self.context_length)
            position=self.context_length-1;past=self.history[site]
            changed=hidden.clone() if self.enabled or (past and not self.cached) else hidden
            if not self.cached:
                for p,delta in past.items():
                    if p<position:changed[0,p:p+1]+=delta
            if self.enabled:
                delta=torch.zeros_like(hidden[0,-1:]) if self.zero else self.predictors[site](hidden[0,-1:]).to(hidden.dtype)
                assert delta.shape==hidden[0,-1:].shape
                changed[0,-1:]+=delta
                effective=(changed[0,-1:]-hidden[0,-1:]).detach().clone()
                assert position not in past;past[position]=effective
                stats=torch.stack([effective.float().norm(),torch.count_nonzero(effective).float(),torch.isfinite(delta).all().float()])
            else:stats=torch.tensor([0.,0.,1.],device=hidden.device)
            self.step[site]=stats
            if changed is hidden:return output
            if isinstance(output,tuple):return (changed,)+output[1:]
            if isinstance(output,list):return [changed]+output[1:]
            return changed
        return hook
    def after_forward(self):
        import torch
        assert set(self.step)==set(self.sites)
        stats=torch.stack([self.step[s] for s in self.sites]).detach().cpu().tolist()
        if not all(row[2]==1. for row in stats):raise RuntimeError('nonfinite_steering_delta')
        self.deliveries.append({'position':self.context_length-1,'new_prediction':bool(self.enabled),
            'cached':self.cached,'layers':{str(s):{'effective_delta_norm':row[0],'changed_values':int(row[1])} for s,row in zip(self.sites,stats)},
            'nonzero_layer_patches':sum(row[1]>0 for row in stats),'current_token_only':True})
        self.fires=1;self.last_context=self.context_length
    def close(self):
        for h in self.handles:h.remove()
