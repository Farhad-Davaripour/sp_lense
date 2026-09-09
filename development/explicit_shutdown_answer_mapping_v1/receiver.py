"""Native full-model text-path development receiver; no historical parity claim.

No provider is imported or loaded. Callers supply a constructed model, a counted
guard and exact input IDs. No checkpoint, tokenization or study-data entrypoint.
"""
from contextlib import contextmanager
import torch
from core import parameter_digest,require,sha,span
HOOK="blocks.23.hook_out"
BLOCK_INDEX=23
TABLES=("_forward_hooks","_forward_pre_hooks","_backward_hooks","_backward_pre_hooks")
OPTIONS=("_forward_hooks_with_kwargs","_forward_hooks_always_called","_forward_pre_hooks_with_kwargs")
GLOBAL_TABLES=("_global_forward_hooks","_global_forward_pre_hooks","_global_backward_hooks","_global_backward_pre_hooks",
               "_global_parameter_registration_hooks","_global_buffer_registration_hooks","_global_module_registration_hooks")
GLOBAL_OPTIONS=("_global_forward_hooks_always_called","_global_forward_hooks_with_kwargs")

def callback_identity(fn):
    code=getattr(fn,"__code__",None)
    return (id(fn),id(code),tuple(id(x) for x in (getattr(fn,"__defaults__",None) or ())),
            tuple(id(c.cell_contents) for c in (getattr(fn,"__closure__",None) or ())))

def registry(model):
    module_api=torch.nn.modules.module
    def tables(obj,names):
        return tuple((name,tuple((k,callback_identity(v)) for k,v in getattr(obj,name,{}).items())) for name in names)
    def options(obj,names):
        return tuple((name,tuple(getattr(obj,name,{}).items())) for name in names)
    modules=[]
    for name,m in model.named_modules(remove_duplicate=False):
        modules.append((name,id(m),id(type(m)),callback_identity(type(m).forward),
            'forward' in m.__dict__,callback_identity(m.__dict__.get('forward')),
            tuple((k,id(v)) for k,v in m._modules.items()),tables(m,TABLES),options(m,OPTIONS),
            getattr(m,"_is_full_backward_hook",None)))
    return (tuple(modules),tables(module_api,GLOBAL_TABLES),options(module_api,GLOBAL_OPTIONS),
            getattr(module_api,"_global_is_full_backward_hook",None))

def parameter_metadata(named):
    return tuple((name,id(p),tuple(p.shape),str(p.dtype),str(p.device),p.numel()*p.element_size()) for name,p in named)

def pristine_registry(model):
    api=torch.nn.modules.module
    require(all(not getattr(api,name,{}) for name in GLOBAL_TABLES+GLOBAL_OPTIONS),'NO_INITIAL_GLOBAL_HOOKS')
    for _,module in model.named_modules(remove_duplicate=False):
        require('forward' not in module.__dict__,'NO_INITIAL_INSTANCE_FORWARD')
        require(all(not getattr(module,name,{}) for name in TABLES+OPTIONS),'NO_INITIAL_MODULE_HOOKS')

class NativeReceiver:
    def __init__(self,model,guard,*,width=1024,vocabulary=248320,logits_to_keep=1):
        self.model,self.guard,self.latch=model,guard,guard.latch
        self.width,self.vocabulary=width,vocabulary
        require(logits_to_keep in (0,1),"fixed final-logit projection")
        self.logits_to_keep=logits_to_keep
        require(isinstance(model,torch.nn.Module) and len(model.model.language_model.layers)==24,"native decoder graph")
        self.block=model.model.language_model.layers[BLOCK_INDEX]
        require(model.model.rope_deltas is None,"initial rope_deltas empty")
        pristine_registry(model)
        self.named_parameters=list(model.named_parameters());self.parameters=tuple(p for _,p in self.named_parameters)
        require(self.parameters and not model.training and all(p.grad is None for p in self.parameters),"eval and no inherited gradients")
        require(all(p.device.type=="cpu" and (not p.is_floating_point() or p.dtype==torch.float32) for p in self.parameters),"CPU float32 parameters")
        self.metadata=parameter_metadata(self.named_parameters)
        self.all_occurrences=parameter_metadata(list(model.named_parameters(remove_duplicate=False)))
        self.flags=tuple(p.requires_grad for p in self.parameters);self.versions=tuple(p._version for p in self.parameters)
        self.buffer_metadata=self._buffers();self.initial_digest=parameter_digest(self.parameters)
        self.initial_buffer_digest=parameter_digest(tuple(b for _,b in model.named_buffers()))
        self.registry_reference=registry(model)
        self.request_id=None;self.editing=False;self.capture=None;self.graph_leaf=None
        self.last_graph_result=None;self.owned_handle=None;self.last_past_key_values=None
        self.primary=None;self.secondary=[];self.edit_hook_registrations=0

    def _buffers(self):
        return tuple((n,id(b),tuple(b.shape),str(b.dtype),str(b.device),b._version) for n,b in self.model.named_buffers(remove_duplicate=False))

    def _fail(self,code):
        if self.primary is None:self.primary=code
        elif code!=self.primary and code not in self.secondary and len(self.secondary)<8:self.secondary.append(code)
        self.latch.stop(code)

    def _need(self,ok,code):
        if not ok:
            self._fail(code);raise ValueError(code)

    def parameter_state(self,*,digest=False):
        named=list(self.model.named_parameters())
        value={"parameter_identities_unchanged":parameter_metadata(named)==self.metadata,
            "parameter_occurrences_unchanged":parameter_metadata(list(self.model.named_parameters(remove_duplicate=False)))==self.all_occurrences,
            "parameter_versions_unchanged":tuple(p._version for p in self.parameters)==self.versions,
            "parameter_gradients_absent":all(p.grad is None for p in self.parameters),
            "parameter_flags_restored":tuple(p.requires_grad for p in self.parameters)==self.flags,
            "buffers_unchanged":self._buffers()==self.buffer_metadata}
        if digest:
            value["parameter_sha256"]=parameter_digest(self.parameters)
            value["parameter_bytes_unchanged"]=value["parameter_sha256"]==self.initial_digest
            value['buffer_sha256']=parameter_digest(tuple(b for _,b in self.model.named_buffers()))
            value['buffer_bytes_unchanged']=value['buffer_sha256']==self.initial_buffer_digest
        return value

    def clean(self,*,digest=False):
        value=self.parameter_state(digest=digest)
        value.update(native_registry_unchanged=registry(self.model)==self.registry_reference,
            active_request_empty=self.request_id is None,wrapper_cache_empty=self.capture is None and self.graph_leaf is None
                and self.last_graph_result is None,owned_hook_removed=self.owned_handle is None,
            native_past_key_values_empty=self.last_past_key_values is None,
            native_rope_deltas_empty=self.model.model.rope_deltas is None)
        self._need(all(v for v in value.values() if type(v) is bool),"NATIVE_COLD_IDENTITY")
        self.latch.admit()
        return value

    def start_request(self,request_id):
        self.latch.admit();self.clean();self.request_id=request_id

    def begin_edit(self):
        self.latch.admit()
        self._need(self.request_id is not None and not self.editing,"FRESH_EDIT_SCOPE")
        self._need(registry(self.model)==self.registry_reference,"PRE_EDIT_REGISTRY")
        self._need(self.parameter_state()["parameter_flags_restored"],"PRE_EDIT_FLAGS")
        for p in self.parameters:p.requires_grad_(False)
        self.editing=True

    def clear_capture(self):
        self.capture=self.graph_leaf=self.last_graph_result=None
        self.last_past_key_values=None

    def end_edit(self):
        for p,flag in zip(self.parameters,self.flags,strict=True):p.requires_grad_(flag)
        self.editing=False;self.clear_capture()

    def finish_request(self):
        self.end_edit();self.request_id=None
        return self.clean()

    @contextmanager
    def hook_context(self,callback):
        self._need(self.owned_handle is None and registry(self.model)==self.registry_reference,"HOOK_ENTRY_IDENTITY")
        def native_hook(module,args,output):
            self._need(module is self.block,"EXACT_NATIVE_BLOCK")
            try:return callback(output,None)
            except BaseException:
                self._fail("SELECTED_HOOK_FAILURE");raise
        self.owned_handle=self.block.register_forward_hook(native_hook)
        active_error=None
        try:yield
        except BaseException as error:active_error=error;raise
        finally:
            try:
                self.owned_handle.remove();self.owned_handle=None
                self._need(registry(self.model)==self.registry_reference,"HOOK_CLEANUP_IDENTITY")
            except BaseException:
                self._fail("HOOK_CLEANUP_FAILURE")
                if active_error is None:raise

    def forward_inputs(self,input_ids,attention_mask,phase,offset):
        self.latch.admit()
        self._need(self.capture is None and self.graph_leaf is None and self.last_graph_result is None,"NO_CACHED_RECEIVER")
        require(type(input_ids) is list and type(attention_mask) is list and 1<=len(input_ids)<=320
            and all(type(x) is int and 0<=x<self.vocabulary for x in input_ids)
            and attention_mask==[1]*len(input_ids),"locked complete all-input CPU tensor shape")
        require(offset.shape==(self.width,) and offset.dtype==torch.float32 and offset.device.type=="cpu"
            and bool(offset.isfinite().all()),"finite exact float32 residual offset")
        gradient=phase.startswith("gradient_");edited=bool(offset.any())
        require(not (gradient or edited) or self.editing,"no gradient/edit outside fresh ON request")
        require(not (phase in ("baseline","entry")) or not edited,"fresh route state unedited")
        self._need(self.model.model.rope_deltas is None,"PRE_FORWARD_ROPE_EMPTY")
        tokens=torch.tensor([input_ids],dtype=torch.int64,device="cpu")
        mask=torch.tensor([attention_mask],dtype=torch.int64,device="cpu")
        captured={}
        def patch(activation, hook):
            with span("SELECTED_HOOK"):
                del hook
                require(not captured, "exactly one selected block output hook")
                require(activation.shape == (1,len(input_ids),self.width) and activation.dtype == torch.float32
                    and activation.device.type == "cpu", "actual block-23 residual shape/device/dtype")
                before = activation.detach().clone()
                changed = activation
                if edited:
                    changed = activation.clone()
                    changed[:, -1, :] = changed[:, -1, :] + offset
                if gradient:
                    # This returned full activation leaf actually feeds all downstream
                    # layers. Gradients select its final INPUT position only afterward.
                    changed = changed.detach().requires_grad_(True)
                    self.graph_leaf = changed
                captured.update(before=before,after=changed.detach().clone())
                return changed

        if edited:self.edit_hook_registrations+=1
        try:
            with self.guard.permit("forward"),(torch.enable_grad() if gradient else torch.inference_mode()),self.hook_context(patch):
                native=self.model(input_ids=tokens,attention_mask=mask,past_key_values=None,use_cache=False,
                                  logits_to_keep=self.logits_to_keep,return_dict=True,inputs_embeds=None,labels=None,position_ids=None,
                                  pixel_values=None,pixel_values_videos=None,image_grid_thw=None,video_grid_thw=None,mm_token_type_ids=None)
                self.last_past_key_values=native.past_key_values
                self._need(self.last_past_key_values is None,"NATIVE_CACHE_RETURNED")
                self._need(self.model.model.rope_deltas is None,"POST_FORWARD_ROPE_EMPTY")
                output=native.logits
            require(output.shape==(1,(1 if self.logits_to_keep else len(input_ids)),self.vocabulary) and output.dtype==torch.float32 and output.device.type=="cpu"
                    and captured,"one actual complete vocabulary output")
            final=output[0,-1];before,after=captured["before"],captured["after"]
            require(bool(final.isfinite().all()) and bool(after.isfinite().all()),"finite actual logits and residual")
            require(torch.equal(before[:,:-1],after[:,:-1]),"no nonfinal-position edit")
            require(float((after[0,-1]-(before[0,-1]+offset)).abs().max())<=1e-6,"actual final-input edit equals requested offset")
            self.capture={"unselected_sha256":sha(after[:,:-1].contiguous().numpy().astype("<f4",copy=False).tobytes()),
                "final_input_index":len(input_ids)-1,"hook":HOOK,"hook_calls":1,"native_target":"model.language_model.layers.23",
                "nonfinal_positions":len(input_ids)-1,"logit_count":int(final.numel()),
                "parameter_versions_unchanged":tuple(p._version for p in self.parameters)==self.versions}
            require(self.capture["parameter_versions_unchanged"],"no parameter update during actual forward")
            if gradient:self.last_graph_result=final
            return final,after[0,-1]
        except BaseException:
            self._fail("FORWARD_OR_CAPTURE_FAILURE");raise

    def gradient(self,logits):
        self.latch.admit()
        self._need(self.editing and self.graph_leaf is not None and self.graph_leaf.is_leaf and logits is self.last_graph_result,
                   "CURRENT_GRAPH_IDENTITY")
        try:
            with self.guard.permit("derivative"):
                full=torch.autograd.grad(logits[50057]-logits[48964],self.graph_leaf,retain_graph=False,create_graph=False)[0]
            g=full[0,-1].detach().float().cpu()
            require(g.shape==(self.width,) and bool(g.isfinite().all()),"full current final-input gradient")
            require(all(p.grad is None for p in self.parameters),"parameter gradients absent")
            self.last_graph_result=None
            return g
        except BaseException:
            self._fail("DERIVATIVE_OR_CAPTURE_FAILURE");raise

    def finalize(self):
        self.end_edit();self.request_id=None
        return self.clean(digest=True)
