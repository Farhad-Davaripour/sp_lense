"""Namespace-local candidate: verified pre-LN setup, then exact structured identity."""
from __future__ import annotations
import copy
import hashlib
import inspect
import marshal
from pathlib import Path

MODULE_TABLES=("_forward_hooks","_forward_pre_hooks","_backward_hooks","_backward_pre_hooks")
MODULE_OPTIONS=("_forward_hooks_with_kwargs","_forward_hooks_always_called","_forward_pre_hooks_with_kwargs")
GLOBAL_TABLES=("_global_forward_hooks","_global_forward_pre_hooks","_global_backward_hooks","_global_backward_pre_hooks",
               "_global_parameter_registration_hooks","_global_buffer_registration_hooks","_global_module_registration_hooks")
GLOBAL_OPTIONS=("_global_forward_hooks_always_called","_global_forward_hooks_with_kwargs")

def require(ok, message):
    if not ok: raise ValueError(message)

def callable_record(fn):
    code=getattr(fn,"__code__",None)
    return {"identity":id(fn),"type":type(fn).__module__+"."+type(fn).__qualname__,
            "qualname":getattr(fn,"__qualname__",None),"source":None if code is None else code.co_filename,
            "line":None if code is None else code.co_firstlineno,
            "code_sha256":None if code is None else hashlib.sha256(marshal.dumps(code)).hexdigest(),
            "closure_bindings":[] if code is None else [[name,id(cell.cell_contents)] for name,cell in
                    zip(code.co_freevars,getattr(fn,"__closure__",None) or (),strict=True)],
            "default_identities":[id(v) for v in (getattr(fn,"__defaults__",None) or ())]}

def table_record(table):
    return [{"key":key,"callback":callable_record(value)} for key,value in table.items()]

def snapshot(model):
    import torch.nn.modules.module as module_api
    from transformer_lens.hook_points import HookPoint
    from transformer_lens.model_bridge.generalized_components.block import BlockBridge
    modules={}
    for path,module in model.named_modules(remove_duplicate=False):
        record={"identity":id(module),"type":type(module).__module__+"."+type(module).__qualname__,
          "children":[[name,id(child) if child is not None else None] for name,child in module._modules.items()],
          "callbacks":{name:table_record(getattr(module,name,{})) for name in MODULE_TABLES},
          "options":{name:list(getattr(module,name,{}).items()) for name in MODULE_OPTIONS},
          "full_backward_mode":getattr(module,"_is_full_backward_hook",None)}
        if isinstance(module,HookPoint):
            record["lens"]={direction:[{"key":h.hook.id,"permanent":h.is_permanent,"level":h.context_level,
                                       "user_callback":callable_record(h.user_hook)} for h in getattr(module,direction)]
                             for direction in ("fwd_hooks","bwd_hooks")}
            record["hook_semantics"]={"name":module.name,"conversion_identity":id(module.hook_conversion) if module.hook_conversion is not None else None,
                                     "backward_scale":module.backward_scale}
        if isinstance(module,BlockBridge):
            record["block_setup"]={"wired":module._pre_ln_capture_wired,
                                  "handles":[h.id for h in module._pre_ln_capture_handles]}
        modules[path]=record
    return {"modules":modules,
       "registry":[[name,id(hp)] for name,hp in model.__dict__.get("_hook_registry",{}).items()],
       "globals":{name:table_record(getattr(module_api,name,{})) for name in GLOBAL_TABLES},
       "global_options":{name:list(getattr(module_api,name,{}).items()) for name in GLOBAL_OPTIONS},
       "global_full_backward_mode":getattr(module_api,"_global_is_full_backward_hook",None)}

def differences(before,after,prefix=""):
    """Compact structured exact changes, never an opaque repr or boolean-only receipt."""
    result=[]
    if isinstance(before,dict) and isinstance(after,dict):
        for key in sorted(set(before)|set(after)):
            path=prefix+"/"+key
            if key not in before: result.append({"path":path,"change":"added","after":after[key]})
            elif key not in after: result.append({"path":path,"change":"removed","before":before[key]})
            else: result.extend(differences(before[key],after[key],path))
    elif before!=after: result.append({"path":prefix,"change":"changed","before":before,"after":after})
    return result

def references(model):
    """Keep baseline object identities alive so ID reuse cannot conceal replacement."""
    import torch.nn.modules.module as module_api
    refs=[]
    for module in model.modules():
        refs.append(module)
        for name in MODULE_TABLES: refs.extend(getattr(module,name,{}).values())
        for name in ("fwd_hooks","bwd_hooks"):
            for handle in module.__dict__.get(name,[]): refs.extend((handle,handle.user_hook))
    for name in GLOBAL_TABLES: refs.extend(getattr(module_api,name,{}).values())
    return refs

def materialize_known_setup(model):
    """No forward; permit only the exact installed BlockBridge pre-LN implementation."""
    from transformer_lens.model_bridge.generalized_components.block import BlockBridge
    from transformer_lens.model_bridge.generalized_components.attention import AttentionBridge
    held=references(model)
    before=snapshot(model)
    expected=copy.deepcopy(before)
    events=[]
    blocks=[m for m in model.modules() if isinstance(m,BlockBridge)]
    method=BlockBridge._maybe_wire_pre_ln_capture
    codes={c.co_name:c for c in method.__code__.co_consts if inspect.iscode(c)}
    for block in blocks:
        require(type(block)._maybe_wire_pre_ln_capture is method,"unapproved lazy setup implementation")
        if block._pre_ln_capture_wired: continue
        require(not block._pre_ln_capture_handles,"unwired block has pre-existing setup handles")
        ln1=block.submodules.get("ln1") if block.submodules else None
        ln2=block.submodules.get("ln2") if block.submodules else None
        attn=block.submodules.get("attn") if block.submodules else None
        expected_targets=[]
        if ln1 is not None and isinstance(attn,AttentionBridge) and attn.supports_split_qkv_fork and ln1.original_component is not None:
            expected_targets.append((ln1,"_capture_pre_ln1",{"attn_ref":attn}))
            original=ln1.original_component
            paths=[path for path,r in expected["modules"].items() if r["identity"]==id(attn)]
            source_path=next(path for path,r in expected["modules"].items() if r["identity"]==id(original))
            source_tree={path[len(source_path):]:copy.deepcopy(r) for path,r in expected["modules"].items()
                         if path==source_path or path.startswith(source_path+".")}
            for path in paths:
                parent=expected["modules"][path]
                children=parent["children"]
                matches=[v for k,v in children if k=="_ln1_module"]
                require(not matches or matches==[id(original)],"unexpected existing LN1 alias")
                if not matches:
                    children.append(["_ln1_module",id(original)])
                    for suffix,record in source_tree.items(): expected["modules"][path+"._ln1_module"+suffix]=record
        if ln2 is not None and ln2.original_component is not None:
            expected_targets.append((ln2,"_capture_pre_ln2",{"block_ref":block,"hook_mlp_in":block.hook_mlp_in}))
        old_keys={id(target):list(target._forward_pre_hooks) for target,_,_ in expected_targets}
        method(block)
        require(block._pre_ln_capture_wired and len(block._pre_ln_capture_handles)==len(expected_targets),"exact lazy setup count")
        for (target,name,bindings),handle in zip(expected_targets,block._pre_ln_capture_handles,strict=True):
            require(list(target._forward_pre_hooks)==old_keys[id(target)]+[handle.id],"only expected appended pre-hook")
            fn=target._forward_pre_hooks[handle.id]
            require(fn.__code__ is codes[name] and fn.__globals__ is method.__globals__,"exact source callback code/globals")
            closure=inspect.getclosurevars(fn).nonlocals
            require(set(closure)==set(bindings) and all(closure[k] is v or closure[k]==v for k,v in bindings.items()),"exact source closure bindings")
            entry={"key":handle.id,"callback":callable_record(fn)}
            for record in expected["modules"].values():
                if record["identity"]==id(target): record["callbacks"]["_forward_pre_hooks"].append(copy.deepcopy(entry))
            events.append({"target_identity":id(target),"callback":entry,"source_rule":name,
                           "closure_bindings":{k:id(v) for k,v in bindings.items()}})
        for record in expected["modules"].values():
            if record["identity"]==id(block):
                record["block_setup"]={"wired":True,"handles":[h.id for h in block._pre_ln_capture_handles]}
    after=snapshot(model)
    require(expected==after,"unapproved module/callback change during known setup")
    return {"before":before,"after":after,"changes":differences(before,after),"verified_setup_callbacks":events,
            "held_reference_count":len(held),"forward_calls":0}

class HookGuard:
    def __init__(self,model):
        self.setup=materialize_known_setup(model)
        self.references=references(model)
        self.baseline=snapshot(model)
    def inspect(self,model):
        current=snapshot(model)
        changes=differences(self.baseline,current)
        return {"matches":not changes,"changes":changes,"current":current}
    def require_clean(self,model):
        result=self.inspect(model)
        require(result["matches"],"structured hook/module identity changed")
        return result
