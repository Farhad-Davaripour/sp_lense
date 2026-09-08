"""Immutable admitted code/default/closure identity, not weight fingerprints."""
import dataclasses
import marshal
import types
from support import need,sha,encoded

def scalar(value,depth=0,strong=None):
    need(depth<=6,"FINGERPRINT_DEPTH")
    if strong is not None:strong.append(value)
    if value is None or type(value) in (bool,int,str):
        need(type(value) is not str or len(value)<=256,"FINGERPRINT_TEXT")
        return {"kind":type(value).__name__,"identity":id(value),"value":value}
    if type(value) is tuple:
        need(len(value)<=32,"FINGERPRINT_TUPLE")
        return {"kind":"tuple","identity":id(value),"items":[scalar(x,depth+1,strong) for x in value]}
    if type(value) is dict:
        need(len(value)<=32 and all(type(k) is str and len(k)<=256 for k in value),"FINGERPRINT_DICT")
        return {"kind":"dict","identity":id(value),"items":{k:scalar(value[k],depth+1,strong) for k in sorted(value)}}
    return {"kind":"opaque_identity","identity":id(value)}

def measure(roots,anchors):
    todo=list(roots);seen=set();nodes=[];strong=list(anchors)+[v for _,v in roots]
    while todo:
        label,fn=todo.pop(0)
        if id(fn) in seen:continue
        seen.add(id(fn));need(len(seen)<=12 and type(fn) is types.FunctionType,"FINGERPRINT_FUNCTION")
        code=fn.__code__;strong.extend((fn,code,fn.__defaults__,fn.__kwdefaults__,fn.__dict__))
        closure=[]
        for key,cell in zip(code.co_freevars,fn.__closure__ or ()):
            empty=False
            try:value=cell.cell_contents
            except ValueError:value=None;empty=True
            strong.extend((cell,value))
            closure.append({"name":key,"cell_identity":id(cell),"empty":empty,"content":scalar(value,strong=strong)})
            if type(value) is types.FunctionType:todo.append((label+".closure."+key,value))
        wrapped=fn.__dict__.get("__wrapped__")
        if wrapped is not None:
            need(type(wrapped) is types.FunctionType,"FINGERPRINT_WRAPPED")
            todo.append((label+".wrapped",wrapped));strong.append(wrapped)
        globals_used={k:id(fn.__globals__[k]) for k in code.co_names if k in fn.__globals__}
        strong.extend(fn.__globals__[k] for k in globals_used)
        nodes.append({"label":label,"function_identity":id(fn),"code_identity":id(code),
            "code_sha256":sha(marshal.dumps(code)),"defaults":scalar(fn.__defaults__,strong=strong),
            "kwdefaults":scalar(fn.__kwdefaults__,strong=strong),"attributes":scalar(fn.__dict__,strong=strong),
            "closure":closure,"globals_identity":id(fn.__globals__),"referenced_global_identities":globals_used})
    need(len(nodes)<=12,"FINGERPRINT_NODES")
    raw=encoded({"schema":"immutable_callable_fingerprint.v1","anchors":[id(x) for x in anchors],"nodes":nodes})
    need(len(raw)<=32768,"FINGERPRINT_CAP")
    return raw,tuple(strong)

@dataclasses.dataclass(frozen=True)
class FrozenFunctions:
    roots:tuple
    anchors:tuple
    raw:bytes
    strong:tuple
    @classmethod
    def capture(cls,targets,installation):
        roots=(("conv",targets.conv),("chunk",targets.chunk),("conv_adapter",installation.conv_adapter))
        anchors=(targets.module,targets.original_type,targets,installation)
        raw,strong=measure(roots,anchors)
        return cls(roots,anchors,raw,strong)
    def inspect(self,targets,installation):
        need(targets.module is self.anchors[0] and targets.original_type is self.anchors[1] and
            targets is self.anchors[2] and installation is self.anchors[3] and
            targets.conv is self.roots[0][1] and targets.chunk is self.roots[1][1] and
            installation.conv_adapter is self.roots[2][1],"CALLABLE_BINDING_DRIFT")
        actual,_=measure(self.roots,self.anchors)
        need(actual==self.raw,"SAME_OBJECT_IMPLEMENTATION_DRIFT")
        return sha(actual)
