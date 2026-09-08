"""Zero-weight installed-definition introspection; never constructs a model."""
import ast, hashlib, importlib, json, os, pathlib, sys, types
ROOT=pathlib.Path(__file__).resolve().parents[2]
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["HF_DATASETS_OFFLINE"]="1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
def audit(event,args):
    if event in {"socket.connect","socket.getaddrinfo"}:raise RuntimeError("OFFLINE_PROBE")
    if event=="open" and args and isinstance(args[0],(str,bytes)):
        p=os.fsdecode(args[0]).lower()
        if p.endswith((".safetensors",".gguf",".pt",".pth",".ckpt")):raise RuntimeError("NO_WEIGHT_ACCESS")
sys.addaudithook(audit)
PINS={"hf":(".venv/Lib/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py","67cf849081143a998f0e189c551e4b5f532365a055805570747385e400372abb"),
"decorator":(".venv/Lib/site-packages/transformers/integrations/hub_kernels.py","50e5b5f938cdb2c5a2f7e90ae1ab3933cb2d505ab38c6c4f4d0226320df4b94a")}
def need(ok,code):
    if not ok:raise ValueError(code)
def catalog(label,optimize=-1):
    path=(ROOT/PINS[label][0]).resolve();raw=path.read_bytes()
    need(hashlib.sha256(raw).hexdigest()==PINS[label][1],"SOURCE_SHA")
    result={}
    def visit(code):
        result[code.co_qualname]=code
        for x in code.co_consts:
            if type(x) is types.CodeType:visit(x)
    visit(compile(raw,str(path),"exec",dont_inherit=True,optimize=optimize))
    return path,result
def fields(a,b):
    result={}
    for name in dir(a):
        if not name.startswith("co_") or callable(getattr(a,name)):continue
        x,y=getattr(a,name),getattr(b,name)
        if x==y:continue
        if isinstance(x,bytes):result[name]={"actual_sha":hashlib.sha256(x).hexdigest(),"expected_sha":hashlib.sha256(y).hexdigest(),"actual_len":len(x),"expected_len":len(y)}
        elif name=="co_consts":
            result[name]={"actual_types":[type(v).__name__ for v in x],"expected_types":[type(v).__name__ for v in y],
                          "different_indices":[i for i in range(min(len(x),len(y))) if x[i]!=y[i]]}
        elif type(x) in (str,int,tuple):result[name]={"actual":x,"expected":y}
        else:result[name]={"different":True}
    return result
def main():
    path,codes=catalog("hf")
    importlib.import_module("transformer_lens.model_bridge")
    module=importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    fn=module.Qwen3_5GatedDeltaNet.__init__;expected=codes["Qwen3_5GatedDeltaNet.__init__"]
    import inspect
    comparisons={}
    for name,f,ref in (("init",fn,expected),("forward",inspect.unwrap(module.Qwen3_5GatedDeltaNet.forward),codes["Qwen3_5GatedDeltaNet.forward"])):
        comparisons[name]={"function_type":type(f) is types.FunctionType,"code_equal":f.__code__==ref,
          "resolved_path_equal":pathlib.Path(f.__code__.co_filename).resolve()==path,"actual_filename":f.__code__.co_filename,
          "expected_filename":ref.co_filename,"differences":fields(f.__code__,ref),"globals_same":f.__globals__ is vars(module),
          "defaults":str(f.__defaults__),"closure_names":list(f.__code__.co_freevars),
          "matches_optimization":{str(o):f.__code__==catalog("hf",o)[1][ref.co_qualname] for o in (0,1,2)}}
    print(json.dumps({"probe":"INSTALLED_DEFINITIONS_ONLY","python":sys.version,"optimize":sys.flags.optimize,
      "module_file":module.__file__,"module_cached":module.__cached__,"comparisons":comparisons,
      "model_constructors":0,"tokenizers":0,"weight_access":0,"parameter_access":0,"forwards":0,"derivatives":0,"encoding":0}))
if __name__=="__main__":main()
