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
    importlib.import_module("transformer_lens.model_bridge")
    module=importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    sys.path.insert(0,str(ROOT))
    from development.gdn_runtime_admission_v1 import compat_candidate,admission
    path,codes=catalog("hf")
    operands=admission.code_conditions(module.Qwen3_5GatedDeltaNet.__init__,codes["Qwen3_5GatedDeltaNet.__init__"],path)
    target=compat_candidate.Targets(module,module.Qwen3_5GatedDeltaNet,module.causal_conv1d_fn,
      module.torch_chunk_gated_delta_rule,ROOT,mode="LIVE_EXISTING_MODULE")
    target.inspect()
    print(json.dumps({"status":"PASS_INSTALLED_DEFINITION_ADMISSION_ONLY","constructor_operands":operands,
      "all_original_helper_defaults_closures_sources_checked":True,"accelerate_wrapper_source_sha256":admission.ACCEL_SHA,
      "target_mode":target.mode,"model_constructors":0,"tokenizers":0,"weight_access":0,"parameters":0,
      "helper_invocations":0,"forwards":0,"derivatives":0,"encoding":0,"real_authority":False,
      "post_model_construction_constructor_discrepancy":"NOT_REPRODUCED_OR_RELAXED"}))
if __name__=="__main__":main()
