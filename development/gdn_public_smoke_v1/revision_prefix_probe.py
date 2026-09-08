"""One authorized fresh-process offline definitions-only prefix/provider check."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
BLOCKED_SUFFIXES=(".safetensors",".gguf",".pt",".pth",".ckpt",".bin")
def need(ok,code):
    if not ok:raise RuntimeError(code)
def child():
    for key in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","HF_DATASETS_OFFLINE","HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key]="1"
    denied=[];activation_initializations=[]
    root=HERE.parents[1]
    activation_source=root/".venv/Lib/site-packages/transformers/activations.py"
    need(hashlib.sha256(activation_source.read_bytes()).hexdigest()=="5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665","PINNED_PARAMETER_FREE_ACTIVATION_SOURCE")
    module_source=root/".venv/Lib/site-packages/torch/nn/modules/module.py"
    need(hashlib.sha256(module_source.read_bytes()).hexdigest()=="caad06c4430304a03920d1cc181828da0e8c4bfb4f0c41d740ab52a36498c1b0","PINNED_MODULE_METADATA_SOURCE")
    activation_classes={"GELUActivation","NewGELUActivation","FastGELUActivation","GELUTanh","QuickGELUActivation","SiLUActivation","MishActivation","LinearActivation"}
    def audit(event,args):
        if event in ("socket.connect","socket.getaddrinfo","socket.bind"):
            denied.append("NETWORK");raise RuntimeError("DEFINITIONS_ONLY_NETWORK_DENIED")
        if event=="open" and args and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).lower().endswith(BLOCKED_SUFFIXES):
            denied.append("CHECKPOINT");raise RuntimeError("DEFINITIONS_ONLY_CHECKPOINT_DENIED")
    sys.addaudithook(audit)
    factories={"tensor","empty","zeros","ones","rand","randn","arange","linspace","full","eye","from_numpy","as_tensor"}
    def profile(frame,event,arg):
        if event=="call":
            name=frame.f_code.co_name;module=frame.f_globals.get("__name__","")
            if not module.startswith(("torch","transformers","transformer_lens")):return
            if module=="torch.nn.parameter" and name=="__new__":
                denied.append("PARAMETER_CREATION");raise RuntimeError("DEFINITIONS_ONLY_PARAMETER_DENIED")
            if name in ("from_pretrained","load_checkpoint","safe_open","load_state_dict"):
                denied.append("LOAD_OR_CHECKPOINT");raise RuntimeError("DEFINITIONS_ONLY_LOAD_DENIED")
            if name in ("forward","backward","generate","encode","batch_encode_plus") and module.startswith(("torch","transformers","transformer_lens")):
                denied.append("NUMERIC_OR_ENCODING");raise RuntimeError("DEFINITIONS_ONLY_NUMERIC_DENIED")
            if name=="__init__":
                obj=frame.f_locals.get("self")
                parameter_free=(obj is not None and type(obj).__module__=="transformers.activations" and type(obj).__name__ in activation_classes)
                if parameter_free:
                    activation_initializations.append(type(obj).__name__)
                elif obj is not None and any((c.__name__,c.__module__) in (("Module","torch.nn.modules.module"),("PreTrainedTokenizerBase","transformers.tokenization_utils_base")) for c in type(obj).__mro__):
                    denied.append("MODULE_OR_TOKENIZER_CONSTRUCTOR");raise RuntimeError("DEFINITIONS_ONLY_CONSTRUCTOR_DENIED")
        elif event=="return" and frame.f_code.co_name=="__init__":
            obj=frame.f_locals.get("self")
            if obj is not None and type(obj).__module__=="transformers.activations" and type(obj).__name__ in activation_classes:
                need(not vars(obj).get("_parameters") and not vars(obj).get("_buffers"),"ACTIVATION_METADATA_ONLY")
        elif event=="c_call" and getattr(arg,"__name__",None) in factories and str(getattr(arg,"__module__","")).startswith("torch"):
            denied.append("TENSOR_FACTORY");raise RuntimeError("DEFINITIONS_ONLY_TENSOR_DENIED")
    sys.setprofile(profile)
    # Exact import statements from the production candidate, not a manually
    # pre-populated concrete HF module. No authority or actual loader is called.
    tree=ast.parse((HERE/"candidate_loader.py").read_bytes())
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="_diagnostic_load_adapter")
    imports=[x for x in fn.body if isinstance(x,ast.ImportFrom) and x.module in ("real_adapter","transformer_lens.model_bridge")]
    need(len(imports)==2,"EXACT_DEFINITION_IMPORT_PREFIX")
    scope={};exec(compile(ast.Module(body=imports,type_ignores=[]),str(HERE/"candidate_loader.py"),"exec"),scope)
    from diagnostic_counter import Counter
    from diagnostic_support import CELL_ID
    from real_adapter import DispatchLatch
    counter=Counter(CELL_ID);counter.reserve_load();latch=DispatchLatch()
    guard=scope["ForwardDerivativeGuard"](scope["TransformerBridge"],counter,latch,time.monotonic()+30)
    before_present="transformers.models.qwen3_5.modeling_qwen3_5" in sys.modules
    guard.install();snapshot=None;restored=False
    try:
        # Execute the exact corrected admission/import statement followed by the
        # actual snapshot provider. Never reach the subsequent backend load.
        body=next(x for x in fn.body if isinstance(x,ast.Try)).body
        start=next(i for i,x in enumerate(body) if isinstance(x,ast.ImportFrom) and x.module=="constructor_operands")
        selected=body[start:start+2]
        need(isinstance(selected[1],ast.Expr) and isinstance(selected[1].value.func,ast.Name) and selected[1].value.func.id=="import_hf_definitions","EXACT_IMPORT_SPLICE")
        scope["guard"]=guard
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(HERE/"candidate_loader.py"),"exec"),scope)
        snapshot=scope["loaded_snapshot"]()
        need(all(snapshot["conditions"][k] is True for k in ("function_type","code_equal","resolved_path_equal")),"UNCHANGED_CONSTRUCTOR_OPERANDS")
        need(guard.forwards==guard.derivatives==guard.rejected==0 and counter.load_calls==0 and not denied,"NO_MODEL_WORK")
    finally:
        guard.restore();restored=not guard.installed;sys.setprofile(None)
    return {"status":"PASS_DEFINITIONS_ONLY_PREFIX","actual_hf_present_after_original_imports":before_present,
        "actual_hf_present_after_explicit_import":"transformers.models.qwen3_5.modeling_qwen3_5" in sys.modules,
        "snapshot":snapshot,"guard_restored":restored,"model_constructors":0,"model_loads":0,"checkpoint_accesses":0,
        "tensor_calls":0,"forwards":0,"derivatives":0,"encoding":0,"denied_operations":denied,
        "allowed_parameter_free_activation_initializations":activation_initializations,"real_authority":False}
def main():
    if "--child" in sys.argv:
        print(json.dumps(child(),sort_keys=True));return 0
    started=time.monotonic();command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child"]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    timed_out=False
    try:stdout,stderr=process.communicate(timeout=28)
    except subprocess.TimeoutExpired:
        timed_out=True;process.kill();stdout,stderr=process.communicate(timeout=2)
    need(len(stdout)+len(stderr)<=1024**2,"PROBE_OUTPUT_CAP")
    value={"schema":"definition_prefix_probe_parent.v1","elapsed_seconds":time.monotonic()-started,
        "child_exit_code":process.returncode,"retained_child_closed":process.poll() is not None,"timed_out":timed_out,
        "stdout":stdout.decode(errors="replace"),"stderr":stderr.decode(errors="replace"),
        "candidate_loader_sha256":hashlib.sha256((HERE/"candidate_loader.py").read_bytes()).hexdigest(),
        "operand_source_sha256":hashlib.sha256((HERE/"constructor_operands.py").read_bytes()).hexdigest(),
        "definition_only_probe_ordinal":3,"maximum_authorized_probes":3,"separate_root_additional_probe_authorized":True,"real_authority":False}
    print(json.dumps(value,sort_keys=True));return 0 if process.returncode==0 and not timed_out else 1
if __name__=="__main__":raise SystemExit(main())
