"""One fixed handoff-only batch: inert instances/callbacks, no numerical work."""
import ast
import contextlib
import copy
import json
import os
import pathlib
import sys
import threading
import time
import types

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parent.parent
BLOCKED={"torch","transformers","transformer_lens","tokenizers","safetensors","numpy","research_backend"}
class BlockImports:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".")[0] in BLOCKED:raise RuntimeError("FORBIDDEN_IMPORT")
sys.meta_path.insert(0,BlockImports())
from support import need,Rejected,encoded,sha,component,authenticate
from handoff import Handoff
from selection import LAYER_TYPES,LABELS
from saved_reader import judge

CASES=("clean_complete_join","instance_set_rejection","same_object_drift","coherent_saved_tamper",
       "setup_failure_joins","cleanup_failure_joins")
LIMITS={"substantive_seconds":45,"shared_cleanup_seconds":15,"absolute_seconds":60,
        "preparation_bytes":33554432,"test_evidence_bytes":8388608,"per_file_bytes":5242880}
DELETE_FAILURES=set()
def put(path,raw):
    need(len(raw)<=5242880,"FILE_CAP");path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    need(path.read_bytes()==raw,"BYTE_READBACK")
    return {"name":path.name,"bytes":len(raw),"sha256":sha(raw)}
def save(path,value):return put(path,encoded(value))
class Publisher:
    def __init__(self,folder,fault=None):self.folder=folder;self.fault=fault
    def write_new(self,name,raw):
        need(name in ("HELPER_SETUP.json","HELPER_TERMINAL.json") and len(raw)<=65536,"NATIVE_RECEIPT")
        if self.fault==name:
            put(self.folder/name,raw[:19]);raise OSError("NEVER_SERIALIZE")
        return put(self.folder/name,raw)
    def read(self,name):return (self.folder/name).read_bytes()
class InertGDN:
    def __init__(self):self.marker=object()
    def __delattr__(self,name):
        if (id(self),name) in DELETE_FAILURES:raise OSError("NEVER_SERIALIZE")
        object.__delattr__(self,name)
class InertBridge:
    def __init__(self,original):self.original_component=original
class InertHF(types.SimpleNamespace):
    def named_modules(self,*,remove_duplicate):
        need(remove_duplicate is False,"DUPLICATES_RETAINED")
        yield from self.named
def conv_base(hidden_states,weight,bias=None,activation=None):raise Rejected("NO_NUMERICAL_EXECUTION")
def chunk_base(query,key,value,g,beta,chunk_size=64,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False):
    raise Rejected("NO_NUMERICAL_EXECUTION")
def wrapped(base):
    implementation=base
    def fn(*args,neutral=None,**kwargs):return implementation(*args,**kwargs)
    fn.__wrapped__=base
    return fn
def alternate_wrapped(base):
    implementation=base
    def fn(*args,neutral=None,**kwargs):
        answer=implementation(*args,**kwargs)
        return answer
    return fn
def new_function(base):
    return types.FunctionType(base.__code__,base.__globals__,base.__name__,base.__defaults__)
def old_cleanup_function():
    p=ROOT/"diagnostics/fresh_confirmation_first_forward_diagnostic_v1/trace_operations.py"
    tree=ast.parse(p.read_bytes())
    body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ("cleanup_step","diagnostic_cleanup")]
    namespace={"span":lambda stage:contextlib.nullcontext()}
    exec(compile(ast.Module(body=body,type_ignores=[]),str(p),"exec",dont_inherit=True),namespace)
    return namespace["diagnostic_cleanup"]

def fixture(folder,tag,fault=None):
    checked=component()
    pair=(wrapped(new_function(conv_base)),wrapped(new_function(chunk_base)))
    module=types.SimpleNamespace(Qwen3_5GatedDeltaNet=InertGDN,causal_conv1d_fn=pair[0],torch_chunk_gated_delta_rule=pair[1])
    targets=checked.Targets(module,InertGDN,*pair,ROOT,mode="INERT_FIXTURE")
    layers=[];blocks=[];named=[]
    for i,kind in enumerate(LAYER_TYPES):
        original=InertGDN() if kind=="linear_attention" else None
        layers.append(types.SimpleNamespace(layer_type=kind,linear_attn=original))
        blocks.append(types.SimpleNamespace(linear_attn=InertBridge(original) if original else None))
        if original:named.append(("model.layers."+str(i)+".linear_attn",original))
    hf=InertHF(config=types.SimpleNamespace(layer_types=LAYER_TYPES,num_hidden_layers=24),
        model=types.SimpleNamespace(layers=layers),named=named)
    bridge=types.SimpleNamespace(blocks=blocks)
    calls=[];latch={"terminal":False}
    def stop(code):latch["terminal"]=True;calls.append("stop")
    def restore():calls.append("original_guard_restore")
    execution={"mode":"INERT_FIXTURE","fixture_id":tag,"real_model_work":False}
    source_lock={"source_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes()),
                 "pins_sha256":sha((HERE/"SOURCE_PINS.json").read_bytes())}
    publisher=Publisher(folder,fault)
    session=Handoff(targets,hf,bridge,InertBridge,publisher,stop,restore,execution=execution,source_lock=source_lock)
    def original_setup():calls.append("original_setup");return ORIGINAL_SETUP_RESULT
    def original_check():
        calls.append("original_pre_forward_check")
        need(not latch["terminal"],"ORIGINAL_REJECTED")
        return ORIGINAL_CHECK_RESULT
    routine=old_cleanup_function()
    def original_cleanup(restore_join):
        def end():calls.append("end_edit")
        def parameter():calls.append("retained_parameter_predicates");return INERT_PARAMETER_RESULT
        def admit():calls.append("original_latch_admit");need(not latch["terminal"],"ORIGINAL_REJECTED")
        def guard():calls.append("unchanged_hook_inspect");return INERT_GUARD_RESULT
        def identity(p,g):
            calls.append("unchanged_cold_identity")
            need(p is INERT_PARAMETER_RESULT and g is INERT_GUARD_RESULT,"ORIGINAL_REJECTED")
            return ORIGINAL_CLEANUP_RESULT
        return routine(end,parameter,admit,guard,identity,restore_join)
    return types.SimpleNamespace(session=session,targets=targets,pair=pair,hf=hf,bridge=bridge,publisher=publisher,
        original_setup=original_setup,original_check=original_check,original_cleanup=original_cleanup,calls=calls,latch=latch,
        execution=execution,source_lock=source_lock)

ORIGINAL_SETUP_RESULT=object();ORIGINAL_CHECK_RESULT=object();ORIGINAL_CLEANUP_RESULT=object()
INERT_PARAMETER_RESULT=object();INERT_GUARD_RESULT=object()
def attempt(action):
    try:return True,action()
    except BaseException as exc:return False,getattr(exc,"code","OTHER")
def finish(f):
    okay,result=attempt(lambda:f.session.close(f.original_cleanup))
    status=f.session.status()
    save(f.publisher.folder/"CONTROLLER_STATUS.json",status)
    save(f.publisher.folder/"ADMITTED_SETUP.json",f.session.admission())
    output=judge(f.publisher.read,execution=f.execution,source_lock=f.source_lock,
        controller_status=status,expected_setup=f.session.admission())
    save(f.publisher.folder/"READER_RESULT.json",output)
    return okay,result,status,output
def clean(folder,tag):
    f=fixture(folder,tag)
    need(f.session.setup(f.original_setup) is ORIGINAL_SETUP_RESULT,"SETUP_RETURN_UNCHANGED")
    need(f.session.checkpoint(f.original_check) is ORIGINAL_CHECK_RESULT,"CHECK_RETURN_UNCHANGED")
    okay,result,status,output=finish(f)
    need(okay and result is ORIGINAL_CLEANUP_RESULT and output["binding_verified"],"CLEAN_JOIN")
    need(f.calls==["original_setup","original_pre_forward_check","end_edit","retained_parameter_predicates",
        "original_latch_admit","unchanged_hook_inspect","unchanged_cold_identity","original_guard_restore"],"ORIGINAL_CLEANUP_ORDER")
    need(all(not any(k in vars(obj) for k in ("causal_conv1d_fn","chunk_gated_delta_rule")) for _,obj in f.hf.named),
        "ALIASES_REMOVED")
    return f,status
def run_group(name,folder):
    DELETE_FAILURES.clear();outcomes=[]
    if name=="clean_complete_join":
        f,status=clean(folder/"clean",name)
        outcomes.append({"join":"complete","instances":18,"checkpoints":3,"guard_restore_count":f.calls.count("original_guard_restore")})
    elif name=="instance_set_rejection":
        for kind in ("omitted_traversal","duplicate_original","wrong_bridge","extra_original"):
            f=fixture(folder/kind,name+"/"+kind)
            if kind=="omitted_traversal":f.hf.named.pop()
            elif kind=="duplicate_original":
                f.hf.model.layers[1].linear_attn=f.hf.model.layers[0].linear_attn
                f.bridge.blocks[1].linear_attn.original_component=f.hf.model.layers[0].linear_attn
            elif kind=="wrong_bridge":f.bridge.blocks[0].linear_attn.original_component=InertGDN()
            else:f.hf.named.append(("extra.unexpected",InertGDN()))
            okay,_=attempt(lambda:f.session.setup(f.original_setup))
            status=f.session.status();save(f.publisher.folder/"CONTROLLER_STATUS.json",status)
            need(not okay and status["primary"]["phase"]=="SELECTION" and status["restore_returned"] and
                 f.calls==["stop","original_guard_restore"],"SELECT_BEFORE_MUTATION_SETUP")
            outcomes.append({"case":kind,"primary":status["primary"]})
    elif name=="same_object_drift":
        for kind in ("wrapper_code","base_code","defaults","kwdefaults","selected_implementation"):
            f=fixture(folder/kind,name+"/"+kind)
            f.session.setup(f.original_setup)
            ref=f.session.frozen.raw;conv=f.pair[0];base=conv.__wrapped__
            if kind=="wrapper_code":conv.__code__=alternate_wrapped(base).__code__
            elif kind=="base_code":base.__code__=chunk_base.__code__
            elif kind=="defaults":base.__defaults__=(None,True)
            elif kind=="kwdefaults":conv.__kwdefaults__["neutral"]=True
            else:conv.__closure__[0].cell_contents=new_function(conv_base)
            f.targets.inspect()  # Old pointer-only inspection still passes.
            okay,_=attempt(lambda:f.session.checkpoint(f.original_check))
            _,_,status,read=finish(f)
            need(not okay and f.session.frozen.raw==ref and status["primary"]["phase"]=="PRE_FORWARD" and
                status["primary"]["code"]=="SAME_OBJECT_IMPLEMENTATION_DRIFT" and not read["binding_verified"] and
                f.calls.count("original_guard_restore")==1,"IMMUTABLE_FUNCTION_DRIFT")
            outcomes.append({"case":kind,"primary":status["primary"],"old_pointer_check_passed":True})
    elif name=="coherent_saved_tamper":
        f,status=clean(folder/"base",name)
        for kind in ("source","execution","instance","fingerprint"):
            setup=json.loads(f.publisher.read("HELPER_SETUP.json"));term=json.loads(f.publisher.read("HELPER_TERMINAL.json"))
            if kind=="source":setup["source_lock"]["source_sha256"]="0"*64
            elif kind=="execution":setup["execution"]["fixture_id"]="coherently_changed"
            elif kind=="instance":setup["selection"][0]["identity"]+=1
            else:setup["fingerprint"]["nodes"][0]["code_sha256"]="0"*64
            target=folder/kind
            ack=put(target/"HELPER_SETUP.json",encoded(setup));term["setup_ack"]=ack
            terminal_ack=put(target/"HELPER_TERMINAL.json",encoded(term))
            forged=dict(term);forged["terminal_ack"]=terminal_ack
            result=judge(lambda p:(target/p).read_bytes(),execution=f.execution,source_lock=f.source_lock,
                controller_status=forged,expected_setup=f.session.admission())
            need(not result["binding_verified"] and result["failure_code"]=="IMMUTABLE_SETUP_ADMISSION","COHERENT_TAMPER_ANCHOR")
            save(target/"COHERENT_STATUS.json",forged);save(target/"READER_RESULT.json",result)
            outcomes.append({"case":kind,"outer_hashes_and_terminal_copy_repaired":True,"reader":result})
    elif name=="setup_failure_joins":
        for kind in ("original_setup","partial_setup_publication"):
            f=fixture(folder/kind,name+"/"+kind,"HELPER_SETUP.json" if kind.startswith("partial") else None)
            def fail():f.calls.append("original_setup");raise Rejected("ORIGINAL_REJECTED")
            okay,_=attempt(lambda:f.session.setup(fail if kind=="original_setup" else f.original_setup))
            status=f.session.status();save(f.publisher.folder/"CONTROLLER_STATUS.json",status)
            reader=judge(f.publisher.read,execution=f.execution,source_lock=f.source_lock,controller_status=status,
                expected_setup=f.session.admission())
            need(not okay and not reader["binding_verified"] and status["restore_returned"] and
                status["component_status"]["rollback_complete"] and f.calls.count("original_guard_restore")==1,"SETUP_FAILURE_CLOSEOUT")
            need(status["primary"]["phase"]==("ORIGINAL_SETUP" if kind=="original_setup" else "SETUP_PUBLICATION"),"SETUP_FIRST_CAUSE")
            save(f.publisher.folder/"READER_RESULT.json",reader);outcomes.append({"case":kind,"primary":status["primary"]})
    elif name=="cleanup_failure_joins":
        for kind in ("latch_refusal_then_rollback_failure","terminal_publication_failure"):
            f=fixture(folder/kind,name+"/"+kind,"HELPER_TERMINAL.json" if kind.startswith("terminal") else None)
            f.session.setup(f.original_setup);f.session.checkpoint(f.original_check)
            if kind.startswith("latch"):
                f.latch["terminal"]=True
                DELETE_FAILURES.add((id(f.hf.model.layers[0].linear_attn),"causal_conv1d_fn"))
            okay,_,status,reader=finish(f)
            need(not okay and not reader["binding_verified"] and f.calls.count("original_guard_restore")==1 and
                status["restore_returned"],"FAILURE_RESTORATION")
            if kind.startswith("latch"):
                need(status["primary"]["phase"]=="ORIGINAL_CLEANUP" and
                    any(x["phase"]=="HELPER_ROLLBACK" for x in status["secondary"]) and
                    "unchanged_hook_inspect" not in f.calls,"FIRST_CAUSE_BEFORE_FINALLY_FAILURE")
            else:need(status["publication_failed"] and status["terminal_ack"] is None,"MISSING_TERMINAL_NO_PASS")
            outcomes.append({"case":kind,"primary":status["primary"],"secondary":status["secondary"]})
            DELETE_FAILURES.clear()
    else:raise Rejected("UNDECLARED_GROUP")
    result={"case":name,"status":"PASS_INERT_HANDOFF_ONLY","outcomes":outcomes,"scientific_pass":False,
        "real_model_imports":0,"model_loads":0,"parameter_accesses":0,"parameter_hashes":0,"forwards":0,
        "derivatives":0,"encodings":0,"original_cleanup_ast_reused":True}
    ack=save(folder/"RESULT.json",result)
    return {"case":name,"status":result["status"],"result":ack}

def bounds():
    files=[p for p in HERE.rglob("*") if p.is_file()]
    test=[p for p in (HERE/"test_evidence").rglob("*") if p.is_file()]
    need(max(p.stat().st_size for p in files)<=5242880 and sum(p.stat().st_size for p in test)<=8388608 and
        sum(p.stat().st_size for p in files)-sum(p.stat().st_size for p in test)<=33554432,"STORAGE_CAP")
def main():
    start=time.monotonic();deadline=start+45;absolute=start+60
    timer=threading.Timer(59,lambda:os._exit(124));timer.daemon=True;timer.start()
    raw=(HERE/"BATCH_LOCK.json").read_bytes();need(len(sys.argv)==2 and sha(raw)==sys.argv[1],"CALLER_LOCK")
    lock=json.loads(raw);freeze_raw=(HERE/"SOURCE_FREEZE.json").read_bytes();freeze=json.loads(freeze_raw)
    need(lock["source_sha256"]==sha(freeze_raw) and lock["cases"]==list(CASES) and lock["limits"]==LIMITS,"FREEZE_LOCK")
    for p,digest in freeze["files"].items():need(sha((HERE/p).read_bytes())==digest,"FROZEN_SOURCE")
    u=(HERE/"USAGE_BEFORE_BATCH.json").read_bytes();need(sha(u)==lock["usage_sha256"],"USAGE_SHA")
    usage=json.loads(u);payload=json.loads(usage["tool_result"]["content"][0]["text"])
    bucket=payload["rateLimitsByLimitId"]["codex"];used=bucket["primary"]["usedPercent"]
    need(type(used) in (int,float) and 0<=used<100 and not bucket["spendControlReached"] and
        0<=time.time()-usage["observed_unix_seconds"]<=120,"ACTUAL_FRESH_USAGE")
    need(not (HERE/"test_evidence").exists(),"ONE_BATCH")
    (HERE/"test_evidence").mkdir()
    save(HERE/"test_evidence/BATCH_STARTED.json",{"source_sha256":lock["source_sha256"],"batch_lock_sha256":sha(raw),
        "started":start,"substantive_deadline":deadline,"absolute_deadline":absolute})
    report={"status":"INCONCLUSIVE","source_sha256":lock["source_sha256"],"batch_lock_sha256":sha(raw),
        "cases":[],"remaining_unrun":list(CASES),"scientific_pass":False,"retry_count":0,
        "real_model_imports":0,"model_loads":0,"parameter_accesses":0,"parameter_hashes":0,"forwards":0,"derivatives":0,"encodings":0}
    try:
        save(HERE/"test_evidence/PINNED_SOURCE_VERIFICATION.json",authenticate())
        for name in CASES:
            need(time.monotonic()<deadline,"SUBSTANTIVE_DEADLINE")
            report["cases"].append(run_group(name,HERE/"test_evidence"/name));report["remaining_unrun"].remove(name)
            need(time.monotonic()<deadline,"SUBSTANTIVE_DEADLINE")
        need(not any(x.split(".")[0] in BLOCKED for x in sys.modules),"NO_MODEL_IMPORTS")
        bounds();report["status"]="PASS_INERT_HANDOFF_ONLY"
    except BaseException as exc:
        report["failure_code"]=getattr(exc,"code","FIXED_BATCH_FAILURE")
        tb=exc.__traceback__;frames=[]
        while tb and len(frames)<8:
            p=pathlib.Path(tb.tb_frame.f_code.co_filename)
            if p.is_absolute() and p.parent==HERE:frames.append({"file":p.name,"line":tb.tb_lineno})
            tb=tb.tb_next
        report["failure_source_frames"]=frames
    finally:
        close=time.monotonic();report["elapsed_seconds"]=close-start
        save(HERE/"test_evidence/BATCH_REPORT.json",report)
        save(HERE/"test_evidence/BATCH_CLOSEOUT.json",{"closed":True,"elapsed_seconds":time.monotonic()-start,
            "shared_closeout_seconds":time.monotonic()-close,"absolute_envelope_met":time.monotonic()<absolute,
            "owned_processes_spawned":0,"real_authority":False})
        bounds();timer.cancel()
    print(json.dumps({"status":report["status"],"completed":len(report["cases"]),
        "remaining_unrun":report["remaining_unrun"],"elapsed_seconds":report["elapsed_seconds"]}))
    return 0 if report["status"]=="PASS_INERT_HANDOFF_ONLY" else 1
if __name__=="__main__":raise SystemExit(main())
