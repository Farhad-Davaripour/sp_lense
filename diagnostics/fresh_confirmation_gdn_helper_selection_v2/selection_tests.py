"""Single locked selection-only batch; no numerical package imports."""
import ast
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
from selection import select,saved_selection,saved_graph,graph_sha,walk
from selection_reader import verify_graph
from saved_reader import judge
from handoff import Handoff
from inert_topology import build_types,topology,EXTRACTIONS
CASES=("valid_post_wrap_join","old_topology_or_field","missing_extra_aliases","rebound_originals_wrappers","selection_receipt_tamper")
LIMITS={"substantive_seconds":45,"shared_cleanup_seconds":15,"absolute_seconds":60,
        "preparation_bytes":33554432,"test_evidence_bytes":8388608,"per_file_bytes":5242880}
TYPES=None
CLEAN=None
def put(path,raw):
    need(len(raw)<=5242880,"FILE_CAP");path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    need(path.read_bytes()==raw,"BYTE_READBACK")
    return {"name":path.name,"bytes":len(raw),"sha256":sha(raw)}
def save(path,value):return put(path,encoded(value))
class Publisher:
    def __init__(self,folder):self.folder=folder
    def write_new(self,name,raw):
        need(name in ("HELPER_SETUP.json","HELPER_TERMINAL.json") and len(raw)<=65536,"RECEIPT_CAP")
        return put(self.folder/name,raw)
    def read(self,name):return (self.folder/name).read_bytes()
def conv_base(hidden_states,weight,bias=None,activation=None):raise Rejected("NO_NUMERICAL_EXECUTION")
def chunk_base(query,key,value,g,beta,chunk_size=64,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False):
    raise Rejected("NO_NUMERICAL_EXECUTION")
def wrapped(base):
    implementation=base
    def fn(*args,**kwargs):return implementation(*args,**kwargs)
    fn.__wrapped__=base
    return fn
def fixture(folder,tag):
    f=topology(TYPES);pair=(wrapped(conv_base),wrapped(chunk_base))
    module=types.SimpleNamespace(Qwen3_5GatedDeltaNet=TYPES.GDN,causal_conv1d_fn=pair[0],torch_chunk_gated_delta_rule=pair[1])
    f.targets=component().Targets(module,TYPES.GDN,*pair,ROOT,mode="INERT_FIXTURE")
    f.calls=[];f.latch={"terminal":False}
    def stop(code):f.latch["terminal"]=True;f.calls.append("stop")
    def restore():f.calls.append("restore")
    f.execution={"mode":"INERT_FIXTURE","fixture_id":tag,"real_model_work":False}
    f.source_lock={"source_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes()),"pins_sha256":sha((HERE/"SOURCE_PINS.json").read_bytes())}
    f.publisher=Publisher(folder)
    f.session=Handoff(f.targets,f.hf,f.bridge,f.bindings,f.publisher,stop,restore,
        execution=f.execution,source_lock=f.source_lock)
    f.pick=lambda:select(f.hf,f.bridge,f.targets,f.bindings)
    def setup():f.calls.append("original_setup");return SETUP_RESULT
    def check():need(not f.latch["terminal"],"ORIGINAL_REJECTED");f.calls.append("original_check");return CHECK_RESULT
    def cleanup(restore_join):
        try:
            f.calls.append("original_cleanup")
            need(not f.latch["terminal"],"ORIGINAL_REJECTED")
            return CLEANUP_RESULT
        finally:restore_join()
    f.setup=setup;f.check=check;f.cleanup=cleanup
    return f
SETUP_RESULT=object();CHECK_RESULT=object();CLEANUP_RESULT=object()
def rejected(action,codes):
    try:action()
    except BaseException as exc:
        code=getattr(exc,"code","UNEXPECTED_EXCEPTION")
        need(code in codes,"EXPECTED_REJECTION_CODE")
        return code
    raise Rejected("INVALID_SELECTION_ACCEPTED")
def finish(f):
    result=f.session.close(f.cleanup);status=f.session.status();admitted=f.session.admission()
    save(f.publisher.folder/"CONTROLLER_STATUS.json",status)
    save(f.publisher.folder/"ADMITTED_SETUP.json",admitted)
    output=judge(f.publisher.read,execution=f.execution,source_lock=f.source_lock,controller_status=status,expected_setup=admitted)
    save(f.publisher.folder/"READER_RESULT.json",output)
    return result,status,admitted,output

def run_group(name,folder):
    global CLEAN
    outcomes=[]
    if name=="valid_post_wrap_join":
        f=fixture(folder/"clean",name);selected=f.pick();graph=saved_graph(selected)
        for root in (f.hf,f.bridge):
            actual=[(p,id(o)) for p,o in root.named_modules(remove_duplicate=False)]
            bounded=[(p,id(o)) for p,o in walk(root)]
            need(actual==bounded,"EXTRACTED_TRAVERSAL_PARITY")
        need(len(selected)==18 and len(graph["layers"])==24 and
             all(len(v)==120 and sum(x[1]=="gdn" for x in v)==36 for v in graph["paths"].values()),"ACTUAL_GRAPH_COUNTS")
        need(f.hf.model.layers is f.bridge.blocks and
             all(f.hf.model.layers[i].original_component is f.decoders[i] for i in range(24)),"ACTUAL_WRAPPED_IDENTITY")
        save(folder/"AST_EXTRACTIONS.json",EXTRACTIONS)
        save(folder/"GRAPH.json",graph)
        need(f.session.setup(f.setup) is SETUP_RESULT and f.session.checkpoint(f.check) is CHECK_RESULT,"UNCHANGED_CALLBACK_RETURNS")
        result,status,admitted,output=finish(f)
        need(result is CLEANUP_RESULT and output["binding_verified"] and f.calls==
             ["original_setup","original_check","original_cleanup","restore"],"CLEAN_SELECTION_JOIN")
        need(all("causal_conv1d_fn" not in vars(obj) and "chunk_gated_delta_rule" not in vars(obj) for _,obj in selected),
             "OWNED_HELPERS_REMOVED")
        CLEAN=(f,status,admitted)
        outcomes.append({"layers":24,"originals":18,"gdn_occurrences_per_root":36,"tracked_paths_per_root":120,
                         "shared_blocks":True,"bounded_traversal_matches_exact_extracted_torch":True,
                         "checkpoints":3,"guard_restorations":1})
    elif name=="old_topology_or_field":
        for kind in ("unwrapped_old_fixture","wrong_decoder_field","wrong_decoder_kind"):
            f=fixture(folder/kind,kind)
            if kind=="unwrapped_old_fixture":f.hf.model.layers=TYPES.ModuleList(f.decoders)
            elif kind=="wrong_decoder_field":
                decoder=f.decoders[0];decoder.layer_type=decoder.block_type;del decoder.block_type
            else:f.decoders[0].block_type="full_attention"
            code=rejected(lambda:f.session.setup(f.setup),{"SHARED_BLOCKS","LAYER_KIND"})
            need("original_setup" not in f.calls and f.calls.count("restore")==1 and f.latch["terminal"],"REJECT_BEFORE_INSTALL")
            save(folder/kind/"STATUS.json",f.session.status())
            outcomes.append({"case":kind,"code":code,"setup_unrun":True})
    elif name=="missing_extra_aliases":
        for kind in ("missing_direct_alias","missing_decoder_alias","extra_root_original","extra_wrapper_alias","wrong_type_extra"):
            f=fixture(folder/kind,kind);block=f.bridge.blocks[0];decoder=block.original_component;wrapper=block.linear_attn
            if kind=="missing_direct_alias":del block._modules["linear_attn"]
            elif kind=="missing_decoder_alias":del decoder._modules["linear_attn"]
            elif kind=="extra_root_original":f.hf.add_module("unexpected_original",wrapper.original_component)
            elif kind=="extra_wrapper_alias":f.bridge.add_module("unexpected_wrapper",wrapper)
            else:
                class Wrong(TYPES.GDN):pass
                f.bridge.add_module("unexpected_subclass",Wrong())
            code=rejected(f.pick,{"MISSING_MODULE","COMPLETE_ALIAS_GRAPH","PATH_TYPE"})
            outcomes.append({"case":kind,"code":code})
    elif name=="rebound_originals_wrappers":
        for kind in ("one_decoder_rebound","duplicate_original","wrong_original_type","shared_list_rebound","post_admission_wrapper_rebound"):
            f=fixture(folder/kind,kind);block=f.bridge.blocks[0];wrapper=block.linear_attn
            if kind=="one_decoder_rebound":
                alternate=TYPES.GDNBridge("linear_attn");alternate.set_original_component(wrapper.original_component)
                block.original_component.linear_attn=alternate
            elif kind=="duplicate_original":wrapper._modules["_original_component"]=f.bridge.blocks[1].linear_attn.original_component
            elif kind=="wrong_original_type":wrapper._modules["_original_component"]=TYPES.Module()
            elif kind=="shared_list_rebound":f.bridge.blocks=TYPES.ModuleList(tuple(f.bridge.blocks))
            else:
                f.session.setup(f.setup)
                alternate=TYPES.GDNBridge("linear_attn")
                alternate.set_original_component(wrapper.original_component)
                block._modules["linear_attn"]=alternate;block.original_component.linear_attn=alternate
                code=rejected(lambda:f.session.checkpoint(f.check),{"COMPLETE_ALIAS_GRAPH"})
                rejected(lambda:finish(f),{"ORIGINAL_REJECTED"})
                need(f.latch["terminal"] and f.calls.count("restore")==1 and f.session.status()["state"]=="FAILED",
                     "ADMITTED_GRAPH_NOT_REFRESHED")
                save(folder/kind/"STATUS.json",f.session.status())
                outcomes.append({"case":kind,"code":code,"reference_not_refreshed":True})
                continue
            code=rejected(f.pick,{"MAPPING_IDENTITY","COMPLETE_UNIQUE_SET","SHARED_BLOCKS"})
            outcomes.append({"case":kind,"code":code})
    elif name=="selection_receipt_tamper":
        f,status,admitted=CLEAN
        for kind in ("omitted_original_path","extra_alias_path","path_identity_rebound","type_source_rebound","terminal_graph_rebound"):
            setup=json.loads(f.publisher.read("HELPER_SETUP.json"));terminal=json.loads(f.publisher.read("HELPER_TERMINAL.json"))
            external=copy.deepcopy(admitted);controller=copy.deepcopy(status)
            graph=setup["selection_graph"]
            if kind=="omitted_original_path":graph["paths"]["hf"].pop(3)
            elif kind=="extra_alias_path":graph["paths"]["bridge"].append(["extra","gdn",graph["layers"][0]["gdn"]])
            elif kind=="path_identity_rebound":graph["paths"]["hf"][3][2]=graph["layers"][1]["gdn"]
            elif kind=="type_source_rebound":graph["types"]["block"]["source_sha256"]="0"*64
            else:terminal["selection_graph_sha256"]="0"*64
            if kind!="terminal_graph_rebound":
                digest=sha(encoded(graph))
                for check in setup["checks"]:check["selection_graph_sha256"]=digest
                for check in terminal["checks"]:check["selection_graph_sha256"]=digest
                terminal["selection_graph_sha256"]=digest
                external["selection_graph_sha256"]=digest
            publisher=Publisher(folder/kind)
            setup_ack=publisher.write_new("HELPER_SETUP.json",encoded(setup))
            terminal["setup_ack"]=setup_ack;external["setup_ack"]=setup_ack
            terminal_ack=publisher.write_new("HELPER_TERMINAL.json",encoded(terminal))
            controller=dict(terminal);controller["terminal_ack"]=terminal_ack
            output=judge(publisher.read,execution=f.execution,source_lock=f.source_lock,
                         controller_status=controller,expected_setup=external)
            expected="GRAPH_TYPE_PIN" if kind=="type_source_rebound" else "SELECTION_GRAPH_TERMINAL" if kind=="terminal_graph_rebound" else "SAVED_ALIAS_GRAPH"
            need(not output["binding_verified"] and output["failure_code"]==expected,"INNER_GRAPH_JOIN_REACHED")
            save(folder/kind/"SYNTHETIC_REBINDING.json",{"synthetic_only":True,"outer_hashes_controller_and_external_fixture_copies_repaired":True,
                "reason":"Reach independent inner graph predicate; not an authentic controller replacement.",
                "expected_failure":expected,"controller_status":controller,"expected_setup":external})
            save(folder/kind/"READER_RESULT.json",output)
            outcomes.append({"case":kind,"code":output["failure_code"],"inner_predicate_reached":True})
    else:raise Rejected("UNDECLARED_GROUP")
    result={"case":name,"status":"PASS_INERT_SELECTION_ONLY","outcomes":outcomes,"scientific_pass":False,
        "real_model_imports":0,"model_loads":0,"parameter_accesses":0,"parameter_hashes":0,"forwards":0,"derivatives":0,"encodings":0}
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
        global TYPES
        TYPES=build_types()
        for name in CASES:
            need(time.monotonic()<deadline,"SUBSTANTIVE_DEADLINE")
            report["cases"].append(run_group(name,HERE/"test_evidence"/name));report["remaining_unrun"].remove(name)
            need(time.monotonic()<deadline,"SUBSTANTIVE_DEADLINE")
        need(not any(x.split(".")[0] in BLOCKED for x in sys.modules),"NO_MODEL_IMPORTS")
        bounds();report["status"]="PASS_INERT_SELECTION_ONLY"
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
    return 0 if report["status"]=="PASS_INERT_SELECTION_ONLY" else 1
if __name__=="__main__":raise SystemExit(main())
