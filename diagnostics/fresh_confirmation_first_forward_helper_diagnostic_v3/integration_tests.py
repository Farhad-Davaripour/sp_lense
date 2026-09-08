"""One integration-only inert batch. Never executes a real loader or baseline."""
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
ROOT=HERE.parents[1]
BLOCKED={"torch","transformers","transformer_lens","tokenizers","safetensors","numpy","research_backend","sp_lense"}
class BlockImports:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".")[0] in BLOCKED:raise RuntimeError("FORBIDDEN_IMPORT")
sys.meta_path.insert(0,BlockImports())
import support
import helper_binding as integration
import helper_reader
import real_boundary
import forward_trace
import trace_reader
from trace_operations import diagnostic_cleanup
from helper_limits import FILE_CAPS,TOTAL_RESERVED,OTHER_ORIGINAL,OTHER_REMAINDER
CASES=("clean_loader_native_join","mandatory_live_admission","native_setup_publication_faults",
       "original_cleanup_refusal","outer_closeout_failure_survival","saved_join_tamper")
LIMITS={"substantive_seconds":45,"shared_cleanup_seconds":15,"absolute_seconds":60,
        "preparation_bytes":33554432,"test_evidence_bytes":8388608,"per_file_bytes":5242880}
def need(ok,code):
    if not ok:raise RuntimeError(code)
def encoded(value):return (json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode()
def sha(raw):return support.sha(raw)
def put(path,raw):
    need(len(raw)<=5242880,"FILE_CAP");path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f:need(f.write(raw)==len(raw),"PARTIAL_TEST_RECEIPT");f.flush();os.fsync(f.fileno())
    need(path.read_bytes()==raw,"READBACK")
    return {"bytes":len(raw),"sha256":sha(raw)}
def save(path,value):return put(path,encoded(value))
PACKAGE=None
TYPES=None
CLEAN=None
ACTIVE_FIXTURE=None
TRACE_RECEIPTS=[]
ORIGINAL_RESOLVER=integration._resolve_loaded
ORIGINAL_WRITE=real_boundary.write_exclusive
ORIGINAL_RUN_DIR=support.run_dir
EVENTS=[]
FAULT=None
PARAMETER_TOKEN=object()
HOOK_TOKEN=object()
SETUP_TOKEN=object()
CLEAN_TOKEN=object()
def conv(hidden_states,weight,bias=None,activation=None):raise RuntimeError("NO_NUMERICAL_WORK")
def chunk(query,key,value,g,beta,chunk_size=64,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False):
    raise RuntimeError("NO_NUMERICAL_WORK")
def wrapped(base):
    implementation=base
    def call(*args,**kwargs):return implementation(*args,**kwargs)
    call.__wrapped__=base
    return call
def fixture(folder,name):
    from test_topology import topology
    f=topology(TYPES);f.folder=folder;f.calls=[];f.stopped=False
    support.run_dir=lambda:folder
    f.execution={"mode":"INERT_FIXTURE","fixture_id":name,"actual_model_work":False}
    f.admitted={"execution":f.execution}
    f.writer=types.SimpleNamespace(root=folder/"evidence/attempt")
    def stop(code):f.stopped=True;f.calls.append("stop")
    def restore():f.calls.append("original_restore")
    f.guard=types.SimpleNamespace(restore=restore)
    context=types.SimpleNamespace(legacy_parameters=None,latch=types.SimpleNamespace(stop=stop))
    def capture(*args):
        need(len(args)==3 and args[0] is f.bridge,"unchanged legacy capture arguments")
        context.legacy_parameters=(PARAMETER_TOKEN,);f.calls.append("legacy_capture")
    context.capture_legacy_before=capture
    f.writer.loader_diagnostics=context
    pair=(wrapped(conv),wrapped(chunk))
    module=types.SimpleNamespace(Qwen3_5GatedDeltaNet=TYPES.GDN,causal_conv1d_fn=pair[0],torch_chunk_gated_delta_rule=pair[1])
    f.targets=PACKAGE["compat"].Targets(module,TYPES.GDN,*pair,ROOT,mode="INERT_FIXTURE")
    def resolver(model,admitted,package):
        need(model is f.bridge and admitted==f.admitted and package is PACKAGE,"explicit inert dependency identities")
        f.calls.append("inert_source_declared_resolver")
        return f.targets,f.hf,f.bridge,f.bindings
    integration._resolve_loaded=resolver
    f.resolver=resolver
    integration.reserve_helpers(f.writer,f.admitted)
    def recorder(*args):
        need(all("causal_conv1d_fn" in vars(f.bridge.blocks[i].linear_attn.original_component) and
                 "chunk_gated_delta_rule" in vars(f.bridge.blocks[i].linear_attn.original_component)
                 for i in range(24) if (i+1)%4),"helpers precede new original setup")
        f.calls.append("original_create_recorder");return SETUP_TOKEN
    f.recorder=recorder
    bind_trace(f)
    return f
def bind_trace(f):
    global ACTIVE_FIXTURE
    need(ACTIVE_FIXTURE is None and forward_trace.ACTIVE is None,"FIXTURE_TRACE_NO_LEAK")
    raw=(HERE/"SOURCE_FREEZE.json").read_bytes()
    source=json.loads(raw)["source_sha256"]
    f.trace_sources={name:digest for name,digest in source.items() if name.endswith(".py")}
    allowlist={str((HERE/name).absolute()).casefold():{"source":name,"sha256":digest}
               for name,digest in f.trace_sources.items()}
    f.trace=forward_trace.Trace(f.execution,sha(raw),allowlist,f.writer.loader_diagnostics.latch.stop)
    f.prior_trace=forward_trace.ACTIVE
    forward_trace.ACTIVE=f.trace
    ACTIVE_FIXTURE=f

def finish_trace(f):
    global ACTIVE_FIXTURE
    need(ACTIVE_FIXTURE is f and forward_trace.ACTIVE is f.trace,"FIXTURE_TRACE_EXACT_BINDING")
    try:
        status=f.trace.publish(lambda raw:put(f.folder/"INERT_TRACE.json",raw)["sha256"])
        save(f.folder/"INERT_TRACE_STATUS.json",status)
        raw=(f.folder/"INERT_TRACE.json").read_bytes()
        proof=trace_reader.interpret(raw,status,f.execution,f.trace.source_sha256,f.trace_sources)
        save(f.folder/"INDEPENDENT_TRACE_RESULT.json",proof)
        need(proof["trace_verified"] and len(raw)<=65536,"INERT_TRACE_AUTHENTICATED")
        f.trace_proof=proof
        TRACE_RECEIPTS.append({"fixture":f.execution["fixture_id"],"path":str(f.folder.relative_to(HERE)),
            "bytes":len(raw),"sha256":sha(raw),"events":status["event_count"],
            "primary_stage":status["primary"]["stage"] if status["primary"] else None,
            "trace_verified":True,"actual_model_work":False})
    finally:
        forward_trace.ACTIVE=f.prior_trace
        ACTIVE_FIXTURE=None

def loader_splice(f):
    # The actual diagnostic core observes the load handoff with this same span.
    with forward_trace.span("LOAD_HANDOFF"):
        return untraced_loader_splice(f)

def untraced_loader_splice(f):
    tree=ast.parse((HERE/"candidate_loader.py").read_bytes())
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="_diagnostic_load_adapter")
    original_try=next(x for x in fn.body if isinstance(x,ast.Try))
    start=next(i for i,x in enumerate(original_try.body) if isinstance(x,ast.Expr) and isinstance(x.value,ast.Call)
        and isinstance(x.value.func,ast.Attribute) and x.value.func.attr=="capture_legacy_before")
    selected=original_try.body[start:start+4]
    need(len(selected)==4 and isinstance(selected[-1],ast.Assign) and selected[-1].targets[0].id=="recorder","exact source splice")
    # Execute only this exact source slice, with explicit inert original callbacks.
    env={"diagnostics":f.writer.loader_diagnostics,"backend":types.SimpleNamespace(model=f.bridge),
         "parameter_digest":PARAMETER_TOKEN,"compatibility":{"weight_sha256":"INERT_NOT_A_WEIGHT_HASH"},
         "writer":f.writer,"guard":f.guard,"admitted":f.admitted,"create_recorder":f.recorder,
         "latch":f.writer.loader_diagnostics.latch,"labels":(),"spec":{},"sys":sys}
    try:exec(compile(ast.Module(body=copy.deepcopy(selected),type_ignores=[]),str(HERE/"candidate_loader.py"),"exec"),env)
    except BaseException:
        # Exact inherited-plus-helper exception path, without any loader/import statements.
        handler=original_try.handlers[0].body[:-1]
        exec(compile(ast.Module(body=copy.deepcopy(handler),type_ignores=[]),str(HERE/"candidate_loader.py"),"exec"),env)
        raise
    need(env["recorder"] is SETUP_TOKEN,"original recorder return identity")
    return env["helper"]
def check(f):
    def original():
        need(not f.stopped,"ORIGINAL_LATCH_REFUSAL");f.calls.append("original_pre_forward_latch");return HOOK_TOKEN
    need(integration.checkpoint(f.writer,original) is HOOK_TOKEN,"original checkpoint return identity")
def cleanup(f):
    need(ACTIVE_FIXTURE is f and forward_trace.ACTIVE is f.trace,"FIXTURE_TRACE_EXACT_BINDING")
    f.trace.cleanup=True
    def end():f.calls.append("original_end_edit")
    def parameter():f.calls.append("original_parameter_predicates_stub");return PARAMETER_TOKEN
    def admit():f.calls.append("original_cleanup_latch");need(not f.stopped,"ORIGINAL_LATCH_REFUSAL")
    def inspect():f.calls.append("original_hook_inspect_stub");return HOOK_TOKEN
    def identity(p,h):
        f.calls.append("original_cold_identity_stub");need(p is PARAMETER_TOKEN and h is HOOK_TOKEN,"identity")
        return CLEAN_TOKEN
    return integration.close(f.writer,lambda restore:diagnostic_cleanup(end,parameter,admit,inspect,identity,restore))
def worker_finally(status,folder,result=None):
    # Execute the actual production worker's unconditional finally body only.
    tree=ast.parse((HERE/"production_run.py").read_bytes())
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="worker")
    final=next(x for x in fn.body if isinstance(x,ast.Try)).finalbody
    prior=sys.modules.get("diagnostic_core")
    core=types.ModuleType("diagnostic_core");core.LAST_HELPER_STATUS=status
    sys.modules["diagnostic_core"]=core
    try:
        exec(compile(ast.Module(body=copy.deepcopy(final),type_ignores=[]),str(HERE/"production_run.py"),"exec"),
             {"write_new":support.write_new,"identity":{"execution":status.get("execution",{})},"result":result or {"inert_only":True}})
    finally:
        if prior is None:sys.modules.pop("diagnostic_core",None)
        else:sys.modules["diagnostic_core"]=prior
    saved=json.loads((folder/"control/WORKER_RESULT.json").read_bytes())
    need(saved["helper_status"]==status,"unconditional worker finally retains exact helper status")
    return saved
def outer(f):
    value=integration.record_outer(f.writer)
    worker=worker_finally(value,f.folder)
    proof=helper_reader.verify_helper(f.folder/"control",f.execution,value,worker["helper_status"])
    save(f.folder/"INDEPENDENT_HELPER_RESULT.json",proof)
    finish_trace(f)
    return value,proof
def expect_failure(call):
    try:call()
    except BaseException:return
    raise RuntimeError("EXPECTED_FIXED_FAILURE")
def set_fault(name):
    global FAULT
    FAULT=name
    def writer(path,raw,root):
        if pathlib.Path(path).name==FAULT:
            ORIGINAL_WRITE(path,raw[:19],root)
            raise OSError("INERT_PARTIAL_NATIVE_IO")
        return ORIGINAL_WRITE(path,raw,root)
    real_boundary.write_exclusive=writer
def clear_fault():
    global FAULT
    FAULT=None;real_boundary.write_exclusive=ORIGINAL_WRITE
def run_group(name,folder):
    global CLEAN
    results=[]
    try:
        if name=="clean_loader_native_join":
            f=fixture(folder/"clean",name)
            bare={k:sys.modules.get(k) for k in ("support","selection","source_contract","fingerprint","handoff")}
            loader_splice(f);check(f);need(cleanup(f) is CLEAN_TOKEN,"unchanged diagnostic cleanup result")
            value,proof=outer(f)
            need(proof["binding_verified"] and not value["first_code"] and
                f.calls==["legacy_capture","inert_source_declared_resolver","original_create_recorder",
                          "original_pre_forward_latch","original_end_edit","original_parameter_predicates_stub",
                          "original_cleanup_latch","original_hook_inspect_stub","original_cold_identity_stub","original_restore"],
                "clean exact integration order")
            need(all(sys.modules.get(k) is v for k,v in bare.items()),"no bare import-name collisions")
            need(TOTAL_RESERVED==172032 and OTHER_REMAINDER+TOTAL_RESERVED==OTHER_ORIGINAL==786432,"existing reservation partition")
            need(f.trace_proof["first_failure"] is None and len(f.trace_proof["events"])==14 and
                 [e["stage"] for e in f.trace_proof["events"] if e["edge"]=="ENTER"]==
                 ["LOAD_HANDOFF","CLEANUP_END_EDIT","CLEANUP_PARAMETER_STATE","CLEANUP_LATCH_INSPECTION",
                  "CLEANUP_HOOK_INSPECTION","CLEANUP_IDENTITY","CLEANUP_GUARD_RESTORE"] and
                 all(e["cleanup"] is (e["stage"]!="LOAD_HANDOFF") for e in f.trace_proof["events"]),
                 "real finite trace wraps original clean callbacks")
            CLEAN=(f,value)
            results.append({"complete":True,"original_order_preserved":True,"native_helper_reserve":TOTAL_RESERVED,
                            "relative_import_namespace_isolated":True,"calls":f.calls})
        elif name=="mandatory_live_admission":
            from launch import preflight
            disabled=preflight("0"*64)
            need(disabled["production_authorized"] is False and not (HERE/"root_release").exists(),"actual disabled preflight")
            results.append({"disabled_preflight":disabled})
            for kind in ("inert_cannot_use_live_resolver","unexpected_existing_helper","private_module_collision"):
                f=fixture(folder/kind,kind)
                prior=sys.modules["helper_pkg.selection"]
                if kind.startswith("inert"):integration._resolve_loaded=ORIGINAL_RESOLVER
                elif kind.startswith("unexpected"):f.bridge.blocks[0].linear_attn.original_component.causal_conv1d_fn=object()
                else:sys.modules["helper_pkg.selection"]=types.ModuleType("INERT_COLLISION")
                try:expect_failure(lambda:loader_splice(f))
                finally:sys.modules["helper_pkg.selection"]=prior
                value,proof=outer(f)
                need(not proof["binding_verified"] and f.stopped and "original_create_recorder" not in f.calls and
                     f.calls.count("original_restore")>=1,"admission denied before new reference")
                results.append({"case":kind,"first_code":value["first_code"],"original_setup_unrun":True})
        elif name=="native_setup_publication_faults":
            for target in ("HELPER_SETUP.json","HELPER_ADMISSION.json"):
                f=fixture(folder/target.split(".")[0],target);set_fault(target)
                expect_failure(lambda:loader_splice(f));clear_fault()
                value,proof=outer(f)
                need(not proof["binding_verified"] and f.stopped and
                     (f.folder/"control"/target).stat().st_size==19,"native partial preserved")
                need(all("causal_conv1d_fn" not in vars(f.bridge.blocks[i].linear_attn.original_component)
                         for i in range(24) if (i+1)%4),"only owned helpers rolled back")
                results.append({"case":target,"partial_bytes":19,"incomplete":True,"restorations":f.calls.count("original_restore")})
        elif name=="original_cleanup_refusal":
            f=fixture(folder/"refusal",name);loader_splice(f);check(f);f.stopped=True
            expect_failure(lambda:cleanup(f));value,proof=outer(f)
            need(not proof["binding_verified"] and "original_hook_inspect_stub" not in f.calls and
                 value["session"]["primary"]["phase"]=="ORIGINAL_CLEANUP" and f.calls.count("original_restore")==1,
                 "original cleanup refusal stays primary and provider unrun")
            need(f.trace_proof["first_failure"]["stage"]=="CLEANUP_LATCH_INSPECTION" and
                 f.trace_proof["first_failure"]["phase"]=="CLEANUP" and
                 f.trace_proof["events"][-1]["stage"]=="CLEANUP_GUARD_RESTORE" and
                 f.trace_proof["events"][-1]["edge"]=="RETURN",
                 "original terminal refusal remains first and restoration is traced")
            results.append({"first_phase":value["session"]["primary"]["phase"],"provider_unrun":True,"restore_count":1})
        elif name=="outer_closeout_failure_survival":
            for target in ("HELPER_TERMINAL.json","HELPER_OUTER_STATUS.json"):
                f=fixture(folder/target.split(".")[0],target);loader_splice(f);check(f)
                if target=="HELPER_TERMINAL.json":
                    set_fault(target);expect_failure(lambda:cleanup(f));clear_fault()
                else:
                    need(cleanup(f) is CLEAN_TOKEN,"clean original before outer IO")
                    set_fault(target)
                value,proof=outer(f);clear_fault()
                need(not proof["binding_verified"] and value["first_code"] is not None and
                     (f.folder/"control"/target).stat().st_size==19,"outer partial cannot pass")
                need(not (f.folder/"control/DIAGNOSTIC_TERMINAL.json").exists(),"missing diagnostic terminal not fabricated")
                results.append({"case":target,"finite_status_retained_in_actual_worker_finally":True,"partial_bytes":19,
                                "missing_original_terminal_stays_incomplete":True})
        elif name=="saved_join_tamper":
            f,clean=CLEAN
            # Read saved clean bytes, apply synthetic rebindings only in fresh output directories.
            for kind in ("coherent_saved_copy","setup_bytes","admitted_packet","execution","outer_status"):
                dest=folder/kind;control=dest/"control";support.run_dir=lambda d=dest:d
                blobs={p.name:p.read_bytes() for p in (f.folder/"control").glob("*.json") if p.name!="WORKER_RESULT.json"}
                live=copy.deepcopy(clean);live.pop("outer_pointer")
                if kind=="setup_bytes":blobs["HELPER_SETUP.json"]+=b" "
                if kind=="admitted_packet":
                    value=json.loads(blobs["HELPER_ADMISSION.json"]);value["selection_graph_sha256"]="0"*64
                    blobs["HELPER_ADMISSION.json"]=encoded(value)
                if kind=="execution":live["execution"]["fixture_id"]="REBOUND"
                if kind=="outer_status":live["session"]["selection_graph_sha256"]="0"*64
                # Repair only outer envelopes; each declared inner corruption remains.
                for key,name in (("reservation_pointer","HELPER_RESERVATION.json"),("admission_pointer","HELPER_ADMISSION.json")):
                    live[key]={"path":(control/name).relative_to(HERE).as_posix(),"bytes":len(blobs[name]),"sha256":sha(blobs[name])}
                blobs["HELPER_OUTER_STATUS.json"]=encoded(live)
                status=dict(live);status["outer_pointer"]={"path":(control/"HELPER_OUTER_STATUS.json").relative_to(HERE).as_posix(),
                    "bytes":len(blobs["HELPER_OUTER_STATUS.json"]),"sha256":sha(blobs["HELPER_OUTER_STATUS.json"])}
                for name,raw in blobs.items():put(control/name,raw)
                proof=helper_reader.verify_helper(control,f.execution,status,status)
                need(proof["binding_verified"] is (kind=="coherent_saved_copy"),"coherent copy passes and inner tamper rejects")
                save(dest/"RESULT.json",{"proof":proof,"synthetic_copy":True,"outer_envelopes_repaired":True,"no_real_authority":True})
                results.append({"case":kind,"binding_verified":proof["binding_verified"]})
            source=ast.parse((HERE/"diagnostic_reader.py").read_bytes())
            assignment=next(x for x in ast.walk(source) if isinstance(x,ast.Assign) and len(x.targets)==1 and
                isinstance(x.targets[0],ast.Name) and x.targets[0].id=="clean" and isinstance(x.value,ast.BoolOp)
                and any(isinstance(y,ast.Subscript) and isinstance(y.value,ast.Name) and y.value.id=="helper_proof" for y in ast.walk(x)))
            for original in (False,True):
                for helper in (False,True):
                    env={"clean":original,"helper_proof":{"binding_verified":helper}}
                    exec(compile(ast.Module(body=[copy.deepcopy(assignment)],type_ignores=[]),"actual_saved_AND","exec"),env)
                    need(env["clean"] is (original and helper),"cannot upgrade original failure")
            results.append({"original_predicate_AND_four_rows_verified":True})
        else:raise RuntimeError("UNDECLARED_CASE")
    finally:
        pending=sys.exception()
        try:
            if ACTIVE_FIXTURE is not None:
                try:finish_trace(ACTIVE_FIXTURE)
                except BaseException:
                    # Preserve the original failing callback as the batch's first error.
                    # Missing/incomplete trace stays a failed group, never PASS.
                    if pending is None:raise
        finally:
            clear_fault();integration._resolve_loaded=ORIGINAL_RESOLVER;support.run_dir=ORIGINAL_RUN_DIR
    raw={"case":name,"status":"PASS_INERT_INTEGRATION_ONLY","outcomes":results,"scientific_pass":False,
         "model_imports":0,"loads":0,"parameter_accesses":0,"parameter_hashes":0,"forwards":0,"derivatives":0,"encodings":0}
    ack=save(folder/"GROUP_RESULT.json",raw)
    return {"case":name,"status":raw["status"],"result":ack}
def main():
    global PACKAGE,TYPES
    start=time.monotonic();end=start+45;absolute=start+60
    timer=threading.Timer(59,lambda:os._exit(124));timer.daemon=True;timer.start()
    raw=(HERE/"BATCH_LOCK.json").read_bytes();need(len(sys.argv)==2 and sha(raw)==sys.argv[1],"CALLER_LOCK")
    lock=json.loads(raw);freeze_raw=(HERE/"SOURCE_FREEZE.json").read_bytes();freeze=json.loads(freeze_raw)
    need(lock["source_sha256"]==sha(freeze_raw) and lock["cases"]==list(CASES) and lock["limits"]==LIMITS,"FROZEN_BATCH")
    for name,digest in freeze["source_sha256"].items():need(sha((HERE/name).read_bytes())==digest,"SOURCE_HASH")
    u=(HERE/"USAGE_BEFORE_BATCH.json").read_bytes();need(sha(u)==lock["usage_sha256"],"USAGE_HASH")
    usage=json.loads(u);payload=json.loads(usage["tool_result"]["content"][0]["text"])
    bucket=payload["rateLimitsByLimitId"]["codex"];used=bucket["primary"]["usedPercent"]
    need(type(used) in (int,float) and 0<=used<100 and not bucket["spendControlReached"] and
         0<=time.time()-usage["observed_unix_seconds"]<=120,"ACTUAL_FRESH_USAGE")
    root=HERE/"test_evidence";need(not root.exists(),"ONE_BATCH");root.mkdir()
    save(root/"BATCH_STARTED.json",{"source_sha256":lock["source_sha256"],"batch_lock_sha256":sha(raw),
        "started":start,"substantive_deadline":end,"absolute_deadline":absolute})
    report={"status":"INCONCLUSIVE","source_sha256":lock["source_sha256"],"batch_lock_sha256":sha(raw),
        "cases":[],"remaining_unrun":list(CASES),"scientific_pass":False,"retry_count":0,
        "model_imports":0,"loads":0,"parameter_accesses":0,"parameter_hashes":0,"forwards":0,"derivatives":0,"encodings":0}
    try:
        PACKAGE=integration.ensure_package()
        from test_topology import build_types,EXTRACTIONS
        TYPES=build_types()
        save(root/"SOURCE_AST_REUSE.json",EXTRACTIONS)
        for case in CASES:
            need(time.monotonic()<end,"SUBSTANTIVE_DEADLINE")
            report["cases"].append(run_group(case,root/case));report["remaining_unrun"].remove(case)
            need(time.monotonic()<end,"SUBSTANTIVE_DEADLINE")
        need(not any(x.split(".")[0] in BLOCKED for x in sys.modules),"NO_MODEL_IMPORTS")
        need(ACTIVE_FIXTURE is None and forward_trace.ACTIVE is None,"NO_TRACE_LEAK_AFTER_BATCH")
        need(len(TRACE_RECEIPTS)==9,"EXACT_NINE_EXISTING_FIXTURES")
        save(root/"INERT_TRACE_RECEIPTS.json",TRACE_RECEIPTS)
        support.bounds();report["status"]="PASS_INERT_INTEGRATION_ONLY"
    except BaseException as exc:
        report["failure_code"]=getattr(exc,"code","FIXED_INTEGRATION_BATCH_FAILURE")
        frames=[];tb=exc.__traceback__
        while tb and len(frames)<12:
            p=pathlib.Path(tb.tb_frame.f_code.co_filename)
            if p.is_absolute() and p.is_relative_to(HERE):frames.append({"source":p.relative_to(HERE).as_posix(),"line":tb.tb_lineno})
            tb=tb.tb_next
        report["failure_frames"]=frames
    finally:
        close=time.monotonic();report["elapsed_seconds"]=close-start
        save(root/"BATCH_REPORT.json",report)
        save(root/"BATCH_CLOSEOUT.json",{"closed":True,"elapsed_seconds":time.monotonic()-start,
             "shared_closeout_seconds":time.monotonic()-close,"absolute_envelope_met":time.monotonic()<absolute,
             "owned_processes_spawned":0,"real_authority":False})
        support.bounds();timer.cancel()
    print(json.dumps({"status":report["status"],"completed":len(report["cases"]),"remaining_unrun":report["remaining_unrun"],
                      "elapsed_seconds":report["elapsed_seconds"]}))
    return 0 if report["status"]=="PASS_INERT_INTEGRATION_ONLY" else 1
if __name__=="__main__":raise SystemExit(main())
