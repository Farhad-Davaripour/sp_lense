"""One frozen six-group inert batch. No numerical functions are executed."""
import ast
import copy
import functools
import hashlib
import inspect
import json
import os
import pathlib
import sys
import threading
import time
import types
from typing import Callable

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BLOCKED = {"torch", "transformers", "transformer_lens", "tokenizers", "safetensors", "research_backend", "numpy"}
class NoModelImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise RuntimeError("FORBIDDEN_MODEL_IMPORT")
sys.meta_path.insert(0, NoModelImports())

from source_contract import Rejected, need, authenticate, parse_sources, source_facts, sha
from compat import Targets, Installation, HELPERS

CASES = ("old_missing_helpers", "transparent_argument_mapping", "existing_or_unexpected_instance",
         "source_and_callable_rejection", "partial_install_rollback", "closeout_identity_failure")
LIMITS = {"substantive_seconds":45,"shared_cleanup_seconds":15,"absolute_seconds":60,
          "preparation_bytes":33554432,"test_evidence_bytes":8388608,"per_file_bytes":5242880}
FAULTS = {}
CALLS = []
RETURN = object()

def put(path, raw):
    need(len(raw) <= LIMITS["per_file_bytes"], "FILE_CAP")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    need(path.read_bytes() == raw, "WRITE_READBACK")
    return {"bytes":len(raw),"sha256":sha(raw)}

def save(path, value):
    return put(path, (json.dumps(value, sort_keys=True, indent=2)+"\n").encode())

def denied(action, expected):
    try:
        action()
    except Rejected as exc:
        need(exc.code == expected, "WRONG_REJECTION_CODE")
        return exc.code
    raise Rejected("EXPECTED_REJECTION")

class InertOriginal:
    def __init__(self):
        self.marker = object()
    def __setattr__(self, name, value):
        fault = FAULTS.get((id(self), name, "set"))
        if fault == "before":
            raise OSError("NOT_RETAINED")
        object.__setattr__(self, name, value)
        if fault == "after":
            raise OSError("NOT_RETAINED")
    def __delattr__(self, name):
        if FAULTS.get((id(self), name, "delete")):
            raise OSError("NOT_RETAINED")
        object.__delattr__(self, name)

def capture(name, values):
    CALLS.append((name, values.copy()))
    return RETURN

def fixtures():
    receipt, trees = parse_sources(ROOT)
    # Preserve the actual decorator implementation AST; replace optional imports
    # with an inert object that reports package absence, and the outer decorator
    # with an identity fixture. No imported HF/kernel code is executed.
    decorator = copy.deepcopy(next(x for x in trees["decorator"].body
        if isinstance(x, ast.FunctionDef) and x.name == "use_kernel_func_from_hub_with_fallback"))
    class InertImports:
        def import_module(self, name):
            raise ImportError("INERT_ABSENT_PACKAGE")
    env = {"Callable":Callable,"inspect":inspect,"functools":functools,"importlib":InertImports(),
           "_KERNELS_INTERNAL_PATH_MAPPINGS":{},"use_kernel_forward_from_hub":lambda name:lambda f:f}
    exec(compile(ast.Module(body=[decorator], type_ignores=[]), "<INERT_EXACT_DECORATOR_AST>", "exec"), env)
    made = []
    for name, tag in (("causal_conv1d_fn","conv"),("torch_chunk_gated_delta_rule","chunk")):
        node = copy.deepcopy(next(x for x in trees["hf"].body if isinstance(x,ast.FunctionDef) and x.name==name))
        for arg in node.args.posonlyargs+node.args.args+node.args.kwonlyargs:
            arg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        node.returns = None
        node.decorator_list = []
        node.body = ast.parse("return capture("+repr(tag)+", locals())").body
        ast.fix_missing_locations(node)
        local = {"capture":capture}
        exec(compile(ast.Module(body=[node],type_ignores=[]), "<INERT_HF_SIGNATURE_BODY_REPLACED>", "exec"), local)
        base = local[name]
        fn = env["use_kernel_func_from_hub_with_fallback"](name,"INERT_PACKAGE")(base)
        made.append(fn)
    module = types.SimpleNamespace(Qwen3_5GatedDeltaNet=InertOriginal,
        causal_conv1d_fn=made[0],torch_chunk_gated_delta_rule=made[1])
    targets = Targets(module, InertOriginal, *made, ROOT, mode="INERT_FIXTURE")
    return module, targets, made, trees, receipt

def run_case(name, folder):
    FAULTS.clear()
    CALLS.clear()
    module, targets, pair, trees, _ = fixtures()
    outcomes = []
    statuses = []
    if name == "old_missing_helpers":
        cls = next(x for x in trees["bridge"].body if isinstance(x, ast.ClassDef) and x.name=="GatedDeltaNetBridge")
        fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name=="_hooked_forward")
        instance = InertOriginal()
        for helper in HELPERS:
            node = next(x for x in ast.walk(fn) if isinstance(x,ast.Attribute) and x.attr==helper
                and isinstance(x.value,ast.Name) and x.value.id=="hf")
            expression = ast.Expression(body=copy.deepcopy(node))
            try:
                eval(compile(expression,"<INERT_BRIDGE_ATTRIBUTE_AST>","eval"), {"hf":instance})
            except AttributeError:
                outcomes.append(helper)
            else:
                raise Rejected("OLD_MISSING_NOT_OBSERVED")
        need(outcomes==list(HELPERS) and not CALLS, "BOTH_OLD_LOOKUPS")
    elif name == "transparent_argument_mapping":
        objs = [InertOriginal(),InertOriginal()]
        original = [dict(vars(x)) for x in objs]
        tx = Installation(targets, list(zip(("L0","L1"),objs)), before_hook_reference=True).install()
        tokens = [object() for _ in range(7)]
        conv_result = objs[0].causal_conv1d_fn(x=tokens[0],weight=tokens[1],bias=tokens[2],activation=tokens[3],seq_idx=None)
        chunk_result = objs[1].chunk_gated_delta_rule(tokens[0],tokens[1],tokens[2],g=tokens[3],beta=tokens[4],
            initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=True)
        need(conv_result is RETURN and chunk_result is RETURN and len(CALLS)==2,"RETURN_IDENTITY")
        conv_values, chunk_values = CALLS[0][1], CALLS[1][1]
        need(all(conv_values[k] is v for k,v in zip(("hidden_states","weight","bias","activation"),tokens))
             and conv_values["kwargs"]=={},"EXACT_CONV_ARGUMENT_IDENTITY")
        need(all(chunk_values[k] is v for k,v in zip(("query","key","value","g","beta"),tokens))
             and chunk_values["chunk_size"]==64 and chunk_values["initial_state"] is None
             and chunk_values["output_final_state"] is False and chunk_values["use_qk_l2norm_in_kernel"] is True
             and chunk_values["kwargs"]=={},"EXACT_CHUNK_ARGUMENT_IDENTITY")
        need(vars(objs[1])["chunk_gated_delta_rule"] is pair[1] and objs[1].chunk_gated_delta_rule is pair[1],
             "NO_METHOD_SELF_BINDING")
        denied(lambda:objs[0].causal_conv1d_fn(x=tokens[0],weight=tokens[1],bias=None,activation=None,seq_idx=1),
               "NONNEUTRAL_SEQ_IDX")
        need(len(CALLS)==2,"NONNEUTRAL_NO_DISPATCH")
        # Exact decorator filtering demonstrates why the x -> hidden_states map is necessary.
        try:
            pair[0](x=tokens[0],weight=tokens[1],bias=None,activation=None,seq_idx=None)
        except TypeError:
            outcomes.append("unmapped_x_rejected_by_exact_decorator_fixture")
        else:
            raise Rejected("UNMAPPED_X_UNEXPECTED")
        tx.inspect()
        statuses.append(tx.status())
        tx.close()
        need([vars(x) for x in objs]==original,"CLEAN_CLOSEOUT")
        statuses.append(tx.status())
        outcomes += ["all_argument_and_return_identities","neutral_only","exact_chunk_object","clean_rollback"]
    elif name == "existing_or_unexpected_instance":
        for value in (None, lambda:None):
            obj = InertOriginal()
            obj.causal_conv1d_fn = value
            tx = Installation(targets,[("L0",obj)],before_hook_reference=True)
            outcomes.append(denied(tx.install,"EXISTING_HELPER"))
            need(obj.causal_conv1d_fn is value and "chunk_gated_delta_rule" not in vars(obj),"NO_OVERWRITE")
            statuses.append(tx.status())
        class Subclass(InertOriginal):
            pass
        outcomes.append(denied(lambda:Installation(targets,[("L0",Subclass())],before_hook_reference=True),"ORIGINAL_TYPE"))
        obj=InertOriginal()
        outcomes.append(denied(lambda:Installation(targets,[("L0",obj),("L1",obj)],before_hook_reference=True),"DUPLICATE_INSTANCE"))
        outcomes.append(denied(lambda:Installation(targets,[("L0",obj)],before_hook_reference=False),"ADMISSION_PHASE"))
        # Class-provided helpers must also be rejected, even if benign-looking.
        InertOriginal.causal_conv1d_fn = staticmethod(pair[0])
        try:
            tx=Installation(targets,[("L0",obj)],before_hook_reference=True)
            outcomes.append(denied(tx.install,"EXISTING_HELPER"))
        finally:
            del InertOriginal.causal_conv1d_fn
    elif name == "source_and_callable_rejection":
        from source_contract import PINS
        raws={k:(ROOT/p).read_bytes() for k,(p,_) in PINS.items()}
        raws["hf"]=raws["hf"]+b"\n"
        put(folder/"altered_hf_source.bin",raws["hf"])
        outcomes.append(denied(lambda:authenticate(ROOT,raws),"SOURCE_SHA"))
        outcomes.append(denied(lambda:Targets(module,InertOriginal,lambda:None,pair[1],ROOT,mode="INERT_FIXTURE"),"TARGET_IDENTITY"))
        outcomes.append(denied(lambda:Targets(module,InertOriginal,pair[0],pair[1],ROOT,mode="LIVE_EXISTING_MODULE"),"LIVE_MODULE"))
        obj=InertOriginal()
        tx=Installation(targets,[("L0",obj)],before_hook_reference=True)
        module.torch_chunk_gated_delta_rule=lambda:None
        outcomes.append(denied(tx.install,"TARGET_DRIFT"))
        need(not any(k in vars(obj) for k in HELPERS) and not CALLS,"DENIED_BEFORE_HELPERS_OR_CALLS")
        statuses.append(tx.status())
    elif name == "partial_install_rollback":
        # Fail before and after a setter's physical assignment. Earlier instances
        # and the just-written failing attribute must all be rolled back.
        for when in ("before","after"):
            objs=[InertOriginal(),InertOriginal()]
            snapshots=[dict(vars(x)) for x in objs]
            FAULTS[(id(objs[1]),"chunk_gated_delta_rule","set")]=when
            tx=Installation(targets,list(zip(("L0","L1"),objs)),before_hook_reference=True)
            outcomes.append(denied(tx.install,"INSTALL_IO_OR_CALLBACK"))
            need(tx.state=="FAILED" and tx.rollback_complete and [vars(x) for x in objs]==snapshots,"ATOMIC_ROLLBACK")
            outcomes.append(denied(tx.install,"NO_RETRY"))
            statuses.append(tx.status())
            FAULTS.clear()
    elif name == "closeout_identity_failure":
        obj=InertOriginal()
        tx=Installation(targets,[("L0",obj)],before_hook_reference=True).install()
        unexpected=lambda:None
        obj.chunk_gated_delta_rule=unexpected
        outcomes.append(denied(tx.close,"CLOSE_INCOMPLETE"))
        need(tx.first_failure=="HELPER_DRIFT" and not tx.rollback_complete and
             obj.chunk_gated_delta_rule is unexpected and "causal_conv1d_fn" not in vars(obj),"PRESERVE_UNEXPECTED")
        outcomes.append(denied(tx.install,"NO_RETRY"))
        statuses.append(tx.status())
        other=InertOriginal()
        FAULTS[(id(other),"chunk_gated_delta_rule","set")]="after"
        FAULTS[(id(other),"causal_conv1d_fn","delete")]=True
        tx2=Installation(targets,[("L0",other)],before_hook_reference=True)
        outcomes.append(denied(tx2.install,"INSTALL_IO_OR_CALLBACK"))
        need(tx2.state=="FAILED" and not tx2.rollback_complete and tx2.first_failure=="INSTALL_IO_OR_CALLBACK",
             "ROLLBACK_FAILURE_NOT_SUCCESS")
        statuses.append(tx2.status())
    else:
        raise Rejected("UNDECLARED_CASE")
    need(not any(x.split(".")[0] in BLOCKED for x in sys.modules),"NO_MODEL_IMPORTS")
    result={"case":name,"status":"PASS_INERT_COMPONENT_ONLY","outcomes":outcomes,"component_statuses":statuses,
        "mode":"INERT_FIXTURE","model_imports":0,"model_loads":0,"model_constructor_calls":0,
        "parameter_accesses":0,"parameter_hashes":0,"forwards":0,"derivatives":0,"encodings":0,
        "scientific_pass":False,"fixture_transformations":["HF signatures retained; numerical bodies replaced with inert captures",
        "exact kernel decorator AST with inert absent-package import and identity outer decorator",
        "source bridge attribute AST evaluated only on inert objects"]}
    receipt=save(folder/"RESULT.json",result)
    saved=json.loads((folder/"RESULT.json").read_bytes())
    need(saved==result and receipt["sha256"]==sha((folder/"RESULT.json").read_bytes()),"SAVED_RECEIPT")
    return {"case":name,"status":result["status"],"result":receipt}

def check_bounds():
    files=[p for p in HERE.rglob("*") if p.is_file()]
    evidence=[p for p in (HERE/"test_evidence").rglob("*") if p.is_file()]
    need(all(p.stat().st_size<=LIMITS["per_file_bytes"] for p in files),"FILE_CAP")
    need(sum(p.stat().st_size for p in evidence)<=LIMITS["test_evidence_bytes"],"EVIDENCE_CAP")
    need(sum(p.stat().st_size for p in files)-sum(p.stat().st_size for p in evidence)<=LIMITS["preparation_bytes"],"PREP_CAP")

def main():
    start=time.monotonic()
    deadline=start+45
    absolute=start+60
    timer=threading.Timer(59,lambda:os._exit(124))
    timer.daemon=True
    timer.start()
    raw=(HERE/"BATCH_LOCK.json").read_bytes()
    need(len(sys.argv)==2 and sha(raw)==sys.argv[1],"CALLER_BATCH_LOCK")
    lock=json.loads(raw)
    source_raw=(HERE/"SOURCE_FREEZE.json").read_bytes()
    freeze=json.loads(source_raw)
    need(lock["source_sha256"]==sha(source_raw) and lock["cases"]==list(CASES) and lock["limits"]==LIMITS,"FROZEN_LOCK")
    for path,digest in freeze["files"].items():
        need(sha((HERE/path).read_bytes())==digest,"FROZEN_SOURCE")
    u=(HERE/"USAGE_BEFORE_BATCH.json").read_bytes()
    need(sha(u)==lock["usage_sha256"],"USAGE_SHA")
    usage=json.loads(u)
    payload=json.loads(usage["tool_result"]["content"][0]["text"])
    bucket=payload["rateLimitsByLimitId"]["codex"]
    used=bucket["primary"]["usedPercent"]
    need(type(used) in (int,float) and 0<=used<100 and not bucket["spendControlReached"] and
         0<=time.time()-usage["observed_unix_seconds"]<=120,"ACTUAL_FRESH_USAGE")
    need(not (HERE/"test_evidence").exists(),"ONE_BATCH_NO_RETRY")
    (HERE/"test_evidence").mkdir()
    save(HERE/"test_evidence/BATCH_STARTED.json",{"source_sha256":lock["source_sha256"],
        "batch_lock_sha256":sha(raw),"started":start,"substantive_deadline":deadline,"absolute_deadline":absolute})
    report={"schema":"gdn_inert_component_batch.v1","status":"INCONCLUSIVE","source_sha256":lock["source_sha256"],
        "batch_lock_sha256":sha(raw),"cases":[],"remaining_unrun":list(CASES),"scientific_pass":False,
        "actual_model_imports":0,"actual_model_loads":0,"actual_parameter_accesses":0,"actual_parameter_hashes":0,
        "actual_forwards":0,"actual_derivatives":0,"actual_encodings":0,"retry_count":0}
    try:
        facts=source_facts(ROOT)
        save(HERE/"test_evidence/SOURCE_AST_EVIDENCE.json",facts)
        for name in CASES:
            need(time.monotonic()<deadline,"SUBSTANTIVE_DEADLINE")
            report["cases"].append(run_case(name,HERE/"test_evidence"/name))
            report["remaining_unrun"].remove(name)
            need(time.monotonic()<deadline,"SUBSTANTIVE_DEADLINE")
        need(not any(x.split(".")[0] in BLOCKED for x in sys.modules),"NO_MODEL_IMPORTS")
        check_bounds()
        report["status"]="PASS_INERT_COMPONENT_ONLY"
    except BaseException as exc:
        report["failure_code"]=exc.code if type(exc) is Rejected else "FIXED_BATCH_FAILURE"
        report["exception_category"]=type(exc).__name__ if type(exc) in (Rejected,KeyError,TypeError,ValueError,OSError) else "OTHER"
        # Fixed source-only locations; no exception text, locals, or repr.
        tb=exc.__traceback__
        frames=[]
        while tb and len(frames)<8:
            path=pathlib.Path(tb.tb_frame.f_code.co_filename)
            if path.is_absolute() and path.parent==HERE:
                frames.append({"file":path.name,"line":tb.tb_lineno})
            tb=tb.tb_next
        report["failure_source_frames"]=frames
    finally:
        close=time.monotonic()
        report["elapsed_seconds"]=close-start
        save(HERE/"test_evidence/BATCH_REPORT.json",report)
        save(HERE/"test_evidence/BATCH_CLOSEOUT.json",{"closed":True,"elapsed_seconds":time.monotonic()-start,
            "shared_closeout_seconds":time.monotonic()-close,"absolute_envelope_met":time.monotonic()<absolute,
            "owned_processes_spawned":0,"no_real_authority":not (HERE/"root_release").exists()})
        check_bounds()
        timer.cancel()
    print(json.dumps({"status":report["status"],"completed":len(report["cases"]),
        "remaining_unrun":report["remaining_unrun"],"elapsed_seconds":report["elapsed_seconds"]}))
    return 0 if report["status"]=="PASS_INERT_COMPONENT_ONLY" else 1

if __name__=="__main__":
    raise SystemExit(main())
