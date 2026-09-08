"""One added-join byte-stand-in batch; the old owned eight-case suite is not run."""
import array
import ast
import copy
import contextlib
import hashlib
import json
import os
import sys
import threading
import time
from types import SimpleNamespace
from support import HERE,ROOT,SOURCES,require,sha,write_new,bounds,check_freeze
from real_boundary import Boundary,encoded,strict,write_exclusive,usage_value

class Parameter:
    def __init__(self,index):
        self.values=array.array("f",[float(index+1),float(index+2)]);self.shape=(2,)
        self.device=SimpleNamespace(type="cpu");self.dtype="float32";self.grad=None;self.requires_grad=True;self._version=0
    def numel(self): return len(self.values)
    def element_size(self): return self.values.itemsize
    def is_floating_point(self): return True
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.values

class Model:
    def __init__(self):
        self.items=[Parameter(i) for i in range(12)];self.after=False;self.duplicate=False
        self.training=False;self._last_hf_cache=None
    def named_parameters(self):
        if not self.after:return iter([("parameter_"+str(i),p) for i,p in enumerate(self.items)])
        pairs=[("blocks.%s._ln1_module.weight"%i,self.items[2*i+1]) for i in range(6)]
        pairs += [("parameter_"+str(i),self.items[i]) for i in range(0,12,2)]
        if self.duplicate:pairs.append(("unexpected_duplicate",self.items[0]))
        return iter(pairs)
    def parameters(self):return (p for _,p in self.named_parameters())

def main():
    require(len(sys.argv)==2,"explicit caller lock")
    lockraw=(HERE/"BATCH_LOCK.json").read_bytes();require(sha(lockraw)==sys.argv[1],"caller frozen batch lock")
    lock=strict(lockraw)
    for name,key in (("SOURCE_FREEZE.json","source_sha256"),("TEST_INPUTS.json","test_inputs_sha256"),("TEST_PROTOCOL.json","test_protocol_sha256")):
        require(sha((HERE/name).read_bytes())==lock[key],"locked source/case bytes")
    check_freeze()
    usage=strict((HERE/"USAGE_BEFORE_BATCH.json").read_bytes())
    require(0<=time.time()-usage["observed_unix_seconds"]<=120 and usage_value(usage["tool_result"])["used_percent"]<100,"fresh actual standard usage")
    from production_run import forbid_research_imports
    forbid_research_imports()
    from source_parity import check
    from loader_diagnostics import Context,encoded as compact,ALGORITHM_SHA
    from order_reader import interpret
    protocol=strict((HERE/"TEST_PROTOCOL.json").read_bytes());inputs=strict((HERE/"TEST_INPUTS.json").read_bytes())
    started=time.monotonic();cutoff=started+45;absolute=started+60
    hard=threading.Timer(59.5,lambda:os._exit(124));hard.daemon=True;hard.start()
    write_new("BATCH_STARTED.json",{"started_monotonic":started,"substantive_cutoff":cutoff,"absolute_deadline":absolute,
        "source_sha256":lock["source_sha256"],"batch_lock_sha256":sys.argv[1],"real_execution":False},preparation=True,critical=True)
    result={"schema":"frozen_order_join_batch.v1","status":"INCONCLUSIVE_MODEL_FREE_ORDER_JOIN",
        "cases":[{"name":name,"status":"UNRUN"} for name in protocol["cases"]],"real_loads":0,"forwards":0,"derivatives":0,
        "tokenizer_calls":0,"old_suite_reruns":0,"actual_root_authority":False,"failure":None}
    active=None
    try:
        result["source_parity"]=check()
        require(result["source_parity"]["digest_function_source_sha256"]==ALGORITHM_SHA,"exact pinned digest function source SHA")
        for pin in strict((HERE/"REUSED_PROOFS.json").read_bytes())["files"]:
            data=(ROOT/pin["path"]).read_bytes();require(len(data)==pin["bytes"] and sha(data)==pin["sha256"],"immutable prior proof hashes, no rerun")
        from launch import preflight
        disabled=preflight("0"*64);require(not disabled["production_authorized"] and not (HERE/"root_release").exists() and not (HERE/"real_evidence").exists(),"actual launch disabled")
        write_new("DISABLED_STATE.json",disabled,preparation=True,critical=True)
        for case in result["cases"]:
            active=case;name=case["name"];require(time.monotonic()<cutoff,"single substantive deadline")
            case_root=HERE/"test_evidence"/name;case_root.mkdir(parents=True,exist_ok=False)
            release=copy.deepcopy(inputs["release"]);auth=copy.deepcopy(inputs["authorization"])
            usage_raw=encoded(inputs["mock_usage"]);release["usage_sha256"]=sha(usage_raw);release_raw=encoded(release)
            auth["release_sha256"]=sha(release_raw);auth_raw=encoded(auth)
            authority=encoded({"schema":"real_root_authority_lock.v1","source_sha256":lock["source_sha256"],"release_sha256":sha(release_raw),"authorization_sha256":sha(auth_raw)})
            for filename,data in (("ROOT_RELEASE.json",release_raw),("AUTHORIZATION.json",auth_raw),("USAGE_RECEIPT.json",usage_raw),("AUTHORITY_LOCK.json",authority)):
                write_exclusive(case_root/"root_release"/filename,data,case_root)
            boundary=Boundary(case_root,mock=True);approved=sha(authority);admission,identity=boundary.admit_once(approved,now=1060.)
            os.environ.update(SP_SETUP_FIXTURE=name,SP_SETUP_BATCH_LOCK=sys.argv[1],SP_CONFIRMATION_AUTHORITY_SHA=approved,SP_CONFIRMATION_ADMISSION_SHA=admission)
            def publisher(data):
                require(len(data)<=2*1024**2,"unchanged finite native diagnostic cap")
                write_new("LOADER_DIAGNOSTICS.json",data,raw=True,critical=True)
            context=Context(identity["execution"],publisher,sha(SOURCES.read("84bfd749876c44dc9bb1bc1e75db89f3e491c159","diagnostics/fresh_confirmation_loader_diagnostics_v1/SOURCE_FREEZE.json")))
            raw=(HERE/"candidate_real_adapter.py").read_text();tree=ast.parse(raw)
            tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ("parameter_digest","RealAdapter")]
            namespace={"torch":SimpleNamespace(float32="float32"),"contextmanager":contextlib.contextmanager,"require":require,"sha":sha,"hashlib":hashlib,"time":time}
            exec(compile(ast.fix_missing_locations(tree),"FROZEN_BYTE_STANDIN_ADAPTER_ONLY","exec"),namespace)
            original_digest=namespace["parameter_digest"];calls=[]
            def measured_digest(parameters):
                calls.append([id(p) for p in parameters])
                if name=="missing_c" and len(calls)==3:raise RuntimeError("FINITE_MISSING_C_FIXTURE")
                return original_digest(parameters)
            namespace["parameter_digest"]=measured_digest
            model=Model()
            if name=="wrong_initial":model.items[0].values[0]=1.5
            context.enter("HOOK_SETUP");context.capture_enumeration("BEFORE_HOOK_SETUP",list(model.named_parameters()))
            context.capture_legacy_before(model,measured_digest)
            retained=context._retained
            model.after=True
            if name=="content_mutation":model.items[0].values[0]=1.5
            if name=="multiplicity_mismatch":model.duplicate=True
            if name=="byte_length_change":model.items[0].values.append(3.0);model.items[0].shape=(3,)
            context.enter("ADAPTER_CONSTRUCTOR");rejected=False
            try:
                namespace["RealAdapter"](model,None,SimpleNamespace(latch=None),expected_weight_sha256=inputs["mock_expected_sha256"],diagnostics=context)
            except BaseException:
                rejected=True;context.failure_closeout()
            require(rejected,"unchanged constructor never accepted these fixed failing cases")
            terminal=context.terminal();terminal_pointer=write_new("ORDER_TERMINAL.json",terminal,raw=False,critical=True)
            saved=strict(boundary.read_control("ORDER_TERMINAL.json"));native=boundary.read_control("LOADER_DIAGNOSTICS.json")
            interpreted=interpret(native,saved,identity["execution"],lock["source_sha256"],inputs["mock_expected_sha256"])
            expected=protocol["expectations"][name]
            require(interpreted["interpretation"]==expected["interpretation"] and interpreted["permits_runtime_admission"] is False
                and interpreted["scientific_pass"] is False,"independent expected interpretation, never runtime permission")
            require(context._retained is retained and type(retained) is tuple and len(retained)==12,"complete strong immutable original list")
            require(terminal["order_fingerprint"]["extra_hash_attempts"]==terminal["order_fingerprint"]["extra_hash_completed"]==expected["additional_hash_calls"],"exact two added passes, incomplete prefixes honest")
            if expected["additional_hash_calls"]==2:require(calls[0]==calls[1] and calls[0]!=calls[2],"B hashes retained order, C current traversal")
            if name=="order_only":require(interpreted["A"]==interpreted["B"]==inputs["mock_expected_sha256"] and interpreted["C"]==inputs["mock_post_order_sha256"],"exact reused byte-fixture digest expectations")
            if name in ("order_only","content_mutation","wrong_initial"):
                require(terminal["first_failure"]["predicate_code"]=="LD_ORDERED_WEIGHT_DIGEST","original failing equality predicate retained")
            # One saved-join tamper negative, no second model/constructor or IO retry.
            forged=copy.deepcopy(saved);forged["receipt_sha256"]="0"*64
            require(interpret(native,forged,identity["execution"],lock["source_sha256"],inputs["mock_expected_sha256"])["interpretation"]=="PROOF_INCOMPLETE","unauthenticated terminal cannot classify ordering")
            case.update(status="PASS",interpretation=interpreted,first_failure=terminal["first_failure"],additional_hash_calls=expected["additional_hash_calls"],
                original_constructor_rejected=True,terminal=terminal_pointer,native_sha256=sha(native),complete_native_readback=True,
                model_work="BYTE_STANDINS_ONLY",scientific_cells_unrun=180,requests_unrun=48,baselines_unrun=24)
            bounds();require(time.monotonic()<cutoff,"substantive work includes every reader")
        result["status"]="PASS_MODEL_FREE_ORDER_JOIN_ONLY"
    except BaseException:
        result["failure"]="FROZEN_ORDER_JOIN_BATCH_FAILURE"
        if active is not None:active["status"]="FAIL"
    finally:
        closeout_started=time.monotonic()
        for key in ("SP_SETUP_FIXTURE","SP_SETUP_BATCH_LOCK","SP_CONFIRMATION_AUTHORITY_SHA","SP_CONFIRMATION_ADMISSION_SHA"):os.environ.pop(key,None)
        result.update(elapsed_seconds=time.monotonic()-started,shared_cleanup_maximum_seconds=15,
            closeout_started_monotonic=closeout_started,absolute_deadline=absolute,source_sha256=lock["source_sha256"],
            batch_lock_sha256=sys.argv[1],area=bounds())
        write_new("BATCH_REPORT.json",result,preparation=True,critical=True)
        closeout={"shared_cleanup_seconds":time.monotonic()-closeout_started,"absolute_envelope_met":time.monotonic()<=absolute,"no_owned_model_processes_started":True}
        write_new("BATCH_CLOSEOUT.json",closeout,preparation=True,critical=True);hard.cancel()
    print(json.dumps(result,sort_keys=True));return 0 if result["status"]=="PASS_MODEL_FREE_ORDER_JOIN_ONLY" else 1

if __name__=="__main__":raise SystemExit(main())
