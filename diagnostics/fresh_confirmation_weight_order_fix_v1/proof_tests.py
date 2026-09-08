"""One retained-order regression batch; no actual model or old suite."""
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
    from weight_reader import interpret
    protocol=strict((HERE/"TEST_PROTOCOL.json").read_bytes());inputs=strict((HERE/"TEST_INPUTS.json").read_bytes())
    started=time.monotonic();cutoff=started+45;absolute=started+60
    hard=threading.Timer(59.5,lambda:os._exit(124));hard.daemon=True;hard.start()
    write_new("BATCH_STARTED.json",{"started_monotonic":started,"substantive_cutoff":cutoff,"absolute_deadline":absolute,
        "source_sha256":lock["source_sha256"],"batch_lock_sha256":sys.argv[1],"real_execution":False},preparation=True,critical=True)
    result={"schema":"frozen_weight_fix_batch.v1","status":"INCONCLUSIVE_MODEL_FREE_WEIGHT_FIX",
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
                return original_digest(parameters)
            namespace["parameter_digest"]=measured_digest
            model=Model();state=None;adapter=None;constructor_accepted=False;boundary_concat_preserved=None
            if name=="wrong_initial":model.items[0].values[0]=1.5
            try:
                context.enter("HOOK_SETUP");context.capture_enumeration("BEFORE_HOOK_SETUP",list(model.named_parameters()))
                context.capture_legacy_before(model,measured_digest,inputs["mock_expected_sha256"])
                retained=context.legacy_parameters
                model.after=name!="original"
                current=list(model.named_parameters())
                if name=="mutation_before":model.items[0].values[0]=1.5
                if name=="omitted_object":current=current[:-1]
                if name=="replaced_object":
                    target=current[0][1];replacement=Parameter(1)
                    require(replacement.values.tobytes()==target.values.tobytes(),"same-byte replacement fixture")
                    current[0]=(current[0][0],replacement)
                if name=="duplicate_object":current.append(("additional_occurrence",model.items[0]))
                if name=="byte_boundaries":
                    before_bytes=b"".join(p.values.tobytes() for p in retained)
                    model.items[0].values=array.array("f",[1.0]);model.items[0].shape=(1,)
                    model.items[1].values=array.array("f",[2.0,2.0,3.0]);model.items[1].shape=(3,)
                    boundary_concat_preserved=before_bytes==b"".join(p.values.tobytes() for p in retained)
                    require(boundary_concat_preserved,"same concatenated bytes but different parameter boundaries")
                model.named_parameters=lambda:iter(current)
                context.enter("ADAPTER_CONSTRUCTOR")
                adapter=namespace["RealAdapter"](model,None,SimpleNamespace(latch=None),expected_weight_sha256=inputs["mock_expected_sha256"],
                    diagnostics=context,legacy_weight_parameters=None if name=="missing_reference" else retained)
                constructor_accepted=True
                require(adapter.legacy_weights.parameters is retained and type(retained) is tuple and len(retained)==12,"exact strong tuple retained")
                require(adapter.named_parameters==current and all(x is y for x,y in zip(adapter.parameters,[p for _,p in current],strict=True)),"current registry arrays unchanged")
                context.ready(adapter)
                if name=="mutation_closeout":model.items[0].values[0]=1.5
                if name=="later_registry":current.reverse()
                # Actual candidate method, including fresh full-content hash, not a simulated result.
                state=adapter.parameter_state(digest=True)
                receiver_ok=all(v for v in state.values() if type(v) is bool)
            except BaseException:
                if constructor_accepted:raise
                context.failure_closeout();receiver_ok=False
            terminal=context.terminal()
            pointer=write_new("FIX_TERMINAL.json",{"diagnostic":terminal,"parameter_state":state,"receiver_ok":receiver_ok,
                "constructor_accepted":constructor_accepted,"hash_argument_identities":calls,"model_work":"BYTE_STANDINS_ONLY",
                "source_sha256":lock["source_sha256"],"execution":identity["execution"]},critical=True)
            saved=strict(boundary.read_control("FIX_TERMINAL.json"));native=boundary.read_control("LOADER_DIAGNOSTICS.json")
            interpreted=interpret(native,saved["diagnostic"],identity["execution"],lock["source_sha256"],inputs["mock_expected_sha256"])
            expected=protocol["expectations"][name]
            require(constructor_accepted==expected["constructor_accepted"] and receiver_ok==expected["receiver_ok"],"frozen constructor/closeout result")
            require(interpreted["interpretation"]==expected["interpretation"],"independent saved native binding interpretation")
            require(len(calls)==expected["hash_calls"] and all(x==calls[0] for x in calls),"fresh hashes all use identical legacy order, never current traversal")
            require(terminal["legacy_weight_binding"]["additional_hash_completed"]==1,"one additional pre-hook full-content pass")
            if state is not None:
                require(saved["receiver_ok"]==all(v for v in saved["parameter_state"].values() if type(v) is bool),"saved all-identity closeout")
                require(state["parameter_sha256"]==inputs["mock_expected_sha256"] if name!="mutation_closeout" else state["parameter_sha256"]!=inputs["mock_expected_sha256"],"fresh closeout digest")
            if name in ("mutation_before","mutation_closeout"):
                require(all(p._version==0 and p.shape==(2,) and p.requires_grad and p.grad is None for p in model.items),"true byte change with unchanged flags/versions/shape")
            if name=="mutation_before":
                require(terminal["first_failure"]["predicate_code"]=="LD_ORDERED_WEIGHT_DIGEST","unchanged constructor equality rejects real content mutation")
            if name=="later_registry":
                require(not state["parameter_identities_unchanged"] and state["parameter_bytes_unchanged"],"registry drift fails even with correct legacy content")
            # Saved receipt tamper only, no second constructor/hash execution or retry.
            forged=copy.deepcopy(saved["diagnostic"]);forged["receipt_sha256"]="0"*64
            require(interpret(native,forged,identity["execution"],lock["source_sha256"],inputs["mock_expected_sha256"])["interpretation"]=="BINDING_INCOMPLETE","forged join cannot pass")
            case.update(status="PASS",interpretation=interpreted,constructor_accepted=constructor_accepted,receiver_ok=receiver_ok,
                first_failure=terminal["first_failure"],hash_calls=len(calls),additional_hash_calls=1,terminal=pointer,native_sha256=sha(native),
                complete_native_readback=True,unchanged_concatenated_bytes_with_boundary_drift=boundary_concat_preserved,
                parameter_state=state,model_work="BYTE_STANDINS_ONLY",scientific_cells_unrun=180,requests_unrun=48,baselines_unrun=24)
            bounds();require(time.monotonic()<cutoff,"substantive work includes every reader")
        result["status"]="PASS_MODEL_FREE_WEIGHT_ORDER_FIX_ONLY"
    except BaseException:
        result["failure"]="FROZEN_WEIGHT_FIX_BATCH_FAILURE"
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
    print(json.dumps(result,sort_keys=True));return 0 if result["status"]=="PASS_MODEL_FREE_WEIGHT_ORDER_FIX_ONLY" else 1

if __name__=="__main__":raise SystemExit(main())
