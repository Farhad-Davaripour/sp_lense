"""Exactly two fake loader/context handoffs; no scientific schedule or parameters."""
import copy
import importlib.abc
import json
import os
import sys
import threading
import time
from types import SimpleNamespace
from support import HERE,SOURCES,require,sha,write_new,bounds,check_freeze
from real_boundary import Boundary,encoded,strict,write_exclusive,usage_value

class NoResearch(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".",1)[0] in {"torch","transformers","transformer_lens","tokenizers","sp_lense","numpy","safetensors","huggingface_hub"}:
            raise RuntimeError("FORBIDDEN_RESEARCH_IMPORT")
        return None

class FakeLatch:
    def __init__(self,writer):self.writer=writer;self.failed=False;self.recorder=None;self.reason=None
    def stop(self,reason):
        self.failed=True;self.reason="H_CAPTURE"
        # Same sticky-writer consequence as the reviewed LatchView hook-fault path;
        # no claim to exercise or repeat the complete hook diagnostic suite.
        self.writer._fail("hook_fault",None)
class FakeRecorder:
    def __init__(self,latch):self.raw=self;self.latch=latch;self.checks=[]
    def status(self):return {"synthetic_handoff_only":True,"terminal":self.latch.failed,"primary_code":self.latch.reason,"permits_pass":False,"normal_hook_checks":0}
    def finish(self):return {"synthetic_handoff_only":True,"real_hook_capacity_proved":False}
class FakeGuard:
    def __init__(self):self.forwards=0;self.derivatives=0;self.rejected=0;self.restores=0
    def restore(self):self.restores+=1
class FakeAdapter:
    def __init__(self,writer,execution,expected,identity):
        self.latch=FakeLatch(writer);self.recorder=FakeRecorder(self.latch);self.latch.recorder=self.recorder
        self.guard=FakeGuard();self.initial_digest=expected;self.request_id=None;self.end_calls=0
        self.runtime_metadata={"model_work":"SYNTHETIC_HANDOFF_NO_MODEL","execution":execution}
        self.identity=identity
    def finalize(self):return copy.deepcopy(self.identity)
    def end_edit(self):self.end_calls+=1

def read_fake_fixture():
    prefix="diagnostics/fresh_confirmation_weight_order_fix_v1/test_evidence/alias_reordered/real_evidence/fresh_confirmation_weight_order_fix_attempt_001/control/"
    commit="e7a689b9c795573c58c72e693579c9acb10ac953"
    raw=SOURCES.read(commit,prefix+"LOADER_DIAGNOSTICS.json")
    saved=strict(SOURCES.read(commit,prefix+"FIX_TERMINAL.json"))
    require(saved["model_work"]=="BYTE_STANDINS_ONLY" and saved["diagnostic"]["receipt_sha256"]==sha(raw),"pinned saved fake evidence only")
    return strict(raw),saved["parameter_state"],{"commit":commit,"native_sha256":sha(raw),"terminal_source_path":prefix+"FIX_TERMINAL.json"}

def main(approved):
    lockraw=(HERE/"BATCH_LOCK.json").read_bytes();require(sha(lockraw)==approved,"caller frozen handoff batch lock")
    lock=strict(lockraw)
    for n,k in (("SOURCE_FREEZE.json","source_sha256"),("TEST_INPUTS.json","test_inputs_sha256"),("TEST_PROTOCOL.json","test_protocol_sha256")):
        require(sha((HERE/n).read_bytes())==lock[k],"locked source/input/case bytes")
    check_freeze();usage=strict((HERE/"USAGE_BEFORE_BATCH.json").read_bytes())
    require(0<=time.time()-usage["observed_unix_seconds"]<=120 and usage_value(usage["tool_result"])["used_percent"]<100,"fresh actual standard usage")
    sys.meta_path.insert(0,NoResearch())
    started=time.monotonic();cutoff=started+45;absolute=started+60
    timer=threading.Timer(59.5,lambda:os._exit(124));timer.daemon=True;timer.start()
    write_new("BATCH_STARTED.json",{"source_sha256":lock["source_sha256"],"batch_lock_sha256":approved,
        "started_monotonic":started,"substantive_cutoff":cutoff,"absolute_deadline":absolute,"model_work":False},preparation=True,critical=True)
    result={"schema":"final_runner_loader_handoff_batch.v1","status":"INCONCLUSIVE_HANDOFF",
        "cases":[{"name":n,"status":"UNRUN"} for n in ("clean_handoff","malformed_binding")],
        "real_model_loads":0,"parameter_accesses":0,"parameter_content_hashes":0,"Qwen_or_torch_imports":0,
        "forwards":0,"derivatives":0,"tokenizer_calls":0,"scientific_schedule_runs":0,"prior_suite_reruns":0,
        "source_sha256":lock["source_sha256"],"batch_lock_sha256":approved,"failure":None}
    active=None
    import support,authority
    old_run_dir=support.run_dir;old_current=authority.current
    try:
        from source_parity import check
        result["source_parity"]=check()
        from launch import preflight
        disabled=preflight("0"*64)
        require(not disabled["production_authorized"] and not (HERE/"root_release").exists() and not (HERE/"real_evidence").exists(),"actual release disabled")
        write_new("DISABLED_STATE.json",disabled,preparation=True,critical=True)
        inputs=strict((HERE/"TEST_INPUTS.json").read_bytes())
        for case in result["cases"]:
            active=case;name=case["name"];require(time.monotonic()<cutoff,"one substantive deadline")
            case_root=HERE/"test_evidence"/name;case_root.mkdir(parents=True,exist_ok=False)
            usage_raw=encoded(inputs["mock_usage"]);release=copy.deepcopy(inputs["mock_release"]);authorization=copy.deepcopy(inputs["mock_authorization"])
            release["usage_sha256"]=sha(usage_raw);release_raw=encoded(release);authorization["release_sha256"]=sha(release_raw);auth_raw=encoded(authorization)
            authority_raw=encoded({"schema":"real_root_authority_lock.v1","source_sha256":lock["source_sha256"],"release_sha256":sha(release_raw),"authorization_sha256":sha(auth_raw)})
            for n,raw in (("USAGE_RECEIPT.json",usage_raw),("ROOT_RELEASE.json",release_raw),("AUTHORIZATION.json",auth_raw),("AUTHORITY_LOCK.json",authority_raw)):
                write_exclusive(case_root/"root_release"/n,raw,case_root)
            boundary=Boundary(case_root,mock=True);ap=sha(authority_raw);ad,admitted=boundary.admit_once(ap,now=1060.)
            execution=admitted["execution"]
            require(not execution["production_authorized"] and execution["permission_scope"]=="MODEL_FREE_SENTINEL_ONLY","synthetic authority never real")
            authority.current=lambda:(boundary,ap,ad)
            support.run_dir=lambda:boundary.output
            boot={"permission_received":True,"execution":execution,"actual_pid":4242,"synthetic_permission_only":True}
            write_exclusive(boundary.control/"owned/production_worker/BOOTSTRAP.json",encoded(boot),boundary.root)
            boundary.role("worker",ap,ad,pid=4242)
            # Build only pure writer and saved-reader components. Engine is never compiled/imported.
            import bind_production
            bind_production.run_dir=support.run_dir
            _,unused_judge=bind_production.bind(include_engine=False)
            area=sys.modules["area"]
            def test_bounds():
                b=bounds();b["evidence_bytes"]=b["test_evidence_bytes"];return b
            area.area_bounds=test_bounds
            writer=area.WorkflowWriter("attempt")
            counters=writer.helper.AttemptCounter(writer.contract);counters.reserve_attempt("load")
            require(counters.limits=={"forward":180,"derivative":48,"load":1},"full scientific counter, never startup0F guard")
            cells=["synthetic_unrun_"+str(i) for i in range(180)]
            state={"execution":execution,"status":"INCONCLUSIVE_STUDY","scientific_failures":[],"technical_failures":[{"code":"SYNTHETIC_HANDOFF_ONLY"}],
                "cell_status":dict.fromkeys(cells,"UNRUN"),"request_status":dict.fromkeys(["synthetic_request_"+str(i) for i in range(48)],"UNRUN"),
                "cleanup_complete":False,"attempts":dict(counters.attempts),"cursor":0,"fresh_routes":0}
            writer.workflow_state=state
            fake_native,identity,origin=read_fake_fixture()
            expected=inputs["frozen_expected_sha256"]
            identity["initial_parameter_sha256"]=identity["parameter_sha256"]=expected
            fake=FakeAdapter(writer,execution,expected,identity)
            write_new("SYNTHETIC_TRANSFORMATION.json",{"schema":"explicit_saved_fake_loader_handoff_rebinding.v1",
                "origin":origin,"execution":execution,"new_source_sha256":lock["source_sha256"],
                "weight_hash_strings_rebound_to_frozen_expected_without_weight_access":True,
                "synthetic_metadata_and_callbacks_only":True,"real_weight_evidence":False,"scientific_slots_unrun":180})
            from loader_handoff import target_loader,run_candidate,verify_loader_binding
            # Authenticate exact future target without invoking it or importing real_adapter.
            selected=target_loader(boundary.bindings["loader"])
            require(selected.__name__=="load_adapter" and "real_adapter" not in sys.modules,"target authenticated without real backend")
            def fake_checked_load(w,c,d,current):
                require(w is writer and c is counters and d==cutoff and current==admitted,"same writer/counter/deadline/identity handoff")
                from loader_diagnostics import new_context
                context=new_context(w,current)
                context.enumerations=copy.deepcopy(fake_native["parameter_enumerations"])
                context.digests=copy.deepcopy(fake_native["already_computed_digests"])
                context.digests["actual_ordered_sha256"]=context.digests["expected_frozen_sha256"]=expected
                context.weight_order=copy.deepcopy(fake_native["legacy_weight_binding"])
                context.weight_order.update(source_sha256=lock["source_sha256"],execution=execution,
                    pre_setup_sha256=expected,constructor_sha256=expected,expected_frozen_sha256=expected)
                if name=="malformed_binding":context.weight_order["source_sha256"]="0"*64
                context.latch=fake.latch;context.enter("ADAPTER_CONSTRUCTOR")
                return context.ready(fake)
            accepted=None;rejected=False
            try:
                accepted=boundary.dispatch(ap,ad,lambda current,target:run_candidate(writer,counters,cutoff,current,fake_checked_load),sentinel=True)
            except BaseException:rejected=True
            require(rejected==(name=="malformed_binding"),"frozen handoff acceptance")
            if rejected:state["technical_failures"].append({"code":"WORKER_EXCEPTION"})
            from runtime_closeout import close_runtime
            compact=lambda x:(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
            event=lambda stream,record:writer.append_event(stream,compact(record))
            capture=close_runtime(accepted,None,writer,state,event,compact)
            pointer=write_new("WORKER_RESULT.json",capture,critical=True)
            saved=strict(boundary.read_control("WORKER_RESULT.json"))
            proof=verify_loader_binding(writer.root,execution,saved["terminal"]["loader_diagnostics"])
            require(proof["binding_verified"]==(name=="clean_handoff"),"independent saved proof handoff")
            # Original scientific saved identity path must stay incomplete for this
            # deliberately non-scientific fixture; the new join cannot upgrade it.
            import saved_runtime
            joined=saved_runtime.verify_runtime_identity(writer.root,state,{"status":"COMPLETE","errors":[]},saved)
            require(not joined["evidence_valid"] and not joined["permits_pass"],"loader proof never upgrades incomplete science")
            require(fake.guard.restores==1 and fake.end_calls==1,"unchanged single effective cleanup/guard restore")
            require(saved["terminal"]["recorder_status"]==fake.recorder.status(),"recorder status survives good/rejected handoff")
            require(counters.attempts=={"load":1,"forward":0,"derivative":0} and set(state["cell_status"].values())=={"UNRUN"},"no scientific dispatch")
            require(saved["status"]==("INCOMPLETE" if rejected else "COMPLETE") and writer.sealed
                and writer.sticky_failure==rejected,"honest actual writer closeout and sticky-fault state")
            case.update(status="PASS",synthetic_only=True,loader_callback_count=1,real_loader_calls=0,handoff_accepted=accepted is fake,
                binding_verified=proof["binding_verified"],source_bound_proof=proof,worker_result=pointer,
                writer_index_sha256=saved["sha256"],guard_restore_count=fake.guard.restores,
                final_counter_limits=counters.limits,attempts=counters.attempts,scientific_cells_unrun=180,requests_unrun=48,baselines_unrun=24,
                native_index_status=saved["status"],writer_sticky=writer.sticky_failure,
                native_closeout_receipt_present=True,scientific_verdict="NOT_TESTED",no_science_upgrade=True)
            write_new("HANDOFF_RESULT.json",case,critical=True)
            require(time.monotonic()<cutoff,"saved handoff audit within shared work deadline")
        result["status"]="PASS_FAKE_LOADER_HANDOFF_ONLY"
    except BaseException:
        result["failure"]="FIXED_HANDOFF_BATCH_FAILURE"
        if active is not None:active["status"]="FAIL"
    finally:
        close_started=time.monotonic()
        authority.current=old_current;support.run_dir=old_run_dir
        result.update(elapsed_seconds=time.monotonic()-started,area=bounds(),shared_cleanup_limit_seconds=15,absolute_deadline=absolute)
        write_new("BATCH_REPORT.json",result,preparation=True,critical=True)
        write_new("BATCH_CLOSEOUT.json",{"shared_cleanup_seconds":time.monotonic()-close_started,
            "absolute_envelope_met":time.monotonic()<=absolute,"no_owned_processes_or_model_calls":True},preparation=True,critical=True)
        timer.cancel()
    print(json.dumps(result,sort_keys=True))
    return 0 if result["status"]=="PASS_FAKE_LOADER_HANDOFF_ONLY" else 1

if __name__=="__main__":
    require(len(sys.argv)==2,"explicit caller batch lock")
    raise SystemExit(main(sys.argv[1]))
