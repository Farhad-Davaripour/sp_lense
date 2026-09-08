"""One locked zero-offset baseline, no scores/routes/requests or continuation."""
import json
import time
import forward_trace
from forward_trace import Trace,span,encoded
from trace_operations import first_call,cleanup_step,diagnostic_cleanup
from diagnostic_counter import Counter
from diagnostic_support import exact_input,allowlist,reserve,publish_trace,CELL_ID
from support import HERE,require,sha,write_new
LAST_FINITE_STATUS=None
LAST_HELPER_STATUS=None
LAST_OPERAND_STATUS=None

def execute(boundary,approved,admission,deadline,mock_case=None):
    global LAST_FINITE_STATUS,LAST_HELPER_STATUS,LAST_OPERAND_STATUS
    require(not boundary.mock and mock_case is None,"real diagnostic has no injection fallback")
    admitted=boundary.role("writer",approved,admission)
    from plan import build_plan
    plan=build_plan();locked,cell=exact_input(plan);cap=reserve()
    from diagnostic_io import make_writer
    writer=make_writer(cap);counter=Counter(CELL_ID);adapter=None;cleanup=None;index=None;failures=[]
    state={"execution":admitted["execution"],"scope":"ONE_LOCKED_UNEDITED_BASELINE_ONLY","scientific_status":"UNRUN",
        "cell_status":{x["cell_id"]:"UNRUN" for x in plan["cells"]},"request_status":{x["request_id"]:"UNRUN" for x in plan["requests"]},
        "diagnostic_cell_id":CELL_ID,"scoring_calls":0,"routes":0,"forward_returned":False,"logits_published":False}
    writer.workflow_state=state
    def stop(code):
        context=getattr(writer,"loader_diagnostics",None)
        if context is not None and context.latch is not None:context.latch.stop(code)
    trace=Trace(admitted["execution"],sha((HERE/"SOURCE_FREEZE.json").read_bytes()),allowlist(),stop)
    forward_trace.ACTIVE=trace
    try:
        from helper_binding import reserve_helpers,checkpoint as helper_checkpoint,close as helper_close,record_outer
        reserve_helpers(writer,admitted)
        from constructor_operands import reserve_operands
        reserve_operands(writer,admitted,write_new)
        require(time.monotonic()<deadline,"one baseline worker cutoff")
        counter.reserve_load()
        with span("LOAD_HANDOFF"):
            from loader import load_adapter
            adapter=load_adapter(writer,counter,deadline,admitted)
        require(counter.load_calls==1,"one counted real backend load")
        require(time.monotonic()<deadline,"no diagnostic forward after deadline")
        helper_checkpoint(writer,adapter.latch.admit);adapter.latch.consume(CELL_ID)
        counter.reserve_forward(CELL_ID);state["cell_status"][CELL_ID]="STARTED"
        writer.append_event("forward_events.jsonl",encoded({"event":"started","cell_id":CELL_ID,"attempt":1}))
        import torch  # Only after authenticated retained ownership and real loader admission.
        delta=torch.zeros(1024)
        try:
            def call():
                value=adapter.forward_inputs(locked["input_ids"],locked["attention_mask"],"baseline",delta)
                state["forward_returned"]=True
                return value
            def serialize(value):
                z,h=value
                return z.detach().numpy().astype("<f4", copy=False).tobytes()
            def publish(raw):
                value=writer.write_logits("logits/001.f32",raw)
                state["logits_published"]=True
                return value
            def validate(value):
                require(adapter.capture["input_int64_le_sha256"]==locked["input_int64_le_sha256"]
                    and adapter.capture["final_input_index"]==locked["final_input_index"],"actual dispatched tensor proof")
            first_call(call,serialize,publish,validate)
            state["cell_status"][CELL_ID]="DIAGNOSTIC_CAPTURED"
            writer.append_event("forward_events.jsonl",encoded({"event":"completed","cell_id":CELL_ID}))
        except BaseException:
            state["cell_status"][CELL_ID]="FAILED";adapter.latch.stop("CELL_FAILURE");raise
        finally:
            trace.cleanup=True
            cleanup_step("CLEAR_CAPTURE",adapter.clear_capture)
    except BaseException:
        failures.append("WORKER_EXCEPTION");stop("WORKER_EXCEPTION")
    finally:
        counter.sealed=True;trace.cleanup=True
        if adapter is not None:
            try:
                def identity(parameter,inspection):
                    current=json.dumps(inspection["current"],sort_keys=True,separators=(",",":"),allow_nan=False).encode()
                    receipt=json.loads((writer.root/"hook_evidence/setup_receipt.json").read_bytes())
                    reference=receipt["values"]["reference"]["raw_sha256"]
                    require(inspection["matches"] is True and inspection["changes"]==[] and sha(current)==reference,"exact setup cleanup registry identity")
                    parameter.update(hook_registry_restored=True,active_request_empty=adapter.request_id is None,
                        wrapper_cache_empty=adapter.capture is None and adapter.graph_leaf is None,
                        bridge_cache_empty=getattr(adapter.model,"_last_hf_cache",None) is None)
                    require(all(v for v in parameter.values() if type(v) is bool),"complete measured cold receiver identity")
                    return {"diagnostic_cleanup_complete":True,"scientific_pass":False,"parameter_state":parameter,
                        "current_registry_sha256":sha(current),"reference_sha256":reference,
                        "normal_scientific_hook_checks":0,"normal_hook_checks_unrun":109,
                        "full_scientific_finalizer_called":False,"latch_admitted_before_provider":True}
                cleanup=helper_close(writer,lambda restore:diagnostic_cleanup(adapter.end_edit,lambda:adapter.parameter_state(digest=True),adapter.latch.admit,
                    lambda:adapter.recorder.guard.inspect(adapter.model),identity,restore))
            except BaseException:
                failures.append("DIAGNOSTIC_CLEANUP_FAILURE");adapter.latch.stop("DIAGNOSTIC_CLEANUP_FAILURE")
        try:index=cleanup_step("WRITER_CLOSEOUT",writer.closeout)
        except BaseException:failures.append("EVIDENCE_CLOSEOUT_FAILURE")
        try:trace.publish(publish_trace)
        except BaseException:failures.append("TRACE_PUBLICATION_FAILURE")
        from helper_binding import record_outer
        from constructor_operands import terminal_status
        LAST_OPERAND_STATUS=terminal_status(writer)
        if not LAST_OPERAND_STATUS.get("complete"):failures.append("CONSTRUCTOR_OPERANDS_INCOMPLETE")
        LAST_HELPER_STATUS=record_outer(writer)
        if LAST_HELPER_STATUS.get("first_code") is not None:failures.append("HELPER_INTEGRATION_INCOMPLETE")
        # Status is copied after every possible write/closeout failure, not inferred from an index.
        context=getattr(writer,"loader_diagnostics",None);diagnostic=context.terminal() if context else None
        guard=counter.guard
        terminal={"schema":"first_forward_diagnostic_terminal.v1","execution":admitted["execution"],
            "status":"INCONCLUSIVE_DIAGNOSTIC" if failures else "FIRST_FORWARD_DIAGNOSTIC_CAPTURED","scientific_pass":False,
            "state":state,"counts":counter.snapshot(),"trace_status":trace.status(),"loader_diagnostics":diagnostic,
            "normal_hook_checks_unrun":109,"full_scientific_finalizer_called":False,
            "recorder_status":adapter.recorder.status() if adapter is not None else (diagnostic.get("recorder_status") if diagnostic else None),
            "guard_restored":not guard.installed if guard else None,"cleanup":cleanup,"failures":failures,"index":index,
            "helper_status":LAST_HELPER_STATUS,"constructor_operands":LAST_OPERAND_STATUS}
        LAST_FINITE_STATUS=trace.status()
        raw=encoded(terminal);require(len(raw)<=cap["terminal"],"reserved complete native diagnostic terminal")
        pointer=write_new("DIAGNOSTIC_TERMINAL.json",raw,raw=True,critical=True)
        forward_trace.ACTIVE=None
    return {"execution":admitted["execution"],"diagnostic_terminal":pointer,"status":terminal["status"],"scientific_pass":False}
