"""One startup-only worker operation. No prompt or scientific finalizer is invoked."""
import json
import time
from support import require,write_new,sha
from setup_counter import Counter
from setup_resources import reserve,bounded_control
from loader_diagnostics import encoded

def execute(boundary,approved,admission,deadline,mock_case=None):
    admitted=boundary.role("writer",approved,admission)
    cap=reserve(boundary.mock)
    from setup_io import make_writer
    writer=make_writer(cap);counter=Counter();adapter=None;cleanup=None;index=None;failures=[]
    def publisher(raw):
        require(len(raw)<=cap["diagnostic"],"reserved finite native diagnostic")
        if mock_case=="partial_publisher":
            from mock_setup import partial_native_publication
            return partial_native_publication(lambda data:write_new("LOADER_DIAGNOSTICS.json",data,raw=True,critical=True),raw)
        write_new("LOADER_DIAGNOSTICS.json",raw,raw=True,critical=True)
    writer.loader_diagnostic_publisher=publisher
    if boundary.mock:
        from mock_setup import mock_plan
        plan=mock_plan()
    else:
        from plan import build_plan
        plan=build_plan()
    state={"execution":admitted["execution"],"scope":"STARTUP_ONLY_NO_QUESTIONS","cell_status":{x["cell_id"]:"UNRUN" for x in plan["cells"]},
        "request_status":{x["request_id"]:"UNRUN" for x in plan["requests"]},"baselines_unrun":len(plan["prompts"]),"scientific_status":"UNRUN"}
    writer.workflow_state=state
    try:
        require(time.monotonic()<deadline,"startup worker deadline")
        counter.reserve_load()
        def invoke(identity,target):
            nonlocal adapter
            require(identity==admitted and target==boundary.bindings["loader"],"exact startup target and authority")
            from setup_loader import load
            adapter=load(writer,counter,deadline,admitted,mock_case)
            return adapter
        adapter=boundary.dispatch(approved,admission,invoke,sentinel=boundary.mock)
        require(counter.load_calls==1,"one actual startup dispatch")
        try:
            if mock_case=="cleanup_failure": raise RuntimeError("FINITE_INJECTED_CLEANUP_FAILURE")
            adapter.end_edit()
            parameter=adapter.parameter_state(digest=True)
            inspection=adapter.recorder.guard.inspect(adapter.model)
            current=json.dumps(inspection["current"],sort_keys=True,separators=(",",":"),allow_nan=False).encode()
            setup=json.loads((writer.root/"hook_evidence/setup_receipt.json").read_bytes())
            reference=setup["values"]["reference"]["raw_sha256"]
            require(inspection["matches"] is True and inspection["changes"]==[] and sha(current)==reference,"exact setup cleanup registry identity")
            require(all(v for v in parameter.values() if type(v) is bool),"exact setup parameter cleanup identity")
            cleanup={"complete":True,"parameter_state":parameter,"current_registry_sha256":sha(current),"reference_sha256":reference,
                "normal_scientific_hook_checks":0,"full_scientific_finalizer_called":False}
        except BaseException:
            failures.append("SETUP_CLEANUP_FAILURE")
            adapter.latch.stop("SETUP_CLEANUP_FAILURE")
        finally:
            try: adapter.guard.restore()
            except BaseException: failures.append("SETUP_GUARD_RESTORE_FAILURE")
    except BaseException:
        failures.append("SETUP_CONSTRUCTION_OR_NATIVE_IO_FAILURE")
        if adapter is not None and adapter.guard.installed:
            adapter.latch.stop("SETUP_NATIVE_CLOSEOUT_FAILURE")
            try: adapter.end_edit()
            except BaseException: failures.append("SETUP_EDIT_CLEANUP_FAILURE")
            try: adapter.guard.restore()
            except BaseException: failures.append("SETUP_GUARD_RESTORE_FAILURE")
    finally:
        counter.sealed=True
        context=getattr(writer,"loader_diagnostics",None)
        diagnostic=context.terminal() if context is not None else None
        if counter.guard is not None:
            guard={"forwards":counter.guard.forwards,"derivatives":counter.guard.derivatives,"rejected":counter.guard.rejected,
                "restored":not counter.guard.installed,"first_dispatch_code":counter.guard.latch.adapter_primary_code}
        else:
            guard={"forwards":0,"derivatives":0,"rejected":None,"restored":"INHERITED_LOADER_EXCEPTION_PATH"}
        try:
            index=writer.closeout()
            require(index["bytes"]<=cap["index"],"reserved setup index cap")
        except BaseException: failures.append("SETUP_WRITER_CLOSEOUT_FAILURE")
        status="SETUP_DIAGNOSTIC_COMPLETE" if not failures and cleanup and diagnostic and not diagnostic["diagnostic_io_failed"] and counter.reason is None else "INCONCLUSIVE_SETUP"
        terminal={"schema":"startup_only_terminal.v1","execution":admitted["execution"],"status":status,"scientific_status":"UNRUN",
            "scientific_pass":False,"state":state,"counts":counter.snapshot(),"guard":guard,"cleanup":cleanup,"failures":failures,
            "loader_diagnostics":diagnostic,"recorder_status":diagnostic.get("recorder_status") if diagnostic else None,"index":index,
            "first_failure":diagnostic.get("first_failure") if diagnostic and diagnostic.get("first_failure") else (failures[0] if failures else None)}
        if mock_case=="missing_terminal": return {"terminal_omitted_by_frozen_fixture":True,"status":"INCONCLUSIVE_SETUP","execution":admitted["execution"]}
        pointer=bounded_control("SETUP_TERMINAL.json",terminal,cap["terminal"])
    return {"execution":admitted["execution"],"setup_terminal":pointer,"status":terminal["status"],"scientific_pass":False}
