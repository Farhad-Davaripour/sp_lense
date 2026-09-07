"""Finite terminal evidence survives failures of ordinary/index/fault writers."""
from support import require


def close_runtime(model,gate,writer,state,event,encoded):
    record=None
    final_identity=None
    hook_capture=None
    codes=[]
    def technical(code):
        if code not in codes: codes.append(code)
        state["status"]="INCONCLUSIVE_STUDY"
        state["technical_failures"].append({"code":code})
    try:
        if model is not None:
            try:
                if model.request_id is not None:
                    model.finish_request()
                final_identity=model.finalize()
                hook_capture=model.recorder.finish()
                state["cleanup_complete"]=all(v for v in final_identity.values() if type(v) is bool)
            except BaseException:
                model.latch.stop("FINAL_IDENTITY_FAILURE")
                technical("FINAL_IDENTITY_FAILURE")
            finally:
                # Restoration is best-effort but never clears either terminal state.
                try: model.end_edit(); model.request_id=None
                except BaseException: technical("PARAMETER_RESTORE_FAILURE")
                try: model.guard.restore()
                except BaseException: technical("DISPATCH_GUARD_RESTORE_FAILURE")
        if gate is not None:
            try: gate.unchanged()
            except BaseException: technical("GATE_IDENTITY_FAILURE")
        if model is not None:
            state["adapter_accounting"]={"forwards":model.guard.forwards,"derivatives":model.guard.derivatives,
                "rejected_dispatches":model.guard.rejected,"dispatch_failed":model.latch.failed}
            record={"execution":state["execution"],"metadata":model.runtime_metadata,
                "initial_parameter_sha256":model.initial_digest,"final_identity":final_identity,
                "hook_checks":model.recorder.checks,"hook_capture":hook_capture,
                "hook_controller_status":model.recorder.status(),"input_bound":True}
        if not writer.sticky_failure:
            try:
                if record is not None: writer.write_source("runtime_adapter.json",encoded(record))
                event("gate_final.json",{"decisions":state["fresh_routes"],"parameters_unchanged":not any(c=="GATE_IDENTITY_FAILURE" for c in codes),"fit_calls":0})
                event("integration_cleanup.json",{"complete":state["cleanup_complete"],"measured_adapter_identity":True})
                writer.append_log("worker_stdout.log",(state["status"]+"\n").encode())
            except BaseException: technical("RUNTIME_RECORD_IO_FAILURE")
    finally:
        # status() is finite/non-I/O and is captured AFTER every above failure.
        status=model.recorder.status() if model is not None else None
        try:
            capture=writer.closeout()
        except BaseException:
            technical("EVIDENCE_CLOSEOUT_FAILURE")
            capture={"path":None,"sha256":None,"status":"INCOMPLETE"}
        try: reconciliation=writer._reconciliation()
        except BaseException: reconciliation={"issues":["RECONCILIATION_UNAVAILABLE"],"actual_total_bytes":None}
        capture["terminal"]={"schema":"confirmation_terminal.v2","execution":state["execution"],
            "state":state,"recorder_status":status,"runtime_record":record,
            "writer_sticky":writer.sticky_failure,"writer_sealed":writer.sealed,
            "actual_total_bytes":reconciliation["actual_total_bytes"],"accounting_available":reconciliation["actual_total_bytes"] is not None,
            "complete_evidence":capture["status"]=="COMPLETE" and state["status"]!="INCONCLUSIVE_STUDY",
            "finite_closeout_codes":codes}
    return capture
