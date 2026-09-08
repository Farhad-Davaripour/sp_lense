"""Independent bounded saved-trace validation; no writer or model imports."""
import hashlib
import json

STAGES={"LOAD_HANDOFF","FORWARD_ADAPTER","BRIDGE_DISPATCH","SELECTED_HOOK",
    "HOOK_CONTEXT_ENTER","HOOK_CONTEXT_EXIT","ADAPTER_POSTCONDITIONS","LOGITS_SERIALIZATION",
    "LOGITS_PUBLICATION","INPUT_CAPTURE_IDENTITY","CLEAR_CAPTURE","CLEANUP_END_EDIT",
    "CLEANUP_PARAMETER_STATE","CLEANUP_PARAMETER_DIGEST","CLEANUP_LATCH_INSPECTION",
    "CLEANUP_HOOK_INSPECTION","CLEANUP_IDENTITY","CLEANUP_GUARD_RESTORE","WRITER_CLOSEOUT"}
CATEGORIES={"VALUE_ERROR","TYPE_ERROR","ATTRIBUTE_ERROR","KEY_ERROR","INDEX_ERROR","OS_ERROR",
    "MEMORY_ERROR","TIMEOUT_ERROR","ASSERTION_ERROR","OVERFLOW_ERROR","RUNTIME_ERROR","OTHER_BASE_EXCEPTION"}

def require(ok,code):
    if not ok:raise ValueError(code)

def strict(raw):
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,"TRACE_DUPLICATE_KEY");value[key]=item
        return value
    def invalid(value):raise ValueError("TRACE_NONFINITE")
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)

def interpret(raw,status,execution,source_sha256,source_files):
    result={"trace_verified":False,"interpretation":"TRACE_INCOMPLETE_UNKNOWN","scientific_pass":False}
    try:
        require(type(raw) is bytes and len(raw)<=65536,"TRACE_RAW_BOUND")
        value=strict(raw)
        require(set(value)=={"schema","execution","source_sha256","primary","secondary_cleanup_failures","events",
            "trace_incomplete","trace_code","open_stages","scope","scientific_pass","exception_text_serialized",
            "locals_or_tensors_serialized","maximum_bytes"},"TRACE_EXACT_SCHEMA")
        require(value["schema"]=="first_forward_finite_trace.v1" and value["scope"]=="ONE_LOCKED_UNEDITED_BASELINE_ONLY"
            and value["execution"]==execution==status["execution"] and value["source_sha256"]==source_sha256==status["source_sha256"],"TRACE_PROVENANCE")
        require(status["schema"]=="first_forward_trace_status.v1" and status["receipt_sha256"]==hashlib.sha256(raw).hexdigest()
            and status["published"] is True and status["closed"] is True and status["io_failed"] is False
            and status["incomplete"] is False and status["retry_allowed"] is False and status["scientific_pass"] is False,"TRACE_CLOSED_ACK")
        require(value["scientific_pass"] is False and value["exception_text_serialized"] is False
            and value["locals_or_tensors_serialized"] is False and value["maximum_bytes"]==65536
            and value["trace_incomplete"] is False and value["trace_code"] is None and status["trace_code"] is None
            and value["open_stages"]==[],"TRACE_COMPLETE_FINITE")
        require(value["primary"]==status["primary"] and value["secondary_cleanup_failures"]==status["secondary_cleanup_failures"],"TRACE_FIRST_CAUSE_COPY")
        events=value["events"];require(type(events) is list and 0<len(events)<=96 and len(events)==status["event_count"],"TRACE_EVENT_COUNT")
        stack=[];raised=[]
        for i,event in enumerate(events):
            require(set(event)=={"ordinal","stage","edge","depth","cleanup"} and event["ordinal"]==i
                and event["stage"] in STAGES and type(event["cleanup"]) is bool,"TRACE_EVENT_SCHEMA")
            if event["edge"]=="ENTER":
                require(event["depth"]==len(stack),"TRACE_ENTER_DEPTH");stack.append(event["stage"])
            else:
                require(event["edge"] in {"RETURN","RAISE"} and stack and stack[-1]==event["stage"] and event["depth"]==len(stack),"TRACE_LEAVE_DEPTH")
                stack.pop()
                if event["edge"]=="RAISE":raised.append(event)
        require(not stack,"TRACE_COMPLETE_STACK")
        failures=([value["primary"]] if value["primary"] is not None else [])+value["secondary_cleanup_failures"]
        require(type(value["secondary_cleanup_failures"]) is list and len(value["secondary_cleanup_failures"])<=16,"TRACE_SECONDARY_BOUND")
        require(bool(raised)==bool(value["primary"]),"TRACE_FAILURE_EVENT_JOIN")
        if raised:require(value["primary"]["stage"]==raised[0]["stage"],"TRACE_FIRST_INNER_FAILURE")
        for finding in failures:
            require(set(finding)=={"stage","category","frames","frames_complete","frame_count","unallowlisted_frames","frame_scan_complete","phase"}
                and finding["stage"] in STAGES and finding["category"] in CATEGORIES and finding["phase"] in {"EXECUTION","CLEANUP"},"TRACE_FINDING_SCHEMA")
            require(any(x["stage"]==finding["stage"] and x["cleanup"]==(finding["phase"]=="CLEANUP") for x in raised),"TRACE_FINDING_RAISED")
            require(type(finding["frames"]) is list and len(finding["frames"])<=8
                and type(finding["frame_count"]) is int and 0<=finding["frame_count"]<=512
                and type(finding["unallowlisted_frames"]) is int and 0<=finding["unallowlisted_frames"]<=finding["frame_count"]
                and type(finding["frames_complete"]) is bool and type(finding["frame_scan_complete"]) is bool,"TRACE_FRAME_BOUND")
            for frame in finding["frames"]:
                require(set(frame)=={"source","sha256","line"} and source_files.get(frame["source"])==frame["sha256"]
                    and type(frame["line"]) is int and 1<=frame["line"]<=1000000,"TRACE_FRAME_SOURCE")
            if finding["frames_complete"]:require(finding["frame_scan_complete"] and finding["unallowlisted_frames"]==0
                and finding["frame_count"]==len(finding["frames"]),"TRACE_NO_HIDDEN_FRAMES")
        require(all(x["phase"]=="CLEANUP" for x in value["secondary_cleanup_failures"]),"TRACE_SECONDARY_CLEANUP_ONLY")
        first=value["primary"]
        result.update(trace_verified=True,interpretation="CLEAN_EXECUTION_TRACE" if first is None else
            ("FIRST_FAILURE_STAGE_AND_ALLOWLISTED_LOCATIONS" if first["frames"] else "FIRST_FAILURE_STAGE_SOURCE_UNKNOWN"),
            first_failure=first,secondary_cleanup_failures=value["secondary_cleanup_failures"],events=events,
            complete_traceback=bool(first and first["frames_complete"]))
    except (ValueError,KeyError,TypeError,IndexError,AttributeError):pass
    return result
