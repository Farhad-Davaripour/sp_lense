"""Finite observational trace. No model dependencies, values, locals or error text."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

MAX_BYTES=16384
MAX_EVENTS=96
MAX_FRAMES=8
STAGES=("LOAD_HANDOFF","FORWARD_ADAPTER","BRIDGE_DISPATCH","SELECTED_HOOK",
    "HOOK_CONTEXT_ENTER","HOOK_CONTEXT_EXIT","ADAPTER_POSTCONDITIONS",
    "LOGITS_SERIALIZATION","LOGITS_PUBLICATION","INPUT_CAPTURE_IDENTITY",
    "CLEAR_CAPTURE","CLEANUP_END_EDIT","CLEANUP_PARAMETER_STATE","CLEANUP_PARAMETER_DIGEST",
    "CLEANUP_LATCH_INSPECTION","CLEANUP_HOOK_INSPECTION","CLEANUP_IDENTITY",
    "CLEANUP_GUARD_RESTORE","WRITER_CLOSEOUT")
EDGES=("ENTER","RETURN","RAISE")
CATEGORIES=("VALUE_ERROR","TYPE_ERROR","ATTRIBUTE_ERROR","KEY_ERROR","INDEX_ERROR",
    "OS_ERROR","MEMORY_ERROR","TIMEOUT_ERROR","ASSERTION_ERROR","OVERFLOW_ERROR",
    "RUNTIME_ERROR","OTHER_BASE_EXCEPTION")
ACTIVE=None

class TraceStopped(RuntimeError): pass

def need(value,code):
    if not value:raise TraceStopped(code)

def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()

def category(error):
    for kind,name in ((MemoryError,"MEMORY_ERROR"),(TimeoutError,"TIMEOUT_ERROR"),
        (AssertionError,"ASSERTION_ERROR"),(OverflowError,"OVERFLOW_ERROR"),
        (ValueError,"VALUE_ERROR"),(TypeError,"TYPE_ERROR"),(AttributeError,"ATTRIBUTE_ERROR"),
        (KeyError,"KEY_ERROR"),(IndexError,"INDEX_ERROR"),(OSError,"OS_ERROR"),(RuntimeError,"RUNTIME_ERROR")):
        if isinstance(error,kind):return name
    return "OTHER_BASE_EXCEPTION"

class Trace:
    def __init__(self,execution,source_sha256,allowlist,stopper):
        need(type(execution) is dict and type(source_sha256) is str and len(source_sha256)==64,"TRACE_IDENTITY")
        need(len(allowlist)<=512,"TRACE_SOURCE_BOUND")
        need(all(type(v) is dict and set(v)=={"source","sha256"} and type(v["source"]) is str
            and len(v["source"].encode())<=384 and type(v["sha256"]) is str and len(v["sha256"])==64
            for v in allowlist.values()),"TRACE_SOURCE_SCHEMA")
        self.execution=execution;self.source_sha256=source_sha256;self.allowlist=allowlist
        self.stopper=stopper;self.events=[];self.stack=[];self.primary=None;self.secondary=[]
        self.incomplete=False;self.io_failed=False;self.published=False;self.receipt_sha256=None
        self.closed=False;self.trace_code=None;self.cleanup=False

    def poison(self,code):
        self.incomplete=True
        if self.trace_code is None:self.trace_code=code
        try:self.stopper("NATIVE_DEVELOPMENT_TRACE_FAILURE")
        except BaseException:pass

    def _event(self,stage,edge):
        if stage not in STAGES or edge not in EDGES or len(self.events)>=MAX_EVENTS or self.closed:
            self.poison("TRACE_EVENT_BOUND");raise TraceStopped("TRACE_EVENT_BOUND")
        self.events.append({"ordinal":len(self.events),"stage":stage,"edge":edge,"depth":len(self.stack),"cleanup":self.cleanup})

    def _failure(self,stage,error):
        frames=[];unknown=0;total=0;tb=error.__traceback__
        while tb is not None and total<512:
            total+=1
            key=str(Path(tb.tb_frame.f_code.co_filename).absolute()).casefold()
            known=self.allowlist.get(key)
            if known is None:unknown+=1
            elif 1<=tb.tb_lineno<=1000000:frames.append({"source":known["source"],"sha256":known["sha256"],"line":tb.tb_lineno})
            else:unknown+=1
            tb=tb.tb_next
        complete=tb is None and unknown==0 and len(frames)<=MAX_FRAMES
        # Explicitly partial source locations are never called complete traceback.
        record={"stage":stage,"category":category(error),"frames":frames[-MAX_FRAMES:],
            "frames_complete":complete,"frame_count":total,"unallowlisted_frames":unknown,
            "frame_scan_complete":tb is None,"phase":"CLEANUP" if self.cleanup else "EXECUTION"}
        if self.primary is None:self.primary=record
        elif self.cleanup and len(self.secondary)<16 and record!=self.primary:self.secondary.append(record)
        elif self.cleanup and len(self.secondary)>=16:self.poison("TRACE_SECONDARY_BOUND")

    @contextmanager
    def observe(self,stage):
        self._event(stage,"ENTER");self.stack.append(stage)
        raised=False
        try:
            yield
        except BaseException as error:
            raised=True
            self._failure(stage,error)
            raise
        else:self._event(stage,"RETURN")
        finally:
            # A failure's innermost stage wins; outer propagation is still visible.
            if raised:self._event(stage,"RAISE")
            self.stack.pop()

    def payload(self):
        return {"schema":"first_forward_finite_trace.v1","execution":self.execution,"source_sha256":self.source_sha256,
            "primary":self.primary,"secondary_cleanup_failures":self.secondary,"events":self.events,
            "trace_incomplete":self.incomplete,"trace_code":self.trace_code,"open_stages":list(self.stack),
            "scope":"ONE_NATIVE_DEVELOPMENT_CALL","scientific_pass":False,"exception_text_serialized":False,
            "locals_or_tensors_serialized":False,"maximum_bytes":MAX_BYTES}

    def publish(self,publisher):
        need(not self.published and not self.closed,"TRACE_SINGLE_PUBLICATION")
        self.published=True
        try:
            raw=encoded(self.payload());need(len(raw)<=MAX_BYTES and not self.stack,"TRACE_COMPLETE_BOUND")
            acknowledged=publisher(raw)
            digest=hashlib.sha256(raw).hexdigest()
            need(acknowledged==digest,"TRACE_NATIVE_ACK")
            self.receipt_sha256=digest
        except BaseException:
            self.io_failed=True;self.poison("TRACE_PUBLICATION_FAILURE")
            raise
        finally:self.closed=True
        return self.status()

    def status(self):
        # Finite non-I/O copy survives even a missing/partial native receipt.
        return {"schema":"first_forward_trace_status.v1","execution":self.execution,"source_sha256":self.source_sha256,
            "receipt_sha256":self.receipt_sha256,"published":self.published,"closed":self.closed,
            "incomplete":self.incomplete,"io_failed":self.io_failed,"trace_code":self.trace_code,
            "primary":self.primary,"secondary_cleanup_failures":list(self.secondary),"event_count":len(self.events),
            "scientific_pass":False,"retry_allowed":False}

@contextmanager
def span(stage):
    need(ACTIVE is not None,"TRACE_REQUIRED")
    with ACTIVE.observe(stage):yield

class traced_context:
    def __init__(self,stage,original):
        need(stage=="HOOK_CONTEXT","TRACE_CONTEXT_STAGE");self.original=original
    def __enter__(self):
        with span("HOOK_CONTEXT_ENTER"):return self.original.__enter__()
    def __exit__(self,*exception):
        with span("HOOK_CONTEXT_EXIT"):return self.original.__exit__(*exception)
