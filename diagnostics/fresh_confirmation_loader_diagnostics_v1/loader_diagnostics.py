"""Finite side-channel diagnostics; no model imports, scientific decisions or retries."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
MAX_RECEIPT=2*1024**2
MAX_ENUMERATION=768*1024
MAX_STATUS=65536
PREDICATES={
 "real-only authority before research imports":"LD_AUTHORITY_REAL",
 "unchanged production authority before loader":"LD_AUTHORITY_MATCH",
 "pinned installed hook implementation before construction":"LD_INSTALLED_SOURCE",
 "pinned model configuration":"LD_CONFIGURATION",
 "one reserved load before all forwards":"LD_RESERVED_LOAD",
 "loader adds no forward":"LD_LOAD_NO_FORWARD",
 "fitted gate runtime metadata equality":"LD_RUNTIME_METADATA",
 "loaded template bytes only; no encoding":"LD_TEMPLATE",
 "exact production hook-check denominator":"LD_HOOK_DENOMINATOR",
 "complete hook setup admitted before first forward":"LD_SETUP_NO_FORWARD",
 "all actual parameters CPU float32":"LD_PARAMETER_DEVICE_DTYPE",
 "eval and no inherited gradients":"LD_EVAL_GRADIENTS",
 "actual weights match frozen gate runtime":"LD_ORDERED_WEIGHT_DIGEST",
 "no inherited bridge cache":"LD_BRIDGE_CACHE"}
STAGES=frozenset(("ENTRY","AUTHORITY","INSTALLED_SOURCE","CONFIGURATION","GUARD_INSTALL","LOAD",
    "RUNTIME_METADATA","TEMPLATE","HOOK_SETUP","ADAPTER_CONSTRUCTOR","ADAPTER_READY"))


class DiagnosticStopped(RuntimeError): pass


def encoded(value): return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def valid_sha(value): return type(value) is str and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def finite(value,depth=0):
    if depth>12: raise DiagnosticStopped("LD_SCHEMA")
    if value is None or type(value) is bool: return
    if type(value) is int and -(2**63)<=value<2**64: return
    if type(value) is str and len(value.encode())<=512: return
    if type(value) is list and len(value)<=2048:
        for x in value: finite(x,depth+1)
        return
    if type(value) is dict and len(value)<=64:
        for k,v in value.items():
            if type(k) is not str or len(k.encode())>128: raise DiagnosticStopped("LD_SCHEMA")
            finite(v,depth+1)
        return
    raise DiagnosticStopped("LD_SCHEMA")


class Context:
    def __init__(self,execution,publish,source_sha256):
        finite(execution)
        if len(encoded(execution))>8192: raise DiagnosticStopped("LD_SOURCE_BINDING")
        if not valid_sha(source_sha256): raise DiagnosticStopped("LD_SOURCE_BINDING")
        self.execution=json.loads(encoded(execution));self.source_sha256=source_sha256;self.publisher=publish
        self.stage="ENTRY";self.events=[];self.first_failure=None;self.enumerations={};self.digests={}
        self.latch=None;self.recorder_status=None;self.status_unavailable=False
        self.publish_attempted=False;self.receipt_sha256=None;self.diagnostic_io_failed=False
    def enter(self,stage):
        if stage not in STAGES or len(self.events)>=32: raise DiagnosticStopped("LD_STAGE_SCHEMA")
        self.stage=stage;self.events.append(stage)
    def fail(self,code):
        if code not in set(PREDICATES.values())|{"LD_EXCEPTION","LD_METADATA_CAPACITY","LD_DIAGNOSTIC_IO","LD_STATUS_UNAVAILABLE"}:
            code="LD_EXCEPTION"
        if self.first_failure is None: self.first_failure={"stage":self.stage,"predicate_code":code}
    def check(self,original,value,message):
        # Original value evaluation and original require exception are preserved.
        try: return original(value,message)
        except BaseException:
            self.fail(PREDICATES.get(message,"LD_EXCEPTION"))
            raise
    def capture_enumeration(self,label,named):
        try:
            if label not in ("BEFORE_HOOK_SETUP","ADAPTER_CAPTURE") or label in self.enumerations or len(named)>1024:
                raise DiagnosticStopped("LD_METADATA_CAPACITY")
            records=[]
            for name,p in named:
                if type(name) is not str or len(name.encode())>256: raise DiagnosticStopped("LD_METADATA_CAPACITY")
                shape=list(p.shape)
                if len(shape)>8 or any(type(n) is not int or not 0<=n<2**63 for n in shape): raise DiagnosticStopped("LD_METADATA_CAPACITY")
                dtype=str(p.dtype)
                if dtype not in ("torch.float32","torch.float64","torch.float16","torch.bfloat16","torch.int64","torch.int32","float32","float64","float16","bfloat16","int64","int32"):
                    raise DiagnosticStopped("LD_METADATA_CAPACITY")
                record={"ordinal":len(records),"name":name,"identity":id(p),"shape":shape,"dtype":dtype,
                    "device_type":p.device.type,"requires_grad":p.requires_grad,"version":p._version}
                finite(record);records.append(record)
            raw=encoded(records)
            if len(raw)>MAX_ENUMERATION: raise DiagnosticStopped("LD_METADATA_CAPACITY")
            self.enumerations[label]={"complete":True,"metadata_sha256":sha(raw),"records":records}
        except BaseException:
            self.fail("LD_METADATA_CAPACITY")
            raise DiagnosticStopped("LD_METADATA_CAPACITY") from None
    def retain_digest(self,actual,expected):
        if not valid_sha(actual) or (expected is not None and not valid_sha(expected)):
            self.fail("LD_METADATA_CAPACITY");raise DiagnosticStopped("LD_METADATA_CAPACITY")
        self.digests={"actual_ordered_sha256":actual,"expected_frozen_sha256":expected,
            "new_parameter_content_hash_calls":0,"algorithm_changed":False}
    def capture_recorder(self):
        raw=getattr(self.latch,"recorder",None) if self.latch is not None else None
        if raw is not None:
            try:
                value=raw.status();finite(value)
                if len(encoded(value))>MAX_STATUS: raise DiagnosticStopped("LD_STATUS_UNAVAILABLE")
                self.recorder_status=json.loads(encoded(value))
            except BaseException:
                self.status_unavailable=True;self.fail("LD_STATUS_UNAVAILABLE")
    def payload(self):
        self.capture_recorder()
        value={"schema":"finite_loader_diagnostics.v1","source_sha256":self.source_sha256,"execution":self.execution,
            "stage":self.stage,"stage_events":list(self.events),"first_failure":self.first_failure,
            "parameter_enumerations":self.enumerations,"already_computed_digests":self.digests,
            "recorder_status":self.recorder_status,"recorder_status_unavailable":self.status_unavailable,
            "exception_text_serialized":False,"parameter_bytes_serialized":False,"scientific_verdict":"NOT_SUPPLIED"}
        if len(encoded(value))>MAX_RECEIPT:
            self.fail("LD_METADATA_CAPACITY");raise DiagnosticStopped("LD_METADATA_CAPACITY")
        return value
    def publish(self):
        if self.publish_attempted: raise DiagnosticStopped("LD_DIAGNOSTIC_IO")
        self.publish_attempted=True
        try:
            value=self.payload()
            if self.status_unavailable: raise DiagnosticStopped("LD_STATUS_UNAVAILABLE")
            raw=encoded(value);self.publisher(raw);self.receipt_sha256=sha(raw)
        except BaseException:
            self.diagnostic_io_failed=True;self.fail("LD_DIAGNOSTIC_IO")
            raise DiagnosticStopped("LD_DIAGNOSTIC_IO") from None
    def ready(self,adapter):
        self.enter("ADAPTER_READY")
        self.publish()  # Still inside the original loader try/cleanup boundary.
        return adapter
    def failure_closeout(self):
        self.fail("LD_EXCEPTION")
        self.capture_recorder()
        if not self.publish_attempted:
            try: self.publish()
            except BaseException: pass  # Original exception stays primary; no retry.
    def terminal(self):
        self.capture_recorder()
        try: value=self.payload()
        except BaseException:
            value={"schema":"finite_loader_diagnostics.v1","source_sha256":self.source_sha256,"execution":self.execution,
                "stage":self.stage,"first_failure":self.first_failure,"complete_metadata":False,"scientific_verdict":"NOT_SUPPLIED"}
        value.update(publish_attempted=self.publish_attempted,receipt_sha256=self.receipt_sha256,
            diagnostic_io_failed=self.diagnostic_io_failed,complete_diagnostic_evidence=bool(self.receipt_sha256 and not self.diagnostic_io_failed and not self.status_unavailable),
            permits_scientific_pass=False,retry_allowed=False)
        return value


def new_context(writer,admitted):
    # The new binding must supply the reviewed reserved native control publisher.
    # It is intentionally absent until a separate real setup-only release.
    publish=writer.loader_diagnostic_publisher
    pin=sha((HERE/"SOURCE_FREEZE.json").read_bytes())
    context=Context(admitted["execution"],publish,pin)
    writer.loader_diagnostics=context
    return context


def retained_terminal(original,model,gate,writer,state,event,encode):
    # No cleanup or scientific predicate is replaced. Only saved terminal fields.
    capture=original(model,gate,writer,state,event,encode)
    context=getattr(writer,"loader_diagnostics",None)
    if context is not None:
        diagnostic=context.terminal()
        capture["terminal"]["loader_diagnostics"]=diagnostic
        if model is None and diagnostic.get("recorder_status") is not None:
            capture["terminal"]["recorder_status"]=diagnostic["recorder_status"]
    return capture
